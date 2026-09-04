// Byte counts become floats only to be printed as "1.4 MiB".
#![allow(clippy::cast_precision_loss)]

//! `indexander` — index a directory of text, then search it.
//!
//! Argument parsing is done by hand on purpose: the binary has two
//! subcommands and adding a dependency to parse them would be the first step
//! toward a startup time we would then have to defend.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use std::sync::Arc;
use std::time::Duration;

use indexander_cluster::coordinator::Coordinator;
use indexander_cluster::leases::LeaseAuthority;
use indexander_core::DocId;
use indexander_core::Document;
use indexander_crawl::Config;
use indexander_crawl::politeness::Politeness;
use indexander_index::builder::SegmentBuilder;
use indexander_index::manifest::Policy as MergePolicy;
use indexander_index::merger::Merger;
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;
use indexander_rank::graph::GraphBuilder;
use indexander_rank::pagerank::{Options as RankOptions, pagerank};
use url::Url;

const USAGE: &str = "\
indexander — a search engine

USAGE:
    indexander crawl <url>... [--out <segment>] [--pages <n>] [--depth <n>]
                              [--delay <ms>] [--concurrency <n>] [--any-host]
    indexander index <directory> [--out <segment>]
    indexander shard  --listen <addr> [--index <segment>]
    indexander leases --listen <addr> [--floor <ms>]
    indexander merge  <directory> [--per-tier <n>] [--once]
    indexander search <query>... [--index <segment>] [--limit <n>]
                                 [--shards <addr,addr,...>]
    indexander stats [--index <segment>]

The default segment path is ./indexander.ixdr

EXAMPLES:
    indexander crawl https://example.com --pages 200 --depth 2
    indexander index ./docs
    indexander search motor de busqueda
    indexander search '\"inverted index\"' -perl --limit 5
    indexander shard --listen 127.0.0.1:7801 --index shard0.ixdr
    indexander leases --listen 127.0.0.1:7900 --floor 1000
    indexander merge ./index --per-tier 8
    indexander crawl https://example.com --leases 127.0.0.1:7900
    indexander search motor --shards 127.0.0.1:7801,127.0.0.1:7802
";

const DEFAULT_SEGMENT: &str = "indexander.ixdr";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("indexander: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let Some(command) = args.first() else {
        print!("{USAGE}");
        return Ok(());
    };
    match command.as_str() {
        "crawl" => cmd_crawl(&args[1..]),
        "shard" => cmd_shard(&args[1..]),
        "leases" => cmd_leases(&args[1..]),
        "merge" => cmd_merge(&args[1..]),
        "index" => cmd_index(&args[1..]),
        "search" => cmd_search(&args[1..]),
        "stats" => cmd_stats(&args[1..]),
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            Ok(())
        }
        "-V" | "--version" => {
            println!("indexander {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    }
}

/// Pulls `--name value` out of `args`, leaving the rest untouched.
fn take_option(args: &[String], name: &str) -> (Option<String>, Vec<String>) {
    let mut rest = Vec::with_capacity(args.len());
    let mut value = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            value = iter.next().cloned();
        } else {
            rest.push(arg.clone());
        }
    }
    (value, rest)
}

/// Crawls the web from one or more seeds and indexes what comes back.
///
/// The crawl and the indexing run at the same time: pages are indexed as they
/// arrive rather than after the crawl finishes, which is what keeps memory
/// proportional to the index and not to the network.
fn cmd_crawl(args: &[String]) -> Result<(), String> {
    let (out, rest) = take_option(args, "--out");
    let (pages, rest) = take_option(&rest, "--pages");
    let (depth, rest) = take_option(&rest, "--depth");
    let (delay, rest) = take_option(&rest, "--delay");
    let (concurrency, rest) = take_option(&rest, "--concurrency");
    let (leases, rest) = take_option(&rest, "--leases");
    let (any_host, seeds): (Vec<String>, Vec<String>) =
        rest.into_iter().partition(|a| a == "--any-host");

    if seeds.is_empty() {
        return Err(format!("crawl needs at least one url\n\n{USAGE}"));
    }
    let seeds: Vec<Url> = seeds
        .iter()
        .map(|s| {
            let with_scheme = if s.contains("://") {
                s.clone()
            } else {
                format!("https://{s}")
            };
            Url::parse(&with_scheme).map_err(|e| format!("{s}: {e}"))
        })
        .collect::<Result<_, _>>()?;

    let config = crawl_config(
        pages.as_deref(),
        depth.as_deref(),
        delay.as_deref(),
        concurrency.as_deref(),
        any_host.is_empty(),
    )?;

    let out = PathBuf::from(out.unwrap_or_else(|| DEFAULT_SEGMENT.to_owned()));
    println!(
        "crawling {} seed{} as {}",
        seeds.len(),
        if seeds.len() == 1 { "" } else { "s" },
        config.user_agent
    );

    let runtime = tokio::runtime::Runtime::new().map_err(|e| format!("starting runtime: {e}"))?;
    let started = std::time::Instant::now();

    // The index is built as pages arrive, but PageRank cannot be: it needs the
    // whole graph. So the crawl builds both, and the ranks are applied at the
    // end, before the segment is written.
    let (mut builder, graph_builder, stats) = runtime.block_on(async {
        let politeness = match &leases {
            Some(address) => Arc::new(
                indexander_cluster::leases::remote_politeness(address)
                    .await
                    .map_err(|e| format!("lease authority {address}: {e}"))?,
            ),
            None => Arc::new(Politeness::local(config.delay)),
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let crawling = indexander_crawl::crawl_with(&config, &seeds, tx, politeness);
        let indexing = async {
            let mut builder = SegmentBuilder::new();
            let mut graph = GraphBuilder::new();
            while let Some(doc) = rx.recv().await {
                if builder.document_count() % 25 == 0 && builder.document_count() > 0 {
                    println!("  {} pages...", builder.document_count());
                }
                graph.node(&doc.uri);
                for link in &doc.links {
                    graph.edge(&doc.uri, link);
                }
                builder.add(&doc);
            }
            (builder, graph)
        };
        let (stats, (builder, graph)) = tokio::join!(crawling, indexing);
        stats.map(|s| (builder, graph, s))
    })?;

    apply_pagerank(&mut builder, graph_builder)?;

    builder
        .write_to(&out)
        .map_err(|e| format!("writing {}: {e}", out.display()))?;

    let size = std::fs::metadata(&out).map_or(0, |m| m.len());
    println!(
        "\nfetched {}, indexed {}, {} terms in {:.2?}",
        stats.fetched,
        stats.indexed,
        builder.term_count(),
        started.elapsed()
    );
    if stats.disallowed_by_robots > 0 {
        println!(
            "{} url(s) skipped by robots.txt",
            stats.disallowed_by_robots
        );
    }
    if stats.skipped_content_type > 0 {
        println!("{} url(s) skipped, not text", stats.skipped_content_type);
    }
    if stats.errors > 0 {
        println!("{} url(s) failed to fetch", stats.errors);
    }
    println!("{} -> {}", out.display(), human_bytes(size));
    Ok(())
}

/// Computes PageRank over the crawled graph and writes each score into the
/// segment being built.
///
/// Separate from the crawl because it cannot start until the crawl is over:
/// a page's authority depends on pages that may not have been fetched yet.
fn apply_pagerank(builder: &mut SegmentBuilder, graph_builder: GraphBuilder) -> Result<(), String> {
    let nodes = graph_builder.node_count();
    let edges = graph_builder.edge_count();
    let graph = graph_builder.build();
    let options = RankOptions::default();
    let ranks = pagerank(&graph, &options);

    println!(
        "link graph: {nodes} nodes, {edges} edges; pagerank {} in {} iteration{}",
        if ranks.converged(&options) {
            "converged"
        } else {
            "stopped early"
        },
        ranks.iterations,
        if ranks.iterations == 1 { "" } else { "s" }
    );

    for i in 0..builder.document_count() {
        let id = DocId(u32::try_from(i).map_err(|_| "too many documents")?);
        if let Some(node) = builder.uri(id).and_then(|uri| graph.id(uri)) {
            builder.set_rank(id, ranks.score(node));
        }
    }

    let mut top = ranks.ranked();
    top.truncate(5);
    if !top.is_empty() {
        println!("\nmost linked-to pages:");
        for (node, score) in top {
            println!("  {score:.5}  {}", graph.uri(node).unwrap_or("?"));
        }
    }
    Ok(())
}

/// Serves one segment to a coordinator.
fn cmd_shard(args: &[String]) -> Result<(), String> {
    let (listen, rest) = take_option(args, "--listen");
    let (index, _) = take_option(&rest, "--index");
    let listen = listen.ok_or_else(|| format!("shard needs --listen\n\n{USAGE}"))?;
    let segment = Arc::new(open_segment(index.as_deref())?);

    println!(
        "shard listening on {listen}: {} documents, {} terms",
        segment.document_count(),
        segment.term_count()
    );

    let runtime = tokio::runtime::Runtime::new().map_err(|e| format!("starting runtime: {e}"))?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(&listen)
            .await
            .map_err(|e| format!("binding {listen}: {e}"))?;
        indexander_cluster::shard::serve(listener, segment)
            .await
            .map_err(|e| format!("serving: {e}"))
    })
}

/// Searches a cluster instead of a local segment.
///
/// This is the same two-round protocol whether there is one shard or fifty,
/// which is the entire point of doing it before there are fifty.
fn search_cluster(addresses: &[String], query_text: &str, limit: usize) -> Result<(), String> {
    let runtime = tokio::runtime::Runtime::new().map_err(|e| format!("starting runtime: {e}"))?;
    runtime.block_on(async {
        let started = std::time::Instant::now();
        let coordinator = Coordinator::connect(addresses)
            .await
            .map_err(|e| e.to_string())?;
        let connected = started.elapsed();

        let searching = std::time::Instant::now();
        let hits = coordinator
            .search(query_text, limit)
            .await
            .map_err(|e| e.to_string())?;
        let elapsed = searching.elapsed();

        if hits.is_empty() {
            println!("no results ({elapsed:.2?})");
            return Ok(());
        }
        for (rank, hit) in hits.iter().enumerate() {
            println!("{:>3}. {:>8.4}  {}", rank + 1, hit.score, hit.uri);
        }
        let (documents, _) = coordinator.stats().await.map_err(|e| e.to_string())?;
        println!(
            "\n{} result{} in {:.2?} across {} shard{} ({documents} documents, connect {:.2?})",
            hits.len(),
            if hits.len() == 1 { "" } else { "s" },
            elapsed,
            coordinator.shard_count(),
            if coordinator.shard_count() == 1 {
                ""
            } else {
                "s"
            },
            connected
        );
        Ok(())
    })
}

/// Turns the crawl command's numeric options into a config.
fn crawl_config(
    pages: Option<&str>,
    depth: Option<&str>,
    delay: Option<&str>,
    concurrency: Option<&str>,
    same_host_only: bool,
) -> Result<Config, String> {
    let parse_num = |value: Option<&str>, name: &str, default: usize| -> Result<usize, String> {
        value
            .map_or(Ok(default), str::parse)
            .map_err(|_| format!("{name} needs a number"))
    };

    let mut config = Config::default();
    config.limits.max_pages = parse_num(pages, "--pages", config.limits.max_pages)?;
    config.limits.max_depth =
        u32::try_from(parse_num(depth, "--depth", 3)?).map_err(|_| "--depth too large")?;
    config.limits.max_pages_per_host = config.limits.max_pages;
    config.limits.same_host_only = same_host_only;
    config.concurrency = parse_num(concurrency, "--concurrency", config.concurrency)?;
    config.delay = Duration::from_millis(
        parse_num(delay, "--delay", 500)?
            .try_into()
            .map_err(|_| "--delay too large")?,
    );
    Ok(config)
}

/// Runs the node that owns a set of hosts' rate limits.
///
/// A crawl spread over several machines needs exactly one of these per host,
/// or each machine paces itself and the site receives the sum. See
/// `docs/DISTRIBUTION.md`.
fn cmd_leases(args: &[String]) -> Result<(), String> {
    let (listen, rest) = take_option(args, "--listen");
    let (floor, _) = take_option(&rest, "--floor");
    let listen = listen.ok_or_else(|| format!("leases needs --listen\n\n{USAGE}"))?;
    let floor: u64 = floor
        .as_deref()
        .map_or(Ok(500), str::parse)
        .map_err(|_| "--floor needs a number of milliseconds".to_owned())?;

    println!("lease authority on {listen}, minimum {floor} ms between requests to a host");

    let runtime = tokio::runtime::Runtime::new().map_err(|e| format!("starting runtime: {e}"))?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(&listen)
            .await
            .map_err(|e| format!("binding {listen}: {e}"))?;
        let authority = Arc::new(LeaseAuthority::new(Duration::from_millis(floor)));
        authority
            .serve(listener)
            .await
            .map_err(|e| format!("serving: {e}"))
    })
}

/// Folds an index's segments together, as a background merger would.
///
/// Safe to interrupt. A merge writes its new segment before replacing the
/// manifest, and the manifest is the only thing that decides what an index is,
/// so a merge killed halfway leaves a file nobody references rather than an
/// index nobody can read.
fn cmd_merge(args: &[String]) -> Result<(), String> {
    let (per_tier, rest) = take_option(args, "--per-tier");
    let (once, rest): (Vec<String>, Vec<String>) = rest.into_iter().partition(|a| a == "--once");
    let directory = rest
        .first()
        .ok_or_else(|| format!("merge needs a directory\n\n{USAGE}"))?;

    let mut policy = MergePolicy::default();
    if let Some(value) = per_tier {
        policy.segments_per_tier = value
            .parse()
            .map_err(|_| "--per-tier needs a number".to_owned())?;
    }

    let merger = Merger::new(Path::new(directory), policy);
    let started = std::time::Instant::now();
    let reports = if once.is_empty() {
        merger.run_to_completion().map_err(|e| e.to_string())?
    } else {
        merger
            .step()
            .map_err(|e| e.to_string())?
            .into_iter()
            .collect()
    };

    if reports.is_empty() {
        println!("nothing to merge");
        return Ok(());
    }
    let mut folded = 0usize;
    for report in &reports {
        println!(
            "  {} segments -> {} ({} documents, {})",
            report.merged.len(),
            report.produced,
            report.documents,
            human_bytes(report.bytes)
        );
        folded += report.merged.len();
        for stuck in &report.undeleted {
            println!("    {stuck} could not be removed; it is unreferenced now");
        }
    }
    println!(
        "{} merge{} folding {folded} segments in {:.2?}",
        reports.len(),
        if reports.len() == 1 { "" } else { "s" },
        started.elapsed()
    );

    let orphans = merger.orphans().map_err(|e| e.to_string())?;
    if !orphans.is_empty() {
        println!(
            "\n{} unreferenced file{} in the directory: {}",
            orphans.len(),
            if orphans.len() == 1 { "" } else { "s" },
            orphans.join(", ")
        );
        println!("left in place: whether they are safe to delete depends on who is still reading");
    }
    Ok(())
}

fn cmd_index(args: &[String]) -> Result<(), String> {
    let (out, rest) = take_option(args, "--out");
    let directory = rest
        .first()
        .ok_or_else(|| format!("index needs a directory\n\n{USAGE}"))?;
    let out = PathBuf::from(out.unwrap_or_else(|| DEFAULT_SEGMENT.to_owned()));

    let started = std::time::Instant::now();
    let files = collect_files(Path::new(directory))?;
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let (builder, bytes_read) = build_index(&files, threads);

    builder
        .write_to(&out)
        .map_err(|e| format!("writing {}: {e}", out.display()))?;

    let elapsed = started.elapsed();
    let size = std::fs::metadata(&out).map_or(0, |m| m.len());
    println!(
        "indexed {} documents, {} terms in {:.2?} on {threads} thread{}",
        builder.document_count(),
        builder.term_count(),
        elapsed,
        if threads == 1 { "" } else { "s" }
    );
    println!(
        "{} -> {} ({:.1}% of {} of text)",
        out.display(),
        human_bytes(size),
        if bytes_read == 0 {
            0.0
        } else {
            size as f64 * 100.0 / bytes_read as f64
        },
        human_bytes(bytes_read)
    );
    Ok(())
}

/// Reads and indexes `files`, using `threads` of them.
///
/// Tokenising a document depends on nothing but that document, so the corpus
/// splits cleanly. Each thread builds a partial index over a contiguous slice
/// and the parts are stitched together with `absorb`, which shifts document
/// ids so the result is exactly what a single pass would have produced -
/// `chunked_building_is_byte_identical_to_one_pass` is the test that says so.
///
/// Slices are contiguous rather than interleaved so document ids stay in file
/// order, which keeps a rebuild reproducible.
fn build_index(files: &[PathBuf], threads: usize) -> (SegmentBuilder, u64) {
    let threads = threads.max(1).min(files.len().max(1));
    let chunk = files.len().div_ceil(threads).max(1);

    let parts: Vec<(SegmentBuilder, u64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = files
            .chunks(chunk)
            .map(|slice| scope.spawn(move || index_files(slice)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| (SegmentBuilder::new(), 0)))
            .collect()
    });

    merge_tree(parts)
}

/// Merges partial indexes pairwise, in parallel, until one is left.
///
/// Absorbing them one after another into the first is correct but serial, and
/// with fourteen threads that tail is most of what is left to save: merging
/// costs roughly the size of what is merged, so a chain does `n` merges of a
/// growing index while a tree does the same total work in `log2(n)` rounds
/// that each run concurrently.
///
/// The order of absorption is preserved exactly — adjacent parts only, left
/// into right — so document ids stay in corpus order and the result is still
/// byte-identical to a single-threaded build.
fn merge_tree(mut parts: Vec<(SegmentBuilder, u64)>) -> (SegmentBuilder, u64) {
    while parts.len() > 1 {
        parts = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(parts.len().div_ceil(2));
            // `into_iter` in chunks of two: each pair merges independently.
            let mut iter = parts.into_iter();
            while let Some(left) = iter.next() {
                match iter.next() {
                    Some(right) => handles.push(scope.spawn(move || {
                        let (mut a, a_bytes) = left;
                        let (b, b_bytes) = right;
                        a.absorb(b);
                        (a, a_bytes + b_bytes)
                    })),
                    // An odd part rides to the next round untouched.
                    None => handles.push(scope.spawn(move || left)),
                }
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| (SegmentBuilder::new(), 0)))
                .collect()
        });
    }
    parts.pop().unwrap_or_else(|| (SegmentBuilder::new(), 0))
}

/// Indexes one slice of the corpus into its own builder.
fn index_files(files: &[PathBuf]) -> (SegmentBuilder, u64) {
    let mut builder = SegmentBuilder::new();
    let mut bytes = 0u64;
    for path in files {
        let Ok(raw) = std::fs::read(path) else {
            continue;
        };
        if is_binary(&raw) {
            continue;
        }
        let body = decode(raw);
        bytes += body.len() as u64;
        let title = path
            .file_stem()
            .map(|s| s.to_string_lossy().replace(['-', '_'], " "))
            .unwrap_or_default();
        builder.add(&Document::new(path.to_string_lossy(), title, body));
    }
    (builder, bytes)
}

fn cmd_search(args: &[String]) -> Result<(), String> {
    let (index, rest) = take_option(args, "--index");
    let (shards, rest) = take_option(&rest, "--shards");
    let (limit, rest) = take_option(&rest, "--limit");
    let limit: usize = limit
        .as_deref()
        .map_or(Ok(10), str::parse)
        .map_err(|_| "--limit needs a number".to_owned())?;
    if rest.is_empty() {
        return Err(format!("search needs a query\n\n{USAGE}"));
    }

    if let Some(shards) = shards {
        let addresses: Vec<String> = shards
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        if addresses.is_empty() {
            return Err("--shards needs at least one address".to_owned());
        }
        return search_cluster(&addresses, &rest.join(" "), limit);
    }

    let segment = open_segment(index.as_deref())?;
    let parsed = query::parse(&rest.join(" "));

    let started = std::time::Instant::now();
    let hits = search(&segment, &parsed, limit).map_err(|e| e.to_string())?;
    let elapsed = started.elapsed();

    if hits.is_empty() {
        println!("no results ({elapsed:.2?})");
        return Ok(());
    }
    for (rank, hit) in hits.iter().enumerate() {
        println!("{:>3}. {:>8.4}  {}", rank + 1, hit.score, hit.uri);
    }
    println!(
        "\n{} result{} in {:.2?} over {} documents",
        hits.len(),
        if hits.len() == 1 { "" } else { "s" },
        elapsed,
        segment.document_count()
    );
    Ok(())
}

fn cmd_stats(args: &[String]) -> Result<(), String> {
    let (index, _) = take_option(args, "--index");
    let segment = open_segment(index.as_deref())?;
    println!("documents          {}", segment.document_count());
    println!("terms              {}", segment.term_count());
    println!(
        "avg doc length     {:.1} tokens",
        segment.average_document_length()
    );
    Ok(())
}

fn open_segment(path: Option<&str>) -> Result<Segment, String> {
    let path = Path::new(path.unwrap_or(DEFAULT_SEGMENT));
    Segment::open(path).map_err(|e| {
        if path.exists() {
            format!("reading {}: {e}", path.display())
        } else {
            format!(
                "no index at {}; run `indexander index <dir>` first",
                path.display()
            )
        }
    })
}

/// Every readable file under `root`, depth first. Hidden entries are skipped.
fn collect_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    if !root.exists() {
        return Err(format!("{} does not exist", root.display()));
    }
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            let hidden = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('.'));
            if hidden {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Decodes file bytes to text.
///
/// UTF-8 when it is valid UTF-8, Latin-1 otherwise. Latin-1 is not a guess so
/// much as a floor: every byte maps to exactly one code point, so nothing is
/// lost and nothing fails, which is more than can be said for refusing to read
/// the file at all. Real charset detection belongs here later; skipping the
/// document does not.
fn decode(raw: Vec<u8>) -> String {
    match String::from_utf8(raw) {
        Ok(text) => text,
        Err(e) => e.into_bytes().into_iter().map(|b| b as char).collect(),
    }
}

/// A NUL byte in the first few KiB means this is not text.
fn is_binary(raw: &[u8]) -> bool {
    raw.iter().take(8192).any(|&b| b == 0)
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
