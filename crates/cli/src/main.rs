// Byte counts become floats only to be printed as "1.4 MiB".
#![allow(clippy::cast_precision_loss)]

//! `indexander` — index a directory of text, then search it.
//!
//! Argument parsing is done by hand on purpose: the binary has two
//! subcommands and adding a dependency to parse them would be the first step
//! toward a startup time we would then have to defend.

mod eval;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use std::sync::Arc;
use std::time::Duration;

use indexander_cluster::coordinator::Coordinator;
use indexander_cluster::leases::LeaseAuthority;
use indexander_cluster::shard::{Replica, ShardIndex};
use indexander_core::DocId;
use indexander_core::Document;
use indexander_crawl::Config;
use indexander_crawl::politeness::Politeness;
use indexander_crawl::robots_cache::RobotsCache;
use indexander_index::builder::SegmentBuilder;
use indexander_index::manifest::{Entry as ManifestEntry, Manifest, Policy as MergePolicy};
use indexander_index::merger::Merger;
use indexander_index::query;
use indexander_index::scoring::Params;
use indexander_index::search::search;
use indexander_index::segment::Segment;
use indexander_index::snippet;
use indexander_rank::graph::GraphBuilder;
use indexander_rank::pagerank::{Options as RankOptions, pagerank};
use url::Url;

const USAGE: &str = "\
indexander — a search engine

USAGE:
    indexander crawl <url>... [--out <segment>] [--pages <n>] [--depth <n>]
                              [--delay <ms>] [--concurrency <n>] [--any-host]
                              [--k1 <n>] [--b <n>]
    indexander index <directory> [--out <segment>] [--k1 <n>] [--b <n>]
    indexander shard  --listen <addr> [--index <segment>] [--from <addr>]
    indexander leases --listen <addr> [--floor <ms>]     rate limits + robots.txt
    indexander manifest <directory>                       assemble an index
    indexander merge  <directory> [--per-tier <n>] [--once] [--every <secs>]
                                  [--notify <addr,addr,...>]
    indexander sync   <directory> --from <addr> [--every <secs>]
    indexander search <query>... [--index <segment>] [--limit <n>]
                                 [--shards <addr,addr,...>]
    indexander stats [--index <segment>]
    indexander eval  --queries <file> --qrels <file> [--index <segment>] [--k <n>]
    indexander known-item <directory> [--index <segment>] [--from body|title]
                          [--sample <n>] [--span <n>] [--seed <n>] [--k <n>]

The default segment path is ./indexander.ixdr

EXAMPLES:
    indexander crawl https://example.com --pages 200 --depth 2
    indexander index ./docs
    indexander index ./docs --b 1.0        # see docs/EVALUATION.md
    indexander search motor de busqueda
    indexander search '\"inverted index\"' -perl --limit 5
    indexander shard --listen 127.0.0.1:7801 --index shard0.ixdr
    indexander leases --listen 127.0.0.1:7900 --floor 1000
    indexander manifest ./index
    indexander merge ./index --per-tier 8 --every 60 --notify 127.0.0.1:7801
    indexander sync ./replica --from 127.0.0.1:7801 --every 300
    indexander sync ./replica --from 127.0.0.1:7801
    indexander crawl https://example.com --leases 127.0.0.1:7900
    indexander search motor --shards 127.0.0.1:7801,127.0.0.1:7802
    indexander known-item ./docs --sample 500
    indexander eval --queries topics.tsv --qrels qrels.txt --k 10
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
        "manifest" => cmd_manifest(&args[1..]),
        "merge" => cmd_merge(&args[1..]),
        "sync" => cmd_sync(&args[1..]),
        "index" => cmd_index(&args[1..]),
        "search" => cmd_search(&args[1..]),
        "stats" => cmd_stats(&args[1..]),
        "eval" => eval::cmd_eval(&args[1..]),
        "known-item" => eval::cmd_known_item(&args[1..]),
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
pub(crate) fn take_option(args: &[String], name: &str) -> (Option<String>, Vec<String>) {
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
    let (params, args) = take_params(args)?;
    let args = &args[..];
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
        let (politeness, robots) = shared_or_local(leases.as_deref(), config.delay).await?;
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let crawling = indexander_crawl::crawl_sharing(&config, &seeds, tx, politeness, robots);
        let indexing = async {
            let mut builder = SegmentBuilder::with_params(params);
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
    let (index, rest) = take_option(&rest, "--index");
    let (from, _) = take_option(&rest, "--from");
    let listen = listen.ok_or_else(|| format!("shard needs --listen\n\n{USAGE}"))?;
    // A directory with a manifest, or a single segment file. A shard that
    // holds a whole index can serve its segments to a replica by name; one
    // holding a lone file can only serve itself.
    let path = std::path::PathBuf::from(index.as_deref().unwrap_or(DEFAULT_SEGMENT));
    if from.is_some() && !path.is_dir() {
        return Err("--from needs a directory to sync into, not a single segment".to_owned());
    }
    let replica = Arc::new(if path.is_dir() {
        Replica::following(&path, from.clone())
            .map_err(|e| format!("opening {}: {e}", path.display()))?
    } else {
        Replica::fixed(ShardIndex::single(open_segment(index.as_deref())?))
    });

    let shard = replica.shard();
    println!(
        "shard listening on {listen}: {} documents in {} segment{}{}",
        shard.index().document_count(),
        shard.index().segment_count(),
        if shard.index().segment_count() == 1 {
            ""
        } else {
            "s"
        },
        match &from {
            Some(source) => format!(", following {source}"),
            None => String::new(),
        }
    );

    let runtime = tokio::runtime::Runtime::new().map_err(|e| format!("starting runtime: {e}"))?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(&listen)
            .await
            .map_err(|e| format!("binding {listen}: {e}"))?;
        indexander_cluster::shard::serve_replica(listener, replica)
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
        let (documents, _, _) = coordinator.stats().await.map_err(|e| e.to_string())?;
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

/// What the crawler should defer to: an authority, or this process.
///
/// One address, two shared things — who may fetch from a host next, and what
/// that host's robots.txt says. They belong together, because whoever owns a
/// host's rate limit is the natural place to remember what it said about being
/// crawled, and it means one address per host rather than two.
async fn shared_or_local(
    leases: Option<&str>,
    delay: Duration,
) -> Result<(Arc<Politeness>, Arc<RobotsCache>), String> {
    match leases {
        Some(address) => {
            let fail = |e: indexander_core::Error| format!("lease authority {address}: {e}");
            Ok((
                Arc::new(
                    indexander_cluster::leases::remote_politeness(address)
                        .await
                        .map_err(fail)?,
                ),
                Arc::new(
                    indexander_cluster::leases::remote_robots_cache(address)
                        .await
                        .map_err(fail)?,
                ),
            ))
        }
        None => Ok((
            Arc::new(Politeness::local(delay)),
            Arc::new(RobotsCache::default()),
        )),
    }
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
/// Writes a MANIFEST naming every segment in a directory.
///
/// The gap this closes: `index` writes one segment, and everything that treats
/// an index as several - merging, replication, a shard serving its pieces by
/// name - needs a manifest to say which they are and in what order. Without
/// this the whole of that machinery was reachable only from a test.
///
/// Segments are taken in file name order, and that order is the index's: the
/// documents of the first come before the documents of the second, and their
/// ids are assigned that way when they merge. Renaming a segment therefore
/// renumbers documents, which is why this refuses to guess and simply sorts.
fn cmd_manifest(args: &[String]) -> Result<(), String> {
    let directory = args
        .first()
        .ok_or_else(|| format!("manifest needs a directory\n\n{USAGE}"))?;
    let directory = Path::new(directory);

    let mut names: Vec<String> = std::fs::read_dir(directory)
        .map_err(|e| format!("{}: {e}", directory.display()))?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        // A case-insensitive match, because a directory copied through a
        // filesystem that does not preserve case would otherwise look empty.
        .filter(|n| {
            std::path::Path::new(n)
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("ixdr"))
        })
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(format!("{} holds no segments", directory.display()));
    }

    let mut manifest = Manifest::new();
    for name in &names {
        let path = directory.join(name);
        let segment = Segment::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        // Every segment is opened rather than trusted: a manifest naming a
        // file that will not parse is a directory that fails at query time
        // instead of here.
        manifest.segments.push(ManifestEntry {
            name: name.clone(),
            digest: segment.digest(),
            documents: segment.document_count(),
            bytes: std::fs::metadata(&path).map_or(0, |m| m.len()),
        });
        println!("  {name}  {} documents", segment.document_count());
    }

    let path = directory.join("MANIFEST");
    manifest
        .write_to(&path)
        .map_err(|e| format!("writing {}: {e}", path.display()))?;
    println!(
        "{} segments, {} documents -> {}",
        manifest.segments.len(),
        manifest.segments.iter().map(|e| e.documents).sum::<usize>(),
        path.display()
    );
    Ok(())
}

fn cmd_merge(args: &[String]) -> Result<(), String> {
    let (per_tier, rest) = take_option(args, "--per-tier");
    let (every, rest) = take_option(&rest, "--every");
    let (notify, rest) = take_option(&rest, "--notify");
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
    let told: Vec<String> = notify
        .as_deref()
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();

    // --every turns this into the background merger: the same step, on a
    // clock. Kept as a loop here rather than a daemon inside the library,
    // because "how often" is an operational decision and belongs where the
    // operator can see it.
    if let Some(seconds) = every {
        let interval: u64 = seconds
            .parse()
            .map_err(|_| "--every needs a number of seconds".to_owned())?;
        println!("merging {directory} every {interval}s; interrupt to stop");
        loop {
            match merger.run_to_completion() {
                Ok(reports) if !reports.is_empty() => {
                    println!("{} merge(s)", reports.len());
                    announce(&told);
                }
                Ok(_) => {}
                // A merge that failed leaves the index as it was, so the right
                // response is to say so and try again rather than to exit.
                Err(e) => eprintln!("merge failed: {e}"),
            }
            std::thread::sleep(Duration::from_secs(interval.max(1)));
        }
    }

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
        report_orphans(&merger)?;
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
    // After the report, not before it: printed the other way round the output
    // reads as though replicas were told about a merge that had not happened.
    announce(&told);
    println!(
        "{} merge{} folding {folded} segments in {:.2?}",
        reports.len(),
        if reports.len() == 1 { "" } else { "s" },
        started.elapsed()
    );

    report_orphans(&merger)?;
    Ok(())
}

/// Names the files in an index directory the manifest does not.
///
/// Reported whether or not anything was merged, because the case where it
/// matters most is the one where nothing was: a replica never merges, and a
/// sync leaves the segments it replaced behind. Nobody would think to ask.
fn report_orphans(merger: &Merger) -> Result<(), String> {
    let orphans = merger.orphans().map_err(|e| e.to_string())?;
    if orphans.is_empty() {
        return Ok(());
    }
    println!(
        "\n{} unreferenced file{} in the directory: {}",
        orphans.len(),
        if orphans.len() == 1 { "" } else { "s" },
        orphans.join(", ")
    );
    println!("left in place: whether they are safe to delete depends on who is still reading");
    Ok(())
}

/// Brings a replica into line with the index served somewhere else.
/// Tells each address to catch up, in the order given.
///
/// The order matters and the tool does not reorder it. A merge happens in this
/// process; every shard *serving* that index opened its manifest at startup
/// and knows nothing about it — including the one whose directory was just
/// merged. So the primary comes first in the list, and the replicas after it:
/// told the other way round, a replica syncs against a manifest that no longer
/// describes the directory and comes away with nothing.
///
/// A nudge that fails is reported and not fatal. The primary has already
/// merged, the index on disk is correct, and refusing to finish because one
/// replica is down would only make an outage worse. What guarantees a replica
/// converges is its own timer, not this.
fn announce(addresses: &[String]) {
    if addresses.is_empty() {
        return;
    }
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        eprintln!("could not start a runtime to notify replicas");
        return;
    };
    for address in addresses {
        match runtime.block_on(indexander_cluster::replication::notify(address)) {
            Ok((fetched, segments)) => {
                println!("  {address}: {segments} segment(s), {fetched} fetched");
            }
            Err(e) => eprintln!("  {address}: not notified: {e}"),
        }
    }
}

fn cmd_sync(args: &[String]) -> Result<(), String> {
    let (from, rest) = take_option(args, "--from");
    let (every, rest) = take_option(&rest, "--every");
    let directory = rest
        .first()
        .ok_or_else(|| format!("sync needs a directory\n\n{USAGE}"))?;
    let from = from.ok_or_else(|| format!("sync needs --from\n\n{USAGE}"))?;

    let runtime = tokio::runtime::Runtime::new().map_err(|e| format!("starting runtime: {e}"))?;

    // The timer, not the notification, is what makes a replica converge. A
    // nudge from the primary is a latency improvement; this is the guarantee.
    if let Some(seconds) = every {
        let interval: u64 = seconds
            .parse()
            .map_err(|_| "--every needs a number of seconds".to_owned())?;
        println!("syncing {directory} from {from} every {interval}s; interrupt to stop");
        loop {
            match runtime.block_on(indexander_cluster::replication::sync_from(
                &from,
                Path::new(directory),
            )) {
                Ok(fetched) if !fetched.is_empty() => {
                    println!("fetched {} segment(s)", fetched.len());
                }
                Ok(_) => {}
                // A source that is down now may be up in a minute, and the
                // replica keeps serving what it has meanwhile.
                Err(e) => eprintln!("sync failed: {e}"),
            }
            std::thread::sleep(Duration::from_secs(interval.max(1)));
        }
    }

    let started = std::time::Instant::now();
    let fetched = runtime
        .block_on(indexander_cluster::replication::sync_from(
            &from,
            Path::new(directory),
        ))
        .map_err(|e| e.to_string())?;

    if fetched.is_empty() {
        println!("already up to date with {from}");
    } else {
        for name in &fetched {
            println!("  fetched {name}");
        }
        println!(
            "{} segment{} from {from} in {:.2?}",
            fetched.len(),
            if fetched.len() == 1 { "" } else { "s" },
            started.elapsed()
        );
    }
    Ok(())
}

/// Reads `--k1` and `--b`, leaving the defaults where a flag is absent.
///
/// They are set when a segment is written and never afterwards: the block-max
/// bounds stored with every postings list are computed with them, so a segment
/// carries them and a reader uses the segment's rather than its own.
/// `docs/EVALUATION.md` has what they are worth and how to find out for a
/// given corpus.
fn take_params(args: &[String]) -> Result<(Params, Vec<String>), String> {
    let (k1, rest) = take_option(args, "--k1");
    let (b, rest) = take_option(&rest, "--b");
    let mut params = Params::default();
    if let Some(k1) = k1 {
        params.k1 = k1.parse().map_err(|_| "--k1 needs a number".to_owned())?;
    }
    if let Some(b) = b {
        params.b = b.parse().map_err(|_| "--b needs a number".to_owned())?;
    }
    if !params.is_usable() {
        return Err(format!(
            "scoring parameters must be finite and not negative, got {params:?}"
        ));
    }
    Ok((params, rest))
}

fn cmd_index(args: &[String]) -> Result<(), String> {
    let (params, args) = take_params(args)?;
    let args = &args[..];
    let (out, rest) = take_option(args, "--out");
    let directory = rest
        .first()
        .ok_or_else(|| format!("index needs a directory\n\n{USAGE}"))?;
    let out = PathBuf::from(out.unwrap_or_else(|| DEFAULT_SEGMENT.to_owned()));

    let started = std::time::Instant::now();
    let files = collect_files(Path::new(directory))?;
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let (builder, bytes_read) = build_index(&files, threads, params);

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
fn build_index(files: &[PathBuf], threads: usize, params: Params) -> (SegmentBuilder, u64) {
    let threads = threads.max(1).min(files.len().max(1));
    let chunk = files.len().div_ceil(threads).max(1);

    let parts: Vec<(SegmentBuilder, u64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = files
            .chunks(chunk)
            .map(|slice| scope.spawn(move || index_files(slice, params)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| (SegmentBuilder::with_params(params), 0))
            })
            .collect()
    });

    merge_tree(parts, params)
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
fn merge_tree(mut parts: Vec<(SegmentBuilder, u64)>, params: Params) -> (SegmentBuilder, u64) {
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
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| (SegmentBuilder::with_params(params), 0))
                })
                .collect()
        });
    }
    parts
        .pop()
        .unwrap_or_else(|| (SegmentBuilder::with_params(params), 0))
}

/// Indexes one slice of the corpus into its own builder.
fn index_files(files: &[PathBuf], params: Params) -> (SegmentBuilder, u64) {
    let mut builder = SegmentBuilder::with_params(params);
    let mut bytes = 0u64;
    for path in files {
        let Ok(raw) = std::fs::read(path) else {
            continue;
        };
        if is_binary(&raw) {
            continue;
        }
        let text = decode(raw);
        bytes += text.len() as u64;
        let stem = || {
            path.file_stem()
                .map(|s| s.to_string_lossy().replace(['-', '_'], " "))
                .unwrap_or_default()
        };
        let (title, body) = if is_html(path, &text) {
            let page = indexander_crawl::extract::extract(&text);
            let title = if page.title.trim().is_empty() {
                stem()
            } else {
                page.title
            };
            (title, page.text)
        } else {
            (stem(), text)
        };
        builder.add(&Document::new(path.to_string_lossy(), title, body));
    }
    (builder, bytes)
}

/// The text of a file as the indexer stored it.
///
/// Everything that reads a document back — a snippet, a known-item question —
/// has to go through here. A searcher that indexes extracted prose and then
/// quotes raw markup underneath the result is showing the reader something the
/// engine does not believe.
pub(crate) fn body_of(path: &Path, text: String) -> String {
    if is_html(path, &text) {
        indexander_crawl::extract::extract(&text).text
    } else {
        text
    }
}

/// Whether to run this file through the HTML extractor.
///
/// Indexing markup as prose is not merely untidy: on the Rust documentation,
/// 86% of the tokens in the raw files are tags and attributes, and the real
/// text is buried in a document six times longer than it is. Extracting it
/// first is worth 0.10 MRR on the prose books and 0.06 on the API docs.
///
/// The extension decides, with a sniff for the files that have none. Content
/// alone would be too eager: a document *about* HTML often opens with a tag.
fn is_html(path: &Path, text: &str) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(e) if e.eq_ignore_ascii_case("html") || e.eq_ignore_ascii_case("htm") => true,
        Some(_) => false,
        None => {
            // Characters, not bytes. Slicing at a byte offset here panics on
            // anything opening with an accent, which is most Spanish prose.
            let head: String = text
                .trim_start()
                .chars()
                .take(15)
                .collect::<String>()
                .to_lowercase();
            head.starts_with("<!doctype html") || head.starts_with("<html")
        }
    }
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
    let terms = parsed.scoring_terms();
    for (rank, hit) in hits.iter().enumerate() {
        println!("{:>3}. {:>8.4}  {}", rank + 1, hit.score, hit.uri);
        if let Some(line) = snippet_for(&hit.uri, &terms) {
            println!("     {line}");
        }
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

/// How much of a document to show under its result.
const SNIPPET_WIDTH: usize = 160;

/// The extract for one hit, or nothing if its text cannot be read back.
///
/// Snippets are not stored, so this re-reads the document. That works when the
/// uri is a path, which is what `indexander index` produces; a crawled `http://`
/// uri is not something to go and fetch again to decorate a result line, so
/// those results print without one rather than pretending.
fn snippet_for(uri: &str, terms: &[String]) -> Option<String> {
    let raw = std::fs::read(Path::new(uri)).ok()?;
    if is_binary(&raw) {
        return None;
    }
    let extract = snippet::best(&body_of(Path::new(uri), decode(raw)), terms, SNIPPET_WIDTH);
    if extract.text.is_empty() {
        return None;
    }
    // Bold, with an explicit reset - not a colour, which a third of readers
    // would see as the same grey as the rest.
    Some(
        extract
            .wrap("\u{1b}[1m", "\u{1b}[0m")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" "),
    )
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
    // Written into the segment, not compiled into this binary: a reader has to
    // be told which ones produced the bounds it is about to trust.
    let params = segment.params();
    println!("scoring k1         {:.3}", params.k1);
    println!("scoring b          {:.3}", params.b);
    println!("authority weight   {:.3}", params.authority_weight);
    Ok(())
}

pub(crate) fn open_segment(path: Option<&str>) -> Result<Segment, String> {
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
pub(crate) fn collect_files(root: &Path) -> Result<Vec<PathBuf>, String> {
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
pub(crate) fn decode(raw: Vec<u8>) -> String {
    match String::from_utf8(raw) {
        Ok(text) => text,
        Err(e) => e.into_bytes().into_iter().map(|b| b as char).collect(),
    }
}

/// A NUL byte in the first few KiB means this is not text.
pub(crate) fn is_binary(raw: &[u8]) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_is_recognised_by_extension() {
        assert!(is_html(Path::new("a/b.html"), ""));
        assert!(is_html(Path::new("a/b.HTM"), ""));
        assert!(!is_html(Path::new("a/b.rs"), "<!doctype html>"));
        assert!(!is_html(Path::new("a/b.md"), "<html>"));
    }

    #[test]
    fn a_file_without_an_extension_is_sniffed() {
        assert!(is_html(Path::new("page"), "<!DOCTYPE html>\n<html>"));
        assert!(is_html(Path::new("page"), "  <html lang=\"es\">"));
        assert!(!is_html(Path::new("notes"), "Un texto cualquiera"));
        assert!(!is_html(Path::new("empty"), ""));
    }

    #[test]
    fn sniffing_does_not_split_a_multibyte_character() {
        // The sniff looks at a fixed number of characters from the front. Byte
        // indexing there panics on any document that opens with an accent,
        // which is most Spanish prose.
        for text in ["ñ", "ñandú", "«Búsqueda»", "\u{1F44E}", "→→→→→→→→→→"]
        {
            assert!(!is_html(Path::new("notes"), text), "{text:?}");
        }
    }
}
