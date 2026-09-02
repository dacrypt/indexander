// Byte counts become floats only to be printed as "1.4 MiB".
#![allow(clippy::cast_precision_loss)]

//! `indexander` — index a directory of text, then search it.
//!
//! Argument parsing is done by hand on purpose: the binary has two
//! subcommands and adding a dependency to parse them would be the first step
//! toward a startup time we would then have to defend.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use indexander_core::Document;
use indexander_index::builder::SegmentBuilder;
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;

const USAGE: &str = "\
indexander — a search engine

USAGE:
    indexander index <directory> [--out <segment>]
    indexander search <query>... [--index <segment>] [--limit <n>]
    indexander stats [--index <segment>]

The default segment path is ./indexander.ixdr

EXAMPLES:
    indexander index ./docs
    indexander search motor de busqueda
    indexander search '\"inverted index\"' -perl --limit 5
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

fn cmd_index(args: &[String]) -> Result<(), String> {
    let (out, rest) = take_option(args, "--out");
    let directory = rest
        .first()
        .ok_or_else(|| format!("index needs a directory\n\n{USAGE}"))?;
    let out = PathBuf::from(out.unwrap_or_else(|| DEFAULT_SEGMENT.to_owned()));

    let started = std::time::Instant::now();
    let mut builder = SegmentBuilder::new();
    let mut bytes_read = 0u64;

    for path in collect_files(Path::new(directory))? {
        let Ok(raw) = std::fs::read(&path) else {
            continue;
        };
        if is_binary(&raw) {
            continue;
        }
        let body = decode(raw);
        bytes_read += body.len() as u64;
        let title = path
            .file_stem()
            .map(|s| s.to_string_lossy().replace(['-', '_'], " "))
            .unwrap_or_default();
        builder.add(&Document::new(path.to_string_lossy(), title, body));
    }

    builder
        .write_to(&out)
        .map_err(|e| format!("writing {}: {e}", out.display()))?;

    let elapsed = started.elapsed();
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "indexed {} documents, {} terms in {:.2?}",
        builder.document_count(),
        builder.term_count(),
        elapsed
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

fn cmd_search(args: &[String]) -> Result<(), String> {
    let (index, rest) = take_option(args, "--index");
    let (limit, rest) = take_option(&rest, "--limit");
    let limit: usize = limit
        .as_deref()
        .map_or(Ok(10), str::parse)
        .map_err(|_| "--limit needs a number".to_owned())?;
    if rest.is_empty() {
        return Err(format!("search needs a query\n\n{USAGE}"));
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
