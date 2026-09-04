//! `indexander eval` and `indexander known-item`.
//!
//! Two ways to ask the only question the rest of the test suite never asks:
//! are the results any good?
//!
//! `eval` takes judgements somebody else made, in TREC format, and does the
//! arithmetic. `known-item` needs no judgements at all: it lifts a span of
//! words out of a document and checks where that document lands, which is a
//! question with exactly one right answer and no opinion in it.

use std::path::Path;

use indexander_core::DocId;
use indexander_eval::metrics::{Judged, Scores, Summary};
use indexander_eval::qrels::{Qrels, parse_topics};
use indexander_eval::sampling::{Rng, sample, span};
use indexander_eval::ties::{Ties, tie_group};
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;
use indexander_index::tokenizer;

use crate::{collect_files, decode, is_binary, open_segment, take_option};

/// How deep a run goes when nothing says otherwise.
///
/// Metrics are reported at `--k`, but reciprocal rank wants to know about a
/// document at rank 40, and truncating the run at 10 would score that the same
/// as never finding it at all.
const DEFAULT_DEPTH: usize = 100;
const DEFAULT_K: usize = 10;

pub fn cmd_eval(args: &[String]) -> Result<(), String> {
    let (index, rest) = take_option(args, "--index");
    let (queries, rest) = take_option(&rest, "--queries");
    let (judgements, rest) = take_option(&rest, "--qrels");
    let k = number(&rest, "--k", DEFAULT_K)?.0;
    let (depth, _) = number(&rest, "--depth", DEFAULT_DEPTH)?;

    let queries = queries.ok_or("eval needs --queries <file>")?;
    let judgements = judgements.ok_or("eval needs --qrels <file>")?;

    let topics = parse_topics(&read(&queries)?).map_err(|e| format!("{queries}: {e}"))?;
    let qrels = Qrels::parse(&read(&judgements)?).map_err(|e| format!("{judgements}: {e}"))?;
    let segment = open_segment(index.as_deref())?;

    let mut summary = Summary::new();
    let mut unscorable = 0usize;
    let started = std::time::Instant::now();
    for topic in &topics {
        let Some(judged) = qrels.get(&topic.id).filter(|j| j.is_scorable()) else {
            // A topic with no relevant document says nothing about the ranker,
            // and averaging it in as a zero would say something false.
            unscorable += 1;
            continue;
        };
        let ranked = run_query(&segment, &topic.text, depth)?;
        summary.add(Scores::of(&ranked, judged, k), ranked.len().min(k));
    }
    let elapsed = started.elapsed();

    if summary.topics() == 0 {
        return Err(format!(
            "none of the {} topics in {queries} had a scorable judgement in {judgements}",
            topics.len()
        ));
    }
    report(&summary, k, segment.document_count());
    println!(
        "\n{} topic{} in {elapsed:.2?}{}",
        summary.topics(),
        if summary.topics() == 1 { "" } else { "s" },
        if unscorable == 0 {
            String::new()
        } else {
            format!(", {unscorable} skipped for having nothing relevant")
        }
    );
    if summary.unjudged() > 0 {
        println!(
            "{} of the {} results shown were judged by nobody, and counted as irrelevant",
            summary.unjudged(),
            summary.topics() * k
        );
    }
    Ok(())
}

pub fn cmd_known_item(args: &[String]) -> Result<(), String> {
    let (index, rest) = take_option(args, "--index");
    let (from, rest) = take_option(&rest, "--from");
    let (k, rest) = number(&rest, "--k", DEFAULT_K)?;
    let (depth, rest) = number(&rest, "--depth", DEFAULT_DEPTH)?;
    let (want, rest) = number(&rest, "--sample", 200)?;
    let (length, rest) = number(&rest, "--span", 6)?;
    let (seed, rest) = number(&rest, "--seed", 1)?;
    let from = match from.as_deref() {
        None | Some("body") => Source::Body,
        Some("title") => Source::Title,
        Some(other) => return Err(format!("--from takes body or title, not {other:?}")),
    };
    let directory = rest
        .first()
        .ok_or("known-item needs the directory the index was built from")?;

    let segment = open_segment(index.as_deref())?;
    let files = collect_files(Path::new(directory))?;
    if files.is_empty() {
        return Err(format!("{directory} has no files"));
    }

    // Every URI the index holds, gathered once. The alternative is a linear
    // scan per sampled document, which is a hundred thousand comparisons to
    // answer a question about one file.
    let known: std::collections::HashSet<&str> = (0..segment.document_count())
        .filter_map(|i| segment.doc_uri(DocId(u32::try_from(i).unwrap_or(u32::MAX))))
        .collect();

    let mut rng = Rng::new(seed as u64);
    let mut summary = Summary::new();
    // The same runs measured at rank 1: with one right answer, "did it come
    // first" is the number a person would actually recognise.
    let mut top = Summary::new();
    let mut skipped = 0usize;
    let mut missing = 0usize;
    let mut beyond = 0usize;
    let mut ties = Ties::new();
    let started = std::time::Instant::now();

    for i in sample(files.len(), want, seed as u64) {
        let path = &files[i];
        let Some(terms) = terms_of(path, from) else {
            skipped += 1;
            continue;
        };
        let Some(question) = choose(&terms, from, length, &mut rng) else {
            skipped += 1;
            continue;
        };
        // The URI the builder stored, so the answer is comparable to the run.
        let answer = path.to_string_lossy().into_owned();
        if !known.contains(answer.as_str()) {
            // The index was built from a different tree, or this file was
            // skipped as binary when it was built. Either way the query has no
            // right answer and scoring it would only measure that mistake.
            missing += 1;
            continue;
        }
        let mut judged = Judged::new();
        judged.judge(&answer, 1);
        let scored = run_scored(&segment, &question.join(" "), depth)?;
        let ranked: Vec<String> = scored.iter().map(|(uri, _)| uri.clone()).collect();
        // Every term is required and the answer contains all of them, so the
        // answer is in the result set by construction. Not finding it means it
        // ranked below `depth` - a distinct failure from ranking badly inside
        // the top ten, and one a mean over ranks would hide.
        match ranked.iter().position(|uri| uri == &answer) {
            Some(at) => {
                let by_score: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();
                ties.add(tie_group(&by_score, at));
            }
            None => beyond += 1,
        }
        summary.add(Scores::of(&ranked, &judged, k), ranked.len().min(k));
        top.add(Scores::of(&ranked, &judged, 1), ranked.len().min(1));
    }
    let elapsed = started.elapsed();

    if summary.topics() == 0 {
        return Err(format!(
            "no query could be built from {directory}: {skipped} file{} too short or unreadable, \
             {missing} not in the index",
            if skipped == 1 { "" } else { "s" }
        ));
    }

    report_known_item(&Run {
        summary: &summary,
        top: &top,
        ties: &ties,
        from,
        length,
        seed,
        k,
        depth,
        documents: segment.document_count(),
        elapsed,
        skipped,
        missing,
        beyond,
    });
    Ok(())
}

/// Everything one known-item run produced, so the reporting is one function
/// rather than twenty lines wedged into the end of the command.
struct Run<'a> {
    summary: &'a Summary,
    top: &'a Summary,
    ties: &'a Ties,
    from: Source,
    length: usize,
    seed: usize,
    k: usize,
    depth: usize,
    documents: usize,
    elapsed: std::time::Duration,
    skipped: usize,
    missing: usize,
    beyond: usize,
}

fn report_known_item(run: &Run) {
    let Run {
        summary,
        top,
        ties,
        from,
        length,
        seed,
        k,
        depth,
        documents,
        elapsed,
        skipped,
        missing,
        beyond,
    } = *run;

    println!(
        "known-item: {} from the {} of each document, seed {seed}",
        match from {
            Source::Body => format!("{length}-word spans"),
            Source::Title => "whole titles".to_owned(),
        },
        match from {
            Source::Body => "body",
            Source::Title => "title",
        }
    );
    // Precision and MAP are not reported here. With exactly one relevant
    // document, precision@10 cannot exceed 0.1 whatever the ranker does, and
    // average precision is arithmetically identical to reciprocal rank. Both
    // would be numbers that look like measurements and are not.
    println!("over {documents} documents\n");
    println!("  MRR                      {:.4}", summary.mrr());
    println!("  nDCG@{k:<19} {:.4}", summary.mean_ndcg());
    println!("  success@1                {:.4}", top.success_rate());
    println!(
        "  success@{k:<16} {:.4}   ({} answer{} outside the top {k}, {beyond} past rank {depth})",
        summary.success_rate(),
        summary.failures(),
        plural(summary.failures()),
    );

    if ties.tied() > 0 {
        println!(
            "\n{} of them had the answer scoring exactly like {:.1} other documents on \
             average, up to {}",
            ties.tied(),
            ties.mean_group() - 1.0,
            ties.largest() - 1
        );
    }
    if ties.is_compromised() {
        println!(
            "the corpus contains duplicates: these numbers describe how many copies of a\n\
             document exist, not how well the engine ranks. Deduplicate it, or do not read\n\
             the figures above."
        );
    }
    println!(
        "\n{} quer{} in {elapsed:.2?}{}{}",
        summary.topics(),
        if summary.topics() == 1 { "y" } else { "ies" },
        if skipped == 0 {
            String::new()
        } else {
            format!(", {skipped} document{} too short to ask", plural(skipped))
        },
        if missing == 0 {
            String::new()
        } else {
            format!(", {missing} not in the index")
        }
    );
}

/// Where a known-item query comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// A span lifted from the document text. The honest one: the body is
    /// weighted 1, so nothing about the query flatters the document it came
    /// from beyond the fact that it contains the words.
    Body,
    /// The whole title. Easier, and knowingly so — titles are weighted 3 —
    /// but it is the shape of the query people actually type.
    Title,
}

fn report(summary: &Summary, k: usize, documents: usize) {
    println!("over {documents} documents\n");
    println!("  P@{k:<22} {:.4}", summary.mean_precision());
    println!("  nDCG@{k:<19} {:.4}", summary.mean_ndcg());
    println!("  MAP                      {:.4}", summary.map());
    println!("  MRR                      {:.4}", summary.mrr());
    println!(
        "  success@{k:<16} {:.4}   ({} quer{} found nothing)",
        summary.success_rate(),
        summary.failures(),
        if summary.failures() == 1 { "y" } else { "ies" }
    );
}

fn run_query(segment: &Segment, text: &str, depth: usize) -> Result<Vec<String>, String> {
    Ok(run_scored(segment, text, depth)?
        .into_iter()
        .map(|(uri, _)| uri)
        .collect())
}

/// The run, with the score that put each document where it is.
///
/// Metrics never look at scores. Tie detection has to: it is the only way to
/// tell a document the engine ranked from one it merely happened to list.
fn run_scored(segment: &Segment, text: &str, depth: usize) -> Result<Vec<(String, f32)>, String> {
    let parsed = query::parse(text);
    if parsed.is_empty() {
        return Ok(Vec::new());
    }
    Ok(search(segment, &parsed, depth)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|hit| (hit.uri, hit.score))
        .collect())
}

/// The terms of a document, tokenised exactly as the index tokenised them.
fn terms_of(path: &Path, from: Source) -> Option<Vec<String>> {
    match from {
        Source::Title => {
            let stem = path.file_stem()?.to_string_lossy().replace(['-', '_'], " ");
            Some(words(&tokenizer::fold(&stem)))
        }
        Source::Body => {
            let raw = std::fs::read(path).ok()?;
            if is_binary(&raw) {
                return None;
            }
            Some(
                tokenizer::tokenize(&decode(raw))
                    .into_iter()
                    .map(|t| t.text)
                    .collect(),
            )
        }
    }
}

fn choose(terms: &[String], from: Source, length: usize, rng: &mut Rng) -> Option<Vec<String>> {
    match from {
        Source::Title if terms.is_empty() => None,
        Source::Title => Some(terms.to_vec()),
        Source::Body => span(terms, length, rng),
    }
}

fn words(folded: &str) -> Vec<String> {
    folded
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn number(args: &[String], name: &str, default: usize) -> Result<(usize, Vec<String>), String> {
    let (value, rest) = take_option(args, name);
    let value = value
        .as_deref()
        .map_or(Ok(default), str::parse)
        .map_err(|_| format!("{name} needs a number"))?;
    Ok((value, rest))
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))
}
