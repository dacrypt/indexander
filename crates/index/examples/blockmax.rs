// An analysis tool, not a library: it mirrors search.rs's arithmetic on
// purpose, and its casts and its length are the same ones search.rs allows.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]

//! Does block-max scoring have anything left to save here?
//!
//! Block-max stores an upper bound on each block's contribution to a score, so
//! a query can skip a whole block once it can prove nothing in it reaches the
//! current top-k. It is the standard next step after skip lists — but it was
//! designed for *disjunctive* queries, where every posting of every term is a
//! candidate. This engine is conjunctive: a document must contain every term,
//! and the leapfrog already skips blocks that cannot contain the pivot.
//!
//! So the question is not "is block-max a good technique" — it is "how many of
//! the blocks this engine still decodes could a score bound remove?"
//!
//! This measures three things per query:
//!
//! 1. **Postings decoded**, against the total in the query's terms. What skip
//!    lists already save.
//! 2. **Blocks entered**, against the total. What is left to save.
//! 3. **Blocks a perfect score bound could have skipped**, computed with the
//!    *final* top-k threshold — which no real implementation could know from
//!    the start, so it is an upper bound on the benefit, not an estimate of it.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release -p indexander-index --example blockmax -- <segment> <query>...
//! ```

use std::collections::HashMap;

use indexander_core::{DocId, Result};
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;

/// BM25 constants, matching `search.rs`.
const K1: f32 = 1.2;
const B: f32 = 0.75;

fn idf(total_docs: usize, doc_freq: usize) -> f32 {
    let n = total_docs as f32;
    let df = doc_freq as f32;
    ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
}

/// One document's actual contribution for one term.
///
/// A real block-max index stores the maximum of this over each block, computed
/// at index time when both the frequency and the document's length are known.
/// Bounding with the corpus-wide shortest document instead would be correct
/// but far looser, and would make block-max look worse than it is.
fn contribution(term_idf: f32, weighted_tf: f32, length_norm: f32) -> f32 {
    term_idf * (weighted_tf * (K1 + 1.0)) / (weighted_tf + K1 * length_norm)
}

struct TermProfile {
    term: String,
    doc_freq: usize,
    blocks: usize,
    /// Highest weighted term frequency in each block.
    block_max: Vec<f32>,
}

/// Walks a term's whole postings list, recording the maximum *score
/// contribution* per block. This is what a block-max index would store.
fn profile(segment: &Segment, term: &str, total_docs: usize, average: f32) -> Result<TermProfile> {
    let mut cursor = segment.cursor(term, false)?;
    let blocks = cursor.block_count();
    let term_idf = idf(total_docs, cursor.document_frequency());
    let mut block_max = vec![0.0f32; blocks];

    while let Some(doc) = cursor.doc() {
        let slot = cursor.current_block();
        if let Some(meta) = segment.doc(doc) {
            let length_norm = 1.0 - B + B * (meta.total_length() as f32 / average);
            let score = contribution(term_idf, cursor.weighted_frequency(), length_norm);
            if slot < block_max.len() {
                block_max[slot] = block_max[slot].max(score);
            }
        }
        cursor.advance()?;
    }
    Ok(TermProfile {
        term: term.to_owned(),
        doc_freq: cursor.document_frequency(),
        blocks,
        block_max,
    })
}

/// What the leapfrog actually did.
struct Observation {
    /// Postings decoded, per term.
    decoded: HashMap<String, usize>,
    jumps: usize,
    /// Documents matching every term.
    matches: usize,
    /// How many times the cursors aligned on a pivot and had to be examined.
    pivots: usize,
    /// How many a bound could skip knowing the final threshold in advance —
    /// the ceiling, which no implementation can reach.
    skippable_pivots: usize,
    /// How many the real algorithm would skip, with the threshold starting at
    /// zero and rising as documents are scored.
    realistic_pivots: usize,
    /// Distinct blocks entered, per term.
    blocks_entered: HashMap<String, usize>,
}

/// Re-runs the leapfrog, recording what a block bound would have let it skip.
///
/// At every pivot the cursors sit in some block of each term's postings. A
/// block-max index would let the query add up those blocks' upper bounds and,
/// if the sum falls below the current top-k threshold, skip the whole aligned
/// range without decoding it. This counts how often that would have fired,
/// using the *final* threshold — which the real thing could not know at the
/// start, so this overstates the benefit.
fn observe(
    segment: &Segment,
    terms: &[String],
    profiles: &[TermProfile],
    threshold: f32,
    limit: usize,
    total_docs: usize,
    average: f32,
) -> Result<Observation> {
    // The stored bounds are already score contributions.
    let bounds: HashMap<&str, &[f32]> = profiles
        .iter()
        .map(|p| (p.term.as_str(), p.block_max.as_slice()))
        .collect();

    let mut observation = Observation {
        decoded: HashMap::new(),
        jumps: 0,
        matches: 0,
        pivots: 0,
        skippable_pivots: 0,
        realistic_pivots: 0,
        blocks_entered: HashMap::new(),
    };

    let mut cursors = Vec::new();
    for term in terms {
        let cursor = segment.cursor(term, false)?;
        if cursor.doc().is_none() {
            return Ok(observation);
        }
        cursors.push((term.clone(), cursor));
    }
    cursors.sort_by_key(|(_, c)| c.document_frequency());

    let mut seen_blocks: HashMap<String, std::collections::HashSet<usize>> = HashMap::new();

    // The running top-k, as the real algorithm would keep it: a threshold of
    // zero until `limit` documents have been scored, then the k-th best.
    let mut best: Vec<f32> = Vec::with_capacity(limit + 1);
    let mut running_threshold = 0.0f32;
    let idfs: HashMap<&str, f32> = profiles
        .iter()
        .map(|p| (p.term.as_str(), idf(total_docs, p.doc_freq)))
        .collect();

    while let Some(mut pivot) = cursors[0].1.doc() {
        let mut aligned = true;
        for (_, cursor) in &mut cursors[1..] {
            cursor.seek(pivot)?;
            match cursor.doc() {
                None => {
                    aligned = false;
                    pivot = DocId(u32::MAX);
                    break;
                }
                Some(doc) if doc != pivot => {
                    pivot = doc;
                    aligned = false;
                    break;
                }
                Some(_) => {}
            }
        }
        if pivot == DocId(u32::MAX) {
            break;
        }
        if !aligned {
            cursors[0].1.seek(pivot)?;
            continue;
        }

        observation.pivots += 1;
        observation.matches += 1;

        // Would a block bound have let us skip this pivot without scoring it?
        let bound: f32 = cursors
            .iter()
            .map(|(term, cursor)| {
                bounds
                    .get(term.as_str())
                    .and_then(|b| b.get(cursor.current_block()))
                    .copied()
                    .unwrap_or(f32::INFINITY)
            })
            .sum();
        if bound < threshold {
            observation.skippable_pivots += 1;
        }

        // The same decision, made with only what the query knows so far.
        if bound < running_threshold {
            observation.realistic_pivots += 1;
        } else if let Some(meta) = segment.doc(pivot) {
            let length_norm = 1.0 - B + B * (meta.total_length() as f32 / average);
            let score: f32 = cursors
                .iter()
                .map(|(term, cursor)| {
                    contribution(
                        idfs.get(term.as_str()).copied().unwrap_or(0.0),
                        cursor.weighted_frequency(),
                        length_norm,
                    )
                })
                .sum();
            best.push(score);
            best.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            best.truncate(limit);
            if best.len() == limit {
                running_threshold = best[limit - 1];
            }
        }

        for (term, cursor) in &cursors {
            seen_blocks
                .entry(term.clone())
                .or_default()
                .insert(cursor.current_block());
        }

        cursors[0].1.advance()?;
    }

    for (term, cursor) in &cursors {
        observation.decoded.insert(term.clone(), cursor.decoded());
        observation.jumps += cursor.jumps();
    }
    for (term, blocks) in seen_blocks {
        observation.blocks_entered.insert(term, blocks.len());
    }
    Ok(observation)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "indexander.ixdr".to_owned());
    let queries: Vec<String> = args.collect();
    let queries = if queries.is_empty() {
        vec!["the".to_owned()]
    } else {
        queries
    };

    let segment = Segment::open(std::path::Path::new(&path))?;
    let total_docs = segment.document_count();
    let average = segment.average_document_length().max(1.0);

    println!("segment: {total_docs} documents, average {average:.0} tokens\n");

    for text in &queries {
        let parsed = query::parse(text);
        let terms = parsed.scoring_terms();
        if terms.is_empty() {
            continue;
        }

        let profiles: Vec<TermProfile> = terms
            .iter()
            .map(|t| profile(&segment, t, total_docs, average))
            .collect::<Result<_>>()?;

        // The threshold a real query ends up with.
        let hits = search(&segment, &parsed, 10)?;
        let threshold = hits.last().map_or(0.0, |h| h.score);

        let observed = observe(
            &segment, &terms, &profiles, threshold, 10, total_docs, average,
        )?;

        let total_postings: usize = profiles.iter().map(|p| p.doc_freq).sum();
        let total_blocks: usize = profiles.iter().map(|p| p.blocks).sum();
        let decoded: usize = observed.decoded.values().sum();
        let blocks_entered: usize = observed.blocks_entered.values().sum();

        println!("query {text:?}");
        for p in &profiles {
            println!(
                "  term {:<22} df {:>7}  blocks {:>5}  entered {:>5}",
                p.term,
                p.doc_freq,
                p.blocks,
                observed.blocks_entered.get(&p.term).copied().unwrap_or(0)
            );
        }
        println!("  matches                     {:>7}", observed.matches);
        println!(
            "  postings decoded            {decoded:>7} of {total_postings}  ({:.1}%)",
            decoded as f64 * 100.0 / total_postings.max(1) as f64
        );
        println!(
            "  blocks entered              {blocks_entered:>7} of {total_blocks}  ({:.1}%), {} jumps",
            blocks_entered as f64 * 100.0 / total_blocks.max(1) as f64,
            observed.jumps
        );
        println!("  top-10 threshold            {threshold:>7.4}");
        println!(
            "  block-max, realistic        {:>7} of {} pivots skipped  ({:.1}%)",
            observed.realistic_pivots,
            observed.pivots,
            observed.realistic_pivots as f64 * 100.0 / observed.pivots.max(1) as f64
        );
        println!(
            "  block-max, ceiling          {:>7} of {} pivots skipped  ({:.1}%)",
            observed.skippable_pivots,
            observed.pivots,
            observed.skippable_pivots as f64 * 100.0 / observed.pivots.max(1) as f64
        );
        println!();
    }
    Ok(())
}
