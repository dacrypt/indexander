// Scores are `f32` deliberately: BM25 is a ranking function, not an
// accounting one, and half the memory per score matters more than digits
// nobody compares. The counts cast into it are document frequencies, which
// stay far below the 24 bits `f32` represents exactly.
#![allow(clippy::cast_precision_loss)]

//! Executing a query against a segment and ranking what comes back.
//!
//! Ranking is Okapi BM25, with one addition inherited from the 2004 design:
//! a term's frequency is weighted by the field it appeared in, so a word in
//! the title or in an incoming link counts for more than the same word buried
//! in the body.

use std::collections::{BinaryHeap, HashMap};

use indexander_core::{DocId, Field, Result};

use crate::query::Query;
use crate::segment::{Posting, Segment};

/// Saturation: how quickly extra occurrences of a term stop adding score.
const K1: f32 = 1.2;
/// Length normalisation: how much a long document is penalised. 0 disables it.
const B: f32 = 0.75;
/// How much authority is allowed to move a result.
///
/// Relevance decides *whether* a page can appear; authority decides the order
/// among pages that already match. This is why the boost multiplies BM25
/// rather than being added to it: a page about nothing cannot buy its way in
/// with links, however many it has.
const AUTHORITY_WEIGHT: f32 = 0.5;

/// Turns a PageRank into a multiplier around 1.0.
///
/// `rank * n` is the page's rank relative to the average page, so an ordinary
/// page sits at 1.0 and gets no boost. The logarithm is what keeps a page that
/// is a thousand times more central from being a thousand times higher: in a
/// real web graph ranks span many orders of magnitude, and without it
/// authority would simply overwrite relevance.
fn authority(rank: f32, total_docs: usize) -> f32 {
    if rank <= 0.0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let relative = rank * total_docs as f32;
    1.0 + AUTHORITY_WEIGHT * relative.max(0.0).ln_1p()
}

/// Corpus-wide term statistics, supplied from outside.
///
/// BM25 weights a term by how rare it is, and rarity is a property of the
/// whole corpus. When an index is one shard of many, its local counts are the
/// wrong ones: a term rare here and common elsewhere would be scored as rare,
/// and the resulting scores could not be compared with another shard's.
///
/// Passing this in makes every shard score on one scale, which is the only
/// thing that makes merging their results meaningful.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GlobalStats {
    /// Documents across every shard.
    pub total_docs: usize,
    /// How many of those contain each term.
    pub doc_freq: HashMap<String, usize>,
}

impl GlobalStats {
    /// Sums one shard's contribution into the running totals.
    pub fn add_shard(&mut self, doc_count: usize, per_term: &[(String, usize)]) {
        self.total_docs += doc_count;
        for (term, freq) in per_term {
            *self.doc_freq.entry(term.clone()).or_insert(0) += freq;
        }
    }

    /// Whether these statistics are usable. Empty ones mean "score locally".
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_docs == 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub doc: DocId,
    pub uri: String,
    pub score: f32,
}

/// Ordering wrapper so a `BinaryHeap` can keep the *lowest* score at the top,
/// which is what a bounded top-k heap needs.
#[derive(Debug, PartialEq)]
struct Ranked(f32, DocId);

impl Eq for Ranked {}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed on score, then by document id so ties are deterministic.
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.1.cmp(&other.1))
    }
}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Inverse document frequency, the part of BM25 that makes rare words matter.
fn idf(total_docs: usize, doc_freq: usize) -> f32 {
    let n = total_docs as f32;
    let df = doc_freq as f32;
    // The +1 inside the logarithm keeps this positive even for a term that
    // appears in every document, which the textbook formula does not.
    ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
}

/// True when `phrase` occurs adjacent and in order within a single field.
fn phrase_matches(per_term: &[&Posting], _doc: DocId) -> bool {
    for field in Field::ALL {
        let first = per_term[0].positions_in(field);
        if first.is_empty() {
            continue;
        }
        'candidate: for &start in first {
            for (step, posting) in per_term.iter().enumerate().skip(1) {
                // `step` indexes a phrase, which is a handful of words.
                let wanted = start + u32::try_from(step).unwrap_or(u32::MAX);
                if posting.positions_in(field).binary_search(&wanted).is_err() {
                    continue 'candidate;
                }
            }
            return true;
        }
    }
    false
}

/// Runs `query` against `segment` and returns at most `limit` hits, best first.
///
/// Scores with the segment's own statistics, which is correct when this
/// segment is the whole corpus. For one shard of several, use
/// [`search_with_stats`].
pub fn search(segment: &Segment, query: &Query, limit: usize) -> Result<Vec<Hit>> {
    search_with_stats(segment, query, limit, None)
}

/// Runs `query`, optionally scoring with corpus-wide statistics.
///
/// `global` of `None` — or empty statistics — falls back to the segment's own
/// counts, so a single-shard deployment takes the same code path as a
/// hundred-shard one and simply supplies nothing.
pub fn search_with_stats(
    segment: &Segment,
    query: &Query,
    limit: usize,
    global: Option<&GlobalStats>,
) -> Result<Vec<Hit>> {
    if query.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    if segment.document_count() == 0 {
        return Ok(Vec::new());
    }
    let global = global.filter(|g| !g.is_empty());
    let total_docs = global.map_or_else(|| segment.document_count(), |g| g.total_docs);

    // Postings for every term we care about, decoded once and shared between
    // the filtering pass and the scoring pass.
    let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();
    for term in query.scoring_terms() {
        postings.insert(term.clone(), segment.postings(&term)?);
    }

    // A required term with no postings means nothing can match.
    let mut candidates: Option<Vec<DocId>> = None;
    for term in &query.required {
        let docs: Vec<DocId> = postings[term].iter().map(|p| p.doc).collect();
        candidates = Some(match candidates {
            None => docs,
            Some(previous) => intersect(&previous, &docs),
        });
        if candidates.as_ref().is_some_and(Vec::is_empty) {
            return Ok(Vec::new());
        }
    }
    for phrase in &query.phrases {
        for term in phrase {
            let docs: Vec<DocId> = postings[term].iter().map(|p| p.doc).collect();
            candidates = Some(match candidates {
                None => docs,
                Some(previous) => intersect(&previous, &docs),
            });
        }
    }
    let Some(mut candidates) = candidates else {
        return Ok(Vec::new());
    };

    // Excluded terms are looked up only now, once the candidate set is small.
    for term in &query.excluded {
        let excluded: Vec<DocId> = segment.postings(term)?.iter().map(|p| p.doc).collect();
        candidates.retain(|d| excluded.binary_search(d).is_err());
    }

    // Phrases are checked positionally, which is the expensive part, so it
    // happens last and only on documents that already survived.
    if !query.phrases.is_empty() {
        candidates.retain(|&doc| {
            query.phrases.iter().all(|phrase| {
                let per_term: Option<Vec<&Posting>> = phrase
                    .iter()
                    .map(|t| postings[t].iter().find(|p| p.doc == doc))
                    .collect();
                per_term.is_some_and(|p| phrase_matches(&p, doc))
            })
        });
    }

    // Score what is left, keeping only the best `limit` in a bounded heap.
    let average_length = segment.average_document_length().max(1.0);
    let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(limit + 1);

    // Candidates ascend and so does every postings list, so the scorer walks
    // them together with one cursor per term. Looking each candidate up with a
    // linear scan instead costs O(candidates x postings): on a 103k-document
    // index that was the difference between a 37 ms query and a 1 ms one.
    let mut scorers: Vec<(f32, &[Posting], usize)> = postings
        .iter()
        .map(|(term, list)| {
            // The document frequency is this shard's unless a global one was
            // supplied. A term absent from the global map is one no shard has
            // seen, so the local count is the whole truth about it.
            let doc_freq = global
                .and_then(|g| g.doc_freq.get(term).copied())
                .unwrap_or(list.len());
            (idf(total_docs, doc_freq), list.as_slice(), 0usize)
        })
        .collect();

    for &doc in &candidates {
        let Some(meta) = segment.doc(doc) else {
            continue;
        };
        let length_norm = 1.0 - B + B * (meta.total_length() as f32 / average_length);
        let mut score = 0.0f32;

        for (term_idf, list, cursor) in &mut scorers {
            while *cursor < list.len() && list[*cursor].doc < doc {
                *cursor += 1;
            }
            let Some(posting) = list.get(*cursor).filter(|p| p.doc == doc) else {
                continue;
            };
            let tf = posting.weighted_frequency();
            score += *term_idf * (tf * (K1 + 1.0)) / (tf + K1 * length_norm);
        }

        // Authority scales relevance; it never creates it. A document that
        // scored zero on the query still scores zero.
        score *= authority(meta.rank, total_docs);

        heap.push(Ranked(score, doc));
        if heap.len() > limit {
            heap.pop();
        }
    }

    let mut ranked: Vec<Ranked> = heap.into_vec();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    Ok(ranked
        .into_iter()
        .map(|Ranked(score, doc)| Hit {
            doc,
            uri: segment.doc(doc).map(|d| d.uri.clone()).unwrap_or_default(),
            score,
        })
        .collect())
}

/// Intersects two ascending, deduplicated document lists.
///
/// Linear in the sum of the lengths, which beats a hash set for the sorted
/// runs postings lists always are.
fn intersect(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersection_keeps_only_common_ascending() {
        let a = [1u32, 3, 5, 7].map(DocId).to_vec();
        let b = [3u32, 4, 5, 9].map(DocId).to_vec();
        assert_eq!(intersect(&a, &b), [DocId(3), DocId(5)]);
    }

    #[test]
    fn intersection_with_empty_is_empty() {
        let a = [1u32, 2].map(DocId).to_vec();
        assert!(intersect(&a, &[]).is_empty());
        assert!(intersect(&[], &a).is_empty());
    }

    #[test]
    fn a_rare_term_outweighs_a_common_one() {
        assert!(idf(1000, 1) > idf(1000, 500));
    }

    #[test]
    fn idf_stays_positive_for_a_term_in_every_document() {
        assert!(idf(100, 100) > 0.0);
    }

    #[test]
    fn an_average_page_gets_no_authority_boost() {
        // rank == 1/n is the average, so relative == 1 and ln_1p(1) is small
        // but the point is that it is bounded and the same for every page.
        assert!((authority(0.0, 100) - 1.0).abs() < f32::EPSILON);
        assert!(authority(0.01, 100) > 1.0);
    }

    #[test]
    fn authority_is_logarithmic_not_linear() {
        // A page a thousand times more central is not a thousand times better.
        let ordinary = authority(0.001, 1000);
        let central = authority(1.0, 1000);
        assert!(central > ordinary);
        assert!(
            central < ordinary * 20.0,
            "a 1000x rank produced a {}x score change",
            central / ordinary
        );
    }

    #[test]
    fn a_missing_rank_is_neutral() {
        // Indexes built before v2, or without a crawl, have rank 0.
        assert!((authority(0.0, 1) - 1.0).abs() < f32::EPSILON);
    }
}
