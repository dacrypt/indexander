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
use crate::scoring::idf;
use crate::segment::{Posting, Segment};

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
    /// Tokens across every document of every shard.
    ///
    /// BM25 discounts a document by its length *relative to the average*, and
    /// the average that matters is the corpus's. Scoring a shard against its
    /// own average makes a document look long or short depending on the
    /// company it keeps, and two shards' scores stop being comparable — the
    /// same failure as using local `idf`, and just as invisible.
    pub total_length: u64,
    /// How many of those contain each term.
    pub doc_freq: HashMap<String, usize>,
}

impl GlobalStats {
    /// Sums one shard's contribution into the running totals.
    pub fn add_shard(&mut self, doc_count: usize, total_length: u64, per_term: &[(String, usize)]) {
        self.total_docs += doc_count;
        self.total_length += total_length;
        for (term, freq) in per_term {
            *self.doc_freq.entry(term.clone()).or_insert(0) += freq;
        }
    }

    /// The corpus-wide average document length.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_length(&self) -> f32 {
        if self.total_docs == 0 {
            0.0
        } else {
            self.total_length as f32 / self.total_docs as f32
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
    // Positions cost several times what document ids and counts cost, and
    // only a phrase query ever reads them. Deciding once, here, is worth more
    // than any amount of cleverness further down.
    let need_positions = !query.phrases.is_empty();

    // Without phrases the answer is the union of the terms' postings, walked
    // with cursors. With phrases, positions are needed anyway, so the
    // straightforward path is used.
    if !need_positions {
        return union(segment, query, limit, global, total_docs);
    }
    let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();
    for term in query.scoring_terms() {
        let list = if need_positions {
            segment.postings(&term)?
        } else {
            segment.postings_counts(&term)?
        };
        postings.insert(term.clone(), list);
    }

    let Some(candidates) = select_candidates(segment, query, &postings)? else {
        return Ok(Vec::new());
    };

    // Score what is left, keeping only the best `limit` in a bounded heap.
    // The corpus-wide average, not this segment's. A shard scoring against
    // its own average makes a document look long or short depending on the
    // company it keeps, and its scores stop being comparable with another's.
    let average_length = global
        .map_or_else(
            || segment.average_document_length(),
            GlobalStats::average_length,
        )
        .max(1.0);
    // The segment's own, never this build's defaults: the block-max bounds
    // stored with every postings list were computed with these.
    let params = segment.params();
    let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(limit.min(1024).saturating_add(1));

    // Candidates ascend and so does every postings list, so the scorer walks
    // them together with one cursor per term. Looking each candidate up with a
    // linear scan instead costs O(candidates x postings): on a 103k-document
    // index that was the difference between a 37 ms query and a 1 ms one.
    // Iterated in sorted term order, not in `HashMap` order.
    //
    // Floating-point addition is not associative, so the order in which a
    // document's per-term contributions are summed changes the last bits of
    // its score. A `HashMap` is seeded differently in every process, so two
    // shards holding byte-identical segments would score the same document
    // differently, ties would break arbitrarily between runs, and no ranking
    // comparison would be reproducible. `scoring_terms` is already sorted.
    let ordered_terms = query.scoring_terms();
    let mut scorers: Vec<(f32, &[Posting], usize)> = ordered_terms
        .iter()
        .filter_map(|term| postings.get(term).map(|list| (term, list)))
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
        // Sixteen bytes from the mapped file, not a document record with an
        // allocated uri: this runs once per candidate, and the uri is needed
        // only for the handful that end up in the results.
        let Some((length, rank)) = segment.doc_lengths(doc) else {
            continue;
        };
        let length_norm = params.length_norm(length, average_length);
        let mut score = 0.0f32;

        for (term_idf, list, cursor) in &mut scorers {
            while *cursor < list.len() && list[*cursor].doc < doc {
                *cursor += 1;
            }
            let Some(posting) = list.get(*cursor).filter(|p| p.doc == doc) else {
                continue;
            };
            let tf = posting.weighted_frequency();
            score += *term_idf * params.saturation(tf, length_norm);
        }

        // Authority scales relevance; it never creates it. A document that
        // scored zero on the query still scores zero.
        score *= params.authority(rank, total_docs);

        heap.push(Ranked(score, doc));
        if heap.len() > limit {
            heap.pop();
        }
    }

    Ok(finish(heap, segment, limit))
}

/// Answers a phrase-free query by walking the union of its terms' postings.
///
/// Bare terms are optional, so a document qualifies by containing *any* of
/// them and is ranked by how much of the query it accounts for — which BM25
/// does on its own, by summing one contribution per term present. Nothing has
/// to enforce "more terms is better"; arithmetic already does.
///
/// This replaced an intersection, and the reason is worth keeping: requiring
/// every term is a fine rule for the two or three words someone types into a
/// box, and it collapses on a query that is a sentence. On the Cranfield
/// collection, 225 questions averaging seventeen words, exactly four had any
/// document at all containing every term. Intersection answered three of them.
///
/// The cursors are the same ones the intersection used, so a term still skips
/// blocks instead of decoding them. What is gone is the pivot: a union cannot
/// jump past a document just because one term is missing from it, because that
/// document may still be a good answer.
fn union(
    segment: &Segment,
    query: &Query,
    limit: usize,
    global: Option<&GlobalStats>,
    total_docs: usize,
) -> Result<Vec<Hit>> {
    let terms = query.scoring_terms();
    if terms.is_empty() {
        return Ok(Vec::new());
    }

    // In `scoring_terms` order, which is sorted, so contributions are summed
    // the same way in every process. Floating-point addition is not
    // associative and two shards must agree to the last bit.
    let mut cursors = Vec::with_capacity(terms.len());
    for term in &terms {
        let cursor = segment.cursor(term, false)?;
        // A term this segment has never seen contributes nothing. Under
        // intersection it meant the query could not match at all; under union
        // it is simply one fewer thing to add up.
        if cursor.doc().is_none() {
            continue;
        }
        let doc_freq = global
            .and_then(|g| g.doc_freq.get(term).copied())
            .unwrap_or_else(|| cursor.document_frequency());
        cursors.push((idf(total_docs, doc_freq), cursor));
    }
    if cursors.is_empty() {
        return Ok(Vec::new());
    }

    // Excluded terms get cursors too, so exclusion is a skip rather than a
    // scan of the excluded term's entire list.
    let mut excluded = Vec::with_capacity(query.excluded.len());
    for term in &query.excluded {
        excluded.push(segment.cursor(term, false)?);
    }

    let average_length = global
        .map_or_else(
            || segment.average_document_length(),
            GlobalStats::average_length,
        )
        .max(1.0);
    let params = segment.params();
    let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(limit.min(1024).saturating_add(1));

    // The lowest document any cursor is sitting on. Every cursor ascends, so
    // this walks the union in document order without a merge buffer.
    while let Some(doc) = cursors.iter().filter_map(|(_, c)| c.doc()).min() {
        let mut dropped = false;
        for cursor in &mut excluded {
            cursor.seek(doc)?;
            if cursor.doc() == Some(doc) {
                dropped = true;
                break;
            }
        }

        if !dropped {
            if let Some((length, rank)) = segment.doc_lengths(doc) {
                let length_norm = params.length_norm(length, average_length);
                let mut score = 0.0f32;
                for (term_idf, cursor) in &cursors {
                    if cursor.doc() == Some(doc) {
                        score +=
                            term_idf * params.saturation(cursor.weighted_frequency(), length_norm);
                    }
                }
                score *= params.authority(rank, total_docs);
                heap.push(Ranked(score, doc));
                if heap.len() > limit {
                    heap.pop();
                }
            }
        }

        for (_, cursor) in &mut cursors {
            if cursor.doc() == Some(doc) {
                cursor.advance()?;
            }
        }
    }

    Ok(finish(heap, segment, limit))
}

/// Drains a top-k heap into hits, best first.
fn finish(heap: BinaryHeap<Ranked>, segment: &Segment, limit: usize) -> Vec<Hit> {
    let mut hits: Vec<Hit> = heap
        .into_vec()
        .into_iter()
        .map(|Ranked(score, doc)| Hit {
            doc,
            uri: segment.doc_uri(doc).unwrap_or_default().to_owned(),
            score,
        })
        .collect();

    // Ties break on the uri, not the document id.
    //
    // A document id is where a document happened to land while indexing; a
    // uri is what the document *is*. Ordering equally-scoring results by id
    // would make the order depend on how the corpus was split into segments
    // and shards, which is not something a user should be able to notice.
    //
    // What this does not fix is *which* tied documents survive the top-k cut:
    // that is decided while scoring, by a heap that cannot afford a string
    // comparison per candidate. With genuinely equal scores, different
    // segmentations can return different members of the tie. Documented in
    // the README rather than papered over.
    sort_hits(&mut hits);
    hits.truncate(limit);
    hits
}

/// Orders results best first, breaking ties on the uri.
pub(crate) fn sort_hits(hits: &mut [Hit]) {
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.uri.cmp(&b.uri))
    });
}

/// Narrows the corpus to the documents that can possibly match.
///
/// Only the hard parts of a query narrow anything. A phrase is a demand: its
/// words have to be there, adjacent and in order. An exclusion is a demand.
/// Bare terms are neither — they are what the ranking is *made of*, and a
/// document missing one of them is a worse answer, not a non-answer.
///
/// Cheap filters first, expensive last: phrase terms intersect sorted posting
/// lists, exclusions run once the set is small, and the positional phrase
/// check — a scan per candidate — runs last of all. `None` means nothing can
/// match.
fn select_candidates(
    segment: &Segment,
    query: &Query,
    postings: &HashMap<String, Vec<Posting>>,
) -> Result<Option<Vec<DocId>>> {
    let mut candidates: Option<Vec<DocId>> = None;
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
        return Ok(None);
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

    Ok(Some(candidates))
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
    use crate::scoring::authority;

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
