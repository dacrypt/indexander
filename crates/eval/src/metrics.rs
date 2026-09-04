//! The arithmetic of relevance.
//!
//! Each function takes a ranked list of URIs — best first, as a search engine
//! returns them — and the judgements for that one query, and produces a number
//! between 0 and 1. None of them look at scores: two rankers that order the
//! same documents the same way measure identically, however differently they
//! arrived there. That is deliberate. Scores are an implementation detail and
//! ordering is the product.
//!
//! ## The assumption that can make all of this a lie
//!
//! A document nobody judged is treated as not relevant. This is what TREC
//! does, and it is fine when the judgements were pooled from the systems being
//! compared, because then anything unjudged really was missed by everyone. It
//! is *not* fine when a new system surfaces good documents the pool never saw:
//! those get scored as garbage, and the new system looks worse than it is.
//!
//! So a number from here is only comparable against another number from the
//! same judgements. It is a regression detector and a knob-tuner. It is not a
//! claim about the world.

// Counts become fractions everywhere in this file. A corpus with more than
// 2^52 documents would round; a corpus that large has other problems.
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// What is relevant for one query, and how relevant.
///
/// Grade 0 means judged and not relevant, which is different from unjudged
/// only in that somebody looked. Both score as zero; the distinction matters
/// when counting how much of a run was covered by judgements at all.
#[derive(Debug, Clone, Default)]
pub struct Judged {
    grades: HashMap<String, u8>,
}

impl Judged {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a judgement. The last one for a URI wins.
    pub fn judge(&mut self, uri: impl Into<String>, grade: u8) {
        self.grades.insert(uri.into(), grade);
    }

    /// The grade of a URI; zero if it was never judged.
    #[must_use]
    pub fn grade(&self, uri: &str) -> u8 {
        self.grades.get(uri).copied().unwrap_or(0)
    }

    /// Whether anyone looked at this URI at all.
    #[must_use]
    pub fn was_judged(&self, uri: &str) -> bool {
        self.grades.contains_key(uri)
    }

    /// How many documents are relevant to any degree.
    ///
    /// This is the denominator of recall, and the reason recall is the metric
    /// that lies most readily: it is only the count of relevant documents
    /// *somebody judged*, never the count that exists.
    #[must_use]
    pub fn relevant_count(&self) -> usize {
        self.grades.values().filter(|&&g| g > 0).count()
    }

    /// Grades of every relevant document, best first. The ideal ranking.
    #[must_use]
    pub fn ideal(&self) -> Vec<u8> {
        let mut grades: Vec<u8> = self.grades.values().copied().filter(|&g| g > 0).collect();
        grades.sort_unstable_by(|a, b| b.cmp(a));
        grades
    }

    /// Whether anything at all is relevant.
    ///
    /// A query with no relevant document cannot be scored — every metric would
    /// be 0 out of 0 — and averaging such a query in drags every mean towards
    /// zero for a reason that has nothing to do with the ranker.
    #[must_use]
    pub fn is_scorable(&self) -> bool {
        self.relevant_count() > 0
    }
}

/// Fraction of the top `k` that are relevant.
///
/// The rank cutoff is what a person actually sees, so this is the metric
/// closest to their experience — and the one most sensitive to `k`. At `k`
/// larger than the number of relevant documents it cannot reach 1 no matter
/// how perfect the ranking is.
#[must_use]
pub fn precision_at(ranked: &[String], judged: &Judged, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let hits = ranked
        .iter()
        .take(k)
        .filter(|u| judged.grade(u) > 0)
        .count();
    hits as f64 / k as f64
}

/// Fraction of the relevant documents that made it into the top `k`.
#[must_use]
pub fn recall_at(ranked: &[String], judged: &Judged, k: usize) -> f64 {
    let total = judged.relevant_count();
    if total == 0 {
        return 0.0;
    }
    let hits = ranked
        .iter()
        .take(k)
        .filter(|u| judged.grade(u) > 0)
        .count();
    hits as f64 / total as f64
}

/// 1/rank of the first relevant result; 0 if there is none.
///
/// The right metric when there is one answer and the user stops as soon as
/// they see it — which is exactly the known-item case.
#[must_use]
pub fn reciprocal_rank(ranked: &[String], judged: &Judged) -> f64 {
    ranked
        .iter()
        .position(|u| judged.grade(u) > 0)
        .map_or(0.0, |i| 1.0 / (i + 1) as f64)
}

/// Whether anything relevant appears in the top `k`.
#[must_use]
pub fn success_at(ranked: &[String], judged: &Judged, k: usize) -> bool {
    ranked.iter().take(k).any(|u| judged.grade(u) > 0)
}

/// Mean of the precision measured at each rank where a relevant document sits.
///
/// Unlike precision@k this has no cutoff to argue about, and it rewards moving
/// a relevant document from rank 9 to rank 2 even when both are "in the top
/// ten". Averaged over queries it is the classic MAP.
#[must_use]
pub fn average_precision(ranked: &[String], judged: &Judged) -> f64 {
    let total = judged.relevant_count();
    if total == 0 {
        return 0.0;
    }
    let mut found = 0usize;
    let mut sum = 0.0;
    for (i, uri) in ranked.iter().enumerate() {
        if judged.grade(uri) > 0 {
            found += 1;
            sum += found as f64 / (i + 1) as f64;
        }
    }
    // Relevant documents the run never returned count as precision 0, which is
    // what dividing by the judged total rather than by `found` achieves.
    sum / total as f64
}

/// Discounted cumulative gain over the top `k`, normalised by the best
/// possible ranking of the same judgements.
///
/// Gain is `2^grade - 1`, so a grade-2 document is worth three grade-1s rather
/// than two — the convention that makes graded judgements worth collecting at
/// all. The discount is `1/log2(rank+1)`.
#[must_use]
pub fn ndcg_at(ranked: &[String], judged: &Judged, k: usize) -> f64 {
    let ideal = discounted_gain(judged.ideal().into_iter(), k);
    if ideal <= 0.0 {
        return 0.0;
    }
    discounted_gain(ranked.iter().take(k).map(|u| judged.grade(u)), k) / ideal
}

fn discounted_gain(grades: impl Iterator<Item = u8>, k: usize) -> f64 {
    grades
        .take(k)
        .enumerate()
        .map(|(i, g)| gain(g) / ((i + 2) as f64).log2())
        .sum()
}

fn gain(grade: u8) -> f64 {
    if grade == 0 {
        0.0
    } else {
        (f64::from(2u32.pow(u32::from(grade.min(31)))) - 1.0).max(0.0)
    }
}

/// Every metric for one query, so a run is measured in a single pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scores {
    pub precision: f64,
    pub recall: f64,
    pub reciprocal_rank: f64,
    pub average_precision: f64,
    pub ndcg: f64,
    pub success: bool,
    /// How many of the returned documents anybody had judged, at all. Low
    /// coverage is the warning that the numbers above are describing the
    /// judgements more than the ranker.
    pub judged_in_top_k: usize,
}

impl Scores {
    #[must_use]
    pub fn of(ranked: &[String], judged: &Judged, k: usize) -> Self {
        Self {
            precision: precision_at(ranked, judged, k),
            recall: recall_at(ranked, judged, k),
            reciprocal_rank: reciprocal_rank(ranked, judged),
            average_precision: average_precision(ranked, judged),
            ndcg: ndcg_at(ranked, judged, k),
            success: success_at(ranked, judged, k),
            judged_in_top_k: ranked
                .iter()
                .take(k)
                .filter(|u| judged.was_judged(u))
                .count(),
        }
    }
}

/// The mean of every metric over a set of queries.
///
/// Means over queries, not over documents: a query with a thousand relevant
/// documents gets the same vote as one with two. That is the standard choice
/// and the defensible one, because the unit a person cares about is the
/// search they did, not the corpus behind it.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    topics: usize,
    precision: f64,
    recall: f64,
    reciprocal_rank: f64,
    average_precision: f64,
    ndcg: f64,
    successes: usize,
    unjudged: usize,
}

impl Summary {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, scores: Scores, returned: usize) {
        self.topics += 1;
        self.precision += scores.precision;
        self.recall += scores.recall;
        self.reciprocal_rank += scores.reciprocal_rank;
        self.average_precision += scores.average_precision;
        self.ndcg += scores.ndcg;
        self.successes += usize::from(scores.success);
        self.unjudged += returned.saturating_sub(scores.judged_in_top_k);
    }

    #[must_use]
    pub fn topics(&self) -> usize {
        self.topics
    }

    /// Queries where nothing relevant came back in the top `k`. The number
    /// worth reading first: a mean hides a total failure, a count does not.
    #[must_use]
    pub fn failures(&self) -> usize {
        self.topics - self.successes
    }

    /// Documents returned in a top `k` that nobody had judged either way.
    #[must_use]
    pub fn unjudged(&self) -> usize {
        self.unjudged
    }

    #[must_use]
    pub fn mean_precision(&self) -> f64 {
        self.mean(self.precision)
    }
    #[must_use]
    pub fn mean_recall(&self) -> f64 {
        self.mean(self.recall)
    }
    /// Mean reciprocal rank.
    #[must_use]
    pub fn mrr(&self) -> f64 {
        self.mean(self.reciprocal_rank)
    }
    /// Mean average precision.
    #[must_use]
    pub fn map(&self) -> f64 {
        self.mean(self.average_precision)
    }
    #[must_use]
    pub fn mean_ndcg(&self) -> f64 {
        self.mean(self.ndcg)
    }
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        self.mean(self.successes as f64)
    }

    fn mean(&self, total: f64) -> f64 {
        if self.topics == 0 {
            0.0
        } else {
            total / self.topics as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked(uris: &[&str]) -> Vec<String> {
        uris.iter().map(|s| (*s).to_owned()).collect()
    }

    fn judged(pairs: &[(&str, u8)]) -> Judged {
        let mut j = Judged::new();
        for (uri, grade) in pairs {
            j.judge(*uri, *grade);
        }
        j
    }

    #[test]
    fn a_perfect_ranking_scores_one_everywhere() {
        let j = judged(&[("a", 1), ("b", 1), ("c", 0)]);
        let run = ranked(&["a", "b", "c"]);
        assert!((precision_at(&run, &j, 2) - 1.0).abs() < 1e-12);
        assert!((recall_at(&run, &j, 2) - 1.0).abs() < 1e-12);
        assert!((average_precision(&run, &j) - 1.0).abs() < 1e-12);
        assert!((ndcg_at(&run, &j, 3) - 1.0).abs() < 1e-12);
        assert!((reciprocal_rank(&run, &j) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_worst_ranking_of_the_same_documents_scores_worse() {
        let j = judged(&[("a", 1), ("b", 1)]);
        let good = Scores::of(&ranked(&["a", "b", "x", "y"]), &j, 4);
        let bad = Scores::of(&ranked(&["x", "y", "a", "b"]), &j, 4);
        // Same documents, same recall at 4 - only the order differs.
        assert!((good.recall - bad.recall).abs() < 1e-12);
        assert!(good.ndcg > bad.ndcg);
        assert!(good.average_precision > bad.average_precision);
        assert!(good.reciprocal_rank > bad.reciprocal_rank);
    }

    #[test]
    fn reciprocal_rank_is_one_over_the_first_hit() {
        let j = judged(&[("c", 1)]);
        assert!((reciprocal_rank(&ranked(&["a", "b", "c"]), &j) - 1.0 / 3.0).abs() < 1e-12);
        assert!(reciprocal_rank(&ranked(&["a", "b"]), &j).abs() < 1e-12);
    }

    #[test]
    fn a_grade_two_document_outweighs_two_grade_ones() {
        // gain(2) = 3, gain(1) = 1: putting the grade-2 first must win.
        let j = judged(&[("hi", 2), ("lo1", 1), ("lo2", 1)]);
        let best = ndcg_at(&ranked(&["hi", "lo1", "lo2"]), &j, 3);
        let worse = ndcg_at(&ranked(&["lo1", "lo2", "hi"]), &j, 3);
        assert!((best - 1.0).abs() < 1e-12);
        assert!(worse < best);
    }

    #[test]
    fn average_precision_counts_relevant_documents_never_returned() {
        // Two relevant, one returned: even at perfect rank 1, AP is 0.5.
        let j = judged(&[("a", 1), ("b", 1)]);
        let ap = average_precision(&ranked(&["a"]), &j);
        assert!((ap - 0.5).abs() < 1e-12, "{ap}");
    }

    #[test]
    fn unjudged_documents_score_as_zero_and_are_counted_as_such() {
        let j = judged(&[("a", 1)]);
        let s = Scores::of(&ranked(&["a", "mystery"]), &j, 2);
        assert!((s.precision - 0.5).abs() < 1e-12);
        assert_eq!(s.judged_in_top_k, 1);
    }

    #[test]
    fn a_query_with_nothing_relevant_is_not_scorable() {
        let j = judged(&[("a", 0)]);
        assert!(!j.is_scorable());
        assert_eq!(j.relevant_count(), 0);
        assert!(ndcg_at(&ranked(&["a"]), &j, 1).abs() < 1e-12);
    }

    #[test]
    fn an_empty_run_fails_every_metric_without_panicking() {
        let j = judged(&[("a", 1)]);
        let s = Scores::of(&[], &j, 10);
        assert!(s.precision.abs() < 1e-12);
        assert!(s.ndcg.abs() < 1e-12);
        assert!(!s.success);
    }

    #[test]
    fn a_summary_averages_over_queries_not_documents() {
        let mut s = Summary::new();
        s.add(Scores::of(&ranked(&["a"]), &judged(&[("a", 1)]), 1), 1);
        s.add(Scores::of(&ranked(&["x"]), &judged(&[("a", 1)]), 1), 1);
        assert_eq!(s.topics(), 2);
        assert_eq!(s.failures(), 1);
        assert!((s.mean_precision() - 0.5).abs() < 1e-12);
        assert!((s.success_rate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn precision_at_zero_is_zero_not_a_division_by_zero() {
        let j = judged(&[("a", 1)]);
        assert!(precision_at(&ranked(&["a"]), &j, 0).abs() < 1e-12);
    }

    #[test]
    fn ndcg_is_capped_at_one_even_when_the_run_is_longer_than_the_judgements() {
        let j = judged(&[("a", 1)]);
        let n = ndcg_at(&ranked(&["a", "b", "c", "d"]), &j, 4);
        assert!(n <= 1.0 + 1e-12, "{n}");
        assert!((n - 1.0).abs() < 1e-12);
    }
}
