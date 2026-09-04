//! Detecting when the question had no single right answer.
//!
//! [`metrics`](crate::metrics) ignores scores on purpose: ordering is the
//! product and two rankers that order the same way are the same ranker. This
//! module is the one exception, and it exists because of a measurement that
//! was wrong without being obviously wrong.
//!
//! A known-item query assumes the document its words came from is the only
//! document that could be the answer. On a corpus containing forty copies of
//! that document, it is one of forty, ranking among them is arbitrary, and the
//! resulting MRR — around `H(40)/40`, or 0.11 — describes the duplication and
//! nothing else. It is a small number that looks exactly like a bad ranker.
//!
//! Exactly equal scores are what copies produce: same term frequencies, same
//! length, same arithmetic. So an exact comparison, which would be the wrong
//! tool for almost anything else involving floats, is the right one here.
//! Near-duplicates that differ by a word score slightly differently and will
//! not be caught, which is correct — they are different documents.

/// How many entries share the score at `index`, counting that one.
///
/// 1 means the score is unique and the ranking at that position is a decision
/// the engine actually made.
///
/// The exact float comparison is the point, not an oversight: copies produce
/// bit-identical scores, and any margin of error would start folding merely
/// similar documents into the same group.
#[allow(clippy::float_cmp)]
#[must_use]
pub fn tie_group(scores: &[f32], index: usize) -> usize {
    let Some(&target) = scores.get(index) else {
        return 0;
    };
    scores.iter().filter(|&&s| s == target).count()
}

/// How often a run had no single right answer to find.
#[derive(Debug, Clone, Default)]
pub struct Ties {
    queries: usize,
    tied: usize,
    largest: usize,
    total_group: usize,
}

impl Ties {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one query, given the size of the answer's tie group.
    pub fn add(&mut self, group: usize) {
        self.queries += 1;
        if group > 1 {
            self.tied += 1;
            self.total_group += group;
            self.largest = self.largest.max(group);
        }
    }

    #[must_use]
    pub fn tied(&self) -> usize {
        self.tied
    }

    #[must_use]
    pub fn largest(&self) -> usize {
        self.largest
    }

    /// Mean size of a tie group, over the queries that had one.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_group(&self) -> f64 {
        if self.tied == 0 {
            0.0
        } else {
            self.total_group as f64 / self.tied as f64
        }
    }

    /// Whether enough of the run was tied that its metrics describe the
    /// corpus rather than the ranker.
    ///
    /// A tenth is a judgement call, not a law: below it the odd duplicate
    /// moves a mean a little, above it the number is not about ranking any
    /// more. It is a threshold for *saying something*, never for changing a
    /// number, so being roughly right is enough.
    #[must_use]
    pub fn is_compromised(&self) -> bool {
        self.queries > 0 && self.tied * 10 > self.queries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_unique_score_is_a_group_of_one() {
        assert_eq!(tie_group(&[3.0, 2.0, 1.0], 0), 1);
        assert_eq!(tie_group(&[3.0, 2.0, 1.0], 2), 1);
    }

    #[test]
    fn copies_of_a_document_score_identically_and_are_counted() {
        assert_eq!(tie_group(&[5.0, 5.0, 5.0, 1.0], 1), 3);
    }

    #[test]
    fn an_index_past_the_end_is_no_group_rather_than_a_panic() {
        assert_eq!(tie_group(&[1.0], 7), 0);
        assert_eq!(tie_group(&[], 0), 0);
    }

    #[test]
    fn a_run_with_no_ties_is_not_compromised() {
        let mut t = Ties::new();
        for _ in 0..100 {
            t.add(1);
        }
        assert!(!t.is_compromised());
        assert_eq!(t.tied(), 0);
        assert!(t.mean_group().abs() < 1e-12);
    }

    #[test]
    fn a_run_where_most_answers_are_tied_is_compromised() {
        let mut t = Ties::new();
        for _ in 0..90 {
            t.add(40);
        }
        for _ in 0..10 {
            t.add(1);
        }
        assert!(t.is_compromised());
        assert_eq!(t.tied(), 90);
        assert_eq!(t.largest(), 40);
        assert!((t.mean_group() - 40.0).abs() < 1e-12);
    }

    #[test]
    fn one_duplicate_in_a_hundred_queries_is_not_worth_a_warning() {
        let mut t = Ties::new();
        t.add(2);
        for _ in 0..99 {
            t.add(1);
        }
        assert!(!t.is_compromised());
        assert_eq!(t.tied(), 1);
    }

    #[test]
    fn an_empty_run_is_not_compromised() {
        assert!(!Ties::new().is_compromised());
    }
}
