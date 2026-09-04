//! Building queries whose right answer nobody had to decide.
//!
//! A *known-item* query is a span of words lifted out of one document. The
//! document it came from is the answer, and it is the only answer, and no
//! human judged anything — which is what makes this the one measurement here
//! that cannot be quietly tuned to flatter the ranker.
//!
//! It is not free of assumptions. Every term of the span is required, so the
//! source document always matches; the question is only where it lands among
//! the other documents that also contain all of those words. When the span is
//! distinctive it lands first and the query measures nothing. When it is
//! common — `the of a` — hundreds of documents qualify and only ranking
//! separates them. The mean over many spans is dominated by the middle of that
//! range, which is the part worth measuring.
//!
//! **Spans are picked uniformly at random and no attempt is made to prefer
//! distinctive ones.** Preferring them is exactly where a benchmark gets tuned
//! without anybody deciding to tune it.
//!
//! Everything is seeded, so a run is a number somebody else can reproduce
//! rather than a number that moved.

/// `SplitMix64`: the smallest generator that is good enough to sample with.
///
/// Not cryptographic, not trying to be. It is here so a report carries a seed
/// instead of a footnote about which machine produced it, and so that adding a
/// dependency for eleven lines of arithmetic is not necessary.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number below `n`, or 0 when `n` is 0.
    ///
    /// Uses the multiply-shift reduction rather than a modulo: the bias is
    /// under one part in 2^64 for any `n` a corpus could have, and it avoids
    /// the division.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let r = u128::from(self.next_u64());
        usize::try_from((r * n as u128) >> 64).unwrap_or(0)
    }
}

/// Picks `take` distinct indices below `count`, in ascending order.
///
/// A partial Fisher-Yates over a map of the swaps, so sampling 200 documents
/// out of a hundred thousand costs 200 steps rather than shuffling the corpus.
/// Asking for more than exists returns everything.
#[must_use]
pub fn sample(count: usize, take: usize, seed: u64) -> Vec<usize> {
    if take >= count {
        return (0..count).collect();
    }
    let mut rng = Rng::new(seed);
    // Only the entries actually touched are stored; the rest are the identity.
    let mut swapped: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut picked = Vec::with_capacity(take);
    for i in 0..take {
        let j = i + rng.below(count - i);
        let at = |m: &std::collections::HashMap<usize, usize>, k: usize| *m.get(&k).unwrap_or(&k);
        let (vi, vj) = (at(&swapped, i), at(&swapped, j));
        swapped.insert(j, vi);
        swapped.insert(i, vj);
        picked.push(vj);
    }
    picked.sort_unstable();
    picked
}

/// Lifts a contiguous run of `len` terms out of `terms`.
///
/// Returns `None` when the document is too short to supply one — a document of
/// three words cannot pose a six-word question, and padding it with words it
/// does not contain would make a query it cannot answer.
#[must_use]
pub fn span(terms: &[String], len: usize, rng: &mut Rng) -> Option<Vec<String>> {
    if len == 0 || terms.len() < len {
        return None;
    }
    let start = rng.below(terms.len() - len + 1);
    Some(terms[start..start + len].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_sample() {
        assert_eq!(sample(1000, 20, 7), sample(1000, 20, 7));
        assert_ne!(sample(1000, 20, 7), sample(1000, 20, 8));
    }

    #[test]
    fn a_sample_has_no_repeats_and_stays_in_range() {
        let picked = sample(500, 100, 42);
        assert_eq!(picked.len(), 100);
        let mut sorted = picked.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), 100, "repeats in {picked:?}");
        assert!(picked.iter().all(|&i| i < 500));
        assert!(picked.windows(2).all(|w| w[0] < w[1]), "not ascending");
    }

    #[test]
    fn asking_for_more_than_exists_returns_everything() {
        assert_eq!(sample(3, 10, 1), [0, 1, 2]);
        assert_eq!(sample(0, 10, 1), Vec::<usize>::new());
    }

    #[test]
    fn sampling_reaches_the_whole_range_not_just_the_front() {
        // A partial shuffle done wrong picks only from the first `take`
        // entries; this is the assertion that catches it.
        let picked = sample(10_000, 50, 3);
        assert!(picked.iter().any(|&i| i > 5_000), "{picked:?}");
    }

    #[test]
    fn a_span_is_contiguous_and_in_order() {
        let terms: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        let mut rng = Rng::new(11);
        for _ in 0..50 {
            let s = span(&terms, 4, &mut rng).unwrap();
            assert_eq!(s.len(), 4);
            let start: usize = s[0].parse().unwrap();
            assert!(start + 4 <= 20);
            assert_eq!(s, terms[start..start + 4]);
        }
    }

    #[test]
    fn a_document_too_short_to_ask_a_question_asks_none() {
        let terms: Vec<String> = vec!["one".into(), "two".into()];
        assert!(span(&terms, 6, &mut Rng::new(1)).is_none());
        assert!(span(&terms, 0, &mut Rng::new(1)).is_none());
        assert!(span(&terms, 2, &mut Rng::new(1)).is_some());
    }

    #[test]
    fn below_stays_below() {
        let mut rng = Rng::new(99);
        assert_eq!(rng.below(0), 0);
        assert_eq!(rng.below(1), 0);
        for _ in 0..1000 {
            assert!(rng.below(7) < 7);
        }
    }

    #[test]
    fn below_does_not_get_stuck_on_one_value() {
        let mut rng = Rng::new(5);
        let seen: std::collections::HashSet<usize> = (0..200).map(|_| rng.below(10)).collect();
        assert!(seen.len() > 5, "only saw {seen:?}");
    }
}
