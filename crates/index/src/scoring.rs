//! The scoring function, in one place.
//!
//! It lives here rather than inside the searcher because the *indexer* needs
//! it too: block-max bounds are computed when a segment is written, and a
//! bound computed with a different formula than the one that scores queries is
//! not a bound at all — it is a source of missing results.

/// Saturation: how quickly extra occurrences of a term stop adding score.
pub const K1: f32 = 1.2;
/// Length normalisation: how much a long document is penalised. 0 disables it.
pub const B: f32 = 0.75;
/// How much authority is allowed to move a result.
pub const AUTHORITY_WEIGHT: f32 = 0.5;

/// The three numbers that decide how a document scores.
///
/// They travel with the segment rather than being compiled in, because the
/// right values are a property of the corpus. On the crate sources measured in
/// `docs/EVALUATION.md`, full length normalisation is worth 0.087 MRR over the
/// textbook 0.75 — and on another corpus it might not be.
///
/// They cannot be changed after a segment is written. Block-max bounds are
/// computed with them at index time, and a bound computed with one formula
/// while queries score with another is not a bound: it is a result that
/// silently never appears. That is also why segments written with different
/// parameters refuse to merge, and why a query across shards that disagree is
/// an error rather than a ranking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Saturation: how quickly extra occurrences stop adding score.
    pub k1: f32,
    /// Length normalisation: how much a long document is penalised.
    pub b: f32,
    /// How much authority is allowed to move a result.
    pub authority_weight: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            k1: K1,
            b: B,
            authority_weight: AUTHORITY_WEIGHT,
        }
    }
}

impl Params {
    /// Whether these could have produced a scoring function at all.
    ///
    /// Checked when a segment is opened, because a corrupt or hostile footer
    /// with a `NaN` here would make every score `NaN` and every comparison
    /// false, which sorts into an arbitrary order instead of failing.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.k1.is_finite()
            && self.b.is_finite()
            && self.authority_weight.is_finite()
            && self.k1 >= 0.0
            && self.b >= 0.0
            && self.authority_weight >= 0.0
    }

    /// How much a document's length discounts its scores.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn length_norm(&self, document_length: u32, average_length: f32) -> f32 {
        1.0 - self.b + self.b * (document_length as f32 / average_length.max(1.0))
    }

    /// The part of a term's contribution that does not depend on the corpus.
    #[must_use]
    pub fn saturation(&self, weighted_tf: f32, length_norm: f32) -> f32 {
        if weighted_tf <= 0.0 {
            return 0.0;
        }
        (weighted_tf * (self.k1 + 1.0)) / (weighted_tf + self.k1 * length_norm)
    }

    /// Turns a `PageRank` into a multiplier around 1.0.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn authority(&self, rank: f32, total_docs: usize) -> f32 {
        if rank <= 0.0 {
            return 1.0;
        }
        let relative = rank * total_docs as f32;
        1.0 + self.authority_weight * relative.max(0.0).ln_1p()
    }
}

/// Inverse document frequency, the part of BM25 that makes rare words matter.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn idf(total_docs: usize, doc_freq: usize) -> f32 {
    let n = total_docs as f32;
    let df = doc_freq as f32;
    // The +1 inside the logarithm keeps this positive even for a term that
    // appears in every document, which the textbook formula does not.
    ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
}

/// How much a document's length discounts its scores, with the defaults.
#[must_use]
pub fn length_norm(document_length: u32, average_length: f32) -> f32 {
    Params::default().length_norm(document_length, average_length)
}

/// The part of a term's contribution that does not depend on the corpus.
///
/// The full contribution is `idf * saturation(...)`. Splitting it here is what
/// lets a block-max bound be computed at index time and still be correct when
/// the query supplies a different `idf` — which it does whenever this segment
/// is one shard of several.
#[must_use]
pub fn saturation(weighted_tf: f32, length_norm: f32) -> f32 {
    Params::default().saturation(weighted_tf, length_norm)
}

/// Turns a `PageRank` into a multiplier around 1.0, with the defaults.
#[must_use]
pub fn authority(rank: f32, total_docs: usize) -> f32 {
    Params::default().authority(rank, total_docs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_the_constants_the_free_functions_use() {
        let p = Params::default();
        assert!((p.k1 - K1).abs() < f32::EPSILON);
        assert!((p.b - B).abs() < f32::EPSILON);
        assert!((p.authority_weight - AUTHORITY_WEIGHT).abs() < f32::EPSILON);
        assert!((p.length_norm(50, 100.0) - length_norm(50, 100.0)).abs() < f32::EPSILON);
        assert!((p.saturation(3.0, 1.2) - saturation(3.0, 1.2)).abs() < f32::EPSILON);
        assert!((p.authority(0.01, 500) - authority(0.01, 500)).abs() < f32::EPSILON);
    }

    #[test]
    fn a_higher_b_penalises_a_long_document_more() {
        let gentle = Params {
            b: 0.3,
            ..Params::default()
        };
        let harsh = Params {
            b: 1.0,
            ..Params::default()
        };
        let long = 400;
        assert!(harsh.length_norm(long, 100.0) > gentle.length_norm(long, 100.0));
        // A higher norm means a lower saturation, hence a lower score.
        assert!(
            harsh.saturation(3.0, harsh.length_norm(long, 100.0))
                < gentle.saturation(3.0, gentle.length_norm(long, 100.0))
        );
    }

    #[test]
    fn b_of_zero_ignores_length_entirely() {
        let flat = Params {
            b: 0.0,
            ..Params::default()
        };
        assert!(
            (flat.length_norm(1, 100.0) - flat.length_norm(10_000, 100.0)).abs() < f32::EPSILON
        );
    }

    #[test]
    fn nan_and_negative_parameters_are_not_usable() {
        assert!(Params::default().is_usable());
        assert!(
            !Params {
                k1: f32::NAN,
                ..Params::default()
            }
            .is_usable()
        );
        assert!(
            !Params {
                b: f32::INFINITY,
                ..Params::default()
            }
            .is_usable()
        );
        assert!(
            !Params {
                b: -0.5,
                ..Params::default()
            }
            .is_usable()
        );
        assert!(
            !Params {
                authority_weight: f32::NAN,
                ..Params::default()
            }
            .is_usable()
        );
        // Zero is a legitimate choice for all three: it turns a feature off.
        assert!(
            Params {
                k1: 0.0,
                b: 0.0,
                authority_weight: 0.0
            }
            .is_usable()
        );
    }

    #[test]
    fn a_rare_term_outweighs_a_common_one() {
        assert!(idf(1000, 1) > idf(1000, 500));
        assert!(idf(100, 100) > 0.0);
    }

    #[test]
    fn saturation_is_monotonic_in_frequency() {
        let norm = 1.0;
        let mut previous = 0.0;
        for tf in 1i16..50 {
            let value = saturation(f32::from(tf), norm);
            assert!(value > previous, "saturation fell at tf {tf}");
            previous = value;
        }
    }

    #[test]
    fn saturation_saturates() {
        // Doubling a large frequency barely moves the score. That is the point.
        let a = saturation(50.0, 1.0);
        let b = saturation(100.0, 1.0);
        assert!(b > a);
        assert!(b < a * 1.1, "{a} -> {b} is not saturation");
    }

    #[test]
    fn a_longer_document_scores_less_for_the_same_frequency() {
        let short = saturation(3.0, length_norm(10, 100.0));
        let long = saturation(3.0, length_norm(1000, 100.0));
        assert!(short > long);
    }

    #[test]
    fn zero_frequency_contributes_nothing() {
        assert!((saturation(0.0, 1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn authority_is_neutral_without_a_rank_and_logarithmic_with_one() {
        assert!((authority(0.0, 100) - 1.0).abs() < f32::EPSILON);
        let ordinary = authority(0.001, 1000);
        let central = authority(1.0, 1000);
        assert!(central > ordinary);
        assert!(central < ordinary * 20.0);
    }
}
