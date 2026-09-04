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

/// How much a document's length discounts its scores.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn length_norm(document_length: u32, average_length: f32) -> f32 {
    1.0 - B + B * (document_length as f32 / average_length.max(1.0))
}

/// The part of a term's contribution that does not depend on the corpus.
///
/// The full contribution is `idf * saturation(...)`. Splitting it here is what
/// lets a block-max bound be computed at index time and still be correct when
/// the query supplies a different `idf` — which it does whenever this segment
/// is one shard of several.
#[must_use]
pub fn saturation(weighted_tf: f32, length_norm: f32) -> f32 {
    if weighted_tf <= 0.0 {
        return 0.0;
    }
    (weighted_tf * (K1 + 1.0)) / (weighted_tf + K1 * length_norm)
}

/// Turns a PageRank into a multiplier around 1.0.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn authority(rank: f32, total_docs: usize) -> f32 {
    if rank <= 0.0 {
        return 1.0;
    }
    let relative = rank * total_docs as f32;
    1.0 + AUTHORITY_WEIGHT * relative.max(0.0).ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

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
