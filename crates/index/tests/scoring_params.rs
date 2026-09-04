//! The scoring parameters travel with the segment, and everything that could
//! quietly mix two sets of them refuses to.
//!
//! The failure being defended against has no symptom. Block-max bounds are
//! computed when a segment is written; a reader scoring with different
//! parameters uses bounds that do not bound its own formula, and the result is
//! not an error or an empty page but a ranking with documents silently missing
//! from it. So every one of these tests is here to watch a refusal happen, not
//! to watch a success.

use indexander_core::Document;
use indexander_index::builder::SegmentBuilder;
use indexander_index::index::Index;
use indexander_index::query;
use indexander_index::scoring::Params;
use indexander_index::search::search;
use indexander_index::segment::Segment;

/// Documents of deliberately different lengths, so that `b` — which is the
/// length knob — has something to act on.
fn corpus(params: Params) -> Segment {
    let mut builder = SegmentBuilder::with_params(params);
    builder.add(&Document::new("/short", "short", "motor"));
    builder.add(&Document::new(
        "/long",
        "long",
        "motor de busqueda con muchas palabras alrededor que hacen este documento \
         considerablemente mas largo que el otro y por tanto mas penalizado motor",
    ));
    builder.add(&Document::new(
        "/other",
        "other",
        "algo distinto sin el termino",
    ));
    Segment::from_bytes(builder.encode()).expect("the builder writes readable segments")
}

#[test]
fn the_parameters_survive_a_round_trip_through_the_footer() {
    let chosen = Params {
        k1: 0.9,
        b: 1.0,
        authority_weight: 0.25,
    };
    let segment = corpus(chosen);
    assert_eq!(segment.params(), chosen);
    // And the default path still records the defaults.
    assert_eq!(corpus(Params::default()).params(), Params::default());
}

#[test]
fn the_parameters_actually_change_the_ranking() {
    // With no length normalisation the long document wins on frequency; with
    // full normalisation the short one wins on brevity. If this ever stops
    // being true, the parameters are being written and then ignored.
    let flat = corpus(Params {
        b: 0.0,
        ..Params::default()
    });
    let harsh = corpus(Params {
        b: 1.0,
        ..Params::default()
    });
    let q = query::parse("motor");

    let flat_hits = search(&flat, &q, 10).expect("searching");
    let harsh_hits = search(&harsh, &q, 10).expect("searching");
    assert_eq!(flat_hits[0].uri, "/long", "{flat_hits:?}");
    assert_eq!(harsh_hits[0].uri, "/short", "{harsh_hits:?}");
}

#[test]
fn a_segment_whose_footer_holds_nonsense_parameters_will_not_open() {
    let mut bytes = corpus(Params::default()).as_bytes().to_vec();
    // The three f32 parameters sit just after the digest, which is the tenth
    // u64 of the footer; the version and magic are the last eight bytes.
    let at = bytes.len() - (3 * 4 + 4 + 4);
    bytes[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());

    let err = Segment::from_bytes(bytes).expect_err("NaN parameters must not open");
    assert!(
        err.to_string().contains("not usable"),
        "unexpected error: {err}"
    );
}

#[test]
fn segments_written_with_different_parameters_refuse_to_merge() {
    let a = corpus(Params::default());
    let b = corpus(Params {
        b: 1.0,
        ..Params::default()
    });

    let err = SegmentBuilder::from_segments(&[&a, &b]).expect_err("a merge here is not exact");
    assert!(
        err.to_string().contains("different scoring parameters"),
        "unexpected error: {err}"
    );

    // The same two parameters merge fine, which is what says the refusal is
    // about disagreement and not about merging.
    let c = corpus(Params::default());
    assert!(SegmentBuilder::from_segments(&[&a, &c]).is_ok());
}

#[test]
#[should_panic(expected = "different scoring parameters")]
fn absorbing_a_builder_with_other_parameters_is_a_programming_error() {
    let mut a = SegmentBuilder::with_params(Params::default());
    let b = SegmentBuilder::with_params(Params {
        k1: 2.0,
        ..Params::default()
    });
    a.absorb(b);
}

#[test]
fn an_index_of_segments_that_disagree_refuses_to_search() {
    let mut index = Index::new();
    index.push(corpus(Params::default()));
    index.push(corpus(Params {
        b: 1.0,
        ..Params::default()
    }));

    let err = index
        .search(&query::parse("motor"), 10)
        .expect_err("scores from two formulas are not comparable");
    assert!(
        err.to_string().contains("different scoring parameters"),
        "unexpected error: {err}"
    );
}

#[test]
fn an_index_whose_segments_agree_searches_normally() {
    let mut index = Index::new();
    index.push(corpus(Params::default()));
    index.push(corpus(Params::default()));
    let hits = index.search(&query::parse("motor"), 10).expect("searching");
    assert_eq!(hits.len(), 4, "two copies of two matching documents");
    assert_eq!(index.params().expect("they agree"), Params::default());
}

#[test]
fn an_empty_index_has_the_defaults_rather_than_an_error() {
    assert_eq!(
        Index::new().params().expect("nothing to disagree"),
        Params::default()
    );
}

/// The invariant that makes all of the above matter: bounds written with one
/// set of parameters must agree with the scorer using them.
///
/// Block-max pruning only happens once the heap is full, so a query asking for
/// more results than exist never prunes at all — which is why this compares a
/// small limit against a large one rather than against a document count. The
/// large one is the unpruned truth; if the stored bounds disagree with the
/// scorer, the small one silently loses documents that belong in the top ten.
#[test]
fn pruning_with_unusual_parameters_loses_nothing() {
    // `k1 = 0` is deliberately absent: it makes saturation exactly 1 for every
    // document, term frequency cancels out, and every score is identical. That
    // is what `k1 = 0` means, and it leaves nothing here to compare.
    for params in [
        Params::default(),
        Params {
            b: 0.0,
            ..Params::default()
        },
        Params {
            b: 1.0,
            ..Params::default()
        },
        Params {
            k1: 0.5,
            b: 0.9,
            authority_weight: 0.0,
        },
        Params {
            k1: 4.0,
            b: 1.0,
            authority_weight: 0.0,
        },
    ] {
        let mut builder = SegmentBuilder::with_params(params);
        // Several skip blocks. Every document gets a *different* number of
        // occurrences, because a corpus where documents tie makes this
        // comparison meaningless: among equally-scoring documents, which ten
        // survive legitimately depends on which the pruner saw first, and the
        // test would fail on correct behaviour. Distinct frequencies mean
        // distinct scores and one right answer.
        for i in 0..300 {
            let body = "motor ".repeat(i + 1);
            builder.add(&Document::new(
                format!("/doc{i:03}"),
                format!("doc {i}"),
                body,
            ));
        }
        let segment = Segment::from_bytes(builder.encode()).expect("readable");
        let q = query::parse("motor");

        let unpruned = search(&segment, &q, 1000).expect("searching");
        assert_eq!(unpruned.len(), 300, "the corpus itself is wrong");
        // Proof the corpus is not degenerate: if scores tied, the comparison
        // below would be about tie-breaking rather than about bounds.
        let distinct: std::collections::HashSet<u32> =
            unpruned.iter().map(|h| h.score.to_bits()).collect();
        assert_eq!(
            distinct.len(),
            300,
            "scores tie with {params:?}; the corpus is degenerate"
        );
        let pruned = search(&segment, &q, 10).expect("searching");

        let expected: Vec<&str> = unpruned.iter().take(10).map(|h| h.uri.as_str()).collect();
        let got: Vec<&str> = pruned.iter().map(|h| h.uri.as_str()).collect();
        assert_eq!(
            got, expected,
            "pruning dropped results with {params:?}: bounds and scorer disagree"
        );
    }
}
