//! Metrics are only worth having if they move when the ranking gets worse.
//!
//! The unit tests in `metrics` prove the arithmetic on hand-written lists.
//! These prove the loop closes: a real index, the real searcher, and a
//! measurement that separates its output from a deliberately spoiled version
//! of the same output. A metric that scores both the same is a metric that
//! would never have caught a regression.

use indexander_core::Document;
use indexander_eval::metrics::{Judged, Scores, Summary};
use indexander_index::builder::SegmentBuilder;
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;

/// Documents that differ in how much they say about ferrets, so that ranking
/// them has a right answer and getting it wrong is visible.
fn corpus() -> Segment {
    let mut builder = SegmentBuilder::new();
    builder.add(&Document::new(
        "/ferrets",
        "ferrets",
        "ferrets sleep. ferrets dig. a ferret is a ferret and ferrets are ferrets.",
    ));
    builder.add(&Document::new(
        "/mammals",
        "mammals",
        "a survey of mammals, among them the ferret, the badger and the otter, \
         each described at some length so that this document is long.",
    ));
    builder.add(&Document::new(
        "/badgers",
        "badgers",
        "badgers dig too, and a badger is not a ferret.",
    ));
    builder.add(&Document::new("/otters", "otters", "otters swim."));
    Segment::from_bytes(builder.encode()).expect("the builder writes readable segments")
}

fn run(segment: &Segment, text: &str) -> Vec<String> {
    search(segment, &query::parse(text), 10)
        .expect("searching a segment built in this test")
        .into_iter()
        .map(|hit| hit.uri)
        .collect()
}

fn judged(uri: &str) -> Judged {
    let mut j = Judged::new();
    j.judge(uri, 1);
    j
}

#[test]
fn the_engine_puts_the_right_document_first_and_the_metric_says_so() {
    let segment = corpus();
    let ranked = run(&segment, "ferret");
    assert!(!ranked.is_empty(), "no results at all");

    let scores = Scores::of(&ranked, &judged("/ferrets"), 10);
    assert!(
        (scores.reciprocal_rank - 1.0).abs() < 1e-12,
        "the document about ferrets should rank first, got {ranked:?}"
    );
    assert!((scores.ndcg - 1.0).abs() < 1e-12);
    assert!(scores.success);
}

#[test]
fn spoiling_the_order_lowers_every_metric_that_reads_order() {
    let segment = corpus();
    let good = run(&segment, "ferret");
    assert!(
        good.len() > 1,
        "need more than one hit to reorder: {good:?}"
    );

    let mut spoiled = good.clone();
    spoiled.reverse();

    let j = judged(&good[0]);
    let before = Scores::of(&good, &j, 10);
    let after = Scores::of(&spoiled, &j, 10);

    // Same documents, same set, same recall - only the order changed.
    assert!((before.recall - after.recall).abs() < 1e-12);
    assert!(
        after.reciprocal_rank < before.reciprocal_rank,
        "{before:?} -> {after:?}"
    );
    assert!(after.ndcg < before.ndcg);
}

#[test]
fn a_ranker_that_returns_nothing_measures_as_a_total_failure() {
    let segment = corpus();
    // A term no document contains: the searcher is right to return nothing,
    // and the measurement must call that a failure rather than a zero it
    // quietly averages away.
    let ranked = run(&segment, "capybara");
    assert!(ranked.is_empty(), "expected no hits, got {ranked:?}");

    let mut summary = Summary::new();
    summary.add(Scores::of(&ranked, &judged("/ferrets"), 10), 0);
    assert_eq!(summary.failures(), 1);
    assert!(summary.mrr().abs() < 1e-12);
}

#[test]
fn a_known_item_query_lifted_from_a_document_finds_that_document() {
    let segment = corpus();
    // The words are the badger document's own, in order, and every one is
    // required - so it must be in the results, and it should be first.
    let ranked = run(&segment, "badger is not a ferret");
    let scores = Scores::of(&ranked, &judged("/badgers"), 10);
    assert!(
        (scores.reciprocal_rank - 1.0).abs() < 1e-12,
        "the document the words came from should win, got {ranked:?}"
    );
}
