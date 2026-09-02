//! Splitting a corpus into shards must not change the answer.
//!
//! These tests build one corpus three ways — whole, and split in two, scored
//! locally and then globally — and compare the rankings. The local-statistics
//! case is expected to be *wrong*; that is the point of the test, and the
//! reason the protocol has two rounds.

use std::collections::HashMap;

use indexander_core::Document;
use indexander_index::builder::SegmentBuilder;
use indexander_index::query;
use indexander_index::search::{GlobalStats, Hit, search, search_with_stats};
use indexander_index::segment::Segment;

/// A corpus built so that local statistics genuinely lie.
///
/// Two documents match the query `rust perl`, one in each shard, and they are
/// deliberately unbalanced:
///
/// * Shard A is small and full of "rust", so within it "perl" looks rare.
/// * Shard B is large and full of "perl", so within it "rust" looks *very*
///   rare — rarer than it is in the corpus as a whole.
///
/// Scored locally, shard B's document rides an inflated `idf` for "rust" and
/// wins. Scored globally, "rust" is the rarer term overall, and the document
/// that repeats it wins instead. The two rankings disagree, which is the
/// whole reason the protocol has a first round.
fn shard_a() -> Vec<Document> {
    let mut docs: Vec<Document> = (0..9)
        .map(|i| Document::new(format!("doc://a{i}"), "articulo", "rust motor de busqueda"))
        .collect();
    docs.push(Document::new(
        "doc://a-match",
        "articulo",
        "rust rust rust perl motor",
    ));
    docs
}

fn shard_b() -> Vec<Document> {
    let mut docs: Vec<Document> = (0..99)
        .map(|i| Document::new(format!("doc://b{i}"), "articulo", "perl motor de busqueda"))
        .collect();
    docs.push(Document::new(
        "doc://b-match",
        "articulo",
        "rust perl perl perl motor",
    ));
    docs
}

fn corpus() -> Vec<Document> {
    let mut all = shard_a();
    all.extend(shard_b());
    all
}

fn segment_of(docs: &[Document]) -> Segment {
    let mut builder = SegmentBuilder::new();
    for doc in docs {
        builder.add(doc);
    }
    Segment::from_bytes(builder.encode()).expect("segment")
}

/// The two halves, as separate shards.
fn shards() -> (Segment, Segment) {
    (segment_of(&shard_a()), segment_of(&shard_b()))
}

/// Merge shard results the way a coordinator does: concatenate, sort, cut.
fn merge(mut all: Vec<Hit>, limit: usize) -> Vec<String> {
    all.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.uri.cmp(&b.uri))
    });
    all.truncate(limit);
    all.into_iter().map(|h| h.uri).collect()
}

/// The global statistics a coordinator would assemble in round one.
fn global_stats(shards: &[&Segment], terms: &[&str]) -> GlobalStats {
    let mut stats = GlobalStats::default();
    for shard in shards {
        let per_term: Vec<(String, usize)> = terms
            .iter()
            .map(|t| ((*t).to_owned(), shard.document_frequency(t).unwrap_or(0)))
            .collect();
        stats.add_shard(shard.document_count(), &per_term);
    }
    stats
}

#[test]
fn local_statistics_produce_a_different_ranking_than_one_index() {
    // This is the bug the two-round protocol exists to prevent. If this test
    // ever starts passing trivially, the corpus stopped exercising the case.
    let whole = segment_of(&corpus());
    let (a, b) = shards();
    let q = query::parse("rust perl");

    let single = merge(search(&whole, &q, 5).unwrap(), 5);
    assert_eq!(single.len(), 2, "expected both matching documents");

    let mut naive = search(&a, &q, 5).unwrap();
    naive.extend(search(&b, &q, 5).unwrap());
    let naive = merge(naive, 5);

    assert_eq!(naive.len(), 2, "each shard should contribute one match");
    assert_ne!(
        single, naive,
        "the corpus no longer exercises incomparable idf across shards"
    );
}

#[test]
fn global_statistics_make_shards_agree_with_one_index() {
    let whole = segment_of(&corpus());
    let (a, b) = shards();
    let q = query::parse("rust perl");
    let stats = global_stats(&[&a, &b], &["rust", "perl"]);

    let single = merge(search(&whole, &q, 5).unwrap(), 5);

    let mut sharded = search_with_stats(&a, &q, 5, Some(&stats)).unwrap();
    sharded.extend(search_with_stats(&b, &q, 5, Some(&stats)).unwrap());
    let sharded = merge(sharded, 5);

    assert_eq!(
        single, sharded,
        "sharded ranking diverged from the single index"
    );
}

#[test]
fn the_global_document_count_is_the_sum_of_the_shards() {
    let (a, b) = shards();
    let stats = global_stats(&[&a, &b], &["rust"]);
    assert_eq!(stats.total_docs, a.document_count() + b.document_count());
    assert_eq!(
        stats.doc_freq["rust"],
        a.document_frequency("rust").unwrap() + b.document_frequency("rust").unwrap()
    );
}

#[test]
fn empty_statistics_fall_back_to_the_segment_itself() {
    // A single-shard deployment takes the distributed code path and supplies
    // nothing; it must behave exactly like the local one.
    let whole = segment_of(&corpus());
    let q = query::parse("rust");
    let local = search(&whole, &q, 5).unwrap();
    let empty = search_with_stats(&whole, &q, 5, Some(&GlobalStats::default())).unwrap();
    let none = search_with_stats(&whole, &q, 5, None).unwrap();

    assert_eq!(local, empty);
    assert_eq!(local, none);
}

#[test]
fn a_term_missing_from_the_global_map_still_scores() {
    let whole = segment_of(&corpus());
    let stats = GlobalStats {
        total_docs: 42,
        doc_freq: HashMap::new(),
    };
    let hits = search_with_stats(&whole, &query::parse("rust"), 5, Some(&stats)).unwrap();
    assert!(
        !hits.is_empty(),
        "an unknown term should fall back, not vanish"
    );
}

#[test]
fn one_shard_holding_everything_equals_no_sharding() {
    // shard_count = 1: the whole point of step one is that this path works.
    let whole = segment_of(&corpus());
    let stats = global_stats(&[&whole], &["rust", "perl"]);
    let q = query::parse("rust perl");

    let direct = search(&whole, &q, 5).unwrap();
    let through_protocol = search_with_stats(&whole, &q, 5, Some(&stats)).unwrap();
    assert_eq!(direct, through_protocol);
}
