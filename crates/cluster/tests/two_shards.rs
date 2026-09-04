//! A coordinator and real shards, over real sockets.
//!
//! The point of step one of `docs/DISTRIBUTION.md` is that a one-shard
//! deployment takes the distributed code path. These tests run it with one
//! shard and with two, and assert both give the answer a single index gives.

use std::sync::Arc;

use indexander_cluster::coordinator::Coordinator;
use indexander_cluster::shard::{self, ShardIndex};
use indexander_core::Document;
use indexander_index::builder::SegmentBuilder;
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;
use tokio::net::TcpListener;

/// Shard A: small, full of "rust", so "perl" looks rare inside it.
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

/// Shard B: large, full of "perl", so "rust" looks far rarer than it is.
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

fn segment_of(docs: &[Document]) -> Segment {
    let mut builder = SegmentBuilder::new();
    for doc in docs {
        builder.add(doc);
    }
    Segment::from_bytes(builder.encode()).expect("segment")
}

/// Starts a shard on an ephemeral port and returns its address.
async fn start(docs: &[Document]) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    let shard = Arc::new(ShardIndex::single(segment_of(docs)));
    tokio::spawn(async move {
        let _ = shard::serve(listener, shard).await;
    });
    address
}

/// What a single, unsharded index answers.
fn single_index_ranking(limit: usize) -> Vec<String> {
    let mut all = shard_a();
    all.extend(shard_b());
    let segment = segment_of(&all);
    search(&segment, &query::parse("rust perl"), limit)
        .expect("search")
        .into_iter()
        .map(|h| h.uri)
        .collect()
}

#[tokio::test]
async fn one_shard_answers_exactly_like_a_local_index() {
    let mut all = shard_a();
    all.extend(shard_b());
    let address = start(&all).await;

    let coordinator = Coordinator::connect(&[address]).await.expect("connect");
    assert_eq!(coordinator.shard_count(), 1);

    let hits = coordinator.search("rust perl", 5).await.expect("search");
    let uris: Vec<String> = hits.into_iter().map(|h| h.uri).collect();
    assert_eq!(uris, single_index_ranking(5));
}

#[tokio::test]
async fn two_shards_agree_with_one_index() {
    // The corpus is built so that local statistics would rank these the other
    // way round; getting the single-index order back is the proof that round
    // one actually happened.
    let a = start(&shard_a()).await;
    let b = start(&shard_b()).await;

    let coordinator = Coordinator::connect(&[a, b]).await.expect("connect");
    assert_eq!(coordinator.shard_count(), 2);

    let hits = coordinator.search("rust perl", 5).await.expect("search");
    let uris: Vec<String> = hits.into_iter().map(|h| h.uri).collect();
    assert_eq!(uris, single_index_ranking(5));
    assert_eq!(uris.len(), 2);
}

#[tokio::test]
async fn round_one_sums_the_shards() {
    let a = start(&shard_a()).await;
    let b = start(&shard_b()).await;
    let coordinator = Coordinator::connect(&[a, b]).await.expect("connect");

    let stats = coordinator
        .term_statistics(&["rust".to_owned(), "perl".to_owned()])
        .await
        .expect("term stats");

    assert_eq!(stats.total_docs, 110, "10 + 100 documents");
    assert_eq!(stats.doc_freq["rust"], 11, "10 in shard a, 1 in shard b");
    assert_eq!(
        stats.doc_freq["perl"], 101,
        "1 in shard a, 99 plus the matching document in shard b"
    );
}

#[tokio::test]
async fn the_cluster_reports_its_totals() {
    let a = start(&shard_a()).await;
    let b = start(&shard_b()).await;
    let coordinator = Coordinator::connect(&[a, b]).await.expect("connect");
    let (documents, terms) = coordinator.stats().await.expect("stats");
    assert_eq!(documents, 110);
    assert!(terms > 0);
}

#[tokio::test]
async fn an_unreachable_shard_fails_the_query_rather_than_silently_shrinking_the_corpus() {
    let a = start(&shard_a()).await;
    // Nothing listening here.
    let dead = "127.0.0.1:1".to_owned();
    let result = Coordinator::connect(&[a, dead]).await;
    assert!(result.is_err(), "a missing shard must not be ignored");
    assert!(format!("{}", result.unwrap_err()).contains("127.0.0.1:1"));
}

#[tokio::test]
async fn a_coordinator_needs_at_least_one_shard() {
    assert!(Coordinator::connect(&[]).await.is_err());
}

#[tokio::test]
async fn empty_and_zero_limit_queries_ask_the_shards_nothing() {
    let a = start(&shard_a()).await;
    let coordinator = Coordinator::connect(&[a]).await.expect("connect");
    assert!(coordinator.search("", 10).await.expect("search").is_empty());
    assert!(
        coordinator
            .search("rust", 0)
            .await
            .expect("search")
            .is_empty()
    );
}

#[tokio::test]
async fn the_same_connection_serves_many_queries() {
    // Connections are long-lived; a second query must not need a reconnect.
    let a = start(&shard_a()).await;
    let coordinator = Coordinator::connect(&[a]).await.expect("connect");
    for _ in 0..5 {
        let hits = coordinator.search("rust", 3).await.expect("search");
        assert!(!hits.is_empty());
    }
}

#[tokio::test]
async fn many_shards_give_the_same_answer_as_one_index() {
    // Eight shards, so the fan-out is doing real work. Splitting the corpus
    // further must not change the ranking.
    let mut all = shard_a();
    all.extend(shard_b());
    let chunk = all.len().div_ceil(8);

    let mut addresses = Vec::new();
    for slice in all.chunks(chunk) {
        addresses.push(start(slice).await);
    }
    assert_eq!(addresses.len(), 8);

    let coordinator = Coordinator::connect(&addresses).await.expect("connect");
    let hits = coordinator.search("rust perl", 5).await.expect("search");
    let uris: Vec<String> = hits.into_iter().map(|h| h.uri).collect();
    assert_eq!(uris, single_index_ranking(5));
}

#[tokio::test]
async fn results_do_not_depend_on_which_shard_answers_first() {
    // The fan-out is concurrent, so replies arrive in whatever order the
    // network gives them. The merge must not inherit that order.
    let a = start(&shard_a()).await;
    let b = start(&shard_b()).await;
    let coordinator = Coordinator::connect(&[a, b]).await.expect("connect");

    let first = coordinator.search("rust perl", 5).await.expect("search");
    for _ in 0..20 {
        let again = coordinator.search("rust perl", 5).await.expect("search");
        assert_eq!(
            again, first,
            "a repeated query returned a different ranking"
        );
    }
}
