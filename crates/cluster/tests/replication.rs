//! Replicas: copying a segment, and surviving the loss of one.
//!
//! Two properties, and the second is the one replication is *for*:
//!
//! * A copy is a copy. Byte-identical, checked against a digest, and refused
//!   if it is not — a transfer that says it finished is not a replica.
//! * A dead replica is not a dead shard. Earlier, a missing shard failed the
//!   whole query on purpose, because a fifth of the corpus silently absent is
//!   worse than an error. A missing *replica* is different: another copy holds
//!   the same segment, so failing over returns the same answer rather than a
//!   quieter one.

use std::sync::Arc;

use indexander_cluster::coordinator::Coordinator;
use indexander_cluster::replication::fetch_segment;
use indexander_cluster::shard;
use indexander_core::Document;
use indexander_index::builder::SegmentBuilder;
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;
use tokio::net::TcpListener;

fn corpus() -> Vec<Document> {
    (0..120)
        .map(|i| {
            Document::new(
                format!("doc://{i}"),
                format!("documento {i}"),
                format!("motor de busqueda distribuido con replicas numero {i}"),
            )
        })
        .collect()
}

fn built() -> SegmentBuilder {
    let mut builder = SegmentBuilder::new();
    for doc in &corpus() {
        builder.add(doc);
    }
    builder
}

/// Serves a segment, returning its address and a handle to kill it.
async fn serve(segment: Segment) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    let segment = Arc::new(segment);
    let handle = tokio::spawn(async move {
        let _ = shard::serve(listener, segment).await;
    });
    (address, handle)
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("indexander-repl-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir
}

#[tokio::test]
async fn a_fetched_replica_is_byte_identical() {
    let source = Segment::from_bytes(built().encode()).expect("segment");
    let digest = source.digest();
    let (address, task) = serve(source).await;

    let dir = scratch("identical");
    let path = dir.join("replica.ixdr");
    let info = fetch_segment(&address, &path).await.expect("fetch");
    task.abort();

    assert_eq!(info.digest, digest);
    let replica = Segment::open(&path).expect("open replica");
    assert_eq!(replica.digest(), digest);
    assert!(
        replica.verify(),
        "the replica does not match its own digest"
    );
    assert_eq!(replica.as_bytes(), built().encode());

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn a_replica_answers_exactly_like_its_source() {
    let source = Segment::from_bytes(built().encode()).expect("segment");
    let (address, task) = serve(source).await;

    let dir = scratch("answers");
    let path = dir.join("replica.ixdr");
    fetch_segment(&address, &path).await.expect("fetch");
    task.abort();

    let original = Segment::from_bytes(built().encode()).expect("segment");
    let replica = Segment::open(&path).expect("open");
    for text in [
        "motor",
        "replicas numero 42",
        "\"busqueda distribuido\"",
        "motor -numero",
    ] {
        let parsed = query::parse(text);
        assert_eq!(
            search(&replica, &parsed, 10).expect("search"),
            search(&original, &parsed, 10).expect("search"),
            "replica disagreed on {text:?}"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn a_failed_transfer_leaves_nothing_behind() {
    // Nothing is listening, so the fetch fails before it writes anything that
    // could later be opened as a segment.
    let dir = scratch("failed");
    let path = dir.join("replica.ixdr");
    assert!(fetch_segment("127.0.0.1:1", &path).await.is_err());
    assert!(!path.exists(), "a failed fetch left a file behind");

    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .expect("readdir")
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(leftovers.is_empty(), "left behind {leftovers:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn a_dead_replica_is_not_a_dead_shard() {
    // The point of the whole exercise. Two replicas of one shard, the first
    // one down: the query is answered, and answered identically.
    let segment = Segment::from_bytes(built().encode()).expect("segment");
    let (live, task) = serve(segment).await;

    let coordinator =
        Coordinator::connect_replicated(&[vec!["127.0.0.1:1".to_owned(), live.clone()]])
            .await
            .expect("a live replica should have been found");
    assert_eq!(coordinator.shard_count(), 1);

    let hits = coordinator.search("motor", 5).await.expect("search");
    assert_eq!(hits.len(), 5);
    task.abort();
}

#[tokio::test]
async fn a_shard_with_every_replica_down_still_fails_the_query() {
    // Replication buys tolerance of a lost copy, not of a lost shard. Losing
    // every copy of a shard is a fifth of the corpus silently missing, and
    // that must still be an error.
    let segment = Segment::from_bytes(built().encode()).expect("segment");
    let (live, task) = serve(segment).await;

    let result = Coordinator::connect_replicated(&[
        vec![live.clone()],
        vec!["127.0.0.1:1".to_owned(), "127.0.0.1:2".to_owned()],
    ])
    .await;
    task.abort();

    let error = result.expect_err("a shard with no live replica must fail");
    let text = format!("{error}");
    assert!(text.contains("127.0.0.1:1"), "got {text}");
    assert!(
        text.contains("127.0.0.1:2"),
        "every attempt should be reported: {text}"
    );
}

#[tokio::test]
async fn replicas_and_single_addresses_give_the_same_answers() {
    let a = Segment::from_bytes(built().encode()).expect("segment");
    let b = Segment::from_bytes(built().encode()).expect("segment");
    let (first, task_a) = serve(a).await;
    let (second, task_b) = serve(b).await;

    let plain = Coordinator::connect(std::slice::from_ref(&first))
        .await
        .expect("connect");
    let replicated = Coordinator::connect_replicated(&[vec![first, second]])
        .await
        .expect("connect");

    let expected = plain.search("motor distribuido", 5).await.expect("search");
    let got = replicated
        .search("motor distribuido", 5)
        .await
        .expect("search");
    assert_eq!(expected, got);

    task_a.abort();
    task_b.abort();
}

#[tokio::test]
async fn an_empty_replica_list_is_refused() {
    assert!(Coordinator::connect_replicated(&[]).await.is_err());
    assert!(
        Coordinator::connect_replicated(&[Vec::new()])
            .await
            .is_err()
    );
}
