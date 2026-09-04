//! PageRank across real sockets must equal PageRank in one process.
//!
//! The in-process version of this property is asserted in
//! `indexander_rank::distributed`'s tests. This one adds the transport, which
//! is where an ordering mistake would live: dangling mass summed too late,
//! emit and absorb interleaved, convergence decided per shard. None of those
//! error — the run finishes, produces numbers, and the numbers are wrong.

use std::collections::HashMap;
use std::sync::Arc;

use indexander_cluster::ranking::{RankCoordinator, RankShard};
use indexander_rank::distributed::ShardGraph;
use indexander_rank::graph::{GraphBuilder, NodeId};
use indexander_rank::pagerank::{Options, pagerank};
use tokio::net::TcpListener;

fn route(uri: &str, shards: usize) -> usize {
    if shards <= 1 {
        return 0;
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in uri.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    usize::try_from(hash % shards as u64).unwrap_or(0)
}

/// A graph with a hub, chains, a cycle and dead ends.
fn edges() -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for i in 0..30 {
        pairs.push((format!("crowd{i}"), "hub".to_owned()));
    }
    pairs.push(("hub".to_owned(), "endorsed".to_owned()));
    for i in 0..12 {
        pairs.push((format!("chain{i}"), format!("chain{}", i + 1)));
    }
    pairs.push(("cycle0".to_owned(), "cycle1".to_owned()));
    pairs.push(("cycle1".to_owned(), "cycle0".to_owned()));
    for i in 0..8 {
        pairs.push((format!("crowd{i}"), format!("deadend{i}")));
    }
    pairs
}

fn single_process(options: &Options) -> HashMap<String, f32> {
    let mut builder = GraphBuilder::new();
    for (from, to) in edges() {
        builder.edge(&from, &to);
    }
    let graph = builder.build();
    let ranks = pagerank(&graph, options);
    graph
        .uris()
        .iter()
        .enumerate()
        .map(|(i, uri)| (uri.clone(), ranks.score(NodeId(u32::try_from(i).unwrap()))))
        .collect()
}

/// Starts `shards` rank shard processes and returns their addresses and the
/// cluster-wide node count.
async fn start(shards: usize) -> (Vec<String>, usize) {
    let pairs = edges();
    let mut all: Vec<String> = Vec::new();
    for (from, to) in &pairs {
        all.push(from.clone());
        all.push(to.clone());
    }
    all.sort();
    all.dedup();
    let total = all.len();

    let mut graphs: Vec<ShardGraph> = (0..shards).map(|_| ShardGraph::new()).collect();
    // Ownership before edges: a page with no outgoing links still holds rank,
    // and a page nobody owns holds none.
    for uri in &all {
        graphs[route(uri, shards)].own(uri);
    }
    for (from, to) in &pairs {
        graphs[route(from, shards)].link(from, to);
    }

    let mut addresses = Vec::new();
    for (index, graph) in graphs.into_iter().enumerate() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        addresses.push(listener.local_addr().expect("addr").to_string());
        let shard = Arc::new(RankShard::new(graph, index));
        tokio::spawn(async move {
            let _ = shard.serve(listener).await;
        });
    }
    (addresses, total)
}

async fn assert_agrees(shards: usize) {
    let options = Options::default();
    let (addresses, total) = start(shards).await;
    let coordinator = RankCoordinator::connect(&addresses).await.expect("connect");
    assert_eq!(coordinator.shard_count(), shards);

    let outcome = coordinator
        .run(total, &options, |uri| route(uri, shards))
        .await
        .expect("run");

    assert!(
        outcome.converged,
        "did not converge in {} iterations",
        outcome.iterations
    );

    let expected = single_process(&options);
    assert_eq!(outcome.ranks.len(), expected.len());
    for (uri, want) in &expected {
        let got = outcome.ranks.get(uri).copied().unwrap_or(0.0);
        assert!(
            (got - want).abs() < 1e-4,
            "{uri}: {got} across {shards} shards, {want} in one process"
        );
    }
}

#[tokio::test]
async fn one_shard_over_a_socket_matches_one_process() {
    assert_agrees(1).await;
}

#[tokio::test]
async fn several_shards_over_sockets_match_one_process() {
    for shards in [2usize, 3, 5] {
        assert_agrees(shards).await;
    }
}

#[tokio::test]
async fn the_ranking_order_survives_the_wire() {
    let options = Options::default();
    let (addresses, total) = start(4).await;
    let coordinator = RankCoordinator::connect(&addresses).await.expect("connect");
    let outcome = coordinator
        .run(total, &options, |uri| route(uri, 4))
        .await
        .expect("run");

    let order = |map: &HashMap<String, f32>| {
        let mut v: Vec<(String, f32)> = map.iter().map(|(k, s)| (k.clone(), *s)).collect();
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        v.into_iter().map(|(k, _)| k).collect::<Vec<_>>()
    };
    assert_eq!(order(&single_process(&options)), order(&outcome.ranks));
}

#[tokio::test]
async fn ranks_sum_to_one_across_the_cluster() {
    let (addresses, total) = start(5).await;
    let coordinator = RankCoordinator::connect(&addresses).await.expect("connect");
    let outcome = coordinator
        .run(total, &Options::default(), |uri| route(uri, 5))
        .await
        .expect("run");
    let sum: f32 = outcome.ranks.values().sum();
    assert!((sum - 1.0).abs() < 1e-3, "ranks summed to {sum}");
}

#[tokio::test]
async fn a_missing_shard_fails_the_run() {
    // A ranking run short a shard is not a slightly worse ranking; it is a
    // graph with a piece cut out, and every rank in it is wrong.
    let (mut addresses, _) = start(2).await;
    addresses.push("127.0.0.1:1".to_owned());
    let result = RankCoordinator::connect(&addresses).await;
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("127.0.0.1:1"));
}

#[tokio::test]
async fn a_coordinator_needs_at_least_one_shard() {
    assert!(RankCoordinator::connect(&[]).await.is_err());
}

#[tokio::test]
async fn a_shard_refuses_a_second_initialisation() {
    // Two runs against the same shard would rank the second against the
    // first's leftovers.
    let (addresses, total) = start(1).await;
    let coordinator = RankCoordinator::connect(&addresses).await.expect("connect");
    let options = Options::default();
    coordinator
        .run(total, &options, |_| 0)
        .await
        .expect("first run");
    let second = coordinator.run(total, &options, |_| 0).await;
    assert!(second.is_err(), "a shard let itself be initialised twice");
}

#[tokio::test]
async fn a_capped_run_reports_that_it_did_not_converge() {
    let options = Options {
        max_iterations: 2,
        ..Options::default()
    };
    let (addresses, total) = start(3).await;
    let coordinator = RankCoordinator::connect(&addresses).await.expect("connect");
    let outcome = coordinator
        .run(total, &options, |uri| route(uri, 3))
        .await
        .expect("run");
    assert_eq!(outcome.iterations, 2);
    assert!(!outcome.converged);
}
