//! Partitioning the graph must not change the ranking.
//!
//! This is the only property that matters. A distributed PageRank that is
//! merely *close* is a search engine that returns different results depending
//! on how many machines happened to be running, and the difference is
//! invisible without something to compare against. So every test here compares
//! against the single-process answer.

use std::collections::HashMap;

use indexander_rank::distributed::{ShardGraph, ShardRanker, run};
use indexander_rank::graph::GraphBuilder;
use indexander_rank::pagerank::{Options, pagerank};

/// A deterministic partition: the same URL always lands in the same shard.
#[allow(clippy::cast_possible_truncation)]
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

/// Ranks computed in one process, keyed by uri.
fn single_process(edges: &[(String, String)], options: &Options) -> HashMap<String, f32> {
    let mut builder = GraphBuilder::new();
    for (from, to) in edges {
        builder.edge(from, to);
    }
    let graph = builder.build();
    let ranks = pagerank(&graph, options);
    graph
        .uris()
        .iter()
        .enumerate()
        .map(|(i, uri)| {
            (
                uri.clone(),
                ranks.score(indexander_rank::graph::NodeId(u32::try_from(i).unwrap())),
            )
        })
        .collect()
}

/// The same graph, split across `shards`, run to convergence.
fn distributed(
    edges: &[(String, String)],
    shards: usize,
    options: &Options,
) -> (HashMap<String, f32>, usize) {
    // Every node in the graph, so each shard can be told the global count.
    let mut all: Vec<String> = Vec::new();
    for (from, to) in edges {
        all.push(from.clone());
        all.push(to.clone());
    }
    all.sort();
    all.dedup();
    let total = all.len();

    let mut graphs: Vec<ShardGraph> = (0..shards).map(|_| ShardGraph::new()).collect();
    // Ownership first, so a node with no outgoing edges still belongs to
    // somebody: a dangling page holds rank and has to be counted.
    for uri in &all {
        graphs[route(uri, shards)].own(uri);
    }
    for (from, to) in edges {
        graphs[route(from, shards)].link(from, to);
    }

    let mut ranked: Vec<ShardRanker> = graphs
        .into_iter()
        .map(|g| ShardRanker::new(g, total, *options))
        .collect();

    let iterations = run(&mut ranked, |uri| route(uri, shards), options);

    let mut out = HashMap::new();
    for shard in &ranked {
        for (uri, score) in shard.ranks() {
            out.insert(uri.to_owned(), score);
        }
    }
    (out, iterations)
}

fn edges(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect()
}

/// A graph with a hub, a chain, a cycle and dead ends: every shape that
/// behaves differently when it straddles a partition.
fn mixed_graph() -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for i in 0..40 {
        pairs.push((format!("crowd{i}"), "hub".to_owned()));
    }
    pairs.push(("hub".to_owned(), "endorsed".to_owned()));
    for i in 0..15 {
        pairs.push((format!("chain{i}"), format!("chain{}", i + 1)));
    }
    pairs.push(("cycle0".to_owned(), "cycle1".to_owned()));
    pairs.push(("cycle1".to_owned(), "cycle2".to_owned()));
    pairs.push(("cycle2".to_owned(), "cycle0".to_owned()));
    for i in 0..10 {
        pairs.push((format!("crowd{i}"), format!("deadend{i}")));
    }
    pairs
}

fn assert_agrees(edges: &[(String, String)], shards: usize) {
    let options = Options::default();
    let single = single_process(edges, &options);
    let (spread, _) = distributed(edges, shards, &options);

    assert_eq!(
        spread.len(),
        single.len(),
        "{shards} shards held {} nodes, one process had {}",
        spread.len(),
        single.len()
    );

    for (uri, expected) in &single {
        let got = spread.get(uri).copied().unwrap_or(0.0);
        assert!(
            (got - expected).abs() < 1e-4,
            "{uri} ranked {got} across {shards} shards, {expected} in one process"
        );
    }
}

#[test]
fn one_shard_matches_one_process() {
    assert_agrees(&mixed_graph(), 1);
}

#[test]
fn many_shards_match_one_process() {
    for shards in [2usize, 3, 5, 8, 17] {
        assert_agrees(&mixed_graph(), shards);
    }
}

#[test]
fn the_ranking_order_survives_partitioning() {
    // Scores agreeing to four decimals is one thing; the order is what a user
    // sees.
    let options = Options::default();
    let graph = mixed_graph();
    let single = single_process(&graph, &options);
    let (spread, _) = distributed(&graph, 7, &options);

    let order = |map: &HashMap<String, f32>| {
        let mut v: Vec<(String, f32)> = map.iter().map(|(k, s)| (k.clone(), *s)).collect();
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        v.into_iter().map(|(k, _)| k).collect::<Vec<_>>()
    };
    assert_eq!(order(&single), order(&spread));
}

#[test]
fn ranks_still_sum_to_one_across_shards() {
    // The invariant dangling-mass handling exists to preserve, now with the
    // dangling pages scattered across shards.
    let options = Options::default();
    for shards in [1usize, 4, 9] {
        let (spread, _) = distributed(&mixed_graph(), shards, &options);
        let total: f32 = spread.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "{shards} shards summed to {total}"
        );
    }
}

#[test]
fn a_link_across_a_partition_still_carries_rank() {
    // Two pages, deliberately in different shards, one linking to the other.
    // If boundary exchange were missing this would look like two isolated
    // pages and they would rank equally.
    let graph = edges(&[("alpha", "beta"), ("gamma", "beta"), ("delta", "beta")]);
    let options = Options::default();
    let (spread, _) = distributed(&graph, 4, &options);

    let beta = spread["beta"];
    let alpha = spread["alpha"];
    assert!(
        beta > alpha * 1.5,
        "beta ranked {beta} against alpha's {alpha}; rank did not cross the partition"
    );
}

#[test]
fn a_dangling_page_in_one_shard_feeds_every_shard() {
    // "sink" has no outlinks. Its mass must be redistributed over the whole
    // cluster, not over whichever shard happens to hold it.
    let graph = edges(&[("a", "sink"), ("b", "c"), ("c", "b")]);
    let options = Options::default();
    let single = single_process(&graph, &options);
    let (spread, _) = distributed(&graph, 4, &options);

    for (uri, expected) in &single {
        assert!(
            (spread[uri] - expected).abs() < 1e-4,
            "{uri}: {} across shards, {expected} in one process",
            spread[uri]
        );
    }
}

#[test]
fn convergence_is_decided_on_the_sum_not_per_shard() {
    // A graph where one shard settles immediately and another does not: a
    // per-shard stopping rule would freeze the quiet one against stale
    // numbers from the busy one.
    let mut pairs = edges(&[("still0", "still1"), ("still1", "still0")]);
    for i in 0..30 {
        pairs.push((format!("busy{i}"), format!("busy{}", (i + 1) % 30)));
    }
    pairs.push(("busy0".to_owned(), "still0".to_owned()));

    let options = Options::default();
    let single = single_process(&pairs, &options);
    let (spread, iterations) = distributed(&pairs, 5, &options);

    assert!(
        iterations > 1,
        "converged in one iteration, nothing was tested"
    );
    for (uri, expected) in &single {
        assert!(
            (spread[uri] - expected).abs() < 1e-4,
            "{uri}: {} vs {expected} after {iterations} iterations",
            spread[uri]
        );
    }
}

#[test]
fn an_empty_cluster_and_an_empty_graph_are_handled() {
    let options = Options::default();
    let mut none: Vec<ShardRanker> = Vec::new();
    assert_eq!(run(&mut none, |_| 0, &options), 0);

    let mut one = vec![ShardRanker::new(ShardGraph::new(), 0, options)];
    let _ = run(&mut one, |_| 0, &options);
    assert!(one[0].ranks().is_empty());
}

#[test]
fn self_links_are_dropped_the_same_way_as_in_one_process() {
    let graph = edges(&[("a", "a"), ("a", "b"), ("b", "a")]);
    let options = Options::default();
    let single = single_process(&graph, &options);
    let (spread, _) = distributed(&graph, 3, &options);
    for (uri, expected) in &single {
        assert!((spread[uri] - expected).abs() < 1e-4, "{uri}");
    }
}
