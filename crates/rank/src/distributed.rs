//! PageRank when the graph does not fit in one process.
//!
//! Everything else in this engine shards cleanly: a shard indexes its own
//! documents, answers about its own postings, and needs to be told only the
//! corpus-wide term statistics. PageRank does not. It is a *global fixed
//! point*: a page's rank depends on the ranks of the pages linking to it,
//! wherever those live, and the answer is not the concatenation of local
//! answers.
//!
//! So each iteration has three exchanges, and skipping any of them gives a
//! result that looks plausible and is wrong:
//!
//! * **Rank across the boundary.** A shard sends every other shard the mass
//!   flowing along the edges it owns that point there. Without this, a link
//!   between shards counts for nothing and the whole point of PageRank —
//!   that importance flows through links — stops at a partition nobody chose
//!   for editorial reasons.
//! * **Dangling mass, globally.** A page with no outlinks holds mass that must
//!   be put back into circulation across *every* node, not just the ones in
//!   its shard. Summing it locally concentrates rank in whichever shard
//!   happens to hold the most dead ends.
//! * **Convergence, globally.** A shard cannot decide the iteration is done. It
//!   can be perfectly still while another is still moving, and stopping early
//!   leaves it ranked against a neighbour's stale numbers.
//!
//! This module is the algorithm, with the exchanges as plain data. Whether
//! they cross a socket or a function call is the caller's business — the same
//! separation that lets a one-shard search take the distributed path.

use std::collections::HashMap;

use crate::pagerank::Options;

/// One shard's slice of the graph.
///
/// Nodes are named by URI rather than by a dense id, because a dense id is
/// only dense within a shard and the whole difficulty here is edges that leave
/// one. A production graph would assign global ids once and send those; this
/// sends the names, which is heavier on the wire and impossible to get subtly
/// wrong.
#[derive(Debug, Default)]
pub struct ShardGraph {
    uris: Vec<String>,
    index: HashMap<String, usize>,
    /// Targets of each owned node, in any shard.
    outlinks: Vec<Vec<String>>,
}

impl ShardGraph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares that this shard owns `uri`.
    pub fn own(&mut self, uri: &str) -> usize {
        if let Some(id) = self.index.get(uri) {
            return *id;
        }
        let id = self.uris.len();
        self.uris.push(uri.to_owned());
        self.index.insert(uri.to_owned(), id);
        self.outlinks.push(Vec::new());
        id
    }

    /// Records a link from an owned page to anywhere.
    ///
    /// Self-links are dropped, as they are in the single-process graph: a page
    /// voting for itself is not evidence of anything.
    pub fn link(&mut self, from: &str, to: &str) {
        if from == to {
            return;
        }
        let id = self.own(from);
        if !self.outlinks[id].iter().any(|t| t == to) {
            self.outlinks[id].push(to.to_owned());
        }
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.uris.len()
    }

    #[must_use]
    pub fn uris(&self) -> &[String] {
        &self.uris
    }
}

/// Rank flowing from one shard to another, for one iteration.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Boundary {
    /// `(target uri, contribution)`, for targets this shard does not own.
    pub contributions: Vec<(String, f32)>,
}

/// What the coordinator needs from every shard each iteration, and what it
/// hands back.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Round {
    /// Mass held by this shard's pages that have no outlinks.
    pub dangling: f32,
    /// How far this shard's vector moved.
    pub residual: f32,
}

/// One shard's participant in a distributed run.
#[derive(Debug)]
pub struct ShardRanker {
    graph: ShardGraph,
    /// This shard's current ranks, by local index.
    ranks: Vec<f32>,
    /// Accumulator for the iteration being built.
    incoming: Vec<f32>,
    options: Options,
    /// Nodes across every shard. Ranks are a distribution over all of them.
    total_nodes: usize,
}

impl ShardRanker {
    /// `total_nodes` is the count across the whole cluster: a shard cannot
    /// know it, and a shard that guesses it produces a vector that does not
    /// sum to one.
    #[must_use]
    pub fn new(graph: ShardGraph, total_nodes: usize, options: Options) -> Self {
        let local = graph.node_count();
        #[allow(clippy::cast_precision_loss)]
        let uniform = if total_nodes == 0 {
            0.0
        } else {
            1.0 / total_nodes as f32
        };
        Self {
            graph,
            ranks: vec![uniform; local],
            incoming: vec![0.0; local],
            options,
            total_nodes,
        }
    }

    /// The mass this shard's dangling pages are holding right now.
    #[must_use]
    pub fn dangling(&self) -> f32 {
        self.ranks
            .iter()
            .enumerate()
            .filter(|(i, _)| self.graph.outlinks[*i].is_empty())
            .map(|(_, r)| *r)
            .sum()
    }

    /// Step one of an iteration: push rank along every edge.
    ///
    /// Contributions to pages this shard owns are kept; the rest come back as
    /// one bundle per destination, keyed by whatever `route` says.
    pub fn emit<F: Fn(&str) -> usize>(&mut self, route: F, shard_count: usize) -> Vec<Boundary> {
        self.incoming.iter_mut().for_each(|slot| *slot = 0.0);
        let mut outgoing = vec![Boundary::default(); shard_count.max(1)];

        for (i, targets) in self.graph.outlinks.iter().enumerate() {
            if targets.is_empty() {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let share = self.options.damping * self.ranks[i] / targets.len() as f32;
            for target in targets {
                if let Some(local) = self.graph.index.get(target) {
                    self.incoming[*local] += share;
                } else {
                    let destination = route(target).min(shard_count.saturating_sub(1));
                    outgoing[destination]
                        .contributions
                        .push((target.clone(), share));
                }
            }
        }
        outgoing
    }

    /// Step two: fold in what other shards sent.
    pub fn absorb(&mut self, boundary: &Boundary) {
        for (uri, share) in &boundary.contributions {
            if let Some(local) = self.graph.index.get(uri) {
                self.incoming[*local] += share;
            }
            // A contribution to a page nobody owns is dropped. That can only
            // happen if routing disagrees between shards, and silently
            // creating the page here would make two shards own it.
        }
    }

    /// Step three: apply the iteration, given the cluster-wide dangling mass.
    ///
    /// Returns how far this shard moved, for the coordinator to sum.
    pub fn apply(&mut self, global_dangling: f32) -> Round {
        if self.total_nodes == 0 {
            return Round::default();
        }
        #[allow(clippy::cast_precision_loss)]
        let count = self.total_nodes as f32;
        let base =
            (1.0 - self.options.damping) / count + self.options.damping * global_dangling / count;

        let mut residual = 0.0f32;
        for (rank, arrived) in self.ranks.iter_mut().zip(&self.incoming) {
            let next = base + arrived;
            residual += (*rank - next).abs();
            *rank = next;
        }
        Round {
            dangling: self.dangling(),
            residual,
        }
    }

    /// This shard's ranks, by uri.
    #[must_use]
    pub fn ranks(&self) -> Vec<(&str, f32)> {
        self.graph
            .uris
            .iter()
            .map(String::as_str)
            .zip(self.ranks.iter().copied())
            .collect()
    }

    #[must_use]
    pub fn options(&self) -> Options {
        self.options
    }
}

/// Runs a whole distributed computation in one process.
///
/// Useful for testing, and it *is* the algorithm: a networked version replaces
/// the three loops below with three exchanges and changes nothing else. That
/// the two produce the same numbers is what
/// `crates/rank/tests/distributed.rs` asserts.
#[must_use]
pub fn run<F: Fn(&str) -> usize + Copy>(
    shards: &mut [ShardRanker],
    route: F,
    options: &Options,
) -> usize {
    let shard_count = shards.len();
    if shard_count == 0 {
        return 0;
    }

    for iteration in 1..=options.max_iterations {
        // Dangling mass is summed across every shard before anyone applies it.
        let global_dangling: f32 = shards.iter().map(ShardRanker::dangling).sum();

        // Everyone emits, then everyone absorbs. Two passes, because a shard
        // must not see this iteration's contributions while still producing
        // its own from last iteration's ranks.
        let mut mail: Vec<Vec<Boundary>> = Vec::with_capacity(shard_count);
        for shard in shards.iter_mut() {
            mail.push(shard.emit(route, shard_count));
        }
        for (from, bundles) in mail.iter().enumerate() {
            for (to, boundary) in bundles.iter().enumerate() {
                if from != to && !boundary.contributions.is_empty() {
                    shards[to].absorb(boundary);
                }
            }
        }

        let residual: f32 = shards
            .iter_mut()
            .map(|s| s.apply(global_dangling).residual)
            .sum();

        // Convergence is decided on the sum. A shard that has stopped moving
        // while another has not is not done.
        if residual < options.tolerance {
            return iteration;
        }
    }
    options.max_iterations
}
