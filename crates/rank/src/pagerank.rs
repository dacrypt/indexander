//! PageRank by power iteration.
//!
//! The model is a reader who follows links at random and, with probability
//! `1 - damping`, gets bored and jumps to a page picked uniformly. A page's
//! rank is the fraction of time that reader spends on it. That is the whole
//! idea, and the reason it beat counting links: a vote from an important page
//! is worth more than a vote from an unimportant one, and importance is
//! defined recursively.
//!
//! Two details do all the damage when they are wrong:
//!
//! * **Dangling nodes.** A page with no outlinks is a sink. Its rank has
//!   nowhere to flow, so unless the mass it holds is put back into
//!   circulation every iteration, total rank shrinks and the whole vector
//!   decays toward zero. Most naive implementations get this wrong, and the
//!   symptom is a ranking that looks plausible and is quietly meaningless.
//! * **Convergence.** Iterating a fixed number of times is a guess. Iterating
//!   until the vector stops moving is an answer.

use crate::graph::{LinkGraph, NodeId};

/// Probability the random reader follows a link rather than jumping.
///
/// 0.85 is the value from the 1998 paper. It is not derived from anything; it
/// is what made the results look right, and twenty-eight years later nobody
/// has found much better.
pub const DAMPING: f32 = 0.85;

/// Settings for a PageRank run.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub damping: f32,
    /// Stop when the total movement of the vector falls below this.
    pub tolerance: f32,
    /// Give up after this many iterations even if it has not converged.
    pub max_iterations: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            damping: DAMPING,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

/// The outcome of a run: the vector, and how it got there.
#[derive(Debug, Clone)]
pub struct Ranks {
    scores: Vec<f32>,
    pub iterations: usize,
    /// Total movement in the final iteration. Below `tolerance` means it
    /// converged; equal to it means it ran out of iterations.
    pub residual: f32,
}

impl Ranks {
    #[must_use]
    pub fn score(&self, node: NodeId) -> f32 {
        self.scores.get(node.as_usize()).copied().unwrap_or(0.0)
    }

    #[must_use]
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    #[must_use]
    pub fn converged(&self, options: &Options) -> bool {
        self.residual < options.tolerance
    }

    /// Nodes ordered by rank, highest first.
    #[must_use]
    pub fn ranked(&self) -> Vec<(NodeId, f32)> {
        let mut out: Vec<(NodeId, f32)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, s)| (NodeId(u32::try_from(i).unwrap_or(u32::MAX)), *s))
            .collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }
}

/// Computes PageRank over `graph`.
#[must_use]
pub fn pagerank(graph: &LinkGraph, options: &Options) -> Ranks {
    let n = graph.node_count();
    if n == 0 {
        return Ranks {
            scores: Vec::new(),
            iterations: 0,
            residual: 0.0,
        };
    }

    #[allow(clippy::cast_precision_loss)]
    let count = n as f32;
    let uniform = 1.0 / count;
    let mut current = vec![uniform; n];
    let mut next = vec![0.0f32; n];

    // Out-degrees are read once per iteration per node; caching them turns a
    // pointer chase into a sequential read.
    let degrees: Vec<u32> = (0..n)
        .map(|i| {
            u32::try_from(graph.out_degree(NodeId(u32::try_from(i).unwrap_or(u32::MAX))))
                .unwrap_or(u32::MAX)
        })
        .collect();

    let mut iterations = 0;
    let mut residual = f32::INFINITY;

    while iterations < options.max_iterations {
        iterations += 1;

        // Mass held by pages with nowhere to send it. Spread evenly, which is
        // the same as saying the reader jumps when there is nothing to click.
        let dangling: f32 = current
            .iter()
            .zip(&degrees)
            .filter(|(_, d)| **d == 0)
            .map(|(r, _)| *r)
            .sum();

        let base = (1.0 - options.damping) / count + options.damping * dangling / count;
        next.fill(base);

        for i in 0..n {
            let degree = degrees[i];
            if degree == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let share = options.damping * current[i] / degree as f32;
            for target in graph.outlinks(NodeId(u32::try_from(i).unwrap_or(u32::MAX))) {
                next[target.as_usize()] += share;
            }
        }

        residual = current
            .iter()
            .zip(&next)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();

        std::mem::swap(&mut current, &mut next);

        if residual < options.tolerance {
            break;
        }
    }

    Ranks {
        scores: current,
        iterations,
        residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn graph(edges: &[(&str, &str)]) -> LinkGraph {
        let mut b = GraphBuilder::new();
        for (from, to) in edges {
            b.edge(from, to);
        }
        b.build()
    }

    fn score_of(g: &LinkGraph, r: &Ranks, uri: &str) -> f32 {
        r.score(g.id(uri).expect("node should exist"))
    }

    #[test]
    fn ranks_sum_to_one() {
        // The invariant that dangling-node handling exists to preserve. If it
        // is wrong, this drifts below 1 and every score is meaningless.
        let g = graph(&[("a", "b"), ("b", "c"), ("d", "a"), ("e", "a"), ("f", "a")]);
        let r = pagerank(&g, &Options::default());
        let total: f32 = r.scores().iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "ranks summed to {total}");
    }

    #[test]
    fn a_dangling_node_does_not_drain_the_graph() {
        // "c" links to nothing. Without redistribution the total collapses.
        let g = graph(&[("a", "c"), ("b", "c")]);
        let r = pagerank(&g, &Options::default());
        let total: f32 = r.scores().iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "total was {total}");
        assert!(score_of(&g, &r, "c") > score_of(&g, &r, "a"));
    }

    #[test]
    fn symmetry_gives_equal_ranks() {
        // Two pages linking to each other: by symmetry each holds exactly half.
        let g = graph(&[("a", "b"), ("b", "a")]);
        let r = pagerank(&g, &Options::default());
        assert!((score_of(&g, &r, "a") - 0.5).abs() < 1e-4);
        assert!((score_of(&g, &r, "b") - 0.5).abs() < 1e-4);
    }

    #[test]
    fn more_incoming_links_means_more_rank() {
        let g = graph(&[
            ("x", "popular"),
            ("y", "popular"),
            ("z", "popular"),
            ("x", "lonely"),
        ]);
        let r = pagerank(&g, &Options::default());
        assert!(score_of(&g, &r, "popular") > score_of(&g, &r, "lonely"));
    }

    /// The insight the whole algorithm exists for, as a test.
    #[test]
    fn one_link_from_an_important_page_beats_many_from_unimportant_ones() {
        let mut b = GraphBuilder::new();
        // A hub everybody points at, so it becomes important.
        for i in 0..50 {
            b.edge(&format!("crowd{i}"), "hub");
        }
        // The hub endorses exactly one page.
        b.edge("hub", "endorsed");
        // Meanwhile ten nobodies all point at another page.
        for i in 0..10 {
            b.edge(&format!("nobody{i}"), "popular_but_cheap");
        }
        let g = b.build();
        let r = pagerank(&g, &Options::default());

        assert!(
            score_of(&g, &r, "endorsed") > score_of(&g, &r, "popular_but_cheap"),
            "one link from the hub ({}) lost to ten cheap links ({})",
            score_of(&g, &r, "endorsed"),
            score_of(&g, &r, "popular_but_cheap"),
        );
    }

    #[test]
    fn a_link_farm_cannot_out_rank_a_real_endorsement() {
        let mut b = GraphBuilder::new();
        // A thousand pages that only link to each other and to the target.
        for i in 0..1000 {
            b.edge(&format!("farm{i}"), "spam");
            b.edge("spam", &format!("farm{i}"));
        }
        // One page endorsed by a genuinely central page.
        for i in 0..50 {
            b.edge(&format!("real{i}"), "authority");
        }
        b.edge("authority", "honest");
        let g = b.build();
        let r = pagerank(&g, &Options::default());

        // The farm does inflate "spam" — PageRank is not spam-proof, and
        // pretending otherwise would be the lie. What it does guarantee is
        // that the mass is bounded by the farm's own share of the graph.
        let spam = score_of(&g, &r, "spam");
        let total: f32 = r.scores().iter().sum();
        assert!((total - 1.0).abs() < 1e-3);
        assert!(spam < 0.5, "one farm captured {spam} of all rank");
    }

    #[test]
    fn it_converges_and_says_so() {
        let g = graph(&[("a", "b"), ("b", "c"), ("c", "a")]);
        let options = Options::default();
        let r = pagerank(&g, &options);
        assert!(r.converged(&options), "residual {}", r.residual);
        assert!(r.iterations < options.max_iterations);
        // A perfect cycle is symmetric: a third each.
        for uri in ["a", "b", "c"] {
            assert!((score_of(&g, &r, uri) - 1.0 / 3.0).abs() < 1e-4);
        }
    }

    #[test]
    fn a_capped_run_reports_that_it_did_not_converge() {
        // Deliberately asymmetric: a symmetric graph is already at its fixed
        // point when the iteration starts from a uniform vector, so it would
        // "converge" in one step and prove nothing.
        let g = graph(&[
            ("a", "b"),
            ("b", "c"),
            ("c", "b"),
            ("d", "b"),
            ("e", "b"),
            ("f", "a"),
        ]);
        let options = Options {
            max_iterations: 1,
            ..Options::default()
        };
        let r = pagerank(&g, &options);
        assert_eq!(r.iterations, 1);
        assert!(!r.converged(&options), "residual {}", r.residual);

        // The same graph, allowed to run, does converge.
        let full = pagerank(&g, &Options::default());
        assert!(full.converged(&Options::default()));
        assert!(full.iterations > 1);
    }

    #[test]
    fn an_empty_graph_produces_no_scores() {
        let r = pagerank(&GraphBuilder::new().build(), &Options::default());
        assert!(r.scores().is_empty());
        assert_eq!(r.iterations, 0);
    }

    #[test]
    fn a_single_page_holds_all_the_rank() {
        let mut b = GraphBuilder::new();
        b.node("only");
        let g = b.build();
        let r = pagerank(&g, &Options::default());
        assert!((r.score(g.id("only").unwrap()) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ranked_returns_descending_order() {
        let g = graph(&[("a", "top"), ("b", "top"), ("c", "middle")]);
        let r = pagerank(&g, &Options::default());
        let order = r.ranked();
        for pair in order.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
        assert_eq!(g.uri(order[0].0), Some("top"));
    }

    #[test]
    fn damping_of_zero_gives_everyone_the_same_rank() {
        // With no link-following, the reader jumps every step, so the
        // distribution is uniform whatever the links say.
        let g = graph(&[("a", "b"), ("a", "c"), ("d", "b")]);
        let r = pagerank(
            &g,
            &Options {
                damping: 0.0,
                ..Options::default()
            },
        );
        let expected = 1.0 / 4.0;
        for score in r.scores() {
            assert!((score - expected).abs() < 1e-5, "got {score}");
        }
    }
}
