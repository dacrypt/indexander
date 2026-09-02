//! The link graph, stored as compressed sparse rows.
//!
//! A web graph is enormous and almost entirely empty: a page links to a
//! handful of others out of billions. Storing it as a matrix is impossible and
//! storing it as a list of `(from, to)` pairs makes iteration jump all over
//! memory. CSR stores the destinations of every node end to end in one array,
//! with an index saying where each node's run begins — so walking one node's
//! links is a sequential read, which is the only access pattern PageRank has.

use std::collections::HashMap;

/// A node in the link graph. Dense, assigned in insertion order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Accumulates edges, then freezes into a [`LinkGraph`].
///
/// Edges may name pages that were never crawled: a link to a page outside the
/// crawl still says something about it, and PageRank flows through it.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    ids: HashMap<String, NodeId>,
    uris: Vec<String>,
    edges: Vec<(NodeId, NodeId)>,
}

impl GraphBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the id for `uri`, assigning one if this is the first sighting.
    ///
    /// # Panics
    ///
    /// If more than `u32::MAX` distinct URLs are seen. A graph that large does
    /// not belong in one process; see `docs/DISTRIBUTION.md`.
    pub fn node(&mut self, uri: &str) -> NodeId {
        if let Some(id) = self.ids.get(uri) {
            return *id;
        }
        let id = NodeId(u32::try_from(self.uris.len()).expect("more than u32::MAX pages"));
        self.ids.insert(uri.to_owned(), id);
        self.uris.push(uri.to_owned());
        id
    }

    /// Records that `from` links to `to`.
    ///
    /// Self-links are dropped: a page voting for itself is not evidence of
    /// anything, and leaving them in lets a page pump its own score.
    pub fn edge(&mut self, from: &str, to: &str) {
        let (a, b) = (self.node(from), self.node(to));
        if a != b {
            self.edges.push((a, b));
        }
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.uris.len()
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Freezes the graph, deduplicating repeated edges.
    ///
    /// Ten links from one page to another are one vote, not ten. Without this
    /// a navigation menu repeated on every page would dominate the ranking.
    #[must_use]
    pub fn build(mut self) -> LinkGraph {
        self.edges.sort_unstable();
        self.edges.dedup();

        let node_count = self.uris.len();
        let mut offsets = vec![0u32; node_count + 1];
        for (from, _) in &self.edges {
            offsets[from.as_usize() + 1] += 1;
        }
        for i in 1..=node_count {
            offsets[i] += offsets[i - 1];
        }

        // Edges are already sorted by source, so a single pass fills the array.
        let targets: Vec<NodeId> = self.edges.iter().map(|(_, to)| *to).collect();

        LinkGraph {
            uris: self.uris,
            ids: self.ids,
            offsets,
            targets,
        }
    }
}

/// An immutable link graph.
#[derive(Debug, Clone)]
pub struct LinkGraph {
    uris: Vec<String>,
    ids: HashMap<String, NodeId>,
    /// `offsets[i]..offsets[i + 1]` is node `i`'s slice of `targets`.
    offsets: Vec<u32>,
    targets: Vec<NodeId>,
}

impl LinkGraph {
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.uris.len()
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.targets.len()
    }

    #[must_use]
    pub fn uri(&self, node: NodeId) -> Option<&str> {
        self.uris.get(node.as_usize()).map(String::as_str)
    }

    #[must_use]
    pub fn id(&self, uri: &str) -> Option<NodeId> {
        self.ids.get(uri).copied()
    }

    /// The nodes `node` links to.
    #[must_use]
    pub fn outlinks(&self, node: NodeId) -> &[NodeId] {
        let i = node.as_usize();
        if i + 1 >= self.offsets.len() {
            return &[];
        }
        let (from, to) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        &self.targets[from..to]
    }

    #[must_use]
    pub fn out_degree(&self, node: NodeId) -> usize {
        self.outlinks(node).len()
    }

    /// Every uri in the graph, indexed by node id.
    #[must_use]
    pub fn uris(&self) -> &[String] {
        &self.uris
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodes_are_assigned_once_per_uri() {
        let mut b = GraphBuilder::new();
        let a = b.node("http://a/");
        let again = b.node("http://a/");
        let other = b.node("http://b/");
        assert_eq!(a, again);
        assert_ne!(a, other);
        assert_eq!(b.node_count(), 2);
    }

    #[test]
    fn edges_land_in_the_right_rows() {
        let mut b = GraphBuilder::new();
        b.edge("http://a/", "http://b/");
        b.edge("http://a/", "http://c/");
        b.edge("http://b/", "http://c/");
        let g = b.build();

        let a = g.id("http://a/").unwrap();
        let bb = g.id("http://b/").unwrap();
        let c = g.id("http://c/").unwrap();

        assert_eq!(g.out_degree(a), 2);
        assert_eq!(g.out_degree(bb), 1);
        assert_eq!(g.out_degree(c), 0, "c links to nothing");
        assert!(g.outlinks(a).contains(&bb));
        assert!(g.outlinks(a).contains(&c));
    }

    #[test]
    fn a_repeated_link_is_one_vote() {
        // A navigation menu on every page would otherwise dominate.
        let mut b = GraphBuilder::new();
        for _ in 0..100 {
            b.edge("http://a/", "http://b/");
        }
        let g = b.build();
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn self_links_are_dropped() {
        let mut b = GraphBuilder::new();
        b.edge("http://a/", "http://a/");
        assert_eq!(b.edge_count(), 0);
    }

    #[test]
    fn a_link_to_an_uncrawled_page_still_creates_a_node() {
        let mut b = GraphBuilder::new();
        b.edge("http://a/", "http://never-fetched/");
        let g = b.build();
        assert_eq!(g.node_count(), 2);
        assert!(g.id("http://never-fetched/").is_some());
    }

    #[test]
    fn an_empty_graph_is_valid() {
        let g = GraphBuilder::new().build();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert!(g.outlinks(NodeId(0)).is_empty());
    }

    #[test]
    fn out_of_range_nodes_return_nothing_rather_than_panicking() {
        let mut b = GraphBuilder::new();
        b.edge("http://a/", "http://b/");
        let g = b.build();
        assert!(g.outlinks(NodeId(999)).is_empty());
        assert!(g.uri(NodeId(999)).is_none());
    }
}
