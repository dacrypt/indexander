//! Several segments, searched as one corpus.
//!
//! A segment is written once and never modified, which is what makes it cheap
//! to memory map, replicate and serve. The cost of that is that adding a
//! document means writing another segment, not editing the one that exists —
//! so an index is a *list* of segments, and a shard accumulates them until a
//! background merge folds them into one.
//!
//! Searching across them is the same problem the cluster has across shards,
//! for the same reason: BM25 weights a term by how rare it is, and rarity is a
//! property of the whole corpus. A term appearing in one document of a small
//! new segment is not rare — it is rare only if it is also rare in the large
//! old one. So the local counts are summed first and every segment is scored
//! with the total, which is exactly [`GlobalStats`] doing the job it was built
//! for one level up.

use std::path::Path;

use indexander_core::Result;

use crate::builder::SegmentBuilder;
use crate::manifest::Manifest;
use crate::query::Query;
use crate::search::{GlobalStats, Hit, search_with_stats};
use crate::segment::Segment;

/// An index made of one or more segments.
#[derive(Debug, Default)]
pub struct Index {
    segments: Vec<Segment>,
}

impl Index {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens every segment named, in order. Order decides nothing about
    /// results; it decides document order in a merge.
    pub fn open(paths: &[impl AsRef<Path>]) -> Result<Self> {
        let mut segments = Vec::with_capacity(paths.len());
        for path in paths {
            segments.push(Segment::open(path.as_ref())?);
        }
        Ok(Self { segments })
    }

    pub fn push(&mut self, segment: Segment) {
        self.segments.push(segment);
    }

    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    #[must_use]
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Documents across every segment.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.segments.iter().map(Segment::document_count).sum()
    }

    /// Corpus-wide statistics for the terms a query cares about.
    ///
    /// One pass over the segments' dictionaries, which is a binary search
    /// each, not a scan.
    pub fn statistics(&self, query: &Query) -> Result<GlobalStats> {
        let terms = query.scoring_terms();
        let mut stats = GlobalStats::default();
        for segment in &self.segments {
            let per_term: Vec<(String, usize)> = terms
                .iter()
                .map(|term| {
                    segment
                        .document_frequency(term)
                        .map(|freq| (term.clone(), freq))
                })
                .collect::<Result<_>>()?;
            stats.add_shard(segment.document_count(), segment.total_length(), &per_term);
        }
        Ok(stats)
    }

    /// Runs `query` across every segment and returns the global top `limit`.
    pub fn search(&self, query: &Query, limit: usize) -> Result<Vec<Hit>> {
        if self.segments.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        // One segment is the whole corpus, so its own counts are the right
        // ones and gathering statistics would only cost a pass.
        if self.segments.len() == 1 {
            return search_with_stats(&self.segments[0], query, limit, None);
        }

        let stats = self.statistics(query)?;
        let mut all: Vec<Hit> = Vec::new();
        for segment in &self.segments {
            // Each segment's own top `limit`: the global top `limit` is a
            // subset of the union, so asking for more from any one of them
            // would be work thrown away.
            all.extend(search_with_stats(segment, query, limit, Some(&stats))?);
        }

        crate::search::sort_hits(&mut all);
        all.truncate(limit);
        Ok(all)
    }

    /// Opens every segment a manifest names, from `directory`.
    ///
    /// Checks each one's digest against what the manifest recorded. A segment
    /// that does not match is not a slightly different segment: it is a file
    /// that will answer queries with documents quietly missing, which is
    /// exactly what the digest is there to catch.
    pub fn open_manifest(directory: &Path, manifest: &Manifest) -> Result<Self> {
        let mut segments = Vec::with_capacity(manifest.segments.len());
        for entry in &manifest.segments {
            let segment = Segment::open(&directory.join(&entry.name))?;
            if segment.digest() != entry.digest {
                return Err(indexander_core::Error::Corrupt(format!(
                    "{} does not match the digest the manifest recorded",
                    entry.name
                )));
            }
            segments.push(segment);
        }
        Ok(Self { segments })
    }

    /// Folds the segments a plan names into one, returning its bytes.
    ///
    /// The plan comes from [`Policy::next_merge`]; this only does what it
    /// says. Deciding and doing are kept apart because deciding is cheap and
    /// testable and doing is neither.
    pub fn merge_plan(&self, plan: &[usize]) -> Result<Vec<u8>> {
        let chosen: Vec<&Segment> = plan.iter().filter_map(|i| self.segments.get(*i)).collect();
        Ok(SegmentBuilder::from_segments(&chosen)?.encode())
    }

    /// Folds every segment into one.
    ///
    /// Returns the bytes rather than writing them, so a caller can write and
    /// rename them itself: replacing a segment that another process may have
    /// mapped has to go through a rename, and this module should not decide
    /// where.
    pub fn merge(&self) -> Result<Vec<u8>> {
        let refs: Vec<&Segment> = self.segments.iter().collect();
        Ok(SegmentBuilder::from_segments(&refs)?.encode())
    }
}
