// A simulation: it counts bytes and compares them to a curve, and the casts
// below are that arithmetic, not data handling.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

//! The property a merge policy exists for: bounded total work.
//!
//! Without one there are two options and both are bad. Merge after every
//! write and every document is rewritten every time — quadratic total work,
//! an index that gets slower to build the bigger it is. Never merge and a
//! query opens thousands of segments and sums statistics across all of them.
//!
//! Tiering is the way out, and what it buys is that a document is rewritten
//! about once per tier it climbs. These tests simulate a long-running index
//! and measure that, because "it should be logarithmic" is not a thing an
//! implementation can be trusted to do without being asked.

use indexander_index::manifest::{Entry, Manifest, Policy};

fn entry(id: usize, bytes: u64) -> Entry {
    Entry {
        name: format!("{id}.ixdr"),
        digest: id as u64,
        documents: bytes.max(1) as usize,
        bytes,
    }
}

/// Adds `flushes` equal-sized segments, merging whenever the policy says so,
/// and returns `(segments left, total bytes rewritten)`.
fn simulate(policy: &Policy, flushes: usize, flush_bytes: u64) -> (usize, u64) {
    let mut manifest = Manifest::new();
    let mut next_id = 0usize;
    let mut written = 0u64;

    for _ in 0..flushes {
        manifest.segments.push(entry(next_id, flush_bytes));
        next_id += 1;

        // Merge until the policy is satisfied, as a background merger would.
        while let Some(plan) = policy.next_merge(&manifest) {
            let merged: u64 = plan.iter().map(|i| manifest.segments[*i].bytes).sum();
            let documents: usize = plan.iter().map(|i| manifest.segments[*i].documents).sum();
            written += merged;

            let mut keep: Vec<Entry> = Vec::new();
            for (i, segment) in manifest.segments.iter().enumerate() {
                if !plan.contains(&i) {
                    keep.push(segment.clone());
                }
            }
            keep.push(Entry {
                name: format!("{next_id}.ixdr"),
                digest: next_id as u64,
                documents,
                bytes: merged,
            });
            next_id += 1;
            manifest.segments = keep;
        }
    }
    (manifest.segments.len(), written)
}

#[test]
fn the_number_of_segments_stays_logarithmic() {
    let policy = Policy::default();
    for flushes in [100usize, 1_000, 10_000] {
        let (segments, _) = simulate(&policy, flushes, 1000);
        // Eight per tier, ten times bigger each tier: a handful of tiers, each
        // holding at most seven leftovers.
        assert!(
            segments < 40,
            "{flushes} flushes left {segments} segments; a query would open all of them"
        );
    }
}

#[test]
fn total_work_is_linearithmic_not_quadratic() {
    // The number that separates a policy from no policy. Merging on every
    // write rewrites about n²/2 segment-sizes; tiering rewrites about
    // n·log(n). At ten thousand flushes those differ by a factor of a
    // thousand, so the bound does not need to be tight to be meaningful.
    let policy = Policy::default();
    let flush = 1000u64;

    for flushes in [1_000usize, 10_000] {
        let (_, written) = simulate(&policy, flushes, flush);
        let n = flushes as f64;
        let corpus = n * flush as f64;
        let quadratic = corpus * n / 2.0;
        let linearithmic = corpus * n.log2();

        assert!(
            (written as f64) < linearithmic * 1.5,
            "{flushes} flushes rewrote {written} bytes, more than n·log n ({linearithmic:.0})"
        );
        assert!(
            (written as f64) < quadratic / 100.0,
            "{flushes} flushes rewrote {written} bytes, close to quadratic ({quadratic:.0})"
        );
    }
}

#[test]
fn every_document_survives_every_merge() {
    // A policy that dropped or duplicated a segment would still look tidy.
    let policy = Policy::default();
    let mut manifest = Manifest::new();
    for id in 0..500 {
        manifest.segments.push(entry(id, 1000));
        while let Some(plan) = policy.next_merge(&manifest) {
            let documents: usize = plan.iter().map(|i| manifest.segments[*i].documents).sum();
            let bytes: u64 = plan.iter().map(|i| manifest.segments[*i].bytes).sum();
            let mut keep: Vec<Entry> = manifest
                .segments
                .iter()
                .enumerate()
                .filter(|(i, _)| !plan.contains(i))
                .map(|(_, s)| s.clone())
                .collect();
            keep.push(Entry {
                name: format!("m{id}.ixdr"),
                digest: 0,
                documents,
                bytes,
            });
            manifest.segments = keep;
        }
        assert_eq!(
            manifest.document_count(),
            (id + 1) * 1000,
            "documents went missing after flush {id}"
        );
    }
}

#[test]
fn a_plan_never_names_a_segment_twice() {
    let policy = Policy {
        segments_per_tier: 3,
        ..Policy::default()
    };
    let manifest = Manifest {
        segments: (0..20).map(|i| entry(i, 100)).collect(),
    };
    let plan = policy.next_merge(&manifest).expect("a merge");
    let mut sorted = plan.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        plan.len(),
        "a segment appeared twice in a plan"
    );
    assert!(plan.iter().all(|i| *i < manifest.segments.len()));
}

#[test]
fn merging_more_eagerly_leaves_fewer_segments_and_writes_more() {
    // The knob does what it says, in both directions. A policy where this is
    // not true is one nobody can tune.
    let eager = Policy {
        segments_per_tier: 2,
        ..Policy::default()
    };
    let lazy = Policy {
        segments_per_tier: 16,
        ..Policy::default()
    };
    let (eager_segments, eager_written) = simulate(&eager, 2000, 1000);
    let (lazy_segments, lazy_written) = simulate(&lazy, 2000, 1000);

    assert!(
        eager_segments <= lazy_segments,
        "{eager_segments} vs {lazy_segments}"
    );
    assert!(
        eager_written > lazy_written,
        "merging more often should write more: {eager_written} vs {lazy_written}"
    );
}

#[test]
fn an_index_that_is_never_written_to_is_never_merged() {
    let policy = Policy::default();
    let manifest = Manifest {
        segments: vec![entry(0, 5_000_000)],
    };
    assert!(policy.next_merge(&manifest).is_none());
    assert!(policy.next_merge(&Manifest::new()).is_none());
}
