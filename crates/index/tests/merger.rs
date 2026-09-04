//! Merging on disk, including when it is interrupted.
//!
//! A merge takes minutes on a real index, so the question that matters is not
//! whether it works but what a half-finished one leaves behind. The design
//! answer is that the manifest is the only thing that decides what an index
//! is: the new segment is written first, the manifest last, and until that
//! rename lands the index is exactly what it was. These tests kill a merge at
//! each of those points and check it.

use indexander_core::Document;
use indexander_index::builder::SegmentBuilder;
use indexander_index::index::Index;
use indexander_index::manifest::{Entry, Manifest, Policy};
use indexander_index::merger::Merger;
use indexander_index::query;
use indexander_index::search::Hit;
use indexander_index::segment::Segment;

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("indexander-merger-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir
}

fn documents(from: usize, count: usize) -> Vec<Document> {
    (from..from + count)
        .map(|i| {
            Document::new(
                format!("doc://{i:05}"),
                format!("documento {i}"),
                format!("motor de busqueda numero {i} con palabras suficientes para variar"),
            )
        })
        .collect()
}

/// Writes one segment and appends it to the manifest, as a flush would.
fn flush(dir: &std::path::Path, docs: &[Document], manifest: &mut Manifest) {
    let mut builder = SegmentBuilder::new();
    for doc in docs {
        builder.add(doc);
    }
    let bytes = builder.encode();
    let segment = Segment::from_bytes(bytes.clone()).expect("segment");
    let name = format!("flush{:03}.ixdr", manifest.segments.len());
    std::fs::write(dir.join(&name), &bytes).expect("write");
    manifest.segments.push(Entry {
        name,
        digest: segment.digest(),
        documents: segment.document_count(),
        bytes: bytes.len() as u64,
    });
}

fn uris(hits: &[Hit]) -> Vec<String> {
    hits.iter().map(|h| h.uri.clone()).collect()
}

fn answers(dir: &std::path::Path) -> Vec<String> {
    let manifest = Manifest::open(&dir.join("MANIFEST")).expect("manifest");
    let index = Index::open_manifest(dir, &manifest).expect("index");
    uris(
        &index
            .search(&query::parse("motor busqueda"), 20)
            .expect("search"),
    )
}

fn document_count(dir: &std::path::Path) -> usize {
    Manifest::open(&dir.join("MANIFEST"))
        .expect("manifest")
        .document_count()
}

#[test]
fn merging_leaves_the_index_answering_the_same_thing() {
    let dir = scratch("same");
    let mut manifest = Manifest::new();
    for i in 0..12 {
        flush(&dir, &documents(i * 20, 20), &mut manifest);
    }
    manifest.write_to(&dir.join("MANIFEST")).expect("write");

    let before = answers(&dir);
    assert_eq!(document_count(&dir), 240);

    let merger = Merger::new(
        &dir,
        Policy {
            segments_per_tier: 4,
            ..Policy::default()
        },
    );
    let reports = merger.run_to_completion().expect("merge");
    assert!(!reports.is_empty(), "nothing was merged");

    assert_eq!(document_count(&dir), 240, "documents went missing");
    assert_eq!(answers(&dir), before, "the merge changed the answers");

    let after = Manifest::open(&dir.join("MANIFEST")).expect("manifest");
    assert!(
        after.segments.len() < 12,
        "still {} segments",
        after.segments.len()
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_merge_that_dies_before_the_manifest_changes_nothing() {
    // The crash that matters. Everything a merge does before replacing the
    // manifest has to be invisible: the segment it wrote is an orphan, and
    // the index is exactly what it was.
    let dir = scratch("crash");
    let mut manifest = Manifest::new();
    for i in 0..8 {
        flush(&dir, &documents(i * 10, 10), &mut manifest);
    }
    manifest.write_to(&dir.join("MANIFEST")).expect("write");
    let before = answers(&dir);
    let manifest_before = std::fs::read_to_string(dir.join("MANIFEST")).expect("read");

    // Simulate the merge writing its segment and then dying: build exactly
    // what the merger would have produced, write it, stop there.
    let index = Index::open_manifest(&dir, &manifest).expect("index");
    let bytes = index.merge_plan(&[0, 1, 2, 3]).expect("merge");
    let produced = Segment::from_bytes(bytes.clone()).expect("segment");
    std::fs::write(dir.join(format!("{:016x}.ixdr", produced.digest())), &bytes).expect("write");

    assert_eq!(
        std::fs::read_to_string(dir.join("MANIFEST")).expect("read"),
        manifest_before,
        "the manifest changed when it should not have"
    );
    assert_eq!(document_count(&dir), 80);
    assert_eq!(answers(&dir), before, "a dead merge changed the answers");

    // And the file it left is visible as an orphan rather than silently
    // taking up space forever.
    let merger = Merger::new(&dir, Policy::default());
    let orphans = merger.orphans().expect("orphans");
    assert_eq!(orphans.len(), 1, "expected exactly one orphan: {orphans:?}");

    // Re-running the merge is safe, and produces the same file name.
    let merger = Merger::new(
        &dir,
        Policy {
            segments_per_tier: 4,
            ..Policy::default()
        },
    );
    let report = merger.step().expect("merge").expect("a merge");
    assert_eq!(
        report.produced,
        format!("{:016x}.ixdr", produced.digest()),
        "a retried merge should write the same name"
    );
    assert_eq!(answers(&dir), before);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_merge_is_idempotent_because_segments_are_named_by_content() {
    let dir = scratch("idempotent");
    let mut manifest = Manifest::new();
    for i in 0..4 {
        flush(&dir, &documents(i * 10, 10), &mut manifest);
    }
    manifest.write_to(&dir.join("MANIFEST")).expect("write");

    let index = Index::open_manifest(&dir, &manifest).expect("index");
    let first = Segment::from_bytes(index.merge_plan(&[0, 1]).expect("merge")).expect("segment");
    let second = Segment::from_bytes(index.merge_plan(&[0, 1]).expect("merge")).expect("segment");
    assert_eq!(first.digest(), second.digest());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn merging_keeps_going_until_the_policy_is_satisfied() {
    let dir = scratch("until");
    let mut manifest = Manifest::new();
    for i in 0..40 {
        flush(&dir, &documents(i * 5, 5), &mut manifest);
    }
    manifest.write_to(&dir.join("MANIFEST")).expect("write");

    let merger = Merger::new(
        &dir,
        Policy {
            segments_per_tier: 4,
            ..Policy::default()
        },
    );
    merger.run_to_completion().expect("merge");
    assert!(
        merger.step().expect("step").is_none(),
        "the policy still wants a merge after running to completion"
    );
    assert_eq!(document_count(&dir), 200);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn nothing_to_merge_is_not_an_error() {
    let dir = scratch("nothing");
    let mut manifest = Manifest::new();
    flush(&dir, &documents(0, 10), &mut manifest);
    manifest.write_to(&dir.join("MANIFEST")).expect("write");

    let merger = Merger::new(&dir, Policy::default());
    assert!(merger.step().expect("step").is_none());
    assert!(merger.orphans().expect("orphans").is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_missing_manifest_is_an_error_not_an_empty_index() {
    // An index whose manifest is gone is not an empty index; treating it as
    // one would merge nothing, report success, and hide a lost file.
    let dir = scratch("missing");
    let merger = Merger::new(&dir, Policy::default());
    assert!(merger.step().is_err());
    assert!(merger.orphans().is_err());
    std::fs::remove_dir_all(&dir).ok();
}
