//! Building an index in pieces must give exactly the index built in one go.
//!
//! This is the property that makes parallel indexing safe. If it holds, the
//! number of threads is an implementation detail; if it does not, every
//! parallel build produces a subtly different index and nobody notices.

use indexander_core::{DocId, Document};
use indexander_index::builder::SegmentBuilder;
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;

fn corpus(n: usize) -> Vec<Document> {
    (0..n)
        .map(|i| {
            Document::new(
                format!("doc://{i}"),
                format!("documento {i}"),
                format!("el robeiro rastrea y el indexander indexa la pagina numero {i} con palabras repetidas y posiciones"),
            )
            .with_anchor(format!("enlace a {i}"))
        })
        .collect()
}

fn build_whole(docs: &[Document]) -> Vec<u8> {
    let mut b = SegmentBuilder::new();
    for doc in docs {
        b.add(doc);
    }
    b.encode()
}

fn build_in_chunks(docs: &[Document], chunks: usize) -> Vec<u8> {
    let size = docs.len().div_ceil(chunks.max(1));
    let mut parts: Vec<SegmentBuilder> = docs
        .chunks(size.max(1))
        .map(|chunk| {
            let mut b = SegmentBuilder::new();
            for doc in chunk {
                b.add(doc);
            }
            b
        })
        .collect();

    let mut first = parts.remove(0);
    for part in parts {
        first.absorb(part);
    }
    first.encode()
}

#[test]
fn chunked_building_is_byte_identical_to_one_pass() {
    let docs = corpus(200);
    let whole = build_whole(&docs);
    for chunks in [1usize, 2, 3, 7, 16, 200] {
        assert_eq!(
            build_in_chunks(&docs, chunks),
            whole,
            "building in {chunks} chunks produced a different segment"
        );
    }
}

#[test]
fn absorbing_preserves_document_order_and_identity() {
    let docs = corpus(50);
    let segment = Segment::from_bytes(build_in_chunks(&docs, 5)).expect("segment");
    assert_eq!(segment.document_count(), 50);
    for (i, doc) in docs.iter().enumerate() {
        let id = DocId(u32::try_from(i).unwrap());
        assert_eq!(segment.doc(id).expect("doc").uri, doc.uri);
    }
}

#[test]
fn search_results_are_the_same_however_it_was_built() {
    let docs = corpus(300);
    let whole = Segment::from_bytes(build_whole(&docs)).expect("segment");
    let chunked = Segment::from_bytes(build_in_chunks(&docs, 8)).expect("segment");

    for q in [
        "indexander",
        "robeiro indexa",
        "\"la pagina numero 42\"",
        "enlace",
    ] {
        let parsed = query::parse(q);
        let a = search(&whole, &parsed, 10).expect("search");
        let b = search(&chunked, &parsed, 10).expect("search");
        assert_eq!(a, b, "results differed for {q:?}");
    }
}

#[test]
fn absorbing_an_empty_builder_changes_nothing() {
    let docs = corpus(10);
    let mut a = SegmentBuilder::new();
    for doc in &docs {
        a.add(doc);
    }
    let before = a.encode();
    a.absorb(SegmentBuilder::new());
    assert_eq!(a.encode(), before);
}

#[test]
fn an_empty_builder_absorbing_another_becomes_it() {
    let docs = corpus(10);
    let mut source = SegmentBuilder::new();
    for doc in &docs {
        source.add(doc);
    }
    let expected = source.encode();

    let mut empty = SegmentBuilder::new();
    empty.absorb(source);
    assert_eq!(empty.encode(), expected);
}

#[test]
fn ranks_set_before_absorbing_follow_their_documents() {
    let docs = corpus(4);
    let mut a = SegmentBuilder::new();
    a.add(&docs[0]);
    a.add(&docs[1]);
    let mut b = SegmentBuilder::new();
    b.add(&docs[2]);
    b.set_rank(DocId(0), 0.5);
    a.absorb(b);

    let segment = Segment::from_bytes(a.encode()).expect("segment");
    assert!((segment.doc(DocId(2)).unwrap().rank - 0.5).abs() < 1e-6);
    assert!((segment.doc(DocId(0)).unwrap().rank - 0.0).abs() < f32::EPSILON);
}

/// Scores must not depend on hash-map iteration order.
///
/// This test exists because it once failed: two byte-identical segments
/// scored the same document 9.72467 and 9.724671, because the per-term
/// contributions were summed in `HashMap` order and a `HashMap` is seeded
/// differently in every process. Two shards holding identical data would
/// have disagreed, and no ranking comparison would have been reproducible.
#[test]
fn scores_are_bit_for_bit_reproducible() {
    let docs = corpus(300);
    let whole = Segment::from_bytes(build_whole(&docs)).expect("segment");
    let chunked = Segment::from_bytes(build_in_chunks(&docs, 8)).expect("segment");

    // Several terms, so the sum has several addends and the order matters.
    let parsed = query::parse("indexander robeiro pagina numero palabras posiciones");
    let a = search(&whole, &parsed, 20).expect("search");
    let b = search(&chunked, &parsed, 20).expect("search");
    assert_eq!(a, b);

    // And repeating the same search must give bit-identical scores.
    for _ in 0..5 {
        assert_eq!(search(&whole, &parsed, 20).expect("search"), a);
    }
}
