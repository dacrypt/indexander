//! The skipping cursor must agree with full decoding, always.
//!
//! A cursor that is merely *usually* right produces an index that returns
//! slightly wrong results for slightly unusual queries, which is the worst
//! kind of broken. These tests compare it against the decode-everything path
//! on corpora large enough to span many skip blocks.

use indexander_core::{DocId, Document, Field};
use indexander_index::builder::SegmentBuilder;
use indexander_index::segment::Segment;

/// Enough documents to cross several 128-posting skip blocks, with terms of
/// very different frequencies.
fn segment(n: usize) -> Segment {
    let mut b = SegmentBuilder::new();
    for i in 0..n {
        let mut body = format!("comun palabra numero {i}");
        if i % 7 == 0 {
            body.push_str(" septimo");
        }
        if i % 101 == 0 {
            body.push_str(" raro");
        }
        b.add(&Document::new(format!("doc://{i}"), "titulo comun", body));
    }
    Segment::from_bytes(b.encode()).expect("segment")
}

/// Walks a cursor to the end, collecting what it saw.
fn walk(segment: &Segment, term: &str) -> Vec<(DocId, f32)> {
    let mut cursor = segment.cursor(term, false).expect("cursor");
    let mut out = Vec::new();
    while let Some(doc) = cursor.doc() {
        out.push((doc, cursor.weighted_frequency()));
        cursor.advance().expect("advance");
    }
    out
}

fn decoded(segment: &Segment, term: &str) -> Vec<(DocId, f32)> {
    segment
        .postings_counts(term)
        .expect("postings")
        .into_iter()
        .map(|p| (p.doc, p.weighted_frequency()))
        .collect()
}

#[test]
fn walking_a_cursor_sees_exactly_what_decoding_sees() {
    let s = segment(1000);
    for term in ["comun", "septimo", "raro", "titulo", "ausente"] {
        assert_eq!(
            walk(&s, term),
            decoded(&s, term),
            "disagreement on {term:?}"
        );
    }
}

#[test]
fn document_frequency_matches() {
    let s = segment(1000);
    for term in ["comun", "septimo", "raro"] {
        let cursor = s.cursor(term, false).expect("cursor");
        assert_eq!(cursor.document_frequency(), decoded(&s, term).len());
    }
}

#[test]
fn seeking_to_every_document_lands_correctly() {
    // The exhaustive version: for every possible target, the cursor must stop
    // on the first posting at or after it.
    let s = segment(600);
    let expected = decoded(&s, "septimo");

    for target in 0..600u32 {
        let mut cursor = s.cursor("septimo", false).expect("cursor");
        cursor.seek(DocId(target)).expect("seek");
        let wanted = expected
            .iter()
            .find(|(d, _)| d.0 >= target)
            .map(|(d, _)| *d);
        assert_eq!(cursor.doc(), wanted, "seeking to {target}");
    }
}

#[test]
fn seeking_across_block_boundaries_works() {
    // Targets around every skip block boundary, where an off-by-one lives.
    let s = segment(2000);
    let expected = decoded(&s, "comun");
    for block in 0..=(expected.len() / 128) {
        for offset in [0usize, 1, 2] {
            // The posting just before a block boundary, the one on it, and the
            // one after: where an off-by-one in the skip index would live.
            let Some(index) = (block * 128 + offset).checked_sub(1) else {
                continue;
            };
            if index >= expected.len() {
                continue;
            }
            let target = expected[index].0;
            let mut cursor = s.cursor("comun", false).expect("cursor");
            cursor.seek(target).expect("seek");
            assert_eq!(cursor.doc(), Some(target), "block {block} offset {offset}");
        }
    }
}

#[test]
fn seeking_never_moves_backwards() {
    let s = segment(500);
    let mut cursor = s.cursor("comun", false).expect("cursor");
    cursor.seek(DocId(300)).expect("seek");
    let at = cursor.doc();
    cursor.seek(DocId(10)).expect("seek back");
    assert_eq!(cursor.doc(), at, "a backwards seek moved the cursor");
}

#[test]
fn seeking_past_the_end_exhausts_the_cursor() {
    let s = segment(300);
    let mut cursor = s.cursor("raro", false).expect("cursor");
    cursor.seek(DocId(u32::MAX)).expect("seek");
    assert_eq!(cursor.doc(), None);
    // And advancing an exhausted cursor is harmless.
    cursor.advance().expect("advance");
    assert_eq!(cursor.doc(), None);
}

#[test]
fn a_cursor_over_a_missing_term_is_empty_immediately() {
    let s = segment(50);
    let mut cursor = s.cursor("kubernetes", false).expect("cursor");
    assert_eq!(cursor.doc(), None);
    assert_eq!(cursor.document_frequency(), 0);
    cursor.seek(DocId(0)).expect("seek");
    assert_eq!(cursor.doc(), None);
}

#[test]
fn a_cursor_can_read_positions_when_asked() {
    let s = segment(400);
    let mut cursor = s.cursor("septimo", true).expect("cursor");
    cursor.seek(DocId(70)).expect("seek");
    assert_eq!(cursor.doc(), Some(DocId(70)));
    let body: Vec<_> = cursor
        .fields()
        .iter()
        .filter(|f| f.field == Field::Body)
        .collect();
    assert_eq!(body.len(), 1);
    assert_eq!(body[0].count, 1);
    assert!(
        !cursor.positions_in(Field::Body).is_empty(),
        "positions were requested but came back empty"
    );
}

#[test]
fn a_single_document_index_still_works() {
    let mut b = SegmentBuilder::new();
    b.add(&Document::new("doc://solo", "uno", "una sola palabra"));
    let s = Segment::from_bytes(b.encode()).expect("segment");
    let mut cursor = s.cursor("palabra", false).expect("cursor");
    assert_eq!(cursor.doc(), Some(DocId(0)));
    cursor.advance().expect("advance");
    assert_eq!(cursor.doc(), None);
}

// --- the leapfrog path against brute force -------------------------------

use indexander_index::query;
use indexander_index::search::search;

/// Intersects postings the slow, obvious way, for comparison.
fn brute_force(segment: &Segment, terms: &[&str], excluded: &[&str]) -> Vec<DocId> {
    let mut sets: Vec<Vec<DocId>> = terms
        .iter()
        .map(|t| {
            segment
                .postings_counts(t)
                .expect("postings")
                .into_iter()
                .map(|p| p.doc)
                .collect()
        })
        .collect();
    let Some(mut out) = sets.pop() else {
        return Vec::new();
    };
    for set in sets {
        out.retain(|d| set.binary_search(d).is_ok());
    }
    for term in excluded {
        let drop: Vec<DocId> = segment
            .postings_counts(term)
            .expect("postings")
            .into_iter()
            .map(|p| p.doc)
            .collect();
        out.retain(|d| drop.binary_search(d).is_err());
    }
    out.sort_unstable();
    out
}

#[test]
fn leapfrog_finds_exactly_what_brute_force_finds() {
    let s = segment(3000);
    let cases: &[(&str, &[&str], &[&str])] = &[
        ("comun", &["comun"], &[]),
        ("raro", &["raro"], &[]),
        ("comun raro", &["comun", "raro"], &[]),
        ("comun septimo raro", &["comun", "septimo", "raro"], &[]),
        ("septimo -raro", &["septimo"], &["raro"]),
        ("comun -septimo", &["comun"], &["septimo"]),
        ("comun ausente", &["comun", "ausente"], &[]),
    ];

    for (query_text, terms, excluded) in cases {
        let expected = brute_force(&s, terms, excluded);
        // A limit past the corpus size, so nothing is cut off.
        let hits = search(&s, &query::parse(query_text), 5000).expect("search");
        let mut got: Vec<DocId> = hits.into_iter().map(|h| h.doc).collect();
        got.sort_unstable();
        assert_eq!(got, expected, "disagreement on {query_text:?}");
    }
}

#[test]
fn the_rarest_term_drives_whatever_order_it_is_written_in() {
    // "comun raro" and "raro comun" are the same query; the cursors get
    // sorted by frequency either way.
    let s = segment(2000);
    let a = search(&s, &query::parse("comun raro"), 50).expect("search");
    let b = search(&s, &query::parse("raro comun"), 50).expect("search");
    assert_eq!(a, b);
    assert!(!a.is_empty());
}

#[test]
fn a_limit_smaller_than_the_matches_keeps_the_best() {
    let s = segment(1000);
    let all = search(&s, &query::parse("septimo"), 5000).expect("search");
    let top = search(&s, &query::parse("septimo"), 3).expect("search");
    assert_eq!(top.len(), 3);
    assert_eq!(top, all[..3]);
}
