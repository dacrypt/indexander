//! Build a segment, write it, read it back, search it.
//!
//! These are the tests that would catch a mismatch between the writer and the
//! reader, which unit tests on either side cannot.

use indexander_core::{DocId, Document, Field};
use indexander_index::builder::SegmentBuilder;
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;

fn corpus() -> Vec<Document> {
    vec![
        Document::new(
            "doc://parasearch",
            "Parasearch, un motor de búsqueda",
            "Un buscador escrito en Perl en Colombia en 2004. El indexador se llamaba Indexander.",
        )
        .with_anchor("el buscador colombiano"),
        Document::new(
            "doc://indexander",
            "Indexander",
            "Motor de búsqueda escrito en Rust. Índice invertido posicional y ranking BM25.",
        )
        .with_anchor("motor de búsqueda en Rust"),
        Document::new(
            "doc://perl",
            "Perl",
            "Un lenguaje de programación. Nada que ver con motores de búsqueda distribuidos.",
        ),
    ]
}

fn build() -> Segment {
    let mut builder = SegmentBuilder::new();
    for doc in &corpus() {
        builder.add(doc);
    }
    Segment::from_bytes(builder.encode()).expect("segment should parse")
}

#[test]
fn segment_roundtrips_through_bytes() {
    let segment = build();
    assert_eq!(segment.document_count(), 3);
    assert!(segment.term_count() > 20, "got {}", segment.term_count());
    assert_eq!(segment.doc(DocId(0)).unwrap().uri, "doc://parasearch");
    assert_eq!(segment.doc(DocId(2)).unwrap().uri, "doc://perl");
    assert!(segment.doc(DocId(99)).is_none());
}

#[test]
fn segment_roundtrips_through_a_file() {
    let mut builder = SegmentBuilder::new();
    for doc in &corpus() {
        builder.add(doc);
    }
    let dir = std::env::temp_dir().join(format!("indexander-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("segment.ixdr");
    builder.write_to(&path).unwrap();

    let segment = Segment::open(&path).unwrap();
    assert_eq!(segment.document_count(), 3);
    let hits = search(&segment, &query::parse("indexander"), 10).unwrap();
    assert_eq!(hits.len(), 2);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn postings_carry_positions_and_fields() {
    let segment = build();
    // "indexander" is in doc 0's body and doc 1's title.
    let postings = segment.postings("indexander").unwrap();
    assert_eq!(postings.len(), 2);
    assert_eq!(postings[0].doc, DocId(0));
    assert!(!postings[0].positions_in(Field::Body).is_empty());
    assert!(postings[0].positions_in(Field::Title).is_empty());
    assert_eq!(postings[1].doc, DocId(1));
    assert!(!postings[1].positions_in(Field::Title).is_empty());
}

#[test]
fn accented_text_is_findable_without_accents_and_with_them() {
    let segment = build();
    let without = search(&segment, &query::parse("busqueda"), 10).unwrap();
    let with = search(&segment, &query::parse("búsqueda"), 10).unwrap();
    assert!(!without.is_empty());
    assert_eq!(without.len(), with.len());
    assert_eq!(without[0].uri, with[0].uri);
}

#[test]
fn the_2004_bug_would_fail_this_test() {
    // parasearch folded "ñ" to "c", so a document containing "español"
    // was indexed as "espacol" and this search returned nothing.
    let mut builder = SegmentBuilder::new();
    builder.add(&Document::new(
        "doc://es",
        "Español",
        "Compañia Colombiana de Años",
    ));
    let segment = Segment::from_bytes(builder.encode()).unwrap();

    for term in ["espanol", "compania", "anos", "español", "años"] {
        let hits = search(&segment, &query::parse(term), 10).unwrap();
        assert_eq!(hits.len(), 1, "searching for {term:?} found nothing");
    }
}

#[test]
fn a_document_with_more_of_the_query_ranks_above_one_with_less() {
    let segment = build();
    // Three documents, holding three, two and one of these words. Nothing
    // enforces the resulting order: each term present adds a contribution, so
    // the document accounting for more of the query wins by arithmetic.
    let hits = search(&segment, &query::parse("motor busqueda rust"), 10).unwrap();
    let uris: Vec<&str> = hits.iter().map(|h| h.uri.as_str()).collect();
    assert_eq!(
        uris,
        ["doc://indexander", "doc://parasearch", "doc://perl"],
        "{hits:?}"
    );
}

#[test]
fn a_word_no_document_has_does_not_empty_the_results() {
    let segment = build();
    // The old rule answered nothing here, which is the behaviour that made a
    // seventeen-word question unanswerable.
    let hits = search(&segment, &query::parse("rust cobol"), 10).unwrap();
    assert!(
        !hits.is_empty(),
        "the documents about rust are still answers"
    );
    let only_cobol = search(&segment, &query::parse("cobol"), 10).unwrap();
    assert!(only_cobol.is_empty(), "no document mentions it at all");
}

#[test]
fn a_phrase_requires_adjacency_in_order() {
    let segment = build();
    let hits = search(&segment, &query::parse(r#""motor de busqueda""#), 10).unwrap();
    assert!(!hits.is_empty(), "the phrase should be found");

    // The same words in the wrong order are not the same phrase.
    let hits = search(&segment, &query::parse(r#""busqueda de motor""#), 10).unwrap();
    assert!(hits.is_empty(), "word order must matter inside a phrase");
}

#[test]
fn a_phrase_does_not_span_two_fields() {
    // "Indexander" ends the title; "Motor" starts the body. Adjacent only if
    // the two fields wrongly share a position space.
    let mut builder = SegmentBuilder::new();
    builder.add(&Document::new("doc://x", "Indexander", "Motor de busqueda"));
    let segment = Segment::from_bytes(builder.encode()).unwrap();
    let hits = search(&segment, &query::parse(r#""indexander motor""#), 10).unwrap();
    assert!(hits.is_empty(), "a phrase must not straddle two fields");
}

#[test]
fn exclusion_removes_documents() {
    let segment = build();
    let all = search(&segment, &query::parse("busqueda"), 10).unwrap();
    let filtered = search(&segment, &query::parse("busqueda -perl"), 10).unwrap();
    assert!(filtered.len() < all.len());
    assert!(filtered.iter().all(|h| h.uri != "doc://perl"));
}

#[test]
fn title_and_anchor_matches_outrank_body_matches() {
    let mut builder = SegmentBuilder::new();
    // Same word, same document length, different field.
    builder.add(&Document::new("doc://body", "algo", "rust"));
    builder.add(&Document::new("doc://title", "rust", "algo"));
    let segment = Segment::from_bytes(builder.encode()).unwrap();

    let hits = search(&segment, &query::parse("rust"), 10).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(
        hits[0].uri, "doc://title",
        "a title match should rank first"
    );
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn results_come_back_in_descending_score_order() {
    let segment = build();
    let hits = search(&segment, &query::parse("busqueda"), 10).unwrap();
    for pair in hits.windows(2) {
        assert!(pair[0].score >= pair[1].score, "results are not sorted");
    }
}

#[test]
fn limit_is_respected() {
    let segment = build();
    let hits = search(&segment, &query::parse("de"), 1).unwrap();
    assert!(hits.len() <= 1);
    assert!(search(&segment, &query::parse("de"), 0).unwrap().is_empty());
}

#[test]
fn unknown_terms_and_empty_queries_return_nothing() {
    let segment = build();
    assert!(
        search(&segment, &query::parse("kubernetes"), 10)
            .unwrap()
            .is_empty()
    );
    assert!(search(&segment, &query::parse(""), 10).unwrap().is_empty());
    assert!(segment.postings("kubernetes").unwrap().is_empty());
}

#[test]
fn an_empty_index_is_searchable_and_returns_nothing() {
    let segment = Segment::from_bytes(SegmentBuilder::new().encode()).unwrap();
    assert_eq!(segment.document_count(), 0);
    assert!(
        search(&segment, &query::parse("cualquier cosa"), 10)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_truncated_segment_is_rejected_rather_than_misread() {
    let mut builder = SegmentBuilder::new();
    builder.add(&Document::new("doc://a", "t", "b"));
    let mut bytes = builder.encode();
    bytes.truncate(bytes.len() - 8);
    assert!(Segment::from_bytes(bytes).is_err());
    assert!(Segment::from_bytes(vec![0u8; 4]).is_err());
    assert!(Segment::from_bytes(Vec::new()).is_err());
}

#[test]
fn document_frequency_matches_the_postings_list() {
    let segment = build();
    for term in ["de", "indexander", "rust", "perl"] {
        assert_eq!(
            segment.document_frequency(term).unwrap(),
            segment.postings(term).unwrap().len(),
            "disagreement for {term:?}"
        );
    }
}

#[test]
fn the_index_is_smaller_than_the_text_it_indexes() {
    // The point of delta plus varint encoding, stated as a test.
    //
    // The corpus has to have real vocabulary variety, because the ratio is a
    // property of the text as much as of the encoding: five hundred copies of
    // one sentence produce five hundred postings and five hundred position
    // lists for each of a dozen terms, and an index that is *larger* than the
    // text. Real prose is nothing like that. A degenerate corpus is a fair
    // thing to measure, but not a fair thing to assert this about.
    let words = [
        "buscador",
        "indice",
        "invertido",
        "posicional",
        "rastreo",
        "robots",
        "frontera",
        "ancla",
        "relevancia",
        "consulta",
        "termino",
        "documento",
        "segmento",
        "bloque",
        "salto",
        "postings",
        "ranking",
        "autoridad",
        "distribuido",
        "fragmento",
        "cortesia",
        "latencia",
        "memoria",
        "disco",
    ];
    let mut builder = SegmentBuilder::new();
    let mut raw = 0usize;
    for i in 0..500 {
        // A different slice of the vocabulary per document.
        let body: String = (0..30)
            .map(|w| words[(i * 7 + w * 13) % words.len()])
            .collect::<Vec<_>>()
            .join(" ");
        let doc = Document::new(format!("doc://{i}"), format!("documento numero {i}"), body);
        raw += doc.title.len() + doc.body.len() + doc.uri.len();
        builder.add(&doc);
    }
    let encoded = builder.encode().len();
    assert!(
        encoded < raw,
        "index is {encoded} bytes for {raw} bytes of text"
    );
}

// --- authority -----------------------------------------------------------
//
// PageRank only matters if it changes the answer. These build the same corpus
// twice, with and without ranks, and assert that the order moves.

use indexander_index::builder::SegmentBuilder as Builder;

/// Two pages that match a query equally well. One is authoritative.
fn two_equal_pages() -> Builder {
    let mut builder = Builder::new();
    builder.add(&Document::new(
        "doc://obscure",
        "buscador",
        "un buscador colombiano",
    ));
    builder.add(&Document::new(
        "doc://famous",
        "buscador",
        "un buscador colombiano",
    ));
    builder
}

#[test]
fn without_ranks_two_identical_pages_tie() {
    let segment = Segment::from_bytes(two_equal_pages().encode()).unwrap();
    let hits = search(&segment, &query::parse("buscador"), 10).unwrap();
    assert_eq!(hits.len(), 2);
    assert!(
        (hits[0].score - hits[1].score).abs() < 1e-6,
        "identical pages scored differently: {} vs {}",
        hits[0].score,
        hits[1].score
    );
}

#[test]
fn authority_breaks_the_tie_and_reorders_results() {
    let mut builder = two_equal_pages();
    // doc 1 is the authoritative one; doc 0 is ordinary.
    builder.set_rank(DocId(0), 0.01);
    builder.set_rank(DocId(1), 0.90);
    let segment = Segment::from_bytes(builder.encode()).unwrap();

    let hits = search(&segment, &query::parse("buscador"), 10).unwrap();
    assert_eq!(hits[0].uri, "doc://famous", "authority did not win");
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn authority_cannot_make_an_irrelevant_page_match() {
    let mut builder = Builder::new();
    builder.add(&Document::new("doc://authority", "portada", "nada que ver"));
    builder.add(&Document::new(
        "doc://relevant",
        "perl",
        "un motor escrito en perl",
    ));
    // Give the irrelevant page overwhelming authority.
    builder.set_rank(DocId(0), 0.99);
    builder.set_rank(DocId(1), 0.00001);
    let segment = Segment::from_bytes(builder.encode()).unwrap();

    let hits = search(&segment, &query::parse("perl"), 10).unwrap();
    assert_eq!(hits.len(), 1, "an authoritative page bought its way in");
    assert_eq!(hits[0].uri, "doc://relevant");
}

#[test]
fn ranks_survive_a_write_and_read() {
    let mut builder = two_equal_pages();
    builder.set_rank(DocId(1), 0.75);
    let segment = Segment::from_bytes(builder.encode()).unwrap();
    let meta = segment.doc(DocId(1)).expect("doc 1");
    assert!(
        (meta.rank - 0.75).abs() < 1e-6,
        "rank came back as {}",
        meta.rank
    );
    assert!((segment.doc(DocId(0)).unwrap().rank - 0.0).abs() < f32::EPSILON);
}

#[test]
fn a_version_1_segment_is_refused_rather_than_misread() {
    // The rank field changed the document store; an old segment read with the
    // new parser would silently produce nonsense lengths and ranks.
    let mut bytes = two_equal_pages().encode();
    let len = bytes.len();
    // The version sits just before the 4-byte magic at the very end.
    bytes[len - 8..len - 4].copy_from_slice(&1u32.to_le_bytes());
    let err = Segment::from_bytes(bytes).unwrap_err();
    assert!(format!("{err}").contains("version"), "got {err}");
}

// --- memory mapping ------------------------------------------------------

/// A mapped segment must answer exactly like an in-memory one.
#[test]
fn a_mapped_segment_answers_like_an_owned_one() {
    let mut builder = Builder::new();
    for doc in &corpus() {
        builder.add(doc);
    }
    let dir = std::env::temp_dir().join(format!("indexander-mmap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("segment.ixdr");
    builder.write_to(&path).unwrap();

    let owned = Segment::from_bytes(builder.encode()).unwrap();
    let mapped = Segment::open(&path).unwrap();

    assert_eq!(mapped.document_count(), owned.document_count());
    assert_eq!(mapped.term_count(), owned.term_count());
    for q in [
        "indexander",
        "busqueda",
        r#""motor de busqueda""#,
        "perl -rust",
    ] {
        let parsed = query::parse(q);
        assert_eq!(
            search(&mapped, &parsed, 10).unwrap(),
            search(&owned, &parsed, 10).unwrap(),
            "mapped and owned disagreed on {q:?}"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Writing a segment must replace it atomically, never truncate in place.
///
/// This is what makes mapping sound: truncating a file another process has
/// mapped is undefined behaviour, so `write_to` renames a new file over the
/// old one instead. The old inode survives for whoever still holds it.
#[test]
fn rewriting_a_segment_does_not_disturb_an_open_one() {
    let dir = std::env::temp_dir().join(format!("indexander-atomic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("segment.ixdr");

    let mut first = Builder::new();
    first.add(&Document::new("doc://one", "uno", "el primer indice"));
    first.write_to(&path).unwrap();

    // Map it, then replace the file underneath.
    let mapped = Segment::open(&path).unwrap();
    let mut second = Builder::new();
    for i in 0..50 {
        second.add(&Document::new(
            format!("doc://{i}"),
            "otro",
            "un indice distinto y mas grande",
        ));
    }
    second.write_to(&path).unwrap();

    // The open segment still sees what it opened.
    assert_eq!(mapped.document_count(), 1);
    assert_eq!(mapped.doc(DocId(0)).unwrap().uri, "doc://one");
    assert_eq!(
        search(&mapped, &query::parse("primer"), 5).unwrap().len(),
        1
    );

    // And a fresh open sees the new one.
    let reopened = Segment::open(&path).unwrap();
    assert_eq!(reopened.document_count(), 50);

    // No temporary file left behind.
    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("tmp"))
        .collect();
    assert!(leftovers.is_empty(), "left behind {leftovers:?}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn opening_a_missing_or_corrupt_file_errors() {
    assert!(Segment::open(std::path::Path::new("/nonexistent/segment.ixdr")).is_err());

    let dir = std::env::temp_dir().join(format!("indexander-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("garbage.ixdr");
    std::fs::write(&path, b"not a segment at all").unwrap();
    assert!(Segment::open(&path).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

// --- digests and replicas ------------------------------------------------

#[test]
fn the_same_corpus_produces_the_same_digest() {
    // What makes a replica checkable: build it twice, get the same segment.
    let a = Segment::from_bytes(build_whole_corpus()).unwrap();
    let b = Segment::from_bytes(build_whole_corpus()).unwrap();
    assert_eq!(a.digest(), b.digest());
    assert!(a.verify());
}

#[test]
fn a_different_corpus_produces_a_different_digest() {
    let a = Segment::from_bytes(build_whole_corpus()).unwrap();
    let mut extra = Builder::new();
    for doc in &corpus() {
        extra.add(doc);
    }
    extra.add(&Document::new("doc://extra", "otro", "un documento mas"));
    let b = Segment::from_bytes(extra.encode()).unwrap();
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn verify_catches_a_flipped_bit() {
    // The failure a replica has to survive: bytes arrived, but not all of
    // them are the ones that were sent.
    let mut bytes = build_whole_corpus();
    // Somewhere in the postings, well clear of the footer.
    bytes[64] ^= 0x01;
    let segment = Segment::from_bytes(bytes).unwrap();
    assert!(
        !segment.verify(),
        "a corrupt segment reported itself intact"
    );
}

#[test]
fn verify_catches_appended_or_removed_padding() {
    // Length is mixed into the digest, so trailing bytes cannot be added or
    // taken away unnoticed. Tested at the body's end rather than the file's,
    // because the footer is not part of what the digest covers.
    let bytes = build_whole_corpus();
    let mut padded = bytes.clone();
    let footer_at = padded.len() - 60;
    padded.splice(footer_at..footer_at, std::iter::repeat_n(0u8, 8));
    let segment = Segment::from_bytes(padded);
    // Either it fails to parse, or it parses and fails to verify. Both are
    // detections; silently serving it would not be.
    match segment {
        Err(_) => {}
        Ok(s) => assert!(!s.verify(), "padded segment reported itself intact"),
    }
}

fn build_whole_corpus() -> Vec<u8> {
    let mut builder = Builder::new();
    for doc in &corpus() {
        builder.add(doc);
    }
    builder.encode()
}
