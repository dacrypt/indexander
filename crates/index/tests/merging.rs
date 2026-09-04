//! Splitting an index into segments must not change the answer.
//!
//! Three ways to hold the same corpus — one segment, several segments, and
//! several merged back into one — and all three have to agree. If they do,
//! how many segments an index happens to have is an implementation detail; if
//! they do not, results depend on when the last merge ran, and nobody would
//! ever notice.

use indexander_core::{DocId, Document};
use indexander_index::builder::SegmentBuilder;
use indexander_index::index::Index;
use indexander_index::query;
use indexander_index::search::search;
use indexander_index::segment::Segment;

/// A corpus where term frequencies differ sharply between the halves, so that
/// scoring a segment with its own counts gives a different answer from scoring
/// it with the corpus's — and where no two documents score the same.
///
/// The lengths vary on purpose. BM25 discounts a document by its length
/// relative to the *corpus* average, so a corpus of uniform-length documents
/// cannot tell whether an implementation is using the right average. This one
/// can, and it did: it is how `GlobalStats` came to carry a total length.
fn corpus() -> Vec<Document> {
    let filler = [
        "posicional",
        "invertido",
        "frontera",
        "ancla",
        "relevancia",
        "consulta",
        "segmento",
        "bloque",
        "salto",
        "cortesia",
    ];
    let mut docs: Vec<Document> = (0..60)
        .map(|i| {
            // Length varies from short to long, so no two documents score
            // alike and a wrong average length shows up as a wrong order.
            let extra: String = filler[..=(i % filler.len())].join(" ");
            Document::new(
                format!("doc://early{i:03}"),
                format!("documento {i}"),
                format!("rust rust motor de busqueda {extra}"),
            )
        })
        .collect();
    docs.push(Document::new(
        "doc://early-perl",
        "raro",
        "perl rust motor de busqueda",
    ));
    docs.extend((0..60).map(|i| {
        let extra: String = filler[..=(i % filler.len())].join(" ");
        Document::new(
            format!("doc://late{i:03}"),
            format!("documento tardio {i}"),
            format!("perl perl motor de busqueda {extra}"),
        )
    }));
    docs.push(Document::new(
        "doc://late-rust",
        "raro",
        "rust perl perl perl motor",
    ));
    docs
}

fn segment_of(docs: &[Document]) -> Segment {
    let mut builder = SegmentBuilder::new();
    for doc in docs {
        builder.add(doc);
    }
    Segment::from_bytes(builder.encode()).expect("segment")
}

/// The corpus in `parts` segments, split in order.
fn split(parts: usize) -> Index {
    let docs = corpus();
    let size = docs.len().div_ceil(parts.max(1));
    let mut index = Index::new();
    for chunk in docs.chunks(size.max(1)) {
        index.push(segment_of(chunk));
    }
    index
}

fn uris(hits: &[indexander_index::search::Hit]) -> Vec<String> {
    hits.iter().map(|h| h.uri.clone()).collect()
}

const QUERIES: [&str; 6] = [
    "rust",
    "perl",
    "rust perl",
    "motor busqueda",
    r#""motor de busqueda""#,
    "motor -perl",
];

#[test]
fn several_segments_answer_like_one() {
    let whole = segment_of(&corpus());
    for parts in [2usize, 3, 5, 11] {
        let index = split(parts);
        assert_eq!(index.document_count(), whole.document_count());
        for text in QUERIES {
            let parsed = query::parse(text);
            assert_eq!(
                uris(&index.search(&parsed, 10).expect("search")),
                uris(&search(&whole, &parsed, 10).expect("search")),
                "{parts} segments disagreed on {text:?}"
            );
        }
    }
}

/// The reason `statistics` exists, stated as a test.
///
/// Segments of very different sizes are what make local counts lie: a term in
/// one document of a sixty-document segment looks rare, and in one document of
/// a six-document segment looks common, when it is the same term appearing
/// once. This builds exactly that imbalance and checks the two answers
/// disagree — if they ever stop, `several_segments_answer_like_one` is
/// proving nothing.
#[test]
fn scoring_each_segment_alone_would_give_a_different_answer() {
    let mut small: Vec<Document> = (0..9)
        .map(|i| Document::new(format!("doc://s{i}"), "chico", "rust motor de busqueda"))
        .collect();
    small.push(Document::new(
        "doc://s-match",
        "chico",
        "rust rust rust perl motor",
    ));

    let mut large: Vec<Document> = (0..99)
        .map(|i| Document::new(format!("doc://l{i:03}"), "grande", "perl motor de busqueda"))
        .collect();
    large.push(Document::new(
        "doc://l-match",
        "grande",
        "rust perl perl perl motor",
    ));

    let mut everything = small.clone();
    everything.extend(large.clone());
    let whole = segment_of(&everything);

    let mut index = Index::new();
    index.push(segment_of(&small));
    index.push(segment_of(&large));

    let parsed = query::parse("rust perl");
    let correct = uris(&search(&whole, &parsed, 10).expect("search"));
    assert_eq!(
        correct.len(),
        10,
        "the limit, since every document has a term"
    );
    assert_eq!(
        uris(&index.search(&parsed, 10).expect("search")),
        correct,
        "corpus-wide statistics should reproduce the single-segment answer"
    );

    let mut naive: Vec<indexander_index::search::Hit> = Vec::new();
    for segment in index.segments() {
        naive.extend(search(segment, &parsed, 10).expect("search"));
    }
    naive.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.uri.cmp(&b.uri))
    });

    assert_ne!(
        uris(&naive),
        correct,
        "segment-local statistics happened to agree; the corpus no longer tests this"
    );
}

#[test]
fn a_merge_produces_exactly_the_single_pass_segment() {
    // Byte-identical, not merely equivalent: a merge that reorders or drops
    // anything would still answer most queries correctly.
    let mut single = SegmentBuilder::new();
    for doc in &corpus() {
        single.add(doc);
    }
    let expected = single.encode();

    for parts in [2usize, 3, 7, 130] {
        assert_eq!(
            split(parts).merge().expect("merge"),
            expected,
            "merging {parts} segments produced a different segment"
        );
    }
}

#[test]
fn a_merged_index_answers_like_the_segments_it_came_from() {
    let index = split(4);
    let merged = Segment::from_bytes(index.merge().expect("merge")).expect("segment");
    for text in QUERIES {
        let parsed = query::parse(text);
        assert_eq!(
            uris(&search(&merged, &parsed, 10).expect("search")),
            uris(&index.search(&parsed, 10).expect("search")),
            "the merge changed the answer to {text:?}"
        );
    }
}

#[test]
fn merging_keeps_positions_so_phrases_still_work() {
    // A merge that dropped positions would answer phrase queries with nothing
    // and look merely unlucky.
    let merged = Segment::from_bytes(split(5).merge().expect("merge")).expect("segment");
    let hits = search(&merged, &query::parse(r#""motor de busqueda""#), 10).expect("search");
    assert!(!hits.is_empty());
    let wrong = search(&merged, &query::parse(r#""busqueda de motor""#), 10).expect("search");
    assert!(
        wrong.is_empty(),
        "word order stopped mattering after a merge"
    );
}

#[test]
fn merging_keeps_ranks() {
    let mut first = SegmentBuilder::new();
    first.add(&Document::new("doc://a", "uno", "motor"));
    first.set_rank(DocId(0), 0.25);
    let mut second = SegmentBuilder::new();
    second.add(&Document::new("doc://b", "dos", "motor"));
    second.set_rank(DocId(0), 0.75);

    let mut index = Index::new();
    index.push(Segment::from_bytes(first.encode()).expect("segment"));
    index.push(Segment::from_bytes(second.encode()).expect("segment"));

    let merged = Segment::from_bytes(index.merge().expect("merge")).expect("segment");
    assert!((merged.doc(DocId(0)).expect("a").rank - 0.25).abs() < 1e-6);
    assert!((merged.doc(DocId(1)).expect("b").rank - 0.75).abs() < 1e-6);
    // And the more authoritative page ranks first.
    let hits = search(&merged, &query::parse("motor"), 5).expect("search");
    assert_eq!(hits[0].uri, "doc://b");
}

/// What happens to documents that score exactly the same.
///
/// Their *order* is stable, because ties break on the uri rather than on a
/// document id that depends on how the corpus was split. Which of them
/// survives a top-k cut is not guaranteed across segmentations: that is
/// decided by a heap while scoring, and it cannot afford a string comparison
/// per candidate. Asserting the weaker, true property rather than the
/// stronger, false one.
#[test]
fn tied_documents_come_back_in_a_stable_order() {
    let tied: Vec<Document> = (0..40)
        .map(|i| Document::new(format!("doc://{i:03}"), "igual", "motor de busqueda"))
        .collect();

    let whole = segment_of(&tied);
    let from_one = uris(&search(&whole, &query::parse("motor"), 40).expect("search"));

    // Every document, so the cut is not involved and only the order is tested.
    let mut index = Index::new();
    for chunk in tied.chunks(7) {
        index.push(segment_of(chunk));
    }
    let from_many = uris(&index.search(&query::parse("motor"), 40).expect("search"));

    assert_eq!(from_one, from_many);
    let mut sorted = from_one.clone();
    sorted.sort();
    assert_eq!(from_one, sorted, "ties did not break on the uri");
}

#[test]
fn an_empty_index_and_an_empty_segment_are_handled() {
    let empty = Index::new();
    assert_eq!(empty.document_count(), 0);
    assert!(
        empty
            .search(&query::parse("motor"), 10)
            .expect("search")
            .is_empty()
    );
    assert!(
        !empty.merge().expect("merge").is_empty(),
        "even nothing has a footer"
    );

    let mut with_empty = Index::new();
    with_empty.push(Segment::from_bytes(SegmentBuilder::new().encode()).expect("segment"));
    with_empty.push(segment_of(&corpus()[..5]));
    assert_eq!(with_empty.document_count(), 5);
    assert!(
        !with_empty
            .search(&query::parse("motor"), 5)
            .expect("search")
            .is_empty()
    );
}

#[test]
fn terms_come_back_sorted_and_complete() {
    let segment = segment_of(&corpus());
    let terms: Vec<String> = segment.terms().collect::<Result<_, _>>().expect("terms");
    assert_eq!(terms.len(), segment.term_count());
    assert!(
        terms.windows(2).all(|w| w[0] < w[1]),
        "terms are not sorted"
    );
    assert!(terms.iter().any(|t| t == "rust"));
    assert!(terms.iter().any(|t| t == "busqueda"));
}

// --- manifests -----------------------------------------------------------

use indexander_index::manifest::{Entry, Manifest, Policy};

#[test]
fn an_index_opens_from_a_manifest_and_checks_the_digests() {
    let dir = std::env::temp_dir().join(format!("indexander-manifest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");

    let docs = corpus();
    let mut manifest = Manifest::new();
    for (i, chunk) in docs.chunks(40).enumerate() {
        let mut builder = SegmentBuilder::new();
        for doc in chunk {
            builder.add(doc);
        }
        let name = format!("part{i}.ixdr");
        builder.write_to(&dir.join(&name)).expect("write");
        let segment = Segment::open(&dir.join(&name)).expect("open");
        manifest.segments.push(Entry {
            name,
            digest: segment.digest(),
            documents: segment.document_count(),
            bytes: segment.as_bytes().len() as u64,
        });
    }

    let path = dir.join("MANIFEST");
    manifest.write_to(&path).expect("write manifest");
    let read_back = Manifest::open(&path).expect("read manifest");
    assert_eq!(read_back, manifest);

    let index = Index::open_manifest(&dir, &read_back).expect("open index");
    assert_eq!(index.document_count(), docs.len());
    let whole = segment_of(&docs);
    for text in QUERIES {
        let parsed = query::parse(text);
        assert_eq!(
            uris(&index.search(&parsed, 10).expect("search")),
            uris(&search(&whole, &parsed, 10).expect("search")),
            "an index opened from a manifest disagreed on {text:?}"
        );
    }

    // A manifest whose digest does not match the file on disk is refused.
    let mut wrong = read_back.clone();
    wrong.segments[0].digest ^= 1;
    assert!(Index::open_manifest(&dir, &wrong).is_err());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn merging_a_plan_matches_merging_by_hand() {
    let index = split(6);
    let policy = Policy {
        segments_per_tier: 2,
        tier_factor: 1_000_000,
        ..Policy::default()
    };
    // Everything lands in one tier, so the plan is the first two segments.
    let manifest = Manifest {
        segments: index
            .segments()
            .iter()
            .enumerate()
            .map(|(i, s)| Entry {
                name: format!("{i}.ixdr"),
                digest: s.digest(),
                documents: s.document_count(),
                bytes: s.as_bytes().len() as u64,
            })
            .collect(),
    };
    let plan = policy.next_merge(&manifest).expect("a merge");
    assert!(plan.len() >= 2);

    let merged = Segment::from_bytes(index.merge_plan(&plan).expect("merge")).expect("segment");
    let expected: usize = plan
        .iter()
        .map(|i| index.segments()[*i].document_count())
        .sum();
    assert_eq!(merged.document_count(), expected);
}
