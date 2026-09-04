// Lengths and counts are `usize` in memory and varints on disk; widening them
// to `u64` to encode is the whole point. The one narrowing conversion that
// could actually lose data is checked explicitly.
#![allow(clippy::cast_possible_truncation)]

//! Building a segment in memory, then writing it to disk.
//!
//! An index is built by accumulating postings for every term, then flushing
//! the whole thing once, sorted. Sorting at the end rather than maintaining
//! order throughout is what makes indexing linear-ish instead of quadratic;
//! it is also why indexing is a batch job and searching is not.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use indexander_core::{DocId, Document, Error, Field, Position, Result};

use crate::segment::Segment;

use crate::scoring::{authority, length_norm, saturation};

use crate::codec::{write_deltas, write_varint};
use crate::segment::{FOOTER_LEN, MAGIC, SKIP_INTERVAL, VERSION};
use crate::tokenizer::tokenize_into;

/// Where one term occurs inside one document, per field.
#[derive(Debug, Default, Clone)]
struct FieldOccurrences {
    positions: Vec<Position>,
}

/// Everything known about one term inside one document.
#[derive(Debug, Default, Clone)]
struct DocOccurrences {
    /// Indexed by `Field as usize`; empty entries are simply never written.
    fields: [FieldOccurrences; 3],
}

/// Per-document data the scorer needs but the postings do not carry.
#[derive(Debug, Clone)]
struct StoredDoc {
    uri: String,
    /// Token counts per field, for length normalisation.
    lengths: [u32; 3],
    /// PageRank, or `1/n` when no link graph was computed. Stored rather than
    /// recomputed because ranking a query must not depend on the whole graph
    /// being in memory.
    rank: f32,
}

/// The most one posting can contribute to a score, with the `idf` left out.
///
/// Leaving `idf` out is what keeps a stored bound correct when the query
/// supplies a different one, which it does whenever this segment is one shard
/// of several: `idf` is a positive factor common to every document in a
/// postings list, so `idf * max(rest)` still bounds `idf * rest` for all of
/// them. Authority is folded in, because it multiplies the whole score.
#[allow(clippy::cast_precision_loss)]
fn posting_bound(
    occurrences: &DocOccurrences,
    meta: &StoredDoc,
    average_length: f32,
    doc_count: usize,
) -> f32 {
    let weighted: f32 = occurrences
        .fields
        .iter()
        .enumerate()
        .map(|(i, f)| Field::ALL[i].weight() * f.positions.len() as f32)
        .sum();
    let norm = length_norm(meta.lengths.iter().sum(), average_length);
    saturation(weighted, norm) * authority(meta.rank, doc_count)
}

/// Accumulates documents and produces a segment.
#[derive(Debug, Default)]
pub struct SegmentBuilder {
    /// `BTreeMap` keeps terms sorted as we go, which is exactly the order the
    /// term dictionary needs on disk — no separate sort pass.
    terms: BTreeMap<String, BTreeMap<DocId, DocOccurrences>>,
    docs: Vec<StoredDoc>,
}

impl SegmentBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn document_count(&self) -> usize {
        self.docs.len()
    }

    #[must_use]
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Adds a document and returns the id it was given.
    ///
    /// Ids are handed out densely in insertion order, so postings come out
    /// already ascending and delta encoding just works.
    ///
    /// # Panics
    ///
    /// If more than `u32::MAX` documents are added to one segment. Segments
    /// are meant to be merged, not grown without bound.
    pub fn add(&mut self, doc: &Document) -> DocId {
        let id = DocId(u32::try_from(self.docs.len()).expect("more than u32::MAX documents"));

        let mut lengths = [0u32; 3];
        let mut tokens = Vec::new();

        for field in Field::ALL {
            tokens.clear();
            let count = match field {
                Field::Title => tokenize_into(&doc.title, 0, &mut tokens),
                Field::Body => tokenize_into(&doc.body, 0, &mut tokens),
                Field::Anchor => {
                    // All anchor texts share one position space, so a phrase
                    // cannot accidentally straddle two different links.
                    let mut next = 0;
                    for anchor in &doc.anchors {
                        next = tokenize_into(anchor, next, &mut tokens);
                    }
                    next
                }
            };
            lengths[field as usize] = count;

            for token in &tokens {
                self.terms
                    .entry(token.text.clone())
                    .or_default()
                    .entry(id)
                    .or_default()
                    .fields[field as usize]
                    .positions
                    .push(token.position);
            }
        }

        self.docs.push(StoredDoc {
            uri: doc.uri.clone(),
            lengths,
            rank: 0.0,
        });
        id
    }

    /// Sets a document's PageRank. Called after the crawl, once the link graph
    /// is complete and the ranks have been computed over it.
    pub fn set_rank(&mut self, id: DocId, rank: f32) {
        if let Some(doc) = self.docs.get_mut(id.as_usize()) {
            doc.rank = rank;
        }
    }

    /// The uri of a document, so a caller can match ids to graph nodes.
    #[must_use]
    pub fn uri(&self, id: DocId) -> Option<&str> {
        self.docs.get(id.as_usize()).map(|d| d.uri.as_str())
    }

    /// Rebuilds a builder from finished segments, as one index.
    ///
    /// This is what makes indexing incremental. Without it a crawl that adds a
    /// thousand pages has to re-tokenise the whole corpus, because a segment
    /// keeps postings, not text — the words are gone, only where they were is
    /// left. So a merge works at the postings level: it reads what each
    /// segment knows and writes it out as one.
    ///
    /// Documents keep their order, segment by segment, and their ids are
    /// shifted the same way [`SegmentBuilder::absorb`] shifts them, so the
    /// result is exactly the segment a single pass over all the documents in
    /// that order would have produced.
    pub fn from_segments(segments: &[&Segment]) -> Result<Self> {
        let mut builder = Self::new();

        for segment in segments {
            let shift = u32::try_from(builder.docs.len())
                .map_err(|_| Error::Corrupt("more than u32::MAX documents".into()))?;

            for i in 0..segment.document_count() {
                let id = DocId(u32::try_from(i).unwrap_or(u32::MAX));
                let meta = segment
                    .doc(id)
                    .ok_or_else(|| Error::Corrupt("document store is short".into()))?;
                builder.docs.push(StoredDoc {
                    uri: meta.uri.clone(),
                    lengths: meta.lengths,
                    rank: meta.rank,
                });
            }

            for term in segment.terms() {
                let term = term?;
                // Positions are needed: a merged segment that lost them would
                // answer phrase queries with nothing and look merely unlucky.
                let postings = segment.postings(&term)?;
                let entry = builder.terms.entry(term).or_default();
                for posting in postings {
                    let occurrences = entry.entry(DocId(posting.doc.0 + shift)).or_default();
                    for field in posting.fields {
                        occurrences.fields[field.field as usize].positions =
                            field.positions().to_vec();
                    }
                }
            }
        }
        Ok(builder)
    }

    /// Absorbs another builder, as if its documents had been added after
    /// these ones.
    ///
    /// This is what makes indexing parallelisable: tokenising a document is
    /// independent of every other document, so a corpus can be split, built on
    /// several threads, and stitched back together here. Document ids from
    /// `other` are shifted by however many documents this builder already
    /// holds, which keeps them dense, ascending and in corpus order — exactly
    /// the invariant delta encoding depends on.
    ///
    /// # Panics
    ///
    /// If the two builders together hold more than `u32::MAX` documents.
    pub fn absorb(&mut self, other: Self) {
        let shift = u32::try_from(self.docs.len()).expect("more than u32::MAX documents");

        for (term, docs) in other.terms {
            let entry = self.terms.entry(term).or_default();
            for (id, occurrences) in docs {
                // The shifted id cannot collide: `other`'s ids start at zero
                // and this builder's stop at `shift - 1`.
                entry.insert(DocId(id.0 + shift), occurrences);
            }
        }
        self.docs.extend(other.docs);
    }

    /// Serialises the segment. See `segment.rs` for the layout.
    ///
    /// # Panics
    ///
    /// If the term dictionary alone would exceed 4 GiB, which the fixed-width
    /// offset table cannot address.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // Needed to compute block bounds, and only knowable once every
        // document is in: which is why bounds are written, not computed live.
        let doc_count = self.docs.len();
        // Computed exactly as `Segment::average_document_length` does. A bound
        // derived from a different average is not a bound, and the failure
        // mode is a result quietly missing from a ranking.
        #[allow(clippy::cast_precision_loss)]
        let average_length = if doc_count == 0 {
            0.0
        } else {
            self.docs
                .iter()
                .map(|d| d.lengths.iter().sum::<u32>() as f32)
                .sum::<f32>()
                / doc_count as f32
        };

        // --- postings block -------------------------------------------------
        let postings_offset = out.len() as u64;
        // Offset and document frequency for each term, in term order.
        let mut term_meta: Vec<(u64, u64)> = Vec::with_capacity(self.terms.len());

        for docs in self.terms.values() {
            let offset = out.len() as u64;
            write_varint(docs.len() as u64, &mut out);

            // Postings are written in blocks with a skip index in front, so a
            // query can jump to the block that might hold the document it is
            // looking for instead of decoding everything before it. Without
            // this, intersecting a rare term with a common one costs the
            // common term's entire list, however few documents match both.
            //
            // Each block starts with an absolute document id rather than a
            // delta, which is what makes it independently decodable and so
            // what makes jumping to it possible at all.
            let mut body: Vec<u8> = Vec::new();
            let mut blocks: Vec<(u32, u64, f32)> = Vec::new();
            let mut previous_doc = 0u32;

            for (position, (doc_id, occurrences)) in docs.iter().enumerate() {
                if position % SKIP_INTERVAL == 0 {
                    blocks.push((doc_id.0, body.len() as u64, 0.0));
                    write_varint(u64::from(doc_id.0), &mut body);
                } else {
                    write_varint(u64::from(doc_id.0 - previous_doc), &mut body);
                }
                previous_doc = doc_id.0;

                // The most this posting could contribute, minus the `idf`.
                //
                // Leaving `idf` out is what keeps the bound correct when the
                // query supplies a different one, which it does whenever this
                // segment is a shard: `idf` is a positive factor common to
                // every document in the list, so `idf * max(rest)` still
                // bounds `idf * rest` for every one of them.
                //
                // Authority is folded in, because it multiplies the score.
                if let Some(meta) = self.docs.get(doc_id.as_usize()) {
                    let bound = posting_bound(occurrences, meta, average_length, doc_count);
                    if let Some(last) = blocks.last_mut() {
                        last.2 = last.2.max(bound);
                    }
                }

                let out = &mut body;

                let present = occurrences
                    .fields
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| !f.positions.is_empty());
                write_varint(present.clone().count() as u64, out);

                for (field_index, field) in present {
                    out.push(field_index as u8);
                    write_varint(field.positions.len() as u64, out);

                    // The byte length of the position block, so a reader that
                    // does not want positions can jump the whole thing with an
                    // addition instead of walking every varint in it. Costs
                    // one or two bytes per field per document; saves reading
                    // most of the index on any query without a phrase.
                    let mut positions = Vec::new();
                    write_deltas(&field.positions, &mut positions);
                    write_varint(positions.len() as u64, out);
                    out.extend_from_slice(&positions);
                }
            }

            write_varint(blocks.len() as u64, &mut out);
            for (first_doc, at, bound) in &blocks {
                write_varint(u64::from(*first_doc), &mut out);
                write_varint(*at, &mut out);
                out.extend_from_slice(&bound.to_le_bytes());
            }
            out.extend_from_slice(&body);
            term_meta.push((offset, docs.len() as u64));
        }

        // --- document store -------------------------------------------------
        let doc_store_offset = out.len() as u64;
        write_varint(self.docs.len() as u64, &mut out);
        for doc in &self.docs {
            write_varint(doc.uri.len() as u64, &mut out);
            out.extend_from_slice(doc.uri.as_bytes());
            for length in doc.lengths {
                write_varint(u64::from(length), &mut out);
            }
            // Raw f32 bits: ranks are tiny fractions, and a varint of a scaled
            // integer would lose precision exactly where it matters, among the
            // many pages whose ranks differ in the sixth decimal.
            out.extend_from_slice(&doc.rank.to_le_bytes());
        }

        // --- term dictionary ------------------------------------------------
        let term_dict_offset = out.len() as u64;
        let mut entry_offsets: Vec<u32> = Vec::with_capacity(self.terms.len());
        for (term, (postings_at, doc_freq)) in self.terms.keys().zip(&term_meta) {
            let relative = out.len() as u64 - term_dict_offset;
            entry_offsets.push(
                u32::try_from(relative).expect("term dictionary larger than 4 GiB in one segment"),
            );
            write_varint(term.len() as u64, &mut out);
            out.extend_from_slice(term.as_bytes());
            write_varint(*postings_at, &mut out);
            write_varint(*doc_freq, &mut out);
        }

        // --- term offset table ----------------------------------------------
        // Fixed-width so the reader can binary search the dictionary without
        // decoding it, and without allocating anything.
        let term_offsets_offset = out.len() as u64;
        for offset in &entry_offsets {
            out.extend_from_slice(&offset.to_le_bytes());
        }

        // --- footer ---------------------------------------------------------
        // Everything written so far is what the digest covers: the footer
        // carries it and therefore cannot be part of it.
        let digest = crate::segment::digest_of(&out);

        let footer_start = out.len();
        for value in [
            postings_offset,
            doc_store_offset,
            term_dict_offset,
            term_offsets_offset,
            self.terms.len() as u64,
            self.docs.len() as u64,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&digest.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(MAGIC);
        debug_assert_eq!(out.len() - footer_start, FOOTER_LEN);

        out
    }

    /// Writes the segment to `path`, replacing whatever was there.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        let bytes = self.encode();

        // Written to a sibling and renamed, never in place.
        //
        // Two reasons, and the second is not optional. First, a reader never
        // sees a half-written segment: rename is atomic, so the file at `path`
        // is either the old segment or the new one. Second, and this is what
        // makes memory mapping sound: `File::create` on an existing path
        // *truncates* it, and truncating a file that another process has
        // mapped pulls the memory out from under it — the exact undefined
        // behaviour `Segment::open`'s safety comment says cannot happen.
        // Renaming replaces the directory entry and leaves the old inode
        // alone, so a reader that already mapped it keeps a valid mapping
        // until it lets go.
        let temporary = path.with_extension("ixdr.tmp");
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)?;
        Ok(())
    }
}
