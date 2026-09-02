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

use indexander_core::{DocId, Document, Field, Position, Result};

use crate::codec::{write_deltas, write_varint};
use crate::segment::{FOOTER_LEN, MAGIC, VERSION};
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

    /// Serialises the segment. See `segment.rs` for the layout.
    ///
    /// # Panics
    ///
    /// If the term dictionary alone would exceed 4 GiB, which the fixed-width
    /// offset table cannot address.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // --- postings block -------------------------------------------------
        let postings_offset = out.len() as u64;
        // Offset and document frequency for each term, in term order.
        let mut term_meta: Vec<(u64, u64)> = Vec::with_capacity(self.terms.len());

        for docs in self.terms.values() {
            let offset = out.len() as u64;
            write_varint(docs.len() as u64, &mut out);

            let mut previous_doc = 0u32;
            for (doc_id, occurrences) in docs {
                write_varint(u64::from(doc_id.0 - previous_doc), &mut out);
                previous_doc = doc_id.0;

                let present = occurrences
                    .fields
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| !f.positions.is_empty());
                write_varint(present.clone().count() as u64, &mut out);

                for (field_index, field) in present {
                    out.push(field_index as u8);
                    write_varint(field.positions.len() as u64, &mut out);
                    write_deltas(&field.positions, &mut out);
                }
            }
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
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(MAGIC);
        debug_assert_eq!(out.len() - footer_start, FOOTER_LEN);

        out
    }

    /// Writes the segment to `path`, replacing whatever was there.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        let bytes = self.encode();
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(())
    }
}
