// Lengths become `f32` only to feed BM25, which is a ranking function; see the
// note in `search.rs`.
#![allow(clippy::cast_precision_loss)]
// Offsets are stored as `u64` and used as `usize`. On the 64-bit targets this
// engine supports the conversion is lossless; a 32-bit target could not mmap a
// segment that large anyway, and every offset is bounds-checked against the
// buffer before use.
#![allow(clippy::cast_possible_truncation)]

//! Reading a segment back.
//!
//! Layout, in write order:
//!
//! ```text
//! [postings]      one variable-length block per term, in term order
//! [doc store]     uri and per-field token counts for every document
//! [term dict]     sorted terms, each with its postings offset and doc frequency
//! [term offsets]  u32 little-endian offset of each dictionary entry
//! [footer]        six u64 offsets, a version, and a magic number
//! ```
//!
//! The dictionary is searched by binary search over the fixed-width offset
//! table, so looking up a term touches `log2(n)` cache lines and allocates
//! nothing. Terms are UTF-8 and compared as bytes, which for folded lowercase
//! terms is the same order as comparing them as strings.
//!
//! The whole segment is read into memory. Memory mapping would avoid the copy
//! and is the natural next step, but it needs `unsafe` and the copy is not yet
//! what makes anything slow.

use std::path::Path;

use indexander_core::{DocId, Error, Field, Position, Result};

use crate::codec::{read_deltas, read_varint};

pub(crate) const MAGIC: &[u8; 4] = b"IXDR";
pub(crate) const VERSION: u32 = 3;
/// Six u64 offsets, a u32 version, a 4-byte magic.
pub(crate) const FOOTER_LEN: usize = 6 * 8 + 4 + 4;

/// One term's appearance in one field of one document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldPosting {
    pub field: Field,
    /// How many times the term occurs in this field.
    pub count: u32,
    /// Where it occurs — empty unless the postings were read with positions.
    ///
    /// Positions are the bulk of an index: a common term has one document
    /// gap per posting and dozens of positions behind it. Only phrase queries
    /// ever look at them, so decoding them for a query that will not is the
    /// single most expensive thing this engine can do for no reason.
    positions: Vec<Position>,
}

/// One document's appearance in a term's postings list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Posting {
    pub doc: DocId,
    /// One entry per field the term appears in, in written order.
    pub fields: Vec<FieldPosting>,
}

impl Posting {
    /// Total occurrences across all fields, unweighted.
    #[must_use]
    pub fn term_frequency(&self) -> u32 {
        self.fields.iter().map(|f| f.count).sum()
    }

    /// Occurrences weighted by field, which is what the scorer wants.
    ///
    /// Reads counts, never positions, so it works on postings read either way.
    #[must_use]
    pub fn weighted_frequency(&self) -> f32 {
        self.fields
            .iter()
            .map(|f| f.field.weight() * f.count as f32)
            .sum()
    }

    /// Where the term occurs in `field`. Empty if positions were not read.
    #[must_use]
    pub fn positions_in(&self, field: Field) -> &[Position] {
        self.fields
            .iter()
            .find(|f| f.field == field)
            .map_or(&[], |f| f.positions.as_slice())
    }
}

/// Stored metadata for one document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocMeta {
    pub uri: String,
    /// Token count per field, indexed by `Field as usize`.
    pub lengths: [u32; 3],
    /// PageRank of this document, or 0 if the index was built without a graph.
    pub rank: f32,
}

impl DocMeta {
    #[must_use]
    pub fn total_length(&self) -> u32 {
        self.lengths.iter().sum()
    }
}

/// An immutable, searchable segment.
#[derive(Debug)]
pub struct Segment {
    bytes: Vec<u8>,
    postings_offset: usize,
    term_dict_offset: usize,
    term_offsets_offset: usize,
    num_terms: usize,
    docs: Vec<DocMeta>,
    average_length: f32,
}

impl Segment {
    /// Parses a segment from raw bytes, validating the footer.
    ///
    /// # Panics
    ///
    /// Never, for any input: the footer is length-checked before it is sliced,
    /// and the fixed-width reads inside that slice cannot fail. The
    /// `expect`s below document that invariant rather than relying on it.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < FOOTER_LEN {
            return Err(Error::Corrupt("file shorter than a footer".into()));
        }
        let footer = &bytes[bytes.len() - FOOTER_LEN..];
        if &footer[FOOTER_LEN - 4..] != MAGIC {
            return Err(Error::Corrupt("magic number missing".into()));
        }
        let version = u32::from_le_bytes(
            footer[FOOTER_LEN - 8..FOOTER_LEN - 4]
                .try_into()
                .expect("4 bytes"),
        );
        if version != VERSION {
            return Err(Error::Corrupt(format!(
                "segment version {version}, this build reads {VERSION}"
            )));
        }

        let read_u64 = |i: usize| -> usize {
            u64::from_le_bytes(footer[i * 8..i * 8 + 8].try_into().expect("8 bytes")) as usize
        };
        let postings_offset = read_u64(0);
        let doc_store_offset = read_u64(1);
        let term_dict_offset = read_u64(2);
        let term_offsets_offset = read_u64(3);
        let num_terms = read_u64(4);
        let num_docs = read_u64(5);

        let body = bytes.len() - FOOTER_LEN;
        if term_offsets_offset > body
            || term_dict_offset > term_offsets_offset
            || doc_store_offset > term_dict_offset
            || postings_offset > doc_store_offset
        {
            return Err(Error::Corrupt("footer offsets out of order".into()));
        }
        if term_offsets_offset + num_terms * 4 > body {
            return Err(Error::Corrupt("term offset table runs past end".into()));
        }

        let docs = Self::read_doc_store(&bytes, doc_store_offset, num_docs)?;
        let average_length = if docs.is_empty() {
            0.0
        } else {
            docs.iter().map(|d| d.total_length() as f32).sum::<f32>() / docs.len() as f32
        };

        Ok(Self {
            bytes,
            postings_offset,
            term_dict_offset,
            term_offsets_offset,
            num_terms,
            docs,
            average_length,
        })
    }

    pub fn open(path: &Path) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    fn read_doc_store(bytes: &[u8], offset: usize, expected: usize) -> Result<Vec<DocMeta>> {
        let mut cursor = offset;
        let count = read_varint(bytes, &mut cursor)? as usize;
        if count != expected {
            return Err(Error::Corrupt(
                "document count disagrees with footer".into(),
            ));
        }
        let mut docs = Vec::with_capacity(count);
        for _ in 0..count {
            let uri_len = read_varint(bytes, &mut cursor)? as usize;
            let end = cursor
                .checked_add(uri_len)
                .filter(|e| *e <= bytes.len())
                .ok_or_else(|| Error::Corrupt("uri runs past end".into()))?;
            let uri = String::from_utf8(bytes[cursor..end].to_vec())
                .map_err(|_| Error::Corrupt("uri is not utf-8".into()))?;
            cursor = end;
            let mut lengths = [0u32; 3];
            for slot in &mut lengths {
                *slot = u32::try_from(read_varint(bytes, &mut cursor)?)
                    .map_err(|_| Error::Corrupt("field length exceeds u32".into()))?;
            }
            let rank_at = cursor;
            let rank_bytes: [u8; 4] = bytes
                .get(rank_at..rank_at + 4)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| Error::Corrupt("document store ends mid-rank".into()))?;
            cursor += 4;
            docs.push(DocMeta {
                uri,
                lengths,
                rank: f32::from_le_bytes(rank_bytes),
            });
        }
        Ok(docs)
    }

    #[must_use]
    pub fn document_count(&self) -> usize {
        self.docs.len()
    }

    #[must_use]
    pub fn term_count(&self) -> usize {
        self.num_terms
    }

    #[must_use]
    pub fn average_document_length(&self) -> f32 {
        self.average_length
    }

    #[must_use]
    pub fn doc(&self, id: DocId) -> Option<&DocMeta> {
        self.docs.get(id.as_usize())
    }

    /// Reads the dictionary entry at index `i`: term bytes, postings offset,
    /// document frequency.
    fn entry(&self, i: usize) -> Result<(&[u8], usize, usize)> {
        let table_at = self.term_offsets_offset + i * 4;
        let relative = u32::from_le_bytes(
            self.bytes[table_at..table_at + 4]
                .try_into()
                .map_err(|_| Error::Corrupt("truncated term offset".into()))?,
        ) as usize;
        let mut cursor = self.term_dict_offset + relative;
        let term_len = read_varint(&self.bytes, &mut cursor)? as usize;
        let end = cursor
            .checked_add(term_len)
            .filter(|e| *e <= self.bytes.len())
            .ok_or_else(|| Error::Corrupt("term runs past end".into()))?;
        let term = &self.bytes[cursor..end];
        cursor = end;
        let postings_at = read_varint(&self.bytes, &mut cursor)? as usize;
        let doc_freq = read_varint(&self.bytes, &mut cursor)? as usize;
        Ok((term, postings_at, doc_freq))
    }

    /// How many documents contain `term`. Zero if it is not in the segment.
    pub fn document_frequency(&self, term: &str) -> Result<usize> {
        Ok(self.lookup(term)?.map_or(0, |(_, df)| df))
    }

    /// Binary search for a term, returning its postings offset and frequency.
    fn lookup(&self, term: &str) -> Result<Option<(usize, usize)>> {
        let needle = term.as_bytes();
        let (mut low, mut high) = (0usize, self.num_terms);
        while low < high {
            let mid = low + (high - low) / 2;
            let (candidate, postings_at, doc_freq) = self.entry(mid)?;
            match candidate.cmp(needle) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Ok(Some((postings_at, doc_freq))),
            }
        }
        Ok(None)
    }

    /// Decodes the postings list for `term`, positions and all.
    ///
    /// Only phrase queries need this; everything else should use
    /// [`Segment::postings_counts`], which is several times cheaper.
    pub fn postings(&self, term: &str) -> Result<Vec<Posting>> {
        self.decode_postings(term, true)
    }

    /// Decodes the postings list for `term` without its positions.
    ///
    /// The position blocks are still walked — their lengths are varints, so
    /// they cannot be jumped over without reading them — but nothing is
    /// stored and nothing is allocated, which is where the cost actually was.
    pub fn postings_counts(&self, term: &str) -> Result<Vec<Posting>> {
        self.decode_postings(term, false)
    }

    fn decode_postings(&self, term: &str, want_positions: bool) -> Result<Vec<Posting>> {
        let Some((offset, _)) = self.lookup(term)? else {
            return Ok(Vec::new());
        };
        let mut cursor = self.postings_offset.max(offset);
        let doc_count = read_varint(&self.bytes, &mut cursor)? as usize;

        let mut out = Vec::with_capacity(doc_count);
        let mut doc = 0u32;
        let mut positions = Vec::new();
        for _ in 0..doc_count {
            doc += u32::try_from(read_varint(&self.bytes, &mut cursor)?)
                .map_err(|_| Error::Corrupt("document gap exceeds u32".into()))?;
            let field_count = read_varint(&self.bytes, &mut cursor)? as usize;
            let mut fields = Vec::with_capacity(field_count);
            for _ in 0..field_count {
                let raw = *self
                    .bytes
                    .get(cursor)
                    .ok_or_else(|| Error::Corrupt("postings end mid-field".into()))?;
                cursor += 1;
                let field = match raw {
                    0 => Field::Title,
                    1 => Field::Body,
                    2 => Field::Anchor,
                    other => return Err(Error::Corrupt(format!("unknown field tag {other}"))),
                };
                let count = read_varint(&self.bytes, &mut cursor)? as usize;
                let block_len = read_varint(&self.bytes, &mut cursor)? as usize;
                if want_positions {
                    let before = cursor;
                    read_deltas(&self.bytes, &mut cursor, count, &mut positions)?;
                    if cursor - before != block_len {
                        return Err(Error::Corrupt("position block length disagrees".into()));
                    }
                } else {
                    // One addition, whatever the block holds.
                    cursor = cursor
                        .checked_add(block_len)
                        .filter(|c| *c <= self.bytes.len())
                        .ok_or_else(|| Error::Corrupt("position block runs past end".into()))?;
                    positions.clear();
                }
                fields.push(FieldPosting {
                    field,
                    count: u32::try_from(count)
                        .map_err(|_| Error::Corrupt("term frequency exceeds u32".into()))?,
                    positions: std::mem::take(&mut positions),
                });
            }
            out.push(Posting {
                doc: DocId(doc),
                fields,
            });
        }
        Ok(out)
    }
}
