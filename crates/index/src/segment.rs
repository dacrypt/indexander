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
//! A segment is memory mapped, not read. The difference is not the copy — it
//! is that a shard holding a 236 MiB index no longer holds 236 MiB of resident
//! memory, and no longer waits to read all of it before answering anything.
//! Pages arrive as they are touched, and a query touches a term dictionary
//! entry and one postings list.
//!
//! [`Segment::from_bytes`] still exists for indexes built in memory, and both
//! paths are the same code behind a slice.

use std::path::Path;

use indexander_core::{DocId, Error, Field, Position, Result};

use crate::codec::{read_deltas, read_varint};

pub(crate) const MAGIC: &[u8; 4] = b"IXDR";
pub(crate) const VERSION: u32 = 6;
/// Postings per skip block.
///
/// Small enough that decoding one block to find a document is cheap; large
/// enough that the skip index stays a small fraction of the postings. 128 is
/// the number every implementation of this lands on, for the same reasons.
pub(crate) const SKIP_INTERVAL: usize = 128;
/// Six u64 offsets, a u64 digest, a u32 version, a 4-byte magic.
pub(crate) const FOOTER_LEN: usize = 6 * 8 + 8 + 4 + 4;

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

impl FieldPosting {
    /// Where the term occurs, or empty if positions were not read.
    #[must_use]
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }
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

/// Where a segment's bytes live.
///
/// Both arms hand out a `&[u8]`, so nothing downstream knows or cares which
/// one it got.
#[derive(Debug)]
enum Backing {
    /// Built in memory, or read from a file that could not be mapped.
    Owned(Vec<u8>),
    /// Mapped from a file. The `Mmap` owns the mapping and unmaps on drop.
    Mapped(memmap2::Mmap),
}

impl std::ops::Deref for Backing {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Mapped(map) => map,
        }
    }
}

/// An immutable, searchable segment.
#[derive(Debug)]
pub struct Segment {
    bytes: Backing,
    digest: u64,
    postings_offset: usize,
    term_dict_offset: usize,
    term_offsets_offset: usize,
    num_terms: usize,
    docs: Vec<DocMeta>,
    average_length: f32,
}

impl Segment {
    /// Parses a segment from bytes already in memory.
    ///
    /// # Panics
    ///
    /// Never, for any input: the footer is length-checked before it is sliced,
    /// and the fixed-width reads inside that slice cannot fail. The
    /// `expect`s below document that invariant rather than relying on it.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_backing(Backing::Owned(bytes))
    }

    /// Opens a segment by memory mapping it.
    ///
    /// Falls back to reading the file when it cannot be mapped — an empty
    /// file, a filesystem that does not support it — because a slower segment
    /// is better than no segment.
    ///
    /// # Safety
    ///
    /// `Mmap::map` is unsafe because the mapping reflects the file: if another
    /// process truncates or rewrites it, the bytes behind a live `&[u8]`
    /// change, which is undefined behaviour. This engine never does that.
    /// Segments are written once and never modified, and
    /// [`SegmentBuilder::write_to`](crate::builder::SegmentBuilder::write_to)
    /// writes to a temporary file and renames it into place, so replacing a
    /// segment creates a new inode and leaves any existing mapping pointing at
    /// the old one until its reader drops it.
    ///
    /// What remains outside that guarantee is somebody editing a segment file
    /// by hand while a shard is serving it. That is the documented contract:
    /// segment files are immutable while a process holds them.
    #[allow(unsafe_code)]
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: see the contract above - segments are never modified in
        // place, and writes go through a rename.
        match unsafe { memmap2::Mmap::map(&file) } {
            Ok(map) => Self::from_backing(Backing::Mapped(map)),
            Err(_) => Self::from_bytes(std::fs::read(path)?),
        }
    }

    fn from_backing(bytes: Backing) -> Result<Self> {
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
        let digest = u64::from_le_bytes(footer[6 * 8..7 * 8].try_into().expect("8 bytes"));
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
            digest,
            postings_offset,
            term_dict_offset,
            term_offsets_offset,
            num_terms,
            docs,
            average_length,
        })
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

    /// The segment's raw bytes, exactly as they are on disk.
    ///
    /// Exposed so a node can serve a copy of itself to a replica without the
    /// replication code needing to know anything about the format. Reading
    /// them all touches every page of a memory-mapped segment, which is fine
    /// for a transfer and would not be for a query.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The digest recorded when this segment was written.
    ///
    /// Reading it is free — it sits in the footer. Two segments with the same
    /// digest are the same segment, which is what makes a replica checkable
    /// rather than assumed.
    #[must_use]
    pub fn digest(&self) -> u64 {
        self.digest
    }

    /// Recomputes the digest over the actual bytes and compares.
    ///
    /// Not done on open: a 248 MB segment is memory mapped precisely so that
    /// a query touches a few pages, and hashing it would touch all of them.
    /// This is for after a transfer, when the question is whether what
    /// arrived is what was sent.
    ///
    /// It detects corruption, not tampering. Anyone who can rewrite a segment
    /// can rewrite its footer, and defending against that is a different
    /// problem needing a different tool.
    #[must_use]
    pub fn verify(&self) -> bool {
        let body = self.bytes.len().saturating_sub(FOOTER_LEN);
        digest_of(&self.bytes[..body]) == self.digest
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

    /// Tokens across every document, which is what an average is made of.
    ///
    /// A corpus-wide average cannot be assembled from per-segment averages
    /// without also knowing how many documents each was over, so this is the
    /// number that travels.
    #[must_use]
    pub fn total_length(&self) -> u64 {
        self.docs.iter().map(|d| u64::from(d.total_length())).sum()
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

    /// Every term in the segment, in sorted order.
    ///
    /// Sorted because the dictionary is, which is what lets a merge walk two
    /// segments together instead of loading either one into memory.
    pub fn terms(&self) -> impl Iterator<Item = Result<String>> + '_ {
        (0..self.num_terms).map(move |i| {
            let (bytes, _, _) = self.entry(i)?;
            String::from_utf8(bytes.to_vec())
                .map_err(|_| Error::Corrupt("term is not utf-8".into()))
        })
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

    /// Reads a term's header: where its postings body starts and where each
    /// skip block begins.
    fn term_blocks(&self, term: &str) -> Result<Option<TermLayout>> {
        let Some((offset, _)) = self.lookup(term)? else {
            return Ok(None);
        };
        let mut cursor = self.postings_offset.max(offset);
        let doc_count = read_varint(&self.bytes, &mut cursor)? as usize;
        let block_count = read_varint(&self.bytes, &mut cursor)? as usize;

        let mut blocks = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let first_doc = u32::try_from(read_varint(&self.bytes, &mut cursor)?)
                .map_err(|_| Error::Corrupt("block start exceeds u32".into()))?;
            let at = read_varint(&self.bytes, &mut cursor)? as usize;
            let bound: [u8; 4] = self
                .bytes
                .get(cursor..cursor + 4)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| Error::Corrupt("skip index ends mid-bound".into()))?;
            cursor += 4;
            blocks.push((first_doc, at, f32::from_le_bytes(bound)));
        }
        // Offsets in the header are relative to the body, which begins here.
        let body = cursor;
        for (_, at, _) in &mut blocks {
            *at += body;
            if *at > self.bytes.len() {
                return Err(Error::Corrupt("skip offset runs past end".into()));
            }
        }
        Ok(Some((doc_count, body, blocks)))
    }

    /// A cursor over `term`'s postings that can skip forward.
    pub fn cursor(&self, term: &str, want_positions: bool) -> Result<PostingsCursor<'_>> {
        let Some((doc_count, body, blocks)) = self.term_blocks(term)? else {
            return Ok(PostingsCursor::empty(self));
        };
        let mut cursor = PostingsCursor {
            segment: self,
            doc_count,
            blocks,
            want_positions,
            at: body,
            index: 0,
            doc: 0,
            fields: Vec::new(),
            exhausted: doc_count == 0,
            decoded: 0,
            jumps: 0,
        };
        if doc_count > 0 {
            cursor.read_here(true)?;
        }
        Ok(cursor)
    }

    fn decode_postings(&self, term: &str, want_positions: bool) -> Result<Vec<Posting>> {
        let Some((doc_count, body, _)) = self.term_blocks(term)? else {
            return Ok(Vec::new());
        };
        let mut cursor = body;

        let mut out = Vec::with_capacity(doc_count);
        let mut doc = 0u32;
        let mut positions = Vec::new();
        for position in 0..doc_count {
            let value = u32::try_from(read_varint(&self.bytes, &mut cursor)?)
                .map_err(|_| Error::Corrupt("document gap exceeds u32".into()))?;
            // Every block restarts from an absolute id.
            doc = if position % SKIP_INTERVAL == 0 {
                value
            } else {
                doc + value
            };
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

/// A 64-bit digest of a segment's contents.
///
/// FNV-1a with an avalanche step at the end: fast enough to run over a
/// quarter of a gigabyte without thinking about it, and good enough to catch
/// a truncated transfer, a flipped bit or a half-written file. It is not a
/// cryptographic hash and is not used as one.
///
/// # Panics
///
/// Never: the only `expect` is on a slice `chunks_exact(8)` has already
/// guaranteed is eight bytes long.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn digest_of(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    // Eight bytes at a time: the compiler turns this into far fewer
    // instructions than a byte-at-a-time loop, and the mixing is the same.
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        hash ^= word;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    for byte in chunks.remainder() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    // Length is mixed in so that trailing zeroes cannot be added or removed
    // without changing the digest.
    hash ^= bytes.len() as u64;
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

/// How a term's postings are laid out: how many documents, where the body
/// starts, and `(first document, byte offset)` for each skip block.
/// `(first document, byte offset, score bound)` for one skip block.
///
/// The bound is the largest contribution any posting in the block can make,
/// with the `idf` factored out — see `SegmentBuilder::encode` for why.
pub type Block = (u32, usize, f32);

type TermLayout = (usize, usize, Vec<Block>);

/// A forward-only cursor over one term's postings, able to skip.
///
/// This is what makes intersecting a rare term with a common one cost the rare
/// term. Decoding both lists in full and intersecting the results costs the
/// common one — for a query like `kubernetes the`, that is a hundred thousand
/// postings decoded to find a handful of documents that hold both.
///
/// [`seek`](PostingsCursor::seek) binary searches the skip index for the block
/// that could contain the target and decodes forward from there, so the work
/// is one block plus a logarithm rather than the whole list.
#[derive(Debug)]
pub struct PostingsCursor<'a> {
    segment: &'a Segment,
    doc_count: usize,
    /// `(first document, absolute byte offset)` for each block.
    blocks: Vec<Block>,
    want_positions: bool,
    /// Byte offset of the posting the cursor is sitting on.
    at: usize,
    /// Its index within the term's postings.
    index: usize,
    doc: u32,
    fields: Vec<FieldPosting>,
    exhausted: bool,
    /// Postings actually decoded, and blocks jumped into. Diagnostics, and
    /// the only way to answer "would skipping more blocks help?" with a
    /// number instead of an opinion.
    decoded: usize,
    jumps: usize,
}

impl<'a> PostingsCursor<'a> {
    fn empty(segment: &'a Segment) -> Self {
        Self {
            segment,
            doc_count: 0,
            blocks: Vec::new(),
            want_positions: false,
            at: 0,
            index: 0,
            doc: 0,
            fields: Vec::new(),
            exhausted: true,
            decoded: 0,
            jumps: 0,
        }
    }

    /// How many postings this cursor has decoded so far.
    #[must_use]
    pub fn decoded(&self) -> usize {
        self.decoded
    }

    /// How many times it jumped to a block rather than walking to it.
    #[must_use]
    pub fn jumps(&self) -> usize {
        self.jumps
    }

    /// Which skip block the cursor is currently inside.
    #[must_use]
    pub fn current_block(&self) -> usize {
        self.index / SKIP_INTERVAL
    }

    /// How many skip blocks this term's postings occupy.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// The skip blocks: where each starts and what it can contribute.
    #[must_use]
    pub fn block_starts(&self) -> &[Block] {
        &self.blocks
    }

    /// The most any posting in the current block can contribute to a score,
    /// before the query's `idf` is applied.
    ///
    /// `f32::INFINITY` when the cursor is past the end, so a caller that
    /// forgets to check for exhaustion never skips something it should read.
    #[must_use]
    pub fn block_bound(&self) -> f32 {
        if self.exhausted {
            return f32::INFINITY;
        }
        self.blocks
            .get(self.current_block())
            .map_or(f32::INFINITY, |(_, _, bound)| *bound)
    }

    /// Jumps to the first posting of the next block.
    ///
    /// This is what a bound buys: the whole block goes unread.
    pub fn skip_block(&mut self) -> Result<()> {
        if self.exhausted {
            return Ok(());
        }
        let next = self.current_block() + 1;
        if let Some(&(first_doc, at, _)) = self.blocks.get(next) {
            self.at = at;
            self.index = next * SKIP_INTERVAL;
            self.doc = first_doc;
            self.jumps += 1;
            self.read_here(true)
        } else {
            self.exhausted = true;
            Ok(())
        }
    }

    /// How many documents contain this term.
    #[must_use]
    pub fn document_frequency(&self) -> usize {
        self.doc_count
    }

    /// The document the cursor is on, or `None` once it has run out.
    #[must_use]
    pub fn doc(&self) -> Option<DocId> {
        if self.exhausted {
            None
        } else {
            Some(DocId(self.doc))
        }
    }

    /// The term's per-field counts and positions in the current document.
    #[must_use]
    pub fn fields(&self) -> &[FieldPosting] {
        &self.fields
    }

    /// Where the term occurs in `field` in the current document.
    #[must_use]
    pub fn positions_in(&self, field: Field) -> &[Position] {
        self.fields
            .iter()
            .find(|f| f.field == field)
            .map_or(&[], |f| f.positions.as_slice())
    }

    /// Occurrences weighted by field, for the current document.
    #[must_use]
    pub fn weighted_frequency(&self) -> f32 {
        self.fields
            .iter()
            .map(|f| f.field.weight() * f.count as f32)
            .sum()
    }

    /// Advances to the next posting.
    pub fn advance(&mut self) -> Result<()> {
        if self.exhausted {
            return Ok(());
        }
        self.index += 1;
        if self.index >= self.doc_count {
            self.exhausted = true;
            return Ok(());
        }
        let starts_block = self.index % SKIP_INTERVAL == 0;
        self.read_here(starts_block)
    }

    /// Moves to the first document at or after `target`.
    ///
    /// Never moves backwards: a cursor already past `target` stays where it
    /// is, which is what lets callers leapfrog without tracking who is ahead.
    pub fn seek(&mut self, target: DocId) -> Result<()> {
        if self.exhausted || self.doc >= target.0 {
            return Ok(());
        }

        // The last block whose first document is at or before the target. Its
        // predecessor cannot hold the target, and its successor starts past it.
        let candidate = self
            .blocks
            .partition_point(|(first, _, _)| *first <= target.0)
            .saturating_sub(1);
        let block_index = candidate * SKIP_INTERVAL;

        // Only jump forward: the cursor may already be inside a later block.
        if block_index > self.index {
            let (first_doc, at, _) = self.blocks[candidate];
            self.at = at;
            self.index = block_index;
            self.doc = first_doc;
            self.jumps += 1;
            self.read_here(true)?;
        }

        while !self.exhausted && self.doc < target.0 {
            self.advance()?;
        }
        Ok(())
    }

    /// Decodes the posting at the current byte offset.
    fn read_here(&mut self, absolute: bool) -> Result<()> {
        let bytes = &*self.segment.bytes;
        let mut cursor = self.at;

        self.decoded += 1;
        let value = u32::try_from(read_varint(bytes, &mut cursor)?)
            .map_err(|_| Error::Corrupt("document gap exceeds u32".into()))?;
        self.doc = if absolute { value } else { self.doc + value };

        let field_count = read_varint(bytes, &mut cursor)? as usize;
        self.fields.clear();
        let mut positions = Vec::new();
        for _ in 0..field_count {
            let raw = *bytes
                .get(cursor)
                .ok_or_else(|| Error::Corrupt("postings end mid-field".into()))?;
            cursor += 1;
            let field = match raw {
                0 => Field::Title,
                1 => Field::Body,
                2 => Field::Anchor,
                other => return Err(Error::Corrupt(format!("unknown field tag {other}"))),
            };
            let count = read_varint(bytes, &mut cursor)? as usize;
            let block_len = read_varint(bytes, &mut cursor)? as usize;
            if self.want_positions {
                read_deltas(bytes, &mut cursor, count, &mut positions)?;
            } else {
                cursor = cursor
                    .checked_add(block_len)
                    .filter(|c| *c <= bytes.len())
                    .ok_or_else(|| Error::Corrupt("position block runs past end".into()))?;
                positions.clear();
            }
            self.fields.push(FieldPosting {
                field,
                count: u32::try_from(count)
                    .map_err(|_| Error::Corrupt("term frequency exceeds u32".into()))?,
                positions: std::mem::take(&mut positions),
            });
        }
        self.at = cursor;
        Ok(())
    }
}
