//! Core types shared by every part of the engine.
//!
//! Deliberately dependency-free: this crate defines the vocabulary
//! (`DocId`, `Document`, `Error`) and nothing else.

use std::fmt;

/// Internal, dense document identifier.
///
/// Dense and `u32` on purpose: postings lists are the hottest data in the
/// engine, and a 4-byte id keeps them small enough to stay in cache.
/// Four billion documents per shard is the same ceiling the 2004 design
/// projected for the whole web.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocId(pub u32);

impl DocId {
    /// Sentinel used by postings iterators to signal exhaustion.
    pub const INVALID: Self = Self(u32::MAX);

    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for DocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Position of a token within a document, in token counts (not bytes).
pub type Position = u32;

/// A document as handed to the indexer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Document {
    /// Stable external identity: a URL, a file path, anything.
    pub uri: String,
    /// Short text weighted higher at scoring time.
    pub title: String,
    /// The main body.
    pub body: String,
    /// Text of the links pointing *at* this document, from elsewhere.
    ///
    /// This is the 2004 design's `anchor_queue` idea, kept: a page is often
    /// better described by how others link to it than by what it says.
    pub anchors: Vec<String>,
    /// Absolute URLs this document links *to*.
    ///
    /// Not indexed. This is the raw material of the link graph, and therefore
    /// of PageRank: the crawler is the only component that sees these, so it
    /// has to carry them out.
    pub links: Vec<String>,
}

impl Document {
    #[must_use]
    pub fn new(uri: impl Into<String>, title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            title: title.into(),
            body: body.into(),
            anchors: Vec::new(),
            links: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_anchor(mut self, anchor: impl Into<String>) -> Self {
        self.anchors.push(anchor.into());
        self
    }
}

/// The fields a document is split into. Each is indexed separately so that
/// scoring can weight them differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Field {
    Title = 0,
    Body = 1,
    Anchor = 2,
}

impl Field {
    pub const ALL: [Self; 3] = [Self::Title, Self::Body, Self::Anchor];

    /// Weight applied to a term occurrence in this field.
    ///
    /// Title and anchor text are strong signals of what a page is *about*;
    /// body text is the baseline.
    #[must_use]
    pub const fn weight(self) -> f32 {
        match self {
            Self::Title => 3.0,
            Self::Body => 1.0,
            Self::Anchor => 2.0,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The bytes on disk are not a segment this build understands.
    Corrupt(String),
    /// A query could not be parsed.
    Query(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "i/o error: {e}"),
            Self::Corrupt(m) => write!(f, "corrupt segment: {m}"),
            Self::Query(m) => write!(f, "bad query: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docid_is_four_bytes() {
        assert_eq!(size_of::<DocId>(), 4);
    }

    #[test]
    fn title_and_anchor_outweigh_body() {
        assert!(Field::Title.weight() > Field::Body.weight());
        assert!(Field::Anchor.weight() > Field::Body.weight());
    }

    #[test]
    fn builder_collects_anchors() {
        let d = Document::new("http://a", "T", "B").with_anchor("click here");
        assert_eq!(d.anchors, ["click here"]);
    }
}
