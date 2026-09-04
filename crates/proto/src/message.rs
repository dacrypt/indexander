//! What a coordinator and a shard say to each other.
//!
//! Two round trips per query, and the first one is the interesting half.
//!
//! BM25 weights a term by how rare it is: `idf` depends on the number of
//! documents containing the term and the number of documents in total. Both
//! are **local** to a shard. A term that is rare in shard 3 and common in
//! shard 7 gets a different weight in each, so merging their top-k compares
//! scores that were never on the same scale — and the merge silently produces
//! a plausible, wrong ranking.
//!
//! So: round one asks every shard for its document count and its frequency
//! for each query term; the coordinator sums them; round two runs the search
//! with those global numbers substituted for the local ones. Every shard then
//! scores on the same scale, and the merge is meaningful.

use indexander_core::Result;

use crate::codec::{Reader, Writer};

/// Bumped whenever the wire format changes in a way an older peer would
/// misread. Both sides check it on the first frame.
pub const PROTOCOL_VERSION: u32 = 1;

/// What the coordinator asks a shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// Round one: how many documents do you hold, and how many contain each
    /// of these terms?
    TermStats { terms: Vec<String> },
    /// Round two: search, scoring with these global numbers rather than your
    /// own. `global_doc_count` of zero means "use your own", which is what a
    /// single-shard deployment does.
    Search {
        query: String,
        limit: usize,
        global_doc_count: usize,
        /// `(term, document frequency)` summed across every shard.
        global_doc_freq: Vec<(String, usize)>,
    },
    /// Diagnostics.
    Stats,
    /// May I make `permits` requests to `host`?
    ///
    /// Sent to whichever node owns that host's rate limit, which is a
    /// different mapping from the one that owns its URLs — see
    /// `docs/DISTRIBUTION.md`.
    Lease {
        host: String,
        /// The delay the asker believes the host wants, in milliseconds.
        /// The authority may enforce more, never less.
        requested_delay_ms: u64,
        permits: usize,
    },
}

/// What a shard answers.
#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    TermStats {
        doc_count: usize,
        /// Parallel to the requested terms, in the same order.
        doc_freq: Vec<usize>,
    },
    Hits {
        hits: Vec<Hit>,
    },
    Stats {
        documents: usize,
        terms: usize,
        average_length: f32,
    },
    /// Permission granted, starting in `wait_ms` from now.
    Lease {
        wait_ms: u64,
        permits: usize,
        /// The gap to leave between the granted requests.
        spacing_ms: u64,
    },
    /// The shard could not do what was asked. Carried rather than dropped so
    /// the coordinator can report which shard failed and why.
    Error {
        message: String,
    },
}

/// One result, with enough to merge it without asking again.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub uri: String,
    pub score: f32,
}

// Discriminants are written out rather than derived from declaration order:
// reordering the enum must not silently change the wire format.
const REQ_TERM_STATS: u8 = 1;
const REQ_SEARCH: u8 = 2;
const REQ_STATS: u8 = 3;
const REQ_LEASE: u8 = 4;

const RES_TERM_STATS: u8 = 1;
const RES_HITS: u8 = 2;
const RES_STATS: u8 = 3;
const RES_ERROR: u8 = 4;
const RES_LEASE: u8 = 5;

impl Request {
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            Self::TermStats { terms } => {
                w.u8(REQ_TERM_STATS);
                w.usize(terms.len());
                for term in terms {
                    w.str(term);
                }
            }
            Self::Search {
                query,
                limit,
                global_doc_count,
                global_doc_freq,
            } => {
                w.u8(REQ_SEARCH);
                w.str(query);
                w.usize(*limit);
                w.usize(*global_doc_count);
                w.usize(global_doc_freq.len());
                for (term, freq) in global_doc_freq {
                    w.str(term);
                    w.usize(*freq);
                }
            }
            Self::Stats => w.u8(REQ_STATS),
            Self::Lease {
                host,
                requested_delay_ms,
                permits,
            } => {
                w.u8(REQ_LEASE);
                w.str(host);
                w.varint(*requested_delay_ms);
                w.usize(*permits);
            }
        }
        w.finish()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let message = match r.u8()? {
            REQ_TERM_STATS => {
                let count = r.count()?;
                let mut terms = Vec::with_capacity(count);
                for _ in 0..count {
                    terms.push(r.string()?);
                }
                Self::TermStats { terms }
            }
            REQ_SEARCH => {
                let query = r.string()?;
                let limit = r.usize()?;
                let global_doc_count = r.usize()?;
                let count = r.count()?;
                let mut global_doc_freq = Vec::with_capacity(count);
                for _ in 0..count {
                    let term = r.string()?;
                    global_doc_freq.push((term, r.usize()?));
                }
                Self::Search {
                    query,
                    limit,
                    global_doc_count,
                    global_doc_freq,
                }
            }
            REQ_STATS => Self::Stats,
            REQ_LEASE => Self::Lease {
                host: r.string()?,
                requested_delay_ms: r.varint()?,
                permits: r.usize()?,
            },
            tag => {
                return Err(indexander_core::Error::Corrupt(format!(
                    "unknown request tag {tag}"
                )));
            }
        };
        r.finish()?;
        Ok(message)
    }
}

impl Response {
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            Self::TermStats {
                doc_count,
                doc_freq,
            } => {
                w.u8(RES_TERM_STATS);
                w.usize(*doc_count);
                w.usize(doc_freq.len());
                for freq in doc_freq {
                    w.usize(*freq);
                }
            }
            Self::Hits { hits } => {
                w.u8(RES_HITS);
                w.usize(hits.len());
                for hit in hits {
                    w.str(&hit.uri);
                    w.f32(hit.score);
                }
            }
            Self::Stats {
                documents,
                terms,
                average_length,
            } => {
                w.u8(RES_STATS);
                w.usize(*documents);
                w.usize(*terms);
                w.f32(*average_length);
            }
            Self::Lease {
                wait_ms,
                permits,
                spacing_ms,
            } => {
                w.u8(RES_LEASE);
                w.varint(*wait_ms);
                w.usize(*permits);
                w.varint(*spacing_ms);
            }
            Self::Error { message } => {
                w.u8(RES_ERROR);
                w.str(message);
            }
        }
        w.finish()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let message = match r.u8()? {
            RES_TERM_STATS => {
                let doc_count = r.usize()?;
                let count = r.count()?;
                let mut doc_freq = Vec::with_capacity(count);
                for _ in 0..count {
                    doc_freq.push(r.usize()?);
                }
                Self::TermStats {
                    doc_count,
                    doc_freq,
                }
            }
            RES_HITS => {
                let count = r.count()?;
                let mut hits = Vec::with_capacity(count);
                for _ in 0..count {
                    let uri = r.string()?;
                    hits.push(Hit {
                        uri,
                        score: r.f32()?,
                    });
                }
                Self::Hits { hits }
            }
            RES_STATS => Self::Stats {
                documents: r.usize()?,
                terms: r.usize()?,
                average_length: r.f32()?,
            },
            RES_LEASE => Self::Lease {
                wait_ms: r.varint()?,
                permits: r.usize()?,
                spacing_ms: r.varint()?,
            },
            RES_ERROR => Self::Error {
                message: r.string()?,
            },
            tag => {
                return Err(indexander_core::Error::Corrupt(format!(
                    "unknown response tag {tag}"
                )));
            }
        };
        r.finish()?;
        Ok(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_request(request: &Request) {
        let decoded = Request::decode(&request.encode()).expect("should decode");
        assert_eq!(&decoded, request);
    }

    #[test]
    fn every_request_roundtrips() {
        roundtrip_request(&Request::Stats);
        roundtrip_request(&Request::Lease {
            host: "example.com".into(),
            requested_delay_ms: 500,
            permits: 20,
        });
        roundtrip_request(&Request::TermStats { terms: Vec::new() });
        roundtrip_request(&Request::TermStats {
            terms: vec!["motor".into(), "búsqueda".into()],
        });
        roundtrip_request(&Request::Search {
            query: r#""motor de busqueda" -perl"#.into(),
            limit: 10,
            global_doc_count: 1_000_000,
            global_doc_freq: vec![("motor".into(), 4321), ("perl".into(), 7)],
        });
    }

    #[test]
    fn every_response_roundtrips() {
        for response in [
            Response::TermStats {
                doc_count: 500,
                doc_freq: vec![1, 2, 3],
            },
            Response::Hits {
                hits: vec![Hit {
                    uri: "http://example.com/ñ".into(),
                    score: 1.25,
                }],
            },
            Response::Stats {
                documents: 10,
                terms: 20,
                average_length: 33.5,
            },
            Response::Lease {
                wait_ms: 1500,
                permits: 20,
                spacing_ms: 500,
            },
            Response::Error {
                message: "no index loaded".into(),
            },
        ] {
            let decoded = Response::decode(&response.encode()).expect("should decode");
            assert_eq!(decoded, response);
        }
    }

    #[test]
    fn an_unknown_tag_is_an_error_not_a_panic() {
        assert!(Request::decode(&[99]).is_err());
        assert!(Response::decode(&[99]).is_err());
        assert!(Request::decode(&[]).is_err());
    }

    #[test]
    fn a_truncated_frame_never_panics() {
        let bytes = Request::Search {
            query: "motor de busqueda".into(),
            limit: 10,
            global_doc_count: 99,
            global_doc_freq: vec![("motor".into(), 5)],
        }
        .encode();
        for cut in 0..bytes.len() {
            let _ = Request::decode(&bytes[..cut]);
        }
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = Request::Stats.encode();
        bytes.push(0);
        assert!(Request::decode(&bytes).is_err());
    }
}
