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
#[derive(Debug, Clone, PartialEq)]
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
        /// Tokens across every document of every shard.
        global_total_length: u64,
        /// `(term, document frequency)` summed across every shard.
        global_doc_freq: Vec<(String, usize)>,
    },
    /// Diagnostics.
    Stats,
    /// Prepare for a ranking run. Sent once, before any iteration.
    ///
    /// `total_nodes` is the count across the whole cluster: a shard cannot
    /// know it, and one that guesses produces a vector that does not sum to
    /// one.
    RankInit {
        total_nodes: usize,
        shard_count: usize,
        damping: f32,
        tolerance: f32,
    },
    /// How much rank are your dangling pages holding?
    ///
    /// Asked of everyone before any shard applies an iteration, because the
    /// mass has to be redistributed across the cluster and not within a shard.
    RankDangling,
    /// Push rank along your edges and tell me what leaves you.
    RankEmit,
    /// Here is rank arriving from somewhere else.
    RankAbsorb { contributions: Vec<(String, f32)> },
    /// Apply the iteration, now that the cluster-wide dangling mass is known.
    RankApply { global_dangling: f32 },
    /// Give me your ranks; the run is over.
    RankResults,
    /// What is known about this host's `robots.txt`?
    ///
    /// `learned` empty means the asker is asking; otherwise it is reporting
    /// what it fetched so nobody else has to. `state` is 0 unknown, 1 rules,
    /// 2 unreachable.
    Robots {
        host: String,
        learned: bool,
        state: u8,
        text: String,
    },
    /// What segments make up your index?
    ///
    /// The reply is the manifest as text — the same bytes on disk, so a
    /// replica can write what it was given rather than re-render it.
    Manifest,
    /// What is this segment's digest, and how big is it?
    ///
    /// An empty `name` means "the one you serve", which is what a shard
    /// holding a single segment answers. The digest is what makes a copy
    /// checkable: a replica that fetched the bytes and got a different digest
    /// fetched something else.
    SegmentInfo { name: String },
    /// Send me `len` bytes of a segment starting at `offset`.
    ///
    /// Chunked because a segment is hundreds of megabytes and a frame is not.
    SegmentChunk { name: String, offset: u64, len: u32 },
    /// Catch up with the source this replica was configured to follow, and
    /// serve what it finds afterwards.
    ///
    /// Deliberately empty. It carries no address, no manifest and no segment:
    /// it is a nudge, not an instruction. A replica pulls from the one place
    /// it was started with, so the worst a stranger can do by sending this is
    /// make it check for work it was going to do anyway.
    ///
    /// Losing one costs staleness until the next, which is why the primary
    /// sending them is a convenience and `indexander sync --every` is the
    /// thing that actually guarantees a replica converges.
    Refresh,
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
        /// Tokens across every document this shard holds.
        ///
        /// BM25 discounts a document by its length relative to the *corpus*
        /// average, so the average has to be assembled the same way the
        /// document frequencies are. Without this a shard scores against its
        /// own average and its numbers stop meaning the same thing as another
        /// shard's — the same failure as local `idf`, and just as quiet.
        total_length: u64,
        /// Parallel to the requested terms, in the same order.
        doc_freq: Vec<usize>,
        /// The scoring parameters this shard's segments were written with,
        /// as `[k1, b, authority_weight]`.
        ///
        /// Shards score locally and the coordinator merges the results, so
        /// two shards using different parameters produce numbers that cannot
        /// be compared - a ranking assembled from them would be arbitrary
        /// without being empty, which is the worst way for this to fail. The
        /// coordinator checks these agree and refuses the query if they do
        /// not.
        params: [f32; 3],
    },
    Hits {
        hits: Vec<Hit>,
    },
    Stats {
        documents: usize,
        terms: usize,
        average_length: f32,
        /// How many segments this shard is serving right now.
        ///
        /// Not a curiosity: it is the only thing that changes visibly when a
        /// replica catches up with a merge. Documents survive a merge by
        /// definition, so a replica still on the old segments answers every
        /// query identically to one that has caught up — and the count is what
        /// tells them apart, from outside, without reading its disk.
        segments: usize,
    },
    /// Permission granted, starting in `wait_ms` from now.
    Lease {
        wait_ms: u64,
        permits: usize,
        /// The gap to leave between the granted requests.
        spacing_ms: u64,
    },
    /// Answer to [`Request::Manifest`].
    /// What a replica holds after catching up.
    Refreshed {
        /// Segments fetched from the source. Zero means it was already current.
        fetched: usize,
        segments: usize,
        documents: usize,
    },
    Manifest {
        text: String,
    },
    /// Answer to [`Request::Robots`]: 0 unknown, 1 rules, 2 unreachable.
    Robots {
        state: u8,
        text: String,
    },
    /// Answer to [`Request::SegmentInfo`].
    SegmentInfo {
        digest: u64,
        len: u64,
    },
    /// Answer to [`Request::SegmentChunk`]. Shorter than asked for at the end.
    SegmentChunk {
        bytes: Vec<u8>,
    },
    /// Answer to [`Request::RankDangling`].
    RankDangling {
        dangling: f32,
    },
    /// One bundle of contributions per destination shard, indexed by shard.
    RankBoundaries {
        per_shard: Vec<Vec<(String, f32)>>,
    },
    /// Answer to [`Request::RankApply`]: how far this shard moved.
    RankRound {
        dangling: f32,
        residual: f32,
    },
    /// Answer to [`Request::RankResults`].
    RankResults {
        ranks: Vec<(String, f32)>,
    },
    /// Acknowledgement for a request with nothing to return.
    Ok,
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
const REQ_RANK_INIT: u8 = 5;
const REQ_RANK_DANGLING: u8 = 6;
const REQ_RANK_EMIT: u8 = 7;
const REQ_RANK_ABSORB: u8 = 8;
const REQ_RANK_APPLY: u8 = 9;
const REQ_RANK_RESULTS: u8 = 10;
const REQ_SEGMENT_INFO: u8 = 11;
const REQ_SEGMENT_CHUNK: u8 = 12;
const REQ_ROBOTS: u8 = 13;
const REQ_MANIFEST: u8 = 14;
const REQ_REFRESH: u8 = 15;

const RES_TERM_STATS: u8 = 1;
const RES_HITS: u8 = 2;
const RES_STATS: u8 = 3;
const RES_ERROR: u8 = 4;
const RES_LEASE: u8 = 5;
const RES_RANK_DANGLING: u8 = 6;
const RES_RANK_BOUNDARIES: u8 = 7;
const RES_RANK_ROUND: u8 = 8;
const RES_RANK_RESULTS: u8 = 9;
const RES_OK: u8 = 10;
const RES_SEGMENT_INFO: u8 = 11;
const RES_SEGMENT_CHUNK: u8 = 12;
const RES_ROBOTS: u8 = 13;
const RES_MANIFEST: u8 = 14;
const RES_REFRESHED: u8 = 15;

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
                global_total_length,
                global_doc_freq,
            } => {
                w.u8(REQ_SEARCH);
                w.str(query);
                w.usize(*limit);
                w.usize(*global_doc_count);
                w.varint(*global_total_length);
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
            Self::RankInit {
                total_nodes,
                shard_count,
                damping,
                tolerance,
            } => {
                w.u8(REQ_RANK_INIT);
                w.usize(*total_nodes);
                w.usize(*shard_count);
                w.f32(*damping);
                w.f32(*tolerance);
            }
            Self::RankDangling => w.u8(REQ_RANK_DANGLING),
            Self::RankEmit => w.u8(REQ_RANK_EMIT),
            Self::RankAbsorb { contributions } => {
                w.u8(REQ_RANK_ABSORB);
                w.usize(contributions.len());
                for (uri, share) in contributions {
                    w.str(uri);
                    w.f32(*share);
                }
            }
            Self::RankApply { global_dangling } => {
                w.u8(REQ_RANK_APPLY);
                w.f32(*global_dangling);
            }
            Self::RankResults => w.u8(REQ_RANK_RESULTS),
            Self::Robots {
                host,
                learned,
                state,
                text,
            } => {
                w.u8(REQ_ROBOTS);
                w.str(host);
                w.u8(u8::from(*learned));
                w.u8(*state);
                w.str(text);
            }
            Self::Manifest => w.u8(REQ_MANIFEST),
            Self::Refresh => w.u8(REQ_REFRESH),
            Self::SegmentInfo { name } => {
                w.u8(REQ_SEGMENT_INFO);
                w.str(name);
            }
            Self::SegmentChunk { name, offset, len } => {
                w.u8(REQ_SEGMENT_CHUNK);
                w.str(name);
                w.varint(*offset);
                w.varint(u64::from(*len));
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
                let global_total_length = r.varint()?;
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
                    global_total_length,
                    global_doc_freq,
                }
            }
            REQ_STATS => Self::Stats,
            REQ_LEASE => Self::Lease {
                host: r.string()?,
                requested_delay_ms: r.varint()?,
                permits: r.usize()?,
            },
            REQ_RANK_INIT => Self::RankInit {
                total_nodes: r.usize()?,
                shard_count: r.usize()?,
                damping: r.f32()?,
                tolerance: r.f32()?,
            },
            REQ_RANK_DANGLING => Self::RankDangling,
            REQ_RANK_EMIT => Self::RankEmit,
            REQ_RANK_ABSORB => {
                let count = r.count()?;
                let mut contributions = Vec::with_capacity(count);
                for _ in 0..count {
                    let uri = r.string()?;
                    contributions.push((uri, r.f32()?));
                }
                Self::RankAbsorb { contributions }
            }
            REQ_RANK_APPLY => Self::RankApply {
                global_dangling: r.f32()?,
            },
            REQ_RANK_RESULTS => Self::RankResults,
            REQ_ROBOTS => Self::Robots {
                host: r.string()?,
                learned: r.u8()? != 0,
                state: r.u8()?,
                text: r.string()?,
            },
            REQ_MANIFEST => Self::Manifest,
            REQ_REFRESH => Self::Refresh,
            REQ_SEGMENT_INFO => Self::SegmentInfo { name: r.string()? },
            REQ_SEGMENT_CHUNK => Self::SegmentChunk {
                name: r.string()?,
                offset: r.varint()?,
                len: u32::try_from(r.varint()?)
                    .map_err(|_| indexander_core::Error::Corrupt("chunk too large".into()))?,
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
    // One arm per variant, each a handful of lines, for both of these.
    // Splitting either would put the encoder and its decoder further apart,
    // and that pairing is what has to stay readable together.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            Self::TermStats {
                doc_count,
                total_length,
                doc_freq,
                params,
            } => {
                w.u8(RES_TERM_STATS);
                w.usize(*doc_count);
                w.varint(*total_length);
                w.usize(doc_freq.len());
                for freq in doc_freq {
                    w.usize(*freq);
                }
                for value in params {
                    w.f32(*value);
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
                segments,
            } => {
                w.u8(RES_STATS);
                w.usize(*documents);
                w.usize(*terms);
                w.f32(*average_length);
                w.usize(*segments);
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
            Self::RankDangling { dangling } => {
                w.u8(RES_RANK_DANGLING);
                w.f32(*dangling);
            }
            Self::RankBoundaries { per_shard } => {
                w.u8(RES_RANK_BOUNDARIES);
                w.usize(per_shard.len());
                for bundle in per_shard {
                    w.usize(bundle.len());
                    for (uri, share) in bundle {
                        w.str(uri);
                        w.f32(*share);
                    }
                }
            }
            Self::RankRound { dangling, residual } => {
                w.u8(RES_RANK_ROUND);
                w.f32(*dangling);
                w.f32(*residual);
            }
            Self::RankResults { ranks } => {
                w.u8(RES_RANK_RESULTS);
                w.usize(ranks.len());
                for (uri, score) in ranks {
                    w.str(uri);
                    w.f32(*score);
                }
            }
            Self::Manifest { text } => {
                w.u8(RES_MANIFEST);
                w.str(text);
            }
            Self::Refreshed {
                fetched,
                segments,
                documents,
            } => {
                w.u8(RES_REFRESHED);
                w.usize(*fetched);
                w.usize(*segments);
                w.usize(*documents);
            }
            Self::Robots { state, text } => {
                w.u8(RES_ROBOTS);
                w.u8(*state);
                w.str(text);
            }
            Self::SegmentInfo { digest, len } => {
                w.u8(RES_SEGMENT_INFO);
                w.varint(*digest);
                w.varint(*len);
            }
            Self::SegmentChunk { bytes } => {
                w.u8(RES_SEGMENT_CHUNK);
                w.usize(bytes.len());
                w.bytes(bytes);
            }
            Self::Ok => w.u8(RES_OK),
            Self::Error { message } => {
                w.u8(RES_ERROR);
                w.str(message);
            }
        }
        w.finish()
    }

    #[allow(clippy::too_many_lines)]
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let message = match r.u8()? {
            RES_TERM_STATS => {
                let doc_count = r.usize()?;
                let total_length = r.varint()?;
                let count = r.count()?;
                let mut doc_freq = Vec::with_capacity(count);
                for _ in 0..count {
                    doc_freq.push(r.usize()?);
                }
                let params = [r.f32()?, r.f32()?, r.f32()?];
                Self::TermStats {
                    doc_count,
                    total_length,
                    doc_freq,
                    params,
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
                segments: r.usize()?,
            },
            RES_LEASE => Self::Lease {
                wait_ms: r.varint()?,
                permits: r.usize()?,
                spacing_ms: r.varint()?,
            },
            RES_RANK_DANGLING => Self::RankDangling { dangling: r.f32()? },
            RES_RANK_BOUNDARIES => {
                let shards = r.count()?;
                let mut per_shard = Vec::with_capacity(shards);
                for _ in 0..shards {
                    let count = r.count()?;
                    let mut bundle = Vec::with_capacity(count);
                    for _ in 0..count {
                        let uri = r.string()?;
                        bundle.push((uri, r.f32()?));
                    }
                    per_shard.push(bundle);
                }
                Self::RankBoundaries { per_shard }
            }
            RES_RANK_ROUND => Self::RankRound {
                dangling: r.f32()?,
                residual: r.f32()?,
            },
            RES_RANK_RESULTS => {
                let count = r.count()?;
                let mut ranks = Vec::with_capacity(count);
                for _ in 0..count {
                    let uri = r.string()?;
                    ranks.push((uri, r.f32()?));
                }
                Self::RankResults { ranks }
            }
            RES_MANIFEST => Self::Manifest { text: r.string()? },
            RES_REFRESHED => Self::Refreshed {
                fetched: r.usize()?,
                segments: r.usize()?,
                documents: r.usize()?,
            },
            RES_ROBOTS => Self::Robots {
                state: r.u8()?,
                text: r.string()?,
            },
            RES_SEGMENT_INFO => Self::SegmentInfo {
                digest: r.varint()?,
                len: r.varint()?,
            },
            RES_SEGMENT_CHUNK => {
                let len = r.count()?;
                Self::SegmentChunk {
                    bytes: r.bytes(len)?,
                }
            }
            RES_OK => Self::Ok,
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
        roundtrip_request(&Request::RankInit {
            total_nodes: 1_000_000,
            shard_count: 17,
            damping: 0.85,
            tolerance: 1e-6,
        });
        roundtrip_request(&Request::RankApply {
            global_dangling: 0.125,
        });
        roundtrip_request(&Request::Robots {
            host: "example.com".into(),
            learned: false,
            state: 0,
            text: String::new(),
        });
        roundtrip_request(&Request::Robots {
            host: "example.com".into(),
            learned: true,
            state: 1,
            text: "User-agent: *\nDisallow: /ñ".into(),
        });
        roundtrip_request(&Request::Manifest);
        roundtrip_request(&Request::SegmentInfo {
            name: String::new(),
        });
        roundtrip_request(&Request::SegmentInfo {
            name: "44906b3a549fc495.ixdr".into(),
        });
        roundtrip_request(&Request::SegmentChunk {
            name: "flush000.ixdr".into(),
            offset: 1 << 40,
            len: 65536,
        });
        roundtrip_request(&Request::RankDangling);
        roundtrip_request(&Request::RankEmit);
        roundtrip_request(&Request::RankResults);
        roundtrip_request(&Request::RankAbsorb {
            contributions: vec![("http://a/ñ".into(), 0.25), ("http://b/".into(), 0.5)],
        });
        roundtrip_request(&Request::RankAbsorb {
            contributions: Vec::new(),
        });
        roundtrip_request(&Request::Refresh);
        roundtrip_request(&Request::TermStats { terms: Vec::new() });
        roundtrip_request(&Request::TermStats {
            terms: vec!["motor".into(), "búsqueda".into()],
        });
        roundtrip_request(&Request::Search {
            query: r#""motor de busqueda" -perl"#.into(),
            limit: 10,
            global_doc_count: 1_000_000,
            global_total_length: 880_000_000,
            global_doc_freq: vec![("motor".into(), 4321), ("perl".into(), 7)],
        });
    }

    #[test]
    fn every_response_roundtrips() {
        for response in [
            Response::Refreshed {
                fetched: 2,
                segments: 3,
                documents: 4000,
            },
            Response::TermStats {
                doc_count: 500,
                total_length: 4_400_000,
                doc_freq: vec![1, 2, 3],
                params: [1.2, 0.75, 0.5],
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
                segments: 3,
            },
            Response::Lease {
                wait_ms: 1500,
                permits: 20,
                spacing_ms: 500,
            },
            Response::Ok,
            Response::Manifest {
                text: "indexander-manifest 1\nsegments 0\n".into(),
            },
            Response::Robots {
                state: 1,
                text: "User-agent: *".into(),
            },
            Response::Robots {
                state: 2,
                text: String::new(),
            },
            Response::SegmentInfo {
                digest: 0xdead_beef_cafe_f00d,
                len: 260_000_000,
            },
            Response::SegmentChunk {
                bytes: vec![0u8, 255, 7, 0, 0, 42],
            },
            Response::SegmentChunk { bytes: Vec::new() },
            Response::RankDangling { dangling: 0.0625 },
            Response::RankRound {
                dangling: 0.5,
                residual: 0.25,
            },
            Response::RankResults {
                ranks: vec![("http://a/".into(), 0.75)],
            },
            Response::RankBoundaries {
                per_shard: vec![
                    vec![("http://x/".into(), 0.5)],
                    Vec::new(),
                    vec![("http://y/ñ".into(), 0.25), ("http://z/".into(), 0.125)],
                ],
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
            global_total_length: 880,
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
