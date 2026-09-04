//! The shard role: owns one segment, answers questions about it.
//!
//! A shard knows nothing about the cluster. It does not know how many shards
//! there are, which URLs belong to it, or who is asking. It answers about the
//! documents it holds, and scores with whatever statistics it is given. That
//! ignorance is deliberate: it is what makes one shard and a hundred shards
//! the same program.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use indexander_core::Result;
use indexander_index::index::Index;
use indexander_index::manifest::Manifest;
use indexander_index::query;
use indexander_index::search::{GlobalStats, search_with_stats};
use indexander_index::segment::Segment;
use indexander_proto::message::{Hit, PROTOCOL_VERSION, Request, Response};
use tokio::net::{TcpListener, TcpStream};

use crate::frame::{read_frame, read_hello, write_frame, write_hello};

/// What a shard serves: an index, and — when it came from a directory — the
/// manifest and files behind it.
///
/// A shard used to hold a single segment, which was true right up until an
/// index became a list of them. Holding the manifest as well is what lets a
/// replica ask "what should I have?" and fetch the pieces it is missing by
/// name.
#[derive(Debug)]
pub struct ShardIndex {
    index: Index,
    /// Where the segments live, for serving their bytes. `None` for an index
    /// built in memory, which can still answer queries but cannot be copied.
    directory: Option<PathBuf>,
    manifest: Manifest,
}

impl ShardIndex {
    /// One segment, held in memory. What a test or a single-file shard uses.
    #[must_use]
    pub fn single(segment: Segment) -> Self {
        let mut index = Index::new();
        index.push(segment);
        Self {
            index,
            directory: None,
            manifest: Manifest::new(),
        }
    }

    /// Every segment a directory's manifest names.
    pub fn open(directory: &Path) -> Result<Self> {
        let manifest = Manifest::open(&directory.join("MANIFEST"))?;
        let index = Index::open_manifest(directory, &manifest)?;
        Ok(Self {
            index,
            directory: Some(directory.to_path_buf()),
            manifest,
        })
    }

    #[must_use]
    pub fn index(&self) -> &Index {
        &self.index
    }

    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The bytes of a named segment, for a replica copying it.
    ///
    /// An empty name means the only segment this shard holds, which is what a
    /// single-segment shard answers about itself.
    fn segment_bytes(&self, name: &str) -> Option<Vec<u8>> {
        if name.is_empty() {
            return self.index.segments().first().map(|s| s.as_bytes().to_vec());
        }
        // Only names the manifest lists. A shard is not a file server, and
        // reading whatever path a caller asks for is how one becomes an
        // accidental one.
        if !self.manifest.segments.iter().any(|e| e.name == name) {
            return None;
        }
        let directory = self.directory.as_ref()?;
        std::fs::read(directory.join(name)).ok()
    }

    fn digest_of(&self, name: &str) -> Option<u64> {
        if name.is_empty() {
            return self.index.segments().first().map(Segment::digest);
        }
        self.manifest
            .segments
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.digest)
    }
}

/// Answers one request against `shard`.
///
/// Pure: no i/o, no clock, no state. Every protocol behaviour can therefore be
/// tested without a socket, and the socket code has nothing left to get wrong
/// but framing.
#[must_use]
pub fn handle(shard: &ShardIndex, request: &Request) -> Response {
    match request {
        // Serving a copy of the index is a different job from answering
        // queries about it. It shares this socket because a replica pulls from
        // whichever node already holds the data.
        Request::Manifest | Request::SegmentInfo { .. } | Request::SegmentChunk { .. } => {
            serve_bytes(shard, request)
        }
        Request::TermStats { terms } => {
            let doc_freq = terms
                .iter()
                .map(|t| {
                    shard
                        .index
                        .segments()
                        .iter()
                        .map(|s| s.document_frequency(t).unwrap_or(0))
                        .sum()
                })
                .collect();
            Response::TermStats {
                doc_count: shard.index.document_count(),
                total_length: shard
                    .index
                    .segments()
                    .iter()
                    .map(Segment::total_length)
                    .sum(),
                doc_freq,
            }
        }
        Request::Search {
            query,
            limit,
            global_doc_count,
            global_total_length,
            global_doc_freq,
        } => {
            let stats = GlobalStats {
                total_docs: *global_doc_count,
                total_length: *global_total_length,
                doc_freq: global_doc_freq.iter().cloned().collect(),
            };
            let parsed = query::parse(query);
            // Across every segment this shard holds, scored with whatever
            // statistics the coordinator supplied.
            let mut hits = Vec::new();
            let mut failure = None;
            for segment in shard.index.segments() {
                match search_with_stats(segment, &parsed, *limit, Some(&stats)) {
                    Ok(found) => hits.extend(found),
                    Err(e) => failure = Some(e.to_string()),
                }
            }
            if let Some(message) = failure {
                return Response::Error { message };
            }
            hits.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.uri.cmp(&b.uri))
            });
            hits.truncate(*limit);
            Response::Hits {
                hits: hits
                    .into_iter()
                    .map(|h| Hit {
                        uri: h.uri,
                        score: h.score,
                    })
                    .collect(),
            }
        }
        Request::Stats => Response::Stats {
            documents: shard.index.document_count(),
            terms: shard.index.segments().iter().map(Segment::term_count).sum(),
            average_length: shard
                .index
                .segments()
                .first()
                .map_or(0.0, Segment::average_document_length),
        },
        // A shard serves an index. Rate limits belong to a lease authority and
        // ranking to a rank shard: different roles, different addresses.
        // Answering anyway would be worse than refusing — a crawler would
        // believe it had permission nobody coordinated, and a ranking run
        // would believe it had a partition of the graph that is not here.
        Request::Lease { .. } | Request::Robots { .. } => Response::Error {
            message: "a shard does not grant fetch leases or hold robots.txt".to_owned(),
        },
        Request::RankInit { .. }
        | Request::RankDangling
        | Request::RankEmit
        | Request::RankAbsorb { .. }
        | Request::RankApply { .. }
        | Request::RankResults => Response::Error {
            message: "a shard does not hold a partition of the link graph".to_owned(),
        },
    }
}

/// Answers the requests a replica makes when copying this index.
fn serve_bytes(shard: &ShardIndex, request: &Request) -> Response {
    match request {
        Request::Manifest => Response::Manifest {
            text: shard.manifest.encode(),
        },
        Request::SegmentInfo { name } => match (shard.digest_of(name), shard.segment_bytes(name)) {
            (Some(digest), Some(bytes)) => Response::SegmentInfo {
                digest,
                len: bytes.len() as u64,
            },
            _ => Response::Error {
                message: format!("no segment called {name:?}"),
            },
        },
        Request::SegmentChunk { name, offset, len } => {
            let Some(bytes) = shard.segment_bytes(name) else {
                return Response::Error {
                    message: format!("no segment called {name:?}"),
                };
            };
            let start = usize::try_from(*offset).unwrap_or(usize::MAX);
            if start >= bytes.len() {
                // Past the end is an empty chunk, not an error: it is how a
                // reader learns it has everything.
                return Response::SegmentChunk { bytes: Vec::new() };
            }
            let end = start.saturating_add(*len as usize).min(bytes.len());
            Response::SegmentChunk {
                bytes: bytes[start..end].to_vec(),
            }
        }
        other => Response::Error {
            message: format!("not a transfer request: {other:?}"),
        },
    }
}

/// Serves `shard` on `listener` until the task is dropped.
pub async fn serve(listener: TcpListener, shard: Arc<ShardIndex>) -> Result<()> {
    loop {
        let (stream, _) = listener.accept().await?;
        let shard = Arc::clone(&shard);
        // One task per connection. A slow or hostile client cannot block the
        // others, and a panic in one connection cannot take down the shard.
        tokio::spawn(async move {
            let _ = connection(stream, &shard).await;
        });
    }
}

async fn connection(mut stream: TcpStream, shard: &ShardIndex) -> Result<()> {
    write_hello(&mut stream, PROTOCOL_VERSION).await?;
    read_hello(&mut stream, PROTOCOL_VERSION).await?;

    // Connections are long-lived: a coordinator sends both rounds of a query,
    // and every subsequent query, down the same socket.
    loop {
        // A read error here means the coordinator hung up, which is the
        // normal way a connection ends, not a failure to report.
        let Ok(payload) = read_frame(&mut stream).await else {
            return Ok(());
        };
        let response = match Request::decode(&payload) {
            Ok(request) => handle(shard, &request),
            Err(e) => Response::Error {
                message: format!("undecodable request: {e}"),
            },
        };
        write_frame(&mut stream, &response.encode()).await?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexander_core::Document;
    use indexander_index::builder::SegmentBuilder;

    fn segment() -> ShardIndex {
        let mut b = SegmentBuilder::new();
        b.add(&Document::new(
            "doc://uno",
            "motor",
            "un motor de busqueda en rust",
        ));
        b.add(&Document::new(
            "doc://dos",
            "perl",
            "un motor escrito en perl",
        ));
        ShardIndex::single(Segment::from_bytes(b.encode()).expect("segment"))
    }

    #[test]
    fn term_stats_answers_in_the_order_asked() {
        let s = segment();
        let response = handle(
            &s,
            &Request::TermStats {
                terms: vec!["motor".into(), "perl".into(), "kubernetes".into()],
            },
        );
        assert_eq!(
            response,
            Response::TermStats {
                doc_count: 2,
                total_length: s.index().segments()[0].total_length(),
                doc_freq: vec![2, 1, 0],
            }
        );
    }

    #[test]
    fn search_returns_hits() {
        let s = segment();
        let response = handle(
            &s,
            &Request::Search {
                query: "perl".into(),
                limit: 10,
                global_doc_count: 0,
                global_total_length: 0,
                global_doc_freq: Vec::new(),
            },
        );
        let Response::Hits { hits } = response else {
            panic!("expected hits, got {response:?}");
        };
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].uri, "doc://dos");
    }

    #[test]
    fn a_shard_scores_with_the_statistics_it_is_given() {
        let s = segment();
        let local = handle(
            &s,
            &Request::Search {
                query: "motor".into(),
                limit: 10,
                global_doc_count: 0,
                global_total_length: 0,
                global_doc_freq: Vec::new(),
            },
        );
        // Pretend the corpus is a million documents in which "motor" is rare.
        let global = handle(
            &s,
            &Request::Search {
                query: "motor".into(),
                limit: 10,
                global_doc_count: 1_000_000,
                global_total_length: 8_000_000,
                global_doc_freq: vec![("motor".into(), 3)],
            },
        );
        let (Response::Hits { hits: a }, Response::Hits { hits: b }) = (local, global) else {
            panic!("expected hits from both");
        };
        assert!(
            b[0].score > a[0].score * 2.0,
            "global rarity did not raise the score: {} vs {}",
            a[0].score,
            b[0].score
        );
    }

    #[test]
    fn stats_reports_the_segment() {
        let s = segment();
        let Response::Stats {
            documents, terms, ..
        } = handle(&s, &Request::Stats)
        else {
            panic!("expected stats");
        };
        assert_eq!(documents, 2);
        assert!(terms > 5);
    }
}
