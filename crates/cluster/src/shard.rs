//! The shard role: owns one segment, answers questions about it.
//!
//! A shard knows nothing about the cluster. It does not know how many shards
//! there are, which URLs belong to it, or who is asking. It answers about the
//! documents it holds, and scores with whatever statistics it is given. That
//! ignorance is deliberate: it is what makes one shard and a hundred shards
//! the same program.

use std::sync::Arc;

use indexander_core::Result;
use indexander_index::query;
use indexander_index::search::{GlobalStats, search_with_stats};
use indexander_index::segment::Segment;
use indexander_proto::message::{Hit, PROTOCOL_VERSION, Request, Response};
use tokio::net::{TcpListener, TcpStream};

use crate::frame::{read_frame, read_hello, write_frame, write_hello};

/// Answers one request against `segment`.
///
/// Pure: no i/o, no clock, no state. Every protocol behaviour can therefore be
/// tested without a socket, and the socket code has nothing left to get wrong
/// but framing.
#[must_use]
pub fn handle(segment: &Segment, request: &Request) -> Response {
    // Serving a copy of the segment is a different job from answering
    // queries about it, and lives in `replication`. It shares this socket
    // because a replica pulls from whichever node already has the data.
    if let Some(response) = crate::replication::handle(segment.as_bytes(), segment, request) {
        return response;
    }
    match request {
        Request::TermStats { terms } => {
            let doc_freq = terms
                .iter()
                .map(|t| segment.document_frequency(t).unwrap_or(0))
                .collect();
            Response::TermStats {
                doc_count: segment.document_count(),
                total_length: segment.total_length(),
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
            match search_with_stats(segment, &parsed, *limit, Some(&stats)) {
                Ok(hits) => Response::Hits {
                    hits: hits
                        .into_iter()
                        .map(|h| Hit {
                            uri: h.uri,
                            score: h.score,
                        })
                        .collect(),
                },
                Err(e) => Response::Error {
                    message: e.to_string(),
                },
            }
        }
        Request::Stats => Response::Stats {
            documents: segment.document_count(),
            terms: segment.term_count(),
            average_length: segment.average_document_length(),
        },
        // A shard serves an index. Rate limits belong to a lease authority and
        // ranking to a rank shard: different roles, different addresses.
        // Answering anyway would be worse than refusing — a crawler would
        // believe it had permission nobody coordinated, and a ranking run
        // would believe it had a partition of the graph that is not here.
        Request::Lease { .. } => Response::Error {
            message: "a shard does not grant fetch leases".to_owned(),
        },
        // Handled above.
        Request::SegmentInfo | Request::SegmentChunk { .. } => Response::Error {
            message: "unreachable: handled by replication".to_owned(),
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

/// Serves `segment` on `listener` until the task is dropped.
pub async fn serve(listener: TcpListener, segment: Arc<Segment>) -> Result<()> {
    loop {
        let (stream, _) = listener.accept().await?;
        let segment = Arc::clone(&segment);
        // One task per connection. A slow or hostile client cannot block the
        // others, and a panic in one connection cannot take down the shard.
        tokio::spawn(async move {
            let _ = connection(stream, &segment).await;
        });
    }
}

async fn connection(mut stream: TcpStream, segment: &Segment) -> Result<()> {
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
            Ok(request) => handle(segment, &request),
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

    fn segment() -> Segment {
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
        Segment::from_bytes(b.encode()).expect("segment")
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
                total_length: s.total_length(),
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
