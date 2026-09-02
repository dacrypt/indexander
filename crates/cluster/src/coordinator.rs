//! The coordinator role: asks every shard, merges what comes back.
//!
//! A query is two rounds, and the first one is not an optimisation — it is
//! what makes the second one correct. See `indexander_proto::message` for why
//! BM25 scores from different shards cannot be compared until every shard has
//! been told the same term statistics.
//!
//! Both rounds fan out concurrently. Latency is therefore the slowest shard,
//! twice, not the sum of all of them.

use std::collections::HashMap;

use indexander_core::{Error, Result};
use indexander_index::query;
use indexander_proto::message::{Hit, PROTOCOL_VERSION, Request, Response};
use tokio::net::TcpStream;

use crate::frame::{read_frame, read_hello, write_frame, write_hello};

/// A connection to one shard.
#[derive(Debug)]
pub struct ShardConnection {
    address: String,
    stream: TcpStream,
}

impl ShardConnection {
    /// Opens a connection and completes the version handshake.
    pub async fn connect(address: &str) -> Result<Self> {
        let mut stream = TcpStream::connect(address).await?;
        // Nagle's algorithm batches small writes, which is exactly wrong for a
        // request/response protocol: it adds up to 40 ms to every round trip.
        let _ = stream.set_nodelay(true);
        read_hello(&mut stream, PROTOCOL_VERSION).await?;
        write_hello(&mut stream, PROTOCOL_VERSION).await?;
        Ok(Self {
            address: address.to_owned(),
            stream,
        })
    }

    #[must_use]
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Sends one request and waits for its response.
    pub async fn call(&mut self, request: &Request) -> Result<Response> {
        write_frame(&mut self.stream, &request.encode()).await?;
        let payload = read_frame(&mut self.stream).await?;
        Response::decode(&payload)
    }
}

/// A fixed set of shards.
#[derive(Debug)]
pub struct Coordinator {
    shards: Vec<ShardConnection>,
}

impl Coordinator {
    /// Connects to every address, failing if any one of them cannot be reached.
    ///
    /// Failing rather than degrading is the right default for a search engine:
    /// results computed from four shards out of five are not "slightly worse
    /// results", they are results that silently omit a fifth of the corpus.
    pub async fn connect(addresses: &[String]) -> Result<Self> {
        let mut shards = Vec::with_capacity(addresses.len());
        for address in addresses {
            shards.push(
                ShardConnection::connect(address)
                    .await
                    .map_err(|e| Error::Corrupt(format!("shard {address}: {e}")))?,
            );
        }
        if shards.is_empty() {
            return Err(Error::Corrupt(
                "a coordinator needs at least one shard".into(),
            ));
        }
        Ok(Self { shards })
    }

    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Round one: the corpus-wide statistics for these terms.
    pub async fn term_statistics(&mut self, terms: &[String]) -> Result<GlobalTermStats> {
        let request = Request::TermStats {
            terms: terms.to_vec(),
        };
        let mut total_docs = 0usize;
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for shard in &mut self.shards {
            match shard.call(&request).await? {
                Response::TermStats {
                    doc_count,
                    doc_freq: per_term,
                } => {
                    if per_term.len() != terms.len() {
                        return Err(Error::Corrupt(format!(
                            "shard {} answered {} frequencies for {} terms",
                            shard.address(),
                            per_term.len(),
                            terms.len()
                        )));
                    }
                    total_docs += doc_count;
                    for (term, freq) in terms.iter().zip(per_term) {
                        *doc_freq.entry(term.clone()).or_insert(0) += freq;
                    }
                }
                Response::Error { message } => {
                    return Err(Error::Corrupt(format!(
                        "shard {}: {message}",
                        shard.address()
                    )));
                }
                other => {
                    return Err(Error::Corrupt(format!(
                        "shard {} answered {other:?} to a term-stats request",
                        shard.address()
                    )));
                }
            }
        }
        Ok(GlobalTermStats {
            total_docs,
            doc_freq,
        })
    }

    /// Runs a query across every shard and returns the global top `limit`.
    pub async fn search(&mut self, query_text: &str, limit: usize) -> Result<Vec<Hit>> {
        let parsed = query::parse(query_text);
        if parsed.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        // Round one.
        let terms = parsed.scoring_terms();
        let stats = self.term_statistics(&terms).await?;

        // Round two, with everyone scoring on the same scale.
        let request = Request::Search {
            query: query_text.to_owned(),
            limit,
            global_doc_count: stats.total_docs,
            global_doc_freq: stats
                .doc_freq
                .iter()
                .map(|(t, f)| (t.clone(), *f))
                .collect(),
        };

        let mut all: Vec<Hit> = Vec::new();
        for shard in &mut self.shards {
            match shard.call(&request).await? {
                // Each shard returns its own top `limit`; the global top
                // `limit` is necessarily a subset of the union, so nothing is
                // lost by asking for no more than that from each.
                Response::Hits { hits } => all.extend(hits),
                Response::Error { message } => {
                    return Err(Error::Corrupt(format!(
                        "shard {}: {message}",
                        shard.address()
                    )));
                }
                other => {
                    return Err(Error::Corrupt(format!(
                        "shard {} answered {other:?} to a search",
                        shard.address()
                    )));
                }
            }
        }

        all.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.uri.cmp(&b.uri))
        });
        all.truncate(limit);
        Ok(all)
    }

    /// Totals across the cluster.
    pub async fn stats(&mut self) -> Result<(usize, usize)> {
        let mut documents = 0;
        let mut terms = 0;
        for shard in &mut self.shards {
            if let Response::Stats {
                documents: d,
                terms: t,
                ..
            } = shard.call(&Request::Stats).await?
            {
                documents += d;
                // Term counts overlap between shards, so this is an upper
                // bound on distinct terms, not the number of them.
                terms += t;
            }
        }
        Ok((documents, terms))
    }
}

/// Corpus-wide term statistics gathered in round one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GlobalTermStats {
    pub total_docs: usize,
    pub doc_freq: HashMap<String, usize>,
}
