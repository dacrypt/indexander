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
use std::sync::Arc;

use indexander_core::{Error, Result};
use indexander_index::query;
use indexander_proto::message::{Hit, PROTOCOL_VERSION, Request, Response};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

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
///
/// Each connection sits behind its own lock, so the coordinator can hold all
/// of them at once and drive every shard concurrently. There is no contention:
/// exactly one task ever touches a given connection at a time.
#[derive(Debug)]
pub struct Coordinator {
    shards: Vec<Arc<Mutex<ShardConnection>>>,
}

impl Coordinator {
    /// Connects to one address per shard.
    ///
    /// Failing rather than degrading is the right default for a search engine:
    /// results computed from four shards out of five are not "slightly worse
    /// results", they are results that silently omit a fifth of the corpus.
    ///
    /// See [`Coordinator::connect_replicated`] when a shard has more than one
    /// copy — then a dead address is not a dead shard.
    pub async fn connect(addresses: &[String]) -> Result<Self> {
        // Connecting is itself fanned out: with fifty shards, doing it one at
        // a time makes startup fifty round trips deep.
        let attempts: Vec<_> = addresses
            .iter()
            .map(|address| {
                let address = address.clone();
                tokio::spawn(async move {
                    ShardConnection::connect(&address)
                        .await
                        .map_err(|e| Error::Corrupt(format!("shard {address}: {e}")))
                })
            })
            .collect();

        let mut shards = Vec::with_capacity(addresses.len());
        for attempt in attempts {
            let connection = attempt
                .await
                .map_err(|e| Error::Corrupt(format!("connect task failed: {e}")))??;
            shards.push(Arc::new(Mutex::new(connection)));
        }
        if shards.is_empty() {
            return Err(Error::Corrupt(
                "a coordinator needs at least one shard".into(),
            ));
        }
        Ok(Self { shards })
    }

    /// Connects to one live replica of each shard.
    ///
    /// Each entry is the replicas of one shard, tried in order. This is the
    /// distinction replication buys, and it is worth being precise about:
    /// a missing *shard* still fails the query, because a fifth of the corpus
    /// would be silently absent; a missing *replica* does not, because another
    /// copy holds the same segment and the answer is the same one.
    ///
    /// A shard whose replicas are all unreachable fails the whole connection,
    /// with every address it tried, rather than quietly returning results from
    /// the rest.
    pub async fn connect_replicated(replicas: &[Vec<String>]) -> Result<Self> {
        if replicas.is_empty() {
            return Err(Error::Corrupt(
                "a coordinator needs at least one shard".into(),
            ));
        }

        // Shards are tried concurrently; the replicas within a shard are tried
        // in order, because the first live one is the answer and racing them
        // would open connections nobody uses.
        let attempts: Vec<_> = replicas
            .iter()
            .map(|group| {
                let group = group.clone();
                tokio::spawn(async move {
                    let mut failures = Vec::new();
                    for address in &group {
                        match ShardConnection::connect(address).await {
                            Ok(connection) => return Ok(connection),
                            Err(e) => failures.push(format!("{address}: {e}")),
                        }
                    }
                    Err(Error::Corrupt(format!(
                        "every replica of a shard is unreachable ({})",
                        failures.join("; ")
                    )))
                })
            })
            .collect();

        let mut shards = Vec::with_capacity(replicas.len());
        for attempt in attempts {
            let connection = attempt
                .await
                .map_err(|e| Error::Corrupt(format!("connect task failed: {e}")))??;
            shards.push(Arc::new(Mutex::new(connection)));
        }
        Ok(Self { shards })
    }

    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Round one: the corpus-wide statistics for these terms.
    pub async fn term_statistics(&self, terms: &[String]) -> Result<GlobalTermStats> {
        let request = Request::TermStats {
            terms: terms.to_vec(),
        };
        let mut total_docs = 0usize;
        let mut total_tokens = 0u64;
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for (address, response) in self.broadcast(&request).await? {
            match response {
                Response::TermStats {
                    doc_count,
                    total_length,
                    doc_freq: per_term,
                } => {
                    if per_term.len() != terms.len() {
                        return Err(Error::Corrupt(format!(
                            "shard {address} answered {} frequencies for {} terms",
                            per_term.len(),
                            terms.len()
                        )));
                    }
                    total_docs += doc_count;
                    total_tokens += total_length;
                    for (term, freq) in terms.iter().zip(per_term) {
                        *doc_freq.entry(term.clone()).or_insert(0) += freq;
                    }
                }
                Response::Error { message } => {
                    return Err(Error::Corrupt(format!("shard {address}: {message}")));
                }
                other => {
                    return Err(Error::Corrupt(format!(
                        "shard {address} answered {other:?} to a term-stats request"
                    )));
                }
            }
        }
        Ok(GlobalTermStats {
            total_docs,
            total_length: total_tokens,
            doc_freq,
        })
    }

    /// Runs a query across every shard and returns the global top `limit`.
    pub async fn search(&self, query_text: &str, limit: usize) -> Result<Vec<Hit>> {
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
            global_total_length: stats.total_length,
            global_doc_freq: stats
                .doc_freq
                .iter()
                .map(|(t, f)| (t.clone(), *f))
                .collect(),
        };

        let mut all: Vec<Hit> = Vec::new();
        for (address, response) in self.broadcast(&request).await? {
            match response {
                // Each shard returns its own top `limit`; the global top
                // `limit` is necessarily a subset of the union, so nothing is
                // lost by asking for no more than that from each.
                Response::Hits { hits } => all.extend(hits),
                Response::Error { message } => {
                    return Err(Error::Corrupt(format!("shard {address}: {message}")));
                }
                other => {
                    return Err(Error::Corrupt(format!(
                        "shard {address} answered {other:?} to a search"
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

    /// Sends one request to every shard at once.
    ///
    /// Concurrent, so a round costs the slowest shard rather than the sum of
    /// all of them — the difference between sharding helping and hurting.
    /// Results come back in shard order regardless of who answered first, so
    /// that a query is reproducible.
    async fn broadcast(&self, request: &Request) -> Result<Vec<(String, Response)>> {
        let calls: Vec<_> = self
            .shards
            .iter()
            .map(|shard| {
                let shard = Arc::clone(shard);
                let request = request.clone();
                tokio::spawn(async move {
                    let mut guard = shard.lock().await;
                    let address = guard.address().to_owned();
                    guard
                        .call(&request)
                        .await
                        .map(|response| (address, response))
                })
            })
            .collect();

        let mut out = Vec::with_capacity(calls.len());
        for call in calls {
            out.push(
                call.await
                    .map_err(|e| Error::Corrupt(format!("shard task failed: {e}")))??,
            );
        }
        Ok(out)
    }

    /// Totals across the cluster.
    pub async fn stats(&self) -> Result<(usize, usize)> {
        let mut documents = 0;
        let mut terms = 0;
        for (_, response) in self.broadcast(&Request::Stats).await? {
            if let Response::Stats {
                documents: d,
                terms: t,
                ..
            } = response
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
    /// Tokens across every document of every shard, so that every shard can
    /// score against the corpus's average length rather than its own.
    pub total_length: u64,
    pub doc_freq: HashMap<String, usize>,
}
