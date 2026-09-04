//! Driving a PageRank run across several processes.
//!
//! The algorithm and the reasons behind its three exchanges are in
//! `indexander_rank::distributed`. This is only the wiring: the coordinator
//! asks every shard the same three questions per iteration, in the order that
//! makes the answers mean anything.
//!
//! The order is not an implementation detail. Dangling mass has to be summed
//! across every shard *before* anyone applies the iteration; every shard has
//! to emit from last iteration's ranks *before* anyone absorbs this
//! iteration's contributions; and the residual has to be summed across every
//! shard before anyone decides to stop. Get any of those out of order and the
//! run still finishes, still produces numbers, and the numbers are wrong.

use std::collections::HashMap;
use std::sync::Arc;

use indexander_core::{Error, Result};
use indexander_proto::message::{PROTOCOL_VERSION, Request, Response};
use indexander_rank::distributed::{Boundary, ShardGraph, ShardRanker};
use indexander_rank::pagerank::Options;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

/// A process holding one shard of the link graph.
#[derive(Debug)]
pub struct RankShard {
    /// The graph is loaded at startup; the ranker is created by `RankInit`,
    /// because only the coordinator knows the cluster-wide node count.
    graph: Mutex<Option<ShardGraph>>,
    ranker: Mutex<Option<ShardRanker>>,
    shard: usize,
}

impl RankShard {
    #[must_use]
    pub fn new(graph: ShardGraph, shard: usize) -> Self {
        Self {
            graph: Mutex::new(Some(graph)),
            ranker: Mutex::new(None),
            shard,
        }
    }

    /// Which shard this is, for routing boundary contributions.
    #[must_use]
    pub fn shard(&self) -> usize {
        self.shard
    }

    /// Answers one request.
    pub async fn handle(&self, request: &Request) -> Response {
        match request {
            Request::RankInit {
                total_nodes,
                shard_count,
                damping,
                tolerance,
            } => {
                let Some(graph) = self.graph.lock().await.take() else {
                    return Response::Error {
                        message: "this shard has already been initialised".to_owned(),
                    };
                };
                let options = Options {
                    damping: *damping,
                    tolerance: *tolerance,
                    ..Options::default()
                };
                *self.ranker.lock().await = Some(ShardRanker::new(graph, *total_nodes, options));
                let _ = shard_count;
                Response::Ok
            }
            Request::RankDangling => {
                self.with_ranker(|r| Response::RankDangling {
                    dangling: r.dangling(),
                })
                .await
            }
            Request::RankEmit => {
                // Routing lives on the coordinator: a shard sending a
                // contribution does not need to know who owns the target, and
                // a shard that guessed differently from its peers would drop
                // rank on the floor. It labels everything "not mine" and the
                // coordinator delivers.
                self.with_ranker_mut(|r| {
                    let bundles = r.emit(|_| 0, 1);
                    Response::RankBoundaries {
                        per_shard: bundles.into_iter().map(|b| b.contributions).collect(),
                    }
                })
                .await
            }
            Request::RankAbsorb { contributions } => {
                self.with_ranker_mut(|r| {
                    r.absorb(&Boundary {
                        contributions: contributions.clone(),
                    });
                    Response::Ok
                })
                .await
            }
            Request::RankApply { global_dangling } => {
                self.with_ranker_mut(|r| {
                    let round = r.apply(*global_dangling);
                    Response::RankRound {
                        dangling: round.dangling,
                        residual: round.residual,
                    }
                })
                .await
            }
            Request::RankResults => {
                self.with_ranker(|r| Response::RankResults {
                    ranks: r
                        .ranks()
                        .into_iter()
                        .map(|(uri, score)| (uri.to_owned(), score))
                        .collect(),
                })
                .await
            }
            other => Response::Error {
                message: format!("a rank shard cannot answer {other:?}"),
            },
        }
    }

    async fn with_ranker<F: FnOnce(&ShardRanker) -> Response>(&self, f: F) -> Response {
        match self.ranker.lock().await.as_ref() {
            Some(ranker) => f(ranker),
            None => Response::Error {
                message: "this shard has not been initialised".to_owned(),
            },
        }
    }

    async fn with_ranker_mut<F: FnOnce(&mut ShardRanker) -> Response>(&self, f: F) -> Response {
        match self.ranker.lock().await.as_mut() {
            Some(ranker) => f(ranker),
            None => Response::Error {
                message: "this shard has not been initialised".to_owned(),
            },
        }
    }

    /// Serves on `listener` until the task is dropped.
    pub async fn serve(self: Arc<Self>, listener: TcpListener) -> Result<()> {
        loop {
            let (stream, _) = listener.accept().await?;
            let shard = Arc::clone(&self);
            tokio::spawn(async move {
                let _ = shard.connection(stream).await;
            });
        }
    }

    async fn connection(&self, mut stream: TcpStream) -> Result<()> {
        write_hello(&mut stream, PROTOCOL_VERSION).await?;
        read_hello(&mut stream, PROTOCOL_VERSION).await?;
        loop {
            let Ok(payload) = read_frame(&mut stream).await else {
                return Ok(());
            };
            let response = match Request::decode(&payload) {
                Ok(request) => self.handle(&request).await,
                Err(e) => Response::Error {
                    message: format!("undecodable request: {e}"),
                },
            };
            write_frame(&mut stream, &response.encode()).await?;
        }
    }
}

use crate::frame::{read_frame, read_hello, write_frame, write_hello};

/// Drives a run across a set of rank shards.
#[derive(Debug)]
pub struct RankCoordinator {
    shards: Vec<Mutex<TcpStream>>,
    addresses: Vec<String>,
}

/// What a finished run produced.
#[derive(Debug, Clone)]
pub struct RankOutcome {
    pub ranks: HashMap<String, f32>,
    pub iterations: usize,
    pub converged: bool,
}

impl RankCoordinator {
    /// Connects to every shard, failing if any cannot be reached.
    ///
    /// A ranking run missing a shard is not a slightly worse ranking; it is a
    /// graph with a piece cut out, and every rank in it is wrong.
    pub async fn connect(addresses: &[String]) -> Result<Self> {
        let mut shards = Vec::with_capacity(addresses.len());
        for address in addresses {
            let mut stream = TcpStream::connect(address)
                .await
                .map_err(|e| Error::Corrupt(format!("rank shard {address}: {e}")))?;
            let _ = stream.set_nodelay(true);
            read_hello(&mut stream, PROTOCOL_VERSION).await?;
            write_hello(&mut stream, PROTOCOL_VERSION).await?;
            shards.push(Mutex::new(stream));
        }
        if shards.is_empty() {
            return Err(Error::Corrupt(
                "a ranking run needs at least one shard".into(),
            ));
        }
        Ok(Self {
            shards,
            addresses: addresses.to_vec(),
        })
    }

    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    async fn call(&self, shard: usize, request: &Request) -> Result<Response> {
        let mut guard = self.shards[shard].lock().await;
        let stream: &mut TcpStream = &mut guard;
        write_frame(stream, &request.encode()).await?;
        let payload = read_frame(stream).await?;
        let response = Response::decode(&payload)?;
        if let Response::Error { message } = &response {
            return Err(Error::Corrupt(format!(
                "rank shard {}: {message}",
                self.addresses[shard]
            )));
        }
        Ok(response)
    }

    /// Runs PageRank to convergence, or to `max_iterations`.
    ///
    /// `route` says which shard owns a URL, and it must agree with however the
    /// graph was partitioned. A disagreement does not error: contributions
    /// arrive at a shard that does not own the page and are dropped, and the
    /// ranking is quietly short of some rank. `ShardRanker::absorb` refuses to
    /// create the page rather than let two shards own it.
    pub async fn run<F: Fn(&str) -> usize>(
        &self,
        total_nodes: usize,
        options: &Options,
        route: F,
    ) -> Result<RankOutcome> {
        let shard_count = self.shards.len();
        for shard in 0..shard_count {
            self.call(
                shard,
                &Request::RankInit {
                    total_nodes,
                    shard_count,
                    damping: options.damping,
                    tolerance: options.tolerance,
                },
            )
            .await?;
        }

        let mut iterations = 0;
        let mut converged = false;

        for iteration in 1..=options.max_iterations {
            iterations = iteration;

            // Every shard's dangling mass, before anyone applies anything.
            let mut global_dangling = 0.0f32;
            for shard in 0..shard_count {
                if let Response::RankDangling { dangling } =
                    self.call(shard, &Request::RankDangling).await?
                {
                    global_dangling += dangling;
                }
            }

            // Everyone emits from last iteration's ranks, and only then does
            // anyone absorb. Interleaving the two would let a shard rank
            // itself against numbers from half of this iteration.
            let mut mail: Vec<Vec<(String, f32)>> = vec![Vec::new(); shard_count];
            for shard in 0..shard_count {
                let Response::RankBoundaries { per_shard } =
                    self.call(shard, &Request::RankEmit).await?
                else {
                    return Err(Error::Corrupt("a shard did not answer RankEmit".into()));
                };
                for bundle in per_shard {
                    for (uri, share) in bundle {
                        let destination = route(&uri).min(shard_count.saturating_sub(1));
                        mail[destination].push((uri, share));
                    }
                }
            }
            for (destination, contributions) in mail.into_iter().enumerate() {
                if !contributions.is_empty() {
                    self.call(destination, &Request::RankAbsorb { contributions })
                        .await?;
                }
            }

            let mut residual = 0.0f32;
            for shard in 0..shard_count {
                if let Response::RankRound {
                    residual: moved, ..
                } = self
                    .call(shard, &Request::RankApply { global_dangling })
                    .await?
                {
                    residual += moved;
                }
            }

            // Summed across the cluster. A shard that has stopped moving while
            // another has not is not done.
            if residual < options.tolerance {
                converged = true;
                break;
            }
        }

        let mut ranks = HashMap::new();
        for shard in 0..shard_count {
            if let Response::RankResults { ranks: theirs } =
                self.call(shard, &Request::RankResults).await?
            {
                ranks.extend(theirs);
            }
        }

        Ok(RankOutcome {
            ranks,
            iterations,
            converged,
        })
    }
}
