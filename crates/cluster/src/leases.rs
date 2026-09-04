//! The node that owns a host's rate limit.
//!
//! Crawl work is sharded by URL, which scatters one host across every node.
//! Each of them would independently decide it is being polite, and the site
//! would get the sum. So a second, independent mapping says which node owns
//! each *host's* rate limit, and every node asks that one before fetching.
//!
//! The authority does not fetch anything itself and does not read
//! `robots.txt`: it hands out slots. The asker supplies the delay it believes
//! the host wants, from that host's own `Crawl-delay`, and the authority
//! enforces the larger of that and its own floor — so a misconfigured or
//! malicious crawler cannot talk the cluster into hammering a site.

use std::sync::Arc;
use std::time::Duration;

use indexander_core::Result;
use indexander_crawl::politeness::{LeaseRequest, LocalPolicy, Politeness};
use indexander_crawl::robots_cache::{Known, LocalRobotsCache, RobotsCache, RobotsRequest};
use indexander_proto::message::{PROTOCOL_VERSION, Request, Response};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

use crate::frame::{read_frame, read_hello, write_frame, write_hello};

/// Milliseconds, rounded up.
///
/// The wire carries whole milliseconds and `as_millis` truncates, which grants
/// a slot slightly *earlier* than the authority computed. Sub-millisecond on
/// one lease, but it is the wrong direction: it compounds across a crawl into
/// a rate above what the site asked for, which is the exact thing this whole
/// mechanism exists to prevent. Waiting a fraction too long is always safe.
fn millis_rounding_up(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos().div_ceil(1_000_000)).unwrap_or(u64::MAX)
}

/// Serves lease requests, and remembers each host's `robots.txt`.
///
/// The two belong together: whoever owns a host's rate limit is the natural
/// place to remember what that host said about being crawled, and it means one
/// address per host rather than two.
///
/// The authority still holds no HTTP client. The first crawler to fetch a
/// host's `robots.txt` reports what it got, and everyone after is told — which
/// keeps fetching where fetching already happens and keeps this a pure
/// bookkeeping service.
#[derive(Debug)]
pub struct LeaseAuthority {
    policy: LocalPolicy,
    robots: LocalRobotsCache,
}

impl LeaseAuthority {
    /// `floor` is the shortest gap this authority will ever grant, whatever a
    /// crawler asks for.
    #[must_use]
    pub fn new(floor: Duration) -> Self {
        Self::with_robots_ttl(floor, Duration::from_secs(24 * 60 * 60))
    }

    /// As [`LeaseAuthority::new`], choosing how long a `robots.txt` is
    /// remembered. A cache that never expires is a promise to obey yesterday's
    /// rules forever.
    #[must_use]
    pub fn with_robots_ttl(floor: Duration, robots_ttl: Duration) -> Self {
        Self {
            policy: LocalPolicy::new(floor),
            robots: LocalRobotsCache::new(robots_ttl),
        }
    }

    /// Answers one request.
    pub async fn handle(&self, request: &Request) -> Response {
        match request {
            Request::Lease {
                host,
                requested_delay_ms,
                permits,
            } => {
                let lease = self
                    .policy
                    .lease(host, Duration::from_millis(*requested_delay_ms), *permits)
                    .await;
                let wait = lease
                    .not_before
                    .saturating_duration_since(std::time::Instant::now());
                Response::Lease {
                    wait_ms: millis_rounding_up(wait),
                    permits: lease.permits,
                    spacing_ms: millis_rounding_up(lease.spacing),
                }
            }
            Request::Robots {
                host,
                learned,
                state,
                text,
            } => {
                if *learned {
                    self.robots.learn(host, decode_known(*state, text)).await;
                }
                let known = self.robots.get(host).await;
                let (state, text) = encode_known(&known);
                Response::Robots { state, text }
            }
            other => Response::Error {
                message: format!("a lease authority cannot answer {other:?}"),
            },
        }
    }

    /// Serves on `listener` until the task is dropped.
    pub async fn serve(self: Arc<Self>, listener: TcpListener) -> Result<()> {
        loop {
            let (stream, _) = listener.accept().await?;
            let authority = Arc::clone(&self);
            tokio::spawn(async move {
                let _ = authority.connection(stream).await;
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

/// The wire encoding of what is known about a host.
fn encode_known(known: &Known) -> (u8, String) {
    match known {
        Known::Unknown => (0, String::new()),
        Known::Rules(text) => (1, text.clone()),
        Known::Unreachable => (2, String::new()),
    }
}

fn decode_known(state: u8, text: &str) -> Known {
    match state {
        1 => Known::Rules(text.to_owned()),
        2 => Known::Unreachable,
        // Anything else is treated as "nobody knows", which makes the asker
        // fetch for itself. A state this build does not understand must not
        // become permission.
        _ => Known::Unknown,
    }
}

/// Connects to a lease authority and returns a [`RobotsCache`] that defers to
/// it.
///
/// A separate connection from the one [`remote_politeness`] opens, so a slow
/// `robots.txt` exchange never sits in front of a lease.
pub async fn remote_robots_cache(address: &str) -> Result<RobotsCache> {
    let mut stream = TcpStream::connect(address).await?;
    let _ = stream.set_nodelay(true);
    read_hello(&mut stream, PROTOCOL_VERSION).await?;
    write_hello(&mut stream, PROTOCOL_VERSION).await?;

    let (tx, mut rx) = mpsc::channel::<RobotsRequest>(64);
    tokio::spawn(async move {
        while let Some(request) = rx.recv().await {
            let (state, text) = request
                .learned
                .as_ref()
                .map_or((0, String::new()), encode_known);
            let wire = Request::Robots {
                host: request.host,
                learned: request.learned.is_some(),
                state,
                text,
            };
            // If the authority is gone, dropping the reply leaves the crawler
            // with "nobody knows", so it fetches for itself rather than
            // stopping.
            let Ok(answer) = ask_robots(&mut stream, &wire).await else {
                return;
            };
            let _ = request.reply.send(answer);
        }
    });

    Ok(RobotsCache::Delegated(tx))
}

async fn ask_robots(stream: &mut TcpStream, request: &Request) -> Result<Known> {
    write_frame(stream, &request.encode()).await?;
    let payload = read_frame(stream).await?;
    match Response::decode(&payload)? {
        Response::Robots { state, text } => Ok(decode_known(state, &text)),
        other => Err(indexander_core::Error::Corrupt(format!(
            "lease authority answered {other:?} to a robots request"
        ))),
    }
}

/// Connects to a lease authority and returns a [`Politeness`] that defers to
/// it.
///
/// The returned value looks exactly like a local policy to the crawler, which
/// is the whole design: a distributed crawl and a single-node crawl run the
/// same code with a different value here.
pub async fn remote_politeness(address: &str) -> Result<Politeness> {
    let mut stream = TcpStream::connect(address).await?;
    let _ = stream.set_nodelay(true);
    read_hello(&mut stream, PROTOCOL_VERSION).await?;
    write_hello(&mut stream, PROTOCOL_VERSION).await?;

    // One task owns the connection and serialises requests onto it, so the
    // crawler's workers never contend for the socket.
    let (tx, mut rx) = mpsc::channel::<LeaseRequest>(64);
    tokio::spawn(async move {
        while let Some(request) = rx.recv().await {
            let wire = Request::Lease {
                host: request.host,
                requested_delay_ms: u64::try_from(request.requested_delay.as_millis())
                    .unwrap_or(u64::MAX),
                permits: request.permits,
            };
            // If the authority is gone, dropping the reply lets the crawler
            // fall back to its own delay rather than stall.
            let Ok(lease) = ask(&mut stream, &wire).await else {
                return;
            };
            let _ = request.reply.send(lease);
        }
    });

    Ok(Politeness::Delegated(tx))
}

async fn ask(
    stream: &mut TcpStream,
    request: &Request,
) -> Result<indexander_crawl::politeness::Lease> {
    write_frame(stream, &request.encode()).await?;
    let payload = read_frame(stream).await?;
    match Response::decode(&payload)? {
        Response::Lease {
            wait_ms,
            permits,
            spacing_ms,
        } => Ok(indexander_crawl::politeness::Lease {
            not_before: std::time::Instant::now() + Duration::from_millis(wait_ms),
            permits,
            spacing: Duration::from_millis(spacing_ms),
        }),
        other => Err(indexander_core::Error::Corrupt(format!(
            "lease authority answered {other:?}"
        ))),
    }
}
