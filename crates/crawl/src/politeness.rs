//! Who decides when a host may be contacted.
//!
//! Politeness is per host, and a crawl sharded by URL scatters one host across
//! every node. Five nodes each holding pages of `example.com` will each
//! independently conclude they are being polite, and the site will receive
//! five times what it asked for. The rate limit has to be owned by exactly one
//! process per host, and it is not the process that owns the URL.
//!
//! So the crawler does not decide. It asks for a **lease**: permission to make
//! some number of requests to a host, starting at a particular moment. Who
//! answers is deliberately not the crawler's business — [`Politeness::Local`]
//! answers from a map in this process, [`Politeness::Delegated`] forwards the
//! question down a channel to whoever is on the other end. The crawler cannot
//! tell the difference, which is the point: a one-node crawl and a fifty-node
//! crawl run the same code.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, mpsc, oneshot};

/// Permission to fetch from a host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lease {
    /// Not before this moment.
    pub not_before: Instant,
    /// How many requests this lease covers. Leases batch because asking costs
    /// a round trip and fetching does not: twenty permits, twenty fetches.
    pub permits: usize,
    /// The gap to leave between those requests.
    pub spacing: Duration,
}

impl Lease {
    /// A lease that may be used immediately, once.
    #[must_use]
    pub fn now() -> Self {
        Self {
            not_before: Instant::now(),
            permits: 1,
            spacing: Duration::ZERO,
        }
    }
}

/// A request for permission, and where to send the answer.
#[derive(Debug)]
pub struct LeaseRequest {
    pub host: String,
    /// The delay this crawler believes the host wants, from its `robots.txt`
    /// or from configuration. An authority is free to enforce more.
    pub requested_delay: Duration,
    pub permits: usize,
    pub reply: oneshot::Sender<Lease>,
}

/// Rate limiting for a set of hosts, in this process.
///
/// Hands out a slot per host and moves the host's next slot forward, holding
/// the lock across both so two askers cannot be told the same moment.
#[derive(Debug, Default)]
pub struct LocalPolicy {
    next_allowed: Mutex<HashMap<String, Instant>>,
    /// The shortest gap this policy will enforce, whatever is asked for.
    floor: Duration,
}

impl LocalPolicy {
    #[must_use]
    pub fn new(floor: Duration) -> Self {
        Self {
            next_allowed: Mutex::new(HashMap::new()),
            floor,
        }
    }

    /// Reserves `permits` requests to `host`, spaced by at least the floor.
    pub async fn lease(&self, host: &str, requested_delay: Duration, permits: usize) -> Lease {
        let permits = permits.max(1);
        let spacing = requested_delay.max(self.floor);

        let mut guard = self.next_allowed.lock().await;
        let now = Instant::now();
        let slot = guard.get(host).copied().filter(|t| *t > now).unwrap_or(now);
        // The whole batch is reserved at once: the next asker is told a moment
        // after every request in this lease has been made.
        let after = slot + spacing.saturating_mul(u32::try_from(permits).unwrap_or(u32::MAX));
        guard.insert(host.to_owned(), after);
        drop(guard);

        Lease {
            not_before: slot,
            permits,
            spacing,
        }
    }
}

/// Where a crawler goes for permission.
#[derive(Debug)]
pub enum Politeness {
    /// This process decides. What a single-node crawl uses.
    Local(LocalPolicy),
    /// Somebody else decides, reached through this channel.
    ///
    /// The crawler knows nothing about how: a socket, another task, a test
    /// harness. That ignorance is what keeps the distributed path and the
    /// local path the same code.
    Delegated(mpsc::Sender<LeaseRequest>),
}

impl Politeness {
    /// A local policy with the given minimum gap between requests.
    #[must_use]
    pub fn local(floor: Duration) -> Self {
        Self::Local(LocalPolicy::new(floor))
    }

    /// Asks for permission to make `permits` requests to `host`.
    ///
    /// A delegated authority that has gone away falls back to permission now:
    /// a crawl that stops because its rate limiter crashed is worse than one
    /// that carries on at the rate its own configuration asks for, and the
    /// per-request spacing in the returned lease still applies.
    pub async fn lease(&self, host: &str, requested_delay: Duration, permits: usize) -> Lease {
        match self {
            Self::Local(policy) => policy.lease(host, requested_delay, permits).await,
            Self::Delegated(sender) => {
                let (reply, answer) = oneshot::channel();
                let request = LeaseRequest {
                    host: host.to_owned(),
                    requested_delay,
                    permits: permits.max(1),
                    reply,
                };
                if sender.send(request).await.is_err() {
                    return Lease {
                        not_before: Instant::now(),
                        permits: permits.max(1),
                        spacing: requested_delay,
                    };
                }
                answer.await.unwrap_or_else(|_| Lease {
                    not_before: Instant::now(),
                    permits: permits.max(1),
                    spacing: requested_delay,
                })
            }
        }
    }
}

impl Default for Politeness {
    fn default() -> Self {
        Self::local(Duration::from_millis(500))
    }
}

/// Waits until a lease's moment arrives.
pub async fn wait_for(lease: Lease) {
    let now = Instant::now();
    if lease.not_before > now {
        tokio::time::sleep(lease.not_before - now).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_first_lease_is_immediate() {
        let policy = LocalPolicy::new(Duration::from_millis(100));
        let lease = policy
            .lease("example.com", Duration::from_millis(100), 1)
            .await;
        assert!(lease.not_before <= Instant::now());
        assert_eq!(lease.permits, 1);
    }

    #[tokio::test]
    async fn consecutive_leases_for_one_host_are_spaced() {
        let policy = LocalPolicy::new(Duration::from_millis(50));
        let first = policy.lease("a.com", Duration::ZERO, 1).await;
        let second = policy.lease("a.com", Duration::ZERO, 1).await;
        let third = policy.lease("a.com", Duration::ZERO, 1).await;

        assert!(second.not_before >= first.not_before + Duration::from_millis(50));
        assert!(third.not_before >= second.not_before + Duration::from_millis(50));
    }

    #[tokio::test]
    async fn different_hosts_do_not_wait_for_each_other() {
        let policy = LocalPolicy::new(Duration::from_millis(500));
        let a = policy.lease("a.com", Duration::ZERO, 1).await;
        let b = policy.lease("b.com", Duration::ZERO, 1).await;
        // Both may go now: politeness is per host, not global.
        let now = Instant::now();
        assert!(a.not_before <= now && b.not_before <= now);
    }

    #[tokio::test]
    async fn a_batch_reserves_the_whole_batch() {
        // Twenty permits must push the next asker twenty slots out, or two
        // crawlers each holding a batch would interleave and double the rate.
        let policy = LocalPolicy::new(Duration::from_millis(10));
        let batch = policy.lease("a.com", Duration::ZERO, 20).await;
        let next = policy.lease("a.com", Duration::ZERO, 1).await;
        assert_eq!(batch.permits, 20);
        assert!(
            next.not_before >= batch.not_before + Duration::from_millis(200),
            "the batch did not reserve its own slots"
        );
    }

    #[tokio::test]
    async fn the_floor_beats_a_smaller_request() {
        // A client asking for no delay does not get no delay.
        let policy = LocalPolicy::new(Duration::from_millis(100));
        let lease = policy.lease("a.com", Duration::ZERO, 1).await;
        assert_eq!(lease.spacing, Duration::from_millis(100));
    }

    #[tokio::test]
    async fn a_larger_request_beats_the_floor() {
        // A host asking for more through Crawl-delay gets more.
        let policy = LocalPolicy::new(Duration::from_millis(100));
        let lease = policy.lease("a.com", Duration::from_secs(2), 1).await;
        assert_eq!(lease.spacing, Duration::from_secs(2));
    }

    #[tokio::test]
    async fn a_delegated_authority_is_asked() {
        let (tx, mut rx) = mpsc::channel(4);
        let politeness = Politeness::Delegated(tx);

        tokio::spawn(async move {
            while let Some(request) = rx.recv().await {
                assert_eq!(request.host, "example.com");
                let _ = request.reply.send(Lease {
                    not_before: Instant::now() + Duration::from_millis(30),
                    permits: request.permits,
                    spacing: Duration::from_millis(30),
                });
            }
        });

        let lease = politeness
            .lease("example.com", Duration::from_millis(10), 5)
            .await;
        assert_eq!(lease.permits, 5);
        assert_eq!(lease.spacing, Duration::from_millis(30));
    }

    #[tokio::test]
    async fn a_dead_authority_does_not_stop_the_crawl() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let politeness = Politeness::Delegated(tx);
        let lease = politeness
            .lease("example.com", Duration::from_millis(20), 3)
            .await;
        // Falls back to the crawler's own idea of the delay, not to no delay.
        assert_eq!(lease.spacing, Duration::from_millis(20));
        assert_eq!(lease.permits, 3);
    }
}
