//! Sharing one host's `robots.txt` across a cluster.
//!
//! A crawl spread over fifty nodes fetches every host's `robots.txt` fifty
//! times. It is a small file and a cheap request, but it is still fifty
//! requests to a site that has done nothing to deserve them, and it is the
//! *first* thing each node does to a host it has never touched — so it is the
//! request most likely to look like a swarm.
//!
//! The fix mirrors [`Politeness`](crate::politeness::Politeness): a crawler
//! asks somebody before fetching, and does not know or care who answers. The
//! authority holds no HTTP client — the **first node to fetch a host reports
//! what it got**, and everyone after it is told. That keeps the authority
//! simple and keeps the fetching where the fetching already happens.
//!
//! Entries expire. A cache of `robots.txt` with no expiry is a promise to
//! obey yesterday's rules forever, and a site that adds a `Disallow` would
//! have no way of being heard.

use std::collections::HashMap;
use std::time::Duration;

use tokio::sync::{Mutex, mpsc, oneshot};
// Tokio's clock rather than the standard library's. They behave identically in
// a running program, and under `tokio::time::pause` this one can be advanced
// on demand — which is the difference between testing an expiry deterministically
// and sleeping for a while and hoping the machine was not busy.
use tokio::time::Instant;

/// What is known about a host's `robots.txt`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Known {
    /// Nobody has fetched it. The asker should, and should report back.
    Unknown,
    /// Fetched: the file's text, empty if the host has none.
    Rules(String),
    /// The host could not be asked — a server error, a dead connection. Not
    /// being able to ask is not permission, so this means skip the host.
    Unreachable,
}

/// A question about a host, and where to send the answer.
#[derive(Debug)]
pub struct RobotsRequest {
    pub host: String,
    /// `Some` when the asker is reporting what it fetched rather than asking.
    pub learned: Option<Known>,
    pub reply: oneshot::Sender<Known>,
}

/// A cache of `robots.txt` by host, in this process.
#[derive(Debug)]
pub struct LocalRobotsCache {
    entries: Mutex<HashMap<String, (Known, Instant)>>,
    ttl: Duration,
}

impl LocalRobotsCache {
    #[must_use]
    pub fn new(ttl: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
        }
    }

    /// What is known about `host` right now.
    pub async fn get(&self, host: &str) -> Known {
        let entries = self.entries.lock().await;
        match entries.get(host) {
            Some((known, at)) if at.elapsed() < self.ttl => known.clone(),
            // Expired or absent are the same answer: go and look.
            _ => Known::Unknown,
        }
    }

    /// Records what a crawler found.
    pub async fn learn(&self, host: &str, known: Known) {
        self.entries
            .lock()
            .await
            .insert(host.to_owned(), (known, Instant::now()));
    }

    /// Forgets everything past its time, so the map does not grow with every
    /// host ever crawled.
    pub async fn sweep(&self) -> usize {
        let mut entries = self.entries.lock().await;
        let before = entries.len();
        entries.retain(|_, (_, at)| at.elapsed() < self.ttl);
        before - entries.len()
    }

    pub async fn len(&self) -> usize {
        self.entries.lock().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

/// Where a crawler asks about `robots.txt`.
#[derive(Debug)]
pub enum RobotsCache {
    /// This process remembers. What a single-node crawl uses.
    Local(LocalRobotsCache),
    /// Somebody else remembers, reached through this channel.
    Delegated(mpsc::Sender<RobotsRequest>),
}

impl RobotsCache {
    #[must_use]
    pub fn local(ttl: Duration) -> Self {
        Self::Local(LocalRobotsCache::new(ttl))
    }

    /// Asks what is known about `host`.
    ///
    /// A delegated authority that has gone away answers `Unknown`, which makes
    /// the crawler fetch for itself: a lost cache should cost extra requests,
    /// not a stopped crawl.
    pub async fn get(&self, host: &str) -> Known {
        match self {
            Self::Local(cache) => cache.get(host).await,
            Self::Delegated(sender) => self.ask(sender, host, None).await,
        }
    }

    /// Reports what this crawler fetched, so nobody else has to.
    pub async fn learn(&self, host: &str, known: Known) {
        match self {
            Self::Local(cache) => cache.learn(host, known).await,
            Self::Delegated(sender) => {
                let _ = self.ask(sender, host, Some(known)).await;
            }
        }
    }

    async fn ask(
        &self,
        sender: &mpsc::Sender<RobotsRequest>,
        host: &str,
        learned: Option<Known>,
    ) -> Known {
        let (reply, answer) = oneshot::channel();
        let request = RobotsRequest {
            host: host.to_owned(),
            learned,
            reply,
        };
        if sender.send(request).await.is_err() {
            return Known::Unknown;
        }
        answer.await.unwrap_or(Known::Unknown)
    }
}

impl Default for RobotsCache {
    fn default() -> Self {
        // A day: long enough that a crawl of any length fetches each host once,
        // short enough that a site changing its mind is heard the next day.
        Self::local(Duration::from_secs(24 * 60 * 60))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_unknown_host_is_unknown() {
        let cache = LocalRobotsCache::new(Duration::from_secs(60));
        assert_eq!(cache.get("example.com").await, Known::Unknown);
    }

    #[tokio::test]
    async fn what_is_learned_is_remembered() {
        let cache = LocalRobotsCache::new(Duration::from_secs(60));
        cache
            .learn(
                "example.com",
                Known::Rules("User-agent: *\nDisallow: /x".into()),
            )
            .await;
        let Known::Rules(text) = cache.get("example.com").await else {
            panic!("expected rules");
        };
        assert!(text.contains("Disallow: /x"));
    }

    #[tokio::test]
    async fn an_unreachable_host_is_remembered_as_unreachable() {
        // Not as "no rules". Being unable to ask is not permission, and a
        // cache that forgot the difference would turn a failed fetch into a
        // licence to crawl everything.
        let cache = LocalRobotsCache::new(Duration::from_secs(60));
        cache.learn("down.example", Known::Unreachable).await;
        assert_eq!(cache.get("down.example").await, Known::Unreachable);
    }

    #[tokio::test]
    async fn an_expired_entry_is_unknown_again() {
        let cache = LocalRobotsCache::new(Duration::from_millis(20));
        cache
            .learn("example.com", Known::Rules(String::new()))
            .await;
        assert_ne!(cache.get("example.com").await, Known::Unknown);
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert_eq!(
            cache.get("example.com").await,
            Known::Unknown,
            "a cache with no expiry obeys yesterday's rules forever"
        );
    }

    #[tokio::test]
    async fn sweeping_drops_only_what_expired() {
        let cache = LocalRobotsCache::new(Duration::from_millis(30));
        cache
            .learn("old.example", Known::Rules(String::new()))
            .await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        cache
            .learn("new.example", Known::Rules(String::new()))
            .await;

        assert_eq!(cache.len().await, 2);
        assert_eq!(cache.sweep().await, 1);
        assert_eq!(cache.len().await, 1);
        assert_ne!(cache.get("new.example").await, Known::Unknown);
    }

    #[tokio::test]
    async fn a_delegated_cache_is_asked_and_told() {
        let (tx, mut rx) = mpsc::channel(8);
        let cache = RobotsCache::Delegated(tx);

        tokio::spawn(async move {
            let store = LocalRobotsCache::new(Duration::from_secs(60));
            while let Some(request) = rx.recv().await {
                let answer = match request.learned {
                    Some(known) => {
                        store.learn(&request.host, known.clone()).await;
                        known
                    }
                    None => store.get(&request.host).await,
                };
                let _ = request.reply.send(answer);
            }
        });

        assert_eq!(cache.get("example.com").await, Known::Unknown);
        cache
            .learn("example.com", Known::Rules("Disallow: /x".into()))
            .await;
        assert_eq!(
            cache.get("example.com").await,
            Known::Rules("Disallow: /x".into())
        );
    }

    #[tokio::test]
    async fn a_dead_authority_makes_the_crawler_fetch_for_itself() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let cache = RobotsCache::Delegated(tx);
        // Unknown, not Unreachable: a lost cache costs extra requests, it does
        // not stop the crawl or forbid a host nobody has asked about.
        assert_eq!(cache.get("example.com").await, Known::Unknown);
    }
}
