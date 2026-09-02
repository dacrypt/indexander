//! A polite, asynchronous web crawler.
//!
//! What "polite" means here, concretely:
//!
//! * `robots.txt` is fetched once per host and obeyed. A host whose
//!   `robots.txt` could not be fetched because of a *server* error is not
//!   crawled at all — being unable to ask is not permission.
//! * Requests to one host are spaced by a delay, and the host's own
//!   `Crawl-delay` overrides ours when it asks for more.
//! * The crawler identifies itself, with a URL a site owner can visit and an
//!   address they can write to.
//! * Response bodies are capped, redirects are bounded, and everything has a
//!   timeout, so one hostile server cannot stall the crawl.
//!
//! Concurrency is across hosts, not within them: several workers run at once,
//! but each waits its turn per host.

pub mod extract;
pub mod frontier;
pub mod normalize;
pub mod robots;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use indexander_core::Document;
use tokio::sync::{Mutex, mpsc};
use url::Url;

use crate::frontier::{Frontier, Limits};
use crate::normalize::{host_of, resolve};
use crate::robots::Robots;

/// How the crawler behaves.
#[derive(Debug, Clone)]
pub struct Config {
    /// Sent as `User-Agent`. Include a URL or an address: a site owner who
    /// wants you to stop should not have to guess who you are.
    pub user_agent: String,
    pub limits: Limits,
    /// Workers running at once. Politeness is per host, so this is how many
    /// *different* hosts can be in flight, not how hard one host is hit.
    pub concurrency: usize,
    /// Minimum gap between two requests to the same host.
    pub delay: Duration,
    pub timeout: Duration,
    /// Bodies larger than this are truncated. A crawler that will read any
    /// number of bytes can be made to read forever.
    pub max_body_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            user_agent: concat!(
                "indexander/",
                env!("CARGO_PKG_VERSION"),
                " (+https://github.com/dacrypt/indexander)"
            )
            .to_owned(),
            limits: Limits::default(),
            concurrency: 4,
            delay: Duration::from_millis(500),
            timeout: Duration::from_secs(15),
            max_body_bytes: 4 * 1024 * 1024,
        }
    }
}

/// What happened during a crawl.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub fetched: usize,
    pub indexed: usize,
    pub disallowed_by_robots: usize,
    pub skipped_content_type: usize,
    pub errors: usize,
}

/// Shared mutable state. One lock, held briefly, rather than one per field:
/// the contended resource is the frontier and the rest travels with it.
#[derive(Debug)]
struct Shared {
    frontier: Frontier,
    robots: HashMap<String, Robots>,
    next_allowed: HashMap<String, Instant>,
    stats: Stats,
}

/// Crawls from `seeds`, sending every indexable page to `sink`.
///
/// Returns when the frontier is exhausted or the page budget is spent.
pub async fn crawl(
    config: &Config,
    seeds: &[Url],
    sink: mpsc::Sender<Document>,
) -> Result<Stats, String> {
    let client = reqwest::Client::builder()
        .user_agent(config.user_agent.clone())
        .timeout(config.timeout)
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| format!("building http client: {e}"))?;

    let mut frontier = Frontier::new(config.limits.clone());
    for seed in seeds {
        frontier.seed(seed.clone());
    }

    let shared = Arc::new(Mutex::new(Shared {
        frontier,
        robots: HashMap::new(),
        next_allowed: HashMap::new(),
        stats: Stats::default(),
    }));

    let mut workers = tokio::task::JoinSet::new();
    for _ in 0..config.concurrency.max(1) {
        let shared = Arc::clone(&shared);
        let client = client.clone();
        let config = config.clone();
        let sink = sink.clone();
        workers.spawn(async move { worker(&shared, &client, &config, &sink).await });
    }
    while workers.join_next().await.is_some() {}

    let stats = shared.lock().await.stats;
    Ok(stats)
}

/// Convenience wrapper that collects every page into a vector.
pub async fn crawl_to_vec(
    config: &Config,
    seeds: &[Url],
) -> Result<(Vec<Document>, Stats), String> {
    // Bounded so a slow consumer applies backpressure instead of growing
    // without limit; here the consumer is instant, so the bound is nominal.
    let (tx, mut rx) = mpsc::channel(256);
    let crawling = crawl(config, seeds, tx);
    let collecting = async {
        let mut out = Vec::new();
        while let Some(doc) = rx.recv().await {
            out.push(doc);
        }
        out
    };
    let (stats, docs) = tokio::join!(crawling, collecting);
    Ok((docs, stats?))
}

async fn worker(
    shared: &Arc<Mutex<Shared>>,
    client: &reqwest::Client,
    config: &Config,
    sink: &mpsc::Sender<Document>,
) {
    loop {
        let Some(pending) = shared.lock().await.frontier.next_url() else {
            return;
        };
        let host = host_of(&pending.url);

        // Ask this host's robots.txt once, before anything else.
        let Some(rules) = ensure_robots(shared, client, config, &pending.url).await else {
            continue;
        };
        let path_and_query = pending.url[url::Position::BeforePath..].to_owned();
        if !rules.allows(&path_and_query) {
            shared.lock().await.stats.disallowed_by_robots += 1;
            continue;
        }

        wait_turn(shared, &host, rules.crawl_delay().unwrap_or(config.delay)).await;

        let fetched = fetch(client, &pending.url, config.max_body_bytes).await;
        let Some(page) = fetched else {
            shared.lock().await.stats.errors += 1;
            continue;
        };
        shared.lock().await.stats.fetched += 1;

        if !page.is_html && !page.is_text {
            shared.lock().await.stats.skipped_content_type += 1;
            continue;
        }

        let parsed = extract::extract(&page.body);

        // Links are queued before the page is emitted, so the crawl keeps
        // moving even if the consumer is slow.
        // Every outlink, resolved. These are collected even for links the
        // frontier refuses to queue — an off-host link is still an edge in the
        // graph, and PageRank flows along it whether or not we fetch the page.
        let mut outlinks: Vec<String> = Vec::new();
        if !parsed.nofollow {
            let base = parsed
                .base
                .as_deref()
                .and_then(|b| Url::parse(b).ok())
                .unwrap_or_else(|| page.url.clone());
            let mut guard = shared.lock().await;
            for link in &parsed.links {
                if let Some(target) = resolve(&base, &link.href) {
                    outlinks.push(target.to_string());
                    guard.enqueue_link(target, pending.depth + 1, &link.text);
                }
            }
        }

        if parsed.noindex {
            continue;
        }

        let anchors = shared.lock().await.frontier.take_anchors(&page.url);
        let title = if parsed.title.is_empty() {
            page.url.path().to_owned()
        } else {
            parsed.title
        };
        let mut document = Document::new(page.url.to_string(), title, parsed.text);
        document.anchors = anchors;
        document.links = outlinks;

        shared.lock().await.stats.indexed += 1;
        if sink.send(document).await.is_err() {
            // The consumer hung up; nothing left to do.
            return;
        }
    }
}

impl Shared {
    fn enqueue_link(&mut self, url: Url, depth: u32, anchor: &str) {
        self.frontier.enqueue(url, depth, Some(anchor));
    }
}

/// Fetches and caches a host's `robots.txt`. `None` means skip this URL.
async fn ensure_robots(
    shared: &Arc<Mutex<Shared>>,
    client: &reqwest::Client,
    config: &Config,
    url: &Url,
) -> Option<Robots> {
    let host = host_of(url);
    if let Some(cached) = shared.lock().await.robots.get(&host) {
        return Some(cached.clone());
    }

    let mut robots_url = url.clone();
    robots_url.set_path("/robots.txt");
    robots_url.set_query(None);

    // robots.txt is itself a request to the host, so it waits its turn too.
    wait_turn(shared, &host, config.delay).await;

    let rules = match client.get(robots_url).send().await {
        Ok(response) if response.status().is_success() => response.text().await.map_or_else(
            |_| Robots::allow_all(),
            |t| Robots::parse(&t, &config.user_agent),
        ),
        // No robots.txt, or it is gone: the site has no rules for us.
        Ok(response) if response.status().is_client_error() => Robots::allow_all(),
        // A server error or a failed connection is not consent.
        _ => Robots::deny_all(),
    };

    shared.lock().await.robots.insert(host, rules.clone());
    Some(rules)
}

/// Blocks until this host may be contacted again, then reserves the next slot.
async fn wait_turn(shared: &Arc<Mutex<Shared>>, host: &str, delay: Duration) {
    let sleep_for = {
        let mut guard = shared.lock().await;
        let now = Instant::now();
        let slot = guard
            .next_allowed
            .get(host)
            .copied()
            .filter(|t| *t > now)
            .unwrap_or(now);
        // Reserve while still holding the lock, so two workers cannot both
        // decide it is their turn.
        guard.next_allowed.insert(host.to_owned(), slot + delay);
        slot.saturating_duration_since(now)
    };
    if !sleep_for.is_zero() {
        tokio::time::sleep(sleep_for).await;
    }
}

struct Page {
    /// The URL after redirects, which is the page's real identity.
    url: Url,
    body: String,
    is_html: bool,
    is_text: bool,
}

async fn fetch(client: &reqwest::Client, url: &Url, max_bytes: usize) -> Option<Page> {
    let response = client.get(url.clone()).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let final_url = response.url().clone();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let is_html = content_type.contains("html") || content_type.is_empty();
    let is_text = content_type.starts_with("text/");

    if !is_html && !is_text {
        return Some(Page {
            url: final_url,
            body: String::new(),
            is_html: false,
            is_text: false,
        });
    }

    // `text()` honours the charset in the Content-Type header, which is how a
    // latin-1 page comes back as correct Rust strings instead of mojibake.
    let mut body = response.text().await.ok()?;
    if body.len() > max_bytes {
        // Truncate on a character boundary, not a byte one.
        let mut cut = max_bytes;
        while cut > 0 && !body.is_char_boundary(cut) {
            cut -= 1;
        }
        body.truncate(cut);
    }

    Some(Page {
        url: final_url,
        body,
        is_html,
        is_text,
    })
}
