//! The set of URLs waiting to be fetched, and the bookkeeping around it.
//!
//! Three jobs, all of which a crawler gets wrong at least once:
//!
//! * **Never fetch the same page twice.** A URL is seen the moment it is
//!   *queued*, not when it is fetched, or a page linked from ten others is
//!   queued ten times.
//! * **Hold anchor text for pages not yet fetched.** When page A links to B
//!   with the words "el buscador colombiano", those words describe B — and A
//!   is usually crawled long before B. The text has to wait somewhere.
//! * **Stay bounded.** Depth, page count and per-host limits, because the web
//!   is infinite and calendars generate URLs forever.

use std::collections::{HashMap, HashSet, VecDeque};

use url::Url;

use crate::normalize::host_of;

/// Limits that keep a crawl finite.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Stop after this many pages have been handed out.
    pub max_pages: usize,
    /// How many links deep from a seed to follow. Seeds are depth 0.
    pub max_depth: u32,
    /// Never hand out more than this many pages from any one host.
    pub max_pages_per_host: usize,
    /// Refuse URLs whose host is not the host of a seed.
    pub same_host_only: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_pages: 1_000,
            max_depth: 3,
            max_pages_per_host: 500,
            same_host_only: true,
        }
    }
}

/// A URL waiting to be fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pending {
    pub url: Url,
    pub depth: u32,
}

#[derive(Debug)]
pub struct Frontier {
    queue: VecDeque<Pending>,
    seen: HashSet<String>,
    /// Anchor text collected for a URL from the pages that link to it.
    anchors: HashMap<String, Vec<String>>,
    per_host: HashMap<String, usize>,
    seed_hosts: HashSet<String>,
    /// Handed to a worker and not yet accounted for, with the depth it was
    /// queued at.
    ///
    /// A URL leaves the queue the moment a worker takes it and enters `seen`
    /// when it is first queued, so between those two a crash loses it from
    /// both: not queued, never to be queued again. That is a page silently
    /// missing from the index, which is the one failure a checkpoint must not
    /// have. Anything still in here goes back into the queue when the frontier
    /// is saved.
    in_flight: HashMap<String, u32>,
    handed_out: usize,
    limits: Limits,
}

impl Frontier {
    #[must_use]
    pub fn new(limits: Limits) -> Self {
        Self {
            queue: VecDeque::new(),
            seen: HashSet::new(),
            in_flight: HashMap::new(),
            anchors: HashMap::new(),
            per_host: HashMap::new(),
            seed_hosts: HashSet::new(),
            handed_out: 0,
            limits,
        }
    }

    /// Adds a starting point. Seeds are depth 0 and define the allowed hosts.
    pub fn seed(&mut self, url: Url) -> bool {
        self.seed_hosts.insert(host_of(&url));
        self.enqueue(url, 0, None)
    }

    /// Offers a URL discovered at `depth`, with the anchor text that pointed
    /// at it. Returns whether it was newly queued.
    ///
    /// Anchor text is recorded even when the URL was already seen: the second
    /// page to link somewhere describes it just as well as the first.
    pub fn enqueue(&mut self, url: Url, depth: u32, anchor: Option<&str>) -> bool {
        let key = url.as_str().to_owned();

        if let Some(text) = anchor {
            let text = text.trim();
            // A cap per URL: some sites link the same target thousands of
            // times, and the hundredth "click here" adds nothing.
            if !text.is_empty() {
                let list = self.anchors.entry(key.clone()).or_default();
                if list.len() < 32 && !list.iter().any(|t| t == text) {
                    list.push(text.to_owned());
                }
            }
        }

        if depth > self.limits.max_depth {
            return false;
        }
        if self.limits.same_host_only
            && !self.seed_hosts.is_empty()
            && !self.seed_hosts.contains(&host_of(&url))
        {
            return false;
        }
        if !self.seen.insert(key) {
            return false;
        }
        self.queue.push_back(Pending { url, depth });
        true
    }

    /// Takes the next URL to fetch, or `None` when the crawl is done.
    ///
    /// Not called `next`: this is not an iterator, and a caller who treats it
    /// like one will be surprised by the per-host limits skipping entries.
    ///
    /// Breadth first: shallow pages tend to be the ones that matter, and a
    /// depth-first crawler disappears down one directory and never returns.
    pub fn next_url(&mut self) -> Option<Pending> {
        while let Some(pending) = self.queue.pop_front() {
            if self.handed_out >= self.limits.max_pages {
                return None;
            }
            let host = host_of(&pending.url);
            let count = self.per_host.entry(host).or_insert(0);
            if *count >= self.limits.max_pages_per_host {
                continue;
            }
            *count += 1;
            self.handed_out += 1;
            self.in_flight
                .insert(pending.url.to_string(), pending.depth);
            return Some(pending);
        }
        None
    }

    /// The anchor text gathered for a URL, consumed as the page is indexed.
    pub fn take_anchors(&mut self, url: &Url) -> Vec<String> {
        self.anchors.remove(url.as_str()).unwrap_or_default()
    }

    #[must_use]
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    #[must_use]
    pub fn seen_count(&self) -> usize {
        self.seen.len()
    }

    #[must_use]
    /// Rebuilds a frontier from its parts. Only [`crate::checkpoint`] calls
    /// this, and it exists so the fields can stay private everywhere else.
    pub(crate) fn from_parts(
        limits: Limits,
        queue: VecDeque<Pending>,
        seen: HashSet<String>,
        anchors: HashMap<String, Vec<String>>,
        per_host: HashMap<String, usize>,
        seed_hosts: HashSet<String>,
        handed_out: usize,
    ) -> Self {
        Self {
            queue,
            seen,
            anchors,
            per_host,
            seed_hosts,
            // A restored frontier owes nothing: whatever was in flight when it
            // was saved is back in the queue.
            in_flight: HashMap::new(),
            handed_out,
            limits,
        }
    }

    /// What is waiting to be fetched, in the order it will be handed out.
    pub(crate) fn pending(&self) -> impl Iterator<Item = &Pending> {
        self.queue.iter()
    }

    /// Every URL this crawl has decided it will not queue again.
    pub(crate) fn seen_urls(&self) -> impl Iterator<Item = &str> {
        self.seen.iter().map(String::as_str)
    }

    /// Anchor text still waiting for the page it describes.
    pub(crate) fn pending_anchors(&self) -> impl Iterator<Item = (&str, &[String])> {
        self.anchors
            .iter()
            .map(|(url, texts)| (url.as_str(), texts.as_slice()))
    }

    pub(crate) fn host_counts(&self) -> impl Iterator<Item = (&str, usize)> {
        self.per_host.iter().map(|(host, n)| (host.as_str(), *n))
    }

    pub(crate) fn seed_host_names(&self) -> impl Iterator<Item = &str> {
        self.seed_hosts.iter().map(String::as_str)
    }

    /// Marks a URL as accounted for: fetched and written somewhere durable, or
    /// definitively given up on.
    ///
    /// Until this is called the URL is treated as still owed, and a checkpoint
    /// will put it back in the queue. Calling it for a page that has been
    /// fetched but not yet spooled is how a resume grows a gap.
    pub fn completed(&mut self, url: &str) {
        self.in_flight.remove(url);
    }

    /// URLs handed out and not yet accounted for, with their depths.
    pub(crate) fn in_flight(&self) -> impl Iterator<Item = (&str, u32)> {
        self.in_flight
            .iter()
            .map(|(url, depth)| (url.as_str(), *depth))
    }

    #[must_use]
    pub fn handed_out(&self) -> usize {
        self.handed_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(s: &str) -> Url {
        Url::parse(s).expect("valid test url")
    }

    fn frontier() -> Frontier {
        Frontier::new(Limits::default())
    }

    #[test]
    fn a_seed_comes_back_out() {
        let mut f = frontier();
        assert!(f.seed(url("https://example.com/")));
        let next = f.next_url().expect("the seed");
        assert_eq!(next.url.as_str(), "https://example.com/");
        assert_eq!(next.depth, 0);
        assert!(f.next_url().is_none());
    }

    #[test]
    fn a_url_is_queued_once_however_many_pages_link_to_it() {
        let mut f = frontier();
        f.seed(url("https://example.com/"));
        assert!(f.enqueue(url("https://example.com/a"), 1, None));
        assert!(!f.enqueue(url("https://example.com/a"), 1, None));
        assert!(!f.enqueue(url("https://example.com/a"), 2, None));
        assert_eq!(f.queued(), 2);
    }

    #[test]
    fn anchor_text_accumulates_from_every_page_that_links() {
        let mut f = frontier();
        f.seed(url("https://example.com/"));
        let target = url("https://example.com/parasearch");
        f.enqueue(target.clone(), 1, Some("el buscador colombiano"));
        // Already seen, but the words still count.
        f.enqueue(target.clone(), 1, Some("motor de busqueda en perl"));
        f.enqueue(target.clone(), 1, Some("el buscador colombiano"));

        let anchors = f.take_anchors(&target);
        assert_eq!(anchors.len(), 2, "duplicates should collapse: {anchors:?}");
        assert!(anchors.iter().any(|a| a.contains("colombiano")));
        // Taking them empties the store.
        assert!(f.take_anchors(&target).is_empty());
    }

    #[test]
    fn empty_anchor_text_is_not_stored() {
        let mut f = frontier();
        let target = url("https://example.com/a");
        f.seed(url("https://example.com/"));
        f.enqueue(target.clone(), 1, Some("   "));
        assert!(f.take_anchors(&target).is_empty());
    }

    #[test]
    fn anchors_per_url_are_capped() {
        let mut f = frontier();
        let target = url("https://example.com/a");
        f.seed(url("https://example.com/"));
        for i in 0..100 {
            f.enqueue(target.clone(), 1, Some(&format!("anchor {i}")));
        }
        assert_eq!(f.take_anchors(&target).len(), 32);
    }

    #[test]
    fn depth_is_a_hard_limit() {
        let mut f = Frontier::new(Limits {
            max_depth: 1,
            ..Limits::default()
        });
        f.seed(url("https://example.com/"));
        assert!(f.enqueue(url("https://example.com/a"), 1, None));
        assert!(!f.enqueue(url("https://example.com/b"), 2, None));
    }

    #[test]
    fn other_hosts_are_refused_when_asked_to_stay_home() {
        let mut f = frontier();
        f.seed(url("https://example.com/"));
        assert!(!f.enqueue(url("https://other.org/a"), 1, None));

        let mut f = Frontier::new(Limits {
            same_host_only: false,
            ..Limits::default()
        });
        f.seed(url("https://example.com/"));
        assert!(f.enqueue(url("https://other.org/a"), 1, None));
    }

    #[test]
    fn the_page_budget_stops_the_crawl() {
        let mut f = Frontier::new(Limits {
            max_pages: 2,
            ..Limits::default()
        });
        f.seed(url("https://example.com/"));
        for i in 0..10 {
            f.enqueue(url(&format!("https://example.com/{i}")), 1, None);
        }
        assert!(f.next_url().is_some());
        assert!(f.next_url().is_some());
        assert!(
            f.next_url().is_none(),
            "budget of 2 handed out a third page"
        );
        assert_eq!(f.handed_out(), 2);
    }

    #[test]
    fn one_host_cannot_consume_the_whole_crawl() {
        let mut f = Frontier::new(Limits {
            max_pages_per_host: 2,
            same_host_only: false,
            ..Limits::default()
        });
        f.seed(url("https://a.com/"));
        for i in 0..5 {
            f.enqueue(url(&format!("https://a.com/{i}")), 1, None);
        }
        f.enqueue(url("https://b.com/x"), 1, None);

        let mut hosts = Vec::new();
        while let Some(p) = f.next_url() {
            hosts.push(host_of(&p.url));
        }
        assert_eq!(hosts.iter().filter(|h| *h == "a.com").count(), 2);
        assert_eq!(hosts.iter().filter(|h| *h == "b.com").count(), 1);
    }

    #[test]
    fn the_queue_is_breadth_first() {
        let mut f = frontier();
        f.seed(url("https://example.com/"));
        f.enqueue(url("https://example.com/shallow"), 1, None);
        f.enqueue(url("https://example.com/deep"), 2, None);
        let order: Vec<u32> = std::iter::from_fn(|| f.next_url())
            .map(|p| p.depth)
            .collect();
        assert_eq!(order, [0, 1, 2]);
    }
}
