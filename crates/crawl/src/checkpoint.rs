//! Making a crawl survivable.
//!
//! A crawl that stops after four hours and starts over is not a crawl, it is a
//! way to annoy other people's servers. What has to survive is both halves of
//! the state, and they have to agree:
//!
//! * **The pages already fetched.** Not the index built from them — the index
//!   cannot be written incrementally, because every document's authority comes
//!   from `PageRank` over the *whole* link graph and that graph is not finished
//!   until the crawl is. So the checkpoint holds the pages, and the index is
//!   built at the end exactly as it is today.
//! * **The frontier**: what is queued, what has been seen, and the anchor text
//!   waiting for pages nobody has fetched yet.
//!
//! ## Why they must agree, and how
//!
//! The two are written separately and a crash can land between them. Written
//! frontier-first, a crash loses pages the frontier already calls seen — gaps,
//! silent, permanent. Written pages-first, a crash re-fetches them — but they
//! are already in the spool, so the index gets each of them twice.
//!
//! Neither is acceptable, so the frontier records *how many spooled pages it
//! accounts for*, and a resume reads exactly that many and ignores the rest.
//! Records written after the last frontier save are dropped and their URLs
//! come back through the queue. A crash costs a re-fetch of the last few pages
//! and nothing else.
//!
//! ## The format
//!
//! The spool is append-only, one length-prefixed record per page, each with a
//! checksum. Append-only is what makes it safe to write during a crawl: a
//! process killed mid-write corrupts at most the record it was writing, and
//! the checksum finds it. Nothing rewrites what is already there.
//!
//! The frontier file is rewritten whole and moved into place, because it has
//! no meaningful partial state — half a frontier is not a smaller frontier.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use indexander_core::{Document, Error, Result};
use url::Url;

use crate::frontier::{Frontier, Limits, Pending};

const SPOOL: &str = "SPOOL";
const FRONTIER: &str = "FRONTIER";
const FRONTIER_HEADER: &str = "indexander frontier 1";

/// FNV-1a, to notice a record that was half written.
///
/// Not a cryptographic hash and not trying to be: the only adversary here is a
/// process that died holding the pen.
fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    let bytes = value.as_bytes();
    out.extend_from_slice(&u32::try_from(bytes.len()).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn put_list(out: &mut Vec<u8>, values: &[String]) {
    out.extend_from_slice(
        &u32::try_from(values.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for value in values {
        put_str(out, value);
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn u32(&mut self) -> Result<usize> {
        let raw: [u8; 4] = self
            .bytes
            .get(self.at..self.at + 4)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| Error::Corrupt("checkpoint record ends early".into()))?;
        self.at += 4;
        Ok(u32::from_le_bytes(raw) as usize)
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u32()?;
        let raw = self
            .bytes
            .get(self.at..self.at + len)
            .ok_or_else(|| Error::Corrupt("checkpoint string runs past the record".into()))?;
        self.at += len;
        // Lossy on purpose: a page whose text arrived mangled is still a page,
        // and refusing to resume a four-hour crawl over one bad byte is worse
        // than indexing a replacement character.
        Ok(String::from_utf8_lossy(raw).into_owned())
    }

    fn list(&mut self) -> Result<Vec<String>> {
        let count = self.u32()?;
        (0..count).map(|_| self.string()).collect()
    }
}

fn encode(doc: &Document) -> Vec<u8> {
    let mut out = Vec::new();
    put_str(&mut out, &doc.uri);
    put_str(&mut out, &doc.title);
    put_str(&mut out, &doc.body);
    put_list(&mut out, &doc.anchors);
    put_list(&mut out, &doc.links);
    out
}

fn decode(bytes: &[u8]) -> Result<Document> {
    let mut c = Cursor { bytes, at: 0 };
    let uri = c.string()?;
    let title = c.string()?;
    let body = c.string()?;
    let anchors = c.list()?;
    let links = c.list()?;
    Ok(Document {
        uri,
        title,
        body,
        anchors,
        links,
    })
}

/// The pages fetched so far, on disk.
#[derive(Debug)]
pub struct Spool {
    file: BufWriter<File>,
    written: usize,
}

impl Spool {
    /// Opens the spool for appending, creating the directory if needed.
    ///
    /// # Errors
    ///
    /// If the directory cannot be made or the file cannot be opened.
    pub fn open(directory: &Path) -> Result<Self> {
        std::fs::create_dir_all(directory)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join(SPOOL))?;
        Ok(Self {
            file: BufWriter::new(file),
            written: 0,
        })
    }

    /// Appends one page.
    ///
    /// # Errors
    ///
    /// If the write fails.
    pub fn append(&mut self, doc: &Document) -> Result<()> {
        let payload = encode(doc);
        let len = u32::try_from(payload.len()).unwrap_or(u32::MAX);
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.write_all(&checksum(&payload).to_le_bytes())?;
        self.written += 1;
        Ok(())
    }

    /// Pushes everything buffered out to the file.
    ///
    /// Called before the frontier is saved, and in that order: the frontier
    /// promises that this many records exist, so they had better exist.
    ///
    /// # Errors
    ///
    /// If the flush fails.
    pub fn flush(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }

    #[must_use]
    pub fn appended(&self) -> usize {
        self.written
    }
}

/// Reads back the first `limit` pages of a spool.
///
/// Stops at the first record that is short or fails its checksum, which is
/// what a process killed mid-append leaves behind. Reading fewer than `limit`
/// records is a corrupt checkpoint and says so, because the frontier is about
/// to claim those pages were fetched.
///
/// # Errors
///
/// If the file cannot be read, or holds fewer sound records than promised.
pub fn read_spool(directory: &Path, limit: usize) -> Result<Vec<Document>> {
    let path = directory.join(SPOOL);
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut file = BufReader::new(File::open(&path)?);
    let mut docs = Vec::with_capacity(limit);
    let mut header = [0u8; 4];
    let mut digest = [0u8; 8];

    while docs.len() < limit {
        if file.read_exact(&mut header).is_err() {
            break;
        }
        let len = u32::from_le_bytes(header) as usize;
        let mut payload = vec![0u8; len];
        if file.read_exact(&mut payload).is_err() || file.read_exact(&mut digest).is_err() {
            break;
        }
        if checksum(&payload) != u64::from_le_bytes(digest) {
            break;
        }
        docs.push(decode(&payload)?);
    }

    if docs.len() < limit {
        return Err(Error::Corrupt(format!(
            "the frontier accounts for {limit} pages but the spool holds {}",
            docs.len()
        )));
    }
    Ok(docs)
}

/// Everything a crawl needs to pick up where it stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub queue: Vec<(String, u32)>,
    pub seen: Vec<String>,
    pub anchors: Vec<(String, Vec<String>)>,
    pub per_host: Vec<(String, usize)>,
    pub seed_hosts: Vec<String>,
    pub handed_out: usize,
    /// How many spooled pages this frontier accounts for. The number that
    /// keeps the two files consistent; see the note at the top of this module.
    pub spooled: usize,
}

impl Checkpoint {
    /// A line-based text format, so an interrupted crawl can be inspected with
    /// `cat` rather than a hex editor. URLs cannot contain a raw newline once
    /// parsed, which is what makes one-per-line safe.
    #[must_use]
    pub fn encode(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(out, "{FRONTIER_HEADER}");
        let _ = writeln!(out, "spooled {}", self.spooled);
        let _ = writeln!(out, "handed-out {}", self.handed_out);
        for host in &self.seed_hosts {
            let _ = writeln!(out, "seed-host {host}");
        }
        for (host, count) in &self.per_host {
            let _ = writeln!(out, "host {count} {host}");
        }
        for (url, depth) in &self.queue {
            let _ = writeln!(out, "queued {depth} {url}");
        }
        for url in &self.seen {
            let _ = writeln!(out, "seen {url}");
        }
        for (url, texts) in &self.anchors {
            for text in texts {
                // Anchor text is arbitrary and may contain anything, so it is
                // the last field and newlines in it are flattened.
                let _ = writeln!(out, "anchor {url} {}", text.replace(['\n', '\r'], " "));
            }
        }
        out
    }

    /// # Errors
    ///
    /// If the header is missing or a line cannot be understood. A frontier
    /// that half-parses would resume a crawl with the wrong idea of what it
    /// has already seen, so nothing is skipped quietly.
    pub fn decode(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        if lines.next().map(str::trim) != Some(FRONTIER_HEADER) {
            return Err(Error::Corrupt("not an indexander frontier".into()));
        }
        let mut out = Self {
            queue: Vec::new(),
            seen: Vec::new(),
            anchors: Vec::new(),
            per_host: Vec::new(),
            seed_hosts: Vec::new(),
            handed_out: 0,
            spooled: 0,
        };
        let mut anchors: HashMap<String, Vec<String>> = HashMap::new();
        let mut order: Vec<String> = Vec::new();

        for (n, line) in lines.enumerate() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let bad = |what: &str| Error::Corrupt(format!("frontier line {}: {what}", n + 2));
            let (tag, rest) = line.split_once(' ').ok_or_else(|| bad("no value"))?;
            match tag {
                "spooled" => out.spooled = rest.parse().map_err(|_| bad("not a number"))?,
                "handed-out" => out.handed_out = rest.parse().map_err(|_| bad("not a number"))?,
                "seed-host" => out.seed_hosts.push(rest.to_owned()),
                "host" => {
                    let (count, host) = rest.split_once(' ').ok_or_else(|| bad("no host"))?;
                    out.per_host.push((
                        host.to_owned(),
                        count.parse().map_err(|_| bad("not a number"))?,
                    ));
                }
                "queued" => {
                    let (depth, url) = rest.split_once(' ').ok_or_else(|| bad("no url"))?;
                    out.queue.push((
                        url.to_owned(),
                        depth.parse().map_err(|_| bad("not a depth"))?,
                    ));
                }
                "seen" => out.seen.push(rest.to_owned()),
                "anchor" => {
                    let (url, text) = rest.split_once(' ').ok_or_else(|| bad("no anchor text"))?;
                    if !anchors.contains_key(url) {
                        order.push(url.to_owned());
                    }
                    anchors
                        .entry(url.to_owned())
                        .or_default()
                        .push(text.to_owned());
                }
                other => return Err(bad(&format!("unknown field {other:?}"))),
            }
        }
        out.anchors = order
            .into_iter()
            .map(|url| {
                let texts = anchors.remove(&url).unwrap_or_default();
                (url, texts)
            })
            .collect();
        Ok(out)
    }

    /// Writes the frontier, atomically.
    ///
    /// Through a temporary file and a rename, because a frontier truncated
    /// halfway is not a smaller frontier — it is one that has forgotten what
    /// it has seen and will crawl those pages again.
    ///
    /// # Errors
    ///
    /// If the file cannot be written or moved into place.
    pub fn write_to(&self, directory: &Path) -> Result<()> {
        let final_path = directory.join(FRONTIER);
        let temporary = directory.join(format!("{FRONTIER}.writing"));
        std::fs::write(&temporary, self.encode())?;
        std::fs::rename(&temporary, &final_path)?;
        Ok(())
    }

    /// Reads a frontier, or `None` if there is none to read.
    ///
    /// # Errors
    ///
    /// If the file exists but cannot be read or understood.
    pub fn open(directory: &Path) -> Result<Option<Self>> {
        let path: PathBuf = directory.join(FRONTIER);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(Self::decode(&std::fs::read_to_string(&path)?)?))
    }
}

impl Frontier {
    /// Everything about this frontier that has to outlive the process.
    #[must_use]
    pub fn checkpoint(&self, spooled: usize) -> Checkpoint {
        Checkpoint {
            // The queue, plus everything handed out and not yet accounted
            // for. Those left the queue when a worker took them and can never
            // be queued again, because they are already in `seen`; without
            // this they are simply lost.
            queue: self
                .pending()
                .map(|p| (p.url.to_string(), p.depth))
                .chain(self.in_flight().map(|(url, depth)| (url.to_owned(), depth)))
                .collect(),
            seen: self.seen_urls().map(ToOwned::to_owned).collect(),
            anchors: self
                .pending_anchors()
                .map(|(url, texts)| (url.to_owned(), texts.to_vec()))
                .collect(),
            per_host: self
                .host_counts()
                .map(|(host, n)| (host.to_owned(), n))
                .collect(),
            seed_hosts: self.seed_host_names().map(ToOwned::to_owned).collect(),
            handed_out: self.handed_out(),
            spooled,
        }
    }

    /// Rebuilds a frontier from a checkpoint.
    ///
    /// # Errors
    ///
    /// If a queued URL will not parse. That means the file was edited or
    /// corrupted, and continuing would silently drop part of the crawl.
    pub fn restore(limits: Limits, saved: &Checkpoint) -> Result<Self> {
        let mut queue = VecDeque::with_capacity(saved.queue.len());
        for (url, depth) in &saved.queue {
            let parsed = Url::parse(url)
                .map_err(|e| Error::Corrupt(format!("queued url {url:?} will not parse: {e}")))?;
            queue.push_back(Pending {
                url: parsed,
                depth: *depth,
            });
        }
        Ok(Self::from_parts(
            limits,
            queue,
            saved.seen.iter().cloned().collect::<HashSet<_>>(),
            saved.anchors.iter().cloned().collect::<HashMap<_, _>>(),
            saved.per_host.iter().cloned().collect::<HashMap<_, _>>(),
            saved.seed_hosts.iter().cloned().collect::<HashSet<_>>(),
            saved.handed_out,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("indexander-ckpt-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    fn page(n: usize) -> Document {
        Document::new(
            format!("https://example.com/{n}"),
            format!("Página {n}"),
            format!("Un cuerpo con acentos, ñandú y un salto\nde línea, número {n}."),
        )
        .with_anchor("el buscador colombiano")
    }

    #[test]
    fn a_page_survives_the_round_trip_intact() {
        let dir = scratch("roundtrip");
        let mut spool = Spool::open(&dir).expect("open");
        let mut written = Vec::new();
        for n in 0..5 {
            let mut doc = page(n);
            doc.links.push(format!("https://example.com/{}", n + 1));
            spool.append(&doc).expect("append");
            written.push(doc);
        }
        spool.flush().expect("flush");

        let read = read_spool(&dir, 5).expect("read");
        assert_eq!(read, written);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_record_cut_off_mid_write_is_not_read_back() {
        // What a process killed while appending leaves behind. The point is
        // that it is detected, not that it is recovered.
        let dir = scratch("torn");
        let mut spool = Spool::open(&dir).expect("open");
        for n in 0..3 {
            spool.append(&page(n)).expect("append");
        }
        spool.flush().expect("flush");
        drop(spool);

        let path = dir.join(SPOOL);
        let whole = std::fs::read(&path).expect("read");
        // Chop the file in the middle of the last record.
        std::fs::write(&path, &whole[..whole.len() - 12]).expect("truncate");

        assert_eq!(read_spool(&dir, 2).expect("two are sound").len(), 2);
        let err = read_spool(&dir, 3).expect_err("the third is torn");
        assert!(err.to_string().contains("spool holds 2"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_flipped_byte_fails_its_checksum() {
        let dir = scratch("flipped");
        let mut spool = Spool::open(&dir).expect("open");
        spool.append(&page(0)).expect("append");
        spool.append(&page(1)).expect("append");
        spool.flush().expect("flush");
        drop(spool);

        let path = dir.join(SPOOL);
        let mut whole = std::fs::read(&path).expect("read");
        // Somewhere inside the first record's payload.
        whole[10] ^= 0xff;
        std::fs::write(&path, &whole).expect("write");

        let err = read_spool(&dir, 2).expect_err("the first record is damaged");
        assert!(err.to_string().contains("spool holds 0"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reading_stops_at_what_the_frontier_accounts_for() {
        // The rule that keeps the two files consistent: pages appended after
        // the last frontier save are ignored, and their URLs come back through
        // the queue instead of being indexed twice.
        let dir = scratch("limit");
        let mut spool = Spool::open(&dir).expect("open");
        for n in 0..10 {
            spool.append(&page(n)).expect("append");
        }
        spool.flush().expect("flush");

        let read = read_spool(&dir, 4).expect("read");
        assert_eq!(read.len(), 4);
        assert_eq!(read[3].uri, "https://example.com/3");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_frontier_survives_the_round_trip() {
        let mut frontier = Frontier::new(Limits::default());
        frontier.seed(Url::parse("https://example.com/").expect("url"));
        frontier.enqueue(
            Url::parse("https://example.com/a").expect("url"),
            1,
            Some("el buscador"),
        );
        frontier.enqueue(
            Url::parse("https://example.com/b").expect("url"),
            2,
            Some("motor de búsqueda"),
        );
        if let Some(p) = frontier.next_url() {
            frontier.completed(p.url.as_str());
        }

        let saved = frontier.checkpoint(7);
        let text = saved.encode();
        let parsed = Checkpoint::decode(&text).expect("decode");
        assert_eq!(parsed.spooled, 7);
        assert_eq!(parsed.handed_out, saved.handed_out);

        let restored = Frontier::restore(Limits::default(), &parsed).expect("restore");
        assert_eq!(restored.queued(), frontier.queued());
        assert_eq!(restored.seen_count(), frontier.seen_count());
        assert_eq!(restored.handed_out(), frontier.handed_out());
    }

    #[test]
    fn a_restored_frontier_does_not_hand_out_a_seen_url_again() {
        // The property the whole thing exists for.
        let mut frontier = Frontier::new(Limits::default());
        frontier.seed(Url::parse("https://example.com/").expect("url"));
        frontier.enqueue(Url::parse("https://example.com/a").expect("url"), 1, None);
        while let Some(p) = frontier.next_url() {
            frontier.completed(p.url.as_str());
        }

        let mut restored =
            Frontier::restore(Limits::default(), &frontier.checkpoint(2)).expect("restore");
        assert!(restored.next_url().is_none(), "the queue should be empty");
        assert!(
            !restored.enqueue(Url::parse("https://example.com/a").expect("url"), 1, None),
            "a url fetched before the restart was queued again"
        );
    }

    /// The gap that cost a page the first time this was run against a real
    /// site, and the reason `completed` exists.
    ///
    /// A URL leaves the queue when a worker takes it and is already in `seen`,
    /// so a crash between those two moments loses it from both: not waiting,
    /// and never to be queued again. The page simply never appears in the
    /// index and nothing says so.
    #[test]
    fn a_page_handed_out_but_never_written_comes_back() {
        let mut frontier = Frontier::new(Limits::default());
        frontier.seed(Url::parse("https://example.com/").expect("url"));
        frontier.enqueue(Url::parse("https://example.com/a").expect("url"), 1, None);
        frontier.enqueue(Url::parse("https://example.com/b").expect("url"), 1, None);

        // Three handed out; only the first was written before the crash.
        let first = frontier.next_url().expect("one");
        frontier.completed(first.url.as_str());
        let second = frontier.next_url().expect("two");
        let third = frontier.next_url().expect("three");

        let saved = frontier.checkpoint(1);
        let mut restored = Frontier::restore(Limits::default(), &saved).expect("restore");

        let mut recovered: Vec<String> = Vec::new();
        while let Some(p) = restored.next_url() {
            recovered.push(p.url.to_string());
        }
        recovered.sort();
        let mut expected = vec![second.url.to_string(), third.url.to_string()];
        expected.sort();
        assert_eq!(
            recovered, expected,
            "the pages in flight when it stopped were lost"
        );
        assert!(
            !recovered.contains(&first.url.to_string()),
            "a page already written was fetched again"
        );
    }

    #[test]
    fn anchor_text_waiting_for_an_unfetched_page_survives() {
        let mut frontier = Frontier::new(Limits::default());
        frontier.seed(Url::parse("https://example.com/").expect("url"));
        let target = Url::parse("https://example.com/target").expect("url");
        frontier.enqueue(target.clone(), 1, Some("el buscador colombiano"));
        frontier.enqueue(target.clone(), 1, Some("indexander"));

        let mut restored =
            Frontier::restore(Limits::default(), &frontier.checkpoint(0)).expect("restore");
        let mut anchors = restored.take_anchors(&target);
        anchors.sort();
        assert_eq!(anchors, ["el buscador colombiano", "indexander"]);
    }

    #[test]
    fn a_frontier_file_is_written_atomically_and_read_back() {
        let dir = scratch("atomic");
        let mut frontier = Frontier::new(Limits::default());
        frontier.seed(Url::parse("https://example.com/").expect("url"));
        frontier.checkpoint(3).write_to(&dir).expect("write");

        let loaded = Checkpoint::open(&dir).expect("open").expect("present");
        assert_eq!(loaded.spooled, 3);
        assert!(
            !dir.join("FRONTIER.writing").exists(),
            "the temporary file was left behind"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_frontier_is_not_an_error() {
        let dir = scratch("absent");
        assert!(Checkpoint::open(&dir).expect("open").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_frontier_that_is_not_one_is_refused() {
        assert!(Checkpoint::decode("some other file\n").is_err());
        let err = Checkpoint::decode(&format!("{FRONTIER_HEADER}\nnonsense 1 2\n")).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
        let err = Checkpoint::decode(&format!("{FRONTIER_HEADER}\nspooled lots\n")).unwrap_err();
        assert!(err.to_string().contains("not a number"), "{err}");
    }
}
