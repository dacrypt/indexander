//! What segments make up an index, and when to fold them together.
//!
//! Segments are immutable, so adding documents means adding a segment. Left
//! alone that ends badly in one of two ways: merging after every write costs
//! quadratic total work, because every document is rewritten every time; never
//! merging leaves a query opening thousands of files and summing statistics
//! across all of them.
//!
//! The way out is to merge in *tiers*. A segment belongs to the tier of its
//! size, roughly `log(bytes)`, and a tier is folded into one segment once it
//! holds enough of them. A document is then rewritten about once per tier it
//! climbs — `log(n)` times over the life of an index rather than `n`.
//!
//! The manifest is the only mutable state in a replicated store, and it is
//! kilobytes: a list of file names and digests. It is written as text on
//! purpose. "Which segments make up this shard" is exactly the question
//! somebody asks at three in the morning, and answering it should not need
//! this program to be working.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

use indexander_core::{Error, Result};

/// One segment, as the manifest records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// File name, relative to the index directory.
    pub name: String,
    /// The digest in the segment's footer, so a manifest can be checked
    /// against the files it names.
    pub digest: u64,
    pub documents: usize,
    pub bytes: u64,
}

/// The segments making up one index, in the order their documents were added.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Manifest {
    pub segments: Vec<Entry>,
}

const HEADER: &str = "indexander-manifest 1";

impl Manifest {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn document_count(&self) -> usize {
        self.segments.iter().map(|s| s.documents).sum()
    }

    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.bytes).sum()
    }

    /// Renders the manifest.
    #[must_use]
    pub fn encode(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{HEADER}");
        // The count is what makes a truncated file detectable: a manifest that
        // lost its last lines would otherwise parse as a smaller, valid index
        // and quietly drop documents.
        let _ = writeln!(out, "segments {}", self.segments.len());
        for entry in &self.segments {
            let _ = writeln!(
                out,
                "segment {} {:016x} {} {}",
                entry.name, entry.digest, entry.documents, entry.bytes
            );
        }
        out
    }

    /// Parses a manifest, refusing anything it does not fully understand.
    pub fn decode(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        if lines.next() != Some(HEADER) {
            return Err(Error::Corrupt("not an indexander manifest".into()));
        }
        let declared = lines
            .next()
            .and_then(|l| l.strip_prefix("segments "))
            .and_then(|n| n.parse::<usize>().ok())
            .ok_or_else(|| Error::Corrupt("manifest has no segment count".into()))?;

        let mut segments = Vec::with_capacity(declared);
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            if parts.next() != Some("segment") {
                return Err(Error::Corrupt(format!("unknown manifest line: {line}")));
            }
            let mut next = || {
                parts
                    .next()
                    .ok_or_else(|| Error::Corrupt(format!("short manifest line: {line}")))
            };
            let name = next()?.to_owned();
            let digest = u64::from_str_radix(next()?, 16)
                .map_err(|_| Error::Corrupt(format!("bad digest in: {line}")))?;
            let documents = next()?
                .parse()
                .map_err(|_| Error::Corrupt(format!("bad document count in: {line}")))?;
            let bytes = next()?
                .parse()
                .map_err(|_| Error::Corrupt(format!("bad byte count in: {line}")))?;
            segments.push(Entry {
                name,
                digest,
                documents,
                bytes,
            });
        }

        if segments.len() != declared {
            return Err(Error::Corrupt(format!(
                "manifest declares {declared} segments and lists {}",
                segments.len()
            )));
        }
        Ok(Self { segments })
    }

    /// Writes the manifest, replacing any previous one atomically.
    ///
    /// Through a temporary file and a rename, for the reason segments are:
    /// a reader must see the old manifest or the new one, never a file being
    /// written. Here it matters more, because a half-read manifest names
    /// segments that exist and omits others that do too.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        let temporary = path.with_extension("manifest.tmp");
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(self.encode().as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)?;
        Ok(())
    }

    pub fn open(path: &Path) -> Result<Self> {
        Self::decode(&std::fs::read_to_string(path)?)
    }
}

/// When to merge.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    /// Segments within this factor of each other in size are one tier.
    ///
    /// Ten means a tier spans an order of magnitude, which is what keeps the
    /// number of tiers — and so the number of times a document is rewritten —
    /// logarithmic in the size of the index.
    pub tier_factor: u64,
    /// How many segments a tier holds before it is folded into one.
    ///
    /// Lower merges more often, keeping queries fast and writing more; higher
    /// does the reverse. This is the only real knob.
    pub segments_per_tier: usize,
    /// Never merge past this, so one enormous segment is not rewritten to
    /// absorb a small one.
    pub max_merged_bytes: u64,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            tier_factor: 10,
            segments_per_tier: 8,
            max_merged_bytes: 4 << 30,
        }
    }
}

impl Policy {
    /// Which tier a segment of `bytes` belongs to.
    #[must_use]
    pub fn tier_of(&self, bytes: u64) -> u32 {
        let factor = self.tier_factor.max(2);
        let mut tier = 0u32;
        let mut ceiling = factor;
        // A tiny segment and an empty one are the same tier; there is no
        // reason to distinguish them and `ilog` would have to handle zero.
        while bytes >= ceiling && tier < 63 {
            ceiling = ceiling.saturating_mul(factor);
            tier += 1;
        }
        tier
    }

    /// The next merge to do, as indices into `manifest.segments`, or `None`
    /// when nothing needs merging.
    ///
    /// One merge at a time, deliberately. A plan naming several merges is a
    /// plan that is stale after the first one finishes, and merging is slow
    /// enough that it will be.
    #[must_use]
    pub fn next_merge(&self, manifest: &Manifest) -> Option<Vec<usize>> {
        let mut by_tier: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, entry) in manifest.segments.iter().enumerate() {
            if entry.bytes >= self.max_merged_bytes {
                continue;
            }
            by_tier
                .entry(self.tier_of(entry.bytes))
                .or_default()
                .push(i);
        }

        // Smallest tier first: merging the small ones is cheap and removes the
        // most segments per byte rewritten, which is the whole point.
        for (_, mut group) in by_tier {
            if group.len() < self.segments_per_tier.max(2) {
                continue;
            }
            group.truncate(self.segments_per_tier.max(2));
            // Would the result exceed the ceiling? Then merge fewer.
            let mut total = 0u64;
            let mut taken = Vec::new();
            for i in group {
                let next = total.saturating_add(manifest.segments[i].bytes);
                if !taken.is_empty() && next > self.max_merged_bytes {
                    break;
                }
                total = next;
                taken.push(i);
            }
            if taken.len() >= 2 {
                return Some(taken);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, bytes: u64) -> Entry {
        Entry {
            name: name.to_owned(),
            digest: 0xdead_beef,
            documents: (bytes / 100).max(1) as usize,
            bytes,
        }
    }

    #[test]
    fn a_manifest_roundtrips() {
        let manifest = Manifest {
            segments: vec![entry("a.ixdr", 1024), entry("b.ixdr", 2_000_000)],
        };
        assert_eq!(Manifest::decode(&manifest.encode()).unwrap(), manifest);
    }

    #[test]
    fn an_empty_manifest_roundtrips() {
        let empty = Manifest::new();
        assert_eq!(Manifest::decode(&empty.encode()).unwrap(), empty);
        assert_eq!(empty.document_count(), 0);
    }

    #[test]
    fn a_truncated_manifest_is_refused_rather_than_read_short() {
        // The failure this is for: a manifest that lost its last lines parses
        // as a smaller, valid index and drops documents without a word.
        let manifest = Manifest {
            segments: vec![entry("a.ixdr", 1), entry("b.ixdr", 2), entry("c.ixdr", 3)],
        };
        let text = manifest.encode();
        let truncated: String = text.lines().take(4).collect::<Vec<_>>().join("\n");
        let error = Manifest::decode(&truncated).unwrap_err();
        assert!(format!("{error}").contains("declares 3"), "got {error}");
    }

    #[test]
    fn garbage_is_refused() {
        assert!(Manifest::decode("").is_err());
        assert!(Manifest::decode("hello").is_err());
        assert!(Manifest::decode("indexander-manifest 1\n").is_err());
        assert!(Manifest::decode("indexander-manifest 1\nsegments 1\nnonsense a b c").is_err());
        assert!(
            Manifest::decode("indexander-manifest 1\nsegments 1\nsegment a zz 1 1").is_err(),
            "a digest that is not hex should be refused"
        );
    }

    #[test]
    fn tiers_span_an_order_of_magnitude() {
        let policy = Policy::default();
        assert_eq!(policy.tier_of(0), 0);
        assert_eq!(policy.tier_of(9), 0);
        assert_eq!(policy.tier_of(10), 1);
        assert_eq!(policy.tier_of(99), 1);
        assert_eq!(policy.tier_of(100), 2);
        assert!(policy.tier_of(u64::MAX) < 64, "tier must not run away");
    }

    #[test]
    fn nothing_to_merge_when_a_tier_is_not_full() {
        let policy = Policy::default();
        let manifest = Manifest {
            segments: (0..7).map(|i| entry(&format!("{i}.ixdr"), 100)).collect(),
        };
        assert!(policy.next_merge(&manifest).is_none());
    }

    #[test]
    fn a_full_tier_is_merged() {
        let policy = Policy::default();
        let manifest = Manifest {
            segments: (0..8).map(|i| entry(&format!("{i}.ixdr"), 100)).collect(),
        };
        let plan = policy.next_merge(&manifest).expect("a merge");
        assert_eq!(plan.len(), 8);
    }

    #[test]
    fn segments_of_different_sizes_are_not_merged_together() {
        // Folding one small segment into a huge one rewrites the huge one to
        // save one file. Tiering exists to stop exactly that.
        let policy = Policy::default();
        let mut segments: Vec<Entry> = (0..7)
            .map(|i| entry(&format!("small{i}.ixdr"), 100))
            .collect();
        segments.push(entry("huge.ixdr", 10_000_000));
        let manifest = Manifest { segments };
        assert!(
            policy.next_merge(&manifest).is_none(),
            "a small tier of seven should not be padded out with a huge segment"
        );
    }

    #[test]
    fn the_smallest_full_tier_goes_first() {
        let policy = Policy {
            segments_per_tier: 2,
            ..Policy::default()
        };
        let manifest = Manifest {
            segments: vec![
                entry("big0.ixdr", 100_000),
                entry("big1.ixdr", 100_000),
                entry("small0.ixdr", 10),
                entry("small1.ixdr", 10),
            ],
        };
        let plan = policy.next_merge(&manifest).expect("a merge");
        assert_eq!(plan, vec![2, 3], "the cheap merge should be chosen first");
    }

    #[test]
    fn a_segment_past_the_ceiling_is_left_alone() {
        let policy = Policy {
            max_merged_bytes: 1000,
            segments_per_tier: 2,
            ..Policy::default()
        };
        let manifest = Manifest {
            segments: vec![entry("a.ixdr", 5000), entry("b.ixdr", 5000)],
        };
        assert!(policy.next_merge(&manifest).is_none());
    }
}
