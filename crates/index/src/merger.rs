//! Doing the merges the policy asks for, one at a time, restartably.
//!
//! A merge reads several segments, writes one, and replaces the manifest. It
//! takes minutes on a real index, which means the interesting question is not
//! how it works but what happens when it stops halfway — a machine reboots, a
//! deploy kills the process, a disk fills.
//!
//! The answer here is that **the manifest is the only thing that decides what
//! an index is**. A merge writes its new segment first, under a name derived
//! from the segment's own digest, and only then replaces the manifest. Before
//! that moment the index is exactly what it was; after it, exactly what it
//! became. There is no in-between state to recover from, and a merge that died
//! leaves a file nobody references — wasted space, not a broken index.
//!
//! Naming segments after their digest also makes a retry free: a merge run
//! twice writes the same bytes to the same name.

use std::path::{Path, PathBuf};

use indexander_core::Result;

use crate::index::Index;
use crate::manifest::{Entry, Manifest, Policy};
use crate::segment::Segment;

/// What one merge did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeReport {
    /// The segments that were folded together.
    pub merged: Vec<String>,
    /// What they became.
    pub produced: String,
    pub documents: usize,
    pub bytes: u64,
    /// Files that could not be removed and are still taking up space.
    ///
    /// Deleting a segment another process has mapped fails on Windows and
    /// succeeds-but-keeps-the-inode on Unix. Either way the merge is done and
    /// the manifest no longer names the file, so this is reported rather than
    /// treated as a failure.
    pub undeleted: Vec<String>,
}

/// Runs merges in one index directory.
#[derive(Debug)]
pub struct Merger {
    directory: PathBuf,
    manifest_path: PathBuf,
    policy: Policy,
}

impl Merger {
    #[must_use]
    pub fn new(directory: &Path, policy: Policy) -> Self {
        Self {
            directory: directory.to_path_buf(),
            manifest_path: directory.join("MANIFEST"),
            policy,
        }
    }

    #[must_use]
    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    /// Does at most one merge, and returns what it did.
    ///
    /// One at a time on purpose: a plan naming several merges is stale the
    /// moment the first one finishes, and each one takes long enough that it
    /// will be.
    pub fn step(&self) -> Result<Option<MergeReport>> {
        let manifest = Manifest::open(&self.manifest_path)?;
        let Some(plan) = self.policy.next_merge(&manifest) else {
            return Ok(None);
        };

        // Only the segments being merged are opened. The rest of the index
        // could be hundreds of gigabytes and has nothing to do with this.
        let mut index = Index::new();
        let mut merged_names = Vec::with_capacity(plan.len());
        for i in &plan {
            let entry = &manifest.segments[*i];
            index.push(Segment::open(&self.directory.join(&entry.name))?);
            merged_names.push(entry.name.clone());
        }

        let bytes = index.merge()?;
        let produced = Segment::from_bytes(bytes.clone())?;
        // Named after its contents: a merge that is retried after a crash
        // writes the same bytes to the same name, and two nodes merging the
        // same segments agree on what to call the result.
        let name = format!("{:016x}.ixdr", produced.digest());
        let entry = Entry {
            name: name.clone(),
            digest: produced.digest(),
            documents: produced.document_count(),
            bytes: bytes.len() as u64,
        };
        write_atomically(&self.directory.join(&name), &bytes)?;

        // The new manifest takes the merged segment's place at the position of
        // the first one it replaces, so documents stay in the order they were
        // added.
        let mut segments: Vec<Entry> = Vec::with_capacity(manifest.segments.len());
        let first = *plan.first().unwrap_or(&0);
        for (i, existing) in manifest.segments.iter().enumerate() {
            if i == first {
                segments.push(entry.clone());
            }
            if !plan.contains(&i) {
                segments.push(existing.clone());
            }
        }
        // Nothing below this line can leave the index in a worse state than
        // above it: until this rename lands, the old manifest is still the
        // truth and the file just written is an orphan.
        Manifest { segments }.write_to(&self.manifest_path)?;

        // Only now are the old files removable. Any that resist are reported.
        drop(index);
        let mut undeleted = Vec::new();
        for old in &merged_names {
            if std::fs::remove_file(self.directory.join(old)).is_err() {
                undeleted.push(old.clone());
            }
        }

        Ok(Some(MergeReport {
            merged: merged_names,
            produced: name,
            documents: entry.documents,
            bytes: entry.bytes,
            undeleted,
        }))
    }

    /// Merges until the policy is satisfied.
    pub fn run_to_completion(&self) -> Result<Vec<MergeReport>> {
        let mut done = Vec::new();
        while let Some(report) = self.step()? {
            done.push(report);
        }
        Ok(done)
    }

    /// Files in the directory that the manifest does not name.
    ///
    /// Left by a merge that died after writing its segment, or by a delete
    /// that could not proceed. Listing them is separate from removing them,
    /// because "this file is unreferenced" and "it is safe to delete now" are
    /// different questions and only the operator can answer the second.
    pub fn orphans(&self) -> Result<Vec<String>> {
        let manifest = Manifest::open(&self.manifest_path)?;
        let named: std::collections::HashSet<&str> =
            manifest.segments.iter().map(|s| s.name.as_str()).collect();

        let mut orphans = Vec::new();
        for entry in std::fs::read_dir(&self.directory)? {
            let entry = entry?;
            let path = entry.path();
            // Compared as a path extension rather than a string suffix, so a
            // file called `SOMETHING.IXDR` on a case-insensitive filesystem is
            // still recognised as a segment rather than left behind forever.
            let is_segment = path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("ixdr"));
            let name = entry.file_name().to_string_lossy().into_owned();
            if is_segment && !named.contains(name.as_str()) {
                orphans.push(name);
            }
        }
        orphans.sort();
        Ok(orphans)
    }
}

/// Writes through a temporary file and a rename, as segments require.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write as _;
    let temporary = path.with_extension("ixdr.writing");
    let mut file = std::fs::File::create(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, path)?;
    Ok(())
}
