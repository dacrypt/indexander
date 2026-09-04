//! Copying a segment to another node.
//!
//! This is the easy half of the distributed store, and it is easy for one
//! reason: segments never change. A file written once and never modified can
//! be copied, served read-only and replicated without a lock, a version or a
//! conversation about who wins. All the mutable state a replicated store needs
//! is the list of which segments make up a shard and where their copies live —
//! kilobytes, and off the data path entirely.
//!
//! What this module refuses to do is trust a copy. A transfer that says it
//! finished is not a replica; a transfer whose digest matches the source is.
//! The failure being defended against is not an attacker — anyone who can
//! rewrite a segment can rewrite its digest — but a truncated stream, a full
//! disk, a flipped bit, all of which produce a file that opens, parses, and
//! answers queries with silently missing documents.

use std::path::Path;

use indexander_core::{Error, Result};
use indexander_index::manifest::Manifest;
use indexander_index::segment::Segment;
use indexander_proto::message::{PROTOCOL_VERSION, Request, Response};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::frame::{read_frame, read_hello, write_frame, write_hello};

/// How much of a segment to ask for at a time.
///
/// Large enough that a 250 MB segment is a few thousand round trips rather
/// than a few million; small enough to stay well inside the frame limit and
/// not to hold a megabyte-and-a-half of it in memory per chunk.
const CHUNK: u32 = 1 << 20;

/// What a source node says about the segment it serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentInfo {
    pub digest: u64,
    pub len: u64,
}

/// Pulls the segment served at `address` and writes it to `destination`.
///
/// Returns what was fetched. The file is written to a temporary name and
/// renamed into place only after its digest matches, so a failed transfer
/// leaves no half-segment anybody could open — the same reason
/// `SegmentBuilder::write_to` renames rather than truncating.
pub async fn fetch_segment(address: &str, destination: &Path) -> Result<SegmentInfo> {
    fetch_named(address, "", destination).await
}

/// Pulls one named segment from `address` into `destination`.
///
/// An empty name means "the segment you serve", which is what a shard holding
/// a single one answers about itself.
pub async fn fetch_named(address: &str, name: &str, destination: &Path) -> Result<SegmentInfo> {
    let mut stream = TcpStream::connect(address).await?;
    let _ = stream.set_nodelay(true);
    read_hello(&mut stream, PROTOCOL_VERSION).await?;
    write_hello(&mut stream, PROTOCOL_VERSION).await?;

    let info = Request::SegmentInfo {
        name: name.to_owned(),
    };
    let Response::SegmentInfo { digest, len } = call(&mut stream, &info).await? else {
        return Err(Error::Corrupt(format!(
            "{address} did not answer what segment it serves"
        )));
    };

    let temporary = destination.with_extension("ixdr.fetching");
    let mut file = tokio::fs::File::create(&temporary).await?;
    let mut fetched = 0u64;

    while fetched < len {
        let want = u32::try_from(len - fetched).unwrap_or(CHUNK).min(CHUNK);
        let Response::SegmentChunk { bytes } = call(
            &mut stream,
            &Request::SegmentChunk {
                name: name.to_owned(),
                offset: fetched,
                len: want,
            },
        )
        .await?
        else {
            return Err(Error::Corrupt(format!("{address} stopped sending chunks")));
        };
        if bytes.is_empty() {
            // The source ran out before it said it would. Better to fail here
            // than to rename a short file into place.
            break;
        }
        fetched += bytes.len() as u64;
        file.write_all(&bytes).await?;
    }
    file.flush().await?;
    file.sync_all().await?;
    drop(file);

    if fetched != len {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(Error::Corrupt(format!(
            "{address} promised {len} bytes and sent {fetched}"
        )));
    }

    // Read back what was actually written, not what was believed to be sent,
    // and let the segment check itself.
    //
    // An earlier version recomputed the digest here over "everything but the
    // footer", with the footer's length written out as a constant. The footer
    // then grew by three fields and this went on subtracting the old number:
    // every transfer failed, which was the lucky direction. Knowing the format
    // in two places is the bug; `Segment::verify` knows it in one.
    let written = tokio::fs::read(&temporary).await?;
    let intact = Segment::from_bytes(written)
        .is_ok_and(|segment| segment.verify() && segment.digest() == digest);
    if !intact {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(Error::Corrupt(format!(
            "the copy from {address} does not match its digest"
        )));
    }

    tokio::fs::rename(&temporary, destination).await?;
    Ok(SegmentInfo { digest, len })
}

/// Brings `directory` into line with the index served at `address`.
///
/// Asks what segments the source has, fetches the ones this copy is missing,
/// verifies each against the digest the manifest recorded, and installs the
/// new manifest **last**. Until that rename lands, this replica is still
/// serving exactly what it was — the same rule a merge follows, for the same
/// reason.
///
/// Returns the names it fetched. Segments already present with the right
/// digest are not refetched, which is what makes syncing after a merge cost
/// the merged segment rather than the whole index.
pub async fn sync_from(address: &str, directory: &Path) -> Result<Vec<String>> {
    let mut stream = TcpStream::connect(address).await?;
    let _ = stream.set_nodelay(true);
    read_hello(&mut stream, PROTOCOL_VERSION).await?;
    write_hello(&mut stream, PROTOCOL_VERSION).await?;

    let Response::Manifest { text } = call(&mut stream, &Request::Manifest).await? else {
        return Err(Error::Corrupt(format!("{address} did not send a manifest")));
    };
    drop(stream);
    let wanted = Manifest::decode(&text)?;

    tokio::fs::create_dir_all(directory).await?;
    let mut fetched = Vec::new();
    for entry in &wanted.segments {
        let path = directory.join(&entry.name);
        // Already here and intact? Then it is the same segment: names come
        // from digests for merged segments, and the digest is checked anyway.
        if let Ok(existing) = tokio::fs::read(&path).await {
            if Segment::from_bytes(existing).is_ok_and(|s| s.digest() == entry.digest) {
                continue;
            }
        }
        let got = fetch_named(address, &entry.name, &path).await?;
        if got.digest != entry.digest {
            return Err(Error::Corrupt(format!(
                "{} arrived with a digest the manifest does not name",
                entry.name
            )));
        }
        fetched.push(entry.name.clone());
    }

    // Last, so a sync that dies partway leaves this replica serving what it
    // was serving, with some extra files nobody references yet.
    wanted.write_to(&directory.join("MANIFEST"))?;
    Ok(fetched)
}

async fn call(stream: &mut TcpStream, request: &Request) -> Result<Response> {
    write_frame(stream, &request.encode()).await?;
    let payload = read_frame(stream).await?;
    let response = Response::decode(&payload)?;
    if let Response::Error { message } = &response {
        return Err(Error::Corrupt(message.clone()));
    }
    Ok(response)
}
