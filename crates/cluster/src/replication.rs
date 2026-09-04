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
use indexander_index::segment::{Segment, digest_of};
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
    let mut stream = TcpStream::connect(address).await?;
    let _ = stream.set_nodelay(true);
    read_hello(&mut stream, PROTOCOL_VERSION).await?;
    write_hello(&mut stream, PROTOCOL_VERSION).await?;

    let Response::SegmentInfo { digest, len } = call(&mut stream, &Request::SegmentInfo).await?
    else {
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

    // Read back what was actually written, not what was believed to be sent.
    let written = tokio::fs::read(&temporary).await?;
    if digest_of(&written[..written.len().saturating_sub(footer_len())]) != digest {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(Error::Corrupt(format!(
            "the copy from {address} does not match its digest"
        )));
    }

    tokio::fs::rename(&temporary, destination).await?;
    Ok(SegmentInfo { digest, len })
}

/// The footer is not part of what the digest covers, so it has to be excluded
/// when checking a copy. Derived from a segment rather than hardcoded here.
fn footer_len() -> usize {
    // Six u64 offsets, the u64 digest, a u32 version and a 4-byte magic.
    6 * 8 + 8 + 4 + 4
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

/// Answers the two transfer requests from a segment held in memory.
///
/// Kept beside the shard role rather than inside it, because serving bytes and
/// serving queries are different jobs that happen to share a socket.
#[must_use]
pub fn handle(segment_bytes: &[u8], segment: &Segment, request: &Request) -> Option<Response> {
    match request {
        Request::SegmentInfo => Some(Response::SegmentInfo {
            digest: segment.digest(),
            len: segment_bytes.len() as u64,
        }),
        Request::SegmentChunk { offset, len } => {
            let start = usize::try_from(*offset).unwrap_or(usize::MAX);
            if start >= segment_bytes.len() {
                // Past the end is an empty chunk, not an error: it is how a
                // reader learns it has everything.
                return Some(Response::SegmentChunk { bytes: Vec::new() });
            }
            let end = start.saturating_add(*len as usize).min(segment_bytes.len());
            Some(Response::SegmentChunk {
                bytes: segment_bytes[start..end].to_vec(),
            })
        }
        _ => None,
    }
}
