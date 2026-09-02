//! Framing: turning a stream of bytes into a stream of messages.
//!
//! TCP has no message boundaries, so the protocol supplies its own: a
//! four-byte little-endian length, then that many bytes. The first frame of a
//! connection carries the protocol version, checked before anything is
//! interpreted, so that two incompatible builds fail with a clear message
//! rather than a corrupt decode.

use indexander_core::{Error, Result};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Refuse to allocate for a frame larger than this. The largest legitimate
/// message is a response full of hits, which is kilobytes.
const MAX_FRAME: u32 = 64 * 1024 * 1024;

/// Writes one length-prefixed frame.
pub async fn write_frame<W>(writer: &mut W, payload: &[u8]) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let len = u32::try_from(payload.len())
        .map_err(|_| Error::Corrupt("frame larger than 4 GiB".into()))?;
    if len > MAX_FRAME {
        return Err(Error::Corrupt(format!("refusing to send {len}-byte frame")));
    }
    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Reads one length-prefixed frame.
pub async fn read_frame<R>(reader: &mut R) -> Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut header = [0u8; 4];
    reader.read_exact(&mut header).await?;
    let len = u32::from_le_bytes(header);
    // Checked before allocating: the length comes from the far side.
    if len > MAX_FRAME {
        return Err(Error::Corrupt(format!("refusing {len}-byte frame")));
    }
    let mut payload = vec![0u8; len as usize];
    reader.read_exact(&mut payload).await?;
    Ok(payload)
}

/// Sends the version handshake.
pub async fn write_hello<W>(writer: &mut W, version: u32) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_frame(writer, &version.to_le_bytes()).await
}

/// Reads and checks the version handshake.
pub async fn read_hello<R>(reader: &mut R, expected: u32) -> Result<()>
where
    R: AsyncRead + Unpin,
{
    let payload = read_frame(reader).await?;
    let bytes: [u8; 4] = payload
        .as_slice()
        .try_into()
        .map_err(|_| Error::Corrupt("malformed handshake".into()))?;
    let their_version = u32::from_le_bytes(bytes);
    if their_version == expected {
        Ok(())
    } else {
        Err(Error::Corrupt(format!(
            "peer speaks protocol {their_version}, this build speaks {expected}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn frames_roundtrip_over_a_pipe() {
        let (mut client, mut server) = tokio::io::duplex(4096);
        let payloads: Vec<Vec<u8>> = vec![b"hola".to_vec(), Vec::new(), vec![7u8; 5000]];

        let sent = payloads.clone();
        tokio::spawn(async move {
            for p in &sent {
                write_frame(&mut client, p).await.unwrap();
            }
        });

        for expected in &payloads {
            assert_eq!(&read_frame(&mut server).await.unwrap(), expected);
        }
    }

    #[tokio::test]
    async fn a_matching_handshake_succeeds_and_a_mismatch_is_reported() {
        let (mut a, mut b) = tokio::io::duplex(64);
        tokio::spawn(async move { write_hello(&mut a, 1).await.unwrap() });
        assert!(read_hello(&mut b, 1).await.is_ok());

        let (mut a, mut b) = tokio::io::duplex(64);
        tokio::spawn(async move { write_hello(&mut a, 99).await.unwrap() });
        let err = read_hello(&mut b, 1).await.unwrap_err();
        assert!(format!("{err}").contains("protocol 99"), "got {err}");
    }

    #[tokio::test]
    async fn an_oversized_length_is_refused_without_allocating() {
        let (mut a, mut b) = tokio::io::duplex(64);
        tokio::spawn(async move {
            // Claim a 4 GiB frame, then send nothing.
            a.write_all(&u32::MAX.to_le_bytes()).await.unwrap();
        });
        assert!(read_frame(&mut b).await.is_err());
    }

    #[tokio::test]
    async fn a_closed_connection_is_an_error_not_a_hang() {
        let (a, mut b) = tokio::io::duplex(64);
        drop(a);
        assert!(read_frame(&mut b).await.is_err());
    }
}
