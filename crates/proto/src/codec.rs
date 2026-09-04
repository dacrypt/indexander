// Varints are defined in terms of truncation: each byte carries the low seven
// bits of what is left, and the `as u8` casts below are that definition.
#![allow(clippy::cast_possible_truncation)]

//! The bytes on the wire.
//!
//! Hand-rolled rather than pulled from a serialisation crate, for the same
//! reason the index format is: this is a protocol that has to stay stable
//! across versions of two programs, and a format nobody can read without a
//! derive macro is a format nobody audits.
//!
//! Everything is little-endian. Lengths and counts are LEB128 varints, so the
//! common small values cost one byte. Strings are a varint length followed by
//! UTF-8. Every read is bounds-checked: the far side of a socket is untrusted
//! input, even when it is a process you started.

use indexander_core::{Error, Result};

/// Refuse to allocate for a length that could only come from a corrupt or
/// hostile frame. A query with a million terms is not a query.
const MAX_COLLECTION: u64 = 1 << 22;
/// The longest string this protocol will accept: URLs and queries, not bodies.
const MAX_STRING: u64 = 1 << 20;

#[derive(Debug, Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u8(&mut self, value: u8) {
        self.buf.push(value);
    }

    pub fn varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.buf.push((value as u8) | 0x80);
            value >>= 7;
        }
        self.buf.push(value as u8);
    }

    pub fn usize(&mut self, value: usize) {
        self.varint(value as u64);
    }

    pub fn f32(&mut self, value: f32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Raw bytes, length written by the caller.
    pub fn bytes(&mut self, value: &[u8]) {
        self.buf.extend_from_slice(value);
    }

    pub fn str(&mut self, value: &str) {
        self.varint(value.len() as u64);
        self.buf.extend_from_slice(value.as_bytes());
    }

    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

#[derive(Debug)]
pub struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .cursor
            .checked_add(n)
            .filter(|e| *e <= self.bytes.len())
            .ok_or_else(|| Error::Corrupt("frame ends mid-value".into()))?;
        let slice = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(slice)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.u8()?;
            if shift >= 64 {
                return Err(Error::Corrupt("varint wider than u64".into()));
            }
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
    }

    pub fn usize(&mut self) -> Result<usize> {
        usize::try_from(self.varint()?).map_err(|_| Error::Corrupt("value exceeds usize".into()))
    }

    /// A length that is about to drive an allocation, so it is capped.
    pub fn count(&mut self) -> Result<usize> {
        let value = self.varint()?;
        if value > MAX_COLLECTION {
            return Err(Error::Corrupt(format!("collection of {value} refused")));
        }
        usize::try_from(value).map_err(|_| Error::Corrupt("count exceeds usize".into()))
    }

    pub fn f32(&mut self) -> Result<f32> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| Error::Corrupt("truncated f32".into()))?;
        Ok(f32::from_le_bytes(bytes))
    }

    /// Reads `len` raw bytes.
    pub fn bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        Ok(self.take(len)?.to_vec())
    }

    pub fn string(&mut self) -> Result<String> {
        let len = self.varint()?;
        if len > MAX_STRING {
            return Err(Error::Corrupt(format!("string of {len} bytes refused")));
        }
        let len = usize::try_from(len).map_err(|_| Error::Corrupt("string too long".into()))?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::Corrupt("string is not utf-8".into()))
    }

    /// Errors unless every byte of the frame has been consumed.
    ///
    /// A frame with bytes left over is a version mismatch or a bug, and
    /// ignoring the tail is how those go unnoticed for months.
    pub fn finish(self) -> Result<()> {
        if self.cursor == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::Corrupt(format!(
                "{} unread bytes at end of frame",
                self.bytes.len() - self.cursor
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_roundtrip() {
        let mut w = Writer::new();
        w.u8(7);
        w.varint(0);
        w.varint(u64::MAX);
        w.usize(300);
        w.f32(-1.5);
        w.str("hola ñandú");
        w.str("");
        let bytes = w.finish();

        let mut r = Reader::new(&bytes);
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.varint().unwrap(), 0);
        assert_eq!(r.varint().unwrap(), u64::MAX);
        assert_eq!(r.usize().unwrap(), 300);
        assert!((r.f32().unwrap() + 1.5).abs() < f32::EPSILON);
        assert_eq!(r.string().unwrap(), "hola ñandú");
        assert_eq!(r.string().unwrap(), "");
        r.finish().unwrap();
    }

    #[test]
    fn a_truncated_frame_errors_rather_than_panicking() {
        let mut w = Writer::new();
        w.str("una cadena razonablemente larga");
        let bytes = w.finish();
        for cut in 0..bytes.len() {
            let mut r = Reader::new(&bytes[..cut]);
            assert!(r.string().is_err(), "cut at {cut} did not error");
        }
    }

    #[test]
    fn leftover_bytes_are_an_error() {
        let mut w = Writer::new();
        w.u8(1);
        w.u8(2);
        let bytes = w.finish();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.u8().unwrap(), 1);
        assert!(r.finish().is_err(), "an unread byte should be caught");
    }

    #[test]
    fn a_hostile_length_is_refused_before_allocating() {
        // A varint claiming four billion elements must not become a Vec.
        let mut w = Writer::new();
        w.varint(u64::MAX);
        let bytes = w.finish();
        assert!(Reader::new(&bytes).count().is_err());
        assert!(Reader::new(&bytes).string().is_err());
    }

    #[test]
    fn invalid_utf8_is_rejected() {
        let mut bytes = vec![2u8];
        bytes.extend_from_slice(&[0xFF, 0xFE]);
        assert!(Reader::new(&bytes).string().is_err());
    }

    #[test]
    fn an_overlong_varint_is_rejected() {
        let bytes = [0x80u8; 12];
        assert!(Reader::new(&bytes).varint().is_err());
    }
}
