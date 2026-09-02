// Varints are defined in terms of truncation: each byte carries the low seven
// bits of what is left. The `as u8` casts below are the encoding, not an
// accident of it.
#![allow(clippy::cast_possible_truncation)]

//! Byte-level encoding for postings.
//!
//! Two ideas do most of the compression work, and both are older than the
//! engine that first inspired this one:
//!
//! * **Delta encoding.** Postings are stored in ascending document order, so
//!   writing the gap to the previous entry instead of the absolute value keeps
//!   the numbers small.
//! * **Varints (LEB128).** Small numbers then cost one byte instead of four.
//!
//! Together they are why an inverted index is a fraction of the size of the
//! corpus it indexes.

use indexander_core::{Error, Result};

/// Appends `value` to `out` as a LEB128 varint.
pub fn write_varint(value: u64, out: &mut Vec<u8>) {
    let mut v = value;
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Reads a varint from `bytes` at `*cursor`, advancing it past what it read.
pub fn read_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    loop {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| Error::Corrupt("varint runs past end of buffer".into()))?;
        *cursor += 1;
        // 10 groups of 7 bits is the most a u64 can need; more means garbage.
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

/// Writes an ascending sequence as deltas. Panics in debug if not ascending.
pub fn write_deltas(values: &[u32], out: &mut Vec<u8>) {
    let mut previous = 0u32;
    for &v in values {
        debug_assert!(v >= previous, "delta encoding needs ascending input");
        write_varint(u64::from(v - previous), out);
        previous = v;
    }
}

/// Reads `count` delta-encoded values back into absolute form.
pub fn read_deltas(
    bytes: &[u8],
    cursor: &mut usize,
    count: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    out.clear();
    out.reserve(count);
    let mut running = 0u64;
    for _ in 0..count {
        running += read_varint(bytes, cursor)?;
        let value = u32::try_from(running)
            .map_err(|_| Error::Corrupt("delta sequence exceeds u32".into()))?;
        out.push(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrips_across_byte_boundaries() {
        // Values either side of every 7-bit boundary, plus the extremes.
        let mut cases = vec![0, 1, 127, 128, 129, 16_383, 16_384, u64::MAX];
        for shift in 0..64 {
            cases.push(1u64 << shift);
        }
        for value in cases {
            let mut buf = Vec::new();
            write_varint(value, &mut buf);
            let mut cursor = 0;
            assert_eq!(
                read_varint(&buf, &mut cursor).unwrap(),
                value,
                "value {value}"
            );
            assert_eq!(cursor, buf.len(), "cursor left mid-varint for {value}");
        }
    }

    #[test]
    fn small_numbers_cost_one_byte() {
        let mut buf = Vec::new();
        write_varint(127, &mut buf);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn deltas_roundtrip() {
        let docs = [0u32, 1, 5, 5, 900, 1_000_000];
        let mut buf = Vec::new();
        write_deltas(&docs, &mut buf);
        let mut cursor = 0;
        let mut back = Vec::new();
        read_deltas(&buf, &mut cursor, docs.len(), &mut back).unwrap();
        assert_eq!(back, docs);
    }

    #[test]
    fn deltas_are_smaller_than_absolutes_for_dense_runs() {
        let dense: Vec<u32> = (100_000..100_500).collect();
        let mut delta_encoded = Vec::new();
        write_deltas(&dense, &mut delta_encoded);
        let mut absolute = Vec::new();
        for &v in &dense {
            write_varint(u64::from(v), &mut absolute);
        }
        assert!(
            delta_encoded.len() * 2 < absolute.len(),
            "deltas {} vs absolutes {}",
            delta_encoded.len(),
            absolute.len()
        );
    }

    #[test]
    fn truncated_varint_is_an_error_not_a_panic() {
        let buf = [0x80u8, 0x80];
        let mut cursor = 0;
        assert!(read_varint(&buf, &mut cursor).is_err());
    }

    #[test]
    fn overlong_varint_is_rejected() {
        let buf = [0x80u8; 12];
        let mut cursor = 0;
        assert!(read_varint(&buf, &mut cursor).is_err());
    }
}
