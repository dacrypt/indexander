// Jump consistent hashing is specified in terms of f64 arithmetic over values
// that never approach the mantissa limit: `b` counts shards, not keys.
#![allow(clippy::cast_precision_loss)]

//! Deciding which shard owns a URL.
//!
//! Two properties matter, and both rule out `hash % n`:
//!
//! * **Every node computes the same answer without asking anyone.** Ownership
//!   is a pure function of the URL and the shard count.
//! * **Adding a shard moves as little as possible.** With `hash % n`, growing
//!   from 4 shards to 5 moves about 80% of all URLs, which means re-crawling
//!   or shipping most of the index. Jump consistent hashing moves exactly
//!   `1/n` — the minimum that any correct scheme can move.
//!
//! Jump consistent hashing needs no lookup table and no memory: it is a short
//! loop over a linear congruential generator, from Lamping and Veach (2014).

/// A cheap, well-mixing 64-bit hash. Not cryptographic, and does not need to
/// be: it decides placement, not access.
#[must_use]
pub fn hash_url(url: &str) -> u64 {
    // FNV-1a, then an avalanche step so that URLs sharing a long prefix — as
    // every URL on one site does — land in different shards.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

/// Jump consistent hash: maps `key` to one of `buckets`.
#[must_use]
pub fn jump_hash(mut key: u64, buckets: u32) -> u32 {
    if buckets == 0 {
        return 0;
    }
    let (mut b, mut j) = (-1i64, 0i64);
    while j < i64::from(buckets) {
        b = j;
        key = key.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(1);
        #[allow(clippy::cast_precision_loss)]
        let denominator = ((key >> 33) + 1) as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            j = ((b + 1) as f64 * (2_147_483_648.0 / denominator)) as i64;
        }
    }
    u32::try_from(b).unwrap_or(0)
}

/// The shard that owns `url` when there are `shard_count` of them.
#[must_use]
pub fn shard_for(url: &str, shard_count: u32) -> u32 {
    jump_hash(hash_url(url), shard_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_shard_owns_everything() {
        for url in ["http://a/", "http://b/", "https://c/d?e=f"] {
            assert_eq!(shard_for(url, 1), 0);
        }
    }

    #[test]
    fn placement_is_deterministic() {
        let url = "https://example.com/docs/page.html";
        let first = shard_for(url, 16);
        for _ in 0..100 {
            assert_eq!(shard_for(url, 16), first);
        }
    }

    #[test]
    fn every_shard_is_in_range() {
        for count in [1u32, 2, 3, 7, 64, 1024] {
            for i in 0..500 {
                let shard = shard_for(&format!("https://example.com/{i}"), count);
                assert!(shard < count, "shard {shard} out of range for {count}");
            }
        }
    }

    #[test]
    fn the_distribution_is_even() {
        let shards = 8u32;
        let n = 40_000;
        let mut counts = vec![0usize; shards as usize];
        for i in 0..n {
            counts[shard_for(&format!("https://site{}.example/page/{i}", i % 97), shards)
                as usize] += 1;
        }
        let expected = n / shards as usize;
        for (shard, count) in counts.iter().enumerate() {
            let drift = (*count as f64 - expected as f64).abs() / expected as f64;
            assert!(
                drift < 0.05,
                "shard {shard} got {count}, expected ~{expected}"
            );
        }
    }

    /// The property that rules out `hash % n`.
    #[test]
    fn growing_the_cluster_moves_only_one_nth() {
        let n = 20_000;
        let urls: Vec<String> = (0..n).map(|i| format!("https://example.com/{i}")).collect();

        let before: Vec<u32> = urls.iter().map(|u| shard_for(u, 4)).collect();
        let after: Vec<u32> = urls.iter().map(|u| shard_for(u, 5)).collect();
        let moved = before.iter().zip(&after).filter(|(a, b)| a != b).count();

        // Adding the fifth shard should move about 1/5 of the keys.
        let fraction = moved as f64 / f64::from(n);
        assert!(
            (fraction - 0.20).abs() < 0.02,
            "moving from 4 to 5 shards moved {:.1}% of urls",
            fraction * 100.0
        );

        // And everything that moved, moved *to* the new shard: nothing is
        // shuffled between existing shards.
        for (a, b) in before.iter().zip(&after) {
            assert!(a == b || *b == 4, "a url moved from shard {a} to {b}");
        }
    }

    #[test]
    fn urls_sharing_a_long_prefix_still_scatter() {
        // Every URL on a site shares its host; if the hash did not avalanche,
        // a whole site would land on one shard.
        let mut seen = std::collections::HashSet::new();
        for i in 0..200 {
            seen.insert(shard_for(
                &format!("https://www.example.com/very/deep/path/page{i}.html"),
                8,
            ));
        }
        assert_eq!(
            seen.len(),
            8,
            "a single site reached only {} shards",
            seen.len()
        );
    }

    #[test]
    fn zero_buckets_does_not_divide_by_zero() {
        assert_eq!(jump_hash(12345, 0), 0);
    }
}
