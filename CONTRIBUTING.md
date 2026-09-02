# Contributing

Patches are welcome. A few things worth knowing before you start.

## The bar

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --all-features
cargo test --workspace
```

CI runs exactly these with `-D warnings`, on Linux, macOS and Windows, plus a
build against the minimum supported Rust version in `rust-version`.

## What gets a test

Anything that reads or writes bytes. The segment format has a writer and a
reader that must agree, and unit tests on either side alone cannot prove that
they do — so format changes need a round-trip test in
`crates/index/tests/end_to_end.rs`.

Anything that once was wrong. `the_2004_bug_is_fixed` exists because the engine
this one descends from folded `ñ` to `c` for years without anyone noticing. A
bug that had a test would have been a bug that lasted an afternoon.

## Performance claims

If a change is meant to make something faster, say how it was measured and
against what. "Should be faster" is not a benchmark. Numbers in a commit
message should be reproducible by whoever reads it.

## Commit messages

Say what changed and why. The why is the part that is not in the diff.
