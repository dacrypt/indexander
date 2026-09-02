# indexander

A search engine written in Rust: crawler-ready inverted index, positional
postings, field-weighted BM25, and a command line that indexes a directory and
searches it in about a millisecond.

It is the successor to [parasearch](https://github.com/dacrypt/parasearch), a
search engine written in Perl in Colombia in 2004 that never shipped. That one
is an archive. This one is meant to work.

```console
$ indexander index ~/corpus
indexed 103257 documents, 1285312 terms in 51.52s
indexander.ixdr -> 222.6 MiB (31.7% of 702.2 MiB of text)

$ indexander search "inverted index" -perl --limit 5
  1.  11.8402  /corpus/design/postings.md
  2.   9.6631  /corpus/notes/lucene.md
  ...
5 results in 1.20ms over 103257 documents
```

## Status

**Working, and honest about its edges.** The index, the query language and the
ranking are real and tested. The crawler is not written yet — today the input is
a directory of files.

| | |
|---|---|
| Indexing | 702 MiB of text into a 222 MiB index in 51 s, single threaded |
| Index size | 27–36% of the text it indexes, across three real corpora |
| Query latency | 1.2–1.4 ms over 103,257 documents; 376 µs for a phrase |
| Tests | 47, including a full write-then-read round trip |
| `unsafe` | none |
| Dependencies | none outside the standard library |

Those numbers come from `indexander index` and `indexander search` on the
machine at hand, not from a model of what should be fast. Reproduce them with
any directory of text you have.

## Install

```sh
git clone https://github.com/dacrypt/indexander
cd indexander
cargo install --path crates/cli
```

Requires Rust 1.85 or newer.

## How it works

**Tokenizing.** Text splits on non-alphanumerics, lowercases, and folds
diacritics — `Años` and `anos` are the same term, in the index and in the query,
which is the only way both spellings can find each other.

**The index is inverted and positional.** For every term, the documents that
contain it; for every document, the positions where it appears, per field. That
is what makes `"motor de busqueda"` different from those three words scattered
across a page, and it is what the 2004 design got right and is worth keeping.

**Fields carry weight.** A document is a title, a body, and the text of the
links pointing at it. A term in the title scores 3×, in an anchor 2×, in the
body 1×. Anchor text is the idea that a page is often better described by how
others link to it than by what it says about itself — the `anchor_queue` of the
original design, still here.

**Storage is delta plus varint.** Postings are written in ascending document
order as gaps, and gaps as LEB128 varints, so a document id that follows its
predecessor costs one byte instead of four. That is most of why the index is a
third of the size of the corpus.

**Lookup is a binary search over a fixed-width offset table.** The term
dictionary is sorted on disk; a `u32` offset table beside it means finding a
term touches `log2(n)` cache lines and allocates nothing.

**Ranking is Okapi BM25** with `k1 = 1.2` and `b = 0.75`, over the
field-weighted term frequency.

**Scoring walks sorted lists together.** Candidates ascend, postings ascend, so
one cursor per term advances monotonically. The first version looked each
candidate up with a linear scan; replacing that took a query over 103k documents
from 37.31 ms to 1.43 ms.

## Query syntax

```text
motor de busqueda      every term must appear
"motor de busqueda"    adjacent, in order, within one field
-perl                  drop documents containing this term
```

Accents and case are folded the same way as at index time, so `Búsqueda`,
`BUSQUEDA` and `busqueda` are one query.

## Layout

```
crates/core    vocabulary: DocId, Document, Field, Error — no dependencies
crates/index   tokenizer, codec, segment format, query parser, BM25 search
crates/cli     the `indexander` binary
```

## What is not built yet

Stated plainly, because a README that only lists what works is a sales page:

- **The crawler.** The whole point of the name it inherits. Async, polite,
  `robots.txt`-aware, with a frontier and per-host rate limiting.
- **Segment merging.** One segment per index today. Real corpora need many,
  written incrementally and merged in the background.
- **Memory mapping.** Segments are read into memory. `mmap` avoids the copy but
  needs `unsafe`, and the copy is not yet what makes anything slow.
- **Concurrency.** Indexing is single threaded on a 14-core machine. The
  per-document work is independent; this is the largest easy win available.
- **Snippets.** Positions are stored, so highlighted extracts are a query away.
- **An HTTP API and a UI.** There is a CLI and a library.

## Lineage

parasearch measured itself in August 2004: 34,436 URLs crawled, 2.2 million
positional entries, 442 MB of MySQL, on a server with 512 MB of RAM. Then its
authors worked out that indexing the web at Google's scale would need 69 TB —
433 hard drives — and stopped.

One thing they did not find out for twenty-two years: their accent-folding table
had 68 source characters and 66 replacements. Perl pads silently with the last
one, so every letter from `ö` onward folded wrong and `ñ` became `c`. A
Colombian search engine indexed *español* as *espacol*, for years.

`crates/index/src/tokenizer.rs` has a test called `the_2004_bug_is_fixed`, and
another called `fold_is_total` that asserts the property the old table violated,
across the whole Latin-1 and Latin Extended-A range. That is what this project
inherits: not the code, the receipts.

## Licence

MIT or Apache-2.0, at your option. See [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE).
