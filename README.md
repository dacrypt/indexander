# indexander

A search engine written in Rust: a polite asynchronous crawler, a positional
inverted index, PageRank over the link graph, field-weighted BM25, and a
command line that crawls a site and searches it in about a millisecond.

It is the successor to [parasearch](https://github.com/dacrypt/parasearch), a
search engine written in Perl in Colombia in 2004 that never shipped. That one
is an archive. This one is meant to work.

```console
$ indexander crawl https://example.com --pages 200 --depth 2
crawling 1 seed as indexander/0.1.0 (+https://github.com/dacrypt/indexander)
  25 pages...
fetched 198, indexed 191, 48210 terms in 1m 42s

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

**Working, and honest about its edges.** The crawler, the index, the query
language and the ranking are real and tested end to end.

| | |
|---|---|
| Indexing | 702 MiB of text into a 222 MiB index in 9.5–15.2 s on 14 threads, from 52.9–61.4 s on one |
| Index size | 32–36% of the text it indexes, across three real corpora |
| Query latency | 24 µs to 1.5 ms over 103,257 documents, by term frequency; 350 µs for a phrase |
| Ranking | BM25 for relevance, PageRank for authority, combined multiplicatively |
| Tests | 187, including full crawls and eight-shard queries over real sockets |
| `unsafe` | none |
| Dependencies | `core` and `index` have none outside `std`; the crawler needs `tokio`, `reqwest` and `url` |

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

## The crawler

`indexander crawl <url>` walks a site and indexes it as it goes — pages are
handed to the indexer as they arrive, so memory tracks the index, not the
network.

What "polite" means here is written down rather than implied:

- **`robots.txt` is fetched once per host and obeyed.** The most specific
  matching `User-agent` group wins and *replaces* the wildcard group rather than
  adding to it; the longest matching path rule wins, and `Allow` breaks a tie.
  `Disallow:` and `Disallow: /` are opposites, which is the single most common
  way to get this wrong, so it has its own test.
- **A host that cannot answer is not crawled.** If `robots.txt` fails with a
  server error or a dead connection, that host is skipped entirely. Being unable
  to ask is not permission.
- **Requests to one host are spaced out**, and the host's own `Crawl-delay`
  overrides ours when it asks for more. Concurrency is *across* hosts.
- **Everything is bounded**: body size, redirect count, request timeout, crawl
  depth, total pages, and pages per host.
- **The crawler says who it is**, with a URL a site owner can visit.

`rel="nofollow"`, `<meta name="robots">` and `<base href>` are all honoured.

### Anchor text is the point

When the home page links to `/motor.html` with the words *"el buscador
colombiano de los noventa"*, those words describe the target, not the source.
The frontier holds them until that page is fetched, then attaches them to it:

```console
$ indexander search colombiano
  1.   0.5909  http://localhost/motor.html
  2.   0.4832  http://localhost/
```

`motor.html` does not contain the word "colombiano" anywhere. It ranks first for
it anyway, above the page that *does* contain it, because anchor text is
weighted 2× and body text 1×. That is the `anchor_queue` of the 2004 design
doing exactly what it was designed to do.

## PageRank

The crawler already sees every link, so it carries them out with each page and
the whole crawl becomes a graph. PageRank runs over it before the segment is
written, and each document's score is stored beside its length.

```console
$ indexander crawl http://localhost:8732/
link graph: 7 nodes, 10 edges; pagerank converged in 24 iterations

most linked-to pages:
  0.46578  http://localhost:8732/autoridad.html
  0.09125  http://localhost:8732/hoja1.html
  ...
```

Six of those pages have **byte-identical text**. Only their position in the
graph differs:

```console
$ indexander search "motor de busqueda distribuido"
  1.   2.3576  http://localhost:8732/autoridad.html
  2.   1.6530  http://localhost:8732/hoja1.html
  3.   1.6530  http://localhost:8732/hoja2.html
```

Three things about the implementation are worth stating, because they are where
PageRank is usually got wrong:

- **Dangling nodes do not drain the graph.** A page with no outlinks is a sink;
  its mass has nowhere to flow. Unless it is put back into circulation every
  iteration, the whole vector decays toward zero and the ranking becomes
  plausible-looking noise. `ranks_sum_to_one` is the test that would catch it.
- **It iterates until it stops moving**, not a fixed number of times, and it
  reports whether it converged and after how many iterations.
- **A repeated link is one vote.** A navigation menu appearing on every page
  would otherwise decide the ranking of the whole site.

### Authority scales relevance; it never creates it

```rust
score = bm25 * (1.0 + 0.5 * ln_1p(rank * document_count))
```

The multiplication is the point. A page that does not match the query scores
zero, and zero times any amount of authority is still zero — there is a test
called `authority_cannot_make_an_irrelevant_page_match` that says so. The
logarithm is the other half: real web graphs span many orders of magnitude of
rank, and without it authority would simply overwrite relevance.

## Running it as a cluster

```console
$ indexander shard --listen 127.0.0.1:7801 --index shard0.ixdr
shard listening on 127.0.0.1:7801: 200 documents, 410 terms

$ indexander search "indices invertidos" --shards 127.0.0.1:7801,127.0.0.1:7802
  1.   0.0025  /corpus/a/doc0.txt
  ...
3 results in 491.29µs across 2 shards (400 documents, connect 1.41ms)
```

Both rounds fan out concurrently, and so does connecting, so a round costs the
slowest shard rather than the sum of all of them: four shards holding 8,000
documents answer in 0.79–1.05 ms. Replies are merged in shard order whatever
order they arrive in, so a repeated query returns exactly the same ranking —
there is a test that runs one twenty times to check.

This is step 1 of [`docs/DISTRIBUTION.md`](docs/DISTRIBUTION.md): the process
is split into roles that talk over a socket, so every call that will one day
cross a network already does. One shard and fifty take the same code path.

### A query is two rounds, and the first one is not an optimisation

BM25 weights a term by how rare it is, and rarity is a property of the corpus.
On one shard of several, the local counts are the wrong ones: a term rare in
shard 3 and common in shard 7 is scored as rare in one and common in the other,
so merging their top-k compares numbers that were never on the same scale. The
merge does not fail — it quietly returns a plausible, wrong ranking.

So round one asks every shard for its document count and its frequency for each
query term, the coordinator sums them, and round two runs the search with those
global numbers substituted. `crates/index/tests/sharding.rs` has both halves as
tests: `local_statistics_produce_a_different_ranking_than_one_index` proves the
bug is real, and `global_statistics_make_shards_agree_with_one_index` proves the
fix works. `crates/cluster/tests/two_shards.rs` does the same over real sockets.

### Routing

Shard ownership is jump consistent hashing over the hash of the whole
normalised URL. Not `hash % n`: growing from four shards to five moves 80% of
the corpus with modulo, and exactly one fifth with jump hashing — there is a
test that measures it, and another that checks everything which moved went *to*
the new shard rather than being shuffled between the old ones.

Hashing the whole URL rather than the host is deliberate. By host, a large site
lands entirely on one shard, and the web's host-size distribution is such that
one shard ends up holding a hundred times what another does. The cost is that a
site's pages scatter, which matters for crawl politeness — and that is solved
with per-host fetch leases, described in `docs/DISTRIBUTION.md`, not by
sharding differently.

### A missing shard fails the query

Results computed from four shards out of five are not slightly worse results;
they are results that silently omit a fifth of the corpus. The coordinator
refuses to connect rather than degrade.

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

**Positions are decoded only when something needs them.** Positions are the
bulk of an index — a common term has one document gap and dozens of positions
behind it — and only a phrase query ever reads them. Skipping them for every
other query, and storing each position block's byte length so skipping is one
addition rather than a walk over every varint in it, is where the query time
went:

| query | before | after |
|---|---|---|
| a rare term | 525 µs | **24 µs** |
| a term in most documents | 3.38 ms | **0.99 ms** |
| four terms, one of them common | 4.77 ms | **1.45 ms** |

The block lengths cost 6.2% more index — 222.6 MiB became 236.4 MiB — for
between 3.4× and 17.5× on the queries that do not need what they skip. Phrase
queries pay the same as before, because they read the positions anyway.

**Storage is delta plus varint.** Postings are written in ascending document
order as gaps, and gaps as LEB128 varints, so a document id that follows its
predecessor costs one byte instead of four. That is most of why the index is a
third of the size of the corpus.

**Lookup is a binary search over a fixed-width offset table.** The term
dictionary is sorted on disk; a `u32` offset table beside it means finding a
term touches `log2(n)` cache lines and allocates nothing.

**Ranking is Okapi BM25** with `k1 = 1.2` and `b = 0.75`, over the
field-weighted term frequency.

**Indexing runs on every core.** Tokenising a document depends on nothing but
that document, so the corpus splits across threads and the partial indexes are
stitched back together. The stitching is itself a parallel tree merge rather
than a chain, because with fourteen threads the serial tail is most of what is
left to save: chaining gave 4.1×, the tree gives about 5×.

The output is **byte-identical to the single-threaded build** — same MD5 — which
is the only verification of a parallel rewrite worth having, and
`chunked_building_is_byte_identical_to_one_pass` asserts it for every chunk
count from 1 to one-document-per-chunk.

**Scores are reproducible bit for bit.** Floating-point addition is not
associative, so summing a document's per-term contributions in `HashMap` order
made the same document score `9.72467` in one process and `9.724671` in
another — two shards holding identical data would have disagreed, and no
ranking comparison would have been trustworthy. Terms are now summed in sorted
order. `scores_are_bit_for_bit_reproducible` exists because this bug was real.

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
crates/crawl   robots.txt, HTML extraction, URL normalisation, frontier, fetch
crates/rank    the link graph as CSR, and PageRank over it
crates/proto   the coordinator/shard wire protocol and shard routing
crates/cluster the coordinator and shard roles, over TCP
crates/cli     the `indexander` binary
```

`core` and `index` deliberately depend on nothing but the standard library. The
crawler is where the dependencies live, because writing an HTTP client and a
TLS stack from scratch would be a different project.

## What is not built yet

Stated plainly, because a README that only lists what works is a sales page:

- **Distribution beyond step 1.** The roles, the protocol, the routing and the
  two-round query exist. What does not: fetch leases so a multi-node crawl is
  polite, distributed PageRank with boundary exchange, and replication. Steps 3
  to 5 of `docs/DISTRIBUTION.md`.
- **Segment merging.** One segment per index today. Real corpora need many,
  written incrementally and merged in the background.
- **Memory mapping.** Segments are read into memory: a 236 MiB index is 236 MiB
  of RSS and a read before a shard can answer anything. `mmap` avoids both, and
  is now the largest remaining win.
- **Skipping within a postings list.** A query still walks every posting of
  every term. Block-max indexes let a scorer prove a whole block cannot reach
  the current top-k and jump it.
- **Concurrency.** Indexing is single threaded on a 14-core machine. The
  per-document work is independent; this is the largest easy win available.
- **Snippets.** Positions are stored, so highlighted extracts are a query away.
- **An HTTP API and a UI.** There is a CLI and a library.
- **A real HTML parser.** `crates/crawl/src/extract.rs` is a scanner, not a
  parser: it walks the bytes once and never builds a tree. It handles the
  malformed markup it has been shown, and every case has a test, but a
  `html5ever`-grade tokenizer would handle more.
- **Crawl state that survives a restart.** The frontier is in memory, so a
  crawl that stops starts over.

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
