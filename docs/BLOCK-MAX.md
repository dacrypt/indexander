# Is block-max worth building?

Block-max stores an upper bound on each block's contribution to a score, so a
query can skip a whole block once it can prove nothing inside it reaches the
current top-k. It is the standard next step after skip lists.

It was also designed for **disjunctive** queries, where every posting of every
term is a candidate. This engine is **conjunctive**: a document must contain
every term, and the leapfrog already skips blocks that cannot contain the
pivot. So the question is not whether block-max is a good technique. It is:
*how many of the blocks this engine still decodes could a score bound remove?*

`cargo run --release -p indexander-index --example blockmax -- <segment> <query>...`
answers that against a real index. What follows is 103,257 documents, 702 MiB
of text, asking for the top 10.

## What it found

| query | postings decoded today | block-max would skip |
|---|---|---|
| `the` (29,815 documents) | 100% | **88%** |
| `index` (16,477) | 100% | **86%** |
| `control` (2,224) | 100% | **66%** |
| `kubernetes the` | 3.9% | **0%** |
| `index the control plane` | 11.6% | **0%** |

Two findings, and they point in opposite directions.

**For single-term queries it is worth a great deal.** They decode every posting
of the term today, because there is no second list to leapfrog against, and a
score bound would remove roughly two thirds to seven eighths of that. These are
also the *slowest* queries the engine has: `the` alone costs 570 µs, against
170 µs for a four-term query. Block-max targets exactly what is left slow.

**For multi-term queries it is worth nothing at all.** Not "a little" — zero
pivots, in every multi-term query measured. The leapfrog has already reduced
`index the control plane` from 49,008 postings to 5,696, and every one of the
77 documents that survives scores above the threshold. There is nothing left to
skip, because the skipping already happened.

## Two things the measurement got wrong first

Worth recording, because both would have produced a confident wrong answer.

**Counting the wrong blocks.** The first version counted how many of *all*
blocks a bound could skip, and reported 232 of 233 for a two-term query — which
looked like an overwhelming case for building it. But those were blocks the
leapfrog already skips. The question is only about blocks that are still being
decoded, and against that denominator the answer was zero.

**A bound that was too loose.** The second version bounded a block by its
largest term frequency and the shortest document in the corpus, which is
correct but generous to the point of uselessness — the shortest document in
this corpus is one token long. It reported 16% for `the`. A real block-max
index stores the maximum *score* per block, computable at index time when both
the frequency and each document's length are known. With that, the same query
reports 88%.

The difference between "16%, not worth it" and "88%, clearly worth it" was
entirely in how the bound was computed.

## The numbers are honest about their ceiling

The tool reports two figures. The **ceiling** assumes the query knows its final
top-k threshold from the start, which no implementation can. The **realistic**
figure starts the threshold at zero and raises it as documents are scored,
which is what the algorithm actually does — and is the figure in the table
above. For `the` the two are 97% and 88%; the gap is what a smarter traversal
order, scoring likely winners first, could recover.

## Verdict, and what happened when it was built

Worth building, for the case it addresses and no other.

It was built, in segment format v5: one `f32` per block, beside the skip index,
holding the largest contribution any posting in the block can make with the
`idf` factored out. What the prediction was worth:

| query | predicted skip | before | after | speedup |
|---|---|---|---|---|
| `the` | 88% | 570 µs | 135 µs | 4.2× |
| `index` | 86% | — | 83 µs | — |
| `kubernetes the` | 0% | 62 µs | 56 µs | none |
| `index the control plane` | 0% | 170 µs | 155 µs | none |

The prediction held on both sides: large for one term, nothing for several.
The index grew 2.1% rather than the half percent estimated here, because the
bound is one `f32` per block and blocks are 128 postings, not the 1,000 that
estimate assumed.

Two things the build itself turned up, neither predicted by the experiment:

- **The bound has to leave `idf` out.** Baking it in would have been correct
  for a single index and silently wrong for a shard, which is told to score
  with corpus-wide statistics instead of its own. A bound computed with the
  wrong `idf` does not fail — it drops results.
- **The bound has to use the same average document length the searcher uses.**
  The first version computed its own, slightly differently, which would have
  made some bounds too low and lost results in exactly the queries that would
  have found them.

Both are the same failure in different clothes: a bound derived from a
different formula than the one that scores is not a bound.
