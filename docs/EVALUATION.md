# Are the results any good?

Every other test in this repository checks that two paths agree. A merged
index scores like the segments it came from. A cluster returns what a single
index would. A rebuild is byte-identical to the last one. Those are worth
having and they are all satisfied by a ranker that puts the worst document
first, as long as it does so reproducibly.

Nothing measured whether the order was *good*. This is that.

## Two ways to get an answer to compare against

**Judged collections.** `indexander eval` reads TREC-format topics and qrels —
the format every published test collection already speaks — and reports P@k,
nDCG@k, MAP, MRR and success@k. It has no opinion about relevance; somebody
else decided, and this does the arithmetic.

**Known-item queries.** `indexander known-item` needs no judgements at all. It
lifts a span of words out of a document and asks where that document lands.
There is exactly one right answer, nobody decided it, and it cannot be argued
with.

The second is what the numbers below come from, because the first requires a
collection this repository does not ship and should not.

### Why a known-item query measures anything

Every term of the span is required, and the document it came from contains all
of them, so that document is in the result set by construction. The question is
only where it ranks among the *other* documents that also contain all of those
words. When the span is distinctive it comes first and the query measures
nothing. When it is ordinary — `the value of the first argument` — hundreds
qualify and only ranking separates them. The mean over many spans is dominated
by the middle of that range, which is the part worth measuring.

Spans are picked uniformly at random and **no attempt is made to prefer
distinctive ones**. Preferring them is exactly where a benchmark gets tuned
without anybody deciding to tune it.

## The corpus

The Rust crate sources cached on the machine that ran this: 8,147 files,
136.4 MiB of text, 7,661 of them indexed — the rest skipped as binary. Not a
research collection, but real prose and real code, and available to anyone with
a Rust toolchain.

```console
$ indexander index ~/.cargo/registry/src/index.crates.io-*
indexed 7661 documents, 338085 terms in 1.44s on 14 threads
```

## What it scores

Five seeds, 400 documents sampled each, six-word spans:

| | MRR | success@1 | success@10 |
|---|---|---|---|
| six words from the body | **0.557** | 0.434 | 0.787 |
| the whole title | 0.345 | 0.223 | 0.577 |

Seed to seed, MRR moves by about ±0.02. Any claim below smaller than that is
noise unless it is paired against the same seeds.

More words, more evidence, better ranking — which is the first thing a
measurement like this has to do before it is worth trusting:

| span | 2 | 4 | 6 | 10 | 20 |
|---|---|---|---|---|---|
| MRR | 0.237 | 0.449 | 0.592 | 0.728 | 0.800 |

### Titles score worse, and it is not the ranker's fault

A title is weighted three times a body, so titles ranking *worse* looks wrong
until you look at the corpus. `indexander index` takes a document's title from
its filename, and this corpus has 3,848 distinct filenames across 8,147 files:
`mod` appears 531 times, `lib` 140, `README` 132. Querying `mod` puts five
hundred documents in an exact tie on the only field that distinguishes them.

That number measures the corpus, not the engine. It is here because leaving it
out would have been the more interesting-looking choice.

## Does it actually catch a regression?

A metric that never moves when the ranker gets worse is worse than no metric,
because it certifies. So: break the scorer on purpose, on the same seed, and
see.

| ranker | MRR | success@1 |
|---|---|---|
| as shipped | 0.5920 | 0.4658 |
| no length normalisation (`B = 0`) | 0.3734 | 0.2397 |
| no saturation (`K1 = 1000`) | 0.4884 | 0.3493 |
| no idf | 0.5752 | 0.4384 |
| no field weights | 0.5910 | 0.4658 |

The first two are unambiguous: a third and a fifth of the ranking, gone.

The last two are **inside the seed-to-seed variation**, so that table cannot
tell them from noise. Running them paired instead — same five seeds, same
queries, one thing changed — can:

| ablation | MRR lost, per seed | mean |
|---|---|---|
| no idf | 0.017, 0.023, 0.006, 0.009, 0.014 | 0.014 |
| no field weights | 0.002, 0.000, 0.002, 0.001, 0.002 | 0.002 |

Five disjoint samples, same sign every time, for both. So idf is worth about
0.014 MRR here and field weights about 0.002 — real, consistent, and small
enough that this measurement can establish the direction and not the size.

That field weights barely register is expected rather than alarming: a query
lifted from a body is answered from the body, and weighting titles higher
cannot help a document win a contest its title never entered. It is a limit of
the *measurement*, and it is the reason `--from title` exists at all.


## When the question has no single right answer

A known-item query assumes the document its words came from is the only
document that could be the answer. Point it at a directory holding forty git
worktrees of one repository and that assumption is false forty times over: the
answer is one of forty byte-identical copies, ranking among them is arbitrary,
and MRR lands near `H(40)/40` — about 0.11.

That is a small number that looks exactly like a bad ranker. It was measured
here before it was understood, which is the only reason the check below exists.

`known-item` now counts how many answers scored *exactly* the same as other
documents, and says so:

```console
131 of them had the answer scoring exactly like 44.8 other documents on average, up to 99
the corpus contains duplicates: these numbers describe how many copies of a
document exist, not how well the engine ranks. Deduplicate it, or do not read
the figures above.
```

The comparison is exact rather than approximate, which would be wrong almost
anywhere else involving floats and is right here: copies produce bit-identical
scores, and any margin of error would start folding merely similar documents
into the same group. Near-duplicates that differ by a word will not be caught,
and should not be — they are different documents.

On the crate sources the warning stays quiet, but the count still appears: 13
of 194 answers tie, which are the identical `LICENSE` files and empty `mod.rs`
stubs. A corpus with none at all would be the surprising one.

## Tuning: what the parameters are actually worth

BM25 has two knobs. `K1` controls how quickly repeated occurrences of a term
stop adding score; `B` controls how much a long document is penalised for
being long. The shipped values are the textbook ones, 1.2 and 0.75.

Twenty combinations, five seeds each, paired:

**`K1` does not matter.** At `B = 1.0`, the four values 0.9, 1.2, 1.5 and 2.0
give MRR 0.6432, 0.6438, 0.6438 and 0.6432 — a spread of 0.0006, thirty times
smaller than the seed-to-seed noise.

**`B` is the whole game.** At `K1 = 1.2`:

| B | 0.30 | 0.50 | 0.75 | 0.90 | **1.00** | 1.10 | 1.25 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|---|---|---|
| MRR | 0.447 | 0.490 | 0.557 | 0.601 | **0.644** | 0.621 | 0.598 | 0.574 | 0.472 |

Full length normalisation beats the textbook 0.75 by 0.087 MRR here, four
times the noise.

That result needed one more test before it could be believed. `B` penalises
long documents, and known-item queries are easier to answer when the target is
short — so a number that simply kept climbing with `B` would be measuring
document length, not ranking. It does not: past 1.0, where BM25's
normalisation is already total, MRR falls monotonically. The peak is real and
it sits exactly at the boundary of the range that means anything.

### Does it generalise?

| corpus, query shape | B = 0.75 | B = 1.00 |
|---|---|---|
| crate sources, 6-word spans | 0.5566 | **0.6438** |
| crate sources, 3-word spans | 0.3347 | **0.4102** |
| crate sources, titles | 0.3445 | **0.3780** |
| rustdoc HTML, 6-word spans † | 0.1514 | 0.1573 |
| rustdoc HTML, 3-word spans † | 0.0877 | 0.0956 |
| 41 worktrees of one repository † | 0.0674 | 0.0728 |

† flagged by the duplicate check above, so read only the direction.

Every measurement points the same way and none reverses. But five of the six
come from corpora the harness itself says are compromised, and the sixth is one
collection of one genre — source files, whose lengths span three orders of
magnitude, which is exactly the situation where heavy length normalisation
should win.

**So the default stays at 0.75.** Not because the evidence is weak — on the one
clean corpus it is strong — but because "the best `B` for this corpus" is a
different claim from "the best `B`", and only the second one justifies changing
what everyone gets. What would change it: the same result on a judged
collection, where the thing being measured is relevance rather than
findability.

What the finding does justify is making the knob reachable, and it now is:

```bash
indexander index ~/corpus --b 1.0
```

The parameters are written into the segment rather than compiled into the
binary, because block-max bounds are computed with them when the segment is
written. A reader scoring by a different formula would be trusting bounds that
do not bound it, and that failure has no symptom — the query returns a ranking,
with documents missing from it. So a searcher uses the segment's parameters and
never its own defaults, and everywhere two segments could be mixed refuses when
they disagree: `from_segments` will not merge them, an `Index` holding both will
not search, and a coordinator whose shards do not match returns an error instead
of a ranking assembled from two formulas.

Segments written before this are version 7 and will not open. There is no
migration: the parameters that produced their bounds were not recorded, so
there is nothing to migrate them *to* — reindex.

Measuring the knob through the knob, with no recompile between these two lines:

```console
$ indexander index ~/.cargo/registry/src/index.crates.io-* --out b075.ixdr
$ indexander index ~/.cargo/registry/src/index.crates.io-* --out b100.ixdr --b 1.0
$ indexander known-item ... --index b075.ixdr    MRR 0.5397  0.5724  0.5624
$ indexander known-item ... --index b100.ixdr    MRR 0.6278  0.6407  0.6744
```

## What these numbers are not

**Comparable to anyone else's.** Different corpus, different queries, different
definition of a document. An MRR of 0.557 is a baseline to regress against, not
a position in a league table.

**A statement about relevance.** Known-item measures whether the engine can
find a document from its own words. A person searching wants documents that are
*about* something, which is a harder question and needs judgements.

**Free of the unjudged-is-irrelevant assumption**, in the `eval` path. A
document nobody judged scores as garbage, so a system that surfaces good
documents the judgement pool never saw is punished for it. `eval` prints how
many results in a run were unjudged, because that count is the one that says
how much to believe the rest.

## Running it

```bash
indexander index ~/corpus --out corpus.ixdr
indexander known-item ~/corpus --index corpus.ixdr --sample 400 --span 6 --seed 1
```

```bash
indexander eval --queries topics.tsv --qrels qrels.txt --index corpus.ixdr --k 10
```

Topics are `id`, whitespace, query. Qrels are TREC's four columns: `topic`,
`iteration`, `document`, `relevance`. Blank lines and `#` comments are skipped
in both — a judgement file should say who judged it and when, given how much
every number above depends on that.
