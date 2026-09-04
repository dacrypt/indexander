# Distribution

The 2004 design was distributed from its first line. `parasearch.conf` listed
the servers, `get_location_server` decided which one owned a URL, and
`add_to_queue` posted work to whoever that was:

```
SERVERS = "[texto_plano, dev.conexcol.com, dev1.conexcol.com]"
```

This engine is not distributed yet. This document says what it would take, in
the order the pieces have to arrive, so the shape is decided before code is
written against the single-process assumption.

## What already anticipates it

- **`DocId` is dense and per-segment**, not global. A shard can number its own
  documents from zero without coordinating with anyone.
- **Segments are immutable.** Once written they can be copied, served
  read-only, and replicated without locking.
- **The query executor takes a `&Segment`.** Nothing in it assumes there is
  only one.
- **PageRank is separate from indexing**, and reads a graph rather than an
  index.

## 1. Sharding: who owns a URL

Ownership must be computable by any node without asking, and stable as nodes
come and go.

```
shard = jump_consistent_hash(hash(normalised_url), shard_count)
```

Hashing the **whole normalised URL**, not the host, is deliberate. Hashing by
host puts all of a large site on one node, and the web's host size distribution
is such that one shard would hold a hundred times what another does. The cost
is that a site's pages scatter, which matters only for crawl politeness — and
that is solved below rather than by sharding differently.

Jump consistent hashing over rendezvous or a ring: it needs no lookup table and
moves exactly `1/n` of the keys when a shard is added.

## 2. Crawl: politeness is per host, and hosts cross shards

This is the real difficulty and the reason to write this down first.

If URLs are sharded by URL, five nodes can each hold pages of `example.com`,
and each will independently decide it is polite. Five nodes at one request per
second is five requests per second to a site that asked for one.

Politeness has to be owned by exactly one node per host, and it is not the
node that owns the URL:

```
politeness_owner = shard_for(host)   # a second, independent mapping
url_owner        = shard_for(url)
```

A node with a URL to fetch requests a **fetch lease** from the host's
politeness owner. Leases are the only cross-node chatter in the crawl, they are
small, and they batch: ask for twenty, fetch twenty.

**A slot crosses the wire as a relative wait, and that costs jitter.** The
authority computes an absolute moment on its own clock and sends "in N
milliseconds"; the asker turns that back into an absolute moment using its
clock, after the reply arrives. The difference in round-trip time between two
requests lands directly in the difference between the two reconstructed
moments, so a crawler can fire a millisecond or two early. Nothing removes
this — transferring an instant between two processes costs the jitter between
them — and it does not compound, because the authority's own reservations
never overlap and each is computed fresh. The milliseconds on the wire are
rounded **up** for the same reason: truncating grants a slot early, and early
is the direction that abuses a site.

As built, the authority hands out slots and nothing else. It does not fetch
`robots.txt` and does not know what a host asked for: the crawler supplies the
delay it read from that host's `Crawl-delay`, and the authority grants the
larger of that and its own floor. That keeps the authority free of an HTTP
client, and it means a misconfigured crawler asking for no delay does not get
no delay. What it does not yet do is share the `robots.txt` fetch, so each node
still makes that one request per host itself.

## 3. Index: scatter, gather, and the score that does not compose

Querying is the easy half:

- A coordinator sends the query to every shard.
- Each returns its top `k` with scores.
- The coordinator merges and returns the global top `k`.

The trap is that **BM25 is not comparable across shards**. `idf` depends on the
document frequency of a term and the total document count, and both are local.
A term that is rare in shard 3 and common in shard 7 gets a different weight in
each, so merging the scores compares numbers that were never on the same scale.

Two ways out, and the choice should be deliberate:

- **Two-round querying.** Round one collects `(df, doc_count)` per term from
  every shard; round two runs the search with the global values substituted.
  Correct, and doubles the latency.
- **A gossiped global dictionary.** Shards periodically publish their term
  statistics; every shard scores with a slightly stale global view. Fast, and
  approximately right, which for ranking is usually enough.

Start with the first, measure, and only move to the second when the numbers say
the extra round trip matters.

## 4. PageRank: the part that genuinely does not shard

PageRank is a global fixed point. Every iteration needs rank mass to cross
shard boundaries, because the whole point is that a link from another shard
counts.

The standard answer is to partition the graph and exchange only the boundary
after each iteration:

1. Partition nodes by the same URL hash the index uses.
2. Each shard iterates locally over the edges it owns.
3. After each iteration, each shard sends every other shard the mass flowing
   along its outgoing cross-shard edges — one `(node, contribution)` list per
   destination shard, which compresses well because it is sorted.
4. Iterate until the *global* residual, summed across shards, is under
   tolerance. Convergence is global; a shard cannot decide it alone.

Roughly forty iterations, one all-to-all exchange each. This is a batch job
measured in minutes, run on a schedule, not on the query path — which is
exactly why ranks are **stored in the segment** rather than computed at query
time.

As implemented, boundary messages carry the target's **URL**, not a global id.
That is heavier on the wire than it needs to be — a production version assigns
dense global ids once and sends those — and it is impossible to get subtly
wrong, which for a first version of an algorithm whose failures are invisible
is the better trade. `crates/rank/src/distributed.rs` says so where it does it.

One thing the implementation makes plainer than the description above: a shard
must own every node it holds *before* the first iteration, including pages with
no outgoing links. A dangling page holds rank; a page nobody owns holds none,
and the vector quietly stops summing to one.

**Routing lives on the coordinator, not on the shards.** A shard emitting a
contribution labels it "not mine" and the coordinator delivers it. Letting each
shard route would mean every shard needs the same routing function *and the
same shard count*, and a disagreement between two of them does not error: the
contribution arrives somewhere that does not own the page and is dropped, and
the ranking is quietly short of some rank. `ShardRanker::absorb` refuses to
create a page it does not own rather than let two shards own it — but not
creating the problem is better than detecting it.

**The order of the three exchanges is the whole correctness argument.** Dangling
mass has to be summed across every shard before anyone applies an iteration;
every shard has to emit from the previous iteration's ranks before anyone
absorbs this one's; the residual has to be summed before anyone stops. Get any
of them out of order and the run still finishes, still produces numbers, and
the numbers are wrong.

## 5. The storage layer

Segments are immutable files. That makes the "distributed database" the easiest
piece, not the hardest:

- Write a segment once, replicate it to `r` nodes, serve it read-only.
- A shard is a list of segment ids and where their replicas live.
- Merging is a background job that produces a new segment and atomically swaps
  it into the shard's list.
- The only mutable, consensus-needing state is that list: which segments make
  up a shard, and which replicas hold them. That is kilobytes, and it belongs
  in a small consistent store, not in the data path.

Keeping the mutable metadata tiny and the data immutable is what lets the
storage layer be boring, and the storage layer should be boring.

**BM25 needs two corpus-wide numbers, not one.** The document frequency of
each term, and the average document length. The second was missing until an
index of several segments made it visible: a shard scoring against its own
average makes a document look long or short depending on the company it keeps,
and its scores stop being comparable with another shard's — the same failure as
using local `idf`, and just as quiet. `Response::TermStats` carries both.

**A copy is not a replica until it has been checked.** Segments carry a digest
in their footer, written when the segment is built and free to read back. A
transfer that reports success is not evidence: the file is written under a
temporary name, read back from disk, digested, and only renamed into place if
it matches. The failure being defended against is not an attacker — anyone who
can rewrite a segment can rewrite its footer — but a truncated stream, a full
disk or a flipped bit, each of which produces a file that opens, parses, and
answers queries with documents quietly missing.

**A dead replica is not a dead shard, and the difference is deliberate.**
Losing a shard still fails the query, because a fifth of the corpus silently
absent is worse than an error. Losing one *copy* of a shard does not, because
another copy holds the same segment and the answer is the same answer rather
than a quieter one. A shard whose every replica is unreachable fails, and the
error names every address that was tried.

## Order of work

1. ~~Split the single process into `coordinator` and `shard` roles that talk
   over a local socket, with `shard_count = 1`.~~ **Done.** `crates/proto`
   carries the wire format and the routing; `crates/cluster` has both roles;
   `indexander shard` and `indexander search --shards` drive them.
2. ~~Global term statistics, two-round.~~ **Done**, and tested both in-process
   and over sockets.
3. ~~Fetch leases, so a multi-node crawl is polite before there is a
   multi-node crawl.~~ **Done.** `indexander leases --listen <addr>` runs an
   authority; `indexander crawl --leases <addr>` defers to it. The crawler
   asks a channel and cannot tell whether the answer came from this process or
   a socket, so a one-node crawl and a fifty-node crawl run the same code.
4. ~~Distributed PageRank with boundary exchange.~~ **Done.** The algorithm is
   in `crates/rank/src/distributed.rs` and the transport in
   `crates/cluster/src/ranking.rs`. Splitting a graph across up to 17 shards
   in process, or 5 over real sockets, gives the same ranks and the same
   *order* as computing it in one.
5. **Replication done; merge scheduling not.** `crates/cluster/src/replication.rs`
   copies a segment between nodes and refuses a copy whose digest does not
   match, and `Coordinator::connect_replicated` falls over to another replica
   of a shard. Segment merging is built — `Index::merge` folds several segments
   into exactly the segment a single pass would have written — and so is the
   manifest that records which segments make up an index and the tiered policy
   that decides what to fold next. What is missing is the daemon that runs them
   on a schedule and replicates a new manifest to a shard's other copies.

Step 1 is most of the value: it is what stops the single-process assumption
from being baked into another year of code.
