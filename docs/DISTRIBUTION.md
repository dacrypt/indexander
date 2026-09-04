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
4. Distributed PageRank with boundary exchange.
5. Replication and merge scheduling.

Step 1 is most of the value: it is what stops the single-process assumption
from being baked into another year of code.
