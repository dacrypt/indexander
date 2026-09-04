"""Cranfield 1400 -> a directory of documents, TREC topics, and TREC qrels.

Three things about this collection bite anyone who loads it naively, and all
three are handled here rather than assumed away:

1. The relevance scale is INVERTED. Cleverdon's 1 is "a complete answer" and 4
   is "minimum interest" (cranqrel.readme). Loading it as if higher were better
   ranks the best documents last.
2. Query ids in cran.qry skip (001, 002, 004, ...) while cranqrel numbers
   queries 1..225 consecutively. The mapping is by ORDER, not by the .I value.
3. cranqrel has three columns; TREC qrels have four.
"""
import os, re, sys

here = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(here, "docs")
os.makedirs(docs_dir, exist_ok=True)

# --- documents -------------------------------------------------------------
raw = open(os.path.join(here, "cran.all.1400"), encoding="latin-1").read()
parts = re.split(r'^\.I (\d+)\s*$', raw, flags=re.M)[1:]
written = {}
for i in range(0, len(parts), 2):
    doc_id = int(parts[i])
    fields, cur = {}, None
    for line in parts[i + 1].splitlines():
        m = re.match(r'^\.([TABW])\s*$', line)
        if m:
            cur = m.group(1); fields[cur] = []
        elif cur:
            fields[cur].append(line)
    title = " ".join(fields.get("T", [])).strip()
    body = "\n".join(fields.get("W", [])).strip()
    # .W usually opens with the title verbatim (1396 of 1400). Where it does
    # not - including two records whose .W is empty - the title is prepended,
    # so no document loses its title and none carries it twice.
    norm = lambda s: " ".join(s.split()).lower()
    text = body if norm(body).startswith(norm(title)) and title else (title + "\n" + body).strip()
    path = os.path.join(docs_dir, f"{doc_id:04d}.txt")
    open(path, "w", encoding="utf-8").write(text + "\n")
    written[doc_id] = path
print(f"documents: {len(written)}")
print(f"empty after conversion: {sum(1 for p in written.values() if os.path.getsize(p) <= 1)}")

# --- topics ----------------------------------------------------------------
raw = open(os.path.join(here, "cran.qry"), encoding="latin-1").read()
parts = re.split(r'^\.I (\d+)\s*$', raw, flags=re.M)[1:]
queries, declared = [], []
for i in range(0, len(parts), 2):
    text = parts[i + 1].replace(".W", " ", 1)
    queries.append(" ".join(text.split()))
    declared.append(parts[i])
print(f"queries: {len(queries)}  (declared ids run {declared[0]}..{declared[-1]}, "
      f"so they are numbered by order, not by that)")

with open(os.path.join(here, "topics.tsv"), "w", encoding="utf-8") as f:
    for n, q in enumerate(queries, start=1):
        # A leading hyphen means "exclude this term" to the query parser. No
        # Cranfield query means that; it would silently drop the word.
        q = re.sub(r'(?<![\w])-', ' ', q).replace('"', ' ')
        f.write(f"{n}\t{q}\n")

# --- judgements ------------------------------------------------------------
# Cleverdon 1 (best) .. 4 (weakest); -1 is the "no interest" marker, one per
# query. Flipped so that larger means more relevant, which is what every metric
# in the eval crate assumes.
GRADE = {1: 4, 2: 3, 3: 2, 4: 1, -1: 0}
seen, lines = set(), []
for line in open(os.path.join(here, "cranqrel"), encoding="latin-1"):
    fields = line.split()
    if len(fields) != 3:
        continue
    qid, doc_id, code = int(fields[0]), int(fields[1]), int(fields[2])
    if doc_id not in written:
        sys.exit(f"judgement for unknown document {doc_id}")
    if code not in GRADE:
        sys.exit(f"unknown relevance code {code}")
    seen.add(qid)
    lines.append(f"{qid} 0 {written[doc_id]} {GRADE[code]}")
open(os.path.join(here, "qrels.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"judgements: {len(lines)} over {len(seen)} queries")
print(f"queries with judgements but no topic: {sorted(seen - set(range(1, len(queries)+1)))}")
