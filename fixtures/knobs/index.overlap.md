# index.overlap — units repeated between the pieces the budget makes

- **Step:** Index. **Fingerprinted:** yes. **Default:** 0.
- **Read by:** the budget's division, whatever the plan. Where
  `index.chunk_chars` divides a piece, each new piece begins with the tail of
  the one before it. An overlap at or above the budget is halved rather than
  looping forever; a piece nothing divided repeats nothing.

## What the knob does
Repeats the tail of each piece at the head of the next, so a sentence sitting
on a boundary appears whole in at least one chunk. `document` alone with a
budget and an overlap is the sliding window the shipped assistant used.

## What it means scientifically
A sliding window with stride `chunk_chars − overlap`. The purpose is to remove
the **boundary artefact** of a hard partition: with a fixed grid, whether a
claim is retrievable depends on an arbitrary offset, which injects variance
that has nothing to do with the retrieval method being measured. Overlapping
windows make the index a redundant covering rather than a partition, so every
span of text is contained in some unit. The cost is duplication: identical text
now lives in two vectors, so near-duplicates compete for the top-k slots and
the effective diversity of a result set falls.

## Why RAG architectures have this knob
Because boundary loss is silent. A pipeline with a good embedder and a good
reranker can still fail on a question whose answer straddled a cut, and no
downstream stage can repair it — the evidence was never stored as a unit.
Overlap is the cheapest insurance against that class of failure.

## When it is useful
- **Useful** on continuous prose with long sentences or claims that span
  sentence boundaries, and whenever `index.chunk_chars` is small relative to
  how much text one fact occupies.
- **Wasteful** on a plan whose pieces already fit the budget — `document /
  part` over short turns — where nothing is divided and the knob is inert.
- **Counterproductive** when large: index size grows, and duplicate hits crowd
  the top-k. If duplicates are the problem, lower `retrieval.mmr_lambda` to
  diversify rather than removing the overlap that is protecting recall.

## Interactions
Trades against `retrieval.k` (duplicates consume slots) and
`retrieval.mmr_lambda` (which can suppress the duplicates overlap creates).
A separator stage in `index.split_plan` is the cheaper answer to the same
boundary problem: cutting at a seam the text already has costs no duplication
at all, and an overlap still worth keeping beside it is one covering claims
that span a seam. `index.chunk_unit` says what this number counts.
