# index.hierarchy — group the chunks and index one summary per group beside them

- **Step:** Index. **Fingerprinted:** yes (and with it, the six knobs that
  describe it — they are dropped from the fingerprint when the hierarchy is
  flat, so a stale value cannot cost a rebuild).
- **Values:** `''` (flat, the default), `louvain`, `leiden`, `label-prop`
  (graph partitions), `raptor`, `agglomerative`, `kmeans` (vector clusterings),
  `metadata` (the declared control).

## What the knob does
Groups the chunks and writes one summary per group **into the same index,
beside the leaves** — the leaves always stay. A summary that drops a detail
would otherwise make the question about that detail unanswerable forever.

## What it means scientifically
This is the **multi-level / hierarchical retrieval** idea, whose reference point
is RAPTOR: recursively cluster chunks, summarise each cluster, index the
summaries, and let retrieval choose its level of abstraction. The motivation is
that some questions have no single supporting chunk — "what changed over the
year", "what were the recurring themes" are *sensemaking* queries, and any
top-k over leaves answers them with an arbitrary handful of examples.

The three families are three different theories of what a group is:

- **Graph partitions** treat chunks as nodes and maximise modularity. Louvain
  is the standard greedy multi-level method; **Leiden** adds refinement passes
  that guarantee communities are internally well-connected — an advantage that
  needs scale to show, so expect louvain and leiden to tie on a small corpus.
  **Label propagation** is near-linear and parameter-free, which makes it the
  honest control: it reads no `granularity` at all.
- **Vector clusterings** work in embedding space (k-means, agglomerative,
  RAPTOR's own recursive scheme) and assume clusters are geometric, not
  relational.
- **`metadata`** groups by whatever storylines the corpus itself declares. It is
  the control that asks whether any learned grouping beats the one the data came
  with.

**These are not GraphRAG.** GraphRAG extracts entities and relations with a
model; this lab builds offline, so the nodes are chunks, and
`index.graph_source = bipartite-terms` is the closest honest analogue.

## Why RAG architectures have this knob
Flat top-k retrieval has a fixed granularity, and question granularity varies.
Summaries give the retriever a coarser unit to return when the question is
coarse — without deleting the fine units the specific questions need.

## When it is useful
- **Useful** for broad, thematic, cross-document questions, and for corpora with
  many short, repetitive parts.
- **Neutral or harmful** for pure factoid lookup: a summary competes with its own
  members in the search and can displace the leaf that held the fact.
- **Remember:** building a hierarchy changes nothing about retrieval until
  `retrieval.summary_scope` moves off `mixed`'s default behaviour — and even in
  one pool, leaves outnumber summaries and outvote them.

## Interactions
`index.graph_source`/`index.graph_knn` (graph families only),
`index.granularity` (tuned families only), `index.hierarchy_levels`
(multi-level families only), `index.min_group`, `index.summarizer`; and on the
retrieval side `retrieval.summary_scope`, `retrieval.summary_boost`,
`retrieval.summary_levels`.
