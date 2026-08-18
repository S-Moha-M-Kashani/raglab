---
name: hierarchical-graph-rag
description: Index summaries and structure above the raw chunks — RAPTOR's recursive clustering tree, GraphRAG's entity graph with community summaries, LightRAG's dual-level index — so questions that span many chunks have something to retrieve. Use when questions are global, aggregate, comparative or multi-hop and flat chunk retrieval returns ten fragments that each answer a tenth of the question. Covers what each method builds, the LLM cost that makes GraphRAG impractical at scale, the cheap LLM-free variants, and why a summary layer must be measured on the question types it targets rather than on a corpus mean.
---

# Hierarchical and Graph RAG

**What it is.** Flat retrieval returns *k* chunks. Some questions cannot be
answered by any *k* chunks — "what were the main themes this year", "how often
did X happen", "how did the position on Y change over time". The evidence for
those is distributed across the corpus, and no individual chunk contains it.

Every method here answers the same way: build something above the chunks at index
time — summaries, clusters, a graph, community reports — and index that too, so
there is a row to retrieve that actually contains the aggregate.

## RAPTOR

Recursive Abstractive Processing for Tree-Organized Retrieval. Bottom-up:

1. Embed the leaf chunks.
2. Reduce dimensionality (UMAP) and cluster softly (a Gaussian mixture), so a
   chunk can belong to more than one cluster.
3. Summarise each cluster with an LLM into a new node.
4. Repeat over the new nodes until one node remains.

Retrieval then queries **all levels at once** (collapsed-tree) or walks down from
the root. A question about a theme matches a level-3 summary; a question about a
detail matches a leaf. Reported gain: **+20% absolute accuracy on QuALITY** when
retrieving from the appropriate level.

## GraphRAG

Microsoft's method, and a different object entirely. It uses an LLM to *extract*
entities and relationships from every chunk, assembles them into a knowledge
graph, partitions the graph into communities (Leiden), and has an LLM write a
report for each community at each level of the hierarchy.

Its strength is genuinely global sensemaking — "what are the main themes in this
corpus" is answered by map-reducing over community reports, not by retrieval at
all. Its weakness is stated plainly in every follow-up paper: the reliance on
frequent LLM calls and extensive summarisation makes it **prohibitively expensive
and inefficient for practical deployment**. Extraction is a model call per chunk;
report generation is a model call per community per level; and any corpus update
re-runs a large part of it.

Note what the graph is made of. In GraphRAG the nodes are *LLM-extracted
entities*. A system that builds a graph over chunks with embedding-similarity
edges is doing something else — legitimate, often useful, but not GraphRAG, and
calling it that overclaims.

## LightRAG and the cheap successors

LightRAG keeps the graph idea and drops most of the bill: lightweight
graph-enhanced indexing plus a **dual-level retrieval** strategy — a low level for
specific entity relationships, a high level for broad themes — and an incremental
update algorithm, so new data does not force a rebuild.

The 2026 literature is largely a race to keep the accuracy and delete the LLM
calls: `LiteSemRAG` (LLM-free semantic-aware graph retrieval), `LinearRAG`
(linear-cost graph retrieval on large corpora), token co-occurrence graphs,
`TagRAG` (tag-guided hierarchical KG retrieval), `HiRAG` and `ArchRAG`
(hierarchical community summaries).

## The measurement trap

A summary layer helps a *minority* of questions and helps them a lot. Averaged
over a mixed question set, that is a small number, and a small number is easily
read as "the hierarchy did nothing".

Three rules follow:

1. **Report per question type.** Counting, aggregate and multi-hop questions are
   where the layer lives. If those are 15% of the sample, a large win there moves
   the mean by very little.
2. **Report whether summaries were retrieved at all.** "Scored flat because
   nothing retrieved the summaries" and "scored flat because the summaries did
   not help" are different findings and need different fixes. Count summary rows
   in the retrieved set, on every row.
3. **Never apply a uniform boost to the summary layer.** It promotes whichever
   layer has the most rows, not whichever layer is most relevant, and it makes
   the mean worse while looking like a fix.

Also keep two controls: grouping by whatever structure the corpus already
declares (sections, threads, authors) and a naive clustering such as k-means. A
graph method that cannot beat metadata grouping and k-means is not a graph
finding.

## In this lab

This is implemented, and the design is a direct response to the trap above
(`raglab/hierarchy.py`, `IndexConfig.hierarchy`):

- **Eight groupings**, defaulting to `''` (flat): `louvain`, `leiden`,
  `label-prop` over a chunk graph; `raptor`, `agglomerative`, `kmeans` over
  vectors; `metadata` over the corpus's declared storylines; and flat.
- **`metadata` and `kmeans` are the two controls** — the corpus's own grouping
  and naive clustering. The earlier deleted rollup layer grouped by declared
  structure, so `metadata` makes the 2026-07-31 finding and any new one two rows
  of one table instead of two dates seven months apart.
- **No build calls a model.** All four summarisers (`centroid`, `lead-idf`,
  `mmr`, `card`) are extractive. Measured build cost across 167 sessions: 0.07 s
  for `metadata` to 7.3 s for `raptor`. That is what makes the grouping sweepable
  at all, and it is the deliberate difference from GraphRAG.
- **Not GraphRAG, and the help text says so first.** The nodes are chunks, not
  LLM-extracted entities. `graph_source='bipartite-terms'` promotes rare terms to
  nodes and is the closest honest analogue — the only source under which a
  community has a nameable subject.
- **Leiden and Louvain tie here**: 0.2587 against 0.2543 modularity. Leiden's
  advantage is over badly-connected communities and needs scale to show.
  `leiden` needs `uv sync --extra graph-index` and is refused when absent, never
  silently served by Louvain.
- **Retrieval does nothing until asked.** `summary_scope` defaults to `mixed`,
  `summary_boost` to 1.0, so a hierarchy build moves no number by itself.
  `leaves` is the control; `drill-down` retrieves among summaries then expands to
  members. None are index fields, so all three sweep free against one build.
- **`n_summaries` and `n_expanded` are on every row** — rule 2 above, made
  structural.
- **The 2026-07-31 finding**: deleting all six rollup layers scored within 0.006
  of keeping them. That is why the hierarchy is a knob and not a default.
- **A determinism warning**: `.runs/` rows from a *graph* hierarchy written
  before 2026-08-13 name an index a rebuild does not reproduce — `_term_postings`
  broke IDF ties in `set` iteration order. Same config gave 8, 8 and 6 groups
  across three processes. A seed fixes a method's RNG, not the order of its input.

## Sources

- [Enable RAPTOR — RAGFlow docs](https://ragflow.io/docs/enable_raptor)
- [LightRAG and hierarchical retrieval, as surveyed in LinearRAG](https://arxiv.org/pdf/2510.10114)
- [LiteSemRAG: Lightweight LLM-Free Semantic-Aware Graph Retrieval](https://arxiv.org/pdf/2604.16350)
- [TagRAG: Tag-guided Hierarchical Knowledge Graph RAG](https://arxiv.org/pdf/2601.05254)
- [Efficient RAG via Token Co-occurrence Graphs](https://arxiv.org/pdf/2606.30093)
- [GraphSearch: An Agentic Deep Searching Workflow for Graph RAG](https://arxiv.org/pdf/2509.22009)
