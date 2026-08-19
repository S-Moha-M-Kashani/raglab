---
name: hybrid-retrieval-fusion
description: Run a dense (embedding) retriever and a sparse (BM25) retriever over the same corpus and merge their ranked lists, usually with Reciprocal Rank Fusion. Use when queries mix paraphrase with exact tokens — identifiers, rare proper nouns, error codes, numbers — because dense retrieval misses the exact ones and lexical retrieval misses the paraphrased ones. Covers RRF and its k parameter, why rank fusion avoids score normalisation, weighted alternatives, and how to tell whether the hybrid is actually beating its better half.
---

# Hybrid Retrieval and Rank Fusion

**What it is.** Two retrievers with different failure modes, run in parallel over
one corpus, their ranked lists merged into one. In practice: a dense bi-encoder
and BM25.

They fail in opposite directions, which is the entire argument:

- **Dense** matches meaning. It finds a paraphrase with no shared words, and it
  reliably fails on a rare literal token — an order id, a function name, a
  misspelling, a number — because that token contributes almost nothing to a
  pooled 768-dimensional vector.
- **Sparse (BM25)** matches terms. It nails the rare literal token, precisely
  because rarity is what IDF rewards, and it fails completely when the query and
  the document share no vocabulary.

A corpus of real questions contains both kinds. Choosing one retriever means
choosing which half of your queries to serve.

## Reciprocal Rank Fusion

The standard merge, from Cormack, Clarke and Buettcher (2009):

```
RRF(d) = Σ over retrievers i of  1 / (k + rank_i(d))
```

`rank_i(d)` is the 1-based rank of document *d* in retriever *i*'s list; a
document missing from a list contributes nothing. Documents are then sorted by
`RRF(d)` descending.

**Why ranks and not scores.** A cosine similarity of 0.83 and a BM25 score of
14.2 are not on the same scale, are not on the same scale as each other across
*queries* either, and any normalisation you invent (min-max, z-score, softmax)
is a modelling choice you would then have to tune and defend. RRF discards the
magnitudes and keeps only the ordering, which both retrievers agree is
meaningful. That is a real loss of information — a retriever that is *certain*
cannot say so — traded for a fusion that needs no calibration and no tuning.

**What `k` does.** It flattens the curve. At `k=0` the top hit is worth 1.0 and
the second 0.5, so rank 1 dominates absolutely. At `k=60` — the paper's value and
the near-universal default — rank 1 is worth 1/61 and rank 2 is worth 1/62, so
the top of each list barely outranks its neighbours and consensus across
retrievers decides instead. Small `k` trusts each retriever's top hit; large `k`
trusts agreement. It is worth sweeping and almost never is.

## Alternatives to RRF

- **Weighted score fusion** — normalise each retriever's scores and take a convex
  combination. Strictly more expressive than RRF, and strictly more to get wrong:
  the normaliser is a per-corpus choice and the weight is a per-corpus choice.
- **Relative score fusion** — min-max within each result list per query, then
  weight. A common middle ground; still a normalisation.
- **Learned fusion** — train a small model on the two rank lists. Needs labels
  and re-training per corpus, and at that point a cross-encoder reranker is a
  better use of the same supervision.
- **Single-model hybrids** — sparse-dense models such as SPLADE learn a sparse
  representation with an encoder, folding both behaviours into one index. One
  index instead of two, but a different retrieval engine.

## Reading the result honestly

The trap: a hybrid is compared against the *worse* of its two halves, or against
neither, and declared a win.

Report three rows every time — dense alone, sparse alone, fused. A fusion that
does not beat both halves has bought you nothing but a second index. And check
per query type, not only on the mean: fusion often loses a little on the queries
dense already handled and wins a lot on the literal-token ones, which is exactly
the trade you wanted, and is invisible on an average.

## When it pays and when it does not

Pays on mixed-vocabulary corpora, on anything with identifiers, on multilingual
or code corpora, and as a cheap first stage feeding a reranker.

Does not pay when queries are consistently conversational paraphrase with no
literal anchors — the sparse arm contributes noise and you have doubled index
cost and query latency for nothing. Also does not pay when the corpus is small
enough that a cross-encoder over everything is affordable.

## In this lab

- `retriever='hybrid-rrf'` is the sweep baseline and part of candidate F. The
  dense arm reads `MemoryVectors` (brute-force cosine, `1 - distance`).
- **`rrf_k` has no control on either panel.** It is one of the three fields in
  `UNSHOWN`, carried through a preset but not rendered — so it is a knob nobody
  can sweep from the UI, and the `k` discussion above applies to it entirely in
  theory here. That is a gap, not a decision.
- Do not confuse `rrf_k` with the `k=8` of candidate F: the latter is `top_k`,
  how many contexts reach the answerer.
- The lab's per-question rows make the three-row honesty check above cheap to run
  — `retriever='dense'`, `'sparse'`, `'hybrid-rrf'` is a legitimate one-knob
  candidate triple under `sweep.py`'s rule.

## Sources

- [12 Advanced RAG Techniques: Beyond Naive Retrieval](https://atlan.com/know/advanced-rag-techniques/)
- [9 advanced RAG techniques and how to implement them — Meilisearch](https://www.meilisearch.com/blog/rag-techniques)
- [Contextual Retrieval — Anthropic Engineering](https://www.anthropic.com/engineering/contextual-retrieval) (hybrid is the baseline its table builds on)
