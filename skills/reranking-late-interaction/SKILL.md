---
name: reranking-late-interaction
description: Re-score a shortlist of retrieved candidates with a model that reads the query and the document together, instead of comparing two independently-computed vectors. Use when recall at depth 50-200 is good but the top few results are wrong, which is the most common single defect in a working RAG pipeline. Covers cross-encoder rerankers, listwise LLM rerankers, ColBERT-style late interaction and multi-vector retrieval, the latency and storage each costs, and how a silent reranker failure produces a row that lies about what ran.
---

# Reranking and Late Interaction

**What it is.** First-stage retrieval is a similarity search between two vectors
computed in ignorance of each other — the document was embedded months ago and
the query a moment ago, and neither saw the other. That is what makes it fast
enough to run over a whole corpus, and it is also why its ordering is coarse.
Reranking spends real computation on a shortlist to fix the ordering.

The three architectures, in order of how late the query and document meet:

| | Query and document meet | Index holds | Cost |
| --- | --- | --- | --- |
| Bi-encoder (first stage) | never — only their vectors | 1 vector per chunk | cheap, corpus-wide |
| Late interaction (ColBERT) | at scoring, per token | ~1 vector per token | mid, corpus-wide with an ANN trick |
| Cross-encoder (rerank) | inside the model, full attention | nothing — scored live | expensive, shortlist only |

## Cross-encoder reranking

Concatenate query and document, run one forward pass, get one relevance score.
Full attention between every query token and every document token, which is why
it is markedly more accurate — and why it can never be precomputed, since the
score does not exist until the query arrives.

The shape is therefore always two-stage: retrieve `rerank_depth` candidates
cheaply (50–200 is the usual band), rerank them, keep `top_k`. The depth is the
knob that matters: reranking cannot recover a document the first stage never
returned, so depth sets the ceiling and the reranker only decides how much of it
you reach.

Anthropic's stack measured reranking as 2.9% → 1.9% top-20 failure — a 34%
relative cut *on top of* an already contextual, already hybrid pipeline. It is
usually the highest-yield single addition to a pipeline that has a decent first
stage.

## Listwise and LLM rerankers

A cross-encoder scores each document alone, so it cannot say "this one is
redundant given that one". Listwise rerankers read several candidates together
and emit an ordering. Recent work (`jina-reranker-v3`, "last but not late
interaction") does this in one pass over the concatenated candidates.

An LLM can also be used directly as a reranker or as a relevance *gate* — asked
per document whether it is relevant, with a threshold. That is slower and costs
*k* model calls per question, and it buys something a scorer does not: a stated
reason, and the ability to drop everything when nothing is relevant.

## Late interaction and multi-vector retrieval

ColBERT keeps one vector per token instead of one per chunk, and scores with
MaxSim: for each query token, take its best match among the document's token
vectors, then sum. Query and document are still encoded independently — so the
document side is precomputable — but they interact at token granularity rather
than after pooling. That recovers most of the cross-encoder's accuracy at
something much closer to first-stage cost, and it is notably strong
out-of-domain, where a bi-encoder trained on one distribution degrades most.

The bill is storage and index complexity: hundreds of vectors per document.
`Jina-ColBERT-v2` is the general-purpose multilingual option and cuts storage by
up to 50% through output-dimension reduction; 2026 work on principled token
pruning (a Voronoi-cell formulation) attacks the same cost. The area has enough
momentum to have its own venue — LIR, the first workshop on Late Interaction and
Multi Vector Retrieval, at ECIR 2026.

## The failure mode to design against

A reranker sits in the middle of a pipeline and has a natural fallback: if it
fails, pass the pre-rerank order through. That is the most dangerous possible
behaviour, because the run still completes, still produces numbers, and every
field on the row still says `reranker=cross-encoder`. You have measured the
first stage and labelled it as the reranked stage.

A reranker that cannot run must **refuse**, loudly, and the refusal must reach
the row. This is the same argument as an unreachable relevance grader scoring
every document 0.5 and thereby clearing its own threshold.

## When it pays and when it does not

Pays whenever recall@50 is much better than recall@5 — which is the common case,
and is the one measurement that tells you a reranker will help before you buy
one. Pays most on long or heterogeneous chunks, where a pooled vector is a poor
summary.

Does not pay when the first stage already puts the gold document at rank 1, when
latency budget is tight and the corpus is small enough to widen `top_k` instead,
or when `rerank_depth` is set so low that the reranker is only reordering results
you were going to keep anyway.

## In this lab

- `reranker` offers six values — `'lexical'`, `'none'`, `'recency'`,
  `'agentic'` (a weighted blend, `agentic_weights`), `'cross-encoder'`, and
  `'llm'` (a per-document model scorer, `reranker_model`) — with `rerank_depth`
  as the shortlist size. Candidate F uses the lexical rerank.
- **Known debt, stated in `CLAUDE.md`:** `pipeline._rerank` swallows a
  cross-encoder failure and falls back to the pre-rerank order — precisely the
  defect above. Fixing it properly needs `cross_encoder_available()` wired into
  `LabConfig.validate()`, which is a design change rather than a patch.
- The `grader='llm'` / `grade_threshold=0.4` pair in candidate F is the LLM-gate
  variant, not a reranker: it filters rather than reorders, and it is the only
  part of candidate F that acts *after* retrieval.
- Late interaction is **not implemented here** and would not fit the current
  store: `MemoryVectors` is one vector per row with brute-force cosine, and
  MaxSim needs one vector per token. It is a store change, not a knob.

## Sources

- [jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking](https://arxiv.org/pdf/2509.25085)
- [Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever](https://aclanthology.org/2024.mrl-1.11/)
- [LIR: The First Workshop on Late Interaction and Multi Vector Retrieval, ECIR 2026](https://arxiv.org/pdf/2511.00444)
- [A Voronoi Cell Formulation for Principled Token Pruning in Late-Interaction Retrieval](https://arxiv.org/pdf/2603.09933)
- [Contextual Retrieval — Anthropic Engineering](https://www.anthropic.com/engineering/contextual-retrieval)
