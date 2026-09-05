# retrieval.candidates — how deep each retriever looks before fusion and reranking

- **Step:** Retrieval. **Default:** 40.

## What the knob does
The number of hits taken from each retriever before Reciprocal Rank Fusion,
reranking and the cut to `retrieval.k`.

## What it means scientifically
This sets the **recall ceiling** of everything downstream. Retrieval pipelines
are cascades (the telescoping architecture of classic web ranking): each stage
is more expensive and more accurate than the last, and each can only reorder
what the previous stage passed on. A reranker cannot recover a document that was
never a candidate, so the candidate pool defines the best result the pipeline is
capable of, no matter how good the reranker is.

Two consequences worth stating:

- Recall@candidates is the quantity to watch when diagnosing, not recall@k. If
  the evidence is not in the pool, the fix is here — not in the reranker.
- Depth is cheap *at this stage* because nothing reads these candidates: the
  cost of reading is controlled separately by `retrieval.rerank_depth` and
  `retrieval.k`. Fusion over a deeper pool also gives RRF more evidence of
  agreement between the two retrievers.

## Why RAG architectures have this knob
Because the cheap first stage and the expensive second stage should be tuned
against different budgets. Collapsing them into one number forces a false
choice between recall and latency.

## When it is useful
- **Raise it** when the Inspector shows the right chunk existing but never
  entering the pool, and whenever a reranker is enabled — a strong reranker with
  a shallow pool is a wasted stage.
- **Raise it** for `hybrid-rrf` specifically: each retriever contributes its own
  depth, and fusion benefits from overlap.
- **Leave it** when the retriever is a hash-embedder ablation or the corpus is
  tiny; deeper pools on a small corpus soon mean "the whole corpus".

## Interactions
Feeds `retrieval.rrf_k` (fusion), `retrieval.rerank_depth` (what actually gets
read), `retrieval.grader` (candidates are gated after reranking) and finally
`retrieval.k`.
