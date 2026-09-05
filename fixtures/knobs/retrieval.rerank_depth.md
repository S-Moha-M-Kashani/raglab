# retrieval.rerank_depth — how many candidates the reranker actually reads

- **Step:** Retrieval. **Default:** 20.

## What the knob does
The reranker is the expensive stage, so this is the cost dial: depth 20 with
`k` 8 means twenty chunks scored to choose eight.

## What it means scientifically
This is the **cascade** parameter. Multi-stage ranking works because cost and
accuracy rise together: stage one is cheap and shallow over everything, stage
two is expensive and accurate over a shortlist. Depth sets where that boundary
falls, and it has a clean interpretation:

- The reranker can only fix mistakes **inside** the window. A relevant chunk
  ranked 35th by the first stage is invisible to a reranker reading 20, however
  good the reranker is.
- The marginal value of depth decreases, because the probability that a relevant
  document sits at rank *r* falls with *r*, while the cost per extra document is
  constant. There is therefore an optimum, and it is usually a small multiple of
  `k` — commonly 2× to 5×.
- Depth also bounds the *harm* a bad reranker can do: with a large window, a
  miscalibrated model can promote junk from deep in the pool into the final `k`.

For a cross-encoder or LLM reranker, cost is linear in depth and dominates
per-question latency; for `lexical` it is free, so depth can be as large as the
candidate pool.

## Why RAG architectures have this knob
Because the accuracy/latency trade of the expensive stage should be tunable
without changing what the reader sees (`k`) or what was retrieved
(`candidates`). Keeping the three separate is what makes cost and quality
measurable independently.

## When it is useful
- **Raise it** when the Inspector shows relevant chunks sitting just outside the
  reranked window, and when the reranker is free (`lexical`).
- **Lower it** when per-question latency or spend is the binding constraint with
  `cross-encoder` or `llm` — this is the first knob to turn, before weakening
  the model.
- **Keep depth ≥ 2× k**; at depth = k the reranker has nothing to choose and the
  stage is decorative.

## Interactions
Bounded above by `retrieval.candidates` and below by `retrieval.k`. With
`retrieval.reranker = llm` it multiplies the number of model calls per question,
so it interacts with `run.workers` and with the model named in
`retrieval.reranker_model`.
