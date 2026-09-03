# retrieval.reranker_model — which model reads and scores the candidates

- **Step:** Retrieval (model role `rerank`). **Only consulted when**
  `retrieval.reranker = llm`. **Default:** `''` = the lab's configured model.

## What the knob does
Names the model that reads the top candidates and scores each one against the
question.

## What it means scientifically
Two facts about this role dominate everything else:

1. **It sits in the latency path of every query**, and it is called
   `rerank_depth` times per question. A slow model here is felt on every single
   question, and the cost is multiplicative rather than additive — unlike a
   judge, which runs once per answer.
2. **What is being asked of the model is judgment, not generation.** Pointwise
   LLM relevance scoring is a classification task, and its usefulness depends on
   *calibration*: scores must be comparable across chunks so the ordering means
   something. Models that answer well can still be poor rankers, typically by
   compressing everything into two or three distinct values — a constant or
   near-constant predictor produces an ordering that is really the first stage's
   ordering with noise.

A purpose-built cross-encoder is the alternative worth naming here: it is
trained for exactly this comparison, usually far cheaper per candidate, and
often better. The LLM reranker earns its place when the relevance criterion
needs *instructions* — "relevant only if it names a decision", say — which a
fixed cross-encoder cannot be told.

## Why RAG architectures have this knob
Because reranking is the highest-leverage precision stage and its cost is
per-query, the model choice here is the sharpest quality/latency trade in the
pipeline. Recording it per row is what keeps two measurements comparable.

## When it is useful
- **Prefer a small, fast, well-calibrated model**; verify it is not a constant
  predictor before trusting a row it produced.
- **Prefer `cross-encoder`** over `llm` unless the criterion needs prose
  instructions.
- **Leave it blank** to inherit the lab default when this stage is not what you
  are studying.

## Interactions
Inert unless `retrieval.reranker = llm`. Its call volume is set by
`retrieval.rerank_depth`; a model the active backend does not serve is a refusal
rather than a substitution, so a row never lies about which model ranked it.
