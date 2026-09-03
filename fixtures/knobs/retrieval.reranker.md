# retrieval.reranker — re-score the candidates before the cut to k

- **Step:** Retrieval. **Default:** `lexical`.
- **Values:** `lexical`, `none`, `recency`, `agentic`, `cross-encoder`, `llm`.

## What the knob does
Re-scores the candidate pool and reorders it before the cut to `retrieval.k`.
`lexical` is free IDF coverage; `recency` prefers recently dated chunks;
`agentic` is the Generative Agents mix of relevance + recency + importance;
`cross-encoder` reads question and chunk *together* with a real model; `llm`
asks a model to score each one.

## What it means scientifically
Reranking exists because of a specific limitation of first-stage retrieval. A
bi-encoder embeds query and document **independently**, so the score is a
similarity between two summaries of meaning, computed without either seeing the
other. A **cross-encoder** (the monoBERT line of work) concatenates question and
passage and runs full attention across both, which lets it model term-level
interaction — negation, qualifiers, who-did-what-to-whom — that a dot product
cannot express. That accuracy gain is why the cascade exists at all, and its
quadratic cost is why it runs on 20 candidates rather than the whole corpus.

The non-model rerankers encode explicit priors instead of learned interaction:

- **`lexical`** rewards coverage of the question's rare terms — a cheap
  approximation of "does this passage actually contain the asked-about
  entities".
- **`recency`** applies an exponential decay in age; see
  `retrieval.recency_half_life_days`.
- **`agentic`** is the retrieval score from Park et al.'s *Generative Agents*:
  a weighted sum of relevance, recency and importance, i.e. a linear utility
  over three signals rather than one.
- **`llm`** is pointwise LLM relevance judging — flexible, slow, and only as
  consistent as the model's calibration.

## Why RAG architectures have this knob
Because first-stage retrieval optimises for recall at low cost and gets the fine
ordering wrong, and the fine ordering is what survives the cut to `k`. Reranking
is where most of the practical precision gain in modern RAG comes from.

## When each option is useful
- **`cross-encoder`** when precision matters and a local model is affordable —
  usually the best accuracy per unit of complexity.
- **`lexical`** as the free default and the honest baseline: it often captures
  much of the gain on corpora full of rare terms.
- **`recency`/`agentic`** on corpora with dates and importance labels, and for
  questions about the recent state of things rather than the whole record.
- **`llm`** when the relevance criterion is subtle enough to need instructions,
  accepting the latency on every question.
- **`none`** to measure what reranking bought.

## Interactions
`retrieval.rerank_depth` decides how many candidates are read (the cost dial);
`retrieval.reranker_model` names the model for `cross-encoder`/`llm`;
`recency`/`agentic` need corpus labels (date-time, ranks) and are greyed out
without them.
