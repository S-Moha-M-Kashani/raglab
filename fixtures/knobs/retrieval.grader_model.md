# retrieval.grader_model — which model decides whether a chunk is relevant at all

- **Step:** Retrieval (model role `grade`). **Only consulted when**
  `retrieval.grader = llm`. **Default:** `''` = the lab's configured model.

## What the knob does
Names the model behind the relevance gate — the stage that lets the lab abstain
rather than answer from noise.

## What it means scientifically
The gate is a binary classifier implemented by prompting, and the properties that
matter are classifier properties, not chat quality:

- **Separation and calibration.** The gate is only useful if relevant and
  irrelevant chunks land in different score ranges (see
  `retrieval.grade_threshold`). A model that outputs 0.9 for everything is a
  constant predictor: it has an accuracy number and no discriminative power.
- **Direction of error under uncertainty.** This lab reads an unparsable verdict
  in the direction that *costs work* rather than the direction that passes, and a
  stage that cannot reach its model refuses to score instead of returning a
  passing default. A gate that silently defaults to "relevant" when the model is
  unavailable is not a gate — it is a gate-shaped no-op.
- **Cross-lingual competence.** Judging relevance between a question and a chunk
  in another language is a harder task than either monolingual case, and a weak
  multilingual model fails at it quietly.

Measured here, an LLM gate at threshold 0.4 refused all five unanswerable
questions on the bundled diary while wrongly refusing 3% of the answerable ones
— the lexical gate had no threshold that could do both.

## Why RAG architectures have this knob
Because abstention is the safety-relevant behaviour of a RAG system, and it is
delegated to a model. Naming that model per row is the difference between a
reproducible refusal policy and an anecdote.

## When it is useful
- **Screen the model first.** A gate model should be verified as a non-constant
  predictor before any row it produced is trusted — the same discipline the lab
  applies to judges.
- **Prefer a small fast model** for the volume (it runs per candidate), but not
  one that cannot read the corpus's language.
- **Leave it blank** to inherit the lab default when studying other stages.

## Interactions
Inert unless `retrieval.grader = llm`. Its threshold scale is not comparable to
the lexical gate's; call volume follows `retrieval.candidates`/`rerank_depth`.
