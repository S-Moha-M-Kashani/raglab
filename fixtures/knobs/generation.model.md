# generation.model — which model writes the answer

- **Step:** Generation (model role `answer`). **Only consulted when**
  `generation.answerer = llm`. **Default:** `''` = the lab's configured model.

## What the knob does
Names the model that writes the answer from the retrieved context, cites the
document ids it drew from, and must refuse when the corpus is silent.

## What it means scientifically
This role is *grounded* generation, not open-ended generation, and the
competences it needs are specific:

- **Instruction adherence over knowledge.** The model is asked to answer from
  the provided context only. What matters is whether it stays inside that
  context — parametric knowledge here is a liability, because a plausible fact
  the corpus never stated is a faithfulness failure that reads like a good
  answer.
- **Abstention.** Refusing when the context does not support an answer is a
  behaviour models are notoriously reluctant about; a model that always produces
  something turns every unanswerable question into a fabrication.
- **Extraction *and* aggregation.** Faithfulness and fact coverage are separate
  axes: staying inside the context is necessary but not sufficient, because the
  answer must also contain the atomic facts the question asked for. Measured on
  the bundled diary, faithfulness was 0.743 while fact coverage was 0.261 —
  answers that were true and incomplete. That is why this is the interesting
  dropdown on that corpus.
- **Language.** The answer is written in the corpus's language, so a model with
  weak generation in that language degrades every row even when its English is
  excellent.

## Why RAG architectures have this knob
Because the answerer is where retrieval quality is either used or wasted, and
because it must be *nameable*: an evaluation row that does not record which model
wrote the answer cannot be compared to another. The lab also refuses to
substitute — a model the active backend does not serve is a refusal, never a
silent swap — so a row never lies about what produced it.

## When it is useful
- **Compare a small and a mid model** when fact coverage is the failing metric:
  incompleteness usually responds to capacity more than faithfulness does.
- **Keep it distinct from the judges.** The sweep refuses a configuration where
  the answerer and the judge are the same model, because a model grading its own
  output is not evidence.
- **Leave it blank** to inherit the lab default when studying retrieval.

## Interactions
Inert unless `generation.answerer = llm`. Must differ from
`generation.ragas_model` (and in practice from `generation.judge_model`);
`retrieval.max_context_chars` and `retrieval.k` decide what it gets to read.
