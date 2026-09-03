# generation.ragas_model — the model that grades the four deciding metrics

- **Step:** Generation (model role `ragas`). **Only consulted when**
  `run.ragas_mode = judged`. **Default:** `''` = the lab's configured model.
- **Kept separate from the answerer on purpose.**

## What the knob does
Names the model RAGAS uses for faithfulness, answer relevancy and factual
correctness — and, with judged context precision and recall, this is the model
behind four of the metrics a configuration is actually chosen by.

## What it means scientifically
This is the most consequential model in the lab, because it is not part of the
system under test — it *is the measuring instrument*.

- **Judge independence.** A model grading its own output is not evidence. LLM
  judges exhibit self-preference: they recognise and prefer their own
  generations, which biases every comparison in which the answerer and judge
  coincide. The sweep therefore refuses that pairing outright rather than warning
  about it.
- **Screening before use.** A judge must pass `raglab-judgescreen` before it may
  grade. The screen's purpose is to detect constant or near-constant predictors:
  a judge that scores everything 0.8 produces means, spreads and rankings that
  are pure artefact.
- **Metric semantics depend on the judge.** RAGAS's faithfulness and relevancy
  are defined by prompted decompositions and verifications, so two judges are
  two instruments. That is why a leaderboard row records its judge, why the board
  refuses to rank across judges, and why `group()` partitions by judge before any
  winner may be named.
- **Refuse rather than substitute.** A judge the active backend does not serve is
  a refusal; a stage that cannot reach its judge reports `GradeUnavailable`
  rather than a passing default. A metric that silently degrades is worse than a
  missing one, because it still votes.

## Why RAG architectures have this knob
Because judged metrics are the only practical way to score faithfulness and
relevancy at scale, and they inherit all the properties of the model producing
them. Naming the judge explicitly turns "our RAG scores 0.78" into a statement
someone else could reproduce.

## When it is useful
- **Choose one screened judge and keep it fixed** across every configuration you
  intend to compare; changing it invalidates comparability more thoroughly than
  changing almost any pipeline knob.
- **Prefer capacity over speed** here, within the call budget: this is where
  measurement error enters the experiment.
- **Never reuse the answerer's model.**

## Interactions
`run.ragas_mode` turns judged metrics on; `run.ragas_limit` bounds how many
questions it grades; `decision_score()` averages exactly four of its metrics and
is never shown without `decision_spread()`.
