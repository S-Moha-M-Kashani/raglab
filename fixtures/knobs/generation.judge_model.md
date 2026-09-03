# generation.judge_model — which model checks the answer against the atomic facts

- **Step:** Generation (model role `judge`). **Only consulted when**
  `generation.fact_judge` is on. **Default:** `''` = the lab's configured model.

## What the knob does
Names the model behind the fact judge: for each declared `derived_fact`, decide
whether the answer contains it.

## What it means scientifically
The fact judge is an **LLM-as-a-judge** classifier, and its reliability is a
measurable property rather than an assumption. Three failure modes matter:

- **Constant prediction.** A model that answers "present" (or "absent") for
  nearly everything yields a coverage number with no information in it. This lab
  has caught real models doing exactly that on its own screen, which is why a
  judge must be screened (`raglab-judgescreen`) before it may grade at all.
- **Cross-lingual entailment.** When the answer and the fact are in different
  languages, the judge is translating and judging in one step. That is strictly
  harder than monolingual entailment, and a weak multilingual model fails it
  silently — producing confidently wrong scores rather than obvious noise. It is
  also the reason no deterministic metric can replace this judge.
- **Unreachable or unparsable verdicts.** A stage that cannot reach its model
  must refuse to score rather than return a passing default, and an unparsable
  verdict is read in the direction that costs work. A judge that defaults to
  "pass" manufactures quality.

## Why RAG architectures have this knob
Because the measuring instrument is itself a model, and instruments need
calibration and identification. Naming the judge per row — and keeping it
separate from the answerer — is what makes a coverage number evidence instead of
an impression.

## When it is useful
- **Use a screened model**, and prefer the strongest affordable one: judging is
  low-volume per run compared to reranking, so capacity is cheaper here than
  anywhere else in the pipeline.
- **Never the same model as the answerer.** A model grading its own output is not
  evidence; LLM judges show a measurable preference for their own outputs.
- **Watch the volume anyway**: facts per question times questions is a real call
  count, which is why a reasoning-heavy model can be the wrong choice on
  wall-clock grounds.

## Interactions
Inert unless `generation.fact_judge` is on. Distinct from
`generation.ragas_model`, which grades the four deciding metrics; both are
distinct from `generation.model`.
