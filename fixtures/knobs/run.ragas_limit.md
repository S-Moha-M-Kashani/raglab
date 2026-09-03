# run.ragas_limit — how many questions RAGAS scores

- **Step:** Run control. Applies when judged metrics make the full set too slow
  or too expensive.

## What the knob does
Caps the number of questions the RAGAS judge grades, independently of how many
questions the pipeline answered (`run.limit`).

## What it means scientifically
This is a **sample size**, and it behaves like one. The four deciding metrics
are means over questions, so their standard error shrinks roughly as `1/√n`:
grading 25 questions instead of 100 does not halve the precision of the estimate,
it doubles the error bar. Three consequences:

- **Small samples cannot resolve small differences.** The lab's ranking refuses
  to name a winner when the lead is inside the combined error, so a tight
  `ragas_limit` does not produce a fast answer — it produces "too close to
  call", which is the correct answer at that sample size.
- **Which questions were graded matters as much as how many.** A cap interacts
  with `run.balance`: a stride sample spreads across the set, while naming a
  question label takes an equal share of that label's values. A cap on a skewed
  sample measures one band and reports it as the pipeline.
- **The cap belongs on the row.** Two rows graded at different sample sizes carry
  different uncertainties, so the setting is recorded rather than assumed.

## Why RAG architectures have this knob
Because judged metrics are the expensive stage — a judge is called several times
per question — and the cost is what usually decides whether an evaluation happens
at all. A cap makes "measure something judged, now" possible without committing
to the full set.

## When it is useful
- **Exploratory runs** where the point is to see whether a configuration is in
  the right neighbourhood at all.
- **Cost-bounded reruns** against a remote judge.
- **Raise it before deciding.** For any comparison that will be ranked or
  published, grade as many questions as the budget allows, and always read the
  spread beside the score.

## Interactions
`run.ragas_mode = judged` is what makes it relevant; `run.limit` and
`run.balance` decide the pool it samples from; `decision_spread()` is how its
consequence becomes visible.
