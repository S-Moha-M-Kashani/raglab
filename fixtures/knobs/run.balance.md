# run.balance — how a limited run chooses its questions

- **Step:** Run control. **Default:** `''` (stride).
- **Recorded on every row rather than assumed, because it has not always been
  the same.**

## What the knob does
Naming a question label takes an equal share of that label's own values —
"difficulty" takes an equal share of easy, medium and hard on a corpus that
declares it. `''` (stride) spreads across the set as it is, which means the
corpus's most common band dominates the sample. On the bundled diary that band
is *medium*: 57 of 112 questions, about half.

## What it means scientifically
The two options are **systematic sampling** and **stratified sampling**, and
they estimate different things:

- **Stride** preserves the population's composition, so the mean it produces is
  an unbiased estimate of the corpus mean — the number you want if the question
  set's mix reflects the questions you expect in use. Its weakness is
  resolution: a rare band contributes few questions, so the pipeline's behaviour
  on that band is measured with a very wide error bar or not at all.
- **Stratified by label** allocates equally across a label's values, which
  estimates each band's performance with comparable precision. Its mean is *not*
  the corpus mean — it is the mean of a re-weighted population — so it answers
  "how does this configuration do across the difficulty range" rather than "how
  will it do on this corpus".

This matters because the four deciding metrics are means over questions. A skewed
sample measures one band and reports it as the pipeline; an equalised sample
reports a composite the corpus does not have. Neither is wrong, and that is
exactly why the setting is recorded on the row: two rows sampled differently are
two measurements, not two configurations.

## Why RAG architectures have this knob
Because evaluation sets are almost never balanced, and the imbalance is invisible
in the final score. Making the sampling scheme explicit is what lets a reader
know whether a difference between rows could be a difference in who was asked.

## When it is useful
- **Stride** for a headline number meant to describe the corpus, and for paired
  A/B comparisons at the same limit.
- **Stratified** when a band is rare but important (hard questions, unanswerable
  questions), or when you suspect a configuration helps one band and hurts
  another.
- **Never change it mid-comparison.** A sampling default must never move
  underneath recorded runs, or old rows stop being comparable.

## Interactions
Applies when `run.limit` is set; `run.labels` restricts the pool first;
`run.ragas_limit` samples again for the judged metrics. Part of what defines a
row's question set, and therefore its comparability group.
