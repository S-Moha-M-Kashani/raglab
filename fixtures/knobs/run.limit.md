# run.limit — how many ground-truth questions to score

- **Step:** Run control. The subset is **never the first n**.

## What the knob does
Caps the number of ground-truth questions the run evaluates. The subset is drawn
by `run.balance` — striding across the set, or taking an equal share of a
labelled band — so a limit of 10 still covers the whole set instead of ten of
one kind.

## What it means scientifically
This is the **sampling** knob, and the design decision worth understanding is
the refusal to take a prefix. Ground-truth files are not in random order: they
are usually authored in thematic or chronological batches, and difficulty and
topic correlate with position. Taking the first *n* is therefore a
*convenience sample* with systematic bias — it measures the beginning of the
file and reports it as the corpus.

Two further points:

- **Every deciding metric is a mean over questions**, so the sample defines the
  estimand. A biased sample does not merely add noise; it estimates a different
  quantity, and no amount of repetition corrects it.
- **Sample size sets resolution.** With ten questions, differences of a few
  points are indistinguishable from noise, which is exactly what the ranking's
  refusal-inside-the-error rule expresses. A limit is a statement about how
  precisely you intend to measure.

Systematic (stride) sampling is used rather than simple random sampling because
it is *reproducible*: the same limit yields the same subset, so two
configurations are compared on identical questions — a paired comparison, which
removes question difficulty as a source of variance between rows.

## Why RAG architectures have this knob
Because a full judged run over a large question set is slow and expensive, and
most iterations do not need it. The knob makes the cheap version honest by
controlling *how* the subset is chosen, not just how big it is.

## When it is useful
- **Small limits** while wiring a pipeline and checking it runs end to end.
- **Moderate limits** for A/B comparisons where both arms use the same limit and
  balance — the paired structure does most of the work.
- **No limit** for a row you intend to publish or rank.

## Interactions
`run.balance` decides the sampling scheme; `run.labels` restricts the pool first;
`run.ragas_limit` can cap the judged subset further; `run.workers` decides how
fast the sample is scored.
