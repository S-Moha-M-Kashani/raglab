# retrieval.recency_half_life_days — how fast the recency reranker forgets

- **Step:** Retrieval. **Default:** 180 days.
- **Greyed out** on a corpus that declares no date-time label: with no date on a
  chunk there is nothing to weigh by age.

## What the knob does
At 180 days, a chunk dated six months back counts half as much as the newest
one. Read by `retrieval.reranker` in `recency` and `agentic` modes.

## What it means scientifically
Exponential decay with a **half-life** parameter: `weight = 0.5 ^ (age /
half_life)`. Two reasons this particular functional form is the standard choice:

- **Memory models use it.** Human retrieval-from-memory models (Ebbinghaus's
  forgetting curve, ACT-R's activation decay) are exponential or power-law in
  age, and the *Generative Agents* retrieval score adopts exponential decay for
  the same reason: recent experience is more likely to be what a query is about.
- **It is scale-free in the right way.** Half-life expresses the decay in the
  corpus's own time units, so the same setting means "half as important per
  season" on a diary and "half as important per sprint" on an engineering log.
  A linear decay would need a hard cutoff; an exponential never quite zeroes,
  so old evidence stays reachable when nothing newer exists.

The important caveat is that recency is a **prior about the query**, not a
property of relevance. It is right for "how are things lately" and wrong for
"what happened that November" — where it actively suppresses the correct
evidence. That is why it is a reranker weight rather than a filter: a wrong
prior costs ranking positions, not the evidence itself.

## Why RAG architectures have this knob
Because most real corpora accumulate over time and most questions about them are
implicitly about the present state. Pure similarity ranking has no notion of
"stale", so a two-year-old note about a since-changed situation outranks the
correction.

## When it is useful
- **Short half-life** (days–weeks) for logs, monitoring notes, changing status —
  where old text is actively misleading.
- **Long half-life** (months–years) for reference corpora and journals, where age
  is weak evidence.
- **Turn recency off entirely** for period-pinned questions, or use
  `retrieval.time_filter` to name the period instead: a filter states the
  constraint exactly, while decay only nudges.

## Interactions
Requires a corpus date-time label (the same absence disables
`retrieval.time_filter` and blanks a summary's date span). Read by
`retrieval.reranker = recency`, and weighted as the second term of
`retrieval.agentic_weights`.
