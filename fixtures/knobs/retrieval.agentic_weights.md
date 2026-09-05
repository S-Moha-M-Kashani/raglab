# retrieval.agentic_weights — the three weights of the agentic reranker

- **Step:** Retrieval. **Default:** (1.0, 0.3, 0.2) = relevance, recency,
  importance. **Read by** `retrieval.reranker = agentic`.
- **Greyed out** when the corpus declares no numeric label with `ranks: true`.

## What the knob does
Weights the three signals of the agentic reranker. Importance is the corpus's
own numeric label declaring `ranks: true`, rescaled to 0–1 by its declared
minimum and maximum — a rating, a severity, whatever that corpus chose to rank
chunks by. A corpus with no such label gives every chunk importance 0.0, so the
third weight shifts all scores equally and changes no ranking.

## What it means scientifically
This is the retrieval function from Park et al.'s **Generative Agents**: a
memory is retrieved by a weighted sum of *relevance* (similarity to the query),
*recency* (exponential decay in age) and *importance* (how significant the
memory is, independent of the query). Three things are worth noticing:

- **It is a linear utility over normalised signals.** Because the three
  components are on comparable 0–1 scales, the weights are directly
  interpretable as an exchange rate: at (1.0, 0.3, 0.2), a chunk must be 0.3
  similarity-units better to outweigh being one half-life older.
- **Two of the three are query-independent.** Recency and importance are
  properties of the chunk, so they act as a *prior* over the corpus. That is
  what makes the mix powerful (it encodes what generally matters) and what makes
  it dangerous (it can systematically bury the one old, unremarkable chunk that
  answers the question).
- **A degenerate signal is not a neutral signal.** With importance constant at
  0.0, that weight adds the same amount to every score: harmless here, but the
  general lesson is that a constant feature in a linear scorer contributes
  nothing while looking like a live knob. The lab greys it out for exactly that
  reason.

## Why RAG architectures have this knob
Because relevance alone is a poor model of what a query wants from an
accumulating corpus. Agent memory research made the case explicitly, and the
same argument applies to any corpus with timestamps and a notion of severity or
priority.

## When it is useful
- **Support tickets, incidents, alerts:** raise the importance weight — severity
  is genuine evidence about what matters.
- **Journals, logs:** raise recency for "lately" questions, and lower both
  non-relevance weights for archival questions.
- **Always compare against `lexical` or `cross-encoder`**: a tuned three-weight
  mix that only wins at one setting is a fragile result.

## Interactions
Needs a date-time label (recency) and a `ranks: true` numeric label
(importance); `retrieval.recency_half_life_days` sets the decay the second
weight applies to.
