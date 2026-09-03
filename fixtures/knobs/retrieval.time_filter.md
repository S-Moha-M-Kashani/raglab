# retrieval.time_filter — resolve time words in the question into a date range

- **Step:** Retrieval. **Default:** on.
- **Dataset-specific:** written for the bundled Farsi diary. Greyed out on a
  corpus that declares no date-time label.

## What the knob does
Reads Persian time language — «آذر», «تابستون», «سه ماه پیش» — as a Jalali date
range and restricts the search to it. A question in any other language matches
nothing, so the filter never fires and the search is simply unrestricted.

## What it means scientifically
This is **temporal information retrieval**: recognising and normalising temporal
expressions (the task standardised by TimeML and tools such as HeidelTime),
then using them as a hard constraint rather than a soft relevance signal. Two
properties make it powerful and dangerous in equal measure:

- **A filter changes the candidate universe, not the ranking.** Restricting to
  one month removes the other eleven months from competition entirely, which
  raises precision by far more than any reranker can — no re-scoring can beat
  simply not having the distractors. This is why metadata filtering is the
  highest-leverage retrieval technique on corpora that have usable metadata.
- **A filter is unrecoverable when wrong.** A mis-parsed expression or an
  off-by-one calendar conversion deletes the evidence, and no later stage sees
  it. Calendar conversion (Jalali↔Gregorian here) is exactly the kind of code
  where an error is systematic rather than random.

The language dependence is the honest general lesson the lab keeps it for: a
filter that knows its corpus's language and metadata beats a language-neutral
pipeline — and it only works on the corpus it was written for.

## Why RAG architectures have this knob
Because "what did I do in November" is not a similarity question. Time words in
a query are constraints, and treating them as embedding content asks a vector to
express something a `WHERE` clause expresses exactly. Every serious RAG system
ends up with some version of pre-filtering by structured metadata.

## When it is useful
- **Time-scoped questions** on corpora whose parts carry dates — journals,
  logs, tickets, meeting notes.
- **Turn it off** to measure how much of a result comes from filtering rather
  than from retrieval, and on any corpus in another language, where it is inert
  anyway.

## Interactions
Requires a corpus label typed `date-time`; the same absence leaves
`retrieval.recency_half_life_days` inert and a summary's date span blank.
Applied before retrieval, so it shrinks the pool `retrieval.candidates` draws
from.
