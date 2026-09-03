# retrieval.max_context_chars — the budget for the assembled context

- **Step:** Retrieval. **Default:** 6000 characters.

## What the knob does
Caps the total size of the context handed to the answerer. When the budget is
exceeded, **whole chunks are dropped, never truncated**.

## What it means scientifically
Two separate ideas meet in this knob.

**1. Context is a scarce resource with sub-linear returns.** More retrieved text
does not monotonically improve answers: models use evidence in the middle of a
long context less reliably than evidence at its edges (**lost in the middle**),
and irrelevant passages measurably degrade generated answers. So a budget is not
merely a cost control — it is a quality control.

**2. Truncation is worse than omission.** Half a chunk reads as a complete one:
it presents a claim whose qualifier, negation or continuation was cut off, and it
invites an answer from a sentence whose second half changed the meaning. Dropping
whole chunks keeps every unit in the context *semantically intact*, so the
answerer's errors are errors of missing evidence rather than of corrupted
evidence — and missing evidence is at least visible in the context-recall
metric, while corrupted evidence looks like a faithful answer to a false premise.

This is why the budget is expressed in characters rather than tokens here: it is
an assembly rule, not a model limit, and it must behave identically across
backends whose tokenizers differ.

## Why RAG architectures have this knob
Because `retrieval.k` chooses *how many* chunks and chunk size decides *how big*
they are, so the actual context length is a product of two other knobs and can
blow past the model's window (or its useful window) without either knob looking
wrong. The budget is the backstop, and it makes the failure explicit.

## When it is useful
- **Raise it** with long chunks and a large-context model, when the Inspector
  shows chunks being dropped that the answer needed.
- **Lower it** to test whether a long context is helping at all — a frequent
  finding is that a tighter, better-ordered context beats a bigger one.
- **Watch it whenever `retrieval.k` rises**: a `k` of 20 at 500-character chunks
  is 10,000 characters, so the effective `k` silently becomes 12.

## Interactions
Interacts with `retrieval.k` (the budget can cut below it), `index.chunk_chars`
(chunk size), and the summary knobs (a summary is another unit competing for the
budget).
