# index.split_plan — where a document is cut, as an ordered list of stages

- **Step:** Index. **Fingerprinted:** yes — the whole list, in order.
- **Default:** `document / drift`.
- **Stages:** `document` (always first, cannot be removed), `part`, a label
  boundary (`role=user`), a separator (`"\n\n"`, `". "`), `drift`.

## What the knob does
A plan is a list of stages. The first is always the document — no chunk ever
spans two documents, and stating it makes the rule visible. Each later stage
subdivides the pieces the stage before it produced, and the budget
(`index.chunk_chars`) closes the plan: whatever is still too big after the last
stage is divided at word boundaries to fit.

- `part` keeps one piece per part of the document.
- A **label boundary**, `role=user`, opens a new piece at every part carrying
  that label and runs it to the part before the next one. It names only the
  opener, so a question and its replies stay together whatever the other
  speakers are called, and it works on a corpus with three speakers. It cuts
  at part boundaries, so every part-level label the corpus declares still
  reaches the chunk.
- A **separator** is a literal string — `"\n\n"`, `"\n"`, `". "` — that cuts
  text. It is not a regular expression and not sentence detection: a corpus
  that wants sentences supplies `". "` itself. A separator cuts *only a piece
  still over the budget* by default, so a paragraph that fits is left whole;
  once text has been cut inside a part, no part span survives.
- `drift` cuts where consecutive parts stop resembling each other — the bottom
  third of that document's own similarity distribution, so no absolute
  threshold is assumed — at a ceiling of twice the budget, and at any
  topic-change markers it is given (`drift or "ولش کن"`). None are built in.

Within one stage, atoms combine with `or` (cut wherever any matches) or with
`and` (cut only where every one holds), never both — so no precedence rule is
needed. Coarse-to-fine is written as successive stages: `document / "\n\n" /
"\n" / ". "` applies the blank line first and the single newline only within
the paragraphs that resulted. `"\n\n" and role=assistant` cuts at a blank line
only inside a part the label selects.

Each stage says when it applies: `always`, or only `over-budget`. Structural
stages default to always, separators to over budget; either is overridable
(`part over-budget`, `"\n\n" always`).

## What the lab refuses
- A plan without the document stage first.
- A stage mixing `or` and `and`.
- A `drift` or label stage after a separator — drift compares parts through
  their embeddings and a label boundary needs part identity, and once text
  has been cut mid-part neither exists.
- A label boundary on a label the selected corpus never declares at the part
  level, or a value outside the label's declared set. Cutting nothing and
  saying so nowhere would be a row lying about what produced it.

## What it means scientifically
Chunking is the choice of **indexing unit**, the oldest question in passage
retrieval. An embedding is one fixed-length vector: a chunk that mixes two
topics has a centroid near neither and is retrievable by neither query. A
label boundary encodes the adjacency-pair structure of dialogue; a list of
separators is recursive character splitting, boundary-preserving rather than
offset-driven; drift is TextTiling's idea with a relative threshold, which is
what makes it portable across documents of different densities.

## When each plan is useful
- **Dialogue, tickets, interviews:** `document / part` or `document /
  role=user` — structure already marks the semantic unit, so a character grid
  would destroy it.
- **Continuous prose, reports, diaries:** `document / drift` — boundaries are
  latent and worth detecting.
- **Sections and paragraphs:** `document / "\n\n" / ". "` with
  `index.part_join` set to a blank line so the paragraph break can match
  between parts.
- **Baselines and ablations:** `document` alone with a budget is the honest
  control — plain word packing — and with `index.overlap` the sliding window;
  `document` at a budget larger than any document is the "no chunking"
  extreme that shows how much chunking bought.

## Interactions
`index.chunk_chars` closes every plan and decides whether an over-budget stage
cuts at all; `index.chunk_unit` says what it counts; `index.overlap` repeats a
tail wherever the budget divides. `index.part_join` decides whether a
separator can match between two parts, and `index.part_prefix` what text a
part contributes. Chunk size then sets how many pieces `retrieval.k` and
`retrieval.max_context_chars` can afford.
