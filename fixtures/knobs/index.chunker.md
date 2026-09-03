# index.chunker — how a document is cut into the pieces that get embedded

- **Step:** Index. **Fingerprinted:** yes.
- **Values:** `semantic-drift` (default), `fixed`, `fixed-overlap`, `message`,
  `turn-pair`, `session`.

## What the knob does
Decides the retrieval unit. `fixed` packs words up to a character budget;
`fixed-overlap` slides a window so a sentence on a boundary appears in both
neighbours; `message` keeps one document part per piece; `turn-pair` keeps a
part together with the one that answers it; `session` stores a whole document
as one piece. `semantic-drift` cuts where consecutive parts stop resembling
each other — at the bottom third of that document's *own* similarity
distribution, so no absolute threshold is assumed — and also at a size ceiling
and at a short list of topic-change phrases (the only language-specific part).

## What it means scientifically
Chunking is the choice of **indexing unit**, the oldest question in passage
retrieval. An embedding is a single fixed-length vector: it summarises whatever
text it is given, so a chunk that mixes two topics has a centroid near neither
and is retrievable by neither query. Cutting on similarity drift is
TextTiling's idea — detect topic boundaries from a lexical/semantic cohesion
signal — with a relative rather than absolute threshold, which is what makes it
portable across documents of different densities. `turn-pair` encodes the
adjacency-pair structure of dialogue: a question and its answer are one
semantic unit even though they are two messages.

## Why RAG architectures have this knob
Retrieval can only return what was stored as a unit. Every failure mode of RAG
has a chunking version: the answer split across a boundary (recall loss), the
answer buried in a page of unrelated text (precision loss), the answer stored
without the context that makes it interpretable (an unusable hit).

## When each option is useful
- **Dialogue, chat, tickets, interviews:** `message` or `turn-pair` — structure
  already marks the semantic unit, so imposing a character grid destroys it.
- **Continuous prose, reports, diaries:** `semantic-drift` — boundaries are
  latent and worth detecting.
- **Baselines and ablations:** `fixed` is the honest control; `fixed-overlap`
  isolates the boundary effect; `session` is the "no chunking" extreme that
  shows how much chunking bought.

## Interactions
Only `semantic-drift`, `fixed` and `fixed-overlap` read `index.chunk_chars`;
only `fixed-overlap` reads `index.overlap`; only `fixed` and `fixed-overlap`
read `index.delimiters`, the boundaries they may stop a piece at — the others
cut on structure, or on a drift signal of their own, and grey all three out.
Chunk size then sets how many pieces `retrieval.k` and
`retrieval.max_context_chars` can afford.
