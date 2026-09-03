# index.delimiters — the boundaries a piece is allowed to stop at, coarsest first

- **Step:** Index. **Fingerprinted:** yes, but dropped from the payload while
  empty, so a config that never sets it hashes exactly as it did before the
  knob existed. **Default:** empty.
- **Read by:** `fixed` and `fixed-overlap` only — the two chunkers that cut on
  a character budget and had no boundary signal of their own. `message`,
  `turn-pair` and `session` cut on structure, `semantic-drift` cuts on
  similarity drift, and all four grey it out.

## What the knob does
Gives the two character-budget splitters a list of strings they may cut on,
tried **in the order given, coarsest boundary first**. A piece that already
fits `index.chunk_chars` is kept whole; only a piece still over budget is
split again on the next delimiter. When the list runs out, what is left falls
back to the plain word packing these chunkers do today — which is also what an
empty list does, byte for byte, so the `fixed` baseline stays the baseline
every recorded run measured against.

The worked example, and the order to copy:

```
["\n\n", "\n", ". ", " "]
```

Paragraph break, then line break, then sentence end, then word. Reversing it
is the easy mistake and it is silent: split on `" "` first and the document is
words, so the coarser boundaries never match and the knob buys nothing but a
rebuild. A delimiter that never appears is inert in the same quiet way — not
an error.

The match is a **literal string**, not a regular expression (`.` is a full
stop here, not "any character") and not sentence detection: no model, no
punkt, no language rules. A corpus that wants sentences supplies `". "` — and
`"? "`, `"! "` — itself, because the lab does not know the punctuation of a
language it has never been shown.

## What it means scientifically
This is **recursive character splitting**, the same construction as
LangChain's `RecursiveCharacterTextSplitter`: a priority list of separators
applied top-down, with a size test deciding whether to descend. As
segmentation it is **boundary-preserving** — cut points drawn from a set the
author already marked, rather than from the offset where the budget ran out.

That matters because of what an embedding is: one fixed-length vector
summarising whatever text it is handed. A unit beginning mid-sentence carries
a fragment whose referents sit in the previous piece, so its vector describes
a half-thought and matches no query cleanly. Cutting on structure does not
make units smaller or larger — it makes them *coherent*, a different axis from
`index.chunk_chars` that composes with it rather than replacing it. It is also
the cheapest way to buy coherence: string search, no model call, no
dependency. `semantic-drift` buys a stronger version with an embedding
comparison per part and needs a corpus already segmented into parts; this
works on a wall of prose imported as one block.

## Why RAG architectures have this knob
Because a hard character grid discards structure already present in the text,
for free, before anything downstream can use it. Retrieval returns what was
stored as a unit, and no reranker or answerer repairs a claim cut in half at
index time — the evidence was never stored whole. It stays a knob rather than
a default because the right separators are a fact about the corpus: chat logs,
Markdown reports, transcripts and single-block prose mark their boundaries
differently.

## When it is useful
- **Useful** on prose with real paragraph structure — reports, articles,
  documentation, diaries — where `["\n\n", "\n", ". ", " "]` costs nothing,
  stops chunks opening mid-sentence, and isolates boundary quality cleanly,
  since the budget is unchanged and only the cut points move.
- **Pointless** where the corpus has no such markers (one long unpunctuated
  block, or a structural chunker doing the cutting instead) — the list matches
  nothing and the build is the one you already had.
- **Watch the piece count**: keeping paragraphs whole makes chunks less
  uniform, so the same `index.chunk_chars` yields fewer, longer pieces. Read
  it in the build stats before blaming a retrieval knob.

## Interactions
Only `index.chunker` decides whether it is read at all — `fixed` and
`fixed-overlap` and nothing else. `index.chunk_chars` is the test each piece
is measured against, so the two are one decision: the budget says how big, the
delimiters say where. With `fixed-overlap` the pieces are what the window
slides over, so `index.overlap` still repeats a tail across neighbours on top
of it.
