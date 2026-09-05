# index.contextual — prepend a situating header to every chunk before embedding

- **Step:** Index. **Fingerprinted:** yes. **Default:** on.

## What the knob does
Writes a one-line header onto each chunk before it is embedded, built from the
corpus's own declared document-level labels — every label the document actually
carries, in declaration order, list values joined by the corpus's own comma
character, in the corpus's declared language. Labels the document lacks are
omitted rather than shown as placeholders, and a label that merely rates
another label's confidence is never embedded: a caveat about a label is not
content. It costs no model call and no summary.

## What it means scientifically
This is **contextual retrieval** (Anthropic's 2024 formulation): chunks are
context-dependent, and embedding them in isolation discards the discourse
context that makes them interpretable. A chunk that says "it got better" has no
referent — *anaphora* without an antecedent — so no query can match it on
meaning. Prepending the situating facts re-grounds the chunk's vector in the
document it came from.

It is also a form of **document expansion**: enriching the indexed
representation rather than the query, the same family as doc2query/docTTTTTquery,
but here the expansion is deterministic metadata instead of generated text. That
matters for this lab: a model-written header would make the build unsweepable
and would let the offline fake backend fill the index with invention that no
field contradicts.

## Why RAG architectures have this knob
Because the failure it fixes is invisible in the corpus and total in the index:
the text is present, the embedder works, and the chunk is still unreachable.
Metadata-only headers are the cheap version of the fix, so a lab can measure
whether the expensive version (a model-written context line) is worth anything.

## When it is useful
- **Strongly useful** on corpora of short, context-dependent parts — chat
  messages, diary entries, ticket comments — and whenever documents carry
  informative labels (date, participants, topic, product area).
- **Near-neutral** on self-contained documents (encyclopedic articles, standalone
  reports) whose chunks already name their subject.
- **Watch for** header text dominating short chunks: every chunk then shares a
  common prefix, which compresses the differences between their vectors.

## Interactions
Depends entirely on what the corpus declares in `label_fields` (see
`run.dataset-file`); a corpus with no document labels gets an empty header and
the knob becomes a no-op. Pairs with `index.split_plan`: the smaller the chunk, the
more the header buys and the more it can drown out.
