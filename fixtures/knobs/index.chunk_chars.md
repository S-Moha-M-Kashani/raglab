# index.chunk_chars — how large one chunk may be, in characters

- **Step:** Index. **Fingerprinted:** yes. **Default:** 500.
- **Read by:** the three length-based chunkers only (`semantic-drift`, `fixed`,
  `fixed-overlap`); the structural ones grey it out.

## What the knob does
For `fixed` and `fixed-overlap` it is the target size of a piece. For
`semantic-drift` it is a maximum: a segment is cut once adding the next part
would take it past twice this value.

## What it means scientifically
This is the classic **granularity trade-off** of passage retrieval, and it is
one dial moving two quantities in opposite directions:

- Small chunks concentrate a single claim, so the embedding is sharp and
  precision is high — but the sentence that actually answers the question is
  often in the neighbouring chunk (recall loss), and the chunk may be
  uninterpretable on its own.
- Large chunks almost always contain the answer (high recall) but dilute it:
  the vector is an average over many claims, similarity to any one query drops,
  and the answerer receives more distracting text per retrieved slot.

There is a signal-processing reading too: the chunk is the window, and window
length trades resolution against stability, exactly as in spectral estimation.

## Why RAG architectures have this knob
Because the reader's context is finite and the retriever's scores are
comparisons between whole units. Chunk size is how a pipeline chooses where to
sit between "many precise fragments" and "few complete passages", and the right
point depends on how densely the corpus states facts.

## When it is useful
- **Raise it** when answers are being cut in half, when chunks read as
  fragments, or when the corpus states one fact over several sentences.
- **Lower it** when judged context precision is poor while recall is fine, or
  when a few long chunks are eating the whole context budget.
- **Sweep it early**: it is cheap to reason about and it interacts with almost
  everything downstream, but note that it is an index knob — each value is a
  rebuild, unlike the retrieval knobs.

## Interactions
Sets the effective corpus size in chunks, which changes `index.graph_knn`'s
connectivity, the cluster count derived from `index.granularity`, and how many
chunks fit under `retrieval.max_context_chars` at a given `retrieval.k`.
