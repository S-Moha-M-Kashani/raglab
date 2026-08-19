---
name: chunking-strategies
description: Choose how documents are split before indexing — fixed-size with overlap, recursive, semantic-drift, structure-aware, late chunking, parent-document — and know what the empirical studies actually found about whether the clever ones beat the simple ones. Use when setting up a new corpus, when retrieval returns fragments that are individually useless, or when someone proposes semantic chunking as an obvious improvement. Covers the strategies, the measured cost-benefit, why the answer is corpus-dependent, and how chunk size interacts with every other knob in the pipeline.
---

# Chunking Strategies

**What it is.** The first irreversible decision in a RAG pipeline. Everything
downstream — what a vector can represent, what a reranker can score, what the
generator is handed — is bounded by what a chunk is. And it is genuinely
irreversible in the practical sense: changing it invalidates the index, every
cached embedding, and every measurement taken against the old one.

## The strategies

**Fixed-size with overlap.** N tokens per chunk, M tokens of overlap. Overlap
exists so a fact straddling a boundary appears whole in at least one chunk. The
baseline everyone should beat, and often nobody does.

**Recursive character/token splitting.** Fixed-size, but the split point is
chosen by trying a list of separators in order — paragraph, then line, then
sentence, then word — so boundaries land on natural breaks when they can. The
default in most frameworks, and a strong one.

**Structure-aware.** Split on the document's own markup: headings, list items,
table rows, code blocks, function definitions. Where the structure exists this is
the best cheap option, because the author already decided where the units are.
Requires a parser per format, which is the real cost.

**Semantic / drift-based.** Embed sentences, walk through the document, and cut
where consecutive-sentence similarity drops below a threshold — a topic boundary.
Intuitive and popular. See the measurements below before adopting it.

**Late chunking.** Run the *whole* document through a long-context encoder first,
then pool token embeddings over chunk spans. Boundaries are ordinary, but each
chunk's vector is computed with attention over the entire document, so it carries
context the chunk text does not contain. This is the cheap non-generative cousin
of contextual retrieval — no LLM call, one long encoder pass.

**Parent-document / small-to-big.** Index small chunks for retrieval precision;
return the enclosing parent for generation context. Decouples "what matches the
query" from "what the model needs to answer", which are genuinely different sizes.

**Sentence-window.** The same idea at finer grain: index single sentences, return
the sentence plus its k neighbours.

## What the studies found

The uncomfortable summary from the 2026 empirical work: **semantic chunking's
gains are frequently not proportional to its cost**, and simple recursive
splitting is hard to beat.

- Systematic evaluations of chunking against computational cost report that
  semantic chunking improves contextual coherence and sometimes retrieval, but
  the improvement often does not justify the extra embedding pass over every
  sentence.
- A domain evaluation on enterprise documents found recursive token-based
  chunking at 100 tokens with no overlap (`R100-0`) consistently outperforming
  more elaborate approaches — strong performance at minimal overhead.
- Structurally informed and adaptive strategies *do* frequently beat fixed-size
  baselines and are more robust across corpora, at additional computational cost.
- The effect is strongly corpus-dependent. On small, focused documents that
  already match the shape of user questions, chunking at all can *hurt* accuracy.
  On long, multi-topic, messy documents, chunking is one of the largest available
  levers.

So the honest position is: chunking matters enormously, and *which clever method*
you pick matters much less than people assume. Measure on your corpus; do not
import a result from someone else's.

## Chunk size, which is the knob that actually moves

Smaller chunks give a more precise vector — one topic per vector rather than a
blurred average — and better retrieval precision. They also fragment facts, lose
referents, and force `top_k` up to compensate.

Larger chunks preserve context and reduce the number of rows, and their pooled
vector represents each individual topic worse, which is precisely the defect
late interaction exists to fix (see `reranking-late-interaction`).

The interactions to hold in mind, because a chunk-size sweep in isolation is
misleading:

- **`top_k` must be swept with chunk size**, not held fixed. Half the size means
  roughly twice the rows for the same context budget.
- **Context budget is the real constraint.** Chunk size × top_k is what the
  generator sees, and that is what its faithfulness responds to.
- **Overlap inflates the index** and creates near-duplicate retrievals that
  consume top_k slots with the same text.
- **The embedding model has a token limit.** Chunks longer than it are silently
  truncated by most encoders, and the tail is simply not represented.

## When each pays

| Situation | Reach for |
| --- | --- |
| Structured docs — markdown, HTML, code | structure-aware |
| Long homogeneous prose | recursive, size swept |
| Chunks lose their referents | late chunking, or a static header |
| Retrieved fragments individually useless | parent-document / sentence-window |
| Dialogue, records, short independent units | do not chunk; one unit per row |
| Someone proposes semantic chunking | measure it against recursive first |

## In this lab

- The chunker is an `IndexConfig` field, so it is **in the fingerprint** — changing
  it forces a rebuild, correctly, since it changes what is indexed.
- `fixed-overlap` and `semantic-drift` are both available. **Candidate F uses
  `semantic-drift`**; `PRODUCTION_CONFIG` uses `fixed-overlap` at 500/100 with a
  recursive splitter, because that is what the shipped Assistant does. The preset
  mirrors what ships, not what wins, and this is one of the two stated honest
  differences between them.
- `chunking.SPEAKERS` and `HEADERS` prepend speaker tags and a situating header
  **in the corpus's own language** — a Farsi header over an English corpus adds a
  constant foreign phrase to every vector. Both are prepended before embedding,
  so they are part of the chunk, not metadata beside it.
- The diary corpus is dialogue-shaped and session-scoped, which puts it near the
  "short independent units" row above — one reason the chunking knobs have moved
  the numbers less here than the embedder choice did.
- **Late chunking is not implemented and is the most interesting unexplored
  option here**, because it buys the contextual-retrieval effect without breaking
  the standing rule that no build calls a generative model. It needs a
  long-context encoder and a pooling step, both inside `embedding.py`.
- `chunks_by_session` in the Inspector is where a chunking change is actually
  *read* rather than scored — a chunk list is the fastest way to see that a
  splitter did something absurd.

## Sources

- [Chunking Methods on RAG: Effectiveness Evaluation Against Computational Cost and Limitations](https://arxiv.org/html/2606.00881v1)
- [A Systematic Investigation of Document Chunking Strategies and Embedding Sensitivity](https://arxiv.org/html/2603.06976)
- [Evaluating Chunking Strategies for RAG in Oil and Gas Enterprise Documents](https://arxiv.org/pdf/2603.24556)
- [Chunk Twice, Embed Once: Segmentation and Representation Trade-offs in Chemistry-Aware RAG](https://arxiv.org/html/2506.17277v1)
- [Evaluating Chunking Strategies for RAG on Academic Texts](https://arxiv.org/pdf/2607.01852)
