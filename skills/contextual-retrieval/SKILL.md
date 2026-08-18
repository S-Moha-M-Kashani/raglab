---
name: contextual-retrieval
description: Prepend a short LLM-written situating blurb to every chunk before it is indexed, so a chunk that says "revenue grew 3%" also says which company and which quarter. Use when chunks lose their referents on being split — pronouns, bare figures, "the above" — and retrieval misses them because the query names the referent the chunk dropped. Covers Anthropic's contextual embeddings plus contextual BM25, the measured failure-rate reductions, the index-time cost, and the cheaper non-LLM substitutes (static headers, late chunking).
---

# Contextual Retrieval

**What it is.** Before a chunk is embedded and indexed, an LLM is shown the whole
document and asked to write 50–100 tokens explaining where this chunk sits. That
blurb is prepended to the chunk text, and the combined string is what gets
embedded *and* what goes into the BM25 index. Nothing changes at query time.

The problem it attacks is that chunking destroys reference. A chunk reading
"The company's revenue grew by 3% over the previous quarter" cannot be retrieved
by "ACME Q2 2023 revenue" because it names neither the company nor the quarter —
the enclosing document did, and the chunk boundary threw that away.

## The mechanism

1. For each chunk, prompt: here is the whole document, here is one chunk, write a
   short blurb situating the chunk within the document. Answer with the blurb only.
2. Prepend the blurb to the chunk.
3. Index the result twice — once as a dense vector, once as BM25 terms.
4. Retrieve normally, fuse the two lists (see `hybrid-retrieval-fusion`), and
   optionally rerank (see `reranking-late-interaction`).

The two halves are separable and were measured separately: *contextual
embeddings* is step 2 feeding the dense index, *contextual BM25* is the same
string feeding the lexical index.

## What is measured

Anthropic's own numbers, on top-20 chunk retrieval failure rate — the share of
queries whose gold chunk is not in the top 20:

| Configuration | Failure rate | Reduction |
| --- | --- | --- |
| Baseline (embeddings + BM25) | 5.7% | — |
| + contextual embeddings | 3.7% | 35% |
| + contextual embeddings and contextual BM25 | 2.9% | 49% |
| + reranking on top | 1.9% | 67% |

Read the table as a stack, not as four options: each row adds to the one above.
The last row is the one usually quoted, and two thirds of its win is the first
two rows.

## Cost, which is the whole objection

This is **one LLM call per chunk at index time**. On a corpus of any size that is
the most expensive thing in the pipeline, and it is paid again on every rebuild.
Prompt caching is what makes it tractable — the document is the same across all
of its chunks, so it is cached once and only the chunk varies. Anthropic quote
roughly $1.02 per million document tokens with caching on.

Two consequences worth stating before adopting it:

- **A rebuild is now a model run.** Build time goes from seconds to minutes or
  hours, and a build can now *fail* for a reason that has nothing to do with the
  corpus. Anything that sweeps configurations will sweep far fewer of them.
- **A weak or unreachable model fills the index with confident invention**, and
  no field on a retrieval row contradicts it. The blurb is text that gets
  embedded; a wrong blurb is a permanently mis-placed chunk.

## Cheaper substitutes, in increasing order of cost

- **A static situating header.** Prepend fields the corpus already knows — date,
  speaker, source, section title — with no model involved. Captures the part of
  the win that is "the chunk forgot its metadata", which is often most of it, and
  costs nothing.
- **Late chunking.** Embed the entire document with a long-context encoder, then
  mean-pool per chunk span. Each chunk vector is computed with attention over the
  whole document, so it carries document context without any generation step. See
  `chunking-strategies`.
- **Parent-document / small-to-big.** Index the small chunk, but hand the model
  its enclosing parent at generation time. Fixes the generation half of the
  problem, not the retrieval half.
- **Full contextual retrieval.** The above table.

## When it pays and when it does not

Pays when chunks are anaphoric and the corpus is entity-dense — financial
filings, legal contracts, technical manuals, anything where "the above clause"
and bare numbers are common.

Does not pay when chunks are already self-contained: a corpus of independent
short records, FAQ pairs, or dialogue turns that each name their own subject.
Nor when the corpus rebuilds often relative to how often it is queried — you are
paying a per-chunk model call to serve a handful of questions.

## In this lab

- `chunking.HEADERS` already prepends a situating header, and `SPEAKERS` tags who
  is talking — both **static**, written in the corpus's own language, no model
  involved. That is the cheap substitute above, not Anthropic's method.
- The `contextual` chunker knob is the lab's nearest control. `PRODUCTION_CONFIG`
  has it **off**, because the shipped Assistant prepends no header, and the preset
  is a mirror of what ships rather than of what wins.
- Adopting the real thing would break the standing rule that **no build calls a
  model** (see `raglab/hierarchy.py` and the paragraph in `CLAUDE.md` about why
  all four summarisers are extractive). That rule exists because a build nobody
  can afford to sweep is a knob nobody will measure, and because the `fake`
  provider would fill an index with invention. Contextual retrieval is the
  strongest known argument against that rule, and is worth running **once**, as a
  named candidate, against the static-header baseline — not as a default.
- The honest comparison is three rows: no header, static header, LLM blurb. Only
  the third one costs a build a model, and the second is free.

## Sources

- [Contextual Retrieval — Anthropic Engineering](https://www.anthropic.com/engineering/contextual-retrieval)
- [Implementing Contextual Retrieval with async processing — Instructor](https://python.useinstructor.com/blog/2024/09/26/implementing-anthropics-contextual-retrieval-with-async-processing/)
- [How Anthropic's Contextual Retrieval changes RAG architecture](https://ninadpathak.com/blog/how-anthropics-contextual-retrieval-changes-rag-architecture/)
