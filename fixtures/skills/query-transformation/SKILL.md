---
name: query-transformation
description: Rewrite the query before it reaches the retriever — HyDE's hypothetical answer, multi-query fan-out, step-back abstraction, decomposition into sub-questions, and conversational rewriting. Use when the query and the documents are written in different registers: a short keyword query against prose, a follow-up that depends on the previous turn, or a comparison question whose two halves live in different documents. Covers what each transform fixes, the per-query LLM cost that separates them from index-time methods, and when a transform makes retrieval worse.
---

# Query Transformation

**What it is.** Everything else in a RAG pipeline changes the documents or the
scoring. These change the *query*, at run time, before retrieval happens.

The premise is an asymmetry: users write short, elliptical, context-dependent
questions, and corpora contain long declarative prose. Embedding a five-word
question and a three-hundred-word passage and expecting their vectors to be close
is optimistic. Each transform below closes that gap in a different direction.

The shared cost is the thing to keep in view: these are **per-query LLM calls**,
paid on every question forever, unlike index-time methods which are paid once.
They also add latency directly to the user-visible path.

## HyDE — Hypothetical Document Embeddings

Ask the model to *answer* the question — inventing freely, with no retrieval —
then embed that hypothetical answer and use it as the query vector. Throw the
text away; only its embedding is used.

The insight is that a wrong answer is still written in the *shape* of the right
one: same domain vocabulary, same register, same length. It lands in the
neighbourhood of real answer passages in a way the bare question does not.

Use it when the query is too short or too keyword-like to retrieve well. Its
failure mode is exactly its mechanism — on a domain the model knows nothing
about, the hypothetical document is confidently off-topic and drags the query
vector into the wrong region. It is also strictly worse than the plain query when
the corpus is written in question form (FAQs, ground-truth pairs).

## Multi-query fan-out

Generate N reformulations of the question, retrieve for each, and fuse the result
lists (RRF is the usual merge — see `hybrid-retrieval-fusion`). Different
phrasings surface different documents; consensus across them is a decent
relevance signal.

Costs one generation plus N retrievals. It is the most reliably positive of these
transforms and the least clever.

## Step-back prompting

Ask a more general question first — abstract to the underlying principle or
category — retrieve for that, and use it as background beside the specific
retrieval. Use when the user asks about a specific instance but the corpus
explains the general rule elsewhere.

## Decomposition

Break a compound question into atomic sub-questions, retrieve for each, and
assemble. This is the right tool for comparisons ("how did A differ from B") and
multi-hop questions where the bridge entity is only named in the first hop's
result. Decomposition can be a single up-front split or interleaved with
retrieval, at which point it has stopped being a transform and become a loop
around retrieval — a different technique, with a different cost model.

## Conversational rewriting

In a multi-turn session, resolve the query against the history into a
self-contained question before retrieval: "and the year after?" becomes "what
happened in 2019?". Cheap, unglamorous, and usually the highest-value transform
in any chat-shaped product, because without it every follow-up retrieves noise.

## Choosing whether to transform at all

Transforming every query pays the cost on the many queries that did not need it.
The usual trigger conditions are cheap to compute: query length, presence of
multiple clauses or question marks, no history to resolve, and — best of all —
low first-stage retrieval confidence, which means the transform is only paid when
plain retrieval already looked weak. That routing decision is itself a topic:
whether to retrieve at all, and what to do when the evidence comes back bad.

## The encoder contract, which transforms break quietly

Whatever you send the retriever must respect the embedding model's input format.
Asymmetric encoders — the E5 family and most instruction-tuned retrievers — expect
`query:` on the query side and `passage:` on the document side, and silently lose
accuracy if you swap or drop them. A transform stage that builds a new query
string and forgets the prefix has introduced a loss that no error message
reports.

The same applies to HyDE in particular: the hypothetical document is a *document*
in register but is being used as a *query*. Which prefix it should carry is an
empirical question, and both answers are defensible — just not both at once,
unmeasured.

## In this lab

- `embedding.query_vectors()` is where the E5 prefix contract is enforced. Any
  query transform added here must route through it, not around it — losing the
  prefix is a silent accuracy loss, which is the exact class of failure this lab
  exists to catch.
- **Both headline transforms are already knobs on `RetrievalConfig`**:
  `multi_query` (default **True** — fan-out is part of the baseline here, so the
  interesting candidate is the one that turns it *off*) and `hyde` (default
  False, with `expansion_model` naming which model writes the hypothetical).
  Both are retrieval fields, outside the index fingerprint, so they sweep free
  against one build.
- **The agent's `retrieve` scope contains a second, conditional transform**: its
  loop is plan → retrieve → judge the evidence → rewrite → retry, so the rewrite
  step is a retrieval-confidence-triggered reformulation with a stopping rule.
  A `multi_query`/`hyde` candidate and a `scope='retrieve'` candidate are
  therefore two prices for the same idea — unconditional and cheap against
  conditional and expensive — and are worth reading side by side.
- The lab has a Farsi time-scope filter on the query side already, which is a
  reminder that some query-side machinery is corpus-specific rather than general
  — the four control corpora in `fixtures/corpus_groundtruth_datasets/` exist to tell
  those two apart.
- **Missing here**: step-back as its own knob, decomposition outside the agent
  loop, and conversational rewriting — the last because the lab evaluates
  single questions, not multi-turn sessions.

## Sources

- [ARAGOG: Advanced RAG Output Grading](https://arxiv.org/pdf/2404.01037)
- [Query Transform: HyDE, Multi-Query, Step-Back](https://neelmishra.github.io/blog/mlops/rag/query-transformation.html)
- [Hypothetical Document Embeddings (HyDE) — Haystack docs](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
- [RAGSmith: Finding the Optimal Composition of RAG Methods Across Datasets](https://arxiv.org/pdf/2511.01386)
