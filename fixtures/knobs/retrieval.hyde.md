# retrieval.hyde — write a hypothetical answer with a model and search with that

- **Step:** Retrieval. **Default:** off. **Costs** one LLM call per query.
- **Dataset-specific:** the prompt asks in Persian for a diary-style paragraph,
  so on another corpus it does not go quiet — it searches with text in the wrong
  language and register, which is worse than leaving it off.

## What the knob does
Generates a plausible answer to the question and uses *that text* as the search
query instead of the question.

## What it means scientifically
This is **HyDE** — Hypothetical Document Embeddings (Gao et al., 2022). The
argument is about the geometry of the embedding space: retrieval with a
bi-encoder is an **asymmetric** match between a short interrogative string and a
long declarative passage, and questions and passages occupy different regions of
the space even when they are about the same thing. Generating a fake passage
moves the query into the *document* region, making the comparison
document-to-document — a symmetric match, which is what these encoders do best.

Notably, the hypothetical answer does not need to be *correct*. It needs the
right vocabulary, topic and register; the real evidence is then retrieved by
similarity to that surface. This also explains its failure modes:

- **Hallucinated specifics** (a wrong name, a wrong date) pull retrieval toward
  the wrong neighbourhood, confidently.
- **Register or language mismatch** poisons every query, systematically. A
  Persian diary-style prompt on a German meeting corpus produces a query whose
  style matches nothing in the index.
- **Cost and latency**: one generation before every search, in the latency path
  of every question.

## Why RAG architectures have this knob
Because query-document asymmetry is a real limitation of cheap dense retrieval,
and HyDE is the best-known way to trade compute for a better query
representation. It is also the natural comparison for `retrieval.multi_query`:
same goal, one with a model and one without.

## When it is useful
- **Useful** for underspecified or jargon-poor questions on corpora written in a
  distinctive style the model can imitate, and where the encoder is a
  general-purpose one.
- **Not useful** when the question already shares vocabulary with the corpus
  (BM25 is doing fine), when latency or cost per question matters, or when the
  prompt's language does not match the corpus.
- **Always measure it against multi-query**: if the free rewriter gets most of
  the gain, HyDE is paying a model call for very little.

## Interactions
`retrieval.expansion_model` names the model that writes the hypothetical answer;
under the `fake` backend HyDE generates nonsense and measures nothing.
