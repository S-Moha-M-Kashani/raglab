# retrieval.multi_query — search several rule-based rewrites of the question

- **Step:** Retrieval. **Default:** on — it is free and costs no model call.
- **Dataset-specific:** the word lists are Persian and hand-written, so on
  another corpus every rewrite collapses back to the original question and the
  knob is a no-op.

## What the knob does
Searches several variants of the question — the question itself, a keyword-only
form with the interrogatives stripped, and a synonym variant — and merges the
hits.

## What it means scientifically
**Query expansion**, the oldest remedy for vocabulary mismatch in IR. The
lineage runs from Rocchio relevance feedback through RM3-style pseudo-relevance
feedback to today's LLM query rewriting; this implementation is the rule-based,
zero-cost end of that family. Three mechanisms are at work:

1. **Interrogative stripping** removes tokens that carry no topical
   information but do carry weight in a bag-of-words match — "what", "when",
   "how many" appear in questions and never in the answering text, so they are
   pure noise for BM25 and mild noise for a bi-encoder.
2. **Synonym substitution** covers the case where the corpus and the asker chose
   different words for the same thing.
3. **Merging several result lists** is rank fusion again: a chunk retrieved by
   more than one phrasing is more likely relevant than one retrieved by a single
   lucky variant.

The deterministic version is scientifically useful precisely *because* it is
free: it establishes how much of an LLM rewriter's benefit comes from expansion
per se rather than from the model.

## Why RAG architectures have this knob
A single query string is one sample from the space of ways to ask a question,
and retrieval is brittle with respect to that sample. Averaging over several
phrasings reduces the variance of retrieval — the same reason ensembles work.

## When it is useful
- **Useful** whenever asker vocabulary differs from corpus vocabulary, and when
  BM25 is contributing significantly (the interrogative stripping mostly helps
  the lexical half).
- **Measured here:** on the bundled Farsi diary it moved quote recall from 0.489
  to 0.512 with no model call — a small, free gain on that corpus.
- **Inert** on any corpus whose language the word lists do not cover; the
  general lesson (rewrite rules are language-specific assets) is the point.

## Interactions
Multiplies the retrieval work per question (several searches), so it interacts
with `run.workers` and wall-clock. Unlike `retrieval.hyde` it needs no model, so
it is the cheap alternative to try first.
