---
name: multilingual-rag
description: Build retrieval that works in a language other than English — choosing an encoder that can represent the script at all, handling cross-lingual query-document mismatch, and avoiding the tokeniser traps that make a pipeline score confidently on nothing. Use whenever the corpus, the queries or both are not English, and especially for lower-resource languages such as Persian, where an English-tuned component can fail silently rather than loudly. Covers encoder selection and benchmarks, the same-language retrieval bias, cross-lingual retrieval, prompt and header language, and evaluation in the corpus language.
---

# Multilingual RAG

**What it is.** Everything in a RAG pipeline was probably tuned on English. Each
component degrades differently outside it, and the dangerous property they share
is that **most of them fail silently** — returning a number rather than an error.

## The tokeniser trap, which comes first

Before any modelling question: can your embedder represent the script at all?

A tokeniser with an ASCII-only pattern — `[a-z0-9]+` and similar — produces *zero
tokens* for Persian, Arabic, Chinese, Hebrew, Greek, Cyrillic. Zero tokens means
the zero vector. Every document embeds identically, cosine similarity is
undefined or constant, and retrieval returns whatever the index iterates first.
The pipeline runs. It produces scores. The scores mean nothing.

This is not a hypothetical: a hash embedder with exactly that tokeniser has been
measured at roughly **0.01 recall against 0.617** for a real Persian encoder on
the same corpus — about **60×** — where no other knob in an entire sweep was worth
2%.

The check is one line and belongs in the test suite, not in a review: embed a
string in the target script and assert the vector is not all zeros.

## Choosing an encoder

Order of preference:

1. **A model trained for the language**, if a good one exists. For Persian,
   `heydariAI/persian-embeddings` and `MatinaSRoberta` are the current names, the
   latter reported as outperforming previous Persian embeddings on retrieval
   accuracy and contextual relevance.
2. **A strong multilingual model** — the multilingual E5 family, BGE-M3,
   Jina-ColBERT-v2 for late interaction. Broad coverage, usually behind a good
   monolingual model on that language.
3. **An English model on translated text** — translate the corpus or the query.
   Adds a translation stage that can fail, and translation error compounds with
   retrieval error.

**Verify the benchmark, do not trust the model card.** Language-specific MTEB
variants exist for this: `FaMTEB` is the massive Persian text embedding
benchmark, and it is what should decide a Persian encoder rather than a general
multilingual leaderboard position.

**Respect the encoder's input contract.** E5-family and instruction-tuned models
need `query:` / `passage:` prefixes. Dropping them is a silent accuracy loss in
any language, and it is easier to lose when the surrounding text is not in the
prefix's language.

## Same-language bias and cross-lingual retrieval

Multilingual retrievers strongly prefer to retrieve documents in the *query's*
language. Measured: 80.8% Arabic retrievals for an Arabic test set, 76.4% Hindi
for Hindi. Persian queries were observed retrieving primarily from German and
French data, which is worth pausing on — the bias is real but not reliable, and
it is not a language *understanding*.

This matters because of where knowledge lives. For a lower-resource language, the
answer to a factual question is often only written down in English. So:

- **Monolingual retrieval is marginally useful or actively harmful** for
  knowledge-intensive questions in low-resource languages, because the retriever
  is confined to the smaller, thinner half of the corpus.
- **Cross-lingual RAG improves knowledge-intensive tasks for both high- and
  low-resource languages** — retrieve across languages, generate in the user's.

The caveat: this applies to *world knowledge*. If the corpus is personal,
proprietary or in-domain — a diary, a company's tickets, a legal archive — the
answer exists only in that corpus, in its own language, and cross-lingual
retrieval has nothing to fetch. Same-language retrieval is correct there, and the
finding above does not transfer.

## The parts of the pipeline that are secretly in a language

Audit every string that gets embedded or prompted:

- **Chunk headers and speaker tags.** These are prepended before embedding. A
  Farsi header over an English corpus adds a constant foreign phrase to every
  vector — a systematic bias affecting every comparison equally, which makes it
  invisible in relative scores and real in absolute ones.
- **Prompts.** Answer language, refusal strings, and grader instructions.
  A generator prompted in English over a Persian corpus will often answer in
  English regardless of the user's language.
- **The canonical refusal.** If it is in one language and the corpus is in
  another, faithfulness scoring sees a language switch as well as a refusal.
- **Text normalisation.** Persian and Arabic need normalisation of visually
  identical codepoints (Arabic ye/kaf vs Persian, zero-width non-joiner,
  diacritics). Without it, exact-match metrics and BM25 undercount.
- **Numerals.** Eastern Arabic digits and Western digits are different
  codepoints; any numeric matching must normalise them or it silently fails.

## Evaluating in the corpus language

An LLM judge is a model too, and its quality varies by language far more than its
confidence does. A judge that is weak in the target language produces scores that
look normal and separate nothing.

Screen the judge in the corpus's own language before trusting it to rank
anything — see `rag-evaluation`. Also keep at least one control corpus in a
different language: it is the only way to tell a finding *about retrieval* from a
finding *about this language*.

## In this lab

- **The founding measurement is the tokeniser trap**: `ascii-hash` embeds Farsi to
  the zero vector, ~0.01 recall against 0.617 for `heydariAI/persian-embeddings`.
  That measurement moved production's default embedder and retired `hash` **by
  name**, so a stale config raises rather than silently selecting a replacement.
  `test_primitives.py::test_ascii_hash_embedder_is_blind_to_farsi` still asserts
  the zero vector — the measurement is what justifies paying 2.2 GB for the real
  encoder.
- **The default is `sentence-transformers` + `heydariAI/persian-embeddings`**,
  which is why the launch line carries `--extra local-embeddings`. A model picked
  against the wrong backend is a validation error, never a silent fall back — a
  run labelled one encoder that measured another is the worst artefact possible.
- `EMBEDDER_HINTS` states language coverage per model, because an English-only
  embedder returns confident numbers that measure nothing on Farsi.
- `embedding.query_vectors()` enforces the E5 prefix contract.
- `textnorm.py` is the shared normaliser — a **vendored copy** of the production
  tokeniser, carrying the commit it came from, and able to drift with nothing to
  notice. Re-snapshotting is a deliberate act.
- `chunking.SPEAKERS` / `HEADERS` default to Farsi and are written in the corpus's
  own language, exactly for the reason above.
- **The four control corpora exist for this**: English support tickets, German
  meeting notes, English research notes, and a five-session smoke set. Every
  finding here is a finding about a Farsi diary until a second corpus says
  otherwise — some are obviously general (the zero-vector result), some obviously
  are not (the Farsi time-scope filter), and with one corpus there is no way to
  tell which is which.
- **The cross-lingual finding does not apply here.** The diary is personal and
  synthetic; its answers exist nowhere else. Same-language retrieval is correct.

## Sources

- [FaMTEB: Massive Text Embedding Benchmark in Persian Language](https://arxiv.org/pdf/2502.11571)
- [Advancing RAG for Persian: Language Models, Benchmarks, and Best Practices](https://aclanthology.org/2026.lrec-1.580/)
- [Multilingual RAG for Knowledge-Intensive Tasks](https://arxiv.org/html/2504.03616v1)
- [Investigating Language Preference of Multilingual RAG Systems](https://arxiv.org/pdf/2502.11175)
- [On the Consistency of Multilingual Context Utilization in RAG](https://arxiv.org/pdf/2504.00597)
- [Multilingual RAG for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness](https://arxiv.org/html/2410.01171)
