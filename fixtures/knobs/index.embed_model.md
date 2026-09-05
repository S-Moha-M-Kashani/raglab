# index.embed_model — which embedding checkpoint the chosen backend loads

- **Step:** Index. **Fingerprinted:** yes (blanked when the embedder loads no
  model, so it cannot cost a rebuild it does not cause).
- **Default:** `''` — whatever the chosen backend recommends.

## What the knob does
Names the concrete encoder. This is where a non-English corpus is won or lost:
the best-known checkpoints (bge-small-en, all-MiniLM-L6) are English-only and
will return confident numbers on text they cannot read. Every entry in the
catalogue states its language coverage, and multilingual ones are listed by
name. The E5 family needs its `query:`/`passage:` prefixes, which the lab
applies automatically.

## What it means scientifically
Three distinctions matter more than parameter count:

1. **Tokenizer and pretraining coverage.** A model whose vocabulary never saw a
   script fragments it into unknown pieces; the resulting vectors are noise
   with a valid norm. Coverage is a hard prerequisite, not a quality dimension.
2. **Training objective.** Retrieval encoders (the E5/GTE/BGE families) are
   trained contrastively on query–passage pairs, i.e. for **asymmetric** search
   where a short question must match a long passage. Paraphrase/STS models are
   trained for **symmetric** similarity between two sentences of the same kind.
   Using an STS model for retrieval is a task mismatch, and it usually shows up
   as a plausible-but-flat ranking.
3. **Prompt/prefix conventions.** E5-style models were trained with role
   prefixes; dropping them shifts every vector off the manifold the model was
   optimised on. This is a silent, systematic error, not a small one.

Benchmarks such as MTEB rank checkpoints in aggregate, but the aggregate hides
language and domain: bigger is not automatically better, and a small
language-tuned encoder routinely beats a large general one on its own language.

## Why RAG architectures have this knob
The encoder defines the geometry that every later stage inherits. A reranker can
reorder what was retrieved; it cannot retrieve what the encoder made invisible.
Making the checkpoint explicit — and recording it on every row — is what lets
two runs be compared at all.

## When it is useful
- **Always state it explicitly** for a corpus that is not English.
- **A/B within a family** (small vs base) to price capacity against latency.
- **Across families** (retrieval-trained vs paraphrase-trained) when scores look
  oddly compressed.

## Interactions
Only meaningful for `index.embedder` in {`sentence-transformers`, `fastembed`}.
Changing it rebuilds the index, because it changes what is stored.
