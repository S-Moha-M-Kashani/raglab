# index.dataset — which corpus the experiment measures against

- **Step:** Index. **Fingerprinted:** yes — changing it rebuilds the index.
- **Values:** a dataset id: `diary-en` (bundled default, a synthetic one-year
  English diary, 167 sessions, 112 ground-truth questions), `diary-fa` (its
  Farsi original, and what an empty value still means for legacy
  fingerprints), plus smaller control corpora and anything imported through
  the panel. The id is storage identity, never a file path.

## What the knob does
Selects the corpus **and** the ground truth measured against it — the two are
paired by id. It is the first thing the leaderboard groups by, and an index
built over one corpus can never be handed a question from another, because the
corpus id is inside the index fingerprint.

## What it means scientifically
This is the knob that sets **external validity**. Every other knob's effect is
measured *conditional on a corpus*: a chunker that wins on dialogue transcripts
is not thereby a better chunker. In IR terms, the dataset is the test
collection, and a result on one collection is a single observation, not a law.
The BEIR line of work exists precisely because retrieval rankings reshuffle
when the domain changes; heterogeneous zero-shot evaluation is the standard
answer to overfitting a pipeline to one corpus.

## Why RAG architectures have this knob
A RAG system is not evaluable in the abstract. Retrieval quality depends on
document length, vocabulary, language, redundancy, whether documents carry
dates, and how questions relate to evidence. Making the corpus a first-class
knob turns "does this help?" into "does this help *here*, and does it still
help *there*?".

## When it is useful
- **Replication across corpora** is the main use: run the same configuration on
  a second corpus to tell a general finding from a fact about one dataset.
- **Language coverage checks**: the bundled set spans Farsi, German and English,
  which is how an English-only embedder gets caught returning confident
  numbers that measure nothing.
- **Cheap iteration**: prefer the small smoke corpus while the pipeline is being
  wired, and the full corpus only when the claim is about the corpus.

## Interactions
Groups the leaderboard (`by_dataset()`), defines a sweep's comparability class
(`group()` partitions by dataset, question set and judge before any winner may
be named), and gates several knobs: no date-time label in the corpus greys out
`retrieval.time_filter` and `retrieval.recency_half_life_days`; no ranks label
neutralises `retrieval.agentic_weights`.
