# index.summarizer — how a group of chunks becomes one piece of text

- **Step:** Index. **Fingerprinted:** yes, whenever a hierarchy is built.
- **Values:** `centroid` (default), `lead-idf`, `mmr`, `card`. All extractive —
  none calls a model.

## What the knob does
`centroid` concatenates the members nearest the group's centre. `lead-idf` takes
the sentences covering the most rare words. `mmr` picks members for coverage
without repetition. `card` writes no prose at all: top terms, date span, member
count, document ids.

## What it means scientifically
These are four classical **multi-document extractive summarisation** strategies,
each with a different notion of what "representative" means:

- **Centroid-based** (the MEAD tradition): the group's mean vector is its
  meaning, and the best summary is the text closest to it. Cheap and stable, but
  it systematically drops the periphery — exactly the unusual members a question
  might be about.
- **Lead + IDF coverage** (the LexRank/TextRank family of salience scoring):
  prefer sentences carrying rare, distinctive terms. This maximises the summary's
  *lexical* coverage of the group, which is what a BM25 half of a hybrid
  retriever can actually match on.
- **MMR** (Carbonell and Goldstein): greedily add the member that is most
  relevant *and* least similar to what is already selected — explicit
  redundancy control, so coverage per character is highest.
- **Card:** aggregation instead of selection. It answers a different class of
  question, because it *states* a count, a span and a set of ids rather than
  hoping the reader infers them from prose. Counting and "how often / over what
  period" questions are a known weak spot of retrieval-and-read pipelines, and a
  structured fact sheet is the cheapest available fix.

The hard constraint is that all four are extractive. An abstractive
(model-written) summary would take hours per build instead of seconds, would be
unsweepable, and — under the offline fake backend — would fill the index with
confident invention that no field in the corpus contradicts. Extractive
summaries can be wrong by omission, never by fabrication.

## Why RAG architectures have this knob
The summary layer is only as good as its summariser, and the choice changes what
the layer is *for*: centroid summaries answer "what is this group about",
lead-idf summaries are findable by keyword, MMR summaries maximise coverage, and
cards answer aggregate questions.

## When it is useful
- **`centroid`** as the default and for topical "what was going on" questions.
- **`lead-idf`** when the retriever is `bm25` or `hybrid-rrf` and summaries are
  never being retrieved — lexical findability is the problem.
- **`mmr`** for large, heterogeneous groups where a centroid would report only
  the majority theme.
- **`card`** for counting, dating and enumeration questions, and as the cheapest
  summary layer to try first.

## Interactions
`index.min_group` decides which groups get summarised at all;
`retrieval.summary_scope` decides whether these summaries are ever searched;
`retrieval.summary_levels` decides which of them may be retrieved.
