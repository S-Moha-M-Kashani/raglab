# index.normalizer — which text normaliser the lexical stages tokenise with

- **Step:** Index. **Fingerprinted:** yes, dropped from the payload at its
  default. **Default:** `""` — the corpus's declared language decides.
  **Values:** `""`, `persian`, `neutral`.

## What the knob does
Every lexical stage of the build — BM25, the two hash embedders, the
hierarchy's rare-word edges and extractive summaries, a drift stage's
markers — tokenises text through one normaliser, and a query is tokenised
through the same one. `persian` folds Arabic letterforms and Persian and
Arabic-Indic digits to one spelling, strips harakat and drops a short Persian
stop list — the normaliser the lab applied to every corpus before this knob
existed. `neutral` normalises Unicode (NFKC) and folds nothing. Left empty, a
corpus declaring `fa` gets `persian` and every other language `neutral`. An
unknown name is refused, never replaced by another.

## What it means scientifically
Normalisation decides which spellings are one term. Folding is right where the
variants are one word (`ي`/`ی`, `۱۴۰۵`/`1405`) and wrong where they are not;
a stop list written for one language deletes content words in another. A
lexical retriever's whole claim — exact names, numbers, rare terms — rests on
the tokeniser matching the corpus's own conventions.

## When it is useful
- **Leave it** for the bundled corpora: each declares its language.
- **Name `persian`** for a Farsi corpus that declares another language code,
  or to measure whether Persian folding helps a corpus it was not written
  for — the default is a default, not a lock.
- **Name `neutral`** to remove the folds from a Farsi corpus and measure what
  they bought.

## Interactions
Reads the corpus's `language`; changes the tokens `retrieval.retriever`'s
BM25 side scores, what `index.embedder`'s hash kinds hash, and the edges
`index.graph_source` builds from rare words.
