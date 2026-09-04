# index.chunk_unit — what the size budget counts

- **Step:** Index. **Fingerprinted:** yes, dropped from the payload at its
  default. **Default:** `characters`. **Values:** `characters`, `tokens`.

## What the knob does
Says what `index.chunk_chars` and `index.overlap` count. `characters` is a
character count, exactly as before the knob existed. `tokens` counts the units
the selected embedding model reads — its own tokeniser's output, summed over
words — so the budget is measured in what the model actually consumes.

## What it means scientifically
Five hundred characters is a different quantity of content in every script:
Farsi, German and English do not spend letters at the same rate, and a
subword tokeniser trained mostly on English spends more tokens per word on
either of the others. A budget in model units is what makes one number mean
one amount of content across corpora, and it is the unit the embedder's own
context window is stated in.

## Why RAG architectures have this knob
Because an embedding model truncates at a token limit, not a character limit,
and a chunk that runs past it is embedded from its first half only — silently.

## When it is useful
- **Comparing corpora in different scripts** at the same budget.
- **Approaching the model's context window**, where characters mislead.
- **Refused** with a hash embedder: it sees characters and reports no model
  units, so a budget in tokens over one is refused rather than quietly
  counted in characters under the wrong name.

## Interactions
Reinterprets `index.chunk_chars` and `index.overlap`; needs an
`index.embedder` that loads a model, whose `index.embed_model` supplies the
tokeniser.
