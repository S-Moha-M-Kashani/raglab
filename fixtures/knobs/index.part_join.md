# index.part_join — the string that joins a document's parts into its text

- **Step:** Index. **Fingerprinted:** yes, dropped from the payload at its
  default. **Default:** a single newline.

## What the knob does
A document's parts are joined into one text before any stage cuts it. By
default they sit one per line. A corpus whose parts are sections rather than
turns can join them on a blank line (`"\n\n"`) — which is what lets a
separator stage naming a blank line cut between two parts at all; joined on a
single newline, `"\n\n"` can never match there, and the knob's own worked
example is inert.

## What it means scientifically
The join is part of the text the embedder reads and the separators search. It
is not content, so it should carry no content — whitespace only — but it is
the seam between parts, and a seam a separator cannot see is a boundary the
plan cannot use.

## When it is useful
- **Sections, paragraphs, articles:** a blank line, with `document / "\n\n"`.
- **Turns:** leave it — one part per line is what every dialogue plan expects.

## Interactions
Decides whether a separator in `index.split_plan` can match between parts;
`index.part_prefix` decides what each part contributes before the join.
