# index.part_prefix — which declared label, if any, is written in front of a part's text

- **Step:** Index. **Fingerprinted:** yes, dropped from the payload at its
  default. **Default:** none.

## What the knob does
Names a part-level label the corpus declares — `role`, `speaker` — whose value
is written at the start of each part's text (`user: …`). By default nothing
is: a part is indexed exactly as the corpus recorded it. The corpus template
asks that a part's text carry no speaker prefix and no markup, since anything
an embedder should not read as content belongs in labels, and the lab holds
itself to the same rule. A label the selected corpus does not declare at the
part level is refused.

## What it means scientifically
A prefix is text the embedder reads: it moves every vector towards the word
`user` or `assistant` and away from what was said. The information it carries
already reaches every chunk as a label, where it can filter and group without
polluting the vector. Whether a pipeline gains from the speaker being inside
the vector — a reranker or an LLM reading the context sees it there — is a
measurable question, which is why the knob exists rather than the assumption.

## When it is useful
- **An LLM answerer** that needs to know who said what inside the context it
  is handed.
- **Reproducing an older row** built when the lab wrote `role:` in front of
  every part.

## Interactions
Changes the text every stage of `index.split_plan` cuts and every embedder
reads; the chunk's labels are unchanged by it.
