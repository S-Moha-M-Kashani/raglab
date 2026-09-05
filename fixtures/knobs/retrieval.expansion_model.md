# retrieval.expansion_model — which model writes the HyDE hypothetical answer

- **Step:** Retrieval (model role `expand`). **Default:** `''` = the lab's
  configured model. **Only consulted when** `retrieval.hyde` is on.

## What the knob does
Names the model that generates the pseudo-answer HyDE searches with. Multi-query
expansion is rule-based and uses no model at all; this role exists only for
HyDE.

## What it means scientifically
The expansion model's job is not to be *right* — it is to produce text in the
same distribution as the corpus. That reframes what "a good model here" means:

- **Style and language fidelity matter more than reasoning.** The generated
  passage is thrown away after embedding; only its surface form is used. A small
  model that writes fluent text in the corpus's language beats a strong reasoner
  that writes in the wrong register.
- **Hallucination is only harmful when it is specific.** A generic plausible
  paragraph retrieves the right neighbourhood; an invented proper noun or figure
  is a confident wrong turn, because dense retrieval will happily find whatever
  is nearest to it.
- **It sits in the query path**, so its latency is paid on every question — the
  same cost structure as the reranker model, and quite unlike a judge that runs
  once per answer.

## Why RAG architectures have this knob
Because each stage should be able to name its own model. A single global model
choice couples an expansion step (cheap, high volume, style-sensitive) to a
judging step (low volume, accuracy-critical), and the right model for the two is
rarely the same. Recording the model per stage is also what makes a row
reproducible.

## When it is useful
- **Pick the cheapest model** that writes the corpus's language convincingly;
  this is high-volume, low-stakes generation.
- **Under a CLI backend**, remember a call is unbilled but not free — process
  spawn plus wall clock on every question — so the lightest alias belongs here.
- **Leave it blank** to inherit the lab default when you are not studying this
  stage.

## Interactions
Inert unless `retrieval.hyde` is on. Independent of `generation.model`, and
deliberately so: a model good at answering from context is not automatically
good at imitating a corpus's voice.
