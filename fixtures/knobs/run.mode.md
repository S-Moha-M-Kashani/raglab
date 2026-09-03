# run.mode — where the LLM stages run

- **Step:** Run control (not a config field). **Default:** Local (Ollama).
- **Values:** Local (Ollama), OpenRouter.

## What the knob does
Picks the backend for every LLM stage. "Local (Ollama)" is free and private and
resets every stage to the lab's own defaults. "OpenRouter" switches the backend
and presets the full pipeline onto a small remote model (HyDE, LLM reranker,
relevance gate, answerer and both judges); the relevance gate prefers a
purpose-built reranker when OpenRouter's model list verifies one. **No mode
touches the index**, so the embedder stays exactly where you set it. Picking a
mode overwrites those stage choices; every knob can still be changed afterwards.

## What it means scientifically
The backend is part of the experimental apparatus, and three properties of it
affect the numbers:

- **Availability is verified, never guessed.** A model the active backend does
  not serve is a refusal, not a substitution. Otherwise a row would silently
  report a configuration that never ran — the single most corrosive thing that
  can happen to a comparison.
- **Reproducibility differs by backend.** A local model with pinned weights is
  reproducible in a way a hosted endpoint is not: remote model ids are moving
  targets, and the same slug can change behind the scenes between two runs weeks
  apart. Local runs trade capability for stability; remote runs trade stability
  for capability.
- **Cost structure differs, and it shapes what you can measure.** Judged metrics
  are the high-volume stages; a per-token backend makes the judge the dominant
  cost, while a local backend makes wall clock the constraint. That is why the
  presets change entire pipelines rather than a single field: mode is a budget
  decision as much as a model decision.

Keeping the index untouched is the deliberate separation: switching where
*generation* runs must not invalidate an index build, or every backend
comparison would also be an index comparison.

## Why RAG architectures have this knob
Because a pipeline is usually developed against something cheap and evaluated
against something capable, and the switch has to be one action rather than eight
consistent edits — while remaining fully overridable afterwards.

## When it is useful
- **Local** for development, for offline work, and for privacy-bound corpora.
- **OpenRouter** when the judged metrics need a stronger grader than the machine
  can host, and when latency per call matters less than quality.
- **Set the stage models explicitly** afterwards for any run you intend to
  publish, so the row records intent rather than a preset.

## Interactions
Presets `retrieval.hyde`, `retrieval.reranker`, `retrieval.grader`,
`generation.answerer` and both judge models. Needs `run.openrouter_key` (or the
environment variable) for the remote backend; the sweep entry points refuse the
`fake` backend outright.
