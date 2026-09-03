# `knobs/` — one page per knob of this lab

Fifty Markdown pages, one per control the panel offers: what the knob does,
what it means scientifically, why RAG architectures have such a knob at all,
which scenarios it is useful in, and which other knobs it interacts with.
They exist to be read by a person and retrieved by the panel's widget through
`search_knobs` and `read_knob` — one copy, two consumers, no drift.

Like `skills/`, this folder is **not** part of the measured seam: nothing here
changes a number. Unlike `skills/`, which is about the field, every page here
is about a control that exists in *this* lab. Numbers quoted as measured here
are facts about the corpus they were taken on, and each page says which.

## The shape a page must have

The first line is the contract:

```
# <group>.<field> — <one-line summary>
```

The filename is `<group>.<field>.md`, so the key names its own file, and the
summary after the em dash is the cheap layer the widget searches before it
pays for a whole page. A file whose first line does not match its own stem is
skipped by the loader rather than served half-read — one malformed page must
not take the corpus down. A `## Interactions` section closes every page, and
the knob keys it names are the corpus's own edges: they are what lets the
widget answer "if I raise this, what else moves" without a graph engine.

`raglab/agents/widget/knob_reference.py` is the loader and
`agents/widget/tests/test_knob_reference.py` pins the coverage in both
directions — a knob the lab explains but this folder does not, and a page for
a knob the lab no longer has, both fail the suite.

## Where the same knob lives in the code

| What | File | What it holds |
| --- | --- | --- |
| The field | `configuration/lab_config.py` | `IndexConfig`, `RetrievalConfig`, `GenerationConfig` — name, type, default, and `IndexConfig.fingerprint()`, which decides whether a change costs a rebuild |
| The allowed values | `configuration/option_vocabularies.py` | every dropdown's closed vocabulary, each leading with the default |
| The panel help | `configuration/knob_help_text.py` | `HELP['<group>.<field>']`, served over `/api/explain` |
| When it is inert | `configuration/knob_dependencies.py` | which knobs another knob's value greys out, and why |
| The model roles | `llm_backends/model_role_catalogue.py` | `ROLES` — the six `*_model` knobs, each with the `only_when` that says when its stage is consulted |

## The fifty, by step

**Index** — runs once per corpus; every one of these is inside the index
fingerprint, so a change rebuilds: `index.dataset`, `index.chunker`,
`index.chunk_chars`, `index.overlap`, `index.contextual`, `index.embedder`,
`index.embed_model`, `index.hierarchy`, `index.graph_source`,
`index.graph_knn`, `index.granularity`, `index.hierarchy_levels`,
`index.min_group`, `index.summarizer`.

**Retrieval** — runs per question against a build that already exists, so
these sweep free against one index: `retrieval.retriever`, `retrieval.k`,
`retrieval.candidates`, `retrieval.rrf_k`, `retrieval.time_filter`,
`retrieval.multi_query`, `retrieval.hyde`, `retrieval.expansion_model`,
`retrieval.mmr_lambda`, `retrieval.reranker`, `retrieval.rerank_depth`,
`retrieval.reranker_model`, `retrieval.recency_half_life_days`,
`retrieval.agentic_weights`, `retrieval.grader`, `retrieval.grade_threshold`,
`retrieval.grader_model`, `retrieval.max_context_chars`,
`retrieval.summary_scope`, `retrieval.summary_boost`,
`retrieval.summary_levels`.

**Generation** — turns retrieved context into an answer and grades it:
`generation.answerer`, `generation.model`, `generation.fact_judge`,
`generation.judge_model`, `generation.ragas_model`.

**Run controls** — belong to one evaluation run rather than to a
configuration: `run.mode`, `run.openrouter_key`, `run.ragas_mode`,
`run.ragas_limit`, `run.limit`, `run.labels`, `run.balance`, `run.workers`,
`run.label`, `run.dataset-file`.
