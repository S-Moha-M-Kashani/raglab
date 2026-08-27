# RAG lab

RAG lab is an offline-first retrieval workbench. It measures chunking,
retrieval, reranking, embedding, and generation choices against corpora with
known answers, so architecture decisions are based on evidence rather than
preference.

The repository includes six datasets: `diary-en`, `diary-fa`, `support-en`,
`meetings-de`, `research-multihop`, and `smoke-mini`. The fresh panel default
is `diary-en`; `diary-fa` remains the legacy identity for older fingerprints.

## Quick start

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), and Node.js for
the browser-contract tests. Docker Compose is optional.

```sh
uv sync --extra local-embeddings
cp .env.example .env
uv run raglab
```

Open <http://localhost:9002>.

The default backend is local Ollama:

```sh
ollama pull 4skl/gemma4-e2b-mtp
```

Set `RAGLAB_LLM` in `.env` to choose another backend:

- `ollama` — local model server; set `RAGLAB_MODEL` if needed.
- `openrouter` — remote model; set `OPENROUTER_API_KEY`.
- `claude` or `codex` — a locally installed and authenticated CLI.
- `fake` — offline test backend only.

The first local index build downloads the embedding model, approximately
2.2 GB. The panel can start before a model is available; answering and judging
require a working backend.

## Run an experiment

1. Start the panel and choose or import a dataset.
2. Configure the index, retrieval, and generation settings.
3. Press **Build**, then **Run evaluation**.
4. Change one knob at a time and run again to make comparisons meaningful.
5. Open an experiment to inspect its retrieved evidence, answers, and metrics.
6. Export a completed experiment when you need to share its settings and
   evidence.

The four deciding metrics are faithfulness, answer relevancy, context
precision, and context recall. A ranking is valid only within the same dataset,
question set, and judge. The leaderboard groups experiments by dataset but
does not claim a universal winner.

## Datasets

Built-in datasets are ordinary corpus/ground-truth JSON pairs under
`fixtures/corpus_groundtruth_datasets/`. Import your own pair from the panel
or through `POST /api/datasets`. The pair must satisfy the declared dataset
contract; invalid data is rejected rather than repaired silently.

## The Ask widget

The **✳ Ask** helper is available on the Laboratory, Inspector, and Leaderboard.
It answers questions about the project, RAG techniques, and recorded
experiments. OpenRouter models can use the experiment and skills tools. Claude
and Codex CLI models are keyless but cannot use tools or retain graph
conversation state.

The widget keeps three kinds of memory in its local SQLite store:

- Short-term memory: the current `thread_id`, normally one experiment ID or
  `general`.
- Dataset memory: accepted summaries from all experiments on one dataset,
  including extracted subtopics such as reranking or semantic drift.
- Global memory: compact patterns that generalize across validated datasets.

Only relevant RAG-lab questions may enter long-term memory. A policy step
decides relevance, dataset, subtopic, and whether saving is allowed. Irrelevant
questions are refused and are not persisted. The summary writer runs after the
answer is delivered.

### Developer trace page

Set `RAGLAB_DEV_KEY` before starting the server and open
`/dev/trace?key=<that key>` to see every widget conversation step by step:
the system lines the model was handed, the question, each tool call with its
arguments and reply, the answer, and a token account where one was reported.
`&thread=<id>` opens one thread; without it the page lists them. It reads the
conversation log and writes nothing, and it is a 404 whenever the key is
unset or wrong.

## Stored data

All durable application data is local and git-ignored:

- `.runs/` — one JSON result per evaluation.
- `databases/raglab.db` — experiment/job ledger.
- `databases/corpora.db` — content-addressed imported corpora.
- `databases/widget.db` — widget checkpoints, readable turn logs, and long-term
  memory. Set `RAGLAB_WIDGET_DB` to override it.
- `.datasets/` — imported dataset files.

Inside `widget.db`:

- `checkpoints` and `writes` are internal LangGraph state tables.
- `widget_turn_log` is the readable one-row-per-question audit log, including
  messages, tool steps, token totals, latency, status, and memory link.
- `dataset_memory` stores one accumulated summary per dataset.
- `global_memory` stores cross-dataset patterns.
- `memory_updates` stores summary provenance: experiment, dataset, subtopic,
  question, and answer.

The widget never writes experiment runs, ledger rows, scores, or ranking data.
Its OpenRouter key, when entered in the panel, stays in process memory only.

## Commands

```sh
uv run raglab                         # serve the application on :9002
uv run raglab-lab --test-only         # run the offline preflight suite
uv run raglab-sweep                   # run one-knob-at-a-time candidates
uv run raglab-judgescreen --models MODEL
uv run raglab-leaderboard             # print recorded experiments
```

Extra tools are under construction.

## Development and testing

The canonical offline check is:

```sh
uv run pytest src/raglab -q
```

It uses the `fake` backend, temporary databases, and blank credentials. The
intentional live probes are excluded unless run explicitly:

```sh
uv run pytest src/raglab/agents/widget/tests/test_skills_live.py -v -s
```

Browser contracts can be run directly from their directory:

```sh
cd src/raglab/dashboard/tests
node --test panel_open.test.js board_reveal.test.js
```

Branch from `development`; `master` is the squash-merged release branch.

## Docker

```sh
mkdir -p .runs databases .datasets
docker compose up -d --build
```

Open <http://localhost:9002>. Stop it with:

```sh
docker compose down
```

The container uses host Ollama through `host.docker.internal:11434` by
default. Embedding data is kept in the named `hf-cache` volume.
