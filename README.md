# RAG lab

## The project

This project presents and refines a flexible RAG laboratory for users who want
to evaluate RAG approaches against their own corpus and ground-truth dataset,
with metrics and tables for each step, while leaving the decision to launch a
new experiment with the user. The laboratory accepts these datasets, runs the
selected approach, retrieves answers for comparison with the ground truth,
reports poor results thoroughly, and provides a helper assistant with
conversation memory, cost tracking, safety guards, and LangSmith tracing. The
helper responds only when asked, selects tools for the question, and,
depending on each result, chains another lookup, answers with insights, or
asks for clarification so the user can decide what to try next. It calls
OpenRouter and tools for experiment data, per-experiment and long-term memory,
and the stored RAG skills in `fixtures/skills/`, while reading the corpus,
ground-truth data, experiment results, metrics, and tables.

**Why it is useful.** Chunk size, retriever, reranker, and embedder choices are
usually made by taste; here they are made by measurement, and the helper turns
a table of numbers into a next step without ever making that step for you.

**Target users.** Engineers and students building a RAG system over their own
documents who need evidence for a design decision, and reviewers who want to
audit how an answer was produced.

## Overview

RAG lab is an offline-first retrieval workbench. It measures chunking,
retrieval, reranking, embedding, and generation choices against corpora with
known answers, so architecture decisions are based on evidence rather than
preference.

The repository includes seven datasets: `diary-en`, `diary-fa`, `support-en`,
`meetings-de`, `research-multihop`, `smoke-mini`, and `nosrat-fa`. Each is a
corpus together with the ground-truth questions a run is scored against. The
fresh panel default is `diary-en`; `diary-fa` remains the legacy identity for
older fingerprints.

## Quick start

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), and Node.js for
the browser-contract tests. Docker Compose is optional.

```sh
uv sync --extra local-embeddings
cp .env.example .env
uv run raglab
```

Open <http://localhost:9002> — the panel at `/`, the Inspector at
`/inspector`, the board at `/leaderboard`. With `RAGLAB_DEV_KEY` set in `.env`,
`/dev/trace` shows every widget conversation step by step — it asks for the
key on the page.

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

With `RAGLAB_DEV_KEY` set, `/dev/trace` asks for that key in a masked field
(it never appears in the address bar) and then lists every widget thread; open
one to see the system lines, the question, each tool call and reply, the
answer, and the token account. The browser stays unlocked until you press
**Lock** or the server restarts. Read-only; a 404 when no key is configured.

Widget turns can also be traced to LangSmith — the widget alone, since it
writes no run and no ledger row. Set the four `LANGSMITH_*` variables from
`.env.example` and restart the server. Two details that silently disable it:
`LANGSMITH_TRACING` must be the lowercase word `true`, and
`LANGSMITH_ENDPOINT` must match your key's region (US keys use
`https://api.smith.langchain.com`, EU keys `https://eu.api.smith.langchain.com`;
a mismatch is a 403).

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

`raglab-sweep`, `raglab-judgescreen` and `raglab-leaderboard` live in
`agents/extra_tools/`; they import the lab and no frontend route reaches them.

## Examples

Questions the Ask widget is built for:

- On an experiment: *"Why is context recall low here?"* — it reads the
  experiment's retrieved evidence and metrics, then answers with what the
  retriever missed and which knob touches that.
- On the Leaderboard: *"Which of these runs is comparable to exp 12?"* — it
  refuses to name a winner across judges and explains which columns must match.
- On the Laboratory: *"What does reranking buy me on a Farsi corpus?"* — it
  looks up the stored skill in `fixtures/skills/` and any dataset memory from
  earlier runs on that corpus.
- Ambiguous: *"Is this good?"* — it asks which experiment and which metric you
  mean instead of guessing.
- Off-topic: *"Write me a poem"* — the relevance guard declines, and nothing is
  written to long-term memory.

## Design decisions

- **No vector database.** The index lives in process memory and dies with the
  process, so a build opens no socket and every experiment is reproducible
  from its recorded fingerprint alone.
- **A row never lies about what produced it.** A model the backend does not
  serve is refused, never substituted; a judge that cannot be reached refuses
  to score rather than returning a passing default.
- **Exactly four judged metrics decide** (faithfulness, answer relevancy,
  context precision, context recall). Everything else is reported and does not
  vote, so a change in one knob cannot be argued into a win by a metric nobody
  agreed on in advance.
- **One knob per sweep candidate,** models held fixed, and the answerer and the
  judge must be different models — otherwise a difference cannot be attributed.
- **The helper is outside the measured seam.** It writes no run, no ledger row
  and no score, which is the only reason it may keep memory, bill tokens and
  trace to LangSmith without contaminating an experiment.
- **Plain HTML/JS front end on a FastAPI backend** rather than a framework: the
  three pages share one token file and one widget bundle, and the browser
  contract is tested with `node --test` without a build step.
- **Prompts, tool descriptions and skills are fixtures, not code**
  (`fixtures/prompts/*.yaml`, `fixtures/skills/*/SKILL.md`), pinned byte-equal
  by a test, so a prompt change is a reviewed diff.

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
