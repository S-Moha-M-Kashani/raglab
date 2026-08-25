# RAG lab

A generic retrieval workbench. Build an index over a ground-truth corpus,
retrieve against its questions, score what comes back, and keep the account of
every experiment, so a RAG architecture for any use case can be chosen by
measurement instead of by taste. The corpus is a config field, not an
assumption: point `dataset` at an imported dataset id matching the stated
contract, or use one of the six that ship. The bundled default is `diary-en`
(`fixtures/corpus_groundtruth_datasets/diary_year_en_corpus.json` and
`diary_year_en_groundtruth.json`), a year of synthetic colloquial diary chat
with ground-truth questions and cited evidence, translated from its Farsi
original (`diary-fa`), which ships beside it — one case study among the
shipped corpora, not the project's scope. For compatibility with older
fingerprints, an empty `IndexConfig.dataset` still means `diary-fa`; the fresh
panel setting explicitly selects `diary-en`.

**Who this is for:** anyone who has to choose a RAG architecture and wants the
choice to rest on numbers — an engineer picking chunking and retrieval settings
for a product, a researcher ablating one knob at a time, or a learner who wants
to see what each advanced-RAG technique actually buys on a corpus with known
answers.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/).
- Node.js, for the standalone browser-contract tests.
- Docker with Compose v2, if you want to run the containerized service.
- Optional model backends: Ollama, a logged-in Claude or Codex CLI, or an
  OpenRouter API key. The test suite uses the offline `fake` backend.

## Quick start

From the repository root:

```sh
uv sync --extra local-embeddings
cp .env.example .env
uv run raglab
```

The default backend is Ollama. To use another backend, set `RAGLAB_LLM` in
`.env`; the panel starts without a chat backend, but answering and judging
require one. Open <http://localhost:9002>.

The `local-embeddings` extra installs the default embedder; its ~2.2 GB model
downloads on the first index build. The panel and index builds need no chat
backend, but answering and judging do. Set `RAGLAB_LLM` in `.env` to one of
the following values:

- `ollama` (the default) — a local model server, so a fresh install can never
  silently spend credit. Install it from <https://ollama.com>, then pull the
  lab's default model once:

  ```sh
  ollama pull 4skl/gemma4-e2b-mtp
  ```

  Any other pulled model works too: name it in `RAGLAB_MODEL`. The lab expects
  Ollama at `localhost:11434` (`RAGLAB_OLLAMA_BASE_URL` overrides).

- `claude` / `codex` — no API key and nothing to pull: each drives the coding
  CLI already installed **and logged in** on this machine (`npm install -g
  @anthropic-ai/claude-code` or `npm install -g @openai/codex`, then run it
  once to log in). Set `RAGLAB_LLM=claude` or `RAGLAB_LLM=codex`;
  `RAGLAB_CLI_EFFORT` sets how hard it thinks (claude takes
  `low|medium|high|xhigh|max`, codex `low|medium|high|none`).

- `openrouter` — a remote model, paid per call. Set `RAGLAB_LLM=openrouter`
  and put an `OPENROUTER_API_KEY` in `.env` — or skip the file and enter the
  key in **⚙** at the right of the panel masthead, where it lives in process
  memory only, for the life of that process. The default model is
  `openai/gpt-5-nano`.

### The widget

The corner helper (the Ask widget on every surface) has its own model picker
and needs no setup beyond the backends above. Its tool-using default,
`openai/gpt-5-nano`, reads the same OpenRouter key — from `.env` or typed into
**⚙**; without one it answers 502 naming the missing variable. The keyless
fallback is picking the claude or codex CLI in its picker, which answers in
one call but cannot run tools.

## Configuration

`RAGLAB_LLM` and its backends are covered in Quick start; the fifth value,
`fake`, is offline, answers and judges without ever failing, and is for tests
only. Everything else the lab reads is in `.env.example`, commented out and
kept complete by `test_conventions.py`.

**⚙** also holds the theme: **Day**, **Night**, or **Auto**, which follows the
machine and is the default. The panel, the Inspector and the board are one
origin, so the choice travels between them rather than being remembered three
times over.

## A first experiment

1. Start the panel (`uv run --extra local-embeddings raglab`) and open
   `http://localhost:9002`.
2. Open **⚙** at the right of the masthead and press **Import JSON** under
   *Experiment archive*, then pick `fixtures/loadstar-rag-setting.json` — the
   measured preset as a settings-only archive: it restores every knob (each
   explained by the `!` beside it) and nothing else — no database row, no
   run, no job started.
3. In the **Index** card press **Build**. The first build downloads the
   default embedding model (~2.2 GB); later builds with the same fingerprint
   are reused.
4. In the **Generation** card leave RAGAS on `offline` and press
   **Run evaluation**. The default `ollama` backend answers locally; the
   Readings card fills with the deterministic scores and the offline RAGAS
   pair, and the run lands in **Every experiment**.
5. Change exactly one knob — say k from 8 to 5 in the **Retrieval** card —
   and run again. Both rows now sit in the experiments table for comparison.
   Ranking by the four judged decision metrics needs a real judge: switch
   RAGAS to `judged` with an `openrouter` backend (answerer and judge are
   separate models on purpose), and the leaderboard shows
   `decision score ± spread` per run. The board's table sorts on any column
   and narrows on any of them from the **Filter** box above it — one term per
   column, all of which must hold: `state!=failed questions>30 decision>=0.6`,
   `when>2026-08-01`, `judge~sonnet`, `kind:run`. A column name followed by a
   bare colon asks whether that column was measured at all (`ctx-recall:` for
   the rows that have it, `!ctx-recall:` for the rows that do not); a bare word
   searches the whole row and `!word` excludes it. The filter is in the URL, so
   a narrowed board is a link.
6. Press the **open** arrow on any row. It reads that experiment in the
   Inspector at `/inspector` — which chunks were retrieved, which were gold,
   what the answer was graded — and makes the same experiment's settings the
   Laboratory's, so the next run starts from it. Every knob this installation
   can serve is set; anything it cannot (an embedder that is not installed, a
   model the current backend does not offer, a corpus since removed) is left
   where you had it and named in the lab helper, along with how much of the
   panel moved. A row with no run file behind it recorded only a handful of
   knobs, and the helper says that too.
7. Back under **⚙** → *Experiment archive*, press **Export experiment** to
   write the whole experiment to one JSON archive. Until an evaluation has
   completed under the current settings the file carries settings only; after
   one it also carries the result and the Inspector evidence. A completed
   archive contains the corpus text, ground truth, generated answers and
   traces — share it as carefully as the data inside it. Importing a completed
   archive renders its results read-only and lands it once in **Every
   experiment**; it never enters the ranked leaderboard.

The **✳ Ask** widget in the panel's corner answers questions about the lab
itself, about the technique it implements, and about the experiments this
installation has already recorded. It is described below.

## Datasets

The lab measures whatever corpus it is pointed at. Six ship with it:
`diary-en` (the fresh panel default), `diary-fa` (the legacy built-in identity),
`support-en`, `meetings-de`, `research-multihop`, and `smoke-mini`. They live in
`fixtures/corpus_groundtruth_datasets/` as ordinary corpus/ground-truth pairs.
Each maps to a different use case, so a finding can be checked against a corpus
of a different language, domain or question shape.

Bring your own with **Import a dataset** in the panel, or `POST
/api/datasets`. Import a pair of JSON files: one corpus file and one
ground-truth file, joined by the declared dataset id. The panel states the
shape it expects behind the `!` beside the file picker. The lab refuses a pair
that does not match it rather than repairing it, and names every problem it
found.

## Choosing an architecture

The lab prescribes none — every pipeline stage is a config knob, and the right
combination depends on the use case. `fixtures/skills/` is the guidance layer: twelve
skill files covering the advanced-RAG technique landscape, a use-case →
starting-architecture map with a per-use-case experiment ladder, the
experiment methodology (dev/test discipline, error analysis), and the sources
to watch for new work. Start at `fixtures/skills/README.md`.

## The Ask agent

**✳** at the right of the masthead opens a tool-calling agent over two corpora
kept deliberately distinct: a knowledge base about *this project* — ports,
metrics, what each knob does — and `fixtures/skills/`, twelve skills about *the
field*. It must not answer "what does this lab do" out of a technique paper, or
present a literature claim as a measurement taken here.

Nine tools:

| Tool | What it does |
| --- | --- |
| `search_knowledge_base` | Keyword match over the project's own facts. |
| `search_rag_skills` | Names and descriptions of the twelve skills — the routing layer, twelve short lines. |
| `read_rag_skill` | Skill bodies, at most three per call; the bodies are the expensive layer. |
| `calculate` | Arithmetic over an AST whitelist, never `eval` — a tool handed to a model must not be a Python prompt. |
| `measure_bilingual_alignment` | Runs the EN–Farsi probe over a real encoder and returns pair cosines, mixed-pool retrieval and a verdict. |
| `list_experiments` | Recorded experiments, newest first, with each decision score beside its own error. |
| `read_experiment` | One experiment: its knobs step by step, the four judged metrics, the judge, the deterministic summary and the backend that answered. |
| `read_experiment_questions` | The per-question rows of one evaluation — by default the questions whose gold evidence retrieval did not fully find inside k. |
| `recall_conversation` | What was said about another experiment in an earlier conversation, read back from `databases/widget.db` by that experiment's id. |

Three of them — `list_experiments`, `read_experiment`,
`read_experiment_questions` — read the two durable records, the ledger and
`.runs/`, and nothing else: they compute no score, rank nothing, and carry no
chunk text, trace or hierarchy summary. That is what makes "what went wrong in
the last run, and what should change" answerable in the widget, against the
run's own evidence rather than from the model's memory. Nothing was written for
the widget to read: it is handed the board's own rows and the same projection
the leaderboard's open button resolves, and all it adds is the formatting that
makes them readable to a model. The widget imports no evaluation module, so the
panel injects those functions at startup; started without them, the three
tools say the records are unavailable rather than answering from an empty
list. `recall_conversation` reads a different record — the widget's own
conversation log, not evidence of anything measured — so it answers from what
was *said* about an experiment, never from what was scored.

The agent is `langchain.agents.create_agent` with six middleware hooks, and the
model picker offers four options that state what they can do. **gpt-5-nano**
(the default) and **gpt-5-mini** run over OpenRouter and can use the tools; a
key is required, and without one the widget answers 502 naming what is missing.
**claude** and **codex** drive a CLI already logged in on this machine, so they
need no key at all — and cannot run tools, because a CLI chat has no
`bind_tools`, and keep no memory of their own, because a CLI call is one
process with no graph and no checkpointer behind it. Those two answer in one
call with the knowledge base inlined and the skill names in the prompt, the
bodies out of reach. Every reply carries its token account, or `None` where
the backend reported nothing — "0 tokens" would be a claim about the bill.

An answer is typed out as it is written: the two OpenRouter models stream, and
the widget shows the pieces as they land, with a caret while the answer is
still coming. The pieces are only how it arrived — the last thing the lab sends
is the reply as its own conversation log now holds it, and the bubble adopts
that, so what stays on screen is the transcript rather than a second private
copy of it. A stream that dies part-way keeps what did arrive, marked as
stopped with the reason beneath it, because a fragment must never be handed
over as a whole answer. The two CLI options cannot stream: one subprocess
reports one complete reply, so their answers land in a single piece, and their
labels say so.

Conversation memory is one thread per experiment, plus one `general` thread
for whenever the lab has none open, shared by all three surfaces; each call
sees the last twenty messages of the thread it lands on. Threads persist in
`databases/widget.db`, so a reload or a restart of the lab does not lose them
— New Chat is the only thing that does, and it ends only the one thread it was
pressed in. A CLI turn is the exception: it answers inside that thread but
writes nothing to it, so a conversation carried on `claude` or `codex` is not
remembered on the next turn even though the lab keeps running. The key typed
into **⚙** lives in process memory only — no file, no environment variable, no
log.

The widget is outside the measured seam. It writes no run, no ledger row and no
number, and that is the only reason it may trace to LangSmith. Tracing is
`LANGSMITH_TRACING`, which is process-global: leave it off while a scored run
shares the process, or the run is traced too.

## What gets written where

- `.runs/` — one JSON file per evaluation run. Git-ignored.
- `databases/raglab.db` — one ledger row per job, including terminal success,
  error and cancellation. `RAGLAB_DB` overrides the path.
- `databases/corpora.db` — content-addressed corpus and ground-truth versions
  used by archived experiments. `RAGLAB_CORPORA_DB` overrides the path.
- `.datasets/` — imported corpus/ground-truth pairs. `RAGLAB_DATASETS`
  overrides the directory.
- `databases/widget.db` — the widget's conversations, one thread per
  experiment plus `general`; see `.env.example` for what it is and is not.
  `RAGLAB_WIDGET_DB` overrides the path — the suite redirects it
  automatically, but a hand-run server does not, so set it yourself or a
  manual check writes into your real conversation store.
- `.screens/` — one JSON file per judge screen, the evidence for which model
  was allowed to grade the deciding metrics. Git-ignored, so keep it if you
  care which judge produced a number.

## Command-line tools

| Command | What it does |
| --- | --- |
| `uv run raglab` | Serve the Laboratory, Inspector and Leaderboard on port 9002. |
| `uv run raglab-lab` | Run the Python suite, then serve the panel. Use `--test-only` to stop after tests or `--no-test` to skip the preflight suite. |
| `uv run raglab-sweep` | Run one-knob-at-a-time candidates and report the decision-score comparison. |
| `uv run raglab-judgescreen --models MODEL [MODEL ...]` | Screen judge models before allowing them to grade. `--pairs` controls claim pairs per model. |
| `uv run raglab-leaderboard` | Print the recorded board; supports `--json`, `--limit` and `--write PATH`. |

## Development

| Command | What it does |
| --- | --- |
| `uv run pytest` | the suite |
| `uv run raglab-lab` | the suite, then the panel — refuses to serve on a red suite |
| `uv run raglab-lab --test-only` | run the suite and stop |
| `uv run raglab-lab --no-test` | start the panel without the preflight suite |

The suite is offline and safe to run anywhere: fixtures pin the `fake`
backend, blank any API keys, and redirect every artifact path, so no test can
call a model, spend credit, or write into the real `.runs/`, ledger or
conversation store. The Python command is the canonical full check:

```sh
uv run pytest src/raglab -q
```

The browser-facing JavaScript contracts are normally exercised by the Python
tests that invoke Node. To run the standalone dashboard contracts that do not
need generated fixtures:

```sh
cd src/raglab/dashboard/tests
node --test panel_open.test.js board_reveal.test.js
```

`archive_ladder.test.js` is generated and run by its Python companion, which
sets `RAGLAB_LADDER` to a temporary fixture. Running every `.test.js` file
directly without that environment variable will fail before the test starts.

The only Python tests excluded from the offline suite are the five intentional
live probes in `src/raglab/agents/widget/tests/test_skills_live.py`: one real
Codex CLI call, three real LLM calls, and one test that loads the 2.2 GB
encoder. Run them explicitly when credentials, network access, and the model
download are available:

```sh
uv run pytest src/raglab/agents/widget/tests/test_skills_live.py -v -s
```

Tests are colocated — each section's `tests/` folder holds its own, and
`src/raglab/tests/test_conventions.py` holds the repo-wide guards. Branch from
`development`, never `master`; `master` is the squash-merged release.

## Docker

Docker runs the same composed application as one service: the Laboratory at
`/`, Inspector at `/inspector`, and Leaderboard at `/leaderboard`. It binds the
durable run files, databases and imported datasets from the host, and keeps the
embedding cache in a named volume.

Before the first start, create the bind-mounted directories so Docker does not
create them as root:

```sh
mkdir -p .runs databases .datasets
docker compose up -d --build
```

Open <http://localhost:9002>. Useful lifecycle commands:

```sh
docker compose logs -f panel
docker compose down
```

The container expects Ollama on the host at `host.docker.internal:11434` by
default, as configured in `compose.yaml`. Alternatively set another backend in
`.env`; do not put a panel-entered OpenRouter key in Compose, because the panel
keeps that key only in process memory. The first index build downloads the
local embedding checkpoint into the persistent `hf-cache` volume.
