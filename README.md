# RAG lab

A generic retrieval workbench. Build an index over a ground-truth corpus,
retrieve against its questions, score what comes back, and keep the account of
every experiment, so a RAG architecture for any use case can be chosen by
measurement instead of by taste. The corpus is a config field, not an
assumption: point `dataset` at any file matching the stated contract, or use
one of the five that ship. The bundled default is `fixtures/corpus_groundtruth_datasets/diary_year_fa.json`,
a year of synthetic colloquial Farsi diary chat with ground-truth questions and
cited evidence — one case study among the shipped corpora, not the project's
scope.

**Who this is for:** anyone who has to choose a RAG architecture and wants the
choice to rest on numbers — an engineer picking chunking and retrieval settings
for a product, a researcher ablating one knob at a time, or a learner who wants
to see what each advanced-RAG technique actually buys on a corpus with known
answers.

## Quick start

```sh
uv run --extra local-embeddings raglab      # the panel on :9002
uv run raglab-inspector                     # the read-only Inspector on :9003
```

The extra is required because the default embedder is a sentence-transformers
checkpoint. Without it the service starts fine and then fails on the first
index build. The ~2.2 GB model downloads on first index build, not at boot.

Add `--extra graph-index` for the `leiden` hierarchy grouping.

| Command | What it does |
| --- | --- |
| `uv run raglab-lab` | the suite, then the panel — refuses to serve on a red suite |
| `uv run raglab-inspector` | the read-only Inspector on :9003, where one question is traced |
| `uv run raglab-sweep` | the one-change-at-a-time ablation ladder |
| `uv run raglab-judgescreen` | score a candidate judge model before trusting it |
| `uv run raglab-leaderboard` | rank the runs in `.runs/`, refusing to rank the incomparable |
| `uv run pytest` | the suite |

## Configuration

`RAGLAB_LLM` picks the chat backend:

- `ollama` (default) — a local model; a judged run can make hundreds of calls,
  so the default must never silently spend credit.
- `openrouter` — a remote model; set `OPENROUTER_API_KEY` in `.env`, or open
  **⚙** at the right of the panel masthead and enter it there, for the life of
  that process. The corner helper defaults to the GPT-5 Nano OpenRouter tool
  agent.
- `claude` / `codex` — drives a CLI already installed on this machine, no API
  key needed.
- `fake` — offline, answers and judges without ever failing; for tests only.

Everything else the lab reads is in `.env.example`, commented out and kept
complete by `test_conventions.py`.

**⚙** also holds the theme: **Day**, **Night**, or **Auto**, which follows the
machine and is the default. The choice is remembered per service, so the panel
on :9002 and the Inspector on :9003 are set separately.

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
6. Trace any single question in the Inspector on :9003 — which chunks were
   retrieved, which were gold, what the answer was graded.
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

The lab measures whatever corpus it is pointed at. Five ship with it: the
default Farsi diary, and four controls in `fixtures/corpus_groundtruth_datasets/` —
English support tickets, German meeting notes, English research notes weighted
to multi-hop, and a five-session smoke set. Each maps to a different use case,
so a finding can be checked against a corpus of a different language, domain
or question shape.

Bring your own with **Import a dataset** in the panel, or `POST
/api/datasets`. One JSON file; the panel states the shape it expects behind the
`!` beside the file picker. The lab refuses a file that does not match it
rather than repairing it, and names every problem it found.

## Choosing an architecture

The lab prescribes none — every pipeline stage is a config knob, and the right
combination depends on the use case. `fixtures/skills/` is the guidance layer: fourteen
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

Eight tools:

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

The last three read the two durable records — the ledger and `.runs/` — and
nothing else: they compute no score, rank nothing, and carry no chunk text,
trace or hierarchy summary. That is what makes "what went wrong in the last
run, and what should change" answerable in the widget, against the run's own
evidence rather than from the model's memory. Nothing was written for the widget
to read: it is handed the board's own rows and the same projection the
leaderboard's open button resolves, and all it adds is the formatting that makes
them readable to a model. The widget imports no evaluation module, so the panel
injects those functions at startup; started without them, the three tools say
the records are unavailable rather than answering from an empty list.

The agent is `langchain.agents.create_agent` with six middleware hooks, and the
model picker offers four options that state what they can do. **gpt-5-nano**
(the default) and **gpt-5-mini** run over OpenRouter and can use the tools; a
key is required, and without one the widget answers 502 naming what is missing.
**claude** and **codex** drive a CLI already logged in on this machine, so they
need no key at all — and cannot run tools, because a CLI chat has no
`bind_tools`. Those two answer in one call with the knowledge base inlined and
the skill names in the prompt, the bodies out of reach. Every reply carries its
token account, or `None` where the backend reported nothing — "0 tokens" would
be a claim about the bill.

Conversation memory is one thread per page: each call sees the last twenty
messages, and the thread lives in process memory, so a reload or a restart
starts a fresh conversation. The key typed into **⚙** lives in process memory
too — no file, no environment variable, no log.

The widget is outside the measured seam. It writes no run, no ledger row and no
number, and that is the only reason it may trace to LangSmith. Tracing is
`LANGSMITH_TRACING`, which is process-global: leave it off while a scored run
shares the process, or the run is traced too.

## What gets written where

- `.runs/` — one JSON file per evaluation run. Git-ignored.
- `databases/raglab.db` — one row per finished experiment. `RAGLAB_DB`
  overrides the path.
- `.screens/` — one JSON file per judge screen, the evidence for which model
  was allowed to grade the deciding metrics. Git-ignored, so keep it if you
  care which judge produced a number.
