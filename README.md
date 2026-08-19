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

Add `--extra agent` for the agent scopes, or `--extra graph-index` for the
`leiden` hierarchy grouping.

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
- `openrouter` — a remote model; needs `OPENROUTER_API_KEY`, or type the key
  into the panel.
- `claude` / `codex` — drives a CLI already installed on this machine, no API
  key needed.
- `fake` — offline, answers and judges without ever failing; for tests only.

Everything else the lab reads is in `.env.example`, commented out and kept
complete by `tests/test_conventions.py`.

## A first experiment

1. Start the panel (`uv run --extra local-embeddings raglab`) and open
   `http://localhost:9002`.
2. Press **Load the shipped settings** — the measured preset, every knob
   explained by the `!` beside it.
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
   `decision score ± spread` per run.
6. Trace any single question in the Inspector on :9003 — which chunks were
   retrieved, which were gold, what the answer was graded.

The **✳ Ask** widget in the panel's corner answers questions about the lab
itself — a tool-calling agent over the project's knowledge base and the
`skills/` corpus, deliberately outside the measured pipeline (it writes no
run and no number).

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
combination depends on the use case. `skills/` is the guidance layer: fourteen
skill files covering the advanced-RAG technique landscape, a use-case →
starting-architecture map with a per-use-case experiment ladder, the
experiment methodology (dev/test discipline, error analysis), and the sources
to watch for new work. Start at `skills/README.md`.

## What gets written where

- `.runs/` — one JSON file per evaluation run. Git-ignored.
- `databases/raglab.db` — one row per finished experiment. `RAGLAB_DB`
  overrides the path.
- `.screens/` — one JSON file per judge screen, the evidence for which model
  was allowed to grade the deciding metrics. Git-ignored, so keep it if you
  care which judge produced a number.
