# RAG lab

A retrieval workbench. Build an index over synthetic diary chat, retrieve
against ground-truth questions, score what comes back, and keep the account of
every experiment, so a retrieval choice can be made by measurement instead of
by taste. The default corpus is `fixtures/diary_year_fa.json`, a year of
synthetic colloquial Farsi diary chat with ground-truth questions and cited
evidence.

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
complete by `tests/test_config.py`.

## Datasets

The built-in corpus is the Farsi diary. Four more ship as controls in
`fixtures/groundtruth_datasets/`: English support tickets, German meeting
notes, English research notes weighted to multi-hop, and a five-session smoke
set.

Bring your own with **Import a dataset** in the panel, or `POST
/api/datasets`. One JSON file; the panel states the shape it expects behind the
`!` beside the file picker. The lab refuses a file that does not match it
rather than repairing it, and names every problem it found.

## What gets written where

- `.runs/` — one JSON file per evaluation run. Git-ignored.
- `databases/raglab.db` — one row per finished experiment. `RAGLAB_DB`
  overrides the path.
- `.screens/` — judge screens. Committed, not git-ignored.

## Background

This code lived at `brain/tests/raglab/` in the Lodestar repository until
2026-08-11, where it decided that project's retrieval architecture. The
reasoning behind individual decisions is in `CLAUDE.md`. The measured argument
with run ids, the design notes and the per-question walkthroughs are kept
outside the repository.
