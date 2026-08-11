# RAG lab

A retrieval workbench. Build an index over a year of synthetic Farsi diary chat,
retrieve against ground-truth questions, score what comes back, and keep the
account of every experiment — so a retrieval choice can be made by measurement
instead of by taste.

```sh
uv run --extra local-embeddings raglab      # the panel on :9002
```

The extra is not optional in practice: the default embedder is a
sentence-transformers checkpoint, and without it the service starts happily and
then fails on the first index build. The embedding model (~2.2 GB) downloads on
first retrieval, not at boot, so `/api/health` keeps answering while it does.

| Command | What it does |
| --- | --- |
| `uv run raglab-lab` | the suite, then the panel — refuses to serve on a red suite |
| `uv run raglab-inspector` | the read-only Inspector on :9003, where one question is traced |
| `uv run raglab-sweep` | the one-change-at-a-time ablation ladder |
| `uv run raglab-judgescreen` | score a candidate judge model before trusting it |
| `uv run raglab-leaderboard` | rank the runs in `.runs/`, refusing to rank the incomparable |
| `uv run pytest` | the suite |

## What is durable and what is not

**An experiment's material is not a record.** The index, the retrieved contexts
and the generated answers exist to produce one number and are discarded with the
process — so there is **no vector database here, deliberately**: nothing to start
first, and nothing a later run can inherit from an earlier one by accident.

What is written down is the account of the work:

- `.runs/` — one JSON file per evaluation run. Git-ignored: ~130 KB each,
  machine-specific, and reproducible from the fixtures.
- `databases/raglab.db` — one row per finished experiment, build and retrieval
  included, so "what have I already tried?" outlives the process that tried it.
  `RAGLAB_DB` overrides the path.
- `.screens/` — judge screens, deliberately **not** git-ignored. Which model was
  allowed to grade the four deciding metrics is part of the argument, not an
  artifact of a machine.

Nothing backs any of this up. The ledger is the only thing that would be missed.

## The corpus

`fixtures/diary_year_fa.json` — 167 sessions of synthetic colloquial Farsi diary
chat over one year (2025-08-02 → 2026-07-27), 18 recurring storylines, 5 tracked
habits, with ground-truth questions and cited evidence beside it. Colloquial on
purpose: Arabic ي/ك mixed with Persian ی/ک, half-spaces present or missing,
Persian and ASCII digits. Two texts a reader would call identical have to
tokenise identically or every lexical score silently under-counts, which is what
`textnorm.py` is for.

## Where this came from

This code lived at `brain/tests/raglab/` in the Lodestar repository until
2026-08-11, where it decided that project's retrieval architecture.
`docs/rag-architecture.md` is the measured argument with run ids;
`docs/rag-test/` walks the winning configuration question by question.

It still measures against what Lodestar shipped, but that configuration is now a
dated snapshot in `src/raglab/baseline.py` rather than a live import — **if
Lodestar's retrieval changes, this repository will not notice.** The same applies
to `textnorm.py`, which is a vendored copy. Both carry the commit they came from.

The snapshot was checked rather than assumed: on 2026-08-11 both labs were run
side by side and their `/api/options` compared, and the preset came back
value-for-value identical across all four groups, as did the model roles, model
lists, modes, embedders, chunkers and metrics. The only intended differences were
the label — which now carries its own date — and the two storage paths.

Two things were lost in the move and are worth knowing about:

- The board's own lab view is gone. There was a second frontend over this API
  inside Lodestar, and eight tests existed to pin that the two could not
  disagree; each now covers the one panel that remains.
- The ports below are a hand-maintained copy. Nothing here can detect a change on
  Lodestar's side.

## Ports

`:9002` the panel, `:9003` the Inspector. `tests/test_ports.py` holds the list of
ports Lodestar's stack owns on this machine and asserts these two avoid them.

## Configuration

Every variable the lab reads is in `.env.example`, commented out, and
`tests/test_config.py` asserts that list is complete in both directions — a
variable missing from it is undiscoverable, and one lingering after the code
stopped reading it is a lie. A repo-root `.env` is read without overriding what
is already in the environment.

The chat backend is `RAGLAB_LLM`: `ollama` (the default — a judged run can make
hundreds of calls, so a default must never silently spend credit), `openrouter`,
or `fake`. An unknown value raises; there is no auto mode anywhere in this
project.
