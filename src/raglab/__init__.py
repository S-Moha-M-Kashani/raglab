"""RAG lab — a retrieval workbench for diary-memory retrieval.

The lab reads synthetic fixtures (`fixtures/diary_year_fa.json` and its ground
truth), indexes them **in process memory**, and exposes a JSON API with a panel
over it.

**An experiment's material is not a record.** The index, the retrieved contexts
and the generated answers exist to produce one number and are discarded with the
process. What is written down is the account of the work: one JSON file per run
in `.runs/`, and one row per finished experiment in `ledger.py`'s SQLite. So the
lab needs **no vector database** and no service running first — there is nothing
to start, and nothing a later run can inherit from an earlier one by accident.

It serves on :9002, with the read-only Inspector on :9003. Both numbers avoid the
ports Lodestar's stack owns on this machine; `tests/test_ports.py` holds the list.

    uv run --extra local-embeddings raglab

The extra is not optional in practice: the default embedder is a
sentence-transformers checkpoint, and without it the service starts and then
fails on the first index build. The embedding model (~2.2 GB) downloads on first
retrieval, not at boot.

This code lived at `brain/tests/raglab/` in the Lodestar repository until
2026-08-11. It still measures against Lodestar's shipped retrieval config, which
is now a dated snapshot in `baseline.py` rather than a live import.
"""
