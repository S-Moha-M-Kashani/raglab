"""RAG lab — a retrieval workbench for diary-memory retrieval, indexed in process memory with no vector database and no service to start first.
Serves the panel at :9002, with the read-only Inspector mounted at /inspector on the same port, via `uv run --extra local-embeddings raglab` — the extra is not optional, the default embedder needs it to build an index.
Moved from Lodestar's `brain/tests/raglab/` on 2026-08-11; still measured against Lodestar's shipped config, snapshotted in `production_baseline_snapshot.py` rather than imported live.
"""
