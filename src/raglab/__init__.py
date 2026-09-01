"""RAG lab — a retrieval workbench where chunking and retrieval choices are decided by measurement, indexed in process memory with no vector database and no service to start first.
Serves the panel at :9002, with the read-only Inspector mounted at /inspector on the same port, via `uv run --extra local-embeddings raglab` — the extra is not optional, the default embedder needs it to build an index.
Extracted from the production assistant it began life inside on 2026-08-11; still measured against the config that assistant shipped, snapshotted in `production_baseline_snapshot.py` rather than imported live.
"""
