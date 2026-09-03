"""The panel's routes, one module per section this project already names.

`panel_server.create_app` builds the state, packs it into a `PanelContext` and
calls each module's `register(app, context)` — so a reader looking for the
dataset routes opens `datasets.py` and finds every one of them and nothing
else. The modules point inward: they read the lab and the shared plumbing, and
nothing in the lab imports them back.

- `assets.py` — the frontend files this service makes public, and the
  allowlist that says which those are.
- `configuration.py` — what the panel needs to render itself, and whether the
  lab is up.
- `pipeline.py` — the measured stages: build an index, retrieve, ask one
  question, and the job table all three run on.
- `experiments.py` — evaluations, the ledger, one experiment's archive, and
  the board built from both records.
- `datasets.py` — the corpora this installation can measure against, the
  templates for adding one, and imported archives.
- `credentials.py` — the OpenRouter key the panel types, held in memory.
- `widget.py` — the helper in the corner, which is outside the measured seam.
- `dev_trace.py` — the developer's step-by-step checkout of one widget thread.
"""
# Imported here so the factory can name each section once — `routes.pipeline`
# rather than a bare `pipeline`, which would collide with the lab modules
# `panel_server.py` imports under those same words (the widget, the datasets,
# the credentials, the question pipeline).
from raglab.dashboard.routes import (  # noqa: F401
    assets,
    configuration,
    credentials,
    datasets,
    dev_trace,
    experiments,
    pipeline,
    widget)
