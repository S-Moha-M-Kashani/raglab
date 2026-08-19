"""The experiment's knobs — the four stage configs, their vocabularies,
dependency rules and help text. Feeds every panel section.

The whole knob surface re-exports through this package (it was `lab_config.py`'s
hub role); `lab_config` holds the dataclasses themselves.
"""
from .lab_config import *          # noqa: F401,F403 — the hub, including re-exports
from . import lab_config           # noqa: F401 — the other submodules load on
                                   # demand; importing explainer_assembly here
                                   # would close a cycle through evaluation


def __getattr__(name: str):
    """`PRODUCTION_CONFIG` is built lazily in lab_config (PEP 562); delegate."""
    return getattr(lab_config, name)
