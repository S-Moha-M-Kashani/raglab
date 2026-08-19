"""Whether this installation can run the loop, verified by import.

Every langgraph import lives here and is deferred into a function, so
`import raglab.agentic_rag` still succeeds when the `agent` extra is not
installed — only a call to `runner.run` needs langgraph, and
`LabConfig.validate()` already refused a scope this installation cannot run
before `run` is ever reached.
"""

AGENT_EXTRA = 'uv sync --extra agent'


def agent_available() -> bool:
    """Verified by import, not read off a list, so NA means one thing: this installation cannot load it."""
    try:
        import langgraph                                    # noqa: F401
        from langgraph.graph import StateGraph              # noqa: F401
    except Exception:
        return False
    return True


def _stategraph():
    """The one place `_Loop.build_graph` reaches into langgraph. Deferred like
    `agent_available`'s own import above rather than a module-level import —
    see the module docstring."""
    from langgraph.graph import END, StateGraph
    return END, StateGraph


def available() -> dict:
    """Every scope → whether this installation can run it, and what to install if not."""
    from raglab.configuration.lab_config import SCOPES
    ready = agent_available()
    return {scope: {'available': True if not scope else ready,
                    'install': '' if not scope else AGENT_EXTRA}
            for scope in SCOPES}
