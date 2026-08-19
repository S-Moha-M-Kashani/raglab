"""The scoped RAG agent: a bounded LangGraph loop around the pipeline's own
stages. `AgentConfig.scope` picks retrieve/generate/full; the loop always calls
`pipeline.retrieve` with the run's own `RetrievalConfig`, never its own
retrieval.

One module per concern, the widget package's layout: `availability` holds the
deferred langgraph imports and the can-this-installation-run-it answer,
`prompts` the five node prompts, `verdicts` the conservative reading of a
model's verdict, `shape` the graph's nodes and edges as data, `loop` the
per-question loop itself, `runner` the `run()` seam callers use. `tests/`
holds the manual live probe, run only when named on the pytest command line.

Unlike the widget, this package sits *inside* the measured seam: no LangSmith,
no checkpointer, every hop through `pipeline.retrieve`, every row carrying its
stop reason.
"""
# The submodules stay reachable (agentic_rag.loop, agentic_rag.verdicts, ...)
# because a test that monkeypatches an internal must patch the module that
# defines it, not this package's re-export of it.
from raglab.agents.agentic_rag import availability
from raglab.agents.agentic_rag import loop
from raglab.agents.agentic_rag import prompts
from raglab.agents.agentic_rag import runner
from raglab.agents.agentic_rag import shape
from raglab.agents.agentic_rag import verdicts
from raglab.agents.agentic_rag.availability import (
    AGENT_EXTRA,
    _stategraph,
    agent_available,
    available)
from raglab.agents.agentic_rag.loop import (
    CRITIC_BAR,
    _ask,
    _contexts_block,
    _Exhausted,
    _guard,
    _Loop,
    _terminal)
from raglab.agents.agentic_rag.prompts import (
    ASSESS_PROMPT,
    COMPLETENESS_PROMPT,
    CRITIQUE_PROMPT,
    PLAN_PROMPT,
    REWRITE_PROMPT)
from raglab.agents.agentic_rag.runner import note_for, run
from raglab.agents.agentic_rag.shape import (
    State,
    _shape,
    graph_edges,
    graph_nodes,
    owns_generation,
    owns_retrieval)
from raglab.agents.agentic_rag.verdicts import verdict
