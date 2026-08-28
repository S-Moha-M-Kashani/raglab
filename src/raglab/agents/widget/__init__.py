"""The panel's LLM widget: one self-contained package, deliberately outside
`chat_model_factory.py`'s measured seam.

It answers questions about this project from a small knowledge base, a
calculator, a skills corpus, and — read-only, injected by the panel because
this package imports no evaluation module — the experiments the lab has
already recorded. It retrieves nothing, judges nothing, computes no score, and
writes no run, no ledger row and no number. That is why it may do two things the measured path
must not: talk to OpenRouter through its own `ChatOpenAI`, and trace to
LangSmith when tracing is switched on (backends.TRACING_ENV). Removing the
widget is deleting this
folder and the one route in panel_server.py; a convention test pins that no other
lab module reaches in, and that this package reaches the lab only through
its unmeasured edges (skills, clichat, settings).

One module per concern: `prompts` loads the model-facing fixtures,
`hooks` holds the four middleware, `tools` the project-knowledge tools and the
registry the agent is handed, `experiment_tools` the three read-only windows
onto what this lab has already measured, `probe` the bilingual measurement
those tools wrap, `backends` the catalogue and the two answer paths,
`__main__` the real-call harness.

The agent is `langchain.agents.create_agent` with four middleware hooks, taken
2026-08-18 when this project moved to langchain 1.x. Before that the pin said
`langchain<1` and the agent was langgraph's `create_react_agent` — the same
prebuilt loop under its pre-1.0 name, wired through `pre_model_hook`,
`post_model_hook` and a callable model because `AgentMiddleware` was a major
version away. There were six hooks from 2026-08-18 until 2026-08-28, when
`before_model`/`after_model` folded into the `wrap_model_call` wrapper so a
tool hop stopped costing two graph nodes it never needed (`hooks.RECURSION_LIMIT`).
"""
# The submodules stay reachable (widget.backends, widget.probe, ...) because
# a test that monkeypatches an internal must patch the module that defines
# it, not this package's re-export of it.
from raglab.agents.widget import backends
from raglab.agents.widget import conversation_memory
from raglab.agents.widget import experiment_tools
from raglab.agents.widget import hooks
from raglab.agents.widget import long_term_memory
from raglab.agents.widget import probe
from raglab.agents.widget import prompts
from raglab.agents.widget import tools
from raglab.agents.widget.backends import (
    DEFAULT_MODEL,
    REQUIRED_ENV,
    TRACING_ENV,
    WIDGET_MODELS,
    WidgetUnavailable,
    _AGENTS,
    _build_agent,
    _cli_system,
    _openrouter_url,
    ask,
    reset,
    set_openrouter_key_resolver,
    stream)
from raglab.agents.widget.conversation_memory import (
    GENERAL,
    MAX_RECALLED,
    WidgetState,
    forget,
    history,
    recall_conversation,
    thread_summaries,
    threads,
    trace)
from raglab.agents.widget.long_term_memory import (
    MAX_SUMMARY_CHARS,
    clear_long_term_memory,
    db_path,
    memory_context,
    save_memory_update)
from raglab.agents.widget.hooks import (
    HOOK_LOG,
    MAX_HISTORY,
    MAX_QUESTION,
    MIDDLEWARE,
    RECURSION_LIMIT,
    _validate,
    check_request,
    close_the_log,
    log_tool_call,
    stop_repeated_tool_hops,
    trim_and_call)
from raglab.agents.widget.probe import _read_pairs
from raglab.agents.widget.prompts import (
    KNOWLEDGE_BASE,
    PROMPTS_DIR,
    STARTERS,
    SYSTEM_PROMPT)
from raglab.agents.widget.experiment_tools import (
    EXPERIMENT_TOOLS,
    MAX_LISTED,
    list_experiments,
    read_experiment,
    read_experiment_questions,
    set_experiment_reader)
from raglab.agents.widget.tools import (
    MAX_SKILL_READS,
    TOOLS,
    calculate,
    measure_bilingual_alignment,
    read_rag_skill,
    search_knowledge_base,
    search_rag_skills)
