"""The widget's model catalogue and the two answer paths.

The OpenRouter path is `langchain.agents.create_agent` with the six
middleware hooks (hooks.py), one cached agent per model; the CLI path is a
process per call with the knowledge base inlined, because `CliChat` has no
`bind_tools`. Both paths enter through `ask`.
"""
import os

from langchain_core.messages import HumanMessage

from .. import skills
from ..clichat import CliChat, checked_effort, cli_available
from ..settings import PROVIDER_MODELS
from .hooks import MIDDLEWARE, _account, _validate
from .prompts import _PROMPTS, KNOWLEDGE_BASE, SYSTEM_PROMPT
from .tools import TOOLS

# Read at build time, never at import: the suite runs offline, and a missing
# variable must become a stated refusal rather than a KeyError at import.
# These are the *OpenRouter path's* requirement — the whole point of a CLI
# backend is that it needs no key at all.
REQUIRED_ENV = ('OPENROUTER_API_KEY', 'LANGSMITH_API_KEY',
                'LANGSMITH_ENDPOINT', 'LANGSMITH_PROJECT', 'LANGSMITH_TRACING')

# The widget's own catalogue: value -> (kind, label). The two OpenRouter
# models run the tool loop; the two CLIs cannot (`CliChat` has no
# `bind_tools`), answer in one call with the knowledge base inlined, and
# their labels say so — an option states what it can do.
WIDGET_MODELS = {
    'openai/gpt-5-nano': ('openrouter', 'gpt-5-nano · OpenRouter, tools'),
    'openai/gpt-5-mini': ('openrouter', 'gpt-5-mini · OpenRouter, tools'),
    'claude': ('cli', 'claude · CLI, no key, no tools'),
    'codex': ('cli', 'codex · CLI, no key, no tools'),
}
# The codex CLI: gpt-5.6-luna, the lightest draw on the membership, no key
# involved. Among the OpenRouter pair the cheaper nano leads the list.
DEFAULT_MODEL = 'codex'


def _openrouter_url() -> str:
    """`.env` already carries OPENROUTER_BASE_URL for the lab's own backend;
    the widget reads the same variable rather than keeping a second copy."""
    return (os.environ.get('OPENROUTER_BASE_URL', '').strip()
            or 'https://openrouter.ai/api/v1')


class WidgetUnavailable(RuntimeError):
    """The lab is up; its widget is not. The route answers this as a 502."""


# One cached agent per OpenRouter model — a CLI is a process per call and
# caches nothing.
_AGENTS: dict = {}


def reset() -> None:
    """Drop the cached agents so the next ask() rebuilds (tests, key changes)."""
    _AGENTS.clear()


def _build_agent(model: str):
    # `load_env` strips values, so a bare `KEY= ` line in .env lands here as
    # '' — an empty variable is a missing one, not a present one.
    for name in REQUIRED_ENV:
        if not os.environ.get(name, '').strip():
            raise WidgetUnavailable(
                f'{name} is not set — the widget needs it in .env')
    # Present, and read the way the spec states them. LangSmith picks its
    # four up from the environment by itself; only the key is passed on.
    openrouter_api_key = os.environ['OPENROUTER_API_KEY']
    os.environ['LANGSMITH_API_KEY']
    os.environ['LANGSMITH_ENDPOINT']
    os.environ['LANGSMITH_PROJECT']
    os.environ['LANGSMITH_TRACING']

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, api_key=openrouter_api_key,
                     base_url=_openrouter_url())
    # A static model, so `create_agent` binds the tools itself. It was a
    # *callable* under langgraph's prebuilt loop, which binds tools only to the
    # static kind — and the agent then answered from its own knowledge, called
    # neither tool, and said nothing about it. Interception lives in
    # `trim_and_call` now, which is where 1.x puts it.
    return create_agent(llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT,
                        middleware=MIDDLEWARE)


def _cli_system() -> str:
    """The tool-less prompt: project facts in full, the skills corpus as its
    index only. The full bodies cannot be inlined, so the prompt says what a
    CLI can do — name the right skill — and what it cannot: read one. The
    template is fixtures/prompts/widget.yaml's `cli_system`."""
    facts = '\n'.join(f'- {key}: {text}' for key, text in KNOWLEDGE_BASE.items())
    return _PROMPTS['cli_system'].format(facts=facts,
                                         skills_index=skills.index_text())


def _cli_answer(cli: str, message: str) -> str:
    """One CLI call, the knowledge base inlined: no tool loop exists here, so
    a prompt that does not carry the facts is a CLI answering about a project
    it has never seen."""
    if not cli_available(cli):
        raise WidgetUnavailable(
            f'the {cli} command is not on this machine — install and log in, '
            'or pick an OpenRouter model')
    system = _cli_system()
    effort = checked_effort(cli, os.environ.get('RAGLAB_CLI_EFFORT', '').strip()
                            or 'low')
    chat = CliChat(cli=cli, model=PROVIDER_MODELS[cli], effort=effort)
    try:
        return str(chat.invoke([('system', system), ('user', message)]).content)
    except Exception as error:
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error


def ask(message: str, model: str = '') -> str:
    """One question in, one answer out. `model` picks from WIDGET_MODELS;
    empty means the default. Agents build on first use, one per model."""
    choice = model or DEFAULT_MODEL
    kind, _ = WIDGET_MODELS.get(choice) or (None, None)
    if kind is None:
        raise ValueError(f'{choice!r} is not a widget model; expected one of '
                         + ', '.join(repr(v) for v in WIDGET_MODELS))
    if kind == 'cli':
        # The two agent-level hooks bracket a CLI too, through the halves they
        # were factored into: a CLI has no loop for the middle four, and no
        # graph to hang middleware on at all.
        return _account(_cli_answer(choice, _validate(message)))
    if choice not in _AGENTS:
        _AGENTS[choice] = _build_agent(choice)
    try:
        # A real HumanMessage rather than a dict, so it carries an id that
        # `check_request` can write a capped question back over.
        # Measured 2026-08-18: with the six middleware nodes a tool hop costs
        # ~4 supersteps, so 12 allowed exactly one hop — a run that searched,
        # then searched and read, then answered (13 steps) died *after* its
        # final answer, one node short of close_the_log. 24 gives the loop
        # about five hops, still a hard ceiling rather than a budget.
        result = _AGENTS[choice].invoke(
            {'messages': [HumanMessage(content=message)]},
            config={'recursion_limit': 24})
    except WidgetUnavailable:
        raise
    except Exception as error:
        # A UI helper's failure is a stated 502, never a bare 500 — but the
        # reason travels with it.
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error
    reply = result['messages'][-1].content
    if isinstance(reply, list):
        reply = ' '.join(part.get('text', '') if isinstance(part, dict) else str(part)
                         for part in reply)
    # `close_the_log` already accounted for this run from inside the graph.
    return str(reply)
