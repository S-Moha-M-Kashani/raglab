"""The widget's model catalogue and the two answer paths.

The OpenRouter path is `langchain.agents.create_agent` with the six
middleware hooks (hooks.py), one cached agent per model; the CLI path is a
process per call with the knowledge base inlined, because `CliChat` has no
`bind_tools`. Both paths enter through `ask`.
"""
import os
from threading import RLock

from langchain_core.messages import HumanMessage

from raglab.agents.widget import conversation_memory as memory
from raglab.agents.widget import skills_corpus_loader as skills
from raglab.llm_backends.cli_subprocess_chat import (
    CliChat,
    checked_effort,
    cli_available)
from raglab.configuration.env_settings import PROVIDER_MODELS
from raglab.agents.widget.hooks import MIDDLEWARE, _account, _validate
from raglab.agents.widget.prompts import (
    _PROMPTS,
    KNOWLEDGE_BASE,
    SYSTEM_PROMPT)
from raglab.agents.widget.tools import TOOLS

# Read at build time, never at import: the suite runs offline, and a missing
# variable must become a stated refusal rather than a KeyError at import.
# These are the *OpenRouter path's* requirement — the whole point of a CLI
# backend is that it needs no key at all.
REQUIRED_ENV = ('OPENROUTER_API_KEY',)

# The LangSmith four exist only to serve tracing, so they are demanded only
# when LANGSMITH_TRACING reads as on — requiring them with tracing off would
# be requiring a credential for a disabled feature.
TRACING_ENV = ('LANGSMITH_API_KEY', 'LANGSMITH_ENDPOINT',
               'LANGSMITH_PROJECT', 'LANGSMITH_TRACING')

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
DEFAULT_MODEL = 'openai/gpt-5-nano'


def _openrouter_url() -> str:
    """`.env` already carries OPENROUTER_BASE_URL for the lab's own backend;
    the widget reads the same variable rather than keeping a second copy."""
    return (os.environ.get('OPENROUTER_BASE_URL', '').strip()
            or 'https://openrouter.ai/api/v1')


def _environment_key(environment_key: str = '') -> str:
    return (environment_key or '').strip()


_openrouter_key_resolver = _environment_key


def set_openrouter_key_resolver(resolver=None) -> None:
    """Set the panel's key resolver; standalone use reads the environment."""
    global _openrouter_key_resolver
    _openrouter_key_resolver = resolver or _environment_key


def _openrouter_key() -> str:
    key = _openrouter_key_resolver(os.environ.get('OPENROUTER_API_KEY', ''))
    if not key:
        raise WidgetUnavailable(
            'OPENROUTER_API_KEY is not set — enter it under Settings or set it in .env')
    return key


class WidgetUnavailable(RuntimeError):
    """The lab is up; its widget is not. The route answers this as a 502."""


# One cached agent per OpenRouter model — a CLI is a process per call and
# caches nothing.
_AGENTS: dict = {}
_AGENTS_LOCK = RLock()


def reset() -> None:
    """Drop the cached clients so the next ask() rebuilds (tests, key changes).
    Not the memory: that outlives every client, and lives in widget.db."""
    with _AGENTS_LOCK:
        _AGENTS.clear()


def _tracing_on() -> bool:
    # The spellings the LangSmith SDK itself reads as on; anything else —
    # 'False', '0', '', or the variable simply not set — is off.
    return os.environ.get('LANGSMITH_TRACING', '').strip().lower() in ('true', '1')


def _build_agent(model: str):
    # `load_env` strips values, so a bare `KEY= ` line in .env lands here as
    # '' — an empty variable is a missing one, not a present one. With
    # tracing on the traces really leave the machine, so only then do the
    # LangSmith four join the demand.
    required = TRACING_ENV if _tracing_on() else ()
    for name in required:
        if not os.environ.get(name, '').strip():
            raise WidgetUnavailable(
                f'{name} is not set — the widget needs it in .env')
    openrouter_api_key = _openrouter_key()
    # Present, and read the way the spec states them. LangSmith picks its
    # four up from the environment by itself; only the key is passed on.
    if _tracing_on():
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
    # One checkpointer for the process, not one per agent: `reset()` clears
    # this cache on every credential change, and a memory living inside an
    # agent went with it — typing a key ended the conversation. The state is
    # two fields, deliberately (conversation_memory.WidgetState).
    # `trim_and_call` keeps the model's window at MAX_HISTORY however long a
    # remembered thread grows.
    return create_agent(llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT,
                        middleware=MIDDLEWARE,
                        state_schema=memory.WidgetState,
                        checkpointer=memory.saver())


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
        # The whole message, not its text: `CliChat` puts the CLI's own token
        # report on `usage_metadata`, and the account travels with the reply.
        return chat.invoke([('system', system), ('user', message)])
    except Exception as error:
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error


def _accounted(reply: str, used: list) -> dict:
    """The reply with its token account. No usage reported lands as None,
    never 0 — "0 tokens" is a claim about the bill, None says the backend
    did not account for it."""
    return {'reply': reply,
            'input_tokens': (sum(u.get('input_tokens') or 0 for u in used)
                             if used else None),
            'output_tokens': (sum(u.get('output_tokens') or 0 for u in used)
                              if used else None)}


def ask(message: str, model: str = '', thread: str = '') -> dict:
    """One question in, one answer out: `{'reply', 'input_tokens',
    'output_tokens'}` — the account read from the `usage_metadata` LangChain
    puts on every AI message. `model` picks from WIDGET_MODELS; empty means
    the default. Agents build on first use, one per model. `thread` names the
    conversation to continue — the lab's active experiment, or `general` when
    it has none; empty lands on `general` rather than on a fresh id, because a
    reader who asked without an experiment open twice is having one
    conversation, not two."""
    choice = model or DEFAULT_MODEL
    kind, _ = WIDGET_MODELS.get(choice) or (None, None)
    if kind is None:
        raise ValueError(f'{choice!r} is not a widget model; expected one of '
                         + ', '.join(repr(v) for v in WIDGET_MODELS))
    if kind == 'cli':
        # The two agent-level hooks bracket a CLI too, through the halves they
        # were factored into: a CLI has no loop for the middle four, and no
        # graph to hang middleware on at all. One process per call means no
        # memory either — the thread is accepted and ignored, the label
        # already says what a CLI cannot do.
        answer = _cli_answer(choice, _validate(message))
        used = getattr(answer, 'usage_metadata', None)
        return _accounted(_account(str(answer.content)),
                          [used] if used else [])
    with _AGENTS_LOCK:
        if choice not in _AGENTS:
            _AGENTS[choice] = _build_agent(choice)
        agent = _AGENTS[choice]
    try:
        # A real HumanMessage rather than a dict, so it carries an id that
        # `check_request` can write a capped question back over.
        # Measured 2026-08-18: with the six middleware nodes a tool hop costs
        # ~4 supersteps, so 12 allowed exactly one hop — a run that searched,
        # then searched and read, then answered (13 steps) died *after* its
        # final answer, one node short of close_the_log. 24 gives the loop
        # about five hops, still a hard ceiling rather than a budget.
        result = agent.invoke(
            {'messages': [HumanMessage(content=message)]},
            config={'recursion_limit': 24,
                    'configurable': {'thread_id': thread or memory.GENERAL}})
    except WidgetUnavailable:
        raise
    except Exception as error:
        # A UI helper's failure is a stated 502, never a bare 500 — but the
        # reason travels with it.
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error
    messages = result['messages']
    # This turn's messages are the ones after the question just sent: under
    # memory the state carries every earlier turn too, and an account summing
    # the whole thread would bill the old turns again on every ask.
    turn = messages
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            turn = messages[i + 1:]
            break
    used = [m.usage_metadata for m in turn
            if getattr(m, 'usage_metadata', None)]
    # The same rendering `conversation_memory.history()` reads back with —
    # `_text` is the one place a message's content becomes a string, so the
    # reply the reader sees live and the turn the log shows later cannot
    # drift into two different accounts of the same answer.
    reply = memory._text(messages[-1].content)
    # `close_the_log` already accounted for this run from inside the graph.
    return _accounted(reply, used)
