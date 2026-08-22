"""The widget's model catalogue and the two answer paths.

The OpenRouter path is `langchain.agents.create_agent` with the six
middleware hooks (hooks.py), one cached agent per model; the CLI path is a
process per call with the knowledge base inlined, because `CliChat` has no
`bind_tools`. Both paths enter through `ask` for a whole answer at once, or
through `stream` for the same answer handed over as it is written.
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
# models run the tool loop and are held by the checkpointer, so a thread
# picks up where it left off; the two CLIs cannot run tools (`CliChat` has no
# `bind_tools`) and keep nothing — a CLI call is one process with no graph and
# no checkpointer, so it writes nothing to widget.db at all, and any earlier
# OpenRouter turns on that thread stay put while the CLI's own turn is never
# added. They cannot stream either — one subprocess reports one complete reply,
# so the answer lands in a single piece where an OpenRouter model's is typed
# out. Their labels say all of this — an option states what it cannot do.
WIDGET_MODELS = {
    'openai/gpt-5-nano': ('openrouter', 'gpt-5-nano · OpenRouter, tools'),
    'openai/gpt-5-mini': ('openrouter', 'gpt-5-mini · OpenRouter, tools'),
    'claude': ('cli', 'claude · CLI, no key, no tools, no memory, no stream'),
    'codex': ('cli', 'codex · CLI, no key, no tools, no memory, no stream'),
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

    # `stream_usage=True` is not a default here and has to be asked for:
    # langchain_openai enables it only for OpenAI's own base url, and this one
    # is OpenRouter's, so a streamed turn would arrive with no usage at all —
    # every reply reporting no bill, which reads on the page as "the backend
    # did not account for it" when the truth is that nobody asked. OpenRouter
    # serves `stream_options.include_usage`, so asking is enough; a backend
    # that still reports nothing lands as None, which is the honest reading.
    llm = ChatOpenAI(model=model, api_key=openrouter_api_key,
                     base_url=_openrouter_url(), stream_usage=True)
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


def _cli_turn(cli: str, message: str) -> dict:
    """One CLI call, accounted. `ask` and `stream` both enter here rather than
    each reading the message themselves: a CLI reply the page streamed and the
    same reply asked for outright must be the same turn, down to its bill."""
    answer = _cli_answer(cli, message)
    used = getattr(answer, 'usage_metadata', None)
    return _accounted(_account(str(answer.content)), [used] if used else [])


def _accounted(reply: str, used: list) -> dict:
    """The reply with its token account. No usage reported lands as None,
    never 0 — "0 tokens" is a claim about the bill, None says the backend
    did not account for it."""
    return {'reply': reply,
            'input_tokens': (sum(u.get('input_tokens') or 0 for u in used)
                             if used else None),
            'output_tokens': (sum(u.get('output_tokens') or 0 for u in used)
                              if used else None)}


def _model_kind(model: str) -> tuple[str, str]:
    """The choice and what kind of backend serves it. A name this installation
    does not serve is a refusal here, at the one gate both answer paths pass,
    never a quiet substitution of the default."""
    choice = model or DEFAULT_MODEL
    kind, _ = WIDGET_MODELS.get(choice) or (None, None)
    if kind is None:
        raise ValueError(f'{choice!r} is not a widget model; expected one of '
                         + ', '.join(repr(v) for v in WIDGET_MODELS))
    return choice, kind


def _run(message: str, thread: str) -> tuple[dict, dict]:
    """The graph's input and its config for one turn — read once, so `ask` and
    `stream` cannot come to disagree about which thread a turn ran under or
    what it stamped on the way in.

    One reading of the thread's name for both the id the graph runs under and
    the state written into it, stripped the way `history` and `forget` already
    strip theirs — a turn that ran under `' abc '` while the reader read back
    `'abc'` would be two threads wearing one name.

    A real HumanMessage rather than a dict, so it carries an id that
    `check_request` can write a capped question back over. `thread_stamp`
    rides in on the same input: `WidgetState`'s two fields are channels like
    `messages`, so writing them here is one checkpoint write rather than a
    second writer racing the graph, and `/api/widget/history` can report them
    as facts about the thread because a turn is what put them there.

    Measured 2026-08-18: with the six middleware nodes a tool hop costs ~4
    supersteps, so 12 allowed exactly one hop — a run that searched, then
    searched and read, then answered (13 steps) died *after* its final answer,
    one node short of close_the_log. 24 gives the loop about five hops, still a
    hard ceiling rather than a budget.
    """
    name = (thread or '').strip() or memory.GENERAL
    return ({'messages': [HumanMessage(content=message)],
             **memory.thread_stamp(name)},
            {'recursion_limit': 24, 'configurable': {'thread_id': name}})


def _turn_account(messages: list) -> tuple[str, list]:
    """This turn's reply and the usage behind it.

    The turn is the messages after the question just sent: under memory the
    state carries every earlier turn too, and an account summing the whole
    thread would bill the old turns again on every ask.

    The reply is rendered with `conversation_memory._text` — the one place a
    message's content becomes a string — so the reply the reader sees live and
    the turn the log shows later cannot drift into two different accounts of
    the same answer.
    """
    turn = messages
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            turn = messages[i + 1:]
            break
    used = [m.usage_metadata for m in turn
            if getattr(m, 'usage_metadata', None)]
    return memory._text(messages[-1].content), used


def _agent_for(model: str):
    """The cached agent for one model, built on first use."""
    with _AGENTS_LOCK:
        if model not in _AGENTS:
            _AGENTS[model] = _build_agent(model)
        return _AGENTS[model]


def ask(message: str, model: str = '', thread: str = '') -> dict:
    """One question in, one answer out: `{'reply', 'input_tokens',
    'output_tokens'}` — the account read from the `usage_metadata` LangChain
    puts on every AI message. `model` picks from WIDGET_MODELS; empty means
    the default. Agents build on first use, one per model. `thread` names the
    conversation to continue — the lab's active experiment, or `general` when
    it has none; empty lands on `general` rather than on a fresh id, because a
    reader who asked without an experiment open twice is having one
    conversation, not two. On the OpenRouter path the turn also stamps the
    thread's own two state fields (`conversation_memory.thread_stamp`); a CLI
    keeps nothing at all, so it stamps nothing either — the label says so.

    `stream` is the same turn, arriving as it is written. This is the whole
    answer at once, which is what a caller with nowhere to put a half-written
    one wants — the `__main__` harness, a test, a future non-browser client."""
    choice, kind = _model_kind(model)
    if kind == 'cli':
        # The two agent-level hooks bracket a CLI too, through the halves they
        # were factored into: a CLI has no loop for the middle four, and no
        # graph to hang middleware on at all. One process per call means no
        # memory either — the thread is accepted and ignored, the label
        # already says what a CLI cannot do.
        return _cli_turn(choice, _validate(message))
    agent = _agent_for(choice)
    payload, config = _run(message, thread)
    try:
        result = agent.invoke(payload, config=config)
    except WidgetUnavailable:
        raise
    except Exception as error:
        # A UI helper's failure is a stated 502, never a bare 500 — but the
        # reason travels with it.
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error
    reply, used = _turn_account(result['messages'])
    # `close_the_log` already accounted for this run from inside the graph.
    return _accounted(reply, used)


def _delta(chunk) -> str:
    """One streamed piece of the answer, or `''` for a piece that is not answer
    text. A tool call's arguments arrive token by token on the very channel the
    answer does, and so does the tool's result; neither is what the reader is
    watching being written, and the widget's log has never shown them
    (`conversation_memory._turns` drops the same two on the way back out)."""
    if getattr(chunk, 'type', '') not in ('ai', 'AIMessageChunk'):
        return ''
    if (getattr(chunk, 'tool_call_chunks', None)
            or getattr(chunk, 'tool_calls', None)):
        return ''
    return memory._text(getattr(chunk, 'content', ''))


def _stream_cli(cli: str, message: str):
    """A CLI streaming: one piece, because one subprocess reports one complete
    reply and there is no partial output to forward. It still travels this path
    so the page keeps one way to ask rather than two, and the catalogue label
    says which models actually type their answer out."""
    done = _cli_turn(cli, message)
    yield {'delta': done['reply']}
    yield done


def _stream_agent(agent, message: str, thread: str):
    """The graph streaming: `stream_mode=['messages', 'values']` on the same
    run `ask` invokes — the same nodes, the same middleware, the same
    checkpoint write, so a streamed turn is in widget.db exactly as an asked
    one is.

    Two modes and not one, because the pieces and the account answer different
    questions. `messages` is what the reader watches arrive. `values` is the
    state the run ended in, and that is where the final event comes from: the
    reply is read back out of the log with `_turn_account`, so what the page
    settles on is what the lab now holds for that turn rather than whatever the
    concatenated pieces happened to spell. A run that streamed pieces but ended
    with no state is a refusal, not a reply assembled here from the fragments —
    the same rule the rest of the lab keeps about a row that cannot say what
    produced it."""
    payload, config = _run(message, thread)
    final = None
    try:
        for mode, event in agent.stream(payload, config=config,
                                        stream_mode=['messages', 'values']):
            if mode == 'values':
                final = event
                continue
            text = _delta(event[0])
            if text:
                yield {'delta': text}
    except WidgetUnavailable:
        raise
    except Exception as error:
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error
    if not (final or {}).get('messages'):
        raise WidgetUnavailable(
            'the widget streamed no answer — the run ended with no state to '
            'read the reply back from')
    reply, used = _turn_account(final['messages'])
    yield _accounted(reply, used)


def stream(message: str, model: str = '', thread: str = ''):
    """The same turn as `ask`, handed over as it is written: an iterator of
    events, `{'delta': text}` for each piece and then exactly one final
    `{'reply', 'input_tokens', 'output_tokens'}` — the very dict `ask` returns.
    The last event is the authoritative one; the deltas are how it arrived.

    A call, not a generator function, and that distinction is the point:
    everything knowable before the first piece — an unserved model, a missing
    key, a CLI this machine does not have — is raised from *here*, while the
    route can still answer it with a status code. A generator would defer all
    three past the response's headers and turn a refusal into a 200 whose body
    apologises."""
    choice, kind = _model_kind(model)
    if kind == 'cli':
        text = _validate(message)
        if not cli_available(choice):
            raise WidgetUnavailable(
                f'the {choice} command is not on this machine — install and '
                'log in, or pick an OpenRouter model')
        return _stream_cli(choice, text)
    return _stream_agent(_agent_for(choice), message, thread)
