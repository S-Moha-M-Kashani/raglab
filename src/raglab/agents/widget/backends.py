"""The widget's model catalogue and the two answer paths.

The OpenRouter path is `langchain.agents.create_agent` with the four
middleware hooks (hooks.py), one cached agent per model; the CLI path is a
process per call with the knowledge base inlined, because `CliChat` has no
`bind_tools`. Both paths enter through `ask` for a whole answer at once, or
through `stream` for the same answer handed over as it is written.

One turn, one order, both paths: what the answer needs runs before the answer,
what the *filing* needs runs after it. First the deterministic relevance
guard, then the agent — everything a keyless or unserved install can refuse
without reading a durable file refuses here — and only then `_preflight`: the
thread's trusted dataset, resolved *once*, because each resolution is a ledger
query plus a run-file read, and the long-term memory filed under that dataset.
After the answer (`_finish_memory`, on a thread of its own for both paths):
the memory policy and every provenance gate, then the summary and the write.

Until 2026-08-28 the policy ran first, so the reader waited two model round
trips for the first word: the policy was being read as a gate on *answering*.
It is a gate on *filing*. The widget writes no measured number, so its
relevance check is a scope guard; every provenance guarantee — the validated
experiment context, the dataset match, the foreign-experiment refusal,
fail-closed saving — stands in front of the *write*, which is where CLAUDE.md
puts them. Two stated consequences: a question the deterministic guard does
not catch but the policy would have called irrelevant is now answered, and a
thread naming an experiment the records cannot yet identify — a job still
running is the common case, and the moment a reader is most likely to ask — is
answered too. Neither is filed.

Whatever the decision turns out to be, its reason is written on the turn's own
`widget_turn_log` row (`_record_memory_outcome`): the deciding thread has no
caller left to return to, and a turn left unfiled with no record of why would
be exactly the silence CLAUDE.md forbids. For the same reason every one of
this file's operational writes goes through `_safely` — three writers share
widget.db, and a row that could not be written is reported, never fatal to an
answer the reader already has.

A turn that never answers says so on a row of its own
(`_log_interrupted_turn`). A run can die after the graph has already written
the question and a tool exchange, and until 2026-08-29 that turn was recorded
nowhere at all while its tokens were still spent — the same silence, one step
earlier.
"""
import os
import time
import uuid
from threading import RLock, Thread
from typing import NamedTuple

from langchain_core.messages import HumanMessage, SystemMessage

from raglab.agents.widget import conversation_memory as memory
from raglab.agents.widget import experiment_tools
from raglab.agents.widget import hooks
from raglab.agents.widget import skills_corpus_loader as skills
from raglab.llm_backends.cli_subprocess_chat import (
    CliChat,
    checked_effort,
    cli_available)
from raglab.configuration.env_settings import PROVIDER_MODELS
from raglab.agents.widget.hooks import (MIDDLEWARE, _account, _validate,
                                        evaluate_memory_policy,
                                        summarize_memory_update)
from raglab.agents.widget.prompts import (
    _PROMPTS,
    ACTIVE_EXPERIMENT_PROMPT,
    KNOWLEDGE_BASE,
    SYSTEM_PROMPT)
from raglab.agents.widget import tools
from raglab.agents.widget import turn_logger
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
# no checkpointer, so it writes no short-term conversation state; its readable
# turn is still recorded in widget_turn_log. Any earlier OpenRouter turns on
# that thread stay put while the CLI's own turn is never added to the thread.
# They cannot stream either — one subprocess reports one complete reply,
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
# caches nothing. Beside it, one cached policy/summarizer client per model:
# the same key, the same base url, the same lock, and the same fate on
# `reset()`, because both are built from the credential a reader can retype.
_AGENTS: dict = {}
_MEMORY_MODELS: dict = {}
_AGENTS_LOCK = RLock()


def reset() -> None:
    """Drop the cached clients so the next ask() rebuilds (tests, key changes).
    Both caches, together: a key typed into the panel invalidates the agent and
    the policy client alike, and one of them surviving would be a turn answered
    by the new credential and filed by the old one.
    Not the memory: that outlives every client, and lives in widget.db."""
    with _AGENTS_LOCK:
        _AGENTS.clear()
        _MEMORY_MODELS.clear()


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


def _run(message: str, thread: str, *, dataset: str = '',
         memory_state: dict | None = None,
         memory_text: str = '') -> tuple[dict, dict]:
    """The graph's input and its config for one turn — read once, so `ask` and
    `stream` cannot come to disagree about which thread a turn ran under or
    what it stamped on the way in.

    `dataset` is the thread's trusted dataset, handed in rather than looked up:
    `_preflight` resolved it for this turn already, and resolving it a second
    time here would be a second ledger query and a second run-file read for an
    answer that cannot have changed mid-turn.

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

    Measured 2026-08-18: with six middleware nodes a tool hop cost ~4
    supersteps, so 12 allowed exactly one hop — a run that searched, then
    searched and read, then answered (13 steps) died *after* its final answer,
    one node short of close_the_log. Raising the number bought headroom but
    never a real budget, because the ceiling and `hooks.MAX_TOOL_HOPS` were
    two numbers nobody had tied together.

    Folded on 2026-08-28: `note_prompt`/`check_reply` moved from graph nodes
    into `hooks.trim_and_call`, so a hop now costs `hooks.SUPERSTEPS_PER_HOP`
    (2) supersteps instead of 4. `hooks.RECURSION_LIMIT` is computed from that
    and `MAX_TOOL_HOPS` — see the arithmetic there — so it admits every
    sequential hop the guard allows *and* the answer after them. That makes
    `stop_repeated_tool_hops` the thing that stops a pathological loop, not
    this ceiling, and it is why the two can no longer silently drift apart.
    """
    name = (thread or '').strip() or memory.GENERAL
    messages: list = []
    # An experiment's thread opens by saying which experiment it is — from the
    # validated record, so an unknown or general thread says nothing. Without
    # it "the last experiment" could only mean the newest on the board, and a
    # meetings-de thread was once told about a smoke-import-check run.
    # ...and said once. The graph appends every input message to the thread,
    # so a line added on each turn was reread on each turn — a five-turn
    # thread carried ten system messages. A line that is already the *newest*
    # standing line of its kind is not written again; anything else is.
    #
    # Newest of its kind, not "somewhere in the thread". The weaker test read
    # fine while the model was handed every line, and became a quality bug the
    # moment only the newest reached it: memory can shrink as well as grow —
    # `long_term_memory` can be cleared and regrow — so a turn whose context
    # byte-matched an older line wrote nothing, and the call then carried the
    # newer, longer line instead of this turn's memory. The one guarantee this
    # whole change exists to give is that the model reads the current memory.
    #
    # It is the *thread* that keeps them all. The identity line cannot change,
    # so there is only ever one; the memory context grows on every accepted
    # turn, so a long thread accumulates several, each superseded by the next.
    # That accumulation is deliberate and stays: the conversation log is a
    # record of what the widget was actually told, `long_term_memory._bounded`
    # caps the stored aggregate so an older line can be the last surviving copy
    # of a note the store has since truncated away, and deleting from a channel
    # two concurrent turns both read is a race — the second turn's delete
    # arrives after the id is gone and langgraph raises. What is bounded is the
    # *prompt*: `hooks.trim_and_call` sends the newest line of each kind and
    # leaves the rest out of that call, which is why every line written here
    # carries a marker saying which standing line it is.
    #
    # The two lines are independent — a turn may write either, both or neither.
    # A thread whose dataset could not be resolved writes no identity line and
    # still carries whatever memory context it was handed.
    newest: dict = {}   # the thread's newest standing line, per kind
    unmarked = set()    # system lines that carry no marker, by their text
    for held in (memory._channels(name).get('messages') or []):
        if getattr(held, 'type', '') != 'system':
            continue
        mark = memory.standing_mark(held)
        if mark:
            newest[mark] = str(held.content)
        else:
            unmarked.add(str(held.content))
    opening = (ACTIVE_EXPERIMENT_PROMPT.format(experiment_id=name,
                                               dataset=dataset)
               if dataset else '')
    for line, mark in ((opening, memory.IDENTITY_LINE),
                       (memory_text, memory.MEMORY_LINE)):
        if not line or line == newest.get(mark):
            continue
        # A thread whose lines predate the marker says the same thing in an
        # unmarked line. Nothing can supersede one of those, so writing a
        # marked copy beside it would put the same text in the call twice.
        if mark not in newest and line in unmarked:
            continue
        messages.append(SystemMessage(
            content=line, additional_kwargs={memory.STANDING_LINE: mark}))
    messages.append(HumanMessage(content=message))
    return ({'messages': messages, **memory.thread_stamp(name),
             **(memory_state or {})},
            {'recursion_limit': hooks.RECURSION_LIMIT,
             'configurable': {'thread_id': name}})


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


def _turn_steps(messages: list) -> list[dict]:
    """Turn final graph messages into a small, readable execution trace."""
    steps = []
    for message in messages:
        kind = getattr(message, 'type', '')
        if kind == 'human':
            step_kind = 'human'
        elif kind == 'tool':
            step_kind = 'tool'
        elif kind == 'ai':
            step_kind = 'ai'
        else:
            continue
        step = {'id': str(getattr(message, 'id', '') or uuid.uuid4()),
                'kind': step_kind,
                'message_id': getattr(message, 'id', None),
                'latency_ms': None}
        if step_kind == 'tool':
            step['name'] = getattr(message, 'name', '') or ''
        if step_kind in ('human', 'ai'):
            step['text'] = memory._text(getattr(message, 'content', ''))
        usage = getattr(message, 'usage_metadata', None) or {}
        metadata = getattr(message, 'response_metadata', None) or {}
        step['latency_ms'] = metadata.get('latency_ms') or metadata.get('duration_ms')
        if usage:
            step['input_tokens'] = usage.get('input_tokens')
            step['output_tokens'] = usage.get('output_tokens')
        steps.append(step)
    return steps


def _policy_transcript(thread: str) -> str:
    """The turns already in the thread, for short follow-ups such as ``and?``.

    Read in `_preflight`, before the graph writes this turn, and carried to
    `_finish_memory` — which is where the policy is asked now. Reading it there
    would read the checkpoint the turn has just been written into: the question
    would appear twice in the prompt, once as itself and once as the last line
    of its own "prior conversation", the answer being judged would be part of
    the context judging it, and the exchange the follow-up actually depends on
    would be pushed out of the `MAX_RECALLED` window by the pair that arrived
    while it was being judged.
    """
    if not (thread or '').strip():
        return ''
    try:
        turns = memory.history(thread).get('turns') or []
    except Exception:
        return ''
    return '\n'.join(f"{turn.get('role', 'message')}: {turn.get('text', '')}"
                     for turn in turns[-memory.MAX_RECALLED:]
                     if turn.get('text'))


def _safely(work, note: str, fallback=None):
    """Run an operational write that must never cost a reader their answer.

    The rule this lab already keeps for the ledger, applied to the widget's own
    files: a failed write is reported, never fatal (CLAUDE.md, "a failed write
    is reported on the job, never fatal"). Three writers share `widget.db` —
    the checkpointer, the long-term store and the turn log — each behind its
    own lock, and since the memory pass moved onto a daemon thread one turn's
    writer can hold the file while the next turn's does not. They wait
    (`BUSY_TIMEOUT_SECONDS`), which settles it in every ordinary case; what
    this covers is the one that outlasts the wait, where the alternative is a
    502 on a turn that answered perfectly well. `HOOK_LOG` is the only reader
    available to a write that failed: the row it would have gone on is the
    thing that could not be written.
    """
    try:
        return work()
    except Exception as error:
        hooks._fired('operational', f'{note}: {error}')
        return fallback


def _log_turn(*, message: str, reply: str, thread: str, started: float,
              input_tokens=None, output_tokens=None, messages=None,
              dataset_id='', status='answered', status_reason='',
              invent_steps=True, ai_message_id='') -> str:
    """Write the human-readable operational row after the answer exists.

    `dataset_id` is the turn's trusted dataset, not the policy's opinion of
    one: the row says which corpus the thread stood on while it answered, and
    that is known before the answer exists rather than after.

    Returns `''` when the row could not be written — see `_safely`. Every
    caller already treats an empty turn id as "there is no row to amend",
    which is exactly what is true then.

    `invent_steps` is the one thing a caller may switch off. A turn that
    answered always has at least a question and an answer, so standing a
    two-step trace in for a state nobody kept is a fair reading of a run that
    finished. A turn that did *not* answer has no such pair to assume: writing
    an `ai` step for an answer that never existed would be the row lying about
    what produced it, in the smallest possible way. An interrupted turn's
    steps are what happened, or nothing."""
    name = (thread or '').strip() or memory.GENERAL
    experiment_id = name if name != memory.GENERAL else ''
    user_id = ''
    if messages:
        for item in messages:
            if isinstance(item, HumanMessage):
                user_id = str(getattr(item, 'id', '') or '')
    trace = _turn_steps(messages or [])
    if not trace and invent_steps:
        trace = [{'id': str(uuid.uuid4()), 'kind': 'human',
                  'text': message, 'latency_ms': 0},
                 {'id': str(uuid.uuid4()), 'kind': 'ai', 'text': reply,
                  'latency_ms': None}]
    return _safely(
        lambda: turn_logger.log_turn(
            thread_id=name, experiment_id=experiment_id, dataset_id=dataset_id,
            user_message_id=user_id or str(uuid.uuid4()), user_message=message,
            ai_message_id=ai_message_id, ai_message=reply,
            steps=trace,
            total_input_tokens=input_tokens, total_output_tokens=output_tokens,
            total_latency_ms=max(0, round((time.monotonic() - started) * 1000)),
            status=status, status_reason=status_reason),
        'the turn-log row was not written', '')


def _log_interrupted_turn(*, message: str, asked, thread: str,
                          started: float, dataset_id: str,
                          reason: str) -> str:
    """The row a turn that never answered still owes.

    Until 2026-08-29 a run that died after the graph had written something
    left nothing behind but the conversation: no row, no status, and its
    tokens billed nowhere. One real thread showed three questions in
    `/dev/trace` against two logged turns, the missing one having spent 2,192
    input and 603 output tokens on a `read_experiment` call whose answer the
    reader never saw. A turn that fails is still a turn that cost money, and a
    lab that writes one row per job does not get to skip the row when the job
    goes wrong — that is precisely when the row is worth having.

    The account comes from the checkpoint rather than from a returned state,
    because there is no returned state: the run raised. langgraph writes a
    checkpoint per superstep, so whatever the run managed — the question, a
    tool call, its reply, each with the usage the model reported — is in the
    thread, and this reads the question and everything after it, the same span
    `_turn_account` calls a turn. The reply is empty because there was none;
    `status` says `interrupted` and `status_reason` says what raised.

    The span is claimed **by the question's own id**, and `asked` is the very
    `HumanMessage` `_run` built for this turn. Two facts make that the right
    key rather than the message's text. `add_messages` stamps a uuid onto the
    object it was handed, in place, so a `None` id here is proof that the
    input write never happened — a locked checkpointer, a `before_agent` that
    raised, a bad config — and there is then no span of ours to claim at all.
    And `check_request` writes a capped question back under the *same* id, so
    the id survives the one edit the graph makes to it.

    Text cannot do this job, and the case it fails is the one this whole task
    came from. The live trace is a reader who asked, watched the turn die, and
    **asked the same thing again**: on that retry the previous turn's question
    is identical text, so a text check claims its span and bills its tokens and
    its steps to the new row. That is the double-billed account and the row
    lying about what produced it, arriving through the most likely path there
    is. An id is different every turn, so it cannot.

    A span that is not this turn's is no span at all: the run wrote nothing of
    its own, the account is `None` rather than someone else's number, and the
    row carries no steps.

    Never fatal, and never the reason a reader sees. Every write here goes
    through `_safely`, and the caller re-raises the original failure the moment
    this returns: what the reader is owed is the error, and what the lab is
    owed is the row.
    """
    name = (thread or '').strip() or memory.GENERAL
    held = _safely(lambda: memory._channels(name).get('messages') or [],
                   'the interrupted turn could not be read back', []) or []
    question_id = str(getattr(asked, 'id', '') or '')
    turn: list = []
    # No id means no write, so there is nothing to look for — and looking
    # anyway would match the first message the thread holds with no id of its
    # own, which is the very mistake this check exists to stop.
    for i, held_message in enumerate(held if question_id else []):
        if str(getattr(held_message, 'id', '') or '') == question_id:
            turn = list(held[i:])
            break
    if question_id and not turn:
        hooks._fired('operational',
                     'the interrupted turn wrote nothing of its own: its '
                     'question is not in the thread')
    elif not question_id:
        hooks._fired('operational',
                     'the interrupted turn wrote nothing of its own: the '
                     'graph never wrote its question')
    used = [m.usage_metadata for m in turn
            if getattr(m, 'usage_metadata', None)]
    account = _accounted('', used)
    return _log_turn(message=message, reply='', thread=thread, started=started,
                     input_tokens=account['input_tokens'],
                     output_tokens=account['output_tokens'], messages=turn,
                     dataset_id=dataset_id, status='interrupted',
                     status_reason=reason, invent_steps=False)


def _agent_for(model: str):
    """The cached agent for one model, built on first use."""
    with _AGENTS_LOCK:
        if model not in _AGENTS:
            _AGENTS[model] = _build_agent(model)
        return _AGENTS[model]


def _build_memory_model(model: str):
    """Build the structured-output seam the memory policy and the summarizer
    speak through — a separate client from the agent's, so policy availability
    is never confused with answer-model availability."""
    openrouter_api_key = _openrouter_key()
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, api_key=openrouter_api_key,
                      base_url=_openrouter_url())


def _memory_model(model: str):
    """The cached policy/summarizer client for one model, built on first use.

    Cached the way `_agent_for` caches agents, and for the same reason: it was
    a fresh `ChatOpenAI` on every turn while the agent beside it was reused, so
    every turn paid for a new connection and its handshake to say one
    structured sentence."""
    with _AGENTS_LOCK:
        if model not in _MEMORY_MODELS:
            _MEMORY_MODELS[model] = _build_memory_model(model)
        return _MEMORY_MODELS[model]


class _Preflight(NamedTuple):
    """What a turn settles before any model is asked anything.

    `dataset` is the thread's trusted dataset id, resolved once for the whole
    turn; `context` is the long-term memory filed under it; `state` is what the
    turn stamps on the thread's own policy channels; `transcript` is what the
    thread held before this turn, which only a reading taken before the turn
    can say. Nothing here refuses: the one refusal a turn can raise without a
    model — `hooks.relevance_guard` — is the caller's, and every other refusal
    this widget has is about the write.
    """
    dataset: str
    context: str
    state: dict
    transcript: str


def _refused(reason: str) -> dict:
    """A decision that is its own answer: nothing ran, so nothing was filed.

    It used to carry `'blocked': True` as well. Two pre-flight branches read
    it while the memory policy still ran before the answer; both went with the
    policy on 2026-08-28, leaving a field written on every refusal and read by
    a test assertion and nothing else. A key nobody reads is not documentation,
    so it is gone and `relevant`/`saved` say the same thing to the one reader
    there is (`panel_server._safe_widget_event`).
    """
    return {'relevant': False, 'should_save': False, 'dataset_id': '',
            'subtopic': '', 'reason': reason, 'saved': False}


def _unfiled(reason: str) -> dict:
    """A turn that ran to the end and still must not be filed.

    One case, and it is the hop guard's: `hooks.trim_and_call` answers with
    `HOP_GUARD_REFUSAL` instead of calling the model, so the run produces a
    well-formed reply that is the widget apologising rather than anything
    about the lab. Filing it would put the apology in long-term memory as a
    fact about an experiment, and `pending` would promise a decision that must
    never be taken.

    Its own status word rather than `_refused`'s: the deterministic guard
    refuses a question and this one abandons a lookup, and a route that called
    the second `irrelevant` would be describing the turn as something it was
    not. Statuses in this lab say what happened.
    """
    return {'status': 'not_filed', 'saved': False, 'reason': reason}


def _experiment_of(thread: str) -> str:
    """The experiment a thread is about, or `''` for the general one.

    One reading of the thread's name, used by everything that has to agree
    about it: the dataset lookup, the state stamp, and the save gate that asks
    whether this thread names an experiment at all.
    """
    name = (thread or '').strip() or memory.GENERAL
    return '' if name == memory.GENERAL else name


def _preflight(thread: str) -> _Preflight:
    """Everything one turn can decide without asking a model.

    The deterministic refusal (`hooks.relevance_guard`) has already been
    answered by the caller — a CLI turn needs it too, and never reaches here.
    What is left needs no model at all: the validated records say which dataset
    this thread stands on, and the long-term memory is filed by that id.

    A thread whose experiment the records cannot identify is *not* refused
    here, and that is a ruling rather than an omission (2026-08-28). It reads
    as a provenance guard, but refusing to answer protects nothing the write
    gate does not already protect: the record-reading tools say honestly when
    they cannot validate a thread, and `foreign_experiments` is what actually
    stops one dataset's thread being told about another's run. Refusing was
    also worst exactly when it hurt most — an experiment still running is not
    in the ledger yet, and that is when a reader asks about it. So the check
    moved to `_finish_memory`, beside the other guards on the write.
    """
    experiment = _experiment_of(thread)
    dataset = experiment_tools.trusted_dataset_id(experiment)
    context = ''
    if dataset:
        reader = tools.read_long_term_memory
        context = (reader.invoke({'dataset_id': dataset})
                   if hasattr(reader, 'invoke') else reader(dataset))
    # `WidgetState`'s own channels record what this turn stood on while it
    # ran: the trusted dataset, and which experiment's thread it is. The
    # verdict that arrives after the answer travels on the returned event and
    # on the `widget_turn_log` row, not back into these channels — a second
    # writer amending them afterwards would be racing the graph over the same
    # checkpoint, which is why the four channels that could only ever have
    # held that verdict were removed rather than left stamped with defaults.
    state = memory.dataset_stamp(dataset, experiment)
    return _Preflight(dataset, context, state, _policy_transcript(thread))


def _summarize_memory_update(**kwargs):
    return summarize_memory_update(**kwargs)


def _finish_memory(question: str, answer: str, model: str, thread: str,
                   dataset: str, turn_id: str = '',
                   transcript: str = '', policy_model=None) -> dict | None:
    """The memory decision, taken after the answer exists, and the write it
    permits.

    The policy runs here rather than before the agent because what it judges is
    whether a *finished* turn is worth keeping — a question about an answer
    that did not exist yet. It no longer gates the answer; it gates the file.

    Four things must all hold before anything is written: the policy calls the
    turn relevant and asks to save it, the records could identify the
    experiment this thread is about, the dataset the policy names is the one
    those records gave the thread, and the answer names no experiment from
    another corpus. A policy that is unavailable or malformed satisfies none of
    them and carries its own reason on the decision it returns.

    `transcript` is what the thread held *before* this turn, read in
    `_preflight`: a follow-up like `and?` is judged against the exchange it
    depends on, not against the answer it just produced.

    `policy_model` is the client `_defer_memory` already built — building one
    is how it tells a decision that can be made from one that cannot, and
    building a second here would be the same cached lookup twice for one turn.
    A direct caller hands nothing and this builds its own.

    `None` means there was no policy to ask at all — no client could be built —
    so the turn says nothing about memory rather than inventing a verdict.

    Every branch below sets a `reason`, and that is load-bearing rather than
    decorative: `_record_memory_outcome` is what carries it to the turn's row,
    and a branch that returned without one would leave a turn unfiled with
    nothing anywhere to say why.
    """
    if policy_model is None:
        try:
            policy_model = _memory_model(model)
        except Exception:
            return None
    policy = evaluate_memory_policy(
        question, policy_model, experiment_id=thread, dataset_id=dataset,
        trusted_dataset_id=dataset, conversation=transcript)
    decision = {**policy.model_dump(), 'saved': False}
    if hooks.policy_unreached(policy):
        # No verdict was reached — the judge was missing or its answer could
        # not be read. Saving fails closed either way, which is why this looks
        # like a refusal on every other field; but a reader told "not relevant"
        # when nobody judged their question has been told something untrue, so
        # the decision says which of the two happened and the panel reads it.
        decision['unavailable'] = True
        return decision
    if not policy.relevant or not policy.should_save:
        return decision
    experiment = _experiment_of(thread)
    if experiment and experiment_tools.experiment_reader_wired() and not dataset:
        # The thread is about an experiment and the validated records cannot
        # say which one — an experiment still running is not in the ledger yet.
        # A row filed under a thread whose provenance nobody could read would
        # be memory about an experiment this lab cannot name, so it fails
        # closed here rather than in front of the answer: the reader asking
        # about a run in progress is served, and nothing is written.
        decision.update({
            'should_save': False, 'saved': False, 'dataset_id': '',
            'reason': ('The active experiment context could not be validated; '
                       'nothing was filed as memory.')})
        return decision
    if not dataset or policy.dataset_id != dataset:
        # Filing is a claim of provenance, so the only dataset a turn may be
        # filed under is the validated one. A policy that named another — or
        # named one on a thread that has none — is answered and not filed; its
        # model-supplied identity is not evidence. (`evaluate_memory_policy`
        # already refuses the mismatch when a trusted dataset was passed to it;
        # this is the rule stated where the write happens.)
        decision.update({
            'should_save': False, 'saved': False, 'dataset_id': dataset,
            'reason': (f'The memory policy named dataset '
                       f'{policy.dataset_id or "none"}, but this thread stands '
                       f'on {dataset or "no validated dataset"}, so nothing '
                       'was filed.')})
        return decision
    # One reading of the board for this whole pass, handed to both things that
    # need it — the foreign-experiment refusal below and the validated ids the
    # write is checked against. Each used to take its own, so a saved turn read
    # up to `SCAN` run files off disk twice over.
    board = experiment_tools.board_snapshot()
    foreign = experiment_tools.foreign_experiments(answer, dataset, rows=board)
    if foreign:
        named = ', '.join(f'{i} ({d})' for i, d in sorted(foreign.items()))
        decision.update({
            'saved': False, 'should_save': False,
            'reason': (f'The answer describes experiments on another dataset '
                       f'— {named} — so it was not filed as memory about '
                       f'{dataset!r}.')})
        return decision
    try:
        summary = _summarize_memory_update(
            question=question, answer=answer,
            dataset_id=dataset, experiment_id=thread,
            subtopic=decision.get('subtopic', ''), model=policy_model)
        dataset_summary = (summary.get('dataset_summary', '')
                           if isinstance(summary, dict)
                           else summary.dataset_summary)
        global_summary = (summary.get('global_summary', '')
                          if isinstance(summary, dict)
                          else summary.global_summary)
        arguments = {
            'dataset_id': dataset,
            'experiment_id': thread, 'subtopic': decision.get('subtopic', ''),
            'question': question, 'answer': answer,
            'dataset_summary': dataset_summary,
            'global_summary': global_summary,
            # The board's ids plus this thread's own, added here rather than
            # asked for by name. `validated_dataset_ids(thread)` would resolve
            # `trusted_dataset_id` a second time — another ledger query and
            # run-file read for a value `pre.dataset` settled once at the top
            # of the turn — and it would hand it the raw thread, where
            # everything else in this file reads the experiment through
            # `_experiment_of` and so knows that `general` names no
            # experiment. `dataset` is non-empty by the gate above.
            'validated_dataset_ids': (
                experiment_tools.validated_dataset_ids(rows=board) | {dataset})}
        writer = tools.save_widget_memory
        stored = (writer.invoke(arguments) if hasattr(writer, 'invoke')
                  else writer(**arguments))
        decision.update({'saved': bool(stored.get('saved')), 'save': stored})
        # Each summary is checked on its own way in
        # (`long_term_memory.names_one_corpus`) and each can be refused on its
        # own: a global note may name no corpus, a dataset note may name only
        # the one it is filed under. Both reasons ride on the decision as
        # their own fields so `_record_memory_outcome` can put them on the
        # turn's row — `saved` already follows the dataset half, and loading it
        # with the global half too would make it a lie in one direction or the
        # other.
        for field in ('dataset_refused', 'global_refused'):
            refused = str(stored.get(field) or '')
            if refused:
                decision[field] = refused
        if turn_id and stored.get('update_id') is not None:
            # Outside the `except` below on purpose: the summary is already in
            # widget.db by now, so a failure to link it to the turn's row is a
            # missing cross-reference, not a failed save. Reporting it as
            # `saved: False` would be the row lying about what it holds.
            _safely(lambda: turn_logger.attach_memory_update(
                turn_id, stored['update_id']),
                'the memory update was written but not linked to its turn')
    except Exception as error:
        decision.update({'saved': False, 'save_error': str(error)})
    return decision


def _record_memory_outcome(turn_id: str, decision: dict | None) -> None:
    """Put the memory decision's reason on the turn's own row.

    The decision is taken after the reader has the answer, on a thread with no
    caller to return to, so until 2026-08-28 every outcome that was not a save
    went nowhere at all: the foreign-experiment refusal, the dataset mismatch,
    the unvalidated experiment, and a `save_error` from a summarizer that
    raised were each set on a dict the daemon then dropped. A turn was left
    unfiled with no record anywhere of why — which is the one thing a record
    in this lab may not do.

    `widget_turn_log` and not the alternatives: `turn_logger.attach_memory_outcome`
    says why that seam. The line is written for a save too, so the column
    means "what the memory pass decided" rather than "what went wrong" — a
    column only ever filled by failures reads as a failure list, and a blank
    one then cannot be told from a pass that never happened.
    """
    if decision is None:
        # No policy client could be built at all, so nothing was judged. The
        # turn says so rather than inventing a verdict — the same distinction
        # `_defer_memory` draws before it starts a thread.
        turn_logger.attach_memory_outcome(
            turn_id, 'not filed: no memory policy was available to judge this '
                     'turn.')
        return
    # A refused summary is a partial outcome, and this row is the record the
    # read-time filter deliberately does not duplicate into the prompt: a note
    # held back is a thing a reader may need to find, and the row is where a
    # reader looks. The dataset half decides the opening word, because it is
    # what `saved` follows — a turn whose only stored trace is its provenance
    # row was not filed, and the policy's own "reusable" would be a strange
    # thing to print after "not filed".
    error = decision.get('save_error')
    if error:
        reason = f'not filed: the memory write failed ({error}).'
    elif decision.get('dataset_refused'):
        reason = f"not filed: the dataset note {decision['dataset_refused']}."
    elif decision.get('saved'):
        reason = f"filed: {decision.get('reason') or 'the policy accepted it.'}"
    else:
        reason = f"not filed: {decision.get('reason') or 'the policy declined.'}"
    if decision.get('global_refused'):
        reason = (f'{reason} The cross-dataset note was not kept: it '
                  f"{decision['global_refused']}.")
    turn_logger.attach_memory_outcome(turn_id, reason)


def _memory_after(question: str, answer: str, model: str, thread: str,
                  pre: _Preflight, turn_id: str, stopped: bool) -> dict:
    """What happens to memory once the answer exists — one decision, both paths.

    `stopped` says the run ended in `hooks.stop_repeated_tool_hops`'s refusal
    rather than in an answer. That refusal is new since 2026-08-28: the guard
    used to append a message and let the run die on the recursion ceiling, so
    a hop-stopped turn never reached here at all. It now produces a real reply
    and would flow straight into `_defer_memory`, be judged, and be summarized
    into long-term memory as though the widget's apology were a fact about an
    experiment. Low probability and wrong in kind, so it is stopped here: a
    refusal is not an answer, and the row says `refused` rather than
    `answered` for the same reason.
    """
    if stopped:
        reason = ('not filed: the tool-hop guard stopped this run, so its '
                  'reply is the widget refusing rather than an answer about '
                  'the lab.')
        _safely(lambda: turn_logger.attach_memory_outcome(turn_id, reason),
                'the hop-guard note did not reach the turn-log row')
        return _unfiled(reason)
    return _defer_memory(question, answer, model, thread, pre.dataset,
                         turn_id, pre.transcript)


def _defer_memory(question: str, answer: str, model: str, thread: str,
                  dataset: str, turn_id: str = '',
                  transcript: str = '') -> dict:
    """Start the memory decision after the caller has received the answer.

    What comes back is a status and not a verdict, because the verdict is not
    made yet: the policy runs on the thread this starts. `saved: False` is the
    honest reading of that moment — nothing has been filed — and `pending` is
    what the panel shows for it (`panel_server._safe_widget_event`), because a
    turn whose memory line is simply missing reads as a turn nobody wanted to
    keep.

    Both paths defer, since 2026-08-28. A streamed turn does have a later event
    to spend, and used to spend it on the resolved decision — but it could only
    resolve one by holding the event stream open through the policy call, the
    summarizer call and two full readings of the board, all after the reader
    had every word of the answer. It now sends this same status on that event.

    `pending` is a promise that a decision is coming, so it is not made where
    none can be: an installation with no policy client to build will never
    resolve anything, and that turn reports `unavailable` — the word the panel
    already uses for a judge nobody could reach — rather than a wait with no
    end. Building the client is that test, and the one built here is the one
    the deferred work uses — the same turn asks for its judge once.

    Whatever the decision turns out to be, its reason lands on the turn's row
    (`_record_memory_outcome`). The status returned here is what was true when
    the answer left; the row is where a reader can find out how it ended."""
    try:
        policy_model = _memory_model(model)
    except Exception:
        _safely(lambda: _record_memory_outcome(turn_id, None),
                'the unavailable-policy note was lost')
        return {'status': 'unavailable', 'saved': False}

    def finish():
        try:
            decision = _finish_memory(question, answer, model, thread, dataset,
                                      turn_id, transcript, policy_model)
            _record_memory_outcome(turn_id, decision)
        except Exception as error:
            # The last boundary, and the reason it is not a bare `pass`: the
            # answer is already with the reader and this thread has no caller
            # left to raise to, but a decision nobody can read is a turn
            # silently unfiled forever. The row above is the durable place for
            # it; if even that write failed, the in-process log is what is
            # left, and `HOOK_LOG` at least reaches `__main__` and a developer
            # attached to this process.
            hooks._fired('memory', f'the deferred decision was lost: {error}')

    Thread(target=finish, name='widget-memory', daemon=True).start()
    return {'status': 'pending', 'saved': False}


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

    The order is the module's: the deterministic guard, then the agent, then
    `_preflight`, and only afterwards the memory policy — started on a thread
    of its own, so the answer is not held behind a judgement about whether to
    keep it. The `memory` field of a turn that ran therefore says `pending`,
    and the verdict lands in the long-term store and on the turn's row rather
    than in this return value; a streamed turn now reports that same status on
    its own memory event, because it defers the very same work. The one turn
    that ran and still says something else is a run the tool-hop guard
    stopped: it says `not_filed`, because there is no decision coming — see
    `_memory_after`.

    `stream` is the same turn, arriving as it is written. This is the whole
    answer at once, which is what a caller with nowhere to put a half-written
    one wants — the `__main__` harness, a test, a future non-browser client."""
    started = time.monotonic()
    choice, kind = _model_kind(model)
    refusal = hooks.relevance_guard(message)
    if refusal:
        return {'reply': refusal, 'input_tokens': None,
                'output_tokens': None, 'memory': _refused(refusal)}
    if kind == 'cli':
        # The two agent-level hooks bracket a CLI too, through the halves they
        # were factored into: a CLI has no loop for the middle four, and no
        # graph to hang middleware on at all. One process per call means no
        # memory either — the thread is accepted and ignored, the label
        # already says what a CLI cannot do.
        output = _cli_turn(choice, _validate(message))
        _log_turn(message=message, reply=output['reply'], thread=thread,
                  started=started, input_tokens=output['input_tokens'],
                  output_tokens=output['output_tokens'], status='answered')
        return output
    # The agent first, then the records: everything knowable without touching
    # a durable file — an unserved model, a missing key — must refuse before a
    # ledger query and a run-file read are spent on a turn that cannot run.
    # `stream` builds it in the same order, and a test pins that they agree.
    agent = _agent_for(choice)
    pre = _preflight(thread)
    payload, config = _run(message, thread, dataset=pre.dataset,
                           memory_state=pre.state, memory_text=pre.context)
    try:
        result = agent.invoke(payload, config=config)
    except Exception as error:
        # The graph may already have written a question and a tool exchange
        # into the thread, so the turn is recorded before the failure travels:
        # see `_log_interrupted_turn`. Then a UI helper's failure is a stated
        # 502, never a bare 500 — but the reason travels with it.
        _log_interrupted_turn(message=message, asked=payload['messages'][-1],
                              thread=thread, started=started,
                              dataset_id=pre.dataset, reason=str(error))
        if isinstance(error, WidgetUnavailable):
            raise
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error
    reply, used = _turn_account(result['messages'])
    # `close_the_log` already accounted for this run from inside the graph.
    output = _accounted(reply, used)
    stopped = hooks.hop_guard_refused(result['messages'])
    turn_id = _log_turn(message=message, reply=reply, thread=thread,
                        started=started, input_tokens=output['input_tokens'],
                        output_tokens=output['output_tokens'],
                        messages=result['messages'],
                        dataset_id=pre.dataset,
                        status='refused' if stopped else 'answered',
                        ai_message_id=str(getattr(result['messages'][-1], 'id', '') or ''))
    output['memory'] = _memory_after(message, reply, choice, thread, pre,
                                     turn_id, stopped)
    return output


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


def _tool_named(chunk) -> list:
    """The tool names this chunk states for the first time — usually none. A
    tool call announces its name exactly once, on the chunk that opens it; the
    argument tokens that follow carry `name=None` and name nothing. That one
    moment is the only part of a tool call the reader gets to see, as an
    ephemeral `{'status': <name>}` the page shows while it waits — the log
    never holds it, the same way `_delta` keeps the call itself off the
    answer channel."""
    if getattr(chunk, 'type', '') not in ('ai', 'AIMessageChunk'):
        return []
    return [piece['name']
            for piece in getattr(chunk, 'tool_call_chunks', None) or []
            if piece.get('name')]


def _stream_cli(cli: str, message: str, thread: str = '', started=None):
    """A CLI streaming: one piece, because one subprocess reports one complete
    reply and there is no partial output to forward. It still travels this path
    so the page keeps one way to ask rather than two, and the catalogue label
    says which models actually type their answer out."""
    done = _cli_turn(cli, message)
    yield {'delta': done['reply']}
    _log_turn(message=message, reply=done['reply'], thread=thread,
              started=started or time.monotonic(),
              input_tokens=done['input_tokens'],
              output_tokens=done['output_tokens'], status='answered')
    yield done


def _stream_agent(agent, message: str, thread: str, model: str,
                  pre: _Preflight, started=None):
    """The graph streaming: `stream_mode=['messages', 'values']` on the same
    run `ask` invokes — the same nodes, the same middleware, the same
    checkpoint write, so a streamed turn is in widget.db exactly as an asked
    one is.

    Two modes and not one, because the pieces and the account answer different
    questions. `messages` is what the reader watches arrive. `values` is the
    state the run ended in, and that is where the authoritative reply event
    comes from: the
    reply is read back out of the log with `_turn_account`, so what the page
    settles on is what the lab now holds for that turn rather than whatever the
    concatenated pieces happened to spell. A run that streamed pieces but ended
    with no state is a refusal, not a reply assembled here from the fragments —
    the same rule the rest of the lab keeps about a row that cannot say what
    produced it.

    `pre` is `ask`'s pre-flight, handed over rather than repeated: the two
    paths are one turn, and a streamed answer that stood on a different dataset
    from an asked one would be two turns wearing one name. It is required
    rather than defaulted: a default would be a turn quietly running with no
    dataset, no memory context and no policy stamp.

    The generator ends one event after the reply, and that event is a status
    rather than a verdict — see `_defer_memory`. Nothing this connection stays
    open for happens after the answer any more."""
    payload, config = _run(message, thread, dataset=pre.dataset,
                           memory_state=pre.state, memory_text=pre.context)
    final = None
    try:
        for mode, event in agent.stream(payload, config=config,
                                        stream_mode=['messages', 'values']):
            if mode == 'values':
                final = event
                continue
            for name in _tool_named(event[0]):
                yield {'status': name}
            text = _delta(event[0])
            if text:
                yield {'delta': text}
    except GeneratorExit:
        # The reader closed the tab, or the route stopped iterating. The run
        # dies here exactly as a raising one does and leaves the same dangling
        # question behind, so it earns the same row; nothing is yielded from a
        # generator being closed, and the close is passed straight on.
        _log_interrupted_turn(message=message, asked=payload['messages'][-1],
                              thread=thread,
                              started=started or time.monotonic(),
                              dataset_id=pre.dataset,
                              reason='the reader closed the stream')
        raise
    except Exception as error:
        _log_interrupted_turn(message=message, asked=payload['messages'][-1],
                              thread=thread,
                              started=started or time.monotonic(),
                              dataset_id=pre.dataset, reason=str(error))
        if isinstance(error, WidgetUnavailable):
            raise
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error
    if not (final or {}).get('messages'):
        stateless = ('the widget streamed no answer — the run ended with no '
                     'state to read the reply back from')
        _log_interrupted_turn(message=message, asked=payload['messages'][-1],
                              thread=thread,
                              started=started or time.monotonic(),
                              dataset_id=pre.dataset, reason=stateless)
        raise WidgetUnavailable(stateless)
    reply, used = _turn_account(final['messages'])
    output = _accounted(reply, used)
    stopped = hooks.hop_guard_refused(final['messages'])
    turn_id = _log_turn(message=message, reply=reply, thread=thread,
                        started=started or time.monotonic(),
                        input_tokens=output['input_tokens'],
                        output_tokens=output['output_tokens'],
                        messages=final['messages'], dataset_id=pre.dataset,
                        status='refused' if stopped else 'answered',
                        ai_message_id=str(getattr(final['messages'][-1], 'id', '') or ''))
    yield output
    # The reply is authoritative and must reach the caller before the memory
    # policy is even asked. Until 2026-08-28 the whole decision was made right
    # here, inline, so the connection stayed open through the policy call, the
    # summarizer call and two readings of the board — a cost paid after the
    # reader already had every word, and one that grows with `.runs/`. The work
    # is now handed to a thread of its own, exactly as `ask` hands it over, and
    # this last event says a decision is pending rather than reporting one:
    # that is what is true at the moment it is sent. `_memory_after` is the
    # shared seam so the two paths cannot come to disagree about which turns
    # are filed at all.
    yield {'memory': _memory_after(message, reply, model, thread, pre,
                                   turn_id, stopped)}


def stream(message: str, model: str = '', thread: str = ''):
    """The same turn as `ask`, handed over as it is written: an iterator of
    events, `{'delta': text}` for each piece and then an authoritative
    `{'reply', 'input_tokens', 'output_tokens'}` event, followed by a separate
    memory event carrying the status of the decision `ask` reports the same
    way: one taken after the answer, on a thread of its own. The reply event is
    the very dict `ask` returns without memory metadata; the deltas are how it
    arrived.

    A call, not a generator function, and that distinction is the point:
    everything knowable before the first piece — an unserved model, a missing
    key, a CLI this machine does not have — is raised from *here*, while the
    route can still answer it with a status code. A generator would defer all
    three past the response's headers and turn a refusal into a 200 whose body
    apologises."""
    started = time.monotonic()
    choice, kind = _model_kind(model)
    refusal = hooks.relevance_guard(message)
    if refusal:
        return iter([{'delta': refusal}, {
            'reply': refusal, 'input_tokens': None,
            'output_tokens': None, 'memory': _refused(refusal)}])
    if kind == 'cli':
        text = _validate(message)
        if not cli_available(choice):
            raise WidgetUnavailable(
                f'the {choice} command is not on this machine — install and '
                'log in, or pick an OpenRouter model')
        return _stream_cli(choice, text, thread, started)
    # The same order as `ask`, spelled out rather than left to how Python
    # happens to evaluate arguments: the agent, then the records.
    agent = _agent_for(choice)
    return _stream_agent(agent, message, thread, choice, _preflight(thread),
                         started)
