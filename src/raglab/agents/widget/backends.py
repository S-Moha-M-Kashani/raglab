"""The widget's model catalogue and the two answer paths.

The OpenRouter path is `langchain.agents.create_agent` with the four
middleware hooks (hooks.py), one cached agent per model; the CLI path is a
process per call with the knowledge base inlined, because `CliChat` has no
`bind_tools`. Both paths enter through `ask` for a whole answer at once, or
through `stream` for the same answer handed over as it is written.

One turn, one order, both paths: what the answer needs runs before the agent,
what the *filing* needs runs after it. Before the agent (`_preflight`): the
relevance guard, the thread's trusted dataset — resolved *once*, because each
resolution is a ledger query plus a run-file read — and the long-term memory
filed under that dataset. After the answer (`_finish_memory`): the memory
policy and every provenance gate, then the summary and the write.

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
    messages = []
    # An experiment's thread opens by saying which experiment it is — from the
    # validated record, so an unknown or general thread says nothing. Without
    # it "the last experiment" could only mean the newest on the board, and a
    # meetings-de thread was once told about a smoke-import-check run.
    # ...and said once. The graph appends every input message to the thread,
    # so a line added on each turn was reread on each turn — a five-turn
    # thread carried ten system messages. A line the thread already holds,
    # word for word, is not added again; a memory context that changed is.
    said = {str(m.content) for m in (memory._channels(name).get('messages')
                                     or []) if getattr(m, 'type', '') == 'system'}
    opening = (ACTIVE_EXPERIMENT_PROMPT.format(experiment_id=name,
                                               dataset=dataset)
               if dataset else '')
    for line in (opening, memory_text):
        if line and line not in said:
            messages.append(SystemMessage(content=line))
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
    """Render prior visible turns for short follow-ups such as ``and?``."""
    if not (thread or '').strip():
        return ''
    try:
        turns = memory.history(thread).get('turns') or []
    except Exception:
        return ''
    return '\n'.join(f"{turn.get('role', 'message')}: {turn.get('text', '')}"
                     for turn in turns[-memory.MAX_RECALLED:]
                     if turn.get('text'))


def _log_turn(*, message: str, reply: str, thread: str, started: float,
              input_tokens=None, output_tokens=None, messages=None,
              dataset_id='', status='answered', ai_message_id='') -> str:
    """Write the human-readable operational row after the answer exists.

    `dataset_id` is the turn's trusted dataset, not the policy's opinion of
    one: the row says which corpus the thread stood on while it answered, and
    that is known before the answer exists rather than after."""
    name = (thread or '').strip() or memory.GENERAL
    experiment_id = name if name != memory.GENERAL else ''
    user_id = ''
    if messages:
        for item in messages:
            if isinstance(item, HumanMessage):
                user_id = str(getattr(item, 'id', '') or '')
    trace = _turn_steps(messages or [])
    if not trace:
        trace = [{'id': str(uuid.uuid4()), 'kind': 'human',
                  'text': message, 'latency_ms': 0},
                 {'id': str(uuid.uuid4()), 'kind': 'ai', 'text': reply,
                  'latency_ms': None}]
    return turn_logger.log_turn(
        thread_id=name, experiment_id=experiment_id, dataset_id=dataset_id,
        user_message_id=user_id or str(uuid.uuid4()), user_message=message,
        ai_message_id=ai_message_id, ai_message=reply,
        steps=trace,
        total_input_tokens=input_tokens, total_output_tokens=output_tokens,
        total_latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        status=status)


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
    turn stamps on the thread's own policy channels. Nothing here refuses: the
    one refusal a turn can raise without a model — `hooks.relevance_guard` — is
    the caller's, and every other refusal this widget has is about the write.
    """
    dataset: str
    context: str
    state: dict


def _refused(reason: str) -> dict:
    """A decision that is its own answer: nothing ran, so nothing was filed."""
    return {'relevant': False, 'should_save': False, 'dataset_id': '',
            'subtopic': '', 'reason': reason, 'saved': False, 'blocked': True}


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
    # `WidgetState`'s policy channels record what this turn stood on while it
    # ran: the trusted dataset and the safe defaults. The verdict that arrives
    # after the answer travels on the returned event and in `turn_logger`, not
    # back into these channels — a second writer amending them afterwards would
    # be racing the graph over the same checkpoint.
    state = memory.policy_state(memory.MemoryPolicy(dataset_id=dataset),
                                experiment)
    return _Preflight(dataset, context, state)


def _summarize_memory_update(**kwargs):
    return summarize_memory_update(**kwargs)


def _finish_memory(question: str, answer: str, model: str, thread: str,
                   dataset: str, turn_id: str = '') -> dict | None:
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

    `None` means there was no policy to ask at all — no client could be built —
    so the turn says nothing about memory rather than inventing a verdict.
    """
    try:
        policy_model = _memory_model(model)
    except Exception:
        return None
    policy = evaluate_memory_policy(
        question, policy_model, experiment_id=thread, dataset_id=dataset,
        trusted_dataset_id=dataset, conversation=_policy_transcript(thread))
    decision = {**policy.model_dump(), 'saved': False}
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
    foreign = experiment_tools.foreign_experiments(answer, dataset)
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
            'validated_dataset_ids': experiment_tools.validated_dataset_ids(thread)}
        writer = tools.save_widget_memory
        stored = (writer.invoke(arguments) if hasattr(writer, 'invoke')
                  else writer(**arguments))
        decision.update({'saved': bool(stored.get('saved')), 'save': stored})
        if turn_id and stored.get('update_id') is not None:
            turn_logger.attach_memory_update(turn_id, stored['update_id'])
    except Exception as error:
        decision.update({'saved': False, 'save_error': str(error)})
    return decision


def _defer_memory(question: str, answer: str, model: str, thread: str,
                  dataset: str, turn_id: str = '') -> dict:
    """Start the memory decision after the caller has received the answer.

    What comes back is a status and not a verdict, because the verdict is not
    made yet: the policy runs on the thread this starts. `saved: False` is the
    honest reading of that moment — nothing has been filed — and the absence of
    `relevant`/`should_save` is what keeps the panel from rendering a guess
    (`panel_server._safe_widget_event` omits a memory value whose three
    booleans are not all there). A streamed turn does not need this: it has a
    later event to carry the resolved decision."""
    def finish():
        try:
            _finish_memory(question, answer, model, thread, dataset, turn_id)
        except Exception:  # defensive boundary: a daemon thread has nobody to
            pass           # tell, and the answer is already with the reader.

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

    The order is the module's: the deterministic guard, then `_preflight`, then
    the agent, and only afterwards the memory policy — started on a thread of
    its own, so the answer is not held behind a judgement about whether to keep
    it. The `memory` field of a turn that ran therefore says `pending`, and the
    verdict lands in the long-term store rather than in this return value; a
    streamed turn is the one that reports it back, on its own last event.

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
    pre = _preflight(thread)
    agent = _agent_for(choice)
    payload, config = _run(message, thread, dataset=pre.dataset,
                           memory_state=pre.state, memory_text=pre.context)
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
    output = _accounted(reply, used)
    turn_id = _log_turn(message=message, reply=reply, thread=thread,
                        started=started, input_tokens=output['input_tokens'],
                        output_tokens=output['output_tokens'],
                        messages=result['messages'],
                        dataset_id=pre.dataset,
                        ai_message_id=str(getattr(result['messages'][-1], 'id', '') or ''))
    output['memory'] = (_defer_memory(message, reply, choice, thread,
                                      pre.dataset, turn_id)
                        or {'status': 'pending', 'saved': False})
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


def _stream_agent(agent, message: str, thread: str, model: str = '',
                  pre: _Preflight | None = None, started=None):
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
    from an asked one would be two turns wearing one name."""
    pre = pre or _Preflight('', '', {})
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
    except WidgetUnavailable:
        raise
    except Exception as error:
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error
    if not (final or {}).get('messages'):
        raise WidgetUnavailable(
            'the widget streamed no answer — the run ended with no state to '
            'read the reply back from')
    reply, used = _turn_account(final['messages'])
    output = _accounted(reply, used)
    turn_id = _log_turn(message=message, reply=reply, thread=thread,
                        started=started or time.monotonic(),
                        input_tokens=output['input_tokens'],
                        output_tokens=output['output_tokens'],
                        messages=final['messages'], dataset_id=pre.dataset,
                        ai_message_id=str(getattr(final['messages'][-1], 'id', '') or ''))
    yield output
    # The reply is authoritative and must reach the caller before the memory
    # policy is even asked. Memory is a separate event so a streamed reader can
    # render the answer without waiting for a judgement about keeping it — and
    # because this path has a later event to spend, it reports the resolved
    # decision rather than `ask`'s `pending`. No decision at all (no policy
    # client to ask) is no event: the turn says nothing rather than a guess.
    finished = _finish_memory(message, reply, model, thread, pre.dataset,
                              turn_id)
    if finished is not None:
        yield {'memory': finished}


def stream(message: str, model: str = '', thread: str = ''):
    """The same turn as `ask`, handed over as it is written: an iterator of
    events, `{'delta': text}` for each piece and then an authoritative
    `{'reply', 'input_tokens', 'output_tokens'}` event, followed — when there
    was a memory policy to ask at all — by a separate memory event carrying the
    decision it came to. The reply event is the very dict `ask` returns without
    memory metadata; the deltas are how it arrived.

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
    return _stream_agent(_agent_for(choice), message, thread, choice,
                         _preflight(thread), started)
