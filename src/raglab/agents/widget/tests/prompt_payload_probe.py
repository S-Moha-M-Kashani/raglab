"""Offline probe: what one model call actually carries, and how it grows.

Why this exists
----------------
The turn-latency branch this one is cut from could only argue its
improvements from two live traces read by eye: one 29-step thread that
carried twelve system messages, a single `read_rag_skill` call that put
roughly 23 KB into the window. Nobody had measured what a turn actually
sends the same way twice, so those numbers were anecdotes with a count
attached. Every later task on this branch claims to shrink the prompt;
without a number taken the same way before and after, that claim is
decoration.

Round 1 of this file reimplemented production's turn construction instead of
driving it — an invented opening line, a hand-copied dedup rule, and no
`system_prompt` at all — and drifted at exactly the two points it did not
copy, undercounting the real payload by about 13%. This version drives
production's own seams instead of restating them:

- **`backends._run`** builds the turn's `(payload, config)` — the opening
  line (`ACTIVE_EXPERIMENT_PROMPT`), the standing-line rule that keeps one
  identity line and one memory line per thread, `thread_stamp`,
  `RECURSION_LIMIT` — so when a later task changes any of those, this probe
  changes with it instead of silently misreporting reality. It already has:
  the standing-line rule replaced an append-if-not-said one on 2026-08-29,
  and this file needed no edit for the measured system count to drop from a
  climbing 13 to a flat 3.
- **`create_agent(..., system_prompt=SYSTEM_PROMPT, tools=TOOLS, ...)`** is
  built the way `backends._build_agent` builds it, substituting only the
  model (a scripted stand-in, never a real network call) for `_build_agent`'s
  `ChatOpenAI`. Everything else — `SYSTEM_PROMPT`, the real `TOOLS` (so a
  scripted tool call runs the real `read_rag_skill` over the real skills
  corpus), `hooks.MIDDLEWARE`, `memory.WidgetState` — is production's own.
  `system_prompt` matters here specifically: langchain's `_execute_model_sync`
  prepends it to the message list *after* `trim_and_call` (a `wrap_model_call`
  wrapper) has already run, so a probe that built its agent without it would
  never see that message at all — the bug round 1 shipped.
- **`memory.saver()`**, the same checkpointer `_build_agent` uses, rather than
  a substitute. This one part of `_build_agent` cannot be swapped for an
  in-memory stand-in: `backends._run`'s dedup reads "what this thread has
  already been told" via `memory._channels`, which is hardwired to this one
  module-level saver — not a parameter `_run` accepts — so an agent driven
  under a *different* checkpointer would have its transcript and `_run`'s
  idea of that transcript permanently disagree. Tests already redirect
  `RAGLAB_WIDGET_DB` to a per-test temp file (`conftest.py`), so this stays
  fully offline; `memory.forget(thread)` below keeps two probe runs, or a
  probe run and a real test, from sharing a thread name by accident.

What is captured is what the model actually receives on every call —
`trim_and_call`'s trimmed window plus the framework-prepended system prompt —
as the count of system messages, the count of other messages, and the total
characters of their rendered content (no tokenizer dependency; see
`_content_chars`). A tool-calling turn costs two model calls, the hop and the
answer, so the unit measured here is the call, not the turn; `probe_thread`
returns one `TurnPayload` per call, tagged with which turn it belongs to, so a
caller can look at either grain.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from raglab.agents.widget import backends
from raglab.agents.widget import conversation_memory as memory
from raglab.agents.widget import hooks
from raglab.agents.widget import long_term_memory
from raglab.agents.widget import skills_corpus_loader as skills
from raglab.agents.widget.prompts import SYSTEM_PROMPT
from raglab.agents.widget.tools import MAX_SKILL_READS, TOOLS

#: The probe's own thread and dataset — fixed rather than parameters most
#: callers would need to invent, since the only thing later tasks compare is
#: the shape of the numbers, not which dataset produced them.
DEFAULT_THREAD = 'prompt-payload-probe'
DEFAULT_DATASET = 'diary-en'

#: `MAX_SKILL_READS` real names, so a tool-calling turn exercises the real
#: worst case of one `read_rag_skill` call — every body it can carry at once,
#: with the real `=== name ===` headers — rather than a single body, which is
#: at most a third of what one call can actually cost.
_SKILL_NAMES = ','.join(sorted(skills.index())[:MAX_SKILL_READS])


def _content_chars(message) -> int:
    """The characters this file counts as payload: rendered text content, the
    same rendering the reader and the turn log already agree on
    (`conversation_memory._text`). Tool-call argument JSON and message
    metadata are left out on purpose — content is the dominant cost by a wide
    margin (three skill bodies are tens of KB; a tool call's arguments are a
    few dozen bytes) and a proxy whose own definition changed between two
    runs would be worse than an approximate one that stays put.
    """
    return len(memory._text(getattr(message, 'content', '')))


class _RecordingModel(BaseChatModel):
    """A scripted chat model that records the exact messages it is handed on
    every call — system prompt included, since `create_agent` prepends it
    inside `_execute_model_sync`, after `trim_and_call` has already run —
    before playing back one line of a fixed script.

    Modeled on `test_widget_regressions._scripted_tool_model`: just enough of
    `BaseChatModel` for `create_agent` to bind tools to it and drive the real
    graph, offline, with no network call anywhere.
    """

    script: list
    calls: int = 0
    _raw: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return 'recording-scripted-model'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = sum(1 for m in messages if getattr(m, 'type', '') == 'system')
        chars = sum(_content_chars(m) for m in messages)
        self._raw.append((system, len(messages) - system, chars))
        reply = self.script[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @property
    def raw(self) -> list:
        """`(system_messages, other_messages, total_chars)` per call so far."""
        return self._raw


def build_agent(model):
    """The production agent with one part swapped.

    Built the way `backends._build_agent` builds it — `SYSTEM_PROMPT`, the
    real `TOOLS`, `hooks.MIDDLEWARE`, `memory.WidgetState`, `memory.saver()` —
    substituting only the model, which is the one part of `_build_agent` that
    cannot run offline (`ChatOpenAI` needs a real key and a real network
    call). A function rather than a line inside `probe_thread`, because a test
    that drives one scripted turn through the real graph needs the same agent
    and a second copy of this call is a second thing to keep in step with
    `_build_agent`.
    """
    return create_agent(model, tools=TOOLS, system_prompt=SYSTEM_PROMPT,
                        middleware=hooks.MIDDLEWARE,
                        state_schema=memory.WidgetState,
                        checkpointer=memory.saver())


@dataclass(frozen=True)
class TurnPayload:
    """What one model call carried. One entry per call `trim_and_call`
    wrapped — a tool-calling turn contributes more than one row, `call_in_turn`
    says which."""

    turn: int
    call_in_turn: int
    system_messages: int
    other_messages: int
    total_chars: int

    def __str__(self) -> str:
        return (f'turn {self.turn:>2} call {self.call_in_turn}: '
                f'{self.system_messages:>2} system, '
                f'{self.other_messages:>3} other, '
                f'{self.total_chars:>6} chars')


def probe_thread(*, turns: int, tool_turns: frozenset[int] = frozenset(),
                 thread: str = DEFAULT_THREAD,
                 dataset_id: str = DEFAULT_DATASET) -> list[TurnPayload]:
    """Drive `turns` turns of one thread through the real compiled agent and
    return one `TurnPayload` per model call, in order.

    `tool_turns` names which turns (0-based) call `read_rag_skill` — the real
    tool, over the real skills corpus — for `MAX_SKILL_READS` bodies before
    answering: each of those costs two model calls, the hop and the answer,
    the same shape a real tool-calling turn has. Every other turn answers in
    one call.

    Memory grows the way `long_term_memory` really grows it: after each turn
    this calls the real `save_memory_update` under `dataset_id`, mirroring an
    accepted `_finish_memory` outcome, so the standing memory line a later
    turn sees is the widget's own aggregate — the same `_aggregate`/`_bounded`
    behaviour production uses — not a hand-written stand-in. Each turn's
    `(payload, config)` is `backends._run`'s own output, so the opening line,
    the marked standing lines and `thread_stamp` are whatever production says
    they are today — and the numbers include `hooks.trim_and_call`'s filter,
    since the calls are recorded from inside it.

    Starts from a clean long-term-memory table and a freshly forgotten thread,
    so two calls in the same test session — or a probe run and a real test
    sharing `memory.saver()` — cannot see each other's growth.
    """
    long_term_memory.clear_long_term_memory()
    memory.forget(thread)
    script = []
    for i in range(turns):
        if i in tool_turns:
            script.append(AIMessage(content='', tool_calls=[
                {'name': 'read_rag_skill', 'args': {'names': _SKILL_NAMES},
                 'id': f'call-{i}'}]))
        script.append(AIMessage(content=f'Answer number {i}.'))

    model = _RecordingModel(script=script)
    agent = build_agent(model)

    payloads: list[TurnPayload] = []
    for i in range(turns):
        context = long_term_memory.memory_context(dataset_id)
        payload, config = backends._run(
            f'Question {i}?', thread, dataset=dataset_id,
            memory_state=memory.dataset_stamp(dataset_id, thread),
            memory_text=context)
        before = len(model.raw)
        agent.invoke(payload, config=config)
        for call_in_turn, (system, other, chars) in enumerate(model.raw[before:]):
            payloads.append(TurnPayload(turn=i, call_in_turn=call_in_turn,
                                        system_messages=system,
                                        other_messages=other,
                                        total_chars=chars))
        # Grow memory as an accepted turn would, so the *next* turn's standing
        # line differs from this one's — real growth, not a static stub.
        long_term_memory.save_memory_update(
            dataset_id=dataset_id, experiment_id=thread,
            subtopic=f'topic-{i}', question=f'Question {i}?',
            answer=f'Answer number {i}.',
            dataset_summary=f'Turn {i} found a stable result on {dataset_id}.')
    return payloads


def render(payloads: list[TurnPayload]) -> str:
    """One line per model call, and a total — the form a developer reads at a
    glance (`pytest -s` on the characterisation test prints this)."""
    lines = [str(p) for p in payloads]
    lines.append(f'-- {len(payloads)} calls, '
                 f'{sum(p.total_chars for p in payloads)} chars total')
    return '\n'.join(lines)
