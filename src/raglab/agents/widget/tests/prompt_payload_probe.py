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

This drives the real compiled agent — `hooks.MIDDLEWARE`, a real
`create_agent` graph, a real (in-memory) checkpointer, the same shape
`test_widget_regressions._drive_real_graph` already uses to prove
`RECURSION_LIMIT` against the graph instead of recomputing its arithmetic —
through a scripted thread of turns, some of which call a tool and some of
which do not. Between turns, `long_term_memory` grows the standing memory
line exactly the way an accepted `_finish_memory` outcome grows it: this
calls the real `save_memory_update`/`memory_context`, not a stand-in string,
because the whole point is a number a later task's real change would move.

What is captured is what `trim_and_call` actually hands the model on each
call — the count of system messages, the count of other messages, and the
total characters of their rendered content (no tokenizer dependency; see
`_content_chars`). A tool-calling turn costs two model calls, the hop and
the answer, so the unit measured here is the call, not the turn; `probe_thread`
returns one `TurnPayload` per call, tagged with which turn it belongs to, so a
caller can look at either grain.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

from raglab.agents.widget import conversation_memory as memory
from raglab.agents.widget import hooks
from raglab.agents.widget import long_term_memory
from raglab.agents.widget import skills_corpus_loader as skills

#: The probe's own thread and dataset — fixed rather than parameters most
#: callers would need to invent, since the only thing later tasks compare is
#: the shape of the numbers, not which dataset produced them.
DEFAULT_THREAD = 'prompt-payload-probe'
DEFAULT_DATASET = 'diary-en'


def _content_chars(message) -> int:
    """The characters this file counts as payload: rendered text content, the
    same rendering the reader and the turn log already agree on
    (`conversation_memory._text`). Tool-call argument JSON and message
    metadata are left out on purpose — content is the dominant cost by a wide
    margin (one skill body is ~23 KB; a tool call's arguments are a few dozen
    bytes) and a proxy whose own definition changed between two runs would be
    worse than an approximate one that stays put.
    """
    return len(memory._text(getattr(message, 'content', '')))


@tool
def read_skill_body(name: str = '') -> str:
    """Stand-in for `tools.read_rag_skill`, over the real skills corpus.

    This is what the probe's tool-calling turns call, and it is what backs
    this file's own docstring claim about a ~23 KB tool reply: the number
    comes from a real `SKILL.md` body, not an invented filler string. It does
    not import `tools.TOOLS` — that list carries `measure_bilingual_alignment`,
    which needs a real encoder and the `local-embeddings` extra, and a
    characterisation test has no business requiring either.
    """
    catalogue = skills.index()
    picked = name if name in catalogue else sorted(catalogue)[0]
    return skills.body(picked)


class _RecordingModel(BaseChatModel):
    """A scripted chat model that records the exact messages it is handed on
    every call — the same list `trim_and_call`'s `handler(request)` passes
    down, after the trim and the standing-system-line bookkeeping have already
    run — before playing back one line of a fixed script.

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

    `tool_turns` names which turns (0-based) call a tool before answering:
    each of those costs two model calls — the hop, then the answer — the same
    shape a real tool-calling turn has. Every other turn answers in one call.

    Memory grows the way `long_term_memory` really grows it: after each turn
    this calls the real `save_memory_update` under `dataset_id`, mirroring an
    accepted `_finish_memory` outcome, so the standing memory line a later
    turn sees is the widget's own aggregate — the same `_aggregate`/`_bounded`
    behaviour production uses — not a hand-written stand-in.

    The standing-system-line bookkeeping — a line already sent to the model is
    not sent again — is `backends._run`'s own rule, reproduced here rather
    than re-derived: it is exactly what decides how many system messages a
    long thread accumulates, which is the number this whole file exists to
    pin.

    Starts from a clean long-term-memory table (`clear_long_term_memory`) and
    a fresh in-memory checkpointer every call, so two calls in the same test
    session cannot see each other's growth.
    """
    long_term_memory.clear_long_term_memory()
    script = []
    for i in range(turns):
        if i in tool_turns:
            script.append(AIMessage(content='', tool_calls=[
                {'name': 'read_skill_body', 'args': {}, 'id': f'call-{i}'}]))
        script.append(AIMessage(content=f'Answer number {i}.'))

    model = _RecordingModel(script=script)
    agent = create_agent(model, tools=[read_skill_body],
                         middleware=hooks.MIDDLEWARE,
                         state_schema=memory.WidgetState,
                         checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': thread},
             'recursion_limit': hooks.RECURSION_LIMIT}

    said: set[str] = set()
    payloads: list[TurnPayload] = []
    opening = f'This thread is about experiment {thread!r} on dataset {dataset_id!r}.'
    for i in range(turns):
        context = long_term_memory.memory_context(dataset_id)
        lines = []
        for text in (opening, context):
            if text and text not in said:
                lines.append(SystemMessage(content=text))
                said.add(text)
        state = {'messages': lines + [HumanMessage(content=f'Question {i}?')],
                 'experiment_id': thread, 'dataset_id': dataset_id}
        if i == 0:
            state['started_at'] = '2026-01-01T00:00:00+00:00'
        before = len(model.raw)
        agent.invoke(state, config=config)
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
