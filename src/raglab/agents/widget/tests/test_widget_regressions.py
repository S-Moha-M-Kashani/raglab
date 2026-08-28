# this is an integration test
"""Reproductions for widget lifecycle, follow-up, and tool-loop bugs."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from raglab.agents import widget


def test_follow_up_policy_receives_the_existing_thread_transcript(monkeypatch):
    """A short follow-up is judged with the exchange before it — and only that.

    Recorded twice. First: `and?` alone is ambiguous, so the policy is handed
    the thread's earlier turns. Then, when the policy moved to after the answer
    (2026-08-28), the reading it was handed came from a checkpoint that already
    held this very turn — the question appeared a second time as the last line
    of its own "prior conversation", the answer being judged was part of the
    context judging it, and an older turn fell out of the recall window to make
    room for the pair that had just arrived.

    The whole thread is real here — a real compiled graph over a scripted chat
    model, the real checkpointer, and `history` untouched — because the bug was
    in *when* `history` is read, and a test that monkeypatches it away cannot
    see that at all.
    """
    from langchain_core.language_models import GenericFakeChatModel
    from langchain.agents import create_agent

    from raglab.agents.widget import conversation_memory as memory

    judged = []

    class Structured:
        def invoke(self, messages):
            judged.append(messages[-1][1])
            return memory.MemoryPolicy(relevant=True, should_save=False)

    class PolicyModel:
        def with_structured_output(self, _schema):
            return Structured()

    agent = create_agent(
        GenericFakeChatModel(messages=iter([
            AIMessage(content='Yes, recall improved.'),
            AIMessage(content='By four points.')])),
        system_prompt='x', middleware=widget.hooks.MIDDLEWARE,
        state_schema=memory.WidgetState, checkpointer=memory.saver())
    monkeypatch.setattr(widget.backends, '_memory_model', lambda _model: PolicyModel())
    monkeypatch.setattr(widget.backends, '_agent_for', lambda _model: agent)
    # The deferred decision, run on this thread instead of its own: the order
    # is unchanged — it still happens after the answer and after the graph has
    # written the turn — and the assertion is not racing a daemon.
    monkeypatch.setattr(widget.backends, '_defer_memory',
                        lambda *args: widget.backends._finish_memory(*args) or {})

    memory.forget('follow-up-thread')
    widget.ask('Did reranking help?', model='openai/gpt-5-nano',
               thread='follow-up-thread')
    widget.ask('and?', model='openai/gpt-5-nano', thread='follow-up-thread')

    first, second = judged
    assert 'Prior conversation:\n(none)' in first
    assert 'Did reranking help?' in second
    assert 'Yes, recall improved.' in second
    # The turn being judged is not its own context: its answer is not there,
    # and its question appears once — as the question, not as a transcript line.
    assert 'By four points.' not in second
    assert second.count('and?') == 1


def _scripted_tool_model(script):
    """A chat model that plays back a fixed script of replies, one per call —
    just enough of `BaseChatModel` for `create_agent` to bind tools to it and
    drive a real graph through a chosen number of tool-calling hops. No
    network, no real model: this is what makes it safe to run the real
    compiled graph offline instead of recomputing `RECURSION_LIMIT`'s own
    arithmetic and calling that a proof."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    class _Model(BaseChatModel):
        script: list
        calls: int = 0

        @property
        def _llm_type(self):
            return 'scripted-tool-model'

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            message = self.script[self.calls]
            self.calls += 1
            return ChatResult(generations=[ChatGeneration(message=message)])

    return _Model(script=list(script))


def _drive_real_graph(hops: int, recursion_limit: int):
    """Run the real compiled agent — `hooks.MIDDLEWARE`, a real `create_agent`
    graph, a real (in-memory) checkpointer — through exactly `hops`
    tool-calling turns and then a final answer, at a chosen recursion limit.
    Raises `GraphRecursionError` exactly when the ceiling is too small; this
    is the proof `test_recursion_limit_admits_max_tool_hops_and_the_closing_answer`
    needs and a recomputed formula cannot give.
    """
    import uuid

    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    @tool
    def noop(x: str = '') -> str:
        """A tool that does nothing; it only exists to cost a hop."""
        return 'ok'

    script = [AIMessage(content='', tool_calls=[
                  {'name': 'noop', 'args': {}, 'id': f'call-{i}'}])
              for i in range(hops)]
    script.append(AIMessage(content='final answer'))
    agent = create_agent(_scripted_tool_model(script), tools=[noop],
                         middleware=widget.hooks.MIDDLEWARE,
                         checkpointer=InMemorySaver())
    return agent.invoke(
        {'messages': [HumanMessage(content='drive the hops')]},
        config={'recursion_limit': recursion_limit,
                'configurable': {'thread_id': str(uuid.uuid4())}})


def test_recursion_limit_admits_max_tool_hops_and_the_closing_answer():
    """Proof against the real compiled graph, not a recomputed formula: a run
    that calls tools `MAX_TOOL_HOPS` times must reach its closing model call
    (a genuine answer, or the guard's own refusal once the count trips) and
    `close_the_log` without `GraphRecursionError` — the ceiling must never be
    what stops a pathological loop; `stop_repeated_tool_hops` must be.

    This replaced an assertion that only recomputed `RECURSION_LIMIT`'s own
    formula and so could never have caught the off-by-one a prior version of
    that formula had: it was one graph-start superstep short, and this test
    is what a reviewer used to prove it empirically.
    """
    from langgraph.errors import GraphRecursionError

    assert widget.hooks.MAX_TOOL_HOPS <= 8
    assert callable(widget.hooks.stop_repeated_tool_hops)

    hops = widget.hooks.MAX_TOOL_HOPS
    result = _drive_real_graph(hops, widget.hooks.RECURSION_LIMIT)
    assert result['messages'][-1].content

    with pytest.raises(GraphRecursionError):
        _drive_real_graph(hops, widget.hooks.RECURSION_LIMIT - 1)


def test_stream_request_is_kept_alive_when_the_page_navigates():
    """Changing surfaces must not cancel the server-side widget request."""
    from pathlib import Path

    source = Path(widget.__file__).parents[2].joinpath(
        'dashboard', 'frontend', 'widget.js').read_text(encoding='utf-8')
    stream_start = source.index('async function widgetStream')
    stream_end = source.index('// --- what you can ask')
    stream_source = source[stream_start:stream_end]
    assert 'keepalive: true' in stream_source

