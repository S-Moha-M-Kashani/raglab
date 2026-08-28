# this is an integration test
"""Reproductions for widget lifecycle, follow-up, and tool-loop bugs."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from raglab.agents import widget


def test_follow_up_policy_receives_the_existing_thread_transcript(monkeypatch):
    """A short follow-up must be judged with the preceding exchange.

    The policy is asked after the answer now, so this drives `_finish_memory`
    directly; what it is judging — a question that means nothing on its own —
    is unchanged, and so is the transcript it must be given."""
    seen = []

    class Structured:
        def invoke(self, messages):
            seen.append(messages[-1][1])
            text = messages[-1][1]
            if 'Did reranking help?' in text and 'Yes, recall improved.' in text:
                return widget.conversation_memory.MemoryPolicy(
                    relevant=True, should_save=False, dataset_id='')
            return widget.conversation_memory.MemoryPolicy(
                relevant=False, should_save=False,
                reason='A follow-up without its conversation is ambiguous.')

    class PolicyModel:
        def with_structured_output(self, _schema):
            return Structured()

    monkeypatch.setattr(widget.backends, '_memory_model', lambda _model: PolicyModel())
    monkeypatch.setattr(widget.memory if hasattr(widget, 'memory') else
                        widget.conversation_memory, 'history',
                        lambda _thread: {'turns': [
                            {'role': 'you', 'text': 'Did reranking help?'},
                            {'role': 'bot', 'text': 'Yes, recall improved.'}]})

    decision = widget.backends._finish_memory(
        'and?', 'Recall improved with reranking.', 'model', 'general', '')

    assert decision['relevant'] is True
    assert 'Did reranking help?' in seen[0]
    assert 'Yes, recall improved.' in seen[0]


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

