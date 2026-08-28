# this is an integration test
"""Reproductions for widget lifecycle, follow-up, and tool-loop bugs."""

import json

from langchain_core.messages import AIMessage, HumanMessage

from raglab.agents import widget


def test_follow_up_policy_receives_the_existing_thread_transcript(monkeypatch):
    """A short follow-up must be judged with the preceding exchange."""
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

    decision, _, _ = widget.backends._memory_turn('and?', 'model', 'general')

    assert decision['relevant'] is True
    assert 'Did reranking help?' in seen[0]
    assert 'Yes, recall improved.' in seen[0]


def test_repeated_tool_hops_are_stopped_before_graph_recursion_limit():
    """A tool loop must receive a bounded stop signal, not GRAPH_RECURSION_LIMIT.

    The old `MAX_TOOL_HOPS <= 8` assertion never checked that the recursion
    ceiling could actually be reached by that many hops — this does: the
    budget must admit every sequential hop the guard allows, at
    `SUPERSTEPS_PER_HOP` supersteps each, plus the closing answer and the
    fixed before/after-agent overhead, so the guard is what stops a
    pathological loop and never this ceiling.
    """
    assert widget.hooks.MAX_TOOL_HOPS <= 8
    assert callable(widget.hooks.stop_repeated_tool_hops)
    hops_and_answer = (widget.hooks.MAX_TOOL_HOPS
                       * widget.hooks.SUPERSTEPS_PER_HOP) + 1
    fixed_overhead = 2  # before_agent, after_agent
    assert widget.hooks.RECURSION_LIMIT >= hops_and_answer + fixed_overhead


def test_stream_request_is_kept_alive_when_the_page_navigates():
    """Changing surfaces must not cancel the server-side widget request."""
    from pathlib import Path

    source = Path(widget.__file__).parents[2].joinpath(
        'dashboard', 'frontend', 'widget.js').read_text(encoding='utf-8')
    stream_start = source.index('async function widgetStream')
    stream_end = source.index('// --- what you can ask')
    stream_source = source[stream_start:stream_end]
    assert 'keepalive: true' in stream_source

