# this is an integration test
"""Characterisation: what one model call carries today, and how it grows.

This pins the numbers `prompt_payload_probe.probe_thread` measures against a
fixed, twelve-turn scripted thread — three of those turns call `read_rag_skill`
(the real tool, over the real skills corpus) for `MAX_SKILL_READS` bodies
before answering. The scenario reproduced, offline and deterministically, the
two live findings that motivated this branch: a long thread's system messages
kept accumulating, and a tool call can put tens of kilobytes into a single
window.

The first of those is fixed as of 2026-08-29. `backends._run` writes the
memory context under an id that is the same on every turn of a thread, so
`add_messages` replaces the line instead of adding a longer version beside it;
the counts below say 3 system messages on every call after the first, where
this file's previous round measured 2 climbing to 13. What the model loses is
stale duplicates of its own memory; what it keeps is the newest, which is
strictly better context as well as a smaller prompt.

Round 1 of this test pinned numbers from a probe that reimplemented
production's turn construction instead of driving it — no `system_prompt`,
an invented opening line, a single skill body instead of the real worst case.
The corrected probe drives `backends._run` and builds the agent the way
`backends._build_agent` does, so the numbers below are what the real seams
say today, not what a hand-rolled stand-in guessed.

Four later tasks on this branch each change what a turn sends: global memory
filtering and one standing memory line (both landed), then a size-bounded
window with stubbed closed-turn tool replies, and interrupted-turn handling.
Every number pinned below is a claim about *today's* behaviour, so a later
task's deliberate change shows up here as an assertion a reviewer can see
move — and why it moved — rather than as silence.
"""
from raglab.agents.widget import hooks
from raglab.agents.widget.prompts import SYSTEM_PROMPT
from raglab.agents.widget.tests.prompt_payload_probe import (
    _RecordingModel, build_agent, probe_thread, render)


def test_prompt_payload_grows_across_a_scripted_thread(capsys):
    """Twelve turns, three of which call a tool, over a growing memory
    context. The measured numbers, and why each one is what it is:

    - 15 calls, not 12: turns 2, 5 and 9 each cost the tool hop plus the
      answer, so `probe_thread` returns one `TurnPayload` per model call.
    - The very first call already carries two system messages: the real
      `SYSTEM_PROMPT` (prepended by `create_agent` itself, after
      `trim_and_call` runs) and the thread's opening line
      (`ACTIVE_EXPERIMENT_PROMPT`, which experiment and dataset this is),
      stamped before any memory exists to summarise.
    - System messages reach 3 on the second call and never move again:
      `SYSTEM_PROMPT`, the thread's opening line, and exactly one memory
      line. `long_term_memory` still produces a longer aggregate on every
      accepted turn, but `backends._run` now writes it under the thread's
      standing id, so each turn's line overwrites the last rather than
      joining it. Before that change this row read 13 and was still
      climbing — the mechanism behind the live finding of twelve system
      messages on a 29-step thread — and nothing capped it, since
      `trim_and_call` exempts system lines from the window by design.
    - Other-message count grows with the transcript and then flattens at 20
      from turn 8 onward: `hooks.MAX_HISTORY` trims the non-system window to
      its last 20 messages, so the raw transcript keeps growing but what
      reaches the model does not.
    - The three tool-hop calls (turns 2, 5, 9) each jump by roughly 20 KB —
      one real `read_rag_skill` call returning its full `MAX_SKILL_READS`
      bodies (chunking-strategies, contextual-retrieval,
      hierarchical-graph-rag: ~20,000 chars of real skill text plus their
      `=== name ===` headers) in one call. That is this branch's other live
      finding, at the real worst case rather than a third of it: one tool
      reply can cost tens of kilobytes of one call's window.
    """
    assert hooks.MAX_HISTORY == 20  # the ceiling this test's plateau assumes
    assert len(SYSTEM_PROMPT) > 1_000  # the message this test's floor assumes

    payloads = probe_thread(turns=12, tool_turns=frozenset({2, 5, 9}))
    print('\n' + render(payloads))
    out = capsys.readouterr().out
    assert 'chars total' in out  # the helper is readable under `pytest -s`

    assert len(payloads) == 15
    assert [p.turn for p in payloads if p.call_in_turn > 0] == [2, 5, 9]

    first, last = payloads[0], payloads[-1]
    assert (first.system_messages, first.other_messages, first.total_chars) \
        == (2, 1, 1722)
    assert (last.system_messages, last.other_messages, last.total_chars) \
        == (3, 20, 42599)

    # The standing set is bounded now: SYSTEM_PROMPT and the opening line from
    # the first call, plus one memory line from the first turn that has memory
    # to state — and never a fourth, however long the thread or how much the
    # memory aggregate grows.
    system_counts = [p.system_messages for p in payloads]
    assert system_counts == [2] + [3] * 14
    assert max(system_counts) == 3

    # The window's non-system side is trimmed to MAX_HISTORY once the
    # transcript passes it — it must never exceed the ceiling.
    assert max(p.other_messages for p in payloads) == hooks.MAX_HISTORY
    assert [p.other_messages for p in payloads[-4:]] == [20, 20, 20, 20]

    # Each tool hop costs roughly the real worst case of one `read_rag_skill`
    # call — three full skill bodies, not one — over the very next call in
    # the same turn.
    hop_turn_2, answer_turn_2 = payloads[2], payloads[3]
    assert answer_turn_2.total_chars - hop_turn_2.total_chars > 15_000

    # Today's whole-thread total: the baseline every later task's version of
    # this same scripted thread is judged against. It was 466,458 while every
    # stale memory line was still being reread; one standing line saves 12,982
    # characters over these fifteen calls. The saving is small next to the
    # three tool replies that dominate the total — that is the next task's
    # finding, not this one's — and the point of this change is as much the
    # curve as the count: the system side no longer grows without a ceiling.
    assert sum(p.total_chars for p in payloads) == 453_476


def test_a_legacy_threads_stale_memory_lines_are_collapsed_by_the_real_graph():
    """The reducer's half of the same change, driven through the compiled
    agent rather than asserted on the input dict.

    A thread recorded under the old rule holds one identity line and a stack
    of memory lines, each an older version of the newest. The next turn sends
    a `RemoveMessage` for each of them beside the current line — through its
    own input, so the graph is still the only writer of `WidgetState` — and
    what the checkpointer holds afterwards is the identity line and exactly
    one memory line. This is the part a unit test on `backends._run` cannot
    show: that langgraph accepts a deletion arriving in a turn's input and
    persists the result.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from raglab.agents.widget import backends
    from raglab.agents.widget import conversation_memory as memory
    from raglab.agents.widget.tests.widget_examples import write_messages

    thread, dataset = 'legacy-standing-lines', 'diary-en'
    memory.forget(thread)
    identity = backends.ACTIVE_EXPERIMENT_PROMPT.format(experiment_id=thread,
                                                        dataset=dataset)
    stale = [SystemMessage(content=f'Dataset memory ({dataset}):\nnote {i}',
                           id=f'stale-{i}') for i in range(4)]
    write_messages(thread, [SystemMessage(content=identity, id='old-opening'),
                            *stale,
                            HumanMessage(content='old?', id='old-q'),
                            AIMessage(content='old.', id='old-a')])

    current = f'Dataset memory ({dataset}):\nnote 0; note 1; note 2; note 3'
    agent = build_agent(_RecordingModel(script=[AIMessage(content='New.')]))
    payload, config = backends._run('next?', thread, dataset=dataset,
                                    memory_text=current)
    agent.invoke(payload, config=config)

    kept = [str(m.content) for m in memory._channels(thread)['messages']
            if getattr(m, 'type', '') == 'system']
    assert kept == [identity, current]
