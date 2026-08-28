# this is an integration test
"""Characterisation: what one model call carries today, and how it grows.

This pins the numbers `prompt_payload_probe.probe_thread` measures against a
fixed, twelve-turn scripted thread — three of those turns call `read_rag_skill`
(the real tool, over the real skills corpus) for `MAX_SKILL_READS` bodies
before answering. The scenario reproduces, offline and deterministically, the
two live findings that motivated this branch: a long thread's system messages
keep accumulating (`backends._run`'s standing-line rule only ever adds, never
drops, a line the memory context has not said before) and a tool call can put
tens of kilobytes into a single window.

Round 1 of this test pinned numbers from a probe that reimplemented
production's turn construction instead of driving it — no `system_prompt`,
an invented opening line, a single skill body instead of the real worst case.
The corrected probe drives `backends._run` and builds the agent the way
`backends._build_agent` does, so the numbers below are what the real seams
say today, not what a hand-rolled stand-in guessed.

Four later tasks on this branch each change what a turn sends: global memory
filtering, one standing memory line instead of one per turn, a size-bounded
window with stubbed closed-turn tool replies, and interrupted-turn handling.
Every number pinned below is a claim about *today's* behaviour, so a later
task's deliberate change shows up here as an assertion a reviewer can see
move — and why it moved — rather than as silence.
"""
from raglab.agents.widget import hooks
from raglab.agents.widget.prompts import SYSTEM_PROMPT
from raglab.agents.widget.tests.prompt_payload_probe import probe_thread, render


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
    - System messages climb to 13 by the last turn and never fall, because
      `backends._run`'s own rule is additive — a memory line already sent
      stays in the standing set, and `long_term_memory` produces a new line
      on every accepted turn (its aggregate keeps growing). This is the
      mechanism behind the live finding of twelve system messages on a
      29-step thread — one more here because `SYSTEM_PROMPT` itself is
      always present and this thread's opening line is also standing.
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
        == (13, 20, 45169)

    # System messages only ever grow, one per turn whose memory line changed,
    # plus the constant SYSTEM_PROMPT and opening line from the first call.
    system_counts = [p.system_messages for p in payloads]
    assert system_counts == sorted(system_counts)
    assert system_counts[-1] - system_counts[0] == 11

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
    # this same scripted thread is judged against.
    assert sum(p.total_chars for p in payloads) == 466_458
