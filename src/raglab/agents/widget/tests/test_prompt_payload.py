# this is an integration test
"""Characterisation: what one model call carries today, and how it grows.

This pins the numbers `prompt_payload_probe.probe_thread` measures against a
fixed, twelve-turn scripted thread — three of those turns call a tool before
answering. The scenario is chosen to reproduce, offline and deterministically,
the two live findings that motivated this branch: a long thread's system
messages keep accumulating (`_run`'s standing-line rule only ever adds, never
drops, a line the memory context has not said before) and a tool call can put
tens of kilobytes into a single window (`read_skill_body`, over a real skill
body).

Four later tasks on this branch each change what a turn sends: global memory
filtering, one standing memory line instead of one per turn, a size-bounded
window with stubbed closed-turn tool replies, and interrupted-turn handling.
Every number pinned below is a claim about *today's* behaviour, so a later
task's deliberate change shows up here as an assertion a reviewer can see
move — and why it moved — rather than as silence.
"""
from raglab.agents.widget import hooks
from raglab.agents.widget.tests.prompt_payload_probe import probe_thread, render


def test_prompt_payload_grows_across_a_scripted_thread(capsys):
    """Twelve turns, three of which call a tool, over a growing memory
    context. The measured numbers, and why each one is what it is:

    - 15 calls, not 12: turns 2, 5 and 9 each cost the tool hop plus the
      answer, so `probe_thread` returns one `TurnPayload` per model call.
    - The very first call already carries one system message: the thread's
      opening line (which experiment and dataset this is), stamped before
      any memory exists to summarise.
    - System messages climb to 12 by the last turn and never fall, because
      `_run`'s own rule is additive — a memory line already sent stays in the
      standing set, and `long_term_memory` produces a new line on every
      accepted turn (its aggregate keeps growing). This is the mechanism
      behind the live finding of twelve system messages on a 29-step thread.
    - Other-message count grows with the transcript and then flattens at 20
      from turn 8 onward: `hooks.MAX_HISTORY` trims the non-system window to
      its last 20 messages, so the raw transcript keeps growing but what
      reaches the model does not.
    - The two tool-hop calls (turns 2, 5) jump by roughly 7 KB each — one real
      skill body read through `read_skill_body` — and turn 9's second call
      jumps further still, past 20 KB, because turn 9 both reads a skill body
      and is deep enough in the transcript to also carry a full 20-message
      window. That single number is this branch's other live finding: one
      tool reply can cost tens of kilobytes of one call's window.
    """
    assert hooks.MAX_HISTORY == 20  # the ceiling this test's plateau assumes

    payloads = probe_thread(turns=12, tool_turns=frozenset({2, 5, 9}))
    print('\n' + render(payloads))
    out = capsys.readouterr().out
    assert 'chars total' in out  # the helper is readable under `pytest -s`

    assert len(payloads) == 15
    assert [p.turn for p in payloads if p.call_in_turn > 0] == [2, 5, 9]

    first, last = payloads[0], payloads[-1]
    assert (first.system_messages, first.other_messages, first.total_chars) \
        == (1, 1, 88)
    assert (last.system_messages, last.other_messages, last.total_chars) \
        == (12, 20, 17747)

    # System messages only ever grow, one per turn whose memory line changed.
    system_counts = [p.system_messages for p in payloads]
    assert system_counts == sorted(system_counts)
    assert system_counts[-1] - system_counts[0] == 11

    # The window's non-system side is trimmed to MAX_HISTORY once the
    # transcript passes it — it must never exceed the ceiling.
    assert max(p.other_messages for p in payloads) == hooks.MAX_HISTORY
    assert [p.other_messages for p in payloads[-4:]] == [20, 20, 20, 20]

    # The two tool hops each cost roughly one skill body's worth of characters
    # over the very next call in the same turn.
    hop_turn_2, answer_turn_2 = payloads[2], payloads[3]
    assert answer_turn_2.total_chars - hop_turn_2.total_chars > 5_000

    # Today's whole-thread total: the baseline every later task's version of
    # this same scripted thread is judged against.
    assert sum(p.total_chars for p in payloads) == 171_174
