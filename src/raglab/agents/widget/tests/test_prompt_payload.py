# this is an integration test
"""Characterisation: what one model call carries today, and how it grows.

This pins the numbers `prompt_payload_probe.probe_thread` measures against a
fixed, twelve-turn scripted thread — three of those turns call `read_rag_skill`
(the real tool, over the real skills corpus) for `MAX_SKILL_READS` bodies
before answering. The scenario reproduced, offline and deterministically, the
two live findings that motivated this branch: a long thread's system messages
kept accumulating, and a tool call can put tens of kilobytes into a single
window.

Both are fixed as of 2026-08-29, and both at the prompt rather than in the
log. The thread still accumulates a memory line per accepted turn
— it is the record of what the widget was told — but `backends._run` marks each
one, and `hooks.trim_and_call` leaves every superseded line out of the call.
The counts below say 3 system messages on every call after the first, where
this file's previous round measured 2 climbing to 13, while the thread those
calls came from ends up holding twelve. What the model loses is stale
duplicates of its own memory; what it keeps is the newest, which is strictly
better context as well as a smaller prompt.

The second finding is fixed the same way. The thread still holds every tool
reply whole, and the call that reads one still carries every character of it —
that is the quality fence — but once the model has answered from a turn, that
turn's tool replies ride on as a stub naming the tool and what it was asked.
A twenty-message window is also a twenty-thousand-character window now, spent
on history and never on the turn being answered. Together they take the
thread's fifteen calls from 453,476 characters to 95,276.

Round 1 of this test pinned numbers from a probe that reimplemented
production's turn construction instead of driving it — no `system_prompt`,
an invented opening line, a single skill body instead of the real worst case.
The corrected probe drives `backends._run` and builds the agent the way
`backends._build_agent` does, so the numbers below are what the real seams
say today, not what a hand-rolled stand-in guessed.

Four later tasks on this branch each change what a turn sends: global memory
filtering and one standing memory line per call, then a size-bounded window
with stubbed closed-turn tool replies (all three landed), and interrupted-turn
handling.
Every number pinned below is a claim about *today's* behaviour, so a later
task's deliberate change shows up here as an assertion a reviewer can see
move — and why it moved — rather than as silence.
"""
from raglab.agents.widget import conversation_memory as memory
from raglab.agents.widget import hooks
from raglab.agents.widget.prompts import SYSTEM_PROMPT
from raglab.agents.widget.tests.prompt_payload_probe import (
    DEFAULT_THREAD, _RecordingModel, build_agent, probe_thread, render)


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
      accepted turn and the thread still keeps every version, but a
      superseded standing line is left out of the call. Before that change
      this row read 13 and was still climbing — the mechanism behind the
      live finding of twelve system messages on a 29-step thread — and
      nothing capped it, since `trim_and_call` exempts system lines from
      the window by design.
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
      reply can cost tens of kilobytes of one call's window. What changed on
      2026-08-29 is how long it costs it *for*: the jump now lands on the one
      call that reads the reply, and the calls after it carry a stub instead
      of the bodies. Each of the three spikes used to raise the floor under
      every later call; now the floor stays around 2.7 KB.
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
        == (3, 20, 2799)

    # The standing set is bounded now: SYSTEM_PROMPT and the opening line from
    # the first call, plus one memory line from the first turn that has memory
    # to state — and never a fourth, however long the thread or how much the
    # memory aggregate grows.
    system_counts = [p.system_messages for p in payloads]
    assert system_counts == [2] + [3] * 14

    # And the guarantee behind that flat line, which the counts alone cannot
    # show: the *thread* went on accumulating. It holds twelve system messages
    # — the opening line and one memory context per accepted turn, the same
    # twelve the live finding counted — and no call carried more than three of
    # them. The log keeps the record; the prompt is what got bounded.
    held = [m for m in memory._channels(DEFAULT_THREAD)['messages']
            if getattr(m, 'type', '') == 'system']
    assert len(held) == 12
    assert [memory.standing_mark(m) for m in held] == (
        [memory.IDENTITY_LINE] + [memory.MEMORY_LINE] * 11)

    # The window's non-system side is trimmed to MAX_HISTORY once the
    # transcript passes it — it must never exceed the ceiling.
    assert max(p.other_messages for p in payloads) == hooks.MAX_HISTORY
    assert [p.other_messages for p in payloads[-4:]] == [20, 20, 20, 20]

    # Each tool hop costs roughly the real worst case of one `read_rag_skill`
    # call — three full skill bodies, not one — over the very next call in
    # the same turn. That call is the one the model answers from, so it still
    # carries every character: the quality fence, measured.
    hop_turn_2, answer_turn_2 = payloads[2], payloads[3]
    assert answer_turn_2.total_chars - hop_turn_2.total_chars > 15_000

    # And the call after it — a new turn, so turn 2 is now closed — does not.
    # The 20 KB is in the log, not in the prompt; what rides on is a stub.
    assert payloads[4].total_chars < hop_turn_2.total_chars + 1_000
    # Three tool turns, three stubs, and the floor barely moves: no call after
    # a tool turn is anywhere near the 42,599 the last call used to carry.
    plateau = [p.total_chars for p in payloads if p.call_in_turn == 0][3:]
    assert max(plateau) < 3_000

    # Today's whole-thread total, and the two steps that brought it here. It
    # was 466,458 while every superseded memory line was still being reread;
    # sending one of each saved 12,982 characters and left 453,476, of which
    # three `read_rag_skill` replies re-sent on every later call were by far
    # the largest part. Bounding the window by size as well as count, and
    # letting a closed turn's tool replies travel as a stub, takes 358,200
    # characters off that — 79% of the whole thread — without shortening the
    # reply being read. The distinction matters and the numbers say it: turn 2's
    # answering call is 21,976 → 21,976, flat, since it had no earlier tool turn
    # to shed — but turn 5's is 42,273 → 22,373 and turn 9's 62,574 → 22,774.
    # Those two calls did get shorter, and what went out of them is the
    # *earlier* turns' bodies that used to ride along beside the reply being
    # read. That is the result, and it is a better one than "no tool-reading
    # call changed" would have been.
    assert sum(p.total_chars for p in payloads) == 95_276


def test_a_thread_that_accumulated_memory_lines_still_sends_one():
    """The prompt-time filter driven through the compiled agent rather than
    asserted on a hand-built request.

    A thread that has run for a while holds one identity line and a stack of
    memory contexts, each superseded by the next, and — the case a text
    heuristic gets wrong — possibly a system line the widget never wrote. The
    model is handed the identity line, the newest memory context and the
    foreign line; the thread keeps all of it, because the log is the record of
    what the widget was told and only the prompt is shaped.
    """
    from langchain_core.messages import (AIMessage, HumanMessage,
                                         SystemMessage)

    from raglab.agents.widget import backends
    from raglab.agents.widget.tests.widget_examples import write_messages

    def standing(text, mark):
        return SystemMessage(content=text,
                             additional_kwargs={memory.STANDING_LINE: mark})

    thread, dataset = 'accumulated-standing-lines', 'diary-en'
    memory.forget(thread)
    identity = backends.ACTIVE_EXPERIMENT_PROMPT.format(experiment_id=thread,
                                                        dataset=dataset)
    older = [f'Dataset memory ({dataset}):\nnote {i}' for i in range(4)]
    seeded = [standing(identity, memory.IDENTITY_LINE),
              *[standing(text, memory.MEMORY_LINE) for text in older],
              SystemMessage(content='SAFETY: never quote a key'),
              HumanMessage(content='old?', id='old-q'),
              AIMessage(content='old.', id='old-a')]
    write_messages(thread, seeded)

    current = f'Dataset memory ({dataset}):\nnote 0; note 1; note 2; note 3'
    model = _RecordingModel(script=[AIMessage(content='New.')])
    agent = build_agent(model)
    payload, config = backends._run('next?', thread, dataset=dataset,
                                    memory_text=current)
    agent.invoke(payload, config=config)

    # What the one call carried: SYSTEM_PROMPT, the identity line, the foreign
    # line and the newest memory context — four, not eight.
    assert model.raw[0][0] == 4
    # What the thread kept: everything it had, plus the line just written.
    kept = [str(m.content) for m in memory._channels(thread)['messages']
            if getattr(m, 'type', '') == 'system']
    assert kept == [identity, *older, 'SAFETY: never quote a key', current]


def test_a_follow_up_is_answerable_from_a_thread_whose_tool_reply_was_stubbed():
    """The quality falsifier for the stub, driven through the real agent.

    Reducing a closed turn's tool reply is the change on this branch most able
    to hurt an answer, so the claim it has to survive is not "the prompt got
    smaller" — that is the characterisation above — but "a later turn can still
    do its job". Two turns: the first reads three real skill bodies and answers
    from them; the second asks a follow-up about the same subject.

    What is asserted is what the model was *handed* on that second turn, since
    that is the whole of what it has to work with. If the stub dropped the
    tool's name, or the arguments it was called with, the model would know only
    that something had been read and would have no way to read it again — and
    these assertions would fail while a size-only test went on passing.
    """
    from langchain_core.messages import AIMessage

    from raglab.agents.widget import backends
    from raglab.agents.widget import long_term_memory
    from raglab.agents.widget.tests.prompt_payload_probe import _SKILL_NAMES
    from raglab.agents.widget.tools import read_rag_skill

    thread, dataset = 'stubbed-follow-up', 'diary-en'
    long_term_memory.clear_long_term_memory()
    memory.forget(thread)
    bodies = read_rag_skill.invoke({'names': _SKILL_NAMES})
    assert len(bodies) > 15_000  # the real worst case, not a stand-in

    def reads_the_skills(call_id):
        return AIMessage(content='', tool_calls=[
            {'name': 'read_rag_skill', 'args': {'names': _SKILL_NAMES},
             'id': call_id}])

    model = _RecordingModel(script=[
        reads_the_skills('c1'), AIMessage(content='Chunking, then.'),
        # The follow-up turn asks the very same tool for the very same thing —
        # which is only a sensible script because the stub told it how.
        reads_the_skills('c2'), AIMessage(content='And rerankers after that.')])
    agent = build_agent(model)
    for question in ('what do the skills say about chunking?',
                     'and what about the reranking one?'):
        payload, config = backends._run(question, thread, dataset=dataset)
        agent.invoke(payload, config=config)

    hop, answer, follow_up, second_answer = [
        [memory._text(m.content) for m in call] for call in model.seen]

    # Turn 1's answering call reads the bodies whole: the current turn is never
    # reduced, so the model reasons over everything the tool returned.
    assert bodies in answer

    # Turn 2 opens with turn 1 closed. The bodies are gone from the prompt...
    assert bodies not in ' '.join(follow_up)
    # ...replaced by one line that names the tool and the exact subject it was
    # asked for, which is what makes the same call re-issuable.
    stub = next(text for text in follow_up if text.startswith('[read_rag_skill'))
    assert _SKILL_NAMES in stub and str(len(bodies)) in stub
    assert len(stub) < 400
    # The reader's question and the answer written from those bodies survive in
    # full — what was reduced is the evidence, not the conversation.
    assert 'what do the skills say about chunking?' in follow_up
    assert 'Chunking, then.' in follow_up

    # And the follow-up's own tool reply arrives whole, in the call that reads
    # it: asking again really does get the bodies back.
    assert bodies in ' '.join(second_answer)
    # One stub only — the second turn's reply is still open when it is read.
    assert sum(1 for t in second_answer if t.startswith('[read_rag_skill')) == 1

    # The log is untouched: both replies are in the thread, whole.
    held = [memory._text(m.content)
            for m in memory._channels(thread)['messages']
            if getattr(m, 'type', '') == 'tool']
    assert held == [bodies, bodies]
