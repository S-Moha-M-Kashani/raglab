# this is a unit test
"""What the widget remembers, and where.

The claim worth testing hardest is that a conversation outlives the process.
Everything else in this lab dies with it — the index by design — and the whole
reason this one file exists is that a reader's chat should not.
"""
import os
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from raglab.agents.widget import conversation_memory as memory
from raglab.agents.widget.tests.widget_examples import write_messages as _write_messages


def _write(thread: str, said: str, replied) -> None:
    """The ordinary two-turn exchange: a question and the reply to it. `replied`
    is not annotated `str` on purpose — a model's content is a string or a list
    of blocks, and both are things this log has to survive."""
    _write_messages(thread, [HumanMessage(content=said),
                             AIMessage(content=replied)])


def test_a_conversation_reads_back_in_order():
    _write('exp-1', 'what is the decision score?', 'the unweighted mean of four')
    read = memory.history('exp-1')
    assert read['thread'] == 'exp-1'
    assert read['turns'] == [
        {'role': 'you', 'text': 'what is the decision score?'},
        {'role': 'bot', 'text': 'the unweighted mean of four'}]


def test_the_two_state_fields_name_the_thread_and_when_it_began():
    """`WidgetState`'s two fields were declared and written by nothing, so the
    history route reported two empty strings as facts about every thread. A
    turn stamps them now, and the stamp is what these assert: the thread's own
    id, and a moment. The general thread belongs to no experiment, so its
    `experiment_id` is the empty string on purpose — that is a statement, not
    an absence."""
    assert memory.thread_stamp('exp-stamp')['experiment_id'] == 'exp-stamp'
    assert memory.thread_stamp(memory.GENERAL)['experiment_id'] == ''
    assert memory.thread_stamp('')['experiment_id'] == ''
    # ISO 8601 with an offset and to the second — the same precision the
    # leaderboard identifies a run by, and enough to say when a conversation
    # began without pretending to microseconds nobody asked for. Handed a
    # moment rather than reading the clock, so the format is what is pinned.
    fixed = datetime(2026, 8, 22, 9, 30, tzinfo=timezone.utc)
    assert memory.thread_stamp('exp-stamp', now=fixed)['started_at'] == (
        '2026-08-22T09:30:00+00:00')


def test_when_a_thread_began_is_stamped_once_and_never_moves():
    """A "when this began" that crept forward to the latest turn would be a
    field naming itself after something it is not — worse than the empty
    string it replaced, which at least admitted to knowing nothing. So the
    stamp is written only while the thread has none, and a thread that already
    carries one gets no `started_at` key back at all: langgraph writes what it
    is handed, and the only way to leave a channel alone is not to name it."""
    _write('exp-began', 'first question', 'first answer')
    began = memory.history('exp-began')['started_at']
    assert began
    assert 'started_at' not in memory.thread_stamp('exp-began')
    _write('exp-began', 'a later question', 'a later answer')
    assert memory.history('exp-began')['started_at'] == began


def test_a_seeded_thread_reports_the_fields_a_real_turn_would_have_written():
    """The route's whole claim about a thread: not just its turns but which
    experiment it is about and when it started. Seeded through the same stamp
    a real turn goes through, so this fails if the stamp stops producing
    values rather than passing on hard-coded ones."""
    _write('exp-fields', 'q', 'a')
    read = memory.history('exp-fields')
    assert read['experiment_id'] == 'exp-fields'
    assert read['started_at']


def test_a_thread_nobody_has_used_reads_as_empty_not_as_an_error():
    """A conversation that has not happened yet is not a failure. The empty log
    with its starters is the honest rendering of it."""
    assert memory.history('never-asked')['turns'] == []


def test_a_conversation_outlives_the_process():
    """The point of SQLite. The saver is dropped and reopened against the same
    file, which is what a restart of the lab does to it."""
    _write('exp-2', 'is leiden available here?', 'not on this installation')
    memory.close()
    assert memory.history('exp-2')['turns'][0]['text'] == 'is leiden available here?'


def test_forgetting_one_thread_leaves_every_other_alone():
    """New Chat resets the conversation you are in. A button that quietly wiped
    the other experiments' conversations would be a different button."""
    _write('exp-3', 'a', 'b')
    _write('exp-4', 'c', 'd')
    memory.forget('exp-3')
    assert memory.history('exp-3')['turns'] == []
    assert memory.history('exp-4')['turns'] != []


def test_the_database_is_never_the_developers_own_during_a_test():
    """The guard this file's own fixture provides, asserted rather than trusted:
    the suite must not deposit the developer's conversations, the same rule the
    ledger and .runs/ already live under."""
    assert os.environ['RAGLAB_WIDGET_DB'] != ''
    assert 'databases/widget.db' not in str(memory.db_path())


def test_a_reply_that_arrives_in_content_blocks_is_kept_not_dropped():
    """A reasoning or multi-block model answers with a list of content blocks
    rather than a string, and the reader watched that answer arrive: the live
    path (`backends.ask`) joins the blocks before the panel renders them. A log
    that dropped it would show a question with nothing underneath, which
    misrepresents the conversation as surely as inventing a turn would — and
    the joined text must be the same text, not a second wording of it."""
    _write('exp-5', 'why exactly four judged metrics?',
           [{'type': 'text', 'text': 'because'},
            {'type': 'text', 'text': 'a decision score needs all four'}])
    assert memory.history('exp-5')['turns'] == [
        {'role': 'you', 'text': 'why exactly four judged metrics?'},
        {'role': 'bot', 'text': 'because a decision score needs all four'}]


def test_a_turn_with_a_reported_account_reads_it_back():
    """`usage_metadata` rides on the stored `AIMessage` and survives the
    checkpointer round-trip untouched — it is not written here, only carried
    through. A reader who opens the widget later must see the same bill the
    live reply showed, or the account exists only until the next redraw."""
    _write_messages('exp-billed', [
        HumanMessage(content='how many tokens did that cost?'),
        AIMessage(content='1692 total', usage_metadata={
            'input_tokens': 1630, 'output_tokens': 62, 'total_tokens': 1692})])
    assert memory.history('exp-billed')['turns'] == [
        {'role': 'you', 'text': 'how many tokens did that cost?'},
        {'role': 'bot', 'text': '1692 total',
         'input_tokens': 1630, 'output_tokens': 62}]


def test_a_turn_with_no_reported_account_carries_no_keys():
    """A human turn never carries `usage_metadata` at all, and a CLI backend's
    `AIMessage` carries none either — `_accounted` in `backends.py` already
    reads that absence as "the backend did not account for it", never as
    zero, and the log has to keep that same distinction: a turn with nothing
    reported gets no `input_tokens`/`output_tokens` keys at all, not zeros a
    reader could mistake for a real, free answer."""
    _write('exp-unbilled', 'no account for this one', 'plain reply')
    turns = memory.history('exp-unbilled')['turns']
    assert turns == [
        {'role': 'you', 'text': 'no account for this one'},
        {'role': 'bot', 'text': 'plain reply'}]
    assert 'input_tokens' not in turns[1]
    assert 'output_tokens' not in turns[1]


def test_a_tool_call_is_a_record_of_its_own_never_the_answer():
    """A tool call is how an answer was reached rather than part of it, so it is
    still not a `bot` turn — but it is no longer nothing. The reader gets one
    `tool` row naming what was asked of the lab, which is the difference between
    an answer that stands on a real record and one the model wrote alone.

    What stays out is the tool's own body. `167 sessions` is the material the
    reply was built from, not something the reader watched arrive, and putting
    it in the log would make the chat a second, worse copy of `/dev/trace`."""
    _write_messages('exp-6', [
        HumanMessage(content='how many sessions in the diary?'),
        AIMessage(content='', tool_calls=[{'name': 'search_knowledge_base',
                                           'args': {'query': 'diary'},
                                           'id': 'call-1'}]),
        ToolMessage(content='167 sessions', tool_call_id='call-1'),
        AIMessage(content='167')])
    assert memory.history('exp-6')['turns'] == [
        {'role': 'you', 'text': 'how many sessions in the diary?'},
        {'role': 'tool', 'text': memory.tool_line('search_knowledge_base')},
        {'role': 'bot', 'text': '167'}]


def test_every_tool_a_turn_called_gets_its_own_row_in_the_order_called():
    """Two calls in one assistant message and a third in the next hop. The count
    and the order are the whole claim: a reader scrolling back is asking which
    records an answer stands on, and a log that collapses three calls into one
    line answers a question they did not ask."""
    _write_messages('exp-6b', [
        HumanMessage(content='compare the two runs'),
        AIMessage(content='', tool_calls=[
            {'name': 'read_experiment', 'args': {'id': 'a'}, 'id': 'c1'},
            {'name': 'read_experiment', 'args': {'id': 'b'}, 'id': 'c2'}]),
        ToolMessage(content='run a', tool_call_id='c1'),
        ToolMessage(content='run b', tool_call_id='c2'),
        AIMessage(content='', tool_calls=[{'name': 'search_knowledge_base',
                                           'args': {'query': 'recall'},
                                           'id': 'c3'}]),
        ToolMessage(content='a page about recall', tool_call_id='c3'),
        AIMessage(content='b recalls more')])
    assert [turn['role'] for turn in memory.history('exp-6b')['turns']] == [
        'you', 'tool', 'tool', 'tool', 'bot']
    assert memory.history('exp-6b')['turns'][1:4] == [
        {'role': 'tool', 'text': memory.tool_line('read_experiment')},
        {'role': 'tool', 'text': memory.tool_line('read_experiment')},
        {'role': 'tool', 'text': memory.tool_line('search_knowledge_base')}]


def test_a_turn_that_called_nothing_says_nothing_about_tools():
    """The other direction, and the one a reader is trusting: silence has to
    mean silence. If a log grew a `tool` row for a turn the model answered out
    of its own head, the row would be the lie the rest of this project's records
    are not allowed to tell."""
    _write('exp-6c', 'what is the decision score?', 'the unweighted mean of four')
    assert [turn['role'] for turn in memory.history('exp-6c')['turns']] == [
        'you', 'bot']


def test_a_tool_call_with_no_name_is_left_out_rather_than_named_nothing():
    """A malformed call — a chunk that never stated its name — is dropped. A row
    reading "called" with nothing after it would claim a tool ran and refuse to
    say which, which is worse than the silence, and the ephemeral live line
    already skips the same nameless chunk (`backends._tool_named`)."""
    _write_messages('exp-6d', [
        HumanMessage(content='anything?'),
        AIMessage(content='', tool_calls=[{'name': '', 'args': {}, 'id': 'c1'}]),
        ToolMessage(content='nothing', tool_call_id='c1'),
        AIMessage(content='no')])
    assert [turn['role'] for turn in memory.history('exp-6d')['turns']] == [
        'you', 'bot']


def test_a_tool_row_carries_no_token_account():
    """The account is a bill for a reply, and a tool row is not a reply. Pinned
    because `_turns` builds all three roles in one loop, and a stray
    `usage_metadata` on the message that carried the call would otherwise ride
    out on a row no reader could match it to."""
    _write_messages('exp-6e', [
        HumanMessage(content='how many sessions?'),
        AIMessage(content='', tool_calls=[{'name': 'search_knowledge_base',
                                           'args': {}, 'id': 'c1'}],
                  usage_metadata={'input_tokens': 11, 'output_tokens': 2,
                                  'total_tokens': 13}),
        ToolMessage(content='167', tool_call_id='c1'),
        AIMessage(content='167')])
    tool_row = memory.history('exp-6e')['turns'][1]
    assert tool_row == {'role': 'tool',
                        'text': memory.tool_line('search_knowledge_base')}


def test_recall_reads_another_experiments_conversation():
    """The gap thread-keying leaves: sitting on the board with nothing open,
    "what did I conclude about abc123?" must still be answerable."""
    _write('abc123', 'is the recall worth it?', 'the thread key already does it')
    said = memory.recall_conversation.invoke({'experiment_id': 'abc123'})
    assert 'is the recall worth it?' in said
    assert 'the thread key already does it' in said


def test_recall_says_so_when_there_is_no_conversation():
    """An experiment nobody has discussed comes back saying so. Returning
    nothing would read to the model as "there is nothing to say about it",
    which is a different claim from "nothing was said about it"."""
    said = memory.recall_conversation.invoke({'experiment_id': 'never-discussed'})
    assert 'never-discussed' in said
    assert 'no recorded conversation' in said.lower()


def test_recall_caps_what_it_hands_the_model_and_says_it_capped():
    """The cap is stated for the reason every other reader here states its own:
    a truncation nobody mentions reads as the whole of it.

    `write_messages` seeds one checkpoint per call, so building a long
    conversation is one call with every turn's messages rather than a loop of
    calls each overwriting the last — the same shape `write_messages`'s other
    callers already use."""
    long_thread = 'exp-long'
    messages = []
    for turn in range(memory.MAX_RECALLED + 5):
        messages += [HumanMessage(content=f'question {turn}'),
                     AIMessage(content=f'answer {turn}')]
    _write_messages(long_thread, messages)
    said = memory.recall_conversation.invoke({'experiment_id': long_thread})
    assert str(memory.MAX_RECALLED) in said


def test_what_the_widget_wrote_is_in_the_file_a_reader_opens():
    """LangGraph's checkpointer switches the file to write-ahead logging, which
    parks every write in a `-wal` sidecar until something checkpoints it — so
    a reader opening `widget.db` in a viewer saw a 4 KB shell with two empty
    tables while months of memory sat beside it. The saver puts the file back
    in rollback mode the moment it is built, so the main file is always the
    whole record and no sidecar outlives a transaction."""
    _write('exp-wal', 'is this in the file?', 'it is now')
    memory.close()
    path = memory.db_path()
    assert not path.with_name(path.name + '-wal').exists()
    import sqlite3
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA journal_mode').fetchone()[0] == 'delete'
        names = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'checkpoints', 'writes'} <= names


def test_a_trace_shows_every_step_the_model_took_not_only_the_two_the_reader_saw():
    """`history` renders the conversation the reader saw — question, reply.
    `trace` renders the conversation the *model* had: the system lines it was
    handed, the tool it called and with what, what the tool said back, and the
    reply. A developer checking why an answer went wrong needs the middle."""
    from langchain_core.messages import SystemMessage
    _write_messages('exp-trace', [
        SystemMessage(content='This conversation belongs to experiment x.'),
        HumanMessage(content='what ran?'),
        AIMessage(content='', tool_calls=[{
            'name': 'read_experiment', 'args': {'experiment_id': 'x'},
            'id': 'call-1'}]),
        ToolMessage(content='experiment x — baseline', tool_call_id='call-1',
                    name='read_experiment'),
        AIMessage(content='x ran the baseline.', usage_metadata={
            'input_tokens': 10, 'output_tokens': 4, 'total_tokens': 14})])
    steps = memory.trace('exp-trace')['steps']
    assert [s['kind'] for s in steps] == ['system', 'human', 'ai', 'tool', 'ai']
    assert steps[2]['tool_calls'] == [
        {'name': 'read_experiment', 'args': {'experiment_id': 'x'}, 'id': 'call-1'}]
    assert steps[3] == {'kind': 'tool', 'name': 'read_experiment',
                        'tool_call_id': 'call-1', 'text': 'experiment x — baseline'}
    assert steps[4]['input_tokens'] == 10 and steps[4]['output_tokens'] == 4
    assert 'exp-trace' in memory.threads()


def test_a_second_process_opening_a_busy_file_still_gets_its_saver():
    """The mode switch needs the file to itself; a second process — the CLI
    beside the server, a developer's one-off check — finds it busy. That must
    not be a crash on `saver()`: the record is still readable and writable in
    whatever mode the file is in, and the switch simply waits for a quieter
    opener."""
    import sqlite3
    memory.close()
    path = memory.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    other = sqlite3.connect(path, isolation_level=None)
    other.execute('PRAGMA journal_mode=WAL')
    # A read transaction held open: the shape of a server mid-request. Writes
    # from another connection still go through under WAL; leaving WAL does not.
    other.execute('BEGIN')
    other.execute('SELECT count(*) FROM sqlite_master').fetchall()
    try:
        assert memory.saver() is not None
    finally:
        other.execute('ROLLBACK')
        other.close()
        memory.close()


def test_every_widget_connection_waits_for_a_busy_file():
    """Three writers share widget.db — this checkpointer, the long-term store
    and the turn log — each behind its own lock, and the deferred memory pass
    now runs on a thread that can overlap the next turn. Waiting is the right
    answer; a 500 on a turn that answered is not. The value is pinned rather
    than inherited from whatever the standard library defaults to."""
    from raglab.agents.widget import long_term_memory, turn_logger

    assert memory.saver().conn.execute('PRAGMA busy_timeout').fetchone()[0] == (
        int(memory.BUSY_TIMEOUT_SECONDS * 1000))
    for module in (long_term_memory, turn_logger):
        with module._connect() as db:
            assert db.execute('PRAGMA busy_timeout').fetchone()[0] == int(
                module.BUSY_TIMEOUT_SECONDS * 1000)


def test_a_thread_recorded_with_the_removed_policy_channels_still_reads():
    # this is an integration test
    """`WidgetState` lost four channels on 2026-08-28 — `relevant`,
    `should_save`, `subtopic` and `reason` — once nothing evaluated a policy
    before the stamp and all four were written empty on every checkpoint.

    Threads recorded before that still hold them, and a conversation a reader
    had last week must not become unreadable because the schema shrank. A real
    graph writes the old shape here and the current one continues the same
    thread: a channel the schema no longer declares is simply not restored, and
    every turn is still there afterwards."""
    from langchain.agents import AgentState, create_agent
    from langchain_core.language_models import GenericFakeChatModel

    class LegacyState(AgentState):
        experiment_id: str
        started_at: str
        relevant: bool
        should_save: bool
        dataset_id: str
        subtopic: str
        reason: str

    thread = 'exp-legacy-channels'
    memory.forget(thread)
    create_agent(
        GenericFakeChatModel(messages=iter([AIMessage(content='the old answer')])),
        state_schema=LegacyState, checkpointer=memory.saver()).invoke(
            {'messages': [HumanMessage(content='the old question')],
             'experiment_id': thread, 'started_at': '2026-08-01T00:00:00+00:00',
             'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
             'subtopic': 'retrieval', 'reason': 'the policy accepted it'},
            config={'configurable': {'thread_id': thread}})
    stored = memory._channels(thread)
    assert stored['reason'] == 'the policy accepted it'

    read = memory.history(thread)
    assert [turn['text'] for turn in read['turns']] == [
        'the old question', 'the old answer']
    assert read['experiment_id'] == thread
    assert read['started_at'] == '2026-08-01T00:00:00+00:00'

    create_agent(
        GenericFakeChatModel(messages=iter([AIMessage(content='the new answer')])),
        state_schema=memory.WidgetState, checkpointer=memory.saver()).invoke(
            {'messages': [HumanMessage(content='the new question')],
             **memory.thread_stamp(thread),
             **memory.dataset_stamp('diary-en', thread)},
            config={'configurable': {'thread_id': thread}})

    continued = memory.history(thread)
    assert [turn['text'] for turn in continued['turns']] == [
        'the old question', 'the old answer',
        'the new question', 'the new answer']
    assert continued['started_at'] == '2026-08-01T00:00:00+00:00'
    assert memory.trace(thread)['dataset_id'] == 'diary-en'


def test_a_turn_is_closed_once_the_model_has_answered_from_it():
    """The one definition of a closed turn, stated where both readers of it
    take it from.

    A turn runs from a reader's question to just before the next one, and it is
    closed when its last message is an answer — an assistant message that asked
    for nothing further. Two things on this branch depend on exactly that:
    which tool replies may travel as a stub (`hooks._as_stubs`), and which turn
    was interrupted (a turn that is neither closed nor the thread's last).
    """
    from langchain_core.messages import SystemMessage

    def shapes(messages):
        return [memory.turn_shape(m) for m in messages]

    calls = [{'name': 'read_rag_skill', 'args': {'names': 'chunking'},
              'id': 'c1'}]
    answered = [HumanMessage(content='q1'),
                AIMessage(content='', tool_calls=calls),
                ToolMessage(content='body', tool_call_id='c1',
                            name='read_rag_skill'),
                AIMessage(content='a1')]
    asking = [HumanMessage(content='q2'),
              AIMessage(content='', tool_calls=calls),
              ToolMessage(content='body', tool_call_id='c1',
                          name='read_rag_skill')]

    # The shape vocabulary is what the split reads, and an assistant message is
    # two different things depending on whether it asked for anything.
    assert shapes(answered) == [memory.TURN_HUMAN, memory.TURN_CALL,
                                memory.TURN_TOOL, memory.TURN_ANSWER]

    # A finished turn followed by one still in flight: only the first is closed,
    # which is the state every model call is made in.
    turns = memory.conversation_turns(shapes(answered + asking))
    assert turns == [memory.Turn(0, 4, True), memory.Turn(4, 7, False)]
    assert memory.closed_turn_tool_replies(shapes(answered + asking)) == {2}

    # A standing line written between two turns decides nothing. It arrives
    # just before the question it was written for, and a split that let it
    # close or open a turn would answer differently depending on which turn
    # happened to write a memory line.
    with_line = answered + [SystemMessage(content='memory')] + asking
    assert memory.conversation_turns(shapes(with_line)) == [
        memory.Turn(0, 5, True), memory.Turn(5, 8, False)]

    # A thread that begins somewhere other than a question — seeded, repaired,
    # or resumed — still splits, rather than losing its opening messages.
    assert memory.conversation_turns(shapes(
        [AIMessage(content='hello')] + answered)) == [
        memory.Turn(0, 1, True), memory.Turn(1, 5, True)]
    assert memory.conversation_turns([]) == [memory.Turn(0, 0, False)]


def test_an_interrupted_turn_is_an_unclosed_turn_that_is_not_the_last():
    """The rule the prompt and the trace page both take their answer from.

    A run that dies after the graph has written something leaves a question and
    a tool exchange with nothing after them. Once the reader asks again, that
    turn is unclosed and no longer last, which is exactly what makes it
    recognisable — and what keeps the *current* turn, unclosed for the ordinary
    reason that the model is still working, out of it.
    """
    from langchain_core.messages import SystemMessage

    def shapes(messages):
        return [memory.turn_shape(m) for m in messages]

    calls = [{'name': 'read_experiment', 'args': {'experiment_id': 'e1'},
              'id': 'c1'}]
    abandoned = [HumanMessage(content='q1'),
                 AIMessage(content='', tool_calls=calls),
                 ToolMessage(content='the row', tool_call_id='c1',
                             name='read_experiment')]
    asked_again = [HumanMessage(content='q1 again'), AIMessage(content='a1')]

    # In flight: the only turn there is, so nothing is called interrupted —
    # the model is on its way to read that tool reply.
    assert memory.interrupted_turn_cuts(shapes(abandoned)) == {}

    # The live finding: the same three messages, followed by a turn that did
    # answer. The question keeps its place; the two messages after it are what
    # no call carries.
    assert memory.interrupted_turn_cuts(shapes(abandoned + asked_again)) \
        == {0: [1, 2]}

    # A turn that died before the model wrote anything is still an abandoned
    # question, and still says so.
    assert memory.interrupted_turn_cuts(
        shapes([HumanMessage(content='q1')] + asked_again)) == {0: []}

    # A standing line inside the span belongs to no turn and is never cut: it
    # is exempt from the window by design, and dropping one would take a
    # memory context away from the call that needed it.
    #
    # Defensive, not a guarantee anything relies on today: both callers strip
    # system messages before they ask (`hooks.trim_and_call` splits them off,
    # `dev_trace_page` filters them out), so neither can reach this branch.
    # It is pinned because the rule is written over shapes rather than over
    # either caller's list, and a third reader handing it a whole thread is
    # the obvious next use.
    with_line = [HumanMessage(content='q1'),
                 SystemMessage(content='memory'),
                 AIMessage(content='', tool_calls=calls),
                 ToolMessage(content='the row', tool_call_id='c1',
                             name='read_experiment')] + asked_again
    assert memory.interrupted_turn_cuts(shapes(with_line)) == {0: [2, 3]}

    # Whatever a thread holds before its first question is a fragment, not an
    # abandoned question, so nothing is marked under it.
    assert memory.interrupted_turn_cuts(
        shapes([AIMessage(content='hello')] + asked_again)) == {}

    # The shape the widget already knows about by name — an assistant message
    # that asked for nothing and said nothing, the `clichat` finding
    # `hooks.trim_and_call` reports as 'empty reply'. It answered nothing, so
    # it closes nothing, so a turn that died on one is interrupted like any
    # other. Reading it as an answer would leave the turn closed, exempt from
    # every reshaping here, and its abandoned tool body riding along whole.
    assert memory.turn_shape(AIMessage(content='')) == memory.TURN_OTHER
    assert memory.turn_shape({'kind': 'ai', 'text': ''}) == memory.TURN_OTHER
    empty = [HumanMessage(content='q1'),
             AIMessage(content='', tool_calls=calls),
             ToolMessage(content='the row', tool_call_id='c1',
                         name='read_experiment'),
             AIMessage(content='')]
    assert memory.conversation_turns(shapes(empty))[-1].closed is False
    assert memory.interrupted_turn_cuts(shapes(empty + asked_again)) \
        == {0: [1, 2, 3]}

    # The line that stands where the cut work was names how much went.
    note = memory.interrupted_note(2)
    assert '2 unfinished step(s)' in note and 'never answered' in note


def test_a_dead_turn_and_one_in_flight_are_told_apart_by_the_turn_log():
    """`interrupted_turn_cuts` exempts the last turn of whatever it is handed,
    which is right for a payload — a new question is always its end — and no
    answer at all for a *stored* thread, where the last turn is the one a
    reader of the log most needs the truth about.

    A thread that stops mid-turn has two meanings and one shape: a run still
    going, whose next model call continues it, or a run that died, whose next
    model call is the reader's next question. Nothing in the messages can say
    which. What can is the row `backends._log_interrupted_turn` writes under
    the question's own id — a run that died owes one and a run still going does
    not — and that is what `next_call_continues` reads.
    """
    from raglab.agents.widget import turn_logger

    calls = [{'name': 'read_experiment', 'args': {'experiment_id': 'e1'},
              'id': 'c1'}]
    mid_turn = [HumanMessage(content='q1', id='q-1'),
                AIMessage(content='', id='a-1', tool_calls=calls),
                ToolMessage(content='the row', tool_call_id='c1', id='t-1',
                            name='read_experiment')]

    # A thread nobody has used continues nothing.
    assert memory.next_call_continues('never-used') is False

    # A finished turn is followed by a new question, whatever else is true.
    _write_messages('closed-last', [HumanMessage(content='q1', id='q-1'),
                                    AIMessage(content='a1', id='a-1')])
    assert memory.next_call_continues('closed-last') is False

    # The same three messages under two threads: one running, one dead.
    _write_messages('still-running', mid_turn)
    _write_messages('run-died', mid_turn)
    assert memory.next_call_continues('still-running') is True
    turn_logger.log_turn(thread_id='run-died', experiment_id='run-died',
                         dataset_id='diary-en', user_message_id='q-1',
                         user_message='q1', status='interrupted',
                         status_reason='the model connection dropped')
    assert memory.next_call_continues('run-died') is False
    # The row is claimed by the question's id, so a row about some other turn
    # of the same thread says nothing about this one.
    assert turn_logger.interrupted_question_ids('run-died') == {'q-1'}
    assert memory.next_call_continues('still-running') is True

    # An answered row is not a dead one — only an interrupted status counts.
    turn_logger.log_turn(thread_id='still-running',
                         experiment_id='still-running',
                         dataset_id='diary-en', user_message_id='q-1',
                         user_message='q1', ai_message='a1', status='answered')
    assert memory.next_call_continues('still-running') is True


def test_a_tool_stub_names_the_tool_and_what_it_was_asked_for():
    """What a closed turn's tool reply travels as. The stub has one job beyond
    being short: a model reading it must be able to re-issue the same call, so
    it names the tool and the arguments — a reply that only said "20,000
    characters were here" would leave a follow-up question with no way back to
    the evidence."""
    body = 'x' * 20_000
    stub = memory.tool_stub('read_rag_skill', {'names': 'chunking,rerankers'},
                            body)
    assert stub.startswith('[read_rag_skill(')
    assert 'names=chunking,rerankers' in stub
    assert '20000 characters' in stub
    assert len(stub) < 200

    # A long argument is trimmed rather than reinstating the cost it replaced.
    wide = memory.tool_stub('read_rag_skill', {'names': 'a' * 500}, body)
    assert len(wide) < memory.MAX_STUB_ARGS + 200 and wide.endswith(']')

    # And a reply already shorter than any sentence about it is left alone:
    # the caller gets back exactly what it passed in, so nothing is spent to
    # lose an answer.
    assert memory.tool_stub('calculate', {'expression': '2+2'}, '4') == '4'
