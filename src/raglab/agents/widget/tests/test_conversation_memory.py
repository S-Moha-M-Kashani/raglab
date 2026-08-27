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


def test_the_moment_the_model_calls_a_tool_is_not_a_turn():
    """The other half of the same rule, pinned so widening the first cannot
    quietly widen this one too. A tool call is how an answer was reached, not
    part of it, and the reader was never shown it — so neither the empty
    message that carries the call nor the tool's own result is a turn."""
    _write_messages('exp-6', [
        HumanMessage(content='how many sessions in the diary?'),
        AIMessage(content='', tool_calls=[{'name': 'search_knowledge_base',
                                           'args': {'query': 'diary'},
                                           'id': 'call-1'}]),
        ToolMessage(content='167 sessions', tool_call_id='call-1'),
        AIMessage(content='167')])
    assert memory.history('exp-6')['turns'] == [
        {'role': 'you', 'text': 'how many sessions in the diary?'},
        {'role': 'bot', 'text': '167'}]


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
