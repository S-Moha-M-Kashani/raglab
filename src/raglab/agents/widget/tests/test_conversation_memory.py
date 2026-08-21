# this is a unit test
"""What the widget remembers, and where.

The claim worth testing hardest is that a conversation outlives the process.
Everything else in this lab dies with it — the index by design — and the whole
reason this one file exists is that a reader's chat should not.
"""
import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import empty_checkpoint

from raglab.agents.widget import conversation_memory as memory


def _write(thread: str, said: str, replied: str) -> None:
    """Two turns straight into the checkpointer, without an LLM: the memory is
    what is under test, not the agent that fills it.

    The checkpoint is `empty_checkpoint()` filled in rather than a dict written
    out here, so this helper cannot drift from whatever shape the installed
    checkpointer actually keeps — and the config carries `checkpoint_ns`,
    which `SqliteSaver.put` reads without a default. It is the real saver
    doing the writing either way, which is the part that matters."""
    config = {'configurable': {'thread_id': thread, 'checkpoint_ns': ''}}
    saver = memory.saver()
    checkpoint = empty_checkpoint()
    checkpoint['id'] = f'{thread}-1'
    checkpoint['ts'] = '2026-08-21T00:00:00+00:00'
    checkpoint['channel_values'] = {
        'messages': [HumanMessage(content=said), AIMessage(content=replied)],
        'experiment_id': '', 'started_at': '2026-08-21T00:00:00+00:00'}
    saver.put(config, checkpoint, {'source': 'update', 'step': 1}, {})


def test_a_conversation_reads_back_in_order():
    _write('exp-1', 'what is the decision score?', 'the unweighted mean of four')
    read = memory.history('exp-1')
    assert read['thread'] == 'exp-1'
    assert read['turns'] == [
        {'role': 'you', 'text': 'what is the decision score?'},
        {'role': 'bot', 'text': 'the unweighted mean of four'}]


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
