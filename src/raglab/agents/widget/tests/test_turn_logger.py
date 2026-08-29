# this is a unit test
"""The widget's readable one-row-per-question operational log."""
import json
import sqlite3

from raglab.agents.widget import turn_logger


def test_one_turn_is_readable_and_keeps_nested_steps_and_totals():
    turn_id = turn_logger.log_turn(
        thread_id='exp-1', experiment_id='exp-1', dataset_id='diary-fa',
        user_message_id='human-1', user_message='Did reranking help?',
        ai_message_id='ai-3', ai_message='Yes, on the missed questions.',
        steps=[
            {'id': 'step-1', 'kind': 'human', 'message_id': 'human-1',
             'text': 'Did reranking help?', 'latency_ms': 0},
            {'id': 'step-2', 'kind': 'tool', 'name': 'read_experiment',
             'latency_ms': 31},
            {'id': 'step-3', 'kind': 'ai', 'message_id': 'ai-3',
             'text': 'Yes, on the missed questions.', 'latency_ms': 1180,
             'input_tokens': 410, 'output_tokens': 75},
        ],
        total_input_tokens=410, total_output_tokens=75,
        total_latency_ms=1211, status='answered',
        memory_update_id=7)

    row = turn_logger.read_turn(turn_id)
    assert row['turn_id'] == turn_id
    assert row['thread_id'] == 'exp-1'
    assert row['dataset_id'] == 'diary-fa'
    assert row['user_message_id'] == 'human-1'
    assert row['ai_message_id'] == 'ai-3'
    assert row['total_input_tokens'] == 410
    assert row['total_output_tokens'] == 75
    assert row['total_tokens'] == 485
    assert row['total_latency_ms'] == 1211
    assert row['memory_update_id'] == 7
    assert row['status'] == 'answered'
    assert json.loads(row['steps_json'])[1]['name'] == 'read_experiment'


def test_missing_usage_stays_null_and_each_question_is_one_row():
    first = turn_logger.log_turn(
        thread_id='general', experiment_id='', dataset_id='',
        user_message_id='human-1', user_message='hello',
        ai_message_id='ai-1', ai_message='I only help with the RAG lab.',
        steps=[{'kind': 'human'}, {'kind': 'ai'}],
        total_input_tokens=None, total_output_tokens=None,
        total_latency_ms=12, status='irrelevant')
    second = turn_logger.log_turn(
        thread_id='general', experiment_id='', dataset_id='',
        user_message_id='human-2', user_message='Which port is used?',
        ai_message_id='ai-2', ai_message='9002', steps=[],
        total_input_tokens=10, total_output_tokens=2,
        total_latency_ms=22, status='answered')

    assert first != second
    assert turn_logger.read_turn(first)['total_tokens'] is None
    assert turn_logger.read_turn(first)['memory_update_id'] is None
    assert len(turn_logger.list_turns('general')) == 2


def test_schema_is_readable_and_does_not_depend_on_checkpoint_blobs():
    turn_logger.log_turn(
        thread_id='exp-2', experiment_id='exp-2', dataset_id='smoke-mini',
        user_message_id='human-2', user_message='What failed?',
        ai_message_id='ai-2', ai_message='Retrieval missed one quote.',
        steps=[{'kind': 'tool', 'name': 'read_experiment_questions'}],
        total_input_tokens=5, total_output_tokens=8,
        total_latency_ms=44, status='answered')

    with sqlite3.connect(turn_logger.db_path()) as db:
        columns = {row[1] for row in db.execute(
            'PRAGMA table_info(widget_turn_log)')}
    assert {'turn_id', 'thread_id', 'dataset_id', 'user_message',
            'ai_message', 'steps_json', 'total_tokens', 'total_latency_ms',
            'memory_update_id'} <= columns


def test_why_a_turn_was_or_was_not_filed_lands_on_its_own_row():
    """The memory decision is taken on a daemon thread after the answer has
    gone, so it has no caller to return to. Without a column of its own every
    outcome that was not a save — a refused policy, a dataset mismatch, a
    summarizer that raised — reached nobody at all."""
    turn_id = turn_logger.log_turn(
        thread_id='exp-9', experiment_id='exp-9', dataset_id='diary-en',
        user_message_id='human-9', user_message='Did reranking help?',
        ai_message='Yes.', steps=[])

    assert turn_logger.read_turn(turn_id)['memory_reason'] is None
    turn_logger.attach_memory_outcome(
        turn_id, 'not filed: the memory write failed (database is locked).')
    assert turn_logger.read_turn(turn_id)['memory_reason'] == (
        'not filed: the memory write failed (database is locked).')
    # No row named, nothing written and nothing raised: a turn whose own row
    # could not be written still has an outcome, and it has nowhere to go.
    turn_logger.attach_memory_outcome('', 'no row to amend')


def test_what_ended_a_turn_is_its_own_column_beside_what_became_of_its_memory():
    """A turn that never answers is still a turn, and the row it writes has to
    say what stopped it. That is not the same fact as what became of the
    memory it might have been filed as — a row can want both, and one column
    holding either would leave a reader unable to tell which he had."""
    interrupted = turn_logger.log_turn(
        thread_id='exp-8', experiment_id='exp-8', dataset_id='diary-en',
        user_message_id='human-8', user_message='what did that run score?',
        steps=[{'kind': 'human'}, {'kind': 'ai'}, {'kind': 'tool'}],
        total_input_tokens=2192, total_output_tokens=603,
        status='interrupted',
        status_reason='the model connection dropped mid-turn')

    row = turn_logger.read_turn(interrupted)
    assert row['status'] == 'interrupted'
    assert row['status_reason'] == 'the model connection dropped mid-turn'
    assert row['memory_reason'] is None      # nothing decided it either way
    assert row['ai_message'] is None         # there was no answer to record
    assert row['total_tokens'] == 2795       # and it was billed all the same

    # An ordinary turn says nothing there, which is what "nothing stopped it"
    # reads as.
    answered = turn_logger.log_turn(
        thread_id='exp-8', experiment_id='exp-8', dataset_id='diary-en',
        user_message_id='human-9', user_message='and now?', ai_message='0.71',
        steps=[])
    assert turn_logger.read_turn(answered)['status_reason'] is None


def test_a_log_written_before_the_reason_columns_existed_gains_them():
    """`CREATE TABLE IF NOT EXISTS` leaves an existing file alone, so a
    developer's widget.db from before a column would keep losing the value
    forever. One ALTER per missing column, checked on connect."""
    path = turn_logger.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE IF EXISTS widget_turn_log')
        db.execute(turn_logger.SCHEMA.replace(
            '  memory_reason TEXT,\n', '').replace(
            '  status_reason TEXT,\n', ''))
        db.commit()
    with sqlite3.connect(path) as db:
        have = {row[1] for row in db.execute(
            'PRAGMA table_info(widget_turn_log)')}
    assert 'memory_reason' not in have and 'status_reason' not in have

    turn_id = turn_logger.log_turn(
        thread_id='exp-old', experiment_id='exp-old', dataset_id='diary-en',
        user_message_id='human-old', user_message='and?', ai_message='yes',
        steps=[], status='interrupted', status_reason='the run died')
    turn_logger.attach_memory_outcome(turn_id, 'not filed: the policy declined.')
    row = turn_logger.read_turn(turn_id)
    assert row['memory_reason'] == 'not filed: the policy declined.'
    assert row['status_reason'] == 'the run died'


def test_the_log_waits_for_a_busy_file_instead_of_failing_the_turn():
    """Three writers share widget.db and the deferred memory writer now runs on
    a thread of its own, so one turn's write can overlap the next turn's. The
    timeout is stated rather than inherited from whatever the standard library
    happens to default to."""
    with turn_logger._connect() as db:
        assert db.execute('PRAGMA busy_timeout').fetchone()[0] == int(
            turn_logger.BUSY_TIMEOUT_SECONDS * 1000)


def test_a_providers_error_page_cannot_grow_the_log_without_bound():
    """`status_reason` is the one column here holding words nobody in this
    package wrote: it is `str(error)`, and what raised may be a provider
    handing back an HTML error page rather than a sentence. Stored verbatim,
    every failed turn on a bad afternoon put kilobytes of markup into a
    conversation log.

    The bound is `long_term_memory.MAX_SUMMARY_CHARS` rather than a number
    invented for this column — that is already this package's answer to how
    much prose one row of widget.db may hold. The head is what survives,
    because an exception says what it is in its first line, and the ellipsis
    says a reader is looking at a cut rather than at where the message ended.
    """
    from raglab.agents.widget import long_term_memory

    assert turn_logger.MAX_STATUS_REASON == long_term_memory.MAX_SUMMARY_CHARS
    page = '<!DOCTYPE html><html><body>' + 'Bad Gateway. ' * 4_000

    turn_id = turn_logger.log_turn(
        thread_id='exp-huge', experiment_id='exp-huge', dataset_id='diary-en',
        user_message_id='human-huge', user_message='and?', steps=[],
        status='interrupted', status_reason=page)

    stored = turn_logger.read_turn(turn_id)['status_reason']
    assert len(stored) == turn_logger.MAX_STATUS_REASON
    assert stored.startswith('<!DOCTYPE html>')  # the head, which says what it is
    assert stored.endswith('…')
    # A reason that already fits is stored exactly as it was raised.
    short = turn_logger.log_turn(
        thread_id='exp-huge', experiment_id='exp-huge', dataset_id='diary-en',
        user_message_id='human-short', user_message='and?', steps=[],
        status='interrupted', status_reason='the model connection dropped')
    assert turn_logger.read_turn(short)['status_reason'] == \
        'the model connection dropped'
