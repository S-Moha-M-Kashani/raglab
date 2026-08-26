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

