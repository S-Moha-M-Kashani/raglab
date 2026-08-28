# this is a unit test
"""The widget's durable, selective memory is separate from its transcript."""
import sqlite3

from raglab.agents.widget import long_term_memory as memory


def test_creates_long_term_schema_alongside_the_checkpointer():
    assert memory.memory_context('diary-en') == ''
    with sqlite3.connect(memory.db_path()) as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {'dataset_memory', 'global_memory', 'memory_updates'} <= tables


def test_updates_aggregate_by_dataset_and_keep_provenance():
    memory.save_memory_update(
        'diary-en', 'exp-1', 'chunking', 'Which chunker won?',
        'Session chunks preserved the relevant context.',
        'Chunking: session chunks preserved relevant context.')
    result = memory.save_memory_update(
        'diary-en', 'exp-2', 'retrieval', 'Did reranking help?',
        'Reranking improved the focused questions.',
        'Retrieval: reranking improved focused questions.')

    context = memory.memory_context('diary-en')
    assert 'Chunking: session chunks preserved relevant context.' in context
    assert 'Retrieval: reranking improved focused questions.' in context
    assert result['saved'] is True
    assert result['dataset_id'] == 'diary-en'
    with sqlite3.connect(memory.db_path()) as db:
        row = db.execute(
            'SELECT dataset_id, experiment_id, subtopic, question, answer '
            'FROM memory_updates ORDER BY id').fetchall()
    assert row == [
        ('diary-en', 'exp-1', 'chunking', 'Which chunker won?',
         'Session chunks preserved the relevant context.'),
        ('diary-en', 'exp-2', 'retrieval', 'Did reranking help?',
         'Reranking improved the focused questions.'),
    ]


def test_global_summary_is_stored_and_context_includes_both_scopes():
    memory.clear_long_term_memory()
    with memory._connect() as db:
        db.execute("INSERT INTO global_memory(id, summary, updated_at) "
                   "VALUES (1, 'Existing global context.', 'now')")
        db.commit()
    memory.save_memory_update(
        'diary-fa', 'exp-fa', 'chunking', 'q', 'a', 'Farsi finding.')
    memory.save_memory_update(
        'diary-en', 'exp-1', 'chunking', 'q', 'a', 'Dataset finding.',
        'Across datasets, semantic drift favored session chunking.',
        {'diary-fa', 'diary-en'})
    context = memory.memory_context('diary-en')
    assert 'Dataset finding.' in context
    assert 'Across datasets, semantic drift favored session chunking.' in context


def test_global_summary_requires_existing_context_and_two_dataset_ids():
    memory.clear_long_term_memory()
    memory.save_memory_update('diary-en', 'exp-1', 'chunking', 'q', 'a',
                              'Dataset finding.', 'unsupported global')
    assert 'unsupported global' not in memory.memory_context('diary-en')


def test_subtopic_is_normalized_and_bounded():
    memory.clear_long_term_memory()
    value = '  Retrieval / Reranking!!  ' + ('x' * 500)
    memory.save_memory_update('diary-en', 'exp-1', value, 'q', 'a', 'finding')
    with sqlite3.connect(memory.db_path()) as db:
        subtopic = db.execute(
            'SELECT subtopic FROM memory_updates ORDER BY id DESC').fetchone()[0]
    assert subtopic.startswith('retrieval-reranking-')
    assert len(subtopic) <= memory.MAX_SUBTOPIC_CHARS


def test_subtopic_normalization_keeps_unicode_words_bounded():
    memory.clear_long_term_memory()
    memory.save_memory_update('diary-fa', 'exp-1', 'بازیابی / رتبه‌بندی!!',
                              'q', 'a', 'finding')
    with sqlite3.connect(memory.db_path()) as db:
        subtopic = db.execute(
            'SELECT subtopic FROM memory_updates ORDER BY id DESC').fetchone()[0]
    assert subtopic == 'بازیابی-رتبه-بندی'


def test_summaries_are_bounded_and_empty_dataset_is_not_stored():
    huge = 'x' * (memory.MAX_SUMMARY_CHARS * 3)
    assert memory.save_memory_update(
        '', 'exp-empty', 'misc', 'q', 'a', huge, huge)['saved'] is False
    result = memory.save_memory_update(
        'diary-en', 'exp-1', 'misc', 'q', 'a', huge, huge)
    assert result['saved'] is True
    with sqlite3.connect(memory.db_path()) as db:
        dataset = db.execute(
            'SELECT summary FROM dataset_memory WHERE dataset_id = ?',
            ('diary-en',)).fetchone()[0]
        global_row = db.execute(
            'SELECT summary FROM global_memory WHERE id = 1').fetchone()
        empty_updates = db.execute(
            "SELECT COUNT(*) FROM memory_updates WHERE dataset_id = ''").fetchone()[0]
    assert len(dataset) <= memory.MAX_SUMMARY_CHARS
    assert global_row is None
    assert empty_updates == 0


def test_clear_removes_long_term_rows_but_preserves_other_widget_tables():
    memory.save_memory_update('diary-en', 'exp-1', 'chunking', 'q', 'a', 'summary')
    with sqlite3.connect(memory.db_path()) as db:
        db.execute('CREATE TABLE widget_marker (id INTEGER PRIMARY KEY)')
        db.execute('INSERT INTO widget_marker VALUES (1)')
        db.commit()
    memory.clear_long_term_memory()
    assert memory.memory_context('diary-en') == ''
    with sqlite3.connect(memory.db_path()) as db:
        assert db.execute('SELECT COUNT(*) FROM widget_marker').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM memory_updates').fetchone()[0] == 0


def test_global_summary_naming_a_dataset_is_refused_while_the_dataset_stores():
    """The evidence this guard exists for: one corpus's run in the row every
    corpus reads."""
    memory.clear_long_term_memory()
    memory.save_memory_update('nosrat-fa', 'exp-fa', 'chunking', 'q', 'a',
                              'Farsi finding.')
    result = memory.save_memory_update(
        'smoke-import-check', '20260828-160758-305a19', 'indexing',
        'What happened?', 'A flat structure.',
        'Indexing produced a flat structure here.',
        'Last experiment details for smoke-import-check: 6 questions analyzed.',
        {'nosrat-fa', 'smoke-import-check'})

    assert result['saved'] is True
    assert 'smoke-import-check' in result['global_refused']
    assert 'Indexing produced a flat structure here.' in \
        memory.memory_context('smoke-import-check')
    assert 'smoke-import-check' not in memory.memory_context('nosrat-fa')
    assert '6 questions analyzed' not in memory.memory_context('nosrat-fa')


def test_global_summary_naming_an_experiment_id_is_refused():
    memory.clear_long_term_memory()
    memory.save_memory_update('diary-fa', 'exp-fa', 'chunking', 'q', 'a',
                              'Farsi finding.')
    result = memory.save_memory_update(
        'diary-en', 'exp-en', 'chunking', 'q', 'a', 'English finding.',
        'Run 20260828-160758-305a19 shows reranking helps.',
        {'diary-fa', 'diary-en'})

    assert '20260828-160758-305a19' in result['global_refused']
    assert '20260828-160758-305a19' not in memory.memory_context('diary-fa')


def test_a_cross_dataset_pattern_still_reaches_every_thread():
    """What the guard must not cost: a finding that holds across corpora."""
    memory.clear_long_term_memory()
    memory.save_memory_update('diary-fa', 'exp-fa', 'chunking', 'q', 'a',
                              'Farsi finding.')
    result = memory.save_memory_update(
        'diary-en', 'exp-en', 'chunking', 'q', 'a', 'English finding.',
        'Session-aware chunking beat fixed windows in three of four corpora.',
        {'diary-fa', 'diary-en'})

    assert result['global_refused'] == ''
    for dataset in ('diary-fa', 'diary-en'):
        assert 'Session-aware chunking beat fixed windows in three of four ' \
            'corpora.' in memory.memory_context(dataset)


def test_older_global_rows_are_withheld_at_the_read_and_left_on_disk():
    """Rows written before the write gate existed: held back, never rewritten."""
    memory.clear_long_term_memory()
    memory.save_memory_update('nosrat-fa', 'exp-fa', 'chunking', 'q', 'a',
                              'Farsi finding.')
    memory.save_memory_update('smoke-import-check', 'exp-smoke', 'indexing',
                              'q', 'a', 'Smoke finding.')
    stored = ('Retrieval depth mattered more than the reranker everywhere.\n'
              'Last experiment details for smoke-import-check: 6 questions '
              'analyzed; min_group=3 was not met.')
    with memory._connect() as db:
        db.execute('INSERT INTO global_memory(id, summary, updated_at) '
                   'VALUES (1, ?, ?)', (stored, 'now'))
        db.commit()

    context = memory.memory_context('nosrat-fa')
    assert 'Retrieval depth mattered more than the reranker everywhere.' in context
    assert 'smoke-import-check' not in context
    assert 'min_group=3' not in context
    assert 'Withheld from global memory: 1 older note' in context
    with sqlite3.connect(memory.db_path()) as db:
        assert db.execute(
            'SELECT summary FROM global_memory WHERE id = 1').fetchone()[0] == stored


def test_a_dataset_id_is_matched_as_a_token_not_as_letters():
    assert memory.cross_dataset_violation('diary-fashion notes', {'diary-fa'}) == ''
    assert memory.cross_dataset_violation('diary-fa notes', {'diary-fa'})
