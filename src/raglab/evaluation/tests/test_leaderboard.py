"""The leaderboard: `group()`/`verdict()`, what a sweep refuses to rank
together (grouping by question set and judge before ranking anything); and
the board — every experiment that touched one corpus, in one flat table,
ranking nothing at all."""
import json
import re

import pytest

from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import leaderboard
from raglab.agents.extra_tools import sweep


# --- the leaderboard, and what it refuses to rank together ------------------
# A decision score is comparable only against rows that scored the same questions
# with the same judge. One flat ranking over everything in `.runs/` is exactly
# where that gets forgotten, so the producer groups first and ranks second.

def _row(run_id, label, decision, ids, judge, stderr=None):
    return {'run_id': run_id, 'label': label, 'ragas_decision': decision,
            'ragas_decision_stderr': stderr, 'started_at': '2026-07-31 10:00:00',
            'seconds': 60, 'n_questions': len(ids), 'summary': {}, 'ragas': {},
            'config': {}, 'judge': judge,
            'selection': {'balance': 'difficulty', 'n': len(ids),
                          'question_ids': ids}}


def test_the_sweeps_own_ranking_reads_the_same_tie_leaderboard_verdict_would():
    # this is a unit test
    """A lead inside the combined error must read as a tie in the sweep's own
    printed ranking too, or the first thing anyone reads is the conclusion
    the leaderboard's own error test rejects. Asserted against the returned
    data — the same group `leaderboard.verdict` computes 'tie' for, and the
    exact combined-error arithmetic — with exactly one substring check left
    for the wording, since the words are the one part a computed value cannot
    stand in for."""
    judge = {'model': 'gemma4:e2b', 'provider': 'ollama'}
    ids = ['q1', 'q2']
    rows = [_row('r2', 'F llm relevance gate', 0.7375, ids, judge, stderr=0.0333),
            _row('r1', 'A baseline', 0.7222, ids, judge, stderr=0.0341)]
    lines = sweep.ranking_verdict(rows)
    assert len(lines) == 1, 'one comparability group, one line of verdict'
    found, = leaderboard.group(rows)
    assert leaderboard.verdict(found) == 'tie'
    lead = 0.7375 - 0.7222
    combined_error = (0.0333 ** 2 + 0.0341 ** 2) ** 0.5
    assert f'{lead:.4f}' in lines[0]
    assert f'{combined_error:.4f}' in lines[0]
    assert 'do not separate' in lines[0]


def test_the_sweeps_ranking_names_a_winner_when_there_is_one():
    # this is a unit test
    """Same shape as the tie test, on rows the leaderboard actually ranks."""
    judge = {'model': 'gemma4:e2b', 'provider': 'ollama'}
    ids = ['q1', 'q2']
    rows = [_row('r2', 'F', 0.90, ids, judge, stderr=0.01),
            _row('r1', 'A', 0.50, ids, judge, stderr=0.01)]
    lines = sweep.ranking_verdict(rows)
    assert len(lines) == 1
    found, = leaderboard.group(rows)
    assert leaderboard.verdict(found) == 'F'
    assert 'Winner: F' in lines[0]


def _different_questions():
    judge = {'model': 'gemma4:e2b', 'provider': 'ollama'}
    return [_row('r1', 'A', 0.61, ['q1', 'q2'], judge),
            _row('r2', 'F', 0.72, ['q3', 'q4'], judge)]


def _different_judges():
    ids = ['q1', 'q2']
    return [_row('r1', 'A', 0.61, ids, {'model': 'gemma4:e2b', 'provider': 'ollama'}),
            _row('r2', 'F', 0.72, ids, {'model': 'openai/gpt-5-mini',
                                        'provider': 'openrouter'})]


def _different_unrecorded_counts():
    judge = {'model': 'openai/gpt-5-mini', 'provider': 'openrouter'}
    rows = [_row('r1', 'A 24q', 0.6385, [], judge),
            _row('r2', 'A 3q', 0.5488, [], judge)]
    rows[0]['n_questions'] = 24
    rows[1]['n_questions'] = 3
    for r in rows:
        r['selection'] = {}
    return rows


@pytest.mark.parametrize('build_rows, reason', [
    (_different_questions, 'two samples are two measurements'),
    (_different_judges, 'a judge swap is a different measurement'),
    (_different_unrecorded_counts, 'different counts, different samples'),
], ids=['different-questions', 'different-judges', 'different-unrecorded-counts'])
def test_rows_are_grouped_only_with_what_they_can_be_compared_against(build_rows,
                                                                      reason):
    # this is a unit test
    """A decision score is comparable only against rows scored on the same
    questions by the same judge: different question ids, a judge swap, or two
    runs that never recorded which questions they scored (so they key on a
    bare count instead — a row measured on 30 balanced questions must never
    read as beating one measured on 24 strided ones) all land in separate
    tables, never one ranked together."""
    groups = leaderboard.group(build_rows())
    assert len(groups) == 2, reason
    assert all(len(g.rows) == 1 for g in groups)


def test_a_group_ranks_by_decision_score_and_keeps_the_unranked_rows_last():
    # this is a unit test
    judge = {'model': 'gemma4:e2b', 'provider': 'ollama'}
    ids = ['q1', 'q2']
    group, = leaderboard.group([
        _row('r1', 'A', 0.61, ids, judge),
        _row('r2', 'no judged metrics', None, ids, judge),
        _row('r3', 'F', 0.72, ids, judge),
    ])
    assert [r['label'] for r in group.rows] == ['F', 'A', 'no judged metrics']
    # Present, not dropped: a run that measured nothing is a fact about the run.
    assert group.rows[-1]['ragas_decision'] is None


def test_a_lead_inside_the_error_is_reported_as_a_tie():
    # this is a unit test
    """The margin has to be compared to the error, or the leaderboard
    manufactures conclusions a bare ranking cannot support."""
    judge = {'model': 'gemma4:e2b', 'provider': 'ollama'}
    ids = ['q1', 'q2']
    group, = leaderboard.group([
        _row('r1', 'A', 0.6487, ids, judge, stderr=0.03),
        _row('r2', 'F', 0.6501, ids, judge, stderr=0.03),
    ])
    assert leaderboard.verdict(group) == 'tie'
    clear, = leaderboard.group([
        _row('r1', 'A', 0.50, ids, judge, stderr=0.01),
        _row('r2', 'F', 0.72, ids, judge, stderr=0.01),
    ])
    assert leaderboard.verdict(clear) == 'F'


def _unrecorded_sample_rows():
    judge = {'model': 'openai/gpt-5-mini', 'provider': 'openrouter'}
    rows = [_row('r1', 'D', 0.6501, [], judge, stderr=0.01),
            _row('r2', 'A', 0.5000, [], judge, stderr=0.01)]
    for r in rows:
        r['selection'] = {}
        r['n_questions'] = 24
    return rows


def _no_measured_error_rows():
    judge = {'model': 'openai/gpt-5-mini', 'provider': 'openrouter'}
    ids = ['q1', 'q2']
    return [_row('r1', 'A', 0.50, ids, judge, stderr=None),
            _row('r2', 'F', 0.72, ids, judge, stderr=None)]


@pytest.mark.parametrize('build_rows', [
    _unrecorded_sample_rows, _no_measured_error_rows,
], ids=['sample-not-recorded', 'no-measured-error'])
def test_a_group_that_cannot_be_ranked_reports_unknown(build_rows):
    # this is a unit test
    """Two different reasons a group refuses a winner. Two runs of 24
    questions apiece may still be two *different* 24 — nothing on those rows
    says which, so even with errors measured the comparison is not
    established (`selection` predates `RunResult`, so equal counts are not a
    shared sample). Or the rows recorded a sample but no error, and `± 0` on
    the oldest rows would present them as the most precise. Both must read as
    'unknown' rather than manufacturing a decision. `leaderboard.markdown` no
    longer renders a per-group verdict line for either reason — that
    rendering retired along with per-group markdown, not moved anywhere in
    particular — so what stays checkable here is the verdict computation
    itself."""
    found, = leaderboard.group(build_rows())
    assert leaderboard.verdict(found) == 'unknown'


def test_the_group_that_decides_something_is_printed_first():
    # this is a unit test
    """Sorting by question count would put a group of unrecorded samples,
    which cannot be ranked at all, above the group that decides something —
    a reader opens this for the live decision."""
    judge = {'model': 'gemma4:e2b', 'provider': 'ollama'}
    stale = _row('r0', 'old 100q', 0.61, [], judge)
    stale['selection'], stale['n_questions'] = {}, 100
    stale['started_at'] = '2026-07-29 10:00:00'
    live_rows = [_row('r1', 'F', 0.90, ['q1', 'q2'], judge, stderr=0.01),
                 _row('r2', 'A', 0.50, ['q1', 'q2'], judge, stderr=0.01)]
    groups = leaderboard.group([stale] + live_rows)
    assert leaderboard.verdict(groups[0]) == 'F', [g.sample for g in groups]


def test_every_judge_label_reads_as_a_noun_after_judged_by():
    # this is a unit test
    """`Group.judge` reads after "judged by" in the sweep's own printer
    (`ranking_verdict`), so every branch has to be a noun phrase — checked
    directly on the property, since `leaderboard.markdown` no longer renders
    a per-group heading to read it through."""
    judge = {'model': '', 'provider': ''}
    unjudged, = leaderboard.group([_row('r1', 'retrieval only', None,
                                        ['q1'], judge)])
    assert unjudged.judge == 'no judge — nothing on these rows was judged'


def test_the_markdown_names_the_judge_and_the_decision_on_every_board_row():
    # this is a unit test
    """Same claim `test_the_markdown_names_the_sample_and_the_judge_on_every_group`
    made before `leaderboard.markdown` stopped taking `Group`s: the judge and
    the decision score are readable on the printed row. The row now lives in
    a `Board`, built through `by_dataset`, rather than a comparability
    `Group`."""
    text = leaderboard.markdown(leaderboard.by_dataset([
        _board_row('r1', 'diary-fa', 0.61, ('q1', 'q2'),
                   {'model': 'gemma4:e2b', 'provider': 'ollama'})]))
    assert 'gemma4:e2b' in text and 'ollama' in text
    assert '0.6100' in text
    assert 'r1' in text, 'the run id is what makes a row checkable'


def test_the_run_list_carries_the_two_fields_comparability_needs(tmp_path,
                                                                 monkeypatch):
    # this is an integration test
    """Writes its own run file rather than reading `.runs/`: a test that
    skips when the developer's disk happens to be empty is not coverage."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    (tmp_path / '20260731-120000-abc123.json').write_text(json.dumps({
        'run_id': '20260731-120000-abc123', 'label': 'A baseline',
        'selection': {'balance': 'difficulty', 'n': 2,
                      'question_ids': ['q1', 'q2']},
        'summary': {'n_questions': 2},
        'ragas': {'metrics': {'faithfulness': 0.9}, 'decision': 0.61,
                  'decision_spread': {'stderr': 0.02},
                  'judge': {'model': 'gemma4:e2b', 'provider': 'ollama'}},
    }), encoding='utf-8')
    row, = evaluate.list_runs(limit=5)
    assert row['selection']['question_ids'] == ['q1', 'q2']
    assert row['judge'] == {'model': 'gemma4:e2b', 'provider': 'ollama'}
    # And the grouping keys off exactly those, so the two travel together.
    found, = leaderboard.group([row])
    assert found.question_ids == ('q1', 'q2')
    assert found.judge_model == 'gemma4:e2b'


# --- the pipeline sentence -------------------------------------------------

def test_the_pipeline_sentence_names_one_fragment_per_step_that_ran():
    # this is a unit test
    """The board's leftmost column is one sentence per row, and each fragment
    is inked with the step it belongs to. Assembled here rather than in the
    page, for the reason `board_dict` exists: two surfaces that each derived
    the sentence could describe one row two ways."""
    config = {
        'index': {'chunker': 'fixed-overlap', 'contextual': True,
                  'hierarchy': 'leiden', 'embedder': 'sentence-transformers',
                  'embed_model': 'intfloat/multilingual-e5-small'},
        'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'lexical',
                      'grader': 'llm'},
        'generation': {'answerer': 'llm'},
        'agent': {'scope': ''},
    }
    assert leaderboard.pipeline_fragments(config) == [
        {'step': 'index',
         'text': 'fixed-overlap+ctx·leiden·sentence-transformers·multilingual-e5-small'},
        {'step': 'retrieval', 'text': 'hybrid-rrf·lexical·llm'},
        {'step': 'generation', 'text': 'llm'},
    ]


def test_a_step_that_did_not_run_is_absent_from_the_sentence():
    # this is a unit test
    """An index build's sentence is its index fragment and nothing else. Padding
    it with em-dashes for the three stages that never ran would draw a row that
    looks like a failed evaluation instead of a successful build."""
    fragments = leaderboard.pipeline_fragments(
        {'index': {'chunker': 'session', 'embedder': 'token-hash'}})
    assert fragments == [{'step': 'index', 'text': 'session·token-hash'}]
    assert leaderboard.pipeline_fragments({}) == []


def test_the_sentence_keeps_the_embedding_model_and_drops_its_vendor():
    # this is a unit test
    """Two `fastembed` rows can be two entirely different representations, so
    the model has to be on the row — but the vendor prefix is the same on every
    row and costs the width the sentence needs."""
    fragments = leaderboard.pipeline_fragments(
        {'index': {'chunker': 'session', 'embedder': 'fastembed',
                   'embed_model': 'BAAI/bge-small-en-v1.5'}})
    assert fragments[0]['text'] == 'session·fastembed·bge-small-en-v1.5'


def test_a_scope_that_is_off_writes_no_agent_fragment():
    # this is a unit test
    """The agent is off by default and off is the common case; a plum fragment
    reading 'off' on every row would spend the sentence's width saying nothing."""
    base = {'index': {'chunker': 'session'}, 'agent': {'scope': ''}}
    assert [f['step'] for f in leaderboard.pipeline_fragments(base)] == ['index']
    lit = {'index': {'chunker': 'session'}, 'agent': {'scope': 'full'}}
    assert leaderboard.pipeline_fragments(lit)[-1] == {'step': 'agent',
                                                      'text': 'full'}


def test_a_knob_set_to_none_is_absent_from_the_sentence():
    # this is a unit test
    """`grader` defaults to the literal 'none' in lab_config (`reranker`
    defaults to 'lexical' — this test passes 'none' for it explicitly, to
    prove the same filter catches both), and a fragment reading
    'rrf·none·none' would spend the sentence's width saying which stages did
    nothing."""
    fragments = leaderboard.pipeline_fragments(
        {'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'none',
                       'grader': 'none'}})
    assert fragments == [{'step': 'retrieval', 'text': 'hybrid-rrf'}]


# --- the board: one table per dataset --------------------------------------
# Deliberately NOT the comparability grouping above. `group()` is what a *sweep*
# ranks by, and it stays. The board is a listing: every experiment that touched
# one corpus, in one table, sorted by whatever column the reader clicks. It
# names no winner, precisely because rows judged differently share it.

def _board_row(run_id, dataset, decision, ids=('q1',), judge=None):
    # `state` the way `_board_row` serves it: every real board row carries one,
    # and the printer's state column reads it.
    return {'run_id': run_id, 'experiment_id': run_id, 'dataset': dataset,
            'state': 'done', 'error': '',
            'label': run_id, 'ragas_decision': decision,
            'ragas_decision_stderr': None, 'started_at': '2026-08-01 10:00:00',
            'seconds': 60, 'n_questions': len(ids), 'config': {},
            'judge': judge or {},
            'selection': {'question_ids': list(ids), 'n': len(ids)}}


def test_one_dataset_is_one_table_whatever_the_questions_or_the_judge():
    # this is a unit test
    """The board's whole point. `group()` would put these three in three
    separate tables — different question sets, different judges. The board puts
    every experiment that touched diary-fa in one, because that is the question
    a reader actually asks of it."""
    boards = leaderboard.by_dataset([
        _board_row('a', 'diary-fa', 0.71, ('q1', 'q2'), {'model': 'sonnet'}),
        _board_row('b', 'diary-fa', 0.68, ('q3',), {'model': 'gpt-4o'}),
        _board_row('c', 'meetings-de', 0.55, ('q1',), {'model': 'sonnet'}),
    ])
    assert [(b.dataset, b.n_experiments) for b in boards] == [
        ('diary-fa', 2), ('meetings-de', 1)]
    assert [r['run_id'] for r in boards[0].rows] == ['a', 'b']


def test_a_board_orders_by_decision_and_keeps_the_unjudged_rows_last():
    # this is a unit test
    """The served order is the ranking, and the sorter's third click restores
    exactly it. A row that judged nothing sorts last rather than as a zero: a
    fabricated 0.0 would read as a measured refusal."""
    boards = leaderboard.by_dataset([
        _board_row('low', 'diary-fa', 0.40),
        _board_row('none', 'diary-fa', None),
        _board_row('high', 'diary-fa', 0.90),
    ])
    assert [r['run_id'] for r in boards[0].rows] == ['high', 'low', 'none']


def test_a_dataset_no_catalogue_describes_still_gets_a_table():
    # this is a unit test
    """Deleting a corpus fixture must not rewrite history. The board names the
    dataset from the rows, so an experiment over a corpus that is gone keeps
    its table under its raw id."""
    boards = leaderboard.by_dataset([_board_row('a', 'deleted-corpus', 0.5)])
    assert [b.dataset for b in boards] == ['deleted-corpus']


def test_a_row_with_no_dataset_is_the_builtin_diary():
    # this is a unit test
    """No dataset predates the field and means the built-in diary — the only
    corpus that existed then. The same fallback `_key` already applies."""
    boards = leaderboard.by_dataset([_board_row('old', '', 0.5)])
    assert [b.dataset for b in boards] == ['diary-fa']


def test_boards_are_ordered_newest_experiment_first():
    # this is a unit test
    """Which corpus you were working on last is the one you want on top."""
    old = _board_row('old', 'meetings-de', 0.5)
    old['started_at'] = '2026-01-01 10:00:00'
    boards = leaderboard.by_dataset([old, _board_row('new', 'diary-fa', 0.5)])
    assert [b.dataset for b in boards] == ['diary-fa', 'meetings-de']


# --- the union of the two durable records ----------------------------------
# The ledger holds every job — builds, retrievals, evaluations — but not the
# four judged metrics and not the judge. `.runs/` holds those, for evaluations
# only. Neither is sufficient, so the board reads both and joins them on the id
# the ledger already stores: `experiment_id = result['run_id'] or job['id']`.

def _write_run(tmp_path, monkeypatch, run_id, dataset, metrics, decision):
    """One run file, in the shape `list_runs` reads back."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    (tmp_path / f'{run_id}.json').write_text(json.dumps({
        'run_id': run_id, 'label': run_id, 'dataset': dataset,
        'started_at': '2026-08-01 10:00:00', 'seconds': 12,
        'config': {'index': {'chunker': 'session', 'embedder': 'token-hash'}},
        'summary': {'n_questions': 2},
        'ragas': {'metrics': metrics, 'decision': decision,
                  'decision_spread': {'stderr': 0.01},
                  'judge': {'model': 'sonnet-4', 'provider': 'openrouter'}},
        'selection': {'question_ids': ['q1', 'q2'], 'n': 2},
    }), encoding='utf-8')


def test_a_row_in_both_records_takes_its_metrics_from_the_run_file(
        tmp_path, monkeypatch):
    # this is an integration test
    """The ledger supplies identity, kind, state and the knobs; the run file
    supplies the four metrics and the judge, because the run file is where the
    number was computed and the one a reader can open to check it."""
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    # `insert_job` does not exist on the ledger: the real write path is
    # `record(job, state, path=...)`, state passed separately from the job.
    ledger.record({'id': 'ignored', 'kind': 'run', 'config': {}, 'seconds': 12,
                   'result': {'run_id': 'r1', 'dataset': 'diary-fa',
                              'label': 'r1', 'started_at': '2026-08-01 10:00:00',
                              'ragas': {'decision': 0.71}}}, 'done', path=db)
    _write_run(tmp_path, monkeypatch, 'r1', 'diary-fa',
               {'faithfulness': 0.8, 'answer_relevancy': 0.7,
                'llm_context_precision_with_reference': 0.6,
                'context_recall': 0.75}, 0.7125)
    rows = leaderboard.board_rows(db_path=db)
    assert len(rows) == 1
    found = rows[0]
    assert found['source'] == 'both'
    assert found['kind'] == 'run'
    assert found['metrics']['faithfulness'] == 0.8
    assert found['judge']['model'] == 'sonnet-4'


def test_a_ledger_row_with_no_run_file_shows_no_metrics_rather_than_zeros(
        tmp_path, monkeypatch):
    # this is an integration test
    """An index build measured nothing, and that is a fact about the build. It
    appears, with the four metric columns empty — never 0.0, which would sort
    below real rows and read as a measured refusal."""
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'empty')
    ledger.record({'id': 'job-1', 'kind': 'index',
                   'config': {'index': {'chunker': 'session',
                                        'dataset': 'smoke-mini',
                                        'embedder': 'token-hash'}},
                   'seconds': 3, 'result': {}}, 'done', path=db)
    rows = leaderboard.board_rows(db_path=db)
    assert rows[0]['source'] == 'ledger'
    assert rows[0]['kind'] == 'index'
    assert rows[0]['metrics'] == {}
    assert rows[0]['decision'] is None
    assert rows[0]['judge'] == {}


def test_a_run_file_with_no_ledger_row_still_appears(tmp_path, monkeypatch):
    # this is an integration test
    """The ledger is written in `Jobs.run`, so every evaluation older than the
    ledger has a run file and no ledger row. Reading only the ledger would drop
    those off the board without saying so — which is why this is a union."""
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    ledger.connect(db).close()          # an empty but real ledger
    _write_run(tmp_path, monkeypatch, 'old-run', 'diary-fa',
               {'faithfulness': 0.5}, 0.5)
    rows = leaderboard.board_rows(db_path=db)
    assert [r['experiment_id'] for r in rows] == ['old-run']
    assert rows[0]['source'] == 'run'
    assert rows[0]['kind'] == 'run'
    assert rows[0]['state'] == 'done'


def test_the_metric_columns_are_exactly_the_four_that_decide(
        tmp_path, monkeypatch):
    # this is an integration test
    """Read from DECISION_METRICS, never spelled out, so a board column cannot
    drift from what actually decides. A fifth metric on the run is reported
    elsewhere and does not belong in these four."""
    from raglab.evaluation import ragas_judged_metrics as judged
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    ledger.connect(db).close()
    _write_run(tmp_path, monkeypatch, 'r1', 'diary-fa',
               {name: 0.5 for name in judged.DECISION_METRICS}
               | {'answer_correctness': 0.9}, 0.5)
    assert set(leaderboard.board_rows(db_path=db)[0]['metrics']) \
        == set(judged.DECISION_METRICS)


def test_every_row_carries_its_pipeline_sentence(tmp_path, monkeypatch):
    # this is an integration test
    """Assembled once, on the way out, so the page never derives it."""
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    ledger.connect(db).close()
    _write_run(tmp_path, monkeypatch, 'r1', 'diary-fa', {}, None)
    assert leaderboard.board_rows(db_path=db)[0]['pipeline'] == [
        {'step': 'index', 'text': 'session·token-hash'}]


def test_a_ledger_only_index_build_still_gets_a_pipeline_sentence(
        tmp_path, monkeypatch):
    # this is an integration test
    """`ledger.experiments()` returns flat columns — `chunker`, `embedder` —
    never a nested config; only a run file carries one, and an index build
    never writes a run file. Without reshaping the flat columns back into a
    config, this row's leftmost column would be blank for exactly the rows
    the board exists to show."""
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'empty')
    ledger.record({'id': 'job-1', 'kind': 'index',
                   'config': {'index': {'chunker': 'session',
                                        'dataset': 'smoke-mini',
                                        'embedder': 'token-hash'}},
                   'seconds': 3, 'result': {}}, 'done', path=db)
    assert leaderboard.board_rows(db_path=db)[0]['pipeline'] == [
        {'step': 'index', 'text': 'session·token-hash'}]


def test_a_ledger_only_retrieval_gets_its_retrieval_fragment(
        tmp_path, monkeypatch):
    # this is an integration test
    """Same gap, for a retrieval job: no run file ever covers one, so the
    retrieval fragment has to come from the ledger's own flat
    retriever/reranker/grader columns rather than a nested config nobody
    wrote down."""
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'empty')
    ledger.record({'id': 'job-2', 'kind': 'retrieve',
                   'config': {'index': {'chunker': 'session',
                                        'embedder': 'token-hash'},
                              'retrieval': {'retriever': 'hybrid-rrf',
                                           'reranker': 'lexical',
                                           'grader': 'none'}},
                   'seconds': 2, 'result': {}}, 'done', path=db)
    assert leaderboard.board_rows(db_path=db)[0]['pipeline'] == [
        {'step': 'index', 'text': 'session·token-hash'},
        {'step': 'retrieval', 'text': 'hybrid-rrf·lexical'}]


# --- the serialised shape and the command line ------------------------------

def test_the_board_serialises_to_one_shape_for_the_page_and_the_command():
    # this is a unit test
    """One serialised shape, so the command line and the panel's route cannot
    come to disagree about what a board is: two callers that each assembled it
    could serve two different answers from one set of rows."""
    board = leaderboard.by_dataset([_board_row('a', 'diary-fa', 0.7)])[0]
    shape = leaderboard.board_dict(board)
    assert set(shape) == {'dataset', 'n_experiments', 'newest', 'rows'}
    assert shape['dataset'] == 'diary-fa'
    assert shape['n_experiments'] == 1


def test_the_markdown_prints_one_table_per_dataset_and_names_no_winner():
    # this is a unit test
    """The board names no winner: rows judged differently share it, so a
    'winner by more than the combined error' claim would compare numbers that
    never met. `verdict()` still says it — for a sweep, whose candidates are
    genuinely comparable — but not here.

    The ban is checked case-insensitively and only inside the tables
    themselves — a case-sensitive check would let a lowercase '**winner: a**'
    row straight through, which is not a guard at all — while the disclaimer
    paragraph above the tables is allowed to use the plain word in explaining
    why, and that disclaimer is asserted present rather than merely assumed."""
    boards = leaderboard.by_dataset([
        _board_row('a', 'diary-fa', 0.71, judge={'model': 'sonnet'}),
        _board_row('b', 'meetings-de', 0.55, judge={'model': 'gpt-4o'}),
    ])
    text = leaderboard.markdown(boards)
    assert '## diary-fa' in text and '## meetings-de' in text
    lowered = text.lower()
    assert 'nothing here names a winner' in lowered
    assert 'are columns you compare on' in lowered
    tables = text.split('## diary-fa', 1)[1].lower()
    for banned in ('winner', 'tie', 'not comparable'):
        assert banned not in tables, (
            f'no table may print {banned!r}: a board mixes judges and question '
            'sets, so no claim of that kind holds across one of its tables')


def test_the_markdown_names_the_judge_and_the_questions_on_every_row():
    # this is a unit test
    """Comparability stops being a table heading and becomes a column, so the
    reason two rows are not comparable is on screen rather than inferred."""
    text = leaderboard.markdown(leaderboard.by_dataset([
        _board_row('a', 'diary-fa', 0.71, ('q1', 'q2'),
                   {'model': 'sonnet-4', 'provider': 'openrouter'})]))
    assert 'sonnet-4' in text
    assert '| judge |' in text and '| questions |' in text


# --- the settings reveal, the state column, and where a blank dataset lands --

def test_a_ledger_only_row_offers_no_stage_it_never_recorded(tmp_path,
                                                             monkeypatch):
    # this is an integration test
    """The reveal is documented as a longer form of the sentence that opened it,
    and the sentence correctly omits a stage that did not run. Emitting the
    step's empty shell had the panel draw RETRIEVAL and GENERATION headings with
    blank knobs under them for every index build on the board — inventing, one
    function after `pipeline_fragments` refuses to, exactly the two stages that
    refusal is about. `'none'` is a recorded value and is not a blank."""
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'empty')
    ledger.record({'id': 'job-1', 'kind': 'index',
                   'config': {'index': {'chunker': 'session',
                                        'embedder': 'token-hash'}},
                   'seconds': 3, 'result': {}}, 'done', path=db)
    config = leaderboard.board_rows(db_path=db)[0]['config']
    assert set(config) == {'index'}, (
        'a step the ledger recorded no knob for is absent, not an empty block')
    assert config['index'] == {'chunker': 'session', 'embedder': 'token-hash'}


def test_a_recorded_none_survives_into_the_reveal(tmp_path, monkeypatch):
    # this is an integration test
    """The other half of the same rule: `'none'` is a grader that ran and
    refused nothing, which is a measurement. Dropping it with the blanks would
    lose the distinction the sentence itself keeps."""
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'empty')
    ledger.record({'id': 'job-2', 'kind': 'retrieve',
                   'config': {'index': {'chunker': 'session',
                                        'embedder': 'token-hash'},
                              'retrieval': {'retriever': 'dense',
                                            'reranker': 'none',
                                            'grader': 'none'}},
                   'seconds': 2, 'result': {}}, 'done', path=db)
    config = leaderboard.board_rows(db_path=db)[0]['config']
    assert config['retrieval'] == {'retriever': 'dense', 'reranker': 'none',
                                   'grader': 'none'}
    assert 'generation' not in config


def test_a_row_agrees_with_the_table_it_is_filed_under(tmp_path, monkeypatch):
    # this is an integration test
    """`by_dataset` files a row with no dataset under the built-in corpus — no
    dataset predates the field, and that was the only corpus there was. The row
    used to answer the question for itself and serve a blank, so it sat on the
    built-in board carrying a cell that said it belonged to no corpus at all."""
    from raglab.evaluation import service_experiment_ledger as ledger
    db = tmp_path / 'l.db'
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'empty')
    ledger.record({'id': 'old-1', 'kind': 'run',
                   'config': {'index': {'chunker': 'session',
                                        'embedder': 'token-hash'}},
                   'seconds': 1, 'result': {}}, 'done', path=db)
    row, = leaderboard.board_rows(db_path=db)
    assert row['dataset'] == datasets.BUILTIN
    board, = leaderboard.by_dataset([row])
    assert board.dataset == datasets.BUILTIN
    assert all(r['dataset'] == board.dataset for r in board.rows), (
        'the cell and the table it is in have to name the same corpus')


def test_the_markdown_says_a_job_did_not_finish_and_why():
    # this is a unit test
    """The command line prints every job now, not only the evaluations a run
    file exists for — so a cancelled run and a failed retrieval print here, and
    without this column they read as ordinary unjudged experiments whose blank
    decision looked like a run nobody had judged yet. The page carries the same
    column with a '!' for the reason; a terminal has nowhere to put a '!', so
    the reason is in the cell."""
    rows = [_board_row('done-1', 'diary-fa', 0.71),
            dict(_board_row('gone-1', 'diary-fa', None), state='cancelled'),
            dict(_board_row('bad-1', 'diary-fa', None), state='error',
                 error='NameError: name | agent | is not defined')]
    text = leaderboard.markdown(leaderboard.by_dataset(rows))
    # Split on unescaped pipes only, which is what a markdown reader does — and
    # is the whole point of escaping them: an unescaped pipe inside a reason
    # ends its cell and shifts every column after it, so the row would then lie
    # about its own seconds and its own id.
    cells = lambda line: [c.strip() for c in re.split(r'(?<!\\)\|', line)]
    header = next(line for line in text.splitlines()
                  if line.startswith('| pipeline '))
    assert '| state |' in header
    columns = cells(header)
    at, id_at = columns.index('state'), columns.index('id')
    printed = [cells(line) for line in text.splitlines()
               if line.startswith('| ') and '`' in line]
    assert {len(row) for row in printed} == {len(columns)}, (
        'every printed row has the same number of columns as the heading')
    cell = {row[id_at].strip('`'): row[at] for row in printed}
    assert cell['done-1'] == 'done'
    assert cell['gone-1'] == '**cancelled**', (
        'a cancelled run has to be distinguishable from a finished one')
    assert 'NameError' in cell['bad-1']
    assert r'\|' in cell['bad-1'], 'the reason keeps its own pipes, escaped'


def test_the_markdown_joins_the_pipeline_the_way_the_page_does():
    # this is a unit test
    """On screen the boundary between two steps is carried by their colours; in
    a terminal it is carried by nothing, and a space had the whole sentence
    reading as one run-on token."""
    row = dict(_board_row('r1', 'diary-fa', 0.5),
               pipeline=[{'step': 'index', 'text': 'session·token-hash'},
                         {'step': 'generation', 'text': 'llm'}])
    text = leaderboard.markdown(leaderboard.by_dataset([row]))
    assert 'session·token-hash · llm' in text


def test_a_board_of_one_says_one_experiment():
    # this is a unit test
    """'1 experiments' in a heading printed by the thing that counted it."""
    text = leaderboard.markdown(leaderboard.by_dataset(
        [_board_row('r1', 'diary-fa', 0.5)]))
    assert '## diary-fa · 1 experiment' in text
    assert '1 experiments' not in text


def test_every_row_is_ordered_by_decision_not_by_dataset_block():
    # this is a unit test
    """The unfiltered view's own prose says the order it was served in is the
    ranking, and the sorter's third click restores exactly that order. Boards
    concatenated are ordered by dataset block instead, so the page would have
    been wrong about itself on the one view that mixes corpora."""
    boards = leaderboard.by_dataset([
        _board_row('mid', 'diary-fa', 0.50),
        _board_row('best', 'meetings-de', 0.90),
        _board_row('worst', 'diary-fa', 0.10),
        _board_row('none', 'meetings-de', None),
    ])
    assert [r['run_id'] for r in leaderboard.every_row(boards)] == [
        'best', 'mid', 'worst', 'none']
