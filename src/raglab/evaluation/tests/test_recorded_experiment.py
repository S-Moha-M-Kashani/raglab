"""One recorded experiment, read back — what a reader that is not the board
gets to see, and what it may never be handed.

Two functions across two modules, one subject: `leaderboard.experiment`
resolves an id through the board's own projection, and
`run_evaluation.question_rows` reads the per-question rows that only a run file
holds. They are tested together because a reader asking "what was this
experiment, and where did it fail" calls both, and because the second is the
only part of the answer the board has no column for."""
import json

import pytest

from raglab.evaluation import leaderboard
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import service_experiment_ledger as ledger

FOUR = {'faithfulness': 0.8, 'answer_relevancy': 0.7,
        'llm_context_precision_with_reference': 0.6, 'context_recall': 0.75}


def _rows(*specs):
    """Per-question rows in the shape a run file holds them."""
    out = []
    for question_id, recall, abstained in specs:
        out.append({'id': question_id, 'type': 'single-hop',
                    'difficulty': 'easy', 'answerable': True,
                    'retrieved_sessions': ['mini-01'], 'n_contexts': 3,
                    'context_chars': 900, 'abstained': abstained,
                    'recall': recall, 'precision': 0.5, 'mrr': 0.5,
                    'ndcg': 0.5, 'hit': 1.0 if recall else 0.0,
                    'quote_recall': recall, 'false_abstention': False,
                    'answer': 'the espresso machine', 'latency_ms': 40})
    return out


def _run_file(tmp_path, monkeypatch, run_id, *, dataset='smoke-mini',
              rows=(), metrics=None, decision=0.7125, stderr=0.01):
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    (tmp_path / f'{run_id}.json').write_text(json.dumps({
        'run_id': run_id, 'label': f'{run_id} label', 'dataset': dataset,
        'started_at': '2026-08-01 10:00:00', 'seconds': 12,
        'config': {'index': {'chunker': 'session', 'embedder': 'token-hash'},
                   'retrieval': {'retriever': 'bm25', 'k': 3},
                   'generation': {'answerer': 'llm', 'model': 'sonnet'}},
        'index': {'chunks': 15, 'embed_dim': 512, 'avg_chars': 243.7},
        'summary': {'overall': {'recall': 0.5}, 'n_questions': len(rows) or 2},
        'rows': list(rows),
        'ragas': {'metrics': FOUR if metrics is None else metrics,
                  'decision': decision,
                  'decision_spread': {'stderr': stderr, 'n': 2},
                  'judge': {'model': 'sonnet-4', 'provider': 'openrouter'}},
        'selection': {'question_ids': [r['id'] for r in rows], 'n': len(rows)},
        'notes': ['ragas sampled 2 of 6 questions'],
    }), encoding='utf-8')


def _ledger_row(db, run_id, *, dataset='smoke-mini', provider='openrouter',
                kind='run', detail_extra=None):
    result = {'run_id': run_id, 'dataset': dataset, 'label': f'{run_id} label',
              'started_at': '2026-08-01 10:00:00',
              'ragas': {'decision': 0.7125}}
    result.update(detail_extra or {})
    ledger.record({'id': run_id, 'kind': kind, 'seconds': 12,
                   'config': {'index': {'chunker': 'session',
                                        'dataset': dataset,
                                        'embedder': 'token-hash'},
                              'retrieval': {'retriever': 'bm25'},
                              'generation': {'answerer': 'llm'},
                              # Where `ledger.row_for` reads it from.
                              'provider': provider},
                   'result': result}, 'done', path=db)


def test_a_listing_row_carries_the_decision_beside_its_own_error(tmp_path,
                                                                monkeypatch):
    # this is an integration test
    """What any listing of recorded experiments needs off one row: the id to
    drill into, the decision beside its own error — never a decision alone —
    and the judge that produced it.

    Asserted on `board_rows` rather than on a compaction of it, because there is
    no longer a compaction: the reader that lists experiments for the widget is
    handed these rows as they are and renders them. Filtering them to one corpus
    is that reader's own business and is pinned in its own suite
    (`agents/widget/tests/test_experiment_tools.py`)."""
    db = tmp_path / 'l.db'
    _run_file(tmp_path, monkeypatch, 'r1')
    _ledger_row(db, 'r1')
    found = leaderboard.board_rows(db_path=db)
    assert [row['experiment_id'] for row in found] == ['r1']
    assert found[0]['decision'] == 0.7125
    assert found[0]['decision_stderr'] == 0.01
    assert found[0]['judge']['model'] == 'sonnet-4'
    assert 'bm25' in ' '.join(f['text'] for f in found[0]['pipeline'])


def test_a_digest_carries_the_four_judged_metrics_with_their_spread(
        tmp_path, monkeypatch):
    # this is an integration test
    """The four that decide, the mean and the error — the whole decision rule
    or none of it."""
    db = tmp_path / 'l.db'
    _run_file(tmp_path, monkeypatch, 'r1')
    _ledger_row(db, 'r1')
    found = leaderboard.experiment('r1', db_path=db)
    assert found['metrics'] == FOUR
    assert found['decision'] == 0.7125
    assert found['decision_stderr'] == 0.01
    assert found['judge']['model'] == 'sonnet-4'
    assert found['summary']['overall']['recall'] == 0.5
    assert found['index']['chunks'] == 15
    assert found['config']['retrieval']['retriever'] == 'bm25'


def test_a_digest_names_the_provider_that_produced_it(tmp_path, monkeypatch):
    # this is an integration test
    """`fake` is not a measurement, so which backend answered travels with the
    numbers rather than being left for the reader to assume."""
    db = tmp_path / 'l.db'
    _run_file(tmp_path, monkeypatch, 'r1')
    _ledger_row(db, 'r1', provider='fake')
    assert leaderboard.experiment('r1', db_path=db)['provider'] == 'fake'


def test_a_digest_of_an_unknown_id_is_none(tmp_path):
    # this is a unit test
    """Neither record holds it, so there is nothing to describe — and nothing
    is what comes back, never an empty experiment that reads as a real one."""
    assert leaderboard.experiment('nope', db_path=tmp_path / 'l.db') is None


def test_a_digest_carries_no_traces_no_summaries_and_no_chunk_text(
        tmp_path, monkeypatch):
    # this is an integration test
    """A ledger row's `detail` holds the whole job result — traces and
    hierarchy summaries included. Those are evidence for the Inspector, not
    for a helper's context window, and the digest is where they stop."""
    db = tmp_path / 'l.db'
    _run_file(tmp_path, monkeypatch, 'r1')
    _ledger_row(db, 'r1', detail_extra={
        'traces': [{'question': 'q', 'chunks': ['a very long chunk body']}],
        'summaries': [{'text': 'a hierarchy summary'}],
        'chunks_by_session': {'mini-01': ['chunk text']}})
    served = json.dumps(leaderboard.experiment('r1', db_path=db))
    assert 'a very long chunk body' not in served
    assert 'a hierarchy summary' not in served
    assert 'chunks_by_session' not in served
    # `index.chunks` is a count and stays: the guard is on evidence text, not
    # on the word.
    assert 'traces' not in served and 'summaries' not in served


def test_the_missed_questions_are_the_ones_evidence_was_not_fully_retrieved(
        tmp_path, monkeypatch):
    # this is an integration test
    """"Not fully retrieved within k" is recall below 1.0 — the question whose
    gold evidence the retriever only partly found is the one worth reading."""
    db = tmp_path / 'l.db'
    _run_file(tmp_path, monkeypatch, 'r1',
              rows=_rows(('mini-001', 1.0, False), ('mini-002', 0.5, False),
                         ('mini-003', 0.0, True)))
    _ledger_row(db, 'r1')
    found = evaluate.question_rows('r1', db_path=db)
    assert [row['id'] for row in found['rows']] == ['mini-002', 'mini-003']
    assert found['n_questions'] == 3
    assert found['n_matched'] == 2


def test_a_missed_question_is_named_by_its_own_text(tmp_path, monkeypatch):
    # this is an integration test
    """A run file's rows carry the question id and no text, so an id alone is
    all a reader would get. The dataset's ground truth is joined in, which is
    what makes 'what can get better' answerable at all.

    The row id is smoke-mini's real `groundtruth_question_id` (1 — "What
    broke in the kitchen?"), not a fabricated string, because the join in
    `run_evaluation._questions` is now by that integer id (D4); a made-up id
    like the old 'mini-001' would simply fail to join under the new schema."""
    db = tmp_path / 'l.db'
    _run_file(tmp_path, monkeypatch, 'r1', dataset='smoke-mini',
              rows=_rows((1, 0.0, False)))
    _ledger_row(db, 'r1')
    row = evaluate.question_rows('r1', db_path=db)['rows'][0]
    assert row['question'] == 'What broke in the kitchen?'
    assert row['expected_sessions'] == ['1']
    assert row['difficulty'] == 'easy'


def test_the_question_rows_say_how_many_they_left_out(tmp_path, monkeypatch):
    # this is an integration test
    """A capped listing that did not say it was capped would read as the whole
    failure set, and a recommendation drawn from it would be about a sample the
    reader thought was a census."""
    db = tmp_path / 'l.db'
    _run_file(tmp_path, monkeypatch, 'r1',
              rows=_rows(*[(f'mini-{n:03d}', 0.0, False) for n in range(1, 8)]))
    _ledger_row(db, 'r1')
    found = evaluate.question_rows('r1', limit=3, db_path=db)
    assert len(found['rows']) == 3
    assert found['n_matched'] == 7


def test_an_experiment_with_no_run_file_has_no_question_rows(tmp_path,
                                                             monkeypatch):
    # this is an integration test
    """Per-question rows live in the run file only. A ledger-only experiment —
    every index build, every evaluation older than the ledger — says so rather
    than reporting zero failures, which would read as a clean sweep."""
    db = tmp_path / 'l.db'
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'empty')
    _ledger_row(db, 'build-1', kind='index')
    found = evaluate.question_rows('build-1', db_path=db)
    assert found['rows'] == []
    assert 'no run file' in found['reason']


def test_the_digest_and_the_board_describe_one_experiment_identically(
        tmp_path, monkeypatch):
    # this is an integration test
    """The invariant the two readers exist to share: the ledger-versus-run
    precedence is one rule, so every fact both of them carry must agree.

    Two records describe one experiment and neither is sufficient, so each
    reader has to decide which wins where they overlap. When that decision was
    written twice, agreement was a coincidence maintained by hand — a run file
    key renamed on one side and the board and the widget would have quoted
    different numbers for the same experiment id, each convinced it had read the
    record. One projection makes the agreement structural."""
    db = tmp_path / 'l.db'
    _run_file(tmp_path, monkeypatch, 'r1', rows=_rows(('q1', 1.0, False)))
    _ledger_row(db, 'r1')
    board = leaderboard.board_rows(db_path=db)[0]
    found = leaderboard.experiment('r1', db_path=db)
    shared = set(board) & set(found)
    # Not a subset check: the point is that nothing they both name may differ.
    assert {k: board[k] for k in shared} == {k: found[k] for k in shared}
    # And the overlap is the whole projection, not two keys that happen to match.
    assert shared >= {'experiment_id', 'kind', 'state', 'label', 'dataset',
                      'started_at', 'seconds', 'provider', 'n_questions',
                      'decision', 'decision_stderr', 'metrics', 'judge',
                      'config', 'source'}
