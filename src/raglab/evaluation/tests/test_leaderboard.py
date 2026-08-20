"""The leaderboard, and what it refuses to rank together — grouping by
question set and judge before ranking anything."""
import json

import pytest

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


@pytest.mark.parametrize('build_rows, check_markdown', [
    (_unrecorded_sample_rows, True),
    (_no_measured_error_rows, False),
], ids=['sample-not-recorded', 'no-measured-error'])
def test_a_group_that_cannot_be_ranked_reports_unknown(build_rows, check_markdown):
    # this is a unit test
    """Two different reasons a group refuses a winner. Two runs of 24
    questions apiece may still be two *different* 24 — nothing on those rows
    says which, so even with errors measured the comparison is not
    established (`selection` predates `RunResult`, so equal counts are not a
    shared sample). Or the rows recorded a sample but no error, and `± 0` on
    the oldest rows would present them as the most precise. Both must read as
    'unknown' rather than manufacturing a decision, and the unrecorded case
    additionally must carry no rank numbers, since a numbered row is a rank
    claim this data cannot support."""
    found, = leaderboard.group(build_rows())
    assert leaderboard.verdict(found) == 'unknown'
    if check_markdown:
        text = leaderboard.markdown([found])
        assert 'not recorded' in text
        # And no rank numbers, or the table contradicts the sentence above it.
        assert '| 1 |' not in text, text


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
    judge = {'model': '', 'provider': ''}
    unjudged, = leaderboard.group([_row('r1', 'retrieval only', None,
                                        ['q1'], judge)])
    text = leaderboard.markdown([unjudged])
    assert 'judged by nothing judged' not in text, text
    assert 'judged by no judge —' in text


def test_the_markdown_names_the_sample_and_the_judge_on_every_group():
    # this is a unit test
    judge = {'model': 'gemma4:e2b', 'provider': 'ollama'}
    text = leaderboard.markdown(leaderboard.group([
        _row('r1', 'A baseline', 0.61, ['q1', 'q2'], judge, stderr=0.02)]))
    assert 'gemma4:e2b' in text and 'ollama' in text
    assert '2 questions' in text
    assert '0.610' in text and '0.020' in text
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
    page, for the reason `as_dict` exists: two surfaces that each derived the
    sentence could describe one row two ways."""
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
    """`grader` and `reranker` both default to the literal 'none' in
    lab_config, and a fragment reading 'rrf·none·none' spends the sentence's
    width saying which stages did nothing."""
    fragments = leaderboard.pipeline_fragments(
        {'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'none',
                       'grader': 'none'}})
    assert fragments == [{'step': 'retrieval', 'text': 'hybrid-rrf'}]
