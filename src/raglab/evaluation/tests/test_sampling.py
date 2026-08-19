"""The 49-question sample, balanced across difficulty — so a run's
questions are a stated, deterministic, comparable choice."""
import pytest

from raglab.configuration import lab_config as config
from raglab.evaluation import run_evaluation as evaluate
from raglab.agents.extra_tools import sweep
from raglab.configuration.lab_config import (
    GenerationConfig,
    IndexConfig,
    LabConfig)

from raglab.conftest import LAB_SETTINGS


# --- the balanced sample, at every limit the lab actually uses --------------
# The four deciding metrics are means over questions, so which questions a run
# scored is part of the measurement. The natural distribution is 29 easy / 57
# medium / 26 hard, and a plain stride hands medium about half of any sample —
# which measures the medium pipeline and reports it as the pipeline.

@pytest.mark.parametrize('limit,expected', [
    # 49 does not divide by three; the remainder goes to the earlier bands in
    # DIFFICULTIES order, so the split is 17/16/16 and not "whatever came out".
    (49, {'easy': 17, 'medium': 16, 'hard': 16}),
    # 51 divides evenly, so the split is exactly equal rather than merely
    # as-equal-as-possible.
    (51, {'easy': 17, 'medium': 17, 'hard': 17}),
    # 30 is the sweep's own sample size; it also divides by three.
    (30, {'easy': 10, 'medium': 10, 'hard': 10}),
])
def test_a_balanced_sample_splits_the_difficulty_bands_as_evenly_as_it_can(
        ground_truth, limit, expected):
    # this is a unit test
    picked = evaluate.select_questions(ground_truth, limit=limit,
                                       balance='difficulty')
    assert len(picked) == limit
    counts = {name: sum(1 for q in picked if q['difficulty'] == name)
              for name in config.DIFFICULTIES}
    assert counts == expected, counts


def test_a_balanced_sample_is_deterministic_and_keeps_fixture_order(
        ground_truth):
    # this is a unit test
    """Two candidates are only comparable if they scored the same questions in
    the same order, so selection has to be both deterministic and undiffable
    between two runs of the same config."""
    first = evaluate.select_questions(ground_truth, limit=49,
                                      balance='difficulty')
    second = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    assert [q['id'] for q in first] == [q['id'] for q in second]
    order = [q['id'] for q in ground_truth['questions']]
    picked_ids = {q['id'] for q in first}
    assert [q['id'] for q in first] == [i for i in order if i in picked_ids]


def test_a_balanced_sample_still_spreads_across_the_question_types(ground_truth):
    # this is a unit test
    """Balancing difficulty must not cost type coverage — habit questions are
    last in the file and were the type a bad stride used to lose entirely."""
    picked = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    types = {q['type'] for q in picked}
    assert len(types) >= 9, types
    assert 'habit' in types


def test_a_band_too_small_for_its_share_does_not_shrink_the_sample():
    # this is a unit test
    """A run asked for N questions must produce N whenever the set holds that
    many; what a small band cannot supply is offered to the others."""
    questions = ([{'id': f'e{i}', 'difficulty': 'easy', 'type': 't'}
                  for i in range(2)]
                 + [{'id': f'm{i}', 'difficulty': 'medium', 'type': 't'}
                    for i in range(20)]
                 + [{'id': f'h{i}', 'difficulty': 'hard', 'type': 't'}
                    for i in range(20)])
    picked = evaluate.select_questions({'questions': questions}, limit=12,
                                       balance='difficulty')
    assert len(picked) == 12
    assert sum(1 for q in picked if q['difficulty'] == 'easy') == 2


def test_the_default_sampling_rule_is_unchanged(ground_truth):
    # this is a unit test
    """The runs already in `.runs/` were strided. Changing the default
    underneath the leaderboard would make those rows incomparable rather than
    merely old — so 'stride' stays the default and the sweep opts in."""
    strided = evaluate.select_questions(ground_truth, limit=24)
    explicit = evaluate.select_questions(ground_truth, limit=24,
                                         balance='stride')
    assert [q['id'] for q in strided] == [q['id'] for q in explicit]


# --- stride, at the two limits that exercise its coverage -------------------
# Moved in from test_raglab.py (test-plan step 9): striding with
# `questions[::step][:limit]` silently drops a tail whenever the count is not
# a multiple of the limit — and the set is grouped by type with the newest
# (habit) appended last, so the dropped tail is always the most recently
# added question type. Both cases check the same stride mechanics (full
# length, no duplicates, starts at the fixture's first id, reaches within one
# stride of the end); each also checks the type coverage a naive stride used
# to lose at that limit.
@pytest.mark.parametrize('limit,required_type,min_types', [
    (10, None, 2),
    (20, 'habit', None),
])
def test_a_stride_reaches_the_end_and_covers_the_question_types(
        ground_truth, limit, required_type, min_types):
    # this is a unit test
    questions = ground_truth['questions']
    picked = evaluate.select_questions(ground_truth, limit=limit)
    assert len(picked) == limit
    ids = [q['id'] for q in picked]
    assert len(set(ids)) == limit, f'{limit} produced duplicates'
    assert ids[0] == questions[0]['id'], limit
    # The last pick is within one stride of the end, not 16 short of it.
    stride = -(-len(questions) // limit)          # ceil
    assert questions.index(picked[-1]) >= len(questions) - stride, limit
    if required_type is not None:
        assert any(q['type'] == required_type for q in picked)
    if min_types is not None:
        assert len({q['type'] for q in picked}) >= min_types, picked


@pytest.mark.parametrize('kwargs', [
    {'limit': 10},
    {},   # checked after the early return: a run with no limit must not let a
          # typo pass silently, which is exactly the run where it used to.
])
def test_an_unknown_balance_raises_rather_than_silently_striding(
        ground_truth, kwargs):
    # this is a unit test
    with pytest.raises(ValueError, match='balance'):
        evaluate.select_questions(ground_truth, balance='difficlty', **kwargs)


def test_a_run_saves_the_questions_it_was_measured_on(registry, ground_truth):
    # this is an integration test
    """Neither the config nor the metric means say which questions produced them,
    so the ids travel with the row. Losing them is how two rows get compared
    across two different samples with nothing to reveal it."""
    cfg = LabConfig(index=IndexConfig(chunker='semantic-drift',
                                      embedder='char-hash', contextual=True),
                    generation=GenerationConfig(answerer='none'))
    result = evaluate.run_eval(registry, ground_truth, cfg, LAB_SETTINGS,
                               limit=9, balance='difficulty', ragas_mode='off')
    selection = result.selection
    assert selection['balance'] == 'difficulty' and selection['limit'] == 9
    assert selection['n'] == 9
    assert len(selection['question_ids']) == 9
    assert selection['by_difficulty'] == {'easy': 3, 'medium': 3, 'hard': 3}
    assert result.as_dict()['selection'] == selection
    # And on the leaderboard row — minus the ids, which would swamp it.
    assert result.brief()['selection']['balance'] == 'difficulty'
    assert 'question_ids' not in result.brief()['selection']


def test_the_sweep_measures_every_candidate_on_the_same_balanced_30():
    # this is a unit test
    """The sample is a property of the sweep, not of the invocation: a row
    measured on a different sample is a different measurement."""
    assert sweep.SWEEP_LIMIT == 30
    assert sweep.SWEEP_BALANCE == 'difficulty'
    assert sweep.SWEEP_BALANCE in config.BALANCES
