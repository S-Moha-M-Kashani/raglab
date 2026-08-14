"""The 49-question sample, balanced across difficulty — so a run's
questions are a stated, deterministic, comparable choice."""
import pytest

from raglab import config, evaluate
from raglab.llm_tools import sweep
from raglab.config import GenerationConfig, IndexConfig, LabConfig

from conftest import LAB_SETTINGS


# --- the 49-question sample, balanced across difficulty --------------------
# The four deciding metrics are means over questions, so which questions a run
# scored is part of the measurement. The natural distribution is 29 easy / 57
# medium / 26 hard, and a plain stride hands medium about half of any sample —
# which measures the medium pipeline and reports it as the pipeline.

def test_a_balanced_sample_splits_the_difficulty_bands_as_evenly_as_it_can(
        ground_truth):
    picked = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    assert len(picked) == 49
    counts = {name: sum(1 for q in picked if q['difficulty'] == name)
              for name in config.DIFFICULTIES}
    # 49 does not divide by three; the remainder goes to the earlier bands in
    # DIFFICULTIES order, so the split is 17/16/16 and not "whatever came out".
    assert counts == {'easy': 17, 'medium': 16, 'hard': 16}, counts


def test_a_balanced_sample_that_divides_evenly_is_exactly_equal(ground_truth):
    picked = evaluate.select_questions(ground_truth, limit=51,
                                       balance='difficulty')
    counts = {name: sum(1 for q in picked if q['difficulty'] == name)
              for name in config.DIFFICULTIES}
    assert counts == {'easy': 17, 'medium': 17, 'hard': 17}, counts


def test_a_balanced_sample_is_the_same_questions_every_time(ground_truth):
    """Two candidates are only comparable if they scored the same questions, so
    the selection has to be deterministic rather than merely proportionate."""
    first = evaluate.select_questions(ground_truth, limit=49,
                                      balance='difficulty')
    second = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    assert [q['id'] for q in first] == [q['id'] for q in second]


def test_a_balanced_sample_still_spreads_across_the_question_types(ground_truth):
    """Balancing difficulty must not cost type coverage — habit questions are
    last in the file and were the type a bad stride used to lose entirely."""
    picked = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    types = {q['type'] for q in picked}
    assert len(types) >= 9, types
    assert 'habit' in types


def test_a_balanced_sample_keeps_the_fixture_order(ground_truth):
    """Band-by-band output would make two runs undiffable line by line for no
    reason."""
    picked = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    order = [q['id'] for q in ground_truth['questions']]
    assert [q['id'] for q in picked] == [i for i in order
                                        if i in {q['id'] for q in picked}]


def test_a_band_too_small_for_its_share_does_not_shrink_the_sample():
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
    """The twelve runs already in `.runs/` were strided. Changing the default
    underneath the leaderboard would make those rows incomparable rather than
    merely old — so 'stride' stays the default and the sweep opts in."""
    strided = evaluate.select_questions(ground_truth, limit=24)
    explicit = evaluate.select_questions(ground_truth, limit=24,
                                         balance='stride')
    assert [q['id'] for q in strided] == [q['id'] for q in explicit]


def test_an_unknown_balance_raises_rather_than_silently_striding(ground_truth):
    with pytest.raises(ValueError, match='balance'):
        evaluate.select_questions(ground_truth, limit=10, balance='difficlty')


def test_an_unknown_balance_raises_even_when_there_is_no_limit(ground_truth):
    """Checked after the early return, the validation passed silently on any run
    without a limit — so a typo would only raise on the runs where it happened to
    change something, which is the worst possible place to find it."""
    with pytest.raises(ValueError, match='balance'):
        evaluate.select_questions(ground_truth, balance='difficlty')


def test_a_run_saves_the_questions_it_was_measured_on(registry, ground_truth):
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
    """The sample is a property of the sweep, not of the invocation: a row
    measured on a different sample is a different measurement."""
    assert sweep.SWEEP_LIMIT == 30
    assert sweep.SWEEP_BALANCE == 'difficulty'
    assert sweep.SWEEP_BALANCE in config.BALANCES


def test_the_sweep_sample_is_exactly_ten_of_each_band(ground_truth):
    """30 divides by three, so this sample needs no remainder rule at all — the
    bands are equal rather than merely as-equal-as-possible."""
    picked = evaluate.select_questions(ground_truth, limit=sweep.SWEEP_LIMIT,
                                       balance=sweep.SWEEP_BALANCE)
    counts = {name: sum(1 for q in picked if q['difficulty'] == name)
              for name in config.DIFFICULTIES}
    assert counts == {'easy': 10, 'medium': 10, 'hard': 10}, counts
