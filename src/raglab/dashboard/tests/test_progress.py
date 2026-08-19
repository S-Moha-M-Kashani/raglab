"""Progress reporting — a run that reports nothing is indistinguishable
from a hang, so every phase says where it is."""
import threading
import time

import pytest

from raglab.configuration import lab_config as config
from raglab.evaluation import run_evaluation as evaluate
from raglab.configuration import explainer_assembly as explain
from raglab.evaluation import ragas_judged_metrics as ragas_eval
from raglab.agents.extra_tools import sweep
from raglab.configuration.lab_config import (
    GenerationConfig,
    IndexConfig,
    LabConfig)

from raglab.conftest import LAB_SETTINGS


# --- progress: a run that reports nothing is indistinguishable from a hang ---
# With a local judge the judged phase is hours, not minutes, so every phase has
# to say where it is. The callback carries a human detail beside the fraction
# because "0.92" for two hours tells the reader nothing about what is happening.

PROGRESS_CFG = LabConfig(index=IndexConfig(chunker='message', embedder='char-hash'),
                         generation=GenerationConfig(answerer='extractive'),
                         label='progress')


def test_progress_reports_which_question_it_is_on(registry, ground_truth,
                                                  tmp_path, monkeypatch):
    # this is an integration test
    """The one live `run_eval` in this file — a smoke run standing in for the
    two that used to be here, since the other, `_reporter`'s own arity
    adaptation, is now pinned directly against the function below rather
    than by running a whole evaluation twice to observe it indirectly."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    seen = []
    evaluate.run_eval(registry, ground_truth, PROGRESS_CFG, LAB_SETTINGS,
                      limit=4, balance='difficulty', ragas_mode='off',
                      progress=lambda stage, fraction, detail='': seen.append(
                          (stage, round(fraction, 3), detail)))
    scoring = [row for row in seen if row[0] == 'scoring']
    assert len(scoring) == 4, seen
    # The count is the point: "question 3/4" is checkable against the sample the
    # row itself records, where a bare fraction is not.
    assert scoring[2][2].startswith('question 3/4'), scoring
    assert scoring[-1][2].startswith('question 4/4'), scoring
    # And the band, because a slow phase on hard questions is a different fact
    # from a slow phase overall.
    assert any(band in scoring[0][2] for band in config.DIFFICULTIES), scoring
    # A run reports a terminal ('done', 1.0) too — without it, a poller has
    # no way to tell "still running" from "finished and stopped reporting".
    assert seen[-1][:2] == ('done', 1.0), seen


def test_the_reporter_inspects_a_callbacks_arity_once_rather_than_per_call():
    # this is a unit test
    """The detail is additive: the panel's reporter predates it, and a run
    must not fail because its caller only wants two arguments — checked
    directly against `evaluate._reporter` rather than by running a whole
    evaluation twice to observe the same adaptation indirectly. The arity
    is inspected once (`inspect.signature`), not by trying a 3-argument call
    and catching the `TypeError` per call — the second approach would also
    swallow a real bug raised *inside* a 3-argument callback, mistaking it
    for an arity mismatch."""
    seen_three = []
    report3 = evaluate._reporter(
        lambda stage, fraction, detail='': seen_three.append((stage, fraction, detail)))
    report3('scoring', 0.5, 'question 3/4 · hard')
    assert seen_three == [('scoring', 0.5, 'question 3/4 · hard')]

    seen_two = []
    report2 = evaluate._reporter(lambda stage, fraction: seen_two.append((stage, fraction)))
    report2('scoring', 0.5, 'question 3/4 · hard')
    assert seen_two == [('scoring', 0.5)], 'a 2-arg callback must never see the detail'

    assert evaluate._reporter(None)('scoring', 0.5, 'ignored') is None

    # The mechanism, not just the outcome: a 3-arg-compatible callback that
    # raises for a real reason must not be read as "wrong arity, retry with
    # two arguments" — a per-call `try/except TypeError` would do exactly
    # that, and would report a different (arity) error here instead.
    def genuinely_buggy(stage, fraction, detail):
        raise TypeError('boom from inside the callback, not an arity mismatch')

    with pytest.raises(TypeError, match='boom from inside the callback'):
        evaluate._reporter(genuinely_buggy)('scoring', 0.5, 'question 3/4 · hard')


def test_the_judged_phase_reports_calls_as_they_land():
    # this is a unit test
    """The judged phase is the whole wall clock on a local judge. RAGAS scores a
    batch, so without a per-call hook the bar sits at one number for hours."""
    watch = ragas_eval.JudgeWatch(total=6)
    seen = []
    watch.progress = lambda stage, fraction, detail='': seen.append(detail)
    watch.on_llm_end(None)
    watch.on_llm_end(None)
    assert 'judge call 2' in seen[-1]
    assert '~6' in seen[-1], 'the estimate is marked as one, not stated as fact'
    # A judge that makes more calls than estimated must not report >100%.
    for _ in range(20):
        watch.on_llm_end(None)
    assert watch.fraction() <= 1.0


def test_the_job_carries_the_detail_to_whoever_is_polling():
    # this is an integration test
    """The panel polls a job dict, so the detail has to be a field on it — a
    progress line only the terminal sees leaves the two UIs looking hung."""
    from raglab.dashboard.panel_server import Jobs
    jobs = Jobs()
    captured = {}

    def target(report):
        report('scoring', 0.5, 'question 16/30 · hard')
        captured['snapshot'] = dict(jobs.jobs[jobs.current])
        return {'ok': True}

    job_id = jobs.start('run', target)
    while jobs.jobs[job_id]['state'] == 'running':
        time.sleep(0.01)
    assert captured['snapshot']['detail'] == 'question 16/30 · hard'
    assert captured['snapshot']['stage'] == 'scoring'
    # Present from the start, so a poll landing before the first report reads a
    # blank rather than undefined.
    assert 'detail' in jobs.jobs[job_id]


def test_a_running_job_can_be_cancelled_before_its_next_call():
    # this is an integration test
    """Stopping a run must prevent its next unit of work, not just its polling."""
    from raglab.dashboard.panel_server import Jobs
    jobs = Jobs()
    started = threading.Event()

    def target(report, cancelled):
        started.set()
        while not cancelled():
            time.sleep(0.001)
        # The normal checkpoint used by indexing and evaluation raises the
        # cooperative cancellation exception as soon as the active call ends.
        report('scoring', 0.5, 'would have made another model call')

    job_id = jobs.start('run', target)
    assert started.wait(timeout=1)
    stopped = jobs.cancel(job_id)
    assert stopped['state'] == 'cancelling'
    for _ in range(100):
        if jobs.get(job_id)['state'] == 'cancelled':
            break
        time.sleep(0.01)
    job = jobs.get(job_id)
    assert job['state'] == 'cancelled'
    assert job['cancel_requested'] is True
    assert '_cancel' not in job


def test_the_terminal_bar_renders_its_fields_and_never_leaves_a_stale_tail():
    # this is a unit test
    """The two bar-rendering claims side by side: `sweep.bar` says stage,
    fraction, elapsed and detail in one rewritten line, and `sweep.live`'s
    redraws pad a shorter detail so a stale tail of a longer one cannot
    survive — without that padding a redraw would leave characters behind
    that read as a stale *number* rather than as a drawing artefact."""
    line = sweep.bar('Stage F', 'scoring', 0.5, 'question 16/30 · hard',
                     time.time() - 63)
    assert line.startswith('\r'), 'the line is rewritten in place, not appended'
    assert 'Stage F' in line
    assert '50.0%' in line
    assert '1m03s' in line, line          # elapsed, because a fraction alone
    assert 'question 16/30 · hard' in line   # cannot tell slow from stuck
    filled = line.count('█')
    assert filled == sweep.BAR_WIDTH // 2, filled

    written = []
    report = sweep.live('Stage A', time.time(),
                        stream=type('S', (), {'write': written.append,
                                              'flush': lambda self: None})())
    report('ragas', 0.94, 'judge call 137 of ~420')
    report('done', 1.0, '')
    assert len(written[0]) == len(written[1])
    assert '137' not in written[1]


def test_the_expected_judge_call_count_scales_with_k():
    # this is a unit test
    """Context precision asks one verdict per retrieved chunk, so k is what
    drives the bill — the estimate has to know that or it is decoration."""
    at_k5 = ragas_eval.expected_judge_calls(n_samples=10, k=5)
    at_k12 = ragas_eval.expected_judge_calls(n_samples=10, k=12)
    assert at_k12 > at_k5
    assert at_k12 - at_k5 == 10 * 7, (at_k5, at_k12)


def test_the_balance_control_is_explained_like_every_other_knob():
    # this is a convention test
    """`explain.missing()` covers config fields; a run-level control has to be
    added to the same registry by hand or it reaches the panel unexplained."""
    assert 'run.balance' in explain.topics()
    assert 'run.difficulty' in explain.topics()
