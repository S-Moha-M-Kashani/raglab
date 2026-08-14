"""Progress reporting — a run that reports nothing is indistinguishable
from a hang, so every phase says where it is."""
import threading
import time

from raglab import config, evaluate, explain, ragas_eval, sweep
from raglab.config import GenerationConfig, IndexConfig, LabConfig

from conftest import LAB_SETTINGS


# --- progress: a run that reports nothing is indistinguishable from a hang ---
# With a local judge the judged phase is hours, not minutes, so every phase has
# to say where it is. The callback carries a human detail beside the fraction
# because "0.92" for two hours tells the reader nothing about what is happening.

PROGRESS_CFG = LabConfig(index=IndexConfig(chunker='message', embedder='char-hash'),
                         generation=GenerationConfig(answerer='extractive'),
                         label='progress')


def test_progress_reports_which_question_it_is_on(registry, ground_truth,
                                                  tmp_path, monkeypatch):
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


def test_a_two_argument_progress_callback_still_works(registry, ground_truth,
                                                      tmp_path, monkeypatch):
    """The detail is additive. The panel's reporter predates it, and a run must
    not fail because its caller does not want the third argument."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    seen = []
    evaluate.run_eval(registry, ground_truth, PROGRESS_CFG, LAB_SETTINGS,
                      limit=2, ragas_mode='off',
                      progress=lambda stage, fraction: seen.append(stage))
    assert 'scoring' in seen and 'done' in seen


def test_the_judged_phase_reports_calls_as_they_land():
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
    """The panel polls a job dict, so the detail has to be a field on it — a
    progress line only the terminal sees leaves the two UIs looking hung."""
    from raglab.server import Jobs
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
    """Stopping a run must prevent its next unit of work, not just its polling."""
    from raglab.server import Jobs
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


def test_the_terminal_bar_says_stage_fraction_elapsed_and_detail():
    line = sweep.bar('Stage F', 'scoring', 0.5, 'question 16/30 · hard',
                     time.time() - 63)
    assert line.startswith('\r'), 'the line is rewritten in place, not appended'
    assert 'Stage F' in line
    assert '50.0%' in line
    assert '1m03s' in line, line          # elapsed, because a fraction alone
    assert 'question 16/30 · hard' in line   # cannot tell slow from stuck
    filled = line.count('█')
    assert filled == sweep.BAR_WIDTH // 2, filled


def test_a_shorter_detail_cannot_leave_the_tail_of_a_longer_one_behind():
    """Without the padding a redraw leaves stale characters on the line, which
    reads as a stale *number* rather than as a drawing artefact."""
    written = []
    report = sweep.live('Stage A', time.time(),
                        stream=type('S', (), {'write': written.append,
                                              'flush': lambda self: None})())
    report('ragas', 0.94, 'judge call 137 of ~420')
    report('done', 1.0, '')
    assert len(written[0]) == len(written[1])
    assert '137' not in written[1]


def test_the_expected_judge_call_count_scales_with_k():
    """Context precision asks one verdict per retrieved chunk, so k is what
    drives the bill — the estimate has to know that or it is decoration."""
    at_k5 = ragas_eval.expected_judge_calls(n_samples=10, k=5)
    at_k12 = ragas_eval.expected_judge_calls(n_samples=10, k=12)
    assert at_k12 > at_k5
    assert at_k12 - at_k5 == 10 * 7, (at_k5, at_k12)


def test_the_balance_control_is_explained_like_every_other_knob():
    """`explain.missing()` covers config fields; a run-level control has to be
    added to the same registry by hand or it reaches the panel unexplained."""
    assert 'run.balance' in explain.topics()
    assert 'run.difficulty' in explain.topics()
