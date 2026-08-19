"""The experiment ledger (raglab.db) — one row per job, recording every
build, retrieval and evaluation the lab finishes.

The full HTTP journey — panel payload through a real index job and a real
evaluation job to a ledger row the leaderboard can group — is
`tests/test_e2e.py`'s one job, not repeated here. What stays here drives
`raglab.server.Jobs` directly, no HTTP, because the claims below are about
the job runner and the ledger module, not about the routes that call them.
"""
import sqlite3
import time
from pathlib import Path

from raglab.configuration import lab_config as config
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.configuration.lab_config import (
    GenerationConfig,
    IndexConfig,
    LabConfig,
    RetrievalConfig)
from raglab.dashboard.panel_server import Jobs, _with_backend

from raglab.conftest import LAB_SETTINGS


def _run_to_terminal(jobs: Jobs, job_id: str, timeout: float = 30.0) -> dict:
    """The direct-call equivalent of `conftest._finished`, polling `Jobs`'s own
    table instead of an HTTP job-status route."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job['state'] not in ('running', 'cancelling'):
            return job
        time.sleep(0.005)
    raise AssertionError(f'job {job_id} still running after {timeout}s')


def test_jobs_run_writes_the_ledger_row_before_the_job_goes_terminal_per_kind(
        tmp_path, monkeypatch):
    # this is an integration test
    """Drives `Jobs.run` directly with a stub work function per kind — no
    HTTP, no real index build, no real evaluation — because the claims are
    about the job runner and the ledger, not about what a build or an
    evaluation computes. Three things `Jobs.run` must get right, each
    checked below: the row is written *before* `state` flips to a terminal
    value (a poller that sees 'done' must never look for the row and miss
    it); a build's row carries its index config and nothing else, even when
    the job config it was handed carries a real retrieval/generation block
    (a build never reads them, so recording them would put a reranker on a
    row that never retrieved anything); and the resolved backend, not the
    request, ends up in `provider`.
    """
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))

    # Captured at the moment `Jobs.run` calls `record` — which the source
    # (`panel_server.py`'s `Jobs.start.run`) does before `job['state'] = outcome`,
    # never after. If that ordering ever regressed, this would still read
    # 'done' rather than 'running' and the test would catch it.
    observed_state_at_write: dict[str, str] = {}

    def record(job: dict, state: str) -> None:
        observed_state_at_write[job['kind']] = job['state']
        ledger.record(job, state)

    jobs = Jobs(record=record)

    # A build's config carries a real retrieval/reranker/grader/generation
    # block, deliberately — so `retriever == ''` etc. below is proof the row
    # blanks them because it is a build, not proof they were merely absent.
    # Hand-written, not through `_with_backend`, matching the real index
    # route (`panel_server.py`'s `/api/indexes` passes `config=cfg.to_dict()` with
    # no backend attached at all — a build calls no chat model).
    index_job = jobs.start(
        'index', lambda report: {'chunks': 5, 'leaves': 5},
        config={'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
                'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'lexical',
                              'grader': 'llm'},
                'generation': {'answerer': 'llm'}})
    assert _run_to_terminal(jobs, index_job)['state'] == 'done'

    # The two pipeline-owning kinds' configs are run through `_with_backend`,
    # the real route helper, rather than a hand-written `'provider': 'fake'`
    # — so `provider == 'fake'` below exercises the actual resolution
    # (`run_settings.provider`), not just a round trip of a literal.
    retrieve_cfg = LabConfig(index=IndexConfig(chunker='session', embedder='ascii-hash'),
                             retrieval=RetrievalConfig(retriever='hybrid-rrf'))
    retrieve_job = jobs.start(
        'retrieve', lambda report: {'selection': {'n': 2},
                                    'questions': [{'trace': {'candidates': [1]}}]},
        config=_with_backend(retrieve_cfg, LAB_SETTINGS))
    assert _run_to_terminal(jobs, retrieve_job)['state'] == 'done'

    run_cfg = LabConfig(index=IndexConfig(chunker='session', embedder='ascii-hash'),
                        retrieval=RetrievalConfig(retriever='hybrid-rrf'),
                        generation=GenerationConfig(answerer='extractive'),
                        label='the ledger')

    def run_work(report):
        time.sleep(0.01)   # so `seconds` below is genuinely > 0, not rounded to it
        return {'run_id': '20260101-000000-abcdef', 'summary': {'n_questions': 3},
                'rows': [{'id': 'q1'}, {'id': 'q2'}, {'id': 'q3'}],
                'selection': {'n': 3}, 'chunks_by_session': [{'session_id': 's1'}]}

    run_job = jobs.start('run', run_work, config=_with_backend(run_cfg, LAB_SETTINGS))
    assert _run_to_terminal(jobs, run_job)['state'] == 'done'

    # 1. Written before the job goes terminal, for every kind — not just
    # observed once and generalised.
    assert observed_state_at_write == {'index': 'running', 'retrieve': 'running',
                                       'run': 'running'}

    rows = {row['kind']: row for row in ledger.experiments()}
    assert set(rows) == {'index', 'retrieve', 'run'}
    # Newest-first ordering the panel's table relies on.
    assert [row['kind'] for row in ledger.experiments()] == ['run', 'retrieve', 'index']

    # 2. A build's row stops at its index config: the retrieval/generation
    # block on its job config is real, and still blanked. This is a fact
    # about `ledger.row_for` alone — it says nothing about whether the real
    # `/api/indexes` route ever attaches a provider to a build's config in
    # the first place (it does not: `panel_server.py`'s index route passes
    # `config=cfg.to_dict()` with no `_with_backend` call at all). That
    # route-level fact is pinned by `tests/test_e2e.py`'s
    # `build_row['provider'] == ''`, not here.
    build = rows['index']
    assert build['chunker'] == 'session' and build['embedder'] == 'ascii-hash'
    assert build['retriever'] == '' and build['reranker'] == ''
    assert build['grader'] == '' and build['answerer'] == ''
    assert build['provider'] == '', 'no chat model is involved in chunking'
    assert build['n_questions'] == 0 and build['decision'] is None

    # 3. The two that did retrieve say so, and the resolved backend — not
    # the request — lands in `provider`, genuinely resolved through
    # `_with_backend`/`LAB_SETTINGS.provider` above, not hand-written.
    retrieved = rows['retrieve']
    assert retrieved['retriever'] == 'hybrid-rrf'
    assert retrieved['provider'] == 'fake'
    assert retrieved['n_questions'] == 2 and retrieved['decision'] is None

    evaluated = rows['run']
    # Identified by its own run id, never by the job id, so the row and the
    # JSON file the leaderboard reads are the same measurement.
    assert evaluated['experiment_id'] == '20260101-000000-abcdef'
    assert evaluated['label'] == 'the ledger'
    assert evaluated['seconds'] > 0
    assert evaluated['answerer'] == 'extractive'
    assert evaluated['provider'] == 'fake'
    assert evaluated['n_questions'] == 3
    # `ragas_mode='off'` in the stub result judged nothing, and an unjudged
    # row carries no score rather than a fabricated zero.
    assert evaluated['decision'] is None and evaluated['decision_stderr'] is None

    # 4. The detail stored beside each row: the whole result minus the
    # corpus. `detail_for` is exercised directly here, over the same jobs,
    # rather than through the `/api/experiments/{id}` route — the claim is
    # about what `ledger.detail_for` strips, not about the route.
    run_detail = ledger.experiment(evaluated['experiment_id'])['detail']
    assert run_detail['config']['index']['chunker'] == 'session'
    assert run_detail['summary'] == {'n_questions': 3}
    assert [row['id'] for row in run_detail['rows']] == ['q1', 'q2', 'q3']
    assert run_detail['selection']['n'] == 3
    assert 'chunks_by_session' not in run_detail, (
        'chunk text is byte-identical across every run sharing a build and '
        'reproduced by re-running it, so it does not belong on the row')

    # A retrieval's detail is its traces — the only thing it produced — so
    # unlike `chunks_by_session`, `questions` (carrying the trace) survives.
    retrieve_detail = ledger.experiment(retrieved['experiment_id'])['detail']
    assert retrieve_detail['questions'][0]['trace']['candidates']
    assert 'chunks_by_session' not in retrieve_detail

    assert ledger.experiment('no-such-experiment') is None

    assert (tmp_path / 'raglab.db').exists(), 'the ledger is one SQLite file'


def test_a_ledger_that_cannot_be_written_reports_on_the_job_and_never_loses_it(
        monkeypatch):
    # this is an integration test
    """A judged run costs hours, and an unwritable database must not be able
    to turn one into an error the panel reports over a result nobody can
    read — the same call `ragas_eval.JudgeWatch` makes about its progress
    counter. Both halves of that claim are checked: the job still finishes
    ('never fatal'), and the failure is reported on it rather than silently
    swallowed ('reports on the job')."""
    def refuse(*_args, **_kwargs):
        raise sqlite3.OperationalError('unable to open database file')

    monkeypatch.setattr(ledger, 'connect', refuse)
    jobs = Jobs(record=ledger.record)
    job_id = jobs.start('index', lambda report: {'chunks': 5},
                        config={'index': {'chunker': 'session'}})
    job = _run_to_terminal(jobs, job_id)
    assert job['state'] == 'done', job.get('error')
    assert job['result']['chunks'] == 5
    assert 'ledger_error' in job, (
        'a broken ledger must report on the job it could not record, not '
        'merely fail to break it')
    assert 'OperationalError' in job['ledger_error']


def test_the_ledger_is_not_kept_beside_the_code_that_writes_it():
    # this is a convention test
    """Where a `.db` goes is a settled question, and the answer is not
    "next to the code that writes it" — a durable record inside `src/`
    reads as build output and is the first thing a clean-up deletes."""
    default = ledger.db_path(env={})
    assert default == config.ROOT / 'databases' / 'raglab.db'
    assert 'src' not in default.parts
    # Overridable, which is what lets the suite guard itself in conftest.
    assert ledger.db_path(env={'RAGLAB_DB': '/tmp/x.db'}) == Path('/tmp/x.db')
