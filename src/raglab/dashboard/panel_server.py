"""The RAG Lab service: settings panel, ad-hoc query inspector, eval runner. Binds :9002.

Depends on no other service, so no route probes anything before creating a
job. Runs are jobs, not requests — creating one answers 202 with a job id and a
Location, and the panel polls that — one at a time, since concurrent runs
would fight over the same index.
"""
import json
import threading
import time
import traceback
import uuid
from urllib.parse import parse_qs
import inspect
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, StreamingResponse)

from raglab.llm_backends import openrouter_key_memory as credentials
from raglab.corpora import dataset_import_contract as datasets
from raglab.corpora import corpus_reading
from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import experiment_archive as archive
from raglab.configuration import explainer_assembly as explain
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.evaluation import experiment_archive_store as archive_store
from raglab.evaluation import leaderboard
from raglab.evaluation import deterministic_metrics as metrics
from raglab.llm_backends import model_role_catalogue as models
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.evaluation import ragas_judged_metrics as ragas_eval
from raglab.rag_components.retrieval import (
    retrieve_fuse_rerank_grade as retrieval)
from raglab.agents import widget
from raglab.dashboard import dev_trace_page
from raglab.configuration.lab_config import (
    ANSWERERS,
    CHUNKERS,
    DEPENDENCIES,
    EMBEDDERS,
    GRADERS,
    GRAPH_SOURCES,
    HIERARCHIES,
    LLM_PROVIDERS,
    RERANKERS,
    RETRIEVERS,
    ROOT,
    RUNS_DIR,
    STEPS,
    SUMMARIZERS,
    SUMMARY_SCOPES,
    GenerationConfig,
    IndexConfig,
    LabConfig,
    LabSettings,
    RetrievalConfig,
    load_lab_settings,
    settings_for_provider)
from raglab.rag_components.indexing import (
    summary_hierarchy_builder as hierarchy)
from raglab.rag_components.indexing.index_builder_registry import IndexRegistry
from raglab.llm_backends.chat_model_factory import lab_llm
from raglab.dashboard.service_presentation import (
    chunks_by_session,
    gold_available,
    mark_gold,
    summary_rows)
from raglab.dashboard.imported_archive_store import ImportedArchiveStore
from raglab.dashboard import routes
from raglab.dashboard.service_route_plumbing import (
    STATIC,
    JobCancelled,
    _accepted,
    _find_question,
    _with_backend,
    cancel_checker,
    ground_truth_for,
    install_no_store,
    screen,
    scaled_progress)
class JobCancelled(Exception):
    """A cooperative stop requested from the RAG Lab panel."""

class Jobs:
    """In-process job table, bounded. A lab restart loses running jobs;
    finished runs are on disk, which is the part that matters — and that is
    also why the table may forget the oldest of them past `max_history`."""

    def __init__(self, record=None, max_history: int = LabSettings.max_job_history):
        """`record(job, state)` is called once per finished job, or nothing is.

        `max_history` is how many *finished* jobs the table keeps, defaulting
        to the ceiling `LabSettings` states; 0 is unbounded. The panel polls
        this table, so an unbounded one is a poll that walks further every
        hour the lab stays up.

        A hook rather than a direct call to `ledger.record`, because the
        Inspector runs this same class read-only (`inspector_server.py` imports `Jobs`
        from here) and must not become a second writer of the ledger — so the
        service that owns the ledger passes the recorder, and the one that
        does not owns nothing to pass."""
        self.lock = threading.Lock()
        self.record = record
        self.max_history = max_history
        self.jobs: dict[str, dict] = {}
        self.current: str | None = None

    def _prune(self) -> None:
        """Drop the oldest finished jobs past the ceiling. Called under
        `self.lock`, on insert. Nothing is lost: every finished job has a
        ledger row and every evaluation a run file, and the two of them are the
        record — this table is only the live view of it. A running or
        cancelling job is never a candidate, whatever its age, because the live
        view is the only place it exists yet."""
        if not self.max_history:
            return
        terminal = [job_id for job_id, job in self.jobs.items()
                    if job['state'] not in ('running', 'cancelling')]
        for job_id in terminal[:-self.max_history]:
            del self.jobs[job_id]

    def start(self, kind: str, target, config: dict | None = None,
              archive=None) -> str:
        """`archive(job)` is called once, and only for a job that finished.

        Per job rather than per instance, because only the route that started
        the work knows what an archive of it would have to say — the panel
        controls a result cannot be read back off. A cancelled or errored job
        never reaches it: an archive is a record of an experiment that ran, and
        there is no honest archive of work that stopped half way.
        """
        with self.lock:
            if self.current and self.jobs[self.current]['state'] in ('running', 'cancelling'):
                # One message per state: wait, versus wait then retry.
                running = self.jobs[self.current]
                article = 'an' if running['kind'][0] in 'aeiou' else 'a'
                state = ('is still cancelling'
                         if running['state'] == 'cancelling'
                         else 'is already running')
                raise HTTPException(
                    409, f'{article} {running["kind"]} job {state} — '
                         'wait for it to finish, or cancel it first')
            job_id = uuid.uuid4().hex[:10]
            self.jobs[job_id] = {'id': job_id, 'kind': kind, 'state': 'running',
                                 'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                                 'stage': 'starting', 'progress': 0.0,
                                 'detail': '', 'config': config,
                                 'result': None, 'error': None,
                                 'cancel_requested': False,
                                 '_cancel': threading.Event(),
                                 '_archive': archive}
            self.current = job_id
            self._prune()

        cancel = self.jobs[job_id]['_cancel']

        def report(stage: str, fraction: float, detail: str = '') -> None:
            if cancel.is_set():
                raise JobCancelled()
            job = self.jobs[job_id]
            job['stage'] = stage
            job['progress'] = round(min(1.0, max(0.0, fraction)), 3)
            # e.g. "question 16/30 · hard" — a stage-only bar looks like a hang
            # on a judged local run that spends hours inside one stage.
            job['detail'] = detail

        def run() -> None:
            job = self.jobs[job_id]
            began = time.time()
            outcome = 'done'
            try:
                # Targets that make external calls receive a cancellation probe;
                # one-argument targets (small callers, tests) still work.
                wants_cancel = len(inspect.signature(target).parameters) >= 2
                job['result'] = target(report, cancel.is_set) if wants_cancel else target(report)
                if cancel.is_set():
                    raise JobCancelled()
            except JobCancelled:
                outcome = 'cancelled'
                job['detail'] = 'stopped before the next model call'
            except Exception as error:              # surfaced, never swallowed
                outcome = 'error'
                job['error'] = f'{type(error).__name__}: {error}'
                job['traceback'] = traceback.format_exc()[-2000:]
            # The only duration a build or retrieval has: neither has a `RunResult`.
            job['seconds'] = round(time.time() - began, 2)
            # Recorded here, before the state goes terminal — a row written
            # after `state = 'done'` is one a polling follower can miss.
            try:
                if self.record is not None:
                    self.record(job, outcome)
            except Exception as error:
                # A ledger records the work; it is never a condition of it —
                # reported on the job rather than swallowed.
                job['ledger_error'] = f'{type(error).__name__}: {error}'
            # The archive, on the ledger's terms exactly: written here, before
            # the state goes terminal, and never able to fail the job that
            # produced it. Only a job that *finished* has one — a cancelled or
            # errored run measured nothing whole, and an archive of it would be
            # a record of an experiment that never happened.
            try:
                if outcome == 'done' and job.get('_archive') is not None:
                    job['_archive'](job)
            except Exception as error:
                job['archive_error'] = f'{type(error).__name__}: {error}'
            job['state'] = outcome
            if outcome == 'done':
                job['progress'] = 1.0
                job['stage'] = 'done'
            elif outcome == 'cancelled':
                job['stage'] = 'cancelled'

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(404, 'unknown job')
        # The event and the archive hook are implementation details, not JSON
        # the browser can read — every private key is dropped by one rule
        # rather than by a list that has to be remembered.
        return {key: value for key, value in job.items()
                if not key.startswith('_')}

    def list(self) -> list[dict]:
        """Newest first, deliberately thin (id/kind/state/config) — not every job's result or traceback.

        Under the lock: the panel polls this while a starting job prunes the
        oldest finished ones, and walking a dict another thread is deleting
        from raises rather than answering."""
        with self.lock:
            snapshot = list(self.jobs.values())
        return [{'id': job['id'], 'kind': job['kind'], 'state': job['state'],
                 'started_at': job.get('started_at', ''),
                 'config': job.get('config')}
                for job in reversed(snapshot)]

    def cancel(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(404, 'unknown job')
        if job['state'] == 'running':
            job['cancel_requested'] = True
            job['_cancel'].set()
            job['state'] = 'cancelling'
            job['stage'] = 'stopping'
            job['detail'] = 'stopping before the next model call'
        return self.get(job_id)


@dataclass(frozen=True)
class PanelContext:
    """The state the panel's routes read, assembled once by `create_app` and
    handed to each route module's `register`.

    A container, and deliberately not a service layer: it holds and it does not
    decide. Every field is either data or a callable the factory built, so a
    route that needs a configuration screened calls the shared plumbing rather
    than a method here — the moment one of these names starts choosing a
    backend or shaping a response it has become the layer this project's
    complexity gate refuses. A convention test pins that.

    A dataclass rather than the app's own `state`, because `app.state` is
    untyped: a route reaching for a name nobody put there would fail at request
    time, while a missing field here fails when the app is built."""
    settings_now: Callable[[], LabSettings]
    corpus: dict
    ground_truth: dict
    questions_for: Callable[[LabConfig], dict]
    registry: IndexRegistry
    dataset_lock: Callable[..., threading.Lock]
    jobs: Jobs
    archives: ImportedArchiveStore


def create_app() -> FastAPI:
    widget.set_openrouter_key_resolver(credentials.active)
    # The widget imports no evaluation module — a convention test pins it as a
    # sealed leaf — so the two durable records reach it the same way the key
    # does: injected here, by the one module that already reads both. Three
    # functions off the modules that own them, deliberately not a fourth module
    # written for the widget: what it is handed is what the board is built from
    # and what this service's own route answers with, and all it adds is the
    # formatting that makes them readable to a model.
    widget.set_experiment_reader(SimpleNamespace(
        board_rows=leaderboard.board_rows,
        experiment=leaderboard.experiment,
        question_rows=evaluate.question_rows))
    boot_settings = load_lab_settings()

    def settings_now():
        """The boot settings, carrying whatever key the panel has since typed — the one place that decides."""
        return credentials.apply(boot_settings)

    settings = boot_settings
    diary, ground_truth = datasets.load()

    def questions_for(cfg: LabConfig) -> dict:
        """The ground truth of the corpus this config names — resolved by id, so index and questions match."""
        return ground_truth_for(cfg, ground_truth)

    # A dataset id names mutable machine-local content. Keep its snapshot,
    # in-memory index use and replacement atomic without making unrelated
    # datasets wait for one another. Lock order is deliberately one-way: the
    # guard is held only while looking up a per-id lock and is always released
    # before that dataset lock is acquired.
    dataset_locks_guard = threading.Lock()
    dataset_locks: dict[str, threading.Lock] = {}

    def dataset_lock(dataset_id: str = '') -> threading.Lock:
        key = dataset_id or datasets.BUILTIN
        with dataset_locks_guard:
            return dataset_locks.setdefault(key, threading.Lock())

    # Every job below opens `with dataset_lock(...), registry.hold(cfg.index):`.
    # The second half is the memory bound's other side: the registry keeps only
    # the newest few indexes, and a job says here that the one it is about to
    # work against is in use — so a build elsewhere can never take it away
    # mid-run and make the next question rebuild what is already resident.

    context = PanelContext(
        settings_now=settings_now,
        corpus=diary,
        ground_truth=ground_truth,
        questions_for=questions_for,
        registry=IndexRegistry(settings, diary),
        dataset_lock=dataset_lock,
        # This service owns the ledger, so this is the one place a recorder is
        # passed.
        jobs=Jobs(record=ledger.record, max_history=settings.max_job_history),
        archives=ImportedArchiveStore())
    # The routes read that state off the context they were handed. Named here
    # once, so a body says `jobs` rather than `context.jobs` fifty times — the
    # same unpacking each route module does at the top of its `register`.
    registry, jobs, archives = context.registry, context.jobs, context.archives

    app = FastAPI(title='RAG Lab')

    install_no_store(app)
    routes.assets.register(app, context)
    routes.configuration.register(app, context)
    routes.pipeline.register(app, context)
    routes.experiments.register(app, context)
    routes.datasets.register(app, context)
    routes.credentials.register(app, context)
    routes.widget.register(app, context)
    routes.dev_trace.register(app, context)

    @app.exception_handler(ValueError)
    def value_error(_request, error: ValueError):
        return JSONResponse({'detail': str(error)}, status_code=400)

    return app


app = create_app()
