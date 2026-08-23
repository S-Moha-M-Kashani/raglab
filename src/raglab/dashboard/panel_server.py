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
import inspect
import sqlite3
from types import SimpleNamespace
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import (FileResponse, JSONResponse,
                               StreamingResponse)

from raglab.llm_backends import openrouter_key_memory as credentials
from raglab.corpora import dataset_import_contract as datasets
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
from raglab.configuration.lab_config import (
    ANSWERERS,
    BALANCES,
    CHUNKERS,
    DEPENDENCIES,
    DIFFICULTIES,
    EMBEDDERS,
    GRADERS,
    GRAPH_SOURCES,
    HIERARCHIES,
    RERANKERS,
    RETRIEVERS,
    ROOT,
    RUNS_DIR,
    STEPS,
    SUMMARIZERS,
    SUMMARY_SCOPES,
    LabConfig,
    load_lab_settings,
    settings_for_provider)
from raglab.corpora.diary_corpus_loader import load_diary, load_ground_truth
from raglab.rag_components.indexing import (
    summary_hierarchy_builder as hierarchy)
from raglab.rag_components.indexing.index_builder_registry import IndexRegistry
from raglab.llm_backends.chat_model_factory import lab_llm
from raglab.dashboard.service_presentation import (
    chunks_by_session,
    mark_gold,
    summary_rows)
from raglab.dashboard.imported_archive_store import ImportedArchiveStore
from raglab.dashboard.service_route_plumbing import (
    _accepted,
    cancel_checker,
    ground_truth_for,
    screen,
    scaled_progress)

STATIC = Path(__file__).resolve().parent / 'frontend'


class JobCancelled(Exception):
    """A cooperative stop requested from the RAG Lab panel."""


def _relative(path: Path) -> str:
    """Repo-relative path for the panel, or absolute when it's outside the repo (`relative_to` raises)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _with_backend(cfg: LabConfig, run_settings) -> dict:
    """A job's config, plus the *resolved* backend it runs on — never the payload's possibly-blank request."""
    return cfg.to_dict() | {'provider': run_settings.provider}


def _archive_ui(payload: dict) -> dict:
    """The panel controls of one run request, in the shape an archive records
    them (`settings.ui`).

    The knobs are a `LabConfig` and travel as one; these five are not — they
    are how much of the corpus was scored and by which backend, which the
    config cannot say. Read back off the request that carried them rather than
    off the result, because the result never held them.

    `mode` is the panel's dropdown, and the request carries the *provider* that
    dropdown resolved to, so it is resolved back through the same table
    (`models.MODES`) rather than guessed at. A provider no mode offers leaves
    it blank, which is what a run started outside the panel honestly is.
    """
    provider = payload.get('provider') or ''
    return {
        'mode': next((mode.key for mode in models.MODES
                      if mode.provider == provider), ''),
        'ragas_mode': payload.get('ragas_mode', 'offline'),
        'limit': int(payload.get('limit') or 0),
        'ragas_limit': int(payload.get('ragas_limit') or 0),
        'types': list(payload.get('types') or []),
    }


def _catalogue_vocab() -> dict:
    """The closed vocabularies for the three pipeline stages: chunker/embedder, retriever/reranker, grader/answerer."""
    return {
        'chunkers': list(CHUNKERS), 'embedders': list(EMBEDDERS),
        'retrievers': list(RETRIEVERS), 'rerankers': list(RERANKERS),
        'graders': list(GRADERS), 'answerers': list(ANSWERERS),
    }


def _hierarchy_options() -> dict:
    """The summary hierarchy: grouping, graph edges, summariser, and what retrieval may do with the rows written."""
    return {
        'hierarchies': list(HIERARCHIES),
        'graph_sources': list(GRAPH_SOURCES),
        'summarizers': list(SUMMARIZERS),
        'summary_scopes': list(SUMMARY_SCOPES),
        # Verified by import, never guessed, so NA keeps meaning one thing:
        # this installation cannot load it.
        'hierarchy_support': hierarchy.available(),
    }


def _question_vocab() -> dict:
    return {
        'question_types': list(metrics.TYPES),
        'difficulties': list(DIFFICULTIES),
        # The sample is part of the measurement: two rows on different
        # samples are not two results of the same one.
        'balances': list(BALANCES),
    }


def _config_defaults() -> dict:
    return {
        # Served rather than duplicated per panel, so both grey out the same
        # knobs for the same stated reason.
        'dependencies': DEPENDENCIES,
        'defaults': LabConfig().to_dict(),
    }


def _step_list() -> dict:
    return {'steps': [{'key': step.key, 'short': step.short, 'label': step.label,
                       'note': step.note} for step in STEPS]}


def _model_catalogues(live) -> dict:
    return {
        'embedder_hints': embedding.embedder_hints(live),
        'embed_models': embedding.embed_model_catalogue(live),
        'models': models.catalogue(live),
        'model_roles': [role.as_dict() for role in models.ROLES],
        'modes': models.mode_catalogue(live),
    }


def _metric_help() -> dict:
    # Label, step, formula and library per metric, so a name cannot
    # drift from its definition.
    return {'metrics': explain.measures(), 'help': explain.topics()}


def _corpus_summary(diary: dict, ground_truth: dict) -> dict:
    return {'corpus': {
        'sessions': len(diary['sessions']),
        'messages': sum(len(s['messages']) for s in diary['sessions']),
        'from': diary['meta']['period']['from'],
        'to': diary['meta']['period']['to'],
        'threads': len(diary['threads']),
        'habits': len(diary.get('habits', {})),
        'questions': len(ground_truth['questions']),
        'query_date': ground_truth['meta'].get('query_date'),
    }}


def _capabilities(live) -> dict:
    return {'capabilities': {
        'fastembed': embedding.fastembed_available(),
        'sentence_transformers': embedding.sentence_transformers_available(),
        'cross_encoder': retrieval.cross_encoder_available(
            live.cross_encoder_model),
        'cross_encoder_model': live.cross_encoder_model,
        'fastembed_model': live.fastembed_model,
        # 'a real model is reachable', not 'a key exists' — under
        # RAGLAB_LLM=ollama every stage runs locally with no key.
        'llm': live.llm_ready,
        'llm_provider': live.provider,
        'llm_model': live.llm_model,
        'ollama_base_url': live.ollama_base_url,
        'ragas': ragas_eval.availability(live).as_dict(),
        'openrouter_key': credentials.state(live),
        # Stated positively, since the index is thrown away with the
        # process rather than merely "no service named".
        'storage': {'index': 'memory',
                    'runs': str(RUNS_DIR.relative_to(ROOT)),
                    'experiments': _relative(ledger.db_path())},
    }}


def _sent_events(events):
    """One iterator of dicts, encoded as server-sent events: `data: ` and the
    JSON, one object per line, blank line between. The lab's only streaming
    route, so this lives beside it rather than in the shared plumbing.

    A failure part-way through is encoded as an `error` event and the stream
    ends there. It cannot be a status code — those were spent on the first
    piece — and it must not be silence either: a page that saw the pieces stop
    with no word would have to guess whether the answer had finished, and
    guessing "finished" would show a fragment as a whole reply. Every
    exception is caught, not only the widget's own: an iterator that dies
    unexpectedly must still say so on the wire it was writing to.
    """
    try:
        for event in events:
            yield f'data: {json.dumps(event)}\n\n'
    except Exception as error:
        yield f'data: {json.dumps({"error": str(error)})}\n\n'


def _dataset_options() -> dict:
    return {
        'datasets': [found.as_dict() for found in datasets.catalogue()],
    }


class Jobs:
    """In-process job table. A lab restart loses running jobs; finished runs are
    on disk, which is the part that matters."""

    def __init__(self, record=None):
        """`record(job, state)` is called once per finished job, or nothing is.

        A hook rather than a direct call to `ledger.record`, because the
        Inspector runs this same class read-only (`inspector_server.py` imports `Jobs`
        from here) and must not become a second writer of the ledger — so the
        service that owns the ledger passes the recorder, and the one that
        does not owns nothing to pass."""
        self.lock = threading.Lock()
        self.record = record
        self.jobs: dict[str, dict] = {}
        self.current: str | None = None

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
                                 'stage': 'starting', 'progress': 0.0,
                                 'detail': '', 'config': config,
                                 'result': None, 'error': None,
                                 'cancel_requested': False,
                                 '_cancel': threading.Event(),
                                 '_archive': archive}
            self.current = job_id

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
        """Newest first, deliberately thin (id/kind/state/config) — not every job's result or traceback."""
        return [{'id': job['id'], 'kind': job['kind'], 'state': job['state'],
                 'config': job.get('config')}
                for job in reversed(list(self.jobs.values()))]

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
    diary = load_diary()
    ground_truth = load_ground_truth()

    def questions_for(cfg: LabConfig) -> dict:
        """The ground truth of the corpus this config names — resolved by id, so index and questions match."""
        return ground_truth_for(cfg, ground_truth)

    registry = IndexRegistry(settings, diary)
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

    # This service owns the ledger, so this is the one place a recorder is passed.
    jobs = Jobs(record=ledger.record)
    archives = ImportedArchiveStore()
    app = FastAPI(title='Lodestar RAG Lab')

    @app.middleware('http')
    async def never_serve_yesterdays_page(request, call_next):
        """The frontend is read from disk on every request, so an edit is live
        the moment it is saved — but `FileResponse` sends no `Cache-Control`,
        which leaves a browser free to reuse a page it already has without ever
        asking. That turns an edited panel into "nothing changed", and the
        reader has no way to tell that from a broken change. A workbench serves
        what is on disk or it is lying about what it is running."""
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def panel():
        return FileResponse(STATIC / 'panel.html')

    @app.get('/sorttable.js')
    def sorttable():
        """The column sorter, shared with the Inspector — one of three static files served outside the one page."""
        return FileResponse(STATIC / 'sorttable.js',
                            media_type='application/javascript')

    @app.get('/filtertable.js')
    def filtertable():
        """The leaderboard's row filter, which reads a cell with the sorter's own parser rather than a second one."""
        return FileResponse(STATIC / 'filtertable.js',
                            media_type='application/javascript')

    @app.get('/tokens.css')
    def tokens_css():
        """The design tokens shared with the Inspector, so a colour cannot drift apart on either page."""
        return FileResponse(STATIC / 'tokens.css', media_type='text/css')

    @app.get('/chrome.css')
    def chrome_css():
        """The bar and surface switcher shared with the Inspector, so the top of a page means one thing on both ports."""
        return FileResponse(STATIC / 'chrome.css', media_type='text/css')

    @app.get('/lab.js')
    def lab_js():
        """The utilities shared with the Inspector, so a name like escapeHtml has one behaviour, not two."""
        return FileResponse(STATIC / 'lab.js',
                            media_type='application/javascript')

    @app.get('/widget.css')
    def widget_css():
        """The widget's own rules, served to all three surfaces — the helper is
        not the Laboratory's, so its sheet is not panel.css."""
        return FileResponse(STATIC / 'widget.css', media_type='text/css')

    @app.get('/widget.js')
    def widget_js():
        """The widget itself. One file, three pages: it builds its own markup,
        so a surface gains the helper by loading this and nothing else."""
        return FileResponse(STATIC / 'widget.js',
                            media_type='application/javascript')

    @app.get('/leaderboard')
    def leaderboard_page():
        """The cross-run surface: what earlier runs said, kept off the lab page where the knobs live."""
        return FileResponse(STATIC / 'leaderboard.html')

    @app.get('/leaderboard.js')
    def leaderboard_js():
        """The leaderboard surface's script — it renders what /api/leaderboard serves and re-derives no rank of its own."""
        return FileResponse(STATIC / 'leaderboard.js',
                            media_type='application/javascript')

    @app.get('/panel.css')
    def panel_css():
        """The panel's style, extracted from panel.html's <style> block."""
        return FileResponse(STATIC / 'panel.css', media_type='text/css')

    @app.get('/panel.js')
    def panel_js():
        """The panel's script, extracted from panel.html's <script> block."""
        return FileResponse(STATIC / 'panel.js',
                            media_type='application/javascript')

    @app.get('/archive_io.js')
    def archive_io_js():
        """The versioned archive codec, loaded before the Panel integration."""
        return FileResponse(STATIC / 'archive_io.js',
                            media_type='application/javascript')

    @app.get('/experiment_handoff.js')
    def experiment_handoff_js():
        """The board-to-Laboratory handoff, loaded by both pages: the board writes the slot, the panel decides which recorded knobs this installation can serve."""
        return FileResponse(STATIC / 'experiment_handoff.js',
                            media_type='application/javascript')

    @app.get('/api/options')
    def options():
        """Everything the panel needs to render itself, including what is actually installed."""
        live = settings_now()
        return (_catalogue_vocab() | _hierarchy_options()
                | _question_vocab() | _config_defaults() | _step_list()
                | _model_catalogues(live) | _metric_help()
                | _corpus_summary(diary, ground_truth) | _capabilities(live)
                | _dataset_options() | {'indexes': registry.known()})

    @app.post('/api/indexes')
    def build_index(payload: dict):
        cfg = LabConfig.from_dict(payload)
        force = bool(payload.get('force'))
        # The same screen the run routes apply: a missing library fails as a
        # 400 naming what to install, not a 500 from an import three frames
        # down. Only the index half is checked — a build reads no model.
        problems = [p for p in cfg.validate()
                    if not p.startswith(('unknown retriever', 'unknown reranker',
                                         'unknown grader', 'unknown answerer',
                                         'unknown summary_scope', 'k must be'))]
        if problems:
            raise HTTPException(400, '; '.join(problems))

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled, JobCancelled)
            check_cancelled()
            with dataset_lock(cfg.index.dataset):
                check_cancelled()
                index = registry.get(cfg.index, progress=report, force=force)
                return {'collection': index.stats.collection,
                        'chunks': index.stats.chunks,
                        'leaves': index.stats.leaves,
                        'avg_chars': index.stats.avg_chars,
                        'p95_chars': index.stats.p95_chars,
                        'embed_dim': index.stats.embed_dim,
                        'build_seconds': index.stats.build_seconds,
                        # None on a flat build, distinguishing "no hierarchy"
                        # from "a hierarchy that found nothing".
                        'hierarchy': index.stats.hierarchy,
                        'reused': index.stats.reused, 'notes': index.stats.notes,
                        # So a follower (the Inspector) can render what was
                        # built without holding its own index; both halves,
                        # since `chunks_by_session` alone omits every summary a
                        # grouping wrote.
                        'chunks_by_session': chunks_by_session(index),
                        'summaries': summary_rows(index)}

        return _accepted(jobs.start('index', work, config=cfg.to_dict()))

    @app.post('/api/evaluations')
    def start_evaluation(payload: dict):
        cfg = LabConfig.from_dict(payload)
        # The mode dropdown's backend override, applied before the screen so
        # the settings that refuse a model are the settings that would run it.
        run_settings = settings_for_provider(settings_now(),
                                             payload.get('provider') or '')
        screen(cfg, run_settings)
        run_execution = {
            'provider': run_settings.provider,
            'models': models.resolve(cfg, run_settings).as_dict(),
        }
        metric_catalogue = explain.measures()
        archive_ui = _archive_ui(payload)

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled, JobCancelled)
            check_cancelled()
            with dataset_lock(cfg.index.dataset):
                # Snapshot and index use share this boundary. A replacement of
                # the same id cannot put new corpus evidence beside old chunks.
                check_cancelled()
                run_corpus, run_truth = datasets.load(cfg.index.dataset)
                result = evaluate.run_eval(
                    registry, run_truth, cfg, run_settings,
                    types=payload.get('types') or None,
                    difficulty=payload.get('difficulty') or None,
                    limit=payload.get('limit') or None,
                    balance=payload.get('balance') or 'stride',
                    ragas_mode=payload.get('ragas_mode', 'offline'),
                    ragas_limit=payload.get('ragas_limit') or None,
                    workers=int(payload.get('workers', 1)), progress=report,
                    # Always traced, so the Inspector is never blank after a run —
                    # a recording of the same retrieval, so no score can move.
                    trace=True, cancelled=check_cancelled)
                # Added here, not inside `as_dict`: the run file `save_run`
                # writes stays the summary, with no trace or chunk text in it.
                return result.as_dict() | {
                    'traces': result.traces,
                    'chunks_by_session': result.chunks_by_session,
                    'summaries': result.summaries,
                    'archive_evidence': {
                        'execution': run_execution,
                        'metric_catalogue': metric_catalogue,
                        'inspector': {
                            'dataset': {
                                'id': cfg.index.dataset or datasets.BUILTIN,
                                'corpus': run_corpus,
                                'ground_truth': run_truth,
                            },
                            'chunks_by_session': result.chunks_by_session,
                            'summaries': result.summaries,
                            'traces': result.traces,
                        },
                    }}

        def keep_archive(job: dict) -> None:
            """One finished evaluation, written down as the archive a reader
            would have downloaded.

            The evidence is already assembled for the browser; all this adds is
            the knob surface it was produced under — the config off the result
            itself, so the two can never disagree, and the panel controls this
            request carried. Handed to the store as a whole object rather than
            column by column: `build_completed` is the export codec's own
            server-side twin, so an experiment on record and an experiment
            exported are the same thing said twice.

            Reached only from `Jobs.run`, and only for a job that finished:
            unfinished work is not saved, and a failure here is reported on the
            job rather than failing it.
            """
            result = job.get('result')
            evidence = result.get('archive_evidence') if isinstance(result, dict) else None
            if not isinstance(evidence, dict):
                return
            canonical = {key: value for key, value in result.items()
                         if key != 'archive_evidence'}
            archive_store.store_completed(
                {'config': canonical['config'], 'ui': archive_ui}, canonical,
                evidence)

        return _accepted(jobs.start('run', work,
                                    config=_with_backend(cfg, run_settings),
                                    archive=keep_archive))

    @app.post('/api/retrievals')
    def start_retrieval(payload: dict):
        """Retrieval only, over the questions the eval card has selected — no answering, no judging, no run file.

        Its own route rather than a flag on `/api/evaluations`, so it stays the
        step affordable to repeat while moving one knob. Takes the same
        selection arguments as an evaluation on purpose."""
        cfg = LabConfig.from_dict(payload)
        run_settings = settings_for_provider(settings_now(),
                                             payload.get('provider') or '')
        screen(cfg, run_settings)

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled, JobCancelled)
            check_cancelled()
            with dataset_lock(cfg.index.dataset):
                check_cancelled()
                return evaluate.run_retrieval(
                    registry, questions_for(cfg), cfg, run_settings,
                    types=payload.get('types') or None,
                    difficulty=payload.get('difficulty') or None,
                    limit=payload.get('limit') or None,
                    balance=payload.get('balance') or 'stride',
                    progress=report, cancelled=check_cancelled)

        return _accepted(jobs.start('retrieve', work,
                                    config=_with_backend(cfg, run_settings)))

    @app.get('/api/jobs')
    def list_jobs():
        """Every job this process has run, newest first — id/kind/state/config only, not the full result."""
        return {'jobs': jobs.list()}

    @app.get('/api/jobs/{job_id}')
    def job_status(job_id: str):
        return jobs.get(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel_job(job_id: str):
        return jobs.cancel(job_id)

    @app.get('/api/evaluations')
    def evaluations(limit: int = 50):
        # `total` beside the rows, since this listing is bounded.
        return {'runs': evaluate.list_runs(limit),
                'total': evaluate.count_runs()}

    @app.get('/api/leaderboard')
    def leaderboard_board(dataset: str = '', limit: int = 500):
        """One board per dataset: every experiment that touched one corpus.

        `dataset=''` is the built-in default, `dataset='*'` is every experiment
        — the table that used to sit on the lab page, which is the same
        population with no filter and so an option in the same picker rather
        than a second surface.

        The grouping and the row shape come from `evaluation.leaderboard`, the
        same module `raglab-leaderboard` prints from, so the page and the
        command line cannot describe the same records differently. This route is
        why that module lives in `evaluation/` rather than among the terminal
        tools no route reaches."""
        boards = leaderboard.build_board(limit)
        wanted = dataset or datasets.BUILTIN
        # `every_row`, not the boards concatenated: the page's own prose says the
        # order it was served in is the ranking, and a concatenation is ordered
        # by dataset block instead — so the unfiltered view would have said the
        # served order meant something it did not.
        rows = (leaderboard.every_row(boards) if dataset == '*' else
                next((b.rows for b in boards if b.dataset == wanted), []))
        return {'dataset': dataset or wanted,
                'datasets': [found.as_dict() for found in datasets.catalogue()],
                'rows': rows}

    @app.get('/api/experiments')
    def experiments(limit: int = 200):
        """Everything this lab has ever finished, newest first — beside the leaderboard, never in it.

        An index build has no questions and no score, so a numbered table it
        appeared in would make a rank claim about work that measured nothing."""
        return {'experiments': ledger.experiments(limit)}

    @app.get('/api/experiments/{experiment_id}')
    def experiment_detail(experiment_id: str):
        """One experiment by id, from whichever of the two records holds it.

        The ledger first, then the run file, because the board is built from
        both and a row it lists must be a row this answers. An evaluation older
        than the ledger has only its run file, and those are most of the scored
        rows there are."""
        found = leaderboard.experiment(experiment_id)
        if found is None:
            raise HTTPException(404, 'unknown experiment')
        row = ledger.experiment(experiment_id)
        run = evaluate.load_run(experiment_id)
        # `experiment_record` is the projection the board's own rows are, so the
        # row a reader clicked and the page that opens it cannot give two
        # accounts of one experiment — and any later reader that wants the same
        # experiment without the evidence asks the same function rather than
        # writing the ledger-versus-run precedence a third time. What a page
        # needs on top *is* the evidence: the ledger's own `detail`, which
        # carries the whole job result for a row it recorded, and the run file
        # itself for an evaluation older than the ledger.
        return found | {'detail': (row or {}).get('detail') or run or {}}

    @app.get('/api/experiments/{experiment_id}/archive')
    def experiment_archive_route(experiment_id: str):
        """One experiment as the portable archive the export button writes.

        Its own route rather than a shape change to `/api/experiments/{id}`,
        because that one has a second reader: the Inspector's recorded mode
        reads `detail`, `state`, `kind` and `error` off it, and would go blank
        if it started receiving an archive instead.

        This is what the board's open button hands to the panel, and it is the
        *same object* a downloaded file carries — so opening a row and importing
        its export are one path with one strictness, not two that can drift.

        The corpus is spliced back in on the way out (`archive_store.serve`)
        from the content-addressed corpus store, and a version that store does
        not hold is a refusal rather than a substitution: 409, because the
        archive is intact and it is this installation that has moved.
        """
        db = archive_store.connect(ledger.db_path())
        try:
            found = archive_store.serve(db, experiment_id)
        except archive_store.ArchiveStoreError as error:
            raise HTTPException(409, str(error))
        finally:
            db.close()
        if found is None:
            raise HTTPException(
                404, f'{experiment_id} has no complete archive: only '
                'experiments whose evidence survives in full are archived')
        return found

    @app.post('/api/imported-archives')
    def import_archive(payload: dict):
        try:
            return archives.import_archive(payload)
        except archive.ArchiveError as error:
            raise HTTPException(400, str(error)) from error
        except sqlite3.Error as error:
            raise HTTPException(500, 'archive database persistence failed') from error

    @app.get('/api/imported-archives/active')
    def active_archive():
        return archives.metadata()

    @app.delete('/api/imported-archives/active')
    def clear_active_archive():
        archives.clear()
        return {'archive_id': None}

    @app.get('/api/imported-archives/{archive_id}')
    def imported_archive(archive_id: str):
        found = archives.get(archive_id)
        if found is None:
            raise HTTPException(404, 'unknown imported archive')
        return found

    @app.get('/api/evaluations/{run_id}')
    def evaluation_detail(run_id: str):
        data = evaluate.load_run(run_id)
        if data is None:
            raise HTTPException(404, 'unknown run')
        return data

    @app.post('/api/queries')
    def ad_hoc_query(payload: dict):
        """Run one question through the current settings and return every stage — still a job, since the
        index it builds implicitly can outwait any HTTP timeout; preconditions still refuse synchronously."""
        cfg = LabConfig.from_dict(payload)
        question = (payload.get('question') or '').strip()
        if not question:
            raise HTTPException(400, 'question is required')
        # The same screen /api/evaluations applies, so the two routes cannot
        # disagree about which configs are legal.
        run_settings = settings_for_provider(settings_now(),
                                             payload.get('provider') or '')
        screen(cfg, run_settings)
        requested_query_date = payload.get('query_date')

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled, JobCancelled)
            check_cancelled()
            with dataset_lock(cfg.index.dataset):
                check_cancelled()
                asked = questions_for(cfg)
                query_date = (requested_query_date
                              or asked['meta']['query_date'])
                # The implicit build is the long silent part — hand it the
                # front of the bar, or it all happens on 'starting 0%'.
                index = registry.get(
                    cfg.index, progress=scaled_progress(report, 0.7))
                llm = lab_llm(run_settings)
                roles = models.resolve(cfg, run_settings)
                report('retrieving', 0.75, question[:80])
                # Traced rather than plain `retrieve`, for the per-step ranks
                # the Inspector's table needs.
                outcome, trace = pipeline.retrieve_traced(
                    index, cfg.retrieval, question, query_date,
                    llm=llm, models=roles)
                report('answering', 0.9)
                outcome = pipeline.answer(
                    outcome, cfg.generation, llm=llm, models=roles)
                # Exact match only, never fuzzy: everything else stays plainly
                # ungraded rather than guessed at.
                gt_question = next((q for q in asked['questions']
                                    if q['question_fa'] == question), None)
                if gt_question is not None:
                    quotes = [ev['quote']
                              for ev in gt_question.get('evidence', [])]
                    gold_flags = mark_gold(
                        [c['text'] for c in trace['candidates']], quotes)
                    question_id = gt_question['id']
                else:
                    gold_flags = [False] * len(trace['candidates'])
                    question_id = None
                for candidate, gold in zip(trace['candidates'], gold_flags):
                    candidate['gold'] = gold
                    candidate['question_id'] = question_id
                return (outcome.as_dict()
                        | {'models': roles.as_dict(), 'trace': trace,
                           'question_id': question_id})

        return _accepted(jobs.start('query', work,
                                    config=_with_backend(cfg, run_settings)))

    @app.get('/api/questions')
    def questions(limit: int = 200, dataset: str = ''):
        """The ground truth without its answers, for picking a question in the query panel."""
        asked = (datasets.load(dataset)[1] if dataset else ground_truth)
        return {'questions': [
            {'id': q['id'], 'type': q['type'], 'difficulty': q['difficulty'],
             'question_fa': q['question_fa'], 'question_en': q['question_en'],
             'answerable': q['answerable'],
             'evidence_sessions': [ev['session_id'] for ev in q['evidence']]}
            for q in asked['questions'][:limit]]}

    @app.get('/api/datasets')
    def list_datasets():
        return {'datasets': [found.as_dict() for found in datasets.catalogue()]}

    @app.post('/api/datasets')
    def import_dataset(payload: dict):
        """Take one dataset file, check it against the contract, keep it — 400 with every problem at once."""
        meta = payload.get('dataset') if isinstance(payload, dict) else None
        raw_id = meta.get('id') if isinstance(meta, dict) else ''
        lock_id = raw_id if isinstance(raw_id, str) else ''
        with dataset_lock(lock_id):
            try:
                found = datasets.import_dataset(payload)
            except ValueError as error:
                raise HTTPException(400, str(error))
            # Import writes the file and clears the loader cache first; eviction
            # is the final step under the same lock, so no later index lookup
            # can observe the new file through an old cached index.
            registry.invalidate_dataset(found.id)
        return found.as_dict()

    @app.post('/api/credentials')
    def set_credentials(payload: dict):
        """Take the OpenRouter key from the panel, held for this process only, never recorded on a run."""
        try:
            credentials.set_key(payload.get('api_key') or '')
        except ValueError as error:
            raise HTTPException(400, str(error))
        widget.reset()
        return credentials.state(settings_now())

    @app.delete('/api/credentials')
    def clear_credentials():
        """Forget the key this panel supplied; never unsets the environment's own."""
        credentials.clear()
        widget.reset()
        return credentials.state(settings_now())

    @app.get('/api/widget')
    def widget_options():
        """The widget's own model list and the four questions its empty log
        offers — served, because neither panel keeps a model list of its own,
        and because the starters are model-facing text, which in this project
        is a fixture rather than a string in a page. They ride the response
        that already exists: no new route, and no new import inside the
        widget package, which is a sealed leaf."""
        return {'models': [{'value': value, 'label': label}
                           for value, (_, label) in widget.WIDGET_MODELS.items()],
                'default': widget.DEFAULT_MODEL,
                'starters': widget.STARTERS}

    @app.post('/api/widget')
    def widget_chat(payload: dict):
        """The corner widget's endpoint: a question in, the widget's reply
        out. Synchronous, not a job — a chat turn is a request, not a run.
        An unknown model raises ValueError, answered as a 400 naming it."""
        message = (payload.get('message') or '').strip()
        if not message:
            raise HTTPException(400, 'message is empty')
        try:
            # The thread is the page's claim about which conversation this is —
            # the lab's active experiment, or `general`. The route only carries
            # it. The reply arrives with its token account and is served
            # unchanged.
            return widget.ask(message,
                              (payload.get('model') or '').strip(),
                              thread=(payload.get('thread') or '').strip())
        except widget.WidgetUnavailable as error:
            # The lab is up; its widget is not — the /api/queries split.
            raise HTTPException(502, str(error))

    @app.post('/api/widget/stream')
    def widget_stream(payload: dict):
        """The same turn as POST /api/widget, sent as it is written: one
        server-sent event per piece of the answer, then one final event
        carrying the reply the lab now holds and its token account — the very
        body the other route returns whole. The page renders the pieces as
        they land and adopts the final reply, so what stays on screen is what
        the conversation log holds rather than whatever the pieces spelled.

        `widget.stream` raises before it yields anything, which is what keeps a
        refusal a status code: an unserved model is a 400 and an unreachable
        widget a 502, decided here, before the response opens. Once the first
        piece is out the status code is spent, so a failure after that can only
        be said inside the stream — an `error` event, and no `reply` event
        ever, because a half-written answer must never be handed over as a
        whole one."""
        message = (payload.get('message') or '').strip()
        if not message:
            raise HTTPException(400, 'message is empty')
        try:
            events = widget.stream(message,
                                   (payload.get('model') or '').strip(),
                                   thread=(payload.get('thread') or '').strip())
        except widget.WidgetUnavailable as error:
            raise HTTPException(502, str(error))
        return StreamingResponse(
            _sent_events(events), media_type='text/event-stream',
            # No cache anywhere in front of a conversation, and no proxy
            # buffering: a stream held back until it completes is the sudden
            # printing this route exists to end.
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/api/widget/history')
    def widget_history(thread: str = ''):
        """One conversation, as the widget holds it. This is what a page draws
        after a refresh: the lab is the only copy of the transcript, so a
        reader's log and the model's memory cannot drift apart. A thread nobody
        has used is empty, never a 404 — a conversation that has not happened
        yet is not an error."""
        return widget.history(thread)

    @app.delete('/api/widget/history')
    def widget_forget(thread: str = ''):
        """New Chat. Ends the conversation named and no other — the reader's
        other experiments keep theirs. Answers with the emptied thread, so the
        page redraws from the lab rather than assuming what it now holds."""
        widget.forget(thread)
        return widget.history(thread)

    @app.get('/api/health')
    def health():
        # No dependency to report: the lab is up or it is not running.
        return {'ok': True, 'storage': 'memory'}

    @app.exception_handler(ValueError)
    def value_error(_request, error: ValueError):
        return JSONResponse({'detail': str(error)}, status_code=400)

    return app


app = create_app()
