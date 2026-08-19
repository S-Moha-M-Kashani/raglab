"""The RAG Lab service: settings panel, ad-hoc query inspector, eval runner. Binds :9002.

Depends on no other service, so no route probes anything before creating a
job. Runs are jobs, not requests — creating one answers 202 with a job id and a
Location, and the panel polls that — one at a time, since concurrent runs
would fight over the same index.
"""
import threading
import time
import traceback
import uuid
import inspect
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from raglab.agents import agentic_rag
from raglab.llm_backends import openrouter_key_memory as credentials
from raglab.corpora import dataset_import_contract as datasets
from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.evaluation import run_evaluation as evaluate
from raglab.configuration import explainer_assembly as explain
from raglab.evaluation import service_experiment_ledger as ledger
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
    CRITICS,
    DEPENDENCIES,
    DIFFICULTIES,
    EMBEDDERS,
    GRADERS,
    GRAPH_SOURCES,
    HIERARCHIES,
    PRODUCTION_CONFIG,
    RERANKERS,
    RETRIEVERS,
    ROOT,
    RUNS_DIR,
    SCOPES,
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


def _agent_options() -> dict:
    return {
        'scopes': list(SCOPES),
        'critics': list(CRITICS),
        'agent_support': agentic_rag.available(),
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
        # The shipped Assistant's own settings, for the panel's preset —
        # served rather than copied into the frontend, which would drift.
        'production': PRODUCTION_CONFIG,
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

    def start(self, kind: str, target, config: dict | None = None) -> str:
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
                                 '_cancel': threading.Event()}
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
        # The event is an implementation detail, not JSON the browser can read.
        return {key: value for key, value in job.items() if key != '_cancel'}

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
    # This service owns the ledger, so this is the one place a recorder is passed.
    jobs = Jobs(record=ledger.record)
    app = FastAPI(title='Lodestar RAG Lab')

    @app.get('/')
    def panel():
        return FileResponse(STATIC / 'panel.html')

    @app.get('/sorttable.js')
    def sorttable():
        """The column sorter, shared with the Inspector — one of three static files served outside the one page."""
        return FileResponse(STATIC / 'sorttable.js',
                            media_type='application/javascript')

    @app.get('/tokens.css')
    def tokens_css():
        """The design tokens shared with the Inspector, so a colour cannot drift apart on either page."""
        return FileResponse(STATIC / 'tokens.css', media_type='text/css')

    @app.get('/lab.js')
    def lab_js():
        """The utilities shared with the Inspector, so a name like escapeHtml has one behaviour, not two."""
        return FileResponse(STATIC / 'lab.js',
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

    @app.get('/api/options')
    def options():
        """Everything the panel needs to render itself, including what is actually installed."""
        live = settings_now()
        return (_catalogue_vocab() | _hierarchy_options() | _agent_options()
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

        def work(report, _cancelled):
            index = registry.get(cfg.index, progress=report, force=force)
            return {'collection': index.stats.collection,
                    'chunks': index.stats.chunks,
                    'leaves': index.stats.leaves,
                    'avg_chars': index.stats.avg_chars,
                    'p95_chars': index.stats.p95_chars,
                    'embed_dim': index.stats.embed_dim,
                    'build_seconds': index.stats.build_seconds,
                    # None on a flat build, distinguishing "no hierarchy" from
                    # "a hierarchy that found nothing".
                    'hierarchy': index.stats.hierarchy,
                    'reused': index.stats.reused, 'notes': index.stats.notes,
                    # So a follower (the Inspector) can render what was built
                    # without holding its own index; both halves, since
                    # `chunks_by_session` alone omits every summary a grouping wrote.
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

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled, JobCancelled)
            result = evaluate.run_eval(
                registry, questions_for(cfg), cfg, run_settings,
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
            # Added here, not inside `as_dict`: the run file `save_run` writes
            # stays the summary, with no trace or chunk text in it.
            return result.as_dict() | {
                'traces': result.traces,
                'chunks_by_session': result.chunks_by_session,
                'summaries': result.summaries}

        return _accepted(jobs.start('run', work, config=_with_backend(cfg, run_settings)))

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

    @app.get('/api/experiments')
    def experiments(limit: int = 200):
        """Everything this lab has ever finished, newest first — beside the leaderboard, never in it.

        An index build has no questions and no score, so a numbered table it
        appeared in would make a rank claim about work that measured nothing."""
        return {'experiments': ledger.experiments(limit)}

    @app.get('/api/experiments/{experiment_id}')
    def experiment_detail(experiment_id: str):
        found = ledger.experiment(experiment_id)
        if found is None:
            raise HTTPException(404, 'unknown experiment')
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
        asked = questions_for(cfg)
        query_date = payload.get('query_date') or asked['meta']['query_date']

        def work(report):
            # The implicit build is the long silent part — hand it the front of
            # the bar, or it all happens on 'starting 0%'.
            index = registry.get(cfg.index, progress=scaled_progress(report, 0.7))
            llm = lab_llm(run_settings)
            roles = models.resolve(cfg, run_settings)
            report('retrieving', 0.75, question[:80])
            # Traced rather than plain `retrieve`, for the per-step ranks the
            # Inspector's table needs. Same agent branch `run_eval` takes, so
            # the fast and slow paths cannot disagree about what a config does.
            if cfg.agent.scope:
                trace = {}
                outcome = agentic_rag.run(index, cfg, question, query_date,
                                          llm=llm, models=roles, trace=trace)
                report('answering', 0.9)
            else:
                outcome, trace = pipeline.retrieve_traced(
                    index, cfg.retrieval, question, query_date,
                    llm=llm, models=roles)
                report('answering', 0.9)
                outcome = pipeline.answer(outcome, cfg.generation, llm=llm,
                                          models=roles)
            # Exact match only, never fuzzy: everything else stays plainly
            # ungraded rather than guessed at.
            gt_question = next((q for q in asked['questions']
                                if q['question_fa'] == question), None)
            if gt_question is not None:
                quotes = [ev['quote'] for ev in gt_question.get('evidence', [])]
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
        try:
            found = datasets.import_dataset(payload)
        except ValueError as error:
            raise HTTPException(400, str(error))
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
        """The widget's own model list — served, because neither panel keeps
        a model list of its own."""
        return {'models': [{'value': value, 'label': label}
                           for value, (_, label) in widget.WIDGET_MODELS.items()],
                'default': widget.DEFAULT_MODEL}

    @app.post('/api/widget')
    def widget_chat(payload: dict):
        """The corner widget's endpoint: a question in, the agent's reply
        out. Synchronous, not a job — a chat turn is a request, not a run.
        An unknown model raises ValueError, answered as a 400 naming it."""
        message = (payload.get('message') or '').strip()
        if not message:
            raise HTTPException(400, 'message is empty')
        try:
            # The session is the page's claim about itself; the route only
            # carries it — absent lands as '', the stateless ask. The reply
            # arrives with its token account and is served unchanged.
            return widget.ask(message,
                              (payload.get('model') or '').strip(),
                              session=(payload.get('session') or '').strip())
        except widget.WidgetUnavailable as error:
            # The lab is up; its widget is not — the /api/queries split.
            raise HTTPException(502, str(error))

    @app.get('/api/health')
    def health():
        # No dependency to report: the lab is up or it is not running.
        return {'ok': True, 'storage': 'memory'}

    @app.exception_handler(ValueError)
    def value_error(_request, error: ValueError):
        return JSONResponse({'detail': str(error)}, status_code=400)

    return app


app = create_app()
