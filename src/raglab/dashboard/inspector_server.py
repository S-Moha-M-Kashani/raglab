"""The RAG Lab Inspector — a read-only view over the same fixtures and pipeline as the lab, mounted at :9002/inspector.

Builds its own in-memory index and writes nothing. `GET /api/follow` reads the
lab's newest finished jobs over plain `urllib` at `lab_base_url()`, which after
the mount is this same process; `RAGLAB_INSPECTOR_LAB_URL` still points it at a
lab somewhere else. A lab that is not running comes back as
`{'lab': 'down', ...}`, never an exception.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from raglab.evaluation import run_evaluation as evaluate
from raglab.configuration import explainer_assembly as explain
from raglab.evaluation import deterministic_metrics as metrics
from raglab.llm_backends import model_role_catalogue as models
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.configuration import lab_config
from raglab.configuration.lab_config import (
    LabConfig,
    load_lab_settings,
    settings_for_provider)
from raglab.corpora import dataset_import_contract as datasets
from raglab.rag_components.indexing.index_builder_registry import IndexRegistry
from raglab.llm_backends.chat_model_factory import lab_llm
from raglab.dashboard.service_presentation import (
    chunks_by_session,
    gold_available,
    mark_gold,
    summary_rows)
from raglab.dashboard.panel_server import Jobs
from raglab.dashboard.service_route_plumbing import (
    _accepted,
    _find_question,
    ground_truth_for,
    scaled_progress,
    screen)

STATIC = Path(__file__).resolve().parent / 'frontend'

LAB_URL_ENV = 'RAGLAB_INSPECTOR_LAB_URL'
DEFAULT_LAB_URL = 'http://localhost:9002'
# Short on purpose: the Inspector polls this every ~2s from the page, so a lab
# that is merely slow to answer must not stack up hung requests behind it.
LAB_TIMEOUT = 2.5


def lab_base_url() -> str:
    return os.environ.get(LAB_URL_ENV, DEFAULT_LAB_URL).rstrip('/')


def _lab_get_response(path: str) -> dict | tuple[int, str] | None:
    """GET one lab path, retaining explicit HTTP refusals for proxy routes."""
    url = f'{lab_base_url()}{path}'
    try:
        with urllib.request.urlopen(url, timeout=LAB_TIMEOUT) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode('utf-8')).get('detail', '')
        except (ValueError, OSError):
            detail = ''
        return error.code, detail or f'lab refused request ({error.code})'
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _lab_get(path: str) -> dict | None:
    """GET one path from the lab for legacy callers that treat every failure alike."""
    found = _lab_get_response(path)
    return found if isinstance(found, dict) else None


def _lab_post(path: str, payload: dict) -> dict | tuple[int, str] | None:
    """POST to the lab, preserving its refusal but bounding transport failure."""
    body = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{lab_base_url()}{path}', data=body, method='POST',
        headers={'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=LAB_TIMEOUT) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode('utf-8')).get('detail', '')
        except (ValueError, OSError):
            detail = ''
        return error.code, detail or f'lab refused request ({error.code})'
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _lab_delete(path: str) -> dict | None:
    """DELETE one lab path with the same bounded, failure-as-None policy as GET."""
    request = urllib.request.Request(f'{lab_base_url()}{path}', method='DELETE')
    try:
        with urllib.request.urlopen(request, timeout=LAB_TIMEOUT) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None

# Candidate F — the chosen architecture — as the Inspector's default config:
# the sweep baseline plus the LLM relevance gate. One source for the endpoint
# tests and the frontend so the two cannot drift.
CHOSEN_CONFIG = {
    'index': {'chunker': 'semantic-drift', 'embedder': 'sentence-transformers'},
    'retrieval': {'retriever': 'hybrid-rrf', 'k': 8, 'reranker': 'lexical',
                  'time_filter': True, 'grader': 'llm', 'grade_threshold': 0.4},
}


def _default_query_date(ground_truth: dict) -> str:
    """The 'now' a relative time expression resolves against when the caller
    named none — the ground truth's own `default_question_asked_at`, sliced to
    a plain date, the same default `run_evaluation._query_date` reads."""
    meta = ground_truth.get('groundtruth_dataset_metadata') or {}
    return meta.get('default_question_asked_at', '2026-07-28T00:00:00Z')[:10]


def _newest_done(jobs_index: dict, kind: str) -> dict | None:
    for entry in jobs_index.get('jobs', []):
        if entry.get('kind') == kind and entry.get('state') == 'done':
            return entry
    return None


def _job_view(jobs_index: dict, kind: str, fields: tuple[str, ...]) -> dict | None:
    entry = _newest_done(jobs_index, kind)
    if entry is None:
        return None
    full = _lab_get(f"/api/jobs/{entry['id']}")
    if full is None or full.get('result') is None:
        return None
    result = full['result']
    out = {'job_id': entry['id'], 'config': full.get('config')}
    out.update({field: result.get(field) for field in fields})
    return out


def _question_set(jobs_index: dict) -> dict | None:
    """The newest finished run over a *set* of questions, normalised from either lab route to one shape."""
    for entry in jobs_index.get('jobs', []):
        kind, key = entry.get('kind'), None
        if kind == 'retrieve':
            key = 'questions'
        elif kind == 'run':
            key = 'traces'
        if key is None or entry.get('state') != 'done':
            continue
        full = _lab_get(f"/api/jobs/{entry['id']}")
        result = (full or {}).get('result') or {}
        rows = result.get(key)
        if not rows:
            continue        # predates tracing, or an empty selection
        return {'kind': kind, 'job_id': entry['id'],
                'config': full.get('config'),
                'selection': result.get('selection'),
                'questions': rows}
    return None


def _newest_chunks(jobs_index: dict) -> dict | None:
    """The chunks the lab's newest finished job actually used, whatever kind of job it was.

    Not `kind == 'index'`: a run builds its index implicitly and creates
    no index job, so the rule is "the newest job that reported any
    chunks" — every index-building route reports them for this reason."""
    for entry in jobs_index.get('jobs', []):
        if entry.get('state') != 'done':
            continue
        full = _lab_get(f"/api/jobs/{entry['id']}")
        result = (full or {}).get('result') or {}
        groups = result.get('chunks_by_session')
        if not groups:
            continue        # a query job, or a run from before this
        return {'kind': entry.get('kind'), 'job_id': entry['id'],
                'config': full.get('config'),
                'chunks_by_session': groups,
                # `or []`: a job recorded before the lab reported
                # summaries has no such key, and absent is not an error.
                'summaries': result.get('summaries') or []}
    return None


def _generation_view(jobs_index: dict) -> dict | None:
    """What the newest *evaluation* wrote and how it scored — only an evaluation generates."""
    out = _job_view(jobs_index, 'run', ('rows', 'summary', 'ragas'))
    return out if out and out.get('rows') else None


def _followed_dataset(jobs_index: dict) -> str:
    """Which corpus the lab is working on, from its newest finished job.

    A config that names no index is passed over rather than read as the
    diary — "does not say" and "says the built-in one" are different facts.
    `''` is the built-in diary and also what a lab with nothing to say returns."""
    for entry in jobs_index.get('jobs', []):
        if entry.get('state') != 'done':
            continue
        index_cfg = (entry.get('config') or {}).get('index')
        if index_cfg is None:
            continue
        return index_cfg.get('dataset') or ''
    return ''


def create_inspector_app() -> FastAPI:
    settings = load_lab_settings()
    diary, ground_truth = datasets.load()
    def truth_for(cfg) -> dict:
        """The ground truth of the corpus a followed config names — same dataset as the index it built."""
        return ground_truth_for(cfg, ground_truth)

    registry = IndexRegistry(settings, diary)
    # No recorder: "the Inspector writes nothing" is what makes it safe to
    # point at a lab that is running.
    jobs = Jobs(max_history=settings.max_job_history)
    app = FastAPI(title='RAG Lab Inspector')

    @app.middleware('http')
    async def never_serve_yesterdays_page(request, call_next):
        """Same reason as the panel's: the frontend is read from disk per
        request, so a browser reusing a page without asking turns an edit into
        "nothing changed" — and a read-only window onto evidence must never show
        yesterday's evidence."""
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def page():
        return FileResponse(STATIC / 'inspector.html')

    # Only the Inspector's own two: the four files both surfaces share are
    # served by the panel at the root, and this app is mounted underneath it.
    @app.get('/inspector.css')
    def css():
        return FileResponse(STATIC / 'inspector.css', media_type='text/css')

    @app.get('/inspector.js')
    def js():
        return FileResponse(STATIC / 'inspector.js',
                            media_type='application/javascript')

    @app.get('/api/health')
    def health():
        return {'ok': True, 'storage': 'memory'}

    @app.get('/api/explain')
    def explain_metrics():
        """What every score on the Generation tab means — the same text `/api/options` on :9002 serves."""
        return {'metrics': explain.measures(), 'help': explain.topics(),
                'brief': explain.briefs()}

    @app.get('/api/groundtruth')
    def groundtruth(dataset: str = ''):
        """The pairs for whichever corpus is being followed, asked for by name rather than assumed built-in."""
        asked = datasets.load(dataset)[1] if dataset else ground_truth
        described = datasets.find(dataset)
        return {'meta': asked['groundtruth_dataset_metadata'],
                'questions': asked['groundtruth_dataset'],
                'dataset': dataset or datasets.BUILTIN,
                # Which language the corpus is in, so the page can render its
                # text in the direction that language reads. Said outright
                # rather than left in `meta`: a ground-truth file's meta
                # describes the question set, and no corpus carries a language
                # there — the built-in diary keeps its own on the corpus half,
                # and `_split` writes it only onto that half too. Empty for a
                # dataset the catalogue cannot describe, which the page reads
                # as "unknown" rather than as any particular direction.
                'language': described.language if described else '',
                'datasets': [found.as_dict() for found in datasets.catalogue()]}

    @app.get('/api/config')
    def chosen_config() -> dict:
        return {
            'chosen': CHOSEN_CONFIG,
            'chunkers': list(lab_config.CHUNKERS),
            'embedders': list(lab_config.EMBEDDERS),
            'retrievers': list(lab_config.RETRIEVERS),
            'rerankers': list(lab_config.RERANKERS),
            'graders': list(lab_config.GRADERS),
        }

    @app.post('/api/chunks')
    def chunks(payload: dict):
        cfg = LabConfig.from_dict(payload)

        def work(report):
            index = registry.get(cfg.index, progress=report)
            groups = chunks_by_session(index)
            summaries = summary_rows(index)
            # Both halves in one job, so the toggle needs no second request.
            # `total` stays the leaf count — mixing in summary rows would make
            # the chunk-size knob unreadable against it.
            return {'chunks_by_session': groups,
                    'total': sum(len(g['chunks']) for g in groups),
                    'summaries': summaries,
                    'total_summaries': len(summaries)}

        return _accepted(jobs.start('chunks', work))

    @app.post('/api/trace')
    def trace(payload: dict):
        cfg = LabConfig.from_dict(payload)
        asked = truth_for(cfg)
        qid = payload.get('question_id')
        question = _find_question(asked, qid)
        if question is None:
            raise HTTPException(404, f'unknown question id: {qid!r}')
        run_settings = settings_for_provider(settings,
                                             payload.get('provider') or '')
        screen(cfg, run_settings)
        query_date = payload.get('query_date') or _default_query_date(asked)

        def work(report):
            index = registry.get(cfg.index, progress=scaled_progress(report, 0.7))
            llm = lab_llm(run_settings)
            roles = models.resolve(cfg, run_settings)
            report('retrieving', 0.8, question['question'][:80])
            _outcome, tr = pipeline.retrieve_traced(
                index, cfg.retrieval, question['question'], query_date,
                llm=llm, models=roles)
            quotes = metrics.verbatim_quotes(question)
            flags = mark_gold([c['text'] for c in tr['candidates']], quotes)
            for cand, gold in zip(tr['candidates'], flags):
                cand['gold'] = gold
            return {'question': question, 'trace': tr, 'query_date': query_date}

        return _accepted(jobs.start('trace', work))

    @app.post('/api/questions')
    def run_question(payload: dict):
        """Run one ground-truth question end to end under the config the page follows, shaped as a
        retrieval row and a generation row — the config travels in the request so it matches its neighbours.

        The index is built in *this* process: the price of following rather than
        driving the lab, though the registry keeps it for the rest of the session."""
        cfg = LabConfig.from_dict(payload)
        asked = truth_for(cfg)
        qid = payload.get('question_id')
        question = _find_question(asked, qid)
        if question is None:
            raise HTTPException(404, f'unknown question id: {qid!r}')
        run_settings = settings_for_provider(settings,
                                             payload.get('provider') or '')
        screen(cfg, run_settings)
        query_date = payload.get('query_date') or _default_query_date(asked)

        def work(report):
            index = registry.get(cfg.index, progress=scaled_progress(report, 0.6))
            llm = lab_llm(run_settings)
            roles = models.resolve(cfg, run_settings)
            report('retrieving', 0.65, question['question'][:80])
            outcome, trace = pipeline.retrieve_traced(
                index, cfg.retrieval, question['question'], query_date,
                llm=llm, models=roles)
            quotes = metrics.verbatim_quotes(question)
            retrieval = evaluate.trace_row(
                question, trace,
                gold_present=gold_available(index, quotes))
            report('answering', 0.85)
            outcome = pipeline.answer(outcome, cfg.generation, llm=llm,
                                      models=roles)
            # Scored by the same function an evaluation uses, so the row
            # carries the same metrics under the same names.
            row = evaluate.json_safe(
                metrics.score_question(question, outcome, cfg.retrieval.k))
            return {'config': cfg.to_dict(), 'models': roles.as_dict(),
                    'retrieval': retrieval, 'generation': row}

        return _accepted(jobs.start('question', work, config=cfg.to_dict()))

    @app.get('/api/jobs/{job_id}')
    def job_status(job_id: str):
        return jobs.get(job_id)

    @app.get('/api/runs')
    def runs(limit: int = 50):
        return {'runs': evaluate.list_runs(limit)}

    @app.get('/api/runs/{run_id}')
    def run_detail(run_id: str):
        data = evaluate.load_run(run_id)
        if data is None:
            raise HTTPException(404, 'unknown run')
        return data

    @app.get('/api/imported-archives/{archive_id}')
    def imported_archive(archive_id: str):
        encoded_id = urllib.parse.quote(archive_id, safe='')
        found = _lab_get(f'/api/imported-archives/{encoded_id}')
        if found is None:
            raise HTTPException(
                404, 'imported archive is unavailable from the lab')
        return found

    @app.get('/api/experiments/{experiment_id}')
    def recorded_experiment(experiment_id: str):
        """One recorded experiment, proxied from the lab.

        Proxied rather than read from `raglab.db` directly: this service owns no
        ledger, and a second reader that opened the file would be a second
        thing to keep in step with a record whose whole value is being written
        once. The same reason `/api/imported-archives/{id}` proxies."""
        encoded_id = urllib.parse.quote(experiment_id, safe='')
        found = _lab_get(f'/api/experiments/{encoded_id}')
        if found is None:
            raise HTTPException(
                404, 'that experiment is unavailable from the lab')
        return found

    @app.get('/api/experiments/{experiment_id}/archive')
    def recorded_experiment_archive(experiment_id: str):
        """A finished experiment's own archive — the chunk text it ran over.

        The ledger strips chunk text from a job row, so a record read through
        `/api/experiments/{id}` cannot fill the Chunks tab; the archive the
        experiment wrote when it finished can, and it is the same object the
        export button writes. Proxied like the record, refusals kept: a 404 is
        an experiment from before archiving, a 409 a corpus this installation
        no longer holds — both are the lab's own words, not this service's."""
        encoded_id = urllib.parse.quote(experiment_id, safe='')
        found = _lab_get_response(f'/api/experiments/{encoded_id}/archive')
        if found is None:
            raise HTTPException(503, 'lab is unavailable; the archive cannot load')
        if isinstance(found, tuple):
            raise HTTPException(found[0], found[1])
        return found

    @app.post('/api/experiments/{experiment_id}/questions')
    def add_recorded_question(experiment_id: str, payload: dict):
        encoded_id = urllib.parse.quote(experiment_id, safe='')
        found = _lab_post(f'/api/experiments/{encoded_id}/questions', payload)
        if found is None:
            raise HTTPException(503, 'lab is unavailable; question was not added')
        if isinstance(found, tuple):
            raise HTTPException(found[0], found[1])
        return JSONResponse(found, status_code=202)

    @app.get('/api/experiments/{experiment_id}/questions')
    def recorded_questions(experiment_id: str):
        encoded_id = urllib.parse.quote(experiment_id, safe='')
        found = _lab_get_response(f'/api/experiments/{encoded_id}/questions')
        if found is None:
            raise HTTPException(503, 'lab is unavailable; recorded questions cannot load')
        if isinstance(found, tuple):
            raise HTTPException(found[0], found[1])
        return found

    @app.get('/api/lab-jobs/{job_id}')
    def lab_job_status(job_id: str):
        encoded_id = urllib.parse.quote(job_id, safe='')
        found = _lab_get_response(f'/api/jobs/{encoded_id}')
        if found is None:
            raise HTTPException(503, 'lab is unavailable; job status cannot load')
        if isinstance(found, tuple):
            raise HTTPException(found[0], found[1])
        return found

    @app.delete('/api/imported-archives/active')
    def clear_imported_archive():
        cleared = _lab_delete('/api/imported-archives/active')
        if cleared is None:
            raise HTTPException(
                503, 'lab is unavailable; archive preview was not cleared')
        return cleared

    @app.get('/api/follow')
    def follow():
        """The lab's newest *finished* jobs in one call, or 'down' when :9002 cannot be reached — HTTP 200 either way."""
        active = _lab_get('/api/imported-archives/active')
        archive_id = (active or {}).get('archive_id')
        jobs_index = _lab_get('/api/jobs')
        if jobs_index is None:
            return {'lab': 'down', 'lab_url': lab_base_url(), 'dataset': '',
                    'index': None, 'query': None, 'retrieval': None,
                    'generation': None, 'archive_id': archive_id}

        query_view = _job_view(jobs_index, 'query',
                               ('trace', 'question', 'question_id', 'answer'))
        return {'lab': 'up', 'lab_url': lab_base_url(),
                'dataset': _followed_dataset(jobs_index),
                'index': _newest_chunks(jobs_index), 'query': query_view,
                'retrieval': _question_set(jobs_index),
                'generation': _generation_view(jobs_index),
                'archive_id': archive_id}

    return app


app = create_inspector_app()
