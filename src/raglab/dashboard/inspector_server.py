"""The RAG Lab Inspector — a read-only view over the same fixtures and pipeline as the lab, mounted at :9002/inspector.

Builds its own in-memory index and writes nothing. Everything it shows of the
lab's own records it asks the lab for, through the `LabAccess` seam it is
handed at mount time: mounted, that is `InProcessLabAccess` and the ask is a
function call; pointed at another machine with `RAGLAB_INSPECTOR_LAB_URL`, it
is `HttpLabAccess` and the ask is a bounded `urllib` request. A lab that is not
running comes back as `{'lab': 'down', ...}`, never an exception.

The application is built by whoever mounts or serves it — `served_lab.py`
composes the one this project ships — rather than held here, because the seam
it reads the lab through is decided at that moment and an app built before it
could only ever be the HTTP one.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

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
    Asset,
    LabAccess,
    LabReply,
    _accepted,
    _find_question,
    ground_truth_for,
    install_assets,
    install_no_store,
    scaled_progress,
    screen)

# The Inspector's own two files and the page they dress. The four both surfaces
# share are served by the panel at the root, and this app is mounted underneath
# it, so they are deliberately absent here rather than served twice.
ASSETS = {
    '/': Asset('inspector.html', None,
               'The read-only window onto the evidence a run left, at the '
               'root of the mount — its only address.'),
    '/inspector.css': Asset(
        'inspector.css', 'text/css',
        "The Inspector's own style, on top of the tokens and chrome the panel "
        'serves to both surfaces.'),
    '/inspector.js': Asset(
        'inspector.js', 'application/javascript',
        "The Inspector's own script; the utilities it shares with the panel "
        'come from the lab.js the panel serves.'),
}

LAB_URL_ENV = 'RAGLAB_INSPECTOR_LAB_URL'
DEFAULT_LAB_URL = 'http://localhost:9002'
# Short on purpose: the Inspector polls this every ~2s from the page, so a lab
# that is merely slow to answer must not stack up hung requests behind it.
LAB_TIMEOUT = 2.5


def lab_base_url() -> str:
    return os.environ.get(LAB_URL_ENV, DEFAULT_LAB_URL).rstrip('/')


def _id(value: str) -> str:
    """One path segment, so an id with a slash in it cannot become two."""
    return urllib.parse.quote(value, safe='')


class HttpLabAccess:
    """The nine lab operations over HTTP, for a lab in another process.

    One request helper and not one per verb: the four this replaces differed
    only in the method they sent and in how much of a refusal they bothered to
    keep, and the one that kept least is how a stated 409 used to arrive at the
    reader as "the lab is unavailable". Every verb now keeps the lab's status
    and the lab's words, and reports only transport failure as unavailability.
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or lab_base_url()).rstrip('/')

    def _request(self, method: str, path: str,
                 payload: dict | None = None) -> LabReply:
        body = None if payload is None else json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            f'{self.base_url}{path}', data=body, method=method,
            headers={'content-type': 'application/json'} if body else {})
        try:
            with urllib.request.urlopen(request, timeout=LAB_TIMEOUT) as response:
                # 202 as readily as 200: starting a job is an answer too.
                if not 200 <= response.status < 300:
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

    def imported_archive(self, archive_id: str) -> LabReply:
        return self._request('GET', f'/api/imported-archives/{_id(archive_id)}')

    def active_archive(self) -> LabReply:
        return self._request('GET', '/api/imported-archives/active')

    def clear_active_archive(self) -> LabReply:
        return self._request('DELETE', '/api/imported-archives/active')

    def experiment(self, experiment_id: str) -> LabReply:
        return self._request('GET', f'/api/experiments/{_id(experiment_id)}')

    def experiment_archive(self, experiment_id: str) -> LabReply:
        return self._request(
            'GET', f'/api/experiments/{_id(experiment_id)}/archive')

    def experiment_questions(self, experiment_id: str) -> LabReply:
        return self._request(
            'GET', f'/api/experiments/{_id(experiment_id)}/questions')

    def add_experiment_question(self, experiment_id: str,
                                payload: dict) -> LabReply:
        return self._request(
            'POST', f'/api/experiments/{_id(experiment_id)}/questions', payload)

    def job(self, job_id: str) -> LabReply:
        return self._request('GET', f'/api/jobs/{_id(job_id)}')

    def jobs(self) -> LabReply:
        return self._request('GET', '/api/jobs')


def _document(found: LabReply) -> dict | None:
    """A lab answer for a caller that reads only the document.

    Two of the Inspector's routes and every `/api/follow` helper treat a
    refusal and an outage alike — an empty view either way — so they say so
    here rather than through a second transport that quietly drops the
    difference before they can see it."""
    return found if isinstance(found, dict) else None


def _answered(found: LabReply, outage: str) -> dict:
    """The document, or the same refusal the reader would have got from the lab.

    `outage` is what a lab that could not be reached at all is called; a lab
    that answered keeps its own status and its own words."""
    if found is None:
        raise HTTPException(503, outage)
    if isinstance(found, tuple):
        raise HTTPException(found[0], found[1])
    return found

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


def _job_view(lab: LabAccess, jobs_index: dict, kind: str,
              fields: tuple[str, ...]) -> dict | None:
    entry = _newest_done(jobs_index, kind)
    if entry is None:
        return None
    full = _document(lab.job(entry['id']))
    if full is None or full.get('result') is None:
        return None
    result = full['result']
    out = {'job_id': entry['id'], 'config': full.get('config')}
    out.update({field: result.get(field) for field in fields})
    return out


def _question_set(lab: LabAccess, jobs_index: dict) -> dict | None:
    """The newest finished run over a *set* of questions, normalised from either lab route to one shape."""
    for entry in jobs_index.get('jobs', []):
        kind, key = entry.get('kind'), None
        if kind == 'retrieve':
            key = 'questions'
        elif kind == 'run':
            key = 'traces'
        if key is None or entry.get('state') != 'done':
            continue
        full = _document(lab.job(entry['id']))
        result = (full or {}).get('result') or {}
        rows = result.get(key)
        if not rows:
            continue        # predates tracing, or an empty selection
        return {'kind': kind, 'job_id': entry['id'],
                'config': full.get('config'),
                'selection': result.get('selection'),
                'questions': rows}
    return None


def _newest_chunks(lab: LabAccess, jobs_index: dict) -> dict | None:
    """The chunks the lab's newest finished job actually used, whatever kind of job it was.

    Not `kind == 'index'`: a run builds its index implicitly and creates
    no index job, so the rule is "the newest job that reported any
    chunks" — every index-building route reports them for this reason."""
    for entry in jobs_index.get('jobs', []):
        if entry.get('state') != 'done':
            continue
        full = _document(lab.job(entry['id']))
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


def _generation_view(lab: LabAccess, jobs_index: dict) -> dict | None:
    """What the newest *evaluation* wrote and how it scored — only an evaluation generates."""
    out = _job_view(lab, jobs_index, 'run', ('rows', 'summary', 'ragas'))
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


def create_inspector_app(lab: LabAccess | None = None) -> FastAPI:
    """The read-only window, reading the lab's records through `lab`.

    Handed the panel's own `InProcessLabAccess` when the two are composed into
    one application (`served_lab.py`), so a mounted Inspector opens no socket.
    Omitted, it reaches the lab `RAGLAB_INSPECTOR_LAB_URL` names over HTTP —
    which is the standalone Inspector, and the only arrangement where the lab
    can be somewhere this process is not."""
    lab = lab if lab is not None else HttpLabAccess()
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

    install_no_store(app)
    install_assets(app, ASSETS)

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
        found = _document(lab.imported_archive(archive_id))
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
        once. The same reason `/api/imported-archives/{id}` asks the lab."""
        found = _document(lab.experiment(experiment_id))
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
        return _answered(lab.experiment_archive(experiment_id),
                         'lab is unavailable; the archive cannot load')

    @app.post('/api/experiments/{experiment_id}/questions')
    def add_recorded_question(experiment_id: str, payload: dict):
        started = _answered(lab.add_experiment_question(experiment_id, payload),
                            'lab is unavailable; question was not added')
        return JSONResponse(started, status_code=202)

    @app.get('/api/experiments/{experiment_id}/questions')
    def recorded_questions(experiment_id: str):
        return _answered(
            lab.experiment_questions(experiment_id),
            'lab is unavailable; recorded questions cannot load')

    @app.get('/api/lab-jobs/{job_id}')
    def lab_job_status(job_id: str):
        return _answered(lab.job(job_id),
                         'lab is unavailable; job status cannot load')

    @app.delete('/api/imported-archives/active')
    def clear_imported_archive():
        return _answered(
            lab.clear_active_archive(),
            'lab is unavailable; archive preview was not cleared')

    @app.get('/api/follow')
    def follow():
        """The lab's newest *finished* jobs in one call, or 'down' when the lab cannot be reached — HTTP 200 either way.

        `lab` is kept, and kept honest, in both modes: it means "could I reach
        the lab", which mounted is answered by the call itself and is therefore
        always `up`. The field stays rather than disappearing in-process
        because the page reads it and a remote Inspector still needs it — a
        reader is owed one bit either way, not a key that exists on some
        installations."""
        active = _document(lab.active_archive())
        archive_id = (active or {}).get('archive_id')
        jobs_index = _document(lab.jobs())
        if jobs_index is None:
            return {'lab': 'down', 'lab_url': lab_base_url(), 'dataset': '',
                    'index': None, 'query': None, 'retrieval': None,
                    'generation': None, 'archive_id': archive_id}

        query_view = _job_view(lab, jobs_index, 'query',
                               ('trace', 'question', 'question_id', 'answer'))
        return {'lab': 'up', 'lab_url': lab_base_url(),
                'dataset': _followed_dataset(jobs_index),
                'index': _newest_chunks(lab, jobs_index), 'query': query_view,
                'retrieval': _question_set(lab, jobs_index),
                'generation': _generation_view(lab, jobs_index),
                'archive_id': archive_id}

    return app

