"""Route plumbing shared by the panel and the Inspector (`panel_server.py`, `inspector_server.py`,
composed as one app on :9002 by `served_lab.py`, the Inspector mounted at /inspector):
the frontend folder and the one asset route each service installs over its own
allowlist, the no-store middleware both install, job-acceptance responses,
config screening, the resolved backend a job records, cancellation, progress.
Nothing here may import either service — the dependency points one way, from
the services into this module."""
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from raglab.corpora import dataset_import_contract as datasets
from raglab.llm_backends import model_role_catalogue as models

STATIC = Path(__file__).resolve().parent / 'frontend'


@dataclass(frozen=True)
class Asset:
    """One public frontend file: what is read, as what, and why it is served here.

    `why` is not decoration. Which surface a file belongs to is a project rule
    — the widget's sheet is not the panel's, the tokens are shared so a colour
    cannot drift — and that reasoning used to live in a per-file route's
    docstring. It lives on the entry now, where it is read beside the sharing
    it explains."""
    file: str
    media_type: str | None
    why: str


def install_no_store(app) -> None:
    """The frontend is read from disk on every request, so an edit is live
    the moment it is saved — but `FileResponse` sends no `Cache-Control`,
    which leaves a browser free to reuse a page it already has without ever
    asking. That turns an edited panel into "nothing changed", and the
    reader has no way to tell that from a broken change. A workbench serves
    what is on disk or it is lying about what it is running.

    Installed by each service rather than written by each: a read-only window
    onto evidence must not show yesterday's evidence either, and one
    implementation is what stops the two surfaces disagreeing about that."""

    @app.middleware('http')
    async def never_serve_yesterdays_page(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response


def install_assets(app, assets: dict[str, Asset]) -> None:
    """One route, registered at every path the allowlist names, for a service's
    whole public frontend.

    The allowlist is the thing a reader checks: a file in `frontend/` that is
    not on it is not reachable, so a page has exactly one URL and no request
    path can walk out of the folder. Adding a file is adding a line here, not
    writing an eighteenth route."""

    def asset(request: Request):
        # The path as the allowlist spells it: the Inspector is mounted, so its
        # own '/inspector.css' arrives as '/inspector/inspector.css' with the
        # mount in `root_path`.
        wanted = request.url.path.removeprefix(request.scope.get('root_path', ''))
        entry = assets.get(wanted)
        if entry is None:
            raise HTTPException(404)
        return FileResponse(STATIC / entry.file, media_type=entry.media_type)

    for public_path in assets:
        app.get(public_path)(asset)


def _accepted(job_id: str) -> JSONResponse:
    """202: the work was accepted, not done. Location is the one place the polling url is spelled."""
    return JSONResponse({'job_id': job_id}, status_code=202,
                        headers={'Location': f'/api/jobs/{job_id}'})


def ground_truth_for(cfg, ground_truth: dict) -> dict:
    """The ground truth of the corpus this config names, resolved by id so index and questions match; `ground_truth` is the caller's own fixture, since each service closes over its own copy."""
    if not cfg.index.dataset:
        return ground_truth
    return datasets.load(cfg.index.dataset)[1]


def _find_question(ground_truth: dict, qid) -> dict | None:
    """One ground-truth question, accepting a DOM string for numeric ids."""
    if qid is None:
        return None
    wanted = str(qid)
    return next((question for question in ground_truth['groundtruth_dataset']
                 if str(question.get('groundtruth_question_id')) == wanted), None)


def screen(cfg, run_settings) -> None:
    """Validate a config and check the resolved backend serves what it names; 400s with every problem at once. Not the index-build route's check — a build reads no model."""
    problems = cfg.validate() + models.provider_problems(cfg, run_settings)
    if problems:
        raise HTTPException(400, '; '.join(problems))


class JobCancelled(Exception):
    """A cooperative stop requested from the RAG Lab panel."""


def cancel_checker(cancelled):
    """A `check_cancelled` closure a job's `work` calls before an expensive step; raises the one cancellation signal `Jobs.run` catches."""
    def check_cancelled():
        if cancelled():
            raise JobCancelled()
    return check_cancelled


def _with_backend(cfg, run_settings) -> dict:
    """A job's config, plus the *resolved* backend it runs on — never the payload's possibly-blank request."""
    return cfg.to_dict() | {'provider': run_settings.provider}


def scaled_progress(report, factor: float):
    """A progress callback that reports into the front `factor` share of the bar, for a stage's implicit index build so it doesn't all happen on 'starting 0%'."""
    return lambda stage, fraction, detail='': report(stage, factor * fraction, detail)
