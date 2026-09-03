"""Route plumbing shared by the panel and the Inspector (`panel_server.py`, `inspector_server.py`,
composed as one app on :9002 by `served_lab.py`, the Inspector mounted at /inspector):
the frontend folder and the one asset route each service installs over its own
allowlist, the no-store middleware both install, job-acceptance responses,
config screening, the resolved backend a job records, cancellation, progress,
and the seam the Inspector reads the lab's records through.
Nothing here may import either service — the dependency points one way, from
the services into this module."""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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


# What a lab operation answers with. Three outcomes and not two: the document
# the lab returned, a refusal it stated (its own status and its own words), or
# `None` — it could not be reached at all. A reader told the wrong one of the
# last two is told "this experiment does not exist" about a lab that is merely
# down, or "the lab is down" about one that answered.
LabReply = dict | tuple[int, str] | None


class LabAccess(Protocol):
    """The nine things the Inspector asks of the lab, and nothing else.

    The Inspector owns no ledger, no archive store and no job recorder: every
    record it shows is the lab's own answer, and every write reachable from it
    is the lab's own write. That rule is about *ownership*, so it is satisfied
    equally by calling the lab's function and by calling the lab's route —
    which is what the two implementations of this protocol are.

    Nine named operations rather than the panel's application, which would
    work and would also put every route the panel has, the writing ones
    included, one attribute away from a read-only surface. A boundary you can
    count is a boundary a convention test can pin."""

    def imported_archive(self, archive_id: str) -> LabReply: ...
    def active_archive(self) -> LabReply: ...
    def clear_active_archive(self) -> LabReply: ...
    def experiment(self, experiment_id: str) -> LabReply: ...
    def experiment_archive(self, experiment_id: str) -> LabReply: ...
    def experiment_questions(self, experiment_id: str) -> LabReply: ...
    def add_experiment_question(self, experiment_id: str,
                                payload: dict) -> LabReply: ...
    def job(self, job_id: str) -> LabReply: ...
    def jobs(self) -> LabReply: ...


def _stated(handler: Callable, *args) -> LabReply:
    """One lab handler's answer as a value: its document, or the refusal it raised.

    Two exception types and not one, because the panel states its refusals in
    two ways. `HTTPException` is the obvious one. `ValueError` is the other:
    `create_app` registers a handler for it that answers 400 with the message
    as `detail`, which is how an experiment whose imported dataset was since
    deleted refuses. Over HTTP both arrive as a status and words; catching only
    the first would make the second a bare 500 in-process, and the reader's
    account of what went wrong would depend on how the Inspector was mounted.
    `ArchiveStoreError` subclasses `ValueError`, so it travels this way too.

    Nothing wider: a genuine bug must stay a 500 on both sides, because a
    workbench that answers 400 to its own broken code is lying about whose
    fault it is. Unavailability has no counterpart in-process — a function
    call either answers or refuses."""
    try:
        return handler(*args)
    except HTTPException as refusal:
        return refusal.status_code, str(refusal.detail)
    except ValueError as bad_request:
        return 400, str(bad_request)


class InProcessLabAccess:
    """The nine operations, answered by calling the lab's own handlers.

    Built by the panel out of the functions its routes are, and handed to a
    mounted Inspector at mount time. No socket, no port and no host name: the
    two halves are one process, and a loopback request between them would cost
    a worker thread on each side of a call the process is making to itself."""

    def __init__(self, *, imported_archive: Callable, active_archive: Callable,
                 clear_active_archive: Callable, experiment: Callable,
                 experiment_archive: Callable, experiment_questions: Callable,
                 add_experiment_question: Callable, job: Callable,
                 jobs: Callable):
        self._imported_archive = imported_archive
        self._active_archive = active_archive
        self._clear_active_archive = clear_active_archive
        self._experiment = experiment
        self._experiment_archive = experiment_archive
        self._experiment_questions = experiment_questions
        self._add_experiment_question = add_experiment_question
        self._job = job
        self._jobs = jobs

    def imported_archive(self, archive_id: str) -> LabReply:
        return _stated(self._imported_archive, archive_id)

    def active_archive(self) -> LabReply:
        return _stated(self._active_archive)

    def clear_active_archive(self) -> LabReply:
        return _stated(self._clear_active_archive)

    def experiment(self, experiment_id: str) -> LabReply:
        return _stated(self._experiment, experiment_id)

    def experiment_archive(self, experiment_id: str) -> LabReply:
        return _stated(self._experiment_archive, experiment_id)

    def experiment_questions(self, experiment_id: str) -> LabReply:
        return _stated(self._experiment_questions, experiment_id)

    def add_experiment_question(self, experiment_id: str,
                                payload: dict) -> LabReply:
        return _stated(self._add_experiment_question, experiment_id, payload)

    def job(self, job_id: str) -> LabReply:
        return _stated(self._job, job_id)

    def jobs(self) -> LabReply:
        return _stated(self._jobs)


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
