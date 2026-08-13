"""Route plumbing shared by the two served apps (`server.py` :9002, `inspector.py` :9003).

Presentation helpers (gold marking, chunk rows) live in `present.py`; this
module is the other, unrelated concern the two services independently grew —
job-acceptance responses, config screening, cancellation and progress
wiring. Nothing here may import either service: the dependency points one
way, from the services into this module.
"""
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from . import datasets, models


def _accepted(job_id: str) -> JSONResponse:
    """202: the work was accepted, not done. Location is the one place the polling url is spelled."""
    return JSONResponse({'job_id': job_id}, status_code=202,
                        headers={'Location': f'/api/jobs/{job_id}'})


def ground_truth_for(cfg, ground_truth: dict) -> dict:
    """The ground truth of the corpus this config names — resolved by id, so index and questions match.

    `ground_truth` is the caller's built-in fixture, passed explicitly since
    each service closes over its own copy loaded at startup.
    """
    if not cfg.index.dataset:
        return ground_truth
    return datasets.load(cfg.index.dataset)[1]


def screen(cfg, run_settings) -> None:
    """Validate a config and check the resolved backend serves what it names; 400s with every problem at once.

    Not the index-build route's check: a build reads no model, so that route
    keeps its own filtered call to `cfg.validate()` alone rather than this.
    """
    problems = cfg.validate() + models.provider_problems(cfg, run_settings)
    if problems:
        raise HTTPException(400, '; '.join(problems))


def cancel_checker(cancelled, exc):
    """A `check_cancelled` closure: raises `exc()` when `cancelled()` is true.

    The cooperative-cancellation check a job's `work` closure calls before an
    expensive step. `exc` is passed in rather than imported, so this module
    never needs the caller's own cancellation exception type.
    """
    def check_cancelled():
        if cancelled():
            raise exc()
    return check_cancelled


def scaled_progress(report, factor: float):
    """A progress callback that reports into the front `factor` share of the bar.

    For a stage's implicit index build, which is the long silent part — hand
    it the front of the bar, or it all happens on 'starting 0%'.
    """
    return lambda stage, fraction, detail='': report(stage, factor * fraction, detail)
