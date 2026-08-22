"""Route plumbing shared by the panel and the Inspector (`panel_server.py`, `inspector_server.py`,
composed as one app on :9002 by `served_lab.py`, the Inspector mounted at /inspector):
job-acceptance responses, config screening, cancellation, progress. Nothing here may import
either service — the dependency points one way, from the services into this module."""
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from raglab.corpora import dataset_import_contract as datasets
from raglab.llm_backends import model_role_catalogue as models


def _accepted(job_id: str) -> JSONResponse:
    """202: the work was accepted, not done. Location is the one place the polling url is spelled."""
    return JSONResponse({'job_id': job_id}, status_code=202,
                        headers={'Location': f'/api/jobs/{job_id}'})


def ground_truth_for(cfg, ground_truth: dict) -> dict:
    """The ground truth of the corpus this config names, resolved by id so index and questions match; `ground_truth` is the caller's own fixture, since each service closes over its own copy."""
    if not cfg.index.dataset:
        return ground_truth
    return datasets.load(cfg.index.dataset)[1]


def screen(cfg, run_settings) -> None:
    """Validate a config and check the resolved backend serves what it names; 400s with every problem at once. Not the index-build route's check — a build reads no model."""
    problems = cfg.validate() + models.provider_problems(cfg, run_settings)
    if problems:
        raise HTTPException(400, '; '.join(problems))


def cancel_checker(cancelled, exc):
    """A `check_cancelled` closure a job's `work` calls before an expensive step; `exc` is passed in so this module needs no caller's exception type."""
    def check_cancelled():
        if cancelled():
            raise exc()
    return check_cancelled


def scaled_progress(report, factor: float):
    """A progress callback that reports into the front `factor` share of the bar, for a stage's implicit index build so it doesn't all happen on 'starting 0%'."""
    return lambda stage, fraction, detail='': report(stage, factor * fraction, detail)
