"""Suite-wide guards, autouse so no test has to remember them."""
import os
from pathlib import Path

import pytest

import raglab

RAGLAB_DIR = Path(raglab.__file__).resolve().parent


@pytest.fixture(scope='module')
def client():
    """A TestClient over the lab's own FastAPI app — shared by the service
    tests and the panel tests that read the same served pages."""
    from fastapi.testclient import TestClient

    from raglab.server import create_app
    return TestClient(create_app())


@pytest.fixture(autouse=True, scope='session')
def _the_lab_suite_does_not_read_the_machine():
    """Force `fake`, so the suite doesn't pass or fail depending on whether
    Ollama happens to be running (`LabSettings.llm_provider` defaults to
    `ollama`). A test that wants a real backend states it explicitly."""
    saved = {key: os.environ.get(key)
             for key in ('RAGLAB_LLM', 'RAGLAB_MODEL', 'BRAIN_LLM')}
    os.environ['RAGLAB_LLM'] = 'fake'
    os.environ['BRAIN_LLM'] = 'fake'
    os.environ.pop('RAGLAB_MODEL', None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True, scope='session')
def _the_experiment_ledger_is_never_the_real_one(tmp_path_factory):
    """No test may write into the lab's real `databases/test/raglab.db` —
    every job records itself, so any test that builds or scores would
    otherwise deposit a row in the durable ledger. An env var rather than a
    patched attribute: `ledger.db_path()` resolves it per call, so this
    fixture needs no import of the lab module it guards."""
    saved = os.environ.get('RAGLAB_DB')
    os.environ['RAGLAB_DB'] = str(
        tmp_path_factory.mktemp('raglab-ledger') / 'raglab.db')
    yield
    if saved is None:
        os.environ.pop('RAGLAB_DB', None)
    else:
        os.environ['RAGLAB_DB'] = saved


@pytest.fixture(autouse=True, scope='session')
def _runs_dir_is_never_the_real_one(tmp_path_factory):
    """No test may write into the lab's real `.runs/` — `evaluate.run_eval`
    ends in `save_run`, so any test that evaluates anything would otherwise
    deposit a JSON run file there. Session-scoped so a per-test
    `monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)` still wins.
    `config.RUNS_DIR` is left pointing at the real path on purpose: it is what
    the invariant test compares against."""
    from raglab import evaluate, sweep
    runs = tmp_path_factory.mktemp('raglab-runs')
    saved = {module: module.RUNS_DIR for module in (evaluate, sweep)}
    for module in saved:
        module.RUNS_DIR = runs
    yield
    for module, original in saved.items():
        module.RUNS_DIR = original
