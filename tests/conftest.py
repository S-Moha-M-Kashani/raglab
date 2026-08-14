"""Suite-wide guards, autouse so no test has to remember them, plus the
fixtures and settings shared across more than one test file."""
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

import raglab
from raglab import corpus
from raglab.config import IndexConfig, LabSettings
from raglab.index import IndexRegistry

RAGLAB_DIR = Path(raglab.__file__).resolve().parent

LAB_SETTINGS = LabSettings(openrouter_api_key='', llm_provider='fake')

# The local backend's own settings, read by the service tests (which check
# /api/options against it) and by the provider tests (which build it).
OLLAMA_SETTINGS = replace(LAB_SETTINGS, llm_provider='ollama',
                          llm_model='gemma4:e2b')

# Each of these embedded a Farsi sentence here, through the backend it names —
# read by the catalogue tests that define the claim and by the service tests
# that check `/api/options` reports the same models.
REQUESTED_MODELS = {
    'heydariAI/persian-embeddings': ('sentence-transformers', 1024, 'open'),
    'intfloat/multilingual-e5-small': ('sentence-transformers', 384, 'open'),
}


@pytest.fixture(scope='module')
def client():
    """A TestClient over the lab's own FastAPI app — shared by the service
    tests and the panel tests that read the same served pages."""
    from fastapi.testclient import TestClient

    from raglab.server import create_app
    return TestClient(create_app())


@pytest.fixture(scope='module')
def diary():
    return corpus.load_diary()


@pytest.fixture(scope='module')
def ground_truth():
    return corpus.load_ground_truth()


@pytest.fixture(scope='module')
def registry(diary):
    return IndexRegistry(LAB_SETTINGS, diary)


@pytest.fixture(scope='module')
def index(registry):
    return registry.get(IndexConfig(chunker='semantic-drift', embedder='char-hash',
                                    contextual=True))


@pytest.fixture(scope='module')
def session(diary):
    return next(s for s in diary['sessions'] if len(s['messages']) >= 6)


def _finished(client, job_id: str, timeout: float = 30.0) -> dict:
    """Poll a job to its terminal state, the way both frontends do."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] not in ('running', 'cancelling'):
            return job
        time.sleep(0.01)
    raise AssertionError(f'job {job_id} still running after {timeout}s')


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
    from raglab import evaluate
    from raglab.llm_tools import sweep
    runs = tmp_path_factory.mktemp('raglab-runs')
    saved = {module: module.RUNS_DIR for module in (evaluate, sweep)}
    for module in saved:
        module.RUNS_DIR = runs
    yield
    for module, original in saved.items():
        module.RUNS_DIR = original
