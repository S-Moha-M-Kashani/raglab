"""Suite-wide guards, autouse so no test has to remember them, plus the
fixtures and settings shared across more than one test file."""
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

import raglab
from raglab import corpus, models
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


# The smoke corpus (`fixtures/groundtruth_datasets/smoke-mini.json`, 5
# sessions, 6 questions) with `token-hash`, which needs no model download —
# every integration test that needs *an* index rather than specifically the
# 167-session Farsi diary reaches for this instead of building the big one.
SMOKE_INDEX = {'dataset': 'smoke-mini', 'chunker': 'session',
               'embedder': 'token-hash'}


class _SmokeIndex:
    """`.config` is a fresh `dict` on every access — a new copy each time, so
    one consumer's `smoke_index.config['limit'] = 5` (or `{**smoke_index.config,
    ...}`) cannot corrupt what the next test reads — to POST at
    `/api/indexes` / `/api/evaluations`. `.index` is the built `LabIndex`
    (`.index.cfg.fingerprint()`, `.index.stats.collection`) for tests that
    call the registry directly instead of going through HTTP. No `.registry`:
    `IndexRegistry.get` mutates its cached `LabIndex` in place
    (`stats.reused = True`), so handing out the registry that built `.index`
    alongside `.index` itself would let one test's reuse-checking build flip
    `.index.stats.reused` for every test that reads it afterwards, for the
    rest of the run. A test that needs registry-reuse semantics builds its
    own registry locally, the way step 6's sequence test does."""

    def __init__(self, index):
        self.index = index

    @property
    def config(self) -> dict:
        return dict(SMOKE_INDEX)


@pytest.fixture(scope='session')
def smoke_index():
    """Built at most once per suite run. The brief calls for module scope;
    `scope='session'` is the stricter reading of that same bound, and safe
    because nothing here mutates the built index in place (see `_SmokeIndex`
    for why `.registry` is deliberately not part of the shape). Asserts the
    build actually produced the five expected chunks, so a load-bearing
    fixture nine later tasks depend on fails at setup rather than quietly
    handing out a broken index."""
    from raglab.index import IndexRegistry as _Registry
    built = _Registry(LAB_SETTINGS).get(IndexConfig(**SMOKE_INDEX))
    assert built.stats.chunks == 5, 'the smoke set is five sessions'
    return _SmokeIndex(built)


def _finished(client, job_id: str, timeout: float = 30.0) -> dict:
    """Poll a job to its terminal state, the way both frontends do."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] not in ('running', 'cancelling'):
            return job
        time.sleep(0.01)
    raise AssertionError(f'job {job_id} still running after {timeout}s')


def drain_jobs(client) -> None:
    """Poll every job this client's app still has running to a terminal
    state. The job table allows exactly one job at a time (`Jobs.start`'s
    lock), so a test that starts one and does not wait for it leaves it
    running for whichever test asks next — this is the `finally` (or
    end-of-test) call that prevents that leak."""
    for job in client.get('/api/jobs').json()['jobs']:
        if job['state'] in ('running', 'cancelling'):
            _finished(client, job['id'])


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


#: The developer-secret variables no test may read from a real `.env`.
#: `OPENROUTER_API_KEY` was the original hole; `LANGSMITH_TRACING` and
#: `LANGSMITH_API_KEY` were added once the widget's restore re-sanctioned
#: those four `.env` variables — `LANGSMITH_TRACING=true` makes tracing
#: process-global the first time anything builds a LangChain/LangGraph
#: object (the widget's own agent, or any of `tests/test_agent.py`'s real
#: LangGraph invocations), and it is set the same `setdefault` way as the
#: OpenRouter key, so it is exactly the same class of leak.
_DEVELOPER_SECRET_ENV = ('OPENROUTER_API_KEY', 'LANGSMITH_TRACING',
                         'LANGSMITH_API_KEY')


@pytest.fixture(autouse=True, scope='session')
def _no_test_reads_the_developers_real_key():
    """`settings.load_env_file` uses `os.environ.setdefault`, so a real
    `OPENROUTER_API_KEY` sitting in a developer's `.env` leaks into the
    process the first time anything calls `load_lab_settings` — and stays
    there, because nothing unsets it. The `RAGLAB_LLM=fake` pin above does not
    close this: `models.openrouter_ids` short-circuits on the *key*, not the
    provider, so a route that only builds a model catalogue (`/api/options`
    and friends) reaches the network under `fake` too.

    This has to be session-scoped, not per test, and that is not a style
    choice: `client` (above) is a *module*-scoped fixture, and pytest always
    finishes instantiating a broader-scoped fixture before a narrower one for
    the same test, regardless of declaration order — so a function-scoped
    `monkeypatch.setenv` here would still run *after* the first test in each
    module had already called `create_app()` → `load_lab_settings()` with the
    real key sitting in `os.environ`, protecting nothing. Measured: with the
    pin at function scope, `test_server.py`, `test_panel.py` and
    `test_raglab.py` each still made a real `httpx.get` to
    `https://openrouter.ai/api/v1/models` carrying the `.env` key, because
    their module-scoped `client` had already baked it into `boot_settings`
    before the per-test fixture ever ran. Session scope closes that: it is
    set before *any* fixture of any narrower scope, in any module, ever runs.
    `test_credentials.py` still proves the panel's key wins over the
    environment's: that file drives `credentials.set_key` and passes its own
    explicit env dicts to `load_lab_settings`, neither of which this fixture
    touches, so its premise (a *typed* key beats an *environment* key)
    survives an environment pinned to no key at all.

    Session scope only closes the *module-scope* hazard above; within one
    scope, pytest still instantiates autouse fixtures in roughly declaration
    order, so this being the *last* of the four session-scoped guards in this
    file is itself load-bearing — a future session-scoped autouse fixture
    declared *above* it that calls `create_app()` or `load_lab_settings()`
    would run before this pin took effect and reopen exactly the hole it
    closes. It is safe today only because none of the other three does.

    `LANGSMITH_TRACING` and `LANGSMITH_API_KEY` are pinned alongside the
    OpenRouter key for the same reason, added once the widget's restore put
    those variables back in `.env.example`: LangSmith's SDK batches traces on
    its own background thread over `requests`, entirely outside the `httpx`
    seam every other network guard in this suite watches, so a developer's
    real `LANGSMITH_TRACING=true` would trace the ~10 real LangGraph
    invocations in `tests/test_agent.py` (and any widget test that reaches
    the OpenRouter agent path) to a real LangSmith project with nothing here
    to notice. Pinning `LANGSMITH_TRACING` empty (falsy) is what actually
    stops the SDK from tracing at all; `LANGSMITH_API_KEY` is pinned too so a
    tracing call that somehow still fires cannot carry a real credential."""
    saved = {name: os.environ.get(name) for name in _DEVELOPER_SECRET_ENV}
    for name in _DEVELOPER_SECRET_ENV:
        os.environ[name] = ''
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def _models_live_cache_never_survives_between_tests():
    """`models._LIVE` caches availability per base url and, before this fix,
    was never reset suite-wide — so the *first* test in the whole run to hit
    a catalogue-building route (`/api/options` and friends) paid for one real
    probe of `http://localhost:11434/api/tags` (the `ollama` mode's own
    catalogue entry, built regardless of which backend is actually active)
    and every test after it, in every file, silently read that one test's
    cached answer instead of asking again. Function-scoped, cleared before
    and after, so which test happens to run first can no longer change what
    any other test observes."""
    models._LIVE.clear()
    yield
    models._LIVE.clear()
