"""Plumbing for the browser suite: a lab of its own, on a port of its own.

Only the `browser`-marked journeys ask for these fixtures, and pytest builds a
fixture only when a test requests it, so this file costs the rest of the
dashboard suite nothing and imports no Playwright of its own — the browser
tests skip themselves at import when the extra is absent.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

#: How long the child process gets to bind its port and answer once, and how
#: long a browser step may take before it is called a failure. Generous
#: because CI runners are slow; never a `sleep`, always a poll.
BOOT_TIMEOUT = 60.0
STEP_TIMEOUT = 20_000


def _free_port() -> int:
    """A port the operating system says is free right now.

    The lab's own :9002 is deliberately not used: the suite must not be able
    to reach — or be confused for — the developer's running daemon.
    """
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def _wait_until_serving(process: subprocess.Popen, base_url: str) -> None:
    """Poll the child until it answers, or fail with whatever it printed."""
    deadline = time.monotonic() + BOOT_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f'the lab exited before serving:\n{process.communicate()[0]}')
        try:
            if httpx.get(f'{base_url}/api/health', timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    process.kill()
    raise RuntimeError(f'the lab did not answer on {base_url} in {BOOT_TIMEOUT}s')


#: What the developer owns and no browser test may touch. Snapshotted before
#: the suite's own lab boots and compared again when it stops.
_ROOT = Path(__file__).resolve().parents[3].parent
REAL_PATHS = (_ROOT / 'databases', _ROOT / '.runs', _ROOT / '.datasets')


def _snapshot(paths=REAL_PATHS) -> dict[str, int]:
    """Name and size of every file under the developer's durable directories."""
    return {str(item): item.stat().st_size
            for path in paths if path.exists()
            for item in sorted(path.rglob('*')) if item.is_file()}


@pytest.fixture(scope='session')
def _the_developers_lab_stays_untouched():
    """Fail the session if the browser suite wrote where the developer lives.

    `lab_server` depends on this, so the snapshot is always taken before the
    child process exists and compared after it is gone. If this fails while
    the developer's own lab is serving on :9002, the daemon wrote the
    difference — stop it and run again.
    """
    before = _snapshot()
    yield before
    changed = {name for name in set(before) | set(_snapshot())
               if before.get(name) != _snapshot().get(name)}
    assert not changed, (
        'the browser suite changed the developer\'s durable files: '
        f'{sorted(changed)} — or a lab was serving on :9002 while it ran')


@pytest.fixture(scope='session')
def lab_home(tmp_path_factory) -> Path:
    """The temporary directory that stands in for everything durable."""
    return tmp_path_factory.mktemp('browser-lab')


@pytest.fixture(scope='session')
def lab_server(_the_developers_lab_stays_untouched, lab_home: Path):
    """A lab in a child process, on an ephemeral port, writing only to `lab_home`.

    Session-scoped on purpose: an index build is the expensive step, and every
    journey that only reads can share one. Journeys that write name their own
    dataset, experiment or thread instead of demanding a clean server, so
    their order never matters.
    """
    port = _free_port()
    base_url = f'http://127.0.0.1:{port}'
    env = {
        **os.environ,
        # The four durable paths, redirected exactly as the offline suite
        # redirects them — nothing the developer owns is reachable from here.
        'RAGLAB_DB': str(lab_home / 'raglab.db'),
        'RAGLAB_WIDGET_DB': str(lab_home / 'widget.db'),
        'RAGLAB_CORPORA_DB': str(lab_home / 'corpora.db'),
        'RAGLAB_BROWSER_RUNS': str(lab_home / 'runs'),
        # The fifth durable place, and the one the offline suite redirects
        # per-test rather than per-session: a dataset imported through the
        # page lands here, and the repo's own `.datasets/` must not be it.
        'RAGLAB_DATASETS': str(lab_home / 'datasets'),
        # The Inspector asks the lab about experiments and jobs over HTTP, and
        # its default is :9002 — the developer's own daemon. Pointing it at
        # this lab is what keeps a record-mode journey talking to the
        # experiments this suite actually recorded.
        'RAGLAB_INSPECTOR_LAB_URL': base_url,
        'RAGLAB_BROWSER_PORT': str(port),
        # No live model, and no credential for one. The blanks are set rather
        # than removed because `load_env_file` uses `setdefault`: a name that
        # is already present, even empty, is a name the repo's `.env` cannot
        # fill in.
        'RAGLAB_LLM': 'fake',
        'BRAIN_LLM': 'fake',
        'OPENROUTER_API_KEY': '',
        'OPENAI_API_KEY': '',
        'LANGSMITH_API_KEY': '',
        'LANGSMITH_TRACING': 'false',
        # Saving a key makes the lab ask OpenRouter which models it serves.
        # That is the one call this suite could make to the internet, so it is
        # pointed at a closed local port: the probe fails at once, offline,
        # and the catalogue reports what an installation without OpenRouter
        # really has.
        'OPENROUTER_BASE_URL': 'http://127.0.0.1:1',
    }
    env.pop('RAGLAB_MODEL', None)
    process = subprocess.Popen(
        [sys.executable, '-m', 'raglab.dashboard.tests.browser_lab_server'],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_until_serving(process, base_url)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


#: The corpus every browser journey runs on: five sessions, six questions, and
#: a hashing embedder, so a build costs a fraction of a second and downloads
#: nothing. No browser journey claims anything about the diary.
SMOKE_INDEX = {'dataset': 'smoke-mini', 'chunker': 'session',
               'embedder': 'token-hash'}


def finish_job(base_url: str, job_id: str, timeout: float = 60.0) -> dict:
    """Poll a job to its terminal state, the way both frontends do."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = httpx.get(f'{base_url}/api/jobs/{job_id}', timeout=10.0).json()
        if job['state'] not in ('running', 'cancelling'):
            return job
        time.sleep(0.02)
    raise AssertionError(f'job {job_id} still running after {timeout}s')


def start_job(base_url: str, path: str, body: dict) -> dict:
    """Start a job over HTTP and return it once it has stopped."""
    started = httpx.post(f'{base_url}{path}', json=body, timeout=30.0)
    assert started.status_code == 202, started.text
    job = finish_job(base_url, started.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    return job


@pytest.fixture(scope='session')
def a_recorded_experiment(lab_server) -> str:
    """One finished evaluation, so the board and the Inspector have a row.

    Built through the lab's own HTTP surface rather than through the panel,
    because what these journeys are about is what the *reader* can then do
    with a recorded experiment — the making of it is setup, not the claim.
    Session-scoped: the ledger row and its run file are read, never rewritten.
    """
    start_job(lab_server, '/api/indexes', {'index': dict(SMOKE_INDEX)})
    job = start_job(lab_server, '/api/evaluations', {
        'index': dict(SMOKE_INDEX),
        'retrieval': {'k': 2, 'reranker': 'none', 'grader': 'none'},
        'generation': {'answerer': 'extractive'},
        'ragas_mode': 'off', 'limit': 2})
    return job['result']['run_id']


@pytest.fixture
def panel(lab_server, page):
    """The panel at `/`, loaded and ready."""
    return _surface(page, lab_server, '/')


@pytest.fixture
def inspector(lab_server, page):
    """The Inspector at `/inspector`, loaded and ready."""
    return _surface(page, lab_server, '/inspector')


@pytest.fixture
def board(lab_server, page):
    """The leaderboard at `/leaderboard`, loaded and ready."""
    return _surface(page, lab_server, '/leaderboard')


def _surface(page, base_url: str, path: str):
    page.set_default_timeout(STEP_TIMEOUT)
    page.goto(f'{base_url}{path}')
    return page
