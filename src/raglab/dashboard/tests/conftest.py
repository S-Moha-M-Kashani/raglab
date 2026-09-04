"""Plumbing the dashboard's tests share: the served texts the route tests read,
and a lab of its own on a port of its own for the browser suite.

Only the `browser`-marked journeys ask for the lab fixtures, and pytest builds
a fixture only when a test requests it, so they cost the rest of the dashboard
suite nothing and import no Playwright of their own — the browser tests skip
themselves at import when the extra is absent.
"""
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from raglab.conftest import RAGLAB_DIR

# --- the served panel's conventions, as one table ---------------------------

@pytest.fixture(scope='module')
def panel_texts(client):
    """Every named text the convention table below checks, fetched the one
    way a browser actually reaches it (`client.get`) — a second disk read of
    the same file would be a claim about a copy nobody is served. Several
    entries are carved out of the full page, css and script, because their
    claim is *where* the text sits rather than merely that it exists
    somewhere on the page — the same regions the retired pin tests scoped
    their own reads to. The panel's route modules are the entries read from
    disk: the lab's Python source is never served, so there is no route to
    prefer over it."""
    html = client.get('/').text
    css = client.get('/panel.css').text
    js = client.get('/panel.js').text
    tokens = client.get('/tokens.css').text

    embed_label = re.search(r'<label>Embedding model.*?</label>', html, re.S)
    model_card = re.search(r'<section[^>]*id="modelCard".*?</section>', html, re.S)
    assert embed_label and model_card, 'the panel dropped a section this table reads'

    return {
        'index.html': html,
        # The shared scale, fetched over its own route because both pages link
        # it before their own sheet — a disk read would be a claim about a copy
        # nobody is served.
        'tokens.css': tokens,
        # The shared chrome sheet, over its own route for the same reason as
        # tokens.css. It now holds the table component every table on either
        # surface is built against, so its contract is checked here beside the
        # markup that depends on it.
        'chrome.css': client.get('/chrome.css').text,
        # The leaderboard surface, served by this same lab: the ranking moved
        # off the lab page, so the rows that guard what a ranking must say
        # follow it here rather than being deleted with the old board.
        'leaderboard.html': client.get('/leaderboard').text,
        'leaderboard.js': client.get('/leaderboard.js').text,
        # The corpus viewer, served by this same lab: it reads the dataset the
        # knobs are measured against, so the rows that guard a reader surface —
        # both themes, no step ink, the shared table component — follow it here
        # rather than being written a third time.
        'dataset.html': client.get('/dataset').text,
        'dataset.js': client.get('/dataset.js').text,
        'dataset.css': client.get('/dataset.css').text,
        # The row filter, over its own route: it is the leaderboard's, but it
        # reads a cell with the shared sorter's parser, so what it can be asked
        # is a claim about that pair of files rather than about this page.
        'filtertable.js': client.get('/filtertable.js').text,
        # The script all three pages load before their own, over its own route
        # for the same reason as tokens.css. What both surfaces turned out to
        # need identically lives in it, so a claim about "one implementation"
        # is a claim about this file.
        'lab.js': client.get('/lab.js').text,
        'panel.css': css,
        'panel.js': js,
        'index.html (embedding-model label)': embed_label.group(0),
        'index.html (modelCard section)': model_card.group(0),
        # The widget's own stylesheet and script — whole files, served from
        # the root to all three surfaces, so what the rows below claim about
        # the helper is a claim about these two files and nothing else. They
        # were once slices carved out of panel.css/panel.js and kept those
        # names for a while after they stopped being slices, which sent a
        # maintainer chasing a widget failure into the Laboratory's own
        # script; one key per file is what stops that.
        # The codec, over its own route: what the two knob-coverage tests at
        # the foot of this file claim about the export template is a claim
        # about the file a browser is actually handed.
        'archive_io.js': client.get('/archive_io.js').text,
        # The handoff, over its own route for the same reason: it reads the
        # codec's own `BUILTIN_DATASET` rather than keeping a second copy of
        # that id, which is a claim about this served file.
        'experiment_handoff.js': client.get('/experiment_handoff.js').text,
        'widget.css': client.get('/widget.css').text,
        'widget.js': client.get('/widget.js').text,
        # The panel's route modules, one entry each, read from disk: the lab's
        # Python source is never served, so there is no route to prefer over
        # it. One key per module rather than one for the whole service, so a
        # row that claims a route exists names the section it belongs to.
    } | {f'routes/{module.name}': module.read_text(encoding='utf-8')
         for module in sorted(
             (RAGLAB_DIR / 'dashboard' / 'routes').glob('*.py'))}

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
        # The fifth durable place, redirected here for the same reason and
        # in the same way as the four above: a dataset imported through the
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
SMOKE_INDEX = {'dataset': 'smoke-mini', 'split_plan': [{'kind': 'document'}],
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


@pytest.fixture
def dataset_page(lab_server, page):
    """The corpus viewer at `/dataset`, on the smoke corpus.

    Named in the URL rather than left to the catalogue's first entry, because
    which corpus opens by default is not what any journey here is about — and
    every claim below is about five documents that can be checked by eye.
    """
    return _surface(page, lab_server, '/dataset?dataset=smoke-mini')


def set_plan(page, *kinds: str):
    """Put the split plan control at `document` plus one stage per kind, added
    the way a reader adds them — through the dropdown under the rows.

    Every stage after the document is removed first, so a journey states the
    whole plan it wants rather than what it hopes was there before."""
    remove = page.locator('#split_plan [data-plan="remove"]')
    while remove.count():
        remove.first.click()
    for kind in kinds:
        page.select_option('#plan-add', kind)


def _surface(page, base_url: str, path: str):
    page.set_default_timeout(STEP_TIMEOUT)
    page.goto(f'{base_url}{path}')
    return page
