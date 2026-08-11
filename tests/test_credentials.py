"""The OpenRouter key, entered in the panel and kept nowhere else.

The lab's judged metrics need a model, and the lab is started by hand — so the
key arrived by editing `.env` and restarting the service, which is two steps too
many at the moment you discover the four deciding metrics cannot be measured.
The panel can now take it.

Everything below is about the one property that makes that safe: a credential
this process holds must not reach any of the four places this lab writes to —
`/api/options`, a `.runs/` file, a `raglab.db` row, or the terminal. A secret in
a durable artefact is durable, and this repository's artefacts are meant to be
readable and shareable.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from raglab import credentials, models
from raglab.config import load_lab_settings

KEY = 'sk-or-v1-0123456789abcdef0123456789abcdef'


@pytest.fixture(autouse=True)
def _no_key_survives_a_test():
    """Each test starts and ends with the process holding nothing."""
    credentials.clear()
    yield
    credentials.clear()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_API_KEY', '')
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    from raglab.server import create_app
    return TestClient(create_app())


# This is a unit test.
def test_a_key_is_held_in_the_process_and_read_back_by_the_settings():
    """The point of the store: the settings a run is built from carry the key
    without the environment or any file having changed."""
    settings = load_lab_settings({'RAGLAB_LLM': 'openrouter'})
    assert credentials.apply(settings).openrouter_api_key == ''
    credentials.set_key(KEY)
    assert credentials.apply(settings).openrouter_api_key == KEY
    assert credentials.apply(settings).llm_ready is True


# This is a unit test.
def test_clearing_falls_back_to_whatever_the_environment_had():
    """Clear means "forget what I typed", not "unset the key" — a lab started
    with OPENROUTER_API_KEY in its environment must be exactly as it was."""
    from_env = load_lab_settings({'OPENROUTER_API_KEY': 'sk-or-from-the-shell-0123456789'})
    credentials.set_key(KEY)
    assert credentials.apply(from_env).openrouter_api_key == KEY
    credentials.clear()
    assert credentials.apply(from_env).openrouter_api_key == 'sk-or-from-the-shell-0123456789'


# This is a unit test.
def test_a_key_that_cannot_be_one_is_refused_with_a_readable_reason():
    """A key silently accepted is a run that fails much later, at the first
    model call, with an error about the model rather than about the key."""
    for bad in ('', '   ', 'sk-or-short', 'sk-or-v1 0123456789abcdef0123456789'):
        with pytest.raises(ValueError) as raised:
            credentials.set_key(bad)
        assert 'key' in str(raised.value).lower()
    assert credentials.state(load_lab_settings({}))['set'] is False


# This is a unit test.
def test_the_state_says_set_and_shows_a_hint_but_never_the_key():
    """A masked tail is enough to answer "is this the key I meant?", which is
    the only question the panel has to be able to answer about it."""
    credentials.set_key(KEY)
    state = credentials.state(load_lab_settings({}))
    assert state['set'] is True and state['source'] == 'panel'
    assert state['hint'].endswith(KEY[-4:])
    assert KEY not in json.dumps(state)
    assert state['hint'].count('…') == 1


# This is a unit test.
def test_the_environments_own_key_is_reported_as_the_environments():
    """Where a credential came from decides who can change it: a key from the
    shell is not one this panel put there, and 'Clear' will not remove it."""
    settings = load_lab_settings({'OPENROUTER_API_KEY': 'sk-or-from-the-shell-0123456789'})
    state = credentials.state(settings)
    assert state['set'] is True and state['source'] == 'environment'


# This is a unit test.
def test_setting_a_key_forgets_the_verified_model_lists():
    """`models.openrouter_ids` caches per base url, and with no key the cached
    answer is the empty set — i.e. "nothing is available". Without dropping that
    cache, entering a key left every remote model reading NA until the lab was
    restarted, which is the restart this feature exists to remove."""
    settings = load_lab_settings({})
    models._LIVE[settings.openrouter_base_url] = frozenset()
    credentials.set_key(KEY)
    assert settings.openrouter_base_url not in models._LIVE


# This is an integration test.
def test_the_panel_can_set_a_key_and_the_options_never_carry_it(client):
    """The round trip, and the rule that makes it safe: /api/options is what the
    browser reads on every visit, so the key must not be in it — set-ness and a
    masked tail are, because a panel that cannot say whether a key is in force
    is a panel you check by starting a run."""
    before = client.get('/api/options').json()
    assert before['capabilities']['openrouter_key']['set'] is False

    posted = client.post('/api/credentials', json={'api_key': KEY})
    assert posted.status_code == 200, posted.text
    assert KEY not in posted.text

    body = client.get('/api/options')
    assert KEY not in body.text, 'the key is being served back to the browser'
    state = body.json()['capabilities']['openrouter_key']
    assert state['set'] is True and state['source'] == 'panel'
    assert state['hint'].endswith(KEY[-4:])


# This is an integration test.
def test_a_refused_key_answers_400_with_the_reason(client):
    refused = client.post('/api/credentials', json={'api_key': 'nope'})
    assert refused.status_code == 400
    assert 'key' in refused.json()['detail'].lower()


# This is an integration test.
def test_the_panel_can_take_the_key_back(client):
    client.post('/api/credentials', json={'api_key': KEY})
    cleared = client.request('DELETE', '/api/credentials')
    assert cleared.status_code == 200
    assert cleared.json()['set'] is False
    assert client.get('/api/options').json()[
        'capabilities']['openrouter_key']['set'] is False


# This is an integration test: a real run, a real run file, a real ledger row.
def test_no_artefact_a_run_leaves_behind_contains_the_key(client, tmp_path,
                                                          monkeypatch):
    """The four places this lab writes to, checked against the key itself.

    A credential in a `.runs/` file or a `raglab.db` row is a credential in the
    thing this project treats as shareable evidence — the run files are the
    leaderboard's durable artifact and the ledger is the account of the work.
    Neither has any use for it, so neither may ever hold it.
    """
    from raglab import evaluate
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    client.post('/api/credentials', json={'api_key': KEY})

    started = client.post('/api/evaluations', json={
        'index': {'chunker': 'session', 'embedder': 'token-hash'},
        'retrieval': {'k': 3}, 'generation': {'answerer': 'extractive'},
        'limit': 2, 'ragas_mode': 'off', 'label': 'key-safety'})
    assert started.status_code == 202, started.text
    job_id = started.json()['job_id']
    for _ in range(600):
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] != 'running':
            break
        import time
        time.sleep(0.05)
    assert job['state'] == 'done', job.get('error')

    for path in tmp_path.glob('*.json'):
        assert KEY not in path.read_text(encoding='utf-8'), path
    assert KEY not in client.get('/api/experiments').text
    experiment = client.get('/api/experiments').json()['experiments'][0]
    assert KEY not in client.get(
        '/api/experiments/' + experiment['experiment_id']).text
    assert KEY not in client.get('/api/jobs/' + job_id).text


# This is a configuration invariant.
def test_the_key_is_never_written_to_a_file_by_this_module():
    """Session-only is a promise about code, not about a test run: nothing in
    the credential store may open a file for writing, and no other module may
    reach into it to persist what it holds."""
    from pathlib import Path
    source = (Path(credentials.__file__)).read_text(encoding='utf-8')
    for forbidden in ('open(', 'write_text', 'os.environ[', 'setenv', 'dump('):
        assert forbidden not in source, (
            f'the credential store contains {forbidden!r} — it holds the key in '
            'memory and writes it nowhere')


# This is a configuration invariant.
def test_the_panel_offers_the_key_field_and_never_stores_it_in_the_browser():
    """localStorage survives the tab, is readable by anything served from this
    origin, and is not where a credential goes. The panel remembers the *config*
    on every keystroke — this field has to stay out of that."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    assert 'id="openrouter_key"' in html
    assert '/api/credentials' in html
    assert 'type="password"' in html
    for line in html.splitlines():
        if 'openrouter_key' in line:
            assert 'localStorage' not in line and 'remember(' not in line, line
    # and it is not a field of any config the panel posts with a run
    assert 'openrouter_key:' not in html


# This is a configuration invariant.
def test_the_key_is_a_documented_environment_variable_still():
    """The panel is a second way in, not a replacement: a lab started from a
    shell or a script has no browser to type into."""
    from raglab.config import ROOT
    example = (ROOT / '.env.example').read_text(encoding='utf-8')
    assert 'OPENROUTER_API_KEY' in example
    assert os.environ.get('OPENROUTER_API_KEY') != KEY
