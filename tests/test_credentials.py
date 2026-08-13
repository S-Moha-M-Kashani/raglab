"""The OpenRouter key, entered in the panel and held nowhere else — never in
`/api/options`, a `.runs/` file, a `raglab.db` row, or the terminal."""
import json
import os

import pytest
from fastapi.testclient import TestClient

from raglab import credentials, models
from raglab.config import load_lab_settings

KEY = 'sk-or-v1-0123456789abcdef0123456789abcdef'


@pytest.fixture(autouse=True)
def _no_key_survives_a_test():
    credentials.clear()
    yield
    credentials.clear()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENROUTER_API_KEY', '')
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    from raglab.server import create_app
    return TestClient(create_app())


def test_a_key_is_held_in_the_process_and_read_back_by_the_settings():
    settings = load_lab_settings({'RAGLAB_LLM': 'openrouter'})
    assert credentials.apply(settings).openrouter_api_key == ''
    credentials.set_key(KEY)
    assert credentials.apply(settings).openrouter_api_key == KEY
    assert credentials.apply(settings).llm_ready is True


def test_clearing_falls_back_to_whatever_the_environment_had():
    """Clear means "forget what I typed", not "unset the key"."""
    from_env = load_lab_settings({'OPENROUTER_API_KEY': 'sk-or-from-the-shell-0123456789'})
    credentials.set_key(KEY)
    assert credentials.apply(from_env).openrouter_api_key == KEY
    credentials.clear()
    assert credentials.apply(from_env).openrouter_api_key == 'sk-or-from-the-shell-0123456789'


def test_a_key_that_cannot_be_one_is_refused_with_a_readable_reason():
    for bad in ('', '   ', 'sk-or-short', 'sk-or-v1 0123456789abcdef0123456789'):
        with pytest.raises(ValueError) as raised:
            credentials.set_key(bad)
        assert 'key' in str(raised.value).lower()
    assert credentials.state(load_lab_settings({}))['set'] is False


def test_the_state_says_set_and_shows_a_hint_but_never_the_key():
    credentials.set_key(KEY)
    state = credentials.state(load_lab_settings({}))
    assert state['set'] is True and state['source'] == 'panel'
    assert state['hint'].endswith(KEY[-4:])
    assert KEY not in json.dumps(state)
    assert state['hint'].count('…') == 1


def test_the_environments_own_key_is_reported_as_the_environments():
    """A key from the shell is not one the panel's 'Clear' can remove."""
    settings = load_lab_settings({'OPENROUTER_API_KEY': 'sk-or-from-the-shell-0123456789'})
    state = credentials.state(settings)
    assert state['set'] is True and state['source'] == 'environment'


def test_setting_a_key_forgets_the_verified_model_lists():
    """`models._LIVE` caches availability per base url; with no key the cached
    answer is the empty set, so it must be dropped when a key is entered."""
    settings = load_lab_settings({})
    models._LIVE[settings.openrouter_base_url] = frozenset()
    credentials.set_key(KEY)
    assert settings.openrouter_base_url not in models._LIVE


def test_the_panel_can_set_a_key_and_the_options_never_carry_it(client):
    """/api/options is read on every visit, so the key itself must never be in
    it — only set-ness and a masked tail."""
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


def test_a_refused_key_answers_400_with_the_reason(client):
    refused = client.post('/api/credentials', json={'api_key': 'nope'})
    assert refused.status_code == 400
    assert 'key' in refused.json()['detail'].lower()


def test_the_panel_can_take_the_key_back(client):
    client.post('/api/credentials', json={'api_key': KEY})
    cleared = client.request('DELETE', '/api/credentials')
    assert cleared.status_code == 200
    assert cleared.json()['set'] is False
    assert client.get('/api/options').json()[
        'capabilities']['openrouter_key']['set'] is False


def test_no_artefact_a_run_leaves_behind_contains_the_key(client, tmp_path,
                                                          monkeypatch):
    """A real run, a real run file, a real ledger row, checked against the
    key itself — both are treated as shareable evidence."""
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


def test_the_key_is_never_written_to_a_file_by_this_module():
    """A source scan, since session-only is a promise about the code, not
    about one test run."""
    from pathlib import Path
    source = (Path(credentials.__file__)).read_text(encoding='utf-8')
    for forbidden in ('open(', 'write_text', 'os.environ[', 'setenv', 'dump('):
        assert forbidden not in source, (
            f'the credential store contains {forbidden!r} — it holds the key in '
            'memory and writes it nowhere')


def test_the_panel_offers_the_key_field_and_never_stores_it_in_the_browser():
    """The panel remembers the *config* on every keystroke; this field must
    stay out of that and out of localStorage."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    assert 'id="openrouter_key"' in html
    assert '/api/credentials' in html
    assert 'type="password"' in html
    for line in html.splitlines():
        if 'openrouter_key' in line:
            assert 'localStorage' not in line and 'remember(' not in line, line
    assert 'openrouter_key:' not in html


def test_the_key_is_a_documented_environment_variable_still():
    """The panel is a second way in, not a replacement."""
    from raglab.config import ROOT
    example = (ROOT / '.env.example').read_text(encoding='utf-8')
    assert 'OPENROUTER_API_KEY' in example
    assert os.environ.get('OPENROUTER_API_KEY') != KEY
