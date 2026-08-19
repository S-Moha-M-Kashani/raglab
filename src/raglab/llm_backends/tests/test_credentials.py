"""The OpenRouter key, entered in the panel and held nowhere else — never in
`/api/options`, a `.runs/` file, a `raglab.db` row, or the terminal."""
import json
import os

import pytest
from fastapi.testclient import TestClient

from raglab.llm_backends import openrouter_key_memory as credentials
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.llm_backends import model_role_catalogue as models
from raglab.configuration.lab_config import (
    GenerationConfig,
    IndexConfig,
    LabConfig,
    load_lab_settings)
from raglab.dashboard.panel_server import _with_backend

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
    from raglab.dashboard.panel_server import create_app
    return TestClient(create_app())


def test_a_key_is_held_in_the_process_and_read_back_by_the_settings():
    # this is a unit test
    settings = load_lab_settings({'RAGLAB_LLM': 'openrouter'})
    assert credentials.apply(settings).openrouter_api_key == ''
    credentials.set_key(KEY)
    assert credentials.apply(settings).openrouter_api_key == KEY
    assert credentials.apply(settings).llm_ready is True


def test_the_active_key_prefers_the_panel_then_falls_back_to_the_environment():
    environment = 'sk-or-from-the-shell-0123456789'
    assert credentials.active(environment) == environment
    credentials.set_key(KEY)
    assert credentials.active(environment) == KEY
    credentials.clear()
    assert credentials.active(environment) == environment
    assert credentials.active('') == ''


def test_clearing_falls_back_directly_and_through_the_panels_own_route(client):
    # this is an integration test
    """Clear means "forget what I typed", never "unset the key" — checked
    twice: directly against `credentials.clear()`, and again through the
    panel's own DELETE route, so a route that forgot the environment
    fallback (or reimplemented it) would fail here too."""
    from_env = load_lab_settings({'OPENROUTER_API_KEY': 'sk-or-from-the-shell-0123456789'})
    credentials.set_key(KEY)
    assert credentials.apply(from_env).openrouter_api_key == KEY
    credentials.clear()
    assert credentials.apply(from_env).openrouter_api_key == 'sk-or-from-the-shell-0123456789'

    client.post('/api/credentials', json={'api_key': KEY})
    cleared = client.request('DELETE', '/api/credentials')
    assert cleared.status_code == 200
    assert cleared.json()['set'] is False
    assert client.get('/api/options').json()[
        'capabilities']['openrouter_key']['set'] is False


def test_a_key_that_cannot_be_one_is_refused_directly_and_through_the_panel(client):
    # this is an integration test
    """The same refusal, checked at both entry points: `set_key` itself, and
    the panel's `/api/credentials` route in front of it."""
    for bad in ('', '   ', 'sk-or-short', 'sk-or-v1 0123456789abcdef0123456789'):
        with pytest.raises(ValueError) as raised:
            credentials.set_key(bad)
        assert 'key' in str(raised.value).lower()
    assert credentials.state(load_lab_settings({}))['set'] is False

    refused = client.post('/api/credentials', json={'api_key': 'nope'})
    assert refused.status_code == 400
    assert 'key' in refused.json()['detail'].lower()


def test_the_state_says_set_and_shows_a_hint_but_never_the_key():
    # this is a unit test
    credentials.set_key(KEY)
    state = credentials.state(load_lab_settings({}))
    assert state['set'] is True and state['source'] == 'panel'
    assert state['hint'].endswith(KEY[-4:])
    assert KEY not in json.dumps(state)
    assert state['hint'].count('…') == 1


def test_the_environments_own_key_is_reported_as_the_environments():
    # this is a unit test
    """A key from the shell is not one the panel's 'Clear' can remove."""
    settings = load_lab_settings({'OPENROUTER_API_KEY': 'sk-or-from-the-shell-0123456789'})
    state = credentials.state(settings)
    assert state['set'] is True and state['source'] == 'environment'


def test_setting_a_key_forgets_the_verified_model_lists():
    # this is a unit test
    """`models._LIVE` caches availability per base url; with no key the cached
    answer is the empty set, so it must be dropped when a key is entered."""
    settings = load_lab_settings({})
    models._LIVE[settings.openrouter_base_url] = frozenset()
    credentials.set_key(KEY)
    assert settings.openrouter_base_url not in models._LIVE


def test_the_panel_can_set_a_key_and_the_options_never_carry_it(client):
    # this is an integration test
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


def test_no_artefact_a_run_leaves_behind_contains_the_key():
    # this is a unit test
    """The claim is about serialization, not retrieval quality, so this
    stubs a run rather than scoring one for real. The stub is only a guard
    if the key could plausibly have reached it: a real key is held, real
    settings are built with it applied under a provider that would actually
    use it (`llm_ready` true, not idle), and `models.note_for` — the one
    place a run's free-text notes name the resolved backend — is called
    against those settings for real, not hand-written. What is checked is
    every artefact a finished run leaves behind: the JSON `RunResult`, the
    ledger row and its stored detail, and the job dict the panel polls."""
    credentials.set_key(KEY)
    settings = credentials.apply(load_lab_settings({'RAGLAB_LLM': 'openrouter'}))
    assert settings.llm_ready and settings.openrouter_api_key == KEY, (
        'the stub below is only a guard if the key really was in force')

    cfg = LabConfig(index=IndexConfig(chunker='session', embedder='token-hash'),
                    generation=GenerationConfig(answerer='llm'), label='key-safety')
    notes = [models.note_for(cfg, settings)]

    result = evaluate.RunResult(
        run_id='20260101-000000-abcdef', label='key-safety', config=cfg.to_dict(),
        index={'collection': 'raglab-abc123', 'chunks': 5},
        summary={'n_questions': 2}, dataset='',
        rows=[{'id': 'q1', 'answer': 'x'}, {'id': 'q2', 'answer': 'y'}],
        ragas={}, seconds=1.0, started_at='2026-01-01 00:00:00',
        notes=notes, selection={'n': 2, 'question_ids': ['q1', 'q2']})

    run_json = json.dumps(evaluate.json_safe(result.as_dict()), ensure_ascii=False)
    assert KEY not in run_json, 'the JSON run file `.runs/` would hold'

    job = {'id': 'job1', 'kind': 'run', 'state': 'done',
          'config': _with_backend(cfg, settings),
          'result': result.as_dict()}
    assert KEY not in json.dumps(job, default=str), 'the job dict the panel polls'

    row = ledger.row_for(job, 'done')
    detail = ledger.detail_for(job)
    assert KEY not in json.dumps(row, default=str), 'the ledger row'
    assert KEY not in json.dumps(detail, default=str), 'the ledger row detail'


def test_the_key_is_never_written_to_a_file_by_this_module():
    # this is a convention test
    """A source scan, since session-only is a promise about the code, not
    about one test run."""
    from pathlib import Path
    source = (Path(credentials.__file__)).read_text(encoding='utf-8')
    for forbidden in ('open(', 'write_text', 'os.environ[', 'setenv', 'dump('):
        assert forbidden not in source, (
            f'the credential store contains {forbidden!r} — it holds the key in '
            'memory and writes it nowhere')


def test_the_panel_offers_the_key_field_and_never_stores_it_in_the_browser():
    # this is a convention test
    """The panel remembers the *config* on every keystroke; this field must
    stay out of that and out of localStorage."""
    from raglab.dashboard.panel_server import STATIC
    html = (STATIC / 'panel.html').read_text(encoding='utf-8')
    js = (STATIC / 'panel.js').read_text(encoding='utf-8')
    assert 'id="openrouter_key"' in html
    assert '/api/credentials' in js
    assert 'type="password"' in html
    for line in html.splitlines() + js.splitlines():
        if 'openrouter_key' in line:
            assert 'localStorage' not in line and 'remember(' not in line, line
    assert 'openrouter_key:' not in html and 'openrouter_key:' not in js


def test_the_key_is_a_documented_environment_variable_still():
    # this is a convention test
    """The panel is a second way in, not a replacement."""
    from raglab.configuration.lab_config import ROOT
    example = (ROOT / '.env.example').read_text(encoding='utf-8')
    assert 'OPENROUTER_API_KEY' in example
    assert os.environ.get('OPENROUTER_API_KEY') != KEY
