"""The OpenRouter key, entered in the panel and held nowhere else — never in
`/api/options`, a `.runs/` file, a `raglab.db` row, or the terminal."""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

from raglab.llm_backends import openrouter_key_memory as credentials
from raglab.agents import widget
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.llm_backends import model_role_catalogue as models
from raglab.configuration.lab_config import (
    GenerationConfig,
    IndexConfig,
    LabConfig,
    load_lab_settings)
from raglab.dashboard.service_route_plumbing import _with_backend

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


def test_saving_a_valid_key_drops_cached_widget_agents(client):
    widget._AGENTS['openai/gpt-5-nano'] = object()
    try:
        response = client.post('/api/credentials', json={'api_key': KEY})
        assert response.status_code == 200
        assert widget._AGENTS == {}
    finally:
        widget.reset()


def test_clearing_a_panel_key_drops_cached_widget_agents(client):
    credentials.set_key(KEY)
    widget._AGENTS['openai/gpt-5-nano'] = object()
    try:
        response = client.delete('/api/credentials')
        assert response.status_code == 200
        assert widget._AGENTS == {}
    finally:
        widget.reset()


def test_a_rejected_key_keeps_the_working_widget_agent(client):
    sentinel = object()
    widget._AGENTS['openai/gpt-5-nano'] = sentinel
    try:
        response = client.post('/api/credentials', json={'api_key': 'nope'})
        assert response.status_code == 400
        assert widget._AGENTS['openai/gpt-5-nano'] is sentinel
    finally:
        widget.reset()


@pytest.mark.parametrize('change', ('save', 'clear'))
def test_credential_changes_wait_for_an_old_widget_build_and_agent_selection(
        client, monkeypatch, change):
    # this is an integration test
    """A credential change cannot finish until it has invalidated every
    agent the old key could still install.  The second half gates the exact
    membership-to-lookup gap, so clearing the cache cannot turn a selected
    agent into a KeyError before its invocation begins."""
    old_key = 'sk-or-v1-old-key-0123456789abcdef0123456789'
    model = 'openai/gpt-5-nano'
    credentials.set_key(old_key)
    widget.reset()

    def endpoint(path, method):
        return next(route.endpoint for route in client.app.routes
                    if getattr(route, 'path', None) == path
                    and method in getattr(route, 'methods', set()))

    change_key = endpoint('/api/credentials', 'POST')
    clear_key = endpoint('/api/credentials', 'DELETE')
    build_started = Event()
    allow_build = Event()
    reset_started = Event()
    reset_finished = Event()
    captured_keys = []
    original_reset = widget.reset

    class Stub:
        def invoke(self, payload, config=None):
            return {'messages': [widget.backends.HumanMessage(content='ok')]}

    def build_with_the_old_key(requested_model):
        assert requested_model == model
        captured_keys.append(widget.backends._openrouter_key())
        build_started.set()
        assert allow_build.wait(timeout=2), 'the test must release the build'
        return Stub()

    def observe_reset():
        reset_started.set()
        original_reset()
        reset_finished.set()

    monkeypatch.setattr(widget.backends, '_build_agent', build_with_the_old_key)
    monkeypatch.setattr(widget, 'reset', observe_reset)
    build_executor = ThreadPoolExecutor(max_workers=2)
    build_future = change_future = None
    try:
        build_future = build_executor.submit(widget.ask, 'hello', model)
        assert build_started.wait(timeout=2), 'the widget build did not start'
        if change == 'save':
            change_future = build_executor.submit(change_key, {'api_key': KEY})
        else:
            change_future = build_executor.submit(clear_key)
        assert reset_started.wait(timeout=2), 'the credential route did not reset'
        assert not reset_finished.wait(timeout=.2), (
            'the credential route returned while an old-key build was paused')
        allow_build.set()
        assert build_future.result(timeout=2)['reply'] == 'ok'
        change_future.result(timeout=2)
        assert captured_keys == [old_key]
        assert widget.backends._AGENTS == {}
    finally:
        allow_build.set()
        for future in (build_future, change_future):
            if future is not None:
                try:
                    future.result(timeout=2)
                except Exception:
                    pass
        build_executor.shutdown(wait=True, cancel_futures=True)
        original_reset()
        credentials.clear()

    membership_checked = Event()
    allow_lookup = Event()
    selection_reset_started = Event()
    selection_reset_finished = Event()

    class GatedAgents(dict):
        def __contains__(self, key):
            found = super().__contains__(key)
            membership_checked.set()
            assert allow_lookup.wait(timeout=2), 'the test must release lookup'
            return found

    agents = GatedAgents({model: Stub()})

    def observe_selection_reset():
        selection_reset_started.set()
        original_reset()
        selection_reset_finished.set()

    monkeypatch.setattr(widget.backends, '_AGENTS', agents)
    monkeypatch.setattr(widget, 'reset', observe_selection_reset)
    selection_executor = ThreadPoolExecutor(max_workers=2)
    ask_future = clear_future = None
    try:
        ask_future = selection_executor.submit(widget.ask, 'hello', model)
        assert membership_checked.wait(timeout=2), 'the cache membership was not read'
        clear_future = selection_executor.submit(clear_key)
        assert selection_reset_started.wait(timeout=2), 'the clear route did not reset'
        assert not selection_reset_finished.wait(timeout=.2), (
            'the cache cleared between membership and agent lookup')
        allow_lookup.set()
        assert ask_future.result(timeout=2)['reply'] == 'ok'
        clear_future.result(timeout=2)
        assert agents == {}
    finally:
        allow_lookup.set()
        for future in (ask_future, clear_future):
            if future is not None:
                try:
                    future.result(timeout=2)
                except Exception:
                    pass
        selection_executor.shutdown(wait=True, cancel_futures=True)
        original_reset()
        credentials.clear()


def test_create_app_installs_the_panel_key_resolver_for_the_widget(monkeypatch, tmp_path):
    # this is an integration test
    """The production composition root, not a test-only injection, wires a
    key saved by the masthead panel into the widget's resolver."""
    from raglab.dashboard.panel_server import create_app

    previous_resolver = widget.backends._openrouter_key_resolver
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    credentials.clear()
    widget.reset()
    widget.set_openrouter_key_resolver()
    try:
        with TestClient(create_app()) as panel:
            saved = panel.post('/api/credentials', json={'api_key': KEY})
            assert saved.status_code == 200
            assert widget.backends._openrouter_key() == KEY
    finally:
        credentials.clear()
        widget.reset()
        widget.set_openrouter_key_resolver(previous_resolver)


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
    header = html[html.index('<header'):html.index('</header>')]
    model_card = html[html.index('id="modelCard"'):]
    model_card = model_card[:model_card.index('</section>')]
    assert 'id="app-settings"' in header
    assert 'popovertarget="app-settings-panel"' in header
    panel = re.search(r'<[^>]*\bid=("|\')app-settings-panel\1[^>]*>', header)
    assert panel, 'the masthead must contain the Settings popover itself'
    assert re.search(r'(?<![\w-])popover(?=\s|=|>)', panel.group()), (
        'the Settings panel needs its own standalone popover attribute')
    assert 'id="openrouter_key"' in header
    assert 'id="openrouter_key"' not in model_card
    js = (STATIC / 'panel.js').read_text(encoding='utf-8')
    assert 'id="openrouter_key"' in html
    assert '/api/credentials' in js
    assert 'type="password"' in html
    for line in html.splitlines() + js.splitlines():
        if 'openrouter_key' in line:
            assert 'localStorage' not in line and 'remember(' not in line, line
    assert 'openrouter_key:' not in html and 'openrouter_key:' not in js


def test_the_settings_popover_overrides_ghost_button_chassis_colours():
    # this is a convention test
    """The popover is physically under `.chassis`; its action buttons must
    explicitly return to the light plate's ink in both ordinary and hover
    states rather than inheriting the masthead's light-on-dark treatment."""
    from raglab.dashboard.panel_server import STATIC
    css = (STATIC / 'panel.css').read_text(encoding='utf-8')
    normal = re.search(r'\.settings-popover\s+button\.ghost\s*\{([^}]*)\}',
                       css, re.DOTALL)
    hover = re.search(r'\.settings-popover\s+button\.ghost:hover\s*\{([^}]*)\}',
                      css, re.DOTALL)
    assert normal, 'the Settings popover needs a scoped ghost-button override'
    assert 'color: var(--ink)' in normal.group(1)
    assert hover, 'the Settings popover needs a scoped ghost-button hover rule'


def test_the_key_is_a_documented_environment_variable_still():
    # this is a convention test
    """The panel is a second way in, not a replacement."""
    from raglab.configuration.lab_config import ROOT
    example = (ROOT / '.env.example').read_text(encoding='utf-8')
    assert 'OPENROUTER_API_KEY' in example
    assert os.environ.get('OPENROUTER_API_KEY') != KEY
