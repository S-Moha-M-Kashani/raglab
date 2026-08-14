"""The panel's LLM widget — one module, one route, no measurement.

`raglab/widget.py` is deliberately outside the lab's measured LLM seam: a
self-contained helper the panel pops up in its lower-right corner. These tests
pin what keeps it harmless — importing it costs nothing, missing env is a
stated refusal rather than a bare 500, and no test here reaches a network.
"""
import pytest

from raglab import widget


# --- the knowledge base and the two tools -------------------------------

def test_the_knowledge_base_has_facts_and_they_are_strings():
    assert widget.KNOWLEDGE_BASE
    for key, value in widget.KNOWLEDGE_BASE.items():
        assert isinstance(key, str) and key
        assert isinstance(value, str) and value


def test_search_finds_a_known_fact_by_keyword():
    reply = widget.search_knowledge_base.invoke({'query': 'ports'})
    assert '9002' in reply


def test_search_says_so_when_nothing_matches():
    reply = widget.search_knowledge_base.invoke({'query': 'zeppelin'})
    assert 'no entry' in reply.lower()


def test_calculate_does_arithmetic():
    reply = widget.calculate.invoke({'expression': '68000000 / 551695'})
    assert reply.startswith('123.2')


def test_calculate_refuses_anything_that_is_not_arithmetic():
    """The expression is evaluated over an AST whitelist, never `eval` — a
    tool handed to a model must not be a Python prompt."""
    with pytest.raises(ValueError):
        widget.calculate.func('__import__("os").getcwd()')


# --- the model picker: four choices, each saying what it can do ----------

def test_the_model_catalogue_offers_four_choices_and_each_names_its_kind():
    """Two OpenRouter models that run the tool loop, two CLIs that cannot
    (`CliChat` has no `bind_tools`) — and the CLI labels say so, the
    catalogue rule that an option states what it can do."""
    assert set(widget.WIDGET_MODELS) == {'openai/gpt-5-nano',
                                         'openai/gpt-5-mini',
                                         'claude', 'codex'}
    # The platform default is the codex CLI (gpt-5.6-luna — the lightest draw
    # on the membership, and no key involved); among the OpenRouter pair the
    # cheaper nano leads the list.
    assert widget.DEFAULT_MODEL == 'codex'
    assert next(iter(widget.WIDGET_MODELS)) == 'openai/gpt-5-nano'
    for kind, label in widget.WIDGET_MODELS.values():
        assert kind in ('openrouter', 'cli')
        assert label
    for cli in ('claude', 'codex'):
        assert 'no tools' in widget.WIDGET_MODELS[cli][1]


def test_an_unknown_model_is_refused_by_name():
    with pytest.raises(ValueError) as caught:
        widget.ask('hello', model='gpt-4')
    assert 'gpt-4' in str(caught.value)


def test_a_missing_cli_command_is_a_stated_refusal(monkeypatch):
    """Availability is the binary — there is nothing else to ask a CLI."""
    monkeypatch.setattr(widget, 'cli_available', lambda cli: False)
    with pytest.raises(widget.WidgetUnavailable) as caught:
        widget.ask('hello', model='claude')
    assert 'claude' in str(caught.value)


def test_the_cli_path_needs_no_key_and_carries_the_knowledge_base(monkeypatch):
    """The whole point of a CLI backend is no API key — so the five env
    variables are the OpenRouter path's requirement, not the widget's. And
    with no tool loop, the knowledge base must travel in the prompt or the
    CLI answers about a project it has never seen."""
    for name in widget.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(widget, 'cli_available', lambda cli: True)
    seen = {}

    class FakeCli:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def invoke(self, messages):
            seen['messages'] = messages
            from langchain_core.messages import AIMessage
            return AIMessage(content='from the cli')

    monkeypatch.setattr(widget, 'CliChat', FakeCli)
    assert widget.ask('which ports?', model='codex') == 'from the cli'
    assert seen['cli'] == 'codex'
    assert '9002' in str(seen['messages'])


def test_the_widget_serves_its_own_model_list(client):
    """The panel keeps no list of its own — the rule both panels already
    follow for every other model dropdown."""
    data = client.get('/api/widget').json()
    assert data['default'] == widget.DEFAULT_MODEL
    assert {m['value'] for m in data['models']} == set(widget.WIDGET_MODELS)
    assert all(m['label'] for m in data['models'])


def test_the_route_passes_the_chosen_model_through(client, monkeypatch):
    seen = {}

    def fake_ask(message, model=''):
        seen['model'] = model
        return 'ok'

    monkeypatch.setattr(widget, 'ask', fake_ask)
    answer = client.post('/api/widget', json={'message': 'hello',
                                              'model': 'openai/gpt-5-nano'})
    assert answer.status_code == 200
    assert seen['model'] == 'openai/gpt-5-nano'


# --- laziness: importing the widget costs nothing ------------------------

def test_importing_the_widget_builds_no_agent_and_reads_no_env():
    """The suite runs offline and a build must open no socket, so the agent
    exists only after the first request asks for it."""
    widget.reset()
    assert not widget._AGENTS


def test_a_missing_env_variable_is_a_stated_refusal(monkeypatch):
    """The five variables the widget needs are read at build time, and the
    refusal names the one that is missing — the `GradeUnavailable` pattern,
    never a KeyError half way up a stack trace."""
    widget.reset()
    for name in widget.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(widget.WidgetUnavailable) as caught:
        widget.ask('hello', model='openai/gpt-5-nano')
    assert 'OPENROUTER_API_KEY' in str(caught.value)


def test_the_openrouter_base_url_is_read_from_the_environment(monkeypatch):
    """`.env` already carries OPENROUTER_BASE_URL for the lab's own backend —
    the widget reads the same variable rather than keeping a second copy of
    the endpoint, and falls back to the public one when it is unset."""
    monkeypatch.setenv('OPENROUTER_BASE_URL', 'http://localhost:9999/v1')
    assert widget._openrouter_url() == 'http://localhost:9999/v1'
    monkeypatch.delenv('OPENROUTER_BASE_URL')
    assert widget._openrouter_url() == 'https://openrouter.ai/api/v1'


def test_an_empty_env_variable_is_missing_too(monkeypatch):
    """`load_env` strips values, so a `KEY= ` line in .env lands as ''. An
    empty key would sail past a presence check and die inside the OpenAI
    client naming a variable this lab never reads (OPENAI_API_KEY) — found
    live 2026-08-14, a duplicate OPENROUTER_API_KEY line whose first, junk
    occurrence won `setdefault`."""
    widget.reset()
    for name in widget.REQUIRED_ENV:
        monkeypatch.setenv(name, 'test-value')
    monkeypatch.setenv('OPENROUTER_API_KEY', '   ')
    with pytest.raises(widget.WidgetUnavailable) as caught:
        widget.ask('hello', model='openai/gpt-5-nano')
    assert 'OPENROUTER_API_KEY' in str(caught.value)


# --- the route -----------------------------------------------------------

def test_the_route_answers_with_the_agents_reply(client, monkeypatch):
    monkeypatch.setattr(widget, 'ask',
                        lambda message, model='': f'echo: {message}')
    answer = client.post('/api/widget', json={'message': 'what is this lab?'})
    assert answer.status_code == 200
    assert answer.json() == {'reply': 'echo: what is this lab?'}


def test_the_route_refuses_an_empty_message(client):
    answer = client.post('/api/widget', json={'message': '   '})
    assert answer.status_code == 400


def test_an_unavailable_widget_is_a_502_naming_the_reason(client, monkeypatch):
    """The lab is up, its widget is not — the same split `/api/queries` makes
    for an unreachable grade model."""
    def refuse(message, model=''):
        raise widget.WidgetUnavailable('OPENROUTER_API_KEY is not set')
    monkeypatch.setattr(widget, 'ask', refuse)
    answer = client.post('/api/widget', json={'message': 'hello'})
    assert answer.status_code == 502
    assert 'OPENROUTER_API_KEY' in answer.json()['detail']


# --- the real build, when the extra is installed --------------------------

def test_the_agent_builds_offline_when_the_extra_is_present(monkeypatch):
    """Constructing the agent opens no socket — fake values are enough to
    build it, which is what keeps the lazy path testable at all."""
    pytest.importorskip('langgraph')
    widget.reset()
    for name in widget.REQUIRED_ENV:
        monkeypatch.setenv(name, 'test-value')
    agent = widget._build_agent('openai/gpt-5-nano')
    assert hasattr(agent, 'invoke')
    widget.reset()
