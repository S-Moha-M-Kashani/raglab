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
    # this is a unit test
    assert widget.KNOWLEDGE_BASE
    for key, value in widget.KNOWLEDGE_BASE.items():
        assert isinstance(key, str) and key
        assert isinstance(value, str) and value


def test_search_finds_a_known_fact_by_keyword():
    # this is a unit test
    reply = widget.search_knowledge_base.invoke({'query': 'ports'})
    assert '9002' in reply


def test_search_says_so_when_nothing_matches():
    # this is a unit test
    reply = widget.search_knowledge_base.invoke({'query': 'zeppelin'})
    assert 'no entry' in reply.lower()


def test_calculate_does_arithmetic():
    # this is a unit test
    reply = widget.calculate.invoke({'expression': '68000000 / 551695'})
    assert reply.startswith('123.2')


def test_calculate_refuses_anything_that_is_not_arithmetic():
    # this is a unit test
    """The expression is evaluated over an AST whitelist, never `eval` — a
    tool handed to a model must not be a Python prompt."""
    with pytest.raises(ValueError):
        widget.calculate.func('__import__("os").getcwd()')


# --- the six hooks, as middleware ----------------------------------------

def test_before_agent_refuses_an_empty_question_and_caps_a_long_one():
    # this is a unit test
    """`_validate` is `check_request`'s work, factored out because the CLI
    path has no graph to hang middleware on."""
    with pytest.raises(ValueError):
        widget._validate('   ')
    assert widget._validate('  hello  ') == 'hello'
    assert len(widget._validate('x' * 9000)) == widget.MAX_QUESTION


def test_check_request_caps_by_replacing_the_question_not_appending_to_it():
    # this is a unit test
    """The capped question goes back with the same message id, so
    `add_messages` overwrites it instead of asking twice."""
    from langchain_core.messages import HumanMessage
    asked = HumanMessage(content='x' * 9000)
    update = widget.check_request.before_agent({'messages': [asked]}, None)
    written = update['messages'][0]
    assert written.id == asked.id
    assert len(written.content) == widget.MAX_QUESTION
    short = HumanMessage(content='which ports?')
    assert widget.check_request.before_agent({'messages': [short]}, None) is None


def test_trim_and_call_shortens_the_request_never_the_transcript():
    # this is a unit test
    """1.x has no `llm_input_messages`: the trim is an override on the request
    handed to this hop, and the graph's own messages are left alone."""
    seen = {}

    class FakeRequest:
        def __init__(self, messages):
            self.messages = messages
            self.model = type('M', (), {'model_name': 'openai/gpt-5-nano'})()

        def override(self, **changes):
            return FakeRequest(changes['messages'])

    request = FakeRequest(list(range(widget.MAX_HISTORY + 5)))
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))
    assert len(seen['messages']) == widget.MAX_HISTORY
    assert len(request.messages) == widget.MAX_HISTORY + 5


def test_check_reply_tells_a_tool_hop_from_an_answer():
    # this is a unit test
    from langchain_core.messages import AIMessage
    widget.HOOK_LOG.clear()
    widget.check_reply.after_model({'messages': [AIMessage(content='', tool_calls=[
        {'name': 'calculate', 'args': {'expression': '1+1'}, 'id': 'a'}])]}, None)
    widget.check_reply.after_model({'messages': [AIMessage(content='seven')]}, None)
    widget.check_reply.after_model({'messages': [AIMessage(content='  ')]}, None)
    assert 'calculate' in widget.HOOK_LOG[0]
    assert 'answer' in widget.HOOK_LOG[1]
    assert 'empty reply' in widget.HOOK_LOG[2]


def test_log_tool_call_logs_the_call_and_lets_the_error_through():
    # this is a unit test
    """Logging, never swallowing: a widget tool that hid its own failure would
    answer confidently from nothing."""
    request = type('R', (), {'tool_call': {'name': 'calculate',
                                           'args': {'expression': '1+1'}}})()
    widget.HOOK_LOG.clear()
    assert widget.log_tool_call.wrap_tool_call(request, lambda r: '2') == '2'
    assert any('calculate' in line for line in widget.HOOK_LOG)
    with pytest.raises(ValueError):
        widget.log_tool_call.wrap_tool_call(
            request, lambda r: (_ for _ in ()).throw(ValueError('boom')))
    assert any('raised' in line for line in widget.HOOK_LOG)


def test_all_six_hooks_are_registered_middleware():
    # this is a unit test
    """Each is the framework's own `AgentMiddleware`, not this module's
    imitation of one — and `create_agent` is handed all six."""
    from langchain.agents.middleware import AgentMiddleware
    assert len(widget.MIDDLEWARE) == 6
    assert all(isinstance(m, AgentMiddleware) for m in widget.MIDDLEWARE)
    assert [m.name for m in widget.MIDDLEWARE] == [
        'check_request', 'note_prompt', 'trim_and_call', 'log_tool_call',
        'check_reply', 'close_the_log']


def test_the_two_agent_level_hooks_bracket_a_cli_too(monkeypatch):
    # this is a unit test
    """A CLI has no tool loop for the middle four and no graph to hang
    middleware on — but a request is still validated and a run accounted."""
    monkeypatch.setattr(widget, 'cli_available', lambda cli: True)
    monkeypatch.setattr(widget, '_cli_answer', lambda cli, message: 'from the cli')
    widget.HOOK_LOG.clear()
    widget.ask('which ports?', model='codex')
    assert [line.split(':')[0] for line in widget.HOOK_LOG] == ['before_agent',
                                                                'after_agent']


# --- the model picker: four choices, each saying what it can do ----------

def test_the_model_catalogue_offers_four_choices_and_each_names_its_kind():
    # this is a unit test
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
    # this is a unit test
    with pytest.raises(ValueError) as caught:
        widget.ask('hello', model='gpt-4')
    assert 'gpt-4' in str(caught.value)


def test_a_missing_cli_command_is_a_stated_refusal(monkeypatch):
    # this is a unit test
    """Availability is the binary — there is nothing else to ask a CLI."""
    monkeypatch.setattr(widget, 'cli_available', lambda cli: False)
    with pytest.raises(widget.WidgetUnavailable) as caught:
        widget.ask('hello', model='claude')
    assert 'claude' in str(caught.value)


def test_the_cli_path_needs_no_key_and_carries_the_knowledge_base(monkeypatch):
    # this is a unit test
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
    # this is an integration test
    """The panel keeps no list of its own — the rule both panels already
    follow for every other model dropdown."""
    data = client.get('/api/widget').json()
    assert data['default'] == widget.DEFAULT_MODEL
    assert {m['value'] for m in data['models']} == set(widget.WIDGET_MODELS)
    assert all(m['label'] for m in data['models'])


def test_the_route_passes_the_chosen_model_through(client, monkeypatch):
    # this is an integration test
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
    # this is a unit test
    """The suite runs offline and a build must open no socket, so the agent
    exists only after the first request asks for it."""
    widget.reset()
    assert not widget._AGENTS


def test_a_missing_env_variable_is_a_stated_refusal(monkeypatch):
    # this is a unit test
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
    # this is a unit test
    """`.env` already carries OPENROUTER_BASE_URL for the lab's own backend —
    the widget reads the same variable rather than keeping a second copy of
    the endpoint, and falls back to the public one when it is unset."""
    monkeypatch.setenv('OPENROUTER_BASE_URL', 'http://localhost:9999/v1')
    assert widget._openrouter_url() == 'http://localhost:9999/v1'
    monkeypatch.delenv('OPENROUTER_BASE_URL')
    assert widget._openrouter_url() == 'https://openrouter.ai/api/v1'


def test_an_empty_env_variable_is_missing_too(monkeypatch):
    # this is a unit test
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
    # this is an integration test
    monkeypatch.setattr(widget, 'ask',
                        lambda message, model='': f'echo: {message}')
    answer = client.post('/api/widget', json={'message': 'what is this lab?'})
    assert answer.status_code == 200
    assert answer.json() == {'reply': 'echo: what is this lab?'}


def test_the_route_refuses_an_empty_message(client):
    # this is an integration test
    answer = client.post('/api/widget', json={'message': '   '})
    assert answer.status_code == 400


def test_an_unavailable_widget_is_a_502_naming_the_reason(client, monkeypatch):
    # this is an integration test
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
    # this is a unit test
    """Constructing the agent opens no socket — fake values are enough to
    build it, which is what keeps the lazy path testable at all."""
    pytest.importorskip('langgraph')
    widget.reset()
    for name in widget.REQUIRED_ENV:
        monkeypatch.setenv(name, 'test-value')
    agent = widget._build_agent('openai/gpt-5-nano')
    assert hasattr(agent, 'invoke')
    widget.reset()
