"""The panel's LLM widget — one package, one route, no measurement.

`raglab/agents/widget/` is deliberately outside the lab's measured LLM seam: a
self-contained helper the panel pops up in its lower-right corner. These tests
pin what keeps it harmless — importing it costs nothing, missing env is a
stated refusal rather than a bare 500, and no test here reaches a network.
A monkeypatch targets the submodule that defines the name (widget.backends,
widget.probe), because a patch on the package's re-export would not reach
the module the code actually reads.
"""
import pytest
from langchain_core.messages import AIMessage

from raglab.agents import widget
from raglab.llm_backends import openrouter_key_memory as credentials


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


# --- the prompts are fixtures ---------------------------------------------

def test_every_prompt_the_model_reads_is_the_yaml_fixture():
    # this is a convention test
    """The model-facing text lives under fixtures/prompts/ — one entry per
    tool, plus the two system prompts — and what the running widget serves
    is exactly what the pages say. A tool the page does not name would fail
    the import itself; this pins the other directions: no orphan entries,
    no drift, and the CLI template keeps both slots."""
    import yaml
    tools_page = yaml.safe_load(
        (widget.PROMPTS_DIR / 'widget_tools.yaml').read_text(encoding='utf-8'))
    assert set(tools_page) == {t.name for t in widget.TOOLS}
    for t in widget.TOOLS:
        assert t.description == tools_page[t.name].strip()
    page = yaml.safe_load(
        (widget.PROMPTS_DIR / 'widget.yaml').read_text(encoding='utf-8'))
    assert widget.SYSTEM_PROMPT == page['system'].strip()
    assert '{facts}' in page['cli_system']
    assert '{skills_index}' in page['cli_system']
    # The empty log's starters are sent to the model verbatim, so they are
    # model-facing text and belong on this page with the prompts either side
    # of them — not in panel.js, where nothing would pin them.
    assert widget.STARTERS == [line.strip() for line in page['starters']]
    assert len(widget.STARTERS) == 4, (
        'four starters: they exist to span what the helper can do, and a list '
        'that grew is a menu the reader now has to read instead of a set of '
        'examples they can take in at a glance')
    knowledge_page = yaml.safe_load(
        (widget.PROMPTS_DIR / 'widget_knowledge.yaml').read_text(encoding='utf-8'))
    assert widget.KNOWLEDGE_BASE == {key: text.strip()
                                     for key, text in knowledge_page.items()}


# --- the bilingual probe tool ---------------------------------------------

def test_the_widget_offers_the_bilingual_probe_tool():
    # this is a unit test
    assert 'measure_bilingual_alignment' in {t.name for t in widget.TOOLS}
    assert 'measure_bilingual_alignment' in widget.SYSTEM_PROMPT


def test_the_bundled_pairs_fixture_holds_the_shape_the_docstring_states():
    # this is a convention test
    """The tool's docstring states the pairs contract; the bundled fixture
    is the first data measured against it, so it must satisfy it — a
    default that fails its own shape is a tool no argument can fix."""
    pairs, problem = widget._read_pairs('')
    assert problem is None
    assert len(pairs) >= 2
    # and the contract itself is announced to the model, with an example
    guide = widget.measure_bilingual_alignment.description
    assert 'JSON list' in guide
    assert '[english, farsi]' in guide
    assert 'bilingual_probe_pairs.json' in guide


def _fake_encoder(pairs, split: bool):
    """One-hot fakes over the given pairs: `split=False` maps a sentence and
    its translation to the same axis (perfect alignment), `split=True` to
    orthogonal axes (plausible vectors, no cross-language geometry — the
    silent failure the probe exists to catch)."""
    import numpy as np
    n = len(pairs)
    where = {}
    for i, (en, fa) in enumerate(pairs):
        where[en] = i
        where[fa] = n + i if split else i

    class Fake:
        def encode(self, texts, normalize_embeddings=True):
            eye = np.eye(2 * n if split else n)
            return np.stack([eye[where[t]] for t in texts])

    return Fake()


def test_the_probe_reads_an_aligned_encoder_as_aligned(monkeypatch):
    # this is a unit test
    """The fake keeps the suite offline while the real encoder run lives in
    test_skills_live.py."""
    pairs, _ = widget._read_pairs('')
    monkeypatch.setattr(widget.probe, '_load_encoder',
                        lambda name: _fake_encoder(pairs, split=False))
    reply = widget.measure_bilingual_alignment.invoke({'model_name': 'fake'})
    assert 'Verdict: aligned' in reply
    # derived from the fixture, which is editable without touching Python
    assert f'{len(pairs)}/{len(pairs)}' in reply


def test_the_probe_reads_a_language_split_encoder_as_unaligned(monkeypatch):
    # this is a unit test
    pairs, _ = widget._read_pairs('')
    monkeypatch.setattr(widget.probe, '_load_encoder',
                        lambda name: _fake_encoder(pairs, split=True))
    reply = widget.measure_bilingual_alignment.invoke({'model_name': 'fake'})
    assert 'weak or no alignment' in reply


def test_the_probe_tells_two_models_apart_and_names_each(monkeypatch):
    # this is a unit test
    """`model_name` routes: two encoders measured in one process, opposite
    verdicts, each reply opening with the model it measured — a reading
    labelled one encoder that measured another is the worst artefact this
    lab can produce, the embedder rule applied to the probe."""
    pairs, _ = widget._read_pairs('')
    encoders = {'good-encoder': _fake_encoder(pairs, split=False),
                'bad-encoder': _fake_encoder(pairs, split=True)}
    asked = []

    def load(name):
        asked.append(name)
        return encoders[name]

    monkeypatch.setattr(widget.probe, '_load_encoder', load)
    good = widget.measure_bilingual_alignment.invoke(
        {'model_name': 'good-encoder'})
    bad = widget.measure_bilingual_alignment.invoke(
        {'model_name': 'bad-encoder'})
    assert asked == ['good-encoder', 'bad-encoder']
    assert good.startswith('good-encoder') and 'Verdict: aligned' in good
    assert bad.startswith('bad-encoder') and 'weak or no alignment' in bad


def test_an_empty_model_name_measures_the_lab_default(monkeypatch):
    # this is a unit test
    pairs, _ = widget._read_pairs('')
    asked = []

    def load(name):
        asked.append(name)
        return _fake_encoder(pairs, split=False)

    monkeypatch.setattr(widget.probe, '_load_encoder', load)
    reply = widget.measure_bilingual_alignment.invoke({'model_name': ''})
    assert asked == ['heydariAI/persian-embeddings']
    assert reply.startswith('heydariAI/persian-embeddings')


def test_the_probe_measures_pairs_the_caller_provides(monkeypatch):
    # this is a unit test
    """The pairs argument is the user's way in: their own sentences, in the
    documented shape, measured instead of the bundled fixture."""
    import json
    own = [['I slept badly.', 'بد خوابیدم.'],
           ['It rained all day.', 'تمام روز باران آمد.'],
           ['We argued about money.', 'سر پول بحث کردیم.']]
    monkeypatch.setattr(
        widget.probe, '_load_encoder',
        lambda name: _fake_encoder([tuple(p) for p in own], split=False))
    reply = widget.measure_bilingual_alignment.invoke(
        {'model_name': 'fake', 'pairs': json.dumps(own, ensure_ascii=False)})
    assert '3 English-Farsi sentence pairs' in reply
    assert '3/3' in reply


def test_malformed_pairs_are_refused_with_the_shape_stated(monkeypatch):
    # this is a unit test
    """The refusal quotes the contract, so the model can correct its next
    call — and it fires before any encoder loads, so a bad shape costs
    nothing."""
    def never(name):
        raise AssertionError('a malformed payload must not load an encoder')
    monkeypatch.setattr(widget.probe, '_load_encoder', never)
    for bad in ('not json at all', '["one string"]',
                '[["only one pair", "یک جفت"]]',
                '[["", "خالی"], ["x", "y"]]'):
        reply = widget.measure_bilingual_alignment.invoke(
            {'model_name': 'fake', 'pairs': bad})
        assert reply.startswith('cannot measure'), bad
        assert '[english, farsi]' in reply, bad


def test_a_probe_that_cannot_load_its_encoder_refuses_by_name(monkeypatch):
    # this is a unit test
    """The stated-refusal rule: a missing extra or checkpoint is the whole
    answer, relayed to the model, never a dead tool loop."""
    def boom(name):
        raise ImportError('no module named sentence_transformers')
    monkeypatch.setattr(widget.probe, '_load_encoder', boom)
    reply = widget.measure_bilingual_alignment.invoke({'model_name': 'x'})
    assert reply.startswith('cannot measure x')
    assert 'sentence_transformers' in reply


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
    monkeypatch.setattr(widget.backends, 'cli_available', lambda cli: True)
    monkeypatch.setattr(widget.backends, '_cli_answer',
                        lambda cli, message: AIMessage(content='from the cli'))
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
    # GPT-5 Nano leads because the widget is a small tool-calling helper; the
    # Codex coding-agent CLI remains selectable but is too context-heavy as a default.
    assert widget.DEFAULT_MODEL == 'openai/gpt-5-nano'
    assert next(iter(widget.WIDGET_MODELS)) == 'openai/gpt-5-nano'
    for kind, label in widget.WIDGET_MODELS.values():
        assert kind in ('openrouter', 'cli')
        assert label
    for cli in ('claude', 'codex'):
        assert 'no tools' in widget.WIDGET_MODELS[cli][1]


def test_a_panel_key_satisfies_the_openrouter_widget_without_an_env_key(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    previous = widget.backends._openrouter_key_resolver
    widget.set_openrouter_key_resolver(credentials.active)
    try:
        credentials.set_key('sk-or-v1-widget-test-0123456789abcdef')
        assert widget.backends._openrouter_key() == \
            'sk-or-v1-widget-test-0123456789abcdef'
    finally:
        credentials.clear()
        widget.set_openrouter_key_resolver(previous)


def test_an_unknown_model_is_refused_by_name():
    # this is a unit test
    with pytest.raises(ValueError) as caught:
        widget.ask('hello', model='gpt-4')
    assert 'gpt-4' in str(caught.value)


def test_a_missing_cli_command_is_a_stated_refusal(monkeypatch):
    # this is a unit test
    """Availability is the binary — there is nothing else to ask a CLI."""
    monkeypatch.setattr(widget.backends, 'cli_available', lambda cli: False)
    with pytest.raises(widget.WidgetUnavailable) as caught:
        widget.ask('hello', model='claude')
    assert 'claude' in str(caught.value)


def test_the_cli_path_needs_no_key_and_carries_the_knowledge_base(monkeypatch):
    # this is a unit test
    """The whole point of a CLI backend is no API key — so the OpenRouter
    key (and, with tracing on, the LangSmith four) is the OpenRouter path's
    requirement, not the widget's. And
    with no tool loop, the knowledge base must travel in the prompt or the
    CLI answers about a project it has never seen."""
    for name in widget.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(widget.backends, 'cli_available', lambda cli: True)
    seen = {}

    class FakeCli:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def invoke(self, messages):
            seen['messages'] = messages
            from langchain_core.messages import AIMessage
            return AIMessage(content='from the cli')

    monkeypatch.setattr(widget.backends, 'CliChat', FakeCli)
    assert widget.ask('which ports?', model='codex')['reply'] == 'from the cli'
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

    def fake_ask(message, model='', thread=''):
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
    """The variables the widget needs are read at build time, and the
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


def test_langsmith_env_is_required_only_when_tracing_is_on(monkeypatch):
    # this is a unit test
    """LANGSMITH_TRACING=False is a working configuration, not four missing
    variables. The LangSmith four exist only to serve tracing, so with
    tracing off — unset, or any spelling the tracer itself reads as off —
    the OpenRouter key alone must build the agent; demanding a LangSmith
    account then is requiring a credential for a disabled feature (found
    2026-08-19, a real .env carrying LANGSMITH_TRACING=False)."""
    pytest.importorskip('langgraph')
    assert widget.REQUIRED_ENV == ('OPENROUTER_API_KEY',)
    assert set(widget.TRACING_ENV) == {'LANGSMITH_API_KEY',
                                       'LANGSMITH_ENDPOINT',
                                       'LANGSMITH_PROJECT',
                                       'LANGSMITH_TRACING'}
    widget.reset()
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-value')
    for name in widget.TRACING_ENV:
        monkeypatch.delenv(name, raising=False)
    # the tracer's own falsy spellings, plus the variable simply not set
    for off in ('False', 'false', '0', ''):
        monkeypatch.setenv('LANGSMITH_TRACING', off)
        assert hasattr(widget._build_agent('openai/gpt-5-nano'), 'invoke'), (
            f'LANGSMITH_TRACING={off!r} must not demand a LangSmith account')
    monkeypatch.delenv('LANGSMITH_TRACING')
    assert hasattr(widget._build_agent('openai/gpt-5-nano'), 'invoke')
    widget.reset()


def test_tracing_switched_on_still_demands_the_langsmith_variables(monkeypatch):
    # this is a unit test
    """The old contract survives exactly where it was true: with tracing on
    the traces really leave the machine, so a missing LangSmith variable is
    a stated refusal naming it, before anything is built or sent."""
    widget.reset()
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-value')
    for name in widget.TRACING_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    with pytest.raises(widget.WidgetUnavailable) as caught:
        widget._build_agent('openai/gpt-5-nano')
    assert 'LANGSMITH_API_KEY' in str(caught.value)
    widget.reset()


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
                        lambda message, model='', thread='': {
                            'reply': f'echo: {message}',
                            'input_tokens': None, 'output_tokens': None})
    answer = client.post('/api/widget', json={'message': 'what is this lab?'})
    assert answer.status_code == 200
    assert answer.json() == {'reply': 'echo: what is this lab?',
                             'input_tokens': None, 'output_tokens': None}


def test_the_route_refuses_an_empty_message(client):
    # this is an integration test
    answer = client.post('/api/widget', json={'message': '   '})
    assert answer.status_code == 400


def test_an_unavailable_widget_is_a_502_naming_the_reason(client, monkeypatch):
    # this is an integration test
    """The lab is up, its widget is not — the same split `/api/queries` makes
    for an unreachable grade model."""
    def refuse(message, model='', thread=''):
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


# --- thread memory: a conversation widget.db remembers ---------------------

def test_the_agent_carries_a_checkpointer_so_a_thread_can_continue(monkeypatch):
    # this is a unit test
    """One question in, one answer out was the whole contract until
    2026-08-19; a helper that forgets the question before cannot be asked a
    follow-up. The memory now lives in `databases/widget.db`, shared by every
    agent this module builds, so nothing about it dies with the agent that
    reads it."""
    pytest.importorskip('langgraph')
    widget.reset()
    for name in widget.REQUIRED_ENV:
        monkeypatch.setenv(name, 'test-value')
    agent = widget._build_agent('openai/gpt-5-nano')
    assert getattr(agent, 'checkpointer', None) is not None
    widget.reset()


def test_a_credential_change_rebuilds_the_agent_without_forgetting_anything():
    # this is a unit test
    """The regression this whole change exists to prevent. `reset()` runs on
    every key set and clear; when the checkpointer lived inside the agent it
    went with it, so typing a key ended the conversation."""
    from raglab.agents.widget import backends
    from raglab.agents.widget import conversation_memory as memory
    before = memory.saver()
    backends.reset()
    assert memory.saver() is before, (
        'reset() must drop the cached clients, never the memory behind them')


def test_every_model_shares_one_memory():
    # this is a unit test
    """Switching model in the widget's gear is not starting a new conversation.
    One saver, handed to every agent the cache builds."""
    from raglab.agents.widget import backends
    from raglab.agents.widget import conversation_memory as memory
    saver = memory.saver()
    for model in [name for name, (kind, _) in backends.WIDGET_MODELS.items()
                  if kind == 'openrouter']:
        # Build without a key by asserting the refusal names the key, not a
        # checkpointer — a build that got as far as needing a key got past the
        # saver, which is what this checks.
        try:
            backends._build_agent(model)
        except backends.WidgetUnavailable as refusal:
            assert 'OPENROUTER_API_KEY' in str(refusal)
    assert memory.saver() is saver


def test_ask_threads_the_thread_into_the_agent():
    # this is a unit test
    """`ask` takes a `thread` and hands it to the graph as the thread id: two
    turns naming the same thread share a history, two different threads never
    do. The stub records what the real agent would have been asked."""
    calls = []

    class Stub:
        def invoke(self, payload, config=None):
            calls.append(config)
            return {'messages': [AIMessage(content='ok')]}

    widget.reset()
    widget._AGENTS['openai/gpt-5-nano'] = Stub()
    try:
        widget.ask('hello', model='openai/gpt-5-nano', thread='exp-one')
        widget.ask('again', model='openai/gpt-5-nano', thread='exp-one')
        widget.ask('hello', model='openai/gpt-5-nano', thread='exp-two')
    finally:
        widget.reset()
    threads = [call['configurable']['thread_id'] for call in calls]
    assert threads[0] == threads[1]
    assert threads[2] != threads[0]
    # the recursion cap is a hard ceiling and must survive the new config key
    assert all(call.get('recursion_limit') for call in calls)


def test_no_thread_lands_on_general():
    # this is a unit test
    """A reader who asks without an experiment open twice is having one
    conversation, not two: an empty `thread` lands on the same `general`
    thread every time, never on a fresh id that would scatter one
    conversation across unrelated stateless turns."""
    from raglab.agents.widget import conversation_memory as memory
    calls = []

    class Stub:
        def invoke(self, payload, config=None):
            calls.append(config)
            return {'messages': [AIMessage(content='ok')]}

    widget.reset()
    widget._AGENTS['openai/gpt-5-nano'] = Stub()
    try:
        widget.ask('hello', model='openai/gpt-5-nano')
        widget.ask('hello', model='openai/gpt-5-nano')
    finally:
        widget.reset()
    threads = [call['configurable']['thread_id'] for call in calls]
    assert threads == [memory.GENERAL, memory.GENERAL]


def test_a_cli_model_accepts_a_thread_and_stays_stateless(monkeypatch):
    # this is a unit test
    """`CliChat` runs one process per call with no persistence — by design. A
    thread passed with a CLI model is accepted and ignored, not an error: the
    panel sends one either way."""
    monkeypatch.setattr(widget.backends, '_cli_answer',
                        lambda cli, message: AIMessage(content='ok'))
    answer = widget.ask('hello', model='claude', thread='exp-one')
    assert answer['reply'] == 'ok'


def test_the_route_passes_the_thread_through(client, monkeypatch):
    # this is an integration test
    """The thread is the page's claim about which conversation this is; the
    route's job is only to carry it — absent lands as '', which `ask` reads
    as `general`."""
    seen = {}

    def fake_ask(message, model='', thread=''):
        seen['thread'] = thread
        return 'ok'

    monkeypatch.setattr(widget, 'ask', fake_ask)
    answer = client.post('/api/widget', json={'message': 'hello',
                                              'thread': 'exp-one'})
    assert answer.status_code == 200
    assert seen['thread'] == 'exp-one'
    client.post('/api/widget', json={'message': 'hello'})
    assert seen['thread'] == ''


# --- token usage: the account travels with the reply -----------------------

def test_ask_returns_the_reply_with_its_token_account():
    # this is a unit test
    """A tool-loop run is several model calls; the account is their sum,
    read from the `usage_metadata` LangChain already puts on every AI
    message. The reply stops being a bare string and becomes the one dict
    the route can serve unchanged."""
    class Stub:
        def invoke(self, payload, config=None):
            return {'messages': [
                AIMessage(content='searching', usage_metadata={
                    'input_tokens': 10, 'output_tokens': 5,
                    'total_tokens': 15}),
                AIMessage(content='ok', usage_metadata={
                    'input_tokens': 30, 'output_tokens': 7,
                    'total_tokens': 37}),
            ]}

    widget.reset()
    widget._AGENTS['openai/gpt-5-nano'] = Stub()
    try:
        answer = widget.ask('hello', model='openai/gpt-5-nano')
    finally:
        widget.reset()
    assert answer['reply'] == 'ok'
    assert answer['input_tokens'] == 40
    assert answer['output_tokens'] == 12


def test_a_backend_that_reports_no_usage_is_a_stated_none():
    # this is a unit test
    """No usage reported must land as None, never 0 — "0 tokens" is a claim
    about the bill, None says the backend did not account for it. The
    refusal-over-substitution rule, applied to a number nobody measured."""
    class Stub:
        def invoke(self, payload, config=None):
            return {'messages': [AIMessage(content='ok')]}

    widget.reset()
    widget._AGENTS['openai/gpt-5-nano'] = Stub()
    try:
        answer = widget.ask('hello', model='openai/gpt-5-nano')
    finally:
        widget.reset()
    assert answer['reply'] == 'ok'
    assert answer['input_tokens'] is None
    assert answer['output_tokens'] is None


def test_a_cli_reply_carries_its_account_too(monkeypatch):
    # this is a unit test
    """`CliChat` already builds `usage_metadata` from the CLI's own token
    report; the CLI path hands back the message so `ask` can read it the
    same way it reads the agent's."""
    monkeypatch.setattr(
        widget.backends, '_cli_answer',
        lambda cli, message: AIMessage(content='ok', usage_metadata={
            'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}))
    answer = widget.ask('hello', model='claude', thread='exp-one')
    assert answer['reply'] == 'ok'
    assert answer['input_tokens'] == 100
    assert answer['output_tokens'] == 20


def test_the_route_serves_the_reply_and_the_account_unchanged(client, monkeypatch):
    # this is an integration test
    monkeypatch.setattr(widget, 'ask',
                        lambda message, model='', thread='': {
                            'reply': f'echo: {message}',
                            'input_tokens': 40, 'output_tokens': 12})
    answer = client.post('/api/widget', json={'message': 'hello'})
    assert answer.status_code == 200
    assert answer.json() == {'reply': 'echo: hello',
                             'input_tokens': 40, 'output_tokens': 12}
