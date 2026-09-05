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


@pytest.fixture(autouse=True)
def _no_experiment_reader_leaks_between_tests():
    """The experiment reader is process-wide and `create_app` wires it, so a
    route test in this file leaves the lab's own validated records behind for
    every unit test that follows. What a turn resolves about its thread — the
    trusted dataset, and so the memory it is filed under — depends on them, and
    a unit test naming a thread of its own should not inherit another test's
    provenance."""
    yield
    widget.experiment_tools.set_experiment_reader(None)


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
    assert len(widget.STARTERS) == 5, (
        'five starters: they exist to span what the helper can do, and a list '
        'that grew is a menu the reader now has to read instead of a set of '
        'examples they can take in at a glance')
    # And the sharper rule the set is chosen by: a starter must be a question
    # only *this* lab can answer — from the ledger, the run files, the corpus
    # or its own knob surface. A general RAG question, however good, is one any
    # chat model answers without the lab, so it wastes the one place a reader
    # learns what the helper is for. Each of the five names a record here.
    for starter in widget.STARTERS:
        assert any(word in starter.lower() for word in
                   ('run', 'runs', 'experiments', "lab's")), (
            f'{starter!r} does not reach for anything only this lab holds')
    knowledge_page = yaml.safe_load(
        (widget.PROMPTS_DIR / 'widget_knowledge.yaml').read_text(encoding='utf-8'))
    assert widget.KNOWLEDGE_BASE == {key: text.strip()
                                     for key, text in knowledge_page.items()}


def test_the_recall_prompt_names_the_cap_the_code_actually_applies():
    # this is a convention test
    """The one number in `fixtures/prompts/widget_tools.yaml` that is also a
    constant in Python. `recall_conversation` tells the model "At most 20
    turns" and `conversation_memory.MAX_RECALLED` is what decides how many it
    really gets; the fixture is pinned byte-equal, so nothing stopped the
    constant from moving and leaving the model reading a promise the code no
    longer keeps — a tool description that misstates its own limit is the
    same lie as a row misstating what produced it, told to the model instead
    of to the reader. Reading the fixture rather than the loaded description
    on purpose: the number has to be right in the file a maintainer edits."""
    import yaml
    from raglab.agents.widget import conversation_memory as memory
    tools_page = yaml.safe_load(
        (widget.PROMPTS_DIR / 'widget_tools.yaml').read_text(encoding='utf-8'))
    assert f'At most {memory.MAX_RECALLED} turns' in tools_page['recall_conversation'], (
        'the recall prompt must state MAX_RECALLED, or the model is told a '
        'cap the code does not apply')


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


# --- the four hooks, as middleware ---------------------------------------

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


class _FakeModelRequest:
    """A minimal stand-in for `langchain`'s `ModelRequest`: just the four
    attributes `trim_and_call` reads (`messages`, `model`, `state`,
    `override`)."""

    def __init__(self, messages, state=None):
        self.messages = messages
        self.model = type('M', (), {'model_name': 'openai/gpt-5-nano'})()
        self.state = {'messages': []} if state is None else state

    def override(self, **changes):
        return _FakeModelRequest(changes['messages'], self.state)


def test_trim_and_call_shortens_the_request_never_the_transcript():
    # this is a unit test
    """1.x has no `llm_input_messages`: the trim is an override on the request
    handed to this hop, and the graph's own messages are left alone."""
    seen = {}
    request = _FakeModelRequest(list(range(widget.MAX_HISTORY + 5)))
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))
    assert len(seen['messages']) == widget.MAX_HISTORY
    assert len(request.messages) == widget.MAX_HISTORY + 5


def test_trim_and_call_logs_the_shape_of_what_came_back():
    # this is a unit test
    """`note_prompt`/`check_reply` folded into `trim_and_call` on 2026-08-28:
    the `before_model`/`after_model` lines they used to write as graph nodes
    are now written from inside this wrapper instead."""
    from langchain.agents.middleware import ModelResponse
    from langchain_core.messages import AIMessage

    def last_two(reply):
        widget.HOOK_LOG.clear()
        widget.trim_and_call.wrap_model_call(
            _FakeModelRequest([]), lambda r: ModelResponse(result=[reply]))
        return list(widget.HOOK_LOG)[-2:]

    trim_line, after = last_two(AIMessage(content='', tool_calls=[
        {'name': 'calculate', 'args': {'expression': '1+1'}, 'id': 'a'}]))
    assert trim_line.startswith('wrap_model_call:')
    assert 'calculate' in after
    assert after.startswith('after_model:')
    assert 'answer' in last_two(AIMessage(content='seven'))[1]
    assert 'empty reply' in last_two(AIMessage(content='  '))[1]


def test_trim_and_call_guard_short_circuits_without_calling_handler():
    # this is a unit test
    """At `MAX_TOOL_HOPS` the guard must win before a model call is ever
    spent on it — `wrap_model_call` accepts a plain `AIMessage` back, no
    tool calls, and that alone is what ends the run."""
    from langchain_core.messages import AIMessage

    hop_worth = [AIMessage(content='', tool_calls=[
        {'name': 'calculate', 'args': {}, 'id': str(i)}])
        for i in range(widget.hooks.MAX_TOOL_HOPS)]
    request = _FakeModelRequest([], {'messages': hop_worth})
    called = []
    result = widget.trim_and_call.wrap_model_call(
        request, lambda r: called.append(r))
    assert not called
    assert isinstance(result, AIMessage)
    assert not (result.tool_calls or [])
    assert result.content == widget.hooks.HOP_GUARD_REFUSAL


def test_the_hop_guard_refusal_fits_a_question_that_named_no_experiment():
    # this is a unit test
    """The other half of the 2026-09-03 regression. The guard fired on a knob
    question and told the reader to "ask for the run by its experiment ID" —
    advice for a lookup they had not asked for. A refusal must describe what
    happened and offer a remedy that fits any question the guard can stop, or
    it is a message lying about what produced it."""
    refusal = widget.hooks.HOP_GUARD_REFUSAL
    assert str(widget.hooks.MAX_TOOL_HOPS) in refusal, (
        'the refusal must say how many calls it stopped after')
    assert 'knob' in refusal.lower(), (
        'a knob question can trip this guard, so the remedy must name knobs too')


def test_hook_log_stops_growing_at_its_cap():
    # this is a unit test
    """`HOOK_LOG` lives for the whole process and is shared by every
    concurrent turn, so it must not grow without bound."""
    widget.HOOK_LOG.clear()
    cap = widget.HOOK_LOG.maxlen
    for i in range(cap + 50):
        widget.hooks._fired('test', str(i))
    assert len(widget.HOOK_LOG) == cap
    assert widget.HOOK_LOG[-1] == f'test: {cap + 49}'
    assert widget.HOOK_LOG[0] == f'test: {50}'


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


def test_all_four_hooks_are_registered_middleware():
    # this is a unit test
    """Each is the framework's own `AgentMiddleware`, not this module's
    imitation of one — and `create_agent` is handed all four."""
    from langchain.agents.middleware import AgentMiddleware
    assert len(widget.MIDDLEWARE) == 4
    assert all(isinstance(m, AgentMiddleware) for m in widget.MIDDLEWARE)
    assert [m.name for m in widget.MIDDLEWARE] == [
        'check_request', 'trim_and_call', 'log_tool_call', 'close_the_log']


def test_the_two_agent_level_hooks_bracket_a_cli_too(monkeypatch):
    # this is a unit test
    """A CLI has no tool loop for the middle two and no graph to hang
    middleware on — but a request is still validated and a run accounted."""
    monkeypatch.setattr(widget.backends, 'cli_available', lambda cli: True)
    monkeypatch.setattr(widget.backends, '_cli_answer',
                        lambda cli, message: AIMessage(content='from the cli'))
    widget.HOOK_LOG.clear()
    widget.ask('which ports?', model='codex')
    assert [line.split(':')[0] for line in widget.HOOK_LOG] == ['before_agent',
                                                                'after_agent']


def test_cli_irrelevance_refuses_before_cli_invocation_or_memory(monkeypatch):
    # this is a unit test
    monkeypatch.setattr(widget.backends, '_cli_answer',
                        lambda *args: (_ for _ in ()).throw(
                            AssertionError('irrelevant CLI must not run')))
    monkeypatch.setattr(widget.backends, '_memory_model',
                        lambda *args: (_ for _ in ()).throw(
                            AssertionError('irrelevant CLI must not ask policy')))

    result = widget.ask('Tell me a joke about penguins.', model='codex')

    assert 'RAG lab' in result['reply']
    # The refusal is its own decision: nothing ran, so nothing was filed.
    # Read off the two fields a reader of this API actually has
    # (`routes.widget._safe_widget_event` turns them into `irrelevant`) rather
    # than off a `blocked` flag that no code left reads.
    assert result['memory']['relevant'] is False
    assert result['memory']['saved'] is False


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


def test_the_route_exposes_irrelevant_memory_as_a_safe_status(client, monkeypatch):
    # this is an integration test
    """The model's memory-policy internals are not panel metadata. An
    irrelevant refusal is useful to the reader only as the bounded status,
    while its reason, dataset and save payload stay on the server side."""
    monkeypatch.setattr(widget, 'ask', lambda message, model='', thread='': {
        'reply': 'I can only help with the RAG lab.',
        'input_tokens': None, 'output_tokens': None,
        'memory': {
            'relevant': False, 'should_save': False, 'saved': False,
            'dataset_id': 'private-dataset',
            'reason': 'unrelated request details',
            'save': {'dataset_summary': 'private summary'},
        }})

    answer = client.post('/api/widget', json={'message': 'tell me a joke'})

    assert answer.status_code == 200
    assert answer.json()['memory'] == {'status': 'irrelevant'}


def test_the_route_says_not_filed_when_the_hop_guard_ended_the_run(
        client, monkeypatch):
    # this is an integration test
    """A run the tool-hop guard stopped answers in the widget's own voice, so
    there is nothing about the lab to keep and no decision to wait for.

    It is its own status and not `irrelevant`: the question was a fair one and
    the deterministic guard let it through — what failed was the lookup. A
    route that called this off-topic would describe the turn as something it
    was not."""
    monkeypatch.setattr(widget, 'ask', lambda message, model='', thread='': {
        'reply': widget.hooks.HOP_GUARD_REFUSAL,
        'input_tokens': 8, 'output_tokens': 4,
        'memory': {'status': 'not_filed', 'saved': False,
                   'reason': 'internal reason nobody outside needs'}})

    answer = client.post('/api/widget', json={'message': 'find that run'})

    assert answer.status_code == 200
    assert answer.json()['memory'] == {'status': 'not_filed'}


def test_the_route_emits_no_status_for_a_verdict_no_answer_path_can_produce(
        client, monkeypatch):
    # this is an integration test
    """`saved` and `not_saved` were removed on 2026-08-28, when the verdict
    moved to a thread that outlives the response.

    Nothing can put a resolved, relevant verdict on an event any more — it
    lands on the `widget_turn_log` row instead — so a block shaped like one is
    treated the way a malformed block is: dropped, never turned into a claim
    about what the lab kept. The reply beside it is untouched."""
    monkeypatch.setattr(widget, 'ask', lambda message, model='', thread='': {
        'reply': 'The comparison is worth retaining.',
        'input_tokens': 8, 'output_tokens': 4,
        'memory': {
            'relevant': True, 'should_save': True, 'saved': True,
            'dataset_id': 'private-dataset', 'subtopic': 'private-topic',
            'reason': 'private policy reasoning',
            'save': {'dataset_summary': 'private summary'},
        }})

    answer = client.post('/api/widget', json={'message': 'remember this'})

    assert answer.status_code == 200
    assert answer.json() == {'reply': 'The comparison is worth retaining.',
                             'input_tokens': 8, 'output_tokens': 4}


def test_the_route_says_unavailable_when_no_judge_ever_answered(
        client, monkeypatch):
    # this is an integration test
    """A judge that could not be reached is not a judge that said no. Both
    reach the route as `relevant: False`, because saving fails closed either
    way — but telling a reader their question was off-topic when nobody read it
    is telling them something untrue, which is the one thing no record here
    may do."""
    monkeypatch.setattr(widget, 'ask', lambda message, model='', thread='': {
        'reply': 'The comparison is worth retaining.',
        'input_tokens': 8, 'output_tokens': 4,
        'memory': {
            'relevant': False, 'should_save': False, 'saved': False,
            'unavailable': True, 'dataset_id': '',
            'reason': 'The memory policy could not be reached: unavailable or '
                      'malformed (connection refused); nothing was saved.',
        }})

    answer = client.post('/api/widget', json={'message': 'remember this'})

    assert answer.status_code == 200
    assert answer.json()['memory'] == {'status': 'unavailable'}


def test_the_route_says_pending_while_the_decision_is_still_being_made(
        client, monkeypatch):
    # this is an integration test
    """The ordinary case since both paths answer first and file afterwards.

    The deferred block carries no policy booleans — there is no verdict to put
    in them yet — and it used to be dropped here, leaving the reader with no
    memory line at all. No line is not neutral: it reads as the lab finding
    nothing worth keeping, which is a verdict nobody gave."""
    monkeypatch.setattr(widget, 'ask', lambda message, model='', thread='': {
        'reply': 'The comparison is worth retaining.',
        'input_tokens': 8, 'output_tokens': 4,
        'memory': {'status': 'pending', 'saved': False}})

    answer = client.post('/api/widget', json={'message': 'remember this'})

    assert answer.status_code == 200
    assert answer.json()['memory'] == {'status': 'pending'}


def test_a_policy_nobody_could_reach_is_marked_as_unreached_not_refused(
        monkeypatch):
    # this is a unit test
    """The same distinction one step earlier, where the decision is made:
    `hooks.evaluate_memory_policy` fails closed on every field, so only the
    decision's own flag can tell the two apart afterwards."""
    class Unreachable:
        def with_structured_output(self, schema):
            raise RuntimeError('connection refused')

    monkeypatch.setattr(widget.backends, '_memory_model',
                        lambda model: Unreachable())

    decision = widget.backends._finish_memory(
        'Which index should I use?', 'an answer', 'openai/gpt-5-nano',
        'general', '')

    assert decision['unavailable'] is True
    assert decision['saved'] is False
    assert decision['should_save'] is False
    assert 'could not be reached' in decision['reason']


def test_ask_and_stream_refuse_a_missing_key_before_reading_any_record(
        monkeypatch):
    # this is a unit test
    """`ask` and `stream` are one turn and must do their work in one order.
    They had drifted: `ask` read the thread's records and its long-term memory
    first, while `stream` — through the order Python evaluates arguments in —
    built the agent first. On a keyless install that is a ledger query and a
    run-file read spent on a turn that was never going to run."""
    reads = []
    monkeypatch.setattr(widget.experiment_tools, 'trusted_dataset_id',
                        lambda experiment: reads.append(experiment) or '')

    def unavailable(model):
        raise widget.WidgetUnavailable('OPENROUTER_API_KEY is not set')

    monkeypatch.setattr(widget.backends, '_agent_for', unavailable)

    for call in (widget.ask, widget.stream):
        with pytest.raises(widget.WidgetUnavailable):
            call('which ports?', model='openai/gpt-5-nano', thread='exp-order')
    assert reads == []


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


def test_two_builds_share_one_checkpointer_across_models_and_a_reset(monkeypatch):
    # this is a unit test
    """The deleted precursor to this test asserted the opposite of the
    contract this task establishes — that two builds got two different
    `.checkpointer` objects, so `reset()` (which runs on every credential
    change) forgot the conversation along with the client. This test builds
    two real agents offline, across two different models and across a
    `reset()` in between, and checks they hold the *identical* checkpointer
    object `conversation_memory.saver()` returns — not two `SqliteSaver`
    instances that merely point at the same file. A fake key is enough:
    `_build_agent` opens no socket, `ChatOpenAI(...)` is only constructed
    here, never called."""
    pytest.importorskip('langgraph')
    from raglab.agents.widget import conversation_memory as memory
    widget.reset()
    for name in widget.REQUIRED_ENV:
        monkeypatch.setenv(name, 'test-value')
    first = widget._build_agent('openai/gpt-5-nano').checkpointer
    widget.reset()
    second = widget._build_agent('openai/gpt-5-mini').checkpointer
    widget.reset()
    assert first is second is memory.saver(), (
        'every build must share the one process-wide checkpointer, across '
        'models and across reset() — a fresh one per build is the bug this '
        'task removes')


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


def test_ask_stamps_the_thread_state_through_the_graphs_own_input():
    # this is a unit test
    """`WidgetState`'s two fields are channels like `messages`, and `ask` writes
    them the same way — in the invoke input, so the checkpointer persists them
    in the same write rather than a second writer racing the graph. Without
    this, both fields stayed empty forever and `/api/widget/history` reported
    two blanks as facts about every thread. The stub records the payload the
    real graph would have been handed."""
    payloads = []

    class Stub:
        def invoke(self, payload, config=None):
            payloads.append(payload)
            return {'messages': [AIMessage(content='ok')]}

    widget.reset()
    widget._AGENTS['openai/gpt-5-nano'] = Stub()
    try:
        widget.ask('hello', model='openai/gpt-5-nano', thread='exp-stamped')
        widget.ask('hello', model='openai/gpt-5-nano')
    finally:
        widget.reset()
    assert payloads[0]['experiment_id'] == 'exp-stamped'
    assert payloads[0]['started_at']
    # The general thread belongs to no experiment, and says so with the empty
    # string rather than with the thread's own name or by leaving the field
    # out: it is an answer about that thread, not a gap in what is known.
    assert payloads[1]['experiment_id'] == ''
    assert payloads[1]['started_at']


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


# --- streaming: the answer arrives as it is written ------------------------
# The reply used to appear all at once, one round trip after Send, which reads
# as a stall and then a wall of text. `stream` is the same turn — the same
# graph, the same checkpointer, the same account — handed over in the order it
# was written. What these tests pin is that nothing else changed with it: a
# tool call is still not answer text, the final word on what was said is still
# the log's, and anything knowable before the first piece is still a refusal
# rather than a 200 that turns out to be an apology.

def _chunks(*texts):
    from langchain_core.messages import AIMessageChunk
    return [('messages', (AIMessageChunk(content=text), {})) for text in texts]


class _StreamStub:
    """An agent that streams: `stream` yields the pieces, then the state."""

    def __init__(self, events, final):
        self.events, self.final = events, final
        self.seen = {}

    def stream(self, payload, config=None, stream_mode=None):
        self.seen['payload'] = payload
        self.seen['config'] = config
        self.seen['stream_mode'] = stream_mode
        yield from self.events
        yield ('values', self.final)


def _streaming(agent, model='openai/gpt-5-nano'):
    widget.reset()
    widget._AGENTS[model] = agent
    return model


def test_stream_yields_the_answer_in_pieces_then_the_account():
    # this is a unit test
    """One event per piece while the answer is written, then exactly one reply
    event — the same dict `ask` returns, so a page can render either. The
    memory status follows it and is the last thing said; the reply is picked
    out by name rather than by position, the way the page picks it out."""
    from langchain_core.messages import HumanMessage
    stub = _StreamStub(_chunks('the ', 'ports ', 'are 9002'),
                       {'messages': [HumanMessage(content='which ports?'),
                                     AIMessage(content='the ports are 9002',
                                               usage_metadata={
                                                   'input_tokens': 30,
                                                   'output_tokens': 7,
                                                   'total_tokens': 37})]})
    model = _streaming(stub)
    try:
        events = list(widget.stream('which ports?', model=model, thread='exp-a'))
    finally:
        widget.reset()
    assert [e['delta'] for e in events if 'delta' in e] == ['the ', 'ports ',
                                                            'are 9002']
    replies = [e for e in events if 'reply' in e]
    assert replies == [{'reply': 'the ports are 9002',
                        'input_tokens': 30, 'output_tokens': 7}]
    assert 'memory' in events[-1]
    # The same thread the same way `ask` runs it, stamp included.
    assert stub.seen['config']['configurable']['thread_id'] == 'exp-a'
    assert stub.seen['payload']['experiment_id'] == 'exp-a'


def test_streaming_says_nothing_for_a_tool_call_or_its_result():
    # this is a unit test
    """A tool call's arguments arrive token by token on the very same channel
    the answer does, and so does the tool's reply. Neither is what the reader
    is watching being written — the widget's own log has never shown them."""
    from langchain_core.messages import (AIMessageChunk, HumanMessage,
                                         ToolMessage)
    calling = AIMessageChunk(content='', tool_call_chunks=[
        {'name': 'search_knowledge_base', 'args': '{"query": "por',
         'id': 'call-1', 'index': 0, 'type': 'tool_call_chunk'}])
    events = [('messages', (calling, {})),
              ('messages', (ToolMessage(content='9002', tool_call_id='call-1'), {}))]
    events += _chunks('9002 it is')
    stub = _StreamStub(events, {'messages': [HumanMessage(content='ports?'),
                                             AIMessage(content='9002 it is')]})
    model = _streaming(stub)
    try:
        said = list(widget.stream('ports?', model=model))
    finally:
        widget.reset()
    assert [e['delta'] for e in said if 'delta' in e] == ['9002 it is']


def test_streaming_names_the_tool_being_called():
    # this is a unit test
    """The one thing a tool call does say out loud is its own name — once, the
    moment the first chunk carries it, as `{'status': <name>}`. That is what
    the page shows while it waits; it is ephemeral, so the log never holds it.
    The argument tokens that follow name nothing (`name=None`) and stay
    silent, and the deltas and the final event are exactly what they were."""
    from langchain_core.messages import (AIMessageChunk, HumanMessage,
                                         ToolMessage)
    naming = AIMessageChunk(content='', tool_call_chunks=[
        {'name': 'search_knowledge_base', 'args': '{"query": "por',
         'id': 'call-1', 'index': 0, 'type': 'tool_call_chunk'}])
    continuing = AIMessageChunk(content='', tool_call_chunks=[
        {'name': None, 'args': 'ts?"}',
         'id': None, 'index': 0, 'type': 'tool_call_chunk'}])
    events = [('messages', (naming, {})),
              ('messages', (continuing, {})),
              ('messages', (ToolMessage(content='9002', tool_call_id='call-1'), {}))]
    events += _chunks('9002 it is')
    stub = _StreamStub(events, {'messages': [HumanMessage(content='ports?'),
                                             AIMessage(content='9002 it is')]})
    model = _streaming(stub)
    try:
        said = list(widget.stream('ports?', model=model))
    finally:
        widget.reset()
    statuses = [e for e in said if 'status' in e]
    assert statuses == [{'status': 'search_knowledge_base'}]
    assert said.index(statuses[0]) < said.index({'delta': '9002 it is'})
    assert [e['delta'] for e in said if 'delta' in e] == ['9002 it is']
    assert [e['reply'] for e in said if 'reply' in e] == ['9002 it is']


def test_streaming_says_which_stage_the_graph_just_ran():
    # this is a unit test
    """The third native stream mode: `updates` names the graph node that just
    ran, and each one is handed over as an ephemeral `{'stage': <label>}` in
    its place in the stream — before the deltas that follow it, after the ones
    that came first, and never instead of the reply.

    Two things it is not. It is not a tool announcement: a stage says which
    part of the run is working, a `status` names a tool being called, and the
    page shows them differently. And it never names a node — a node name is
    this package's private vocabulary, so a node the allowlist does not know
    arrives as the one fallback label instead of as itself."""
    from langchain_core.messages import HumanMessage
    events = [('updates', {'check_request.before_agent': {}})]
    events += _chunks('the ')
    events += [('updates', {'model': {}}),
               ('updates', {'tools': {}}),
               ('updates', {'a_node_this_allowlist_never_heard_of': {}})]
    events += _chunks('ports are 9002')
    stub = _StreamStub(events,
                       {'messages': [HumanMessage(content='which ports?'),
                                     AIMessage(content='the ports are 9002')]})
    model = _streaming(stub)
    try:
        said = list(widget.stream('which ports?', model=model, thread='exp-stage'))
    finally:
        widget.reset()
    assert stub.seen['stream_mode'] == ['messages', 'values', 'updates']
    assert [(key, event[key]) for event in said
            for key in ('stage', 'delta', 'reply') if key in event] == [
        ('stage', 'Reading the question'),
        ('delta', 'the '),
        ('stage', 'Working out the answer'),
        ('stage', 'Looking things up'),
        ('stage', 'Working on it'),
        ('delta', 'ports are 9002'),
        ('reply', 'the ports are 9002'),
    ]
    assert 'memory' in said[-1]
    assert 'a_node_this_allowlist_never_heard_of' not in str(said)


def test_every_stage_label_names_a_node_the_real_graph_has(monkeypatch):
    # this is a unit test
    """The allowlist's keys are langchain's and this package's own node names,
    read off the compiled graph rather than guessed. A langchain release that
    renames a node, or a middleware renamed here, would leave every stage
    falling back to the one generic label and nothing would fail — so the
    names are checked against the graph that produces them."""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-value')
    monkeypatch.delenv('LANGSMITH_TRACING', raising=False)
    graph = widget._build_agent('openai/gpt-5-nano').get_graph()
    assert set(widget.backends.STAGE_LABELS) <= set(graph.nodes)


def test_a_stage_is_never_written_down_and_never_replayed():
    # this is a unit test
    """A stage is what the reader watches happening, not what happened. It is
    gone the moment the turn lands: the operational row does not carry it, and
    a redraw of the thread shows the turns the conversation holds and no stage
    among them. The page's live progress is the only place one ever exists."""
    import json

    from langchain_core.messages import HumanMessage

    from raglab.agents.widget import conversation_memory as memory
    from raglab.agents.widget import turn_logger
    from raglab.agents.widget.tests.widget_examples import write_messages

    thread = 'exp-stage-log'
    memory.forget(thread)
    write_messages(thread, [HumanMessage(content='which ports?'),
                            AIMessage(content='the ports are 9002')])
    events = [('updates', {'tools': {}})] + _chunks('9002 too')
    stub = _StreamStub(events,
                       {'messages': [HumanMessage(content='and the board?'),
                                     AIMessage(content='9002 too')]})
    model = _streaming(stub)
    try:
        said = list(widget.stream('and the board?', model=model, thread=thread))
    finally:
        widget.reset()
    assert [event['stage'] for event in said if 'stage' in event] == [
        'Looking things up']
    row = turn_logger.list_turns(thread)[-1]
    assert row['ai_message'] == '9002 too'
    assert 'Looking things up' not in json.dumps(row, default=str)
    assert memory.history(thread)['turns'] == [
        {'role': 'you', 'text': 'which ports?'},
        {'role': 'bot', 'text': 'the ports are 9002'}]


def test_an_interrupted_row_never_bills_the_turn_before_it():
    # this is a unit test
    """The account of an interrupted turn is read back from the checkpoint,
    because a run that raised returned no state. The last question in the
    thread is this turn's in every ordinary failure — the graph writes the
    input before it does anything else — but a run that dies *in front of*
    that write leaves the previous turn at the head of the span.

    Reading it anyway would put the last turn's tokens and the last turn's
    steps on a row whose question is this one: the bill charged twice and a row
    lying about what produced it, which is this repo's first rule. So the span
    is claimed only when its head is the question being logged, and when it is
    not, the honest answer is that this run wrote nothing — no account, no
    steps."""
    import json

    from langchain_core.messages import HumanMessage

    from raglab.agents.widget import conversation_memory as memory
    from raglab.agents.widget import turn_logger
    from raglab.agents.widget.tests.widget_examples import write_messages

    thread = 'exp-before-the-write'
    memory.forget(thread)
    write_messages(thread, [
        HumanMessage(content='the turn before'),
        AIMessage(content='and its answer',
                  usage_metadata={'input_tokens': 900, 'output_tokens': 40,
                                  'total_tokens': 940})])

    class _NeverStarts:
        """A run that dies before the graph writes anything of its own."""

        def invoke(self, payload, config=None):
            raise RuntimeError('the checkpointer is locked')

    model = 'openai/gpt-5-nano'
    widget.reset()
    widget._AGENTS[model] = _NeverStarts()
    try:
        with pytest.raises(widget.WidgetUnavailable):
            widget.ask('the turn that never ran', model=model, thread=thread)
    finally:
        widget.reset()

    row = turn_logger.list_turns(thread)[-1]
    assert row['user_message'] == 'the turn that never ran'
    assert row['status'] == 'interrupted'
    # Not 900/40: those belong to the turn before, and they are already billed
    # on its own row. None says the bill is unknown, which is the truth here.
    assert row['total_input_tokens'] is None
    assert row['total_output_tokens'] is None
    assert row['total_tokens'] is None
    # And no invented trace either: an interrupted turn's steps are what
    # happened, or nothing.
    assert json.loads(row['steps_json']) == []


def test_a_stream_the_reader_walks_away_from_still_writes_its_row():
    # this is a unit test
    """A streamed turn dies three ways and all three owe a row.

    A run that raises mid-stream is the obvious one. The second is a reader who
    closes the tab: the route stops iterating, the generator is closed, and
    what arrives inside it is `GeneratorExit` — not an exception, so an
    ordinary `except Exception` never sees it, and the turn used to end with
    the graph's partial work in the thread and nothing anywhere else. The third
    is the run that streams pieces and then ends with no state to read the
    reply back from: the widget refuses to assemble an answer out of the
    fragments, which makes it a turn that produced no reply and therefore a
    turn that owes the same row. All three land as `interrupted`, each saying
    which it was."""
    class _Halts:
        def stream(self, payload, config=None, stream_mode=None):
            yield from _chunks('half a th')
            raise RuntimeError('the provider hung up')

    model = _streaming(_Halts())
    try:
        with pytest.raises(widget.WidgetUnavailable):
            list(widget.stream('why?', model=model, thread='exp-halt'))
    finally:
        widget.reset()

    class _Endless:
        def stream(self, payload, config=None, stream_mode=None):
            yield from _chunks('one ', 'two ', 'three ', 'four')

    model = _streaming(_Endless())
    try:
        events = widget.stream('and?', model=model, thread='exp-walk-away')
        assert next(events) == {'delta': 'one '}
        events.close()          # the reader closed the tab
    finally:
        widget.reset()

    class _NoState:
        def stream(self, payload, config=None, stream_mode=None):
            yield from _chunks('half an ans')

    model = _streaming(_NoState())
    try:
        with pytest.raises(widget.WidgetUnavailable):
            list(widget.stream('so?', model=model, thread='exp-no-state'))
    finally:
        widget.reset()

    from raglab.agents.widget import turn_logger
    halted = turn_logger.list_turns('exp-halt')
    walked = turn_logger.list_turns('exp-walk-away')
    stateless = turn_logger.list_turns('exp-no-state')
    assert [row['status'] for row in halted + walked + stateless] \
        == ['interrupted'] * 3
    assert halted[0]['status_reason'] == 'the provider hung up'
    assert walked[0]['status_reason'] == 'the reader closed the stream'
    assert walked[0]['user_message'] == 'and?'
    assert 'streamed no answer' in stateless[0]['status_reason']
    # The refusal the reader sees and the reason on the row are the same
    # sentence: a turn does not get to say one thing to a person and another
    # to the record.
    assert stateless[0]['ai_message'] is None


def test_the_last_word_on_the_answer_is_the_log_the_lab_kept():
    # this is a unit test
    """The pieces are how the answer arrived; the final event is what the lab
    now holds for that turn, read from the state the graph checkpointed with
    the same `_text` the history route reads back with. A page that showed the
    concatenated pieces and nothing else could drift from the transcript — so
    the reply is stated again, from the log, and the page adopts it."""
    from langchain_core.messages import HumanMessage
    stub = _StreamStub(_chunks('half an ans'),
                       {'messages': [HumanMessage(content='q'),
                                     AIMessage(content='half an answer, whole')]})
    model = _streaming(stub)
    try:
        events = list(widget.stream('q', model=model))
    finally:
        widget.reset()
    assert [e['reply'] for e in events if 'reply' in e] == [
        'half an answer, whole']


def test_the_event_after_the_streamed_reply_is_the_deferred_status(monkeypatch):
    # this is a unit test
    """The authoritative reply is emitted first, and the event after it says a
    decision is pending — not what the decision was.

    It carried the resolved decision until 2026-08-28, which it could only do
    by holding the event stream open through the policy call, the summarizer
    call and two readings of the board, every bit of it after the reader had
    the whole answer. The judgement is unchanged and still happens; it happens
    on a thread of its own, which is what this last event reports."""
    from langchain_core.messages import HumanMessage
    import threading

    stub = _StreamStub(
        _chunks('answer'),
        {'messages': [HumanMessage(content='q'), AIMessage(content='answer')]})
    judged = threading.Event()
    monkeypatch.setattr(widget.backends, '_memory_model', lambda model: object())
    monkeypatch.setattr(widget.backends, '_finish_memory',
                        lambda *args: judged.set())
    model = _streaming(stub)
    try:
        events = widget.stream('q', model=model)
        assert next(events) == {'delta': 'answer'}
        assert next(events) == {
            'reply': 'answer', 'input_tokens': None, 'output_tokens': None,
        }
        assert next(events) == {'memory': {'status': 'pending',
                                           'saved': False}}
        with pytest.raises(StopIteration):
            next(events)
        # Deferred, not dropped: the decision is being taken elsewhere.
        assert judged.wait(5)
    finally:
        widget.reset()


def test_a_cli_answer_arrives_as_one_piece_because_that_is_what_it_is(monkeypatch):
    # this is a unit test
    """A CLI is one process and one complete reply: there is no partial output
    to forward. It still travels the streaming path — one delta, then the
    account — rather than the page keeping a second way to ask.

    It also says nothing about stages, because it has none to report: a
    subprocess runs no graph, so a stage event here would be this path
    inventing progress it never saw."""
    monkeypatch.setattr(
        widget.backends, '_cli_answer',
        lambda cli, message: AIMessage(content='from the cli', usage_metadata={
            'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15}))
    monkeypatch.setattr(widget.backends, 'cli_available', lambda cli: True)
    events = list(widget.stream('hello', model='claude'))
    assert [e['delta'] for e in events[:-1]] == ['from the cli']
    assert events[-1] == {'reply': 'from the cli',
                          'input_tokens': 12, 'output_tokens': 3}
    assert not [e for e in events if 'stage' in e]


def test_an_unknown_model_is_refused_before_a_single_piece_is_yielded():
    # this is a unit test
    """`stream` is a call that returns an iterator, not a generator function:
    everything knowable up front is raised here, where the route can still
    answer it with a status code."""
    with pytest.raises(ValueError):
        widget.stream('hello', model='gpt-9')


def test_a_missing_key_refuses_before_the_stream_opens(monkeypatch):
    # this is a unit test
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    credentials.clear()
    widget.reset()
    with pytest.raises(widget.WidgetUnavailable) as caught:
        widget.stream('hello', model='openai/gpt-5-nano')
    assert 'OPENROUTER_API_KEY' in str(caught.value)


def _sse(response) -> list[dict]:
    import json
    return [json.loads(line[len('data: '):])
            for line in response.text.splitlines()
            if line.startswith('data: ')]


def test_the_stream_route_sends_the_pieces_then_the_account(client, monkeypatch):
    # this is an integration test
    """Server-sent events, one JSON object per line: the deltas as they come,
    the account last. The route carries them and adds nothing — a stage event
    included, which is why it needs no clause of its own here: the route
    forwards whatever the turn yields, in the order it was yielded, and holds
    nothing back until the answer is whole."""
    monkeypatch.setattr(widget, 'stream',
                        lambda message, model='', thread='': iter(
                            [{'stage': 'Reading the question'},
                             {'delta': 'he'}, {'delta': 'llo'},
                             {'reply': 'hello', 'input_tokens': 4,
                              'output_tokens': 1}]))
    answer = client.post('/api/widget/stream', json={'message': 'hi'})
    assert answer.status_code == 200
    assert answer.headers['content-type'].startswith('text/event-stream')
    assert _sse(answer) == [{'stage': 'Reading the question'},
                            {'delta': 'he'}, {'delta': 'llo'},
                            {'reply': 'hello', 'input_tokens': 4,
                             'output_tokens': 1}]


def test_the_stream_route_refuses_an_empty_message(client):
    # this is an integration test
    answer = client.post('/api/widget/stream', json={'message': '   '})
    assert answer.status_code == 400


def test_the_stream_route_passes_the_model_and_thread_through(client, monkeypatch):
    # this is an integration test
    seen = {}

    def fake_stream(message, model='', thread=''):
        seen.update(message=message, model=model, thread=thread)
        return iter([{'reply': 'ok', 'input_tokens': None,
                      'output_tokens': None}])

    monkeypatch.setattr(widget, 'stream', fake_stream)
    client.post('/api/widget/stream', json={'message': 'hello',
                                            'model': 'openai/gpt-5-mini',
                                            'thread': 'exp-one'})
    assert seen == {'message': 'hello', 'model': 'openai/gpt-5-mini',
                    'thread': 'exp-one'}


def test_the_stream_route_sanitizes_memory_on_the_authoritative_final_event(
        client, monkeypatch):
    # this is an integration test
    """The final event remains the answer the browser adopts, with only the
    safe memory status added beside it; the request's thread is unchanged.

    The shape a streamed turn really sends since 2026-08-28: the reply event
    on its own, then a separate memory event carrying the deferred status. The
    verdict is taken after the connection closes, so no event here can hold
    one, and the internal reason never reaches the wire."""
    seen = {}

    def fake_stream(message, model='', thread=''):
        seen.update(message=message, model=model, thread=thread)
        return iter([
            {'reply': 'answer from the lab', 'input_tokens': 4,
             'output_tokens': 2},
            {'memory': {'status': 'pending', 'saved': False,
                        'reason': 'private policy reasoning'}},
        ])

    monkeypatch.setattr(widget, 'stream', fake_stream)
    answer = client.post('/api/widget/stream', json={
        'message': 'what should I retain?', 'model': 'openai/gpt-5-mini',
        'thread': 'exp-stream'})

    assert answer.status_code == 200
    assert _sse(answer) == [
        {'reply': 'answer from the lab', 'input_tokens': 4,
         'output_tokens': 2},
        {'memory': {'status': 'pending'}},
    ]
    assert seen == {'message': 'what should I retain?',
                    'model': 'openai/gpt-5-mini', 'thread': 'exp-stream'}


@pytest.mark.parametrize('memory', [
    {},
    {'relevant': 'false', 'saved': True},
    {'relevant': True, 'saved': 'yes'},
])
def test_the_stream_route_omits_status_for_malformed_or_missing_policy_booleans(
        client, monkeypatch, memory):
    # this is an integration test
    """Reader metadata must not turn malformed policy fields into a false
    claim that the request was irrelevant."""
    monkeypatch.setattr(widget, 'stream', lambda message, model='', thread='': iter([{
        'reply': 'answer from the lab', 'input_tokens': 4,
        'output_tokens': 2, 'memory': memory,
    }]))

    answer = client.post('/api/widget/stream', json={'message': 'hello'})

    assert answer.status_code == 200
    assert _sse(answer) == [{
        'reply': 'answer from the lab', 'input_tokens': 4,
        'output_tokens': 2,
    }]


def test_an_unavailable_widget_is_a_502_before_the_stream_opens(client, monkeypatch):
    # this is an integration test
    """The refusal is raised by the call, not by the iterator, so it is still a
    status code rather than a 200 whose body apologises."""
    def refuse(message, model='', thread=''):
        raise widget.WidgetUnavailable('OPENROUTER_API_KEY is not set')

    monkeypatch.setattr(widget, 'stream', refuse)
    answer = client.post('/api/widget/stream', json={'message': 'hello'})
    assert answer.status_code == 502
    assert 'OPENROUTER_API_KEY' in answer.json()['detail']


def test_a_failure_halfway_through_arrives_as_an_error_event(client, monkeypatch):
    # this is an integration test
    """Once the first piece is out the status code is spent, so the only
    honest place left to say the answer never finished is the stream itself —
    never a truncated reply the page would render as a whole one."""
    def half_then_fail(message, model='', thread=''):
        def events():
            yield {'stage': 'Working out the answer'}
            yield {'delta': 'the answer beg'}
            raise widget.WidgetUnavailable('the widget could not answer: gone')
        return events()

    monkeypatch.setattr(widget, 'stream', half_then_fail)
    answer = client.post('/api/widget/stream', json={'message': 'hello'})
    assert answer.status_code == 200
    said = _sse(answer)
    assert said[0] == {'stage': 'Working out the answer'}
    assert said[1] == {'delta': 'the answer beg'}
    assert 'could not answer' in said[-1]['error']
    assert not any('reply' in event for event in said)


def _standing_line(text: str, mark: str):
    """A system line as `backends._run` writes one — the marker included, since
    that is what says which standing line it is."""
    from langchain_core.messages import SystemMessage
    from raglab.agents.widget import conversation_memory as memory
    return SystemMessage(content=text,
                         additional_kwargs={memory.STANDING_LINE: mark})


def test_the_trim_keeps_every_system_line_and_the_newest_of_the_rest():
    # this is a unit test
    """The thread's system lines — which experiment it is about, the memory
    context — are written at the top. A trim that kept only the newest twenty
    messages would drop them on the twenty-first, and the model would forget
    mid-conversation which experiment it was discussing. System lines survive
    the trim; the window counts the rest."""
    from langchain_core.messages import AIMessage, HumanMessage
    from raglab.agents.widget import conversation_memory as memory
    seen = {}

    tail = [m for i in range(15) for m in (HumanMessage(content=f'q{i}'),
                                           AIMessage(content=f'a{i}'))]
    request = _FakeModelRequest([
        _standing_line('about exp-1', memory.IDENTITY_LINE),
        _standing_line('memory', memory.MEMORY_LINE), *tail])
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))
    kept = seen['messages']
    assert [m.content for m in kept[:2]] == ['about exp-1', 'memory']
    assert kept[2:] == tail[-widget.MAX_HISTORY:]


def test_only_the_newest_standing_memory_line_is_sent_to_the_model():
    # this is a unit test
    """The finding: system lines are exempt from the window, the memory context
    grows on every accepted turn, and the thread keeps every version — so the
    model was handed several, oldest first, the stale ones contradicting the
    newest. The call carries the newest of each kind. The thread is untouched:
    `request.override` shapes one prompt, it does not rewrite the log."""
    from langchain_core.messages import HumanMessage
    from raglab.agents.widget import conversation_memory as memory
    seen = {}

    lines = [_standing_line('about exp-1', memory.IDENTITY_LINE),
             _standing_line('memory v1', memory.MEMORY_LINE),
             _standing_line('memory v2', memory.MEMORY_LINE),
             _standing_line('memory v3', memory.MEMORY_LINE)]
    given = [*lines, HumanMessage(content='q')]
    request = _FakeModelRequest(given)
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))

    assert [m.content for m in seen['messages']] == ['about exp-1',
                                                     'memory v3', 'q']
    assert request.messages == given


def test_a_system_line_the_widget_did_not_write_is_always_sent():
    # this is a unit test
    """The filter drops a line only when a newer line says it is superseded,
    and only a line the widget marked can say that. A system message from
    anywhere else — a future middleware's instruction, or a line written before
    the marker existed — is not the widget's standing text to supersede, so it
    goes to the model whatever else the thread holds. Guessing from the text
    instead is how such an instruction would silently disappear."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from raglab.agents.widget import conversation_memory as memory
    seen = {}

    request = _FakeModelRequest([
        SystemMessage(content='SAFETY: never quote a key'),
        SystemMessage(content='an unmarked memory line from an old thread'),
        _standing_line('memory v1', memory.MEMORY_LINE),
        _standing_line('memory v2', memory.MEMORY_LINE),
        HumanMessage(content='q')])
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))

    assert [m.content for m in seen['messages']] == [
        'SAFETY: never quote a key',
        'an unmarked memory line from an old thread',
        'memory v2', 'q']


def _turn(question: str, tool_text: str, answer: str, call_id: str) -> list:
    """One tool-using turn: the question, the model asking for something, what
    came back, and the answer written from it."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    return [HumanMessage(content=question),
            AIMessage(content='', tool_calls=[
                {'name': 'read_rag_skill', 'args': {'names': 'chunking'},
                 'id': call_id}]),
            ToolMessage(content=tool_text, tool_call_id=call_id,
                        name='read_rag_skill'),
            AIMessage(content=answer)]


def test_a_closed_turns_tool_reply_is_sent_as_a_stub_and_kept_in_the_thread():
    # this is a unit test
    """The measured finding: one `read_rag_skill` reply is about 20,000
    characters, and the count window then re-sent it on every call for the next
    twenty messages. A turn the model has already answered from does not need
    the bodies again — it needs to know they were read and how to read them
    again — so the prompt carries a stub naming the tool and its subject.

    The request is shaped; the thread is not. `request.messages` is unchanged
    after the call, which is what makes this a smaller prompt rather than a
    rewritten record."""
    from langchain_core.messages import HumanMessage
    seen = {}

    body = 'a skill body. ' * 1_500
    given = _turn('what about chunking?', body, 'chunk by heading.', 'c1') \
        + [HumanMessage(content='and rerankers?')]
    request = _FakeModelRequest(given)
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))

    sent = [str(m.content) for m in seen['messages']]
    assert body not in sent
    assert sent[2].startswith('[read_rag_skill(') and 'names=chunking' in sent[2]
    assert str(len(body)) in sent[2]
    # Everything else the turn said is still there: the reduction is of the
    # evidence the model has finished with, not of the conversation.
    assert sent[0] == 'what about chunking?' and sent[3] == 'chunk by heading.'
    assert request.messages == given


def test_the_tool_replies_of_the_turn_being_answered_are_never_stubbed():
    # this is a unit test
    """The quality fence. A model reasoning over what a tool has just returned
    always has the whole of it — that is the state every answering call is in,
    since a turn the model has not answered yet cannot be closed. Two turns
    here: the finished one travels as a stub, the one in flight travels whole.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    seen = {}

    old, new = 'the old body. ' * 1_500, 'the new body. ' * 1_500
    given = _turn('what about chunking?', old, 'chunk by heading.', 'c1') + [
        HumanMessage(content='and rerankers?'),
        AIMessage(content='', tool_calls=[
            {'name': 'read_rag_skill', 'args': {'names': 'rerankers'},
             'id': 'c2'}]),
        ToolMessage(content=new, tool_call_id='c2', name='read_rag_skill')]
    request = _FakeModelRequest(given)
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))

    sent = [str(m.content) for m in seen['messages']]
    assert old not in sent
    assert sent[-1] == new


def test_the_window_is_bounded_by_characters_as_well_as_by_messages():
    # this is a unit test
    """Twenty messages can be five hundred characters or forty thousand, and
    the count cannot tell them apart. `MAX_HISTORY_CHARS` is the second
    ceiling: history is dropped a whole turn at a time, oldest first, until it
    fits.

    Whole turns rather than single messages, because the count window can
    already cut between an assistant's tool call and the reply to it. And the
    turn being answered is neither trimmed nor counted — a turn that has just
    read 20,000 characters would otherwise spend the entire budget on itself
    and throw away the conversation it belongs to."""
    from langchain_core.messages import AIMessage, HumanMessage
    seen = {}

    # Six turns of two messages each — inside MAX_HISTORY, far outside the
    # character budget.
    fat = 'x' * 6_000
    given = [m for i in range(6)
             for m in (HumanMessage(content=f'q{i}'),
                       AIMessage(content=f'{fat}{i}'))]
    request = _FakeModelRequest(given)
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))

    sent = seen['messages']
    assert len(sent) < len(given) <= widget.MAX_HISTORY
    # What is left starts at a question, never mid-turn — the two oldest turns
    # went, which is the fewest that brings the history inside the budget.
    assert sent[0].content == 'q2'
    # ...the history in front of the current turn is inside the budget...
    history = sum(len(str(m.content)) for m in sent[:-2])
    assert history <= widget.MAX_HISTORY_CHARS
    # ...and the turn being answered survives whatever it costs.
    assert [str(m.content) for m in sent[-2:]] == ['q5', f'{fat}5']
    assert request.messages == given


def test_an_enormous_current_turn_does_not_empty_the_conversation():
    # this is a unit test
    """The budget is spent on history and never on the turn being answered, so
    a turn that has just read three skill bodies does not cost the model every
    earlier turn as well. Counting the current turn would be almost as bad as
    trimming it: the conversation would lose its context at exactly the moment
    the reader asked something that needed a big lookup."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    seen = {}

    chat = [m for i in range(3)
            for m in (HumanMessage(content=f'q{i}'),
                      AIMessage(content=f'a{i}'))]
    given = chat + [
        HumanMessage(content='and the skills?'),
        AIMessage(content='', tool_calls=[
            {'name': 'read_rag_skill', 'args': {'names': 'chunking'},
             'id': 'c9'}]),
        ToolMessage(content='y' * 40_000, tool_call_id='c9',
                    name='read_rag_skill')]
    request = _FakeModelRequest(given)
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))

    sent = [str(m.content) for m in seen['messages']]
    assert sent[:6] == ['q0', 'a0', 'q1', 'a1', 'q2', 'a2']
    assert sent[-1] == 'y' * 40_000


def test_the_budget_drops_whole_turns_so_a_tool_reply_keeps_the_call_that_asked():
    # this is a unit test
    """The reason `_within_budget` drops turns and not messages, fenced.

    A budget that walked the list dropping one message at a time until it fit
    would, on this thread, hand the model `[tool reply, answer, question]` — a
    tool reply with no tool call in front of it. Some providers reject that
    outright and every provider reads it as an answer to a question nobody
    asked, which is the one shape a prompt this careful must never emit. The
    assertions below fail on exactly that mutation: the window either starts at
    a reader's question or it does not."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    seen = {}

    # The bulk sits on the assistant's tool-call message, so stubbing the tool
    # reply cannot bring the turn back inside the budget: it has to go whole.
    given = [HumanMessage(content='the long one?'),
             AIMessage(content='x' * 25_000, tool_calls=[
                 {'name': 'read_rag_skill', 'args': {'names': 'chunking'},
                  'id': 'c1'}]),
             ToolMessage(content='a body', tool_call_id='c1',
                         name='read_rag_skill'),
             AIMessage(content='chunk by heading.'),
             HumanMessage(content='and now?')]
    request = _FakeModelRequest(given)
    widget.trim_and_call.wrap_model_call(
        request, lambda r: seen.setdefault('messages', r.messages))

    sent = seen['messages']

    # The invariant first, because it is the durable one: every tool reply the
    # window carries is preceded by the call it answers, whatever a future trim
    # decides to drop. A message-wise budget fails here, on `[tool, answer,
    # question]`, before the exact-window assertion below is ever reached.
    for i, message in enumerate(sent):
        if getattr(message, 'type', '') != 'tool':
            continue
        asked = {call.get('id') for earlier in sent[:i]
                 for call in (getattr(earlier, 'tool_calls', None) or [])}
        assert message.tool_call_id in asked, (
            'the window handed the model a tool reply whose call it dropped')

    # And what this thread in particular comes to: the turn went whole.
    assert [m.type for m in sent] == ['human']
    assert str(sent[0].content) == 'and now?'
    assert request.messages == given
