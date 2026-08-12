"""The lab's own chat-model seam, after the move off lodestar_brain."""
import json
import subprocess
from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage

from raglab import clichat, llm


@dataclass(frozen=True)
class Stub:
    """Only the fields the factory reads. Task 4 brings the real LabSettings."""
    provider: str = 'fake'
    llm_model: str = 'openai/gpt-5-nano'
    openrouter_api_key: str = ''
    openrouter_base_url: str = 'https://openrouter.ai/api/v1'
    ollama_base_url: str = 'http://localhost:11434/v1'
    cli_effort: str = 'low'


# This is a unit test.
def test_the_fake_backend_answers_without_a_network():
    """`fake` is what the whole suite runs on, so it must never reach a wire and
    must always answer."""
    model = llm.make_chat_model(Stub(provider='fake'))
    reply = model.invoke([{'role': 'user', 'content': 'سلام'}])
    assert isinstance(reply, AIMessage)
    assert reply.content == 'FAKE: سلام'
    # A fake reporting no usage leaves the cost path untestable offline.
    assert reply.usage_metadata['total_tokens'] > 0


# This is a unit test.
def test_each_real_backend_is_built_for_its_own_endpoint_and_patience():
    """The two real backends differ in nothing but where the model runs, so one
    construction site holds both — and a local model gets the longer timeout,
    because the 90s that is generous for a remote API is what lost a local
    judged run three of its four deciding metrics to TimeoutError."""
    remote = llm.make_chat_model(Stub(provider='openrouter',
                                     openrouter_api_key='sk-test'))
    local = llm.make_chat_model(Stub(provider='ollama'))
    assert remote.openai_api_base == 'https://openrouter.ai/api/v1'
    assert local.openai_api_base == 'http://localhost:11434/v1'
    assert remote.request_timeout == llm.REMOTE_TIMEOUT
    assert local.request_timeout == llm.LOCAL_TIMEOUT
    # A real credential must never leave for localhost, however harmless the
    # listener.
    assert local.openai_api_key.get_secret_value() == 'ollama'


# This is a unit test.
def test_an_unknown_backend_raises_instead_of_falling_back():
    """No auto modes: the old behaviour was a silent downgrade to openrouter,
    which is one config quietly billing an API on whichever machine had the
    daemon down."""
    with pytest.raises(ValueError, match='unknown RAGLAB_LLM'):
        llm.make_chat_model(Stub(provider='wat'))


# This is a unit test.
def test_a_per_role_model_is_forwarded_per_request_not_bound():
    """An empty model means "whatever the client was built with"; a named one is
    forwarded per request, so one client still serves every role. Passing '' to
    invoke() would put a null model on the wire, which is why this is a branch."""
    captured = {}

    class Recorder:
        def invoke(self, messages, **kwargs):
            captured.update(kwargs)
            return AIMessage(content='ok')

    llm.lab_chat(Recorder(), [{'role': 'user', 'content': 'hi'}])
    assert 'model' not in captured
    llm.lab_chat(Recorder(), [{'role': 'user', 'content': 'hi'}], model='qwen3.5:2b')
    assert captured['model'] == 'qwen3.5:2b'


# This is a unit test.
def test_a_cli_backend_is_built_through_the_same_one_construction_site():
    """The seam's whole claim: adding a backend is a branch here and never an
    edit to a call site. A CLI is not an endpoint, so it takes no base url — and
    it gets the local timeout, because a process spawn plus an agent turn is not
    a thing to be patient about for 90 seconds only."""
    model = llm.make_chat_model(Stub(provider='claude', llm_model='sonnet'))
    assert model.cli == 'claude' and model.model == 'sonnet'
    assert model.timeout == llm.CLI_TIMEOUT
    assert llm.CLI_TIMEOUT > llm.REMOTE_TIMEOUT
    codex = llm.make_chat_model(Stub(provider='codex',
                                     llm_model='gpt-5.6-terra'))
    assert codex.cli == 'codex'


# This is a unit test.
def test_the_configured_effort_is_the_one_that_reaches_the_argv(monkeypatch):
    """The entire justification for making effort a setting is that it moves the
    numbers — the grade probe scored 8 under `low` where the default scored 9 — and
    nothing pinned it past the factory. `CliChat.effort` defaults to `low` too, so
    hardcoding `'low'` in `make_chat_model` left the whole suite green while the
    knob silently stopped working, and every row it labelled would have been
    measured at a different effort than the one that was asked for. So this follows
    one non-default value from the settings object to the argv the process is
    actually spawned with."""
    calls = []

    def record(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({'is_error': False, 'usage': {},
                                 'result': '1: 8'}), '')

    monkeypatch.setattr(clichat.subprocess, 'run', record)
    model = llm.make_chat_model(Stub(provider='claude', llm_model='sonnet',
                                     cli_effort='xhigh'))
    assert model.effort == 'xhigh'
    model.invoke([{'role': 'user', 'content': 'hi'}])
    assert calls[0][calls[0].index('--effort') + 1] == 'xhigh'


# This is a unit test.
def test_a_reasoning_effort_the_backend_rejects_stops_the_run_at_build_time():
    """Earlier than the call, because codex answers an effort it does not accept
    with exit 0 and no text — and the lab would rather refuse to build a client
    than produce a stage that scored every document 0.5."""
    llm.make_chat_model(Stub(provider='codex', llm_model='gpt-5.6-terra',
                             cli_effort='none'))
    with pytest.raises(ValueError, match='none'):
        llm.make_chat_model(Stub(provider='claude', llm_model='sonnet',
                                 cli_effort='none'))
