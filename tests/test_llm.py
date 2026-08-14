"""The lab's own chat-model seam, after the move off lodestar_brain."""
import json
import subprocess
from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage

from raglab import clichat, llm


@dataclass(frozen=True)
class Stub:
    """Only the fields the factory reads."""
    provider: str = 'fake'
    llm_model: str = 'openai/gpt-5-nano'
    openrouter_api_key: str = ''
    openrouter_base_url: str = 'https://openrouter.ai/api/v1'
    ollama_base_url: str = 'http://localhost:11434/v1'
    cli_effort: str = 'low'


def test_the_fake_backend_answers_without_a_network():
    model = llm.make_chat_model(Stub(provider='fake'))
    reply = model.invoke([{'role': 'user', 'content': 'سلام'}])
    assert isinstance(reply, AIMessage)
    assert reply.content == 'FAKE: سلام'
    # A fake reporting no usage leaves the cost path untestable offline.
    assert reply.usage_metadata['total_tokens'] > 0


def test_each_real_backend_is_built_for_its_own_endpoint_and_patience():
    """openrouter and ollama differ in nothing but endpoint and timeout — a
    local model needs longer than the remote-API 90s allows."""
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


def test_an_unknown_backend_raises_instead_of_falling_back():
    """No silent downgrade to openrouter, which would quietly bill an API on
    whichever machine had the local daemon down."""
    with pytest.raises(ValueError, match='unknown RAGLAB_LLM'):
        llm.make_chat_model(Stub(provider='wat'))


def test_a_per_role_model_is_forwarded_per_request_not_bound():
    """An empty model means "whatever the client was built with" rather than
    a null model literally forwarded to invoke()."""
    captured = {}

    class Recorder:
        def invoke(self, messages, **kwargs):
            captured.update(kwargs)
            return AIMessage(content='ok')

    llm.lab_chat(Recorder(), [{'role': 'user', 'content': 'hi'}])
    assert 'model' not in captured
    llm.lab_chat(Recorder(), [{'role': 'user', 'content': 'hi'}], model='qwen3.5:2b')
    assert captured['model'] == 'qwen3.5:2b'


def test_a_cli_backend_is_built_through_the_same_one_construction_site():
    """A CLI is not an endpoint, so it takes no base url, and gets the local
    timeout since a process spawn plus an agent turn needs more than 90s."""
    model = llm.make_chat_model(Stub(provider='claude', llm_model='sonnet'))
    assert model.cli == 'claude' and model.model == 'sonnet'
    assert model.timeout == llm.CLI_TIMEOUT
    assert llm.CLI_TIMEOUT > llm.REMOTE_TIMEOUT
    codex = llm.make_chat_model(Stub(provider='codex',
                                     llm_model='gpt-5.6-terra'))
    assert codex.cli == 'codex'


def test_the_configured_effort_is_the_one_that_reaches_the_argv(monkeypatch):
    """`CliChat.effort` defaults to `low` too, so hardcoding `'low'` in
    `make_chat_model` would leave the suite green while the knob silently
    stopped reaching the argv."""
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


def test_a_reasoning_effort_the_backend_rejects_stops_the_run_at_build_time():
    """Refused at build time, not at the call: codex answers an unaccepted
    effort with exit 0 and no text, which elsewhere reads as no opinion."""
    llm.make_chat_model(Stub(provider='codex', llm_model='gpt-5.6-terra',
                             cli_effort='none'))
    with pytest.raises(ValueError, match='none'):
        llm.make_chat_model(Stub(provider='claude', llm_model='sonnet',
                                 cli_effort='none'))
