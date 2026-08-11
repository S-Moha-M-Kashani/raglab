"""The lab's own chat-model seam, after the move off lodestar_brain."""
from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage

from raglab import llm


@dataclass(frozen=True)
class Stub:
    """Only the fields the factory reads. Task 4 brings the real LabSettings."""
    provider: str = 'fake'
    llm_model: str = 'openai/gpt-5-nano'
    openrouter_api_key: str = ''
    openrouter_base_url: str = 'https://openrouter.ai/api/v1'
    ollama_base_url: str = 'http://localhost:11434/v1'


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
