"""LabSettings -> a LangChain chat model, chosen by `RAGLAB_LLM`, plus the offline fake.

'openrouter' and 'ollama' both build through `init_chat_model` as OpenAI-compatible
endpoints rather than ChatOllama, which the `langchain-openai<1` pin ragas 0.4
requires does not cover; `claude`/`codex` run a CLI subprocess instead (`clichat.py`)
but arrive through this same function.
"""
from typing import Any, Optional, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from . import clichat

# Separate constants because a local model under concurrent load is far slower
# than a remote API — the right value is a property of where the model runs.
REMOTE_TIMEOUT = 90
LOCAL_TIMEOUT = 600

# A CLI call is a process spawn, throttled by `ragas_eval.JUDGE_LOAD` rather
# than a rate limit; it gets ollama's local patience for the same reason.
CLI_TIMEOUT = 600


def _endpoint(settings) -> tuple[str, str, int]:
    """Base url, api key, and timeout for this backend; raises for anything else — no auto modes."""
    if settings.provider == 'openrouter':
        # `or 'missing'`: init_chat_model refuses to build with no key, and a
        # lab with no key must still start; the failure lands as a 401 at call time.
        return (settings.openrouter_base_url,
                settings.openrouter_api_key or 'missing', REMOTE_TIMEOUT)
    if settings.provider == 'ollama':
        # A placeholder, deliberately not the OpenRouter key: this request goes
        # to localhost, and a real credential must never be sent where it wasn't issued for.
        return settings.ollama_base_url, 'ollama', LOCAL_TIMEOUT
    raise ValueError(f'unknown RAGLAB_LLM {settings.provider!r}')


def make_chat_model(settings, model: str = '') -> BaseChatModel:
    """A model for the configured backend; the provider is never inferred from model text."""
    if settings.provider == 'fake':
        return FakeChat()
    if settings.provider in clichat.CLIS:
        # Not an endpoint: no base url or key, since the credential is this
        # machine's own CLI login. See clichat.py for why each argv flag matters.
        return clichat.CliChat(
            cli=settings.provider, model=model or settings.llm_model,
            effort=clichat.checked_effort(settings.provider,
                                          settings.cli_effort),
            timeout=CLI_TIMEOUT)
    base_url, api_key, timeout = _endpoint(settings)
    return init_chat_model(model or settings.llm_model, model_provider='openai',
                           base_url=base_url, api_key=api_key, timeout=timeout)


def lab_llm(settings):
    """The lab's chat model, or the offline fake — the lab must stay runnable with no network at all."""
    return make_chat_model(settings)


def judge_llm(settings, model: str = ''):
    """The client RAGAS judges with, built the same way so a local judge is possible.

    RAGAS binds the model at construction and never forwards one per request,
    so passing the judge slug here is the only way it reaches the wire."""
    return make_chat_model(settings, model)


def lab_chat(llm, messages: list[dict], model: str = '') -> BaseMessage:
    """Every LLM-backed lab step calls the model through here; empty `model` means the client's own.

    A branch rather than a default argument: passing model='' through to
    invoke() would put a null model on the wire instead of omitting it."""
    return llm.invoke(messages, model=model) if model else llm.invoke(messages)


def _text(message) -> str:
    """Text of a message, whether it's a plain dict (ours) or a LangChain message object."""
    if isinstance(message, dict):
        return message.get('content', '') or ''
    content = getattr(message, 'content', '')
    if isinstance(content, str):
        return content
    # Some providers return content blocks; only the text parts matter here.
    return ''.join(part.get('text', '') for part in content
                   if isinstance(part, dict))


class FakeChat(BaseChatModel):
    """Deterministic offline chat model, selected via `RAGLAB_LLM=fake`; echoes the last user message.

    Lives here rather than in the test tree because 'fake' is a real backend;
    LangChain's own fakes report no usage, leaving the token/cost path untested offline."""

    script: Optional[list[BaseMessage]] = None

    @property
    def _llm_type(self) -> str:
        return 'fake'

    def bind_tools(self, tools: Sequence, **kwargs: Any) -> 'FakeChat':
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        message = self._next(messages)
        # A rough estimate; it only needs to be non-zero and additive to exercise reporting.
        if message.usage_metadata is None:
            spent_in = sum(len(_text(m)) for m in messages) // 4
            spent_out = len(_text(message)) // 4
            message = message.model_copy(update={'usage_metadata': {
                'input_tokens': spent_in, 'output_tokens': spent_out,
                'total_tokens': spent_in + spent_out}})
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        if self.script:
            return self.script.pop(0)
        last_user = next((m for m in reversed(messages)
                          if isinstance(m, HumanMessage)
                          or (isinstance(m, dict) and m.get('role') == 'user')),
                         None)
        if last_user is None:
            return AIMessage(content='FAKE: ')
        return AIMessage(content=f'FAKE: {_text(last_user).strip()}')
