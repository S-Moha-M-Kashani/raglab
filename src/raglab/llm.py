"""The lab's LLM seam: LabSettings -> a LangChain chat model, plus the fake.

One function, chosen by `RAGLAB_LLM`. Adding a backend means adding a branch
here, never editing a call site.

Two backends serve a real model and differ only in *where it runs*: 'openrouter'
pays a remote API, 'ollama' talks to a model on this machine. Both are built
through `init_chat_model` as OpenAI-compatible endpoints, because Ollama serves
an OpenAI-compatible /v1 — so a local model costs no new dependency, and it works
under the `langchain-openai<1` that ragas 0.4 requires, which ChatOllama is not
covered by. Since the two differ in nothing but the endpoint and the patience it
deserves, `_endpoint` holds that difference and there is one construction site: a
knob that applies to both cannot be added to one and forgotten on the other.

A third kind arrived 2026-08-12 and is not an endpoint at all: `claude` and
`codex` run a CLI on this machine as a subprocess (`clichat.py`). They still
arrive through this one function, which is the whole point of it.

Until 2026-08-11 this module translated LabSettings into lodestar_brain's
Settings and called that project's factory, so there would be exactly one LLM
path in one repository. With the lab standalone there is nothing to share with,
and the translation step was the only thing the indirection bought.
"""
from typing import Any, Optional, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from . import clichat

# A remote API answers in seconds; a local model on a laptop does not. Measured
# on gemma4:e2b judging this lab: a single call took ~8s, but under three
# concurrent requests individual calls reached 80–92s — so the 90s that is
# generous for OpenRouter is the exact reason a local judged run lost three of
# its four deciding metrics to TimeoutError. Two constants rather than one
# setting, because the right value is a property of *where the model runs*.
REMOTE_TIMEOUT = 90
LOCAL_TIMEOUT = 600

# A CLI backend is a process spawn *and* an agent turn, and the process is what
# `ragas_eval.JUDGE_LOAD` throttles rather than a rate limit. Local patience for
# the same reason ollama gets it: the right value is a property of where the
# model runs.
CLI_TIMEOUT = 600


def _endpoint(settings) -> tuple[str, str, int]:
    """Where this backend's model lives, what to authenticate with, how long to
    wait. Raises for anything else — no auto modes."""
    if settings.provider == 'openrouter':
        # `or 'missing'`: init_chat_model refuses to build without a key, and a
        # lab with no key must still start and serve the panel. The failure
        # lands as a 401 at call time.
        return (settings.openrouter_base_url,
                settings.openrouter_api_key or 'missing', REMOTE_TIMEOUT)
    if settings.provider == 'ollama':
        # Ollama authenticates nothing, but the client still demands a key, so a
        # placeholder goes on the wire. Deliberately *not* the OpenRouter key:
        # this request leaves for localhost, and a real credential must never be
        # sent somewhere it was not issued for, however harmless the listener.
        return settings.ollama_base_url, 'ollama', LOCAL_TIMEOUT
    raise ValueError(f'unknown RAGLAB_LLM {settings.provider!r}')


def make_chat_model(settings, model: str = '') -> BaseChatModel:
    """Build a model for the configured backend.

    A provider is never inferred from model text: a local-looking slug must not
    quietly redirect to a paid API, and a remote one must not quietly start a
    daemon call.
    """
    if settings.provider == 'fake':
        return FakeChat()
    if settings.provider in clichat.CLIS:
        # Not an endpoint, so no base url and no key: the credential is the
        # login this machine already has. See clichat.py for why each flag in
        # the argv is load-bearing.
        return clichat.CliChat(
            cli=settings.provider, model=model or settings.llm_model,
            effort=clichat.checked_effort(settings.provider,
                                          settings.cli_effort),
            timeout=CLI_TIMEOUT)
    base_url, api_key, timeout = _endpoint(settings)
    return init_chat_model(model or settings.llm_model, model_provider='openai',
                           base_url=base_url, api_key=api_key, timeout=timeout)


def lab_llm(settings):
    """The lab's chat model, or the offline fake when nothing real is
    configured — the lab must remain runnable with no network at all."""
    return make_chat_model(settings)


def judge_llm(settings, model: str = ''):
    """The client RAGAS judges with.

    Its own client rather than the one every other stage shares, because the
    judge is the only stage whose model is *bound* at construction: RAGAS calls
    it through its own wrapper and never forwards a per-request model, so
    passing the judge slug here is the only way it reaches the wire. Building it
    from the same seam is what makes a local judge possible at all."""
    return make_chat_model(settings, model)


def lab_chat(llm, messages: list[dict], model: str = '') -> BaseMessage:
    """Every LLM-backed lab step calls the model through here.

    An empty `model` means "whatever the client was built with", which is the
    lab's convention for per-role model settings (see models.ROLES); a non-empty
    one is forwarded per request, so one client still serves every role without
    being rebuilt. Passing model='' through to invoke() would put a null model in
    the request instead, which is why this is a branch and not a default
    argument.
    """
    return llm.invoke(messages, model=model) if model else llm.invoke(messages)


def _text(message) -> str:
    """The text of a message, whether it arrived as an object or a plain dict.

    The lab hands `lab_chat` OpenAI-shaped dicts, and LangChain hands this class
    message objects, so the fake has to read both.
    """
    if isinstance(message, dict):
        return message.get('content', '') or ''
    content = getattr(message, 'content', '')
    if isinstance(content, str):
        return content
    # Some providers return content blocks; only the text parts matter here.
    return ''.join(part.get('text', '') for part in content
                   if isinstance(part, dict))


class FakeChat(BaseChatModel):
    """Deterministic offline chat model for the suite.

    Echoes the last user message back as 'FAKE: <text>'. `conftest.py` pins
    `RAGLAB_LLM=fake` for the whole suite, so this is what every LLM-backed stage
    runs on when nobody asked for a real model.

    It is here rather than in the test tree because 'fake' is a *backend*,
    selected through `RAGLAB_LLM` like any other. LangChain's own fakes report no
    usage, which would leave the token-and-cost path untestable offline.
    """

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
        # Four characters to the token: an estimate, and it only has to be
        # non-zero and additive for the reporting to be exercised.
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
