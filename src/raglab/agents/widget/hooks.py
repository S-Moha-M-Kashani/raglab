"""The six hooks, as middleware.

One decorator each, from `langchain.agents.middleware`, and the decorated
object *is* the middleware — the framework's own seam rather than this
package's imitation of one. They are declared here at import (measured: 0.07 s
on top of what this package already pays for langchain_core) and handed to
`create_agent` in `backends._build_agent`; the agent itself is still built on
the first request, which is the laziness that matters.

One line of real work each: the point is that they are visible, and that each
has somewhere obvious to put a breakpoint.
"""
from langchain.agents.middleware import (after_agent, after_model,
                                         before_agent, before_model,
                                         wrap_model_call, wrap_tool_call)
from langchain_core.messages import AIMessage

from raglab.agents.widget.conversation_memory import (
    MAX_RELEVANCE_TEXT,
    MemoryPolicy,
    relevance_guard,
)
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from raglab.agents.widget import long_term_memory
from raglab.agents.widget.prompts import MEMORY_POLICY_PROMPT

HOOKS_VERBOSE = False        # __main__ turns this on; the route leaves it off
HOOK_LOG: list[str] = []     # what fired, in order — the whole run at a glance

MAX_QUESTION = MAX_RELEVANCE_TEXT  # the longest request it will accept
MAX_HISTORY = 20             # how much history one model call sees
MAX_TOOL_HOPS = 8             # hard stop before a pathological tool loop


def stop_repeated_tool_hops(state) -> str | None:
    """Return a final fallback once a run has called tools too many times."""
    calls = sum(len(getattr(message, 'tool_calls', None) or [])
                for message in state.get('messages', []))
    if calls >= MAX_TOOL_HOPS:
        return ('I could not complete this lookup safely because the tool '
                'calls started repeating. Please ask for the run by its '
                'experiment ID.')
    return None


class MemoryUpdate(BaseModel):
    """The bounded, structured output accepted by the memory writer."""

    model_config = ConfigDict(extra='forbid', strict=True)
    dataset_summary: StrictStr = Field(default='', max_length=long_term_memory.MAX_SUMMARY_CHARS)
    global_summary: StrictStr = Field(default='', max_length=long_term_memory.MAX_SUMMARY_CHARS)


def evaluate_memory_policy(text: str, model, *, experiment_id: str = '',
                           dataset_id: str = '',
                           trusted_dataset_id: str = '',
                           conversation: str = '') -> MemoryPolicy:
    """Ask a model for the structured memory decision, failing closed.

    ``model`` is injected by the caller so this seam can be tested offline and
    so policy availability is never confused with answer-model availability.
    """
    refusal = relevance_guard(text)
    if refusal:
        return MemoryPolicy(relevant=False, should_save=False, reason=refusal)
    if model is None:
        return MemoryPolicy(reason='Memory policy is unavailable; nothing was saved.')
    try:
        structured = model.with_structured_output(MemoryPolicy)
        result = structured.invoke([
            ('system', MEMORY_POLICY_PROMPT),
            ('user', (f'Experiment: {experiment_id or "none"}\n'
                      f'Dataset context: {dataset_id or "unknown"}\n'
                      f'Prior conversation:\n{conversation or "(none)"}\n'
                      f'Question: {str(text).strip()}'))])
        policy = result if isinstance(result, MemoryPolicy) \
            else MemoryPolicy.model_validate(result)
        if not policy.relevant:
            return policy.model_copy(update={
                'should_save': False,
                'reason': policy.reason or
                'This request is not relevant to the RAG lab, so it was not processed.'})
        if trusted_dataset_id and policy.dataset_id != trusted_dataset_id:
            return MemoryPolicy(
                relevant=False, should_save=False,
                dataset_id=trusted_dataset_id,
                reason=(f'The policy named dataset {policy.dataset_id!r}, but '
                        f'the active experiment uses {trusted_dataset_id!r}; '
                        'nothing was processed.'))
        return policy
    except Exception as error:
        return MemoryPolicy(
            reason=f'Memory policy unavailable or malformed; nothing was saved '
                   f'({error}).')


def summarize_memory_update(question: str, answer: str, *, dataset_id: str,
                            experiment_id: str = '', subtopic: str = '',
                            model=None) -> MemoryUpdate:
    """Summarize an accepted turn after its authoritative answer exists."""
    if model is None:
        raise RuntimeError('memory summarizer is unavailable')
    structured = model.with_structured_output(MemoryUpdate)
    result = structured.invoke([
        ('system', 'Summarize this accepted RAG-lab answer for bounded long-term '
                    'memory. Return only dataset_summary and optional '
                    'global_summary. Do not invent measurements.'),
        ('user', (f'Question: {str(question).strip()}\n'
                  f'Answer: {str(answer).strip()}\n'
                  f'Dataset: {dataset_id or "unknown"}\n'
                  f'Experiment: {experiment_id or "none"}\n'
                  f'Subtopic: {subtopic or "general"}'))])
    return result if isinstance(result, MemoryUpdate) else MemoryUpdate.model_validate(result)


def refusal_for_message(message) -> str | None:
    """The deterministic refusal copy for a human message, if any."""
    return relevance_guard(getattr(message, 'content', ''))


def _fired(hook: str, detail: str) -> None:
    HOOK_LOG.append(f'{hook}: {detail}')
    if HOOKS_VERBOSE:
        print(f'      [{hook}] {detail}')


def _validate(text: str) -> str:
    """What `check_request` does, factored out because the CLI path has no
    loop to hang middleware on and must still be able to do it."""
    text = text.strip()
    if not text:
        raise ValueError('the widget was asked nothing')
    text = text[:MAX_QUESTION]
    _fired('before_agent', f'{len(text)} chars, {len(text.split())} words')
    return text


def _account(reply: str) -> str:
    """Likewise `close_the_log`'s half."""
    _fired('after_agent', f'{len(reply)} chars, {len(HOOK_LOG)} hooks fired')
    return reply


@before_agent
def check_request(state, runtime):
    """Before the agent starts: validate the request. It is the request that
    is checked and not the answer, because an over-long question is the one
    thing here that can cost real money. A capped question is written back as
    a *replacement* — same message id, so `add_messages` overwrites it rather
    than appending a second copy of the question."""
    last = state['messages'][-1]
    text = _validate(str(last.content))
    if text == str(last.content):
        return None
    return {'messages': [last.model_copy(update={'content': text})]}


@before_model
def note_prompt(state, runtime):
    """Before each LLM call: say what the loop is about to send. This is where
    context injection would go; the trim itself belongs one hook further in,
    where it can be applied to the request instead of to the transcript."""
    stop = stop_repeated_tool_hops(state)
    if stop:
        _fired('before_model', 'tool-hop guard stopped the run')
        return {'messages': [AIMessage(content=stop)]}
    _fired('before_model', f'{len(state["messages"])} messages in state')
    return None


@wrap_model_call
def trim_and_call(request, handler):
    """Around each LLM call: trim what this hop sees, name the model, and hand
    it on. `request.override` is 1.x's non-destructive trim — langgraph's
    `llm_input_messages` is gone, and writing `messages` from `before_model`
    would delete the transcript rather than shorten a prompt."""
    if len(request.messages) > MAX_HISTORY:
        request = request.override(messages=request.messages[-MAX_HISTORY:])
    name = getattr(request.model, 'model_name', type(request.model).__name__)
    _fired('wrap_model_call', f'{name}, {len(request.messages)} messages')
    return handler(request)


@wrap_tool_call
def log_tool_call(request, handler):
    """Around each tool call: log it, and let an error through after saying so.
    A widget tool that swallowed its own failure would answer confidently from
    nothing, which is the one thing this package must not do."""
    call = request.tool_call
    _fired('wrap_tool_call', f'{call["name"]}({str(call["args"])[:60]})')
    try:
        result = handler(request)
    except Exception as error:
        _fired('wrap_tool_call', f'{call["name"]} raised {error}')
        raise
    _fired('wrap_tool_call',
           f'{call["name"]} → {str(getattr(result, "content", result))[:60]}')
    return result


@after_model
def check_reply(state, runtime):
    """After each LLM response: look at what came back. A tool-calling hop and
    a final answer are the two shapes, and an empty one is neither — the
    `clichat` finding, stated where it can be seen rather than swallowed."""
    last = state['messages'][-1]
    calls = getattr(last, 'tool_calls', None) or []
    text = str(last.content)
    if calls:
        shape = f'{len(calls)} tool call(s): ' + ', '.join(c['name'] for c in calls)
    else:
        shape = f'{len(text)} chars of answer' if text.strip() else 'empty reply'
    _fired('after_model', shape)
    return None


@after_agent
def close_the_log(state, runtime):
    """After the agent completes: the analytics line. `HOOK_LOG` is the whole
    account of the run, and this is where it is closed."""
    _account(str(state['messages'][-1].content))
    return None


# Order is the order they nest in, and it is the order they are declared in.
MIDDLEWARE = [check_request, note_prompt, trim_and_call, log_tool_call,
              check_reply, close_the_log]
