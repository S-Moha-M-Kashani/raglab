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

HOOKS_VERBOSE = False        # __main__ turns this on; the route leaves it off
HOOK_LOG: list[str] = []     # what fired, in order — the whole run at a glance

MAX_QUESTION = 500           # the longest request it will accept
MAX_HISTORY = 20             # how much history one model call sees


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
