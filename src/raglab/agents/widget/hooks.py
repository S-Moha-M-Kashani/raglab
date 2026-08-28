"""The four hooks, as middleware.

One decorator each, from `langchain.agents.middleware`, and the decorated
object *is* the middleware — the framework's own seam rather than this
package's imitation of one. They are declared here at import (measured: 0.07 s
on top of what this package already pays for langchain_core) and handed to
`create_agent` in `backends._build_agent`; the agent itself is still built on
the first request, which is the laziness that matters.

There were six, until 2026-08-28: `note_prompt` (`before_model`) and
`check_reply` (`after_model`) were each their own graph node, so one tool hop
cost four supersteps and `backends._run`'s recursion budget died a few hops
before `MAX_TOOL_HOPS` could ever fire. Both jobs now live inside
`trim_and_call`, a `wrap_model_call` wrapper rather than a node, which is what
halves the supersteps a hop costs. The `_fired` labels they wrote
(`before_model`, `after_model`) are kept — see the comment on `trim_and_call`
— so the account of a run still names the same moments.

Each is still one obvious place to put a breakpoint — that is what they are
for. It is no longer one line of real work each: the fold gave `trim_and_call`
five jobs, and the paragraph above says why they belong together rather than
in nodes of their own.
"""
import collections

from langchain.agents.middleware import (after_agent, before_agent,
                                         wrap_model_call, wrap_tool_call)
from langchain_core.messages import AIMessage

from raglab.agents.widget.conversation_memory import (
    IDENTITY_LINE,
    MAX_RELEVANCE_TEXT,
    MEMORY_LINE,
    MemoryPolicy,
    STANDING_LINE,
    relevance_guard,
    standing_mark,
    superseded_standing_lines,
)
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from raglab.agents.widget import long_term_memory
from raglab.agents.widget.prompts import MEMORY_POLICY_PROMPT

HOOKS_VERBOSE = False        # __main__ turns this on; the route leaves it off

MAX_QUESTION = MAX_RELEVANCE_TEXT  # the longest request it will accept
MAX_HISTORY = 20             # how much history one model call sees
MAX_TOOL_HOPS = 8             # hard stop before a pathological tool loop

# One model node plus one tools node. It was four until 2026-08-28: two more
# supersteps went to `before_model`/`after_model`, the graph nodes that used
# to carry one logging line each and now write that line from inside
# `trim_and_call` instead.
SUPERSTEPS_PER_HOP = 2

# The recursion budget `backends._run` hands the graph, derived so the guard
# and the ceiling can never drift apart again. A run is one graph-start
# superstep the compiled graph spends before `before_agent` ever runs
# (verified against the real compiled graph — `get_graph()` shows
# `__start__` feeding `check_request.before_agent` as its own transition, and
# driving a scripted model through 0..MAX_TOOL_HOPS real hops always needed
# exactly one more than the count of nodes actually executed, never zero),
# `before_agent` itself, up to `MAX_TOOL_HOPS` hops at `SUPERSTEPS_PER_HOP`
# supersteps each, the model node that writes the final answer, and
# `after_agent`:
#
#   1 + 1 + MAX_TOOL_HOPS * SUPERSTEPS_PER_HOP + 1 + 1
#   = 1 + 1 + 8 * 2 + 1 + 1 = 20
#
# That admits MAX_TOOL_HOPS sequential hops *and* the answer after them —
# proved against the real compiled graph in
# `test_widget_regressions.test_recursion_limit_admits_max_tool_hops_and_the_closing_answer`,
# not just recomputed here — so `stop_repeated_tool_hops` is what stops a
# pathological loop, and this ceiling is never what fires first.
RECURSION_LIMIT = 1 + 1 + MAX_TOOL_HOPS * SUPERSTEPS_PER_HOP + 1 + 1

# What fired, in order — the whole run at a glance. Bounded because it is a
# module-level list shared by every concurrent turn for the life of the
# process, and an unbounded log growing under real traffic would be its own
# quiet leak. One run logs at most a handful of lines per superstep — one or
# two from `trim_and_call`, up to a couple more per tool call inside a single
# tools superstep — so `RECURSION_LIMIT` supersteps times a generous 8 lines
# each covers any ordinary run with room to spare; a run this bounded cannot
# fill it.
HOOK_LOG: collections.deque = collections.deque(maxlen=RECURSION_LIMIT * 8)


#: What the hop guard says instead of an answer. A constant rather than a
#: literal because two other places have to recognise it as a refusal rather
#: than read it as prose — see `HOP_GUARD_MARK`.
HOP_GUARD_REFUSAL = ('I could not complete this lookup safely because the tool '
                     'calls started repeating. Please ask for the run by its '
                     'experiment ID.')

#: The flag `trim_and_call` puts on `response_metadata` when it answers with
#: the refusal above instead of calling the model.
#:
#: Until 2026-08-28 the guard only appended a message and let the run die on
#: the recursion ceiling, so nothing downstream ever saw a finished turn made
#: of it. Now it produces a real, well-formed reply — and a reply is what
#: `backends` files as memory and logs as `answered`. A refusal is neither: it
#: says nothing about the lab, so remembering it would be filing the widget's
#: own apology as a fact about an experiment. Matching on the text would work
#: today and break the first time the copy is reworded; a flag on the message
#: that wrote it cannot drift from it.
HOP_GUARD_MARK = 'widget_hop_guard'


def stop_repeated_tool_hops(state) -> str | None:
    """Return a final fallback once a run has called tools too many times."""
    calls = sum(len(getattr(message, 'tool_calls', None) or [])
                for message in state.get('messages', []))
    if calls >= MAX_TOOL_HOPS:
        return HOP_GUARD_REFUSAL
    return None


def hop_guard_refused(messages) -> bool:
    """Whether this run ended in the hop guard's refusal rather than an answer.

    Read off the last message's own metadata, which is where `trim_and_call`
    stamped it — the run's own account of what it produced, not a guess made
    afterwards from the words."""
    last = (messages or [None])[-1]
    metadata = getattr(last, 'response_metadata', None) or {}
    return bool(metadata.get(HOP_GUARD_MARK))


class MemoryUpdate(BaseModel):
    """The bounded, structured output accepted by the memory writer."""

    model_config = ConfigDict(extra='forbid', strict=True)
    dataset_summary: StrictStr = Field(default='', max_length=long_term_memory.MAX_SUMMARY_CHARS)
    global_summary: StrictStr = Field(default='', max_length=long_term_memory.MAX_SUMMARY_CHARS)


# The prefix on a reason that means nobody judged the turn: no policy model
# was available, or its answer could not be read. "The judge said no" and "the
# judge could not be asked" are both `relevant=False` here — they have to be,
# because saving fails closed either way — but they are not the same thing to
# tell a reader, so the reason says which and `policy_unreached` is how a
# caller asks without reading English.
POLICY_UNREACHED = 'The memory policy could not be reached'


def policy_unreached(policy: MemoryPolicy) -> bool:
    """Whether this is a refusal or the absence of one."""
    return policy.reason.startswith(POLICY_UNREACHED)


def evaluate_memory_policy(text: str, model, *, experiment_id: str = '',
                           dataset_id: str = '',
                           trusted_dataset_id: str = '',
                           conversation: str = '') -> MemoryPolicy:
    """Ask a model for the structured memory decision, failing closed.

    ``model`` is injected by the caller so this seam can be tested offline and
    so policy availability is never confused with answer-model availability.

    It is asked *after* the answer exists (`backends._finish_memory`), so every
    refusal here says nothing was saved rather than nothing was processed: the
    reader has their answer either way, and what this decides is whether the
    turn is filed.
    """
    refusal = relevance_guard(text)
    if refusal:
        return MemoryPolicy(relevant=False, should_save=False, reason=refusal)
    if model is None:
        return MemoryPolicy(
            reason=f'{POLICY_UNREACHED}: no policy model is available; '
                   'nothing was saved.')
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
                'This request is not relevant to the RAG lab, so it was not saved.'})
        if trusted_dataset_id and policy.dataset_id != trusted_dataset_id:
            return MemoryPolicy(
                relevant=False, should_save=False,
                dataset_id=trusted_dataset_id,
                reason=(f'The policy named dataset {policy.dataset_id!r}, but '
                        f'the active experiment uses {trusted_dataset_id!r}; '
                        'nothing was saved.'))
        return policy
    except Exception as error:
        return MemoryPolicy(
            reason=f'{POLICY_UNREACHED}: unavailable or malformed ({error}); '
                   'nothing was saved.')


#: What the summarizer is asked for. Named rather than left inline where it
#: was, because two other places now state the same rule and a reader has to
#: be able to find all three: this instruction, the store's check
#: (`long_term_memory.names_one_corpus`) and the read-time filter.
#: It stays in code, unlike every prompt in `fixtures/prompts/`, for the
#: reason it was already in code: the rule it states is enforced by the store,
#: so wording and check are one change, and a fixture a reader could edit into
#: disagreement with the check would be a promise the store then breaks.
#:
#: The two summaries are not the same kind of sentence, so the instruction
#: says so. `dataset_summary` is filed under one corpus and read only by that
#: corpus's threads, so it may name that corpus — and only that one, since a
#: note about somebody else's corpus filed under this one is the same lie one
#: thread wide. `global_summary` is the single row every dataset's thread is
#: handed, so it may hold only a pattern that holds across corpora — which is
#: why it may name no dataset id, no experiment id and no single run's
#: numbers. Told this, a summarizer still wrote
#: "Last experiment details for smoke-import-check: 6 questions analyzed …"
#: into it, and a `nosrat-fa` thread was handed it as fact. So the instruction
#: is the request and `long_term_memory.names_one_corpus` is the check;
#: neither end is enough alone, because the write gate cannot repair a row
#: written before it existed and the read filter cannot stop the store filling
#: up with notes no thread will ever be shown.
SUMMARIZE_MEMORY_PROMPT = (
    'Summarize this accepted RAG-lab answer for bounded long-term memory. '
    'Return only dataset_summary and optional global_summary. Do not invent '
    'measurements.\n'
    'dataset_summary is about the dataset named below: it may name that '
    'dataset, and it must name no other dataset and no experiment id.\n'
    'global_summary is different: every dataset\'s thread reads it, so write '
    'one only for a pattern that holds across corpora. It must name no '
    'dataset id, no experiment id, and no single run\'s numbers. If this '
    'answer only says something about this one dataset or this one run, leave '
    'global_summary empty — that is the ordinary case, not a failure.')


def summarize_memory_update(question: str, answer: str, *, dataset_id: str,
                            experiment_id: str = '', subtopic: str = '',
                            model=None) -> MemoryUpdate:
    """Summarize an accepted turn after its authoritative answer exists."""
    if model is None:
        raise RuntimeError('memory summarizer is unavailable')
    structured = model.with_structured_output(MemoryUpdate)
    result = structured.invoke([
        ('system', SUMMARIZE_MEMORY_PROMPT),
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
    """Likewise `close_the_log`'s half.

    The second number says how much is *in the shared log*, and no longer
    claims to be this run's hook count. It never could be: `HOOK_LOG` is one
    module-level deque that every concurrent turn writes into, so under two
    turns at once the number already included the other turn's lines, and
    since the deque was bounded it saturates at its cap and stops moving at
    all. Both are fine for what the log is — `__main__` clears it and reads one
    run at a time — but a line has to say the thing it counts.
    """
    _fired('after_agent',
           f'{len(reply)} chars, {len(HOOK_LOG)} lines in the shared log')
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


@wrap_model_call
def trim_and_call(request, handler):
    """Around each LLM call: the tool-hop guard, the "about to call" line, the
    trim, the call itself, and the "what came back" line — five jobs where
    there used to be three hooks. `note_prompt` (`before_model`) and
    `check_reply` (`after_model`) were folded in here on 2026-08-28: each was
    its own graph node for one logging line, which made a tool hop cost four
    supersteps instead of the two a model node and a tools node actually need.
    A wrapper is not a node, so folding them in is what makes
    `RECURSION_LIMIT` an honest budget rather than a ceiling the guard never
    gets to test.

    The `_fired` labels below still read `before_model` and `after_model`
    even though neither is its own graph node anymore. The label names *when
    in a model call* the line was written — before the request goes out,
    after the response comes back — and that stays true regardless of which
    node runs it; a later reader should not infer two graph nodes that no
    longer exist.

    `request.override` is 1.x's non-destructive trim — langgraph's
    `llm_input_messages` is gone, and writing `messages` from a `before_model`
    node would delete the transcript rather than shorten a prompt.
    """
    stop = stop_repeated_tool_hops(request.state)
    if stop:
        _fired('before_model', 'tool-hop guard stopped the run')
        # Stamped, not just worded: this message ends the run as if it were an
        # answer, and `backends` has to be able to tell that it is not one
        # before it files the turn as memory or logs it as answered.
        return AIMessage(content=stop,
                         response_metadata={HOP_GUARD_MARK: True})
    _fired('before_model', f'{len(request.state["messages"])} messages in state')
    # System lines say what the whole thread is about — which experiment it is,
    # what long-term memory holds — so they must outlive the window or the
    # model forgets its subject on the twenty-first message. The window counts
    # everything else.
    #
    # Exempt is not unbounded, and that was the bug. The memory context grows
    # on every accepted turn and `backends._run` appends the new text, so a
    # real 29-step thread handed the model twelve system messages: several
    # versions of one memory, oldest first, the stale ones contradicting the
    # newest. A standing line a newer line of the same kind supersedes is now
    # left out of *this call*. The thread keeps it — `request.override` is
    # 1.x's non-destructive trim, and the reasons for keeping it are in
    # `superseded_standing_lines` — and the model reads one identity line and
    # one memory context.
    #
    # Only marked lines are filtered. A thread whose lines predate the marker
    # still sends all of them: nothing can now tell one of those from a system
    # line the widget did not write, and dropping a line on a guess is how a
    # future middleware's instruction disappears. Those threads stop growing —
    # every line written from now on is marked — which is the bound that
    # matters.
    system = [m for m in request.messages if getattr(m, 'type', '') == 'system']
    rest = [m for m in request.messages if getattr(m, 'type', '') != 'system']
    stale = superseded_standing_lines([standing_mark(m) for m in system])
    standing = [m for i, m in enumerate(system) if i not in stale]
    if stale or len(rest) > MAX_HISTORY:
        window = rest[-MAX_HISTORY:] if len(rest) > MAX_HISTORY else rest
        request = request.override(messages=standing + window)
    name = getattr(request.model, 'model_name', type(request.model).__name__)
    _fired('wrap_model_call', f'{name}, {len(request.messages)} messages')
    response = handler(request)
    # A tool-calling hop and a final answer are the two shapes, and an empty
    # one is neither — the `clichat` finding, stated where it can be seen
    # rather than swallowed.
    last = response.result[-1] if getattr(response, 'result', None) else None
    calls = getattr(last, 'tool_calls', None) or []
    text = str(getattr(last, 'content', ''))
    if calls:
        shape = f'{len(calls)} tool call(s): ' + ', '.join(c['name'] for c in calls)
    else:
        shape = f'{len(text)} chars of answer' if text.strip() else 'empty reply'
    _fired('after_model', shape)
    return response


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


@after_agent
def close_the_log(state, runtime):
    """After the agent completes: the analytics line. `HOOK_LOG` is the whole
    account of the run, and this is where it is closed."""
    _account(str(state['messages'][-1].content))
    return None


# Order is the order they nest in, and it is the order they are declared in.
MIDDLEWARE = [check_request, trim_and_call, log_tool_call, close_the_log]
