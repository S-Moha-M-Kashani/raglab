"""The panel's LLM widget: one self-contained module, deliberately outside
`llm.py`'s measured seam.

It answers questions about this project from a small knowledge base and a
calculator — it retrieves nothing, judges nothing, and writes no run, no
ledger row and no number. That is why it may do two things the measured path
must not: talk to OpenRouter through its own `ChatOpenAI`, and trace to
LangSmith (the env variables below). The architecture is a first cut on
purpose; promoting or removing it later touches this module and one route.

The agent is `langchain.agents.create_agent` with six middleware hooks, taken
2026-08-18 when this project moved to langchain 1.x. Before that the pin said
`langchain<1` and the agent was langgraph's `create_react_agent` — the same
prebuilt loop under its pre-1.0 name, wired through `pre_model_hook`,
`post_model_hook` and a callable model because `AgentMiddleware` was a major
version away.
"""
import ast
import os
import re
import sys
import time

if __name__ == '__main__' and __package__ in (None, ''):
    # A debugger's run-this-file button executes `python src/raglab/widget.py`,
    # where a relative import has no package to be relative to — so stepping
    # through this module died on line one. Naming the package here, before the
    # imports below, is what makes `python -m raglab.widget` and the green
    # arrow the same run.
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = 'raglab'

from langchain.agents.middleware import (after_agent, after_model,
                                         before_agent, before_model,
                                         wrap_model_call, wrap_tool_call)
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from .clichat import CliChat, checked_effort, cli_available
from .settings import PROVIDER_MODELS, load_env_file

# Read at build time, never at import: the suite runs offline, and a missing
# variable must become a stated refusal rather than a KeyError at import.
# These are the *OpenRouter path's* requirement — the whole point of a CLI
# backend is that it needs no key at all.
REQUIRED_ENV = ('OPENROUTER_API_KEY', 'LANGSMITH_API_KEY',
                'LANGSMITH_ENDPOINT', 'LANGSMITH_PROJECT', 'LANGSMITH_TRACING')

# The widget's own catalogue: value -> (kind, label). The two OpenRouter
# models run the tool loop; the two CLIs cannot (`CliChat` has no
# `bind_tools`), answer in one call with the knowledge base inlined, and
# their labels say so — an option states what it can do.
WIDGET_MODELS = {
    'openai/gpt-5-nano': ('openrouter', 'gpt-5-nano · OpenRouter, tools'),
    'openai/gpt-5-mini': ('openrouter', 'gpt-5-mini · OpenRouter, tools'),
    'claude': ('cli', 'claude · CLI, no key, no tools'),
    'codex': ('cli', 'codex · CLI, no key, no tools'),
}
# The codex CLI: gpt-5.6-luna, the lightest draw on the membership, no key
# involved. Among the OpenRouter pair the cheaper nano leads the list.
DEFAULT_MODEL = 'codex'

def _openrouter_url() -> str:
    """`.env` already carries OPENROUTER_BASE_URL for the lab's own backend;
    the widget reads the same variable rather than keeping a second copy."""
    return (os.environ.get('OPENROUTER_BASE_URL', '').strip()
            or 'https://openrouter.ai/api/v1')

KNOWLEDGE_BASE = {
    'purpose': 'The RAG lab is a generic retrieval workbench: chunking and '
               'retrieval choices for any use case are decided by measurement '
               'against a ground-truth corpus, which is itself a config '
               'field. The bundled default is fixtures/diary_year_fa.json, '
               '167 sessions of synthetic Farsi diary chat — one case study '
               'among five shipped corpora, with imports supported.',
    'ports': 'The lab serves its panel on port 9002 '
             '(uv run --extra local-embeddings raglab); the read-only '
             'Inspector runs on port 9003 (uv run raglab-inspector).',
    'architecture': 'The chosen retrieval architecture is candidate F, the '
                    'gated pipeline: semantic-drift chunks, '
                    'heydariAI/persian-embeddings, hybrid-RRF at k=8, lexical '
                    'rerank, time filter, plus an LLM relevance gate '
                    '(grader=llm, grade_threshold=0.4) between retrieval and '
                    'generation.',
    'metrics': 'Exactly four judged metrics choose the architecture: '
               'faithfulness, answer relevancy, LLM context precision and '
               'context recall. Their unweighted mean is the decision score, '
               'and it never appears without its standard error.',
    'storage': 'The index lives in process memory (MemoryVectors, brute-force '
               'cosine) and is discarded when the process ends. The durable '
               'artifacts are one JSON file per run in .runs/ and one ledger '
               'row per job in databases/raglab.db.',
    'commands': 'Entry points: raglab (panel), raglab-inspector, raglab-lab, '
                'raglab-sweep, raglab-judgescreen, raglab-leaderboard. The '
                'sweep changes exactly one knob per candidate and refuses to '
                'run on the fake backend.',
    'embedder': 'The default embedder is heydariAI/persian-embeddings over '
                'sentence-transformers (the local-embeddings extra). The '
                'ascii-hash embedder is kept as a reference point: it embeds '
                'Farsi to the zero vector and scores ~0.01 recall against '
                '0.617 for the real encoder.',
}


# --- the six hooks, as middleware ----------------------------------------
#
# One decorator each, from `langchain.agents.middleware`, and the decorated
# object *is* the middleware — the framework's own seam rather than this
# module's imitation of one. They are declared here at import (measured: 0.07 s
# on top of what this module already pays for langchain_core) and handed to
# `create_agent` in `_build_agent`; the agent itself is still built on the
# first request, which is the laziness that matters.
#
# One line of real work each: the point is that they are visible, and that each
# has somewhere obvious to put a breakpoint.

HOOKS_VERBOSE = False        # main() turns this on; the route leaves it off
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
    nothing, which is the one thing this module must not do."""
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


@tool
def search_knowledge_base(query: str) -> str:
    """Look up facts about the RAG lab project.

    Use this for any question about what the lab is, its ports, commands,
    architecture, metrics, storage or embedder.

    Args:
        query: A few keywords, e.g. "ports" or "decision metrics".
    """
    words = {w for w in re.findall(r'[a-z0-9]+', query.lower()) if len(w) > 2}
    hits = [f'{key}: {text}' for key, text in KNOWLEDGE_BASE.items()
            if any(w in key.lower() or w in text.lower() for w in words)]
    if not hits:
        return f"No entry matches '{query}'. Topics: {', '.join(KNOWLEDGE_BASE)}."
    return '\n\n'.join(hits)


_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                  ast.Mod, ast.Pow, ast.USub, ast.UAdd)


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "68000000 / 551695".

    Args:
        expression: Numbers and the operators + - * / // % ** only.
    """
    # An AST whitelist, never `eval`: a tool handed to a model must not be a
    # Python prompt.
    try:
        parsed = ast.parse(expression, mode='eval')
    except SyntaxError as error:
        raise ValueError(f'not an arithmetic expression: {error}') from error
    for node in ast.walk(parsed):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f'not arithmetic: {type(node).__name__} is not allowed')
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError('only numbers are allowed')
    return str(eval(compile(parsed, '<widget>', 'eval'), {'__builtins__': {}}))


TOOLS = [search_knowledge_base, calculate]

SYSTEM_PROMPT = (
    'You are the RAG lab panel\'s helper. Answer questions about this '
    'project using the search_knowledge_base tool, and use calculate for '
    'any arithmetic. Answer briefly; say so when the knowledge base has no '
    'answer rather than inventing one.')


class WidgetUnavailable(RuntimeError):
    """The lab is up; its widget is not. The route answers this as a 502."""


# One cached agent per OpenRouter model — a CLI is a process per call and
# caches nothing.
_AGENTS: dict = {}


def reset() -> None:
    """Drop the cached agents so the next ask() rebuilds (tests, key changes)."""
    _AGENTS.clear()


def _build_agent(model: str):
    # `load_env` strips values, so a bare `KEY= ` line in .env lands here as
    # '' — an empty variable is a missing one, not a present one.
    for name in REQUIRED_ENV:
        if not os.environ.get(name, '').strip():
            raise WidgetUnavailable(
                f'{name} is not set — the widget needs it in .env')
    # Present, and read the way the spec states them. LangSmith picks its
    # four up from the environment by itself; only the key is passed on.
    openrouter_api_key = os.environ['OPENROUTER_API_KEY']
    os.environ['LANGSMITH_API_KEY']
    os.environ['LANGSMITH_ENDPOINT']
    os.environ['LANGSMITH_PROJECT']
    os.environ['LANGSMITH_TRACING']

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, api_key=openrouter_api_key,
                     base_url=_openrouter_url())
    # A static model, so `create_agent` binds the tools itself. It was a
    # *callable* under langgraph's prebuilt loop, which binds tools only to the
    # static kind — and the agent then answered from its own knowledge, called
    # neither tool, and said nothing about it. Interception lives in
    # `trim_and_call` now, which is where 1.x puts it.
    return create_agent(llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT,
                        middleware=MIDDLEWARE)


def _cli_answer(cli: str, message: str) -> str:
    """One CLI call, the knowledge base inlined: no tool loop exists here, so
    a prompt that does not carry the facts is a CLI answering about a project
    it has never seen."""
    if not cli_available(cli):
        raise WidgetUnavailable(
            f'the {cli} command is not on this machine — install and log in, '
            'or pick an OpenRouter model')
    facts = '\n'.join(f'- {key}: {text}' for key, text in KNOWLEDGE_BASE.items())
    system = ('You are the RAG lab panel\'s helper. Answer briefly from the '
              'knowledge base below; say so when it has no answer rather '
              'than inventing one.\n\nThe knowledge base, in full:\n' + facts)
    effort = checked_effort(cli, os.environ.get('RAGLAB_CLI_EFFORT', '').strip()
                            or 'low')
    chat = CliChat(cli=cli, model=PROVIDER_MODELS[cli], effort=effort)
    try:
        return str(chat.invoke([('system', system), ('user', message)]).content)
    except Exception as error:
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error


def ask(message: str, model: str = '') -> str:
    """One question in, one answer out. `model` picks from WIDGET_MODELS;
    empty means the default. Agents build on first use, one per model."""
    choice = model or DEFAULT_MODEL
    kind, _ = WIDGET_MODELS.get(choice) or (None, None)
    if kind is None:
        raise ValueError(f'{choice!r} is not a widget model; expected one of '
                         + ', '.join(repr(v) for v in WIDGET_MODELS))
    if kind == 'cli':
        # The two agent-level hooks bracket a CLI too, through the halves they
        # were factored into: a CLI has no loop for the middle four, and no
        # graph to hang middleware on at all.
        return _account(_cli_answer(choice, _validate(message)))
    if choice not in _AGENTS:
        _AGENTS[choice] = _build_agent(choice)
    try:
        # A real HumanMessage rather than a dict, so it carries an id that
        # `check_request` can write a capped question back over.
        result = _AGENTS[choice].invoke(
            {'messages': [HumanMessage(content=message)]},
            config={'recursion_limit': 12})
    except WidgetUnavailable:
        raise
    except Exception as error:
        # A UI helper's failure is a stated 502, never a bare 500 — but the
        # reason travels with it.
        raise WidgetUnavailable(f'the widget could not answer: {error}') from error
    reply = result['messages'][-1].content
    if isinstance(reply, list):
        reply = ' '.join(part.get('text', '') if isinstance(part, dict) else str(part)
                         for part in reply)
    # `close_the_log` already accounted for this run from inside the graph.
    return str(reply)


# --- the end-to-end check -------------------------------------------------
#
#     uv run python -m raglab.widget [model] [question ...]
#
# or the file itself, under a debugger. Real calls: a live model over the
# network, or a real CLI process. Not a console entry point and not a test —
# the suite is offline, and this is the thing that is not.
#
# Nothing here catches or exits: a refusal should stop the debugger where it
# was raised, and the two questions are the two tools.

QUESTIONS = ('Which ports do the lab and the Inspector serve on?',
             'What is 174 - 167?')


def main(model: str = DEFAULT_MODEL, questions: tuple = QUESTIONS) -> None:
    global HOOKS_VERBOSE
    HOOKS_VERBOSE = True                 # hooks print as they fire
    load_env_file()                      # the route's server did this already
    kind, label = WIDGET_MODELS[model]
    print(f'{label}  [{kind}]  tools: {[t.name for t in TOOLS]}\n')
    for question in questions:
        HOOK_LOG.clear()
        started = time.perf_counter()
        answer = ask(question, model)
        print(f'\n  {question}\n  → {answer.strip()}'
              f'\n  {time.perf_counter() - started:.1f}s, '
              f'{len(HOOK_LOG)} hooks\n')


if __name__ == '__main__':
    # argv, not argparse: two optional positionals, and a debugger that runs
    # this file with none of them gets the defaults.
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL,
         (' '.join(sys.argv[2:]),) if len(sys.argv) > 2 else QUESTIONS)

