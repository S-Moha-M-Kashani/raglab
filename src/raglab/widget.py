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
import json
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

from . import skills
from .clichat import CliChat, checked_effort, cli_available
from .settings import PROVIDER_MODELS, ROOT, load_env_file

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
               'field. The bundled default is fixtures/corpus_groundtruth_datasets/diary_year_fa.json, '
               '167 sessions of synthetic Farsi diary chat — one case study '
               'among five shipped corpora, with imports supported.',
    'ports': 'The lab serves its panel on port 9002 '
             '(uv run --extra local-embeddings raglab); the read-only '
             'Inspector runs on port 9003 (uv run raglab-inspector).',
    'architecture': 'The lab prescribes no single architecture: every stage '
                    'is a config knob, and the right pipeline depends on the '
                    'use case. For the bundled diary case study, the measured '
                    'choice was candidate F, the gated pipeline: '
                    'semantic-drift chunks, heydariAI/persian-embeddings, '
                    'hybrid-RRF at k=8, lexical rerank, time filter, plus an '
                    'LLM relevance gate (grader=llm, grade_threshold=0.4) '
                    'between retrieval and generation — a finding about that '
                    'corpus, not a default for others.',
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
                'sentence-transformers (the local-embeddings extra), because '
                'the bundled default corpus is Farsi — a different corpus '
                'wants an encoder verified on its own language. The '
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


# The model-facing text is data, not code (the skills-folder and
# bilingual-pairs rule): every prompt lives under fixtures/prompts/, so
# editing what the model reads is editing a fixture. The docstrings on the
# tools below are for readers of this file; what the model sees is assigned
# from widget_tools.yaml right after TOOLS.
PROMPTS_DIR = ROOT / 'fixtures' / 'prompts'


def _prompts(name: str) -> dict:
    """One YAML page from the prompts folder, parsed at import — the pages
    are small, and a missing key must fail loudly here rather than serve a
    tool with no description."""
    import yaml
    return yaml.safe_load(
        (PROMPTS_DIR / f'{name}.yaml').read_text(encoding='utf-8'))


_PROMPTS = _prompts('widget')
_TOOL_PROMPTS = _prompts('widget_tools')


@tool
def search_knowledge_base(query: str) -> str:
    """Facts about this project, matched by keyword; the model-facing
    prompt is fixtures/prompts/widget_tools.yaml's entry."""
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
    """Arithmetic over an AST whitelist, never `eval`; the model-facing
    prompt is fixtures/prompts/widget_tools.yaml's entry."""
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


# How many bodies one read_rag_skill call returns: the bodies are the
# expensive layer, and a call asking for the whole corpus would put ~100 KB
# into the loop.
MAX_SKILL_READS = 3


@tool
def search_rag_skills(query: str) -> str:
    """The skills catalogue's cheap layer, matched literally, whole index
    on a miss; the model-facing prompt — including the search-again-on-thin-
    hits guidance — is fixtures/prompts/widget_tools.yaml's entry."""
    hits = skills.search(query)
    if not hits:
        return (f"No skill matches '{query}'.\n\n" + skills.index_text())
    return '\n'.join(f'{name}: {description}' for name, description in hits)


@tool
def read_rag_skill(names: str) -> str:
    """Full skill bodies, several per call but capped; the model-facing
    prompt is fixtures/prompts/widget_tools.yaml's entry."""
    catalogue = skills.index()
    asked = [n for n in re.split(r'[,\s]+', names.strip()) if n]
    known = [n for n in asked if n in catalogue]
    unknown = [n for n in asked if n not in catalogue]
    served, over_cap = known[:MAX_SKILL_READS], known[MAX_SKILL_READS:]
    parts = [f'=== {name} ===\n{skills.body(name)}' for name in served]
    if over_cap:
        parts.append(f'At most {MAX_SKILL_READS} skills per call — not served: '
                     + ', '.join(over_cap) + '. Ask again for them.')
    if unknown:
        parts.append('Not skills: ' + ', '.join(unknown)
                      + '. The skills are: ' + ', '.join(sorted(catalogue)) + '.')
    if not parts:
        return 'No names given. The skills are: ' + ', '.join(sorted(catalogue)) + '.'
    return '\n\n'.join(parts)


# The bilingual probe's default instrument: twelve diary-like sentence pairs,
# kept as a fixture rather than code — the skills-folder rule, so the pairs
# can change without touching Python — and fixed between calls so two
# measurements stay comparable.
PAIRS_FILE = ROOT / 'fixtures' / 'bilingual_probe_pairs.json'

# The shape a pairs payload must have, quoted by the tool's refusal and
# stated to the model by its YAML prompt — the config.HELP['run.dataset-file']
# pattern: the contract is announced where the data goes in.
PAIRS_SHAPE = ('a JSON list of at least two [english, farsi] pairs, e.g. '
               '[["I slept badly.", "بد خوابیدم."], '
               '["It rained.", "باران آمد."]]')


def _read_pairs(raw: str = ''):
    """The probe's pairs: the caller's JSON, or the bundled fixture. Returns
    (pairs, problem); a problem is a stated string the model can relay and
    correct itself against, never an exception."""
    source = raw.strip()
    try:
        data = json.loads(source) if source else json.loads(
            PAIRS_FILE.read_text(encoding='utf-8'))
    except Exception as error:
        return None, f'unreadable pairs ({error}); expected {PAIRS_SHAPE}'
    well_formed = (isinstance(data, list) and len(data) >= 2 and all(
        isinstance(pair, list) and len(pair) == 2
        and all(isinstance(text, str) and text.strip() for text in pair)
        for pair in data))
    if not well_formed:
        return None, f'malformed pairs; expected {PAIRS_SHAPE}'
    return [(en, fa) for en, fa in data], None

# name -> loaded encoder; the _AGENTS pattern — a 2 GB checkpoint must not be
# reloaded per question, and the cache dies with the process. Bounded,
# unlike _AGENTS, because the keys come from a model or a user naming any
# HuggingFace checkpoint: agents are a few kilobytes and four names, while
# a handful of encoder probes at gigabytes each would exhaust the lab
# process's memory.
_ENCODERS: dict = {}
MAX_ENCODERS = 2


def _load_encoder(name: str):
    """Lazy on both axes: the import needs the local-embeddings extra, and
    neither it nor the checkpoint may cost anything at module import."""
    from sentence_transformers import SentenceTransformer
    if name not in _ENCODERS:
        while len(_ENCODERS) >= MAX_ENCODERS:
            _ENCODERS.pop(next(iter(_ENCODERS)))
        _ENCODERS[name] = SentenceTransformer(name)
    return _ENCODERS[name]


@tool
def measure_bilingual_alignment(model_name: str = '', pairs: str = '') -> str:
    """The EN-Farsi alignment probe over a real encoder — pair cosine,
    mixed-pool retrieval, a verdict; the model-facing prompt, including the
    pairs contract, is fixtures/prompts/widget_tools.yaml's entry."""
    name = model_name.strip() or 'heydariAI/persian-embeddings'
    items, problem = _read_pairs(pairs)
    if problem:
        # Stated refusals the model can relay and correct against, not a
        # dead loop: the shape, the extra or the checkpoint — whichever is
        # missing is the whole answer.
        return f'cannot measure: {problem}'
    english = [en for en, _ in items]
    farsi = [fa for _, fa in items]
    try:
        import numpy as np
        encoder = _load_encoder(name)
        vectors = np.asarray(encoder.encode(english + farsi,
                                            normalize_embeddings=True))
    except Exception as error:
        return f'cannot measure {name}: {error}'
    n = len(items)
    sims = vectors[:n] @ vectors[n:].T
    pairs = np.diag(sims)
    mismatched = sims[~np.eye(n, dtype=bool)]
    pool = vectors @ vectors.T
    np.fill_diagonal(pool, -1.0)          # a query may not retrieve itself
    en_wins = int((pool[:n].argmax(axis=1) == np.arange(n) + n).sum())
    fa_wins = int((pool[n:].argmax(axis=1) == np.arange(n)).sum())
    separation = float(pairs.mean() - mismatched.mean())
    aligned = en_wins == n and fa_wins == n and separation >= 0.3
    verdict = 'aligned' if aligned else 'weak or no alignment'
    return (f'{name}, measured now on {n} English-Farsi sentence pairs: '
            f'translation pairs mean cosine {pairs.mean():.3f} '
            f'(min {pairs.min():.3f}), mismatched pairs mean '
            f'{mismatched.mean():.3f} (max {mismatched.max():.3f}); in a '
            f'mixed-language pool the English query finds its own Farsi '
            f'translation {en_wins}/{n} times and the Farsi query its '
            f'English one {fa_wins}/{n}. Verdict: {verdict}. This is a '
            f'sentence-scale probe on {n} short sentence pairs — '
            f'corpus-scale retrieval can still differ.')


TOOLS = [search_knowledge_base, calculate, search_rag_skills, read_rag_skill,
         measure_bilingual_alignment]

# The YAML page is what the model reads; assigning it here makes the fixture
# the single source, and the import fails on a tool the page does not name.
for _each in TOOLS:
    _each.description = _TOOL_PROMPTS[_each.name].strip()

SYSTEM_PROMPT = _PROMPTS['system'].strip()


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


def _cli_system() -> str:
    """The tool-less prompt: project facts in full, the skills corpus as its
    index only. The full bodies cannot be inlined, so the prompt says what a
    CLI can do — name the right skill — and what it cannot: read one. The
    template is fixtures/prompts/widget.yaml's `cli_system`."""
    facts = '\n'.join(f'- {key}: {text}' for key, text in KNOWLEDGE_BASE.items())
    return _PROMPTS['cli_system'].format(facts=facts,
                                         skills_index=skills.index_text())


def _cli_answer(cli: str, message: str) -> str:
    """One CLI call, the knowledge base inlined: no tool loop exists here, so
    a prompt that does not carry the facts is a CLI answering about a project
    it has never seen."""
    if not cli_available(cli):
        raise WidgetUnavailable(
            f'the {cli} command is not on this machine — install and log in, '
            'or pick an OpenRouter model')
    system = _cli_system()
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
        # Measured 2026-08-18: with the six middleware nodes a tool hop costs
        # ~4 supersteps, so 12 allowed exactly one hop — a run that searched,
        # then searched and read, then answered (13 steps) died *after* its
        # final answer, one node short of close_the_log. 24 gives the loop
        # about five hops, still a hard ceiling rather than a budget.
        result = _AGENTS[choice].invoke(
            {'messages': [HumanMessage(content=message)]},
            config={'recursion_limit': 24})
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

