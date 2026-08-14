"""The panel's LLM widget: one self-contained module, deliberately outside
`llm.py`'s measured seam.

It answers questions about this project from a small knowledge base and a
calculator — it retrieves nothing, judges nothing, and writes no run, no
ledger row and no number. That is why it may do two things the measured path
must not: talk to OpenRouter through its own `ChatOpenAI`, and trace to
LangSmith (the env variables below). The architecture is a first cut on
purpose; promoting or removing it later touches this module and one route.

`langchain.agents.create_agent` cannot install here — ragas 0.4 pins
`langchain<1` — so the agent is langgraph's `create_react_agent`, the same
prebuilt tool loop under its pre-1.0 name, from the `agent` extra.
"""
import ast
import os
import re

from langchain_core.tools import tool

# Read at build time, never at import: the suite runs offline, and a missing
# variable must become a stated refusal rather than a KeyError at import.
REQUIRED_ENV = ('OPENROUTER_API_KEY', 'LANGSMITH_API_KEY',
                'LANGSMITH_ENDPOINT', 'LANGSMITH_PROJECT', 'LANGSMITH_TRACING')

WIDGET_MODEL = 'openai/gpt-5-mini'
OPENROUTER_URL = 'https://openrouter.ai/api/v1'

KNOWLEDGE_BASE = {
    'purpose': 'The RAG lab is a workbench for diary-memory retrieval: '
               'chunking and retrieval choices are decided by measurement '
               'against fixtures/diary_year_fa.json, 167 sessions of '
               'synthetic Farsi diary chat with ground truth.',
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


_AGENT = None


def reset() -> None:
    """Drop the cached agent so the next ask() rebuilds it (tests, key changes)."""
    global _AGENT
    _AGENT = None


def _build_agent():
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

    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError as error:
        raise WidgetUnavailable(
            'langgraph is not installed — launch with --extra agent') from error
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=WIDGET_MODEL, api_key=openrouter_api_key,
                     base_url=OPENROUTER_URL)
    return create_react_agent(llm, tools=TOOLS, prompt=SYSTEM_PROMPT)


def ask(message: str) -> str:
    """One question in, one answer out. Builds the agent on first use."""
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent()
    try:
        result = _AGENT.invoke(
            {'messages': [{'role': 'user', 'content': message}]},
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
    return str(reply)
