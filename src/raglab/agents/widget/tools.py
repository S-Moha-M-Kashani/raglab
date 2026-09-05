"""The widget's tools, and the registry the agent is handed.

The docstrings on the tools below are for readers of this file; what the
model sees is assigned from widget_tools.yaml right after TOOLS — the fixture
is the single source, and the import fails on a tool the page does not name.
A future capability is one new @tool here plus its entry on that page.
"""
import ast
import re

from langchain_core.tools import tool

from raglab.agents.widget import knob_reference as knobs
from raglab.agents.widget import skills_corpus_loader as skills
from raglab.agents.widget.conversation_memory import recall_conversation
from raglab.agents.widget.experiment_tools import EXPERIMENT_TOOLS
from raglab.agents.widget import long_term_memory
from raglab.agents.widget import probe
from raglab.agents.widget.prompts import _TOOL_PROMPTS, KNOWLEDGE_BASE


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


# How many knob pages one read_knob call returns. Same reasoning as the skill
# cap one layer down: the pages are the expensive layer, and a call asking for
# the whole surface would put ~130 KB into the loop.
MAX_KNOB_READS = 3


@tool
def search_knobs(query: str) -> str:
    """This lab's own knobs, matched literally, the whole surface on a miss;
    the model-facing prompt is fixtures/prompts/widget_tools.yaml's entry."""
    hits = knobs.search(query, limit=None)
    if not hits:
        return (f"No knob matches '{query}'.\n\n" + knobs.index_text())
    shown, rest = hits[:knobs.MAX_SEARCH_HITS], hits[knobs.MAX_SEARCH_HITS:]
    lines = [f'{key} — {summary}' for key, summary in shown]
    if rest:
        # A cap the model cannot see is an invitation to reword and search
        # again — which is the loop this whole ranking exists to end.
        lines.append(f'({len(rest)} more matched less closely. Narrow the '
                     'query, or read one of the above with read_knob rather '
                     'than searching again.)')
    return '\n'.join(lines)


@tool
def read_knob(keys: str) -> str:
    """Whole knob pages with the knobs each interacts with, several per call
    but capped; the model-facing prompt is
    fixtures/prompts/widget_tools.yaml's entry."""
    catalogue = knobs.index()
    asked = [k for k in re.split(r'[,\s]+', keys.strip()) if k]
    known = [k for k in asked if k in catalogue]
    unknown = [k for k in asked if k not in catalogue]
    served, over_cap = known[:MAX_KNOB_READS], known[MAX_KNOB_READS:]
    parts = []
    for key in served:
        neighbours = knobs.related(key)
        # The neighbours ride along rather than waiting for a second search:
        # the page names them in prose, and this is the same list as a
        # machine-readable line, so a follow-up question needs no round trip.
        beside = ('\nrelated knobs: ' + ', '.join(neighbours)) if neighbours else ''
        parts.append(f'=== {key} ===\n{knobs.page(key)}{beside}')
    if over_cap:
        parts.append(f'At most {MAX_KNOB_READS} knobs per call — not served: '
                     + ', '.join(over_cap) + '. Ask again for them.')
    if unknown:
        parts.append('Not knobs: ' + ', '.join(unknown)
                     + '. The knobs are: ' + ', '.join(sorted(catalogue)) + '.')
    if not parts:
        return 'No keys given. The knobs are: ' + ', '.join(sorted(catalogue)) + '.'
    return '\n\n'.join(parts)


@tool
def measure_bilingual_alignment(model_name: str = '', pairs: str = '') -> str:
    """The EN-Farsi alignment probe over a real encoder — pair cosine,
    mixed-pool retrieval, a verdict. The measurement itself is probe.py's
    `measure`; the model-facing prompt, including the pairs contract, is
    fixtures/prompts/widget_tools.yaml's entry."""
    return probe.measure(model_name, pairs)


@tool
def read_long_term_memory(dataset_id: str) -> str:
    """Read the applicable dataset and cross-dataset memory context.

    This is deliberately a separate read seam from the transcript recall:
    long-term memory contains only accepted summaries, never evidence or a
    measured result.
    """
    context = long_term_memory.memory_context(dataset_id)
    return context or 'No long-term memory is stored for this dataset.'


@tool
def save_widget_memory(dataset_id: str, experiment_id: str, subtopic: str,
                       question: str, answer: str, dataset_summary: str,
                       global_summary: str = '',
                       validated_dataset_ids: set[str] | None = None) -> dict:
    """Persist one structured, policy-approved summarizer result."""
    return long_term_memory.save_memory_update(
        dataset_id, experiment_id, subtopic, question, answer,
        dataset_summary, global_summary, validated_dataset_ids)


# The recorded-experiment tools are defined in their own module: they are one
# concern (what this lab has already measured), and the only tools whose data
# is injected rather than read from a fixture.
TOOLS = [search_knowledge_base, calculate, search_rag_skills, read_rag_skill,
         search_knobs, read_knob,
         measure_bilingual_alignment, read_long_term_memory,
         recall_conversation] + EXPERIMENT_TOOLS

# The YAML page is what the model reads; assigning it here makes the fixture
# the single source, and the import fails on a tool the page does not name.
for _each in TOOLS:
    _each.description = _TOOL_PROMPTS[_each.name].strip()
