"""The skills corpus's loader: `skills/*/SKILL.md` read from disk, on demand.

The folder is the source of truth — nothing from the Markdown is copied into
Python, so adding a skill is writing a file, not touching code
(`test_skills.py` holds the loader to serving exactly the folders that
exist). Frontmatter is the cheap layer (name, description — the routing
index), the body is the expensive one, and both are cached in process memory
by file mtime: cheap to rebuild, nothing to go stale on disk, dies with the
process — the same call `MemoryVectors` and the widget's agent cache make.

The one hand-written string is `DISTINCTIONS`, the guide to how the
near-neighbour skills differ. It cannot be derived from the files, so a
convention test forces it to name every skill instead.
"""
import re
from pathlib import Path

from raglab.configuration.env_settings import ROOT

SKILLS_DIR = ROOT / 'fixtures' / 'skills'

# path -> (mtime, name, description, body); a malformed file caches as None
# so it is not re-parsed on every call either.
_CACHE: dict = {}

_FRONTMATTER = re.compile(r'^---\n(.*?)\n---\n', re.S)


def _parse(path: Path):
    """One file to (name, description, body), or None when malformed — one
    broken page must not take the corpus down; the widget is a helper."""
    text = path.read_text(encoding='utf-8')
    matched = _FRONTMATTER.match(text)
    if not matched:
        return None
    front = matched.group(1)
    name = re.search(r'^name:\s*(.+)$', front, re.M)
    description = re.search(r'^description:\s*(.+)$', front, re.M)
    if not name or not description:
        return None
    body = text[matched.end():].lstrip('\n')
    return name.group(1).strip(), description.group(1).strip(), body


def _load(root=None) -> dict:
    """Every well-formed skill under `root`, as name -> (description, body)."""
    folder = Path(root) if root else SKILLS_DIR
    corpus = {}
    for path in sorted(folder.glob('*/SKILL.md')):
        stamp = path.stat().st_mtime
        cached = _CACHE.get(path)
        if cached is None or cached[0] != stamp:
            parsed = _parse(path)
            cached = (stamp,) + (parsed if parsed else (None, None, None))
            _CACHE[path] = cached
        if cached[1] is not None:
            corpus[cached[1]] = (cached[2], cached[3])
    return corpus


def index(root=None) -> dict:
    """The catalogue: skill name -> description, one entry per file."""
    return {name: description for name, (description, _) in _load(root).items()}


def body(name: str, root=None) -> str:
    """One skill's full Markdown body, frontmatter stripped. Unknown names
    raise with the valid ones in the message, so a caller can correct itself."""
    corpus = _load(root)
    if name not in corpus:
        raise KeyError(f'{name!r} is not a skill; the skills are: '
                       + ', '.join(sorted(corpus)))
    return corpus[name][1]


def search(query: str, root=None) -> list:
    """(name, description) pairs whose name, description or body mentions any
    query word — the same keyword rule as the widget's knowledge-base search."""
    words = {w for w in re.findall(r'[a-z0-9]+', query.lower()) if len(w) > 2}
    hits = []
    for name, (description, text) in _load(root).items():
        haystack = f'{name}\n{description}\n{text}'.lower()
        if any(w in haystack for w in words):
            hits.append((name, description))
    return hits


# How the near-neighbours differ — the routing judgment the descriptions
# alone cannot carry. Hand-written on purpose; the test that pins it requires
# every skill's name to appear here, so a new folder forces a new line.
DISTINCTIONS = """\
How to tell the near-neighbours apart:
- chunking-strategies splits documents; contextual-retrieval enriches chunks
  after the split. Splitting badly and enriching nothing are different faults.
- hybrid-retrieval-fusion merges two first-stage ranked lists;
  reranking-late-interaction re-scores one shortlist afterwards. Fusion is
  about recall, reranking about the order of the top few.
- query-transformation rewrites the query once, before retrieval;
  agentic-rag loops — retrieve, judge the evidence, rewrite, retry — with
  caps and stop reasons. One price paid always versus a larger price paid
  conditionally.
- adaptive-corrective-rag decides per query (route it, gate its evidence);
  agentic-rag iterates within a query. A router is not a loop.
- hierarchical-graph-rag is index-time structure for questions no single
  chunk answers; it changes what exists to retrieve, not how retrieval runs.
- multilingual-rag is the failure modes of non-English corpora — encoders,
  tokenisers, prompts — wherever they appear in the pipeline.
- rag-evaluation says what a number is allowed to mean (metrics, judges,
  error bars); rag-experiment-methodology says how to iterate without
  fooling yourself (dev/test discipline, error analysis). The instrument
  versus the loop that uses it.
- rag-use-case-architectures picks candidate zero and the experiment ladder
  for a use case; rag-experiment-methodology governs every rung of that
  ladder once picked.
- rag-research-radar is the field-wide procedure for finding and triaging
  new work; rag-source-watchlist is its address book — per use case, the
  named sources and query phrases.
"""


def index_text(root=None) -> str:
    """The whole cheap layer as one block — every name with its full
    description, then the distinctions guide. What the search tool answers
    with when nothing matches, and what the CLI prompt inlines."""
    lines = [f'- {name}: {description}'
             for name, description in index(root).items()]
    return 'The skills:\n' + '\n'.join(lines) + '\n\n' + DISTINCTIONS
