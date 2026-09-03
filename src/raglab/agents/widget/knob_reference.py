"""The knob reference's loader: `knobs/*.md` read from disk, on demand.

The folder is the source of truth — nothing from the Markdown is copied into
Python, so writing a page is the whole of adding a knob's explanation
(`test_knob_reference.py` holds the loader to serving exactly the knobs the
lab explains, in both directions). The title line is the cheap layer
(`# <group>.<field> — <summary>`, the key naming its own file), the page is
the expensive one, and both are cached in process memory by file mtime — the
same call `MemoryVectors`, the skills corpus and the widget's agent cache
make.

`related()` is the third reading, and the reason this corpus needs no graph
engine: each page closes with `## Interactions`, and the knob keys named
there are edges. Parsed rather than hand-listed, so a page and its
neighbours cannot drift apart.
"""
import re
from pathlib import Path

from raglab.configuration.env_settings import ROOT

KNOBS_DIR = ROOT / 'fixtures' / 'knobs'

# path -> (mtime, summary, page); a page whose title line does not match its
# own stem caches as None so it is not re-parsed on every call either.
_CACHE: dict = {}

# The four groups the lab's knob keys belong to. A closed list on purpose: it
# is what lets `related()` read edges out of prose without treating every
# dotted word in a sentence as a knob.
_KEY = re.compile(r'\b(?:index|retrieval|generation|run)\.[a-z][a-z0-9_-]*')

_INTERACTIONS = '## Interactions'


def _parse(path: Path):
    """One file to (summary, page), or None when the title line is malformed.

    The key is the filename stem and the title must state it, so a renamed
    file cannot keep answering under its old key."""
    text = path.read_text(encoding='utf-8')
    title = text.split('\n', 1)[0]
    prefix = f'# {path.stem} — '
    if '.' not in path.stem or not title.startswith(prefix):
        return None
    return title[len(prefix):].strip(), text


def _load(root=None) -> dict:
    """Every well-formed page under *root*, as key -> (summary, page)."""
    folder = Path(root) if root else KNOBS_DIR
    corpus = {}
    for path in sorted(folder.glob('*.md')):
        stamp = path.stat().st_mtime
        cached = _CACHE.get(path)
        if not cached or cached[0] != stamp:
            parsed = _parse(path)
            _CACHE[path] = (stamp, *parsed) if parsed else (stamp, None, None)
            cached = _CACHE[path]
        if cached[1] is not None:
            corpus[path.stem] = (cached[1], cached[2])
    return corpus


def index(root=None) -> dict:
    """knob key -> its one-line summary. The layer a search routes on."""
    return {key: summary for key, (summary, _) in _load(root).items()}


def page(key: str, root=None) -> str:
    """One knob's whole page, or a KeyError naming the keys there are."""
    corpus = _load(root)
    if key not in corpus:
        raise KeyError(f'{key!r} is not a knob; the knobs are: '
                       + ', '.join(sorted(corpus)))
    return corpus[key][1]


def related(key: str, root=None) -> tuple:
    """The knobs *key*'s own Interactions section names, in reading order.

    Itself excluded — a page is not its own neighbour. Not filtered against
    the corpus: a neighbour with no page of its own is rot, and the place to
    fail is the convention test that reads every page, not here, where the
    filter would hide it."""
    text = page(key, root)
    _, _, interactions = text.partition(_INTERACTIONS)
    seen, found = set(), []
    for match in _KEY.findall(interactions):
        if match == key or match in seen:
            continue
        seen.add(match)
        found.append(match)
    return tuple(found)


def search(query: str, root=None) -> list:
    """(key, summary) for every knob whose key, summary or page matches.

    Literal, case-insensitive, one word at a time — the same cheap matcher
    the skills corpus uses, for the same reason: a helper's search must not
    need an index of its own."""
    words = {w for w in re.findall(r'[a-z0-9_.-]+', query.lower()) if len(w) > 2}
    if not words:
        return []
    hits = []
    for key, (summary, text) in _load(root).items():
        haystack = f'{key}\n{summary}\n{text}'.lower()
        if any(word in haystack for word in words):
            hits.append((key, summary))
    return hits


def index_text(root=None) -> str:
    """The whole knob surface as one block, grouped by step.

    What the search tool serves when nothing matched: a miss should leave the
    model knowing what there is to ask for, not guessing."""
    corpus = index(root)
    lines = []
    for group in ('index', 'retrieval', 'generation', 'run'):
        keys = [k for k in corpus if k.startswith(f'{group}.')]
        if not keys:
            continue
        lines.append(f'{group}:')
        lines += [f'  {key} — {corpus[key]}' for key in keys]
    return 'The knobs:\n' + '\n'.join(lines)
