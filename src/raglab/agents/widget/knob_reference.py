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

#: How many knobs one search returns. A shortlist to commit to, not a wall to
#: narrow: the whole surface, unranked, is what the model rewords its way
#: around instead of reading a page.
MAX_SEARCH_HITS = 8

#: What a query word is worth by where it landed. A key match names the knob
#: asked about, a summary match describes it, a body match may be one clause
#: in a page about something else — and the spread has to be wide enough that
#: one key match outranks several passing mentions.
KEY_WEIGHT, SUMMARY_WEIGHT, BODY_WEIGHT = 8, 4, 1

#: Question and function words, dropped before scoring. They carry no topic
#: and they actively mislead: the first ranking measured put the three
#: `*_model` knobs on top of "which knob fixes missed retrieval", because
#: their summaries open with "which model …" and `which` was scoring as a
#: summary match. Domain words are deliberately absent from this list — `run`
#: is a group prefix and `knob` appears in every page body, where a uniform
#: +1 changes no ordering.
_STOPWORDS = frozenset('''
    which what when where why how does did done the and for with that this
    from into are was were has have had can could should would will not you
    your our its about one any all more most than then there their them they
    use used using give gives make makes something anything
'''.split())


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


def search(query: str, root=None, limit=MAX_SEARCH_HITS) -> list:
    """(key, summary) for the knobs a query matches, closest first.

    Literal and case-insensitive, like the skills corpus's matcher, but
    ranked and bounded — which that one does not need to be and this one
    does. A skill catalogue is twelve descriptions; this is fifty pages of
    2.6 KB prose, so "any query word anywhere" matched up to the whole
    surface and returned it alphabetically. Measured on 2026-09-03: the real
    question "which knob fixes missed retrieval" matched all fifty, led by
    the generation knobs, and the model answered by searching again eight
    times until the hop guard stopped the turn. Nothing was wrong with the
    guard; the reply was simply of no use.

    So *where* a word lands decides what it is worth: the key names the
    knob, the summary describes it, and the page merely mentions it. Ties
    break by key, so the ordering is stable rather than dict-ordered, and
    `limit=None` asks for the whole ranked list — which is how the tool
    knows how many it is not showing."""
    words = {w for w in re.findall(r'[a-z0-9_.-]+', query.lower())
             if len(w) > 2 and w not in _STOPWORDS}
    if not words:
        return []
    scored = []
    for key, (summary, text) in _load(root).items():
        low_key, low_summary, low_text = key.lower(), summary.lower(), text.lower()
        score = 0
        for word in words:
            if word in low_key:
                score += KEY_WEIGHT
            elif word in low_summary:
                score += SUMMARY_WEIGHT
            elif word in low_text:
                score += BODY_WEIGHT
        if score:
            scored.append((-score, key, summary))
    scored.sort()
    hits = [(key, summary) for _, key, summary in scored]
    return hits if limit is None else hits[:limit]


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
