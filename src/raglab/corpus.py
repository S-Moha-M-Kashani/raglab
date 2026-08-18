"""Fixture access: the year of diary sessions and its ground-truth questions.

One file since 2026-08-18 (they were `diary_year_fa.json` plus
`diary_year_fa_groundtruth.json`): corpus and ground truth merged under the
`groundtruth` key, in the folder every other corpus already lived in. The
diary keeps its native schema rather than the import contract — `persona`,
`threads` and `habits` are fields the pipeline reads and the contract does
not carry — so `datasets._files` skips this file by path instead of parsing
it as a bundled dataset.
"""
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures'
DIARY_PATH = FIXTURES / 'corpus_groundtruth_datasets' / 'diary_year_fa.json'

# path -> (mtime, parsed). The merge made both loaders read one ~1.4 MB file,
# and every builtin `datasets.load` calls both — without this, two full
# parses per call where the other corpora pay one per process. Mtime-keyed
# like the skills loader, so an edited fixture is still served fresh.
_CACHE: dict = {}


def _read(path: Path) -> dict:
    stamp = path.stat().st_mtime
    cached = _CACHE.get(path)
    if cached is None or cached[0] != stamp:
        with open(path, encoding='utf-8') as f:
            cached = (stamp, json.load(f))
        _CACHE[path] = cached
    return cached[1]


def load_diary(path: Path = DIARY_PATH) -> dict:
    return _read(path)


def load_ground_truth(path: Path = DIARY_PATH) -> dict:
    return _read(path)['groundtruth']


def date_int(date: str) -> int:
    """'2026-03-10' -> 20260310, so time filters can compare numbers instead of date strings."""
    return int(date.replace('-', ''))


def session_text(session: dict) -> str:
    """One session as plain, role-tagged dialogue text, in the corpus's own language, for embedding."""
    from .chunking import _language, _speaker
    language = _language(session)
    lines = []
    for message in session['messages']:
        lines.append(f"{_speaker(message['role'], language)}: {message['content']}")
    return '\n'.join(lines)


def sessions_by_id(diary: dict) -> dict[str, dict]:
    return {s['session_id']: s for s in diary['sessions']}


def evidence_texts(sessions: dict[str, dict], question: dict) -> list[str]:
    """Full text of every evidence message, as `reference_contexts` for RAGAS's whole-string context metrics
    (quote-level precision is `metrics.quote_recall` instead)."""
    out: list[str] = []
    for ev in question.get('evidence', []):
        session = sessions.get(ev['session_id'])
        if not session:
            continue
        for index in ev.get('message_indices', []):
            if 0 <= index < len(session['messages']):
                out.append(session['messages'][index]['content'])
    return out or [ev['quote'] for ev in question.get('evidence', [])]


def evidence_sessions(question: dict) -> list[str]:
    """Distinct evidence session ids, in the order the ground truth lists them."""
    seen: list[str] = []
    for ev in question.get('evidence', []):
        if ev['session_id'] not in seen:
            seen.append(ev['session_id'])
    return seen
