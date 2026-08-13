"""Fixture access: the year of diary sessions and its ground-truth questions."""
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures'
DIARY_PATH = FIXTURES / 'diary_year_fa.json'
GROUND_TRUTH_PATH = FIXTURES / 'diary_year_fa_groundtruth.json'


def load_diary(path: Path = DIARY_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


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
