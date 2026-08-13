"""Presentation helpers shared by the panel (:9002) and the Inspector (:9003).

Live here rather than in `inspector.py`, which `server.py` already imports
from — `server.py` importing back would be circular.
"""
from . import textnorm


def _norm(text: str) -> str:
    return ' '.join(textnorm.tokens(text, drop_stopwords=False))


def mark_gold(candidate_texts: list[str],
              evidence_quotes: list[str]) -> list[bool]:
    """Which candidates contain a question's gold evidence quote.

    Substring match either direction over the shared normaliser, since a chunk
    may be smaller than a quote or larger. Text or quotes that normalise to
    empty are excluded first — the empty string is a substring of everything,
    and would otherwise mark every candidate gold."""
    return _gold_flags([_norm(text) for text in candidate_texts], evidence_quotes)


def _gold_flags(normalised_texts: list[str],
                evidence_quotes: list[str]) -> list[bool]:
    """The matching itself, over pre-normalised text, so a multi-question caller normalises once."""
    quotes = [n for n in (_norm(q) for q in evidence_quotes) if n]
    return [bool(text) and any(q in text or text in q for q in quotes)
            for text in normalised_texts]


def normalised_chunks(index) -> list[str]:
    """Every chunk in the index, normalised once, for `gold_available`."""
    return [_norm(chunk.text) for chunk in index.chunks]


def gold_available(index, evidence_quotes: list[str],
                   norm_chunks: list[str] | None = None) -> int:
    """How many chunks in the index hold this question's evidence — the denominator behind "1 of how many".

    Counted over chunks, not over evidence quotes: a length-based chunker can
    split one quote across chunks or pack two quotes into one, so the counts differ."""
    norms = norm_chunks if norm_chunks is not None else normalised_chunks(index)
    return sum(_gold_flags(norms, evidence_quotes))


def evidence_spans(text: str, evidence_quotes: list[str]) -> list[list[int]]:
    """`[start, end]` character ranges of a question's gold evidence in one candidate's text.

    Verbatim `str.find`, not the normaliser `mark_gold` uses, since a highlight
    must match the exact rendered characters. A candidate smaller than its quote
    is still gold but has no verbatim quote to highlight, so it returns nothing
    rather than a guessed range. Overlapping or touching ranges are merged, since
    two nested `<mark>` elements would render as an unintended third stripe."""
    found: list[list[int]] = []
    for quote in evidence_quotes:
        if not quote:
            continue
        start = text.find(quote)
        while start != -1:
            found.append([start, start + len(quote)])
            start = text.find(quote, start + 1)
    merged: list[list[int]] = []
    for start, end in sorted(found):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def chunks_by_session(index) -> list[dict]:
    """Chunks the *chunker* produced, grouped by session in index order — leaves only.

    `summary_rows` is the other half; between them every row in the index is
    covered exactly once. `by_session` is already in chunk order, so no sorting is needed."""
    groups = []
    for session_id, chunks in index.by_session.items():
        leaves = [c for c in chunks if c.layer != 'summary']
        if not leaves:
            continue
        groups.append({
            'session_id': session_id,
            'date': leaves[0].date,
            'chunks': [{'id': c.id, 'text': c.text} for c in leaves]})
    return groups


def _leaf_sessions(index, chunk) -> set[str]:
    """Sessions a summary ultimately speaks for, resolved transitively through its members.

    A multi-session summary carries `session_id=''`, and at two levels a
    level-2 group's members are themselves summaries with no session id —
    so only recursion reports the sessions a top-level summary spans."""
    if chunk.layer != 'summary':
        return {chunk.session_id} if chunk.session_id else set()
    found: set[str] = set()
    for member_id in chunk.member_ids:
        member = index.by_id.get(member_id)
        if member is not None:
            found |= _leaf_sessions(index, member)
    return found


def summary_rows(index) -> list[dict]:
    """Every row a hierarchy wrote, in index order — the other half of the chunk view.

    Each row states what its text cannot: which group, at which level, over how
    many members and sessions. An empty list on a flat index, never a missing
    key — "no hierarchy" and "a hierarchy that found nothing" are different facts."""
    rows = []
    for chunk in index.chunks:
        if chunk.layer != 'summary':
            continue
        rows.append({
            'id': chunk.id, 'text': chunk.text,
            'group_id': chunk.group_id, 'level': chunk.level,
            'members': len(chunk.member_ids),
            'member_ids': list(chunk.member_ids),
            'sessions': len(_leaf_sessions(index, chunk)),
            'date': chunk.date, 'chars': len(chunk.text)})
    return rows
