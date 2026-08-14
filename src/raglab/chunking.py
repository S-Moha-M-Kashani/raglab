"""Chunking strategies for spoken diary text (fixed, fixed-overlap, message,
turn-pair, session, semantic-drift), plus the per-chunk metadata every chunk
carries into the index. `contextual` is orthogonal to all six: a short
metadata-only header prepended to each chunk before embedding.
"""
from dataclasses import dataclass

import numpy as np

from . import textnorm
from .corpus import date_int, session_text


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Greedy word packing, kept verbatim as the `fixed` baseline so old `.runs/` rows stay comparable."""
    chunks: list[str] = []
    current = ''
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = f'{current} {word}' if current else word
    if current:
        chunks.append(current)
    return chunks

# Phrases the diarist actually uses when he abandons one subject for another.
# A hard cut here is cheap and catches shifts the embedding drift misses.
SHIFT_MARKERS = ('ولش کن', 'بذریم', 'بگذریم', 'راستی', 'یه چیز دیگه',
                 'در کل', 'یادم رفت بگم', 'اینا رو ولش', 'حالا اینا')


@dataclass
class Chunk:
    id: str
    text: str
    session_id: str = ''
    date: str = ''
    span_from: int = 0          # date_int; equals span_to for leaf chunks
    span_to: int = 0
    time: str = ''
    source: str = ''
    mood: str = ''
    valence: int = 0
    arousal: int = 0
    importance: float = 0.0
    topics: tuple[str, ...] = ()
    threads: tuple[str, ...] = ()
    msg_start: int = -1
    msg_end: int = -1
    prefix: str = ''            # the contextual header, kept separately so metrics can measure the body alone
    # A leaf is layer='', level=0; hierarchy.py writes layer='summary' at its
    # level. Present (empty) on every chunk, never absent — an optional field
    # would turn a `where` clause into a silent partial scan.
    layer: str = ''
    level: int = 0
    group_id: str = ''
    member_ids: tuple[str, ...] = ()

    def metadata(self) -> dict:
        """Flat and filterable: lists become space-joined strings since the store can only filter on scalars."""
        return {
            'session_id': self.session_id, 'date': self.date,
            'span_from': self.span_from, 'span_to': self.span_to,
            'time': self.time, 'source': self.source, 'mood': self.mood,
            'valence': self.valence, 'arousal': self.arousal,
            'importance': self.importance,
            'topics': ' '.join(self.topics), 'threads': ' '.join(self.threads),
            'msg_start': self.msg_start, 'msg_end': self.msg_end,
            'chars': len(self.text),
            'layer': self.layer, 'level': self.level,
            'group_id': self.group_id, 'member_ids': ' '.join(self.member_ids),
        }

    @property
    def body(self) -> str:
        """The chunk without its contextual header."""
        return self.text[len(self.prefix):] if self.prefix else self.text


def importance_of(session: dict) -> float:
    """Emotional intensity as a memorability proxy (Generative Agents'
    'importance' term): high arousal or extreme valence marks a memorable session."""
    mood = session['mood']
    arousal = mood['arousal'] / 10.0
    extremity = abs(mood['valence'] - 5.5) / 4.5
    return round(min(1.0, 0.6 * arousal + 0.4 * extremity), 3)


# Both strings are prepended to text that gets embedded, so writing them in
# Farsi over an English corpus would add a constant foreign phrase to every vector.
SPEAKERS = {'fa': ('کاربر', 'دستیار'), 'de': ('Nutzer', 'Assistent')}
SPEAKERS_DEFAULT = ('User', 'Assistant')
HEADERS = {'fa': ('حال', 'موضوع', 'رشته', '، '),
           'de': ('Stimmung', 'Thema', 'Strang', ', ')}
HEADERS_DEFAULT = ('mood', 'topic', 'thread', ', ')


def _language(session: dict) -> str:
    """The corpus's language; absent (every session of the built-in fixture) defaults to Farsi."""
    return session.get('language') or 'fa'


def _speaker(role: str, language: str = 'fa') -> str:
    user, assistant = SPEAKERS.get(language, SPEAKERS_DEFAULT)
    return user if role == 'user' else assistant


def _base(session: dict) -> dict:
    di = date_int(session['date'])
    return dict(session_id=session['session_id'], date=session['date'],
                span_from=di, span_to=di, time=session['time'],
                source=session['source'], mood=session['mood']['label'],
                valence=session['mood']['valence'], arousal=session['mood']['arousal'],
                importance=importance_of(session),
                topics=tuple(session['topics']),
                threads=tuple(session['recurring_threads']))


def contextual_prefix(session: dict) -> str:
    """A short header naming when this was said, how it felt, and what it was
    about. Deliberately short: duplicated into every chunk of the session."""
    mood_word, topic_word, thread_word, comma = HEADERS.get(_language(session),
                                                            HEADERS_DEFAULT)
    threads = comma.join(session['recurring_threads']) or '—'
    head = (f"[{session['date']} | {mood_word}: {session['mood']['label'] or '—'} | "
            f"{topic_word}: {comma.join(session['topics']) or '—'} | "
            f"{thread_word}: {threads}]")
    return head + '\n'


# --- leaf strategies -------------------------------------------------------

def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Sliding character windows snapped to word boundaries."""
    if overlap >= size:
        overlap = size // 2
    step = max(1, size - overlap)
    words = text.split()
    out, i = [], 0
    while i < len(words):
        window, j = '', i
        while j < len(words) and len(window) + len(words[j]) + 1 <= size:
            window = f'{window} {words[j]}' if window else words[j]
            j += 1
        out.append(window)
        if j >= len(words):
            break
        # advance by roughly `step` characters, never by zero words
        consumed, k = 0, i
        while k < j - 1 and consumed + len(words[k]) + 1 < step:
            consumed += len(words[k]) + 1
            k += 1
        i = max(i + 1, k)
    return out


def _semantic_segments(session: dict, embedder, max_chars: int) -> list[list[int]]:
    """Groups consecutive messages into topical segments: cut where similarity
    drops into the bottom of *this session's own* distribution (a relative
    threshold, since embedder scales are not comparable), a shift marker fires, or max_chars is exceeded."""
    messages = session['messages']
    if len(messages) <= 2:
        return [list(range(len(messages)))]
    vectors = embedder.embed([m['content'] for m in messages])
    sims = np.array([float(vectors[i] @ vectors[i + 1])
                     for i in range(len(messages) - 1)], dtype=np.float32)
    cut_at = float(np.percentile(sims, 35)) if sims.size else -1.0
    segments, current, size = [], [0], len(messages[0]['content'])
    for i in range(1, len(messages)):
        marker = any(m in textnorm.normalize(messages[i]['content'])
                     for m in SHIFT_MARKERS)
        too_big = size + len(messages[i]['content']) > max_chars * 2
        drifted = sims[i - 1] <= cut_at
        if current and (marker or too_big or drifted):
            segments.append(current)
            current, size = [i], len(messages[i]['content'])
        else:
            current.append(i)
            size += len(messages[i]['content'])
    if current:
        segments.append(current)
    return segments


def chunk_session(session: dict, cfg, embedder) -> list[Chunk]:
    """One session → chunks, per the configured strategy."""
    base = _base(session)
    prefix = contextual_prefix(session) if cfg.contextual else ''
    messages = session['messages']
    out: list[Chunk] = []
    language = _language(session)

    def emit(text: str, i: int, start: int, end: int) -> None:
        out.append(Chunk(id=f"{session['session_id']}:c{i}", text=prefix + text,
                         prefix=prefix, msg_start=start, msg_end=end, **base))

    if cfg.chunker == 'session':
        emit(session_text(session), 0, 0, len(messages) - 1)
    elif cfg.chunker == 'message':
        for i, m in enumerate(messages):
            emit(f"{_speaker(m['role'], language)}: {m['content']}", i, i, i)
    elif cfg.chunker == 'turn-pair':
        i = 0
        while i < len(messages):
            group = [messages[i]]
            end = i
            if (messages[i]['role'] == 'user' and i + 1 < len(messages)
                    and messages[i + 1]['role'] == 'assistant'):
                group.append(messages[i + 1])
                end = i + 1
            emit('\n'.join(f"{_speaker(m['role'], language)}: {m['content']}"
                           for m in group),
                 len(out), i, end)
            i = end + 1
    elif cfg.chunker == 'semantic-drift':
        for i, segment in enumerate(_semantic_segments(session, embedder,
                                                       cfg.chunk_chars)):
            emit('\n'.join(f"{_speaker(messages[j]['role'], language)}: "
                           f"{messages[j]['content']}" for j in segment),
                 i, segment[0], segment[-1])
    elif cfg.chunker == 'fixed-overlap':
        for i, piece in enumerate(_windows(session_text(session), cfg.chunk_chars,
                                           cfg.overlap)):
            emit(piece, i, -1, -1)
    else:   # 'fixed' — the production baseline, called rather than reimplemented
        for i, piece in enumerate(chunk_text(session_text(session), cfg.chunk_chars)):
            emit(piece, i, -1, -1)
    return out
