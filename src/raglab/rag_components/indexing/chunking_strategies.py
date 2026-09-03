"""Chunking strategies for a corpus's documents (fixed, fixed-overlap,
message, turn-pair, session, semantic-drift), plus the per-chunk metadata
every chunk carries into the index. `contextual` is orthogonal to all six: a
short metadata-only header prepended to each chunk before embedding.

Everything here reads the corpus's own declared vocabulary (D4) — a
document's `document_metadata`, a part's `labels`, and the dataset's
`label_fields` table that says what each one means and where it may land —
rather than a fixed set of diary-specific fields. A label a corpus never
declares simply never appears; nothing here invents a neutral default for
one that is absent.
"""
import json
from dataclasses import dataclass, field

import numpy as np

from raglab.rag_components.retrieval import farsi_text_normalizer as textnorm
from raglab.corpora.corpus_reading import (
    date_int, date_label, document_text, part_line, ranks_label)


def _split_on_delimiters(text: str, delimiters: tuple[str, ...],
                         max_chars: int) -> list[str]:
    """The pieces a character-budget chunker packs, cut at the highest-priority
    boundary that works. Each delimiter is tried in the order given: a piece
    that already fits `max_chars` is kept whole, and only a piece still too big
    is cut again on the next delimiter down. An exhausted list — or an empty
    one, the default — falls through to `text.split()`, so today's plain word
    packing is this function's base case rather than a branch beside it, and
    `delimiters=()` returns exactly the words it always did."""
    if not delimiters:
        return text.split()
    pieces: list[str] = []
    for piece in text.split(delimiters[0]):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= max_chars:
            pieces.append(piece)
        else:
            pieces.extend(_split_on_delimiters(piece, delimiters[1:], max_chars))
    return pieces


def chunk_text(text: str, max_chars: int = 500,
               delimiters: tuple[str, ...] = ()) -> list[str]:
    """Greedy packing of whatever `_split_on_delimiters` hands back, kept verbatim
    as the `fixed` baseline so old `.runs/` rows stay comparable — at the default
    `delimiters=()` those pieces are the words `text.split()` always produced."""
    chunks: list[str] = []
    current = ''
    for piece in _split_on_delimiters(text, delimiters, max_chars):
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f'{current} {piece}' if current else piece
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
    document_id: str = ''
    date: str = ''
    span_from: int = 0          # date_int; equals span_to for leaf chunks
    span_to: int = 0
    importance: float = 0.0
    # Every label the corpus declares that reaches this chunk, flowed down
    # from the document or the parts it was cut from (D4) — never a fixed
    # set of fields, and never filled with a neutral value when absent.
    labels: dict = field(default_factory=dict)
    part_start: int = -1
    part_end: int = -1
    prefix: str = ''            # the contextual header, kept separately so metrics can measure the body alone
    # A leaf is layer='', level=0; summary_hierarchy_builder.py writes layer='summary' at its
    # level. Present (empty) on every chunk, never absent — an optional field
    # would turn a `where` clause into a silent partial scan.
    layer: str = ''
    level: int = 0
    group_id: str = ''
    member_ids: tuple[str, ...] = ()

    def metadata(self) -> dict:
        """Flat and filterable: lists become space-joined strings and objects
        become a JSON string, since the store can only filter on scalars.
        The flattened keys are prefixed (`label.topics`) so a label can never
        collide with a structural key."""
        meta = {
            'document_id': self.document_id, 'date': self.date,
            'span_from': self.span_from, 'span_to': self.span_to,
            'importance': self.importance,
            'part_start': self.part_start, 'part_end': self.part_end,
            'chars': len(self.text),
            'layer': self.layer, 'level': self.level,
            'group_id': self.group_id, 'member_ids': ' '.join(self.member_ids),
        }
        for name, value in self.labels.items():
            flat = _flattened(value)
            if flat is not None:
                meta[f'label.{name}'] = flat
        return meta

    @property
    def body(self) -> str:
        """The chunk without its contextual header."""
        return self.text[len(self.prefix):] if self.prefix else self.text


def is_present(value) -> bool:
    """Whether a label's own value is something to show or group by, not
    nothing recorded — the one presence rule `contextual_prefix` and
    `summary_hierarchy_builder._metadata_groups` both defer to, so neither
    can silently disagree with the other about what "absent" means. `None`
    (a nullable label recorded absent, D4) and an empty string carry
    nothing; `0`, `False`, and any other real value — including one item
    already pulled out of a list — do."""
    return value is not None and value != ''


def _flattened(value):
    """One label value, made chroma-safe: a list space-joined, an object (a
    keyed confidence, or a rolled-up date range) a JSON string, a scalar as
    itself. `None` (a nullable label recorded absent) stays absent rather
    than becoming the string `'None'`."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ' '.join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def importance_of(document: dict, label_fields: dict) -> float:
    """The `ranks` label (D6) rescaled to 0-1, or 0.0 without one declared —
    corrected in place, same name, same call sites. `mood.valence`/`arousal`
    are gone: the new schema deliberately has no such field, and importance
    now reads whatever numeric label the corpus itself declared as the
    source (`ranks: true`, with `minimum`/`maximum`)."""
    name = ranks_label(label_fields)
    if not name:
        return 0.0
    value = (document.get('document_metadata') or {}).get(name)
    if value is None:
        return 0.0
    definition = label_fields[name]
    lo, hi = definition.get('minimum', 0.0), definition.get('maximum', 1.0)
    if hi <= lo:
        return 0.0
    return round(min(1.0, max(0.0, (float(value) - lo) / (hi - lo))), 3)


# The corpus's own declared language picks the comma its list-valued labels
# join by inside `contextual_prefix` — Farsi's is not the ASCII comma. A tiny
# mapping, not an i18n library: every language without a special-cased comma
# falls back to ', '.
_LANGUAGE_COMMA = {'fa': '، '}
_DEFAULT_COMMA = ', '


def _comma_for(language: str) -> str:
    return _LANGUAGE_COMMA.get(language, _DEFAULT_COMMA)


def contextual_prefix(document: dict, label_fields: dict, language: str) -> str:
    """`[<label>: <value> | …]` for every label declared at the document
    level that this document actually carries, in declaration order, list
    values joined by the corpus's own language comma. Deliberately short:
    duplicated into every chunk of the document. A label's own name is its
    name — no HEADERS translation table, since inventing a word for a user's
    label would be guessing. A label declaring `confidence_for` is never
    shown: it is a caveat on another label, never something to embed."""
    meta = document.get('document_metadata') or {}
    comma = _comma_for(language)
    parts: list[str] = []
    for name, definition in (label_fields or {}).items():
        if 'document' not in (definition.get('applies_to') or []):
            continue
        if definition.get('confidence_for'):
            continue
        if name not in meta:
            continue
        value = meta[name]
        # A nullable label recorded `null` (D4: absence, never "recorded as
        # nothing") is skipped here rather than rendered as the literal word
        # "None" — the same presence rule `is_present` states once for every
        # reader of a label value.
        if not is_present(value):
            continue
        if isinstance(value, (list, tuple)):
            shown = comma.join(str(v) for v in value if is_present(v))
        elif isinstance(value, dict):
            shown = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            shown = str(value)
        if not shown:
            continue
        parts.append(f'{name}: {shown}')
    if not parts:
        return ''
    return '[' + ' | '.join(parts) + ']\n'


def _document_date(document: dict, label_fields: dict) -> tuple[str, int, int]:
    """`date`/`span_from`/`span_to` from the date-typed label (D5). No date
    label, or a value that will not parse, means '' and 0/0 — inert, not an
    error: time filtering, recency ranking and summary date ranges all read
    this and must degrade gracefully rather than raise."""
    name = date_label(label_fields)
    if not name:
        return '', 0, 0
    value = (document.get('document_metadata') or {}).get(name)
    if not value:
        return '', 0, 0
    day = str(value)[:10]
    try:
        di = date_int(day)
    except ValueError:
        return '', 0, 0
    return day, di, di


def _flowing(label_fields: dict, source: str, target: str) -> list[str]:
    """Label names declared at both `source` (where the value actually lives
    — `document` or `part`) and `target` (the level they flow down into,
    e.g. `chunk`) — the applies_to table is what makes a label reach a chunk
    at all."""
    return [name for name, definition in (label_fields or {}).items()
           if source in (definition.get('applies_to') or [])
           and target in (definition.get('applies_to') or [])]


def _document_chunk_labels(document: dict, label_fields: dict) -> dict:
    """Document-level labels that reach a chunk, read straight from
    `document_metadata` — absent stays absent."""
    meta = document.get('document_metadata') or {}
    return {name: meta[name] for name in _flowing(label_fields, 'document', 'chunk')
           if name in meta}


def _part_chunk_labels(document: dict, label_fields: dict,
                       start: int, end: int) -> dict:
    """Part-level labels for the parts a chunk actually spans (`role`, most
    often), unioned across the span — a single value stays a scalar, more
    than one becomes a list, the same rule a summary uses to union its
    members. A chunk cut by character window (`start < 0`) has no part span
    to read a part-level label from at all."""
    if start < 0:
        return {}
    names = _flowing(label_fields, 'part', 'chunk')
    if not names:
        return {}
    parts = document.get('document_content') or []
    collected: dict[str, list] = {name: [] for name in names}
    for part in parts[start:end + 1]:
        part_labels = part.get('labels') or {}
        for name in names:
            if name in part_labels and part_labels[name] not in collected[name]:
                collected[name].append(part_labels[name])
    return {name: (values[0] if len(values) == 1 else values)
           for name, values in collected.items() if values}


def _base(document: dict, label_fields: dict) -> dict:
    date, span_from, span_to = _document_date(document, label_fields)
    return dict(document_id=str(document.get('corpus_document_id', '')),
               date=date, span_from=span_from, span_to=span_to,
               importance=importance_of(document, label_fields),
               labels=_document_chunk_labels(document, label_fields))


# --- leaf strategies -------------------------------------------------------

def _windows(text: str, size: int, overlap: int,
             delimiters: tuple[str, ...] = ()) -> list[str]:
    """Sliding character windows snapped to the boundaries `_split_on_delimiters`
    found — word boundaries at the default `delimiters=()`. The window loop
    below admits a piece only while `len(window) + len(piece) + 1 <= size`, so
    the largest piece it can hold on its own is `size - 1` characters; splitting
    to that budget is what keeps a piece exactly `size` long from being packed
    into nothing."""
    if overlap >= size:
        overlap = size // 2
    step = max(1, size - overlap)
    words = _split_on_delimiters(text, delimiters, size - 1)
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


def _semantic_segments(document: dict, embedder, max_chars: int) -> list[list[int]]:
    """Groups consecutive parts into topical segments: cut where similarity
    drops into the bottom of *this document's own* distribution (a relative
    threshold, since embedder scales are not comparable), a shift marker
    fires, or max_chars is exceeded. Reads `document['document_content']`
    (D4: the file's own key, not a renamed one)."""
    parts = document.get('document_content') or []
    if len(parts) <= 2:
        return [list(range(len(parts)))]
    vectors = embedder.embed([p.get('text', '') for p in parts])
    sims = np.array([float(vectors[i] @ vectors[i + 1])
                     for i in range(len(parts) - 1)], dtype=np.float32)
    cut_at = float(np.percentile(sims, 35)) if sims.size else -1.0
    segments, current, size = [], [0], len(parts[0].get('text', ''))
    for i in range(1, len(parts)):
        text = parts[i].get('text', '')
        marker = any(m in textnorm.normalize(text) for m in SHIFT_MARKERS)
        too_big = size + len(text) > max_chars * 2
        drifted = sims[i - 1] <= cut_at
        if current and (marker or too_big or drifted):
            segments.append(current)
            current, size = [i], len(text)
        else:
            current.append(i)
            size += len(text)
    if current:
        segments.append(current)
    return segments


def chunk_document(document: dict, cfg, embedder, label_fields: dict | None = None,
                   language: str = '') -> list[Chunk]:
    """One document → chunks, per the configured strategy."""
    label_fields = label_fields or {}
    base = _base(document, label_fields)
    prefix = contextual_prefix(document, label_fields, language) if cfg.contextual else ''
    parts = document.get('document_content') or []
    document_id = base['document_id']
    out: list[Chunk] = []

    def emit(text: str, i: int, start: int, end: int) -> None:
        labels = dict(base['labels'])
        labels.update(_part_chunk_labels(document, label_fields, start, end))
        out.append(Chunk(
            id=f'{document_id}:c{i}', text=prefix + text, prefix=prefix,
            document_id=document_id, date=base['date'],
            span_from=base['span_from'], span_to=base['span_to'],
            importance=base['importance'], labels=labels,
            part_start=start, part_end=end))

    if cfg.chunker == 'session':
        emit(document_text(document), 0, 0, len(parts) - 1)
    elif cfg.chunker == 'message':
        for i, part in enumerate(parts):
            emit(part_line(part), i, i, i)
    elif cfg.chunker == 'turn-pair':
        i = 0
        while i < len(parts):
            group = [parts[i]]
            end = i
            if ((parts[i].get('labels') or {}).get('role') == 'user' and i + 1 < len(parts)
                    and (parts[i + 1].get('labels') or {}).get('role') == 'assistant'):
                group.append(parts[i + 1])
                end = i + 1
            emit('\n'.join(part_line(p) for p in group), len(out), i, end)
            i = end + 1
    elif cfg.chunker == 'semantic-drift':
        for i, segment in enumerate(_semantic_segments(document, embedder,
                                                       cfg.chunk_chars)):
            emit('\n'.join(part_line(parts[j]) for j in segment),
                 i, segment[0], segment[-1])
    elif cfg.chunker == 'fixed-overlap':
        for i, piece in enumerate(_windows(document_text(document), cfg.chunk_chars,
                                           cfg.overlap, cfg.delimiters)):
            emit(piece, i, -1, -1)
    else:   # 'fixed' — the production baseline, called rather than reimplemented
        for i, piece in enumerate(chunk_text(document_text(document), cfg.chunk_chars,
                                            cfg.delimiters)):
            emit(piece, i, -1, -1)
    return out
