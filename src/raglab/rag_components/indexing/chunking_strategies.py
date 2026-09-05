"""Cutting a corpus's documents into the chunks that get embedded, by the
split plan (`configuration/split_plan.py`): a fold over stages, each mapping
the pieces the previous stage produced to a longer list, and a budget that
closes the plan by dividing whatever is still too big. Every chunker the lab
used to name is one such plan, and `contextual` is orthogonal to all of them:
a short metadata-only header prepended to each chunk before embedding.

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

from raglab.configuration import split_plan as plan
from raglab.corpora.corpus_reading import (
    date_int, date_label, part_line, ranks_label)
from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.rag_components.retrieval import text_normalizers


def _chars(text: str) -> int:
    return len(text)


def budget_measure(cfg, embedder):
    """How the budget measures a text: `len` for characters, or the count of
    the units the embedder's own model reads, summed over words — a word never
    straddles a whitespace boundary in any tokeniser the lab loads, so the sum
    is the count. An embedder that cannot report its units is refused rather
    than measured in characters under the wrong name."""
    if cfg.chunk_unit != 'tokens':
        return _chars
    count = embedding.token_counter(embedder)
    if count is None:
        raise ValueError(
            f'chunk_unit=tokens: the {cfg.embedder} embedder reports no model '
            'units, so a budget in tokens cannot be counted — refused rather '
            'than silently counted in characters')
    return lambda text: sum(count(word) for word in text.split())


def chunk_text(text: str, max_chars: int = 500, measure=_chars) -> list[str]:
    """Greedy word packing to the budget, kept verbatim as the `fixed`
    baseline — a plan of the document alone at a budget is exactly this."""
    chunks: list[str] = []
    current = ''
    for piece in text.split():
        if current and measure(f'{current} {piece}') > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f'{current} {piece}' if current else piece
    if current:
        chunks.append(current)
    return chunks


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


# --- the stages -------------------------------------------------------------

def _windows(text: str, size: int, overlap: int, measure=_chars) -> list[str]:
    """Sliding windows snapped to word boundaries, each holding as many words
    as fit the budget and starting roughly `size - overlap` after the last —
    so a sentence sitting on a boundary appears whole in one of them."""
    if overlap >= size:
        overlap = size // 2
    step = max(1, size - overlap)
    words = text.split()
    out, i = [], 0
    while i < len(words):
        j = i
        while j < len(words) and measure(' '.join(words[i:j + 1])) <= size:
            j += 1
        j = max(j, i + 1)      # a word over the budget on its own still lands
        out.append(' '.join(words[i:j]))
        if j >= len(words):
            break
        # advance by roughly `step`, never by zero words
        k = i
        while k < j - 1 and measure(' '.join(words[i:k + 1])) + 1 < step:
            k += 1
        i = max(i + 1, k)
    return out


@dataclass
class _Piece:
    """What a stage hands the next one: text, and the parts it spans — `-1`
    once a separator has cut inside a part and the span is no longer known."""
    text: str
    start: int
    end: int


class _Document:
    """One document as the stages read it: its parts, how a run of them
    renders as text, and which of them a label boundary selects."""

    def __init__(self, document: dict, cfg, embedder, measure, normalizer):
        self.parts = document.get('document_content') or []
        self.join, self.prefix = cfg.part_join, cfg.part_prefix
        self.budget, self.overlap = cfg.chunk_chars, cfg.overlap
        self.embedder, self.measure, self.normalizer = embedder, measure, normalizer

    def render(self, start: int, end: int) -> _Piece:
        return _Piece(self.join.join(part_line(part, self.prefix)
                                     for part in self.parts[start:end + 1]),
                      start, end)

    def matches(self, i: int, atoms, join: str) -> bool:
        labels = self.parts[i].get('labels') or {}
        holds = [labels.get(atom['label']) == atom['value']
                 for atom in atoms if plan.is_label(atom)]
        return (all(holds) if join == 'and' else any(holds)) if holds else False

    def over_budget(self, piece: _Piece) -> bool:
        return self.measure(piece.text) > self.budget


def _runs(doc: _Document, piece: _Piece, opens) -> list[_Piece]:
    """Pieces from the parts of `piece`, a new one opening at every part
    `opens` says so for; the first part always opens one."""
    out, start = [], piece.start
    for i in range(piece.start + 1, piece.end + 1):
        if opens(i):
            out.append(doc.render(start, i - 1))
            start = i
    out.append(doc.render(start, piece.end))
    return out


def _by_part(doc: _Document, piece: _Piece, stage: dict) -> list[_Piece]:
    return [doc.render(i, i) for i in range(piece.start, piece.end + 1)]


def _by_label(doc: _Document, piece: _Piece, stage: dict) -> list[_Piece]:
    """A new piece opens at each part the boundary selects and runs to the
    part before the next — which is `turn-pair` on `role=user` without ever
    naming the other speaker, and works however many speakers there are."""
    return _runs(doc, piece, lambda i: doc.matches(i, stage['atoms'], stage['join']))


def _by_drift(doc: _Document, piece: _Piece, stage: dict) -> list[_Piece]:
    """Consecutive parts grouped into topical segments: cut where similarity
    drops into the bottom of *this piece's own* distribution (a relative
    threshold, since embedder scales are not comparable), where one of the
    configured markers occurs, or at a size ceiling of twice the budget."""
    parts = doc.parts[piece.start:piece.end + 1]
    if len(parts) <= 2:
        return [piece]
    texts = [p.get('text', '') for p in parts]
    vectors = doc.embedder.embed(texts)
    sims = np.array([float(vectors[i] @ vectors[i + 1])
                     for i in range(len(parts) - 1)], dtype=np.float32)
    cut_at = float(np.percentile(sims, 35)) if sims.size else -1.0
    markers = [doc.normalizer.normalize(m) for m in stage['markers']]
    cuts, size = set(), doc.measure(texts[0])
    for i in range(1, len(parts)):
        seen = doc.normalizer.normalize(texts[i])
        marker = any(m in seen for m in markers)
        too_big = size + doc.measure(texts[i]) > doc.budget * 2
        drifted = sims[i - 1] <= cut_at
        if marker or too_big or drifted:
            cuts.add(piece.start + i)
            size = doc.measure(texts[i])
        else:
            size += doc.measure(texts[i])
    return _runs(doc, piece, cuts.__contains__)


def _offsets(doc: _Document, piece: _Piece) -> list[tuple[int, int, int]]:
    """Where each part of a rendered piece sits in its text: (part, from, to)."""
    out, at = [], 0
    for i in range(piece.start, piece.end + 1):
        line = part_line(doc.parts[i], doc.prefix)
        out.append((i, at, at + len(line)))
        at += len(line) + len(doc.join)
    return out


def _literal_cuts(text: str, literals: list[str], join: str, lo: int, hi: int
                  ) -> list[tuple[int, int]]:
    """`(position, length)` of every cut the literals make in `text[lo:hi]`:
    under `or` wherever any occurs, under `and` only where every one begins
    at the same position, the longest consumed."""
    cuts = []
    for at in range(lo, hi):
        found = [lit for lit in literals if text.startswith(lit, at) and at + len(lit) <= hi]
        if not found or (join == 'and' and len(found) < len(literals)):
            continue
        cuts.append((at, len(max(found, key=len))))
    return cuts


def _by_separator(doc: _Document, piece: _Piece, stage: dict) -> list[_Piece]:
    """Cuts text. A literal cuts wherever it occurs (`or`) or where every
    literal in the stage holds at once (`and`); a label atom beside a literal
    narrows an `and` to the parts it selects and adds a cut at the boundary of
    each such part under `or`. The pieces that come out know no part span."""
    atoms, join = stage['atoms'], stage['join']
    literals = [a['text'] for a in atoms if plan.is_text(a)]
    selective = any(plan.is_label(a) for a in atoms)
    cuts: list[tuple[int, int]] = []
    if selective and piece.start >= 0:
        for i, lo, hi in _offsets(doc, piece):
            selected = doc.matches(i, atoms, join)
            if join == 'and' and selected:
                cuts.extend(_literal_cuts(piece.text, literals, join, lo, hi))
            elif join == 'or':
                cuts.extend(_literal_cuts(piece.text, literals, join, lo, hi))
                if selected and i > piece.start:
                    cuts.append((lo, 0))
    else:
        cuts = _literal_cuts(piece.text, literals, join, 0, len(piece.text))
    out, at = [], 0
    for position, length in sorted(set(cuts)):
        if position < at:
            continue
        out.append(piece.text[at:position])
        at = position + length
    out.append(piece.text[at:])
    return [_Piece(text.strip(), -1, -1) for text in out if text.strip()]


STAGES = {'part': _by_part, 'label': _by_label, 'drift': _by_drift,
          'separator': _by_separator}


def _apply(doc: _Document, stage: dict, pieces: list[_Piece]) -> list[_Piece]:
    out: list[_Piece] = []
    for piece in pieces:
        if stage['when'] == 'over-budget' and not doc.over_budget(piece):
            out.append(piece)
        elif plan.needs_parts(stage) and piece.start < 0:
            out.append(piece)        # validation refuses this plan; never cut blind
        else:
            out.extend(STAGES[stage['kind']](doc, piece, stage))
    return out


def _close(doc: _Document, pieces: list[_Piece]) -> list[_Piece]:
    """The budget: a piece still too big is divided at word boundaries, with
    the overlap repeated between neighbours when one is set. A division of one
    part still spans that part; a division of several no longer knows which."""
    out: list[_Piece] = []
    for piece in pieces:
        if not doc.over_budget(piece):
            out.append(piece)
            continue
        divided = (_windows(piece.text, doc.budget, doc.overlap, doc.measure)
                   if doc.overlap > 0 else
                   chunk_text(piece.text, doc.budget, doc.measure))
        start, end = ((piece.start, piece.end) if 0 <= piece.start == piece.end
                      else (-1, -1))
        out.extend(_Piece(text, start, end) for text in divided)
    return out


def chunk_document(document: dict, cfg, embedder, label_fields: dict | None = None,
                   language: str = '', normalizer=None, measure=None) -> list[Chunk]:
    """One document → chunks, by the plan: one piece to start, every stage
    after the document's applied in turn, the budget closing what is left."""
    label_fields = label_fields or {}
    base = _base(document, label_fields)
    prefix = contextual_prefix(document, label_fields, language) if cfg.contextual else ''
    doc = _Document(document, cfg, embedder,
                    measure or budget_measure(cfg, embedder),
                    normalizer or text_normalizers.resolve(cfg.normalizer, language))
    pieces = [doc.render(0, len(doc.parts) - 1)]
    for stage in plan.normalize(cfg.split_plan)[1:]:
        pieces = _apply(doc, stage, pieces)
    pieces = _close(doc, pieces)

    document_id = base['document_id']
    out: list[Chunk] = []
    for i, piece in enumerate(pieces):
        labels = dict(base['labels'])
        labels.update(_part_chunk_labels(document, label_fields, piece.start, piece.end))
        out.append(Chunk(
            id=f'{document_id}:c{i}', text=prefix + piece.text, prefix=prefix,
            document_id=document_id, date=base['date'],
            span_from=base['span_from'], span_to=base['span_to'],
            importance=base['importance'], labels=labels,
            part_start=piece.start, part_end=piece.end))
    return out
