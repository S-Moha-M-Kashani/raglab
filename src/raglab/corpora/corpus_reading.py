"""Small, pure readers over one already-loaded corpus/ground-truth pair, in
the schema's own vocabulary (D4 — the file's vocabulary is the lab's
vocabulary, so there is no second, translated dialect). This module does no
file I/O of its own: `dataset_import_contract.load()` is where the two JSON
files are read; everything here just answers a question about the dicts it
returns — which document a `corpus_document_id` names, what one document
reads as plain text, which documents and evidence a question cites.
"""


def date_int(date: str) -> int:
    """'2026-03-10' -> 20260310, so time filters can compare numbers instead of date strings."""
    return int(date.replace('-', ''))


def part_line(part: dict, prefix: str = '') -> str:
    """One part as it reads in plain text: the part's own text, exactly as the
    corpus recorded it. `prefix` names a declared part-level label whose
    value is written in front (`user: …`) — asked for by name, never assumed,
    since the corpus template's own rule is that a part's text carries no
    speaker prefix and nothing an embedder should not read as content."""
    text = part.get('text', '')
    value = (part.get('labels') or {}).get(prefix) if prefix else None
    return f'{value}: {text}' if value not in (None, '') else text


def document_text(document: dict, join: str = '\n', prefix: str = '') -> str:
    """One document as plain text, its parts joined by `join` — a bare newline
    unless a corpus of sections rather than turns asks for a blank line, which
    is what lets a paragraph separator match between two parts at all."""
    return join.join(part_line(part, prefix)
                     for part in document.get('document_content') or [])


def date_label(label_fields: dict) -> str:
    """The one label typed `date-time` at the document level (D5) — what a
    build reads `date`/`span_from`/`span_to` off. `''` when the corpus
    declares none: that corpus has no time behaviour at all."""
    for name, definition in (label_fields or {}).items():
        if (definition.get('format') == 'date-time'
                and 'document' in (definition.get('applies_to') or [])):
            return name
    return ''


def ranks_label(label_fields: dict) -> str:
    """The one label declaring `ranks: true` (D6) — the importance source.
    `''` when the corpus declares none: importance is 0.0 for every chunk."""
    for name, definition in (label_fields or {}).items():
        if definition.get('ranks'):
            return name
    return ''


def documents_by_id(corpus: dict) -> dict[int, dict]:
    """Every document in a corpus, keyed by its `corpus_document_id` — the
    only id a ground truth's `relevant_corpus_documents` cites."""
    return {document['corpus_document_id']: document
            for document in corpus.get('corpus_documents') or []}


def evidence_texts(documents: dict[int, dict], question: dict) -> list[str]:
    """Full text of every evidence entry a question's `relevant_corpus_documents`
    names, as `reference_contexts` for RAGAS's offline context metrics
    (quote-level precision is `metrics.quote_recall` instead). `documents` is
    unused now that every piece of evidence carries its own `text` rather
    than an index into one — kept so a caller already holding
    `documents_by_id`'s result does not have to change its call."""
    out: list[str] = []
    for relevant in question.get('relevant_corpus_documents') or []:
        for evidence in relevant.get('evidence') or []:
            text = (evidence.get('text') or '').strip()
            if text:
                out.append(text)
    return out


def evidence_documents(question: dict) -> list[int]:
    """Distinct evidence document ids, in the order the ground truth lists them."""
    seen: list[int] = []
    for relevant in question.get('relevant_corpus_documents') or []:
        document_id = relevant.get('corpus_document_id')
        if document_id is not None and document_id not in seen:
            seen.append(document_id)
    return seen
