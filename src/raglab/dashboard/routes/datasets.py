"""The corpora this installation can measure against.

Holds the catalogue and the declaration table a dataset carries — what labels
its corpus declares, what labels its questions carry, which of them is the date
and which the ranks — read straight off the loaded files rather than hardcoded.
`configuration.py` serves the same reading inside `/api/options`, so the card a
reader opens and the panel that renders itself cannot describe one corpus two
ways.
"""
from raglab.corpora import corpus_reading
from raglab.corpora import dataset_import_contract as datasets

def _question_vocab(ground_truth: dict) -> dict:
    """One switch-group per question label the loaded ground truth declares
    with a closed set of values or a glossary (D7) — data-driven, since the
    labels are the dataset's own, not a vocabulary every corpus must share.
    `balance` may name any of them, or '' for a plain stride."""
    fields = (ground_truth.get('groundtruth_dataset_metadata') or {}
             ).get('question_metadata_fields') or {}
    labels = {name: (declaration.get('values')
                     or list((declaration.get('glossary') or {})))
             for name, declaration in fields.items()
             if declaration.get('values') or declaration.get('glossary')}
    return {
        'question_labels': labels,
        # The sample is part of the measurement: two rows on different
        # samples are not two results of the same one.
        'balances': [''] + sorted(labels),
    }


def _label_declaration(fields: dict) -> list[dict]:
    """One row per declared label — name, type, its closed set of levels (a
    plain `values` list or a `glossary`'s own keys; empty for an open field),
    whether a model extracted it, and the confidence rater that scores it, if
    any. Read straight off the file's own `label_fields`/
    `question_metadata_fields` (D4) — nothing here is guessed or filled in."""
    return [
        {'name': name, 'type': declaration.get('type', ''),
         'levels': declaration.get('values')
                   or list(declaration.get('glossary') or {}),
         'extracted': bool(declaration.get('extracted')),
         'confidence_for': declaration.get('confidence_for', '')}
        for name, declaration in sorted(fields.items())]


def _dataset_declaration(dataset_id: str) -> dict:
    """Everything the dataset card and the run's label filters read about one
    dataset, straight off its loaded files (D4) — never hardcoded, so a
    corpus with no date or ranks label just shows fewer rows and three
    greyed-out knobs rather than a placeholder for what it lacks. The same
    reading an import shows on success and a catalogue entry shows on
    selection, because both are 'this is what the lab read' and must not
    disagree."""
    try:
        corpus, ground_truth = datasets.load(dataset_id)
    except ValueError:
        # Listed but unmeasurable (D1): describe the corpus it does have and
        # declare no question labels, rather than refusing to describe the
        # catalogue at all. The refusal belongs on the run, not on the card.
        corpus, ground_truth = datasets.load_corpus(dataset_id), {}
    label_fields = (corpus.get('corpus_dataset_metadata') or {}
                    ).get('label_fields') or {}
    question_fields = (ground_truth.get('groundtruth_dataset_metadata') or {}
                       ).get('question_metadata_fields') or {}
    return {
        'label_declarations': _label_declaration(label_fields),
        'question_label_declarations': _label_declaration(question_fields),
        # '' means the corpus declares no such label (D5/D6) — the source
        # `knob_dependencies.DEPENDENCIES` reads to grey the three time knobs
        # and the agentic importance weight.
        'date_label': corpus_reading.date_label(label_fields),
        'ranks_label': corpus_reading.ranks_label(label_fields),
    } | _question_vocab(ground_truth)


def _dataset_options() -> dict:
    return {
        'datasets': [found.as_dict() | _dataset_declaration(found.id)
                     for found in datasets.catalogue()],
    }
