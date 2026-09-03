"""The corpora this installation can measure against, and the archives brought
in from elsewhere.

Holds the catalogue and the declaration table a dataset carries — what labels
its corpus declares, what labels its questions carry, which of them is the date
and which the ranks — read straight off the loaded files rather than hardcoded.
`configuration.py` serves the same reading inside `/api/options`, so the card a
reader opens and the panel that renders itself cannot describe one corpus two
ways.

An imported archive is the other direction: a complete experiment from another
machine, kept verbatim and never re-run here.
"""
import sqlite3

from fastapi import HTTPException
from fastapi.responses import FileResponse

from raglab.agents import widget
from raglab.corpora import corpus_reading
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import experiment_archive as archive

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


def register(app, context) -> None:
    registry, dataset_lock, archives = (
        context.registry, context.dataset_lock, context.archives)

    @app.post('/api/imported-archives')
    def import_archive(payload: dict):
        try:
            return archives.import_archive(payload)
        except archive.ArchiveError as error:
            raise HTTPException(400, str(error)) from error
        except sqlite3.Error as error:
            raise HTTPException(500, 'archive database persistence failed') from error

    @app.get('/api/imported-archives/active')
    def active_archive():
        return archives.metadata()

    @app.delete('/api/imported-archives/active')
    def clear_active_archive():
        archives.clear()
        return {'archive_id': None}

    @app.get('/api/imported-archives/{archive_id}')
    def imported_archive(archive_id: str):
        found = archives.get(archive_id)
        if found is None:
            raise HTTPException(404, 'unknown imported archive')
        return found

    @app.get('/api/dataset-templates/corpus')
    def dataset_template_corpus():
        """The corpus template, read from the fixture every time — no copy of
        it lives in code, so the one author (the fixture file, pinned
        byte-equal to the schema by its own convention test) is also the one
        the import section's guidance link hands the browser."""
        return FileResponse(datasets.BUNDLED_DIR / 'corpus_template.json',
                            media_type='application/json',
                            filename='corpus_template.json')

    @app.get('/api/dataset-templates/groundtruth')
    def dataset_template_groundtruth():
        """The ground-truth template, on the same terms as the corpus one above."""
        return FileResponse(datasets.BUNDLED_DIR / 'groundtruth_template.json',
                            media_type='application/json',
                            filename='groundtruth_template.json')

    @app.get('/api/datasets')
    def list_datasets():
        return {'datasets': [found.as_dict() for found in datasets.catalogue()]}

    @app.post('/api/datasets')
    def import_dataset(payload: dict):
        """Take one dataset pair — the corpus file and its ground truth (D1) —
        check it against the contract, keep it. 400 with every problem at once."""
        corpus = payload.get('corpus') if isinstance(payload, dict) else None
        ground_truth = (payload.get('ground_truth')
                        if isinstance(payload, dict) else None)
        if not isinstance(corpus, dict) or not isinstance(ground_truth, dict):
            raise HTTPException(
                400, "a dataset import needs both files: {'corpus': …, "
                     "'ground_truth': …}")
        meta = corpus.get('corpus_dataset_metadata')
        raw_id = meta.get('dataset') if isinstance(meta, dict) else ''
        lock_id = raw_id if isinstance(raw_id, str) else ''
        with dataset_lock(lock_id):
            try:
                found = datasets.import_dataset(corpus, ground_truth)
            except ValueError as error:
                raise HTTPException(400, str(error))
            # Import writes the file and clears the loader cache first; eviction
            # is the final step under the same lock, so no later index lookup
            # can observe the new file through an old cached index.
            registry.invalidate_dataset(found.id)
        # The corpus set this installation knows has just changed, and the
        # widget caches the board's dataset ids for the life of the process —
        # it filters every turn's memory context against them and cannot pay
        # for a board reading per turn. Forgetting them here is what stops an
        # import needing a restart before the filter can see past it. This
        # route, not the store: the widget is a sealed leaf and `corpora/` may
        # not reach into it, while this module is the one that already does.
        widget.forget_board_dataset_ids()
        # The same declaration table a catalogue entry carries, so the panel
        # can show what it just read without a second round trip.
        return found.as_dict() | _dataset_declaration(found.id)
