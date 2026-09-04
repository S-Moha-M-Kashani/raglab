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
import re
import sqlite3

from fastapi import HTTPException
from fastapi.responses import FileResponse

from raglab.agents import widget
from raglab.corpora import corpus_reading
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import experiment_archive as archive

def question_vocab(ground_truth: dict) -> dict:
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
    the levels it may be attached at, whether a model extracted it, and the
    confidence rater that scores it, if any. Read straight off the file's own
    `label_fields`/`question_metadata_fields` (D4) — nothing here is guessed
    or filled in. `applies_to` is what the dataset viewer's columns are
    derived from: a part-level label describes parts whether or not any part
    has been given a value for it yet."""
    return [
        {'name': name, 'type': declaration.get('type', ''),
         'levels': declaration.get('values')
                   or list(declaration.get('glossary') or {}),
         'applies_to': declaration.get('applies_to') or [],
         'extracted': bool(declaration.get('extracted')),
         'confidence_for': declaration.get('confidence_for', '')}
        for name, declaration in sorted(fields.items())]


#: A line with nothing on it but whitespace, between two newlines. What a
#: separator stage cutting on `"\n\n"` needs in order to match anything at
#: all inside a part — a corpus whose parts are single lines can never offer
#: it (only `part_join` can put one between parts), and the reading below is
#: how a reader finds that out before sweeping it.
BLANK_LINE = re.compile(r'\n[^\S\n]*\n')


def readings(corpus: dict, ground_truth: dict) -> list[dict]:
    """The four statements about whether this pair can be measured together at
    all, each with the rows behind it named rather than merely counted — so a
    reported count can be turned into the rows it is about (D3).

    Derived from declared structure only, the way `_label_declaration` above
    is: nothing here guesses a field or fills in one a corpus lacks. A
    reading with nothing to report says zero rather than disappearing, which
    is what lets a reader tell a clean corpus from a check that did not run.
    `grid` names which of the page's tables the identifiers are rows of.
    """
    documents = corpus.get('corpus_documents') or []
    questions = ground_truth.get('groundtruth_dataset') or []
    by_id = corpus_reading.documents_by_id(corpus)

    cited: set = set()
    absent: list[tuple] = []
    for question in questions:
        for document_id in corpus_reading.evidence_documents(question):
            if document_id in by_id:
                cited.add(document_id)
            else:
                absent.append(
                    (question.get('groundtruth_question_id'), document_id))
    uncited = [document_id for document_id in by_id if document_id not in cited]

    # A label is populated by whatever row can carry it: a document through
    # `document_metadata`, a part through its own `labels`, a question through
    # `question_metadata`. Checked wherever a value could be rather than only
    # where `applies_to` says it should be — the claim is that no row anywhere
    # carries this label, and a value in the wrong place is still a value.
    populated: set = set()
    for document in documents:
        populated |= _named(document.get('document_metadata'))
        for part in document.get('document_content') or []:
            populated |= _named(part.get('labels'))
    for question in questions:
        populated |= _named(question.get('question_metadata'))
    declared = set((corpus.get('corpus_dataset_metadata') or {}
                    ).get('label_fields') or {}) | set(
        (ground_truth.get('groundtruth_dataset_metadata') or {}
         ).get('question_metadata_fields') or {})
    unpopulated = sorted(declared - populated)

    blank = [document_id for document_id, document in by_id.items()
             if any(BLANK_LINE.search(part.get('text') or '')
                    for part in document.get('document_content') or [])]

    return [
        {'id': 'uncited-documents', 'grid': 'documents',
         'says': 'documents no ground-truth question cites',
         'ids': uncited, 'count': len(uncited), 'detail': []},
        # The offending id is named, not just counted: which document a
        # question reached for is the whole of what a reader has to fix, and
        # it is a value the corpus does not contain, so no grid can show it.
        {'id': 'absent-citations', 'grid': 'questions',
         'says': 'questions citing a document this corpus does not have',
         'ids': [question_id for question_id, _ in absent],
         'count': len(absent),
         'detail': [f'question {question_id} cites document {document_id}'
                    for question_id, document_id in absent]},
        {'id': 'unpopulated-labels', 'grid': 'labels',
         'says': 'declared labels no row populates',
         'ids': unpopulated, 'count': len(unpopulated), 'detail': []},
        {'id': 'blank-line-documents', 'grid': 'documents',
         'says': 'documents whose part text holds a blank line',
         'ids': blank, 'count': len(blank), 'detail': []},
    ]


def _named(labels: dict | None) -> set:
    """The labels one row actually carries. An empty string, list or object is
    the label declared and not answered, which is not a row populating it."""
    return {name for name, value in (labels or {}).items()
            if value not in (None, '', [], {})}


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
    } | question_vocab(ground_truth)


def dataset_options() -> dict:
    return {
        'datasets': [found.as_dict() | _dataset_declaration(found.id)
                     for found in datasets.catalogue()],
    }


def register(app, context) -> dict:
    """Returns the imported-archive operations the Inspector reads through."""
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

    @app.get('/api/dataset-content/{dataset_id}')
    def dataset_content(dataset_id: str):
        """One whole dataset for the viewer at `/dataset`: the corpus and the
        ground truth exactly as the files hold them (D4), the declaration
        table the panel's own card reads, and the four readings about whether
        the pair can be measured together at all.

        Whole, with no paging: the largest bundled corpus is 269 KB, so a
        cursor and a scroll-driven fetch would be machinery bought for
        nothing — and everything the page sorts and filters is then already
        in the browser. The catalogue decides what may be asked for, so the
        surface cannot be pointed at a file by URL; a listed pair whose
        ground truth will not load is described rather than refused, the same
        fallback `_dataset_declaration` above already makes.
        """
        found = datasets.find(dataset_id)
        if found is None:
            raise HTTPException(
                404, f'unknown dataset {dataset_id!r} — known: '
                     + ', '.join(entry.id for entry in datasets.catalogue()))
        try:
            corpus, ground_truth = datasets.load(dataset_id)
            unmeasurable = ''
        except ValueError as error:
            corpus, ground_truth = datasets.load_corpus(dataset_id), {}
            unmeasurable = str(error)
        return {
            'dataset': found.as_dict() | _dataset_declaration(dataset_id),
            'corpus': corpus,
            'ground_truth': ground_truth,
            'ground_truth_error': unmeasurable,
            'readings': readings(corpus, ground_truth),
        }

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

    # The three of these the Inspector reads through, handed back so the panel
    # can build the seam out of the functions its own routes are. Named for
    # what they do rather than for the route that carries them, because the
    # Inspector asks for an operation and never for a path.
    return {'imported_archive': imported_archive,
            'active_archive': active_archive,
            'clear_active_archive': clear_active_archive}
