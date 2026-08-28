"""The corpora this lab can measure against, and the contract a new pair
meets (`validate()` here is the contract; `config.HELP['run.dataset-file']`
states it for the panel). `IndexConfig.dataset` lands in the fingerprint, so
an index built over one corpus can never be handed a question from another;
`''` means the built-in diary (`diary-fa`), and is also the leaderboard's
coarsest grouping key.

A dataset is two files, paired by id (D1): `<name>_corpus.json` and
`<name>_groundtruth.json`, joined by `corpus_dataset_metadata.dataset ==
groundtruth_dataset_metadata.corpus_ref.dataset` — never by filename, since
the bundled files' names (`diary_year_fa_corpus.json`) do not spell the id
they declare (`diary-fa`). A corpus with no matching ground truth is listed
and refused at run time ("nothing to measure against"); a ground truth with
no corpus is never listed at all, since every listing is reached by scanning
corpus files.

`schema_corpus.json`/`schema_groundtruth.json` are the single source of
structural truth (D9): `validate()` runs them with the `jsonschema` library
and adds only what a JSON Schema cannot express — the `x-consistency` and
`x-cross-file` rules the schemas themselves declare, since a label vocabulary
is data, not shape, and whether a document obeys its own table is not
knowable until the file is open.

`load()` returns the two file payloads exactly as they are (D4) — there is no
second, translated dialect. `document_content`, `text`, `labels`,
`document_metadata`, `label_fields`, `derived_facts`, `evidence`, `fidelity`,
`behavior`, `supports` are the names every consumer (chunker, harness,
metrics, panel) reads.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

from raglab.configuration.lab_config import ROOT
from raglab.corpora import corpus_store as corpora

# Shipped and read-only: a reference point that can be edited in place isn't one.
BUNDLED_DIR = ROOT / 'fixtures' / 'corpus_groundtruth_datasets'
# Imported through the panel. Git-ignored and machine-local, like `.runs/`.
IMPORTED_DIR = ROOT / '.datasets'

# `''`, not this name, is what a config carries by default, so every fingerprint
# and stored run keeps meaning what it meant.
BUILTIN = 'diary-fa'
# What a fresh panel starts on — the English rendering of the diary. A
# different constant than BUILTIN on purpose: BUILTIN is what an *absent*
# dataset id has always meant and must keep meaning; DEFAULT is merely the
# first offer, carried explicitly so it lands in the fingerprint like any
# other non-default knob.
DEFAULT = 'diary-en'

CORPUS_SUFFIX = '_corpus.json'
GROUNDTRUTH_SUFFIX = '_groundtruth.json'
# The schemas themselves live beside the data they describe and happen to
# share both files' suffixes (`schema_corpus.json`, `schema_groundtruth.json`)
# — named explicitly rather than matched, since a suffix match alone would
# read them as a nameless dataset.
SCHEMA_FILES = {'schema_corpus.json', 'schema_groundtruth.json'}

CORPUS_SCHEMA = json.loads(
    (BUNDLED_DIR / 'schema_corpus.json').read_text(encoding='utf-8'))
GROUNDTRUTH_SCHEMA = json.loads(
    (BUNDLED_DIR / 'schema_groundtruth.json').read_text(encoding='utf-8'))


@dataclass(frozen=True)
class Dataset:
    """One corpus with its ground truth, as the panel lists it."""
    id: str
    name: str
    description: str
    language: str
    source: str                  # bundled | imported
    path: str                    # repo-relative, for a reader who wants to look
    documents: int = 0
    parts: int = 0
    questions: int = 0
    query_date: str = ''
    period: dict = field(default_factory=dict)
    # The label names this corpus declares — what the panel renders one
    # switch-group or filter per (D7's successor here: the listing side of it).
    labels: list[str] = field(default_factory=list)
    # Which row of `databases/corpora.db` holds this corpus as it entered the
    # lab — assigned by the database on insert, never carried in the file. Set
    # by `import_dataset`, which is the one place a dataset enters; `0` on a
    # listing, which reads files and looks nothing up, and means "this listing
    # did not ask" rather than "there is no such row". A local storage id, so
    # it means nothing on another machine — the portable address is the
    # fingerprint, which is what an archive carries.
    id_corpora: int = 0

    def as_dict(self) -> dict:
        return {'id': self.id, 'name': self.name, 'description': self.description,
                'language': self.language, 'source': self.source, 'path': self.path,
                'documents': self.documents, 'parts': self.parts,
                'questions': self.questions, 'query_date': self.query_date,
                'period': self.period, 'labels': self.labels,
                'id_corpora': self.id_corpora}


def imported_dir() -> Path:
    """Where imports land. `RAGLAB_DATASETS` overrides, which is what lets the
    suite import into a temp directory without touching the real one."""
    override = (os.environ.get('RAGLAB_DATASETS') or '').strip()
    return Path(override) if override else IMPORTED_DIR


# --- reading ---------------------------------------------------------------

def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _dataset_id(corpus: dict) -> str:
    return (corpus.get('corpus_dataset_metadata') or {}).get('dataset', '')


def _files() -> list[tuple[str, str, Path, Path | None]]:
    """Every corpus this lab can be pointed at, paired with its ground truth
    by the id each file declares (D1) — not by filename — bundled then
    imported. An import with the id of a bundled sample shadows it: it is the
    copy the user made on purpose, and refusing the name would be refusing
    the edit. A corpus with no matching ground truth is still listed here,
    with `None` in its place — `load()` is where that becomes a refusal. A
    ground truth with no corpus is never reached at all, since this scans by
    corpus file, never by ground-truth file."""
    found: list[tuple[str, str, Path, Path | None]] = []
    order: dict[str, int] = {}
    for source, folder in (('bundled', BUNDLED_DIR), ('imported', imported_dir())):
        if not folder.exists():
            continue
        truths: dict[str, Path] = {}
        for path in sorted(folder.glob(f'*{GROUNDTRUTH_SUFFIX}')):
            if path.name in SCHEMA_FILES:
                continue
            try:
                payload = _read(path)
            except Exception:
                continue
            meta = payload.get('groundtruth_dataset_metadata') or {}
            dataset_id = (meta.get('corpus_ref') or {}).get('dataset', '')
            if dataset_id:
                truths[dataset_id] = path
        for path in sorted(folder.glob(f'*{CORPUS_SUFFIX}')):
            if path.name in SCHEMA_FILES:
                continue
            try:
                payload = _read(path)
            except Exception:
                continue
            dataset_id = _dataset_id(payload)
            if not dataset_id:
                continue
            entry = (source, dataset_id, path, truths.get(dataset_id))
            if dataset_id in order:
                found[order[dataset_id]] = entry
            else:
                order[dataset_id] = len(found)
                found.append(entry)
    return found


def _date_label(label_fields: dict) -> str:
    """The one label typed `date-time` at the document level (D5) — what
    `period` is read off. A corpus that declares none has no time behaviour,
    and this returns ''."""
    for name, definition in label_fields.items():
        if (definition.get('format') == 'date-time'
                and 'document' in (definition.get('applies_to') or [])):
            return name
    return ''


def describe(corpus: dict, ground_truth: dict | None, source: str, path: Path,
             id_corpora: int = 0) -> Dataset:
    """One dataset pair as the panel lists it. `ground_truth` is `None` for a
    corpus with nothing to measure against (D1) — still listed, with zero
    questions. `id_corpora` is passed only by the import that just stored the
    corpus and got a row id back; a listing leaves it 0, because a listing
    reads files and asks the corpus store nothing."""
    meta = corpus.get('corpus_dataset_metadata') or {}
    documents = corpus.get('corpus_documents') or []
    label_fields = meta.get('label_fields') or {}
    date_label = _date_label(label_fields)
    dates = sorted(
        value for value in (
            (document.get('document_metadata') or {}).get(date_label)
            for document in documents)
        if value) if date_label else []
    gt_meta = (ground_truth or {}).get('groundtruth_dataset_metadata') or {}
    questions = (ground_truth or {}).get('groundtruth_dataset') or []
    try:
        shown = str(path.relative_to(ROOT))
    except ValueError:
        shown = str(path)
    return Dataset(
        id=meta.get('dataset', path.stem), name=meta.get('name', path.stem),
        description=meta.get('description', ''),
        language=meta.get('language', ''), source=source, path=shown,
        documents=len(documents),
        parts=sum(len(document.get('document_content') or [])
                  for document in documents),
        questions=len(questions),
        query_date=gt_meta.get('default_question_asked_at', ''),
        period={'from': dates[0], 'to': dates[-1]} if dates else {},
        labels=sorted(label_fields),
        id_corpora=int(id_corpora))


def catalogue() -> list[Dataset]:
    """Every dataset this lab can be pointed at. A file that will not parse is
    skipped, never fatal."""
    out: list[Dataset] = []
    for source, dataset_id, corpus_path, truth_path in _files():
        try:
            corpus = _read(corpus_path)
        except Exception:
            continue
        ground_truth = None
        if truth_path is not None:
            try:
                ground_truth = _read(truth_path)
            except Exception:
                ground_truth = None
        out.append(describe(corpus, ground_truth, source, corpus_path))
    return out


def find(dataset_id: str) -> Dataset | None:
    wanted = dataset_id or BUILTIN
    return next((d for d in catalogue() if d.id == wanted), None)


_CACHE: dict[str, tuple[dict, dict]] = {}


def load(dataset_id: str = '') -> tuple[dict, dict]:
    """`(corpus, ground_truth)` for one dataset, in the shape the schema
    declares (D4) — no translation layer. Cached per id, read once per
    process; an import calls `forget()`."""
    wanted = dataset_id or BUILTIN
    if wanted in _CACHE:
        return _CACHE[wanted]
    for source, found_id, corpus_path, truth_path in _files():
        if found_id != wanted:
            continue
        if truth_path is None:
            raise ValueError(
                f'{wanted!r} has no ground truth — nothing to measure '
                f'against. Add a {wanted}{GROUNDTRUTH_SUFFIX} whose '
                'corpus_ref.dataset names it.')
        _CACHE[wanted] = (_read(corpus_path), _read(truth_path))
        return _CACHE[wanted]
    raise ValueError(
        f'unknown dataset {wanted!r} — known: '
        + ', '.join(d.id for d in catalogue()))


def load_corpus(dataset_id: str = '') -> dict:
    """Just the corpus, for a reader that describes rather than scores. A pair
    with no ground truth is listed (D1) and refused only at run time, so the
    panel's dataset card has to be able to read the corpus of a dataset
    `load()` would refuse — otherwise one unmeasurable corpus in the folder
    takes the whole catalogue down with it. Everything that scores still goes
    through `load()`, which is the one door that insists on both files."""
    wanted = dataset_id or BUILTIN
    if wanted in _CACHE:
        return _CACHE[wanted][0]
    for _source, found_id, corpus_path, _truth_path in _files():
        if found_id == wanted:
            return _read(corpus_path)
    raise ValueError(
        f'unknown dataset {wanted!r} — known: '
        + ', '.join(d.id for d in catalogue()))


def forget() -> None:
    """Drop the corpus cache, so a dataset replaced under the same id is the
    one the next build reads."""
    _CACHE.clear()


# --- the contract ------------------------------------------------------------

def validate(corpus: dict, ground_truth: dict) -> list[str]:
    """Everything wrong with a corpus/ground-truth pair, in the order a reader
    would fix it — a list rather than an exception, and *all* of the problems
    rather than the first. D9's composition: the two schemas run first, since
    they are the structural truth; if either has anything to say, the checks
    below (which assume a structurally sound pair) do not run at all, and only
    once both are clean do the `x-consistency`/`x-cross-file` rules the
    schemas declare get checked in Python — the only place a dynamic label
    vocabulary can be checked, since its shape is data, not schema."""
    problems = (_schema_problems(corpus, CORPUS_SCHEMA, 'corpus')
               + _schema_problems(ground_truth, GROUNDTRUTH_SCHEMA, 'ground truth'))
    if problems:
        return problems
    problems.extend(_consistency_problems(corpus))
    problems.extend(_groundtruth_consistency_problems(ground_truth))
    problems.extend(_cross_file_problems(corpus, ground_truth))
    return problems


def _schema_problems(payload: dict, schema: dict, label: str) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload),
                    key=lambda error: [str(part) for part in error.absolute_path])
    return [f'{".".join(str(part) for part in error.absolute_path) or label}: '
            f'{error.message}' for error in errors]


def _consistency_problems(corpus: dict) -> list[str]:
    """`schema_corpus.json`'s `x-consistency`, the parts a JSON Schema cannot
    check because the vocabulary table is itself data: every used label is
    declared at the level it is used, every extracted label has exactly one
    rater and no rater rates an unextracted label, a confidence's own shape
    matches what it rates, and `ranks` is legal only where declared."""
    problems: list[str] = []
    dataset_id = _dataset_id(corpus)
    meta = corpus.get('corpus_dataset_metadata') or {}
    fields = meta.get('label_fields') or {}
    documents = corpus.get('corpus_documents') or []

    used_at: dict[str, set[str]] = {}
    for document in documents:
        for key in (document.get('document_metadata') or {}):
            used_at.setdefault(key, set()).add('document')
        for part in document.get('document_content') or []:
            for key in (part.get('labels') or {}):
                used_at.setdefault(key, set()).add('part')

    for key, levels in used_at.items():
        if key not in fields:
            problems.append(
                f'{dataset_id}: label {key!r} is used but never declared in '
                'label_fields')
            continue
        declared = set(fields[key].get('applies_to') or [])
        stray = levels - declared
        if stray:
            problems.append(
                f'{dataset_id}: label {key!r} is used at '
                f'{", ".join(sorted(stray))}, which its declared applies_to '
                f'{sorted(declared)} does not name')

    raters: dict[str, list[str]] = {}
    for name, definition in fields.items():
        rated = definition.get('confidence_for')
        if rated:
            raters.setdefault(rated, []).append(name)

    for name, definition in fields.items():
        if definition.get('extracted'):
            names = raters.get(name, [])
            if len(names) != 1:
                problems.append(
                    f'{dataset_id}: extracted label {name!r} needs exactly '
                    f'one label declaring confidence_for {name!r}, found '
                    f'{len(names)}')
        rated = definition.get('confidence_for')
        if rated and not (fields.get(rated) or {}).get('extracted'):
            problems.append(
                f'{dataset_id}: {name!r} declares confidence_for {rated!r}, '
                'which is not declared extracted — a rater rates only what '
                'was extracted')
        if definition.get('ranks'):
            if definition.get('type') not in ('number', 'integer'):
                problems.append(
                    f'{dataset_id}: {name!r} declares ranks but type '
                    f'{definition.get("type")!r} is not number or integer')
            if 'minimum' not in definition or 'maximum' not in definition:
                problems.append(
                    f'{dataset_id}: {name!r} declares ranks but is missing '
                    'minimum or maximum')
            if rated:
                problems.append(
                    f'{dataset_id}: {name!r} declares both ranks and '
                    'confidence_for — a rater ranks nothing')

    for document in documents:
        doc_meta = document.get('document_metadata') or {}
        where = f'{dataset_id}#{document.get("corpus_document_id")}'
        for rated, names in raters.items():
            rated_type = (fields.get(rated) or {}).get('type')
            for name in names:
                if name not in doc_meta:
                    continue
                value = doc_meta[name]
                if rated_type == 'array':
                    if not isinstance(value, dict):
                        problems.append(
                            f'{where}: confidence {name!r} for list label '
                            f'{rated!r} must be an object keyed by its '
                            'values, never a second list')
                    else:
                        carried = set(doc_meta.get(rated) or [])
                        stray_keys = set(value) - carried
                        if stray_keys:
                            problems.append(
                                f'{where}: confidence {name!r} rates '
                                f'{sorted(stray_keys)}, which {rated!r} does '
                                'not carry on this document')
                elif not isinstance(value, (int, float)) or isinstance(value, bool):
                    problems.append(
                        f'{where}: confidence {name!r} must be a single '
                        'number')
    return problems


def _groundtruth_consistency_problems(ground_truth: dict) -> list[str]:
    """The same declared-before-used rule as a ground truth's own
    vocabulary tables (`question_metadata_fields`, `evidence_role_values`) —
    a JSON Schema cannot check a dynamic dict either."""
    problems: list[str] = []
    meta = ground_truth.get('groundtruth_dataset_metadata') or {}
    fields = meta.get('question_metadata_fields') or {}
    roles = meta.get('evidence_role_values') or {}
    for question in ground_truth.get('groundtruth_dataset') or []:
        where = f'question {question.get("groundtruth_question_id")}'
        for key in (question.get('question_metadata') or {}):
            if key not in fields:
                problems.append(
                    f'{where}: question_metadata {key!r} is used but never '
                    'declared in question_metadata_fields')
        for relevant in question.get('relevant_corpus_documents') or []:
            for evidence in relevant.get('evidence') or []:
                role = evidence.get('role')
                if role and role not in roles:
                    problems.append(
                        f'{where}: evidence role {role!r} is used but never '
                        'declared in evidence_role_values')
    return problems


def _closed_values(definition: dict) -> set | None:
    """The set of values a label declaration closes itself to, read off
    whichever of the two mechanisms names it — `glossary`'s keys or a bare
    `values` list — or `None` when neither is declared, meaning any value of
    the field's type is open. The representation a file chose is not part of
    the label's meaning; the set it closes on is."""
    if 'glossary' in definition:
        return set(definition['glossary'])
    if 'values' in definition:
        return set(definition['values'])
    return None


def _label_meaning_mismatches(corpus_def: dict, groundtruth_def: dict) -> list[str]:
    """x-cross-file's tenth rule, for one label name both files declare. The
    honest, checkable slice of "meaning" a shared name promises is its
    `type`, its closed set of allowed values however either file spells it,
    and its own prose `description` — never `applies_to`, `extracted` or
    `nullable`, which legitimately differ per file since a ground truth
    reaches only the `question` level and a corpus never does. When both
    sides also carry a `glossary`, the value set agreeing is not enough on
    its own: the schema's own words are "each file declares its own
    glossary so it can be read alone", so two glossaries naming the same
    values with different prose still disagree about what those values
    mean."""
    mismatches: list[str] = []
    if corpus_def.get('type') != groundtruth_def.get('type'):
        mismatches.append(
            f'type ({corpus_def.get("type")!r} in the corpus vs '
            f'{groundtruth_def.get("type")!r} in the ground truth)')
    if corpus_def.get('description') != groundtruth_def.get('description'):
        mismatches.append('description differs between the two files')
    corpus_values = _closed_values(corpus_def)
    groundtruth_values = _closed_values(groundtruth_def)
    if corpus_values != groundtruth_values:
        mismatches.append(
            'allowed values '
            f'({sorted(corpus_values) if corpus_values is not None else "open"} '
            'in the corpus vs '
            f'{sorted(groundtruth_values) if groundtruth_values is not None else "open"} '
            'in the ground truth)')
    elif (corpus_def.get('glossary') is not None
            and groundtruth_def.get('glossary') is not None
            and corpus_def['glossary'] != groundtruth_def['glossary']):
        mismatches.append(
            'glossary text differs between the two files even though the '
            'values agree')
    return mismatches


def _cross_file_problems(corpus: dict, ground_truth: dict) -> list[str]:
    """`schema_groundtruth.json`'s `x-cross-file`: the pair join, every cited
    document exists, every verbatim quote is findable, every `supports` id is
    valid and every derived fact is covered by one, a computed piece names its
    source label, a copied `document_metadata`/`relevant_metadata` is real,
    and a label name declared in both files agrees on its meaning."""
    problems: list[str] = []
    corpus_id = _dataset_id(corpus)
    gt_meta = ground_truth.get('groundtruth_dataset_metadata') or {}
    ref = gt_meta.get('corpus_ref') or {}
    if ref.get('dataset') != corpus_id:
        problems.append(
            f'{ref.get("dataset")!r}: corpus_ref.dataset does not match the '
            f'corpus it is paired with ({corpus_id!r}) — the pair join is '
            'broken')

    corpus_fields = (corpus.get('corpus_dataset_metadata') or {}).get(
        'label_fields') or {}
    groundtruth_fields = gt_meta.get('question_metadata_fields') or {}
    for name in sorted(set(corpus_fields) & set(groundtruth_fields)):
        mismatches = _label_meaning_mismatches(
            corpus_fields[name], groundtruth_fields[name])
        if mismatches:
            problems.append(
                f'label {name!r} is declared in both files but does not '
                f'carry the same meaning in both: {"; ".join(mismatches)}')

    documents = {document['corpus_document_id']: document
                for document in corpus.get('corpus_documents') or []
                if 'corpus_document_id' in document}

    for question in ground_truth.get('groundtruth_dataset') or []:
        where = f'question {question.get("groundtruth_question_id")}'
        expected = question.get('expected_answer') or {}
        derived_fact_list = expected.get('derived_facts') or []
        derived_facts = {fact['derived_fact_id'] for fact in derived_fact_list
                         if 'derived_fact_id' in fact}
        covered: set = set()
        cited_documents: list[dict] = []
        for relevant in question.get('relevant_corpus_documents') or []:
            document_id = relevant.get('corpus_document_id')
            document = documents.get(document_id)
            if document is None:
                problems.append(
                    f'{where}: cites corpus_document_id {document_id}, which '
                    'is not in this corpus')
                continue
            cited_documents.append(document)
            doc_text = ' \n'.join(part.get('text', '')
                                  for part in document.get('document_content') or [])
            for evidence in relevant.get('evidence') or []:
                supports = evidence.get('supports') or []
                bad = [s for s in supports if s not in derived_facts]
                if bad:
                    problems.append(
                        f'{where}: evidence supports {bad}, which are not '
                        'derived_fact_ids of this question')
                covered.update(s for s in supports if s in derived_facts)
                if not supports:
                    # "supports the answer as a whole", per the schema.
                    covered.update(derived_facts)

                fidelity = evidence.get('fidelity')
                text = evidence.get('text', '')
                if fidelity == 'verbatim' and text not in doc_text:
                    problems.append(
                        f'{where}: evidence quotes text that is not in '
                        f'document {document_id} — a verbatim quote must be '
                        'findable character for character, or every lexical '
                        'score in this lab is measured against something '
                        'the corpus never said')
                if fidelity == 'computed' and not evidence.get('relevant_metadata'):
                    problems.append(
                        f'{where}: a computed piece of evidence must carry '
                        'relevant_metadata naming the label it was computed '
                        'from')
                part_labels = evidence.get('part_labels')
                if fidelity == 'verbatim' and not part_labels:
                    problems.append(
                        f'{where}: verbatim evidence needs part_labels for '
                        'the parts its text lies in')
                if fidelity == 'computed' and part_labels:
                    problems.append(
                        f'{where}: computed evidence lies in no part, so '
                        'part_labels must be empty')

                copy = evidence.get('document_metadata')
                actual = document.get('document_metadata') or {}
                if copy is not None and copy != actual:
                    problems.append(
                        f'{where}: evidence.document_metadata does not match '
                        f'what document {document_id} actually carries')

                for key, value in (evidence.get('relevant_metadata') or {}).items():
                    problems.extend(
                        _relevant_metadata_problems(where, document, key, value))

            for key, value in (relevant.get('relevant_metadata') or {}).items():
                problems.extend(
                    _relevant_metadata_problems(where, document, key, value))

        for fact in derived_fact_list:
            problems.extend(
                _derived_fact_metadata_problems(where, cited_documents, fact))

        if derived_facts and expected.get('behavior') != 'abstain':
            missing = derived_facts - covered
            if missing:
                problems.append(
                    f'{where}: derived_fact_id {sorted(missing)} is not '
                    'named by any evidence — a claim no evidence carries is '
                    'a claim the ground truth cannot back')
    return problems


def _metadata_matches(document: dict, key: str, value) -> bool:
    """Whether `document`'s own `document_metadata` actually carries `value`
    under `key` — a scalar must match exactly, a list-valued label must
    contain every value named (a bare, non-list value counts as one to look
    for). Absent entirely (`key` not on the document at all) is never a
    match."""
    actual = (document.get('document_metadata') or {}).get(key)
    if actual is None:
        return False
    if isinstance(actual, list):
        wanted = value if isinstance(value, list) else [value]
        return all(v in actual for v in wanted)
    return actual == value


def _relevant_metadata_problems(where: str, document: dict, key: str,
                                value) -> list[str]:
    """x-cross-file #6 on a document or a piece of evidence, both already
    tied to exactly one document: the value named has to be one that
    document actually carries."""
    actual = (document.get('document_metadata') or {}).get(key)
    document_id = document.get('corpus_document_id')
    if actual is None:
        return [f'{where}: relevant_metadata names {key!r}, which document '
                f'{document_id} does not carry']
    if not _metadata_matches(document, key, value):
        if isinstance(actual, list):
            wanted = value if isinstance(value, list) else [value]
            missing = [v for v in wanted if v not in actual]
            return [f'{where}: relevant_metadata {key!r}={missing} is not '
                    f'among what document {document_id} actually carries']
        return [f'{where}: relevant_metadata {key!r}={value!r} does not '
                f'match what document {document_id} actually carries '
                f'({actual!r})']
    return []


def _derived_fact_metadata_problems(where: str, documents: list[dict],
                                    fact: dict) -> list[str]:
    """x-cross-file #6 on a derived fact — the same rule, but a derived fact
    is not tied to one document the way a piece of evidence is (it names a
    claim, not a citation), so a value counts as real if *any* of the
    question's own `relevant_corpus_documents` actually carries it."""
    problems: list[str] = []
    fact_id = fact.get('derived_fact_id')
    for key, value in (fact.get('relevant_metadata') or {}).items():
        if not any(_metadata_matches(document, key, value)
                  for document in documents):
            problems.append(
                f'{where}: derived_fact {fact_id} relevant_metadata '
                f'{key!r}={value!r} is not carried by any document this '
                'question cites')
    return problems


def import_dataset(corpus: dict, ground_truth: dict) -> Dataset:
    """Validate and keep one dataset pair: the two objects in the corpus
    store, both files in the imported directory. Refuses rather than repairs
    — a silently repaired dataset measures something nobody described.

    A dataset entering the lab is stored as content, once, the moment it
    arrives: the two objects stored are exactly the two file payloads the
    pipeline reads, so the row written here is the row every experiment on
    this corpus will later reference, rather than one written again per
    archive. Neither file carries an id and cannot — the id is storage
    identity, assigned by the database on insert, and a file that named one
    would be claiming a row it knows nothing about.

    Content-addressed, so re-importing the same bytes is the same row and the
    same id, and an *edited* dataset under the same id is a new row beside the
    old one rather than an overwrite of it. That is what keeps an archive of an
    earlier run resolving to the corpus that run actually saw.

    The corpus goes in before the files do, the order
    `experiment_archive_store.shrink` gives: a failure between the two leaves a
    stored corpus nobody reads yet, where the other order would leave a
    dataset this lab can build over but the store has never seen.
    """
    problems = validate(corpus, ground_truth)
    if problems:
        raise ValueError('; '.join(problems))
    dataset_id = corpus['corpus_dataset_metadata']['dataset']
    if dataset_id == BUILTIN:
        raise ValueError(
            f'{BUILTIN!r} is the built-in corpus and cannot be replaced — every '
            'run already recorded was measured against it. Give this one its '
            'own id.')
    id_corpora = corpora.put(dataset_id, corpus, ground_truth)
    folder = imported_dir()
    folder.mkdir(parents=True, exist_ok=True)
    corpus_path = folder / f'{dataset_id}{CORPUS_SUFFIX}'
    truth_path = folder / f'{dataset_id}{GROUNDTRUTH_SUFFIX}'
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    truth_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=1),
                          encoding='utf-8')
    forget()
    return describe(corpus, ground_truth, 'imported', corpus_path, id_corpora)
