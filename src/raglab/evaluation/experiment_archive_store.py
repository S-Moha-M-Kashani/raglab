"""Archives at rest: one complete experiment archive per row, corpus by reference.

The board's open button and the archive export must hand over the *same* object,
so what this stores is the archive itself rather than a handful of columns
transcribed out of it. Columns still exist — the board sorts on them, and
sorting 226 rows by parsing 226 JSON blobs is not a table — but every one of
them is *projected* from the archive by `projection()` on the way in. A column
can therefore be stale only if the archive is, which is the same thing as the
row being wrong.

Two things are deliberately not stored:

`inspector.dataset.corpus` and its `ground_truth` are replaced by a reference.
Every experiment on one corpus embeds a byte-identical copy of it, and the
bundled diary is 690 KB — 207 diary rows would be 145 MB of the same text. The
text itself lives in the content-addressed corpus store
(`corpora/corpus_store.py`), once per distinct version, and `serve()` splices it
back in, so the object a reader receives is byte-identical to the file export
writes; only the encoding at rest differs.

That splice is exactly where a row could start lying about what produced it. A
dataset replaced under the same id would hand a *plausible* corpus to an
experiment that never ran on it — the failure this repo's first rule exists to
prevent. So the reference is written twice, in two currencies. `id_corpora` on
the row is the local foreign key into `databases/corpora.db`, and the archive's
own `corpus_ref = {dataset, fingerprint}` is the content address, which is what
an exported file can still be resolved and checked by on a machine that has
never seen this store's ids. Serving fetches by the id and then verifies the
fingerprint: an id says which row, only content can say what is in it. This used to re-read the tracked fixture file at
serve time, which meant an ordinary edit to a corpus permanently broke every
archive naming it. A version the store does not have is still a refusal —
absence is answered by saying so, never by serving a corpus that merely fits.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from raglab.corpora import corpus_store as corpora
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import experiment_archive as archive
from raglab.evaluation import service_experiment_ledger as ledger


CORPUS_REF = 'corpus_ref'

# `archives`, not `experiments`: the ledger owns a table of that name with a
# different shape, and `CREATE TABLE IF NOT EXISTS` would let whichever module
# opened the file first silently define it for the other. The two coexist until
# the board is rebuilt from archives, at which point the ledger's table is what
# gets dropped — and a rename now is cheaper than a migration that has to guess
# which schema it is looking at.
SCHEMA = """
CREATE TABLE IF NOT EXISTS archives (
  -- Storage identity, assigned by the database, the way `corpora.id` is: the
  -- two tables that hold an experiment's record and the text it ran on are
  -- keyed the same way, so neither has to explain itself differently from the
  -- other. It says which row and nothing about what is in it, and it means
  -- nothing on another machine.
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  -- What the row is *about*, and still the only address any reader uses: every
  -- lookup here is by experiment id. Unique rather than the key, so an
  -- experiment is stored once and re-storing it edits that row in place rather
  -- than replacing it with a new one under a new id.
  experiment_id   TEXT NOT NULL UNIQUE,
  -- `id_corpora` is the one column here that is NOT projected from the
  -- archive, and the exception is worth stating rather than discovering. Every
  -- other column is a reading of the record and may never be an independent
  -- source; this one cannot be a reading of it, because it is a local storage
  -- id — which row of `databases/corpora.db` holds the text — and a local id
  -- is meaningless in a file exported to another machine. So the archive keeps
  -- its own portable counterpart *inside* the JSON: `corpus_ref = {dataset,
  -- fingerprint}`, the content address, which is what makes an exported
  -- archive self-describing anywhere. Both, deliberately: the id joins here,
  -- the fingerprint travels and verifies.
  --
  -- Named for the table it points at (`id_<table>`), and spelled with an
  -- underscore rather than a hyphen because SQL reads `id-corpora` as `id`
  -- minus `corpora` unless every single use site quotes it — and every other
  -- column in this repo is snake_case (`experiment_id`, `decision_stderr`).
  --
  -- It is a foreign key in intent only — the corpora live in a separate
  -- database file and SQLite enforces no foreign key across files, not even
  -- through ATTACH. The relationship is held in code, in two halves: the
  -- corpus row is written and its id obtained before the archive that
  -- references it (`shrink`, called by `put` before the INSERT), so a failure
  -- can only orphan a corpus and never dangle a reference; and `swell`/`serve`
  -- re-hash what the id fetched and refuse unless it matches the fingerprint
  -- the archive recorded, so a reused id or a row edited in place is a refusal
  -- rather than a substitution.
  id_corpora      INTEGER NOT NULL DEFAULT 0,
  -- Every column below is projected from `archive` by `projection()`; none is
  -- written independently, so none can disagree with the record it summarises.
  dataset         TEXT NOT NULL DEFAULT '',
  label           TEXT NOT NULL DEFAULT '',
  started_at      TEXT NOT NULL DEFAULT '',
  seconds         REAL NOT NULL DEFAULT 0,
  provider        TEXT NOT NULL DEFAULT '',
  chunker         TEXT NOT NULL DEFAULT '',
  embedder        TEXT NOT NULL DEFAULT '',
  retriever       TEXT NOT NULL DEFAULT '',
  reranker        TEXT NOT NULL DEFAULT '',
  grader          TEXT NOT NULL DEFAULT '',
  answerer        TEXT NOT NULL DEFAULT '',
  n_questions     INTEGER NOT NULL DEFAULT 0,
  -- NULL, never 0.0, on anything that judged nothing: a fabricated zero sorts
  -- below real rows and reads as a measured refusal.
  decision        REAL,
  decision_stderr REAL,
  archive         TEXT NOT NULL
);
"""


class ArchiveStoreError(ValueError):
    pass


def corpus_fingerprint(corpus: dict) -> str:
    """A stable hash of the corpus as it was when the experiment ran.

    The corpus store's own function, named here too because this module's
    readers and its stored references speak of a *corpus* fingerprint. One
    implementation, so a fingerprint written into an archive and a key in the
    store can never drift apart.
    """
    return corpora.fingerprint(corpus)


def shrink(value: dict, *, keep=None) -> tuple[dict, int]:
    """The archive as stored, and the id of the corpus row it references: the
    corpus and ground truth swapped for a portable reference, the text itself
    written into the corpus store.

    The write happens *here*, before `put()` inserts the row that references
    it, and that order is the point: a failure between the two leaves an
    orphan corpus nobody reads, which is harmless, where the other order
    would leave an archive pointing at text that was never stored — a row
    that cannot answer for what produced it.

    Two halves come out of it because a reference has two jobs. The id is
    returned for the local join and belongs on the row, not in the archive: it
    means nothing anywhere else. The fingerprint goes *into* the archive, so a
    file exported from here can still be resolved and checked on a machine
    where this store's ids do not exist.
    """
    # Resolved per call, never bound as a default: a default is evaluated once
    # at import, which would pin this to whichever store existed then.
    keep = keep or corpora.put
    thin = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    dataset = thin.get('evaluation', {}).get('inspector', {}).get('dataset')
    if dataset is None:
        return thin, 0
    corpus = dataset.get('corpus')
    if corpus is None:
        raise ArchiveStoreError(
            'evaluation.inspector.dataset.corpus: required before shrinking')
    ground_truth = dataset.get('ground_truth')
    if ground_truth is None:
        raise ArchiveStoreError(
            'evaluation.inspector.dataset.ground_truth: required before '
            'shrinking')
    id_corpora = keep(dataset['id'], corpus, ground_truth)
    dataset.pop('corpus')
    dataset.pop('ground_truth')
    dataset[CORPUS_REF] = {
        'dataset': dataset['id'],
        'fingerprint': corpora.fingerprint(corpus),
        # Named as well as the corpus, because one corpus can be graded
        # against more than one question set: without this the reference
        # would be ambiguous the day that happens, and an ambiguous reference
        # is a row that cannot say what it read.
        'ground_truth': corpora.fingerprint(ground_truth),
    }
    return thin, int(id_corpora)


def _stored_corpus(id_corpora: int, reference: dict):
    """The corpus row an archive references, by whichever address the caller
    actually has: the local id when the row that carried the archive had one,
    and the content address when it did not — an archive imported from another
    machine carries a fingerprint and no id of ours. Both answers are checked
    the same way by `swell`, so neither path can splice unverified text."""
    if id_corpora:
        found = corpora.get(int(id_corpora))
        if found is not None:
            return found
    return corpora.find(reference.get('fingerprint') or '',
                        reference.get('ground_truth') or '')


def swell(thin: dict, id_corpora: int = 0, *, fetch=None) -> dict:
    """The stored archive with its corpus and ground truth spliced back in,
    or a refusal.

    Two steps, and the second is the one that matters: the text is fetched by
    `id_corpora` — the archives row's foreign key, holding a `corpora.id` —
    and then hashed again and compared with the fingerprint the archive itself
    recorded. An id says which row and not what is in it, so an id reused after
    a delete, or a row edited in place, would otherwise hand a plausible corpus
    to an experiment that never ran on it. Only content can vouch for content.

    Never a substitution, either way it fails: a version the store does not
    hold is named as missing, and one that does not hash to what was recorded
    is refused, rather than either being stood in for by the corpus that
    currently wears the same dataset id.
    """
    # Resolved per call, for the reason `shrink` gives.
    fetch = fetch or _stored_corpus
    value = json.loads(json.dumps(thin, ensure_ascii=False, allow_nan=False))
    dataset = value.get('evaluation', {}).get('inspector', {}).get('dataset')
    if dataset is None or CORPUS_REF not in dataset:
        return value
    reference = dataset.pop(CORPUS_REF)
    wanted = reference.get('dataset') or ''
    recorded = reference.get('fingerprint') or ''
    try:
        found = fetch(id_corpora, reference)
    except corpora.CorpusStoreError as error:
        raise ArchiveStoreError(f'{wanted}: {error}') from error
    if found is None:
        raise ArchiveStoreError(
            f'{wanted}: the corpus this experiment ran on (fingerprint '
            f'{recorded}, corpora row {id_corpora or "unknown"}) is not in the '
            'corpus store, so its evidence cannot be served. Refused rather '
            'than served from whatever now carries that dataset id, because a '
            'row must never lie about what produced it.')
    corpus, ground_truth = found
    # The check the id cannot make for itself, and the reason the fingerprint
    # is kept inside the archive as well as joined on from outside it.
    served = corpora.fingerprint(corpus)
    if served != recorded:
        raise ArchiveStoreError(
            f'{wanted}: the stored corpus is not the one this experiment ran '
            f'on (fingerprint {served}, recorded {recorded}). Refused rather '
            'than served, because a row must never lie about what produced '
            'it.')
    dataset['corpus'] = corpus
    dataset['ground_truth'] = ground_truth
    return value


def projection(value: dict) -> dict:
    """Every sortable column, read off the archive and nowhere else.

    `id_corpora` is deliberately not among them and cannot be: it is a local
    storage id, and no reading of an archive can produce one. It is the single
    column `put()` writes beside this projection — see the schema comment.
    """
    settings = value['settings']['config']
    evaluation = value.get('evaluation') or {}
    result = evaluation.get('result') or {}
    ragas = result.get('ragas') or {}
    spread = ragas.get('decision_spread') or {}
    index, retrieval, generation = (settings['index'], settings['retrieval'],
                                    settings['generation'])
    return {
        'dataset': result.get('dataset') or index.get('dataset') or datasets.BUILTIN,
        'label': result.get('label', settings.get('label', '')),
        'started_at': result.get('started_at', ''),
        'seconds': float(result.get('seconds') or 0.0),
        'provider': (evaluation.get('execution') or {}).get('provider', ''),
        'chunker': index.get('chunker', ''),
        'embedder': index.get('embedder', ''),
        'retriever': retrieval.get('retriever', ''),
        'reranker': retrieval.get('reranker', ''),
        'grader': retrieval.get('grader', ''),
        'answerer': generation.get('answerer', ''),
        'n_questions': int((result.get('summary') or {}).get('n_questions') or 0),
        'decision': ragas.get('decision'),
        'decision_stderr': spread.get('stderr'),
    }


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute(SCHEMA)
    _migrate(db)
    return db


def _migrate(db: sqlite3.Connection) -> None:
    """Add columns this schema has gained since a table was created —
    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists."""
    have = {row['name'] for row in db.execute('PRAGMA table_info(archives)')}
    # The default is 0, meaning "no corpus row here" — the only value a row
    # written before this column existed could honestly carry, and one no
    # corpus row can ever have (`AUTOINCREMENT` starts at 1). Such a row falls
    # back to the fingerprint the archive carries, which is exactly the case
    # the portable half exists for.
    for name, kind in (('id_corpora', 'INTEGER NOT NULL DEFAULT 0'),):
        if name not in have:
            db.execute(f'ALTER TABLE archives ADD COLUMN {name} {kind}')
            have.add(name)
    if 'id' not in have:
        _rekey(db)


def _rekey(db: sqlite3.Connection) -> None:
    """Give an archives table written before `id` existed one, in place.

    The one migration `ALTER TABLE ... ADD COLUMN` cannot do: SQLite will not
    add a primary key to a table that already exists, so the table is rebuilt
    and its rows copied across. Copied in `rowid` order, so the ids that come
    out follow the order the archives were stored in rather than an arbitrary
    one — an id is storage identity, and the storage order is the only thing it
    could honestly mean.

    Nothing is transcribed on the way: every column is carried by name, so a
    rebuilt row says exactly what it said before, and only its key changed.
    """
    columns = [row['name'] for row in db.execute('PRAGMA table_info(archives)')]
    names = ', '.join(columns)
    db.execute('ALTER TABLE archives RENAME TO archives_keyed_by_experiment')
    db.execute(SCHEMA)
    db.execute(f'INSERT INTO archives ({names}) SELECT {names} FROM '
               'archives_keyed_by_experiment ORDER BY rowid')
    db.execute('DROP TABLE archives_keyed_by_experiment')
    db.commit()


def put(db: sqlite3.Connection, experiment_id: str, value: dict, *,
        keep=None) -> dict:
    """Store one complete archive, validated before it is written.

    The corpus goes into the corpus store first (`shrink`), and only then does
    the row that references it get inserted — see `shrink` for why that order
    is the only safe one.

    An experiment already stored is *updated* rather than replaced, and that is
    not a style choice: `INSERT OR REPLACE` deletes the old row and inserts a
    new one, which under `AUTOINCREMENT` burns an id and hands the same
    experiment a different one every time it is re-stored. Looked up first for
    the reason `corpus_store.put` gives, and the upsert stays on the insert
    branch so a concurrent writer is still handled by the unique constraint
    rather than by luck.
    """
    archive.validate_archive(value)
    # First the corpus, then the row: `shrink` has written the text and come
    # back with the id of the row holding it, so the reference below can only
    # ever point at something that exists.
    thin, id_corpora = shrink(value, keep=keep)
    values = {'experiment_id': experiment_id, 'id_corpora': id_corpora,
              **projection(thin),
              'archive': json.dumps(thin, ensure_ascii=False, allow_nan=False)}
    fields = tuple(values)
    assignments = ', '.join(f'{name} = :{name}' for name in fields
                            if name != 'experiment_id')
    found = db.execute('SELECT id FROM archives WHERE experiment_id = ?',
                       (experiment_id,)).fetchone()
    if found is None:
        db.execute(
            f'INSERT INTO archives ({", ".join(fields)}) '
            f'VALUES ({", ".join(":" + name for name in fields)}) '
            f'ON CONFLICT(experiment_id) DO UPDATE SET {assignments}', values)
    else:
        db.execute(f'UPDATE archives SET {assignments} '
                   'WHERE experiment_id = :experiment_id', values)
    db.commit()
    return thin


def serve(db: sqlite3.Connection, experiment_id: str, *, fetch=None):
    """One archive, whole — the same object the export button writes.

    The join happens here: `id_corpora` off the row, then `swell` fetches by it
    and verifies what came back against the archive's own fingerprint.
    """
    row = db.execute(
        'SELECT archive, id_corpora FROM archives WHERE experiment_id = ?',
        (experiment_id,)).fetchone()
    if row is None:
        return None
    return swell(json.loads(row['archive']), row['id_corpora'], fetch=fetch)


def _encoded(value: dict) -> dict:
    """One object as JSON reads it — the encoding, not a repair: a tuple
    becomes the list it is written as, and anything that is not JSON at all
    raises here rather than reaching the validator as a Python object it has no
    vocabulary for."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def store_completed(settings: dict, result: dict, evidence: dict, *,
                    path: Path | None = None, keep=None) -> str:
    """Build the archive of one finished evaluation and store it; return the
    experiment id it was stored under.

    The seam between "an evaluation finished" and "the experiment is on
    record". `build_completed` is the same assembly the export button's own
    codec performs in the browser, so the row written here is the file a reader
    would have downloaded — one object, one strictness, not two that can drift.

    Nothing is invented on the way: the settings, the result and the evidence
    all come from the run that just happened, and `build_completed` refuses
    rather than repairs anything they do not support. A caller that has an
    unfinished run has nothing to hand this.

    Encoded first, because an archive is a JSON document and the objects a
    running lab holds are not quite JSON — `agentic_weights` is a tuple in the
    config the dataclasses produce, and the browser's own export only ever sees
    it after it has crossed the wire as a list. One encoding for both paths, so
    the archive stored here and the file a reader downloads cannot differ by
    the shape of a value. `allow_nan=False`, so a non-finite number is refused
    here rather than written as something no JSON reader will accept.
    """
    value = archive.build_completed(_encoded(settings), _encoded(result),
                                    _encoded(evidence))
    experiment_id = value['evaluation']['result']['run_id']
    db = connect(path or ledger.db_path())
    try:
        put(db, experiment_id, value, keep=keep)
    finally:
        db.close()
    return experiment_id
