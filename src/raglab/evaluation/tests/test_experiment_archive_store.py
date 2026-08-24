"""What is stored is what is served: one archive per row, corpus by reference."""
import copy
import json
import sqlite3

import pytest

from raglab.corpora import corpus_store as corpora
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import experiment_archive as archive
from raglab.evaluation import experiment_archive_store as store
from raglab.evaluation.tests import archive_examples as examples


def _complete():
    """The ladder's fullest rung — the only shape this store accepts."""
    return examples.generated_rung()['archive']


def test_an_archive_written_before_this_schema_still_serves_unchanged(
        tmp_path, monkeypatch):
    # this is a unit test
    """D2/D8's promise kept at the one seam a schema migration could quietly
    break: an archive stored in the shape every archive carried before this
    branch (`sessions`/`questions`/`session_id`/`message_indices`/`types`,
    not `corpus_documents`/`groundtruth_dataset`/…) still opens exactly as it
    was written.

    `validate_archive` now refuses this shape on the way *in* — that is
    checked below, first — so this row stands in for one already on disk
    from before that refusal existed, written back when this branch's own
    (now retired) validator accepted it. `put()` is bypassed rather than
    called, because calling it would apply *today's* validator to a write
    that never happened under it; `serve()` is not bypassed, because it never
    validates at all — it only splices the corpus back in by
    content-addressed reference, which is exactly why an old-shape row still
    opens.
    """
    old_shape = examples.pre_migration_archive()

    # First: today's codec really does refuse a fresh write in this shape —
    # otherwise this test would not be about an old row at all.
    with pytest.raises(archive.ArchiveError):
        archive.validate_archive(copy.deepcopy(old_shape))

    db = store.connect(tmp_path / 'archives.db')
    monkeypatch.setattr(archive, 'validate_archive', lambda value, **_: value)
    store.put(db, 'pre-migration-run-001', old_shape)
    # Restored *before* `serve()` runs, deliberately: the docstring's claim
    # is that `serve()` itself needs no bypass, and leaving the patch active
    # here would let a future regression that added a `validate_archive` call
    # inside `serve()`/`swell()` pass this test right along with it.
    monkeypatch.undo()

    served = store.serve(db, 'pre-migration-run-001')
    assert served == old_shape, (
        'an archive predating this schema must be served byte-for-byte, '
        'never reinterpreted through the current one')


def test_an_archive_survives_storage_byte_for_byte(tmp_path):
    # this is a unit test
    """The claim the whole change rests on.

    The corpus is dropped on the way in and spliced back on the way out, so the
    object a reader receives has to be indistinguishable from the one export
    wrote — otherwise "open hands over what export writes" is false at the only
    point anybody can check it.
    """
    value = _complete()
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', value)
    served = store.serve(db, 'exp-1')
    assert served == value


def test_the_corpus_is_stored_once_and_not_in_the_row(tmp_path):
    # this is a unit test
    """The reason for the reference: the row must not carry the corpus."""
    value = _complete()
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', value)
    stored = db.execute(
        'SELECT archive FROM archives WHERE experiment_id = ?',
        ('exp-1',)).fetchone()['archive']
    dataset = json.loads(stored)['evaluation']['inspector']['dataset']
    assert 'corpus' not in dataset
    assert store.CORPUS_REF in dataset
    # Not "the corpus text appears nowhere": `chunks_by_session` is that text
    # chunked, and the ground truth quotes from it, and both are this
    # experiment's own evidence rather than a second copy of the corpus. What
    # must be gone is the corpus *body* — every session and message of it.
    whole = json.dumps(_complete(), ensure_ascii=False)
    assert len(stored) < len(whole), 'the stored row must be smaller than the archive'
    assert 'noted, the third shelf' not in stored, (
        'an assistant turn appears only in the corpus body, never in the '
        'chunks or the quotes, so it is the one string that proves the body '
        'itself is gone')


def test_the_corpus_is_in_the_corpus_store_before_the_row_that_names_it(tmp_path):
    # this is a unit test
    """The write order, which is the only thing holding this relationship.

    The corpora live in a second database file, and SQLite enforces no foreign
    key across files — so the guarantee is that the corpus row is written, and
    its id known, before the archive row referencing it exists. A failure
    between them orphans a corpus nobody reads; the other order would leave an
    archive naming a row that was never written.
    """
    written: list[tuple] = []

    def keep(dataset_id, corpus, ground_truth):
        written.append((dataset_id, corpus, ground_truth))
        assert db.execute('SELECT COUNT(*) AS n FROM archives').fetchone()['n'] == 0, (
            'the corpus must be written before the row that references it')
        return corpora.put(dataset_id, corpus, ground_truth)

    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', _complete(), keep=keep)
    assert [entry[0] for entry in written] == [examples.DATASET_ID]
    assert written[0][1] == examples.CORPUS
    assert written[0][2] == examples.GROUND_TRUTH

    id_corpora = db.execute('SELECT id_corpora FROM archives WHERE '
                            'experiment_id = ?', ('exp-1',)).fetchone()['id_corpora']
    assert id_corpora > 0, 'the row must name a corpora row that exists'
    assert corpora.get(id_corpora) == (examples.CORPUS, examples.GROUND_TRUTH)


def test_a_bundled_datasets_real_pair_is_written_by_shrink_with_no_new_code(
        tmp_path):
    # this is an integration test
    """The refill decision (Task 10, D8): no new writer is needed.

    Every experiment that finishes goes through `store.put` -> `shrink` ->
    `keep` -> `corpus_store.put` (`panel_server.keep_archive` reaches this
    same `store.put`) on its way onto the ledger. `shrink` stores whatever
    `evaluation.inspector.dataset.{corpus, ground_truth}` the run actually
    carried, and for a bundled dataset that is
    `dataset_import_contract.load()`'s own real pair — the schema's shape,
    unrewritten (D4) — not a synthetic stand-in. So the first evaluation
    that finishes against each bundled dataset, post-migration, writes its
    real pair through exactly this path: the code every archive has always
    gone through, unchanged by this migration.
    """
    path = tmp_path / 'corpora.db'
    keep = (lambda dataset_id, corpus, ground_truth:
            corpora.put(dataset_id, corpus, ground_truth, path))
    written = {}
    for dataset_id in ('diary-fa', 'support-en', 'meetings-de',
                       'research-multihop', 'smoke-mini'):
        corpus, ground_truth = datasets.load(dataset_id)
        value = {'evaluation': {'inspector': {'dataset': {
            'id': dataset_id, 'corpus': corpus, 'ground_truth': ground_truth,
        }}}}
        thin, id_corpora = store.shrink(value, keep=keep)
        reference = thin['evaluation']['inspector']['dataset'][store.CORPUS_REF]
        assert reference['dataset'] == dataset_id
        assert corpora.get(id_corpora, path) == (corpus, ground_truth)
        written[dataset_id] = id_corpora
    assert len(set(written.values())) == 5, 'five distinct pairs, five rows'


def test_many_archives_over_one_corpus_share_one_corpora_row(tmp_path):
    # this is a unit test
    """The dedup, at the seam that motivated it.

    Seventeen of nineteen real archives are the same 690 KB diary. Twenty
    archives here store one corpus between them and every row carries the same
    `id_corpora`, which is also what makes "who references this corpus" a join
    rather than a walk over twenty JSON blobs.
    """
    db = store.connect(tmp_path / 'archives.db')
    for number in range(20):
        value = _complete()
        value['evaluation']['result']['run_id'] = f'exp-{number}'
        store.put(db, f'exp-{number}', value)

    ids = {row['id_corpora'] for row in
           db.execute('SELECT id_corpora FROM archives')}
    assert len(ids) == 1, 'twenty archives, one corpora row'
    with corpora.connect() as store_db:
        rows = store_db.execute(
            'SELECT COUNT(*) AS n FROM corpora WHERE id = ?',
            (ids.pop(),)).fetchone()['n']
    assert rows == 1


def test_the_corpus_reference_is_the_one_column_not_projected(tmp_path):
    # this is a unit test
    """The exception, checked so it stays deliberate.

    Every other column is a projection of the archive and may never be an
    independent source. `id_corpora` cannot be one: it is a local storage id,
    and no reading of an archive can produce it — an exported file has no
    business naming a row of this machine's database. Its portable counterpart
    lives inside the archive JSON as `corpus_ref`, and the two are checked
    against each other here: the id fetches, the fingerprint proves.
    """
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', _complete())
    assert 'id_corpora' not in store.projection(_complete()), (
        'a local storage id must not be projectable from a portable archive')

    row = db.execute('SELECT id_corpora, archive FROM archives '
                     'WHERE experiment_id = ?', ('exp-1',)).fetchone()
    reference = (json.loads(row['archive'])['evaluation']['inspector']
                 ['dataset'][store.CORPUS_REF])
    assert reference == {'dataset': examples.DATASET_ID,
                         'fingerprint': corpora.fingerprint(examples.CORPUS),
                         'ground_truth': corpora.fingerprint(examples.GROUND_TRUTH)}
    corpus, _ = corpora.get(row['id_corpora'])
    assert corpora.fingerprint(corpus) == reference['fingerprint'], (
        'the id and the fingerprint must name the same text')


def test_a_corpus_the_store_does_not_have_is_refused_rather_than_served(tmp_path):
    # this is a unit test
    """The hazard the fingerprint exists for.

    A dataset replaced under the same id would otherwise hand a plausible
    corpus to an experiment that never ran on it — a row lying about what
    produced it, one step before the row. Content-addressing makes that
    impossible to reach by accident; what remains is absence, and the only
    honest answer to absence is a refusal that names it.
    """
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', _complete())

    with pytest.raises(store.ArchiveStoreError, match='not in the corpus store'):
        store.serve(db, 'exp-1', fetch=lambda *_: None)


def test_a_corpus_the_store_does_not_have_is_refused_through_the_real_store(
        tmp_path, monkeypatch):
    # this is a unit test
    """The same refusal with nothing injected: the archive in one database,
    the corpora in another that does not hold this version."""
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', _complete())
    monkeypatch.setenv('RAGLAB_CORPORA_DB', str(tmp_path / 'other-corpora.db'))
    with pytest.raises(store.ArchiveStoreError, match='not in the corpus store'):
        store.serve(db, 'exp-1')


def test_a_row_the_id_points_at_that_moved_is_refused(tmp_path):
    # this is a unit test
    """Why the id is the join and never the proof.

    An id says which row and not what is in it: reused after a delete, or
    edited in place, it would hand a plausible corpus to an experiment that
    never ran on it. What comes back is hashed again and compared with what
    the archive recorded, so a mismatch is a refusal, not a substitution.
    """
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', _complete())

    def replaced(*_args):
        corpus = copy.deepcopy(examples.CORPUS)
        corpus['corpus_documents'][0]['document_content'][0]['text'] = 'a different diary'
        return corpus, copy.deepcopy(examples.GROUND_TRUTH)

    with pytest.raises(store.ArchiveStoreError, match='not the one this experiment'):
        store.serve(db, 'exp-1', fetch=replaced)


def test_a_row_edited_in_place_under_its_id_is_refused_through_the_real_store(
        tmp_path, monkeypatch):
    # this is a unit test
    """The same claim without an injected lookup: the corpora row this
    archive's id names is rewritten underneath it, and serving refuses.

    Its own corpora file, because this test corrupts a row on purpose and the
    suite's autouse redirect points every test at one shared store — the
    damage would otherwise be every later test's to discover.
    """
    monkeypatch.setenv('RAGLAB_CORPORA_DB', str(tmp_path / 'corrupted.db'))
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', _complete())
    id_corpora = db.execute('SELECT id_corpora FROM archives WHERE '
                            'experiment_id = ?', ('exp-1',)).fetchone()['id_corpora']
    moved = copy.deepcopy(examples.CORPUS)
    moved['corpus_documents'][0]['document_content'][0]['text'] = 'a different diary'
    with corpora.connect() as store_db:
        store_db.execute('UPDATE corpora SET corpus = ? WHERE id = ?',
                         (json.dumps(moved, ensure_ascii=False), id_corpora))

    with pytest.raises(store.ArchiveStoreError, match='not the one this experiment'):
        store.serve(db, 'exp-1')


def test_an_archive_with_no_local_id_is_served_by_content_address(tmp_path):
    # this is a unit test
    """The portable half doing its job.

    An archive that arrived from another machine has a fingerprint and no id
    of ours — and a row written before this column existed carries 0. The
    content address resolves it, and it is verified exactly as the id path is.
    """
    db = store.connect(tmp_path / 'archives.db')
    thin = store.put(db, 'exp-1', _complete())
    db.execute('UPDATE archives SET id_corpora = 0 WHERE experiment_id = ?',
               ('exp-1',))
    db.commit()
    assert store.serve(db, 'exp-1') == _complete()
    assert store.swell(thin) == _complete(), (
        'no id at all still resolves, by the fingerprint the archive carries')


def test_an_ambiguous_reference_is_refused_by_name(tmp_path):
    # this is a unit test
    """One corpus graded against two question sets: refused, never picked."""
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', _complete())

    def ambiguous(*_args):
        raise corpora.CorpusStoreError('two ground truths, none of them named')

    with pytest.raises(store.ArchiveStoreError, match='two ground truths'):
        store.serve(db, 'exp-1', fetch=ambiguous)


def test_every_column_is_projected_from_the_archive(tmp_path):
    # this is a unit test
    """Columns exist so the board can sort; they may never be a second source.

    Each one is read back off the row and compared with the archive it was
    projected from, so a column that drifted from the record would fail here
    rather than sort the board into an order the archives do not support.
    """
    value = _complete()
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', value)
    row = db.execute('SELECT * FROM archives WHERE experiment_id = ?',
                     ('exp-1',)).fetchone()
    for column, expected in store.projection(value).items():
        assert row[column] == expected, f'column {column} disagrees with the archive'
    assert row['dataset'] == examples.DATASET_ID
    assert row['decision'] == examples.RAGAS['decision']


def test_an_incomplete_archive_is_refused_by_the_store(tmp_path):
    # this is a unit test
    """Only complete archives are rows now, so the store is where that is held."""
    db = store.connect(tmp_path / 'archives.db')
    broken = _complete()
    broken['evaluation']['result']['rows'] = []
    with pytest.raises(Exception):
        store.put(db, 'exp-1', broken)


def _archives(db) -> list[dict]:
    return [dict(row) for row in db.execute(
        'SELECT id, experiment_id, label FROM archives ORDER BY id')]


def test_an_archive_row_is_keyed_by_an_id_the_database_assigns(tmp_path):
    # this is a unit test
    """Storage identity, given by the database, exactly as `corpora.id` is.

    The two tables that hold an experiment and the text it ran on are keyed the
    same way, so a reference into either is the same kind of thing. The
    experiment id stays the only address anybody looks a row up by — it is what
    the row is *about* — and it is unique, so an experiment is on record once.
    """
    db = store.connect(tmp_path / 'archives.db')
    for number in range(3):
        value = _complete()
        value['evaluation']['result']['run_id'] = f'exp-{number}'
        store.put(db, f'exp-{number}', value)

    rows = _archives(db)
    assert [row['experiment_id'] for row in rows] == ['exp-0', 'exp-1', 'exp-2']
    assert [row['id'] for row in rows] == [1, 2, 3]
    assert all(isinstance(row['id'], int) for row in rows)

    keys = {row['name'] for row in db.execute('PRAGMA table_info(archives)')
            if row['pk']}
    unique = [{column['name'] for column in
               db.execute(f'PRAGMA index_info({index["name"]})')}
              for index in db.execute('PRAGMA index_list(archives)')
              if index['unique']]
    assert keys == {'id'}
    assert {'experiment_id'} in unique, 'one experiment is one row'
    # And every lookup still goes by experiment id, which is what the route,
    # `serve` and `put` all speak.
    assert store.serve(db, 'exp-1')['evaluation']['result']['run_id'] == 'exp-1'
    assert store.serve(db, 'exp-9') is None


def test_re_storing_one_experiment_edits_its_row_and_burns_no_id(tmp_path):
    # this is a unit test
    """A retry is the same experiment, so it is the same row.

    `INSERT OR REPLACE` would delete the row and insert a new one, which under
    `AUTOINCREMENT` hands the same experiment a different id every time it is
    re-stored and leaves the ids after it counting attempts rather than
    experiments. Looked up before the insert, for the reason `corpus_store.put`
    gives — and the row that comes back is the *new* archive, not a stale one.
    """
    db = store.connect(tmp_path / 'archives.db')
    store.put(db, 'exp-1', _complete())
    first = _archives(db)[0]

    relabelled = _complete()
    relabelled['settings']['config']['label'] = 'run again'
    relabelled['evaluation']['result']['config']['label'] = 'run again'
    relabelled['evaluation']['result']['label'] = 'run again'
    store.put(db, 'exp-1', relabelled)

    rows = _archives(db)
    assert len(rows) == 1, 're-storing an experiment must not leave two rows'
    assert rows[0]['id'] == first['id'], 'the same experiment keeps its id'
    assert rows[0]['label'] == 'run again', 'and the row is the new archive'
    assert store.serve(db, 'exp-1') == relabelled

    # The next experiment takes the next id, rather than one past a burnt one.
    following = _complete()
    following['evaluation']['result']['run_id'] = 'exp-2'
    store.put(db, 'exp-2', following)
    assert [row['id'] for row in _archives(db)] == [first['id'], first['id'] + 1]


def test_a_table_written_before_ids_existed_gains_them_without_losing_a_row(
        tmp_path):
    # this is a unit test
    """The migration, over a table in the shape this store shipped with.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists and
    `ALTER TABLE` cannot add a primary key, so the old table is rebuilt and its
    rows copied in storage order. Every archive has to survive it saying exactly
    what it said — a migration that dropped or rewrote a row would be the
    ledger's own first rule broken by the machinery meant to preserve it.
    """
    path = tmp_path / 'archives.db'
    old = sqlite3.connect(path)
    old.row_factory = sqlite3.Row
    old.execute("""
      CREATE TABLE archives (
        experiment_id TEXT PRIMARY KEY,
        id_corpora INTEGER NOT NULL DEFAULT 0,
        dataset TEXT NOT NULL DEFAULT '', label TEXT NOT NULL DEFAULT '',
        started_at TEXT NOT NULL DEFAULT '', seconds REAL NOT NULL DEFAULT 0,
        provider TEXT NOT NULL DEFAULT '', chunker TEXT NOT NULL DEFAULT '',
        embedder TEXT NOT NULL DEFAULT '', retriever TEXT NOT NULL DEFAULT '',
        reranker TEXT NOT NULL DEFAULT '', grader TEXT NOT NULL DEFAULT '',
        answerer TEXT NOT NULL DEFAULT '', n_questions INTEGER NOT NULL DEFAULT 0,
        decision REAL, decision_stderr REAL, archive TEXT NOT NULL)
    """)
    thin, id_corpora = store.shrink(_complete())
    encoded = json.dumps(thin, ensure_ascii=False, allow_nan=False)
    for number in (0, 1):
        old.execute(
            'INSERT INTO archives (experiment_id, id_corpora, label, archive) '
            'VALUES (?, ?, ?, ?)',
            (f'old-{number}', id_corpora, f'kept {number}', encoded))
    old.commit()
    old.close()

    db = store.connect(path)
    rows = _archives(db)
    assert [row['experiment_id'] for row in rows] == ['old-0', 'old-1']
    assert [row['id'] for row in rows] == [1, 2], (
        'the ids follow the order the archives were stored in')
    assert [row['label'] for row in rows] == ['kept 0', 'kept 1']
    assert store.serve(db, 'old-1') == _complete(), (
        'a migrated row still serves the archive it was written with')
    # And the migration is done once: opening it again changes nothing.
    assert _archives(store.connect(path)) == rows
