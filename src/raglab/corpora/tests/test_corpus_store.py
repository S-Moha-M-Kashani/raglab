"""One corpus, stored once, keyed by a row id and unique by what it says."""
import copy
import json

import pytest

from raglab.corpora import corpus_store as store
from raglab.corpora import dataset_import_contract as datasets


def _corpus(first_line: str = 'the third shelf'):
    return {'meta': {'language': 'en'}, 'persona': {}, 'threads': [],
            'sessions': [{'session_id': 's1', 'date': '2026-08-19',
                          'messages': [{'role': 'user', 'content': first_line}]}]}


def _ground_truth(answer: str = 'the third shelf'):
    return {'meta': {'query_date': '2026-08-19'},
            'questions': [{'id': 'q1', 'type': 'single-hop', 'difficulty': 'easy',
                           'question': 'where', 'answer': answer,
                           'evidence': [{'session_id': 's1', 'quote': answer}]}]}


def _rows(path):
    with store.connect(path) as db:
        return db.execute('SELECT COUNT(*) AS n FROM corpora').fetchone()['n']


def test_storing_the_same_corpus_twice_stores_one_row(tmp_path):
    # this is a unit test
    """The whole reason the store exists.

    Seventeen of nineteen real archives are the same 690 KB diary. If a second
    put wrote a second row, the store would be the duplication it was built to
    remove — so this is the claim, not an optimisation detail. The id is the
    key, but the *content* is what decides whether a row is new: same content,
    same row, and the caller is handed the id that already existed.
    """
    path = tmp_path / 'corpora.db'
    first = store.put('diary-fa', _corpus(), _ground_truth(), path)
    second = store.put('diary-fa', _corpus(), _ground_truth(), path)
    assert first == second, 'the same content must resolve to the same row'
    assert _rows(path) == 1


def test_one_corpus_row_serves_every_archive_that_references_it(tmp_path):
    # this is a unit test
    """Many archives, one corpus: the dedup as an archive-shaped claim.

    Each experiment stores the corpus it ran on, exactly as it will when a
    sweep archives twenty candidates over one dataset. They all come back with
    the same id, and that one row answers every one of them.
    """
    path = tmp_path / 'corpora.db'
    ids = {store.put('diary-fa', _corpus(), _ground_truth(), path)
           for _ in range(20)}
    assert len(ids) == 1
    assert _rows(path) == 1
    corpus, ground_truth = store.get(ids.pop(), path)
    assert corpus == _corpus() and ground_truth == _ground_truth()


def test_two_versions_of_one_dataset_id_coexist(tmp_path):
    # this is a unit test
    """A corpus edited between runs is a different corpus, and both survive.

    This is the failure the old scheme could not avoid: it re-read the tracked
    fixture file, so an edit made the earlier experiment's evidence either a
    refusal or — far worse — the new text served as if it were the old. The
    edit is a new row with a new id, and the archive holding the old id keeps
    resolving to what it actually ran on.
    """
    path = tmp_path / 'corpora.db'
    before = store.put('diary-fa', _corpus('the third shelf'),
                       _ground_truth('the third shelf'), path)
    after = store.put('diary-fa', _corpus('the fourth shelf'),
                      _ground_truth('the fourth shelf'), path)
    assert before != after
    assert _rows(path) == 2

    old_corpus, old_truth = store.get(before, path)
    new_corpus, _ = store.get(after, path)
    assert old_corpus['sessions'][0]['messages'][0]['content'] == 'the third shelf'
    assert new_corpus['sessions'][0]['messages'][0]['content'] == 'the fourth shelf'
    assert old_truth == _ground_truth('the third shelf'), (
        'the older version keeps its own ground truth, not the newer one')
    assert {version['id'] for version in store.versions('diary-fa', path)} \
        == {before, after}


def test_an_unknown_id_is_absent_rather_than_approximated(tmp_path):
    # this is a unit test
    """None means "not here" and never "here is something close".

    The refusal an archive's reader depends on starts at this answer: a
    reference the store cannot satisfy must come back empty, so the layer
    above can say so instead of serving a corpus that merely fits.
    """
    path = tmp_path / 'corpora.db'
    store.put('diary-fa', _corpus(), _ground_truth(), path)
    assert store.get(999, path) is None
    assert store.get(0, path) is None


def test_an_unknown_fingerprint_is_absent_too(tmp_path):
    # this is a unit test
    """The same answer from the portable side, which is what an archive
    imported from another machine has instead of one of our ids."""
    path = tmp_path / 'corpora.db'
    store.put('diary-fa', _corpus(), _ground_truth(), path)
    assert store.find('0000000000000000', path=path) is None
    assert store.locate('0000000000000000', path=path) is None
    assert store.find(store.fingerprint(_corpus()), path=path) == (
        _corpus(), _ground_truth())


def test_a_corpus_graded_against_two_question_sets_is_never_guessed(tmp_path):
    # this is a unit test
    """The one genuine ambiguity, refused rather than resolved by picking.

    The same corpus can be graded against a second question set — two rows,
    two ids, one corpus fingerprint. A reference holding an id is never
    ambiguous; one holding only the corpus fingerprint cannot say which
    experiment read which, and answering it with either would be a row lying
    about what produced it.
    """
    path = tmp_path / 'corpora.db'
    first = store.put('diary-fa', _corpus(), _ground_truth('the third shelf'), path)
    second = store.put('diary-fa', _corpus(),
                       _ground_truth('the third shelf, again'), path)
    assert first != second, 'a second question set is a second row'
    assert _rows(path) == 2
    assert store.get(first, path)[1] == _ground_truth('the third shelf')
    assert store.get(second, path)[1] == _ground_truth('the third shelf, again')

    with pytest.raises(store.CorpusStoreError, match='Refused rather than guessed'):
        store.find(store.fingerprint(_corpus()), path=path)

    exact = store.find(store.fingerprint(_corpus()),
                       store.fingerprint(_ground_truth('the third shelf, again')),
                       path)
    assert exact[1] == _ground_truth('the third shelf, again')


def test_what_comes_out_is_what_went_in(tmp_path):
    # this is a unit test
    """Byte-for-byte, Farsi included: an archive's evidence is only evidence
    while the text it carries is the text that was measured."""
    path = tmp_path / 'corpora.db'
    corpus = _corpus('قفسه سوم')
    ground_truth = _ground_truth('قفسه سوم')
    original = (copy.deepcopy(corpus), copy.deepcopy(ground_truth))
    id_corpora = store.put('diary-fa', corpus, ground_truth, path)
    assert store.get(id_corpora, path) == original


def test_the_id_is_the_key_and_the_content_is_what_is_unique(tmp_path):
    # this is a unit test
    """The shape the join needs and the shape the dedup needs, both at once.

    `id` is the key, because a reference in another database file can carry an
    integer and nothing else. The fingerprints are unique together rather than
    the key, which is what keeps one text one row — including under two
    different dataset names, since a name is what a corpus was called and
    never what it is.
    """
    path = tmp_path / 'corpora.db'
    under_one_name = store.put('diary-fa', _corpus(), _ground_truth(), path)
    under_another = store.put('diary-copy', _corpus(), _ground_truth(), path)
    assert under_one_name == under_another
    assert _rows(path) == 1, 'the same text under two names is still one text'

    with store.connect(path) as db:
        keys = {row['name'] for row in db.execute('PRAGMA table_info(corpora)')
                if row['pk']}
        unique = [db.execute(f'PRAGMA index_info({index["name"]})').fetchall()
                  for index in db.execute('PRAGMA index_list(corpora)')
                  if index['unique']]
    assert keys == {'id'}
    assert [{column['name'] for column in columns} for columns in unique] == \
        [{'fingerprint', 'ground_truth_fingerprint'}]


def test_the_store_is_never_the_developers_own_file(monkeypatch, tmp_path):
    # this is a unit test
    """`RAGLAB_CORPORA_DB` is resolved per call, which is what lets the suite's
    autouse fixture redirect every write without patching this module."""
    monkeypatch.setenv('RAGLAB_CORPORA_DB', str(tmp_path / 'elsewhere.db'))
    assert store.db_path() == tmp_path / 'elsewhere.db'
    monkeypatch.setenv('RAGLAB_CORPORA_DB', '')
    assert store.db_path().name == 'corpora.db'
    assert store.db_path().parent.name == 'databases'


def test_a_row_found_by_fingerprint_is_checked_against_what_it_holds(tmp_path):
    # this is a unit test
    """The key is evidence, and the row is asked to confirm it.

    16 hex characters of SHA-256 is 64 bits — overwhelming evidence that two
    corpora are the same text, and not proof of it. Reusing a row on the
    strength of the key alone would mean that the day two corpora ever
    collided, one row would answer for both and every archive referencing it
    would claim a corpus it never ran on. So the stored text is compared with
    the text being stored, and only then is the id handed back.
    """
    path = tmp_path / 'corpora.db'
    first = store.put('diary-fa', _corpus(), _ground_truth(), path)
    second = store.put('diary-fa', _corpus(), _ground_truth(), path)
    assert first == second and _rows(path) == 1

    # The check passed on content, not on the fingerprints agreeing with
    # themselves: the row really is holding the text the second put offered.
    with store.connect(path) as db:
        row = db.execute('SELECT corpus, ground_truth FROM corpora WHERE id = ?',
                         (first,)).fetchone()
    assert store.canonical(json.loads(row['corpus'])) == store.canonical(_corpus())
    assert store.canonical(json.loads(row['ground_truth'])) == \
        store.canonical(_ground_truth())

    # And the same content written with its keys in another order is still one
    # row — canonically compared, so key order cannot read as a collision.
    reordered = {'sessions': _corpus()['sessions'], 'threads': [],
                 'persona': {}, 'meta': {'language': 'en'}}
    assert store.put('diary-fa', reordered, _ground_truth(), path) == first
    assert _rows(path) == 1


def test_a_fingerprint_collision_is_refused_rather_than_aliased(tmp_path,
                                                               monkeypatch):
    # this is a unit test
    """Two different corpora under one key: refused, and nothing written.

    Forced, because a real 64-bit collision is not something a test can wait
    for — the hash is pinned to a constant so two genuinely different corpora
    address the same row. Both other answers are the failure this store exists
    to prevent: aliasing hands the second corpus's archives the first one's
    text, and inserting anyway leaves two rows under a key that claims to be
    unique, with nothing able to choose between them.
    """
    path = tmp_path / 'corpora.db'
    monkeypatch.setattr(store, 'fingerprint', lambda value: 'c0ffeec0ffeec0ff')
    first = store.put('diary-fa', _corpus('the third shelf'),
                      _ground_truth('the third shelf'), path)

    with pytest.raises(store.CorpusStoreError, match='different corpus'):
        store.put('diary-fa', _corpus('a different diary entirely'),
                  _ground_truth('the third shelf'), path)
    with pytest.raises(store.CorpusStoreError, match='different ground truth'):
        store.put('diary-fa', _corpus('the third shelf'),
                  _ground_truth('a different answer entirely'), path)

    assert _rows(path) == 1, 'a refusal writes nothing'
    corpus, ground_truth = store.get(first, path)
    assert corpus == _corpus('the third shelf'), (
        'the stored row is untouched by the corpus that could not be stored')
    assert ground_truth == _ground_truth('the third shelf')


def test_no_two_rows_are_ever_identical_in_key_or_in_content(tmp_path):
    # this is a unit test
    """The claim the table itself has to satisfy, asserted against the table.

    A batch with repeats and edits in it: the repeats must collapse onto the
    rows they repeat, the edits must each be their own row, and afterwards no
    two rows may agree on their key *or* on what they hold. Read out of SQLite
    rather than inferred from the ids `put` returned, because it is the table
    a later archive resolves against.
    """
    path = tmp_path / 'corpora.db'
    batch = [('diary-fa', 'the third shelf', 'the third shelf'),
             ('diary-fa', 'the third shelf', 'the third shelf'),   # a repeat
             ('diary-fa', 'the fourth shelf', 'the fourth shelf'),  # an edit
             ('diary-copy', 'the third shelf', 'the third shelf'),  # renamed
             ('diary-fa', 'the third shelf', 'the third shelf, again')]
    ids = [store.put(name, _corpus(text), _ground_truth(answer), path)
           for name, text, answer in batch]
    assert ids[0] == ids[1] == ids[3], 'the same content is the same row'
    assert len({*ids}) == 3

    with store.connect(path) as db:
        rows = [dict(row) for row in db.execute(
            'SELECT id, fingerprint, ground_truth_fingerprint, corpus, '
            'ground_truth FROM corpora')]
    assert len(rows) == 3
    keys = [(row['fingerprint'], row['ground_truth_fingerprint']) for row in rows]
    contents = [(store.canonical(json.loads(row['corpus'])),
                 store.canonical(json.loads(row['ground_truth'])))
                for row in rows]
    assert len(set(keys)) == len(rows), 'no two rows share a key'
    assert len(set(contents)) == len(rows), 'no two rows hold the same text'
    assert len({row['id'] for row in rows}) == len(rows)


# --- the refill (D8): the five bundled pairs enter as new rows -------------
#
# Everything above is proven with a small, hand-written corpus so the claims
# stay legible. This store is opaque about shape — it hashes and stores
# whatever dict it is handed — so nothing here needed to change for the
# schema-pair migration. What the tests below pin is that the *refill* the
# migration requires (D8: the old rows in the real `databases/corpora.db`
# hold the pre-migration shape; nothing re-derives them; a new row per
# dataset enters the same way every corpus always has — by being stored) is
# nothing more than storing the five real pairs, and that they coexist beside
# rows already holding the old shape rather than replacing them.

BUNDLED = ('diary-fa', 'support-en', 'meetings-de', 'research-multihop',
          'smoke-mini')


def test_every_bundled_pairs_real_shape_round_trips_through_the_store(tmp_path):
    # this is an integration test
    """`corpora.get(id)` on a new-pair row hands back exactly the two file
    payloads `load()` reads — the schema's own shape (`corpus_documents`,
    `groundtruth_dataset`, …), not the retired `sessions`/`questions` one
    `_corpus()`/`_ground_truth()` above stand in for. One row per dataset,
    because five distinct corpora are five distinct fingerprints.
    """
    path = tmp_path / 'corpora.db'
    ids = set()
    for dataset_id in BUNDLED:
        corpus, ground_truth = datasets.load(dataset_id)
        id_corpora = store.put(dataset_id, corpus, ground_truth, path)
        assert store.get(id_corpora, path) == (corpus, ground_truth)
        ids.add(id_corpora)
    assert len(ids) == len(BUNDLED), 'five distinct corpora, five distinct rows'
    assert _rows(path) == len(BUNDLED)


def test_a_bundled_pairs_fingerprint_still_refuses_altered_content(tmp_path,
                                                                    monkeypatch):
    # this is a unit test
    """The same refusal `_same_or_refuse` makes on the hand-written fixture
    above, pinned again against a real bundled pair — `diary-fa` — so the
    claim is not an artefact of the small synthetic corpus. Forced with a
    pinned fingerprint for the reason
    `test_a_fingerprint_collision_is_refused_rather_than_aliased` gives: a
    real 64-bit collision is not something a test can wait for.
    """
    path = tmp_path / 'corpora.db'
    corpus, ground_truth = datasets.load('diary-fa')
    monkeypatch.setattr(store, 'fingerprint', lambda value: 'c0ffeec0ffeec0ff')
    first = store.put('diary-fa', corpus, ground_truth, path)

    altered = copy.deepcopy(corpus)
    altered['corpus_documents'][0]['document_content'][0]['text'] += ' — edited'
    with pytest.raises(store.CorpusStoreError, match='different corpus'):
        store.put('diary-fa', altered, ground_truth, path)

    assert _rows(path) == 1, 'the refusal writes nothing'
    assert store.get(first, path) == (corpus, ground_truth), (
        'the stored row is untouched by the corpus that could not be stored')


def test_versions_of_diary_fa_lets_a_new_shape_row_coexist_beside_an_old_one(
        tmp_path):
    # this is an integration test
    """The exact claim D8 makes about the real `databases/corpora.db`, pinned
    here on a throwaway one instead (the suite never touches the real file):
    a row already holding the pre-migration shape (`sessions`/`messages`/
    `groundtruth` — the shape every archive before this branch carried, and
    the shape still sitting in the real database's `diary-fa` row) is left
    exactly as it is, and the real pair `load('diary-fa')` now returns enters
    as a second, distinct row under the same dataset id — never a migration,
    never an overwrite.
    """
    path = tmp_path / 'corpora.db'
    old_shape_corpus = _corpus('a diary entry from before the schema pair')
    old_shape_truth = _ground_truth('a diary entry from before the schema pair')
    old_id = store.put('diary-fa', old_shape_corpus, old_shape_truth, path)

    new_corpus, new_truth = datasets.load('diary-fa')
    new_id = store.put('diary-fa', new_corpus, new_truth, path)

    assert old_id != new_id
    versions = {version['id'] for version in store.versions('diary-fa', path)}
    assert versions == {old_id, new_id}
    # The old row is untouched — still the shape nothing re-derives it from.
    assert store.get(old_id, path) == (old_shape_corpus, old_shape_truth)
    # The new row is the schema's own shape, not a translation of it.
    assert 'corpus_documents' in new_corpus and 'sessions' not in new_corpus
    assert store.get(new_id, path) == (new_corpus, new_truth)
