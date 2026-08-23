"""One object, whatever brought it: exported, stored, served, imported.

The whole point of splitting an experiment across two databases is that a
reader can never tell. An archive is written to `raglab.db` with its corpus
lifted out into `corpora.db`; opening it joins the two back and must produce
the *same file* the export button writes — because what happens next is the
import path, unchanged, the one a downloaded file takes.

That is a claim about a loop, so it is tested as a loop, over every shape an
archive takes rather than only the fullest one. A codec that lost the summaries
on rung 2, or the judged metrics on the traceless rung, would satisfy any test
written against `generated` alone — which is exactly how 166 real evaluations
came to look unarchivable.
"""
import json

import pytest

from raglab.evaluation import experiment_archive as archive
from raglab.evaluation import experiment_archive_store as store
from raglab.evaluation.tests import archive_examples as examples


LADDER = examples.ladder()
RUNG_IDS = [rung['name'] for rung in LADDER]


@pytest.mark.parametrize('rung', LADDER, ids=RUNG_IDS)
def test_what_is_stored_split_is_served_whole_and_identical(rung, tmp_path):
    # this is an integration test
    """Two tables in, one file out, and the file is the one that went in.

    Not `contents()` counts and not a spot-check of a few keys: the whole
    object, compared to the archive the export button would have written. The
    corpus and the ground truth are the parts that physically are not in
    `raglab.db` at all, so if the join were lossy this is where it shows.
    """
    exported = rung['archive']
    db = store.connect(tmp_path / 'raglab.db')
    try:
        store.put(db, f'exp-{rung["name"]}', exported)
        served = store.serve(db, f'exp-{rung["name"]}')
    finally:
        db.close()

    assert served == exported, (
        f'the {rung["name"]} rung did not survive the split across two '
        'databases')
    # Byte-for-byte in the encoding the file is actually written in, since
    # "identical JSON file" is the claim and dict equality is weaker than it.
    assert (json.dumps(served, sort_keys=True, ensure_ascii=False)
            == json.dumps(exported, sort_keys=True, ensure_ascii=False))


@pytest.mark.parametrize('rung', LADDER, ids=RUNG_IDS)
def test_the_served_object_is_accepted_by_the_import_that_reads_a_file(
        rung, tmp_path):
    # this is an integration test
    """Opening feeds the import; so what opening produces must be importable.

    The panel proves this on its own side — `openHandedExperiment` hands the
    served archive to `adoptArchive`, the same function `importArchiveFile`
    calls. This is the server-side half of that promise: the object the join
    produces passes the very validation an uploaded file is put through, so
    "open uses the import function" cannot be true in the wiring and false in
    the data.
    """
    db = store.connect(tmp_path / 'raglab.db')
    try:
        store.put(db, 'exp-1', rung['archive'])
        served = store.serve(db, 'exp-1')
    finally:
        db.close()

    # The same call `imported_archive_store` makes at the server's trust
    # boundary for a file someone chose to import.
    assert archive.validate_archive(served) == served

    # And it survives the text it would travel as, since a file is text.
    assert archive.validate_archive(
        json.loads(json.dumps(served, ensure_ascii=False, allow_nan=False)))


@pytest.mark.parametrize('rung', LADDER, ids=RUNG_IDS)
def test_every_rung_carries_its_corpus_and_ground_truth_out_again(
        rung, tmp_path):
    # this is an integration test
    """The two things lifted into the other database, checked by value.

    A rung with no evaluation has no corpus to lift, and must not grow one —
    an archive that gained a corpus it never had would be a row claiming
    evidence it does not hold.
    """
    db = store.connect(tmp_path / 'raglab.db')
    try:
        store.put(db, 'exp-1', rung['archive'])
        served = store.serve(db, 'exp-1')
    finally:
        db.close()

    evaluation = served.get('evaluation')
    if evaluation is None:
        assert rung['archive'].get('evaluation') is None
        return
    dataset = evaluation['inspector']['dataset']
    assert dataset['corpus'] == examples.CORPUS
    assert dataset['ground_truth'] == examples.GROUND_TRUTH
    # The reference the split leaves behind must not survive into the object a
    # reader receives: an exported file has a corpus, not a pointer to one.
    assert store.CORPUS_REF not in dataset


def test_the_five_rungs_are_one_format_not_five(tmp_path):
    # this is a unit test
    """Unified structure, asserted rather than assumed.

    Every rung is the same envelope — same top-level keys, same settings
    shape — differing only in how much evidence hangs off `evaluation`. If a
    rung grew a key of its own, a reader could not treat one import path as
    reading all of them, which is the property this whole design rests on.
    """
    for rung in LADDER:
        value = rung['archive']
        assert set(value) <= {'format', 'version', 'settings', 'evaluation'}
        assert value['format'] == archive.FORMAT
        assert value['version'] == archive.VERSION
        assert set(value['settings']) == {'config', 'ui'}
        assert set(value['settings']['config']) == {
            'label', 'index', 'retrieval', 'generation'}
        if 'evaluation' in value:
            assert set(value['evaluation']) == {
                'execution', 'metric_catalogue', 'stage_results', 'result',
                'inspector'}
            assert set(value['evaluation']['inspector']) == {
                'dataset', 'chunks_by_session', 'summaries', 'traces'}
