"""Contracts for portable RAG Lab experiment archives."""
import copy
import json

import pytest

from raglab.evaluation import experiment_archive as archive
from raglab.evaluation.tests import archive_examples as examples
from raglab.evaluation.tests.archive_examples import completed_archive


def test_settings_only_and_completed_archives_validate_without_fabricated_results():
    # this is a unit test
    full = completed_archive()
    settings_only = {key: copy.deepcopy(full[key])
                     for key in ('format', 'version', 'settings')}
    assert archive.validate_archive(settings_only) == settings_only
    assert 'evaluation' not in settings_only
    assert archive.validate_archive(full) == full


def test_stage_results_are_derived_from_the_metric_catalogue():
    # this is a unit test
    full = completed_archive()
    expected = full['evaluation']['stage_results']
    assert archive.stage_results(full['evaluation']['result'],
                                 full['evaluation']['metric_catalogue']) == expected


@pytest.mark.parametrize('key', ['api_key', 'openrouterKey', 'password',
                                 'client_secret', 'access_token', 'authorization'])
def test_credentials_are_rejected_at_any_depth(key):
    # this is a unit test
    full = completed_archive()
    full['evaluation']['inspector']['dataset']['corpus'][key] = 'never persist'
    with pytest.raises(archive.ArchiveError, match=key):
        archive.validate_archive(full)


def test_archive_size_config_and_stage_mismatches_are_refused():
    # this is a unit test
    full = completed_archive()
    assert archive.validate_archive(full, encoded_size=archive.MAX_BYTES) == full
    with pytest.raises(archive.ArchiveError, match='32 MiB'):
        archive.validate_archive(full, encoded_size=archive.MAX_BYTES + 1)
    full = completed_archive()
    full['evaluation']['result']['config']['label'] = 'different'
    with pytest.raises(archive.ArchiveError, match='result.config'):
        archive.validate_archive(full)
    full = completed_archive()
    full['evaluation']['stage_results']['retrieval']['metrics']['recall'] = 0.2
    with pytest.raises(archive.ArchiveError, match='stage_results'):
        archive.validate_archive(full)


def test_direct_posts_enforce_the_encoded_archive_size_limit(monkeypatch):
    # this is a unit test
    payload = completed_archive()
    encoded_size = len(json.dumps(payload, ensure_ascii=False,
                                  allow_nan=False).encode('utf-8'))
    monkeypatch.setattr(archive, 'MAX_BYTES', encoded_size - 1)
    with pytest.raises(archive.ArchiveError, match='archive.*(encoded|size|MiB)'):
        archive.validate_archive(payload)


def test_direct_post_serialization_errors_are_archive_errors():
    # this is a unit test
    payload = completed_archive()
    value = 10 ** 5_000
    payload['evaluation']['result']['summary']['overall']['large_integer'] = value
    payload['evaluation']['stage_results']['overall']['metrics']['large_integer'] = value
    with pytest.raises(archive.ArchiveError, match='archive.*(serialize|encoded|size)'):
        archive.validate_archive(payload)


def test_result_config_requires_json_type_equality():
    # this is a unit test
    full = completed_archive()
    assert full['settings']['config']['retrieval']['grade_threshold'] == 0.0
    full['evaluation']['result']['config']['retrieval']['grade_threshold'] = False
    with pytest.raises(archive.ArchiveError, match='evaluation.result.config'):
        archive.validate_archive(full)


def test_archive_version_requires_an_integer_type():
    # this is a unit test
    full = completed_archive()
    full['version'] = 1.0
    with pytest.raises(archive.ArchiveError, match='archive.version.*integer'):
        archive.validate_archive(full)


def test_broken_question_chunk_and_span_references_are_refused():
    # this is a unit test
    mutations = [
        (lambda full: full['evaluation']['result']['rows'][0].update(id=99),
         'rows.*99.*outside'),
        (lambda full: full['evaluation']['inspector']['dataset']['ground_truth']
         ['groundtruth_dataset'][0]['relevant_corpus_documents'][0]
         .update(corpus_document_id=999),
         'cites corpus_document_id 999'),
        (lambda full: full['evaluation']['inspector']['dataset']['ground_truth']
         ['groundtruth_dataset'][0]['relevant_corpus_documents'][0]['evidence'][0]
         .update(text='words the corpus never said'),
         'findable character for character'),
        (lambda full: full['evaluation']['inspector']['traces'][0]
         .update(question_id=999),
         'traces.*question_id.*999'),
        (lambda full: full['evaluation']['inspector']['traces'][0]
         ['trace']['candidates'][0].update(chunk_id='missing'), 'chunk_id.*missing'),
        (lambda full: full['evaluation']['inspector']['traces'][0]
         ['trace']['candidates'][0].update(gold_spans=[[0, 99]]), 'gold_spans'),
    ]
    for mutate, message in mutations:
        full = completed_archive()
        mutate(full)
        with pytest.raises(archive.ArchiveError, match=message):
            archive.validate_archive(full)


def test_dataset_ids_duplicate_ids_and_non_finite_metrics_are_refused():
    # this is a unit test
    full = completed_archive()
    full['settings']['config']['index']['dataset'] = ''
    full['evaluation']['result']['config']['index']['dataset'] = ''
    full['evaluation']['result']['dataset'] = 'diary-fa'
    full['evaluation']['inspector']['dataset']['id'] = 'diary-fa'
    full['evaluation']['inspector']['dataset']['corpus'][
        'corpus_dataset_metadata']['dataset'] = 'diary-fa'
    full['evaluation']['inspector']['dataset']['ground_truth'][
        'groundtruth_dataset_metadata']['corpus_ref']['dataset'] = 'diary-fa'
    assert archive.validate_archive(full) == full

    full = completed_archive()
    full['evaluation']['inspector']['dataset']['id'] = 'other'
    with pytest.raises(archive.ArchiveError, match='dataset.*other'):
        archive.validate_archive(full)

    full = completed_archive()
    full['evaluation']['inspector']['dataset']['ground_truth'][
        'groundtruth_dataset'].append(copy.deepcopy(
            full['evaluation']['inspector']['dataset']['ground_truth']
            ['groundtruth_dataset'][0]))
    with pytest.raises(archive.ArchiveError, match='duplicate.*1'):
        archive.validate_archive(full)

    full = completed_archive()
    full['evaluation']['result']['summary']['overall']['recall'] = float('nan')
    with pytest.raises(archive.ArchiveError, match='finite'):
        archive.validate_archive(full)


@pytest.mark.parametrize(('limits', 'mutate', 'message'), [
    ({'depth': 2},
     lambda full: full['evaluation']['inspector']['dataset']['corpus']
     .update(nested={'again': {'too_deep': True}}), 'depth'),
    ({'questions': 0}, lambda full: None, 'questions'),
    ({'chunks': 0}, lambda full: None, 'chunks'),
    ({'traces': 0}, lambda full: None, 'traces'),
    ({'candidates_per_trace': 0}, lambda full: None, 'candidates'),
    ({'list_items': 0}, lambda full: None, 'list'),
    ({'string_chars': 3}, lambda full: None, 'string'),
])
def test_structural_limits_are_enforced_without_large_allocations(
        limits, mutate, message):
    # this is a unit test
    full = completed_archive()
    mutate(full)
    with pytest.raises(archive.ArchiveError, match=message):
        archive.validate_archive(full, limits=limits)


# --- traces are evidence, and evidence may be absent -------------------------
# The format used to require one trace per row, which made an evaluation that
# was scored before traces were ever written down unarchivable: 166 real runs
# held rows, judged metrics and a selection, and no recording of what retrieval
# ranked. The rule relaxed in exactly one direction — trace ids are a subset of
# the row ids, ordered as the selection is — and these are the four ways of
# checking that it relaxed no further than that.


def test_an_archive_scored_with_no_trace_at_all_is_accepted():
    # this is a unit test
    """Rows and metrics with an empty trace list read "scored, trace not kept".

    The measurement is untouched — `selection.question_ids` still equals the
    row ids exactly — so nothing here is invented; what is missing is only the
    recording of how retrieval reached those rows.
    """
    full = completed_archive()
    full['evaluation']['inspector']['traces'] = []
    assert archive.validate_archive(full) == full


def test_an_archive_traced_for_only_some_of_its_rows_is_accepted():
    # this is a unit test
    """A subset is a subset at any size, and the surviving trace still resolves."""
    rung = examples.retrieved_rung()['archive']
    kept = rung['evaluation']['inspector']['traces'][0]
    rung['evaluation']['inspector']['traces'] = [copy.deepcopy(kept)]
    assert archive.validate_archive(rung) == rung
    assert len(rung['evaluation']['result']['rows']) == 2


def test_traces_that_reorder_or_outrun_the_selection_are_refused():
    # this is a unit test
    """Subset, not free-for-all: order is still the run's, and rows still bound it.

    A trace list that ran in a different order from the selection would be a
    quiet claim that the run asked its questions in that order, and a trace for
    a question the run never selected would be a reading with no row behind it.
    """
    rung = examples.retrieved_rung()['archive']
    rung['evaluation']['inspector']['traces'].reverse()
    with pytest.raises(archive.ArchiveError, match='traces.*must follow'):
        archive.validate_archive(rung)

    # The indexed rung asked nothing, so any trace at all is outside its rows.
    indexed = examples.indexed_rung()['archive']
    indexed['evaluation']['inspector']['traces'] = [
        copy.deepcopy(examples.TRACES[0])]
    with pytest.raises(archive.ArchiveError, match='traces.*outside'):
        archive.validate_archive(indexed)


def test_a_trace_that_is_present_is_still_held_to_the_archived_evidence():
    # this is a unit test
    """The relaxation is about absence only; a present trace loosened nothing.

    A candidate naming a chunk the archive does not carry, or quoting text that
    is not byte-equal to the chunk it names, is still refused — otherwise
    dropping the equality would have bought "rows without traces" at the price
    of "traces that cite whatever they like".
    """
    full = completed_archive()
    full['evaluation']['inspector']['traces'][0]['trace']['candidates'][0][
        'chunk_id'] = 'never-archived'
    with pytest.raises(archive.ArchiveError, match='chunk_id.*never-archived'):
        archive.validate_archive(full)

    full = completed_archive()
    candidate = full['evaluation']['inspector']['traces'][0]['trace'][
        'candidates'][0]
    candidate['text'] = candidate['text'] + ' '
    candidate['gold_spans'] = []
    with pytest.raises(archive.ArchiveError, match='text.*differs from archived'):
        archive.validate_archive(full)
