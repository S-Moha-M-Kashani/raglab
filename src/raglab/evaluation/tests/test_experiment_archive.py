"""Contracts for portable RAG Lab experiment archives."""
import copy
import json

import pytest

from raglab.evaluation import experiment_archive as archive
from raglab.evaluation.tests.archive_examples import completed_archive


def test_settings_only_and_completed_archives_validate_without_fabricated_results():
    full = completed_archive()
    settings_only = {key: copy.deepcopy(full[key])
                     for key in ('format', 'version', 'settings')}
    assert archive.validate_archive(settings_only) == settings_only
    assert 'evaluation' not in settings_only
    assert archive.validate_archive(full) == full


def test_stage_results_are_derived_from_the_metric_catalogue():
    full = completed_archive()
    expected = full['evaluation']['stage_results']
    assert archive.stage_results(full['evaluation']['result'],
                                 full['evaluation']['metric_catalogue']) == expected


@pytest.mark.parametrize('key', ['api_key', 'openrouterKey', 'password',
                                 'client_secret', 'access_token', 'authorization'])
def test_credentials_are_rejected_at_any_depth(key):
    full = completed_archive()
    full['evaluation']['inspector']['dataset']['corpus'][key] = 'never persist'
    with pytest.raises(archive.ArchiveError, match=key):
        archive.validate_archive(full)


def test_archive_size_config_and_stage_mismatches_are_refused():
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
    payload = completed_archive()
    encoded_size = len(json.dumps(payload, ensure_ascii=False,
                                  allow_nan=False).encode('utf-8'))
    monkeypatch.setattr(archive, 'MAX_BYTES', encoded_size - 1)
    with pytest.raises(archive.ArchiveError, match='archive.*(encoded|size|MiB)'):
        archive.validate_archive(payload)


def test_result_config_requires_json_type_equality():
    full = completed_archive()
    assert full['settings']['config']['retrieval']['grade_threshold'] == 0.0
    full['evaluation']['result']['config']['retrieval']['grade_threshold'] = False
    with pytest.raises(archive.ArchiveError, match='evaluation.result.config'):
        archive.validate_archive(full)


def test_archive_version_requires_an_integer_type():
    full = completed_archive()
    full['version'] = 1.0
    with pytest.raises(archive.ArchiveError, match='archive.version.*integer'):
        archive.validate_archive(full)


def test_broken_question_chunk_and_span_references_are_refused():
    mutations = [
        (lambda full: full['evaluation']['result']['rows'][0].update(id='missing'),
         'rows.*missing'),
        (lambda full: full['evaluation']['inspector']['dataset']['ground_truth']
         ['questions'][0]['evidence'][0].update(session_id='missing-session'),
         'evidence.*session_id.*missing-session'),
        (lambda full: full['evaluation']['inspector']['dataset']['ground_truth']
         ['questions'][0]['evidence'][0].update(message_indices=[1]),
         'evidence.*message_indices.*1'),
        (lambda full: full['evaluation']['inspector']['traces'][0]
         .update(question_id='missing-question'),
         'traces.*question_id.*missing-question'),
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
    full = completed_archive()
    full['settings']['config']['index']['dataset'] = ''
    full['evaluation']['result']['config']['index']['dataset'] = ''
    full['evaluation']['result']['dataset'] = 'diary-fa'
    full['evaluation']['inspector']['dataset']['id'] = 'diary-fa'
    assert archive.validate_archive(full) == full

    full = completed_archive()
    full['evaluation']['inspector']['dataset']['id'] = 'other'
    with pytest.raises(archive.ArchiveError, match='dataset.*other'):
        archive.validate_archive(full)

    full = completed_archive()
    full['evaluation']['inspector']['dataset']['ground_truth']['questions'] \
        .append(copy.deepcopy(full['evaluation']['inspector']['dataset']
                              ['ground_truth']['questions'][0]))
    with pytest.raises(archive.ArchiveError, match='duplicate.*q1'):
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
    full = completed_archive()
    mutate(full)
    with pytest.raises(archive.ArchiveError, match=message):
        archive.validate_archive(full, limits=limits)
