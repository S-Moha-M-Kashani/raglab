"""Import routes preserve archives without creating normal run artifacts."""
import sqlite3

from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.evaluation.tests.archive_examples import completed_archive


def test_completed_import_is_persisted_once_without_a_run_file(
        client, monkeypatch, tmp_path):
    # this is an integration test
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'completed-import.db'))
    before_runs = list(evaluate.RUNS_DIR.glob('*.json'))
    before_board = client.get('/api/evaluations').json()
    full = completed_archive()

    first = client.post('/api/imported-archives', json=full)
    assert first.status_code == 200
    assert first.json() == {'archive_id': 'imported-run-001',
                            'database': 'created'}
    assert client.get('/api/imported-archives/active').json() == {
        'archive_id': 'imported-run-001', 'source': 'import'}
    assert client.get('/api/imported-archives/imported-run-001').json() == full
    experiments = client.get('/api/experiments').json()['experiments']
    assert len(experiments) == 1
    assert experiments[0]['experiment_id'] == 'imported-run-001'
    assert experiments[0]['label'] == 'imported experiment'
    assert list(evaluate.RUNS_DIR.glob('*.json')) == before_runs
    assert client.get('/api/evaluations').json() == before_board

    changed = completed_archive()
    changed['evaluation']['result']['label'] = 'preview only'
    changed['settings']['config']['label'] = 'preview only'
    changed['evaluation']['result']['config']['label'] = 'preview only'
    assert client.post('/api/imported-archives', json=changed).json()['database'] \
        == 'existing'
    assert client.get('/api/imported-archives/imported-run-001').json() == changed
    stored = client.get('/api/experiments/imported-run-001').json()
    assert stored['label'] == 'imported experiment'
    assert stored['detail'] == full


def test_settings_only_is_not_a_persistence_route_payload(client, monkeypatch, tmp_path):
    # this is an integration test
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'settings-only-import.db'))
    full = completed_archive()
    settings_only = {key: full[key] for key in ('format', 'version', 'settings')}
    response = client.post('/api/imported-archives', json=settings_only)
    assert response.status_code == 400
    assert 'completed archive is required' in response.json()['detail']
    assert client.get('/api/experiments').json()['experiments'] == []


def test_a_failed_archive_insert_leaves_no_row_or_active_archive(
        client, monkeypatch, tmp_path):
    # this is an integration test
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'failed-import.db'))
    client.delete('/api/imported-archives/active')

    def refuse_insert(*_args, **_kwargs):
        raise sqlite3.OperationalError('database is unavailable')

    monkeypatch.setattr(ledger, 'insert_archive', refuse_insert)
    response = client.post('/api/imported-archives',
                           json=completed_archive('failed-import'))
    assert response.status_code == 500
    assert 'database' in response.json()['detail']
    assert client.get('/api/experiments').json()['experiments'] == []
    assert client.get('/api/imported-archives/active').json() == {
        'archive_id': None, 'source': None}


def test_clear_only_returns_inspector_to_live(client):
    # this is an integration test
    client.post('/api/imported-archives', json=completed_archive())
    assert client.delete('/api/imported-archives/active').json() == {
        'archive_id': None}
    assert client.get('/api/imported-archives/active').json()['archive_id'] is None
    assert client.get('/api/imported-archives/imported-run-001').status_code == 200
