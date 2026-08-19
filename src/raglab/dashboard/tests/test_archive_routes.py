"""Import routes preserve archives without creating normal run artifacts."""
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation.tests.archive_examples import completed_archive


def test_completed_import_is_persisted_once_without_a_run_file(client, tmp_path):
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
    assert len(client.get('/api/experiments').json()) == 1
    assert list(evaluate.RUNS_DIR.glob('*.json')) == before_runs
    assert client.get('/api/evaluations').json() == before_board

    changed = completed_archive()
    changed['evaluation']['result']['label'] = 'preview only'
    changed['settings']['config']['label'] = 'preview only'
    assert client.post('/api/imported-archives', json=changed).json()['database'] \
        == 'existing'
    assert client.get('/api/imported-archives/imported-run-001').json() == changed
    stored = client.get('/api/experiments/imported-run-001').json()
    assert stored['label'] == 'imported experiment'
    assert stored['detail'] == full


def test_settings_only_is_not_a_persistence_route_payload(client):
    full = completed_archive()
    settings_only = {key: full[key] for key in ('format', 'version', 'settings')}
    response = client.post('/api/imported-archives', json=settings_only)
    assert response.status_code == 400
    assert 'completed archive is required' in response.json()['detail']
    assert client.get('/api/experiments').json() == []


def test_clear_only_returns_inspector_to_live(client):
    client.post('/api/imported-archives', json=completed_archive())
    assert client.delete('/api/imported-archives/active').json() == {
        'archive_id': None}
    assert client.get('/api/imported-archives/active').json()['archive_id'] is None
    assert client.get('/api/imported-archives/imported-run-001').status_code == 200
