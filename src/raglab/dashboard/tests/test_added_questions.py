"""A question added from a recorded Inspector is its own durable experiment."""
import threading
import pytest
from fastapi.testclient import TestClient

from raglab.configuration.lab_config import IndexConfig, LabConfig
from raglab.conftest import SMOKE_INDEX, _finished, drain_jobs
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.evaluation.tests.archive_examples import completed_archive


RUN = {
    'index': dict(SMOKE_INDEX),
    'retrieval': {'k': 3},
    'generation': {'answerer': 'llm', 'fact_judge': True},
    'ragas_mode': 'off',
    'limit': 1,
    'label': 'recorded-parent',
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    # this is an integration test fixture
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    monkeypatch.setenv('RAGLAB_CORPORA_DB', str(tmp_path / 'corpora.db'))
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'runs')
    from raglab.dashboard.panel_server import create_app
    return TestClient(create_app())


def _recorded_parent(client) -> tuple[str, str]:
    started = client.post('/api/evaluations', json=RUN)
    assert started.status_code == 202, started.text
    finished = _finished(client, started.json()['job_id'])
    assert finished['state'] == 'done', finished.get('error')
    question = client.get('/api/questions?dataset=smoke-mini').json()['questions'][0]
    return finished['result']['run_id'], str(question['id'])


def _record(experiment_id: str, config: dict, *, detail: dict | None = None):
    job = {'id': experiment_id, 'kind': 'run', 'config': config,
           'result': detail or {'config': config, 'dataset': 'smoke-mini'}}
    ledger.record(job, 'done')


def test_added_question_is_a_linked_durable_row_and_never_rewrites_parent(client):
    # this is an end-to-end test
    parent_id, question_id = _recorded_parent(client)
    with ledger.connect() as db:
        original_bytes = db.execute(
            'SELECT detail FROM experiments WHERE experiment_id = ?',
            (parent_id,)).fetchone()['detail']

    try:
        started = client.post(f'/api/experiments/{parent_id}/questions',
                              json={'question_id': question_id})
        assert started.status_code == 202, started.text
        job = _finished(client, started.json()['job_id'])
    finally:
        drain_jobs(client)

    assert job['state'] == 'done', job.get('error')
    result = job['result']
    assert result['label'] == f'adds {question_id} to {parent_id}'
    assert result['annotates'] == parent_id
    assert result['question_id'] == question_id
    assert result['selection'] == {'n': 1, 'question_ids': [question_id]}
    assert len(result['traces']) == len(result['rows']) == 1
    assert result['models']
    assert 'run_id' not in result, 'an addition is ledger-only, never a sweep run'

    added = ledger.experiment(job['id'])
    assert added['kind'] == 'question'
    assert added['state'] == 'done'
    assert added['n_questions'] == 1
    assert added['decision'] is None
    assert added['detail']['annotates'] == parent_id

    with ledger.connect() as db:
        after_bytes = db.execute(
            'SELECT detail FROM experiments WHERE experiment_id = ?',
            (parent_id,)).fetchone()['detail']
    assert after_bytes == original_bytes, 'the original experiment is insert-only'

    listed = client.get(f'/api/experiments/{parent_id}/questions')
    assert listed.status_code == 200
    assert [row['experiment_id'] for row in listed.json()['questions']] == [job['id']]

    judged = LabConfig(index=IndexConfig(**SMOKE_INDEX)).to_dict()
    _record('judged-row', judged, detail={
        'config': judged, 'dataset': 'smoke-mini',
        'selection': {'n': 1, 'question_ids': ['judge-q']},
        'ragas': {'decision': 0.75},
    })
    board = client.get('/api/leaderboard?dataset=smoke-mini').json()['rows']
    row = next(row for row in board if row['experiment_id'] == job['id'])
    assert row['kind'] == 'question'
    assert row['label'] == f'adds {question_id} to {parent_id}'
    assert row['decision'] is None
    assert [row['experiment_id'] for row in board].index('judged-row') \
        < [row['experiment_id'] for row in board].index(job['id'])


def test_added_question_refuses_unknown_parent_question_config_and_retired_knobs(client):
    # this is an integration test
    parent_id, question_id = _recorded_parent(client)
    unknown = client.post('/api/experiments/no-such-experiment/questions',
                          json={'question_id': question_id})
    assert unknown.status_code == 404
    assert 'unknown experiment' in unknown.json()['detail']

    missing_question = client.post(f'/api/experiments/{parent_id}/questions',
                                   json={'question_id': 'not-in-smoke'})
    assert missing_question.status_code == 404
    assert 'unknown question id' in missing_question.json()['detail']

    _record('no-config', {}, detail={})
    no_config = client.post('/api/experiments/no-config/questions',
                            json={'question_id': question_id})
    assert no_config.status_code == 409
    assert 'config' in no_config.json()['detail']

    ledger.insert_archive(completed_archive('archive-shaped'))
    archive_shaped = client.post('/api/experiments/archive-shaped/questions',
                                 json={'question_id': question_id})
    assert archive_shaped.status_code == 409
    assert 'archive' in archive_shaped.json()['detail']

    retired = LabConfig(index=IndexConfig(**SMOKE_INDEX)).to_dict()
    retired['generation']['retired_fact_judge'] = True
    _record('retired-knob', retired)
    retired_response = client.post('/api/experiments/retired-knob/questions',
                                   json={'question_id': question_id})
    assert retired_response.status_code == 409
    assert 'retired_fact_judge' in retired_response.json()['detail']

    invalid = LabConfig(index=IndexConfig(**SMOKE_INDEX)).to_dict()
    invalid['retrieval']['reranker'] = 'not-a-reranker'
    invalid['provider'] = 'fake'
    _record('unservable-config', invalid)
    screened = client.post('/api/experiments/unservable-config/questions',
                           json={'question_id': question_id})
    assert screened.status_code == 400
    assert 'unknown reranker' in screened.json()['detail']


def test_added_question_refuses_retired_top_level_knobs_and_malformed_scalars(
        client):
    # this is an integration test
    _parent_id, question_id = _recorded_parent(client)

    retired = LabConfig(index=IndexConfig(**SMOKE_INDEX)).to_dict()
    retired['retired_mode'] = 'old-default'
    _record('retired-top-level', retired)
    retired_response = client.post(
        '/api/experiments/retired-top-level/questions',
        json={'question_id': question_id})
    assert retired_response.status_code == 409
    assert retired_response.json()['detail'] == (
        'retired_mode is not a knob this lab reads any more')

    malformed = LabConfig(index=IndexConfig(**SMOKE_INDEX)).to_dict()
    malformed['retrieval']['k'] = []
    _record('malformed-k', malformed)
    malformed_response = client.post('/api/experiments/malformed-k/questions',
                                      json={'question_id': question_id})
    assert malformed_response.status_code == 409
    assert malformed_response.json()['detail'] == (
        'recorded retrieval.k has malformed type')


def test_added_question_refuses_an_unknown_recorded_provider(client):
    # this is an integration test
    _parent_id, question_id = _recorded_parent(client)
    recorded = LabConfig(index=IndexConfig(**SMOKE_INDEX)).to_dict()
    recorded['provider'] = 'retired-provider'
    _record('unknown-provider', recorded)

    response = client.post('/api/experiments/unknown-provider/questions',
                           json={'question_id': question_id})
    assert response.status_code == 409
    assert response.json()['detail'] == (
        "unknown recorded provider: 'retired-provider'")


@pytest.mark.parametrize(
    ('experiment_id', 'mutate', 'reason'),
    [
        ('missing-recorded-provider',
         lambda config: config.pop('provider'),
         'recorded provider is missing'),
        ('missing-recorded-group',
         lambda config: config.pop('retrieval'),
         'recorded retrieval is missing'),
        ('missing-recorded-knob',
         lambda config: config['generation'].pop('answerer'),
         'recorded generation.answerer is missing'),
    ],
)
def test_added_question_refuses_incomplete_recorded_settings(
        client, experiment_id, mutate, reason):
    # this is an integration test
    """A recorded row may not inherit a config value from this live lab."""
    _parent_id, question_id = _recorded_parent(client)
    recorded = (LabConfig(index=IndexConfig(**SMOKE_INDEX)).to_dict()
                | {'provider': 'fake'})
    mutate(recorded)
    _record(experiment_id, recorded)

    response = client.post(f'/api/experiments/{experiment_id}/questions',
                           json={'question_id': question_id})

    assert response.status_code == 409
    assert response.json()['detail'] == reason


def test_added_question_refuses_while_the_lab_is_busy(client, monkeypatch):
    # this is an integration test
    parent_id, question_id = _recorded_parent(client)
    from raglab.dashboard import panel_server

    entered, release = threading.Event(), threading.Event()
    original_get = panel_server.IndexRegistry.get

    def wait_before_index(self, *args, **kwargs):
        entered.set()
        assert release.wait(timeout=2), 'the test must release the build'
        return original_get(self, *args, **kwargs)

    monkeypatch.setattr(panel_server.IndexRegistry, 'get', wait_before_index)
    build = client.post('/api/indexes', json={'index': dict(SMOKE_INDEX)})
    assert build.status_code == 202
    assert entered.wait(timeout=2)
    try:
        refused = client.post(f'/api/experiments/{parent_id}/questions',
                              json={'question_id': question_id})
    finally:
        release.set()
        drain_jobs(client)
    assert refused.status_code == 409
    assert 'job is already running' in refused.json()['detail']
