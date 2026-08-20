"""The lab's one end-to-end test: a panel payload through the real HTTP
surface, an index job, an evaluation job, a run file on disk, a ledger row
and a leaderboard group. Everything else that proves itself by running the
whole lab is rewritten as a direct call — see constraints.md.

Smoke set (`fixtures/corpus_groundtruth_datasets/smoke-mini.json`, 5 sessions, 6
questions) with `token-hash` (no model download) and the conftest-pinned
`fake` provider, so this stays fast."""
import json
import threading

from raglab.configuration import explainer_assembly as explain
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import run_evaluation as evaluate
from raglab.llm_backends import model_role_catalogue as models
from raglab.agents.extra_tools import leaderboard

from raglab.conftest import SMOKE_INDEX, _finished


def test_replacing_a_dataset_id_rebuilds_index_and_archive_evidence(
        client, monkeypatch, tmp_path):
    monkeypatch.setenv('RAGLAB_DATASETS', str(tmp_path / 'datasets'))
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'runs')
    datasets.forget()

    def payload(question_id, text):
        return {
            'dataset': {'id': 'archive-same-id', 'name': 'Archive replacement',
                        'language': 'en'},
            'sessions': [{'session_id': 's1', 'date': '2026-08-19',
                          'messages': [{'role': 'user', 'content': text}]}],
            'questions': [{'id': question_id, 'type': 'single-hop',
                           'difficulty': 'easy', 'answerable': True,
                           'question': 'What was recorded?', 'answer': text,
                           'evidence': [{'session_id': 's1',
                                         'message_indices': [0], 'quote': text}]}],
        }

    old = payload('old-question', 'old indexed evidence')
    assert client.post('/api/datasets', json=old).status_code == 200
    index = {'dataset': 'archive-same-id', 'chunker': 'session',
             'embedder': 'token-hash'}
    built = client.post('/api/indexes', json={'index': index})
    assert built.status_code == 202, built.text
    assert _finished(client, built.json()['job_id'])['state'] == 'done'

    replacement = payload('replacement-question', 'replacement archive evidence')
    assert client.post('/api/datasets', json=replacement).status_code == 200
    started = client.post('/api/evaluations', json={
        'index': index, 'retrieval': {'k': 1, 'reranker': 'none', 'grader': 'none'},
        'generation': {'answerer': 'extractive'}, 'ragas_mode': 'off', 'limit': 1})
    assert started.status_code == 202, started.text
    job = _finished(client, started.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    result = job['result']
    evidence = result['archive_evidence']['inspector']['dataset']
    archived_text = evidence['corpus']['sessions'][0]['messages'][0]['content']
    indexed_text = result['chunks_by_session'][0]['chunks'][0]['text']
    assert archived_text == 'replacement archive evidence'
    assert 'replacement archive evidence' in indexed_text
    assert result['index']['reused'] is False


def test_dataset_replacement_waits_for_the_evaluation_snapshot_and_index(
        client, monkeypatch, tmp_path):
    monkeypatch.setenv('RAGLAB_DATASETS', str(tmp_path / 'datasets'))
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'runs')
    datasets.forget()

    def payload(question_id, text):
        return {
            'dataset': {'id': 'archive-race-id', 'name': 'Archive race',
                        'language': 'en'},
            'sessions': [{'session_id': 's1', 'date': '2026-08-19',
                          'messages': [{'role': 'user', 'content': text}]}],
            'questions': [{'id': question_id, 'type': 'single-hop',
                           'difficulty': 'easy', 'answerable': True,
                           'question': 'What was recorded?', 'answer': text,
                           'evidence': [{'session_id': 's1',
                                         'message_indices': [0], 'quote': text}]}],
        }

    original = payload('old-question', 'old evaluation evidence')
    replacement = payload('replacement-question', 'replacement evaluation evidence')
    assert client.post('/api/datasets', json=original).status_code == 200
    index = {'dataset': 'archive-race-id', 'chunker': 'session',
             'embedder': 'token-hash'}
    entered_index = threading.Event()
    release_index = threading.Event()
    entered_import = threading.Event()
    release_import = threading.Event()
    replacement_done = threading.Event()
    replacement_response = {}

    from raglab.rag_components.indexing.index_builder_registry import IndexRegistry

    original_get = IndexRegistry.get

    def paused_get(self, cfg, *args, **kwargs):
        if cfg.dataset == 'archive-race-id' and not entered_index.is_set():
            entered_index.set()
            assert release_index.wait(2), 'test did not release evaluation index work'
        return original_get(self, cfg, *args, **kwargs)

    original_import = datasets.import_dataset

    def paused_import(body):
        entered_import.set()
        assert release_import.wait(2), 'test did not release dataset replacement'
        return original_import(body)

    monkeypatch.setattr(IndexRegistry, 'get', paused_get)
    monkeypatch.setattr(datasets, 'import_dataset', paused_import)

    started = client.post('/api/evaluations', json={
        'index': index, 'retrieval': {'k': 1, 'reranker': 'none', 'grader': 'none'},
        'generation': {'answerer': 'extractive'}, 'ragas_mode': 'off', 'limit': 1})
    assert started.status_code == 202, started.text
    assert entered_index.wait(1), 'evaluation never acquired its dataset index'

    def replace_dataset():
        try:
            replacement_response['response'] = client.post('/api/datasets',
                                                             json=replacement)
        finally:
            replacement_done.set()

    thread = threading.Thread(target=replace_dataset)
    thread.start()
    try:
        assert not entered_import.wait(0.2), (
            'replacement entered dataset import while evaluation still owned '
            'its evidence snapshot and index acquisition')
        release_index.set()
        evaluated = _finished(client, started.json()['job_id'])
        assert evaluated['state'] == 'done', evaluated.get('error')
        assert entered_import.wait(1), 'replacement did not proceed after evaluation'
        release_import.set()
        thread.join(2)
        assert not thread.is_alive(), 'replacement request did not finish'
    finally:
        release_index.set()
        release_import.set()
        thread.join(2)

    assert replacement_done.is_set()
    assert replacement_response['response'].status_code == 200
    rebuilt = client.post('/api/indexes', json={'index': index})
    assert rebuilt.status_code == 202, rebuilt.text
    rebuilt_job = _finished(client, rebuilt.json()['job_id'])
    assert rebuilt_job['state'] == 'done', rebuilt_job.get('error')
    assert rebuilt_job['result']['reused'] is False
    assert 'replacement evaluation evidence' in \
        rebuilt_job['result']['chunks_by_session'][0]['chunks'][0]['text']


def test_dataset_replacement_waits_for_a_standalone_index_build(
        client, monkeypatch, tmp_path):
    monkeypatch.setenv('RAGLAB_DATASETS', str(tmp_path / 'datasets'))
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'runs')
    datasets.forget()

    def payload(question_id, text):
        return {
            'dataset': {'id': 'archive-build-race', 'name': 'Archive build race',
                        'language': 'en'},
            'sessions': [{'session_id': 's1', 'date': '2026-08-19',
                          'messages': [{'role': 'user', 'content': text}]}],
            'questions': [{'id': question_id, 'type': 'single-hop',
                           'difficulty': 'easy', 'answerable': True,
                           'question': 'What was recorded?', 'answer': text,
                           'evidence': [{'session_id': 's1',
                                         'message_indices': [0], 'quote': text}]}],
        }

    original = payload('old-question', 'old standalone evidence')
    replacement = payload('replacement-question', 'replacement standalone evidence')
    assert client.post('/api/datasets', json=original).status_code == 200
    index = {'dataset': 'archive-build-race', 'chunker': 'session',
             'embedder': 'token-hash'}
    entered_index = threading.Event()
    release_index = threading.Event()
    entered_import = threading.Event()
    release_import = threading.Event()
    replacement_done = threading.Event()
    replacement_response = {}

    from raglab.rag_components.indexing.index_builder_registry import IndexRegistry

    original_get = IndexRegistry.get

    def paused_get(self, cfg, *args, **kwargs):
        if cfg.dataset == 'archive-build-race' and not entered_index.is_set():
            entered_index.set()
            assert release_index.wait(2), 'test did not release standalone index work'
        return original_get(self, cfg, *args, **kwargs)

    original_import = datasets.import_dataset

    def paused_import(body):
        entered_import.set()
        assert release_import.wait(2), 'test did not release dataset replacement'
        return original_import(body)

    monkeypatch.setattr(IndexRegistry, 'get', paused_get)
    monkeypatch.setattr(datasets, 'import_dataset', paused_import)

    started = client.post('/api/indexes', json={'index': index})
    assert started.status_code == 202, started.text
    assert entered_index.wait(1), 'standalone build never acquired its dataset index'

    def replace_dataset():
        try:
            replacement_response['response'] = client.post('/api/datasets',
                                                             json=replacement)
        finally:
            replacement_done.set()

    thread = threading.Thread(target=replace_dataset)
    thread.start()
    try:
        assert not entered_import.wait(0.2), (
            'replacement entered dataset import while standalone index build '
            'still owned registry acquisition')
        release_index.set()
        built = _finished(client, started.json()['job_id'])
        assert built['state'] == 'done', built.get('error')
        assert entered_import.wait(1), (
            'replacement did not proceed after standalone index build')
        release_import.set()
        thread.join(2)
        assert not thread.is_alive(), 'replacement request did not finish'
    finally:
        release_index.set()
        release_import.set()
        thread.join(2)

    assert replacement_done.is_set()
    assert replacement_response['response'].status_code == 200
    started = client.post('/api/evaluations', json={
        'index': index, 'retrieval': {'k': 1, 'reranker': 'none', 'grader': 'none'},
        'generation': {'answerer': 'extractive'}, 'ragas_mode': 'off', 'limit': 1})
    assert started.status_code == 202, started.text
    evaluated = _finished(client, started.json()['job_id'])
    assert evaluated['state'] == 'done', evaluated.get('error')
    result = evaluated['result']
    evidence = result['archive_evidence']['inspector']['dataset']
    assert evidence['corpus']['sessions'][0]['messages'][0]['content'] == \
        'replacement standalone evidence'
    assert 'replacement standalone evidence' in \
        result['chunks_by_session'][0]['chunks'][0]['text']
    assert result['index']['reused'] is False


def test_the_lab_runs_one_experiment_end_to_end(client, tmp_path, monkeypatch):
    # this is an end-to-end test
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'runs')

    index_cfg = dict(SMOKE_INDEX)

    # 1. build the index over the smoke set.
    built = client.post('/api/indexes', json={'index': index_cfg})
    assert built.status_code == 202, built.text
    build_job = _finished(client, built.json()['job_id'])
    assert build_job['state'] == 'done', build_job.get('error')
    build_result = build_job['result']
    assert build_result['chunks'] == 5, 'the smoke set is five sessions'
    assert sum(len(group['chunks'])
              for group in build_result['chunks_by_session']) == \
        build_result['leaves']

    # 2. evaluate three of its six questions — answerer llm (fake), key-facts
    # judge on, ragas off.
    started = client.post('/api/evaluations', json={
        'index': index_cfg, 'retrieval': {'k': 3},
        'generation': {'answerer': 'llm', 'key_facts_judge': True},
        'ragas_mode': 'off', 'limit': 3, 'label': 'e2e-smoke'})
    assert started.status_code == 202, started.text
    run_job = _finished(client, started.json()['job_id'])
    assert run_job['state'] == 'done', run_job.get('error')
    result = run_job['result']
    assert len(result['rows']) == 3
    assert result['summary']['n_questions'] == 3
    selection = result['selection']
    assert selection['n'] == 3
    assert len(selection['question_ids']) == 3
    evidence = result['archive_evidence']
    assert evidence['execution']['provider'] == 'fake'
    assert set(evidence['execution']['models']) == {
        role.key for role in models.ROLES}
    assert evidence['metric_catalogue'] == explain.measures()
    assert evidence['inspector']['dataset']['id'] == result['dataset']
    assert evidence['inspector']['dataset']['corpus']['sessions']
    assert evidence['inspector']['dataset']['ground_truth']['questions']
    assert evidence['inspector']['chunks_by_session'] == result['chunks_by_session']
    assert evidence['inspector']['summaries'] == result['summaries']
    assert evidence['inspector']['traces'] == result['traces']
    assert [row['question_id'] for row in evidence['inspector']['traces']] \
        == result['selection']['question_ids']
    saved = json.loads((evaluate.RUNS_DIR / f"{result['run_id']}.json").read_text())
    assert 'archive_evidence' not in saved
    assert 'traces' not in saved

    # 3. exactly one new file appeared in the (conftest-redirected) runs
    # dir, and it round-trips through `evaluate.list_runs`.
    run_files = list((tmp_path / 'runs').glob('*.json'))
    assert [path.stem for path in run_files] == [result['run_id']]
    listed = evaluate.list_runs()
    assert [row['run_id'] for row in listed] == [result['run_id']]

    # 4. the ledger: index row and run row, newest first, the resolved
    # backend named on the row that actually called a model.
    rows = client.get('/api/experiments').json()['experiments']
    assert [row['kind'] for row in rows[:2]] == ['run', 'index']
    run_row, build_row = rows[0], rows[1]
    assert run_row['experiment_id'] == result['run_id']
    assert run_row['provider'] == 'fake'
    assert run_row['dataset'] == 'smoke-mini'
    # A build's row carries index config and nothing else — no chat model
    # is involved in chunking, so its provider stays blank.
    assert build_row['provider'] == ''
    assert build_row['n_questions'] == 0
    assert build_row['dataset'] == 'smoke-mini'

    # 5. the leaderboard groups the run into the smoke-set group; ragas is
    # off, so nothing measured a decision score and no rank number applies.
    groups = leaderboard.group(listed)
    assert len(groups) == 1
    group = groups[0]
    assert group.dataset == 'smoke-mini'
    assert group.rows[0]['run_id'] == result['run_id']
    assert leaderboard.verdict(group) == 'unknown', \
        'no decision score was measured, so nothing can be ranked'
    rank_column = leaderboard.markdown(groups).splitlines()
    data_row = next(line for line in rank_column if result['run_id'] in line)
    assert data_row.strip().startswith('| —'), 'no rank number on an unjudged row'
