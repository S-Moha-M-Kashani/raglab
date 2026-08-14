"""The experiment ledger (raglab.db) — one row per job, recording every
build, retrieval and evaluation the lab finishes."""
import sqlite3
from pathlib import Path

from raglab import config

from conftest import _finished


# --- the experiment ledger (raglab.db) -------------------------------------

# A real SQLite file on a temp path.
def test_every_experiment_the_lab_runs_lands_in_the_ledger(client, tmp_path,
                                                           monkeypatch):
    """Three experiments, three rows: the ledger records every job the lab
    *finishes*, which is what makes "what have I already tried?" a
    question with an answer after the process that tried it is gone."""
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    index = {'chunker': 'session', 'embedder': 'ascii-hash'}
    retrieval_cfg = {'retriever': 'hybrid-rrf', 'reranker': 'none',
                     'grader': 'none', 'k': 3, 'rerank_depth': 20,
                     'time_filter': False, 'multi_query': False}

    built = client.post('/api/indexes', json={'index': index})
    assert _finished(client, built.json()['job_id'])['state'] == 'done'
    got = client.post('/api/retrievals', json={
        'index': index, 'retrieval': retrieval_cfg, 'limit': 2,
        'balance': 'stride'})
    assert _finished(client, got.json()['job_id'])['state'] == 'done'
    ran = client.post('/api/evaluations', json={
        'index': index, 'retrieval': retrieval_cfg,
        'generation': {'answerer': 'extractive'}, 'label': 'the ledger',
        'limit': 2, 'balance': 'stride', 'ragas_mode': 'off'})
    run_job = _finished(client, ran.json()['job_id'], timeout=120.0)
    assert run_job['state'] == 'done', run_job.get('error')

    rows = client.get('/api/experiments').json()['experiments']
    # Newest first, like every other listing the lab serves.
    assert [row['kind'] for row in rows[:3]] == ['run', 'retrieve', 'index']
    evaluation, retrieved, build = rows[0], rows[1], rows[2]

    # An evaluation is identified by its run id, never by its job id: the ledger
    # row and the JSON file the leaderboard reads are then the same measurement,
    # each checkable against the other.
    assert evaluation['experiment_id'] == run_job['result']['run_id']
    assert evaluation['label'] == 'the ledger'
    assert evaluation['n_questions'] == 2 and evaluation['state'] == 'done'
    assert evaluation['seconds'] > 0
    # Recorded before the job goes terminal, so a follower that sees 'done' can
    # never look for the row and miss it.
    assert evaluation['started_at']
    # `ragas_mode='off'` judged nothing, and an unjudged row carries no score
    # rather than a zero — the rule the leaderboard already keeps, because a
    # fabricated 0.0 would rank below every real row and read as a measurement.
    assert evaluation['decision'] is None
    assert evaluation['decision_stderr'] is None

    # A retrieval scored nothing either, but it did choose a sample, and which
    # questions it covered is the whole point of having run it.
    assert retrieved['n_questions'] == 2 and retrieved['decision'] is None
    # An index build has no sample at all: it is a fact about the corpus.
    assert build['n_questions'] == 0 and build['decision'] is None

    # Every row says which index it was over, so the panel's table needs no
    # per-kind branch to render one.
    for row in rows[:3]:
        assert row['chunker'] == 'session'
        assert row['embedder'] == 'ascii-hash'
        assert row['experiment_id']

    # But a build's row stops there. Its job config carries a whole LabConfig, so
    # the retrieval group is populated with defaults the panel happened to be
    # showing and no part of a build reads — recorded, they would put a reranker
    # on a row that never retrieved anything, and a reader comparing rows would
    # attribute a chunk count to it. Same reason `provider` is blank: no chat
    # model is involved in chunking, not even for contextual headers.
    assert build['retriever'] == '' and build['reranker'] == ''
    assert build['grader'] == '' and build['answerer'] == ''
    assert build['provider'] == ''
    # The two that did retrieve say so, and say where the calls went — the one
    # field that separates a measurement from a rehearsal.
    assert retrieved['retriever'] == 'hybrid-rrf'
    assert evaluation['answerer'] == 'extractive'
    assert evaluation['provider'] == 'fake', 'the resolved backend, not the ask'

    assert (tmp_path / 'raglab.db').exists(), 'the ledger is one SQLite file'


def test_the_ledger_explains_a_row_without_storing_the_corpus(client, tmp_path,
                                                              monkeypatch):
    """"With all the details" means the details of the *experiment*. The
    chunk text is not one: it is byte-identical across every experiment
    sharing a fingerprint and rebuilt exactly by re-running the build, so
    storing it per row would store the whole corpus once per experiment."""
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    index = {'chunker': 'session', 'embedder': 'ascii-hash'}
    retrieval_cfg = {'retriever': 'hybrid-rrf', 'reranker': 'none',
                     'grader': 'none', 'k': 3, 'rerank_depth': 20,
                     'time_filter': False, 'multi_query': False}
    ran = client.post('/api/evaluations', json={
        'index': index, 'retrieval': retrieval_cfg,
        'generation': {'answerer': 'extractive'}, 'limit': 2,
        'balance': 'stride', 'ragas_mode': 'off'})
    job = _finished(client, ran.json()['job_id'], timeout=120.0)
    assert job['state'] == 'done', job.get('error')
    run_id = job['result']['run_id']

    stored = client.get(f'/api/experiments/{run_id}').json()
    assert stored['experiment_id'] == run_id
    detail = stored['detail']
    assert detail['config']['index']['chunker'] == 'session'
    assert detail['config']['retrieval']['k'] == 3
    assert detail['summary'] == job['result']['summary']
    assert [row['id'] for row in detail['rows']] == \
        [row['id'] for row in job['result']['rows']]
    assert detail['selection']['n'] == 2
    assert 'chunks_by_session' not in detail

    # A retrieval's detail is its traces: the ranks at every step are the only
    # thing it produced, so dropping them would leave a row that records that
    # something ran and nothing about what it found.
    got = client.post('/api/retrievals', json={
        'index': index, 'retrieval': retrieval_cfg, 'limit': 2,
        'balance': 'stride'})
    retrieval_job = _finished(client, got.json()['job_id'])
    assert retrieval_job['state'] == 'done', retrieval_job.get('error')
    newest = client.get('/api/experiments').json()['experiments'][0]
    kept = client.get(f"/api/experiments/{newest['experiment_id']}").json()['detail']
    assert kept['questions'][0]['trace']['candidates']
    assert 'chunks_by_session' not in kept

    assert client.get('/api/experiments/no-such-experiment').status_code == 404


def test_a_ledger_that_cannot_be_written_does_not_lose_the_experiment(
        client, monkeypatch):
    """A judged run costs hours, and an unwritable database must not be
    able to turn one into an error the panel reports over a result nobody
    can read — the same call `ragas_eval.JudgeWatch` makes about its
    progress counter."""
    from raglab import ledger

    def refuse(*_args, **_kwargs):
        raise sqlite3.OperationalError('unable to open database file')

    monkeypatch.setattr(ledger, 'connect', refuse)
    res = client.post('/api/indexes', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash'}})
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    assert job['result']['chunks'] > 0


def test_the_ledger_is_not_kept_beside_the_code_that_writes_it():
    """Where a `.db` goes is a settled question, and the answer is not
    "next to the code that writes it" — a durable record inside `src/`
    reads as build output and is the first thing a clean-up deletes."""
    from raglab import ledger

    default = ledger.db_path(env={})
    assert default == config.ROOT / 'databases' / 'raglab.db'
    assert 'src' not in default.parts
    # Overridable, which is what lets the suite guard itself in conftest.
    assert ledger.db_path(env={'RAGLAB_DB': '/tmp/x.db'}) == Path('/tmp/x.db')
