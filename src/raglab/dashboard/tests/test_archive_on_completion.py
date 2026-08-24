"""An experiment that ran is on record, and one that did not is not.

Archiving used to be something a reader did — click export, or run the backfill
over what the ledger and `.runs/` still remembered. An evaluation that finished
now writes its own archive as a matter of course, on the ledger row's exact
terms: written in `Jobs.run` before the job goes terminal, and never able to
fail the job that produced it.

Two halves, and both are the same rule about not lying. Only a *finished* run is
archived — there is no honest archive of work that stopped half way, so a
cancelled or errored job leaves no row at all. And a row that could not be
written is said out loud on the job rather than swallowed, because the one thing
worse than a missing archive is a reader who thinks there is one.

Offline throughout: the smoke corpus (5 sessions, 6 questions), `token-hash`
(no model download) and the conftest-pinned `fake` provider.
"""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from raglab.corpora import corpus_store as corpora
from raglab.evaluation import experiment_archive as archive
from raglab.evaluation import experiment_archive_store as archive_store
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import service_experiment_ledger as ledger

from raglab.conftest import SMOKE_INDEX, _finished

# The panel controls this request carries, and what `settings.ui` therefore has
# to say about it: no mode picked, judging off, three of the six questions, no
# type filter. Written out rather than read back from the server, so the
# projection is pinned by something that is not itself.
RUN = {'index': dict(SMOKE_INDEX), 'retrieval': {'k': 3},
       'generation': {'answerer': 'llm', 'fact_judge': True},
       'ragas_mode': 'off', 'limit': 3, 'label': 'archived-on-completion'}
UI = {'mode': '', 'ragas_mode': 'off', 'limit': 3, 'ragas_limit': 0,
      'labels': {}, 'balance': ''}


@pytest.fixture
def client(monkeypatch, tmp_path):
    """This test's own ledger, corpus store and runs directory, so counting
    rows means counting the rows this test wrote."""
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    monkeypatch.setenv('RAGLAB_CORPORA_DB', str(tmp_path / 'corpora.db'))
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path / 'runs')
    from raglab.dashboard.panel_server import create_app
    return TestClient(create_app())


def _archive_rows() -> list[dict]:
    db = archive_store.connect(ledger.db_path())
    try:
        return [dict(row) for row in db.execute(
            'SELECT id, experiment_id, dataset, label, n_questions '
            'FROM archives ORDER BY id')]
    finally:
        db.close()


def _served(experiment_id: str):
    db = archive_store.connect(ledger.db_path())
    try:
        return archive_store.serve(db, experiment_id)
    finally:
        db.close()


def _evaluate(client, **overrides) -> dict:
    started = client.post('/api/evaluations', json=dict(RUN, **overrides))
    assert started.status_code == 202, started.text
    return _finished(client, started.json()['job_id'])


def test_a_finished_evaluation_is_on_record_as_the_file_export_would_write_it(
        client):
    # this is an end-to-end test
    """The claim: running an experiment archives it, and the archive is the
    export.

    Not "a row appeared" — the row is compared with what `build_completed`
    makes of the run's own settings, result and evidence, so the stored
    experiment is the same object a reader would have downloaded rather than a
    second, thinner account of it. Served back through `serve()`, corpus and
    all, which is the form the board's open button and the download share.
    """
    job = _evaluate(client)
    assert job['state'] == 'done', job.get('error')
    assert 'archive_error' not in job, job.get('archive_error')
    result = job['result']
    run_id = result['run_id']

    rows = _archive_rows()
    assert [row['experiment_id'] for row in rows] == [run_id]
    assert rows[0]['id'] == 1, 'the row is keyed by an id the database assigned'
    assert rows[0]['dataset'] == 'smoke-mini'
    assert rows[0]['label'] == 'archived-on-completion'
    assert rows[0]['n_questions'] == 3

    canonical = {key: value for key, value in result.items()
                 if key != 'archive_evidence'}
    # Through JSON on the way in, because that is what an archive is written
    # in — the browser's own export only ever sees these values after they
    # have crossed the wire, and the two paths must not differ by a tuple.
    expected = archive.build_completed(
        *(json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
          for value in ({'config': canonical['config'], 'ui': UI}, canonical,
                        result['archive_evidence'])))
    assert _served(run_id) == expected

    # And the route the board's open button reads answers with the same object.
    over_http = client.get(f'/api/experiments/{run_id}/archive')
    assert over_http.status_code == 200
    assert over_http.json() == expected

    # The corpus travelled by reference, into the store the import writes to —
    # one row, holding the text this run was actually measured against.
    reference = (expected['evaluation']['inspector']['dataset']['corpus'])
    id_corpora = corpora.locate(
        corpora.fingerprint(reference),
        corpora.fingerprint(
            expected['evaluation']['inspector']['dataset']['ground_truth']))
    assert id_corpora is not None
    assert corpora.get(id_corpora)[0] == reference


def test_two_runs_over_one_corpus_are_two_rows_and_one_stored_corpus(client):
    # this is an end-to-end test
    """What archiving every run costs, checked at the seam it would cost it.

    Every experiment on one corpus carries that corpus; storing it inside each
    archive would write the same text once per run. The reference is why this
    is affordable at all, and it only holds while a second run over the same
    corpus adds a row here and nothing there.

    The second run moves one index knob — a run id is a timestamp to the second
    plus the index fingerprint, so two runs of the *same* build inside one
    second are one experiment id and would be one row by definition rather than
    by anything this test could claim.
    """
    first = _evaluate(client, label='first')
    second = _evaluate(client, label='second',
                       index=dict(SMOKE_INDEX, chunker='message'))
    assert first['state'] == 'done' and second['state'] == 'done'

    rows = _archive_rows()
    assert [row['label'] for row in rows] == ['first', 'second']
    assert [row['id'] for row in rows] == [1, 2]
    with corpora.connect() as db:
        stored = db.execute('SELECT COUNT(*) AS n FROM corpora').fetchone()['n']
    assert stored == 1, 'two experiments, one corpus, one stored copy of it'


def test_a_run_that_failed_is_not_archived(client, monkeypatch):
    # this is an end-to-end test
    """Unfinished work is not saved.

    An errored run has a ledger row — the ledger records what was attempted —
    but no archive: an archive is the evidence of an experiment, and there is
    no honest archive of one that stopped part way. Half of it would be an
    experiment on record that never happened.
    """
    def refuses(*args, **kwargs):
        raise RuntimeError('the answerer went away mid-run')

    monkeypatch.setattr(evaluate, 'run_eval', refuses)
    job = _evaluate(client)
    assert job['state'] == 'error'
    assert 'went away' in job['error']
    assert _archive_rows() == []
    recorded = client.get('/api/experiments').json()['experiments']
    assert [row['state'] for row in recorded] == ['error'], (
        'the ledger still records the attempt; only the archive is withheld')


def test_a_cancelled_or_errored_job_never_reaches_the_archive_hook():
    # this is an integration test
    """The same rule at the seam that enforces it, for the state a cancelled
    run leaves and an HTTP test cannot reliably reach.

    `Jobs.run` calls the hook only on `done`. Checked here rather than inferred
    from an empty table, because an empty table is also what a hook that ran
    and quietly failed would leave.
    """
    from raglab.dashboard.panel_server import Jobs
    archived: list[dict] = []
    jobs = Jobs()
    started = threading.Event()

    def waits(report, cancelled):
        started.set()
        while not cancelled():
            time.sleep(0.001)
        # The real targets raise at their next checkpoint; returning under a
        # set cancel flag lands in the same terminal state.
        return {'stopped': True}

    job_id = jobs.start('run', waits, archive=archived.append)
    assert started.wait(timeout=2)
    jobs.cancel(job_id)
    for _ in range(200):
        if jobs.get(job_id)['state'] == 'cancelled':
            break
        time.sleep(0.01)
    assert jobs.get(job_id)['state'] == 'cancelled'
    assert archived == [], 'a cancelled job is not an experiment on record'

    def fails(report):
        raise RuntimeError('no')

    failed_id = jobs.start('run', fails, archive=archived.append)
    for _ in range(200):
        if jobs.get(failed_id)['state'] == 'error':
            break
        time.sleep(0.01)
    assert jobs.get(failed_id)['state'] == 'error'
    assert archived == []

    def finishes(report):
        return {'ok': True}

    done_id = jobs.start('run', finishes, archive=archived.append)
    for _ in range(200):
        if jobs.get(done_id)['state'] == 'done':
            break
        time.sleep(0.01)
    assert [job['id'] for job in archived] == [done_id], (
        'and the job that did finish is archived, so the guard above is not '
        'passing because the hook is never called at all')
    assert '_archive' not in jobs.get(done_id), (
        'the hook is an implementation detail, not JSON the browser reads')


def test_an_archive_that_cannot_be_written_is_reported_not_fatal(client,
                                                                 monkeypatch):
    # this is an end-to-end test
    """The ledger's rule, applied to the archive.

    A record of the work is never a condition of the work. A failing archive
    write leaves the run itself untouched — the same state, the same run file,
    the same ledger row, the same numbers — and says so on the job, where the
    panel's poller can see it. Swallowed instead, a reader would be told the
    experiment was archived when nothing was written.
    """
    def refuses(*args, **kwargs):
        raise archive_store.ArchiveStoreError('the corpus store is read-only')

    monkeypatch.setattr(archive_store, 'store_completed', refuses)
    job = _evaluate(client)

    assert job['state'] == 'done', job.get('error')
    assert 'read-only' in job['archive_error']
    assert job['archive_error'].startswith('ArchiveStoreError'), (
        'the report names what failed, the way ledger_error does')
    assert _archive_rows() == []

    result = job['result']
    assert len(result['rows']) == 3, 'the measurement is untouched'
    assert (evaluate.RUNS_DIR / f"{result['run_id']}.json").exists()
    recorded = client.get('/api/experiments').json()['experiments']
    assert [row['experiment_id'] for row in recorded] == [result['run_id']]
    assert recorded[0]['state'] == 'done'
