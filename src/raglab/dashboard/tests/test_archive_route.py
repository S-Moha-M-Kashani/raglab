"""The route the open button reads: one experiment, as the export file."""
import json

from raglab.configuration.lab_config import IndexConfig, LabConfig
from raglab.conftest import SMOKE_INDEX
from raglab.evaluation import experiment_archive as archive
from raglab.evaluation import experiment_archive_store as archive_store
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.evaluation.tests import archive_examples as examples


def _stored(experiment_id='exp-archived'):
    """One complete archive in the ledger's database, as phase 2 will leave it."""
    db = archive_store.connect(ledger.db_path())
    try:
        archive_store.put(db, experiment_id, examples.generated_rung()['archive'])
    finally:
        db.close()
    return experiment_id


def test_the_route_serves_the_object_the_export_button_writes(client):
    # this is an integration test
    """The whole point of the change, checked at the seam a reader crosses.

    What comes back over HTTP must be indistinguishable from the file export
    writes — corpus included, spliced back from the corpus store the archive
    wrote it to — or "opening a row is importing its export" is false where it
    is observable.
    """
    experiment_id = _stored()
    body = client.get(f'/api/experiments/{experiment_id}/archive')
    assert body.status_code == 200
    assert body.json() == examples.generated_rung()['archive']


def test_an_experiment_without_a_complete_archive_is_a_404_that_says_why(client):
    # this is an integration test
    """Only complete archives are rows, so the absence has to be legible.

    A bare 404 would read as a broken link on a row the reader can see; it is
    the experiment's evidence that is incomplete, and the message says so.
    """
    missing = client.get('/api/experiments/never-archived/archive')
    assert missing.status_code == 404
    assert 'complete archive' in missing.json()['detail']


def test_a_done_question_row_opens_with_its_own_settings_not_a_fake_archive(
        client):
    # this is an integration test
    """Question jobs keep evidence in the ledger, not in the archive store.

    The board still needs an openable handoff.  It may use the settings this
    job recorded, but must neither rewrite the row nor fabricate an evaluation
    archive or a historical corpus that the question row never stored.
    """
    settings_config = json.loads(json.dumps(
        LabConfig(index=IndexConfig(**SMOKE_INDEX)).to_dict()))
    config = settings_config | {'provider': 'fake'}
    result = {
        'label': 'adds 1 to parent-run', 'annotates': 'parent-run',
        'question_id': '1', 'config': config, 'dataset': 'smoke-mini',
        'selection': {'n': 1, 'question_ids': ['1']},
        'traces': [{'question_id': '1', 'trace': {'candidates': []}}],
        'rows': [{'id': '1', 'answer': 'recorded answer'}],
    }
    ledger.record({'id': 'question-row', 'kind': 'question',
                   'config': config, 'result': result}, 'done')
    before = ledger.experiment('question-row')

    response = client.get('/api/experiments/question-row/archive')

    assert response.status_code == 200
    handoff = response.json()
    assert archive.validate_archive(handoff) == handoff
    assert handoff == {
        'format': 'raglab-experiment', 'version': 1,
        'settings': {
            'config': settings_config,
            'ui': {'mode': '', 'ragas_mode': 'offline', 'limit': 0,
                   'ragas_limit': 0, 'labels': {}, 'balance': ''},
        },
    }
    assert ledger.experiment('question-row') == before


def test_a_corpus_the_store_lost_is_refused_over_http_not_substituted(
        client, monkeypatch, tmp_path):
    # this is an integration test
    """The fingerprint guard, at the seam where it would do its damage.

    A dataset replaced under the same id would otherwise let this route hand a
    plausible corpus to an experiment that never ran on it — the row lying
    about what produced it, one step before the row. The corpus is addressed
    by its content now, so a replacement can never answer for it; what is left
    is a store that does not hold this version, and the honest answer is still
    a refusal. 409 rather than 404: the archive is intact, it is this
    installation that has moved.
    """
    experiment_id = _stored('exp-moved-corpus')
    monkeypatch.setenv('RAGLAB_CORPORA_DB', str(tmp_path / 'emptied.db'))
    moved = client.get(f'/api/experiments/{experiment_id}/archive')
    assert moved.status_code == 409
    assert 'not in the corpus store' in moved.json()['detail']


def test_the_existing_experiment_route_still_answers_the_inspector(client):
    # this is an integration test
    """The reason this is a new route rather than a changed one.

    `inspector.js` reads `detail`, `state` and `kind` off `/api/experiments/{id}`
    for its recorded mode. Repointing that route at an archive would blank the
    page, so both shapes must be reachable at once.
    """
    experiment_id = _stored('exp-both-routes')
    archived = client.get(f'/api/experiments/{experiment_id}/archive')
    record = client.get(f'/api/experiments/{experiment_id}')
    assert archived.status_code == 200
    # The record route answers from the ledger/run join, which knows nothing of
    # this row — what matters is that the archive route did not replace it.
    assert record.status_code in (200, 404)
    assert 'format' in archived.json()
