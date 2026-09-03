"""The seam the Inspector reads the lab through: one protocol, two implementations.

The Inspector owns no ledger, no archive store and no job recorder, so every
record it shows is the lab's own answer. *How* it asks is the only thing that
differs between a mounted Inspector (a function call) and one pointed at a lab
in another process (HTTP). These tests pin the two modes to one behaviour: the
same document, and — the part that is easy to lose — the same refusal.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from raglab.configuration.lab_config import LabSettings
from raglab.dashboard import inspector_server as inspector
from raglab.dashboard import service_route_plumbing as plumbing

LAB_SETTINGS = LabSettings(openrouter_api_key='', llm_provider='fake')

# The nine operations, and the panel route each one is the caller's side of.
# Written out here rather than derived, because a derived list would drift
# with the code it is supposed to be checking.
OPERATIONS = {
    'imported_archive': ('GET', '/api/imported-archives/{archive_id}'),
    'active_archive': ('GET', '/api/imported-archives/active'),
    'clear_active_archive': ('DELETE', '/api/imported-archives/active'),
    'experiment': ('GET', '/api/experiments/{experiment_id}'),
    'experiment_archive': ('GET', '/api/experiments/{experiment_id}/archive'),
    'experiment_questions': ('GET', '/api/experiments/{experiment_id}/questions'),
    'add_experiment_question': ('POST', '/api/experiments/{experiment_id}/questions'),
    'job': ('GET', '/api/jobs/{job_id}'),
    'jobs': ('GET', '/api/jobs'),
}


def test_the_seam_names_the_nine_operations_and_nothing_else():
    # this is a convention test
    """The Inspector is handed an object that can do exactly nine things.

    Handing it the panel's whole application would work and would also give a
    read-only surface every route the panel has, including the ones that write
    a ledger row. Nine names is the boundary written down, and this is what
    keeps a tenth from arriving unremarked."""
    named = {name for name in dir(plumbing.LabAccess)
             if not name.startswith('_')}
    assert named == set(OPERATIONS), (
        'the seam must name exactly the operations the Inspector needs')
    for implementation in (plumbing.InProcessLabAccess, inspector.HttpLabAccess):
        missing = [name for name in OPERATIONS
                   if not callable(getattr(implementation, name, None))]
        assert not missing, f'{implementation.__name__} answers none of {missing}'


# --- one canned lab, reachable two ways -------------------------------------
#
# The same documents and the same refusals, served once as functions the way
# the panel's handlers behave (a document returned, a refusal raised) and once
# over a real socket the way a remote lab behaves. Every test below runs
# against both, so neither mode can quietly grow a behaviour of its own.

DOCUMENTS = {
    'GET /api/imported-archives/imported-1': {'archive_id': 'imported-1'},
    'GET /api/imported-archives/active': {'archive_id': 'imported-1'},
    'DELETE /api/imported-archives/active': {'archive_id': None},
    'GET /api/experiments/exp-1': {'experiment_id': 'exp-1', 'kind': 'run'},
    'GET /api/experiments/exp-1/archive': {'format': 'raglab-experiment'},
    'GET /api/experiments/exp-1/questions': {'questions': [{'experiment_id': 'add-1'}]},
    # Echoes what it was asked, so the payload's own trip is proven too.
    'POST /api/experiments/exp-1/questions': {'job_id': 'question-job'},
    'GET /api/jobs/question-job': {'id': 'question-job', 'state': 'done'},
    'GET /api/jobs': {'jobs': []},
}

REFUSALS = {
    'GET /api/experiments/no-such-id/archive': (404, 'no-such-id has no complete archive'),
    'GET /api/experiments/moved/archive': (409, 'this installation no longer holds that corpus'),
    'GET /api/experiments/no-such-id/questions': (404, 'unknown experiment: no-such-id'),
    'POST /api/experiments/no-such-id/questions': (404, 'unknown experiment: no-such-id'),
    'GET /api/jobs/no-such-job': (404, 'unknown job'),
}


def _handler(method: str, path: str):
    """One canned lab handler, behaving the way a panel handler behaves."""
    def handle(payload=None):
        wanted = f'{method} {path}'
        if wanted in REFUSALS:
            raise HTTPException(*REFUSALS[wanted])
        document = DOCUMENTS[wanted]
        return document if payload is None else document | {'asked': payload}
    return handle


def _in_process_lab() -> plumbing.InProcessLabAccess:
    return plumbing.InProcessLabAccess(
        imported_archive=lambda archive_id: _handler(
            'GET', f'/api/imported-archives/{archive_id}')(),
        active_archive=_handler('GET', '/api/imported-archives/active'),
        clear_active_archive=_handler('DELETE', '/api/imported-archives/active'),
        experiment=lambda experiment_id: _handler(
            'GET', f'/api/experiments/{experiment_id}')(),
        experiment_archive=lambda experiment_id: _handler(
            'GET', f'/api/experiments/{experiment_id}/archive')(),
        experiment_questions=lambda experiment_id: _handler(
            'GET', f'/api/experiments/{experiment_id}/questions')(),
        add_experiment_question=lambda experiment_id, payload: _handler(
            'POST', f'/api/experiments/{experiment_id}/questions')(payload),
        job=lambda job_id: _handler('GET', f'/api/jobs/{job_id}')(),
        jobs=_handler('GET', '/api/jobs'))


class _CannedLabHandler(BaseHTTPRequestHandler):
    """The same canned lab over a real socket, refusals included."""

    def log_message(self, *args):
        pass

    def _answer(self, method: str, payload: dict | None = None):
        wanted = f'{method} {self.path}'
        if wanted in REFUSALS:
            status, detail = REFUSALS[wanted]
            body = {'detail': detail}
        elif wanted in DOCUMENTS:
            status, body = (202 if method == 'POST' else 200), DOCUMENTS[wanted]
            if payload is not None:
                body = body | {'asked': payload}
        else:
            status, body = 404, {'detail': 'not found'}
        payload = json.dumps(body).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._answer('GET')

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        payload = json.loads(self.rfile.read(length)) if length else None
        self._answer('POST', payload)

    def do_DELETE(self):
        self._answer('DELETE')


@pytest.fixture(scope='module')
def canned_lab_url():
    server = ThreadingHTTPServer(('127.0.0.1', 0), _CannedLabHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        thread.join(timeout=2)


@pytest.fixture(params=['in-process', 'http'])
def both_modes(request, monkeypatch, canned_lab_url):
    """The Inspector over each access mode in turn — one test body, two transports."""
    monkeypatch.setattr(inspector, 'load_lab_settings', lambda: LAB_SETTINGS)
    lab = (_in_process_lab() if request.param == 'in-process'
           else inspector.HttpLabAccess(canned_lab_url))
    return TestClient(inspector.create_inspector_app(lab=lab))


def test_every_lab_backed_route_returns_the_labs_own_document(both_modes):
    # this is an integration test
    """The document is the lab's, whichever way it was fetched."""
    assert both_modes.get('/api/imported-archives/imported-1').json() == {
        'archive_id': 'imported-1'}
    assert both_modes.delete('/api/imported-archives/active').json() == {
        'archive_id': None}
    assert both_modes.get('/api/experiments/exp-1').json()['experiment_id'] == 'exp-1'
    assert both_modes.get('/api/experiments/exp-1/archive').json() == {
        'format': 'raglab-experiment'}
    assert both_modes.get('/api/experiments/exp-1/questions').json()['questions'] == [
        {'experiment_id': 'add-1'}]
    assert both_modes.get('/api/lab-jobs/question-job').json()['state'] == 'done'

    created = both_modes.post('/api/experiments/exp-1/questions',
                              json={'question_id': 'q7'})
    assert created.status_code == 202
    # The lab's own answer, and the payload arrived at the lab unaltered.
    assert created.json() == {'job_id': 'question-job',
                              'asked': {'question_id': 'q7'}}

    follow = both_modes.get('/api/follow')
    assert follow.status_code == 200
    assert follow.json()['lab'] == 'up'
    assert follow.json()['archive_id'] == 'imported-1'


def test_a_refusal_keeps_its_status_and_its_message_in_both_modes(both_modes):
    # this is an integration test
    """A refusal is the lab speaking, and the reader is owed its words.

    A 404 that became a 503 would read as "the lab is down" about a lab that
    answered, and a 409 that became a 404 would read as "this experiment does
    not exist" about one that does — this installation has simply lost the
    corpus it ran on."""
    missing = both_modes.get('/api/experiments/no-such-id/archive')
    assert missing.status_code == 404
    assert missing.json()['detail'] == 'no-such-id has no complete archive'

    moved = both_modes.get('/api/experiments/moved/archive')
    assert moved.status_code == 409
    assert moved.json()['detail'] == 'this installation no longer holds that corpus'

    questions = both_modes.get('/api/experiments/no-such-id/questions')
    assert questions.status_code == 404
    assert questions.json()['detail'] == 'unknown experiment: no-such-id'

    added = both_modes.post('/api/experiments/no-such-id/questions',
                            json={'question_id': 'q7'})
    assert added.status_code == 404
    assert added.json()['detail'] == 'unknown experiment: no-such-id'

    job = both_modes.get('/api/lab-jobs/no-such-job')
    assert job.status_code == 404
    assert job.json()['detail'] == 'unknown job'


def test_a_lab_that_cannot_be_reached_is_unavailability_not_a_refusal(monkeypatch):
    # this is an integration test
    """Only the remote mode can be unreachable, and it must say so.

    In-process the branch is dead — the call either answers or raises — which
    is exactly why the HTTP mode has to keep proving it."""
    monkeypatch.setattr(inspector, 'load_lab_settings', lambda: LAB_SETTINGS)
    # Port 9 is the discard port: nothing listens, and the connection is
    # refused rather than hanging until the timeout.
    client = TestClient(inspector.create_inspector_app(
        lab=inspector.HttpLabAccess('http://127.0.0.1:9')))

    assert client.get('/api/experiments/exp-1/archive').status_code == 503
    assert client.get('/api/experiments/exp-1/questions').status_code == 503
    assert client.get('/api/lab-jobs/question-job').status_code == 503
    assert client.post('/api/experiments/exp-1/questions',
                       json={'question_id': 'q7'}).status_code == 503
    assert client.delete('/api/imported-archives/active').status_code == 503
    # A record that cannot be fetched is a 404 rather than a 503, because a
    # read-only view pinned to nothing reads as an experiment that recorded no
    # evidence — the one place the two failures are deliberately alike.
    assert client.get('/api/experiments/exp-1').status_code == 404
    assert client.get('/api/imported-archives/imported-1').status_code == 404
    assert client.get('/api/follow').json()['lab'] == 'down'


def test_the_inspector_defaults_to_the_lab_the_environment_names(monkeypatch):
    # this is a unit test
    """No `lab=` means the standalone Inspector: HTTP, at the configured URL."""
    monkeypatch.setattr(inspector, 'load_lab_settings', lambda: LAB_SETTINGS)
    monkeypatch.setenv(inspector.LAB_URL_ENV, 'http://lab.example:9002/')
    assert inspector.HttpLabAccess().base_url == 'http://lab.example:9002'
    monkeypatch.delenv(inspector.LAB_URL_ENV)
    assert inspector.HttpLabAccess().base_url == inspector.DEFAULT_LAB_URL


def test_an_operation_the_inspector_never_calls_is_never_reached(monkeypatch):
    # this is a unit test
    """A structural stand-in is enough, which is the point of a nine-name seam:
    an Inspector route asks for one operation and can reach no other."""
    monkeypatch.setattr(inspector, 'load_lab_settings', lambda: LAB_SETTINGS)
    client = TestClient(inspector.create_inspector_app(
        lab=SimpleNamespace(experiment=lambda _id: {'experiment_id': 'exp-1'})))
    assert client.get('/api/experiments/exp-1').json()['experiment_id'] == 'exp-1'


# --- the panel's own handlers, behind the same seam --------------------------

def _stored_archive(experiment_id: str) -> str:
    """One complete archive in the ledger's database, as a finished run leaves it."""
    from raglab.evaluation import experiment_archive_store as archive_store
    from raglab.evaluation import service_experiment_ledger as ledger
    from raglab.evaluation.tests import archive_examples as examples

    db = archive_store.connect(ledger.db_path())
    try:
        archive_store.put(db, experiment_id, examples.generated_rung()['archive'])
    finally:
        db.close()
    return experiment_id


def test_the_panels_own_refusals_cross_its_seam_unchanged(client, monkeypatch,
                                                          tmp_path):
    # this is an integration test
    """What the panel's route refuses, the panel's seam refuses identically.

    Both refusals are the ones the Inspector's reader actually meets: a 404
    for an experiment with no complete archive, and a 409 for one whose corpus
    this installation no longer holds. Neither may arrive as unavailability,
    and neither may lose the lab's own words on the way — the message is the
    reader's only account of what went wrong."""
    lab = client.app.state.lab_access

    missing = client.get('/api/experiments/never-archived/archive')
    assert missing.status_code == 404
    assert lab.experiment_archive('never-archived') == (
        missing.status_code, missing.json()['detail'])

    experiment_id = _stored_archive('exp-moved-corpus')
    monkeypatch.setenv('RAGLAB_CORPORA_DB', str(tmp_path / 'emptied.db'))
    moved = client.get(f'/api/experiments/{experiment_id}/archive')
    assert moved.status_code == 409
    assert lab.experiment_archive(experiment_id) == (
        moved.status_code, moved.json()['detail'])

    unknown_job = client.get('/api/jobs/no-such-job')
    assert unknown_job.status_code == 404
    assert lab.job('no-such-job') == (unknown_job.status_code,
                                      unknown_job.json()['detail'])


def test_the_panel_answers_its_seam_with_the_document_its_route_serves():
    # this is an integration test
    """A document crosses unchanged too, which is the other half of the claim."""
    from raglab.dashboard.panel_server import create_app

    panel = TestClient(create_app())
    lab = panel.app.state.lab_access
    assert lab.jobs() == panel.get('/api/jobs').json()
    assert lab.active_archive() == panel.get('/api/imported-archives/active').json()
    assert lab.clear_active_archive() == {'archive_id': None}
