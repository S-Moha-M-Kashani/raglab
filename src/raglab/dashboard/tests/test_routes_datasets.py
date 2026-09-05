"""The dataset routes: the templates an import is guided by, and the import.

The two templates are served from the fixture files themselves rather than
from a copy in code, so the guided path and the schema cannot drift apart.
"""
import json

import pytest

from raglab.configuration import lab_config as config
from raglab.corpora import dataset_import_contract as datasets
from raglab.dashboard.routes import datasets as routes


def test_the_dataset_help_points_at_the_import_that_exists(panel_texts, client):
    # this is a convention test
    """The corpus knob's own text tells a reader where to import one, and it had
    gone stale twice over: it named a `!` no page draws any more, and it said
    "the button beside it" after the import moved out of the Index card into the
    setup panel. Text that directs a reader to a place is a claim about the
    layout, and it has to be checked like one."""
    served = client.get('/api/options').json()['help']
    dataset = served['index.dataset']
    assert 'the ! there' not in dataset, (
        'no page draws that mark any more — a text naming it sends the reader '
        'looking for something that is not there')
    assert 'setup panel' in dataset, (
        'the import lives in the left panel now, and the sentence that points '
        'at it has to point there')
    html = panel_texts['index.html']
    sidebar = html[html.index('<aside class="sidebar"'):]
    sidebar = sidebar[:sidebar.index('</aside>')]
    assert 'data-topic="run.dataset-file"' in sidebar, (
        'and what it points at is the trigger on the import block itself')


# --- the templates are the guided path, not the schemas ---------------------

@pytest.mark.parametrize('route, filename', [
    ('/api/dataset-templates/corpus', 'corpus_template.json'),
    ('/api/dataset-templates/groundtruth', 'groundtruth_template.json'),
])
def test_dataset_template_routes_serve_the_fixture_byte_for_byte(
        client, route, filename):
    # this is a convention test
    """One author per artifact: the route reads the fixture at request time
    rather than a copy baked into code, so a byte comparison against the
    file on disk is the whole claim — anything less would let the served
    copy drift from the file the mirror test actually guards."""
    response = client.get(route)
    assert response.status_code == 200
    on_disk = (datasets.BUNDLED_DIR / filename).read_bytes()
    assert response.content == on_disk, (
        f'{route} must serve {filename} byte-identical to the fixture, not '
        'a duplicate or a stale copy')
    assert filename in (response.headers.get('content-disposition') or ''), (
        f'{route} must offer the file under its real name ({filename}), so '
        'a download or a save keeps the name the templates are written under'
    )


def test_dataset_help_text_leads_with_the_templates():
    # this is a convention test
    """The user's complaint was concrete: the schema document is confusing
    and complex. The fix is not deleting the schema help — it is the
    contract validate() actually runs — but making it the second thing
    read, not the first. 'Start from the two templates' must come before
    the schema files are named at all."""
    text = config.HELP['run.dataset-file']
    templates_at = text.index('Start from the two templates')
    schema_at = text.index('schema_corpus.json')
    assert templates_at < schema_at, (
        'run.dataset-file must lead with the templates as the way to start '
        'and name the schemas only afterwards, as the formal contract '
        'behind them')
    assert 'corpus_template.json' in text and 'groundtruth_template.json' in text
    assert 'formal contract' in text, (
        'the schemas must be named as the contract behind the templates, '
        'not as a second starting point')


def test_importing_a_corpus_forgets_the_widgets_cached_board_ids(client):
    # this is an integration test
    """The widget reads the board's dataset ids once and keeps them for the
    life of the process — it filters every turn's memory context against them
    and cannot afford a board reading per turn. Wiring a different reader
    forgets them, and until now nothing else did: a corpus imported into a
    running lab stayed invisible to that filter until somebody restarted the
    server. The import route is where this installation's corpus set changes,
    and it is a route rather than the store because the widget is a sealed leaf
    that `corpora/` may not reach into."""
    from raglab.agents.widget import long_term_memory
    from raglab.corpora.tests.test_datasets import _valid_pair

    long_term_memory.remember_board_dataset_ids({'stale-corpus'})
    assert long_term_memory._BOARD_DATASET_IDS == {'stale-corpus'}

    corpus, ground_truth = _valid_pair()
    response = client.post('/api/datasets',
                           json={'corpus': corpus, 'ground_truth': ground_truth})

    assert response.status_code == 200, response.text
    assert long_term_memory._BOARD_DATASET_IDS is None


# --- the four readings, and the endpoint that serves them --------------------

#: The readings, in the order `readings()` returns them, so a test naming one
#: is naming the same statement the page shows.
READING_IDS = ('uncited-documents', 'absent-citations',
               'unpopulated-labels', 'blank-line-documents')


def test_a_clean_pair_reports_zero_for_every_reading():
    # this is a unit test
    """`smoke-mini` is five documents and six questions that can be checked by
    eye: every document is cited, every citation resolves, every declared
    label is carried by some row, and no part text holds a blank line. So all
    four readings must be present and all four must say zero — a reading that
    vanished when it found nothing would leave a reader unable to tell a clean
    corpus from a check that never ran."""
    corpus, ground_truth = datasets.load('smoke-mini')
    found = routes.readings(corpus, ground_truth)
    assert [reading['id'] for reading in found] == list(READING_IDS)
    for reading in found:
        assert reading['count'] == 0 and reading['ids'] == [], reading['id']
        assert reading['says'] and reading['grid'], reading['id']


def test_a_broken_pair_names_the_rows_behind_each_reading():
    # this is a unit test
    """The pair this page exists for: a question reaching for a document that
    is not there, a document nothing asks about, a label declared and never
    carried, and a part whose text holds a blank line. Each reading has to
    hand back the rows behind its count — and the absent citation has to name
    the document id it cited, because that id is not in the corpus and so no
    grid on the page can show it."""
    corpus = {
        'corpus_dataset_metadata': {'dataset': 'broken', 'label_fields': {
            'role': {'type': 'string'}, 'mood': {'type': 'string'}}},
        'corpus_documents': [
            {'corpus_document_id': 1, 'document_content': [
                {'text': 'one line\n\nand another', 'labels': {'role': 'user'}}]},
            {'corpus_document_id': 2, 'document_content': [
                {'text': 'nobody asks about this', 'labels': {'role': 'user'}}]},
        ]}
    ground_truth = {
        'groundtruth_dataset_metadata': {
            'question_metadata_fields': {'difficulty': {'type': 'string'}}},
        'groundtruth_dataset': [
            {'groundtruth_question_id': 7,
             'relevant_corpus_documents': [{'corpus_document_id': 1},
                                           {'corpus_document_id': 99}]}]}
    found = {reading['id']: reading
             for reading in routes.readings(corpus, ground_truth)}

    assert found['uncited-documents']['ids'] == [2]
    assert found['absent-citations']['ids'] == [7]
    assert found['absent-citations']['detail'] == [
        'question 7 cites document 99'], (
        'the offending document id must be named — it is the whole of what a '
        'reader has to fix, and it is not a value any grid can show')
    assert found['unpopulated-labels']['ids'] == ['difficulty', 'mood']
    assert found['blank-line-documents']['ids'] == [1]
    for reading in found.values():
        assert reading['count'] == len(reading['ids']), reading['id']


def test_dataset_content_serves_the_whole_pair_and_its_readings(client):
    # this is an integration test
    """One request per dataset and everything after it in the browser, so the
    round trip has to carry every document, every part and every question —
    a page that sorted and filtered a partial corpus would answer questions
    about rows it never received."""
    body = client.get('/api/dataset-content/smoke-mini').json()
    corpus, ground_truth = datasets.load('smoke-mini')
    assert body['corpus'] == corpus and body['ground_truth'] == ground_truth
    assert body['ground_truth_error'] == ''
    assert body['dataset']['id'] == 'smoke-mini'
    # The declarations the panel's own card reads, so the page's columns are
    # this corpus's own labels rather than a fixed set.
    assert [row['name'] for row in body['dataset']['label_declarations']] == [
        'role', 'topics']
    assert [reading['id'] for reading in body['readings']] == list(READING_IDS)


def test_dataset_content_refuses_an_id_the_catalogue_does_not_list(client):
    # this is an integration test
    """A surface that loaded whatever `datasets.load` would accept could be
    pointed at a file by URL. So the catalogue decides, the refusal names the
    id asked for, and nothing partial comes back with it."""
    response = client.get('/api/dataset-content/no-such-corpus')
    assert response.status_code == 404
    assert 'no-such-corpus' in response.json()['detail']
    assert 'corpus' not in response.json()


def test_a_pair_with_no_ground_truth_still_describes_its_corpus(client, tmp_path,
                                                                monkeypatch):
    # this is an integration test
    """A corpus listed with nothing to measure against (D1) is described
    rather than refused — the same fallback the dataset card already makes —
    and the reason the ground truth could not be read is stated instead of
    being shown as an empty question list."""
    corpus = {
        'corpus_dataset_metadata': {'dataset': 'lonely', 'name': 'Lonely',
                                    'language': 'en', 'label_fields': {}},
        'corpus_documents': [{'corpus_document_id': 1, 'document_content': [
            {'text': 'no question was ever written about this'}]}]}
    monkeypatch.setenv('RAGLAB_DATASETS', str(tmp_path))
    (tmp_path / 'lonely_corpus.json').write_text(json.dumps(corpus),
                                                 encoding='utf-8')
    datasets.forget()
    body = client.get('/api/dataset-content/lonely').json()
    assert body['corpus'] == corpus
    assert body['ground_truth'] == {} and body['dataset']['questions'] == 0
    assert 'nothing to measure against' in body['ground_truth_error']
    datasets.forget()
