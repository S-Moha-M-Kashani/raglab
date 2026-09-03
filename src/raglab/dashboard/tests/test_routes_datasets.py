"""The dataset routes: the templates an import is guided by, and the import.

The two templates are served from the fixture files themselves rather than
from a copy in code, so the guided path and the schema cannot drift apart.
"""
import json

import pytest

from raglab.configuration import lab_config as config
from raglab.corpora import dataset_import_contract as datasets


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
