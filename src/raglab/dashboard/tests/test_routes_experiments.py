"""The board route: one table per dataset, and what it says while a job runs.

The board is not a ranking. These tests pin the population it draws from, the
filter that narrows it to one corpus, and the ordering it declares — the
markup that renders those rows is asserted in `test_routes_assets.py`.
"""
import json

from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import run_evaluation as evaluate


# --- the leaderboard's bounded view -----------------------------------------

def test_the_leaderboard_says_how_much_of_the_disk_it_shows(client, monkeypatch, tmp_path):
    # this is an integration test
    """A run can rank differently on a bounded page than over the whole
    directory, with nothing on screen explaining the disagreement — a
    bounded view has to say what it left out. That the panel actually asks
    for a stated limit is a row in the convention table above; this is the
    behaviour behind it, exercised through the real route. Writes its own run
    files rather than reading whatever the developer's `.runs/` happens to
    hold, the way `test_leaderboard.py` does — a test that passes on an empty
    directory is not coverage."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    for i in range(4):
        run_id = f'20260731-12000{i}-abc12{i}'
        (tmp_path / f'{run_id}.json').write_text(json.dumps({
            'run_id': run_id, 'label': f'run {i}',
        }), encoding='utf-8')

    body = client.get('/api/evaluations?limit=3').json()
    assert len(body['runs']) == 3
    # Served, not counted in the browser: the page cannot know how many files it
    # was not sent.
    assert body['total'] >= 4


# --- the leaderboard's one table per dataset --------------------------------

def test_the_leaderboard_route_filters_to_one_dataset(client):
    # this is an integration test
    """The picker is a filter on one population, not a switch between two
    surfaces — so the route takes the dataset and answers with one board."""
    body = client.get(f'/api/leaderboard?dataset={datasets.BUILTIN}').json()
    assert body['dataset'] == datasets.BUILTIN
    assert isinstance(body['rows'], list)
    # And every row agrees with the table it was served in. This held on the
    # fixtures and not in production: three places decided what a blank dataset
    # means, and the row was the one that answered differently — so rows with no
    # recorded dataset arrived on the built-in board carrying a cell that said
    # they belonged to no corpus at all.
    assert all(r['dataset'] == datasets.BUILTIN for r in body['rows'])


def test_the_leaderboard_route_offers_every_experiment_unfiltered(client):
    # this is an integration test
    """`*` is every experiment — the table that used to live on the lab page.
    It is the same population with no filter, which is why it is an option in
    the same picker rather than a second surface."""
    body = client.get('/api/leaderboard?dataset=*').json()
    assert body['dataset'] == '*'
    datasets = {r['dataset'] for r in body['rows']}
    assert len(datasets) != 1 or not body['rows'], (
        'the unfiltered view must not be filtered')


def test_the_leaderboard_route_names_every_dataset_the_picker_can_offer(client):
    # this is an integration test
    """The picker's options travel with the board, so the page makes one
    request rather than joining two."""
    body = client.get('/api/leaderboard').json()
    ids = {d['id'] for d in body['datasets']}
    assert 'diary-fa' in ids


def test_the_default_leaderboard_is_the_experiment_population_not_farsi(client):
    # this is an integration test
    """An omitted dataset is a request for the board's experiment population,
    not a request to display the legacy Farsi builtin. The latter makes an
    empty Farsi table appear whenever the user returns from another surface."""
    body = client.get('/api/leaderboard').json()
    assert body['dataset'] == '*'
    assert body['ordering'] == 'score'
    assert body['running'] is False


def test_the_leaderboard_contract_exposes_live_ordering_and_job_state(
        panel_texts):
    # this is a convention test
    """The page must be able to distinguish a live newest-first board from an
    idle score-first board; without these fields it cannot refresh after the
    Laboratory starts or finishes a job."""
    js = panel_texts['leaderboard.js']
    assert 'body.ordering' in js
    assert 'body.running' in js
    assert 'setTimeout' in js
