# this is an end-to-end test
"""The suite owns its server — and that server is not the developer's lab.

The first journey a reader should be able to trust: a real Chromium reaches a
lab this suite started, on a port nobody else has, backed by the fake model and
writing only into a temporary directory. Everything else in the browser suite
rests on that, so it is asserted once, here, rather than repeated.
"""
import json

import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser


def _capabilities(page, lab_server):
    """What the served lab says about itself, read through the browser.

    `/api/options` is the one call the panel makes on load, and the capability
    block rides on it — so this asks the page exactly what the page asks.
    """
    options = page.evaluate('url => fetch(url).then(r => r.json())',
                            f'{lab_server}/api/options')
    return options['capabilities']


def test_all_three_surfaces_load_in_a_real_browser(page, lab_server):
    for path, marker in (('/', '#app-settings'),
                         ('/inspector', 'body'),
                         ('/leaderboard', 'body')):
        response = page.goto(f'{lab_server}{path}')
        assert response is not None and response.ok, path
        page.wait_for_selector(marker)


def test_the_lab_under_test_is_not_the_developers_daemon(page, lab_server):
    assert ':9002' not in lab_server, lab_server
    page.goto(f'{lab_server}/')
    reported = _capabilities(page, lab_server)
    assert reported['llm_provider'] == 'fake', reported['llm_provider']


def test_the_ledger_this_lab_writes_lives_in_the_temporary_home(
        page, lab_server, lab_home):
    page.goto(f'{lab_server}/')
    storage = _capabilities(page, lab_server)['storage']
    assert storage['index'] == 'memory', json.dumps(storage)
    assert str(lab_home) in storage['experiments'], json.dumps(storage)
