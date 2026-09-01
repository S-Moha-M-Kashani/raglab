# this is an end-to-end test
"""The panel's working journey, pressed by a reader in a real browser.

Everything the lab is for happens on one page and in one order: pick a corpus,
build its index, retrieve against it, run an evaluation, read the numbers that
came back. These journeys drive exactly that, through a real Chromium against
the suite's own lab, on the smoke corpus (five sessions, six questions) with a
hashing embedder and the fake backend — so a build costs milliseconds,
downloads nothing, and no claim here is about the diary.

What is asserted is what the panel itself puts on screen: the index summary it
writes after a build, the retrieval line, the score tiles and the three
tables, the per-question rows, and the note the job box carries when the
server refuses a second experiment. Never an internal — if the page did not
say it, it is not proved.
"""
import re

import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser

from playwright.sync_api import expect                      # noqa: E402

#: A build is fast here, but a cold first page load and a job's 700 ms poll
#: are not — every wait below is Playwright's own, never a sleep.
SETTLE = 20_000


def _pick_the_smoke_experiment(page, *, chunker='session', limit='0',
                               ragas_mode='off'):
    """Put the panel on the smoke corpus and a pipeline that needs no model.

    A first visit boots on the codex CLI preset — HyDE, an LLM reranker, an
    LLM gate, an LLM answerer and both judges — so the first thing a reader
    doing offline work does is put the backend back to the one the lab booted
    with, and every stage that would call a model back to one that does not.

    `chunker` is in the index fingerprint, so a journey that has something to
    say about building names its own: the lab under test is shared and its
    index registry is process memory, which would otherwise make "was this
    reused?" a question about which test ran first.
    """
    # Wide enough that the setup panel is a column rather than a drawer, and
    # tall enough that every button is on screen at once.
    page.set_viewport_size({'width': 1440, 'height': 1800})
    # Nothing exists at parse time: every select on this page is empty in the
    # markup and filled once `GET /api/options` answers.
    expect(page.locator('#chunker option').first).to_be_attached(timeout=SETTLE)
    page.select_option('#mode', '')
    page.select_option('#dataset', 'smoke-mini')
    page.select_option('#chunker', chunker)
    page.select_option('#embedder', 'token-hash')
    page.uncheck('#hyde')
    page.select_option('#reranker', 'none')
    page.select_option('#grader', 'none')
    page.fill('#k', '3')
    page.select_option('#answerer', 'extractive')
    page.uncheck('#fact_judge')
    page.select_option('#ragas_mode', ragas_mode)
    page.fill('#limit', limit)


def _build(page, *, force=False):
    """Press Build (or Force rebuild) and wait for the summary to be rewritten.

    Waiting for the *text to change* rather than for a phrase to appear: the
    line already holds a summary from the press before, so a phrase it already
    contains would let a second build pass without ever having happened.
    """
    before = page.locator('#indexInfo').inner_text()
    page.click('#rebuild' if force else '#build')
    page.wait_for_function(
        'before => document.getElementById("indexInfo").innerText !== before',
        arg=before, timeout=SETTLE)
    return page.locator('#indexInfo').inner_text()


def _collection_of(summary: str) -> str:
    """The collection name the index summary leads with."""
    found = re.match(r'(raglab-[0-9a-f]+) —', summary)
    assert found, summary
    return found.group(1)


def test_the_reader_picks_the_smoke_corpus_and_builds_its_index(panel):
    _pick_the_smoke_experiment(panel)
    spine = panel.locator('#chromeProgress')
    summary = _build(panel)

    # The spine stops announcing when the work stops: `data-running` is the
    # panel's own flag, written on every poll tick. It is asserted settled
    # rather than live because a five-session build on a hashing embedder is
    # over long before the panel's first 700 ms tick — there is no live frame
    # to catch, and inventing one would mean faking the server.
    expect(spine).to_have_attribute('data-running', '', timeout=SETTLE)
    caption = panel.locator('#spineCaption')
    expect(caption).to_contain_text('index')
    expect(caption).to_contain_text('100%')

    # And the summary reports what was built, not merely that something was.
    assert '5 chunks' in summary, summary
    assert 'dim 512' in summary, summary
    assert _collection_of(summary)
    assert panel.locator('#jobBox').inner_text().strip() == ''


def test_building_the_same_index_twice_reuses_it_and_a_forced_rebuild_does_not(
        panel):
    # Its own chunker, so this journey owns the collection it talks about.
    _pick_the_smoke_experiment(panel, chunker='turn-pair')
    first = _build(panel)
    collection = _collection_of(first)
    assert 'reused' not in first, \
        'the first build of a collection cannot be a reuse'

    # Same knobs, so the second press is allowed to hand back what is already
    # in memory — and says so.
    assert '(reused)' in _build(panel)

    # Force rebuild is the same collection built again, so the word goes away.
    rebuilt = _build(panel, force=True)
    assert '(reused)' not in rebuilt, rebuilt
    assert _collection_of(rebuilt) == collection, \
        'a forced rebuild rebuilds the same index, it does not name a new one'
    assert '5 chunks' in rebuilt


def test_changing_a_knob_and_building_again_builds_a_different_index(panel):
    _pick_the_smoke_experiment(panel, chunker='message')
    by_message = _collection_of(_build(panel))

    # A chunker is inside the index fingerprint, so cutting the same five
    # sessions a different way is a different index and has to be built.
    panel.select_option('#chunker', 'fixed')
    panel.fill('#chunk_chars', '140')
    rebuilt = _build(panel)
    assert '(reused)' not in rebuilt, \
        'a knob the build reads changed, so nothing could be handed back'
    assert _collection_of(rebuilt) != by_message, rebuilt


def test_retrieving_without_an_evaluation_reports_what_came_back(panel):
    _pick_the_smoke_experiment(panel, limit='2')
    _build(panel)

    panel.click('#retrieve-selected')
    line = panel.locator('#retrieveInfo')
    expect(line).to_contain_text('retrieved for', timeout=SETTLE)
    expect(line).to_contain_text('2 questions')
    expect(line).to_contain_text('gold chunk')
    # A retrieval scores nothing, so the Readings card stays as it was.
    expect(panel.locator('#resultEmpty')).to_be_visible()


def test_running_an_evaluation_fills_the_readings_card(panel):
    # `offline` RAGAS needs no judge, so its two context metrics land in the
    # table on the fake backend — which is what makes the table a table here.
    _pick_the_smoke_experiment(panel, ragas_mode='offline')
    _build(panel)
    panel.fill('#label', 'browser smoke')

    panel.click('#run')
    expect(panel.locator('#resultBody')).to_be_visible(timeout=SETTLE)
    expect(panel.locator('#resultEmpty')).to_be_hidden()

    meta = panel.locator('#resultMeta')
    expect(meta).to_contain_text('browser smoke')
    expect(meta).to_contain_text('6 questions')
    expect(meta).to_contain_text('5 chunks')

    # One tile per metric the run measured, each carrying a number and the
    # sentence that says what it is.
    tiles = panel.locator('#scores .score')
    assert tiles.count() > 5, tiles.count()
    expect(tiles.first.locator('b')).not_to_be_empty()
    expect(tiles.first.locator('.muted')).not_to_be_empty()

    expect(panel.locator('#byType table caption')).to_have_text(
        'Scores by behavior')
    expect(panel.locator('#byType table tbody tr')).not_to_have_count(0)
    expect(panel.locator('#ragas table caption')).to_have_text(
        'RAGAS judged metrics')
    expect(panel.locator('#ragas')).to_contain_text('mode offline')
    expect(panel.locator('#extras table caption')).to_have_text('Selection')
    expect(panel.locator('#extras')).to_contain_text('stride')


def test_the_per_question_rows_open_and_sort(panel):
    _pick_the_smoke_experiment(panel)
    _build(panel)
    panel.click('#run')
    expect(panel.locator('#resultBody')).to_be_visible(timeout=SETTLE)

    rows = panel.locator('details.rowdump')
    expect(rows.locator('#rows table')).to_be_hidden()
    rows.locator('summary').click()
    expect(rows.locator('#rows table caption')).to_have_text(
        'Every question, one row each')
    body = panel.locator('#rows table tbody tr')
    expect(body).to_have_count(6)
    # Every question the run scored has its own row, with its own numbers.
    expect(body.first.locator('td').nth(1)).not_to_be_empty()

    # The id column holds numbers, so it opens on the largest.
    ids = panel.locator('#rows table thead th').first
    ids.click()
    expect(ids).to_have_attribute('aria-sort', 'descending')
    expect(body.first.locator('td').first).to_have_text('6')
    ids.click()
    expect(ids).to_have_attribute('aria-sort', 'ascending')
    expect(body.first.locator('td').first).to_have_text('1')


def test_the_server_refuses_a_second_experiment_and_the_page_says_so(panel):
    _pick_the_smoke_experiment(panel, ragas_mode='offline')
    _build(panel)

    # One job at a time, process-wide: the build pressed while the evaluation
    # is still running is refused with a 409, and the panel puts the server's
    # own sentence in the job box rather than swallowing it.
    panel.click('#run')
    with panel.expect_response(
            lambda r: r.request.method == 'POST'
            and r.url.endswith('/api/indexes')) as refused:
        panel.click('#build')
    assert refused.value.status == 409, refused.value.text()
    expect(panel.locator('#jobBox')).to_contain_text('is already running')

    # And the experiment that was running still finishes and still reports.
    expect(panel.locator('#resultBody')).to_be_visible(timeout=SETTLE)


@pytest.mark.xfail(reason='on the smoke corpus a job outlives the panel\'s '
                          'first poll but not its second, so Stop lands on '
                          'work that has already finished and the stop note '
                          'is never written')
def test_stopping_an_experiment_reports_that_it_stopped(panel):
    _pick_the_smoke_experiment(panel, ragas_mode='offline')
    _build(panel)

    panel.click('#run')
    stop = panel.locator('#cancel')
    expect(stop).to_be_enabled(timeout=SETTLE)
    stop.click()
    expect(panel.locator('#jobBox')).to_contain_text('experiment stopped')


def test_stopping_an_experiment_leaves_the_page_settled(panel, lab_server):
    _pick_the_smoke_experiment(panel, ragas_mode='offline')
    _build(panel)

    panel.click('#run')
    stop = panel.locator('#cancel')
    expect(stop).to_be_enabled(timeout=SETTLE)
    stop.click()

    # However the race falls — stopped in time or finished first — the page
    # must not be left claiming that something is still running.
    expect(stop).to_be_disabled(timeout=SETTLE)
    expect(stop).not_to_have_attribute('data-job-id', re.compile('.*'),
                                       timeout=SETTLE)
    expect(panel.locator('#chromeProgress')).to_have_attribute(
        'data-running', '', timeout=SETTLE)
    jobs = panel.evaluate('url => fetch(url).then(r => r.json())',
                          f'{lab_server}/api/jobs')
    assert not [job for job in jobs['jobs']
                if job['state'] in ('running', 'cancelling')], jobs['jobs']


def test_refresh_re_reads_what_the_server_has(panel):
    _pick_the_smoke_experiment(panel)
    collection = _collection_of(_build(panel))

    # "Refresh" re-boots the page: it asks for the options again and re-renders
    # everything they decide — including the index list, which the server
    # knows about and the build report does not come from.
    with panel.expect_response(
            lambda r: r.url.endswith('/api/options')) as options:
        panel.click('#stopPoll')
    assert options.value.ok

    listing = panel.locator('#indexInfo')
    expect(listing).to_contain_text(f'{collection}: 5 chunks', timeout=SETTLE)
    # What stands there now is the server's own list of every index this lab
    # holds — including ones this page never built — in place of the build
    # report, which came from the job and not from the server.
    assert 'avg' not in listing.inner_text(), listing.inner_text()
    # The reader's own choices survive the re-boot.
    expect(panel.locator('#dataset')).to_have_value('smoke-mini')
    expect(panel.locator('#chunker')).to_have_value('session')
