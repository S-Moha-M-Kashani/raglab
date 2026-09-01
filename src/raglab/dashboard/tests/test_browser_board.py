# this is an end-to-end test
"""The leaderboard as a reader actually uses it, in a real browser.

Everything this surface does happens after the page has loaded: the board is
one table drawn by script from `GET /api/leaderboard`, sorted and filtered by
two more scripts that read cells the renderer wrote, and a settings panel that
only exists once a pointer or a keyboard has asked for it. None of that can be
asserted on markup — an offline test sees a `<div id="board">` and nothing
else. So this journey drives the real thing: it reads the table one recorded
experiment produced, sorts every column it offers, narrows it and clears it
again, picks a corpus that has nothing in it, opens the settings behind a row
and the reason behind a row that failed, and then follows the one link that
leaves the page — the open arrow, which is supposed to land the Laboratory on
that experiment with its knobs set and the Inspector link pointing at it.

The board's own promise is asserted alongside: it groups by dataset and it
names no winner. Rows graded by different judges over different question sets
share it, so it offers columns to compare on and never a verdict.
"""
import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser

import functools  # noqa: E402  (after the skip guard)
import re  # noqa: E402

import httpx  # noqa: E402
from playwright.sync_api import expect  # noqa: E402

from raglab.dashboard.tests.conftest import (  # noqa: E402
    SMOKE_INDEX, finish_job, start_job)


# --- more rows than one, so a sort has something to order ---------------------

@pytest.fixture(scope='session')
def a_second_recorded_experiment(lab_server, a_recorded_experiment) -> str:
    """A second experiment on the same corpus, unlike the first in every column.

    One row proves a table renders; it proves nothing about ordering it. A
    retrieval rather than a second evaluation, because a run file is named
    after the second it started in and two evaluations of this corpus finish
    inside one — and it stays on `smoke-mini`, because another corpus would
    put it in another table.
    """
    job = start_job(lab_server, '/api/retrievals', {
        'index': dict(SMOKE_INDEX),
        'retrieval': {'k': 3, 'retriever': 'dense', 'reranker': 'none',
                      'grader': 'none'},
        'generation': {'answerer': 'extractive'}, 'limit': 3})
    # A retrieval writes no run file, so the ledger row — and the board row —
    # is the job itself.
    return job['id']


@pytest.fixture(scope='session')
def a_failed_experiment(lab_server) -> str:
    """One ledger row that did not finish, and says why.

    A retrieval pointed at a corpus this lab does not have: the request is
    accepted, the job raises when it goes to load the ground truth, and the
    ledger keeps the refusal verbatim. That is the only way to get the board's
    failure mark on screen, since the mark renders from `state != done` plus a
    non-empty `error`.
    """
    started = httpx.post(f'{lab_server}/api/retrievals', json={
        'index': {'dataset': 'no-such-corpus', 'chunker': 'session',
                  'embedder': 'token-hash'},
        'retrieval': {'k': 2, 'reranker': 'none', 'grader': 'none'},
        'generation': {'answerer': 'extractive'}}, timeout=30.0)
    assert started.status_code == 202, started.text
    job_id = started.json()['job_id']
    job = finish_job(lab_server, job_id)
    assert job['state'] == 'error', job
    return job_id


@pytest.fixture(scope='session')
def a_populated_board(a_recorded_experiment, a_second_recorded_experiment,
                      a_failed_experiment) -> dict:
    """The three experiment ids every journey below reads the board through."""
    return {'first': a_recorded_experiment,
            'second': a_second_recorded_experiment,
            'failed': a_failed_experiment}


# --- reading the table the way the two scripts read it ------------------------

#: Every column that sorts, by the name its heading filters under. `open` is
#: the one column marked `data-nosort`, so it is not here.
SORTABLE = ['pipeline', 'dataset', 'questions', 'decision', 'spread', 'faith',
            'ans-rel', 'ctx-prec', 'ctx-recall', 'kind', 'when', 'label',
            'judge', 'backend', 'state', 'seconds']

#: One cell's worth of the page, in JavaScript, because `data-sort` and the
#: cell's own text deliberately differ — the pipeline column shows an
#: abbreviation and sorts on the whole sentence, `questions` shows
#: "12 (easy=4, hard=8)" and sorts on 12. Reading `textContent` to check a sort
#: would disagree with the sorter about half the board.
_COLUMN_JS = """
(name) => {
  const table = document.querySelector('#board table');
  const head = table.tHead.rows[table.tHead.rows.length - 1];
  const names = Array.from(head.cells).map((th) =>
    th.getAttribute('data-filter')
    || th.textContent.toLowerCase().trim().replace(/\\s+/g, '-'));
  const at = names.indexOf(name);
  const rows = Array.from(table.tBodies[0].rows).map((tr) => {
    const cell = tr.cells[at];
    const told = cell.getAttribute('data-sort');
    const link = tr.querySelector('a.open-run');
    return { id: link ? link.dataset.experiment : '',
             key: told === null ? cell.textContent : told };
  });
  return { at, rows };
}
"""


def _ready(page):
    """The board is fetched and drawn after load, so wait for the table itself."""
    page.wait_for_selector('#board table')
    return page


def _column(page, name: str) -> dict:
    return _ready(page).evaluate(_COLUMN_JS, name)


def _heading(page, at: int):
    return page.locator('#board thead th').nth(at)


_MISSING = {'', '—', '–', '·', 'n/a'}
_LEADING_NUMBER = re.compile(r'^([+-]?\d+(?:\.\d+)?)([\s\S]*)$')
_NOT_A_UNIT = re.compile(r'^[-/:.]\d')


def _key(text: str):
    """`sorttable.js`'s `cellKey`, restated — the promise, not the code.

    A dash of any of the four spellings is "never measured"; a leading figure
    with a unit after it is that figure; a timestamp is text, because
    '2026-09-02' read as 2026 would collapse a whole column into one year.
    """
    shown = (text or '').strip()
    if shown.lower() in _MISSING:
        return (True, None, '')
    probe = shown.replace(',', '')
    number = None
    found = _LEADING_NUMBER.match(probe)
    if found and not _NOT_A_UNIT.match(found.group(2)):
        number = float(found.group(1))
    elif not found:
        error = re.match(r'^±\s*([+-]?\d+(?:\.\d+)?)$', probe)
        if error:
            number = float(error.group(1))
    return (False, number, shown)


#: How the browser itself orders two words. `sorttable.js` compares text with
#: `localeCompare(..., {sensitivity: 'base', numeric: true})`, and that is the
#: platform's collation, not the lab's: it puts a middle dot before a plus
#: sign, where a comparison on code points puts it after. So the mirror below
#: keeps every rule the sorter states — missing last, figures as figures, ties
#: in served order — and asks the page for this one ordering rather than
#: guessing at it. Collation-equal words share a rank, so they stay ties.
_RANK_JS = """
(texts) => {
  const collator = new Intl.Collator(undefined,
    { sensitivity: 'base', numeric: true });
  const sorted = Array.from(new Set(texts)).sort(collator.compare);
  const ranked = [];
  let rank = 0;
  sorted.forEach((text, at) => {
    if (at > 0 && collator.compare(sorted[at - 1], text) !== 0) rank += 1;
    ranked.push([text, rank]);
  });
  return ranked;
}
"""


def _compare(left, right, ranks, direction: int) -> int:
    """`sorttable.js`'s `compare`: missing last whichever way the column runs."""
    left_missing, left_number, left_text = left
    right_missing, right_number, right_text = right
    if left_missing or right_missing:
        if left_missing and right_missing:
            return 0
        return 1 if left_missing else -1
    if left_number is not None and right_number is not None:
        return int((left_number > right_number)
                   - (left_number < right_number)) * direction
    low, high = ranks[left_text], ranks[right_text]
    return int((low > high) - (low < high)) * direction


def _ordered(page, rows: list[dict], direction: int) -> list[str]:
    """The row ids in the order the sorter promises, ties keeping served order."""
    keys = [_key(row['key']) for row in rows]
    ranks = dict(page.evaluate(
        _RANK_JS, [key[2] for key in keys if not key[0]]))
    numbered = list(enumerate(zip(rows, keys)))
    numbered.sort(key=functools.cmp_to_key(
        lambda x, y: _compare(x[1][1], y[1][1], ranks, direction)
        or (x[0] - y[0])))
    return [row['id'] for _, (row, _key_of) in numbered]


def _opens(rows: list[dict]) -> int:
    """Numbers open best-first, words open A-Z. No board column overrides it."""
    return -1 if any(_key(row['key'])[1] is not None for row in rows) else 1


def _said(direction: int) -> str:
    return 'ascending' if direction == 1 else 'descending'


def _visible_ids(page) -> list[str]:
    return _ready(page).evaluate("""() => Array.from(
        document.querySelectorAll('#board tbody tr:not([hidden]) a.open-run'))
        .map((a) => a.dataset.experiment)""")


def _visible_kinds(page) -> list[str]:
    rows = _column(page, 'kind')['rows']
    hidden = page.evaluate("""() => Array.from(
        document.querySelectorAll('#board tbody tr'))
        .filter((tr) => tr.hidden)
        .map((tr) => { const a = tr.querySelector('a.open-run');
                       return a ? a.dataset.experiment : ''; })""")
    return [row['key'] for row in rows if row['id'] not in hidden]


def _pick(page, dataset: str) -> None:
    """Choose a corpus the way a reader does — through the native popover."""
    page.click('button.context-scope')
    page.click(f'#pick-list button.pick[data-dataset="{dataset}"]')


def _row(page, experiment_id: str):
    return page.locator(
        f'#board tbody tr:has(a.open-run[data-experiment="{experiment_id}"])')


# --- the board itself ---------------------------------------------------------

def test_the_board_shows_every_recorded_experiment_and_names_no_winner(
        a_populated_board, board):
    _ready(board)
    expect(board.locator('#board table')).to_have_count(1)
    expect(board.locator('#board caption')).to_have_text(
        'every experiment on every dataset')

    for experiment_id in a_populated_board.values():
        expect(_row(board, experiment_id)).to_have_count(1)

    # The count beside the heading is the board's own answer to "is this all of
    # it", so it has to agree with the table under it.
    shown = board.locator('#board tbody tr').count()
    expect(board.locator('#board .section-meta.right')).to_have_text(
        f'{shown} recorded · best score first')

    # The rule, said on the page and kept by it: the prose states it, and no
    # cell, class or mark in the table then contradicts it by naming one.
    expect(board.locator('.table-hint').first).to_contain_text(
        'This table names no winner')
    assert board.locator('[class*="winner"], [data-winner]').count() == 0
    assert 'winner' not in board.locator('#board table').inner_text().lower()


def test_each_dataset_gets_its_own_table_and_the_picker_switches_between_them(
        a_populated_board, board):
    """One table per dataset — chosen, not concatenated, so the picker is it."""
    _ready(board)
    # Every experiment: the corpus is a question, so the column is there.
    expect(board.locator('#board th', has_text=re.compile('^dataset$'))
           ).to_have_count(1)
    expect(_row(board, a_populated_board['failed'])).to_have_count(1)

    _pick(board, 'smoke-mini')

    expect(board.locator('#board h2')).to_have_text('Smoke set — five sessions')
    expect(board.locator('#board caption')).to_have_text(
        'every experiment on Smoke set — five sessions')
    # Inside one corpus's table the dataset column would say the same word on
    # every row, so it is not drawn at all.
    expect(board.locator('#board th', has_text=re.compile('^dataset$'))
           ).to_have_count(0)
    # The evaluation and the retrieval belong to this corpus; the failed one
    # named a corpus this lab does not have, so it is in no corpus's table.
    expect(_row(board, a_populated_board['first'])).to_have_count(1)
    expect(_row(board, a_populated_board['second'])).to_have_count(1)
    expect(_row(board, a_populated_board['failed'])).to_have_count(0)

    # A view somebody wants to come back to is a view they can send.
    assert 'dataset=smoke-mini' in board.url
    board.reload()
    expect(board.locator('button.context-scope')).to_contain_text(
        'Smoke set — five sessions')


def test_a_dataset_with_nothing_recorded_says_so_instead_of_showing_a_table(
        a_populated_board, board):
    _pick(board, 'research-multihop')

    expect(board.locator('#board .prose')).to_contain_text(
        'Nothing recorded for this dataset yet.')
    # No table means no filter bar either: a bar that could narrow nothing.
    expect(board.locator('#board table')).to_have_count(0)
    expect(board.locator('#row-filter')).to_have_count(0)


@pytest.mark.parametrize('column', SORTABLE)
def test_every_sortable_heading_sorts_reverses_and_restores(
        a_populated_board, board, column):
    """Three states per column, and the third is the order it was served in.

    Checked against the keys the sorter itself reads — `data-sort` where the
    renderer wrote one — because the pipeline cell shows an abbreviation of a
    sentence and the questions cell shows a count inside a sample.
    """
    served = _column(board, column)
    heading = _heading(board, served['at'])
    ids = [row['id'] for row in served['rows']]
    direction = _opens(served['rows'])

    heading.click()
    expect(heading).to_have_attribute('aria-sort', _said(direction))
    assert [row['id'] for row in _column(board, column)['rows']] \
        == _ordered(board, served['rows'], direction)

    heading.click()
    expect(heading).to_have_attribute('aria-sort', _said(-direction))
    assert [row['id'] for row in _column(board, column)['rows']] \
        == _ordered(board, served['rows'], -direction)

    heading.click()
    expect(heading).to_have_attribute('aria-sort', 'none')
    assert [row['id'] for row in _column(board, column)['rows']] == ids
    # A column that is no longer the sort key still announces that it can be
    # sorted, so every sortable heading says 'none' and none of them is blank.
    assert board.locator(
        '#board thead th:not([data-nosort])[aria-sort="none"]').count() \
        == board.locator('#board thead th:not([data-nosort])').count()


def test_a_heading_sorts_from_the_keyboard_too(a_populated_board, board):
    """Enter and Space do what a click does — the column is not a mouse control."""
    served = _column(board, 'kind')
    heading = _heading(board, served['at'])
    direction = _opens(served['rows'])

    heading.focus()
    board.keyboard.press('Enter')
    expect(heading).to_have_attribute('aria-sort', _said(direction))
    assert [row['id'] for row in _column(board, 'kind')['rows']] \
        == _ordered(board, served['rows'], direction)

    board.keyboard.press(' ')
    expect(heading).to_have_attribute('aria-sort', _said(-direction))
    assert [row['id'] for row in _column(board, 'kind')['rows']] \
        == _ordered(board, served['rows'], -direction)


# --- narrowing ----------------------------------------------------------------

def test_the_filter_hides_rows_and_clearing_brings_them_back(
        a_populated_board, board):
    total = _ready(board).locator('#board tbody tr').count()
    expect(board.locator('#filter-count')).to_have_text(f'{total} rows')

    board.fill('#row-filter', 'kind:run')

    shown = _visible_ids(board)
    # The evaluation answers `kind:run`; the two retrievals and the index
    # build do not, and every row still standing says `run` in that column.
    assert a_populated_board['first'] in shown
    assert a_populated_board['failed'] not in shown
    assert set(_visible_kinds(board)) == {'run'}
    expect(board.locator('#filter-count')).to_have_text(
        f'{len(shown)} of {total} shown')
    # Hidden, not removed: the sorter's record of the served order depends on
    # every row still being in the body.
    assert board.locator('#board tbody tr').count() == total
    assert 'filter=kind%3Arun' in board.url

    # A filtered board is a link, so it survives being followed again.
    board.reload()
    expect(board.locator('#row-filter')).to_have_value('kind:run')
    assert set(_visible_ids(board)) == set(shown)

    board.click('#filter-clear')

    expect(board.locator('#row-filter')).to_have_value('')
    expect(board.locator('#filter-count')).to_have_text(f'{total} rows')
    assert len(_visible_ids(board)) == total
    assert 'filter=' not in board.url
    assert board.evaluate('() => document.activeElement.id') == 'row-filter'


def test_a_filter_naming_no_column_says_so_and_leaves_the_rows_alone(
        a_populated_board, board):
    """An unanswerable question empties nothing — it is answered in words."""
    before = _visible_ids(board)
    expect(board.locator('#filter-said')).to_be_hidden()

    board.fill('#row-filter', 'nosuchcol>1')

    expect(board.locator('#filter-said')).to_be_visible()
    expect(board.locator('#filter-said')).to_contain_text(
        'no column called “nosuchcol” on this board')
    assert _visible_ids(board) == before

    # '±' is a column nobody can type, so it answers to `spread` — and that
    # spelling is a real column, which is how we know the notice above was
    # about the name and not about the syntax.
    board.fill('#row-filter', 'spread:')
    expect(board.locator('#filter-said')).to_be_hidden()
    # A colon with nothing after it asks whether the column was measured at
    # all, and none of these rows was judged.
    assert not set(_visible_ids(board)) & set(a_populated_board.values())


# --- what a row carries -------------------------------------------------------

def test_the_settings_reveal_publishes_the_knobs_that_produced_the_row(
        a_populated_board, board):
    """The pipeline cell is an abbreviation; the reveal is the whole config."""
    first = _row(board, a_populated_board['first']).locator('td.freeze-1')
    first.hover()

    reveal = first.locator('.settings-reveal')
    expect(reveal).to_be_visible()
    expect(reveal.locator('.reveal-said')).to_contain_text('token-hash')
    expect(reveal.locator('.reveal-step[data-step="index"]')).to_contain_text(
        'session')
    expect(reveal.locator('.reveal-step[data-step="generation"]')).to_contain_text(
        'extractive')
    # A knob the build never read reads `none`, never the value that sat
    # unused, and carries the reason it was never read.
    off = reveal.locator('.reveal-knob-off').first
    expect(off).to_contain_text('none')
    assert 'never used' in (off.get_attribute('title') or '')

    # Two panels of settings at once say nothing about which row is being read.
    second = _row(board, a_populated_board['second']).locator('td.freeze-1')
    second.hover()
    expect(second.locator('.settings-reveal')).to_be_visible()
    expect(reveal).to_be_hidden()

    # And it opens for a keyboard as well, or it publishes to a mouse and to
    # nothing else. The pointer is parked first so hover is not what opens it.
    board.mouse.move(0, 0)
    expect(second.locator('.settings-reveal')).to_be_hidden()
    first.focus()
    expect(reveal).to_be_visible()


def test_the_why_mark_explains_a_row_that_produced_no_score(
        a_populated_board, board):
    failed = _row(board, a_populated_board['failed'])
    expect(failed.locator('span.failed b')).to_have_text('error')

    mark = failed.locator('button.why')
    expect(mark).to_have_attribute(
        'data-help', re.compile("unknown dataset 'no-such-corpus'"))

    # Hovering is the shorter answer: the brief takes the reason's first
    # sentence, which for a refusal is the part that names what went wrong.
    mark.hover()
    expect(board.locator('#help-brief')).to_be_visible()
    expect(board.locator('#help-brief')).to_contain_text('unknown dataset')
    expect(mark).to_have_attribute('aria-describedby', 'help-brief')

    mark.click()
    explain = failed.locator('p.explain')
    expect(explain).to_be_visible()
    expect(explain).to_contain_text("unknown dataset 'no-such-corpus'")

    mark.click()
    expect(explain).to_have_count(0)


# --- the handoff --------------------------------------------------------------

def test_the_open_arrow_lands_the_laboratory_on_that_experiment(
        a_populated_board, board, lab_server):
    """The journey the board exists to end in.

    One click has to do three things at once: leave this tab on the
    Laboratory, make that experiment's recorded settings the knobs there, and
    point the Inspector link at the same experiment. The slot the two pages
    share is consumed on the way, so the panel cannot re-announce days later
    an experiment the reader opened once.
    """
    experiment_id = a_populated_board['first']
    arrow = _row(board, experiment_id).locator('a.open-run')
    expect(arrow).to_have_text('↗')
    expect(arrow).to_have_attribute('href', f'/?experiment={experiment_id}')

    arrow.click()

    # Same tab, and the address carries the experiment — a copied link works
    # where a slot written by a click could not.
    expect(board).to_have_url(f'{lab_server}/?experiment={experiment_id}')

    # The knobs are that experiment's: the corpus first, because a config
    # applied against the wrong corpus is not that experiment at all.
    expect(board.locator('#dataset')).to_have_value('smoke-mini')
    expect(board.locator('#chunker')).to_have_value('session')
    expect(board.locator('#embedder')).to_have_value('token-hash')
    expect(board.locator('#k')).to_have_value('2')
    expect(board.locator('#reranker')).to_have_value('none')
    expect(board.locator('#grader')).to_have_value('none')
    expect(board.locator('#answerer')).to_have_value('extractive')

    # And the Inspector link above now leads to the same experiment.
    expect(board.locator('.topnav a[href^="/inspector"]')).to_have_attribute(
        'href', f'/inspector?experiment={experiment_id}')

    # Taken once: a slot left behind would re-announce this on every reload.
    assert board.evaluate(
        "() => localStorage.getItem('raglab:open-experiment')") is None


# --- the chrome and the failure the page can be handed ------------------------

def test_the_page_chrome_leads_to_the_other_surfaces(board):
    board.keyboard.press('Tab')
    expect(board.locator('a.skip-link')).to_be_focused()
    board.keyboard.press('Enter')
    assert board.evaluate('() => location.hash') == '#main'

    expect(board.locator('a.brand')).to_have_attribute('href', '/')
    expect(board.locator('.topnav a[href="/"]')).to_have_text('Laboratory')
    expect(board.locator('.topnav a[href="/inspector"]')).to_have_text('Inspector')
    expect(board.locator('.topnav a[aria-current="page"]')).to_have_text(
        'Leaderboard')

    board.click('#app-settings')
    expect(board.locator('#app-settings-panel')).to_be_visible()
    expect(board.locator('#theme-control input[name="raglab-theme"]')
           ).to_have_count(3)
    board.keyboard.press('Escape')
    expect(board.locator('#app-settings-panel')).to_be_hidden()

    # The rail in the header is decoration here — the board runs no job, so
    # nothing on this page ever drives it.
    expect(board.locator('.chrome-progress')).to_have_count(1)
    assert board.locator('#chrome-progress').count() == 0


def test_a_leaderboard_the_lab_cannot_answer_says_so(a_populated_board, board):
    """A failing read reads as a failure, never as an empty lab."""
    board.route('**/api/leaderboard*', lambda route: route.fulfill(status=500))
    board.reload()

    expect(board.locator('#board h2')).to_have_text(
        'Could not read the leaderboard')
    expect(board.locator('#board table')).to_have_count(0)

