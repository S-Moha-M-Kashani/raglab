# this is an end-to-end test
"""The corpus viewer as a reader actually uses it, in a real browser.

Everything this surface does happens after the page has loaded: an identity
head carrying four readings, four tabs over one panel,
tables drawn by script from a single `GET /api/dataset-content/<id>`, a
document that expands into the parts it is given as, and a raw view built out
of `<details>`. None of that can be asserted on markup — an offline test sees
`<div id="dataset">` and nothing else, and the column-derivation and narrowing
promises are pinned in JavaScript by `dataset_grids.test.js`. So this journey
drives the real thing: it reads the smoke corpus, opens a document, types a
filter, walks the three tabs with the arrow keys, switches corpora through the
picker, turns a non-zero reading into its rows, and opens the pair as given.

The explaining is asserted too, because it is now the page's only prose: every
heading carries its note on hover, on keyboard focus and on a press, and none
of it is on screen until asked for.

The surface's own promise is asserted alongside: it decides nothing. The
session guard in `conftest.py` fails the run if the developer's durable files
moved while it worked; here the strict form of the claim is what the page
sends, because anything it recorded, wrote or set it would have to ask the lab
for — and every ask it makes is a GET.
"""
import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser

import re  # noqa: E402  (after the skip guard)

from playwright.sync_api import expect  # noqa: E402


def _ready(page):
    """The page fetches and draws itself after load, so wait for a table."""
    page.wait_for_selector('table[data-grid="documents"]')
    return page


def _rows(page, grid: str):
    """The rows of one grid the reader can actually see. `filtertable.js`
    hides a row rather than removing it, so `:not([hidden])` is the reading."""
    return page.locator(f'table[data-grid="{grid}"] tbody tr:not([hidden])')


def _tab(page, name: str):
    """Open one of the three tabs. Only the chosen panel is built, so a table
    on another tab is not hidden — it is not in the document at all."""
    page.click(f'#tab-{name}')
    expect(page.locator(f'#tab-{name}')).to_have_attribute('aria-selected', 'true')
    return page


def test_the_viewer_shows_one_corpus_whole_and_changes_nothing(dataset_page):
    _ready(dataset_page)

    # The smoke corpus, by the name the picker says rather than by its id.
    expect(dataset_page.locator('#dataset h2').first).to_have_text(
        'Smoke set — five sessions')
    # Five documents, and the documents table holds all five: the whole of the
    # dataset arrives in one request, so nothing is a page away.
    expect(_rows(dataset_page, 'documents')).to_have_count(5)
    # Documents opens first, and the other three panels are not built until
    # asked for — which is why the questions table is absent rather than empty.
    expect(dataset_page.locator('table[data-grid="questions"]')).to_have_count(0)
    expect(dataset_page.locator('table[data-grid="labels"]')).to_have_count(0)
    _tab(dataset_page, 'questions')
    expect(_rows(dataset_page, 'questions')).to_have_count(6)
    expect(dataset_page.locator('table[data-grid="documents"]')).to_have_count(0)
    # Four labels declared — two on the corpus, two on the questions. Its own
    # tab, like the other three readings of the pair: it is a table, and a
    # table bounded to five rows inside the head pushed the rows a reader came
    # for below the fold on any corpus that declares more than a handful.
    _tab(dataset_page, 'labels')
    expect(_rows(dataset_page, 'labels')).to_have_count(4)
    _tab(dataset_page, 'documents')
    # No document is open yet, so there are no parts on screen at all.
    headings = dataset_page.locator('table[data-grid="parts"] thead th')
    expect(dataset_page.locator('table[data-grid="parts"]')).to_have_count(0)

    # A document expands into the parts every chunker cuts between. Document 1
    # is two turns — a note and a reply.
    dataset_page.click('button.open-doc[data-document="1"]')
    expect(dataset_page.locator(
        'button.open-doc[data-document="1"]')).to_have_attribute(
        'aria-expanded', 'true')
    expect(_rows(dataset_page, 'parts')).to_have_count(2)
    # The corpus declares `role` at the part level, so that is a column of the
    # parts table — read off the corpus's own declaration, not a fixed set.
    expect(headings.filter(has_text='role')).to_have_count(1)

    # The typed filter is the board's, over these rows: one term, one column.
    dataset_page.fill('#filter-parts', 'role=assistant')
    expect(_rows(dataset_page, 'parts')).to_have_count(1)
    dataset_page.fill('#filter-parts', '')
    expect(_rows(dataset_page, 'parts')).to_have_count(2)

    # Clicking the open document closes it again.
    dataset_page.click('button.open-doc[data-document="1"]')
    expect(dataset_page.locator('table[data-grid="parts"]')).to_have_count(0)

    # The pair as given: `<details>` all the way down, the file itself open and
    # everything under it closed until it is asked for.
    _tab(dataset_page, 'raw')
    corpus = dataset_page.locator('.raw-tree > details').first
    expect(corpus).to_have_attribute('open', '')
    documents = corpus.locator('summary', has_text='corpus_documents').first
    expect(documents).to_be_visible()

    # And nothing the lab keeps could have moved while that happened, because
    # the page never asked it to: reading, sorting, filtering and expanding all
    # happen in the browser, and the only thing that crosses the wire is a GET.
    sent = []
    dataset_page.on('request', lambda request: sent.append(request.method))
    dataset_page.reload()
    _ready(dataset_page)
    _tab(dataset_page, 'raw')
    _tab(dataset_page, 'documents')
    dataset_page.click('button.open-doc[data-document="2"]')
    expect(dataset_page.locator('table[data-grid="parts"]')).to_have_count(1)
    assert sent and set(sent) == {'GET'}, (
        'a read-only surface may not write: it records no row, sets no knob '
        f'and moves no fingerprint, and these were sent — {sorted(set(sent))}')


def test_a_reading_turns_its_count_into_the_rows_behind_it(dataset_page):
    _ready(dataset_page)

    # Every reading on the smoke corpus is zero — every document is cited,
    # every citation resolves, every declared label is carried, and no part
    # holds a blank line. All four are still on the page, saying zero: that is
    # what lets a reader tell a clean corpus from a check that did not run.
    expect(dataset_page.locator('.readings .reading')).to_have_count(4)
    expect(dataset_page.locator('.readings .reading-zero')).to_have_count(4)
    expect(dataset_page.locator('.readings button.reading')).to_have_count(0)

    # A corpus with something to report, reached the way a reader reaches it.
    dataset_page.click('.context-scope')
    dataset_page.click('.pick[data-dataset="support-en"]')
    _ready(dataset_page)
    uncited = dataset_page.locator('button.reading[data-reading="uncited-documents"]')
    expect(uncited).to_have_count(1)
    counted = int(uncited.locator('.reading-count').inner_text())
    assert counted, 'support-en has documents no question cites'

    whole = _rows(dataset_page, 'documents').count()
    assert whole > counted, 'the reading is about some of the documents, not all'
    uncited.click()
    expect(uncited).to_have_attribute('aria-pressed', 'true')
    expect(_rows(dataset_page, 'documents')).to_have_count(counted)

    # And pressing it again gives the whole corpus back.
    uncited.click()
    expect(_rows(dataset_page, 'documents')).to_have_count(whole)


def test_the_page_explains_itself_only_when_asked(dataset_page):
    _ready(dataset_page)

    # No prose on the surface. The page used to open with a paragraph and carry
    # another under each table; what is on screen now is figures and rows.
    expect(dataset_page.locator('#dataset p.explain')).to_have_count(0)
    expect(dataset_page.locator('#help-brief')).to_have_count(0)

    heading = dataset_page.locator('#dataset h2').first
    expect(heading).to_have_class(re.compile(r'why-term'))

    # A pointer resting on the title opens the whole note beside it.
    heading.hover()
    brief = dataset_page.locator('#help-brief')
    expect(brief).to_be_visible()
    assert 'writes no run' in brief.inner_text(), brief.inner_text()
    # And it is the whole note, not an opening sentence with the rest withheld.
    expect(brief).to_have_attribute('data-more', 'false')

    # A keyboard reaches the same words: `lab.js` answers `focusin` with no
    # delay, which is what keeps this from being a mouse-only publication.
    dataset_page.mouse.move(0, 0)
    dataset_page.locator('#tab-questions').focus()
    expect(dataset_page.locator('#help-brief')).to_be_visible()
    assert 'abstain' in dataset_page.locator('#help-brief').inner_text()

    # And a press pins it under the heading, for a touch screen with no hover
    # at all. Pressing again puts it away.
    heading.click()
    pinned = dataset_page.locator('#dataset p.explain')
    expect(pinned).to_have_count(1)
    assert 'writes no run' in pinned.inner_text()
    heading.click()
    expect(dataset_page.locator('#dataset p.explain')).to_have_count(0)


def test_the_arrow_keys_walk_the_four_tabs(dataset_page):
    _ready(dataset_page)
    expect(dataset_page.locator('#tab-documents')).to_have_attribute(
        'aria-selected', 'true')

    # A roving tabindex is what makes four buttons a tablist: the group takes
    # one tab stop and the arrows move inside it.
    expect(dataset_page.locator('#tab-questions')).to_have_attribute(
        'tabindex', '-1')
    dataset_page.locator('#tab-documents').focus()
    dataset_page.keyboard.press('ArrowRight')
    expect(dataset_page.locator('#tab-questions')).to_have_attribute(
        'aria-selected', 'true')
    expect(_rows(dataset_page, 'questions')).to_have_count(6)
    dataset_page.keyboard.press('ArrowRight')
    expect(_rows(dataset_page, 'labels')).to_have_count(4)
    dataset_page.keyboard.press('End')
    expect(dataset_page.locator('#tab-raw')).to_have_attribute(
        'aria-selected', 'true')
    expect(dataset_page.locator('.raw-tree')).to_be_visible()
    dataset_page.keyboard.press('Home')
    expect(dataset_page.locator('#tab-documents')).to_have_attribute(
        'aria-selected', 'true')


def test_a_reading_opens_the_tab_holding_the_rows_it_named(dataset_page):
    # A count is only turned into its rows if those rows are on screen. The
    # smoke corpus reports zero for all four, so this needs a corpus with
    # something to say — and the reading it presses is about questions, which
    # live behind a tab the reader has not opened.
    _ready(dataset_page)
    dataset_page.click('.context-scope')
    dataset_page.click('.pick[data-dataset="support-en"]')
    _ready(dataset_page)
    uncited = dataset_page.locator(
        'button.reading[data-reading="uncited-documents"]')
    counted = int(uncited.locator('.reading-count').inner_text())
    _tab(dataset_page, 'raw')
    uncited.click()
    expect(dataset_page.locator('#tab-documents')).to_have_attribute(
        'aria-selected', 'true')
    expect(_rows(dataset_page, 'documents')).to_have_count(counted)


def test_the_arrows_walk_the_rows_and_enter_opens_one(dataset_page):
    """A table you can read from the keyboard, not only look at.

    A scroll region that takes focus already scrolls with the arrows, which
    moves the viewport and nothing else: there is no "the row I am on", so
    there is nothing for Enter to open. Here the arrows move a row instead —
    the row itself takes focus, so a screen reader is told the whole of it —
    and Enter presses the row's own control, which on this page is the button
    that opens a document into its parts.
    """
    _ready(dataset_page)
    region = dataset_page.locator('.table-scroll').first
    region.focus()

    def on():
        return dataset_page.locator('tr.row-here td').first.inner_text().strip()

    dataset_page.keyboard.press('ArrowDown')
    assert on() == '1', 'arriving from the region lands on the first row'
    dataset_page.keyboard.press('ArrowDown')
    assert on() == '2'
    dataset_page.keyboard.press('End')
    assert on() == '5', 'End is the last row the reader can see'
    dataset_page.keyboard.press('Home')
    assert on() == '1'

    # And the row is what holds focus, which is what makes Enter mean
    # something: it presses the control the row carries.
    assert dataset_page.evaluate(
        '() => document.activeElement.tagName') == 'TR'
    dataset_page.keyboard.press('Enter')
    expect(dataset_page.locator('table[data-grid="parts"]')).to_have_count(1)


def test_every_column_says_what_it_means(dataset_page):
    """No heading is left to be guessed at.

    The same gate the knobs and the metrics are held to: a column named
    `feeling_confidence` or `expects` means nothing until it says so, and a
    `title` attribute would say it to a mouse and to nothing else. So every
    heading and every group above it carries `data-brief`, which `lab.js`
    opens on hover and on keyboard focus alike — and `sorttable.js` leaves its
    own native tooltip off any heading that has one, so a reader never gets two
    tooltips over one word.
    """
    _ready(dataset_page)
    for grid in ('documents', 'questions', 'labels'):
        _tab(dataset_page, grid)
        heads = dataset_page.locator(f'table[data-grid="{grid}"] thead th')
        assert heads.count(), grid
        for at in range(heads.count()):
            head = heads.nth(at)
            said = head.get_attribute('data-brief') or ''
            assert said.strip(), (
                f'{grid}: the {head.inner_text()!r} heading explains nothing')
            assert not head.get_attribute('title'), (
                f'{grid}: {head.inner_text()!r} carries a native tooltip as '
                'well as its note — a mouse would get both')
