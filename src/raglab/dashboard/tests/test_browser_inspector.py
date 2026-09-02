# this is an end-to-end test
"""The Inspector as a reader actually uses it, in a real browser.

Four tabs over one read-only surface, and the two journeys the specification
names outright: a recorded experiment opened by deep link — read-only, with its
own questions and its own retrieved context on screen and every control that
would change something withheld — and a question added through the page, which
is saved with that experiment and is still there after a reload. Around those,
the controls a reader touches on the way: the tab strip and its arrow keys, the
question picker, the build and the rebuild on the Chunks tab, the sortable
candidate tables, the marks that explain a metric, and the two ways out of
read-only mode.

None of that is visible to an offline assertion on markup. Which view is
showing, whether a button is disabled, whether a reveal opened, and whether a
record survives a reload are all facts about a running page, so this journey
drives one — on the suite's own lab, on its own port, against `smoke-mini` and
the fake model.
"""
import json
import re

import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser

from playwright.sync_api import expect  # noqa: E402  (after the skip guard)

from raglab.dashboard.tests.conftest import (  # noqa: E402
    SMOKE_INDEX, STEP_TIMEOUT, start_job)

#: The four views, in the order the tab strip lists them.
VIEWS = ('groundtruth', 'chunks', 'retrieval', 'generation')

#: `smoke-mini` — six questions, ids `1`–`6`, and five sessions in its corpus.
QUESTION_IDS = ('1', '2', '3', '4', '5', '6')
SESSIONS = 5


@pytest.fixture(scope='module')
def an_index_build(lab_server) -> str:
    """One finished index build, which is a record with no archive behind it.

    Only an evaluation archives itself, so this is the shape whose Chunks tab
    has to say that the chunk text was never recorded and offer the rebuild —
    the one that runs under the record's own config rather than today's.
    """
    return start_job(lab_server, '/api/indexes',
                     {'index': dict(SMOKE_INDEX)})['id']


@pytest.fixture(scope='module')
def a_record_of_its_own(lab_server) -> str:
    """A recorded evaluation this file may write to.

    The session-wide `a_recorded_experiment` is read by every browser journey
    there is; adding a question to one leaves a second ledger row pointing at
    it, so the journey that adds gets an experiment nobody else is reading.
    """
    start_job(lab_server, '/api/indexes', {'index': dict(SMOKE_INDEX)})
    job = start_job(lab_server, '/api/evaluations', {
        'index': dict(SMOKE_INDEX),
        'retrieval': {'k': 2, 'reranker': 'none', 'grader': 'none'},
        'generation': {'answerer': 'extractive'},
        'ragas_mode': 'off', 'limit': 2, 'label': 'browser-added-questions'})
    return job['result']['run_id']


def _pin(page, lab: str, experiment_id: str):
    """Open one recorded experiment the way the board's `↗` opens it."""
    page.goto(f'{lab}/inspector?experiment={experiment_id}')
    expect(page.locator('#archive-state')).to_contain_text('read-only')
    return page


def _remembered(page):
    return page.evaluate(
        "() => localStorage.getItem('raglab:inspector-experiment')")


def test_the_live_inspector_opens_every_tab_and_each_shows_its_own_view(
        lab_server, inspector):
    expect(inspector.locator('#follow-state')).to_have_attribute('data-lab', 'up')
    # Ground truth is the view a fresh page lands on, before anything is clicked.
    expect(inspector.locator('#view-groundtruth')).to_be_visible()

    own_content = {
        'groundtruth': '#view-groundtruth .gt-row',
        'chunks': '#build-chunks',
        'retrieval': '#add-question',
        'generation': '#generation-active-config',
    }
    for view in VIEWS:
        inspector.click(f'#tab-{view}')
        expect(inspector.locator(f'#tab-{view}')).to_have_attribute(
            'aria-selected', 'true')
        expect(inspector.locator(f'#view-{view}')).to_be_visible()
        expect(inspector.locator(own_content[view]).first).to_be_visible()
        for other in VIEWS:
            if other != view:
                expect(inspector.locator(f'#view-{other}')).to_be_hidden()


def test_the_tab_strip_is_one_stop_and_the_arrow_keys_walk_it(lab_server, inspector):
    """A tablist is one stop in the page's order, and the arrows move inside it."""
    inspector.focus('#tab-groundtruth')
    inspector.keyboard.press('ArrowRight')
    expect(inspector.locator('#view-chunks')).to_be_visible()
    expect(inspector.locator('#tab-chunks')).to_be_focused()

    inspector.keyboard.press('End')
    expect(inspector.locator('#view-generation')).to_be_visible()
    expect(inspector.locator('#tab-generation')).to_be_focused()

    inspector.keyboard.press('Home')
    expect(inspector.locator('#view-groundtruth')).to_be_visible()
    # Only the selected tab is reachable by Tab; the other three are -1.
    expect(inspector.locator('#tab-groundtruth')).to_have_attribute('tabindex', '0')
    expect(inspector.locator('#tab-retrieval')).to_have_attribute('tabindex', '-1')


def test_the_ground_truth_tab_lists_the_corpus_questions_with_their_evidence(
        a_recorded_experiment, lab_server, inspector):
    """The corpus the lab is working on, which is the one it last ran against.

    A lab that has finished nothing names no corpus, and the tab then reads the
    built-in diary — so the recorded experiment is asked for first, and the
    questions below are the ones it measured.
    """
    rows = inspector.locator('#view-groundtruth .gt-row')
    expect(rows).to_have_count(len(QUESTION_IDS))
    assert inspector.locator('#view-groundtruth .gt-head .q-id')\
        .all_text_contents() == list(QUESTION_IDS)

    first = rows.first
    expect(first.locator('.gt-q')).to_contain_text('kitchen')
    expect(first.locator('.qh-facts li').first).to_be_visible()
    # A verbatim quote is marked because it is really findable in the corpus.
    expect(first.locator('mark.evidence-mark').first).to_be_visible()
    expect(first.locator('.gt-supports').first).to_be_visible()
    # An English corpus reads left to right, and the page says so once.
    expect(inspector.locator('html')).to_have_attribute('data-corpus-dir', 'ltr')


def test_building_an_index_lists_its_chunks_and_says_the_index_is_flat(
        lab_server, inspector):
    """The build button, both mode buttons, and one session opened.

    The button posts whatever config this installation falls back to — the
    built-in diary under a downloaded embedder — and no browser journey builds
    that. The request is redirected at the smoke corpus on its way out, which
    changes what is built and nothing about the control being pressed.
    """
    inspector.route('**/inspector/api/chunks', lambda route: route.continue_(
        post_data=json.dumps({'index': dict(SMOKE_INDEX)})))
    inspector.click('#tab-chunks')
    expect(inspector.locator('#build-chunks')).to_be_enabled()

    inspector.click('#build-chunks')

    expect(inspector.locator('#chunks-status')).to_have_text(
        re.compile(r'^\d+ chunks · \d+ summaries$'), timeout=STEP_TIMEOUT)
    sessions = inspector.locator('#chunks-body details.chunk-session')
    expect(sessions).to_have_count(SESSIONS)
    sessions.first.locator('summary').click()
    expect(sessions.first.locator('.chunk-line').first).to_be_visible()

    inspector.click('#chunks-mode-summaries')
    expect(inspector.locator('#summaries-body')).to_be_visible()
    expect(inspector.locator('#chunks-body')).to_be_hidden()
    expect(inspector.locator('#chunks-mode-summaries')).to_have_attribute(
        'aria-pressed', 'true')
    # Nothing was grouped, and "flat" is a different fact from "found nothing".
    expect(inspector.locator('#summaries-body p.empty-note')).to_contain_text('flat')

    inspector.click('#chunks-mode-chunks')
    expect(inspector.locator('#chunks-body')).to_be_visible()
    expect(inspector.locator('#summaries-body')).to_be_hidden()


def test_the_question_picker_filters_walks_and_closes_where_it_opened(
        a_recorded_experiment, inspector):
    """The picker offers the followed corpus's questions — hence the record."""
    inspector.click('#tab-retrieval')
    inspector.click('#add-question')
    expect(inspector.locator('#add-question')).to_have_attribute(
        'aria-expanded', 'true')
    options = inspector.locator('#question-picker-list .q-option')
    expect(options).to_have_count(len(QUESTION_IDS))

    inspector.fill('#question-picker-filter', 'kitchen')
    expect(options).to_have_count(1)
    expect(options.first).to_have_attribute('data-id', '1')

    inspector.fill('#question-picker-filter', 'nothing matches this')
    expect(inspector.locator('#question-picker-list .q-empty')).to_be_visible()

    inspector.fill('#question-picker-filter', '')
    expect(options).to_have_count(len(QUESTION_IDS))
    # The options are divs with role=option, so only the arrows reach them.
    inspector.keyboard.press('ArrowDown')
    inspector.keyboard.press('ArrowDown')
    assert inspector.evaluate('() => document.activeElement.dataset.id') == '2'
    # Its detail opens on focus alone — CSS, with no attribute to read.
    assert inspector.evaluate(
        "() => getComputedStyle(document.activeElement"
        ".querySelector('.q-option-detail')).display") == 'block'

    inspector.keyboard.press('Escape')
    expect(inspector.locator('#question-picker')).to_be_hidden()
    expect(inspector.locator('#add-question')).to_have_attribute(
        'aria-expanded', 'false')
    expect(inspector.locator('#add-question')).to_be_focused()


def test_a_deep_linked_record_is_read_only_and_shows_its_own_evidence(
        lab_server, inspector, a_recorded_experiment):
    """The journey the board's `↗` starts: one row of the ledger, read back."""
    _pin(inspector, lab_server, a_recorded_experiment)

    banner = inspector.locator('#archive-state')
    expect(banner).to_be_visible()
    expect(banner).to_have_text(
        f'Recorded experiment (run) · read-only · {a_recorded_experiment}')
    expect(inspector.locator('#archive-return-live')).to_have_text('Return to live')
    # Read-only is not a word here: the one control that would build something
    # is actually withheld.
    expect(inspector.locator('#build-chunks')).to_be_disabled()
    assert _remembered(inspector) == a_recorded_experiment

    # Its own questions, and its own retrieved context under them.
    inspector.click('#tab-retrieval')
    questions = inspector.locator('#retrieval-questions details.retrieval-question')
    expect(questions).to_have_count(2)
    expect(inspector.locator('#retrieval-set-config')).to_contain_text('session')
    questions.first.locator('summary').click()
    expect(questions.first.locator('tbody tr.retrieval-row').first).to_be_visible()

    # And the chunks it ran over, from the archive it wrote when it finished —
    # never from a rebuild, which would be today's index under an old id.
    inspector.click('#tab-chunks')
    expect(inspector.locator('#chunks-status')).to_contain_text(
        "from this experiment's own archive")
    expect(inspector.locator('#chunks-body details.chunk-session')).to_have_count(
        SESSIONS)
    expect(inspector.locator('#rebuild-recorded-chunks')).to_have_count(0)


def test_each_recorded_question_keeps_its_own_sortable_table_of_candidates(
        lab_server, inspector, a_recorded_experiment):
    """Moving between the record's questions moves the rows under them."""
    _pin(inspector, lab_server, a_recorded_experiment)
    inspector.click('#tab-retrieval')
    questions = inspector.locator('#retrieval-questions details.retrieval-question')
    expect(questions).to_have_count(2)

    seen = []
    for index in range(2):
        question = questions.nth(index)
        question.locator('summary').click()
        rows = question.locator('tbody tr.retrieval-row')
        expect(rows.first).to_be_visible()
        counted = int(re.search(r'(\d+) candidates',
                                question.locator('.q-tally').inner_text()).group(1))
        expect(rows).to_have_count(counted)
        seen.append(question.locator('summary .q-id').inner_text())
    assert seen[0] != seen[1], 'the two tables are the same question'

    first = questions.first
    # Eight of the nine columns sort; `path` is the same three ranks drawn as a
    # shape, and says so by never being wired.
    headers = first.locator('th.sort-col')
    expect(headers).to_have_count(8)
    for index in range(8):
        headers.nth(index).click()
        expect(headers.nth(index)).not_to_have_attribute('aria-sort', 'none')
    expect(first.locator('th[data-nosort]')).not_to_have_class(re.compile('sort-col'))

    # The full chunk, revealed by hovering anywhere on its row.
    row = first.locator('tbody tr.retrieval-row').first
    row.hover()
    expect(row.locator('.chunk-reveal')).to_be_visible()
    # And the region the nine columns scroll inside takes focus of its own.
    expect(first.locator('div.table-scroll[role="region"]')).to_have_attribute(
        'tabindex', '0')


def test_a_recorded_run_shows_its_answers_and_the_marks_that_explain_them(
        lab_server, inspector, a_recorded_experiment):
    _pin(inspector, lab_server, a_recorded_experiment)
    inspector.click('#tab-generation')

    # Nothing was judged, and the view says which scores exist instead.
    expect(inspector.locator('#generation-ragas')).to_contain_text('ragas_mode=off')
    answers = inspector.locator('#generation-questions details.gen-question')
    expect(answers).to_have_count(2)

    first = answers.first
    # Its own summary, not the nested trace's.
    first.locator('summary').first.click()
    expect(first.locator('.gen-answer--ideal')).to_contain_text('what the ground truth says')
    expect(first.locator('.gen-answer--actual')).to_contain_text('what this run wrote')
    metrics = first.locator('.gen-metric')
    assert metrics.count() >= 1
    # The retrieval this answer was written from, nested under it because it is
    # the same job.
    expect(first.locator('details.gen-trace')).to_have_count(1)

    mark = first.locator('button.why').first
    mark.focus()
    brief = inspector.locator('#help-brief')
    expect(brief).to_be_visible()
    assert brief.inner_text().strip()
    expect(mark).to_have_attribute('aria-describedby', 'help-brief')

    mark.click()
    note = first.locator('span.why-text').first
    expect(note).to_be_visible()
    assert note.inner_text().strip()
    mark.click()
    expect(first.locator('span.why-text')).to_have_count(0)


def test_a_record_without_an_archive_offers_a_rebuild_from_its_own_config(
        lab_server, inspector, an_index_build):
    """An index build measured no questions, and kept no chunk text either."""
    _pin(inspector, lab_server, an_index_build)
    expect(inspector.locator('#archive-state')).to_contain_text(
        'Recorded experiment (index)')

    inspector.click('#tab-generation')
    expect(inspector.locator('#generation-questions p.empty-note')).to_contain_text(
        'an index build chunks and embeds a corpus and stops there')
    inspector.click('#tab-retrieval')
    expect(inspector.locator('#retrieval-questions p.empty-note')).to_contain_text(
        'recorded no per-question retrieval')

    inspector.click('#tab-chunks')
    expect(inspector.locator('#chunks-body p.empty-note').first).to_contain_text(
        'not recorded')
    rebuild = inspector.locator('#rebuild-recorded-chunks')
    expect(rebuild).to_be_visible()

    rebuild.click()

    expect(inspector.locator('#chunks-status')).to_have_text(
        re.compile(r'^\d+ chunks · \d+ summaries$'), timeout=STEP_TIMEOUT)
    expect(inspector.locator('#chunks-body details.chunk-session')).to_have_count(
        SESSIONS)


def test_returning_to_live_forgets_a_record_a_bare_load_would_re_pin(
        lab_server, inspector, a_recorded_experiment):
    """The remembered id is what makes `/inspector` sticky — until it is left."""
    _pin(inspector, lab_server, a_recorded_experiment)

    # A bare load re-pins, because that is what remembering it is for.
    inspector.goto(f'{lab_server}/inspector')
    expect(inspector.locator('#archive-state')).to_contain_text(
        a_recorded_experiment)

    inspector.click('#archive-return-live')

    expect(inspector.locator('#archive-state')).to_be_hidden()
    expect(inspector.locator('#archive-return-live')).to_be_hidden()
    expect(inspector.locator('#build-chunks')).to_be_enabled()
    expect(inspector.locator('#view-groundtruth .gt-row')).to_have_count(
        len(QUESTION_IDS))
    assert _remembered(inspector) is None
    assert 'experiment=' not in inspector.url

    # And now a bare load really is bare.
    inspector.goto(f'{lab_server}/inspector')
    expect(inspector.locator('#archive-state')).to_be_hidden()
    expect(inspector.locator('#build-chunks')).to_be_enabled()


def test_a_record_the_lab_cannot_read_leaves_the_page_live_and_dismissible(
        lab_server, inspector):
    inspector.goto(f'{lab_server}/inspector?experiment=no-such-experiment')

    banner = inspector.locator('#archive-state')
    expect(banner).to_contain_text('could not be read from the lab')
    expect(banner).to_contain_text('showing live instead')
    # Nothing was pinned, so there is nothing to return from — only a message.
    expect(inspector.locator('#archive-return-live')).to_have_text('Dismiss')
    expect(inspector.locator('#build-chunks')).to_be_enabled()
    # Live ground truth, not a record's: which corpus it is has its own test.
    expect(inspector.locator('#view-groundtruth .gt-row').first).to_be_visible()
    assert _remembered(inspector) is None

    inspector.click('#archive-return-live')

    expect(banner).to_be_hidden()
    expect(inspector.locator('#archive-return-live')).to_be_hidden()
    assert 'experiment=' not in inspector.url


def test_a_failed_pin_leaves_the_page_on_the_corpus_the_lab_follows(
        a_recorded_experiment, lab_server, inspector):
    """Saying one experiment could not be read must not change the corpus.

    The failure branch asks for the ground truth again so the tab is not left
    empty, and asks under the corpus the page knew at boot — which is not yet
    the one the lab is following. Two live loads are therefore in flight, and
    the later one wins whichever order they land in.
    """
    inspector.goto(f'{lab_server}/inspector?experiment=no-such-experiment')
    expect(inspector.locator('#archive-state')).to_contain_text('showing live instead')

    expect(inspector.locator('#view-groundtruth .gt-row')).to_have_count(
        len(QUESTION_IDS))


def test_a_question_added_to_a_record_is_saved_with_it_and_survives_a_reload(
        lab_server, inspector, a_record_of_its_own):
    """The second journey the specification names: add one, then reload."""
    _pin(inspector, lab_server, a_record_of_its_own)
    inspector.click('#tab-retrieval')
    # Adding is the one thing a pinned record does allow, since the addition is
    # run under the config that record itself keeps.
    expect(inspector.locator('#add-question')).to_be_enabled()

    inspector.click('#add-question')
    inspector.click('#question-picker-list .q-option[data-id="5"]')

    expect(inspector.locator('#retrieval-status')).to_contain_text(
        'saved with this recorded experiment', timeout=STEP_TIMEOUT)
    added = inspector.locator('#retrieval-added details.retrieval-question')
    expect(added).to_have_count(1)
    expect(added.first.locator('summary .q-id')).to_have_text('5')
    expect(inspector.locator('#retrieval-added .added-label')).to_contain_text(
        "not part of the run's own sample")
    # It is in both tabs, and in neither run's own list.
    expect(inspector.locator('#generation-added details.gen-question')).to_have_count(1)
    expect(inspector.locator(
        '#retrieval-questions details.retrieval-question')).to_have_count(2)

    inspector.reload()

    expect(inspector.locator('#archive-state')).to_contain_text('read-only')
    inspector.click('#tab-retrieval')
    kept = inspector.locator('#retrieval-added details.retrieval-question')
    expect(kept).to_have_count(1)
    expect(kept.first.locator('summary .q-id')).to_have_text('5')
    expect(inspector.locator('#generation-added details.gen-question')).to_have_count(1)
