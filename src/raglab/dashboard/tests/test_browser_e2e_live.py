# this is an end-to-end test
"""The whole lab on a real model, in a real browser — named on the command
line or skipped.

Every other journey in this folder runs on the `fake` backend, which is what
makes them fast, free and safe to run on every merge. That is also their
limit: `fake` returns invention no field contradicts, so a stage that would
have refused, timed out or answered in the wrong language passes there
exactly as a working one does. This file is the twin that closes that gap. It
runs one question through every LLM stage the lab has, on the codex CLI, and
asserts what a reader would actually see at each step.

Two opt-ins guard it, because it spends real calls (a few dozen at several
seconds each, so several minutes) and no sweep may trigger that by accident:

* the `live` marker, which `pyproject.toml` deselects the way it deselects
  `browser`; and
* the file's own name on the command line, which is the same
  invocation-shaped guard every `*_live.py` probe in this repo already
  applies to itself, and which is what stands the suite's secrets pin down.

    uv run pytest src/raglab/dashboard/tests/test_browser_e2e_live.py -m live -s

The corpus is `meetings-de`: fifteen German meeting notes, two parts each,
twelve questions. German rather than English on purpose — an answer in the
corpus's own language is one of the claims here, and an English corpus cannot
carry it. It declares a part-level `role` label (so a label boundary and a
part prefix have something real to name) and a document-level `meeting_date`
(so the time filter is live rather than inert), which is what lets one run put
every knob in the panel to work.

Nothing the developer owns is reachable: the lab runs in a child process with
the same five durable redirections the offline browser lab uses, and the same
session guard fails the run if `databases/`, `.runs/` or `.datasets/` changed
while it worked.
"""
import re

import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = [pytest.mark.browser, pytest.mark.live]

from playwright.sync_api import expect                              # noqa: E402

from raglab.llm_backends.cli_subprocess_chat import cli_available   # noqa: E402
from raglab.dashboard.tests.conftest import serve_lab, set_plan     # noqa: E402

#: The corpus, and how many questions its ground truth declares.
DATASET = 'meetings-de'
QUESTIONS = 12

#: A model call is seconds, and a judged run makes many of them behind one
#: job — so every wait here is minutes where the offline journeys wait
#: seconds. Still Playwright's own waits, never a sleep.
CALL = 300_000
RUN = 1_500_000

#: The backend this journey pays for. `gpt-5.6-luna` at effort low is the
#: lightest alias the codex mode presets, which is what the repo's other live
#: probe uses for the same reason. OpenRouter is blanked and pointed at a
#: closed port exactly as the offline lab blanks it: the codex CLI is the one
#: thing here allowed to reach the network, and a second path to a model would
#: make "which backend answered" a question rather than a fact.
LIVE_BACKEND = {
    'RAGLAB_LLM': 'codex',
    'RAGLAB_MODEL': 'gpt-5.6-luna',
    'RAGLAB_CLI_EFFORT': 'low',
    'BRAIN_LLM': 'codex',
    'OPENROUTER_API_KEY': '',
    'OPENAI_API_KEY': '',
    'OPENROUTER_BASE_URL': 'http://127.0.0.1:1',
    # The lab is outside the measured seam and may trace, but a test run is
    # not a session anybody wants in the project's LangSmith history.
    'LANGSMITH_API_KEY': '',
    'LANGSMITH_TRACING': 'false',
}

#: Words only a German answer would carry. The claim is "the answer came back
#: in the corpus's language", and a model asked a German question can still
#: reply in English — that is exactly the failure worth catching. A handful of
#: function words rather than one, because which content words appear is the
#: model's business and not this test's.
GERMAN = re.compile(r'\b(der|die|das|und|ist|wurde|für|nicht|sind|wird|auf)\b',
                    re.IGNORECASE)


def _named_on_the_command_line(request) -> bool:
    """Whether this run asked for this file by name, as the play button and an
    explicit terminal run do — a directory or bare sweep did not."""
    return any('test_browser_e2e_live' in str(arg)
               for arg in request.config.invocation_params.args)


@pytest.fixture(scope='session', autouse=True)
def _only_when_asked_for_by_name(request):
    """The second opt-in. The marker keeps this out of every sweep; this keeps
    it out of a `-m live` sweep too, so the only way to spend the calls is to
    name the file."""
    if not _named_on_the_command_line(request):
        pytest.skip('a real codex run: name this file to opt in')
    if not cli_available('codex'):
        pytest.skip('the codex CLI is not on PATH')
    # `page.set_default_timeout` governs actions, not assertions: an `expect`
    # keeps its own 5s default, which is under a single model call. Set here
    # rather than at import, so naming some other file never widens it.
    expect.set_options(timeout=CALL)


@pytest.fixture(scope='session')
def live_lab_server(_the_developers_lab_stays_untouched, tmp_path_factory):
    """A lab on the codex CLI, in a child process, writing only to a temporary
    home — the offline lab's plumbing with one backend swapped."""
    home = tmp_path_factory.mktemp('live-lab')
    with serve_lab(home, LIVE_BACKEND) as base_url:
        yield base_url


def _written_answer(host):
    """The answer text out of a generation block, which starts collapsed.

    One renderer draws both the run's own questions and the ones a reader adds
    later, so this reads either. `.gen-answer--actual` is what the run wrote;
    `--ideal` beside it is the ground truth, and asserting on that would prove
    only that the corpus is German.
    """
    block = host.locator('details.gen-question').first
    expect(block).to_have_count(1)
    block.locator('summary').first.click()
    written = block.locator('.gen-answer--actual')
    expect(written).to_be_visible()
    return written.inner_text()


def _page(browser, base_url: str, path: str):
    page = browser.new_page()
    page.set_default_timeout(CALL)
    page.set_viewport_size({'width': 1440, 'height': 1800})
    page.goto(f'{base_url}{path}')
    return page


@pytest.fixture(scope='session')
def live_journey(live_lab_server, browser):
    """One page, one run, walked once and read by every test below.

    Session-scoped and shared rather than one run per test: the run is the
    expensive thing, and re-pressing it per assertion would multiply several
    minutes by the number of claims for no extra confidence. The fixture only
    *performs* the journey; every assertion lives in a test.
    """
    page = _page(browser, live_lab_server, '/')
    expect(page.locator('#plan-add option').first).to_be_attached()

    # --- the index step: every chunking knob the plan control offers --------
    # The codex mode's preset is the full LLM pipeline — HyDE, an LLM
    # reranker, an LLM gate, an LLM answerer and both judges — so picking it
    # is most of what "all knobs activated" means for the two steps below.
    page.select_option('#mode', 'codex')
    page.select_option('#dataset', DATASET)
    # Every stage kind a plan can hold, in the only order that validates: a
    # drift or label boundary needs parts, which a separator has already cut
    # away, so the separator closes the plan.
    set_plan(page, 'part', 'label', 'drift', 'separator')
    # Stage 2, the label boundary: the corpus's own part-level label.
    page.select_option('#split_plan [data-plan="atom-label"][data-at="2"]', 'role')
    page.select_option('#split_plan [data-plan="atom-value"][data-at="2"]', 'user')
    # Stage 3, drift: a marker of its own, so the stage is not just the
    # embedding distribution. A new drift stage carries none, hence the add.
    page.click('#split_plan [data-plan="atom-add"][data-at="3"]')
    page.fill('#split_plan [data-plan="atom-text"][data-at="3"][data-atom="0"]',
              'Ausserdem')
    # Stage 4, the separator that closes the plan, only where a piece is still
    # over the budget.
    page.fill('#split_plan [data-plan="atom-text"][data-at="4"][data-atom="0"]',
              '\\n\\n')
    page.select_option('#split_plan [data-plan="when"][data-at="4"]', 'over-budget')
    page.fill('#chunk_chars', '400')
    page.select_option('#chunk_unit', 'characters')
    page.fill('#overlap', '40')
    page.fill('#part_join', '\\n\\n')
    page.select_option('#part_prefix', 'role')
    page.select_option('#normalizer', 'neutral')
    page.check('#contextual')
    # A grouping, so the build produces summaries as well as leaves — the one
    # index knob whose output the Inspector shows on a mode of its own.
    page.select_option('#hierarchy', 'louvain')
    page.select_option('#graph_source', 'hybrid')
    page.fill('#graph_knn', '6')
    page.fill('#min_group', '2')
    page.select_option('#summarizer', 'centroid')
    # A hashing embedder: this journey is about the model-calling stages, and
    # a downloaded encoder would add minutes that prove nothing here.
    page.select_option('#embedder', 'token-hash')

    page.click('#build')
    expect(page.locator('#indexInfo')).to_contain_text('chunks')
    index_summary = page.locator('#indexInfo').inner_text()

    # --- the retrieval step -------------------------------------------------
    # One question, set before anything retrieves: the retrieve button works
    # on whatever step 3 has selected, so this is what keeps a stage that
    # calls a model per question from doing it twelve times.
    page.fill('#limit', '1')
    page.check('#time_filter')
    page.check('#multi_query')
    page.check('#hyde')
    page.select_option('#retriever', 'hybrid-rrf')
    page.fill('#candidates', '20')
    page.select_option('#summary_scope', 'mixed')
    page.fill('#summary_boost', '1.2')
    page.select_option('#reranker', 'llm')
    page.fill('#rerank_depth', '10')
    page.select_option('#grader', 'llm')
    page.fill('#grade_threshold', '0.2')
    page.fill('#k', '4')

    page.click('#retrieve-selected')
    expect(page.locator('#retrieveInfo')).not_to_be_empty()
    retrieval_line = page.locator('#retrieveInfo').inner_text()

    # --- the generation step, on that one question --------------------------
    page.select_option('#answerer', 'llm')
    page.check('#fact_judge')
    page.select_option('#ragas_mode', 'llm')
    page.fill('#ragas_limit', '1')
    page.fill('#label', 'live codex all knobs')

    page.click('#run')
    expect(page.locator('#resultBody')).to_be_visible(timeout=RUN)

    yield {'page': page, 'base_url': live_lab_server,
           'index_summary': index_summary, 'retrieval_line': retrieval_line}
    page.close()


def test_the_index_step_reports_the_chunks_the_five_stage_plan_produced(
        live_journey):
    """A five-stage plan over fifteen German meetings, with a grouping on top."""
    summary = live_journey['index_summary']
    assert re.search(r'\d+ chunks', summary), summary
    # The plan is the whole point of the index step, so the panel's own
    # one-line rendering of it is read back rather than assumed.
    expect(live_journey['page'].locator('#plan-text')).to_have_text(
        'document / part / role=user / drift or "Ausserdem" / "\\n\\n"')


def test_the_retrieval_step_returns_candidates_through_every_llm_stage(
        live_journey):
    """HyDE, a multi-query expansion, an LLM reranker and an LLM gate all ran.

    What is asserted is the line the panel writes, not an internal: a stage
    that could not reach its model refuses rather than passing something
    through, so a line with a count on it is a line four model calls produced.
    """
    line = live_journey['retrieval_line']
    assert re.search(r'\d', line), line
    assert 'error' not in line.lower(), line


def test_the_run_scores_one_question_with_a_real_judge(live_journey):
    """The four deciding metrics come from a real judge, so the table is real."""
    page = live_journey['page']
    expect(page.locator('#resultMeta')).to_contain_text('live codex all knobs')

    tiles = page.locator('#scores .score')
    assert tiles.count() > 5, tiles.count()
    expect(tiles.first.locator('b')).not_to_be_empty()

    # Judged, not offline: the mode line says which of the two ran, and a
    # judged run is the only one whose four deciding metrics exist at all.
    expect(page.locator('#ragas')).to_contain_text('mode llm')
    expect(page.locator('#ragas table tbody tr')).not_to_have_count(0)

    # One question asked, one row written.
    rows = page.locator('details.rowdump')
    rows.locator('summary').click()
    expect(page.locator('#rows table tbody tr')).to_have_count(1)


def test_the_inspector_shows_the_chunks_summaries_ranking_and_answer(
        live_journey, browser):
    """Every step of the run, opened afterwards on its own tab.

    This is also where the language claim is settled: the panel's per-question
    table holds scores, and the answer text itself is on this page.
    """
    page = _page(browser, live_journey['base_url'], '/inspector')
    expect(page.locator('#follow-state')).to_have_attribute('data-lab', 'up')

    # The chunks the run actually used, and the summaries the grouping built.
    # The live view states both counts on the config line — `#chunks-status`
    # belongs to the Build button beside the tab, and stays empty for a build
    # this page only followed.
    page.click('#tab-chunks')
    told = page.locator('#chunks-active-config')
    expect(told).to_contain_text(re.compile(r'\d+ chunks in \d+ sessions'))
    expect(told).to_contain_text('from the last evaluation')
    counted = told.inner_text()
    assert re.search(r'· [1-9]\d* summar', counted), counted
    page.click('#chunks-mode-summaries')
    expect(page.locator('#summaries-body')).to_be_visible()
    expect(page.locator('#summaries-body')).not_to_contain_text('flat')

    # The ranking, question by question.
    page.click('#tab-retrieval')
    expect(page.locator('#retrieval-questions')).not_to_be_empty()

    # And the answer the run actually wrote, in the corpus's own language. The
    # answerer is the stage that can silently get this wrong: `fake` returns
    # invention in no particular language, so no offline journey can make this
    # claim, and a real model asked in German still sometimes replies in
    # English.
    page.click('#tab-generation')
    expect(page.locator('#generation-active-config')).not_to_be_empty()
    written = _written_answer(page.locator('#generation-questions'))
    assert GERMAN.search(written), f'not German: {written[:500]}'
    page.close()


def test_a_question_added_in_the_inspector_is_retrieved_and_answered_live(
        live_journey, browser):
    """The reader's own follow-up question, run through the same config.

    A second real question, deliberately: the run sampled one, and "can I ask
    it something it did not sample" is the Inspector's own claim. It goes
    through every LLM stage again, so this is the second place a broken
    backend would show.
    """
    page = _page(browser, live_journey['base_url'], '/inspector')
    page.click('#tab-retrieval')
    page.click('#add-question')
    options = page.locator('#question-picker-list .q-option')
    expect(options).to_have_count(QUESTIONS)

    asked = options.nth(1).get_attribute('data-id')
    options.nth(1).click()

    expect(page.locator('#retrieval-status')).to_contain_text(
        f'{asked} added', timeout=RUN)
    expect(page.locator('#retrieval-added')).not_to_be_empty()
    # It lands in both tabs, and the generation half is a real answer in the
    # corpus's language — the same claim as above, on a question nobody sampled.
    page.click('#tab-generation')
    written = _written_answer(page.locator('#generation-added'))
    assert GERMAN.search(written), f'not German: {written[:500]}'
    page.close()


def test_the_run_reaches_the_board_under_its_own_corpus(live_journey, browser):
    """The last step of the journey: a finished experiment is on the board.

    One table per dataset and no winner named — this asserts only that the row
    arrived under the corpus it ran on, which is what "the leaderboard picked
    it up" means.
    """
    page = _page(browser, live_journey['base_url'], '/leaderboard')
    page.wait_for_selector('#board table')
    expect(page.locator('#board')).to_contain_text('live codex all knobs')
    expect(page.locator('#board table')).to_have_count(1)
    page.close()
