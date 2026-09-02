# this is an end-to-end test
"""The knob surface of the three step cards, driven in a real browser.

Every control in Index, Retrieval and Generation is empty in the markup: the
selects are filled, the numbers are set and the labels become explainer
triggers only after `GET /api/options` comes back. So the first thing this
journey proves is that the surface a reader arrives at is complete — every
knob present, reachable, and offering something to choose.

Then the three things a knob has to do. It has to *hold* a choice: the control
shows the new value and the panel writes it into `raglab:config`, which is the
same memory a reload reads back. It has to go inert when the pipeline stops
reading it — greyed and empty rather than accepting a number nothing will use
— and the rules it goes inert by are the served ones in
`knob_dependencies.py`, which this file imports rather than restates. And it
has to explain itself at two lengths: a brief on hover, the whole note on a
click, with the reason it is inert leading both while it is inert, because a
control a reader cannot use is the one that owes them a sentence.

Nothing here builds an index or runs an evaluation — a knob is a knob before
any of that, and keeping the model out is what keeps this file quick.
"""
import json
import re

import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser

from playwright.sync_api import expect  # noqa: E402  (after the skip guard)

from raglab.configuration.knob_dependencies import DEPENDENCIES  # noqa: E402


EXPLAIN_TIMEOUT = 20_000


#: Every knob in the three step cards, in the order the page lays them out:
#: the card it belongs to, its id, and what kind of control it is. The list is
#: the coverage claim — a knob added to a card and not added here is a knob
#: this file stops asserting anything about.
KNOBS = [
    ('card-index', 'dataset', 'select'),
    ('card-index', 'chunker', 'select'),
    ('card-index', 'chunk_chars', 'number'),
    ('card-index', 'overlap', 'number'),
    ('card-index', 'contextual', 'checkbox'),
    ('card-index', 'hierarchy', 'select'),
    ('card-index', 'graph_source', 'select'),
    ('card-index', 'graph_knn', 'number'),
    ('card-index', 'granularity', 'number'),
    ('card-index', 'hierarchy_levels', 'number'),
    ('card-index', 'min_group', 'number'),
    ('card-index', 'summarizer', 'select'),
    ('card-retrieval', 'time_filter', 'checkbox'),
    ('card-retrieval', 'multi_query', 'checkbox'),
    ('card-retrieval', 'hyde', 'checkbox'),
    ('card-retrieval', 'summary_scope', 'select'),
    ('card-retrieval', 'summary_levels', 'text'),
    ('card-retrieval', 'retriever', 'select'),
    ('card-retrieval', 'candidates', 'number'),
    ('card-retrieval', 'summary_boost', 'number'),
    ('card-retrieval', 'reranker', 'select'),
    ('card-retrieval', 'rerank_depth', 'number'),
    ('card-retrieval', 'recency_half_life_days', 'number'),
    ('card-retrieval', 'mmr_lambda', 'number'),
    ('card-retrieval', 'k', 'number'),
    ('card-retrieval', 'grader', 'select'),
    ('card-retrieval', 'grade_threshold', 'number'),
    ('card-generation', 'limit', 'number'),
    ('card-generation', 'answerer', 'select'),
    ('card-generation', 'fact_judge', 'checkbox'),
    ('card-generation', 'ragas_mode', 'select'),
    ('card-generation', 'ragas_limit', 'number'),
    ('card-generation', 'label', 'text'),
]


def _booted(page):
    """Wait for the one call the panel makes on load, by its visible effect.

    Two separate readiness signals, because the page has two: `boot()` fills
    the selects, and the pass at the end of it turns every knob's label into
    an explainer trigger. Waiting on an option and on a trigger waits on both
    without waiting on a clock.
    """
    expect(page.locator('#chunker')).to_contain_text('semantic-drift')
    expect(page.locator('#card-index .why-term').first).to_be_visible()
    return page


def _stored_config(page) -> dict:
    """What the panel remembers, read out of the browser's own storage."""
    kept = page.evaluate("() => localStorage.getItem('raglab:config')")
    assert kept, 'the panel wrote no raglab:config'
    return json.loads(kept)


def _control(page, path: str):
    """A knob by its dotted config path, found the way `controlFor` finds it.

    Most knobs are an element whose id is the field name; the six model roles
    are dropdowns in the setup panel carrying the whole path in `data-field`.
    """
    name = path.split('.')[1]
    by_id = page.locator(f'#{name}')
    if by_id.count():
        return by_id
    return page.locator(f'.rag-model[data-field="{path}"]')


#: The other half of the grey-out: the class the panel toggles on a dead
#: knob's nearest wrapping div, which is what paints its value out. Matched as
#: a word, because most of those divs carry a layout class as well.
OFF = re.compile(r'\brag-field-off\b')


def _wrapper(page, path: str):
    """The div the grey-out class lands on — the control's nearest ancestor."""
    return _control(page, path).locator('xpath=ancestor::div[1]')


def _governed_by(field: str) -> list[str]:
    """The knobs the served table says this one decides — never a list typed here."""
    return sorted(path for path, rule in DEPENDENCIES.items()
                  if rule['field'] == field)


def _shown_dependents(page, field: str) -> list[str]:
    """The governed knobs the panel actually draws a control for.

    A rule may name a config field with no control on the page —
    `retrieval.agentic_weights` is three numbers the panel carries through
    untouched — and `controlFor` skips those, so this skips them too.
    """
    return [path for path in _governed_by(field) if _control(page, path).count()]


def _said(path: str) -> str:
    """A rule's reason as the explainer prints it: capitalised, full-stopped."""
    reason = DEPENDENCIES[path]['reason']
    return reason[0].upper() + reason[1:] + '.'


def _explainer_trigger(page, knob_id: str):
    """A knob's own name, which is the button that explains it.

    Every knob but a checkbox is preceded by a label whose first words became
    the trigger; a checkbox lives inside its label, so it keeps a `?` beside
    its words instead — its own words already toggle it.
    """
    beside = page.locator(f'#{knob_id}').locator(
        'xpath=preceding-sibling::label//button[contains(@class, "why")]')
    if beside.count():
        return beside
    return page.locator(f'#{knob_id}').locator(
        'xpath=parent::label/button[contains(@class, "why")]')


def _open_explainer(page, knob_id: str):
    """Click a knob's name and hand back the paragraph that opens."""
    _explainer_trigger(page, knob_id).click()
    explain = page.locator('p.explain')
    expect(explain).to_have_count(1)
    return explain


def _set(page, knob_id: str, kind: str, value):
    """Drive one knob the way a reader would, by what kind of control it is."""
    if kind == 'select':
        page.select_option(f'#{knob_id}', value)
    elif kind == 'checkbox':
        page.set_checked(f'#{knob_id}', value)
    else:
        page.fill(f'#{knob_id}', value)


@pytest.mark.parametrize('card, knob, kind',
                         KNOBS, ids=[knob for _, knob, _ in KNOBS])
def test_every_knob_in_the_three_step_cards_is_on_the_page_and_filled(
        panel, card, knob, kind):
    """One walk over the whole surface: present, in its own card, and offering.

    A select is empty until `/api/options` resolves, so asserting it holds at
    least one option is both the coverage claim and the wait for the load.
    """
    control = panel.locator(f'#{card} #{knob}')
    expect(control).to_have_count(1)
    expect(control).to_be_visible()
    if kind == 'select':
        expect(panel.locator(f'#{knob} option')).not_to_have_count(0)


def test_the_index_knobs_take_a_choice_and_the_panel_records_it(panel):
    """Every Index knob moved once, then read back off the control and the memory."""
    _booted(panel)
    # Order matters only in that a knob has to be live to be typed into: the
    # chunker and the grouping are what wake the eleven below them.
    _set(panel, 'dataset', 'select', 'smoke-mini')
    _set(panel, 'chunker', 'select', 'fixed-overlap')
    _set(panel, 'chunk_chars', 'number', '760')
    _set(panel, 'overlap', 'number', '140')
    _set(panel, 'contextual', 'checkbox', False)
    _set(panel, 'hierarchy', 'select', 'louvain')
    _set(panel, 'graph_source', 'select', 'knn')
    _set(panel, 'graph_knn', 'number', '12')
    _set(panel, 'granularity', 'number', '1.4')
    _set(panel, 'hierarchy_levels', 'number', '3')
    _set(panel, 'min_group', 'number', '5')
    _set(panel, 'summarizer', 'select', 'mmr')

    expect(panel.locator('#chunker')).to_have_value('fixed-overlap')
    expect(panel.locator('#chunk_chars')).to_have_value('760')
    expect(panel.locator('#contextual')).not_to_be_checked()
    expect(panel.locator('#hierarchy')).to_have_value('louvain')

    assert _stored_config(panel)['index'] == {
        'dataset': 'smoke-mini', 'chunker': 'fixed-overlap', 'chunk_chars': 760,
        'overlap': 140, 'contextual': False, 'embedder': 'sentence-transformers',
        'embed_model': '', 'hierarchy': 'louvain', 'graph_source': 'knn',
        'graph_knn': 12, 'granularity': 1.4, 'hierarchy_levels': 3,
        'min_group': 5, 'summarizer': 'mmr'}


def test_the_retrieval_knobs_take_a_choice_and_the_panel_records_it(panel):
    _booted(panel)
    # A grouping first: three of these fifteen are inert over a flat index,
    # because a flat index holds no summaries to scope, boost or level.
    _set(panel, 'hierarchy', 'select', 'raptor')
    _set(panel, 'time_filter', 'checkbox', False)
    _set(panel, 'multi_query', 'checkbox', False)
    _set(panel, 'hyde', 'checkbox', True)
    _set(panel, 'summary_scope', 'select', 'drill-down')
    _set(panel, 'summary_levels', 'text', '1 2')
    _set(panel, 'retriever', 'select', 'bm25')
    _set(panel, 'candidates', 'number', '60')
    _set(panel, 'summary_boost', 'number', '2.5')
    _set(panel, 'reranker', 'select', 'recency')
    _set(panel, 'rerank_depth', 'number', '30')
    _set(panel, 'recency_half_life_days', 'number', '91')
    _set(panel, 'mmr_lambda', 'number', '0.7')
    _set(panel, 'k', 'number', '5')
    _set(panel, 'grader', 'select', 'lexical')
    _set(panel, 'grade_threshold', 'number', '0.35')

    expect(panel.locator('#retriever')).to_have_value('bm25')
    expect(panel.locator('#hyde')).to_be_checked()
    expect(panel.locator('#summary_levels')).to_have_value('1 2')

    kept = _stored_config(panel)['retrieval']
    assert {key: kept[key] for key in (
        'time_filter', 'multi_query', 'hyde', 'summary_scope', 'summary_levels',
        'retriever', 'candidates', 'summary_boost', 'reranker', 'rerank_depth',
        'recency_half_life_days', 'mmr_lambda', 'k', 'grader',
        'grade_threshold')} == {
        'time_filter': False, 'multi_query': False, 'hyde': True,
        'summary_scope': 'drill-down', 'summary_levels': '1 2',
        'retriever': 'bm25', 'candidates': 60, 'summary_boost': 2.5,
        'reranker': 'recency', 'rerank_depth': 30,
        'recency_half_life_days': 91, 'mmr_lambda': 0.7, 'k': 5,
        'grader': 'lexical', 'grade_threshold': 0.35}


def test_the_generation_knobs_take_a_choice_and_the_panel_records_it(panel):
    """Six knobs, and only three of them are config.

    `limit`, `ragas_mode` and `ragas_limit` describe *this run* rather than the
    pipeline, so they travel in the run's ui block and never enter the config
    the fingerprint is taken over. The control still has to hold them.
    """
    _booted(panel)
    _set(panel, 'limit', 'number', '4')
    _set(panel, 'answerer', 'select', 'none')
    _set(panel, 'fact_judge', 'checkbox', True)
    _set(panel, 'ragas_mode', 'select', 'off')
    _set(panel, 'ragas_limit', 'number', '3')
    _set(panel, 'label', 'text', 'a browser journey')

    expect(panel.locator('#limit')).to_have_value('4')
    expect(panel.locator('#answerer')).to_have_value('none')
    expect(panel.locator('#fact_judge')).to_be_checked()
    expect(panel.locator('#ragas_mode')).to_have_value('off')
    expect(panel.locator('#ragas_limit')).to_have_value('3')

    kept = _stored_config(panel)
    assert kept['label'] == 'a browser journey'
    assert kept['generation']['answerer'] == 'none'
    assert kept['generation']['fact_judge'] is True


def test_the_chunker_decides_whether_the_two_length_knobs_are_live(panel):
    """`index.chunker` governs exactly two knobs, and the table says which."""
    _booted(panel)
    assert _governed_by('index.chunker') == ['index.chunk_chars', 'index.overlap']

    # The lab boots on semantic-drift: it cuts to a budget, so a chunk length
    # is read and an overlap is not.
    expect(panel.locator('#chunk_chars')).to_be_enabled()
    expect(panel.locator('#overlap')).to_be_disabled()

    _set(panel, 'chunker', 'select', 'fixed-overlap')
    expect(panel.locator('#chunk_chars')).to_be_enabled()
    expect(panel.locator('#overlap')).to_be_enabled()

    _set(panel, 'chunker', 'select', 'session')
    expect(panel.locator('#chunk_chars')).to_be_disabled()
    expect(panel.locator('#overlap')).to_be_disabled()


def test_choosing_a_grouping_wakes_every_knob_the_table_says_it_governs(panel):
    """A flat index makes nine knobs inert; one grouping wakes all nine."""
    _booted(panel)
    governed = _shown_dependents(panel, 'index.hierarchy')
    assert governed == ['index.granularity', 'index.graph_source',
                        'index.hierarchy_levels', 'index.min_group',
                        'index.summarizer', 'retrieval.summary_boost',
                        'retrieval.summary_levels', 'retrieval.summary_scope']

    # The grey-out is two signals: the control itself, and a class on its
    # nearest wrapping div, which is what paints the value out.
    for path in governed:
        expect(_control(panel, path)).to_be_disabled()
        expect(_wrapper(panel, path)).to_have_class(OFF)
    # kNN neighbours is nobody's direct dependent — it hangs off graph_source,
    # which hangs off the grouping — so it goes inert transitively.
    expect(panel.locator('#graph_knn')).to_be_disabled()

    _set(panel, 'hierarchy', 'select', 'louvain')

    for path in governed:
        expect(_control(panel, path)).to_be_enabled()
        expect(_wrapper(panel, path)).not_to_have_class(OFF)
    expect(panel.locator('#graph_knn')).to_be_enabled()
    # And nothing else moved: the chunking knobs answer to the chunker.
    expect(panel.locator('#chunk_chars')).to_be_enabled()
    expect(panel.locator('#overlap')).to_be_disabled()


def test_the_reranker_and_the_gate_each_wake_only_their_own_dependents(panel):
    """Two governing knobs whose dependents overlap in the same card."""
    _booted(panel)
    assert _shown_dependents(panel, 'retrieval.reranker') == [
        'retrieval.recency_half_life_days', 'retrieval.rerank_depth',
        'retrieval.reranker_model']
    assert _shown_dependents(panel, 'retrieval.grader') == [
        'retrieval.grade_threshold', 'retrieval.grader_model']

    # Every state is driven here rather than assumed: a first visit boots on
    # the default backend's preset, which already picks an llm reranker and
    # an llm gate, so "what the lab starts at" is not the dataclass default.
    _set(panel, 'reranker', 'select', 'lexical')
    # The lexical reranker reranks, but weighs no age and calls no model.
    expect(panel.locator('#rerank_depth')).to_be_enabled()
    expect(panel.locator('#recency_half_life_days')).to_be_disabled()
    expect(_control(panel, 'retrieval.reranker_model')).to_be_disabled()

    _set(panel, 'reranker', 'select', 'none')
    for path in _shown_dependents(panel, 'retrieval.reranker'):
        expect(_control(panel, path)).to_be_disabled()

    _set(panel, 'reranker', 'select', 'recency')
    expect(panel.locator('#rerank_depth')).to_be_enabled()
    expect(panel.locator('#recency_half_life_days')).to_be_enabled()
    expect(_control(panel, 'retrieval.reranker_model')).to_be_disabled()

    _set(panel, 'grader', 'select', 'none')
    # The gate is off, so the threshold has nothing to threshold.
    expect(panel.locator('#grade_threshold')).to_be_disabled()
    expect(_control(panel, 'retrieval.grader_model')).to_be_disabled()
    _set(panel, 'grader', 'select', 'lexical')
    expect(panel.locator('#grade_threshold')).to_be_enabled()
    expect(_control(panel, 'retrieval.grader_model')).to_be_disabled()
    _set(panel, 'grader', 'select', 'llm')
    expect(panel.locator('#grade_threshold')).to_be_enabled()
    expect(_control(panel, 'retrieval.grader_model')).to_be_enabled()


def test_a_corpus_with_no_date_label_makes_the_time_filter_inert(panel):
    """The one rule that reads a corpus rather than another knob.

    `smoke-mini` declares no date label, so a filter that resolves time
    language into a date range has nothing to resolve against — whatever else
    is set.
    """
    _booted(panel)
    expect(panel.locator('#time_filter')).to_be_enabled()

    _set(panel, 'dataset', 'select', 'smoke-mini')

    expect(panel.locator('#time_filter')).to_be_disabled()
    # The reason is not on the page; it is the first sentence of this knob's
    # own explainer, and a checkbox keeps a `?` to open it with.
    expect(_open_explainer(panel, 'time_filter')).to_contain_text(
        _said('retrieval.time_filter'))


def test_an_inert_knob_renders_empty_and_says_why_in_its_explainer(panel):
    """Both halves of the rule, on the knob the lab boots with inert.

    Empty is the CSS half: the value is painted in transparent ink rather than
    hidden, so the control keeps its size and the reader is not offered a
    number that nothing will read. The reason is the explainer half.
    """
    _booted(panel)
    expect(panel.locator('#overlap')).to_be_disabled()
    expect(_wrapper(panel, 'index.overlap')).to_have_class(OFF)
    assert panel.evaluate(
        "() => getComputedStyle(document.getElementById('overlap'))"
        '.webkitTextFillColor') == 'rgba(0, 0, 0, 0)'

    explain = _open_explainer(panel, 'overlap')
    expect(explain).to_contain_text(_said('index.overlap'))
    # The reason leads; the definition follows it rather than replacing it.
    expect(explain).to_contain_text('Characters repeated between neighbouring')


@pytest.mark.parametrize('knob', ['chunker', 'reranker', 'answerer'])
def test_a_knob_explains_itself_at_two_lengths_one_per_step(panel, knob):
    """Hover for the brief, click for the whole note — on a live knob of each step."""
    # A window tall enough to hold the whole bench, so reaching a knob in the
    # third card scrolls nothing. Any scroll dismisses the brief (`lab.js`), and
    # the brief waits 140 ms before it opens, so a page still settling under a
    # pointer that has already landed cancels the box before it appears. A
    # reader reads a page that has stopped moving; so does this.
    panel.set_viewport_size({'width': 1440, 'height': 2200})
    _booted(panel)
    trigger = _explainer_trigger(panel, knob)

    # A browser context may preserve the pointer position across navigations.
    # Move away first so this hover always emits the mouseover that starts the
    # brief's deliberately delayed opening.
    panel.mouse.move(0, 0)
    trigger.hover()
    brief = panel.locator('#help-brief')
    expect(brief).to_be_visible(timeout=EXPLAIN_TIMEOUT)
    expect(brief).to_have_attribute(
        'data-more', 'true', timeout=EXPLAIN_TIMEOUT)
    expect(trigger).to_have_attribute(
        'aria-describedby', 'help-brief', timeout=EXPLAIN_TIMEOUT)
    sentence = brief.text_content().strip()
    assert sentence, f'{knob} offered no brief'

    trigger.click()

    expect(brief).to_be_hidden(timeout=EXPLAIN_TIMEOUT)
    explain = panel.locator('p.explain')
    expect(explain).to_have_count(1, timeout=EXPLAIN_TIMEOUT)
    whole = explain.text_content().strip()
    assert whole and whole != sentence, (
        f'{knob}: the click said exactly what the hover already said')


def test_clicking_a_knobs_name_twice_puts_its_explainer_away(panel):
    # This claim is about the two clicks, not Playwright auto-scrolling between
    # locating the trigger and pressing it. Keep the whole bench stationary,
    # as the two-length explainer regression above does.
    panel.set_viewport_size({'width': 1440, 'height': 2200})
    _booted(panel)
    trigger = _explainer_trigger(panel, 'k')

    trigger.click()
    expect(panel.locator('p.explain')).to_have_count(
        1, timeout=EXPLAIN_TIMEOUT)

    trigger.click()
    expect(panel.locator('p.explain')).to_have_count(
        0, timeout=EXPLAIN_TIMEOUT)


def test_a_reload_brings_back_the_knobs_the_reader_set(panel):
    """The panel's memory is the point of writing every keystroke to storage."""
    _booted(panel)
    _set(panel, 'dataset', 'select', 'smoke-mini')
    _set(panel, 'chunker', 'select', 'fixed-overlap')
    _set(panel, 'chunk_chars', 'number', '640')
    _set(panel, 'hierarchy', 'select', 'kmeans')
    _set(panel, 'reranker', 'select', 'none')
    _set(panel, 'k', 'number', '3')
    _set(panel, 'answerer', 'select', 'none')
    _set(panel, 'label', 'text', 'remembered')

    panel.reload()
    _booted(panel)

    expect(panel.locator('#dataset')).to_have_value('smoke-mini')
    expect(panel.locator('#chunker')).to_have_value('fixed-overlap')
    expect(panel.locator('#chunk_chars')).to_have_value('640')
    expect(panel.locator('#hierarchy')).to_have_value('kmeans')
    expect(panel.locator('#reranker')).to_have_value('none')
    expect(panel.locator('#k')).to_have_value('3')
    expect(panel.locator('#answerer')).to_have_value('none')
    expect(panel.locator('#label')).to_have_value('remembered')
    # The dependencies come back with the values, not one paint later.
    expect(panel.locator('#overlap')).to_be_enabled()
    expect(panel.locator('#rerank_depth')).to_be_disabled()
    expect(panel.locator('#min_group')).to_be_enabled()
