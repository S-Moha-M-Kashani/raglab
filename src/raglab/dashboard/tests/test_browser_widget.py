# this is an end-to-end test
"""The helper in the corner, driven in a real browser on all three surfaces.

The widget is the one part of the lab that exists only after a script has run:
`widget.js` builds its launcher and its window into `<body>` on every page, so
nothing here can be asserted on served markup. This journey opens it, types in
it, watches an answer arrive, reloads the page to check the conversation was
the lab's and not the tab's, and ends one thread without touching the other.

**What is real and what is stubbed.** The widget sits outside the measured LLM
seam and builds its own client against OpenRouter, so `RAGLAB_LLM=fake` does
not make it answer: with no key, `POST /api/widget` and `POST
/api/widget/stream` refuse with a 502, and there is no backend in its own model
list that answers offline (the two CLI options shell out to a real model, which
this suite may never do). So the two asking routes — and only those two — are
answered inside the browser by `page.route`, with hand-written server-sent
events shaped exactly as the lab writes them. Each test that does it says so in
its own docstring; every other test in this file calls no stub at all.

Everything else is the running lab: the model list and the starters come from
the real `GET /api/widget`, the log is drawn from the real `GET
/api/widget/history`, New Chat sends the real `DELETE`, and the thread it acts
on is a real thread in the suite's own `widget.db`. A conversation that has to
pre-exist is written straight into that file through langgraph's own
checkpointer — the turn is seeded rather than spoken by a model, but the file,
the route that reads it and the rendering are the lab's.

The refusal itself is worth a journey too, and gets one: with no stub at all,
the 502 has to reach the reader as an error line rather than a silence.
"""
import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser

import json  # noqa: E402  (after the skip guard)
import sqlite3  # noqa: E402

import httpx  # noqa: E402
from playwright.sync_api import expect  # noqa: E402


#: The three keys the widget owns in `localStorage`, and nothing else does.
OPEN_KEY = 'raglab-widget-open'
EXPERIMENT_KEY = 'raglab-active-experiment'
SIZE_KEY = 'raglab:widget-size'


@pytest.fixture(autouse=True)
def _the_general_thread_starts_empty(lab_server):
    """One conversation is shared by every surface, and by every test here.

    `widget.db` outlives a page the way it is meant to, so a journey that seeds
    the general thread would otherwise be the previous state of the next one's
    log. Ended through the lab's own DELETE — the same call New Chat makes —
    rather than by deleting a file behind the lab's back.
    """
    httpx.request('DELETE', f'{lab_server}/api/widget/history',
                  params={'thread': 'general'}, timeout=10.0)


# --- a conversation the lab really holds --------------------------------------

def seed_thread(lab_home, thread: str, exchanges) -> None:
    """Put turns into the suite's own `widget.db`, as the lab would hold them.

    A browser test cannot make the widget speak — that needs a model and a key
    — but every journey about *memory* needs a thread that already has
    something in it. So the conversation is written into the same checkpointer
    the lab reads through, under the same thread name, with the token account
    on the reply exactly where a real turn carries it. The route that serves it
    back, and everything the page does with it, is the lab's own.

    `exchanges` is a list of `(question, answer, tokens_in, tokens_out)`;
    `None` for the two counts means a reply nobody billed, which must render
    without a meta line.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.sqlite import SqliteSaver

    messages = []
    for question, answer, tokens_in, tokens_out in exchanges:
        used = None if tokens_in is None else {
            'input_tokens': tokens_in, 'output_tokens': tokens_out,
            'total_tokens': tokens_in + tokens_out}
        messages.append(HumanMessage(question))
        messages.append(AIMessage(answer, usage_metadata=used))

    checkpoint = empty_checkpoint()
    checkpoint['channel_values'] = {
        'messages': messages,
        'experiment_id': '' if thread == 'general' else thread,
        'started_at': '2026-09-02T00:00:00+00:00'}
    connection = sqlite3.connect(str(lab_home / 'widget.db'),
                                 check_same_thread=False, timeout=30.0)
    saver = SqliteSaver(connection)
    saver.setup()
    # The lab keeps this file in rollback mode so the record is always the whole
    # file; a second opener must not leave it in another one behind the lab's
    # back. Busy is not a failure here — the lab is reading it right now.
    try:
        connection.execute('PRAGMA journal_mode=DELETE')
    except sqlite3.OperationalError:
        pass
    saver.put({'configurable': {'thread_id': thread, 'checkpoint_ns': '',
                                'checkpoint_id': None}},
              checkpoint,
              {'source': 'update', 'step': 1, 'writes': {}, 'parents': {}}, {})
    connection.close()


def held_turns(lab_server: str, thread: str) -> list:
    """What the lab says the thread holds, asked the way the page asks."""
    read = httpx.get(f'{lab_server}/api/widget/history',
                     params={'thread': thread}, timeout=10.0)
    assert read.status_code == 200, read.text
    return read.json()['turns']


# --- the two asking routes, answered in the browser ---------------------------

def sse(*events) -> str:
    """The lab's own wire shape: one JSON object per `data:` line, blank line
    between events. What `widgetStream` parses, written by hand."""
    return ''.join(f'data: {json.dumps(event)}\n\n' for event in events)


def stub_the_answer(page, *events) -> None:
    """Answer `POST /api/widget/stream` in the browser, with these events.

    The refusal this replaces is real — see the module docstring — so the stub
    is deliberately narrow: one route, the shape the lab writes, and no other
    call on the page touched.
    """
    page.route('**/api/widget/stream', lambda route: route.fulfill(
        status=200, content_type='text/event-stream', body=sse(*events)))


def open_the_helper(page):
    """Click the launcher, and wait for the draw that opening starts.

    Opening the helper draws the thread from the lab, and that draw clears the
    log before it paints it. A question typed into the window before it settles
    is a question the draw wipes — which the page then handles honestly by
    redrawing from history, and history is where a stubbed answer never went.
    So the wait is for the paint itself: starters on a thread with nothing in
    it, turns on one that has something.
    """
    page.wait_for_selector('#widget-launch')
    page.click('#widget-launch')
    expect(page.locator('#widget-window')).to_be_visible()
    page.wait_for_selector('#widget-log .widget-empty, #widget-log .widget-msg')
    return page.locator('#widget-window')


def ask(page, question: str) -> None:
    """Type a question and press Send, the way a reader does."""
    page.fill('#widget-input', question)
    page.click('#widget-send')


# --- opening, closing, and being found where you left it ----------------------

def test_the_launcher_opens_the_helper_and_a_reload_finds_it_still_open(panel):
    expect(panel.locator('#widget-launch')).to_be_visible()
    expect(panel.locator('#widget-window')).to_be_hidden()

    open_the_helper(panel)
    assert panel.evaluate(f"() => localStorage.getItem('{OPEN_KEY}')") == '1'
    # The input is focused on open, because opening the helper is asking it
    # something.
    expect(panel.locator('#widget-input')).to_be_focused()

    panel.reload()
    expect(panel.locator('#widget-window')).to_be_visible()


def test_the_close_button_shuts_the_helper_and_a_reload_leaves_it_shut(panel):
    open_the_helper(panel)
    panel.click('#widget-close')
    expect(panel.locator('#widget-window')).to_be_hidden()
    assert panel.evaluate(f"() => localStorage.getItem('{OPEN_KEY}')") == ''

    panel.reload()
    panel.wait_for_selector('#widget-launch')
    expect(panel.locator('#widget-window')).to_be_hidden()


# --- the settings row and the model list --------------------------------------

def test_the_settings_row_offers_the_models_this_installation_serves(panel):
    """No stub: the catalogue is the real `GET /api/widget`."""
    open_the_helper(panel)
    config = panel.locator('#widget-config')
    expect(config).to_be_hidden()

    panel.click('#widget-settings')
    expect(config).to_be_visible()

    model = panel.locator('#widget-model')
    values = model.locator('option').evaluate_all(
        'options => options.map(o => o.value)')
    assert values == ['openai/gpt-5-nano', 'openai/gpt-5-mini',
                      'claude', 'codex'], values
    # An option says what it cannot do, so the label carries more than the id.
    expect(model.locator('option', has_text='claude')).to_contain_text('CLI')
    expect(model).to_have_value('openai/gpt-5-nano')

    model.select_option('openai/gpt-5-mini')
    expect(model).to_have_value('openai/gpt-5-mini')

    panel.click('#widget-settings')
    expect(config).to_be_hidden()


# --- asking, and what comes back ----------------------------------------------

def test_the_input_takes_a_question_and_send_puts_a_reply_in_the_log(panel):
    """The stream route is stubbed; everything else on this page is the lab's."""
    stub_the_answer(panel, {'reply': 'Chunking splits a corpus into pieces.'})
    open_the_helper(panel)

    ask(panel, 'what is chunking')
    expect(panel.locator('.widget-msg.you')).to_have_text('what is chunking')
    expect(panel.locator('.widget-msg.bot')).to_have_text(
        'Chunking splits a corpus into pieces.')
    # The box is emptied by the submit, and the caret goes back to it.
    expect(panel.locator('#widget-input')).to_have_value('')
    expect(panel.locator('#widget-input')).to_be_focused()
    # The examples are gone the moment anything is said.
    expect(panel.locator('.widget-empty')).to_have_count(0)


def test_a_starter_chip_asks_its_own_question(panel):
    """The stream route is stubbed; the four chips are the served fixture.

    A chip is not a shortcut into the input box — clicking one sends that exact
    string, which is why the model-facing text lives in `fixtures/prompts/`.
    """
    stub_the_answer(panel, {'reply': 'Both surfaces are served on :9002.'})
    open_the_helper(panel)

    chips = panel.locator('.widget-starter')
    expect(chips).to_have_count(4)
    first = chips.first.inner_text()
    chips.first.click()

    expect(panel.locator('.widget-msg.you')).to_have_text(first)
    expect(panel.locator('.widget-msg.bot')).to_have_text(
        'Both surfaces are served on :9002.')


def test_a_streamed_reply_types_itself_out_and_settles_on_what_the_lab_holds(
        panel):
    """The stream route is stubbed with real SSE frames; the parsing is the page's.

    Two claims, and the second is the one that matters: the pieces are how an
    answer arrived, and the final event is the turn the lab now holds — so the
    bubble must end up holding the reply and not the concatenation it typed.
    The intermediate states are read off the DOM's own mutation records, since
    every one of them is a text node the page put there and then replaced.
    """
    panel.add_init_script("""
      window.__typed = [];
      new MutationObserver((records) => {
        for (const record of records)
          for (const node of record.addedNodes)
            if (node.nodeType === 3 && record.target.classList
                && record.target.classList.contains('widget-msg'))
              window.__typed.push(node.nodeValue);
      }).observe(document, {childList: true, subtree: true});
    """)
    panel.reload()
    stub_the_answer(panel,
                    {'delta': 'A chunker '},
                    {'delta': 'splits a corpus.'},
                    {'reply': 'A chunker splits a corpus into passages.'})
    open_the_helper(panel)

    ask(panel, 'what does a chunker do')
    reply = panel.locator('.widget-msg.bot')
    expect(reply).to_have_text('A chunker splits a corpus into passages.')
    # The caret is gone: nothing on screen still claims to be arriving.
    expect(panel.locator('.widget-msg.bot.streaming')).to_have_count(0)

    typed = panel.evaluate('() => window.__typed')
    assert 'A chunker ' in typed, typed
    assert 'A chunker splits a corpus.' in typed, typed
    # The typed text and the reply the lab holds are deliberately different, so
    # a page that kept its own transcript would fail here rather than pass by
    # coincidence.
    assert typed[-1] == 'A chunker splits a corpus into passages.', typed


def test_the_token_account_rides_under_the_reply_it_bills(panel):
    """The stream route is stubbed; the account is read off its final event.

    A bill, never a measurement: it renders as a `meta` line under the answer
    and votes in nothing.
    """
    stub_the_answer(panel, {'reply': 'Four metrics decide.',
                            'input_tokens': 118, 'output_tokens': 9})
    open_the_helper(panel)

    ask(panel, 'how many metrics decide')
    expect(panel.locator('.widget-msg.meta')).to_have_text('out 9 in 118 tok.')


def test_a_helper_that_cannot_reach_a_model_says_so_in_the_log(panel):
    """No stub at all: this installation has no key, and the refusal is real.

    A row must never lie about what produced it, and the same rule holds for a
    log line — a widget that cannot reach its model says why, rather than
    falling silent or inventing an answer.
    """
    open_the_helper(panel)
    ask(panel, 'anything at all')

    expect(panel.locator('.widget-msg.err')).to_contain_text(
        'OPENROUTER_API_KEY is not set')
    # The question stays where the reader put it; only the answer is missing.
    expect(panel.locator('.widget-msg.you')).to_have_text('anything at all')
    expect(panel.locator('.widget-msg.thinking')).to_have_count(0)


# --- the conversation belongs to the lab, not to the tab ----------------------

def test_a_reply_survives_a_reload_because_the_lab_is_the_one_holding_it(
        panel, lab_home):
    """No stub: the turns are read back over the real history route.

    The page keeps no transcript — every draw rebuilds the log from
    `GET /api/widget/history` — so a reload is the honest test of who holds it.
    """
    seed_thread(lab_home, 'general',
                [('what is context recall', 'It needs ground truth.', 91, 6)])
    panel.reload()
    open_the_helper(panel)

    expect(panel.locator('.widget-msg.you')).to_have_text(
        'what is context recall')
    expect(panel.locator('.widget-msg.bot')).to_have_text(
        'It needs ground truth.')
    # The account came back on the turn itself, not from anything the page kept.
    expect(panel.locator('.widget-msg.meta')).to_have_text('out 6 in 91 tok.')

    panel.reload()
    expect(panel.locator('#widget-window')).to_be_visible()
    expect(panel.locator('.widget-msg.bot')).to_have_text(
        'It needs ground truth.')


def test_the_header_names_the_thread_and_clicking_it_leaves_the_experiment(
        panel, lab_home, a_recorded_experiment):
    """No stub: two real threads, and the header has to name the right one."""
    experiment = a_recorded_experiment
    seed_thread(lab_home, 'general', [('the shared thread', 'general.', 5, 2)])
    seed_thread(lab_home, experiment,
                [('this experiment', 'about the run.', 7, 3)])
    panel.evaluate(f"id => localStorage.setItem('{EXPERIMENT_KEY}', id)",
                   experiment)
    panel.reload()
    open_the_helper(panel)

    expect(panel.locator('#widget-name')).to_have_text(f'About {experiment}')
    expect(panel.locator('#widget-name')).to_have_attribute(
        'title', "Leave this experiment's conversation")
    expect(panel.locator('.widget-msg.bot')).to_have_text('about the run.')

    panel.click('#widget-name')
    expect(panel.locator('#widget-name')).to_have_text('Lab helper')
    expect(panel.locator('#widget-name')).to_have_attribute(
        'title', 'Ask about this lab')
    expect(panel.locator('.widget-msg.bot')).to_have_text('general.')
    assert panel.evaluate(
        f"() => localStorage.getItem('{EXPERIMENT_KEY}')") is None


def test_new_chat_ends_only_the_conversation_it_was_pressed_in(
        panel, lab_server, lab_home, a_recorded_experiment):
    """No stub: a real DELETE against one real thread, and only that one.

    The spec names this journey outright, because a helper that quietly forgot
    a second conversation would be destroying a record nobody asked it to.
    """
    experiment = a_recorded_experiment
    seed_thread(lab_home, 'general',
                [('kept', 'the general thread stays.', None, None)])
    seed_thread(lab_home, experiment,
                [('ended', 'this one is about to go.', None, None)])
    panel.evaluate(f"id => localStorage.setItem('{EXPERIMENT_KEY}', id)",
                   experiment)
    panel.reload()
    open_the_helper(panel)
    expect(panel.locator('.widget-msg.bot')).to_have_text(
        'this one is about to go.')

    panel.click('#widget-new')
    # An ended conversation is an empty one, and an empty thread is where the
    # examples belong.
    expect(panel.locator('.widget-msg')).to_have_count(0)
    expect(panel.locator('.widget-empty')).to_be_visible()

    # The other thread is untouched — on screen, and in the lab's own record.
    panel.click('#widget-name')
    expect(panel.locator('.widget-msg.bot')).to_have_text(
        'the general thread stays.')
    assert held_turns(lab_server, experiment) == []
    assert [turn['text'] for turn in held_turns(lab_server, 'general')] == [
        'kept', 'the general thread stays.']


# --- one helper, three surfaces -----------------------------------------------

@pytest.mark.parametrize('path', ['/', '/inspector', '/leaderboard'])
def test_the_helper_asks_and_answers_on_every_surface(page, lab_server, path):
    """The stream route is stubbed; the widget on each page is the served one.

    One `widget.js`, three surfaces, one origin — which is the whole reason the
    Inspector is mounted here rather than served on a port of its own.
    """
    page.goto(f'{lab_server}{path}')
    stub_the_answer(page, {'reply': f'answered on {path}'})
    open_the_helper(page)

    ask(page, f'where am i on {path}')
    expect(page.locator('.widget-msg.you')).to_have_text(f'where am i on {path}')
    expect(page.locator('.widget-msg.bot')).to_have_text(f'answered on {path}')


# --- the size is a preference, and preferences are remembered -----------------

def test_the_grips_resize_the_helper_and_the_size_is_remembered(panel):
    """No stub: three grips, two of them reachable from the keyboard."""
    window = open_the_helper(panel)
    before = window.bounding_box()

    # The two edges answer only their own axis, which is what an oriented
    # separator promises.
    panel.focus('[data-grip="top"]')
    panel.keyboard.press('ArrowUp')
    taller = window.bounding_box()
    assert taller['height'] > before['height'], (before, taller)
    assert round(taller['width']) == round(before['width']), (before, taller)

    panel.focus('[data-grip="left"]')
    panel.keyboard.press('ArrowLeft')
    wider = window.bounding_box()
    assert wider['width'] > taller['width'], (taller, wider)
    assert round(wider['height']) == round(taller['height']), (taller, wider)

    # The class is what lets the log take the slack a fixed height creates.
    assert 'widget-sized' in window.get_attribute('class')
    kept = json.loads(panel.evaluate(
        f"() => localStorage.getItem('{SIZE_KEY}')"))
    assert kept['w'] == round(wider['width']), (kept, wider)
    assert kept['h'] == round(wider['height']), (kept, wider)

    # The corner is the two edges at once, and takes no focus: a drag is the
    # only way to reach it.
    grip = panel.locator('[data-grip="corner"]').bounding_box()
    panel.mouse.move(grip['x'] + grip['width'] / 2,
                     grip['y'] + grip['height'] / 2)
    panel.mouse.down()
    panel.mouse.move(grip['x'] - 60, grip['y'] - 60, steps=4)
    panel.mouse.up()
    dragged = window.bounding_box()
    assert dragged['width'] > wider['width'], (wider, dragged)
    assert dragged['height'] > wider['height'], (wider, dragged)

    # A size that fits is a size that comes back.
    panel.reload()
    expect(panel.locator('#widget-window')).to_be_visible()
    restored = panel.locator('#widget-window').bounding_box()
    assert round(restored['width']) == round(dragged['width']), (
        dragged, restored)
