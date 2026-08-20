"""Tests for the served panel's own markup and script."""
import json
import shutil
import subprocess
import re

import pytest

from raglab.configuration import lab_config as config
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import deterministic_metrics as metrics

from raglab.conftest import RAGLAB_DIR, _font_size_literals, _radius_literals

PANEL_JS = RAGLAB_DIR / 'dashboard' / 'frontend' / 'panel.js'


def _scope(text: str, anchor: str) -> str:
    """Slice `text` from `anchor` onward, or hand back `''` if the anchor
    itself is gone. A plain `text[text.index(anchor):]` raises `ValueError`
    out of fixture setup the moment the anchor disappears — which would
    error every row in CONVENTIONS at once, table-wide, with a message that
    names neither the widget nor which row cares. Returning `''` instead
    lets each row that reads this scope fail on its own, by its own
    must_contain/must_not_contain and its own reason string — a removed
    widget fails the widget rows by name, not the whole table opaquely."""
    i = text.find(anchor)
    return text[i:] if i >= 0 else ''


# --- the served panel's conventions, as one table ---------------------------

@pytest.fixture(scope='module')
def panel_texts(client):
    """Every named text the convention table below checks, fetched the one
    way a browser actually reaches it (`client.get`) — a second disk read of
    the same file would be a claim about a copy nobody is served. Several
    entries are carved out of the full page, css and script, because their
    claim is *where* the text sits rather than merely that it exists
    somewhere on the page — the same regions the retired pin tests scoped
    their own reads to. `panel_server.py` is the one entry read from disk: the
    lab's Python source is never served, so there is no route to prefer over
    it."""
    html = client.get('/').text
    css = client.get('/panel.css').text
    js = client.get('/panel.js').text
    tokens = client.get('/tokens.css').text

    embed_label = re.search(r'<label>Embedding model.*?</label>', html, re.S)
    model_card = re.search(r'<section[^>]*id="modelCard".*?</section>', html, re.S)
    assert embed_label and model_card, 'the panel dropped a section this table reads'

    return {
        'index.html': html,
        # The shared scale, fetched over its own route because both pages link
        # it before their own sheet — a disk read would be a claim about a copy
        # nobody is served.
        'tokens.css': tokens,
        # The shared chrome sheet, over its own route for the same reason as
        # tokens.css. It now holds the table component every table on either
        # surface is built against, so its contract is checked here beside the
        # markup that depends on it.
        'chrome.css': client.get('/chrome.css').text,
        # The leaderboard surface, served by this same lab: the ranking moved
        # off the lab page, so the rows that guard what a ranking must say
        # follow it here rather than being deleted with the old board.
        'leaderboard.html': client.get('/leaderboard').text,
        'leaderboard.js': client.get('/leaderboard.js').text,
        # The script all three pages load before their own, over its own route
        # for the same reason as tokens.css. What both surfaces turned out to
        # need identically lives in it, so a claim about "one implementation"
        # is a claim about this file.
        'lab.js': client.get('/lab.js').text,
        'panel.css': css,
        'panel.js': js,
        'index.html (embedding-model label)': embed_label.group(0),
        'index.html (modelCard section)': model_card.group(0),
        # The widget's own CSS rules and script, sliced from a real selector
        # / function name rather than the bare word "widget" — both files
        # carry a header *comment* naming the widget first, and a check that
        # started scanning there would still pass with the feature gutted.
        'panel.css (widget block)': _scope(css, '.widget-launch'),
        'panel.js (widget block)': _scope(js, 'widgetSay'),
        'panel_server.py': (RAGLAB_DIR / 'dashboard' / 'panel_server.py').read_text(encoding='utf-8'),
    }


# (file, must_contain, must_not_contain, reason) — one row per retired
# single-substring pin test, each carrying the one line that used to be its
# docstring so a failure names the rule rather than printing a bare
# "assert 'x' in text".
CONVENTIONS = [
    ('panel.css', 'var(--step-index)', None,
     'the index-step ink token must ship in the served stylesheet, or the '
     'colour convention has no token to draw from — checked with the closing '
     'paren so the `-lit` variant (`var(--step-index-lit)`) cannot satisfy it '
     'by prefix collision'),
    ('panel.css', 'var(--step-retrieval)', None,
     'the retrieval-step ink token must ship in the served stylesheet — '
     'checked with the closing paren so the `-lit` variant and the doc '
     'comment a few lines above (which names the token in prose) cannot '
     'satisfy it in place of the real declaration'),
    ('panel.css', 'var(--step-generation)', None,
     'the generation-step ink token must ship in the served stylesheet — '
     'checked with the closing paren so the `-lit` variant '
     '(`var(--step-generation-lit)`) cannot satisfy it by prefix collision'),
    ('index.html', 'data-step="index"', None,
     'the index card must be tagged with its step, so the ink and the stage '
     'cannot disagree'),
    ('index.html', 'data-step="retrieval"', None,
     'the retrieval card must be tagged with its step'),
    ('index.html', 'data-step="generation"', None,
     'the generation card must be tagged with its step'),
    ('index.html', '/panel.css', None,
     'the split-out stylesheet must actually be linked from the page'),
    ('index.html', '/panel.js', None,
     'the split-out script must actually be linked from the page'),
    ('panel.js', 'model_roles', None,
     'the standalone panel must read model roles from the served list rather '
     'than hard-code a picker'),
    ('panel.js', 'rag-model', None,
     'every model role must render through the shared .rag-model markup'),
    ('panel.js', 'embed_models', None,
     'the embedder is a language model too, and must offer the served model '
     'list the same way the other roles do'),
    ('panel.js', 'embedder_hints', None,
     'the embedder picker must carry the served language hints'),
    ('panel.js', 'OPTIONS.metrics', None,
     'the score cards must be read from the service, not a local list'),
    ('panel.js', 'metric.${key}', None,
     'each score must join the one help registry under metric.<key>'),
    ('panel.js', None, 'SCORE_CARDS',
     'the hard-coded score list must not come back'),
    ('index.html (embedding-model label)', 'sentence-transformers', None,
     'the embedding-model label must name the sentence-transformers backend'),
    ('index.html (embedding-model label)', 'fastembed', None,
     'the embedding-model label must name the fastembed backend'),
    ('index.html (embedding-model label)', None, 'openai',
     'the openai backend has no catalogue left and must not be named here'),
    ('index.html (modelCard section)', 'id="embedder"', None,
     'the embedder control must live in the one model column'),
    ('index.html (modelCard section)', 'id="embed_model"', None,
     'the embed-model control must live in the one model column'),
    ('leaderboard.html', 'decision score', None,
     'the leaderboard must say which column chose the architecture'),
    ('leaderboard.js', 'decision_stderr', None,
     'the leaderboard must show the deciding score with its error, never the '
     'mean alone — a board row calls it `decision_stderr`, which is the name '
     'the route serves'),
    ('panel.js', 'job.detail', None,
     'a judged local run spends hours in one stage, and the detail is the '
     'one thing that still moves'),
    ('index.html', 'Stop experiment', None,
     'a run that cannot be stopped is one you kill the process to escape'),
    ('panel.js', "'/api/jobs/' + jobId + '/cancel'", None,
     'the stop button must call the cooperative-cancel route'),
    ('index.html', None, 'retrieving…',
     'the ask must run through the same job box as a build or a run, not a '
     'static note'),
    ('index.html', 'localhost:9003', None,
     'the panel must link to the Inspector, or :9003 is a port you have to '
     'already know about'),
    ('index.html', 'Inspector (:9003)', None,
     'the panel must name the Inspector in its link text, not just point at '
     'the port — checked against the link text itself rather than the bare '
     'word "Inspector", which also appears in three unrelated HTML comments '
     '(the shared-tokens note by the stylesheet links, the retrieval-window '
     'note above the actions row, and the note beside the link itself) that '
     'a rename of the visible link would not touch'),
    ('index.html', None, 'id="question"',
     'asking one question moved to the Inspector; a control left behind is '
     'how a retired feature quietly comes back'),
    ('index.html', None, 'id="gtPick"',
     'the retired ground-truth picker must not come back either'),
    ('index.html', None, 'id="ask"',
     'the retired ask button must not come back'),
    ('index.html', None, 'id="queryOut"',
     'the retired answer box must not come back'),
    ('panel_server.py', 'api/queries', None,
     "the route itself must stay: the Inspector's followed query view reads "
     'whatever runs through it'),
    ('index.html', 'id="mode"', None,
     'the mode dropdown must read the served modes rather than a local copy'),
    ('index.html', 'id="retrieve-selected"', None,
     'retrieval for the selected questions must stay one click away'),
    ('index.html', 'id="archive-import"', None,
     'the archive import control must keep its script hook'),
    ('index.html', 'id="archive-export"', None,
     'the archive export control must keep its script hook'),
    ('index.html', 'id="archive-file"', None,
     'the hidden archive file input must keep its script hook'),
    ('index.html', 'id="archive-status"', None,
     'the archive status must keep its render hook'),
    ('leaderboard.html', 'id="board"', None,
     'the board must stay on its own surface'),
    ('index.html', 'id="experiments"', None,
     'the ledger of every experiment must stay on the lab page — an index '
     'build has no decision score, so it never belonged in the ranking'),
    ('panel.js', '/api/experiments', None,
     'the experiments list must be read from the ledger route'),
    ('index.html', 'sorttable.js', None,
     'the panel must load the shared column sorter'),
    ('index.html', None, 'ragas_decision ▼',
     'the hard-coded sort arrow must not come back to the markup — it cannot '
     'move once the column is sorted a different way'),
    ('panel.js', None, 'ragas_decision ▼',
     'the hard-coded sort arrow must not come back to the script either'),
    ('leaderboard.js', '/api/leaderboard', None,
     'the leaderboard must read the board route, not re-derive its own rows '
     'from the raw run list — two derivations is how two surfaces come to '
     'describe the same records differently'),
    ('leaderboard.js', 'onApply', None,
     'a numbered row is a claim about the order on screen, so `#` is written '
     'from the displayed order after every reorder rather than served with the '
     'row — a static rank travels with its row and reads 1, 3, 2 the moment '
     'another column is sorted'),
    ('leaderboard.js', 'tabindex="0"', None,
     'the table must sit in a focusable scroll region, or there is no keyboard '
     'way to reach the right-hand side of it at all'),
    ('leaderboard.html', 'chrome.css', None,
     'the leaderboard surface must wear the shared bar, or the switcher cannot '
     'lead back out of it'),
    ('panel.js', 'localStorage', None,
     'the grades card and the settings on screen must be remembered across a '
     'reload'),
    ('panel.js', 'lodestar:raglab-last-run', None,
     'the last experiment must be remembered by id'),
    ('panel.js', 'lodestar:raglab-config', None,
     'the settings on screen must be remembered too'),
    ('panel.js', 'restoreLastRun', None,
     'the remembered run must be re-read by id from the service, or a run '
     'file deleted between two visits would render a stale copy'),
    ('index.html', 'id="widget-launch"', None,
     'the widget launcher button must keep a stable id for its script hook'),
    ('index.html', 'id="widget-window"', None,
     'the widget window must keep a stable id — the launcher toggles it by id'),
    ('index.html', 'id="widget-log"', None,
     'the widget must have somewhere to render the conversation'),
    ('index.html', 'id="widget-input"', None,
     'the widget must have a text field to type a question into'),
    ('index.html', 'id="widget-send"', None,
     'the widget must have a button that submits the question'),
    ('index.html', 'id="widget-settings"', None,
     'the gear that reveals the model row must keep a stable id'),
    ('index.html', 'id="widget-config"', None,
     'the row the gear reveals must keep a stable id'),
    ('index.html', 'id="widget-model"', None,
     'the served model list needs somewhere to render into'),
    ('index.html', 'id="widget-close"', None,
     'the close button must keep a stable id — panel.js binds it directly, '
     'and a missing id throws at script load and takes the whole panel down'),
    ('index.html', 'id="widget-form"', None,
     'the form must keep a stable id — panel.js binds its submit handler '
     'directly, and a missing id throws at script load and takes the whole '
     'panel down'),
    ('panel.css (widget block)', 'position: fixed', None,
     'the launcher and its window must be pinned to the viewport, or a '
     'widget that scrolls with the page is a fourth card, not a widget — '
     'scoped to the widget rules so an unrelated `position: fixed` '
     'elsewhere in the sheet cannot satisfy this'),
    ('panel.css (widget block)',
     'right: var(--gutter); bottom: calc(var(--rail-h) + var(--s-2));', None,
     "the launcher's real anchor values, right down to the unit — not just "
     'the property names, which `.widget-config`\'s `border-bottom: 1px '
     'solid var(--rule)` in the same scoped block would otherwise satisfy '
     'even with both real anchors deleted. Measured off --rail-h because the '
     'status rail is fixed to the viewport floor: a literal offset would put '
     'the launcher underneath it'),
    ('panel.css (widget block)',
     'bottom: calc(var(--rail-h) + var(--s-2) + 2.6rem);', None,
     "the window's real anchor values, distinct from the launcher's own — "
     'same collision this guards against as the launcher row above. The '
     "2.6rem is the launcher's own box, which has no ramp step"),
    ('panel.css (widget block)', None, '--step-',
     'the widget is a helper, not a pipeline stage, and must wear no step ink'),
    ('panel.css (widget block)', 'background: var(--card)', None,
     "a reply's bubble must name a colour. It named var(--slab), which is the "
     'slab-serif font stack — an invalid background, so every reply the '
     'helper had ever given rendered on no bubble at all'),
    ('panel.css (widget block)', None, 'background: var(--slab)',
     'and the font stack must not come back as a colour'),
    ('index.html', 'class="widget-grip widget-grip-top"', None,
     'the window grows from its top and left edges, because it is anchored '
     'bottom-right — the handles have to exist on those two edges'),
    ('index.html', 'role="separator"', None,
     'the two straight handles take focus and answer the arrow keys, so a '
     'reader without a mouse can still size the window'),
    ('index.html', None, 'What you can ask',
     'the empty state is built from the served fixture, not written into the '
     'markup — a starter in two places is a starter that drifts'),
    ('panel.js', None, 'Which ports do the lab',
     'and not written into the script either: the starters are the message '
     'sent to the model, which makes them model-facing text and a fixture'),
    ('panel.js', 'data.starters', None,
     'the starters ride the /api/widget response the model list already '
     'ride, so the widget stays a sealed leaf with no new route'),
    ('panel.js', 'widget-empty', None,
     'the empty log offers four questions and clears them on the first '
     'thing said'),
    ('panel.js', 'lodestar:raglab-widget-size', None,
     'an adjusted window is a preference, not a gesture, and survives a '
     'reload under the same prefix as the settings and the last run'),
    ('panel.js', 'setPointerCapture', None,
     'a drag off a six-pixel handle must keep going — without capture the '
     'resize stops the instant the cursor outruns the edge, which is at once'),
    ('panel_server.py', "'starters': widget.STARTERS", None,
     'the four questions are served from the widget package, whose fixture '
     'they live in — a copy in the page would be text nothing pins'),

    # --- the chrome: one bar, one rail ------------------------------------
    # The header was ~250px of chrome carrying four step strips, six
    # always-green capability chips and a 1200-character findings essay. These
    # rows pin what replaced it, and — just as importantly — pin that the
    # replaced things are gone, since a half-finished revert would leave both.
    ('index.html', '/chrome.css', None,
     'the bar and the surface switcher are shared with the Inspector, so the '
     'page must actually link the shared sheet rather than restyle a copy'),
    ('index.html', 'class="topnav"', None,
     'the three surfaces need a switcher on the page, or the Leaderboard and '
     'the Inspector are only reachable by typing a URL'),
    ('index.html', 'href="/leaderboard"', None,
     'the switcher must point at the leaderboard surface — a nav item that '
     'goes nowhere is worse than no nav item'),
    ('index.html', 'aria-current="page"', None,
     'the current surface must say so in markup, not by colour alone'),
    ('index.html', 'class="statusrail"', None,
     'what is running and what this installation can do report from the rail '
     'at the foot of the page, not from the top chrome'),
    ('index.html', 'id="statusPill"', None,
     'the six capability checks roll up into one worst-state pill; six '
     'always-green badges is how the header got crowded'),
    ('index.html', 'id="caps"', None,
     'the capability chips still render, inside the pill\'s popover — rolling '
     'them up must not mean deleting the detail'),
    ('index.html', None, 'class="spine-seg"',
     'the four step strips are retired: they duplicated the four cards below, '
     'which already carry the same step ink and the same served titles'),
    ('index.html', None, 'class="chips" id="caps"></div>\n  <details',
     'the chip row and the findings essay must not both sit in the header '
     'again — the findings moved to the leaderboard surface, which is where '
     'cross-run reading belongs'),
    ('index.html', 'id="chromeProgress"', None,
     "the running step's ink and progress moved to the bar's bottom edge, "
     'which is the one job of the retired step strips worth keeping'),
    ('panel.js', 'renderStatusPill', None,
     'the pill must be rendered from the same served capabilities the chips '
     'are, so the roll-up cannot disagree with the detail it summarises'),
    ('panel.js', "$('chromeProgress')", None,
     'the job progress must drive the bar\'s progress line now that the '
     'separate spine track is gone'),
    ('panel.js', None, '.spine-seg',
     'nothing may still be reaching for the retired step strips'),
    ('panel.css', '.widget-window[hidden] { display: none; }', None,
     "a rule setting `display` beats the browser's own `[hidden] { display: "
     'none }` — found live: the window stayed visible because nothing said '
     'so explicitly. Checked against the exact rule so a bare `[hidden]` '
     'selector with no `display: none` cannot satisfy it'),
    ('panel.css', '.widget-config[hidden] { display: none; }', None,
     'same fix, for the model row the gear toggles'),
    ('panel.js', "api('/api/widget'", None,
     "the widget's script must actually call its route — checked against "
     'the call site itself, not the bare string `/api/widget`, which also '
     "names the route in this block's own header comment"),
    ('panel.js (widget block)', 'escapeHtml', None,
     'a reply is model output rendered into the page, so it must go through '
     'the shared escaper like every other untrusted string'),
    ('panel.js (widget block)', 'widget-model', None,
     "the script must read the gear's model select"),
    ('panel.js (widget block)', 'model:', None,
     'the chosen model must travel with every message — scoped past the '
     "header comment so `embed_model:`, elsewhere in the file, cannot "
     'satisfy this by suffix collision'),
    ('tokens.css', '--s-1: 0.25rem', None,
     'the spacing ramp must ship in the shared sheet: both pages hand-set '
     'every padding and margin today, which is why neither reads as '
     'uncrowded'),
    ('tokens.css', '--t-base: 0.875rem', None,
     'the type scale must ship in the shared sheet, and its base is 14px — '
     'the old 13px body put dense tables under the readable floor'),
    ('tokens.css', '--radius-sm: 3px', None,
     'the radius scale must ship in the shared sheet; six hand-set radii '
     'across two sheets is drift, not design'),
    ('tokens.css', '--measure: 1560px', None,
     'the page measure must be named once — it was written out five times in '
     'panel.css and disagreed with the Inspector'),
    ('tokens.css', '--gutter:', None,
     'the page gutter must be named once, so the two surfaces stop '
     'disagreeing (1.4rem on the panel, 1.25rem on the Inspector)'),
    ('tokens.css', '--bar-h:', None,
     'the top bar height is a shared token because Phase 2 chrome and the '
     "widget's anchor both measure against it"),
    ('tokens.css', '--rail-h:', None,
     'the footer rail height is a shared token for the same reason as '
     '--bar-h'),

    # --- what changed, said out loud ---------------------------------------
    ('index.html', 'id="resultMeta" aria-live="polite"', None,
     'a finished evaluation is the single most important thing that happens '
     'on this page, and it announced itself to nobody'),
    ('index.html', 'id="indexInfo" aria-live="polite"', None,
     "a build's result is the only place the collection name and the chunk "
     'count appear, so it has to say when it arrives'),
    ('index.html', 'id="retrieveInfo" aria-live="polite"', None,
     'the same for a retrieval, which otherwise finishes in silence'),

    # --- a reason is published on the page, not to a mouse -----------------
    ('panel.js', None, 'holder.title = enabled',
     'the disabled-knob reason is written into a visible note; a tooltip '
     'saying the same thing is a second copy only a mouse can reach'),
    ('panel.js', None, 'title="a stub answered',
     'what `fake` means is in the prose above the ledger, on the page — the '
     'tooltip was a hover-only duplicate of it'),
    ('panel.js', None, 'title="${safe(r.error',
     'why a run failed is the reason the row is degraded, and a title '
     'attribute publishes it to a mouse and to nothing else'),
    ('panel.js', 'class="failed"', None,
     'the failed state and the mark that opens its reason travel together, '
     'wrapped — an explainer inserted directly after a <td> is hoisted out '
     'of the table by the parser'),
    ('panel.css', None, '.rag-field-off { opacity',
     'group opacity is the bug: it composites the whole subtree, so the '
     "`opacity: 1` that used to sit on the field's own explanation did "
     'nothing and the sentence rendered at about 2:1'),

    # --- tables: one component, both surfaces ------------------------------
    ('chrome.css', 'position: sticky; top: 0; z-index: 2;', None,
     'a table header declared `position: sticky` with no inset resolves '
     'against nothing and stays in flow, which reads on screen as a header '
     'that is simply not sticky — the inset is the whole rule'),
    ('chrome.css', None, 'position: sticky; top: 0; z-index: 3;',
     "the caption must not be sticky: it and the header row both sat at "
     '`top: 0` and the caption won on z-index, so the column names vanished '
     'under it the moment a table scrolled'),
    ('chrome.css', 'th.sort-col:focus-visible', None,
     'a focused sortable header must show a ring in the shared sheet — while '
     'each page kept its own copy of these rules only the Inspector had one, '
     'so on the lab a keyboard user could not see which column they were on'),
    ('panel.css', None, '#experiments, #byType, #ragas, #extras, #rows',
     'per-host `overflow-x` is what the shared scroll region replaced: it '
     'makes the host a scroll container on both axes with no bounded height, '
     'so the sticky header had nothing to stick against'),
    ('panel.css', None, 'grid-column: span 2',
     "the readings breakdown must not borrow the control bench's grid — that "
     'one reserves a 300px column for a models card the readings card does '
     'not have, so the column sat empty while the tables were squeezed'),

    ('panel.css', None, '1560px',
     'the page measure is a token, not a number typed in five places — the '
     'five copies are exactly how the panel and the Inspector came to '
     'disagree about how wide a page is'),
]


@pytest.mark.parametrize('file, must_contain, must_not_contain, reason', CONVENTIONS)
def test_the_served_panel_keeps_its_conventions(
        panel_texts, file, must_contain, must_not_contain, reason):
    # this is a convention test
    """Roughly a dozen single-substring pin tests, folded into one table.
    Each row is a claim a served asset makes about itself — a colour token, a
    route it must call, a control it must expose, a feature it must have
    retired — and the reason string is what a failure prints instead of a
    bare `assert 'x' in text`."""
    text = panel_texts[file]
    if must_contain is not None:
        assert must_contain in text, reason
    if must_not_contain is not None:
        assert must_not_contain not in text, reason


def test_every_table_on_the_lab_page_is_built_by_one_component(panel_texts):
    # this is a convention test
    """Five tables on this page — the four readings breakdowns and the
    experiment ledger — and until now three of them were built by hand. Two of
    those three took the sortable styling from the stylesheet and none of the
    listeners, so they looked sortable and were inert. One builder, wearing the
    shared region from chrome.css, is what makes a table added later sortable
    and scrollable by having been rendered rather than by someone remembering."""
    js = panel_texts['panel.js']
    assert js.count('<table') == 1, (
        'there must be exactly one place a table is built; found '
        f"{js.count('<table')}")
    assert 'class="table-scroll" tabindex="0" role="region"' in js, (
        'the region must be focusable and labelled, or there is no keyboard '
        'way to reach the right-hand side of a fifteen-column table')
    assert '<table class="data-table">' in js
    for host in ('byType', 'ragas', 'extras', 'rows', 'experiments'):
        assert f"renderTable('{host}'" in js, (
            f'#{host} must be written through renderTable, which wires the '
            'column sorter after insertion — building it with innerHTML is '
            'how #ragas and #extras came to look sortable and do nothing')


def test_a_table_can_freeze_a_column_at_either_edge(panel_texts):
    # this is a convention test
    """The board's identity is at the left edge and its way into the Inspector
    is at the right, and neither may scroll away — a frozen column that only
    works on one side would put one of the two out of reach at exactly the width
    where reaching it matters. Both need an explicit width for the reason the
    left pair already carries: leave the width to the content and a sticky
    column slides over its neighbour instead of beside it."""
    css = panel_texts['chrome.css']
    assert '.data-table .freeze-last' in css
    assert 'right: 0' in css
    assert 'box-shadow: -1px 0 0 var(--rule)' in css, (
        'the divider goes on the inner edge, which for a right-frozen column '
        'is the left one')


def test_a_disabled_knob_keeps_its_reason_at_full_contrast(panel_texts):
    # this is a convention test
    """A knob this pipeline would ignore is dimmed, and the one sentence saying
    why is the only part of it still worth reading — so the dimming is colour on
    the fields, never `opacity` on the group. Opacity composites the whole
    subtree: a child cannot climb back out of an ancestor's, which is why the
    `opacity: 1` this block used to carry on the note did nothing at all and the
    sentence rendered at half of an already soft ink."""
    css = panel_texts['panel.css']
    # The end anchor is the next rule after the block. It used to be
    # `\nbutton.why {` — the *bare* selector, because `.rag-field-off
    # button.why {` is inside this block and would have cut the slice in half.
    # That rule now lives in chrome.css, shared with the Inspector, so the
    # anchor is the explainer paragraph that follows instead.
    block = css[css.index('/* A knob the current pipeline would ignore'):
                css.index('\np.explain {')]
    # Comments out: this block's own comment explains the bug by naming it, and
    # a guard that cannot tell an explanation from a declaration guards nothing.
    block = re.sub(r'/\*.*?\*/', '', block, flags=re.S)
    assert 'opacity' not in block, (
        'no opacity anywhere in this block — not on the group, and not the '
        'countermand on the note that made it look handled')
    assert 'var(--ink-off)' in block, (
        'the dimming is a named ink, so the one value that means "this knob is '
        'out of play" is decided once')
    assert 'color: var(--ink-soft)' in block, (
        'the reason itself stays at the page\'s ordinary soft ink, which is '
        'what full contrast means here')


def test_the_smallest_controls_clear_the_target_floor(panel_texts):
    # this is a convention test
    """`button.why` was about 16×15 and there is one per knob and per metric;
    the widget's gear and close were about 14×19, side by side in the corner of
    a floating window. Both clear 24×24 now, by different means: the widget's
    head bar has room for the size outright, while the `!` keeps a small mark
    and takes its target from a pseudo-element — a 24px disc at the end of an
    eleven-pixel uppercase label would set the line height of every label on
    the page. The `!` is read from chrome.css because there is one of it for
    both surfaces now; the Inspector's half of that claim lives in
    test_inspector.py."""
    css = panel_texts['chrome.css']
    assert 'button.why::after' in css
    assert 'width: 24px; height: 24px' in css
    assert 'button.why::after' not in panel_texts['panel.css'], (
        'the lab must not carry a second copy of the mark — only what it adds '
        'to it, which is the dimmed variant on a locked knob')
    widget = panel_texts['panel.css (widget block)']
    assert 'min-width: 24px; min-height: 24px' in widget, (
        "the widget's own header controls must clear the floor too — the "
        'scoped read is what stops an unrelated 24px elsewhere in the sheet '
        'from satisfying this')


def test_the_run_chip_names_the_run_on_screen_or_is_nothing(panel_texts):
    # this is a convention test
    """One chip built from the run on the Readings card, and nothing when there
    is no run — never a chip that refers to nothing. It is deliberately the
    *lab's* last run rather than the last conversation: widget memory is an
    in-process checkpointer keyed to a page-scoped session id, so a reload
    genuinely forgets, and a chip implying otherwise would be a panel lying
    about what produced it."""
    js = panel_texts['panel.js']
    ask = js[js.index('function widgetRunAsk'):js.index('function widgetOffer')]
    assert 'result.started_at' in ask, (
        'the chip identifies the run by when it started, the same way the '
        "leaderboard's `when` column does")
    assert '.slice(0, 16)' in ask, (
        'and to the same precision — seconds help nobody identify a run'
    )
    assert 'if (!when) return null' in ask, (
        'a run with no start time gets no chip rather than a chip naming a '
        'blank')
    assert 'DECISION_KEYS' in ask, (
        'the metric it names is one of the four that decide, so the chip and '
        'a ranking are asking about the same number')
    assert 'WIDGET_RUN_ASK = widgetRunAsk(result' in js, (
        'renderResult is the one place that holds the run, so it is where the '
        'chip is set — reading it back off the DOM later would be a second '
        'source for the same fact')


def test_the_panel_centres_every_band_on_the_one_measure(panel_texts):
    # this is a convention test
    """Six page-level bands set their own max-width: the banner, the status
    rail, the capability chips and main in panel.css, the top bar and the
    context scope in chrome.css. They are one band at six widths and must read
    from one token, or the next one added drifts. Counted per sheet, each
    against its own number: a count over both sheets together passes with one
    band gone and another added, which is the failure this pins."""
    bands = {'panel.css': 4, 'chrome.css': 2}
    for sheet, expected in bands.items():
        found = panel_texts[sheet].count('max-width: var(--measure)')
        assert found == expected, (
            f'{sheet} centres {found} bands on the page measure, not '
            f'{expected} — a band either stopped naming the token or a new one '
            'arrived that this test has not been told about')


def test_the_panel_sizes_every_type_from_the_shared_scale(panel_texts):
    # this is a convention test
    """22 hand-set sizes in three units is why the panel read as cramped. Each
    must name a --t-* step instead, so a size is a decision recorded once
    rather than a number typed at the point of use. The Inspector's half of
    this claim lives in test_inspector.py."""
    assert _font_size_literals(panel_texts['panel.css']) == []


def test_font_size_literals_catches_shorthand_with_or_without_a_line_height():
    # this is a unit test
    """The shorthand's line-height is optional in real CSS, so the guard must
    not depend on a trailing `/` to notice a hand-set size — that gap is
    exactly how `.findings code` and `.prose code` slipped past review once
    already. Proven against the seven forms the panel conversion actually
    produced (three token forms the guard must leave alone, and three
    literal shorthand forms plus the no-line-height form it must catch), plus
    four forms the unit alternation used to miss entirely: a percentage, a
    viewport unit, a point size and a `calc(...)` value."""
    not_flagged = [
        'font: var(--t-sm)/1.45 var(--mono)',
        'font: 700 var(--t-sm)/1 var(--mono)',
        'font: 600 var(--t-xl)/1 var(--slab)',
        'font: inherit',
    ]
    flagged = [
        'font: .72rem var(--mono)',
        'font: 12.5px/1.45 var(--mono)',
        'font: 600 1.32rem/1 var(--slab)',
        # The unit alternation used to stop at rem|px|em, which is an open
        # path back to a literal size in every other CSS unit and in any
        # computed value — these four are exactly that gap.
        'font-size: 90%',
        'font-size: 1.2vw',
        'font-size: 11pt',
        'font-size: calc(1rem + 2px)',
    ]
    for css in not_flagged:
        assert _font_size_literals(css) == [], css
    for css in flagged:
        assert _font_size_literals(css) != [], css


def test_the_panel_rounds_every_corner_from_the_shared_scale(panel_texts):
    # this is a convention test
    """2px, 3px, 4px, 5px, 6px, 999px and 50% all appeared as literal radii.
    Each must name a --radius-* token, shorthand corners included, so the two
    pages cannot round the same kind of thing differently."""
    assert _radius_literals(panel_texts['panel.css']) == []


def test_radius_literals_catches_shorthand_and_ignores_the_named_tokens():
    # this is a unit test
    """A regex matching nothing at all would pass the convention test above
    just as well as a correct one, so this proves the guard actually guards:
    fed the three literal shorthand forms the panel conversion had to
    remove, the eight corner-longhand forms (four physical, four logical)
    the shorthand-only pin used to miss entirely, plus the five token forms
    — including the shorthand, a longhand and the two shape tokens — every
    one of these must leave alone."""
    flagged = [
        'a{border-radius: 3px;}',
        'a{border-radius: 999px;}',
        'a{border-radius: 0 3px 3px 0;}',
        # The shorthand-only pin missed every corner longhand — physical and
        # logical alike, which is exactly the gap this proves closed.
        'a{border-top-left-radius: 4px;}',
        'a{border-top-right-radius: 4px;}',
        'a{border-bottom-left-radius: 4px;}',
        'a{border-bottom-right-radius: 4px;}',
        'a{border-start-start-radius: 6px;}',
        'a{border-start-end-radius: 6px;}',
        'a{border-end-start-radius: 6px;}',
        'a{border-end-end-radius: 6px;}',
    ]
    not_flagged = [
        'a{border-radius: var(--radius-sm);}',
        'a{border-radius: 0 var(--radius-sm) var(--radius-sm) 0;}',
        'a{border-radius: var(--radius-pill);}',
        'a{border-radius: var(--radius-circle);}',
        'a{border-top-left-radius: var(--radius-sm);}',
    ]
    for css in flagged:
        assert _radius_literals(css) != [], css
    for css in not_flagged:
        assert _radius_literals(css) == [], css


# --- the routes behind the split files --------------------------------------

def test_the_panels_style_and_script_are_served_as_their_own_files(client):
    # this is a convention test
    """The markup, the style and the script were split into three files —
    `index.html`, `panel.css`, `panel.js` — and a split that is not routed is
    just a dead file next to the one still being served. The content itself
    is asserted by the convention table above; this pins that the two new
    routes actually serve it, with the content type a browser needs."""
    css = client.get('/panel.css')
    assert css.status_code == 200
    assert css.headers['content-type'].startswith('text/css')

    js = client.get('/panel.js')
    assert js.status_code == 200
    assert js.headers['content-type'].startswith('application/javascript')


# --- the standalone panel: relationships a substring cannot hold -----------

def test_the_standalone_panel_reads_only_fields_the_lab_still_produces(client):
    # this is a convention test
    """A field the panel reads but the lab no longer sends prints
    "undefined" or throws — checked against what the lab actually returns
    rather than a list of names someone has to remember to prune."""
    html = client.get('/').text

    served = set(metrics.aggregate([]))
    read = set(re.findall(r'result\.summary\.(\w+)', html))
    assert read <= served, (
        'the panel reads summary fields the lab no longer returns: '
        f'{sorted(read - served)}')

    # This panel renders no retrieved context at all now; that risk moved to
    # the Inspector, covered by test_inspector.py against a real trace.
    assert 'out.contexts' not in html, (
        'a contexts loop is back in the standalone panel — either restore the '
        'field check above with it, or move it to :9003 where the rest went')


def test_the_archive_exchange_uses_the_codec_and_no_run_routes(client):
    # this is a convention test
    source = PANEL_JS.read_text()
    exchange = source[source.index('async function importArchiveFile'):
                      source.index('function renderResult')]
    assert 'ArchiveIO.transact' in exchange
    assert "api('/api/imported-archives'" in exchange
    assert 'raglab-experiment.json' in exchange
    assert '32 * 1024 * 1024' in exchange
    assert all(value not in exchange for value in
               ('use-production', 'productionNote', 'OPTIONS.production',
                'api_key', 'openrouterKey', 'password', 'client_secret',
                'access_token', 'authorization', '/api/indexes',
                '/api/retrievals', '/api/evaluations', '/api/credentials'))
    html = client.get('/').text
    assert html.index('src="/archive_io.js"') < html.index('src="/panel.js"')


def test_archive_exchange_escapes_imported_table_labels_and_never_runs_work():
    # this is a convention test
    source = PANEL_JS.read_text()
    exchange = source[source.index('async function importArchiveFile'):
                      source.index('function renderResult')]
    assert 'ArchiveIO.transact' in exchange
    assert "api('/api/imported-archives'" in exchange
    assert all(path not in exchange for path in
               ('/api/indexes', '/api/retrievals', '/api/evaluations',
                '/api/credentials'))
    render = source[source.index('function renderResult'):
                    source.index('function table')]
    for value in ('row.id', 'row.type', 'row.difficulty', 'name'):
        assert f'safe({value})' in render


def test_rendering_a_different_run_makes_export_settings_only():
    # this is a convention test
    """Clicking an older leaderboard or ledger row re-renders the readings
    card without touching any control, so the settings-change invalidation
    never fires — but exporting then must not ship the previous run's private
    evidence while the screen shows a different result. The design pins it:
    export remains settings-only unless the browser holds the complete
    evidence for the run on display."""
    source = PANEL_JS.read_text()
    render = source[source.index('function renderResult'):
                    source.index('function table')]
    assert '.run_id !== result.run_id' in render, (
        'renderResult must compare the displayed run to the held evidence')
    assert 'CURRENT_ARCHIVE = null' in render, (
        'renderResult must drop held evidence that belongs to another run')


def test_unavailable_completed_dataset_is_view_only_but_settings_only_fails():
    # this is a convention test
    source = PANEL_JS.read_text()
    assert 'ArchiveIO.datasetDisposition' in source
    assert "option.dataset.archiveViewOnly = 'true'" in source
    assert 'ARCHIVE_VIEW_ONLY = true' in source
    assert all(f"$('{control}').disabled = ARCHIVE_VIEW_ONLY" in source
               for control in ('build', 'retrieve-selected', 'run'))


def test_boot_keeps_hidden_defaults_before_any_archive_or_run_action():
    # this is a convention test
    source = PANEL_JS.read_text()
    boot = source[source.index('async function boot()'):
                  source.index('async function refreshOptions()')]
    retained = 'keepUnshown(startingConfig(o.defaults));'
    assert retained in boot
    assert boot.index(retained) > boot.index('applyDefaults(startingConfig(o.defaults));')
    assert source.index(retained) < min(
        source.index("$('run').onclick"), source.index('function archiveSettings'),
        source.index('function snapshotDashboard'), source.index('function exportArchive'))


def test_every_experiment_rows_escape_strings_and_bind_detail_clicks():
    # this is a convention test
    source = PANEL_JS.read_text()
    rows = source[source.index('async function loadExperiments'):
                  source.index('// The whole stored payload for one experiment')]
    assert 'const safe = (value) => escapeHtml(String(value ?? \'\'));' in rows
    assert 'safe(r.started_at)' in rows
    assert 'safe(r.experiment_id)' in rows
    assert 'onclick=' not in rows
    assert "document.createElement('a')" in rows
    assert "addEventListener('click'" in rows


def test_imported_results_render_their_archived_metric_catalogue():
    # this is a convention test
    source = PANEL_JS.read_text()
    exchange = source[source.index('async function importArchiveFile'):
                      source.index('function renderResult')]
    assert 'metric_catalogue: imported.evaluation.metric_catalogue' in exchange
    render = source[source.index('function renderResult'):
                    source.index('function table')]
    assert 'const metricCatalogue = options.metric_catalogue || measures();' in render
    assert 'metricCatalogue.filter(' in render


# --- sortable columns and the shared token sheet ---------------------------

def test_both_lab_pages_share_one_column_sorter(client):
    # this is a convention test
    """One file for both pages rather than a copy each, so "what does
    clicking a header do" has one answer instead of two that drift. The
    order it produces is unit tested in `tests/sorttable.test.js`. Whether
    the panel actually loads it, and whether the hard-coded arrow has come
    back, are rows in the convention table above; the Inspector's half of
    this claim lives in test_inspector.py."""
    from raglab.dashboard.panel_server import STATIC

    assert (STATIC / 'sorttable.js').exists()
    js = client.get('/panel.js').text
    # The two tables worth sorting, both marked at the point they are rendered.
    assert js.count('sortable') >= 2


def test_both_lab_pages_share_one_token_sheet_and_one_script(client):
    # this is a convention test
    """`tokens.css` and `lab.js` follow the same pattern as `sorttable.js`:
    one file for both pages rather than a copy each, so a design token or a
    utility cannot drift apart on either page. This pins that the lab
    actually routes them, the panel actually loads them, and each loads
    before the panel's own stylesheet or script — a later link would lose
    the tokens to the page's own overrides instead of feeding them. The
    Inspector's half of this claim moved to test_inspector.py, since :9003
    is not this test's subject."""
    from raglab.dashboard.panel_server import STATIC

    assert (STATIC / 'tokens.css').exists()
    assert (STATIC / 'lab.js').exists()

    panel_html = client.get('/').text
    tokens = client.get('/tokens.css')
    lab = client.get('/lab.js')
    assert tokens.status_code == 200
    assert tokens.headers['content-type'].startswith('text/css')
    assert lab.status_code == 200
    assert lab.headers['content-type'].startswith('application/javascript')
    assert (panel_html.index('href="/tokens.css"')
            < panel_html.index('href="/panel.css"'))
    assert (panel_html.index('src="/lab.js"')
            < panel_html.index('src="/panel.js"'))


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
    body = client.get('/api/leaderboard?dataset=diary-fa').json()
    assert body['dataset'] == 'diary-fa'
    assert isinstance(body['rows'], list)
    assert all(r['dataset'] == 'diary-fa' for r in body['rows'])


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


def test_the_board_is_one_table_with_both_edges_frozen(panel_texts):
    # this is a convention test
    """One table per dataset, its identity frozen left and its Inspector link
    frozen right. The rank is computed from the displayed order, never served
    with the row, because a static rank travels with its row and reads 1, 3, 2
    the moment another column is sorted."""
    js = panel_texts['leaderboard.js']
    assert "'freeze-1'" in js and "'freeze-last'" in js
    assert 'SortTable.make' in js, (
        'the page loaded sorttable.js and never called it, so click-to-sort was '
        'broken here while the lab page had it — this is the pin against that '
        'coming back')
    assert 'onApply' in js


def test_the_context_popover_says_where_it_opens(panel_texts):
    # this is a convention test
    """A popover opens where its sheet says, on both surfaces: the lab page puts
    the trigger under the top bar and the board puts it mid-card, and the
    browser's own default with `margin: 0` is the viewport's top-left corner for
    both — over the lab's identity and the surface switcher. The other two
    popovers on these pages each state an inset; this one states its trigger."""
    css = panel_texts['chrome.css']
    detail = css.split('.context-detail {', 1)[1].split('}', 1)[0]
    assert 'inset:' in detail, (
        'without a placement the popover lands where nobody put it')
    assert 'position-anchor: --context-scope' in css, (
        'the two surfaces put the trigger in different places, so the box is '
        'placed against the trigger rather than at one literal offset')
    assert 'anchor-name: --context-scope' in css


def test_the_settings_reveal_wraps_what_the_cells_around_it_do_not(panel_texts):
    # this is a convention test
    """The reveal hangs off a table cell, and `white-space: nowrap` on those
    cells inherits into it. Unreset, an embedding model's path or a question-set
    id runs off the side of a box that was opened to read it."""
    css = panel_texts['chrome.css']
    reveal = css.split('.settings-reveal {', 1)[1].split('}', 1)[0]
    assert 'white-space: normal' in reveal
    assert 'overflow-wrap: anywhere' in reveal, (
        'a knob value with no spaces in it wraps nowhere without this')


def test_the_frozen_identity_column_sorts_on_the_sentence_it_shows(panel_texts):
    # this is a convention test
    """The pipeline cell carries the settings reveal, so the cell's own text is
    the sentence plus every knob and value of the recorded config. Sorted on
    that, the column orders by a payload the reader cannot see — and two rows
    whose sentences share a prefix are ordered by the knobs alone. `data-sort`
    carries the sentence, which is what the sorter reads instead."""
    js = panel_texts['leaderboard.js']
    assert 'data-sort="${escapeHtml(sentenceText(row))}"' in js
    assert "const sentenceText" in js, (
        'the sort key is the sentence as text, derived from the same '
        '`row.pipeline` the visible fragments are, so the two cannot disagree')


def test_the_board_names_its_dataset_the_way_the_picker_does(panel_texts):
    # this is a convention test
    """The heading and the button under it name the same corpus, so they say
    the same name — a heading reading `diary-fa` over a button reading `Farsi
    diary` makes the reader work out that they are one thing."""
    js = panel_texts['leaderboard.js']
    assert 'shownOption(CURRENT, CATALOGUE).name' in js
    assert 'const shownOption' in js and 'const optionsFor' in js, (
        'both the heading and the picker read one list of options, or the two '
        'can drift apart again')
    assert 'const corpusName' in js and 'corpusName(dataset)' in js, (
        "the caption and the scroll region's name are read aloud, so an id "
        'there gives the screen reader the internal name while the eye gets '
        'the human one')


def test_the_board_leads_with_what_decides(panel_texts):
    # this is a convention test
    """Exactly four judged metrics decide, and the frozen identity column is
    wide enough to push whatever follows it off the screen. So `decision`, its
    error and those four come before the descriptive columns rather than after
    them."""
    js = panel_texts['leaderboard.js']
    order = [js.index(f"key: '{key}'") for key in
             ('decision', 'spread', 'faithfulness', 'answer_relevancy',
              'llm_context_precision_with_reference', 'context_recall',
              'kind', 'when', 'label')]
    assert order == sorted(order), (
        'the four deciding metrics must sit between the frozen sentence and '
        'the descriptive columns, not behind them')


def test_the_board_colours_a_fragment_by_its_pipeline_step(panel_texts):
    # this is a convention test
    """Colour means pipeline step on both surfaces and is defined once. The
    sentence reads its ink from `data-step`, which is how every other coloured
    thing on these pages does it."""
    assert 'data-step' in panel_texts['leaderboard.js']


def test_the_board_offers_a_dataset_picker_naming_every_experiment(panel_texts):
    # this is a convention test
    """The picker is the surface for choosing which corpus is on screen, and
    'every experiment' is one of its options rather than a second page — it is
    the same population with no filter."""
    js = panel_texts['leaderboard.js']
    assert 'every experiment' in js
    assert 'context-scope' in js or 'context-scope' in panel_texts['leaderboard.html']


def test_the_settings_reveal_has_a_keyboard_way_in_at_all(panel_texts):
    # this is a convention test
    """Hover alone is what this project already removed from these pages twice:
    a reveal that answers only a pointer publishes to a mouse and to nothing
    else. This is the half a served file can be asked: that the cell is a tab
    stop, and that no CSS rule opens the box any more — the box is a popover, so
    a selector cannot open it and the mechanism is script. That focus actually
    opens it, and that tabbing into the panel does not close it, is behaviour
    and is pinned as behaviour, in `board_reveal.test.js`. A grep for the word
    `focusin` would pass with focus ignored entirely."""
    js = panel_texts['leaderboard.js']
    assert 'tabindex="0"' in js, (
        'a cell that cannot be focused has no keyboard way to its settings '
        'however good the script is')
    assert 'popover="manual"' in js
    css = panel_texts['chrome.css']
    assert ':hover .settings-reveal' not in css, (
        'a CSS rule cannot open a popover, so one here would be a second, dead '
        'mechanism for the same box — and it would read as the live one')


def test_the_reveal_escapes_the_scroll_region_that_would_clip_it(panel_texts):
    # this is a convention test
    """The reveal hangs off a sticky cell inside a bounded scroll region, so an
    absolutely-positioned box is clipped at every width — the exact defect the
    Inspector's chunk reveal already had and solved by going `fixed` and being
    placed by script. One implementation, in the file both pages load."""
    assert 'position: fixed' in panel_texts['chrome.css']
    assert 'placeReveal' in panel_texts['lab.js']
    # And out of the cell's paint order as well as its clip: the cell is
    # `position: sticky`, which is a stacking context whatever its z-index, so a
    # reveal that flips up across the sticky header was painted underneath it.
    # The top layer is the only place outside every stacking context on a page.
    assert 'popover="manual"' in panel_texts['leaderboard.js']
    assert 'showPopover' in panel_texts['leaderboard.js']


def test_the_board_names_no_winner(panel_texts):
    # this is a convention test
    """A board mixes judges and question sets, so no winner claim holds across
    one of its tables. The claim itself still exists for the sweep."""
    js = panel_texts['leaderboard.js']
    for banned in ('Winner:', 'No winner', 'Not comparable'):
        assert banned not in js, (
            f'{banned!r} is a comparability claim, and the board is not a '
            'comparability group')


# --- the one dynamic, data-driven guard -------------------------------------

def test_the_panels_no_backend_hint_names_every_backend_that_would_fix_it(client):
    # this is a convention test
    """A hint that lists some of the ways out is worse than one that lists
    none, because a reader takes it for the whole set — so this fails the
    day a backend is added and the sentence is not. Built from the live
    provider list rather than a fixed row, since the row's own content is
    the thing under test."""
    page = client.get('/panel.js').text
    hint = [line for line in page.splitlines() if 'no LLM backend' in line]
    assert hint, 'the panel must say what to do when no backend is reachable'
    for provider in config.LLM_PROVIDERS:
        # 'fake' is not a way out: it answers without failing, which is the
        # problem rather than the fix.
        if provider and provider != 'fake':
            assert provider in hint[0], provider


def test_the_widget_sends_one_session_id_per_page(panel_texts):
    # this is a convention test
    """The widget's memory is a browser page's: the client mints one id when
    the script loads and sends it with every ask, so a follow-up lands in the
    same thread and a reloaded page starts clean — nothing persisted, nothing
    shared between tabs."""
    block = panel_texts['panel.js (widget block)']
    assert re.search(r"api\('/api/widget',\s*\{[^}]*\bsession\b", block), (
        'the widget POST must carry the session id')
    assert 'crypto.randomUUID' in block, (
        'one id, minted client-side, once per page load')


def test_the_widget_shows_the_token_account_under_a_reply(panel_texts):
    # this is a convention test
    """The account travels with the reply and the page shows it — a faint
    meta line, only when the backend reported one: an unreported account
    renders nothing rather than a made-up zero."""
    block = panel_texts['panel.js (widget block)']
    assert 'input_tokens' in block, (
        'the widget must read the served token account')
    assert 'output_tokens' in block, (
        'both directions of the account, not just one')


def test_every_served_script_actually_parses():
    # this is a convention test
    """A syntax error in a served script takes the whole page down at load —
    the panel renders as unstyled markup with no controls wired — and the rest
    of this suite cannot see it: every other check reads the script as *text*,
    and text with an unbalanced brace in it still contains every substring the
    table looks for. That gap is not hypothetical; it shipped a stray `}` that
    only a browser caught.

    Skipped rather than failed where `node` is absent: the suite must stay
    runnable offline on a machine with no JavaScript runtime, and a guard that
    cannot run is worth more as a skip that says so than as a silent pass."""
    node = shutil.which('node')
    if node is None:
        pytest.skip('no node on PATH to parse the served scripts with')
    scripts = sorted((RAGLAB_DIR / 'dashboard' / 'frontend').glob('*.js'))
    assert len(scripts) >= 5, (
        'the frontend should have several scripts; a glob finding almost '
        'nothing would let this pass without checking anything')
    for script in scripts:
        done = subprocess.run([node, '--check', str(script)],
                              capture_output=True, text=True)
        assert done.returncode == 0, (
            f'{script.name} does not parse, so the page it belongs to is dead '
            f'on arrival:\n{done.stderr.strip()}')


def test_the_column_sorter_keeps_header_semantics_and_survives_a_restore():
    # this is a convention test
    """Two defects this file guards against, both invisible to a text check
    that only asks whether sorting exists at all.

    `role="button"` on a `<th>` overrides the implicit `columnheader` role, so
    a screen reader stops announcing which column a cell is in — worst exactly
    where the tables are widest. Sortability belongs to `aria-sort`.

    And the "already wired" flag must not live in the DOM: several places here
    save a card's markup and put it back, and a `data-` flag survives that
    round-trip. A restored table came back with the flag, the `.sortable`
    class, the focus rings and the arrows — and no listeners, because `make`
    saw the flag and returned early. It looked sortable and did nothing."""
    source = (RAGLAB_DIR / 'dashboard' / 'frontend' / 'sorttable.js').read_text(
        encoding='utf-8')
    assert "setAttribute('role', 'button')" not in source, (
        'a sortable th must keep its columnheader role')
    assert "aria-sort" in source, 'sortability is announced with aria-sort'
    assert 'WeakSet' in source, (
        'the wired-already flag must not be a DOM attribute, or an innerHTML '
        'restore produces a table that looks sortable and is inert')
    assert 'dataset.sortWired' not in source, (
        'the DOM flag that survived innerHTML must not come back')
