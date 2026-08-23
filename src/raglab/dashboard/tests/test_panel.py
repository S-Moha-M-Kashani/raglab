"""Tests for the served panel's own markup and script."""
import json
import shutil
import subprocess
import re

import pytest

from raglab.configuration import lab_config as config
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import deterministic_metrics as metrics
from raglab.llm_backends import model_role_catalogue as model_roles

from raglab.conftest import RAGLAB_DIR, _font_size_literals, _radius_literals

PANEL_JS = RAGLAB_DIR / 'dashboard' / 'frontend' / 'panel.js'


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
        # The row filter, over its own route: it is the leaderboard's, but it
        # reads a cell with the shared sorter's parser, so what it can be asked
        # is a claim about that pair of files rather than about this page.
        'filtertable.js': client.get('/filtertable.js').text,
        # The script all three pages load before their own, over its own route
        # for the same reason as tokens.css. What both surfaces turned out to
        # need identically lives in it, so a claim about "one implementation"
        # is a claim about this file.
        'lab.js': client.get('/lab.js').text,
        'panel.css': css,
        'panel.js': js,
        'index.html (embedding-model label)': embed_label.group(0),
        'index.html (modelCard section)': model_card.group(0),
        # The widget's own stylesheet and script — whole files, served from
        # the root to all three surfaces, so what the rows below claim about
        # the helper is a claim about these two files and nothing else. They
        # were once slices carved out of panel.css/panel.js and kept those
        # names for a while after they stopped being slices, which sent a
        # maintainer chasing a widget failure into the Laboratory's own
        # script; one key per file is what stops that.
        # The codec, over its own route: what the two knob-coverage tests at
        # the foot of this file claim about the export template is a claim
        # about the file a browser is actually handed.
        'archive_io.js': client.get('/archive_io.js').text,
        'widget.css': client.get('/widget.css').text,
        'widget.js': client.get('/widget.js').text,
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
    ('index.html', 'href="/inspector"', None,
     'the panel must link to the Inspector by path — all three surfaces are '
     'one origin now, so a port is not something a reader ever has to know'),
    ('index.html', 'Inspector &rarr;', None,
     'the panel must still name the Inspector in its link text — checked '
     'against the link text itself rather than the bare word "Inspector", '
     'which also appears in unrelated HTML comments (the shared-tokens note '
     'by the stylesheet links, the retrieval-window note above the actions '
     'row, and the note beside the link itself) that a rename of the '
     'visible link would not touch'),
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
     'the filter must re-run after every reorder: the sorter re-appends every '
     'row it holds, which drops the hidden ones back among the visible ones — '
     'a table whose stripes count rows it is hiding, and whose count was '
     'measured against an order that has since moved'),
    ('leaderboard.js', 'FilterTable.apply', None,
     'the board must filter through the shared engine, which reads a cell with '
     'the sorter\'s own parser — a second reading of the same cell is how '
     '`decision>0.7` comes to disagree with the column it sorts'),
    ('leaderboard.js', None, "label: '#'",
     'the rank column must not come back: position in the current sort is what '
     'the sort itself says, and a column that has to be rewritten after every '
     'reorder to stay true is a column saying nothing the order does not'),
    ('leaderboard.html', 'filtertable.js', None,
     'the leaderboard must load the row filter, or its bar is a box that '
     'narrows nothing'),
    ('leaderboard.js', 'id="filter-syntax"', None,
     'what the filter can be asked must be on the page: a query language whose '
     'help is in a comment in the source is a control only its author can use'),
    ('chrome.css', '.scroll-rail', None,
     'a wide table needs the second scrollbar above it — on a long table the '
     'one below the rows is off the bottom of the screen, so moving the columns '
     'means scrolling away from what you were reading'),
    ('lab.js', 'mountScrollRail', None,
     'the rail is built once, in the script both surfaces load, rather than in '
     'the one page that mounts it today'),
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
    ('panel.js', "get('experiment')", None,
     'the panel must take the experiment from its own address, not only from '
     'the one-shot slot: the board now lands here, and a slot written by a '
     'click cannot survive a reload, a bookmark, a copied link, or a new tab '
     'that boots before the writing page has finished — each of which reads '
     'as the open button doing nothing at all'),
    # The launcher and the window are the two elements widget.js creates rather
    # than writes as markup, so their ids are pinned in the form the file
    # actually spells them. The claim is the one the markup rows make: the id is
    # the hook, and it must not drift.
    ('widget.js', "launcher.id = 'widget-launch'", None,
     'the widget launcher button must keep a stable id for its script hook'),
    ('widget.js', "win.id = 'widget-window'", None,
     'the widget window must keep a stable id — the launcher toggles it by id'),
    ('widget.js', 'id="widget-log"', None,
     'the widget must have somewhere to render the conversation'),
    ('widget.js', 'id="widget-input"', None,
     'the widget must have a text field to type a question into'),
    ('widget.js', 'id="widget-send"', None,
     'the widget must have a button that submits the question'),
    ('widget.js', 'id="widget-settings"', None,
     'the gear that reveals the model row must keep a stable id'),
    ('widget.js', 'id="widget-config"', None,
     'the row the gear reveals must keep a stable id'),
    ('widget.js', 'id="widget-model"', None,
     'the served model list needs somewhere to render into'),
    ('widget.js', 'id="widget-close"', None,
     'the close button must keep a stable id — widget.js binds it directly, '
     'and a missing id throws at script load and takes the whole panel down'),
    ('widget.js', 'id="widget-form"', None,
     'the form must keep a stable id — widget.js binds its submit handler '
     'directly, and a missing id throws at script load and takes the whole '
     'panel down'),
    ('widget.js', 'id="widget-new"', None,
     'the only control that ends a conversation must exist — without it the '
     'thread is unresettable, which is the other half of "it never resets"'),
    ('widget.js', "null, 'DELETE')", None,
     'New Chat must actually forget the thread in widget.db, not merely blank '
     'the log — a cleared screen over a remembered conversation is the widget '
     "lying about what the model holds. Checked against api()'s real "
     "three-argument shape (path, body, method) rather than the brief's "
     "{ method: 'DELETE' } sketch, which this file's api() never took"),
    ('widget.css', 'position: fixed', None,
     'the launcher and its window must be pinned to the viewport, or a '
     'widget that scrolls with the page is a fourth card, not a widget — '
     "read against the widget's own sheet, so an unrelated `position: fixed` "
     'in the Laboratory\'s stylesheet cannot satisfy it'),
    ('widget.css',
     'right: var(--gutter); bottom: calc(var(--rail-h) + var(--s-2));', None,
     "the launcher's real anchor values, right down to the unit — not just "
     'the property names, which `.widget-config`\'s `border-bottom: 1px '
     'solid var(--rule)` in this same sheet would otherwise satisfy '
     'even with both real anchors deleted. Measured off --rail-h because the '
     'status rail is fixed to the viewport floor: a literal offset would put '
     'the launcher underneath it'),
    ('widget.css',
     'bottom: calc(var(--rail-h) + var(--s-2) + 2.6rem);', None,
     "the window's real anchor values, distinct from the launcher's own — "
     'same collision this guards against as the launcher row above. The '
     "2.6rem is the launcher's own box, which has no ramp step"),
    ('widget.css', None, '--step-',
     'the widget is a helper, not a pipeline stage, and must wear no step ink'),
    ('widget.css', 'background: var(--card)', None,
     "a reply's bubble must name a colour. It named var(--slab), which is the "
     'slab-serif font stack — an invalid background, so every reply the '
     'helper had ever given rendered on no bubble at all'),
    ('widget.css', None, 'background: var(--slab)',
     'and the font stack must not come back as a colour'),
    ('widget.js', 'class="widget-grip widget-grip-top"', None,
     'the window grows from its top and left edges, because it is anchored '
     'bottom-right — the handles have to exist on those two edges'),
    ('widget.js', 'role="separator"', None,
     'the two straight handles take focus and answer the arrow keys, so a '
     'reader without a mouse can still size the window'),
    ('index.html', None, 'What you can ask',
     'the empty state is built from the served fixture, not written into the '
     'markup — a starter in two places is a starter that drifts'),
    ('widget.js', None, 'Which ports do the lab',
     'and not written into the script either: the starters are the message '
     'sent to the model, which makes them model-facing text and a fixture'),
    ('widget.js', 'data.starters', None,
     'the starters ride the /api/widget response the model list already '
     'ride, so the widget stays a sealed leaf with no new route'),
    ('widget.js', 'widget-empty', None,
     'the empty log offers four questions and clears them on the first '
     'thing said'),
    ('widget.js', 'lodestar:raglab-widget-size', None,
     'an adjusted window is a preference, not a gesture, and survives a '
     'reload under the same prefix as the settings and the last run'),
    ('widget.js', 'setPointerCapture', None,
     'a drag off a six-pixel handle must keep going — without capture the '
     'resize stops the instant the cursor outruns the edge, which is at once'),
    ('widget.js', 'raglab-widget-open', None,
     'whether the window is open must outlive the page: a helper that closed '
     'itself on every navigation is a helper the reader reopens on every '
     'surface, which is the friction this change removes'),
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
    ('widget.css', '.widget-window[hidden] { display: none; }', None,
     "a rule setting `display` beats the browser's own `[hidden] { display: "
     'none }` — found live: the window stayed visible because nothing said '
     'so explicitly. Checked against the exact rule so a bare `[hidden]` '
     'selector with no `display: none` cannot satisfy it'),
    ('widget.css', '.widget-config[hidden] { display: none; }', None,
     'same fix, for the model row the gear toggles'),
    ('widget.js', "api('/api/widget'", None,
     "the widget's script must actually call its route — checked against "
     'the call site itself, not the bare string `/api/widget`, which also '
     "names the route in this file's own header comment"),
    ('widget.js', 'escapeHtml', None,
     'a reply is model output rendered into the page, so it must go through '
     'the shared escaper like every other untrusted string'),
    ('widget.js', 'widget-model', None,
     "the script must read the gear's model select"),
    ('widget.js', '{ message, model, thread: intended }', None,
     'the chosen model must travel with every message, in the same body as '
     'the message and the thread. Pinned as the POST body itself: this used '
     "to read `model:`, which was scoped past panel.js's `embed_model:` back "
     'when the widget was a block of that file — and once the widget became '
     'its own file the only `model:` left in it was a *comment*, so the row '
     'would have passed with the model dropped from the request entirely'),
    # --- an experiment opened on the board ---------------------------------
    # The board's open button pins the Inspector to one experiment and makes
    # the same experiment's settings the Laboratory's. Every row below guards
    # one half of that: that the shared module reaches both pages, that the
    # panel takes the handoff both ways it can arrive, and that what the reader
    # is told is the lab's own voice and not the model's.
    ('index.html', '/experiment_handoff.js', None,
     'the Laboratory must load the handoff module, or the experiment the '
     'board hands over reaches a page that cannot read it'),
    ('leaderboard.html', '/experiment_handoff.js', None,
     'the board must load the same module it writes the slot with'),
    ('panel_server.py', "'experiment_handoff.js'", None,
     'a script both pages link must have a route serving it'),
    ('panel.js', 'ExperimentHandoff.taken', None,
     'the Laboratory must take what the board handed over'),
    ('panel.js', "'storage'", None,
     'the case this handoff exists for is a board in one tab and the lab in '
     'another, and a `storage` listener is the only thing that reaches an '
     'already-open Laboratory — without it the settings arrive on the next '
     'reload, which is not what the button says it did'),
    ('panel.js', 'ExperimentHandoff.reconcile', None,
     'which knobs this installation can serve is decided in the one module '
     'that decides it. There is one reader of that rule now — the archive '
     'import — because opening a board row goes through the same import, so '
     'the disagreement this used to guard against cannot arise; what it '
     'still pins is that the rule is not reimplemented in the page'),
    ('panel.js', 'adoptArchive(ArchiveIO.normalize', None,
     'opening an experiment on the board is importing its exported archive: '
     'one path, one strictness. Written out twice, an experiment opened here '
     'and the same experiment imported as a file would come to disagree '
     'about what this lab accepts — which is the whole defect this replaced, '
     'where open applied what it could and import refused outright'),
    ('widget.js', "widgetSayAfterDraw('note'", None,
     'the lab writes its own notices in its own voice: a line the page wrote '
     "must never arrive as `bot`, which is the model's. Pinned at the one "
     'call site that writes one — a notice is said only once the redraw it '
     'waited through has finished, so the kind and that wait travel '
     'together'),
    ('widget.css', '.widget-msg.note', None,
     "a message kind with no rule of its own inherits another kind's ink and "
     'reads as something the model said'),
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
    ('leaderboard.js', 'class="failed"', None,
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
    # --- Day and Night --------------------------------------------------
    ('tokens.css', ':root[data-theme="night"]', None,
     'Night must be reachable as an explicit choice, not only as a guess at '
     'what the operating system wants — the switch sets this attribute and '
     'nothing else'),
    ('tokens.css', ':root:not([data-theme="day"])', None,
     'and Day must be able to win against the machine: without this guard on '
     'the media query, picking Day on a machine set to dark would leave the '
     'query in charge and the choice would do nothing'),
    ('tokens.css', 'color-scheme', None,
     'the theme must reach the controls the browser draws itself — a select, '
     'a file picker, a scrollbar. Without this the page is Night and every '
     'native widget on it is still Day'),
    ('lab.js', 'raglab-theme', None,
     'the choice outlives the page, under one key both surfaces know'),
    ('lab.js', 'removeAttribute', None,
     'Auto is not a stored value, it is the absence of one — choosing it must '
     'clear the attribute rather than write a third name. Store "auto" and the '
     'guarded media query never fires, so Auto would silently mean Day; and '
     'the page would stop following a machine that changes its mind at sunset, '
     'which is the one thing Auto is for'),
    # --- Settings, gathered ---------------------------------------------
    ('index.html', 'id="theme-control"', None,
     'the theme control must keep its script hook'),
    ('index.html', 'aria-label="Settings"', None,
     'the round button carries no text, so it must carry a name'),
    ('chrome.css', '.settings-button', None,
     'the round Settings button is chrome both surfaces wear, so it is drawn '
     'once in the shared sheet rather than per page'),
    ('chrome.css', '.theme-control', None,
     'and so is the control inside it — the Inspector offers the same three '
     'choices from the same markup'),
    # --- the surface switcher -------------------------------------------
    ('index.html', None, 'Inspector <span class="port">:9003</span>',
     'the Inspector link drops its port: the reader is on :9002 looking at a '
     'link, not at a terminal, and the port was never the thing they were '
     'choosing'),
    ('leaderboard.html', None, 'Inspector <span class="port">:9003</span>',
     'and the leaderboard drops it for the same reason — the two surfaces '
     'wear one switcher, so a port shown on one and not the other is drift'),
]


def test_the_panel_spells_the_built_in_corpus_one_way(panel_texts):
    # this is a convention test
    """`IndexConfig.fingerprint()` drops `dataset=''` from its payload, so the
    built-in corpus is the empty string and spelling it `diary-fa` instead
    renames every collection already built under it. The panel therefore has to
    say that in exactly one place: the `<option>` values it fills the dataset
    select with, the lookup that reads that value back, and the catalogue it
    tells `ExperimentHandoff.reconcile` it serves must all agree.

    Written out three times, they did not. `servedKnobs()` listed the corpora
    by id, so a recorded config carrying the built-in corpus's own `''` was not
    in the list — and an experiment opened from the board announced the lab's
    own default corpus as *not installed here* while quietly leaving the knob
    alone. A fabricated discrepancy in the one notice whose entire job is to
    report real ones."""
    js = panel_texts['panel.js']
    spelling = js.count("'builtin' ? '' : ")
    assert spelling == 1, (
        'the built-in corpus must be spelled in one place and read from there '
        f'({spelling} copies of the rule found in panel.js)')
    served = re.search(r'function servedKnobs\(.*?\n}', js, re.S)
    assert served, 'servedKnobs must exist for the handoff to ask what is served'
    assert 'datasetValues()' in served.group(0), (
        'what servedKnobs calls a served corpus must be the same value the '
        'dataset select offers, or the empty string that means the built-in '
        'corpus reads as a corpus this installation does not have')


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
    """Four tables on this page — the readings breakdowns — and until now
    three of them were built by hand. Two of those three took the sortable
    styling from the stylesheet and none of the listeners, so they looked
    sortable and were inert. One builder, wearing the shared region from
    chrome.css, is what makes a table added later sortable and scrollable by
    having been rendered rather than by someone remembering."""
    js = panel_texts['panel.js']
    assert js.count('<table') == 1, (
        'there must be exactly one place a table is built; found '
        f"{js.count('<table')}")
    assert 'class="table-scroll" tabindex="0" role="region"' in js, (
        'the region must be focusable and labelled, or there is no keyboard '
        'way to reach the right-hand side of a fifteen-column table')
    assert '<table class="data-table">' in js
    for host in ('byType', 'ragas', 'extras', 'rows'):
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
    widget = panel_texts['widget.css']
    assert 'min-width: 24px; min-height: 24px' in widget, (
        "the widget's own header controls must clear the floor too — read "
        "against the widget's own sheet, so a 24px in the Laboratory's "
        'stylesheet cannot satisfy it')


def test_the_run_chip_names_the_run_on_screen_or_is_nothing(panel_texts):
    # this is a convention test
    """One chip built from the run on the Readings card, and nothing when there
    is no run — never a chip that refers to nothing. It is deliberately the
    *lab's* last run rather than the last conversation: the conversation is
    already on screen — the widget redraws its thread's history from the lab
    on load — so a chip naming it would repeat what the reader can read, while
    the run it was about is the one thing the log says nothing of."""
    js = panel_texts['panel.js']
    # The helper itself is widget.js now; what stays on the Laboratory is this
    # one function, at the foot of the file, because it reads a *run*.
    ask = js[js.index('function widgetRunAsk'):]
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
    assert 'Widget.offer(widgetRunAsk(result' in js, (
        'renderResult is the one place that holds the run, so it is where the '
        'chip is built — reading it back off the DOM later would be a second '
        'source for the same fact — and it is handed to the helper through '
        '`Widget.offer`, which is the whole of what a page may say to it')


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
    this claim lives in test_inspector.py; the widget's sheet is read here
    beside the panel's, because it left panel.css and the guard has to follow
    it or the rules it holds stop being checked at all."""
    for sheet in ('panel.css', 'widget.css'):
        assert _font_size_literals(panel_texts[sheet]) == [], sheet


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
    pages cannot round the same kind of thing differently. The widget's sheet
    is read here for the same reason the type guard reads it."""
    for sheet in ('panel.css', 'widget.css'):
        assert _radius_literals(panel_texts[sheet]) == [], sheet


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
        'field check above with it, or move it to test_inspector.py, where '
        'the rest went')


def test_the_archive_exchange_uses_the_codec_and_no_run_routes(client):
    # this is a convention test
    source = PANEL_JS.read_text()
    exchange = source[source.index('async function adoptArchive'):
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
    exchange = source[source.index('async function adoptArchive'):
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


def test_the_lab_page_no_longer_holds_the_experiment_ledger(panel_texts):
    # this is a convention test
    """It moved to the leaderboard, which is the surface for cross-run reading —
    the lab page is for setting up one run. Pinned as an absence so the move
    cannot half-happen and leave two tables of the same rows drifting apart."""
    html, js = panel_texts['index.html'], panel_texts['panel.js']
    assert 'Every experiment' not in html
    assert 'id="experiments"' not in html
    assert 'id="experimentDetail"' not in html
    assert 'loadExperiments' not in js
    assert 'showExperiment' not in js


def test_an_imported_archive_says_where_it_landed(panel_texts):
    # this is a convention test
    """The import used to say 'saved in Every experiment; leaderboard
    unchanged'. Both halves are now wrong: that table is gone, and the board
    reads the ledger, so the leaderboard *is* changed."""
    js = panel_texts['panel.js']
    assert 'leaderboard unchanged' not in js.lower()


def test_imported_results_render_their_archived_metric_catalogue():
    # this is a convention test
    source = PANEL_JS.read_text()
    exchange = source[source.index('async function adoptArchive'):
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


def test_the_board_is_one_table_with_both_edges_frozen(panel_texts):
    # this is a convention test
    """One table per dataset, its identity frozen left and its Inspector link
    frozen right, sortable by any column and narrowable by any of them — and the
    two composed in the one direction that matters, the filter re-running after
    every reorder."""
    js = panel_texts['leaderboard.js']
    assert "'freeze-1'" in js and "'freeze-last'" in js
    assert 'SortTable.make' in js, (
        'the page loaded sorttable.js and never called it, so click-to-sort was '
        'broken here while the lab page had it — this is the pin against that '
        'coming back')
    assert 'onApply: applyFilter' in js, (
        'the sorter re-appends every row it holds, so a filtered board has to '
        'be re-filtered after each reorder or the hidden rows land back among '
        'the visible ones')


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


def test_the_board_publishes_its_column_help_where_a_reader_can_read_it(
        panel_texts):
    # this is a convention test
    """The board wrote eleven column explanations as `title` attributes and the
    shared sorter wrote its own hint over nine of them — two decisions each
    correct alone. `sorttable.js` now yields to a heading that has something of
    its own to say (pinned in `sorttable.test.js`, because which title survives
    is behaviour), but a `title` was never where any of this is *published*: it
    answers a mouse and nothing else, which is the rule these pages already
    applied twice. So the two sentences a reader actually needs are in the
    page's own text under the table — that the pipeline cell opens, which is the
    only announcement the settings reveal exists at all, and that a `fake`
    backend makes a row a rehearsal rather than a measurement, which was prose
    on the lab page until the card holding it was deleted."""
    js = panel_texts['leaderboard.js']
    # Whitespace-normalised: the paragraph is a wrapped template literal in the
    # source and one flowing sentence on screen, and it is the screen this is a
    # claim about.
    hint = ' '.join(js[js.index('class="table-hint"'):
                       js.index('</p>', js.index('table-hint'))].split())
    assert 'hover it or give it focus' in hint, (
        'the reveal has no other announcement on the page')
    assert 'rehearsal of the pipeline and not a measurement' in hint, (
        'the sentence explaining a fake backend has to exist somewhere a '
        'reader can see it')


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


def test_the_widget_is_two_shared_files_every_surface_can_load(client):
    # this is an integration test
    """The widget is a helper any surface gains by loading it, not a feature of
    one page. Its rules and its script are served from the root like tokens.css
    and lab.js, so there is one definition rather than three copies."""
    css = client.get('/widget.css')
    js = client.get('/widget.js')
    assert css.status_code == 200
    assert css.headers['content-type'].startswith('text/css')
    assert js.status_code == 200
    assert js.headers['content-type'].startswith('application/javascript')
    assert '.widget-launch' in css.text
    assert 'widgetSay' in js.text
    assert '.widget-launch' not in client.get('/panel.css').text, (
        'the widget rules must live in one sheet, not two')


def test_every_surface_carries_the_widget(panel_texts):
    # this is a convention test
    """One helper, three surfaces. A reader who can ask a question on the
    Laboratory and not on the board is reading two different labs.

    Built over `served_lab.app`, not the shared `client` fixture — that
    fixture is `panel_server.create_app()` alone, and the Inspector is only
    mounted where the three surfaces actually come together, the way
    `test_inspector.py`'s own cross-surface checks already do."""
    from fastapi.testclient import TestClient
    from raglab.dashboard import served_lab

    inspector = TestClient(served_lab.app).get('/inspector/').text
    for name, page in (('index.html', panel_texts['index.html']),
                       ('leaderboard.html', panel_texts['leaderboard.html']),
                       ('inspector.html', inspector)):
        assert 'src="/widget.js"' in page, f'{name} does not load the widget'
        assert 'href="/widget.css"' in page, f'{name} does not style the widget'


def test_the_widget_sends_the_thread_it_is_in(panel_texts):
    # this is a convention test
    """The widget's memory is a thread in widget.db, not a page's lifetime. The
    POST must carry which thread, or every question lands in the same one."""
    script = panel_texts['widget.js']
    assert re.search(r"widgetStream\('/api/widget/stream',\s*\{[^}]*\bthread\b",
                     script), ('the widget POST must carry the thread id')
    assert 'crypto.randomUUID' not in script, (
        'a per-page id is exactly the reset this change removed')


def test_the_widget_types_the_answer_out_as_it_arrives(panel_texts):
    # this is a convention test
    """The reply used to land in one piece one round trip after Send. It comes
    from `/api/widget/stream` now, read as it arrives — and the pieces are only
    how it arrived: the final event carries the reply the lab's own log holds,
    and the bubble adopts that, so the screen and the transcript cannot differ.
    A stream that stops part-way leaves what came marked as stopped rather than
    dressed up as a whole answer."""
    script, style = panel_texts['widget.js'], panel_texts['widget.css']
    assert "'/api/widget/stream'" in script, (
        'the widget must ask the streaming route')
    assert 'getReader()' in script, (
        "the answer must be read as it arrives, not awaited as one body")
    assert 'widgetFinish(live, data.reply)' in script, (
        "the reply the lab holds must replace the pieces the page typed")
    assert 'widgetStopped(live)' in script, (
        'a stream that failed must not leave a fragment looking finished')
    assert '.widget-msg.bot.streaming::after' in style, (
        'an answer still being written must say so on screen')
    assert '.widget-msg.bot.stopped' in style, (
        'and one that stopped part-way must say that instead')


def test_the_widget_shows_the_token_account_under_a_reply(panel_texts):
    # this is a convention test
    """The account travels with the reply and the page shows it — a faint
    meta line, only when the backend reported one: an unreported account
    renders nothing rather than a made-up zero."""
    script = panel_texts['widget.js']
    assert 'input_tokens' in script, (
        'the widget must read the served token account')
    assert 'output_tokens' in script, (
        'both directions of the account, not just one')


def test_the_widget_serves_the_conversation_it_holds(client, monkeypatch):
    # this is an integration test
    """A refresh redraws the log from the lab, not from a copy in the browser:
    what a reader sees is exactly what the model remembers, so the two cannot
    drift apart. And what it serves about a thread — which experiment it is
    about, when it began — must be what a turn actually wrote, not two empty
    strings dressed as facts.

    A question is put through the real route, the real `ask`, the real graph
    and the real checkpointer; only the model is a fake, because the suite is
    offline and no test here may reach OpenRouter. That is what makes this an
    honest reading of what a reader would see: nothing between the POST and
    the GET is stubbed."""
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain.agents import create_agent

    from raglab.agents import widget
    from raglab.agents.widget import conversation_memory as memory
    from raglab.agents.widget.hooks import MIDDLEWARE

    memory.forget('exp-route')
    # A thread nobody has used says so with three empty answers rather than
    # with an error — the empty log and its starters are the honest rendering
    # of a conversation that has not happened yet.
    read = client.get('/api/widget/history', params={'thread': 'exp-route'})
    assert read.status_code == 200
    assert read.json() == {'thread': 'exp-route', 'experiment_id': '',
                           'started_at': '', 'turns': []}

    def fake_agent(model):
        # `_build_agent`'s own shape with the one part that needs a key and a
        # network swapped out: same state schema, same middleware, same
        # process-wide checkpointer, so what lands in widget.db is written by
        # the graph exactly as it would be in production. No tools, because a
        # fake chat model cannot bind them and a scripted reply calls none.
        return create_agent(GenericFakeChatModel(messages=iter([
                                AIMessage(content='four judged metrics decide')])),
                            system_prompt='x', middleware=MIDDLEWARE,
                            state_schema=memory.WidgetState,
                            checkpointer=memory.saver())

    widget.reset()
    monkeypatch.setattr(widget.backends, '_build_agent', fake_agent)
    try:
        said = client.post('/api/widget', json={'message': 'which metrics decide?',
                                                'model': 'openai/gpt-5-nano',
                                                'thread': 'exp-route'})
        assert said.status_code == 200, said.text
    finally:
        widget.reset()

    read = client.get('/api/widget/history', params={'thread': 'exp-route'})
    body = read.json()
    assert body['turns'] == [
        {'role': 'you', 'text': 'which metrics decide?'},
        {'role': 'bot', 'text': 'four judged metrics decide'}]
    # The two fields the route reports beside the turns. They were declared on
    # `WidgetState`, written by nothing, and pinned here as empty strings — a
    # route stating as fact about every thread the one thing it did not know.
    assert body['experiment_id'] == 'exp-route'
    assert body['started_at']


def test_the_route_serves_a_turns_token_account_when_one_was_reported(client):
    # this is an integration test
    """The route is a thin pass-through over `conversation_memory.history`, so
    what it proves here is that nothing between the checkpointer and the JSON
    response strips the account back off — the same seeding helper the unit
    tests use, read back through the actual FastAPI route rather than the
    Python function directly."""
    from langchain_core.messages import AIMessage, HumanMessage
    from raglab.agents.widget.tests.widget_examples import write_messages

    write_messages('exp-billed-route', [
        HumanMessage(content='what did that cost?'),
        AIMessage(content='1692 total', usage_metadata={
            'input_tokens': 1630, 'output_tokens': 62, 'total_tokens': 1692})])

    read = client.get('/api/widget/history', params={'thread': 'exp-billed-route'})
    assert read.status_code == 200
    assert read.json()['turns'] == [
        {'role': 'you', 'text': 'what did that cost?'},
        {'role': 'bot', 'text': '1692 total',
         'input_tokens': 1630, 'output_tokens': 62}]


def test_new_chat_empties_one_conversation_and_no_other(client):
    # this is an integration test
    """The only control that ends a conversation, and it ends exactly one."""
    from langchain_core.messages import AIMessage, HumanMessage
    from raglab.agents.widget.tests.widget_examples import write_messages

    # Seeded through the real saver, the same helper
    # `agents/widget/tests/test_conversation_memory.py` seeds its own threads
    # with — one definition of what a checkpoint has to look like, not two
    # that a langgraph upgrade could silently pull apart.
    for thread in ('exp-a', 'exp-b'):
        write_messages(thread, [HumanMessage(content='q'), AIMessage(content='a')])

    gone = client.delete('/api/widget/history', params={'thread': 'exp-a'})
    assert gone.status_code == 200
    assert gone.json()['turns'] == []
    kept = client.get('/api/widget/history', params={'thread': 'exp-b'})
    assert kept.json()['turns'] == [{'role': 'you', 'text': 'q'},
                                    {'role': 'bot', 'text': 'a'}]


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


# --- Day and Night ----------------------------------------------------------
#
# The page had a dark theme before this: a `prefers-color-scheme` block that
# read the machine's setting and offered no way to disagree with it. What is
# new is the disagreeing — an explicit choice, stored, that outranks the
# machine — and the tests below are about the two failure modes that come with
# it: a theme that arrives a frame late, and a palette that has to be written
# twice because a media query and an attribute selector cannot share a block.

THEME_KEY = 'raglab-theme'
SURFACES = ('index.html', 'leaderboard.html')


@pytest.mark.parametrize('surface', SURFACES)
def test_every_surface_stamps_the_stored_theme_before_it_paints(panel_texts, surface):
    # this is a convention test
    """A theme applied by the page's own script arrives after the first paint,
    so a reader who chose Night sees a white page flash on every navigation —
    on this lab, across three surfaces that link to each other, that is a
    flash per click. The fix is the one thing that cannot be deferred: a
    blocking script in the head, before anything renders, whose whole job is
    to copy the stored choice onto the root element. It is inline rather than
    in lab.js for the same reason — a separate file is a request, and the
    flash is exactly as long as that request."""
    head = panel_texts[surface].split('</head>')[0]
    assert THEME_KEY in head, (
        'the stored choice must be read in the head: read it from lab.js at '
        'the foot of the page and every navigation flashes the other theme')
    assert 'documentElement' in head, (
        'and stamped on the root element, which is what both the attribute '
        'selector and the guarded media query in tokens.css hang off')


def test_the_night_palette_is_written_once_and_read_twice(panel_texts):
    # this is a convention test
    """Night has to be reachable two ways — as `[data-theme="night"]`, the
    explicit choice, and through the media query, for a reader on Auto whose
    machine is dark. CSS gives no way to put one declaration block behind both
    selectors, so the obvious spelling duplicates the whole palette, and the
    two copies then drift the way a step ink already drifted between the
    panel and the Inspector. Instead the values live once, on bare `:root`, as
    `--night-*`; the two selectors only assign them. This pins that: every
    Night value appears exactly once in the sheet, and both selectors read
    them through `var()` rather than restating them."""
    tokens = panel_texts['tokens.css']
    values = re.findall(r'--night-[a-z-]+:\s*([^;]+);', tokens)
    assert values, 'the Night palette must be named as --night-* tokens'
    for value in values:
        assert tokens.count(value.strip()) == 1, (
            f'{value.strip()!r} is written more than once — a Night value '
            'restated under the second selector is the drift this shape exists '
            'to prevent')
    assigned = []
    for selector in (':root[data-theme="night"]', ':root:not([data-theme="day"])'):
        block = tokens[tokens.index(selector):]
        block = block[:block.index('}')]
        assert 'var(--night-' in block, (
            f'{selector} must read the named palette, not restate it')
        assigned.append(sorted(re.findall(r'(--[a-z-]+):\s*([^;]+);', block)))
    chosen, inherited = assigned
    assert chosen == inherited, (
        'the two Night blocks must assign the same tokens to the same values. '
        'Naming the palette once stops the values drifting; only this stops '
        'one block gaining a token the other never got — which is the same '
        'bug wearing a different hat, and it would show up only for readers '
        'on Auto, or only for readers who picked Night, never for both')


@pytest.mark.parametrize('sheet', ('tokens.css', 'panel.css', 'widget.css'))
def test_no_dark_block_outranks_an_explicit_choice(panel_texts, sheet):
    # this is a convention test
    """Every `prefers-color-scheme: dark` block on either surface has to carry
    the `:not([data-theme="day"])` guard. An unguarded one is worse than no
    theming at all: the switch would appear to work everywhere except the few
    tokens that block happens to hold, so choosing Day on a dark machine would
    give a light page with, say, dark-mode alert washes still on it — a bug
    that only shows up on one theme on one machine. The guard is not optional
    decoration; it is what makes the choice a choice."""
    css = panel_texts[sheet]
    for match in re.finditer(r'@media\s*\(prefers-color-scheme:\s*dark\)\s*\{', css):
        block = css[match.end():match.end() + 200]
        assert ':not([data-theme="day"])' in block, (
            f'an unguarded dark block in {sheet} at character {match.start()} — '
            'the machine would outrank the reader for whatever tokens it holds')


def test_each_theme_says_which_way_the_browser_should_draw_its_own_controls(panel_texts):
    # this is a convention test
    """`color-scheme` is the only thing a page can say about the widgets it
    does not draw: the select menus on the knob surface, the archive file
    picker, the scrollbars on every table region. Style them all you like and
    they stay light until this property says otherwise — which on Night means
    a page of dark cards holding white dropdowns. Auto gets `light dark` so
    the browser keeps following the machine, exactly as the palette does."""
    tokens = panel_texts['tokens.css']
    assert re.search(r':root\s*\{[^}]*color-scheme:\s*light\s+dark', tokens, re.S), (
        'Auto must hand the decision back to the browser with `light dark`')
    night = tokens[tokens.index(':root[data-theme="night"]'):]
    assert 'color-scheme: dark' in night[:night.index('}')], (
        'Night must say dark, or its native controls stay light')
    day = tokens[tokens.index(':root[data-theme="day"]'):]
    assert 'color-scheme: light' in day[:day.index('}')], (
        'and Day must say light, or Night controls survive the switch back')


def test_the_settings_popover_gathers_every_installation_level_control(panel_texts):
    # this is a convention test
    """Three unrelated controls used to sit loose in the top bar — Settings,
    Import JSON, Export experiment — competing for the same corner as the
    surface switcher. They have one thing in common: none of them is about the
    experiment on screen, they are about this installation. So they are one
    button now, and this pins that the archive pair actually moved inside the
    popover rather than merely surviving somewhere on the page. The ids are
    unchanged on purpose: panel.js and archive_io.js reach for them by id, and
    a move that renames its hooks is a rewrite, not a move."""
    html = panel_texts['index.html']
    popover = html[html.index('id="app-settings-panel"'):]
    popover = popover[:popover.index('</header>')]
    for hook in ('id="theme-control"', 'id="openrouter_key"',
                 'id="archive-import"', 'id="archive-export"', 'id="archive-file"'):
        assert hook in popover, (
            f'{hook} must live inside the Settings popover — a control left '
            'loose in the bar is the crowding this replaces')


@pytest.mark.parametrize('surface', SURFACES)
def test_the_surface_switcher_reads_lab_inspector_leaderboard(panel_texts, surface):
    # this is a convention test
    """The Inspector reads what the Laboratory just produced; the Leaderboard
    ranks across every run there has ever been. So the switcher runs
    Laboratory, Inspector, Leaderboard — nearest first, widest last — rather
    than the order the three surfaces happened to be built in. Checked as
    positions in the nav rather than as a substring of it, because the nav's
    markup differs per surface (each page marks its own entry current)."""
    nav = panel_texts[surface]
    nav = nav[nav.index('<nav class="topnav"'):]
    nav = nav[:nav.index('</nav>')]
    order = [nav.index(name) for name in ('Laboratory', 'Inspector', 'Leaderboard')]
    assert order == sorted(order), (
        'the switcher must run Laboratory, Inspector, Leaderboard')


def test_a_hovered_row_lights_up_whole_and_in_one_direction(panel_texts):
    # this is a convention test
    """Two defects that Night made visible and Day had been hiding. The hover
    background was `--plate`, the page behind the table: on Day that is a
    hundredth of a step off the even-row stripe, so hovering half the rows did
    nothing you could see; on Night the plate is darker than every row, so the
    hovered row read as a hole punched in the table. And the rule never covered
    the two frozen identity columns, whose striping selector matches at the same
    specificity — so a hovered even row lit up everywhere except the columns
    saying which run it was. `--rule` fixes the first (one step past the stripe,
    in whichever direction that theme's rules run) and the pair of frozen
    selectors, placed after the striping rule, fixes the second."""
    css = panel_texts['chrome.css']
    hover = css[css.index('.data-table tbody tr:hover td'):]
    hover = hover[:hover.index('}')]
    assert 'var(--rule)' in hover and 'var(--plate)' not in hover, (
        'row hover must be a step past the stripe, not the page behind the table')
    frozen = ('.data-table tbody tr:hover .freeze-1,\n'
              '.data-table tbody tr:hover .freeze-last')
    assert frozen in css, (
        'the frozen columns must take the hover too, or a hovered even row is '
        'two-tone — its identity column keeps the stripe while the numbers '
        'light up. Both edges: the board freezes a pipeline sentence on the '
        'left and its Inspector link on the right, and the right-hand column '
        'has the same problem for the same reason')
    # After *both* striping rules, not just the first. They match at the same
    # specificity, so order is the only thing deciding which wins — and a hover
    # rule placed between the two stripes loses to the second one, which is a
    # way of half-fixing this that still leaves one edge two-tone.
    for stripe in ('nth-child(even) .freeze-1', 'nth-child(even) .freeze-last'):
        assert css.index(stripe) < css.index(frozen), (
            f'the hover rule must come after {stripe}, or that column keeps '
            'its stripe while the rest of the row lights up')


def test_no_surface_links_to_a_hardcoded_localhost(panel_texts):
    # this is a convention test
    """Three surfaces on one origin reach each other by path. A hardcoded
    http://localhost:9002 was how a second port had to be linked; it now
    breaks the moment the lab is served anywhere but this machine, and
    :9003 points at nothing at all."""
    for name in ('index.html', 'leaderboard.html', 'leaderboard.js'):
        assert 'localhost:900' not in panel_texts[name], (
            f'{name} still links out to a port instead of a path')
    # The board's row link is built at runtime from a template string rather
    # than sitting in markup, so the loop above already covers it once
    # `leaderboard.js` is in scope — this assertion additionally pins the
    # replacement shape, so a fix that merely drops the origin without
    # carrying the experiment id on a path of this origin still fails here.
    # *Which* surface it lands on is a different claim, pinned beside the
    # handoff it belongs to in `board_handoff.test.js`.
    assert 'href="/?experiment=' in panel_texts['leaderboard.js'], (
        'the board must open the experiment by path on this origin, with the '
        'id carried in the query string')
    # The panel's own door to the Inspector, opened from 3 · Generation once a
    # run exists: same requirement, same reason, a second place the origin
    # could have been left behind.
    assert 'id="open-inspector" href="/inspector"' in panel_texts['index.html'], (
        "the panel's lab-link to the Inspector must be a path on this "
        'origin, not a link to a port')


# --- the knob surface, end to end -------------------------------------------
# Three lists have to name the same knobs and are written in three places by
# hand: the dataclasses in `configuration/lab_config.py`, the codec's
# `CONFIG_TEMPLATE` in `archive_io.js`, and the controls the panel renders.
# Nothing tied them together, so a knob added to one and forgotten in another
# was found by a reader whose export threw or whose imported value silently
# stayed at this lab's default. These two tests are that tie.

# The knobs no control shows, carried through untouched by `UNSHOWN` so that
# importing a config that sets them cannot quietly reset them to this lab's
# defaults. Named here rather than derived, because that is the point: a knob
# that joins this list joins it deliberately, in a diff someone reads.
UNSHOWN_KNOBS = frozenset({
    'retrieval.rrf_k', 'retrieval.agentic_weights', 'retrieval.max_context_chars',
})


def _config_knobs():
    """Every knob the lab has, as `group.field`, from the definition itself."""
    return {f'{group}.{field}'
            for group, fields in config.LabConfig().to_dict().items()
            if group != 'label'
            for field in fields}


def _function_body(js, name):
    """One top-level function, from its `function` line to the `}` in column 0.

    Slicing between two *named* functions instead would depend on the order
    they happen to be written in, and `applyDefaults` sits below `keepUnshown`
    — which is how an earlier draft of this file read an empty function body
    and reported every knob in the lab as unwritten.
    """
    start = js.index(f'function {name}(')
    return js[start:js.index('\n}', start)]


def _archive_template_knobs(archive_js):
    """The keys `CONFIG_TEMPLATE` promises, which the codec enforces exactly.

    Brace-aware on purpose: the template closes `generation` and then names a
    top-level `label`, so a scan that only looked for `name:` would file that
    under whichever group it saw last and invent a knob nobody wrote.
    """
    body = archive_js[archive_js.index('const CONFIG_TEMPLATE'):
                      archive_js.index('const VOCABULARIES')]
    token = re.compile(r'(?P<group>\w+)\s*:\s*\{|(?P<close>\})'
                       r'|(?P<field>\w+)\s*:')
    knobs, outside, group = set(), set(), None
    for match in token.finditer(body):
        if match.group('group'):
            group = match.group('group')
        elif match.group('close'):
            group = None
        elif group:
            knobs.add(f'{group}.{match.group("field")}')
        else:
            outside.add(match.group('field'))
    assert outside == {'label'}, (
        'the archive template grew a key outside the three config groups, '
        f'which this reading would drop: {sorted(outside - {"label"})}')
    return knobs


def _controlled_knobs(panel_js):
    """Knobs the panel reads back off a real control.

    Two kinds. Most name their element inline (`chunker: $('chunker').value`);
    the model roles are rendered from the served catalogue and read through
    `data-field` in a loop, so their names live in `model_role_catalogue` and
    not in this file — which is what keeps a role added there from having to
    be added here too.
    """
    body = _function_body(panel_js, 'readShownConfig')
    knobs, group = set(), None
    for line in body.splitlines():
        opened = re.match(r'\s*(index|retrieval|generation):\s*\{', line)
        if opened:
            group = opened.group(1)
        for field in re.findall(r'(\w+):\s*\+?\$\(', line):
            if group and field != 'label':
                knobs.add(f'{group}.{field}')
    return knobs | {role.field for role in model_roles.ROLES}


def test_every_knob_reaches_the_export_file(panel_texts):
    # this is a convention test
    """Exporting is all of the config or none of it.

    `exportArchive` hands `readConfig()` to `ArchiveIO.settings`, whose
    `shape()` refuses a config with a missing *or* an extra key — so a knob
    that fell out of the template does not quietly leave a gap in the file,
    it stops every export on this installation. This asserts the three lists
    agree before that can happen, and that the export path can actually
    produce each one: a knob is either on a control or carried by `UNSHOWN`,
    and a knob that is neither reaches `readConfig()` from nowhere.
    """
    defined = _config_knobs()
    template = _archive_template_knobs(panel_texts['archive_io.js'])
    assert template == defined, (
        'the archive template and the lab config name different knobs — '
        f'only in the config: {sorted(defined - template)}; '
        f'only in the template: {sorted(template - defined)}')

    reachable = _controlled_knobs(panel_texts['panel.js']) | UNSHOWN_KNOBS
    assert defined <= reachable, (
        'these knobs exist in the config but the export path can read them '
        f'from neither a control nor UNSHOWN: {sorted(defined - reachable)}')
    assert UNSHOWN_KNOBS <= defined, (
        'UNSHOWN_KNOBS names a knob the config no longer has: '
        f'{sorted(UNSHOWN_KNOBS - defined)}')


def test_every_knob_can_be_set_by_an_import(panel_texts):
    # this is a convention test
    """Importing is the same promise read backwards.

    `writeArchiveSettings` applies a config with `applyDefaults` and then
    checks that reading the panel back reproduces it exactly, refusing the
    import otherwise. That check is the guarantee at run time; this is the
    same guarantee at build time, which is where a reader would rather meet
    it. Every knob on a control must be written by `applyDefaults`, or an
    imported value lands in an object nothing puts on screen; every knob
    without one must be in `UNSHOWN_KNOBS`, which `keepUnshown` carries.
    """
    panel_js = panel_texts['panel.js']
    applied = {f'{group}.{field}' for group, field in re.findall(
        r'\bd\.(index|retrieval|generation)\.(\w+)',
        _function_body(panel_js, 'applyDefaults'))}
    # The model roles again: written by the same `data-field` loop that reads
    # them, from `(d[group] || {})[field]`, so they carry no `d.<group>.<field>`
    # for the pattern above to find.
    applied |= {role.field for role in model_roles.ROLES}

    controlled = _controlled_knobs(panel_js)
    assert controlled <= applied, (
        'these knobs have a control the panel reads but never writes, so an '
        f'imported value would never reach it: {sorted(controlled - applied)}')

    unreachable = _config_knobs() - applied - UNSHOWN_KNOBS
    assert not unreachable, (
        'these knobs can be neither adjusted nor carried through an import: '
        f'{sorted(unreachable)}')

    # And the run-time half, which is what makes a partial import impossible
    # rather than merely unlikely: the panel refuses an import it cannot
    # reproduce, instead of applying the knobs it understood and keeping quiet
    # about the rest.
    assert 'Imported settings could not be represented exactly by this panel' \
        in panel_js, ('the import must verify it reproduced the config it was '
                      'given, or a knob dropped here is a knob nobody is told '
                      'about')
