"""Tests for the served panel's own markup and script."""
import json
import re

import pytest

from raglab import baseline, config, evaluate, metrics

from conftest import RAGLAB_DIR


# --- the served panel's conventions, as one table ---------------------------

@pytest.fixture(scope='module')
def panel_texts(client):
    """Every named text the convention table below checks, fetched the one
    way a browser actually reaches it (`client.get`) — a second disk read of
    the same file would be a claim about a copy nobody is served. Two entries
    are carved out of the full page, and one out of `panel.js`, because their
    claim is *where* the text sits rather than merely that it exists
    somewhere on the page — the same regions the retired pin tests scoped
    their own reads to. `server.py` is the one entry read from disk: the
    lab's Python source is never served, so there is no route to prefer over
    it."""
    html = client.get('/').text
    css = client.get('/panel.css').text
    js = client.get('/panel.js').text

    embed_label = re.search(r'<label>Embedding model.*?</label>', html, re.S)
    model_card = re.search(r'<section[^>]*id="modelCard".*?</section>', html, re.S)
    assert embed_label and model_card, 'the panel dropped a section this table reads'

    handler = js[js.index("$('use-production').onclick"):]
    handler = handler[:handler.index('\n};')]

    return {
        'index.html': html,
        'panel.css': css,
        'panel.js': js,
        'index.html (embedding-model label)': embed_label.group(0),
        'index.html (modelCard section)': model_card.group(0),
        'panel.js (use-production handler)': handler,
        'server.py': (RAGLAB_DIR / 'server.py').read_text(encoding='utf-8'),
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
    ('index.html', 'ragas_decision', None,
     'the leaderboard must say which column chose the architecture'),
    ('panel.js', 'ragas_decision_stderr', None,
     'the leaderboard must show the deciding score with its error, never the '
     'mean alone'),
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
    ('server.py', 'api/queries', None,
     "the route itself must stay: the Inspector's followed query view reads "
     'whatever runs through it'),
    ('index.html', 'id="mode"', None,
     'the mode dropdown must read the served modes rather than a local copy'),
    ('index.html', 'id="retrieve-selected"', None,
     'retrieval for the selected questions must stay one click away'),
    ('index.html', 'id="use-production"', None,
     "the shipped assistant's settings must stay one click away"),
    ('index.html', 'id="board"', None,
     'the ranked leaderboard must stay on the page'),
    ('index.html', 'id="experiments"', None,
     'the ledger of every experiment must sit beside the ranked leaderboard'),
    ('panel.js', '/api/experiments', None,
     'the experiments list must be read from the ledger route'),
    ('index.html', 'sorttable.js', None,
     'the panel must load the shared column sorter'),
    ('index.html', None, 'ragas_decision ▼',
     'the hard-coded sort arrow must not come back to the markup — it cannot '
     'move once the column is sorted a different way'),
    ('panel.js', None, 'ragas_decision ▼',
     'the hard-coded sort arrow must not come back to the script either'),
    ('panel.js', '/api/evaluations?limit=', None,
     'the leaderboard must ask for a stated limit rather than the whole '
     'directory'),
    ('panel.js', 'OPTIONS.production', None,
     'the production preset must be read from the served options, not kept '
     'a second time in the script'),
    ('panel.js (use-production handler)', None, '/api/indexes',
     'the production preset must not start a build'),
    ('panel.js (use-production handler)', None, 'doRetrieve',
     'the production preset must not start a retrieval'),
    ('index.html', None, 'then run',
     'the preset button must no longer claim to run anything'),
    ('panel.js', None, 'then run',
     'the preset button must no longer claim to run anything, in the script '
     'either'),
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


def test_the_preset_carries_the_fields_the_panel_cannot_show(client):
    # this is a convention test
    """Three fields of a `LabConfig` have no control on either panel —
    `rrf_k`, `agentic_weights` and `max_context_chars` — and the production
    preset sets all three. Dropped, the run would fall back to
    `LabConfig`'s own defaults while the label claims the shipped
    Assistant. The three happen to equal the lab's defaults today, which is
    why this tripwire exists: a field whose preset value silently disagrees
    with the lab default is what has to be noticed."""
    body = client.get('/api/options').json()
    preset, defaults = body['production'], body['defaults']
    # A control may be marked up in the page or wired up in its script, so
    # both are searched — the split moved `$('key')` calls into panel.js.
    panel = client.get('/').text + client.get('/panel.js').text

    unshown = {}
    for group in ('index', 'retrieval', 'generation', 'agent'):
        for key, value in preset[group].items():
            # A control is `$('key')` in the panel, or a model dropdown carrying
            # the dotted path — the two ways this page reads a field.
            if f"$('{key}')" in panel or f'"{group}.{key}"' in panel:
                continue
            unshown[f'{group}.{key}'] = (value, defaults[group].get(key))

    assert unshown, 'if nothing is unshown this guard has become dead weight'
    assert 'UNSHOWN' in panel, 'the panel must carry what it cannot render'
    for path, (wanted, fallback) in unshown.items():
        assert wanted == fallback, (
            f'{path}: the preset wants {wanted!r} but the lab defaults to '
            f'{fallback!r}, and the panel has no control for it — so a run '
            f'labelled "the shipped assistant" would use {fallback!r}. Give it '
            f'a control, or confirm the carry-through still reaches the payload.')


def test_the_panel_fills_the_projects_settings_from_the_served_preset(client):
    # this is a convention test
    """The preset's own label is served alongside it, so its presence in the
    frontend would mean the frontend had a second copy of the preset to go
    stale. The button's own substance — that it reads `OPTIONS.production`
    and starts neither a build nor a retrieval — is the convention table
    above; this is the one part that needs a live value from the service to
    check, so it stays its own test."""
    served = client.get('/api/options').json()['production']
    panel = client.get('/panel.js').text
    assert served['label'] == baseline.LABEL
    assert served['label'] not in panel, 'the panel keeps its own preset'


# --- sortable columns and the shared token sheet ---------------------------

def test_both_lab_pages_share_one_column_sorter(client):
    # this is a convention test
    """One file for both pages rather than a copy each, so "what does
    clicking a header do" has one answer instead of two that drift. The
    order it produces is unit tested in `tests/sorttable.test.js`. Whether
    the panel actually loads it, and whether the hard-coded arrow has come
    back, are rows in the convention table above; the Inspector's half of
    this claim lives in test_inspector.py."""
    from raglab.server import STATIC

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
    from raglab.server import STATIC

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
