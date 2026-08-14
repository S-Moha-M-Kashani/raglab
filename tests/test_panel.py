"""Tests for the served panel's own markup and script."""
import re

from raglab import baseline, config, metrics

from conftest import RAGLAB_DIR


# --- the service is served ---------------------------------------------

def test_panel_is_served(client):
    page = client.get('/')
    assert page.status_code == 200
    assert 'RAG Lab' in page.text


def test_the_panels_style_and_script_are_served_as_their_own_files(client):
    """The markup, the style and the script were split into three files —
    `index.html`, `panel.css`, `panel.js` — and a split that is not routed is
    just a dead file next to the one still being served. This pins the two
    new routes rather than only the files on disk."""
    css = client.get('/panel.css')
    assert css.status_code == 200
    assert css.headers['content-type'].startswith('text/css')
    assert '--step-index' in css.text

    js = client.get('/panel.js')
    assert js.status_code == 200
    assert js.headers['content-type'].startswith('application/javascript')
    assert 'model_roles' in js.text

    html = client.get('/').text
    assert '/panel.css' in html
    assert '/panel.js' in html


# --- the standalone panel: models, colours, metrics ---------------------

def test_the_standalone_panel_offers_the_model_pickers_too():
    """The lab still runs without a board, and that panel must not be the one
    place where a model is hard-coded."""
    from raglab.server import STATIC
    js = (STATIC / 'panel.js').read_text(encoding='utf-8')
    assert 'model_roles' in js and 'rag-model' in js


def test_the_standalone_panel_reads_only_fields_the_lab_still_produces():
    """A field the panel reads but the lab no longer sends prints
    "undefined" or throws — checked against what the lab actually returns
    rather than a list of names someone has to remember to prune."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')

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


def test_the_standalone_panel_offers_the_embedding_models_too():
    from raglab.server import STATIC
    js = (STATIC / 'panel.js').read_text(encoding='utf-8')
    assert 'embed_models' in js and 'embedder_hints' in js


def test_the_standalone_panel_colour_codes_the_steps_too():
    """One ink per step, defined once as a token and applied by data-step, so the
    two panels cannot end up disagreeing about what orange means."""
    from raglab.server import STATIC
    css = (STATIC / 'panel.css').read_text(encoding='utf-8')
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    for token in ('--step-index', '--step-retrieval', '--step-generation'):
        assert token in css, token
    assert 'data-step="index"' in html
    assert 'data-step="retrieval"' in html
    assert 'data-step="generation"' in html


def test_the_standalone_panel_takes_its_metric_definitions_from_the_service():
    """No second list of score labels: the panel that runs without a board has to
    explain a metric the same way the board's page does, or the same number ends
    up with two names and one definition."""
    from raglab.server import STATIC
    js = (STATIC / 'panel.js').read_text(encoding='utf-8')
    assert 'OPTIONS.metrics' in js
    assert 'metric.${key}' in js or "metric.' + key" in js
    assert 'SCORE_CARDS' not in js, 'the hard-coded score list is back'


def test_the_standalone_panel_says_which_backends_consult_the_model():
    """The label must name every backend that can actually load a model, and
    no backend whose catalogue is gone."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    label = re.search(r'<label>Embedding model.*?</label>', html, re.S)
    assert label, 'the standalone panel lost its embedding-model label'
    assert 'sentence-transformers' in label.group(0)
    assert 'fastembed' in label.group(0)
    assert 'openai' not in label.group(0)


def test_the_standalone_panel_keeps_every_model_in_one_place():
    """The embedder is a language model too, so it belongs in the model column
    with the other seven rather than buried among the chunking knobs."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    card = re.search(r'<section[^>]*id="modelCard".*?</section>', html, re.S)
    assert card, 'the standalone panel has no model column'
    assert 'id="embedder"' in card.group(0)
    assert 'id="embed_model"' in card.group(0)


def test_the_standalone_panel_ranks_the_leaderboard_by_the_deciding_score():
    """Two numbers on one row invite ranking by the wrong one, so the panel has
    to say which column chose the architecture."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    js = (STATIC / 'panel.js').read_text(encoding='utf-8')
    assert 'ragas_decision' in html
    # And by the score *with its error*: neither panel may show the mean alone,
    # because the candidates in a sweep sit inside each other's error bars.
    assert 'ragas_decision_stderr' in js


# --- progress and control -------------------------------------------------

def test_the_panel_reads_the_progress_detail():
    """A judged local run spends hours in one stage, and the detail is the
    only thing that moves — the panel may not quietly stop showing it."""
    panel = (RAGLAB_DIR / 'static' / 'panel.js').read_text(encoding='utf-8')
    assert 'job.detail' in panel


def test_the_panel_offers_a_cooperative_stop():
    """A run that cannot be stopped is a run you kill the process to escape,
    and the ledger row it was about to write goes with it."""
    html = (RAGLAB_DIR / 'static' / 'index.html').read_text(encoding='utf-8')
    js = (RAGLAB_DIR / 'static' / 'panel.js').read_text(encoding='utf-8')
    assert 'Stop experiment' in html
    assert "'/api/jobs/' + jobId + '/cancel'" in js


def test_the_panel_watches_the_ask_as_a_job():
    """The panel may not block on a bare fetch behind a static note: the ask
    goes through the same job box as builds and runs, so the reader sees
    stage, fraction and detail instead of guessing whether anything is
    happening at all."""
    panel = (RAGLAB_DIR / 'static' / 'index.html').read_text(encoding='utf-8')
    assert 'retrieving…' not in panel


# --- the inspector door and the retired one-question form ----------------

def test_the_panel_sends_you_to_the_inspector():
    """The lab measures; the Inspector shows why. The panel has to name the
    door, or :9003 is a port you have to already know about."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    assert 'localhost:9003' in html, 'the panel does not link to the Inspector'
    assert 'inspector' in html.lower(), 'the panel does not name the Inspector'


def test_the_panel_no_longer_asks_one_question():
    """Asking one question lives on :9003 now, where the answer arrives
    beside its ranks, gold evidence and scores. Asserted by absence, like
    the repo's other retirements: a control that still exists is exactly
    how a removed feature comes back."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    for gone in ('id="question"', 'id="gtPick"', 'id="ask"', 'id="queryOut"'):
        assert gone not in html, f'the panel still carries {gone}'
    # the route itself stays: it is the lab's API, and the Inspector's followed
    # query view reads whatever runs through it
    assert 'api/queries' in (STATIC.parent / 'server.py').read_text(encoding='utf-8')


def test_the_panel_offers_the_mode_dropdown():
    """The dropdown reads the served modes rather than a local copy — a
    preset kept in a frontend is a preset that will drift."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    assert 'modes' in html


def test_the_panel_offers_retrieve_and_the_production_preset():
    """Both buttons the loop needs: run retrieval for the selected questions,
    and load the shipped assistant's settings in one click."""
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    assert 'id="retrieve-selected"' in html
    assert 'id="use-production"' in html


# --- the leaderboard's own panel ------------------------------------------

def test_the_panel_lists_every_experiment_beside_the_ranked_runs(client):
    """The leaderboard ranks judged runs and must keep doing exactly that — an
    index build has no decision score, and a row that cannot be ranked has no
    business in a numbered table. So the ledger is a second table beside it,
    listing everything that ran."""
    html = client.get('/').text
    js = client.get('/panel.js').text
    assert 'id="board"' in html, 'the ranked leaderboard stays'
    assert 'id="experiments"' in html
    assert '/api/experiments' in js


# The served panel's own markup.
def test_the_panel_ends_its_run_buttons_with_the_inspector(client):
    """The door to :9003 belongs at the end of the row you press to run
    something, not above it: it is where you go *after* an experiment, so it
    reads as the last step rather than a second heading."""
    html = client.get('/').text
    assert html.index('id="use-production"') < html.index('id="open-inspector"')
    anchor = html[html.index('id="open-inspector"') - 200:
                  html.index('id="open-inspector"') + 200]
    assert 'right' in anchor, 'the link sits at the far right of the row'


# --- sortable columns ------------------------------------------------------

# The served pages' own markup.
def test_both_lab_pages_share_one_column_sorter(client):
    """One file for both pages rather than a copy each, so "what does
    clicking a header do" has one answer instead of two that drift. The
    order it produces is unit tested in `tests/sorttable.test.js`; this
    pins that both pages actually load it."""
    from raglab.server import STATIC

    assert (STATIC / 'sorttable.js').exists()
    panel = client.get('/').text
    js = client.get('/panel.js').text
    assert 'sorttable.js' in panel
    # The two tables worth sorting, both marked at the point they are rendered.
    assert js.count('sortable') >= 2
    # The hardcoded arrow is gone from the leaderboard's header: an indicator
    # that cannot move is a lie the moment you sort by anything else, and the
    # column's role is stated in prose beside the table instead.
    assert 'ragas_decision ▼' not in panel
    assert 'ragas_decision ▼' not in js

    inspector = (STATIC / 'inspector.html').read_text(encoding='utf-8')
    assert 'sorttable.js' in inspector
    # `path` draws the three ranks as a shape and the same three numbers follow
    # it, so sorting on the picture would sort on nothing.
    assert 'data-nosort' in inspector


# The served pages' own markup, plus the routes on both services.
def test_both_lab_pages_share_one_token_sheet_and_one_script(client, monkeypatch):
    """tokens.css and lab.js follow the same pattern as sorttable.js: one file
    for both pages rather than a copy each, so a design token or a utility
    cannot drift apart on either page. This pins that both services actually
    route them, both pages actually load them, and each loads before the
    page's own stylesheet or script — a later link would lose the tokens to
    the page's own overrides instead of feeding them."""
    from fastapi.testclient import TestClient

    from raglab import inspector
    from raglab.config import LabSettings
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

    monkeypatch.setattr(
        inspector, 'load_lab_settings',
        lambda: LabSettings(openrouter_api_key='', llm_provider='fake'))
    insp_client = TestClient(inspector.create_inspector_app())
    inspector_html = insp_client.get('/').text
    insp_tokens = insp_client.get('/tokens.css')
    insp_lab = insp_client.get('/lab.js')
    assert insp_tokens.status_code == 200
    assert insp_tokens.headers['content-type'].startswith('text/css')
    assert insp_lab.status_code == 200
    assert insp_lab.headers['content-type'].startswith('application/javascript')
    assert (inspector_html.index('href="/tokens.css"')
            < inspector_html.index('href="/inspector.css"'))
    assert (inspector_html.index('src="/lab.js"')
            < inspector_html.index('src="/inspector.js"'))


# --- the panel does not forget across a reload -----------------------------

# The served panel's own markup.
def test_the_panel_keeps_its_experiment_and_its_settings_across_a_reload(client):
    """Refreshing the page must not throw away the grades card and the
    settings on screen. Both are remembered in localStorage and restored on
    boot — the last experiment by id, re-read from the service so the page
    never renders a stale copy of a run that has since been deleted."""
    html = client.get('/panel.js').text
    assert 'localStorage' in html
    assert 'lodestar:raglab-last-run' in html
    assert 'lodestar:raglab-config' in html
    # Re-read by id rather than stored whole: a run file can be deleted between
    # two visits, and a page rendering a copy of something that is gone is worse
    # than a page that has forgotten it.
    assert 'restoreLastRun' in html


def test_the_leaderboard_says_how_much_of_the_disk_it_shows(client):
    """A run can rank differently on a bounded page than over the whole
    directory, with nothing on screen explaining the disagreement — a
    bounded view has to say what it left out."""
    body = client.get('/api/evaluations?limit=3').json()
    assert len(body['runs']) <= 3
    # Served, not counted in the browser: the page cannot know how many files it
    # was not sent.
    assert body['total'] >= len(body['runs'])
    js = client.get('/panel.js').text
    assert '/api/evaluations?limit=' in js, 'the panel must ask for a stated limit'


# --- the project's own RAG settings, in one click --------------------------

# The panel's own source, against the served preset.
def test_the_panel_fills_the_projects_settings_from_the_served_preset(client):
    """The preset is served from `/api/options`, so a button claiming to be
    the real system reads one source rather than keeping its own copy —
    the same reason the mode dropdown is served. Settings only: a preset
    that also started a job would download a large encoder for someone who
    only wanted to see what the real system uses."""
    html = client.get('/').text
    panel = client.get('/panel.js').text

    assert 'OPTIONS.production' in panel

    # The preset's own label is served with it, so its presence in the frontend
    # would mean the frontend had a second copy of the preset to go stale.
    served = client.get('/api/options').json()['production']
    assert served['label'] == baseline.LABEL
    assert served['label'] not in panel, 'the panel keeps its own preset'

    # The button runs nothing.
    handler = panel[panel.index("$('use-production').onclick"):]
    handler = handler[:handler.index('\n};')]
    assert '/api/indexes' not in handler, 'the preset must not start a build'
    assert 'doRetrieve' not in handler, 'the preset must not start a retrieval'
    assert 'then run' not in panel and 'then run' not in html, (
        'the button no longer claims to run')


def test_the_preset_carries_the_fields_the_panel_cannot_show(client):
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


def test_the_panels_no_backend_hint_names_every_backend_that_would_fix_it():
    """A hint that lists some of the ways out is worse than one that lists
    none, because a reader takes it for the whole set — so this fails the
    day a backend is added and the sentence is not."""
    page = (RAGLAB_DIR / 'static' / 'panel.js').read_text(encoding='utf-8')
    hint = [line for line in page.splitlines() if 'no LLM backend' in line]
    assert hint, 'the panel must say what to do when no backend is reachable'
    for provider in config.LLM_PROVIDERS:
        # 'fake' is not a way out: it answers without failing, which is the
        # problem rather than the fix.
        if provider and provider != 'fake':
            assert provider in hint[0], provider


# --- the LLM widget: a window in the corner, not a stage ----------------

def test_the_widget_pops_up_from_the_lower_right_corner():
    """The launcher and its window are fixed to the page's lower-right
    corner; a widget that scrolls with the panel is a fourth card, not a
    widget."""
    html = (RAGLAB_DIR / 'static' / 'index.html').read_text(encoding='utf-8')
    for element in ('widget-launch', 'widget-window', 'widget-log',
                    'widget-input', 'widget-send'):
        assert f'id="{element}"' in html, element

    css = (RAGLAB_DIR / 'static' / 'panel.css').read_text(encoding='utf-8')
    corner = css[css.index('.widget'):]
    assert 'position: fixed' in corner
    assert 'right:' in corner and 'bottom:' in corner
    # The widget is not a pipeline stage, so it wears no step ink.
    assert '--step-' not in corner


def test_the_widgets_script_talks_to_the_widget_route():
    js = (RAGLAB_DIR / 'static' / 'panel.js').read_text(encoding='utf-8')
    assert "/api/widget" in js
    assert 'escapeHtml' in js[js.index('widget'):], (
        'replies are model output rendered into the page — they go through '
        'the shared escaper like every other untrusted string')
