# this is an end-to-end test
"""Everything on the panel that is not the experiment: the setup column and
the chrome around it.

The bench in the middle of the page is where a run is configured. This journey
drives the rest of it — the collapsing setup panel on the left, the settings
disc in the top bar, and the status rail at the foot — because every one of
those controls is a promise the lab makes about *this installation* rather than
about one run, and none of them can be checked without a real browser: the
panel's collapse is one attribute on `<html>` plus a stored string plus a media
query, the settings disc is a native popover, a file picker needs a real file,
and the export button is a synthetic `<a download>` that exists for one frame.

What it proves, in order: the setup panel folds to a rail and is remembered;
at a narrow width it becomes a drawer over the bench that a scrim click or
Escape closes; the settings disc reaches its three groups; the OpenRouter key
can be typed, saved and cleared, and never reaches a file; the backend,
embedder, embedding-model and model-role pickers are the server's lists rather
than the page's, are settable, and say NA for a model this installation cannot
serve; the dataset picker names every corpus and its two popovers explain the
selected one; a valid pair of files imports and rebuilds the declaration table,
an invalid one is refused with the contract's own words and imports nothing;
the two template links serve the templates; a recorded experiment survives an
export and an import unchanged; and the status rail reports the fake backend
and the in-memory index honestly.
"""
import json

import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser

import httpx  # noqa: E402  (after the skip guard)
from playwright.sync_api import expect  # noqa: E402  (after the skip guard)

from raglab.corpora import dataset_import_contract as datasets  # noqa: E402

#: An obviously fake credential, long enough to pass the length floor the key
#: store applies (20 characters) and shaped like nothing anybody ever issued.
FAKE_KEY = 'sk-or-v1-this-key-is-not-real-000000'

#: The two ids this file imports under — distinct from every bundled corpus,
#: so no journey can be reading one of those and believing it imported it. The
#: import lands in the suite's own temporary dataset directory (`lab_server`
#: redirects `RAGLAB_DATASETS`), so nothing here has to be swept up after.
IMPORTED_ID = 'browser-setup-pair'
REFUSED_ID = 'browser-setup-refused'


def _booted(page):
    """Wait for `boot()`. Every `<select>` on this page is empty in the markup
    and filled only once `GET /api/options` has answered, so the presence of a
    control proves nothing and the presence of an option proves the fetch."""
    expect(page.locator('#dataset option').first).to_be_attached()
    return page


def _options(page) -> dict:
    """What the page itself was handed — asked the way the page asks."""
    return page.evaluate("() => fetch('/api/options').then((r) => r.json())")


def _open_settings(page):
    """The disc's popover, opened once and left open."""
    if not page.locator('#app-settings-panel').is_visible():
        page.click('#app-settings')
    expect(page.locator('#app-settings-panel')).to_be_visible()


def _option_values(page, selector: str) -> list[str]:
    return page.eval_on_selector_all(
        f'{selector} option', '(nodes) => nodes.map((n) => n.value)')


def _option_text(page, selector: str, value: str) -> str:
    return page.eval_on_selector(
        selector, '(select, value) => Array.from(select.options)'
                  '.find((o) => o.value === value).textContent', value)


def _reachable_openrouter_models(served: dict) -> set:
    """The OpenRouter models this installation reports it can actually run.

    Availability there is `verified or served-by-the-live-list`, and the live
    list needs a reachable OpenRouter — which the suite deliberately does not
    have (`OPENROUTER_BASE_URL` is a closed local port). So this is the set a
    key cannot change, and comparing it across a save is what shows the panel
    reporting rather than assuming.
    """
    mode = next((m for m in served['modes'] if m['key'] == 'openrouter'), None)
    models = (mode or {}).get('models') or served['models']
    return {model['id'] for model in models if model['available']}


def _catalogue_on_offer(page, served: dict) -> list[dict]:
    """The chat models the model-role dropdowns are actually offering.

    Not `served['models']`: that is the boot backend's catalogue, and a first
    visit picks a backend of its own (the codex CLI needs no key), so the list
    on screen is that mode's. Read the picker rather than assume either.
    """
    picked = page.input_value('#mode')
    mode = next((m for m in served['modes'] if m['key'] == picked), None)
    return (mode or {}).get('models') or served['models']


def _dataset_pair(dataset_id: str = IMPORTED_ID, corpus_ref: str = ''):
    """The smallest pair that satisfies the import contract, plus one declared
    label — the label is the point, because it is what the declaration table on
    the page is built from and so what proves the table was rebuilt."""
    first = ('The archive room was repainted on Tuesday and the shelves went '
             'back in on Friday.')
    second = ('A second reader asked for the ledger and it was handed over the '
              'same afternoon.')
    corpus = {
        'corpus_dataset_metadata': {
            'dataset': dataset_id,
            'name': 'Browser setup pair',
            'language': 'en',
            'description': 'Two documents, imported through the panel by a '
                           'browser test.',
            'label_fields': {
                'channel': {'type': 'string', 'values': ['note', 'letter'],
                            'description': 'Where the document came from.',
                            'applies_to': ['document', 'chunk', 'summary']},
            },
        },
        'corpus_documents': [
            {'corpus_document_id': 1,
             'document_content': [{'text': first}],
             'document_metadata': {'channel': 'note'}},
            {'corpus_document_id': 2,
             'document_content': [{'text': second}],
             'document_metadata': {'channel': 'letter'}},
        ],
    }
    def question(number, asked, answered, quoted):
        """One question citing one document, quoting it verbatim — the shape
        the cross-file rules check, `part_labels` included."""
        return {
            'groundtruth_question_id': number,
            'question': asked,
            'expected_answer': {'behavior': 'answer', 'text': answered},
            'relevant_corpus_documents': [
                {'corpus_document_id': number,
                 'evidence': [{'text': quoted, 'fidelity': 'verbatim',
                               'part_labels': [{}]}]}],
        }

    ground_truth = {
        'groundtruth_dataset_metadata': {
            'name': 'Browser setup pair questions',
            # The pair join: an id, never a filename. Pointing it elsewhere is
            # exactly how the refusal journey below breaks a pair.
            'corpus_ref': {'dataset': corpus_ref or dataset_id},
        },
        'groundtruth_dataset': [
            question(1, 'When were the shelves put back?', 'On Friday.', first),
            question(2, 'Who asked for the ledger?', 'A second reader.', second),
        ],
    }
    return corpus, ground_truth


def _write_pair(directory, dataset_id: str, corpus_ref: str = ''):
    """The pair on disk, under the two filenames the contract expects, so the
    browser is handed real files rather than a synthetic buffer."""
    corpus, ground_truth = _dataset_pair(dataset_id, corpus_ref)
    corpus_path = directory / f'{dataset_id}_corpus.json'
    truth_path = directory / f'{dataset_id}_groundtruth.json'
    corpus_path.write_text(json.dumps(corpus), encoding='utf-8')
    truth_path.write_text(json.dumps(ground_truth), encoding='utf-8')
    return corpus_path, truth_path


def test_the_setup_panel_folds_to_a_rail_and_is_remembered_across_a_reload(panel):
    _booted(panel)
    assert panel.evaluate('() => document.documentElement.dataset.sidebar') == 'open'
    expect(panel.locator('#sidebar-body')).to_be_visible()
    expect(panel.locator('#sidebar-toggle')).to_have_attribute('aria-expanded', 'true')

    panel.click('#sidebar-toggle')
    expect(panel.locator('#sidebar-body')).to_be_hidden()
    expect(panel.locator('#sidebar-toggle')).to_have_attribute('aria-expanded', 'false')
    # A bare string, not JSON: the inline script in the head reads it too.
    assert panel.evaluate("() => localStorage.getItem('raglab-sidebar')") == 'collapsed'

    panel.reload()
    _booted(panel)
    expect(panel.locator('#sidebar-body')).to_be_hidden()
    expect(panel.locator('#sidebar-toggle')).to_have_attribute('aria-expanded', 'false')

    panel.click('#sidebar-toggle')
    expect(panel.locator('#sidebar-body')).to_be_visible()
    panel.reload()
    _booted(panel)
    expect(panel.locator('#sidebar-body')).to_be_visible()


def test_at_a_narrow_width_the_setup_panel_is_a_drawer_a_scrim_or_escape_closes(
        panel):
    # Below the 1024px breakpoint the panel stops being a column and becomes an
    # overlay; the head's inline script shuts it on a narrow first paint
    # regardless of what was stored, so the reload is the point of the reload.
    panel.set_viewport_size({'width': 820, 'height': 900})
    panel.reload()
    _booted(panel)
    expect(panel.locator('#sidebar-body')).to_be_hidden()
    expect(panel.locator('#sidebar-scrim')).to_be_hidden()

    panel.click('#sidebar-toggle')
    expect(panel.locator('#sidebar-body')).to_be_visible()
    expect(panel.locator('#sidebar-scrim')).to_be_visible()

    panel.keyboard.press('Escape')
    expect(panel.locator('#sidebar-body')).to_be_hidden()
    expect(panel.locator('#sidebar-scrim')).to_be_hidden()

    panel.click('#sidebar-toggle')
    expect(panel.locator('#sidebar-scrim')).to_be_visible()
    panel.click('#sidebar-scrim')
    expect(panel.locator('#sidebar-body')).to_be_hidden()

    # Back at a wide viewport Escape is not a dismissal: nothing is covering
    # the panel there, and taking it away would answer a keypress meant for a
    # popover. The reload comes up collapsed because the scrim click above was
    # a choice and the choice is remembered — so the panel is opened first, and
    # what Escape must not do is close it again.
    panel.set_viewport_size({'width': 1400, 'height': 900})
    panel.reload()
    _booted(panel)
    panel.click('#sidebar-toggle')
    expect(panel.locator('#sidebar-body')).to_be_visible()
    panel.keyboard.press('Escape')
    expect(panel.locator('#sidebar-body')).to_be_visible()


def test_the_settings_disc_opens_and_all_three_of_its_groups_are_reachable(panel):
    _booted(panel)
    expect(panel.locator('#app-settings-panel')).to_be_hidden()
    _open_settings(panel)

    popover = panel.locator('#app-settings-panel')
    expect(popover.get_by_role('heading', name='Theme')).to_be_visible()
    expect(popover.get_by_role('heading', name='Experiment archive')).to_be_visible()
    expect(popover.get_by_role('heading', name='OpenRouter')).to_be_visible()
    # The theme radios are written by lab.js on DOMContentLoaded rather than by
    # boot(), so their presence is a second readiness signal on one page. What
    # they *do* is test_browser_theme.py's claim; that they are reachable from
    # here is this one's.
    for radio in ('#theme-day', '#theme-night', '#theme-auto'):
        expect(panel.locator(radio)).to_be_attached()
    expect(panel.locator('#archive-import')).to_be_visible()
    expect(panel.locator('#archive-export')).to_be_visible()
    expect(panel.locator('#openrouter_key')).to_be_visible()
    expect(panel.locator('#save-key')).to_be_visible()
    expect(panel.locator('#clear-key')).to_be_visible()
    expect(panel.locator('#keyState')).not_to_be_empty()


def test_the_openrouter_key_can_be_saved_and_cleared_and_never_reaches_a_file(
        panel, lab_home):
    _booted(panel)
    _open_settings(panel)
    # Nothing has been typed, so there is nothing this panel could take away.
    expect(panel.locator('#keyState')).to_contain_text('No key')
    expect(panel.locator('#clear-key')).to_be_disabled()
    reachable_before = _reachable_openrouter_models(_options(panel))

    panel.fill('#openrouter_key', FAKE_KEY)
    assert panel.input_value('#openrouter_key') == FAKE_KEY
    panel.click('#save-key')
    # Held for this process and named as such; the field empties itself, because
    # a password box holding a credential is a credential on screen.
    expect(panel.locator('#keyState')).to_contain_text('Key set')
    expect(panel.locator('#keyState')).to_contain_text('forgotten when it stops')
    expect(panel.locator('#openrouter_key')).to_have_value('')
    expect(panel.locator('#clear-key')).to_be_enabled()

    # A reload settles the catalogue refresh the save kicked off, and asks the
    # better question at the same time: the key is held by the lab process, not
    # by this page, so a page that has forgotten everything still finds it set.
    panel.reload()
    _booted(panel)
    _open_settings(panel)
    expect(panel.locator('#keyState')).to_contain_text('Key set')

    # Saving a key re-reads every catalogue, and this is where a panel could
    # start guessing: a key is permission to ask, never an answer. OpenRouter
    # is unreachable from this suite, so the probe comes back empty and the
    # models this lab has not verified for itself stay NA — exactly the set
    # that was reachable before the key was typed.
    panel.select_option('#mode', 'openrouter')
    served = _options(panel)
    assert _reachable_openrouter_models(served) == reachable_before
    catalogue = _catalogue_on_offer(panel, served)
    unavailable = [m for m in catalogue if not m['available']]
    assert unavailable, 'nothing was NA, so this proves nothing about guessing'
    role = served['model_roles'][0]['key']
    for model in unavailable:
        text = _option_text(panel, f'select.rag-model[data-role="{role}"]',
                            model['id'])
        assert 'NA' in text, text

    panel.click('#clear-key')
    expect(panel.locator('#keyState')).to_contain_text('No key')
    expect(panel.locator('#clear-key')).to_be_disabled()

    # The rule the panel states in its own label: no file, no log, no artifact.
    # `lab_home` is every durable path this lab has — the ledger, the widget
    # log, the corpus store, the imported datasets and the run files all live
    # under it.
    for path in sorted(lab_home.rglob('*')):
        if path.is_file():
            assert FAKE_KEY.encode() not in path.read_bytes(), path


def test_the_backend_embedder_and_embedding_model_pickers_are_the_servers_lists(
        panel):
    _booted(panel)
    served = _options(panel)

    # The first backend option is the lab's own boot provider, spelled with an
    # empty value; the rest are the served modes, in order.
    assert _option_values(panel, '#mode') == [''] + [m['key'] for m in served['modes']]
    expect(panel.locator('#mode option').first).to_contain_text(
        served['capabilities']['llm_provider'])
    assert _option_values(panel, '#embedder') == [
        hint['kind'] for hint in served['embedder_hints']]
    assert _option_values(panel, '#embed_model') == [
        model['id'] for model in served['embed_models']]

    # Settable, and the backend choice is remembered under the board's prefix.
    picked = served['modes'][0]['key']
    panel.select_option('#mode', picked)
    expect(panel.locator('#mode')).to_have_value(picked)
    assert panel.evaluate("() => localStorage.getItem('raglab:mode')") == f'"{picked}"'

    panel.select_option('#embedder', 'sentence-transformers')
    expect(panel.locator('#embedder')).to_have_value('sentence-transformers')
    model_id = next(m['id'] for m in served['embed_models'] if m['id'])
    panel.select_option('#embed_model', model_id)
    expect(panel.locator('#embed_model')).to_have_value(model_id)

    # NA means "this installation cannot load it" — never a silent drop from
    # the list, and never a label on something that *is* available.
    for model in served['embed_models']:
        text = _option_text(panel, '#embed_model', model['id'])
        assert ('NA' in text) is (not model['available']), text
    for hint in served['embedder_hints']:
        text = _option_text(panel, '#embedder', hint['kind'])
        assert ('NA' in text) is (hint.get('available') is False), text


def test_the_model_role_dropdowns_are_the_ones_the_server_names(panel):
    _booted(panel)
    served = _options(panel)
    roles = served['model_roles']
    assert roles, 'a lab with no model roles has nothing to prove here'

    selects = panel.locator('select.rag-model')
    expect(selects).to_have_count(len(roles))
    for role in roles:
        control = panel.locator(f'select.rag-model[data-role="{role["key"]}"]')
        expect(control).to_have_count(1)
        expect(control).to_have_attribute('data-field', role['field'])
        # Every role wears its own explainer trigger, which is where the
        # condition it runs under is recorded.
        expect(panel.locator(
            f'.why-term[data-topic="model.{role["key"]}"]')).to_have_count(1)

    # A slug only means something to the backend that serves it, so the list
    # in these dropdowns is the *picked* backend's catalogue — and a first
    # visit picks one (the codex CLI, which needs no key) rather than none.
    catalogue = _catalogue_on_offer(panel, served)
    chosen = catalogue[-1]['id']
    settable = 0
    for role in roles:
        control = panel.locator(f'select.rag-model[data-role="{role["key"]}"]')
        if control.is_disabled():
            # Inert is a served decision, not an accident: the knob dependency
            # table has to name this field for the greying to be legitimate.
            assert role['field'] in served['dependencies'], role['field']
            continue
        control.select_option(chosen)
        expect(control).to_have_value(chosen)
        settable += 1
    assert settable, 'every model role was inert; nothing was actually set'

    # A model the active backend does not serve is offered and marked, never
    # dropped and never offered as if it would run.
    for model in catalogue:
        text = _option_text(
            panel, f'select.rag-model[data-role="{roles[0]["key"]}"]', model['id'])
        assert ('NA' in text) is (not model['available']), text


def test_the_dataset_picker_names_every_corpus_and_the_popovers_explain_it(panel):
    _booted(panel)
    served = _options(panel)
    # The built-in diary is offered as '' — the value every recorded run
    # carries — so the picker's values are not simply the ids.
    expected = {'' if d['id'] == 'diary-fa' else d['id'] for d in served['datasets']}
    assert set(_option_values(panel, '#dataset')) == expected
    expect(panel.locator('#dataset optgroup[label="Bundled"]')).to_have_count(1)
    for dataset in served['datasets']:
        value = '' if dataset['id'] == 'diary-fa' else dataset['id']
        assert dataset['id'] in _option_text(panel, '#dataset', value)

    panel.select_option('#dataset', 'smoke-mini')
    expect(panel.locator('#corpusName')).to_have_text('smoke-mini')

    panel.click('#corpusScope')
    expect(panel.locator('#corpus-detail')).to_be_visible()
    expect(panel.locator('#corpus')).to_contain_text('documents')
    expect(panel.locator('#corpus')).to_contain_text('questions')

    panel.click('#dataset-detail-open')
    expect(panel.locator('#dataset-detail')).to_be_visible()
    expect(panel.locator('#dataset-detail-title')).not_to_be_empty()
    expect(panel.locator('#dataset-detail-description')).not_to_be_empty()
    expect(panel.locator('#dataset-detail-census')).to_contain_text('documents')
    # The declaration table is the corpus's own: these two labels are what the
    # smoke set declares, and nothing here hardcodes a vocabulary.
    labels = panel.locator('#dataset-detail-labels table')
    expect(labels).to_contain_text('topics')
    expect(labels).to_contain_text('role')


def test_a_valid_dataset_pair_imports_and_rebuilds_the_declaration_table(
        panel, tmp_path):
    _booted(panel)
    panel.select_option('#dataset', 'smoke-mini')
    panel.click('#corpusScope')
    panel.click('#dataset-detail-open')
    expect(panel.locator('#dataset-detail-labels table')).to_contain_text('topics')
    panel.keyboard.press('Escape')

    corpus_path, truth_path = _write_pair(tmp_path, IMPORTED_ID)
    panel.set_input_files('#dataset-corpus-file', str(corpus_path))
    panel.set_input_files('#dataset-groundtruth-file', str(truth_path))
    panel.click('#dataset-import')

    expect(panel.locator('#importInfo')).to_contain_text('Browser setup pair')
    expect(panel.locator('#importInfo')).to_contain_text('2 documents, 2 questions')
    # The picker now offers it, under the group that says where it came from,
    # and the page has switched to it.
    expect(panel.locator('#dataset')).to_have_value(IMPORTED_ID)
    expect(panel.locator(
        f'#dataset optgroup[label="Imported"] option[value="{IMPORTED_ID}"]'
    )).to_have_count(1)
    expect(panel.locator('#corpusName')).to_have_text(IMPORTED_ID)

    panel.click('#corpusScope')
    expect(panel.locator('#corpus')).to_contain_text('2 documents')
    panel.click('#dataset-detail-open')
    labels = panel.locator('#dataset-detail-labels table')
    expect(labels).to_contain_text('channel')
    expect(labels).not_to_contain_text('topics')


def test_an_invalid_dataset_pair_is_refused_in_the_contracts_own_words(
        panel, tmp_path):
    _booted(panel)
    before = _option_values(panel, '#dataset')

    # One file is not a dataset, and the page says so before it asks the lab.
    corpus_path, truth_path = _write_pair(
        tmp_path, REFUSED_ID, corpus_ref='not-this-corpus')
    panel.set_input_files('#dataset-corpus-file', str(corpus_path))
    panel.click('#dataset-import')
    expect(panel.locator('#importInfo')).to_contain_text('a dataset is two files')

    panel.set_input_files('#dataset-corpus-file', str(corpus_path))
    panel.set_input_files('#dataset-groundtruth-file', str(truth_path))
    panel.click('#dataset-import')
    # The route's own message, which is the contract's own message: the panel
    # forwards it rather than inventing a friendlier one.
    expect(panel.locator('#importInfo .note')).to_contain_text(
        'corpus_ref.dataset does not match the corpus it is paired with')
    expect(panel.locator('#importInfo .note')).to_contain_text(
        'the pair join is broken')

    # And nothing arrived: not in the picker, and not in what the lab serves.
    assert _option_values(panel, '#dataset') == before
    served = _options(panel)
    assert REFUSED_ID not in {d['id'] for d in served['datasets']}


def test_the_two_dataset_template_links_serve_the_templates(panel):
    _booted(panel)
    for name, filename in (('corpus', 'corpus_template.json'),
                           ('ground truth', 'groundtruth_template.json')):
        link = panel.locator('.template-links a', has_text=name)
        with panel.expect_download() as download:
            link.click()
        served = download.value
        assert served.suggested_filename == filename
        with open(served.path(), encoding='utf-8') as handle:
            offered = json.load(handle)
        # Read from the fixture every time on the way out, so the file the link
        # hands a reader and the file the schema pins are one file.
        expected = json.loads(
            (datasets.BUNDLED_DIR / filename).read_text(encoding='utf-8'))
        assert offered == expected


def test_a_recorded_experiment_survives_an_export_and_an_import_unchanged(
        panel, lab_server, a_recorded_experiment, tmp_path):
    _booted(panel)
    _open_settings(panel)

    # 1. With no readings on screen there is no evidence to carry, and the page
    #    says exactly that rather than exporting a file that implies otherwise.
    with panel.expect_download() as first:
        panel.click('#archive-export')
    settings_only = first.value
    assert settings_only.suggested_filename == 'raglab-experiment.json'
    with open(settings_only.path(), encoding='utf-8') as handle:
        exported = json.load(handle)
    assert exported['format'] == 'raglab-experiment'
    assert 'evaluation' not in exported
    expect(panel.locator('#archive-status')).to_be_visible()
    expect(panel.locator('#archive-status')).to_contain_text(
        'Settings-only experiment exported')
    expect(panel.locator('#archive-status')).to_have_class(
        'banner archive-status success')

    # 2. The recorded experiment as the export button would have written it —
    #    the same object the board's open button hands this page.
    archive = httpx.get(
        f'{lab_server}/api/experiments/{a_recorded_experiment}/archive',
        timeout=30.0).json()
    archive_path = tmp_path / 'recorded-experiment.json'
    archive_path.write_text(json.dumps(archive), encoding='utf-8')

    panel.set_input_files('#archive-file', str(archive_path))
    expect(panel.locator('#archive-status')).to_contain_text(
        'Private evidence included')
    # It is an experiment on this page now: the readings card is filled, and
    # names the row as one this page did not run.
    expect(panel.locator('#resultMeta')).to_contain_text('imported · read-only')
    expect(panel.locator('#resultBody')).to_be_visible()
    active = httpx.get(f'{lab_server}/api/imported-archives/active',
                       timeout=10.0).json()
    assert active['archive_id'] == a_recorded_experiment
    listed = httpx.get(f'{lab_server}/api/experiments', timeout=10.0).json()
    assert a_recorded_experiment in {row['experiment_id']
                                     for row in listed['experiments']}

    # 3. Exported again, it is the same file — evidence included, nothing
    #    reshaped on the way through the page.
    _open_settings(panel)
    with panel.expect_download() as second:
        panel.click('#archive-export')
    with open(second.value.path(), encoding='utf-8') as handle:
        round_tripped = json.load(handle)
    assert round_tripped == archive


def test_the_status_rail_reports_this_installation_honestly(panel, lab_home):
    _booted(panel)
    served = _options(panel)
    caps = served['capabilities']

    expect(panel.locator('#spineCaption')).to_have_text('nothing running')
    expect(panel.locator('#chromeProgress')).to_have_attribute('data-step', '')

    expect(panel.locator('#status-detail')).to_be_hidden()
    panel.click('#statusPill')
    expect(panel.locator('#status-detail')).to_be_visible()

    chips = panel.locator('#caps .chip')
    expect(chips).to_have_count(6)
    texts = chips.all_text_contents()
    # The fake backend is not an LLM backend, and the chip refuses to pretend
    # otherwise — it says so and carries the `off` class that colours it.
    assert caps['llm_provider'] == 'fake', caps['llm_provider']
    assert caps['llm'] is False
    llm_chip = panel.locator('#caps .chip', has_text='no LLM backend')
    expect(llm_chip).to_have_count(1)
    expect(llm_chip).to_have_class('chip off')
    # No vector database anywhere: the index is process memory, and what is
    # durable is named instead — here, the suite's own temporary lab.
    storage_chip = panel.locator('#caps .chip', has_text='index in memory')
    expect(storage_chip).to_have_count(1)
    expect(storage_chip).to_contain_text(str(lab_home))
    assert any(text.startswith('ragas ') for text in texts), texts

    # The pill takes the worst of the five checks and puts the count in words,
    # so its meaning never rests on the dot's colour alone.
    checks = [caps['fastembed'], caps['cross_encoder'], caps['llm'],
              caps['ragas']['installed'], caps['ragas']['llm_ready']]
    missing = len([ok for ok in checks if not ok])
    expect(panel.locator('#statusPillText')).to_have_text(
        f'{missing} of 5 not ready' if missing else 'all systems ready')


def test_the_top_bar_offers_the_three_surfaces_and_a_way_past_them(
        panel, lab_server):
    _booted(panel)
    expect(panel.locator('a.brand')).to_have_attribute('href', '/')
    expect(panel.locator('.topnav a[href="/"]')).to_have_attribute(
        'aria-current', 'page')
    expect(panel.locator('.topnav a[href="/inspector"]')).to_have_count(1)
    expect(panel.locator('.topnav a[href="/leaderboard"]')).to_have_count(1)

    # The skip link is the first stop on a keyboard, and it has to land on the
    # controls rather than merely scroll near them.
    panel.keyboard.press('Tab')
    expect(panel.locator('a.skip-link')).to_be_focused()
    panel.keyboard.press('Enter')
    assert panel.evaluate('() => location.hash') == '#main'
    expect(panel.locator('#main')).to_be_visible()

    panel.click('.topnav a[href="/leaderboard"]')
    expect(panel).to_have_url(f'{lab_server}/leaderboard')
