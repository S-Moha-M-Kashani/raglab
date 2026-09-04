// tests/panel_open.test.js — the board's open arrow, over a real pre-branch
// experiment.
//
// Contract under test: `boot()` -> `takeHandedExperiment()` ->
// `handOver()` -> `openHandedExperiment()` in `dashboard/frontend/panel.js`
// — the exact function panel.js runs on its own when the page loads with
// `?experiment=<id>` in the URL, which is what the leaderboard's `↗` open
// arrow navigates in the same tab. Round 1 and round 2's unit tests exercised
// `ExperimentHandoff.reconcile()`/`reconcileUi()` directly; this is the first
// test that drives the real page-load entry point end to end, over a real
// archive fetched from a running server before this branch's schema existed
// (`pre_branch_archive_fixture.json`, trimmed only of bulk per-row/per-chunk
// evidence — `settings.config`/`settings.ui` are byte-for-byte the server's
// own response) and a real `/api/options` response
// (`pre_branch_options_fixture.json`, untrimmed — `boot()` reads more of it
// than the reconciliation alone needs, and a trimmed fixture that happened
// to omit a field `boot()` touches would make this test pass by accident).
//
// `boot()` runs automatically the moment `panel.js` loads (it is the last
// line of the file), so this harness does not call anything by hand — it
// sets up the smallest DOM and network double the real boot sequence
// touches, and waits for the promise chain `boot()` starts to settle.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (name) => readFileSync(join(HERE, '../frontend', name), 'utf8');
const SOURCE = `${read('lab.js')}\n${read('archive_io.js')}\n`
  + `${read('experiment_handoff.js')}\n${read('panel.js')}`;

const ARCHIVE = JSON.parse(
  readFileSync(join(HERE, 'pre_branch_archive_fixture.json'), 'utf8'));
const REAL_OPTIONS = JSON.parse(
  readFileSync(join(HERE, 'pre_branch_options_fixture.json'), 'utf8'));
const EXPERIMENT_ID = ARCHIVE.evaluation.result.run_id;

// A second real, pre-branch archive — fetched the same way, from a run
// against a *served, non-empty* dataset id rather than the built-in diary's
// `''`. The live re-check that found the `''` gap named this exact row by
// name first, on the reasonable guess that `''`/`'meetings-de'` might be the
// same fact about every pre-branch row; fetching both shows they are not —
// this repository's own pre-branch rows split across both shapes, so a fix
// for one must not regress the other.
const MEETINGS_ARCHIVE = JSON.parse(
  readFileSync(join(HERE, 'pre_branch_archive_meetings_fixture.json'), 'utf8'));
const MEETINGS_EXPERIMENT_ID = MEETINGS_ARCHIVE.evaluation.result.run_id;

// The third case has no real row to fetch — no experiment here has ever run
// against a dataset this installation does not serve — so it is the one
// fixture built by hand, from the real archive's own shape, changed in
// exactly the one field the case is about.
const UNKNOWN_DATASET_ID = 'retired-corpus-xyz';
function unknownDatasetArchive() {
  const archive = JSON.parse(JSON.stringify(ARCHIVE));
  archive.settings.config.index.dataset = UNKNOWN_DATASET_ID;
  archive.evaluation.result.dataset = UNKNOWN_DATASET_ID;
  archive.evaluation.result.run_id = 'unknown-dataset-run-001';
  return archive;
}

const copy = (value) => JSON.parse(JSON.stringify(value));

// The confirming half of "fetched, not hand-written": the fixture must still
// carry the exact defect this round fixes — a retired key and a missing
// renamed one, in the shape the server actually served it.
const RECORDED_GENERATION = ARCHIVE.settings.config.generation;

// --- the smallest DOM `boot()` runs against ---------------------------------
// One generic stub for every element, the way `record_mode.test.js`'s does:
// `boot()` reaches many ids this test never asserts on (chunker selects,
// hierarchy controls, the declaration table, …), and none of them need to be
// anything more specific than "a box that will not throw when touched."
function element(name = '') {
  const held = new Map();
  const self = {
    name, innerHTML: '', textContent: '', hidden: false, disabled: false,
    dir: '', className: '', type: '', id: '', value: '', checked: false,
    min: '', max: '', tabIndex: 0, dataset: {}, style: {}, children: [],
    options: [], tHead: null, tBodies: [],
    classList: { add() {}, remove() {}, contains: () => false, toggle() {} },
    attributes: {},
    setAttribute(key, value) { self.attributes[key] = String(value); },
    getAttribute: (key) => (key in self.attributes ? self.attributes[key] : null),
    hasAttribute: (key) => key in self.attributes,
    addEventListener() {}, removeEventListener() {},
    appendChild(child) { self.children.push(child); return child; },
    insertAdjacentHTML() {}, remove() {}, focus() {}, scrollIntoView() {},
    querySelectorAll: () => [],
    closest: () => null,
    getBoundingClientRect: () => ({ top: 0, bottom: 0, left: 0, right: 0 }),
    cloneNode: () => element(`${name}-clone`),
    querySelector(selector) {
      if (!held.has(selector)) held.set(selector, element(`${name}${selector}`));
      return held.get(selector);
    },
  };
  self.content = { get firstElementChild() { return self.querySelector('table'); } };
  return self;
}

// The six `.rag-model` selects `readShownConfig()` reads by class rather than
// by id — one per model role `/api/options` actually declares, read off the
// fixture rather than hard-coded, so a role added later fails this test
// instead of silently going unmodelled.
function modelRoleElements() {
  return REAL_OPTIONS.model_roles.map((role) => {
    const el = element(`model-${role.key}`);
    el.className = 'rag-model';
    el.dataset = { field: role.field };
    el.value = '';
    return el;
  });
}

// The one control whose options a real page declares statically in
// `panel.html` rather than filling by script (`ragas_mode`; `mode` is built
// by JS from `/api/options`, and `reconcileUi` reads it off `OPTIONS.modes`
// directly for exactly that reason; `labels`/`balance` have no controls at
// all — panel.js carries them in `QUESTION_SELECTION`) — seeded here to
// match that markup, since this harness loads no HTML at all.
function seedStaticControls(byIdOf) {
  const ragasMode = byIdOf('ragas_mode');
  ragasMode.value = 'offline';
  ragasMode.options = ['offline', 'llm', 'off'].map((value) => ({ value }));
  byIdOf('limit').value = '0';
  byIdOf('ragas_limit').value = '0';
}

function panelPage({ search = '', initialSavedConfig = null,
                    initialSavedMode = null,
                    archives = { [EXPERIMENT_ID]: ARCHIVE } } = {}) {
  const byId = new Map();
  const byIdOf = (id) => {
    if (!byId.has(id)) byId.set(id, element(id));
    return byId.get(id);
  };
  seedStaticControls(byIdOf);
  const ragModels = modelRoleElements();
  const requests = [];
  const inspectorLink = element('inspector-link');
  inspectorLink.href = '/inspector';
  // Seeded *before* the script below ever runs, so a saved config already
  // sits in storage the moment the automatic `boot()` call at the foot of
  // panel.js starts — the only way to exercise `startingConfig()`'s own read
  // of it, as opposed to `openHandedExperiment`'s.
  const storage = new Map();
  if (initialSavedConfig) {
    storage.set('raglab:config', JSON.stringify(initialSavedConfig));
  }
  // A remembered backend ('' is the lab-boot one). With nothing saved at all,
  // boot() treats the page as a first visit and applies DEFAULT_MODE's own
  // preset before anything else happens — a test whose claim needs the panel
  // at the *served* defaults seeds this to opt out of that preset.
  if (initialSavedMode !== null) {
    storage.set('raglab:mode', JSON.stringify(initialSavedMode));
  }

  const sandbox = {
    console,
    URLSearchParams,
    setTimeout() {},
    localStorage: {
      getItem: (key) => (storage.has(key) ? storage.get(key) : null),
      setItem: (key, value) => storage.set(key, String(value)),
      removeItem: (key) => storage.delete(key),
    },
    document: {
      documentElement: element('html'),
      activeElement: null,
      getElementById: byIdOf,
      createElement: (tag) => element(`<${tag}>`),
      querySelector: (selector) => selector === '.topnav a[href^="/inspector"]'
        ? inspectorLink : element('querySelector'),
      querySelectorAll: (selector) => (selector === '.rag-model' ? ragModels : []),
      addEventListener() {},
    },
    window: {
      innerHeight: 1000, innerWidth: 1400,
      location: { search, href: `http://localhost:9002/${search}` },
      history: { replaceState() {} },
      addEventListener() {},
      showRun: undefined,
    },
    getComputedStyle: () => ({ position: 'static' }),
    // The widget is a helper outside the measured seam (CLAUDE.md) and shares
    // no state with the reconciliation under test; a bare recorder is enough
    // to assert on what the page told it.
    Widget: { note(message) { requests.notes.push(message); }, about() {}, offer() {} },
    SortTable: { make() {} },
    fetch: (path) => {
      requests.push(path);
      if (path === '/api/options') {
        return Promise.resolve({ ok: true, status: 200, statusText: 'OK',
          json: () => Promise.resolve(copy(REAL_OPTIONS)) });
      }
      // Boot asks the job table for a job already running (reattachRunningJob);
      // this harness models an idle installation, which answers an empty list.
      if (path === '/api/jobs') {
        return Promise.resolve({ ok: true, status: 200, statusText: 'OK',
          json: () => Promise.resolve([]) });
      }
      for (const [id, archive] of Object.entries(archives)) {
        if (path === `/api/experiments/${encodeURIComponent(id)}/archive`) {
          return Promise.resolve({ ok: true, status: 200, statusText: 'OK',
            json: () => Promise.resolve(copy(archive)) });
        }
      }
      return Promise.resolve({ ok: false, status: 404, statusText: 'Not Found',
        json: () => Promise.resolve({ detail: `no route registered for ${path}` }) });
    },
  };
  requests.notes = [];
  sandbox.window.document = sandbox.document;
  runInNewContext(SOURCE, sandbox);
  // `boot()` runs on load and its own promise chain — options, then the
  // handed-over experiment — needs several microtask turns to settle, the
  // same wait `record_mode.test.js`'s harness uses.
  const settled = (async () => { for (let i = 0; i < 40; i += 1) await null; })();
  return { sandbox, byId: byIdOf, requests, inspectorLink, settled };
}

test('the fixture still carries the defect this round fixes', () => {
  // this is a unit test
  assert.equal('key_facts_judge' in RECORDED_GENERATION, true);
  assert.equal('fact_judge' in RECORDED_GENERATION, false);
  assert.equal(RECORDED_GENERATION.key_facts_judge, true);
});

test('loading ?experiment=<pre-branch-id> opens it — the open proceeds, the '
  + 'retired key is dropped and named, and every servable knob is adopted',
  async () => {
    // The remembered lab-boot backend, not a bare first visit: the first-visit
    // codex preset itself sets `fact_judge: true` — the same value the record's
    // retired `key_facts_judge` carries — which would erase the distinction the
    // fact_judge assertion below exists to read. A remembered backend keeps the
    // panel at the served defaults, where the two values still differ.
    const page = panelPage({ search: `?experiment=${EXPERIMENT_ID}`,
                             initialSavedMode: '' });
    await page.settled;

    // The open proceeded: the fetch for this exact archive happened, and no
    // uncaught rejection ever reached `handOver`'s own failure note (which
    // this fixture regressed to before this round's fix — "Opening
    // experiment ... failed (... unexpected keys key_facts_judge)").
    assert.ok(page.requests.includes(
      `/api/experiments/${encodeURIComponent(EXPERIMENT_ID)}/archive`));
    const failed = page.requests.notes.filter((note) => /^Opening experiment/.test(note));
    assert.deepEqual(failed, [], 'the outer handOver failure note must never fire');
    const refused = page.requests.notes.filter((note) => /was not opened/.test(note));
    assert.deepEqual(refused, [], 'the open must not refuse wholesale');

    // Every servable knob applied: the recorded answerer, a value this
    // installation's `openrouter`/`codex`/... catalogues do serve, landed on
    // the control that runs the next evaluation.
    assert.equal(page.byId('answerer').value, RECORDED_GENERATION.answerer);

    // The retired key never reaches the config this lab is about to run —
    // dropped, not translated: `fact_judge` (the field it was renamed to)
    // stays at the panel's own value, never the recorded `true`, because the
    // record predates that field and never named it at all.
    assert.equal(page.byId('fact_judge').checked, false,
      'fact_judge must stay at the panel’s own value, never the recorded '
      + 'key_facts_judge value it was never actually written under');

    // And the drop is named, in the one sentence a reader watching the
    // widget would see — not merely counted, not merely swallowed.
    const opened = page.requests.notes.find((note) => /^Laboratory settings are now/.test(note));
    assert.ok(opened, 'the widget must say the settings are now this experiment’s');
    assert.match(opened, /key_facts_judge = true — not a knob this lab reads any more/);
    assert.match(opened, /The Inspector link above now leads to this experiment\.$/,
      'after a successful open, the widget must name the Inspector destination');

    // Nothing here was reinterpreted onto the running config either: the
    // in-memory config this page would submit next carries no trace of the
    // retired name.
    assert.equal('key_facts_judge' in page.sandbox.readConfig().generation, false);
  });

test('opening an experiment pins the topnav Inspector link to that record',
  async () => {
    // this is an integration test
    const page = panelPage({ search: `?experiment=${EXPERIMENT_ID}`,
                             initialSavedMode: '' });
    await page.settled;
    assert.equal(page.inspectorLink.href,
      `/inspector?experiment=${encodeURIComponent(EXPERIMENT_ID)}`);
  });

test('a fresh page load leaves no poison behind for the next one', async () => {
  // this is an integration test
  // The exact failure mode a live re-check surfaced: a config once merged
  // with `key_facts_judge` in it (an old `localStorage` save, or a
  // config this page itself once wrote before `startingConfig`/`keepUnshown`
  // were hardened) must not keep reintroducing it on every later boot.
  const page = panelPage({ search: `?experiment=${EXPERIMENT_ID}` });
  await page.settled;
  const saved = JSON.parse(page.sandbox.localStorage.getItem('raglab:config'));
  assert.equal('key_facts_judge' in saved.generation, false);
  assert.equal('fact_judge' in saved.generation, true);
});

// This is the exact live reproduction that surfaced the second strict site:
// a browser whose `localStorage` already held a config saved under the old
// field name (from a session that predates `fact_judge` entirely) opened the
// same experiment and the boot itself — `startingConfig()` merging that save
// over the server defaults, then `keepUnshown()` carrying it into every
// `readConfig()` from then on — fed `key_facts_judge` back into the config
// `ArchiveIO.settings()` validates, before `openHandedExperiment` ever ran.
// The widget's own message changed between round 2 and this round for
// exactly that reason: "missing keys fact_judge" (round 1, `fact_judge`
// genuinely absent) became "unexpected keys key_facts_judge" (the live
// re-check, `fact_judge` present *and* the retired key present) — a config
// that was the union of an old save and the current schema, not either shape
// cleanly.
const POISONED_SAVE = {
  label: 'a session from before this schema existed',
  index: { dataset: 'meetings-de' },
  retrieval: {},
  generation: { answerer: 'llm', key_facts_judge: true, model: '',
               judge_model: '', ragas_model: '' },
};

test('a page that boots over an already-poisoned saved config still opens '
  + 'cleanly — the exact scenario a live re-check reproduced', async () => {
  // this is an integration test
  //
  // The poison is seeded *before* the script runs at all, so the one
  // automatic `boot()` call panel.js makes on load — not a second, manually
  // triggered one — is the thing that reads it through `startingConfig()`
  // and carries it into `readConfig()` via `keepUnshown()`, exactly as a
  // second visit to an already-poisoned browser would.
  const page = panelPage({ search: `?experiment=${EXPERIMENT_ID}`,
                           initialSavedConfig: POISONED_SAVE });
  await page.settled;

  const failed = page.requests.notes.filter((note) => /^Opening experiment/.test(note)
    || /was not opened/.test(note));
  assert.deepEqual(failed, [], 'a poisoned save must not make the open refuse');
  assert.equal(page.byId('fact_judge').checked, false);
  assert.equal('key_facts_judge' in page.sandbox.readConfig().generation, false);
  const opened = page.requests.notes.find((note) => /^Laboratory settings are now/.test(note));
  assert.ok(opened);
  assert.match(opened, /key_facts_judge = true — not a knob this lab reads any more/);
});

// --- D3: an absent dataset still means the built-in diary, over the real ---
// page-load path ------------------------------------------------------------
// A live re-check after round 3 found the dropdown reading `'—'` (no
// selection) for an experiment recorded with `index.dataset=''` — the
// pre-D3 way of naming the built-in diary, which the datasets this
// installation now serves list under a real id (`'diary-fa'`) rather than
// `''`. `archive_io.js`'s own `datasetDisposition` already resolves that
// same absence (`|| BUILTIN_DATASET`); `ExperimentHandoff.reconcile` reading
// that one constant (pinned directly in `experiment_handoff.test.js`) is
// the mechanism, and these three tests are that mechanism driven through
// the real page-load entry point, over real archives, for the three cases
// the fix has to tell apart: an absent id resolves and selects, a served
// non-empty id selects exactly as it always did, and a genuinely unknown
// one keeps the ordinary honest naming.

// --- the option values themselves ------------------------------------------
// Which value the dropdown offers for the built-in corpus is the whole of
// this: `''` is that corpus's config identity — `IndexConfig.fingerprint()`
// drops `dataset=''` from its payload, so `''` and `'diary-fa'` name one
// corpus but fingerprint two different collections (`804444ae65db` and
// `cc06e9e8bd3e`). The mapping was written against a `source === 'builtin'`
// test that the service stopped satisfying when the diary became an ordinary
// bundled pair (D3): `source` reads `'bundled'`, so the branch never matched
// and every fresh selection of the diary sent `'diary-fa'`, fingerprinting
// away from every collection already recorded. A count of the source text
// was what guarded it, and text is exactly what a rename like that leaves
// intact — so this asks the served page for its own option values instead.
test('the dataset dropdown offers the built-in corpus as the empty string '
  + 'and every other corpus under its own id', async () => {
  // this is an integration test
  const page = panelPage({});
  await page.settled;
  const options = [...page.byId('dataset').innerHTML
    .matchAll(/<option value="([^"]*)">([^<]*)</g)]
    .map(([, value, text]) => ({ value, text }));
  const rows = REAL_OPTIONS.datasets;
  assert.ok(rows.length >= 5, 'the real /api/options fixture lists the corpora');
  assert.deepEqual(options.map((option) => option.value),
    rows.map((row) => (row.id === 'diary-fa' ? '' : row.id)),
    'one option per served corpus, in order: the built-in corpus under the '
    + 'empty string, every other corpus under its own id');
  // Not by absence of a match, but by the corpus the empty option actually
  // names — a mapping that dropped the diary entirely would also produce no
  // `value="diary-fa"`.
  const builtin = options.find((option) => option.value === '');
  const diary = rows.find((row) => row.id === 'diary-fa');
  assert.ok(builtin.text.startsWith(diary.name),
    'and the empty option is the diary itself, named by the catalogue');
  // The other direction, said outright: the id must not be an option value,
  // or selecting the diary would send a value no recorded collection has.
  assert.equal(page.byId('dataset').innerHTML.includes('value="diary-fa"'), false);

  // What a fresh page would actually submit for a build: the served default
  // — the English diary, named by its own id, never the `''` the built-in
  // Farsi original keeps — read back through the control that shows it.
  assert.equal(REAL_OPTIONS.defaults.index.dataset, 'diary-en');
  assert.equal(page.byId('dataset').value, 'diary-en');
  assert.equal(page.sandbox.readConfig().index.dataset, 'diary-en');
  // And the corpus card renders, which is the visible half of the same fact:
  // the lookup reads the selected value back to a corpus, and under the dead
  // mapping `''` matched no row at all, so `describeDataset` returned early
  // and the header described nothing.
  assert.equal(page.byId('corpusName').textContent, 'diary-en');
  assert.match(page.byId('corpus').textContent, /documents · \d+ parts/);
});

test("an archive recorded with dataset='' adopts '' — the corpus is named, "
  + 'the recorded collection stays reproducible', async () => {
    const page = panelPage({ search: `?experiment=${EXPERIMENT_ID}` });
    await page.settled;
    assert.equal(ARCHIVE.settings.config.index.dataset, '',
      'this fixture must really be the pre-D3 shape the live re-check found');
    assert.equal(page.byId('dataset').value, '',
      "the recorded '' is written through unchanged: it is the value the "
      + "built-in corpus's own option carries, and the value a rebuild must "
      + 'fingerprint under to land on the collection this run used');
    // The config this page would submit next — the run payload's own
    // `index.dataset` — is the promise `IndexConfig.fingerprint()` keeps.
    assert.equal(page.sandbox.readConfig().index.dataset, '');
    // Resolved for the reader, not for the record: the header names the
    // corpus by id even though the knob holds the empty string.
    assert.equal(page.byId('corpusName').textContent, 'diary-fa');
    const failed = page.requests.notes.filter((note) => /^Opening experiment/.test(note)
      || /was not opened/.test(note));
    assert.deepEqual(failed, []);
    const opened = page.requests.notes.find((note) => /^Laboratory settings are now/.test(note));
    assert.ok(opened);
    // Resolved and applied, so it is never named as something this lab
    // could not serve — the built-in corpus is always servable.
    assert.doesNotMatch(opened, /dataset.*not installed here/);
  });

test('an archive recorded with a served, non-empty dataset id selects that '
  + 'dataset exactly as it always did', async () => {
  const page = panelPage({ search: `?experiment=${MEETINGS_EXPERIMENT_ID}`,
                           archives: { [MEETINGS_EXPERIMENT_ID]: MEETINGS_ARCHIVE } });
  await page.settled;
  assert.equal(MEETINGS_ARCHIVE.settings.config.index.dataset, 'meetings-de',
    'this fixture must really carry a served, non-empty id, not the '
    + "'' case the sibling test covers");
  assert.equal(page.byId('dataset').value, 'meetings-de');
  assert.equal(page.sandbox.readConfig().index.dataset, 'meetings-de');
  const opened = page.requests.notes.find((note) => /^Laboratory settings are now/.test(note));
  assert.ok(opened);
  assert.doesNotMatch(opened, /dataset.*not installed here/);
});

test('an archive recorded with a dataset this installation does not serve '
  + 'at all keeps the ordinary unserved naming, and the dropdown stays at '
  + "the panel's own selection", async () => {
  const archive = unknownDatasetArchive();
  const runId = archive.evaluation.result.run_id;
  const page = panelPage({ search: `?experiment=${runId}`,
                           archives: { [runId]: archive } });
  await page.settled;
  assert.equal(page.byId('dataset').value, 'diary-en',
    "the panel's own starting selection (no prior save in this test) — "
    + 'never the unknown id, and never diary-fa either: an unknown id is '
    + "not '' and gets no free resolution");
  const opened = page.requests.notes.find((note) => /^Laboratory settings are now/.test(note));
  assert.ok(opened);
  assert.match(opened, new RegExp(`dataset = ${UNKNOWN_DATASET_ID} — not installed here`));
});
