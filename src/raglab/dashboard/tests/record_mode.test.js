// tests/record_mode.test.js — the Inspector's recorded-experiment mode, driven
// over every shape the ledger actually holds.
//
// Contract under test: `?experiment=<id>` on :9002/inspector, which is what
// the board's frozen `↗` column links to — `followRecordedExperiment` and
// `renderRecordedExperiment` at the foot of `dashboard/frontend/inspector.js`.
//
// What is tested here is which branch a record reaches, because that is where
// this mode is wrong in ways nobody notices. The four views' sentences were
// pinned as strings, which proves only that the sentences exist; the defect was
// that a cancelled run, a one-off query, an errored retrieval and an imported
// archive were all handed an index build's explanation, and one of them was
// told nothing had been retrieved and nothing answered while its own record
// held the trace and the answer. Five real `detail` shapes, one test each, read
// off what each view ends up saying.
//
// Evaluated in a `vm` context, the way sorttable.test.js and board_reveal.test.js
// do, against the smallest fake of what the page actually touches: the served
// files need no module wrapper they would otherwise carry only for this test.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (name) => readFileSync(join(HERE, '../frontend', name), 'utf8');
const SOURCE = `${read('sorttable.js')}\n${read('lab.js')}\n${read('inspector.js')}`;

// --- the smallest DOM the page runs against ---------------------------------
// Every element is the same stub, because what this file asserts on is what a
// named box ends up holding — `innerHTML` and `textContent` — and nothing here
// cares what any of them look like. `querySelector` hands back a child stub so
// the template clone in `retrievalTable` and the tbody it fills both work;
// `tHead` is deliberately absent, which is how `SortTable.make` correctly
// declines to wire a table that has no header row.
function element(name = '') {
  const held = new Map();
  const self = {
    name,
    innerHTML: '',
    textContent: '',
    hidden: false,
    disabled: false,
    dir: '',
    className: '',
    type: '',
    id: '',
    value: '',
    tabIndex: 0,
    dataset: {},
    style: {},
    children: [],
    // A table with no header row is one `SortTable.make` correctly declines to
    // wire — but it reads `tBodies[0]` before it checks, so the arrays have to
    // be there to be found empty.
    tHead: null,
    tBodies: [],
    classList: { add() {}, remove() {}, contains: () => false },
    attributes: {},
    setAttribute(key, value) { self.attributes[key] = String(value); },
    getAttribute: (key) => (key in self.attributes ? self.attributes[key] : null),
    hasAttribute: (key) => key in self.attributes,
    addEventListener() {},
    removeEventListener() {},
    appendChild(child) { self.children.push(child); return child; },
    insertAdjacentHTML() {},
    remove() {},
    focus() {},
    querySelectorAll: () => [],
    closest: () => null,
    getBoundingClientRect: () => ({ top: 0, bottom: 0, left: 0, right: 0 }),
    cloneNode: () => element(`${name}-clone`),
    querySelector(selector) {
      if (!held.has(selector)) held.set(selector, element(`${name}${selector}`));
      return held.get(selector);
    },
  };
  // A <template>'s content, for the one table this page clones rather than
  // writes: `retrievalTable` reaches through `.content.firstElementChild`.
  self.content = { get firstElementChild() { return self.querySelector('table'); } };
  return self;
}

// One fake page. `records` maps an experiment id to what `/api/experiments/{id}`
// answers with; anything absent 404s, which is the deleted-row and mistyped-id
// case. `search` is the page's query string, so a deep link can be exercised
// through the same entry point the browser uses.
function inspector({ records = {}, groundtruth = {}, search = '' } = {}) {
  const byId = new Map();
  const byIdOf = (id) => {
    if (!byId.has(id)) byId.set(id, element(id));
    return byId.get(id);
  };
  const requests = [];

  // Every request the page makes goes through its own `api()` helper now,
  // which prefixes the mount — so the routes this fake answers are the
  // prefixed ones, exactly as a real browser would send them.
  const answer = (path) => {
    requests.push(path);
    if (path === '/inspector/api/config') return { ok: true, body: { chosen: LIVE_CONFIG } };
    if (path === '/inspector/api/explain') return { ok: true, body: { metrics: [], help: {} } };
    if (path.startsWith('/inspector/api/groundtruth')) {
      const dataset = decodeURIComponent(path.split('dataset=')[1] || '');
      const found = groundtruth[dataset];
      return found
        ? { ok: true, body: found }
        : { ok: false, body: { detail: `unknown dataset ${dataset}` } };
    }
    if (path.startsWith('/inspector/api/experiments/')) {
      const id = decodeURIComponent(path.slice('/inspector/api/experiments/'.length));
      return records[id]
        ? { ok: true, body: records[id] }
        : { ok: false, body: { detail: 'unknown experiment' } };
    }
    if (path === '/inspector/api/follow') return { ok: true, body: { lab: 'up', lab_url: 'x' } };
    return { ok: false, body: { detail: `no route ${path}` } };
  };

  const sandbox = {
    console,
    URLSearchParams,
    URL,
    Map,
    Set,
    // Never called back: the follow loop would otherwise re-enter itself for
    // the length of the test run, and nothing here is about the second tick.
    setTimeout() {},
    getComputedStyle: () => ({ position: 'static' }),
    history: { replaceState() {} },
    fetch: (path) => {
      const { ok, body } = answer(path);
      return Promise.resolve({
        ok,
        status: ok ? 200 : 404,
        statusText: ok ? 'OK' : 'Not Found',
        json: () => Promise.resolve(body),
      });
    },
    document: {
      documentElement: element('html'),
      activeElement: null,
      getElementById: byIdOf,
      createElement: (tag) => element(`<${tag}>`),
      querySelector: () => element('querySelector'),
      querySelectorAll: () => [],
      addEventListener() {},
    },
    window: {
      innerHeight: 1000,
      innerWidth: 1400,
      location: { search, href: `http://localhost:9002/inspector/${search}` },
      history: { replaceState() {} },
    },
  };
  sandbox.window.document = sandbox.document;
  runInNewContext(SOURCE, sandbox);
  // Everything the page starts on load — the config fetch, the ground-truth
  // fetch, the first follow tick, and a deep link when there is one — is a
  // promise chain several microtasks deep. Drained here so a test reads the
  // page as it stands once the boot has settled.
  const settled = (async () => { for (let i = 0; i < 30; i += 1) await null; })();
  return { sandbox, byId: byIdOf, requests, settled };
}

const LIVE_CONFIG = {
  index: { dataset: 'live-corpus', chunker: 'session', embedder: 'token-hash' },
  retrieval: { retriever: 'dense', k: 8, reranker: 'none', grader: 'none' },
  generation: { answerer: 'llm' },
};

const RECORDED_CONFIG = {
  index: { dataset: 'smoke-mini', chunker: 'semantic-drift',
           embedder: 'sentence-transformers' },
  retrieval: { retriever: 'hybrid-rrf', k: 8, reranker: 'llm', grader: 'llm' },
  generation: { answerer: 'extractive' },
};

const CORPUS = {
  'smoke-mini': { dataset: 'smoke-mini', language: 'en', questions: [] },
  'live-corpus': { dataset: 'live-corpus', language: 'en', questions: [] },
  '': { dataset: 'live-corpus', language: 'en', questions: [] },
};

const record = (over) => ({
  experiment_id: 'exp-1', kind: 'run', state: 'done', error: '',
  dataset: 'smoke-mini', detail: { config: RECORDED_CONFIG }, ...over,
});

// What each view says, once a record is on screen.
function views(page) {
  return {
    state: page.byId('archive-state').textContent,
    control: page.byId('archive-return-live').textContent,
    retrieval: page.byId('retrieval-questions').innerHTML,
    generation: page.byId('generation-questions').innerHTML,
    chunks: page.byId('chunks-body').innerHTML,
    candidates: page.byId('retrieval-body'),
    answer: page.byId('retrieval-answer').textContent,
    // The rebuild is a real button in the view's control row rather than markup
    // in its `innerHTML`, so whether it was offered is a question about the
    // children — which is the point: a regex over the note would pass on the
    // note's own sentence about rebuilding.
    rebuild: page.byId('chunks-body').children.some((row) =>
      (row.children || []).some((b) => b.id === 'rebuild-recorded-chunks')),
  };
}

async function open(id, options) {
  const page = inspector(options);
  await page.settled;
  await page.sandbox.followRecordedExperiment(id);
  return page;
}

// This is a unit test.
test('a cancelled run says it was cancelled, and carries its reason', async () => {
  // The worst of the five: a read-only view of a cancelled experiment that
  // never says "cancelled" is a row not carrying the reason it is degraded. Its
  // detail holds a config and nothing else, which is indistinguishable from an
  // index build's by row-counting alone — so it was told an index build's story
  // and its state was never mentioned at all.
  const page = await open('cancelled-1', {
    groundtruth: CORPUS,
    records: {
      'cancelled-1': record({
        experiment_id: 'cancelled-1', kind: 'run', state: 'cancelled',
        error: 'stopped from the panel',
      }),
    },
  });
  const shown = views(page);
  assert.match(shown.state, /cancelled/,
    'the state line must say the record was cancelled');
  assert.match(shown.state, /read-only · cancelled-1/);
  for (const [view, text] of [['retrieval', shown.retrieval],
                              ['generation', shown.generation],
                              ['chunks', shown.chunks]]) {
    assert.match(text, /cancelled/, `${view} must say why it is empty`);
    assert.match(text, /stopped from the panel/,
      `${view} must carry the reason the row recorded`);
    assert.doesNotMatch(text, /an index build/,
      `${view} must not explain a cancelled run as an index build`);
  }
});

// This is a unit test.
test('an errored retrieval says it failed, and names the failure', async () => {
  const page = await open('failed-1', {
    groundtruth: CORPUS,
    records: {
      'failed-1': record({
        experiment_id: 'failed-1', kind: 'retrieve', state: 'error',
        error: "NameError: name 'agent' is not defined",
      }),
    },
  });
  const shown = views(page);
  assert.match(shown.state, /retrieve · error/);
  assert.match(shown.retrieval, /NameError/);
  assert.match(shown.generation, /NameError/);
  assert.doesNotMatch(shown.retrieval, /an index build/);
});

// This is a unit test.
test('a one-off query shows the answer and the candidates its record holds',
  async () => {
    // Its detail carries `trace`, `contexts`, `sessions` and `answer` — real
    // evidence — and the page told the reader nothing had been retrieved and
    // nothing answered, which is a view withholding what its own record holds.
    const page = await open('query-1', {
      groundtruth: CORPUS,
      records: {
        'query-1': record({
          experiment_id: 'query-1', kind: 'query',
          detail: {
            config: RECORDED_CONFIG,
            question: 'what did I eat',
            question_id: 'q7',
            answer: 'a recorded answer',
            abstained: false,
            contexts: ['one context'],
            sessions: ['s1'],
            trace: { candidates: [
              { chunk_id: 'c1', text: 'evidence', kept: true, gold: true,
                dense_rank: 1, bm25_rank: 2, fused_rank: 1 },
            ] },
          },
        }),
      },
    });
    const shown = views(page);
    assert.equal(shown.answer, 'a recorded answer',
      'the answer the record holds has to be on screen');
    assert.ok(shown.candidates.children.length,
      'the traced candidates the record holds have to be on screen');
    assert.match(shown.state, /query/);
    assert.match(shown.retrieval, /one-off/,
      'and the list above says why there is no per-question list');
    assert.doesNotMatch(shown.retrieval, /an index build/);
    assert.doesNotMatch(shown.generation, /wrote no answers/,
      'a query that answered must not be told it wrote no answers');
  });

// This is a unit test.
test('an index build is the only record told that a build measures no questions',
  async () => {
    const page = await open('index-1', {
      groundtruth: CORPUS,
      records: {
        'index-1': record({ experiment_id: 'index-1', kind: 'index',
                            detail: { config: RECORDED_CONFIG, chunks: 12 } }),
      },
    });
    const shown = views(page);
    assert.match(shown.retrieval, /an index build measures no questions/);
    assert.match(shown.generation, /chunks and embeds a corpus and stops there/,
      "the generation view states the build's own reason, not a shared one");
    assert.doesNotMatch(shown.state, /cancelled|error/);
  });

// This is a unit test.
test('an evaluation resolved from its run file keeps its answers, and says the '
  + 'rankings were never written down', async () => {
  const page = await open('run-1', {
    groundtruth: CORPUS,
    records: {
      'run-1': record({
        experiment_id: 'run-1', kind: 'run',
        detail: { config: RECORDED_CONFIG, ragas: {},
                  rows: [{ id: 'q1', answer: 'an answer', type: 'single-hop',
                           difficulty: 'easy' }] },
      }),
    },
  });
  const shown = views(page);
  assert.match(shown.retrieval, /not recorded/);
  assert.doesNotMatch(shown.retrieval, /an index build/,
    'an evaluation that answered questions did retrieve; it was not recorded');
  assert.ok(page.byId('generation-questions').children.length,
    'its answered rows are rendered rather than explained away');
});

// This is a unit test.
test('an imported archive is rendered from its own payload, never through a '
  + 'config it did not name', async () => {
  // `insert_archive` stores the whole archive payload as `detail` — by design,
  // it is the one record preserved verbatim — so `detail.config`,
  // `detail.rows` and `detail.traces` are all absent. Read as a job result it
  // emptied four tabs for a row whose decision score the board can show, and
  // offered "Rebuild from this config" that fell through to the *live* config
  // under a note promising this experiment's.
  const archive = {
    format: 'raglab-experiment',
    version: 1,
    settings: { config: RECORDED_CONFIG, ui: {} },
    evaluation: {
      result: { run_id: 'arch-1', config: RECORDED_CONFIG, dataset: 'smoke-mini',
                rows: [{ id: 'q1', answer: 'archived answer', type: 'single-hop',
                         difficulty: 'easy' }],
                summary: { n_questions: 1 }, ragas: {} },
      inspector: {
        dataset: { id: 'smoke-mini', corpus: { meta: { language: 'en' } },
                   ground_truth: { meta: {}, questions: [] } },
        chunks_by_session: [{ session_id: 's1', date: '2026-08-19',
                              chunks: [{ id: 'c1', text: 'evidence' }] }],
        summaries: [],
        traces: [{ question_id: 'q1', trace: { candidates: [] } }],
      },
    },
  };
  const page = await open('arch-1', {
    groundtruth: CORPUS,
    records: {
      'arch-1': record({ experiment_id: 'arch-1', kind: 'run',
                         detail: archive }),
    },
  });
  const shown = views(page);
  assert.match(shown.state, /read-only · arch-1/);
  assert.doesNotMatch(shown.chunks, /not recorded/,
    'an archive carries its chunk text, so nothing here is missing');
  assert.equal(shown.rebuild, false,
    'and no rebuild is offered, because there is no config here to rebuild from');
  assert.ok(page.byId('generation-questions').children.length,
    "the archive's answered rows are on screen");
  assert.doesNotMatch(shown.generation, /wrote no answers/);
  assert.doesNotMatch(shown.retrieval, /an index build/);
});

// This is a unit test.
test("a record's chunk text is stated missing, with a rebuild under its own "
  + 'config and no other', async () => {
  // The pin this replaces read `'not recorded' in js`, which the retrieval
  // empty state satisfied on its own — so the branch's headline honesty claim
  // was advertised by a test that could not fail.
  const page = await open('run-2', {
    groundtruth: CORPUS,
    records: { 'run-2': record({ experiment_id: 'run-2' }) },
  });
  assert.match(views(page).chunks, /chunk text of this experiment was <b>not/,
    'the Chunks tab has to say that this specific evidence was never recorded');
  assert.equal(views(page).rebuild, true,
    'and offer the rebuild, under the config the record actually named');

  // And a record with no config at all is not offered one: `buildChunks(null)`
  // falls back to the live config on purpose, for the button beside the tab —
  // taking that fallback here would rebuild under a config this experiment
  // never ran, under a note promising this experiment's.
  const bare = await open('run-3', {
    groundtruth: CORPUS,
    records: { 'run-3': record({ experiment_id: 'run-3', detail: {} }) },
  });
  assert.equal(views(bare).rebuild, false);
  assert.match(views(bare).chunks, /config was not recorded/);
});

// This is a unit test.
test('a deep link that failed keeps saying so, and keeps its way back', async () => {
  // Observed in a browser: the lab was holding an imported archive, the follow
  // loop wrote "Imported archive · read-only · …" over the failure message, and
  // the reader landed on an unrelated archive and was told nothing — while the
  // control beside it still routed to the record's own exit, which issues no
  // DELETE, so pressing "Return to live" did not return to live.
  const page = inspector({ groundtruth: CORPUS, search: '?experiment=gone-1' });
  await page.settled;
  const stated = page.byId('archive-state').textContent;
  assert.match(stated, /gone-1 could not be read/);
  assert.equal(page.byId('archive-return-live').textContent, 'Dismiss',
    'a record that never loaded was never left, so the control says what it does');

  await page.sandbox.renderFollow({ lab: 'up', lab_url: 'x', dataset: '',
                                    archive_id: 'unrelated-archive' });
  assert.equal(page.byId('archive-state').textContent, stated,
    'a stated failure is not overwritten by an archive nobody asked for');
  assert.equal(page.byId('archive-return-live').textContent, 'Dismiss');
});
