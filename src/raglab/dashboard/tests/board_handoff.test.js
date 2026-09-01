// tests/board_handoff.test.js — the board's open button, and the second thing
// it does.
//
// Contract under test: the open column of `dashboard/frontend/leaderboard.js`
// pins the Inspector to one recorded experiment *and* hands the same experiment
// to the Laboratory, so the knobs there become that experiment's.
//
// The link half is the browser's and must stay the browser's: the cell is an
// `<a>` so that middle-click, ⌘-click and the keyboard all still work, which
// means the handler must not call `preventDefault`. A handler that swallowed
// the click and navigated by script would look identical in a screenshot and
// would have quietly cost every one of those.
//
// Evaluated in a `vm` context against the served files, the way
// `board_reveal.test.js` does.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (name) => readFileSync(join(HERE, '../frontend', name), 'utf8');
// The three scripts the leaderboard page loads, in the order it loads them.
const SOURCE = `${read('experiment_handoff.js')}\n${read('lab.js')}\n`
  + `${read('leaderboard.js')}`;

function board() {
  const listeners = {};
  const kept = {};
  const storage = {
    getItem: (key) => (key in kept ? kept[key] : null),
    setItem: (key, value) => { kept[key] = String(value); },
    removeItem: (key) => { delete kept[key]; },
  };
  const sandbox = {
    document: {
      addEventListener(type, fn) {
        (listeners[type] = listeners[type] || []).push(fn);
      },
      // `loadBoard` runs on load and writes into #board; `fetch` is absent, so
      // the read fails into its own error card and nothing else is reached.
      getElementById: () => ({ innerHTML: '' }),
      querySelector: () => null,
      querySelectorAll: () => [],
      activeElement: null,
    },
    window: {
      innerHeight: 1000,
      innerWidth: 1400,
      location: { search: '', href: 'http://localhost:9002/leaderboard' },
      history: { replaceState() {} },
      localStorage: storage,
    },
    localStorage: storage,
    getComputedStyle: () => ({ position: 'fixed' }),
    URLSearchParams,
  };
  runInNewContext(SOURCE, sandbox);

  const fire = (type, event) => {
    const wired = listeners[type] || [];
    assert.ok(wired.length, `nothing listens for ${type}`);
    for (const fn of wired) fn({ target: null, relatedTarget: null, ...event });
  };
  return { fire, kept, sandbox };
}

// One open link, as the board renders it, and something that is not one.
function openLink(experimentId) {
  const anchor = {
    dataset: { experiment: experimentId },
    closest: (selector) => (selector === '.open-run' ? anchor : null),
  };
  return anchor;
}
const elsewhere = { closest: () => null };

const slot = (kept) => JSON.parse(kept['raglab:open-experiment']);

// This is a unit test.
test('opening an experiment hands it to the Laboratory as well', () => {
  const { fire, kept } = board();
  fire('click', { target: openLink('exp-1') });
  assert.equal(slot(kept).experiment_id, 'exp-1');
});

// This is a unit test.
test('the open cell stays a link the browser follows', () => {
  // Middle-click, ⌘-click and Enter on a focused link are all the browser's,
  // and all of them are lost the moment this handler prevents the default.
  const { fire, kept } = board();
  let prevented = false;
  fire('click', {
    target: openLink('exp-1'),
    preventDefault: () => { prevented = true; },
  });
  assert.equal(prevented, false, 'the Inspector link must still be a link');
  assert.ok(kept['raglab:open-experiment'], 'and the slot still written');
});

// This is a unit test.
test('opening the same experiment twice is heard twice', () => {
  // An already-open Laboratory tab learns about this through a `storage` event,
  // which fires on a change and not on a write. Two clicks on one row writing
  // the same bytes would leave the second one silent.
  const { fire, kept } = board();
  fire('click', { target: openLink('exp-1') });
  const first = kept['raglab:open-experiment'];
  fire('click', { target: openLink('exp-1') });
  assert.notEqual(kept['raglab:open-experiment'], first);
});

// This is a unit test.
test('clicking anything else on the board hands over nothing', () => {
  const { fire, kept } = board();
  fire('click', { target: elsewhere });
  assert.equal(kept['raglab:open-experiment'], undefined);
});

// This is a unit test.
test('the open cell carries the experiment id the handler reads', () => {
  // The two halves of one cell: the `href` the browser follows and the
  // `data-experiment` the handler hands over. A cell that rendered only the
  // href would pass every test above — they build their own anchor — and hand
  // over `undefined` in a browser.
  const source = read('leaderboard.js');
  const openCell = source.slice(source.indexOf("case 'open':"));
  assert.ok(openCell.includes('data-experiment='),
    'the rendered open link must name the experiment it opens');
});

// This is a unit test.
test('the open link lands on the Laboratory, where the knobs are', () => {
  // The slot alone only reaches a Laboratory that is already open. A reader
  // who has just the board up clicked "open" and watched the Inspector appear
  // while the knobs they were promised stayed on another surface entirely.
  const source = read('leaderboard.js');
  const openCell = source.slice(source.indexOf("case 'open':"));
  const href = /href="([^"]*)"/.exec(openCell);
  assert.ok(href, 'the open cell must render an href');
  assert.ok(href[1].startsWith('/?experiment='),
    `the open link must go to the Laboratory, not ${href[1]}`);
});

// This is a convention test.
test('the open cell chooses same-tab navigation by default', () => {
  const source = read('leaderboard.js');
  const start = source.indexOf("case 'open':");
  const openCell = source.slice(start, source.indexOf('default:', start));
  assert.doesNotMatch(openCell, /target=/,
    'the board must leave same-tab navigation as the browser default');
});
