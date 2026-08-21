// Browser contract: which conversation the widget is in.
//
// One thread per experiment is the whole of the recall — open an experiment and
// the conversation you had about it is simply there. The rule worth pinning is
// what happens when nothing is open, because that is most of the time: the
// reader must land in one shared general thread rather than in a new one per
// page, which is the reset this change exists to remove.
//
// `nextGeneration`/`supersedes`/`stillCurrent` are pinned here too: they are
// the two decisions that keep a redraw or a note from painting a screen that
// has already moved on to a different thread, and both are pure — no DOM,
// nothing but `DRAW_SEQ` and a call to `widgetThread()` — which is exactly
// what makes them reachable from this storage-only sandbox at all.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const source = readFileSync(new URL('../frontend/widget.js', import.meta.url), 'utf8');

function load(stored) {
  const store = new Map(Object.entries(stored || {}));
  const context = {
    localStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, String(v)),
      removeItem: (k) => store.delete(k),
    },
    document: { body: { append() {} }, createElement: () => ({ style: {}, classList: { add() {} } }), addEventListener() {}, querySelectorAll: () => [], getElementById: () => null },
    window: {}, fetch: async () => ({ ok: true, json: async () => ({}) }),
  };
  context.window = context;
  // Only the thread half is under test; the DOM half is exercised by test_panel.
  const half = source.slice(source.indexOf('// --- which conversation'),
                            source.indexOf('// --- end of the thread half'));
  runInNewContext(half, context);
  return context;
}

test('with no experiment open, every surface shares the general thread', () => {
  assert.equal(load({}).widgetThread(), 'general');
});

test('with an experiment open, the thread is that experiment', () => {
  const page = load({ 'raglab-active-experiment': 'abc123' });
  assert.equal(page.widgetThread(), 'abc123');
});

test('opening an experiment remembers it for every surface', () => {
  const page = load({});
  page.widgetAbout('def456');
  assert.equal(page.widgetThread(), 'def456');
});

test('leaving an experiment drops back to the general thread, not to a new one', () => {
  const page = load({ 'raglab-active-experiment': 'abc123' });
  page.widgetAbout('');
  assert.equal(page.widgetThread(), 'general');
});

test('a generation is not superseded until a later one starts', () => {
  const page = load({});
  const mine = page.nextGeneration();
  assert.equal(page.supersedes(mine), false);
});

test('starting a new draw supersedes every earlier generation', () => {
  const page = load({});
  const mine = page.nextGeneration();
  page.nextGeneration(); // a second draw starts before the first settles
  assert.equal(page.supersedes(mine), true);
});

test('a thread stops being current the moment another one is opened', () => {
  const page = load({ 'raglab-active-experiment': 'abc123' });
  assert.equal(page.stillCurrent('abc123'), true);
  page.widgetAbout('def456');
  assert.equal(page.stillCurrent('abc123'), false);
  assert.equal(page.stillCurrent('def456'), true);
});
