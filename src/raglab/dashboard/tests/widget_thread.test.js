// Browser contract: which conversation the widget is in.
//
// One thread per experiment is the whole of the recall — open an experiment and
// the conversation you had about it is simply there. The rule worth pinning is
// what happens when nothing is open, because that is most of the time: the
// reader must land in one shared general thread rather than in a new one per
// page, which is the reset this change exists to remove.
//
// `nextGeneration`/`currentGeneration`/`supersedes`/`stillCurrent`, and
// `replyFate` on top of them, are pinned here too: they are the decisions that
// keep a redraw, a note or an answer from painting a screen that has already
// moved on — and, just as importantly, from silently dropping something the
// screen is still waiting for. All of them are pure — no DOM, nothing but
// `DRAW_SEQ`, a call to `widgetThread()` and their arguments — which is
// exactly what makes them reachable from this storage-only sandbox at all.
//
// The last test is the exception, and deliberately so: it reads `widgetAsk`'s
// source rather than running it. What it pins is an *ordering* — that the
// answer's inputs are taken before the post, not after it — and an ordering
// between a synchronous read and an await cannot be observed from a sandbox
// with no DOM, no fetch and no scheduler; building all three to watch one
// variable being read a tick earlier would replace this contract with a fake
// browser. A decision that is only correct because of where it sits in a
// function is fairly pinned by looking at where it sits.
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

// Asking what the current generation is must not be the same thing as
// starting one. `widgetAsk` reads it before it posts, precisely because it is
// not starting a draw of its own; a version that incremented instead would
// stand down every draw in flight the moment a reader hit send, and a version
// that answered a constant would make the answer's own generation check
// meaningless. The first assertion catches both; the second states the
// consequence the caller actually depends on.
test('reading the current generation reports the newest draw without starting one', () => {
  const page = load({});
  const mine = page.nextGeneration();
  assert.equal(page.currentGeneration(), mine);
  assert.equal(page.supersedes(mine), false);
});

test('an answer whose thread the reader has left is not shown at all', () => {
  const page = load({ 'raglab-active-experiment': 'abc123' });
  const mine = page.currentGeneration();
  page.widgetAbout('def456');
  assert.equal(page.replyFate(mine, 'abc123', false), 'gone');
});

// The case that cost a reader their own turn: same thread, so the header is
// right and nothing could be misattributed, but a redraw started while the
// question was in flight — and that redraw's history GET usually beats the
// answer, taking the question with it. The honest way out is the lab's own
// record, so this must ask for a redraw rather than drop.
test('an answer overtaken by a redraw of its own thread is redrawn, not dropped', () => {
  const page = load({ 'raglab-active-experiment': 'abc123' });
  const mine = page.currentGeneration();
  page.nextGeneration();
  assert.equal(page.replyFate(mine, 'abc123', false), 'stale');
});

// The same loss without a supersession: a question asked while the draw that
// opened the widget is still open carries that draw's own generation, so no
// generation check can see it coming — only the flag can.
test('an answer a draw still in flight would erase is redrawn too', () => {
  const page = load({ 'raglab-active-experiment': 'abc123' });
  const mine = page.nextGeneration();
  assert.equal(page.supersedes(mine), false);
  assert.equal(page.replyFate(mine, 'abc123', true), 'stale');
});

test('an answer nothing has moved under is said where it was asked', () => {
  const page = load({ 'raglab-active-experiment': 'abc123' });
  assert.equal(page.replyFate(page.currentGeneration(), 'abc123', false), 'here');
});

// A right answer read at the wrong moment. `intended` and `mine` are captured
// before the post because that is this file's whole discipline; `drawPending`
// was not, and it is the one of the three that changes on its own while the
// post is in flight. A draw running when Send is pressed clears the log — the
// reader's own question with it — and then finishes, long before a model turn
// comes back, so a `drawPending` read only on the way out reports false and
// `replyFate` says `'here'`: the answer paints alone, under no question. The
// reachable route is New Chat, which unlocks the controls in its `finally`
// while the redraw it just started is still running, but the launcher's draw
// and another tab's `storage` echo open the same window.
//
// Pinned as three claims, because each fails a different mistake: that the
// flag is read at all before the post, that the read really does come before
// the await rather than after it, and that both readings reach `replyFate`
// together rather than the later one alone.
test('a question captures whether a draw was already in flight before it posts', () => {
  const ask = source.slice(source.indexOf('async function widgetAsk'),
                           source.indexOf('async function widgetLoadOptions'));
  const captured = ask.indexOf('const wasPending = drawPending;');
  const posted = ask.indexOf('await widgetStream(');
  assert.ok(captured > -1,
    'widgetAsk must read drawPending into a local, not at reply time');
  assert.ok(posted > -1 && captured < posted,
    'the capture must happen before the post, or it is not a capture');
  const fates = ask.match(/replyFate\([^)]*\)/g) || [];
  assert.deepEqual(fates, ['replyFate(mine, intended, wasPending || drawPending)',
                           'replyFate(mine, intended, wasPending || drawPending)'],
    'both the answer and the failure path must weigh the whole wait: a draw '
    + 'in flight when the question was asked counts even once it has settled');
});

// The same discipline, one layer in: the answer no longer arrives all at once,
// so the log is written to over and over while the reader can still leave the
// thread or start a draw that clears it. Every piece has to re-ask the two
// questions `replyFate` asks at the end — read from the source for the same
// reason the capture above is: what is being pinned is *where* the check sits,
// inside the callback that renders a piece, and a sandbox with no DOM, no
// fetch and no stream cannot observe that by running it.
test('every streamed piece re-asks whether it still belongs on this screen', () => {
  const ask = source.slice(source.indexOf('async function widgetAsk'),
                           source.indexOf('async function widgetLoadOptions'));
  const delta = ask.slice(ask.indexOf('(delta) =>'), ask.indexOf('replyFate'));
  assert.ok(delta.includes('supersedes(mine)'),
    'a piece must not paint over a screen a newer draw has taken');
  assert.ok(delta.includes('!stillCurrent(intended)'),
    'a piece must not be typed into a thread the reader has left');
  assert.ok(ask.indexOf('widgetFinish(live, data.reply)') > ask.indexOf('replyFate'),
    'the reply the lab holds must replace the typed pieces, and only after '
    + 'the fate check has said this screen is still the right one');
});
