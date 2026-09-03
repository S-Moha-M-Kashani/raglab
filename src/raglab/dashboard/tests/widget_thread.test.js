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
  // Only the thread half is under test; the DOM half is exercised by test_routes_widget.
  const half = source.slice(source.indexOf('// --- which conversation'),
                            source.indexOf('// --- end of the thread half'));
  runInNewContext(half, context);
  return context;
}

// The SSE reader, run whole rather than read. `widgetStream` itself needs no
// DOM — only `fetch` — and its buffered fallback (a response with no `.body`)
// drains the same events through the same parser as the streaming path, so a
// fake fetch handing back the whole SSE text at once exercises exactly the
// routing under test: which callback each kind of event reaches, and which
// event gets to be final.
function loadStream(sse) {
  const context = {
    fetch: async () => ({ ok: true, text: async () => sse }),
  };
  const reader = source.slice(source.indexOf('async function widgetStream'),
                              source.indexOf('// --- what you can ask'));
  runInNewContext(reader, context);
  return context;
}

// A status event is the lab saying what it is doing — a tool being called —
// not part of the answer: it must reach `onStatus` and nothing else. The trap
// this pins shut is the reader's old shape, where anything without a `delta`
// was captured as the final event; a status event falling into that bucket
// would be ephemeral chatter promoted to the turn the log holds.
test('a status event is routed to onStatus and never becomes the final event', async () => {
  const sse = 'data: {"status": "search_notes"}\n\n'
            + 'data: {"delta": "he"}\n\n'
            + 'data: {"delta": "llo"}\n\n'
            + 'data: {"reply": "hello"}\n\n';
  const deltas = [];
  const statuses = [];
  const final = await loadStream(sse).widgetStream('/api/widget/stream', {},
    (delta) => deltas.push(delta), (status) => statuses.push(status));
  assert.deepEqual(statuses, ['search_notes']);
  assert.deepEqual(deltas, ['he', 'llo']);
  assert.equal(final.reply, 'hello');
  assert.equal(final.status, undefined);
});

test('a memory event is routed after the authoritative reply', async () => {
  const sse = 'data: {"delta": "hello"}\n\n'
            + 'data: {"reply": "hello"}\n\n'
            + 'data: {"memory": {"status": "pending"}}\n\n';
  const deltas = [];
  const memory = [];
  const final = await loadStream(sse).widgetStream('/api/widget/stream', {},
    (delta) => deltas.push(delta), () => {}, (status) => memory.push(status));
  assert.deepEqual(deltas, ['hello']);
  assert.equal(final.reply, 'hello');
  assert.equal(memory.length, 1);
  assert.equal(memory[0].status, 'pending');
});

test('the widget keeps its request alive across a page navigation', () => {
  const start = source.indexOf('async function widgetStream');
  const end = source.indexOf('// --- what you can ask');
  assert.match(source.slice(start, end), /keepalive:\s*true/);
});

test('memory decisions render as safe metadata, never as a model answer', () => {
  const ask = source.slice(source.indexOf('async function widgetAsk'),
                           source.indexOf('async function widgetLoadOptions'));
  assert.ok(source.includes('function widgetMemoryStatus(memory)'),
    'the browser needs one small mapping from API memory metadata to reader copy');
  assert.ok(ask.includes('widgetMemoryStatus(memoryStatus)'),
    'the later memory event must be the source of the displayed status');
  assert.ok(ask.includes("widgetSay('meta', copy)"),
    'memory status must be a concise metadata line, not a conversation turn');
  assert.ok(!ask.includes("widgetSay('bot', data.memory"),
    'internal memory state must never be displayed as the model answer');
});

// The sharper edge of the same rule: a stream that dies after a status event
// has still never said what the lab holds. The old reader would have handed
// the status event back as the reply's own final event — an answer the lab
// never gave — where the honest reading is the "stopped before" refusal.
test('a stream that ends on a status event has no final and is refused', async () => {
  const sse = 'data: {"delta": "hi"}\n\n'
            + 'data: {"status": "search_notes"}\n\n';
  await assert.rejects(
    loadStream(sse).widgetStream('/api/widget/stream', {}, () => {}, () => {}),
    /stopped before the lab said what it holds/);
});

// A stage event is the same kind of fact one step out: which step of the run
// is happening, rather than which tool it reached for. It is transient by the
// same contract — no history holds one — so the reader has to route it like a
// status and refuse it as a reply, at both edges: the ordinary stream, where
// the terminal reply is still the only final event, and the stream that dies
// on a stage, where the honest reading is the "stopped before" refusal rather
// than progress text handed back as the answer the lab holds.
test('a stage event is routed to onStage and never becomes the reply', async () => {
  const sse = 'data: {"stage": "retrieving"}\n\n'
            + 'data: {"status": "search_notes"}\n\n'
            + 'data: {"stage": "answering"}\n\n'
            + 'data: {"delta": "hello"}\n\n'
            + 'data: {"reply": "hello"}\n\n';
  const stages = [];
  const statuses = [];
  const final = await loadStream(sse).widgetStream('/api/widget/stream', {},
    () => {}, (status) => statuses.push(status), () => {},
    (stage) => stages.push(stage));
  assert.deepEqual(stages, ['retrieving', 'answering']);
  assert.deepEqual(statuses, ['search_notes']);
  assert.equal(final.reply, 'hello');
  assert.equal(final.stage, undefined);
  // The same stream with no stage callback at all — what a caller that does
  // not care about progress passes — is still refused rather than crashing on
  // the missing callback.
  await assert.rejects(
    loadStream('data: {"stage": "retrieving"}\n\n')
      .widgetStream('/api/widget/stream', {}, () => {}, () => {}),
    /stopped before the lab said what it holds/);
});

// The renderer, one function wide. `widgetSay` is the single door every line in
// the log comes through, and what it decides — which class a kind is allowed to
// write, and whether an unknown one may write anything at all — is a decision
// about escaping as much as about styling, since the log is fed by a route. It
// needs a `#widget-log` and nothing else, so the double is a box that remembers
// what was appended to it rather than a browser.
function loadSay() {
  const written = [];
  const log = {
    scrollTop: 0, scrollHeight: 0,
    querySelector: () => null,
    insertAdjacentHTML: (where, html) => written.push(html),
  };
  const context = {
    written,
    $: (id) => (id === 'widget-log' ? log : null),
    escapeHtml: (text) => String(text).replace(/&/g, '&amp;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;'),
  };
  const say = source.slice(source.indexOf('  // Every kind of line this log'),
                           source.indexOf('  // --- the answer, as it is written'));
  // `WIDGET_KINDS` is a `const`, so it is lexical and never lands on the
  // sandbox's global the way the function declaration beside it does — handed
  // back as the slice's last expression instead of reached for afterwards.
  context.WIDGET_KINDS = runInNewContext(`${say}\n;WIDGET_KINDS`, context);
  return context;
}

// The change this whole capability is: a tool call is a line in the log, not a
// status that vanished. It has to be its own kind — `note` is the lab speaking
// about its own state, and an answer that stands on a real record is a
// different fact from the lab reporting one.
test('a tool call renders as its own kind, distinct from the lab\'s own voice', () => {
  const page = loadSay();
  page.widgetSay('tool', 'called search_knowledge_base');
  page.widgetSay('note', 'No knob was changed.');
  assert.deepEqual(page.written, [
    '<div class="widget-msg tool">called search_knowledge_base</div>',
    '<div class="widget-msg note">No knob was changed.</div>']);
});

// A thread written before this change is replayed through the same loop, and
// nothing about it may move: the rows it does have are the rows it had.
test('a thread with no tool rows renders exactly as it did before', () => {
  const page = loadSay();
  for (const turn of [{ role: 'you', text: 'what decides?' },
                      { role: 'bot', text: 'four judged metrics' }]) {
    page.widgetSay(turn.role, turn.text);
  }
  assert.deepEqual(page.written, [
    '<div class="widget-msg you">what decides?</div>',
    '<div class="widget-msg bot">four judged metrics</div>']);
});

// Every kind the lab can send, and only those. A seventh arriving from a route
// this page has not caught up with must still show the reader what was said —
// dropping it would make their own history disappear — but it may not write a
// class, because the class is the one part of that markup that is not escaped.
test('a kind this page does not know renders inert, and never writes a class', () => {
  const page = loadSay();
  page.widgetSay('stage', 'retrieving');
  page.widgetSay('bot"><script>alert(1)</script><b class="', 'nice try');
  assert.deepEqual(page.written, [
    '<div class="widget-msg">retrieving</div>',
    '<div class="widget-msg">nice try</div>']);
});

// The list itself, pinned. `thinking` is the one live line the log can hold and
// it is deliberately not here: it is built by `widgetThinking`, lives for the
// length of one wait, and is never said — the same ephemerality the status
// events it is fed by keep.
test('the log holds six kinds and the ephemeral line is not one of them', () => {
  const page = loadSay();
  // Spread first: the array was built inside the sandbox's own realm, so it is
  // an Array from another Array — same structure, different prototype, and
  // `deepStrictEqual` compares that too.
  assert.deepEqual([...page.WIDGET_KINDS],
    ['you', 'bot', 'tool', 'meta', 'note', 'err']);
  assert.ok(!page.WIDGET_KINDS.includes('thinking'));
});

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
                           'replyFate(mine, intended, wasPending || drawPending)',
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

// The wait line is ephemeral by construction, and that is a lifecycle claim
// about widgetAsk — named before the post, ended by the first piece, cleared
// again on every way out — which, like the two pins above, is about *where*
// the calls sit and so is read from the source rather than run: the sandbox
// has no DOM to mount an indicator into, and building one to watch a div be
// removed would replace this contract with a fake browser.
test('the thinking line is born with the post and cannot outlive the turn', () => {
  const ask = source.slice(source.indexOf('async function widgetAsk'),
                           source.indexOf('async function widgetLoadOptions'));
  const born = ask.indexOf('const thinking = widgetThinking()');
  const posted = ask.indexOf('await widgetStream(');
  assert.ok(born > -1 && born < posted,
    'the wait must be named before the post, or the reader watches nothing');
  const delta = ask.slice(ask.indexOf('(delta) =>'), ask.indexOf('(status) =>'));
  assert.ok(delta.includes('widgetThinkingOver(thinking)'),
    'the first piece must end the wait it answers');
  const out = ask.slice(ask.indexOf('} catch'));
  assert.ok((out.match(/widgetThinkingOver\(thinking\)/g) || []).length >= 2,
    'both the catch and the finally must clear a wait no delta ever ended');
  const over = source.slice(source.indexOf('function widgetThinkingOver'),
                            source.indexOf('// One SSE reader'));
  assert.ok(over.includes('contains(el)'),
    'removal must re-ask whether the log still holds the line — the same '
    + 'contains check every other writer to a moved-on log makes');
});

// The swap to "calling <tool>…" answers to the same rule the live bubble
// does: a redraw that wiped the line means the log has moved on, and the
// status handler must fall silent rather than recreate what a draw removed.
test('a status event retitles the indicator only while the log still holds it', () => {
  const ask = source.slice(source.indexOf('async function widgetAsk'),
                           source.indexOf('async function widgetLoadOptions'));
  const status = ask.slice(ask.indexOf('(status) =>'), ask.indexOf('const fate'));
  assert.ok(status.includes('contains(thinking)'),
    'the swap must check the indicator is still on screen');
  assert.ok(status.includes('textContent'),
    'the tool name is untrusted stream data and must be assigned as text, '
    + 'never as markup');
  assert.ok(!status.includes('widgetThinking()'),
    'a wiped indicator is never recreated — the log has moved on');
  assert.ok(!ask.includes("widgetSay('thinking'") && !ask.includes('widgetSay(\'meta\', `calling'),
    'no status text goes through widgetSay: it is not a turn and must never '
    + 'look like one');
});

// The stage label lands in the one wait line the status swap already writes to
// — retitled, not joined by a second line or a second bubble — under the same
// rule: only a line the log still holds is this turn's to retitle, a wiped one
// stays gone, and the label is text because it arrived over the stream. The
// line is a live region so a screen reader hears the step change, and it is
// still ephemeral: `widgetThinkingOver` in the catch and the finally is what
// clears it on the reply, an error, a disconnect and a reload alike, which the
// lifecycle test above pins.
test('a stage event retitles the same wait line, and that line is announced', () => {
  const ask = source.slice(source.indexOf('async function widgetAsk'),
                           source.indexOf('async function widgetLoadOptions'));
  const opens = ask.indexOf('(stage) =>');
  const stage = ask.slice(opens, ask.indexOf('const fate', opens));
  assert.ok(stage.includes('contains(thinking)'),
    'the retitle must check the wait line is still on screen');
  assert.ok(stage.includes('thinking.textContent = stage'),
    'the label is untrusted stream data assigned as text onto the existing '
    + 'line — never markup, and never a second line');
  assert.ok(!stage.includes('widgetThinking()') && !stage.includes('widgetSay('),
    'a wiped line is never recreated and no stage text becomes a turn');
  const born = source.slice(source.indexOf('function widgetThinking()'),
                            source.indexOf('function widgetThinkingOver'));
  assert.match(born, /role', 'status'/);
  assert.match(born, /aria-live', 'polite'/);
});
