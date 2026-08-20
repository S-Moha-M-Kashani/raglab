// tests/sorttable.test.js — the column sorter both RAG lab panels share.
//
// Contract under test: `dashboard/frontend/sorttable.js`, loaded as a plain
// script by the lab (:9002) and the Inspector (:9003) — the two pages are served
// out of the same directory, which is what lets one file be the single answer to
// "what does clicking a header do". Its DOM half is verified in a real
// browser; what is tested here is the part that decides an *order*, because that
// is where a sorter is wrong in ways nobody notices:
//
//   cellKey(text)          -> { missing, number, text }
//   compare(a, b, dir)     -> negative / zero / positive, missing always last
//
// Evaluated in a `vm` context rather than imported, so the browser file needs no
// module wrapper it would otherwise carry only for this test.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE = readFileSync(
  join(HERE, '../frontend/sorttable.js'), 'utf8');
const SortTable = runInNewContext(SOURCE + '\n;SortTable', {});

// Every cell the two panels actually render, with what it has to sort as.
const CELLS = [
  ['0.747', 0.747, 'a bare score'],
  // The leaderboard renders the deciding score with its standard error inside
  // the same cell. Sorting on the text would put 0.9 below 0.75 ± 0.1.
  ['0.747 ± 0.042', 0.747, 'a score carrying its error'],
  ['± 0.042', 0.042, 'an error on its own'],
  ['-0.5', -0.5, 'a negative'],
  ['1,204', 1204, 'a thousands separator'],
  ['83%', 83, 'a percentage'],
  ['12s', 12, 'a duration with its unit'],
  ['2026-08-04 16:00:08', null, 'a timestamp is text, not the year 2026'],
  ['fixed-overlap', null, 'a strategy name'],
  ['C-017', null, 'a ledger id keeps its letters'],
];

// This is a unit test.
test('a number inside a rendered cell sorts as a number', () => {
  for (const [text, expected, what] of CELLS) {
    assert.equal(SortTable.cellKey(text).number, expected, `${what}: ${text}`);
  }
  // A timestamp must sort chronologically all the same, which its own text does:
  // the format is zero-padded, so lexical order is chronological order. Reading
  // 2026 out of it as a number would sort every row in one day as equal.
  const stamps = ['2026-08-04 09:05:00', '2026-08-04 16:00:08',
    '2026-08-03 23:59:59'];
  assert.deepEqual([...stamps].sort((a, b) => SortTable.compare(
    SortTable.cellKey(a), SortTable.cellKey(b), 1)),
    ['2026-08-03 23:59:59', '2026-08-04 09:05:00', '2026-08-04 16:00:08']);
});

// This is a unit test.
test('a cell with no value sorts last in both directions', () => {
  // This is the whole reason the sorter is not two lines. A dash means "this was
  // never measured" — every unjudged run, every retrieval, every build — and a
  // dash that sorts as 0 or as the string '—' takes over one end of the table.
  // Ascending, it would beat every real score; descending, it would look like a
  // ranking of the rows that measured least.
  for (const blank of ['—', '', '  ', '·', 'n/a']) {
    const key = SortTable.cellKey(blank);
    assert.ok(key.missing, `${JSON.stringify(blank)} is a missing value`);
    for (const dir of [1, -1]) {
      assert.ok(SortTable.compare(key, SortTable.cellKey('0.001'), dir) > 0,
        `${JSON.stringify(blank)} sorts after a real number going ${dir}`);
      assert.ok(SortTable.compare(SortTable.cellKey('zebra'), key, dir) < 0,
        `${JSON.stringify(blank)} sorts after text going ${dir}`);
    }
  }
  // Two missing values are equal, so a stable sort leaves them in served order.
  assert.equal(SortTable.compare(SortTable.cellKey('—'),
    SortTable.cellKey('·'), 1), 0);
});

// This is a unit test.
test('text sorts by locale, and a number never sorts as text', () => {
  const words = ['semantic-drift', 'Fixed', 'char-hash', 'ascii-hash'];
  assert.deepEqual([...words].sort((a, b) => SortTable.compare(
    SortTable.cellKey(a), SortTable.cellKey(b), 1)),
    // Case-insensitively, so 'Fixed' files under f and not before every
    // lower-case name — the lab's own labels are typed by hand and mix case.
    ['ascii-hash', 'char-hash', 'Fixed', 'semantic-drift']);
  // The failure a text sort would produce on the column that decides the
  // architecture: '0.9' beats '0.75' numerically and loses lexically.
  const scores = ['0.75', '0.9', '0.125'];
  assert.deepEqual([...scores].sort((a, b) => SortTable.compare(
    SortTable.cellKey(a), SortTable.cellKey(b), -1)),
    ['0.9', '0.75', '0.125']);
});

// This is a unit test.
test('the sorter is stable, so a tie keeps the order it was served in', () => {
  // Ties are not an edge case here: a leaderboard group can hold a dozen rows
  // whose column is identical, and the served order is itself a ranking. A sort
  // that shuffled equal rows would make the same click produce a different table.
  const rows = [['a', '0.5'], ['b', '0.5'], ['c', '0.9'], ['d', '0.5']];
  const sorted = [...rows].sort((x, y) => SortTable.compare(
    SortTable.cellKey(x[1]), SortTable.cellKey(y[1]), -1));
  assert.deepEqual(sorted.map((r) => r[0]), ['c', 'a', 'b', 'd']);
});

// `SortTable.make` is a DOM function, and this file's harness runs sorttable.js
// in an empty `vm` context with no DOM at all. Rather than pull in a real
// document, this is the minimal fake covering exactly what `make` touches: a
// `th` that records attributes and stores its listeners, a body whose
// `appendChild` reorders (removes the row from wherever it sits, then pushes
// it to the end — that's what makes `body.rows` reflect a sort), and a table
// wiring the two together.
function fakeHeader(text) {
  const attrs = {};
  const listeners = {};
  return {
    textContent: text,
    hasAttribute: (name) => Object.prototype.hasOwnProperty.call(attrs, name),
    getAttribute: (name) => (Object.prototype.hasOwnProperty.call(attrs, name)
      ? attrs[name] : null),
    setAttribute: (name, value) => { attrs[name] = String(value); },
    classList: { add: () => {} },
    tabIndex: undefined,
    title: undefined,
    addEventListener: (type, fn) => {
      (listeners[type] = listeners[type] || []).push(fn);
    },
    // Not a real DOM method signature, but named the same: fires whatever
    // `make` wired to 'click', which is all a test needs to trigger a cycle.
    click: () => { for (const fn of (listeners.click || [])) fn(); },
  };
}

function fakeCell(text) {
  const attrs = {};
  return {
    textContent: text,
    getAttribute: (name) => (Object.prototype.hasOwnProperty.call(attrs, name)
      ? attrs[name] : null),
  };
}

function buildTable(columnNames, dataRows) {
  const head = { cells: columnNames.map(fakeHeader) };
  const rows = dataRows.map((cells) => ({ cells: cells.map(fakeCell) }));
  const body = {
    rows,
    appendChild(row) {
      const at = this.rows.indexOf(row);
      if (at !== -1) this.rows.splice(at, 1);
      this.rows.push(row);
    },
  };
  return {
    tHead: { rows: [head] },
    tBodies: [body],
    classList: { add: () => {} },
  };
}

// This is a unit test.
test('onApply reports the displayed order on wiring, on sort, and on the '
  + 'third-click restore', () => {
  const table = buildTable(
    ['pipeline', 'decision'],
    [['a', '0.40'], ['b', '0.90'], ['c', '0.10']]);
  const seen = [];
  // `Array.from`, not `.map`: both `rows` (the callback argument) and
  // `table.tBodies[0].rows` are arrays from the vm sandbox sorttable.js runs
  // in, so their own `.map` would build its result via that sandbox's own
  // `Array`, and comparing it against a plain array literal here fails
  // deepStrictEqual on realm identity alone, despite identical contents.
  // `Array.from` called on this side's `Array` sidesteps that.
  const bodyOrder = () => Array.from(
    table.tBodies[0].rows, (r) => r.cells[0].textContent);
  SortTable.make(table, {
    onApply: (rows) => seen.push(Array.from(rows, (r) => r.cells[0].textContent)),
  });
  // Fires once on wiring, so a rank column starts correct rather than
  // correct-after-the-first-click. The body itself was never touched to get
  // there — nothing needed moving to already match the served order.
  assert.equal(seen.length, 1);
  assert.deepStrictEqual(seen.at(-1), ['a', 'b', 'c']);
  assert.deepStrictEqual(bodyOrder(), ['a', 'b', 'c']);

  const [pipelineHead, decisionHead] = table.tHead.rows[0].cells;
  decisionHead.click();
  assert.deepStrictEqual(seen.at(-1), ['b', 'a', 'c']);
  // The callback's argument is only meaningful if the actual DOM agrees —
  // this is what would have caught a shim (or a browser) that computed the
  // right answer for onApply but never actually moved the rows.
  assert.deepStrictEqual(bodyOrder(), ['b', 'a', 'c']);
  assert.equal(decisionHead.getAttribute('aria-sort'), 'descending');
  assert.equal(pipelineHead.getAttribute('aria-sort'), 'none');

  decisionHead.click();
  assert.deepStrictEqual(seen.at(-1), ['c', 'a', 'b']);
  assert.deepStrictEqual(bodyOrder(), ['c', 'a', 'b']);
  assert.equal(decisionHead.getAttribute('aria-sort'), 'ascending');

  // The third click restores the served order, and onApply must say so — a
  // rank column left showing the reversed numbering there would be a column
  // lying. Same for the body itself.
  decisionHead.click();
  assert.deepStrictEqual(seen.at(-1), ['a', 'b', 'c']);
  assert.deepStrictEqual(bodyOrder(), ['a', 'b', 'c']);
  assert.equal(decisionHead.getAttribute('aria-sort'), 'none');
});

// This is a unit test.
test('make still works with no options at all, which is how both panels call it',
  () => {
    const table = buildTable(['x'], [['1'], ['2']]);
    assert.doesNotThrow(() => SortTable.make(table));
  });

// This is a unit test.
test('a column can say which way it opens, because "best" is not always highest', () => {
  // The generic rule leads with the highest number, which is right for a score and
  // wrong for a rank: rank 1 is the best one. Observed in the browser on
  // 2026-08-04 — clicking the Inspector's dense column led with rank 46, the
  // candidate dense liked least. `data-order` on the header carries the exception,
  // so the knowledge lives where the column is declared.
  const th = (attrs) => ({
    getAttribute: (name) => (name in attrs ? attrs[name] : null),
  });
  assert.equal(SortTable.opening(th({ 'data-order': 'ascending' }), true), 1);
  assert.equal(SortTable.opening(th({ 'data-order': 'descending' }), false), -1);
  // Unmarked: numbers lead with the highest, text reads A to Z.
  assert.equal(SortTable.opening(th({}), true), -1);
  assert.equal(SortTable.opening(th({}), false), 1);
});
