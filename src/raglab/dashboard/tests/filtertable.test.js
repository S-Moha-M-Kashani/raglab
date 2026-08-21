// tests/filtertable.test.js — the leaderboard's row filter.
//
// Contract under test: `dashboard/frontend/filtertable.js`, loaded as a plain
// script by the leaderboard (:9002) beside `sorttable.js`, whose `cellKey` it
// reads a cell with. Filtering and sorting are one question asked twice — which
// rows, in what order — so a filtered cell and a sorted cell must be understood
// identically. A filter that read '0.747 ± 0.042' as text while the sorter read
// it as 0.747 would answer `decision>0.7` with a row the column ordered above
// it.
//
// What is tested here is the part that decides whether a row is *shown*: the
// query a reader types, and the answer for one row. The DOM half (which `tr`
// gets `hidden`) is one loop over these two, and is verified in a browser.
//
//   parse(query)              -> { terms, bad }
//   matches(terms, row)       -> boolean
//   unknown(terms, names)     -> column names this table does not have
//
// Evaluated in a `vm` context, the way sorttable.test.js does, and given the
// real `sorttable.js` as its neighbour rather than a fake of it: the point of
// the file is that one parser reads every cell.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (name) => readFileSync(join(HERE, '../frontend', name), 'utf8');
const SANDBOX = runInNewContext(
  `${read('sorttable.js')}\n${read('filtertable.js')}\n;FilterTable`, {});

// The vm context has its own `Array`, so an array it built is not deep-strict-
// equal to an identical one built here however identical its contents. Every
// list crossing the boundary is copied into this realm at the boundary, once,
// rather than at each of the thirty assertions below.
const FilterTable = {
  ...SANDBOX,
  parse: (query) => {
    const { terms, bad } = SANDBOX.parse(query);
    return { terms: [...terms], bad: [...bad] };
  },
  unknown: (terms, names) => [...SANDBOX.unknown(terms, names)],
};

// One leaderboard row as the page renders it: the cell text under each column's
// filter name. `state` is deliberately the marked-up kind ('failed' with its
// '!' button) reduced to its text, which is what a filter sees.
const ROW = {
  pipeline: 'sem-drift·louv·ST·MiniLM-L12 · rrf·lex·llm · llm',
  decision: '0.7412',
  spread: '0.031',
  faith: '0.880',
  'ctx-recall': '—',
  kind: 'run',
  when: '2026-08-04 16:00',
  label: 'hybrid vs dense',
  judge: 'claude-sonnet-4-5 via anthropic',
  questions: '40',
  provider: 'anthropic',
  state: 'done',
  seconds: '312',
};

const NAMES = Object.keys(ROW);

// The whole row as one string, which is what a bare word searches.
const rowOf = (cells) => ({
  cells,
  text: Object.values(cells).join(' '),
});

const asks = (query, cells = ROW) => {
  const { terms, bad } = FilterTable.parse(query);
  assert.deepEqual(bad, [], `nothing malformed in: ${query}`);
  assert.deepEqual(FilterTable.unknown(terms, NAMES), [],
    `every column named exists: ${query}`);
  return FilterTable.matches(terms, rowOf(cells));
};

// This is a unit test.
test('an empty query shows every row', () => {
  for (const query of ['', '   ', '\t']) {
    const { terms } = FilterTable.parse(query);
    assert.deepEqual(terms, [], `nothing to ask: ${JSON.stringify(query)}`);
    assert.equal(FilterTable.matches(terms, rowOf(ROW)), true);
  }
});

// This is a unit test.
test('a number in a cell is compared as a number, not as text', () => {
  // The four the reader asked for, and the direction each has to fail in.
  assert.equal(asks('questions>30'), true);
  assert.equal(asks('questions>40'), false);
  assert.equal(asks('questions>=40'), true);
  assert.equal(asks('decision>0.6'), true);
  assert.equal(asks('decision>0.8'), false);
  assert.equal(asks('seconds<400'), true);
  assert.equal(asks('seconds<=312'), true);
  // Text order would put '40' below '9': the whole reason this is numeric.
  assert.equal(asks('questions>9'), true);
  assert.equal(asks('questions<9'), false);
});

// This is a unit test.
test('a score carrying its error is compared by the score', () => {
  // The same cell `sorttable` orders by 0.747 rather than by its first
  // character, read through the same parser.
  const cells = { ...ROW, decision: '0.747 ± 0.042' };
  assert.equal(asks('decision>0.7', cells), true);
  assert.equal(asks('decision>0.8', cells), false);
});

// This is a unit test.
test('a timestamp compares chronologically', () => {
  // Not as a number — 2026 is not what '2026-08-04 16:00' means — but as its
  // own zero-padded text, where lexical order is chronological order.
  assert.equal(asks('when>2026-08-01'), true);
  assert.equal(asks('when>2026-09-01'), false);
  assert.equal(asks('when<2026-08-05'), true);
  assert.equal(asks('when>=2026-08-04'), true);
});

// This is a unit test.
test('text equality ignores case, and != is its opposite', () => {
  assert.equal(asks('state=done'), true);
  assert.equal(asks('state=DONE'), true, 'a state is not typed in one case');
  assert.equal(asks('state=failed'), false);
  assert.equal(asks('state!=failed'), true);
  assert.equal(asks('state!=done'), false);
  // Equality on a figure is the figure's, so a rendered 0.7412 answers 0.7412
  // however either side is written.
  assert.equal(asks('questions=40'), true);
  assert.equal(asks('questions=40.0'), true);
});

// This is a unit test.
test('a substring term matches part of a cell', () => {
  assert.equal(asks('judge~sonnet'), true);
  assert.equal(asks('judge~gpt'), false);
  assert.equal(asks('judge!~gpt'), true);
  assert.equal(asks('judge!~sonnet'), false);
  // ':' is the forgiving spelling of the same thing, because it is what a
  // reader types first.
  assert.equal(asks('kind:run'), true);
  assert.equal(asks('kind:index'), false);
});

// This is a unit test.
test('a bare word searches the whole row', () => {
  assert.equal(asks('hybrid'), true);
  assert.equal(asks('anthropic'), true);
  assert.equal(asks('chromadb'), false);
  // And a bare word can be excluded, which is how a reader drops a family of
  // rows without knowing which column names it.
  assert.equal(asks('!chromadb'), true);
  assert.equal(asks('!anthropic'), false);
});

// This is a unit test.
test('a quoted value keeps its spaces', () => {
  assert.equal(asks('label~"hybrid vs"'), true);
  assert.equal(asks('label~"hybrid or"'), false);
  assert.equal(asks('"hybrid vs dense"'), true);
});

// This is a unit test.
test('a column can be asked whether it was measured at all', () => {
  assert.equal(asks('faith:'), true, 'faithfulness was measured');
  assert.equal(asks('ctx-recall:'), false, 'context recall was not');
  assert.equal(asks('!ctx-recall:'), true, 'and that is what !col: finds');
  assert.equal(asks('!faith:'), false);
});

// This is a unit test.
test('a term about a column a row never measured does not match it', () => {
  // A dash is 'never measured', not a low value — the same reading that puts a
  // missing cell last in both sort directions. So a row missing the column is
  // no answer to a question about it, in either direction: `ctx-recall!=0.5`
  // must not quietly hand back every ungraded row.
  for (const query of ['ctx-recall>0.1', 'ctx-recall<0.1', 'ctx-recall=0.5',
    'ctx-recall!=0.5', 'ctx-recall~0', 'ctx-recall!~0']) {
    assert.equal(asks(query), false, query);
  }
});

// This is a unit test.
test('terms are ANDed, so each one can only narrow the table', () => {
  assert.equal(asks('state!=failed questions>30 decision>0.6'), true);
  assert.equal(asks('state!=failed questions>30 decision>0.9'), false);
  assert.equal(asks('kind:run judge~sonnet faith: hybrid'), true);
  assert.equal(asks('kind:run judge~sonnet ctx-recall:'), false);
});

// This is a unit test.
test('a column this table does not have is reported, not guessed at', () => {
  const { terms, bad } = FilterTable.parse('faith>0.5 recall>0.5 quesitons>30');
  assert.deepEqual(bad, []);
  // Named in the order typed, so the message points at the first mistake.
  assert.deepEqual(FilterTable.unknown(terms, NAMES), ['recall', 'quesitons']);
  // A bare word is not a column name and is never reported as one.
  assert.deepEqual(FilterTable.unknown(FilterTable.parse('recall').terms, NAMES),
    []);
});

// This is a unit test.
test('a malformed term is reported rather than silently dropped', () => {
  // An operator with nothing to compare against asks nothing. Reported, so the
  // reader is not left believing a table filtered on it.
  for (const query of ['questions>', 'decision>=', 'state!=', 'judge~', ':done',
    '>30']) {
    const { bad } = FilterTable.parse(query);
    assert.deepEqual(bad, [query.trim()], query);
  }
});

// This is a unit test.
test('a numeric operator on a word compares the words', () => {
  // Not a silent no: '>' on a text column is a question with an answer, and
  // the answer readers expect is alphabetical. `state>c` keeps 'done'.
  assert.equal(asks('state>c'), true);
  assert.equal(asks('state>e'), false);
});
