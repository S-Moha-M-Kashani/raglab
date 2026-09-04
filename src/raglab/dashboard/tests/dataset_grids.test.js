// tests/dataset_grids.test.js — the dataset viewer's columns, its narrowing,
// and its raw tree.
//
// Contract under test: `documentsGrid`, `partsGrid`, `renderGrid` and `tree`
// in `dashboard/frontend/dataset.js`. Three claims that can only be made
// against the served file, because the page draws itself from one JSON payload
// and an offline test of its markup sees `<div id="dataset">` and nothing else:
//
//   1. every column is one of the dataset's own declared labels — a corpus
//      declaring `speaker` gets a `speaker` column, and a label this corpus
//      never declared gets no column at all;
//   2. a reading narrows its grid to exactly the rows it named;
//   3. the raw tree shows a value as recorded, whitespace included.
//
// Evaluated in a `vm` context, the way board_questions.test.js does, against
// the smallest sandbox that lets the served file finish loading.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (name) => readFileSync(join(HERE, '../frontend', name), 'utf8');
const SOURCE = `${read('lab.js')}\n${read('dataset.js')}`;

// `load()` runs on load and fails harmlessly into its own error card because
// `fetch` is absent — the same way the board's suite lets `loadBoard` fail.
function loadPage() {
  const sandbox = {
    document: {
      addEventListener() {},
      getElementById: () => ({ innerHTML: '', querySelectorAll: () => [] }),
      querySelector: () => null,
      querySelectorAll: () => [],
      createElement: () => ({ style: {}, setAttribute() {}, appendChild() {},
                              addEventListener() {}, classList: { add() {} } }),
      body: { appendChild() {} },
    },
    window: {
      addEventListener() {},
      innerHeight: 1000,
      innerWidth: 1400,
      location: { search: '', href: 'http://localhost:9002/dataset' },
      history: { replaceState() {} },
    },
    getComputedStyle: () => ({ position: 'fixed' }),
    URL,
    URLSearchParams,
    setTimeout,
    clearTimeout,
  };
  runInNewContext(SOURCE, sandbox);
  return sandbox;
}

// A payload shaped the way `/api/dataset-content/<id>` serves one: the two
// files as they are on disk, plus the declaration table the route derives.
// This corpus deliberately declares neither of the diary's label names — a
// `speaker` on parts and a `mood` on documents — because that is the claim.
function payload() {
  return {
    dataset: {
      id: 'other', name: 'Another corpus', documents: 2, parts: 2,
      questions: 0, source: 'imported', description: '',
      label_declarations: [
        { name: 'mood', type: 'string', levels: [], applies_to: ['document'],
          extracted: false, confidence_for: '' },
        { name: 'speaker', type: 'string', levels: ['host', 'guest'],
          applies_to: ['part'], extracted: false, confidence_for: '' },
      ],
      question_label_declarations: [],
    },
    corpus: {
      corpus_dataset_metadata: { dataset: 'other' },
      corpus_documents: [
        { corpus_document_id: 1, document_metadata: { mood: 'flat' },
          document_content: [{ text: 'one line\n\nand another',
                               labels: { speaker: 'host' } }] },
        { corpus_document_id: 2, document_metadata: { mood: 'bright' },
          document_content: [{ text: 'nobody asks about this',
                               labels: { speaker: 'guest' } }] },
      ],
    },
    ground_truth: { groundtruth_dataset: [] },
    ground_truth_error: '',
    readings: [],
  };
}

// Joined rather than compared as arrays: an array built inside the `vm`
// context has that context's own `Array` prototype, and a strict deep
// comparison against a host array fails on the prototype alone.
const columnNames = (grid) => grid.columns.map((column) => column.label).join(' ');

test('a part-level label the corpus declares names a column of its own', () => {
  const page = loadPage();
  const parts = page.partsGrid(payload(), '1');
  assert.equal(columnNames(parts), 'part speaker text');
  assert.ok(!columnNames(parts).includes('role'),
    'a label this corpus never declared must have no column');
  // And a document-level label describes documents, not parts: `applies_to`
  // is the file's own statement of where a label may land.
  assert.equal(columnNames(page.documentsGrid(payload(), null)),
    'document parts chars mood');
});

test('a corpus declaring no part-level label still shows its parts', () => {
  const page = loadPage();
  const data = payload();
  data.dataset.label_declarations = [];
  const parts = page.partsGrid(data, '1');
  assert.equal(columnNames(parts), 'part text',
    'position and text alone — never a placeholder column for a label the '
    + 'corpus lacks');
  assert.equal(parts.rows.length, 1);
});

test('a reading narrows its grid to exactly the rows it named', () => {
  const page = loadPage();
  const grid = page.documentsGrid(payload(), null);
  const identities = (html) => (html.match(/data-document="\d+"/g) || []);
  assert.equal(identities(page.renderGrid(grid, null)).length, 2);
  // Identifiers arrive from the route as JSON numbers and are compared as
  // strings, which is what a cell holds.
  const narrowed = page.renderGrid(grid, ['2']);
  assert.equal(identities(narrowed).length, 1);
  assert.ok(narrowed.includes('data-document="2"'));
  assert.ok(!narrowed.includes('data-document="1"'));
});

test('the raw tree shows a value as recorded and collapses past the first level', () => {
  const page = loadPage();
  const html = page.tree(payload().corpus, 'other_corpus.json', 0);
  // The blank line inside a part is the reason this view exists: a grid shows
  // the sentence, and only this shows that there are two newlines in it.
  assert.ok(html.includes('one line\\n\\nand another'),
    'a value must be shown as recorded, not reformatted into two lines');
  assert.equal((html.match(/<details open>/g) || []).length, 1,
    'only the file itself opens; everything under it is collapsed');
  assert.ok(html.includes('<details>'), 'and the levels below it are there');
});
