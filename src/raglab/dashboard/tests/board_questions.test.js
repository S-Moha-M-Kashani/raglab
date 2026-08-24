// tests/board_questions.test.js — the board's `questions` column, and what it
// reads off one row.
//
// Contract under test: `cell(row, 'questions')` and the `questions` column's
// own `data-sort` in `renderTable`, at the top of
// `dashboard/frontend/leaderboard.js`.
//
// The column used to be a bare `row.n_questions` — a count with no identity.
// `evaluation/leaderboard.py`'s `experiment_record` now carries the run's own
// recorded selection note (`run_evaluation.selection_note`) on the row as
// `selection`, and this is the page's reading of it: the note when a row
// carries one, the bare count when it does not — never both, and the sort key
// stays numeric either way so a richer cell text cannot scramble the column's
// order. Evaluated in a `vm` context, the way board_reveal.test.js and
// sorttable.test.js do, against the smallest sandbox that lets the served
// file finish loading.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (name) => readFileSync(join(HERE, '../frontend', name), 'utf8');
const SOURCE = `${read('lab.js')}\n${read('leaderboard.js')}`;

// The smallest sandbox that lets the file finish loading — `loadBoard` runs on
// load and reads into `#board`, and fails harmlessly into its own error card
// because `fetch` is absent, exactly as board_reveal.test.js's does.
function loadPage() {
  const sandbox = {
    document: {
      addEventListener() {},
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
    },
    getComputedStyle: () => ({ position: 'fixed' }),
    URLSearchParams,
  };
  runInNewContext(SOURCE, sandbox);
  return sandbox;
}

// A row shaped the way `evaluation.leaderboard.experiment_record` projects
// one — only the fields `cell(row, 'questions')` and the sort key actually
// read, plus enough for `renderTable` not to choke on the rest of the row.
function row(overrides = {}) {
  return {
    experiment_id: 'exp-1', kind: 'run', state: 'done', error: '',
    label: '', started_at: '2026-08-01 10:00:00', seconds: 12,
    dataset: 'diary-fa', provider: 'fake', n_questions: 0, selection: {},
    decision: null, decision_stderr: null, metrics: {}, judge: {},
    pipeline: [], config: {},
    ...overrides,
  };
}

// This is a unit test.
test('a row with a balanced selection note shows the note, not the count',
  () => {
    const { cell } = loadPage();
    const balanced = row({
      n_questions: 9,
      selection: { balance: 'difficulty', limit: 9, n: 9,
                  by_difficulty: { easy: 3, medium: 3, hard: 3 },
                  question_ids: ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7',
                                'q8', 'q9'] },
    });
    assert.equal(cell(balanced, 'questions'), '9 (easy=3, medium=3, hard=3)');
  });

// This is a unit test.
test('a row with a strided (unbalanced) selection note shows n and the '
  + 'label, with no counts to name', () => {
    const { cell } = loadPage();
    const strided = row({
      n_questions: 5,
      selection: { balance: '', limit: 5, n: 5,
                  question_ids: ['q1', 'q2', 'q3', 'q4', 'q5'] },
    });
    assert.equal(cell(strided, 'questions'), '5');
  });

// This is a unit test.
test('a row with no selection note at all falls back to the bare count',
  () => {
    const { cell } = loadPage();
    // A ledger-only row (an index build, a retrieval, an imported archive)
    // never scored a question, and a run predating `selection_note` recorded
    // none either — both arrive here as `selection: {}`.
    const noNote = row({ n_questions: 24, selection: {} });
    assert.equal(cell(noNote, 'questions'), '24');
  });

// This is a unit test.
test('the questions column sorts on the number, not on the note\'s text',
  () => {
    const { renderTable } = loadPage();
    // "12 (...)" must sort as 12, next to a bare "9" sorting as 9 — not
    // alphabetically, where "12" comes before "9".
    const rows = [
      row({ experiment_id: 'a', n_questions: 12,
            selection: { balance: 'difficulty', n: 12,
                        by_difficulty: { easy: 4, medium: 4, hard: 4 } } }),
      row({ experiment_id: 'b', n_questions: 9, selection: {} }),
    ];
    const html = renderTable('diary-fa', rows);
    const sorts = [...html.matchAll(/data-sort="(\d+)"/g)].map((m) => m[1]);
    assert.deepEqual(sorts, ['12', '9'],
      'expected one numeric data-sort per row, in row order, on the '
      + 'questions column');
    // And the cell text itself still carries the richer note.
    assert.ok(html.includes('12 (easy=4, medium=4, hard=4)'));
    assert.ok(html.includes('>9<'));
  });
