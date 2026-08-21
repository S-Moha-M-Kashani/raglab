// tests/board_reveal.test.js — the board's settings reveal, opened and closed.
//
// Contract under test: the reveal wiring at the foot of
// `dashboard/frontend/leaderboard.js`, loaded as a plain script by the
// leaderboard (:9002) together with `lab.js`, which holds `placeReveal`.
//
// What is tested here is which reveal is open after a sequence of events,
// because that is where this mechanism is wrong in ways nobody notices. It
// stopped being a pair of CSS rules when the box moved into the top layer — a
// `popover` cannot be opened by a selector — so hover, focus and the browser's
// own closing of a popover are now decided in script, and a test that only
// greps the source for the word `focusin` would pass with focus ignored
// entirely.
//
// Evaluated in a `vm` context, the way sorttable.test.js does, against the
// smallest fake of what the two files actually touch: the served files need no
// module wrapper they would otherwise carry only for this test.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (name) => readFileSync(join(HERE, '../frontend', name), 'utf8');
const SOURCE = `${read('lab.js')}\n${read('leaderboard.js')}`;

// One pipeline cell, its reveal, and somewhere else to hover or focus. The
// reveal answers `closest('td.freeze-1')` with the cell, because in the
// document it is a child of that cell however far outside it the top layer
// paints it — which is the whole reason the page reads the cell off an event's
// target rather than off `:hover`.
function board() {
  const listeners = {};
  const reveal = {
    open: false,
    style: {},
    offsetHeight: 460,
    offsetWidth: 520,
    matches: (selector) => (selector === ':popover-open' ? reveal.open : false),
    showPopover() {
      assert.equal(reveal.open, false, 'showPopover on an already-open popover');
      reveal.open = true;
    },
    hidePopover() {
      assert.equal(reveal.open, true, 'hidePopover on an already-closed popover');
      reveal.open = false;
    },
    closest: (selector) => (selector === 'td.freeze-1' ? cell : null),
  };
  const cell = {
    isConnected: true,
    querySelector: (selector) =>
      (selector === '.settings-reveal' ? reveal : null),
    closest: (selector) => (selector === 'td.freeze-1' ? cell : null),
    getBoundingClientRect: () => ({
      top: 600, bottom: 632, left: 100, right: 580,
    }),
  };
  // Anything that is not a pipeline cell and not inside one: another column's
  // cell, the intro card, the top bar.
  const elsewhere = { closest: () => null };

  const sandbox = {
    document: {
      addEventListener(type, fn) {
        (listeners[type] = listeners[type] || []).push(fn);
      },
      // `loadBoard` runs on load and writes into #board; nothing else here is
      // reached, because `fetch` is absent and the read fails into its own
      // error card.
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

  const fire = (type, event) => {
    const wired = listeners[type] || [];
    assert.ok(wired.length, `nothing listens for ${type}`);
    for (const fn of wired) fn({ target: null, relatedTarget: null, ...event });
  };
  return { fire, reveal, cell, elsewhere };
}

// This is a unit test.
test('a reveal opened by a keyboard survives the mouse going elsewhere', () => {
  const { fire, reveal, cell, elsewhere } = board();
  fire('focusin', { target: cell });
  assert.ok(reveal.open, 'focusing a pipeline cell opens its reveal');
  assert.ok(reveal.style.top, 'and the fixed box is told where to go');
  // Hover and focus are separate states, so a pointer moving over the intro
  // card says nothing about a panel the keyboard opened.
  fire('mouseover', { target: elsewhere });
  assert.ok(reveal.open, 'the mouse moving away does not close a focused reveal');
});

// This is a unit test.
test('tabbing into the panel does not close the panel', () => {
  // The panel is a scrollable box, which makes it a tab stop of its own — so
  // the tab after the cell moves focus *into* it, and `focusout` fires on the
  // cell with the panel as its `relatedTarget`. Closing on that hides the very
  // thing focus is arriving at: focus lands on nothing, the keypress is dead,
  // and the 58% of the panel below its own fold can never be reached.
  const { fire, reveal, cell, elsewhere } = board();
  fire('focusin', { target: cell });
  fire('focusout', { target: cell, relatedTarget: reveal });
  assert.ok(reveal.open, 'focus moving into the reveal is not focus leaving it');
  fire('focusin', { target: reveal });
  assert.ok(reveal.open, 'and it stays open with focus inside it');
  // Out the other side: focus leaves both the cell and the panel.
  fire('focusout', { target: reveal, relatedTarget: elsewhere });
  assert.equal(reveal.open, false, 'and closes when focus really does leave');
});

// This is a unit test.
test('a popover the browser closed is opened again, not assumed open', () => {
  // Sorting moves a row, which takes it out of the document and back in, and
  // the browser closes any popover inside it on the way. A page that believed
  // its own record of what is open left the cell that owns it unable to open it
  // ever again: hover it, and both branches short-circuit on a box that is not
  // on screen. Observed by clicking a column heading with a reveal open.
  const { fire, reveal, cell } = board();
  fire('mouseover', { target: cell });
  assert.ok(reveal.open);
  reveal.open = false;                    // what the sort did, behind the page
  fire('mouseover', { target: cell });
  assert.ok(reveal.open, 'the same cell opens its reveal again');
});

// This is a unit test.
test('the pointer leaving the row closes the reveal, and so does leaving the '
  + 'window', () => {
  const { fire, reveal, cell, elsewhere } = board();
  fire('mouseover', { target: cell });
  assert.ok(reveal.open);
  fire('mouseover', { target: elsewhere });
  assert.equal(reveal.open, false, 'hover moved off the row');
  fire('mouseover', { target: cell });
  // A pointer that leaves the window enters nothing, so there is no `mouseover`
  // to say the row was left.
  fire('mouseout', { target: cell, relatedTarget: null });
  assert.equal(reveal.open, false, 'the pointer left the window');
});

// This is a unit test.
test('the reveal stays with its row while anything under it scrolls', () => {
  const { fire, reveal, cell } = board();
  fire('mouseover', { target: cell });
  const placed = reveal.style.top;
  reveal.style.top = '';
  fire('scroll', { target: cell });
  assert.equal(reveal.style.top, placed,
    'a fixed box does not travel with its cell, so a scroll re-places it');
});
