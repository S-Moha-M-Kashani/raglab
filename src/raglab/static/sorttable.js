// Click a column header to sort by it. Loaded by both lab pages, so sorting
// behaves identically on each; evaluated directly in a `vm` context by
// `tests/sorttable.test.js`, hence no module wrapper and no DOM access outside
// a function.
//
// Three states per column: sort, reverse, then back to the order the table
// was served in — that order is itself information (a ranking, or newest-first).
const SortTable = (() => {
  // Not '-': a lone minus sign is not a placeholder anybody here writes.
  const MISSING = new Set(['', '—', '–', '·', 'n/a']);
  const LEADING_NUMBER = /^([+-]?\d+(?:\.\d+)?)([\s\S]*)$/;
  // Excludes dates/times like '2026-08-04' or '16:00:08', which sort correctly
  // as their own zero-padded text and would collapse to 2026/16 as figures.
  const NOT_A_UNIT = /^[-/:.]\d/;

  // `data-sort` on the cell wins where a renderer knows better than its own
  // text; everything else reads what the row *shows*.
  function cellKey(text) {
    const shown = String(text === null || text === undefined ? '' : text).trim();
    if (MISSING.has(shown.toLowerCase())) {
      return { missing: true, number: null, text: '' };
    }
    // Commas dropped for the numeric probe only: '1,204' is 1204, and the text
    // key keeps the cell exactly as rendered.
    const probe = shown.replace(/,/g, '');
    let number = null;
    const found = probe.match(LEADING_NUMBER);
    if (found && !NOT_A_UNIT.test(found[2])) {
      // A trailing unit or qualifier is fine — '83%', '12s', '0.747 ± 0.042'.
      number = parseFloat(found[1]);
    } else if (!found) {
      // '± 0.042' on its own: the mark qualifies the figure, not part of it.
      const error = probe.match(/^±\s*([+-]?\d+(?:\.\d+)?)$/);
      if (error) number = parseFloat(error[1]);
    }
    return { missing: false, number, text: shown };
  }

  // `dir` is 1 for ascending, -1 for descending.
  function compare(a, b, dir) {
    if (a.missing || b.missing) {
      // Last regardless of `dir`: a dash means "never measured", not a low value.
      if (a.missing && b.missing) return 0;
      return a.missing ? 1 : -1;
    }
    if (a.number !== null && b.number !== null) return (a.number - b.number) * dir;
    // A column mixing figures and words sorts as words.
    return a.text.localeCompare(b.text, undefined,
      // Case-insensitive (labels are typed by hand); numeric so 'C-2' < 'C-10'.
      { sensitivity: 'base', numeric: true }) * dir;
  }

  // Numbers open best-first (descending), text opens A-Z (ascending). Not every
  // "best" is highest — a rank column marks itself with `data-order="ascending"`.
  function opening(th, hasNumbers) {
    const told = (th.getAttribute('data-order') || '').toLowerCase();
    if (told === 'ascending') return 1;
    if (told === 'descending') return -1;
    return hasNumbers ? -1 : 1;
  }

  // Wire one table. Safe to call again on the same element — a re-rendered table
  // is a new element, and a second call on the same one would stack listeners.
  function make(table) {
    if (!table || table.dataset.sortWired) return;
    const head = table.tHead && table.tHead.rows[table.tHead.rows.length - 1];
    const body = table.tBodies[0];
    if (!head || !body || !body.rows.length) return;
    table.dataset.sortWired = '1';
    table.classList.add('sortable');
    // Captured once: this is the order the service sent, and the third click
    // restores exactly it.
    const served = Array.from(body.rows);
    const heads = Array.from(head.cells);
    let column = -1;
    let dir = 0;

    const keyOf = (row, at) => {
      const cell = row.cells[at];
      if (!cell) return cellKey('');
      const told = cell.getAttribute('data-sort');
      return cellKey(told === null ? cell.textContent : told);
    };

    const opensAt = (at) => opening(
      heads[at], served.some((row) => keyOf(row, at).number !== null));

    function apply() {
      for (const th of heads) th.removeAttribute('aria-sort');
      let rows = served;
      if (column >= 0) {
        rows = served
          .map((row, at) => ({ row, at }))
          .sort((x, y) => compare(keyOf(x.row, column), keyOf(y.row, column), dir)
            // Explicit, so a tie provably keeps served order rather than relying
            // on the engine's sort being stable.
            || x.at - y.at)
          .map((held) => held.row);
        heads[column].setAttribute('aria-sort',
          dir === 1 ? 'ascending' : 'descending');
      }
      for (const row of rows) body.appendChild(row);
    }

    heads.forEach((th, at) => {
      if (th.hasAttribute('data-nosort')) return;
      th.classList.add('sort-col');
      th.tabIndex = 0;
      th.setAttribute('role', 'button');
      th.title = 'sort by this column · again to reverse · a third time for the '
        + 'order it was served in';
      const cycle = () => {
        if (column !== at) { column = at; dir = opensAt(at); }
        else if (dir === opensAt(at)) { dir = -dir; }
        else { column = -1; dir = 0; }
        apply();
      };
      th.addEventListener('click', cycle);
      th.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          cycle();
        }
      });
    });
  }

  // Every table under a root, for a page that renders several at once.
  function makeAll(root) {
    for (const table of (root || document).querySelectorAll('table')) make(table);
  }

  return { cellKey, compare, opening, make, makeAll };
})();

if (typeof window !== 'undefined') window.SortTable = SortTable;
