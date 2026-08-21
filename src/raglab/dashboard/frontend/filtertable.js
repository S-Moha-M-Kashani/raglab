// Type a question, and the table answers with the rows that satisfy it.
//
// Filtering and sorting are the same question asked twice — which rows, in what
// order — so this file reads a cell with `sorttable.js`'s own `cellKey`, and
// never with a parser of its own. A filter that read '0.747 ± 0.042' as text
// while the column ordered it as 0.747 would answer `decision>0.7` with rows
// the reader can see are below it.
//
// The two compose without knowing about each other. A row is hidden, not
// removed, so the sorter's record of the order it was served in stays whole;
// and hidden rows are parked at the end of the body, so the stripes count the
// rows you can actually see rather than the gaps between them.
//
// Column names come from the headings, so a column added to the board is
// filterable the day it is added and this file does not mention any of them.
const FilterTable = (() => {
  // Longest first: '>=' must be found before '>', and '!=' before '='.
  const OPS = ['>=', '<=', '!=', '!~', '>', '<', '=', '~', ':'];

  // A run of non-space characters, except that a quoted span may hold spaces —
  // which is how `label~"hybrid vs dense"` is one term and not three.
  const TOKENS = /(?:[^\s"]|"[^"]*")+/g;

  const fold = (text) => String(text === null || text === undefined ? '' : text)
    .toLowerCase();
  const same = (a, b) => fold(a) === fold(b);
  const has = (haystack, needle) => fold(haystack).includes(fold(needle));

  // Headings are what a reader sees, so they are what a reader types: 'ctx
  // recall' is asked for as `ctx-recall`. A heading that is a symbol says its
  // own name in `data-filter` instead — '±' is a column nobody can type.
  const nameOf = (text) => fold(text).trim().replace(/\s+/g, '-');

  const unquote = (text) => (text.length > 1 && text.startsWith('"')
    && text.endsWith('"') ? text.slice(1, -1) : text);

  // An operator where the column name should be. `:done` and `>30` are a column
  // name away from being questions, so they are reported as the mistakes they
  // are rather than searched for as words.
  const HEADLESS = /^!?[><=~:]/;

  // One typed term. `col` is null for a bare word, which asks about the whole
  // row rather than about a column.
  function term(token) {
    if (HEADLESS.test(token)) return null;
    const negated = token.startsWith('!') && !OPS.some(
      (op) => op.length > 1 && token.slice(1).startsWith(op));
    for (const op of OPS) {
      const at = token.indexOf(op);
      // `!=` and `!~` carry their own '!', so a leading one is the column's.
      if (at <= 0) continue;
      const col = nameOf(token.slice(0, at).replace(/^!/, ''));
      const value = unquote(token.slice(at + op.length).trim());
      if (!col) return null;
      if (op === ':') {
        // A colon with nothing after it asks whether the column was measured
        // at all — the one question a reader cannot ask any other way.
        if (!value) return { col, op: negated ? 'absent' : 'present', value: '' };
        // And with something after it, it is the forgiving spelling of '~',
        // because `state:done` is what a reader types before reading any help.
        return { col, op: negated ? '!~' : '~', value };
      }
      if (!value) return null;      // an operator with nothing to compare to
      return { col, op, value };
    }
    const word = unquote(negated ? token.slice(1) : token);
    if (!word) return null;
    return { col: null, op: negated ? 'lacks' : 'has', value: word };
  }

  // `bad` holds the tokens that asked nothing, spelled as they were typed. They
  // are reported rather than dropped: a table filtered on two of three terms
  // looks exactly like a table filtered on three.
  function parse(query) {
    const terms = [];
    const bad = [];
    for (const token of String(query || '').match(TOKENS) || []) {
      const asked = term(token);
      if (asked) terms.push(asked);
      else bad.push(token);
    }
    return { terms, bad };
  }

  // The column names a query mentions that this table does not have, in the
  // order they were typed. A misspelling is a question about nothing, and
  // answering it with an empty table would read as 'no row is like that'.
  function unknown(terms, names) {
    const held = new Set(names);
    const out = [];
    for (const asked of terms) {
      if (asked.col && !held.has(asked.col) && !out.includes(asked.col)) {
        out.push(asked.col);
      }
    }
    return out;
  }

  function answers(asked, row) {
    if (asked.op === 'has') return has(row.text, asked.value);
    if (asked.op === 'lacks') return !has(row.text, asked.value);
    const cells = row.cells || {};
    const key = SortTable.cellKey(
      Object.prototype.hasOwnProperty.call(cells, asked.col)
        ? cells[asked.col] : '');
    if (asked.op === 'present') return !key.missing;
    if (asked.op === 'absent') return key.missing;
    // A dash is 'never measured', not a low value — the reading that already
    // puts a missing cell last in both sort directions. So a row that never
    // measured the column is no answer to a question about it, in either
    // direction: `ctx-recall!=0.5` must not quietly hand back every ungraded
    // row as though it had disagreed.
    if (key.missing) return false;
    const other = SortTable.cellKey(asked.value);
    const numeric = key.number !== null && other.number !== null;
    switch (asked.op) {
      case '=': return numeric ? key.number === other.number
        : same(key.text, asked.value);
      case '!=': return !(numeric ? key.number === other.number
        : same(key.text, asked.value));
      case '~': return has(key.text, asked.value);
      case '!~': return !has(key.text, asked.value);
      default: {
        // Words compare as words: '>' on a text column is a real question, and
        // alphabetical is the answer a reader expects from it. Timestamps land
        // here too and come out chronological, because the format is
        // zero-padded and `numeric` collation reads each field as a figure.
        const sign = numeric ? key.number - other.number
          : key.text.localeCompare(asked.value, undefined,
            { sensitivity: 'base', numeric: true });
        if (asked.op === '>') return sign > 0;
        if (asked.op === '>=') return sign >= 0;
        if (asked.op === '<') return sign < 0;
        return sign <= 0;
      }
    }
  }

  // Every term must hold. An AND is the only combination worth having here: a
  // reader narrows a table by adding to what they typed, and each addition
  // showing *more* rows is the opposite of that.
  const matches = (terms, row) => terms.every((asked) => answers(asked, row));

  // --- the DOM half ---------------------------------------------------------

  function columnNames(table) {
    const head = table.tHead && table.tHead.rows[table.tHead.rows.length - 1];
    return Array.from(head ? head.cells : []).map(
      (th) => th.getAttribute('data-filter') || nameOf(th.textContent));
  }

  const readRow = (tr, names) => {
    const cells = {};
    names.forEach((name, at) => {
      const cell = tr.cells[at];
      if (!name || !cell) return;
      // What the column sorts on where the renderer said so, for the same
      // reason the sorter reads it: the pipeline cell shows an abbreviation and
      // carries the whole sentence, and a reader filtering that column means
      // the sentence.
      const told = cell.getAttribute('data-sort');
      cells[name] = told === null ? cell.textContent : told;
    });
    return { cells, text: tr.textContent };
  };

  // Apply `query` to `table`, and say what happened. Safe to call as often as
  // the reader types, and again after every sort: it decides each row from the
  // query alone, so it cannot drift.
  function apply(table, query) {
    const body = table && table.tBodies[0];
    if (!body) return { shown: 0, total: 0, bad: [], unknown: [] };
    const rows = Array.from(body.rows);
    const names = columnNames(table);
    const { terms, bad } = parse(query);
    const missing = unknown(terms, names);
    // A query that asks nothing answerable leaves the table alone rather than
    // emptying it, and the bar says why.
    if (bad.length || missing.length) {
      return { shown: rows.filter((tr) => !tr.hidden).length,
               total: rows.length, bad, unknown: missing };
    }
    const hidden = [];
    let shown = 0;
    for (const tr of rows) {
      const ok = matches(terms, readRow(tr, names));
      tr.hidden = !ok;
      if (ok) shown += 1;
      else hidden.push(tr);
    }
    // Parked at the end, in the order they were in. The stripes are drawn by
    // position in the body, so hidden rows left among the visible ones stripe
    // the gaps and the pattern comes out looking broken.
    for (const tr of hidden) body.appendChild(tr);
    return { shown, total: rows.length, bad: [], unknown: [] };
  }

  return { parse, matches, unknown, apply, columnNames };
})();

if (typeof window !== 'undefined') window.FilterTable = FilterTable;
