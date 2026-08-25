// The leaderboard surface: one table per dataset, every experiment in it.
//
// Everything it renders comes from `GET /api/leaderboard`, which serialises
// `evaluation.leaderboard` — the same module `raglab-leaderboard` prints from,
// so this page and the command line cannot describe the same records
// differently. Nothing here derives the pipeline sentence or a decision score:
// a page that re-derived either could disagree with the file it came from.
//
// The board names no winner. It mixes question sets and judges by design, so
// 'winner by more than the combined error' would compare numbers that never
// met. That claim still exists — `leaderboard.verdict()` — for the sweep, whose
// candidates share a question set and a judge by construction.

const $ = (id) => document.getElementById(id);

// '*' is every experiment: the table that used to sit on the lab page, which is
// this same population with no filter — which is why it is an option in the
// picker rather than a second surface.
const EVERY = '*';

const COLUMNS = [
  // `title` here is the short form for a pointer, and `sorttable.js` now leaves
  // a heading that has one alone. It is not where any of this is *published*,
  // though — a title answers a mouse and nothing else, so the two sentences a
  // reader actually needs (that this cell opens, and that `fake` is a rehearsal)
  // are in the hint prose under the table, in the page's own text.
  { key: 'pipeline', label: 'pipeline', text: true, freeze: 'freeze-1',
    title: 'every step this experiment ran, abbreviated · hover or focus for '
      + 'the whole sentence and every knob behind it' },
  // What the row is *about*, before anything it measured: which corpus (on the
  // board that mixes them — one table per dataset means this column is only a
  // question there) and how many questions were put to it. A metric read
  // without knowing how many questions produced it is a number with no error
  // bar, so the count comes before the numbers rather than after them.
  { key: 'dataset', label: 'dataset', everyOnly: true },
  { key: 'questions', label: 'questions',
    title: 'how many questions were scored, and the sample they were drawn '
      + 'from when the run recorded one' },
  // Then the deciding score, its error, and the four metrics it is the mean of:
  // the only columns that decide anything, kept as close to the identity as the
  // frozen sentence allows. The descriptive columns wait behind them.
  { key: 'decision', label: 'decision', title: 'unweighted mean of the four judged metrics' },
  // '±' is a column no reader can type, so it says its filter name itself. Every
  // other heading is already the word a reader would use for it.
  { key: 'spread', label: '±', filter: 'spread',
    title: 'standard error of the decision score' },
  { key: 'faithfulness', label: 'faith', step: 'generation' },
  { key: 'answer_relevancy', label: 'ans rel', step: 'generation' },
  { key: 'llm_context_precision_with_reference', label: 'ctx prec', step: 'retrieval' },
  { key: 'context_recall', label: 'ctx recall', step: 'retrieval' },
  { key: 'kind', label: 'kind', title: 'index, retrieve, run or query' },
  { key: 'when', label: 'when', title: 'when it started' },
  { key: 'label', label: 'label' },
  { key: 'judge', label: 'judge', step: 'generation',
    title: 'which model graded — rows graded differently are not comparable' },
  { key: 'provider', label: 'backend',
    title: 'where the model calls went · fake is a rehearsal, not a measurement' },
  { key: 'state', label: 'state' },
  { key: 'seconds', label: 'seconds', title: 'wall clock' },
  { key: 'open', label: 'open', nosort: true, freeze: 'freeze-last',
    title: 'open this experiment in the Laboratory · its settings become the '
      + 'knobs, and what this lab cannot serve is named there' },
];

const fmt = (value, digits = 3) =>
  value === null || value === undefined ? '—' : Number(value).toFixed(digits);

// `started_at` is already '%Y-%m-%d %H:%M:%S'. Seconds do not help anyone
// comparing experiments, so they are dropped rather than reformatted.
const when = (row) => (row.started_at || '').slice(0, 16) || '—';

const judgeOf = (row) => {
  const judge = row.judge || {};
  return judge.model ? `${judge.model} via ${judge.provider || '?'}` : '—';
};

// The `questions` column reads a row's own recorded selection identity —
// `run_evaluation.selection_note`'s balance/limit/n/by_<balance> counts —
// when the run carries one, falling back to the bare count for a row that
// does not: an index build, a retrieval, an imported archive (none of which
// score questions) or a run recorded before `selection_note` existed. Never
// both, and never a count reinterpreted as an identity it never claimed.
const selectionText = (row) => {
  const selection = row.selection || {};
  const n = selection.n;
  if (n === null || n === undefined) return String(row.n_questions ?? 0);
  const balance = selection.balance || '';
  const counts = balance ? selection[`by_${balance}`] : null;
  if (counts && Object.keys(counts).length) {
    const parts = Object.entries(counts)
      .map(([value, count]) => `${value}=${count}`).join(', ');
    return `${n} (${parts})`;
  }
  return balance ? `${n}, ${balance}` : String(n);
};

// The sort key stays numeric even when the display text is not — the same
// reason the pipeline column overrides its own `data-sort` below: a richer
// "12 (easy=4, hard=8)" must still sort beside a bare "30" as 12 and 30, not
// alphabetically as strings.
const selectionSort = (row) => {
  const n = (row.selection || {}).n;
  return n === null || n === undefined ? (row.n_questions ?? 0) : n;
};

// The same sentence spelled out, as plain text: what the column sorts on, what
// a filter on it reads, and what the reveal publishes. The board *draws* the
// short form — an abbreviated cell that also sorted and filtered as its
// abbreviation would answer `pipeline~sentence-transformers` with nothing.
const sentenceText = (row) =>
  (row.pipeline || []).map((f) => f.text).join(' · ');

// The sentence, each fragment inked with its own step, in the short form the
// service abbreviated it to — neither surface derives the other's reading, so
// the board cannot come to abbreviate a knob the printer spells out differently.
// `data-step` is how every other coloured thing on these pages takes its ink,
// and the four inks are defined once in tokens.css. Wrapped in `.clip`, which is
// what actually holds the frozen column's width — a cell cannot.
const sentence = (row) => '<span class="clip">' + ((row.pipeline || []).length
  ? (row.pipeline || []).map((f) =>
    `<span data-step="${escapeHtml(f.step)}" class="pipe-part">`
    + `${escapeHtml(f.short || f.text)}</span>`).join('<span class="pipe-sep">·</span>')
  : '<span class="muted">—</span>') + '</span>';

function cell(row, key) {
  const metrics = row.metrics || {};
  switch (key) {
    case 'pipeline': return sentence(row);
    case 'kind': return escapeHtml(row.kind || '—');
    case 'when': return escapeHtml(when(row));
    case 'label': return escapeHtml(row.label || '—');
    case 'decision': return fmt(row.decision, 4);
    case 'spread': return fmt(row.decision_stderr, 3);
    case 'judge': return escapeHtml(judgeOf(row));
    case 'questions': return escapeHtml(selectionText(row));
    case 'dataset': return escapeHtml(row.dataset || '—');
    // Marked rather than merely printed: on `fake` every LLM number on the row
    // came from a stub that cannot fail, so the row is a rehearsal of the
    // pipeline and not a measurement of it.
    case 'provider': return row.provider === 'fake'
      ? '<b>fake</b>' : escapeHtml(row.provider || '—');
    case 'state': return row.state === 'done'
      ? 'done'
      : row.error
        ? `<span class="failed"><b>${escapeHtml(row.state || '?')}</b>`
          + `<button type="button" class="why" data-help="${escapeHtml(row.error)}"`
          + ` aria-label="Why did this ${escapeHtml(row.state || 'fail')}?">!</button></span>`
        : `<b>${escapeHtml(row.state || '—')}</b>`;
    case 'seconds': return Math.round(row.seconds || 0);
    // Two halves of one cell: the `href` the browser follows to the Laboratory,
    // and the id the handoff handler below writes into the slot for it. Both
    // name the experiment, because a link that navigated and a handler that
    // read a row index would be two accounts of one click.
    //
    // Same-tab navigation is the requested in-app move; the browser still
    // keeps middle-click, cmd/ctrl-click and other modifier-key choices.
    case 'open': return `<a class="open-run"`
      + ` href="/?experiment=${encodeURIComponent(row.experiment_id)}"`
      + ` data-experiment="${escapeHtml(row.experiment_id)}"`
      + ` aria-label="Open ${escapeHtml(row.experiment_id)} in the Laboratory,`
      + ` with its settings on the knobs">↗</a>`;
    // The four judged metrics, by their own keys, so a column cannot drift from
    // the metric it names.
    default: return fmt(metrics[key]);
  }
}

let CURRENT = '';        // the dataset in force, '' = the served default
let CATALOGUE = [];      // the corpora the route named, as served

// The corpus by the name the heading and the picker say, never by its id: the
// caption and the region's name are read aloud, so an id there hands the screen
// reader the internal name while the eye gets the human one.
const corpusName = (dataset) => (dataset === EVERY
  ? 'every dataset' : shownOption(dataset, CATALOGUE).name);

const columnsFor = (dataset) =>
  COLUMNS.filter((col) => !col.everyOnly || dataset === EVERY);

function renderTable(dataset, rows) {
  const columns = columnsFor(dataset);

  const head = columns.map((col) => {
    const cls = [col.text ? 'text' : '', col.freeze || ''].filter(Boolean).join(' ');
    return `<th scope="col"${cls ? ` class="${cls}"` : ''}`
      + `${col.step ? ` data-step="${col.step}"` : ''}`
      + `${col.nosort ? ' data-nosort' : ''}`
      + `${col.filter ? ` data-filter="${escapeHtml(col.filter)}"` : ''}`
      + `${col.title ? ` title="${escapeHtml(col.title)}"` : ''}>`
      + `${escapeHtml(col.label)}</th>`;
  }).join('');

  const body = rows.map((row) => columns.map((col) => {
    const cls = [col.text ? 'text' : '', col.freeze || ''].filter(Boolean).join(' ');
    // The settings reveal hangs off the pipeline cell, which is the cell that
    // shows the short form of the same thing. The cell takes focus so there is
    // a keyboard way to it and not only a pointer one.
    // And it says what it sorts as, because the reveal it carries is part of the
    // cell's text: sorted on that, the column reads the whole recorded config
    // after the sentence, and two rows whose sentences share a prefix are
    // ordered by knobs the reader cannot see. `data-sort` is what the sorter
    // reads instead when a renderer knows better than its own text.
    const reveal = col.key === 'pipeline' ? settingsReveal(row) : '';
    // Same reason as the pipeline row above: the questions cell's own text can
    // be a richer "12 (easy=4, hard=8)" now, and a column sorted on that text
    // would order "12 (...)" after "3 (...)" alphabetically. `data-sort` keeps
    // the sort numeric regardless of what the cell says.
    const sortAs = col.key === 'pipeline'
      ? ` data-sort="${escapeHtml(sentenceText(row))}"`
      : col.key === 'questions' ? ` data-sort="${selectionSort(row)}"` : '';
    return `<td${cls ? ` class="${cls}"` : ''}${sortAs}`
      + `${col.key === 'pipeline' ? ' tabindex="0"' : ''}>`
      + `${cell(row, col.key)}${reveal}</td>`;
  }).join('')).map((cells) => `<tr>${cells}</tr>`).join('');

  // The corpus by the name the heading and the picker use. The id is what the
  // route is keyed on, not what anything on the page is supposed to read — and
  // a caption and a region name are read aloud, so the id here would give the
  // screen reader the internal name and the eye the human one.
  const named = `every experiment on ${escapeHtml(corpusName(dataset))}`;
  return `
      <div class="table-scroll" tabindex="0" role="region"
           aria-label="${named}">
        <table class="data-table centred">
          <caption>${named}</caption>
          <thead><tr>${head}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>`;
}

// The whole recorded config, grouped by step and inked by step, so it reads as
// a longer form of the sentence that opened it. Opens on hover AND on keyboard
// focus: a reveal that only answers a mouse publishes to a mouse and to nothing
// else, which is why the tooltips on these pages were removed.
function settingsReveal(row) {
  const config = row.config || {};
  // Which recorded knobs this build never read, and why — `{}` for every row
  // Tasks 1-2 did not touch, and for every row whose config left nothing
  // inert. Keyed the same way the blocks below are built: `${step}.${key}`.
  const inert = row.inert || {};
  // The abbreviation's expansion, first: the cell draws 'sem-drift·louv·ST' and
  // the reader who does not recognise one of those words has to be able to read
  // it here, not infer it from the knob list below. The knobs are still the
  // longer answer — they hold what the sentence never names.
  const said = sentenceText(row);
  const blocks = ['index', 'retrieval', 'generation']
    .filter((step) => config[step] && Object.keys(config[step]).length)
    .map((step) => `<div class="reveal-step" data-step="${step}">`
      + `<b>${step}</b>`
      + Object.entries(config[step]).map(([k, v]) => {
        // A knob this config recorded but the build never read (an overlap
        // under a chunker that never slides a window) is a refusal, not a
        // number to trust — so it reads `none`, never the value that sat
        // unused, and the span carries its own class and the reason a reader
        // would otherwise have to guess at. A knob genuinely recorded as the
        // word 'none' takes neither: it falls through to the plain span
        // below, unchanged, so the two are never mistaken for each other.
        const reason = inert[`${step}.${k}`];
        return reason === undefined
          ? `<span class="reveal-knob">${escapeHtml(k)} <b>${escapeHtml(String(v))}</b></span>`
          : `<span class="reveal-knob-off" title="${escapeHtml(reason)}`
            + ` — recorded value never used">${escapeHtml(k)} <b>none</b></span>`;
      }).join('')
      + '</div>').join('');
  // A tab stop of its own, because the box holds more than it can show and
  // scrolls: the part below its own fold is otherwise readable with a pointer
  // and by nothing else. Some browsers make a scrollable box focusable anyway
  // and some do not, and which of them a reader has is not something the panel
  // should decide the answer for.
  const head = said ? `<div class="reveal-said">${escapeHtml(said)}</div>` : '';
  return blocks || head
    ? `<div class="settings-reveal" popover="manual" tabindex="0">`
      + `${head}${blocks}</div>`
    : '';
}

// --- the filter -------------------------------------------------------------
// One line above the table. What it can be asked is `filtertable.js`'s to say;
// what this holds is the query in force, and where it is published: in the URL,
// so a filtered board is a link, and kept across a dataset pick, because
// 'the failed ones' is a question about the lab and not about one corpus.
let QUERY = '';
let REMEASURE = null;      // the scroll rail's, once there is a table to measure

function renderFilter() {
  return `
    <div class="filter-bar">
      <label for="row-filter">Filter</label>
      <input id="row-filter" class="filter-input" type="text" spellcheck="false"
             autocapitalize="off" autocomplete="off" aria-describedby="filter-syntax"
             placeholder="state!=failed questions&gt;30 decision&gt;0.6"
             value="${escapeHtml(QUERY)}">
      <button type="button" class="filter-clear" id="filter-clear">clear</button>
      <span class="filter-count" id="filter-count" role="status"></span>
      <p class="filter-said" id="filter-said" hidden></p>
    </div>`;
}

const quoted = (words) => words.map((word) => `“${word}”`).join(', ');

// Narrow the table to the query, and say what happened. A count on every state,
// including the unfiltered one: a table with no count is a table that quietly
// might not be all of it.
function applyFilter() {
  const table = document.querySelector('#board table');
  const count = $('filter-count');
  const said = $('filter-said');
  if (!table || !count || !said) return;
  const out = FilterTable.apply(table, QUERY);
  const trouble = out.unknown.length
    ? `no column called ${quoted(out.unknown)} on this board`
    : out.bad.length
      ? `${quoted(out.bad)} asks nothing`
      : '';
  // A query that asked nothing answerable leaves the rows exactly as the last
  // answerable one left them — which while a term is half typed is the useful
  // behaviour and never the obvious one, so it is said rather than assumed.
  said.textContent = trouble ? `${trouble} — the rows below are unchanged.` : '';
  said.hidden = !trouble;
  count.textContent = out.shown === out.total
    ? `${out.total} row${out.total === 1 ? '' : 's'}`
    : `${out.shown} of ${out.total} shown`;
  // The table's width follows what is in it, and the rail above it is sized to
  // that width.
  if (REMEASURE) REMEASURE();
}

// In the URL beside the dataset, for the same reason: a view somebody wants to
// come back to is a view they can send.
function publishQuery() {
  const url = new URL(window.location.href);
  if (QUERY) url.searchParams.set('filter', QUERY);
  else url.searchParams.delete('filter');
  window.history.replaceState({}, '', url);
}

// Delegated and registered once, because the whole board is rebuilt on every
// dataset pick and a listener added per render would stack.
document.addEventListener('input', (event) => {
  if (!event.target || event.target.id !== 'row-filter') return;
  QUERY = event.target.value;
  publishQuery();
  applyFilter();
});
document.addEventListener('click', (event) => {
  const clear = event.target && event.target.closest
    ? event.target.closest('#filter-clear') : null;
  if (!clear) return;
  QUERY = '';
  const box = $('row-filter');
  if (box) { box.value = ''; box.focus(); }
  publishQuery();
  applyFilter();
});

// Which corpus is on screen. The same affordance the lab page uses for its
// corpus scope, down to the classes and the native `popover` — the browser owns
// show, hide, light-dismiss and Escape, so there is no second implementation of
// one control across two pages. Where the box opens is the shared sheet's, said
// once against the button that opens it, so both surfaces get it.
const optionsFor = (datasets) => [
  ...datasets.map((d) => ({ id: d.id, name: d.name || d.id })),
  { id: EVERY, name: 'every experiment' }];

// One list of names for one set of corpora: a heading that printed the id while
// the button under it printed the corpus's name would be two names for the same
// thing, and the reader would have to work out that they are one.
const shownOption = (dataset, datasets) => {
  const options = optionsFor(datasets);
  return options.find((o) => o.id === dataset) || options[0];
};

function renderPicker(dataset, datasets) {
  const options = optionsFor(datasets);
  const shown = shownOption(dataset, datasets);
  return `
    <div class="board-bar">
      <button class="context-scope" type="button"
              popovertarget="pick-list">
        <span class="context-label">Dataset</span>
        <span>${escapeHtml(shown.name)}</span>
        <span class="context-caret" aria-hidden="true">▾</span>
      </button>
      <ul class="context-detail" id="pick-list" popover
          aria-label="Which dataset this board shows">
        ${options.map((o) => `<li><button type="button" class="pick"
             data-dataset="${escapeHtml(o.id)}"
             ${o.id === shown.id ? 'aria-current="true"' : ''}
             >${escapeHtml(o.name)}</button></li>`).join('')}
      </ul>
    </div>`;
}

async function loadBoard(dataset) {
  const box = $('board');
  box.innerHTML = '<div class="card"><p class="prose">Reading the records…</p></div>';
  let body;
  try {
    const res = await fetch('/api/leaderboard?dataset='
                            + encodeURIComponent(dataset || ''));
    if (!res.ok) throw new Error(`the lab answered ${res.status}`);
    body = await res.json();
  } catch (e) {
    // A failing read says so, rather than looking like an empty lab.
    box.innerHTML = '<div class="card"><div class="card-head">'
      + '<h2>Could not read the leaderboard</h2></div>'
      + `<p class="prose">${escapeHtml(e.message)}. The lab on :9002 serves this `
      + 'page and the records behind it, so if it stopped, this is what that '
      + 'looks like.</p></div>';
    return;
  }
  CURRENT = body.dataset || '';
  CATALOGUE = body.datasets || [];
  const rows = body.rows || [];
  box.innerHTML = `
    <section class="card">
      <div class="card-head">
        <h2>${escapeHtml(shownOption(CURRENT, CATALOGUE).name)}</h2>
        <span class="section-meta right">${rows.length} recorded</span>
      </div>
      ${renderPicker(CURRENT, CATALOGUE)}
      ${rows.length ? renderFilter() + renderTable(CURRENT, rows)
        : '<p class="prose">Nothing recorded for '
        + 'this dataset yet. Open the <a href="/">Laboratory</a>, pick it and '
        + 'press <b>Run evaluation</b>.</p>'}
      <p class="table-hint">Click any column heading to sort by it, again to
        reverse, a third time for the order it was served in. The
        <b>pipeline</b> cell is abbreviated and clipped to the column's width —
        hover it or give it focus and the whole sentence opens beside it, with
        every recorded knob under that. <b>backend</b> is where the model calls
        actually went: <code>fake</code> answers and judges without ever
        failing, so those rows are a rehearsal of the pipeline and not a
        measurement of it. This table names no winner: rows graded by different
        judges over different question sets share it, so <b>judge</b> and
        <b>questions</b> are columns you compare on. The <b>open</b> arrow opens
        the <a href="/">Laboratory</a> in this tab with the experiment's
        settings on the knobs, pins the Inspector link above to that same
        experiment, and names any unserved knob in the lab helper rather than
        quietly leaving it behind.</p>
      <p class="table-hint" id="filter-syntax"><b>Filter</b> takes one term per
        column, all of which must hold, each written as the column's own heading
        and what you want of it: <code>questions&gt;30</code>,
        <code>decision&gt;=0.6</code>, <code>seconds&lt;120</code>,
        <code>when&gt;2026-08-01</code>, <code>state!=failed</code>,
        <code>judge~sonnet</code>. A colon is the forgiving spelling of
        <code>~</code> (<code>kind:run</code>); a colon with nothing after it
        asks whether the column was measured at all (<code>ctx-recall:</code>
        for the rows that have it, <code>!ctx-recall:</code> for the rows that do
        not). A bare word searches the whole row, <code>!word</code> excludes it,
        and quotes hold a value together: <code>label~"hybrid vs dense"</code>.
        A term about a column a row never measured does not match it in either
        direction — a dash is <i>never measured</i>, not a low value.</p>
    </section>`;
  const table = box.querySelector('table');
  if (table) {
    // Both readings of the table, wired to each other in one direction only:
    // the sorter re-appends every row it holds on each reorder, which drops the
    // hidden ones back among the visible ones, so the filter re-runs after it.
    // The filter knows nothing about sorting in return — it decides each row
    // from the query alone, so it cannot drift whatever order it is handed.
    SortTable.make(table, { onApply: applyFilter });
    REMEASURE = mountScrollRail(box.querySelector('.table-scroll'));
  }
  applyFilter();
  wirePicker();
}

function wirePicker() {
  const list = $('pick-list');
  if (!list) return;
  // Only the choosing. Opening and closing belong to the browser: the button
  // carries `popovertarget`, which is what buys light-dismiss and Escape.
  list.addEventListener('click', (event) => {
    const pick = event.target.closest('.pick');
    if (!pick) return;
    const chosen = pick.dataset.dataset;
    // In the URL, so a view is linkable and survives a reload.
    const url = new URL(window.location.href);
    url.searchParams.set('dataset', chosen);
    window.history.replaceState({}, '', url);
    loadBoard(chosen);
  });
}

// Which reveal is open is a question about hover and focus, and the two answer
// it together: hover alone is what these pages already removed twice — a reveal
// that answers only a pointer publishes to a mouse and to nothing else, and the
// settings would have no keyboard way in at all. The two states are tracked
// separately because they are separate: a reveal opened by focus does not close
// because the mouse went somewhere else.
//
// The box is a `popover`, which is the only way it can be seen over the sticky
// header: it lives in a `position: sticky` cell, a sticky element is a stacking
// context whatever its z-index, and no z-index lifts a descendant out of its own
// stacking context. So showing it is script's job, not a CSS rule's. Once shown
// it still has to be told where to go, because it is `fixed` and does not travel
// with its cell.
//
// Which cell an event is about is read from the event's own target, not from
// `:hover` — the pointer moving onto the panel is not the pointer leaving the
// row (the panel scrolls, so it has to be reachable), and the panel is a
// descendant of the cell in the document however far out of it the top layer
// paints, so `closest` answers that on its own. `:hover` cannot: it is
// recomputed on its own schedule, and reading it in a listener is a race.
let OPEN = null;
let HOVERED = null;
let FOCUSED = null;

// A board rebuilt under an open reveal leaves both of these pointing at cells
// that are no longer in the document.
const live = (cell) => (cell && cell.isConnected ? cell : null);

const revealCell = (node) => (node && node.closest
  ? node.closest('td.freeze-1') : null);

function syncReveal() {
  const cell = live(HOVERED) || live(FOCUSED);
  const reveal = cell ? cell.querySelector('.settings-reveal') : null;
  // What is open is read back rather than remembered: the browser closes a
  // popover itself whenever the element leaves the document, which sorting a
  // column does to every row it moves and a rebuild does to all of them. A page
  // that trusted its own record then left the cell that owns that reveal unable
  // to open it ever again — hover it, and the panel is already 'open'.
  if (OPEN && !OPEN.matches(':popover-open')) OPEN = null;
  // One at a time, because two panels of settings on screen say nothing about
  // which row you are reading.
  if (OPEN !== reveal) {
    if (OPEN) OPEN.hidePopover();
    if (reveal) reveal.showPopover();
    OPEN = reveal;
  }
  if (OPEN) placeReveal(cell, '.settings-reveal');
}

// Delegated at the document and registered once, because the whole board is
// rebuilt on every pick and a listener added per render would stack.
document.addEventListener('mouseover', (event) => {
  HOVERED = revealCell(event.target);
  syncReveal();
});
// A pointer that leaves the window enters nothing, so there is no `mouseover`
// to say the row was left.
document.addEventListener('mouseout', (event) => {
  if (!event.relatedTarget) { HOVERED = null; syncReveal(); }
});
document.addEventListener('focusin', (event) => {
  FOCUSED = revealCell(event.target);
  syncReveal();
});
document.addEventListener('focusout', (event) => {
  const leaving = revealCell(event.target);
  if (!leaving) return;
  // The panel scrolls, which makes it a tab stop of its own, so the tab after
  // the cell moves focus *into* it — and closing on that hides the very thing
  // focus is arriving at: the keypress lands on nothing, and the part of the
  // panel below its own fold can never be read. Focus moving between the cell
  // and its own reveal is focus staying, the same way the pointer moving onto
  // the panel is the pointer staying.
  if (revealCell(event.relatedTarget) === leaving) return;
  FOCUSED = null;
  syncReveal();
});
// Capturing, so it hears the scroll region as well as the page: a wheel scroll
// does not end a hover, and a reveal left where it opened would drift away from
// the row that owns it.
document.addEventListener('scroll', syncReveal, true);

// Why a row is not `done` is the reason that row is degraded, so it goes through
// the same '!' the lab page uses rather than a `title` — a reason published to a
// mouse and to nothing else is a reason half the readers never get. Delegated at
// the document because the whole board is rebuilt on every pick.
document.addEventListener('click', (event) => {
  const mark = event.target && event.target.closest
    ? event.target.closest('.why') : null;
  if (!mark) return;
  const open = mark.parentElement.nextElementSibling;
  if (open && open.classList.contains('explain')) { open.remove(); return; }
  mark.parentElement.insertAdjacentHTML('afterend',
    `<p class="explain">${escapeHtml(mark.dataset.help || '')}</p>`);
});

// --- handing the experiment to the Laboratory -------------------------------
// The open button opens the Laboratory, and the same click makes that
// experiment's settings the knobs there. The board cannot write those knobs
// itself: only the lab page holds `/api/options`, so only it can tell a knob
// this installation serves from one this row merely recorded, and a value
// written into a `<select>` with no such option reads back as ''. So what
// crosses is an id, in one slot, and `experiment_handoff.js` says the rest.
//
// Nothing here prevents the default. The cell is an `<a>` so that middle-click,
// ⌘-click and Enter on a focused link all still reach the Laboratory; a handler
// that navigated by script instead would look the same and cost all three.
// Delegated at the document like every other listener on this page, because the
// whole board is rebuilt on each dataset pick.
document.addEventListener('click', (event) => {
  const open = event.target && event.target.closest
    ? event.target.closest('.open-run') : null;
  if (!open) return;
  // `Date.now()` is not decoration: a `storage` event fires on a change, so two
  // clicks on one row must not write the same bytes or an already-open
  // Laboratory tab hears the second one as nothing.
  ExperimentHandoff.offer(localStorage, open.dataset.experiment, Date.now());
});

const ASKED = new URLSearchParams(window.location.search);
QUERY = ASKED.get('filter') || '';
loadBoard(ASKED.get('dataset') || '');
