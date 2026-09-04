// The dataset viewer: one corpus, its ground truth, and whether the two can
// be measured together at all.
//
// Everything it draws comes from one request — `GET /api/dataset-content/<id>`
// — which carries the two files exactly as they are on disk plus the four
// readings the route derives beside the declaration table the panel's own
// dataset card reads. The largest bundled corpus is 269 KB, so the whole pair
// arrives once and every sort, filter and expansion after that happens here
// without asking the lab again.
//
// Nothing on this page writes. It is a reader surface: no run file, no ledger
// row, no archive, no knob, no fingerprint. It also takes no step ink — the
// three inks mean index, retrieval and generation, and reading a corpus is
// none of those.
//
// The four grid builders and both renderers take what they draw as arguments
// rather than reading the state below, so what a column is named and which
// rows a reading leaves are questions that can be asked of this file directly
// — `dataset_grids.test.js` asks them.

const $ = (id) => document.getElementById(id);

let DATA = null;      // the served pair, whole
let CATALOGUE = [];   // the corpora this installation lists
let CURRENT = '';     // the dataset on screen
let OPENED = null;    // which document's parts are shown, as a string
// Which reading is in force, if any: a grid name and the row identifiers that
// reading named. One at a time, because two narrowings on screen say nothing
// about which count you are reading.
let ONLY = null;
// The typed filter per grid. Held here rather than read back off the inputs
// because a reading click and a document expansion both rebuild the tables,
// and a query the reader can still see in the box has to still be in force.
const QUERY = {};

// --- one value, as a cell ---------------------------------------------------
// A label's value is whatever JSON the corpus declared it as: a string, a
// number, a list of topics, an object of confidences. A list reads as its
// members and an object as its JSON, because a cell that showed `[object
// Object]` would be hiding the very thing this page exists to show.
function shown(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (Array.isArray(value)) return value.length ? value.join(', ') : '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

// The columns one level of this dataset is described by: its own declared
// labels and no others. `applies_to` is the file's own statement of where a
// label may land, so a part-level label describes parts whether or not any
// part has been given a value for it yet. A declaration naming no level at
// all is shown at every level rather than dropped where nobody would find it.
function labelsAt(declarations, level) {
  return (declarations || []).filter((declaration) => {
    const levels = declaration.applies_to || [];
    return !levels.length || levels.includes(level);
  });
}

function labelColumns(declarations, level) {
  return labelsAt(declarations, level).map((declaration) => ({
    label: declaration.name,
    text: true,
    title: `${declaration.type || 'label'}${(declaration.levels || []).length
      ? ` · one of ${(declaration.levels || []).join(', ')}` : ''}`,
  }));
}

function labelCells(declarations, level, labels) {
  return labelsAt(declarations, level).map(
    (declaration) => shown((labels || {})[declaration.name]));
}

// --- the four grids ---------------------------------------------------------
// Each is a name, a caption, a filter example, its columns and its rows — and
// the first column of every one of them is the identity a reading names its
// rows by. Rows are arrays of plain values; the renderer below escapes them,
// so nothing here builds markup except the one cell that has to be a button.

function documentsGrid(data, opened) {
  const declared = data.dataset.label_declarations;
  return {
    name: 'documents',
    caption: `every document in ${data.dataset.name}`,
    hint: 'parts>1',
    columns: [
      // A button, because opening a document's parts has to be reachable from
      // the keyboard and not only by pointer. Its text is still the id, so the
      // column sorts and filters as the number a reader sees.
      { label: 'document', cell: (id) => `<button type="button" class="open-doc"`
        + ` data-document="${escapeHtml(id)}"`
        + ` aria-expanded="${String(String(id) === opened)}"`
        // A stable name with `aria-expanded` carrying the state, rather than
        // a name that changes to describe the next click: what the control is
        // does not change when it is open.
        + ` aria-label="Parts of document ${escapeHtml(id)}"`
        + `>${escapeHtml(id)}</button>` },
      { label: 'parts', title: 'how many parts this document is given as — a '
        + 'document given as one part can only be cut inside its text' },
      { label: 'chars', title: 'characters of part text — what the split '
        + 'plan\'s budget has to divide' },
      ...labelColumns(declared, 'document'),
    ],
    rows: (data.corpus.corpus_documents || []).map((document) => {
      const parts = document.document_content || [];
      return [
        document.corpus_document_id,
        parts.length,
        parts.reduce((total, part) => total + (part.text || '').length, 0),
        ...labelCells(declared, 'document', document.document_metadata),
      ];
    }),
  };
}

function partsGrid(data, opened) {
  const found = (data.corpus.corpus_documents || []).find(
    (entry) => String(entry.corpus_document_id) === opened);
  if (!found) return null;
  const declared = data.dataset.label_declarations;
  return {
    name: 'parts',
    caption: `the parts of document ${found.corpus_document_id}`,
    hint: 'text~broke',
    // Position and text alone where the corpus declares no part-level label:
    // a placeholder column for a label this corpus lacks would be a claim
    // about a field that does not exist.
    columns: [{ label: 'part' }, ...labelColumns(declared, 'part'),
              { label: 'text', text: true }],
    rows: (found.document_content || []).map((part, at) => [
      at + 1,
      ...labelCells(declared, 'part', part.labels),
      part.text || '',
    ]),
  };
}

function questionsGrid(data) {
  const declared = data.dataset.question_label_declarations;
  return {
    name: 'questions',
    caption: `every ground-truth question for ${data.dataset.name}`,
    hint: 'expects=abstain',
    columns: [
      { label: 'question' },
      { label: 'asks', text: true },
      // The one field the harness branches on, so it is named exactly as the
      // schema names it: answer from the corpus, abstain because the corpus
      // does not hold it, or correct the question's false premise.
      { label: 'expects', title: 'what the ground truth says a correct '
        + 'pipeline does with this question — answer, abstain, or '
        + 'correct_premise' },
      { label: 'cites', text: true,
        title: 'the documents this question names as its evidence' },
      { label: 'facts', title: 'how many derived facts the expected answer '
        + 'is broken into' },
      ...labelColumns(declared, 'question'),
    ],
    rows: (data.ground_truth.groundtruth_dataset || []).map((question) => {
      const expected = question.expected_answer || {};
      const cites = (question.relevant_corpus_documents || []).map(
        (relevant) => relevant.corpus_document_id);
      return [
        question.groundtruth_question_id,
        question.question || '',
        expected.behavior || '—',
        cites.length ? cites.join(', ') : '—',
        (expected.derived_facts || []).length,
        ...labelCells(declared, 'question', question.question_metadata),
      ];
    }),
  };
}

// The declarations themselves, as rows — the same reading the panel's dataset
// card shows, and here because the third reading is about them: a label
// declared and never filled in is a row of this table and of no other.
function labelsGrid(data) {
  return {
    name: 'labels',
    caption: `every label ${data.dataset.name} declares`,
    hint: 'file=corpus',
    columns: [
      { label: 'label', text: true },
      { label: 'file', title: 'the corpus declares the labels its documents '
        + 'and parts carry; the ground truth declares the labels its '
        + 'questions carry' },
      { label: 'type' },
      { label: 'applies to', text: true },
      { label: 'levels', text: true,
        title: 'the closed set of values, where the label declares one' },
      { label: 'extracted', title: 'whether a model produced this label '
        + 'rather than a person' },
      { label: 'rates', title: 'the label this one carries a confidence for' },
    ],
    rows: [
      ...(data.dataset.label_declarations || []).map((row) => [row, 'corpus']),
      ...(data.dataset.question_label_declarations || []).map(
        (row) => [row, 'questions']),
    ].map(([row, file]) => [
      row.name, file, row.type || '—', shown(row.applies_to),
      shown(row.levels), row.extracted ? 'yes' : 'no',
      row.confidence_for || '—',
    ]),
  };
}

// --- one grid, rendered -----------------------------------------------------
// `narrowed` is the row identifiers a reading named, or null for the whole
// grid. Narrowing renders fewer rows rather than hiding some: a reading is a
// question about which rows exist at all, while the typed filter below is a
// question about the rows on screen, and the two compose in that order.

function renderGrid(grid, narrowed) {
  const rows = narrowed
    ? grid.rows.filter((row) => narrowed.includes(String(row[0]))) : grid.rows;

  const head = grid.columns.map((column) => `<th scope="col"`
    + `${column.text ? ' class="text"' : ''}`
    + `${column.title ? ` title="${escapeHtml(column.title)}"` : ''}>`
    + `${escapeHtml(column.label)}</th>`).join('');

  const body = rows.map((row) => '<tr>' + grid.columns.map((column, at) =>
    `<td${column.text ? ' class="text"' : ''}>`
    + `${column.cell ? column.cell(row[at]) : escapeHtml(shown(row[at]))}</td>`
  ).join('') + '</tr>').join('');

  return `
    ${renderFilter(grid)}
    <div class="table-scroll" tabindex="0" role="region"
         aria-label="${escapeHtml(grid.caption)}">
      <table class="data-table centred" data-grid="${escapeHtml(grid.name)}">
        <caption>${escapeHtml(grid.caption)}</caption>
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>`;
}

// One line above each table, asking the same question `filtertable.js` answers
// on the board. Per grid rather than one for the page: `role=user` is a
// question about parts and means nothing about questions.
function renderFilter(grid) {
  return `
    <div class="filter-bar">
      <label for="filter-${grid.name}">Filter</label>
      <input id="filter-${grid.name}" class="filter-input" type="text"
             spellcheck="false" autocapitalize="off" autocomplete="off"
             data-grid="${grid.name}"
             value="${escapeHtml(QUERY[grid.name] || '')}"
             placeholder="${escapeHtml(grid.hint)}">
      <span class="filter-count" id="count-${grid.name}" role="status"></span>
      <p class="filter-said" id="said-${grid.name}" hidden></p>
    </div>`;
}

// --- the readings -----------------------------------------------------------
// Four counts, each one click away from the rows behind it. A reading that
// found nothing says zero and is not a button: there is nothing for it to
// narrow, and a control that does nothing is worse than a number.

function renderReadings(readings, only) {
  return `
    <div class="readings" role="group"
         aria-label="Whether this pair can be measured together">
      ${(readings || []).map((reading) => {
        const detail = (reading.detail || []).length
          ? `<span class="reading-detail">${escapeHtml(
              reading.detail.join(' · '))}</span>` : '';
        const inside = `<b class="reading-count">${reading.count}</b>`
          + `<span class="reading-says">${escapeHtml(reading.says)}</span>`;
        return reading.count
          ? `<button type="button" class="reading" data-reading="${
              escapeHtml(reading.id)}" aria-pressed="${String(
              Boolean(only) && only.id === reading.id)}"`
            + ` title="show only these rows in the ${escapeHtml(reading.grid)}`
            + ` table">${inside}</button>${detail}`
          : `<span class="reading reading-zero">${inside}</span>`;
      }).join('')}
    </div>`;
}

// --- the pair as given ------------------------------------------------------
// A collapsible tree of `<details>`/`<summary>`, which is the platform's own
// disclosure behaviour: keyboard-operable, themable, and found by the
// browser's own find. A viewer library would be a dependency loaded from a
// CDN — which this lab does not do — for something the browser already has.
//
// Scalars are printed as JSON, so a value is shown as recorded: `"one\n\ntwo"`
// reads as the two-newline string it is rather than as two lines a renderer
// decided to draw. That is the whole point of having a raw view beside grids
// that summarise.
function tree(value, key, depth) {
  const named = escapeHtml(key);
  if (value !== null && typeof value === 'object') {
    const entries = Array.isArray(value)
      ? value.map((item, at) => [at, item]) : Object.entries(value);
    const size = Array.isArray(value)
      ? `[${entries.length}]` : `{${entries.length}}`;
    return `<details${depth < 1 ? ' open' : ''}><summary>${named} `
      + `<span class="muted">${size}</span></summary>`
      + (entries.length
        ? entries.map(([name, item]) => tree(item, name, depth + 1)).join('')
        : '<div class="leaf muted">empty</div>')
      + '</details>';
  }
  return `<div class="leaf"><b>${named}</b> `
    + `<span>${escapeHtml(JSON.stringify(value))}</span></div>`;
}

function renderRaw(data, dataset) {
  return `
    <div class="raw-tree">
      ${tree(data.corpus, `${dataset}_corpus.json`, 0)}
      ${data.ground_truth_error
        ? `<p class="prose">No ground truth: ${escapeHtml(
            data.ground_truth_error)}</p>`
        : tree(data.ground_truth, `${dataset}_groundtruth.json`, 0)}
    </div>`;
}

// --- what the headings say when you ask them -------------------------------
// The page used to open with a paragraph, and each table carried another
// underneath it. That is an article, and this is an instrument: a reader who
// already knows what a corpus is should see figures and rows, not prose they
// have read once.
//
// So every sentence moved onto the thing it describes, where `lab.js` shows it
// on hover and on keyboard focus, and a click pins it under the heading for a
// reader with neither. Each entry does exactly one job — nothing here explains
// the page in general.
const SAYS = {
  page: 'The corpus every knob on the Laboratory page is measured against, '
    + 'read here and nowhere changed: this page writes no run, records no row, '
    + 'sets no knob and leaves every index fingerprint where it was. Pick '
    + 'another corpus from the button beside the name.',
  readings: 'The four questions that decide whether a corpus and its questions '
    + 'can be measured together at all, rather than merely describing them. '
    + 'Each is a count you can press to see the rows behind it, and each says '
    + 'zero rather than disappearing — so a clean corpus reads differently '
    + 'from a check that never ran. The last one matters most before a sweep: '
    + 'a separator stage cutting on a blank line can match nothing in a '
    + 'corpus that has none, unless index.part_join puts one between parts.',
  labels: 'Every label this dataset declares, and nothing else. These are the '
    + 'columns the Documents and Questions tables are described by — a corpus '
    + 'that declares speaker where the diary declares role is described by a '
    + 'speaker column, and no column stands for a label this corpus never '
    + 'declared. A label keeps its column whether or not any row has been '
    + 'given a value for it yet.',
  documents: 'Every document in the corpus, with how many parts it is given '
    + 'as and how many characters of text it holds — the two figures a '
    + 'split plan divides. Press a document number to read the parts it is '
    + 'made of, which are the units a part or label stage cuts between.',
  questions: 'Every ground-truth question, what it expects a correct pipeline '
    + 'to do with it, and which documents it names as its evidence. Expects '
    + 'is the one field the harness branches on: answer means the corpus '
    + 'holds it, abstain means it does not and refusing honestly is what is '
    + 'being measured, and correct_premise means the question itself is wrong '
    + 'and saying so is the right answer.',
  raw: 'Both files exactly as recorded, value by value — whitespace and '
    + 'nesting included, nothing reformatted or rounded. The tables summarise; '
    + 'this is where you check the shape of a value, which is the only place '
    + 'a blank line or a stray space is visible.',
};

// What a heading answers with. `lab.js` owns the hover and the focus, and
// `data-brief` carrying the whole note rather than an opening sentence is what
// makes this the long explanation rather than a teaser for it.
//
// The attribute only. The class that makes an element a trigger (`why-term`,
// which is what `lab.js` matches on) is written at each call site instead: a
// helper that emitted its own `class` gave the tabs two class attributes, and
// a browser keeps the first and drops the second — so the tabs looked right
// and explained nothing.
const asked = (topic) => ` data-brief="${escapeHtml(SAYS[topic])}"`;

// --- which corpus ----------------------------------------------------------
// The same affordance the board uses for its dataset scope, down to the
// classes and the native `popover`: the browser owns show, hide,
// light-dismiss and Escape, so there is no second implementation of one
// control across two pages.

function renderPicker() {
  const found = CATALOGUE.find((entry) => entry.id === CURRENT);
  return `
    <div class="board-bar">
      <button class="context-scope" type="button" popovertarget="pick-list">
        <span class="context-label">Dataset</span>
        <span>${escapeHtml(found ? (found.name || found.id) : CURRENT)}</span>
        <span class="context-caret" aria-hidden="true">▾</span>
      </button>
      <ul class="context-detail" id="pick-list" popover
          aria-label="Which dataset this page shows">
        ${CATALOGUE.map((entry) => `<li><button type="button" class="pick"
           data-dataset="${escapeHtml(entry.id)}"
           ${entry.id === CURRENT ? 'aria-current="true"' : ''}
           >${escapeHtml(entry.name || entry.id)}</button></li>`).join('')}
      </ul>
    </div>`;
}

// --- three tabs -------------------------------------------------------------
// A corpus is read one way at a time: you are looking at its documents, or at
// the questions put to it, or at the files themselves. Stacked as five cards
// they were one long scroll where the third thing was never on screen with the
// first; as tabs each is a whole screen and the reader says which.
//
// Only the chosen panel is built. That is not an optimisation for its own
// sake: the diary's raw tree alone is 60,000 nodes, and building all three at
// once would be paying for two screens nobody is looking at.
const TABS = [
  { id: 'documents', label: 'Documents' },
  { id: 'questions', label: 'Questions' },
  { id: 'raw', label: 'The pair as given' },
];

let TAB = 'documents';

function renderTabs() {
  return `
    <div class="subtabs" role="tablist" aria-label="How to read this dataset">
      ${TABS.map((tab) => {
        const on = tab.id === TAB;
        return `<button type="button" role="tab" class="subtab why-term"
          id="tab-${tab.id}" data-tab="${tab.id}"
          aria-selected="${String(on)}" aria-controls="panel-${tab.id}"
          tabindex="${on ? '0' : '-1'}"${asked(tab.id)}
          >${escapeHtml(tab.label)}</button>`;
      }).join('')}
    </div>`;
}

// The panel is one region whatever is in it, labelled by the tab that chose
// it, so a screen reader arriving here is told which of the three it is
// reading rather than finding an unnamed table.
function renderPanel(data) {
  const parts = partsGrid(data, OPENED);
  if (TAB === 'documents') {
    return renderGrid(documentsGrid(data, OPENED), narrowing('documents'))
      // The parts table's own name is read aloud from its caption but drawn
      // nowhere, and a second table appearing under the first with no heading
      // leaves a sighted reader to infer what it is. It is one line.
      + (parts ? '<div class="parts-panel">'
        + `<h3>${escapeHtml(parts.caption)}</h3>`
        + `${renderGrid(parts, null)}</div>` : '');
  }
  if (TAB === 'questions') {
    return data.ground_truth_error
      ? `<p class="prose">${escapeHtml(data.ground_truth_error)}</p>`
      : renderGrid(questionsGrid(data), narrowing('questions'));
  }
  return renderRaw(data, CURRENT);
}

// --- the page ---------------------------------------------------------------

const narrowing = (grid) => (ONLY && ONLY.grid === grid ? ONLY.ids : null);

// The corpus's size, as three figures rather than a sentence of them. Its own
// row because none of it is a control: the readings below are pressable and
// these are not, and one row mixing the two would offer a click that does
// nothing.
const renderSize = (data) => `
  <p class="corpus-size">
    ${['documents', 'parts', 'questions'].map((noun) =>
      `<span><b>${data.dataset[noun]}</b> ${noun}</span>`).join('')}
    <span class="corpus-source">${escapeHtml(data.dataset.source)}</span>
  </p>`;

function render() {
  $('dataset').innerHTML = `
    <section class="card">
      <div class="card-head">
        <h2 class="why-term"${asked('page')}>${escapeHtml(DATA.dataset.name || CURRENT)}</h2>
        ${renderPicker()}
      </div>
      ${renderSize(DATA)}
      <p class="prose">${escapeHtml(DATA.dataset.description || '')}</p>

      <h3 class="why-term"${asked('readings')}>Can this pair be measured?</h3>
      ${renderReadings(DATA.readings, ONLY)}

      <h3 class="why-term"${asked('labels')}>Labels this dataset declares</h3>
      <div class="labels-block">
        ${renderGrid(labelsGrid(DATA), narrowing('labels'))}
      </div>
    </section>

    ${renderTabs()}
    <section class="card subtab-panel" role="tabpanel" id="panel-${TAB}"
             aria-labelledby="tab-${TAB}">
      ${renderPanel(DATA)}
    </section>`;

  // The sorter re-appends every row it holds on each reorder, which drops the
  // hidden ones back among the visible ones, so the filter re-runs after it.
  // `make` calls `onApply` once on wiring too, which is what starts each
  // table's count correct rather than correct-after-the-first-click.
  for (const table of $('dataset').querySelectorAll('table')) {
    SortTable.make(table, { onApply: () => applyFilter(table) });
  }
}

// Narrow one table to what its own box says, and say what happened. The same
// two readings the board publishes: how many rows are shown, and what a term
// asked that this table cannot answer.
function applyFilter(table) {
  const name = table.dataset.grid;
  const count = $(`count-${name}`);
  const said = $(`said-${name}`);
  if (!count || !said) return;
  const out = FilterTable.apply(table, QUERY[name] || '');
  const quoted = (words) => words.map((word) => `“${word}”`).join(', ');
  const trouble = out.unknown.length
    ? `no column called ${quoted(out.unknown)} here`
    : out.bad.length ? `${quoted(out.bad)} asks nothing` : '';
  said.textContent = trouble ? `${trouble} — the rows below are unchanged.` : '';
  said.hidden = !trouble;
  count.textContent = out.shown === out.total
    ? `${out.total} row${out.total === 1 ? '' : 's'}`
    : `${out.shown} of ${out.total} shown`;
}

// `dataset` may be empty, which is the first load with nothing in the URL: the
// catalogue then decides which corpus opens, rather than this file carrying an
// id of its own that would have to be kept in step with the lab's default.
async function load(dataset) {
  const box = $('dataset');
  box.innerHTML = '<div class="card"><p class="prose">Reading the corpus…</p></div>';
  try {
    CATALOGUE = ((await (await fetch('/api/datasets')).json()).datasets) || [];
    const wanted = dataset || (CATALOGUE[0] || {}).id || '';
    const pair = await fetch(`/api/dataset-content/${encodeURIComponent(wanted)}`);
    if (!pair.ok) {
      const body = await pair.json().catch(() => ({}));
      throw new Error(body.detail || `the lab answered ${pair.status}`);
    }
    DATA = await pair.json();
  } catch (error) {
    // A refusal says which id was asked for and what this lab does list, so
    // the page never looks like an installation with no corpora in it.
    box.innerHTML = '<div class="card"><div class="card-head">'
      + '<h2>Could not read that dataset</h2></div>'
      + `<p class="prose">${escapeHtml(error.message)}</p></div>`;
    return;
  }
  CURRENT = DATA.dataset.id;
  OPENED = null;
  ONLY = null;
  TAB = 'documents';
  render();
}

// Delegated at the document and registered once: the whole page is rebuilt on
// a dataset pick, a tab, a reading click and a document expansion, so a
// listener added per render would stack.
document.addEventListener('click', (event) => {
  const target = event.target;
  if (!target || !target.closest) return;

  const pick = target.closest('.pick');
  if (pick) {
    // In the URL, so a corpus somebody is reading is a link they can send.
    const url = new URL(window.location.href);
    url.searchParams.set('dataset', pick.dataset.dataset);
    window.history.replaceState({}, '', url);
    load(pick.dataset.dataset);
    return;
  }

  const tab = target.closest('.subtab');
  if (tab) {
    // A heading that is also a tab: `lab.js` answers the hover, and this
    // answers the press. The two never collide, because the brief is closed
    // on any click before this runs.
    if (tab.dataset.tab !== TAB) { TAB = tab.dataset.tab; render(); }
    return;
  }

  const open = target.closest('.open-doc');
  if (open) {
    // Clicking the open document closes it again, so the panel is a toggle
    // rather than something that can only ever be replaced.
    OPENED = OPENED === open.dataset.document ? null : open.dataset.document;
    render();
    return;
  }

  const reading = target.closest('.reading[data-reading]');
  if (reading) {
    const found = (DATA.readings || []).find(
      (entry) => entry.id === reading.dataset.reading);
    if (!found) return;
    ONLY = ONLY && ONLY.id === found.id
      ? null
      : { id: found.id, grid: found.grid, ids: found.ids.map(String) };
    // A count is only turned into its rows if those rows are on screen: a
    // reading about questions opens the Questions tab, because narrowing a
    // table the reader cannot see would look like nothing happening.
    if (ONLY && (found.grid === 'documents' || found.grid === 'questions')) {
      TAB = found.grid;
    }
    render();
    return;
  }

  // The reader with no pointer, and the reader on a touch screen: hovering is
  // not available to either, so the same note a hover shows is pinned under
  // its heading by a press. Toggled, so a second press puts it away.
  const heading = target.closest('.why-term[data-brief]');
  if (heading && !heading.classList.contains('subtab')) pinned(heading);
});

// One note at a time, under the heading that owns it.
function pinned(heading) {
  const already = heading.parentElement.querySelector(':scope > .explain');
  const mine = already && already.dataset.of === heading.dataset.brief;
  if (already) already.remove();
  if (mine) return;
  heading.insertAdjacentHTML('afterend',
    `<p class="explain" data-of="${escapeHtml(heading.dataset.brief)}">`
    + `${escapeHtml(heading.dataset.brief)}</p>`);
}

// Arrow keys walk the tabs, which is what makes them tabs rather than three
// buttons in a row: a reader on a keyboard reaches the group once and moves
// inside it, instead of tabbing past every one of them to get to the table.
document.addEventListener('keydown', (event) => {
  const tab = event.target && event.target.closest
    ? event.target.closest('.subtab') : null;
  if (!tab) return;
  const step = { ArrowRight: 1, ArrowLeft: -1, Home: -TABS.length,
                 End: TABS.length }[event.key];
  if (step === undefined) return;
  event.preventDefault();
  const at = TABS.findIndex((entry) => entry.id === TAB);
  const to = Math.min(TABS.length - 1, Math.max(0, at + step));
  TAB = TABS[to].id;
  render();
  const landed = document.getElementById(`tab-${TAB}`);
  if (landed) landed.focus();
});

document.addEventListener('input', (event) => {
  const box = event.target;
  if (!box || !box.dataset || !box.dataset.grid) return;
  QUERY[box.dataset.grid] = box.value;
  const table = document.querySelector(`table[data-grid="${box.dataset.grid}"]`);
  if (table) applyFilter(table);
});

load(new URLSearchParams(window.location.search).get('dataset') || '');
