// The Inspector's whole frontend: four views over the read-only :9003 API —
// ground truth, chunks, retrieval, generation — three of which auto-follow
// whatever the lab (:9002) actually ran.
let CHOSEN = null;   // fallback config, served by /api/config so it cannot drift
async function loadChosen() {
  CHOSEN = (await (await fetch('/api/config')).json()).chosen;
}
const chosenReady = loadChosen();

const views = ['groundtruth', 'chunks', 'retrieval', 'generation'];
const tabOf = view => document.getElementById(`tab-${view}`);

// Roving tabindex: a tablist is one stop in the page's tab order, not four, and
// the selected tab is the one that stop lands on. Reached by four buttons that
// only ever set `aria-selected`, which is inert without `role="tab"` — so a
// screen reader was told which of the four views was showing by nothing at all,
// and there was no way between them but Tab, Tab, Tab.
function show(view) {
  for (const v of views) {
    const on = v === view;
    document.getElementById(`view-${v}`).hidden = !on;
    const tab = tabOf(v);
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
  }
}
for (const v of views) tabOf(v).addEventListener('click', () => show(v));

// The arrows walk the strip and switch as they go: all four panels are already
// in the page, so there is nothing to fetch and nothing a reader would gain by
// having to confirm the move. Home and End go to the ends. Same shape as the
// question picker further down, which is the keyboard implementation this page
// already had right.
document.querySelector('.inspector-tabs').addEventListener('keydown', event => {
  const at = views.findIndex(v => tabOf(v) === document.activeElement);
  if (at < 0) return;
  const next = {
    ArrowRight: (at + 1) % views.length,
    ArrowLeft: (at - 1 + views.length) % views.length,
    Home: 0,
    End: views.length - 1,
  }[event.key];
  if (next === undefined) return;
  event.preventDefault();
  show(views[next]);
  tabOf(views[next]).focus();
});
show('groundtruth');

async function pollJob(jobId) {
  for (;;) {
    const job = await (await fetch(`/api/jobs/${jobId}`)).json();
    if (job.state === 'done') return job.result;
    if (job.state === 'error') throw new Error(job.error || 'job failed');
    await new Promise(r => setTimeout(r, 500));
  }
}

// Turn a POST response into its job id, or throw the server's own reason
// (400/404/409) instead of letting callers poll `/api/jobs/undefined` forever.
async function startedJob(response) {
  const body = await response.json();
  if (!body.job_id) throw new Error(body.detail || 'could not start job');
  return body.job_id;
}

// "showing: session · ascii-hash · hybrid-rrf k=8 · lexical · grader=none" —
// the config that produced whatever is on screen, so a reader is never left
// guessing which pipeline the rows in front of them came from.
function formatConfig(cfg) {
  if (!cfg) return '';
  const parts = [];
  // Corpus first when it is not the built-in diary — two corpora are not two
  // configurations of one measurement, and this must not go unstated.
  if (cfg.index && cfg.index.dataset) parts.push(cfg.index.dataset);
  if (cfg.index) parts.push(`${cfg.index.chunker} · ${cfg.index.embedder}`);
  if (cfg.retrieval) {
    parts.push(`${cfg.retrieval.retriever} k=${cfg.retrieval.k}`,
               cfg.retrieval.reranker, `grader=${cfg.retrieval.grader}`);
  }
  return `showing: ${parts.join(' · ')}`;
}

// --- What every score means: the '!' marks, reading the lab's own text -------
// Fetched from /api/explain rather than written here, so this page and the
// panel on :9002 cannot end up explaining the same metric differently.
let EXPLAIN = { metrics: [], help: {} };

// Which followed job each view is currently drawing. Declared up here because
// `loadGroundTruth` clears two of these when the fixture lands, and it runs
// before the follow loop is set up further down.
const followed = { indexJobId: null, queryJobId: null, retrievalJobId: null,
                   generationJobId: null, dataset: '' };

function measureOf(key) {
  return EXPLAIN.metrics.find(m => m.key === key) || { key, label: key };
}

// The sentence the '!' opens: a metric's own note and formula when it has them,
// falling back to the help topic the lab writes for the same key.
function whyText(key) {
  const m = measureOf(key);
  return [m.note, m.formula && `formula: ${m.formula}`, m.source && `computed by ${m.source}`,
          EXPLAIN.help[`metric.${key}`]].filter(Boolean).join(' — ');
}

function whyMark(key) {
  const label = escapeHtml(measureOf(key).label || key);
  return `<button type="button" class="why" data-why="${escapeHtml(key)}"`
    + ` aria-label="What is ${label}?">!</button>`;
}

// One listener for the page: the marks are re-rendered on every poll tick, and
// a listener per button would leak one per render.
document.addEventListener('click', event => {
  const button = event.target.closest('button.why');
  if (!button) return;
  const open = button.nextElementSibling;
  if (open && open.classList.contains('why-text')) { open.remove(); return; }
  const note = document.createElement('span');
  note.className = 'why-text';
  note.textContent = whyText(button.dataset.why) || 'no description for this one yet';
  button.after(note);
});

async function loadExplain() {
  try { EXPLAIN = await (await fetch('/api/explain')).json(); }
  catch (error) { /* the marks fall back to the bare key; not worth failing on */ }
}
loadExplain();

// --- Ground truth ---
// Kept by id as well as rendered, because the retrieval and generation views
// restate a question's own facts and ideal answer beside its rows, and those
// belong to the fixture rather than to any run.
const GT = new Map();

// Which corpus this map holds. `''` is the built-in diary, and is what the page
// starts on so the tab has something in it before the first poll answers.
let FOLLOWED_DATASET = '';

// --- which direction the corpus reads ---
// This page used to answer that with a hardcoded rtl written into fourteen template
// strings and a Persian face pinned into four CSS rules, because the first
// corpus was a Farsi diary. Four of the five bundled corpora are German or
// English, and every one of them rendered right-to-left in Vazirmatn with its
// chunk column against the wrong edge. The dataset has always known its
// language; `/api/groundtruth` says so now, and this is the page asking.
//
// The scripts that run right to left, by language subtag. A list rather than a
// script lookup because there is no way to ask the platform offline, and this
// list is short and does not move.
const RTL_LANGUAGES = new Set(['ar', 'arc', 'ckb', 'dv', 'fa', 'he', 'ku',
                               'ps', 'sd', 'ug', 'ur', 'yi']);

// `auto` until a corpus says otherwise — not `ltr`, and not the diary's `rtl`.
// `auto` is a real answer: the browser takes each block's direction from its
// own first strong character, which is the honest reading for text whose
// language nothing has stated. An archive written without one lands here.
let CORPUS_DIR = 'auto';

function dirFor(language) {
  const base = String(language || '').toLowerCase().split(/[-_]/)[0];
  if (!base) return 'auto';
  return RTL_LANGUAGES.has(base) ? 'rtl' : 'ltr';
}

// One fact, set in one place, read two ways: the attribute is what the
// stylesheet reads to put the chunk column and its header against the right
// edge (that is layout, and belongs to the page), while `CORPUS_DIR` goes onto
// each block of corpus text as its own `dir` (that is the text's own property,
// and is what bidi, the font stack, selection and a screen reader all need).
function setCorpusDir(language) {
  CORPUS_DIR = dirFor(language);
  document.documentElement.dataset.corpusDir = CORPUS_DIR;
}

function renderGroundTruth(body) {
  FOLLOWED_DATASET = body.dataset || '';
  // Before the first row is written, because every render below reads it.
  setCorpusDir(body.language);
  const root = document.getElementById('view-groundtruth');
  GT.clear();
  root.innerHTML = '';
  for (const q of body.questions) {
    GT.set(q.id, q);
    const row = document.createElement('div');
    row.className = 'gt-row';
    // Label above its text, never beside it. A label set inline with a
    // right-aligned Farsi block ends up at the opposite edge of the row from the
    // thing it labels, with the width of the page in between.
    const field = (label, text, corpusText) => text
      ? `<div class="gt-field"><div class="qh-label">${label}</div>`
        + `<div${corpusText ? ` dir="${CORPUS_DIR}"` : ''}>${escapeHtml(text)}</div></div>` : '';
    const quotes = (q.evidence || []).map(e => e.quote);
    row.innerHTML = `<div class="gt-head"><span class="q-id">${escapeHtml(q.id)}</span> `
      + `<span class="q-tally">${escapeHtml(q.type)} · ${escapeHtml(q.difficulty)}`
      + `${q.answerable ? '' : ' · unanswerable'}</span></div>`
      + `<div class="gt-q" dir="${CORPUS_DIR}">${escapeHtml(q.question_fa)}</div>`
      + `<div class="gt-en">${escapeHtml(q.question_en || '')}</div>`
      + field('answer', q.answer_fa, true)
      + (quotes.length
         ? `<div class="gt-field"><div class="qh-label">evidence quoted from the diary</div>`
           + quotes.map(quote =>
               `<div dir="${CORPUS_DIR}" class="gt-quote">${escapeHtml(quote)}</div>`).join('')
           + `</div>` : '');
    root.appendChild(row);
  }
  // The first poll can beat this fetch; forgetting the rendered job ids forces
  // a redraw once the fixture is available, rather than leaving every ideal
  // answer showing '—' until the next run.
  followed.retrievalJobId = null;
  followed.generationJobId = null;
  if (!picker.hidden) renderPicker(pickerFilter.value);
}

// One named corpus's fixture, fetched and handed back unrendered. Split out
// because a read-only mode pinned to a record has to fetch it too, and must not
// be turned away by the guard below — that guard is there to stop a *live*
// fetch landing after something recorded went on screen.
async function fetchGroundTruth(dataset) {
  const requestedDataset = dataset || '';
  // Named on every request, never assumed, so this view can't show one corpus
  // while another view on the page shows a different one.
  const response = await fetch('/api/groundtruth?dataset='
                               + encodeURIComponent(requestedDataset));
  const body = await response.json().catch(() => ({ detail: response.statusText }));
  if (!response.ok) throw new Error(body.detail || response.statusText);
  return body;
}

async function loadGroundTruth(dataset) {
  const body = await fetchGroundTruth(dataset);
  // A live fetch that began before a read-only mode must not land afterwards
  // and replace what is pinned with a different corpus — either mode, since an
  // imported archive and a recorded experiment both put one on screen.
  if (activeArchiveId !== null || archiveLoadingId !== null
      || activeExperimentId !== null || recordLoadingId !== null) return;
  renderGroundTruth(body);
}
loadGroundTruth('');

// Added questions are rows about ids the old corpus had; called only from the
// poll, since the initial load has no previous corpus and `ADDED` doesn't
// exist yet when this file first runs.
async function followDataset(dataset) {
  ADDED.clear();
  renderAdded();
  await loadGroundTruth(dataset);
}

// --- Chunks: shared render, used by both the followed view and the manual one ---
function renderChunkGroups(container, groups) {
  container.innerHTML = '';
  for (const g of groups) {
    const det = document.createElement('details');
    det.className = 'chunk-session';
    det.innerHTML = `<summary><span class="q-id">${escapeHtml(g.session_id)}</span> `
      + `<span class="q-tally">${g.chunks.length} chunk`
      + `${g.chunks.length === 1 ? '' : 's'}${g.date ? ' · ' + escapeHtml(g.date) : ''}`
      + `</span></summary>`;
    g.chunks.forEach((c, i) => {
      const line = document.createElement('div');
      line.className = 'chunk-line';
      // The number is a Latin marker on its own line rather than a prefix inside
      // the Farsi string, where bidi puts it at whichever edge the run ends on.
      line.innerHTML = `<div class="chunk-no">chunk ${i + 1}</div>`
        + `<div dir="${CORPUS_DIR}">${escapeHtml(c.text)}</div>`;
      det.appendChild(line);
    });
    container.appendChild(det);
  }
}

// Each card leads with what its text cannot say — group, level, chunk count,
// and session count. A group spanning several sessions carries no session id
// at all, which is why the session count must be stated rather than shown.
function renderSummaries(container, summaries, absentNote) {
  container.innerHTML = '';
  if (!summaries.length) {
    // "No hierarchy" and "a hierarchy that found nothing" are different facts, and
    // this view must not read as the second when it is the first. A third fact —
    // "nobody wrote this down" — belongs to a record that never reported any,
    // and only the caller holding that record can tell it from a flat build.
    container.innerHTML = absentNote
      || '<p class="empty-note">This index is flat — no grouping was '
      + 'asked for, so there are no summaries to list. Pick a grouping under '
      + '"Summary hierarchy" on the lab to build some.</p>';
    return;
  }
  for (const s of summaries) {
    const det = document.createElement('details');
    det.className = 'chunk-session';
    det.innerHTML = `<summary>`
      + `<span class="layer-badge" data-step="index">L${s.level} `
      + `${escapeHtml(s.group_id || '')} · ${s.members}</span> `
      + `<span class="q-tally">${s.members} chunk${s.members === 1 ? '' : 's'} `
      + `· ${s.sessions} session${s.sessions === 1 ? '' : 's'}`
      + `${s.chars ? ' · ' + s.chars + ' chars' : ''}</span></summary>`;
    const line = document.createElement('div');
    line.className = 'chunk-line';
    line.innerHTML = `<div class="chunk-no">summary</div>`
      + `<div dir="${CORPUS_DIR}">${escapeHtml(s.text)}</div>`;
    det.appendChild(line);
    // The members by id, so a summary can be traced back to the rows it stands
    // for — without them the card is an assertion the reader cannot check.
    const members = document.createElement('div');
    members.className = 'chunk-line';
    members.innerHTML = '<div class="chunk-no">members</div>'
      + `<div class="summary-members">${(s.member_ids || [])
          .map((id) => escapeHtml(id)).join(', ')}</div>`;
    det.appendChild(members);
    container.appendChild(det);
  }
}

// The two halves are held here rather than re-fetched, so switching costs nothing
// and can never show one build's leaves beside another's summaries.
const chunkView = { summaries: [], absentNote: '' };

// One way in, so no caller can leave the previous source's note attached to
// this one's rows: every assignment states what an empty list means here.
function setChunkSummaries(summaries, absentNote = '') {
  chunkView.summaries = summaries || [];
  chunkView.absentNote = absentNote;
}

function showChunkMode(mode) {
  document.getElementById('chunks-body').hidden = mode !== 'chunks';
  document.getElementById('summaries-body').hidden = mode !== 'summaries';
  for (const button of document.querySelectorAll('#chunks-mode button')) {
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  }
  if (mode === 'summaries') {
    renderSummaries(document.getElementById('summaries-body'),
                    chunkView.summaries, chunkView.absentNote);
  }
}

for (const button of document.querySelectorAll('#chunks-mode button')) {
  button.addEventListener('click', () => showChunkMode(button.dataset.mode));
}

// Build an index here and list what it chunked. Takes the config rather than
// reading one, because two controls ask for this: the button beside the tab
// builds under the config this page falls back to, and a recorded experiment's
// Chunks tab offers a rebuild under the config that experiment recorded. One
// request path, so the status line, the config line and the summaries toggle
// cannot be updated by one caller and forgotten by the other.
async function buildChunks(config) {
  const status = document.getElementById('chunks-status');
  try {
    status.textContent = 'building…';
    await chosenReady;
    const asked = config || CHOSEN;
    const response = await fetch('/api/chunks',
      { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(asked) });
    const result = await pollJob(await startedJob(response));
    // Both counts, because "167 chunks" over an index of 174 rows is the exact
    // statement that made seven summaries invisible.
    status.textContent = `${result.total} chunks · `
      + `${result.total_summaries || 0} summaries`;
    // The config line follows what was actually built: after a rebuild these
    // are today's chunks, and the line must not go on describing a record.
    document.getElementById('chunks-active-config').textContent = formatConfig(asked);
    renderChunkGroups(document.getElementById('chunks-body'), result.chunks_by_session);
    setChunkSummaries(result.summaries);
    showChunkMode(document.getElementById('summaries-body').hidden
      ? 'chunks' : 'summaries');
  } catch (error) {
    status.textContent = error.message;
  }
}

// `null`, not `CHOSEN`: the fallback config is still being fetched while this
// listener is being attached, and the build reads it after awaiting that fetch.
document.getElementById('build-chunks').addEventListener('click',
  () => buildChunks(null));

// --- Retrieval: shared render ---
// One candidate per row, cloned from the page's own template so every table
// carries the same header. Row background is ground-truth relevance
// (white/gray) — a different axis from `kept`, the pipeline's own column.
// `gold_spans` always comes from the service, never a browser search: a
// candidate can be gold by the quote *containing* it, with nothing verbatim to
// mark, and a range invented here would paint text the ground truth never
// quoted.
function highlighted(text, spans) {
  const source = text || '';
  if (!spans || !spans.length) return escapeHtml(source);
  let out = '', at = 0;
  for (const [start, end] of spans) {
    out += escapeHtml(source.slice(at, start))
      + `<mark class="evidence-mark">${escapeHtml(source.slice(start, end))}</mark>`;
    at = end;
  }
  return out + escapeHtml(source.slice(at));
}

// Skips a leading [ … ] contextual header, so a 60-char preview isn't the same
// shape of metadata for every chunk; the reveal still shows the header, since
// it was part of what got embedded.
function previewOf(text) {
  const close = text.startsWith('[') ? text.indexOf(']') : -1;
  return (close === -1 ? text : text.slice(close + 1)).trim() || text;
}

function chunkCell(candidate) {
  const text = candidate.text || '';
  const spans = candidate.gold_spans || [];
  const preview = previewOf(text);
  // Gold with no span is a real state, not an error, and saying so is the whole
  // reason the spans are computed where the marking is.
  const footnote = (candidate.gold && !spans.length)
    ? '<span class="no-evidence">gold: this chunk sits inside the evidence quote, '
      + 'so there is no verbatim span to highlight</span>' : '';
  // `data-sort` set explicitly: the cell's own text is preview *and* reveal,
  // and the column must sort by what the reader sees, not by both at once.
  // A summary row wears the index ink and names its group, so it reads as
  // build-written rather than mistaken for the diarist's own words.
  const badge = candidate.layer === 'summary'
    ? `<span class="layer-badge" data-step="index">`
      + `L${candidate.level} ${escapeHtml(candidate.group_id || '')} `
      + `· ${candidate.members}</span> ` : '';
  // What the badge means, in the reveal rather than in a `title`. That this row
  // is not the corpus's own words is the most important thing about it, and a
  // mouse-hover tooltip is not a way to publish that — the reveal is where the
  // row's full text already goes, and it opens to a keyboard as well.
  const layerNote = candidate.layer === 'summary'
    ? '<span class="no-evidence">a summary this build wrote over '
      + `${candidate.members} chunk${candidate.members === 1 ? '' : 's'} `
      + '— not the corpus\'s own words</span>' : '';
  return `<td class="chunk-cell" data-sort="${escapeHtml(preview.slice(0, 60))}">`
    + badge
    + `<span class="chunk-preview" dir="${CORPUS_DIR}" tabindex="0">`
    + `${escapeHtml(preview.slice(0, 60))}${preview.length > 60 ? '…' : ''}</span>`
    + `<div class="chunk-reveal" dir="${CORPUS_DIR}">${highlighted(text, spans)}`
    + `${footnote}${layerNote}</div></td>`;
}

// One bar per rank-producing step (dense, BM25, RRF fusion), height standing
// for how high the chunk placed there. `cap` is this table's own candidate
// count, so the scale is the run's depth, not an arbitrary constant.
// aria-hidden: the same three ranks follow in their own columns.
function ladder(candidate, cap) {
  const steps = [['dense', candidate.dense_rank], ['bm25', candidate.bm25_rank],
                 ['RRF', candidate.fused_rank]];
  if (steps.every(([, rank]) => !rank)) {
    return '<td><span class="ladder-none">·</span></td>';
  }
  const cells = steps.map(([, rank]) => {
    // Rank 1 is full height; anything past the table's depth keeps a stub, so a
    // bar is never absent in a way that could read as "no data".
    const height = rank ? Math.max(8, Math.round(100 * (1 - (rank - 1) / Math.max(1, cap)))) : 0;
    return `<i class="ladder-step" style="--h:${height}%"></i>`;
  }).join('');
  const label = steps.map(([name, rank]) => `${name} ${rank || '—'}`).join(' · ');
  return `<td><span class="ladder" title="${escapeHtml(label)}" aria-hidden="true">`
    + `${cells}</span></td>`;
}

// A table inside the scroll region both surfaces share (chrome.css): nine
// columns move the table and never the page, the header stays put because the
// region's height is bounded, and the region takes focus so the arrow keys,
// PageUp/PageDown, Home and End reach the right-hand side of it. It used to
// scroll only at narrow widths, because on a wide screen it clipped the reveal
// hanging below a row; the reveal is fixed to the viewport now (`placeReveal`),
// so the region can be real at every width.
function scrollable(table, label) {
  const box = document.createElement('div');
  box.className = 'table-scroll';
  box.tabIndex = 0;
  box.setAttribute('role', 'region');
  box.setAttribute('aria-label', label || 'retrieved candidates');
  box.appendChild(table);
  return box;
}

// Where an open reveal goes is `placeReveal`, in lab.js: the board on :9002
// grew a reveal off a cell in the same bounded region, and the placement is the
// same problem on both surfaces, so it is answered once in the file both pages
// load rather than copied. The selector is passed because only the caller knows
// which reveal its cell holds.

// The cell whose reveal a pointer or a focus has opened. Hover opens on the
// whole row (`.retrieval-row:hover .chunk-reveal`), so the event target is
// usually some rank cell three columns away from the text it reveals.
function revealCell(node) {
  const row = node && node.closest && node.closest('.retrieval-row');
  return row ? row.querySelector('.chunk-cell') : null;
}

// Delegated at the document, because every one of these tables is rebuilt each
// time the lab finishes a job. The scroll listener is capturing: a wheel scroll
// does not end a hover, so a reveal left where it opened would drift away from
// the row that owns it.
document.addEventListener('pointerover',
  event => placeReveal(revealCell(event.target), '.chunk-reveal'));
document.addEventListener('focusin',
  event => placeReveal(revealCell(event.target), '.chunk-reveal'));
document.addEventListener('scroll', () => {
  for (const cell of document.querySelectorAll(
    '.retrieval-row:hover .chunk-cell, .chunk-cell:focus-within')) {
    placeReveal(cell, '.chunk-reveal');
  }
}, true);

function retrievalTable(candidates) {
  const table = document.getElementById('retrieval-table-template')
    .content.firstElementChild.cloneNode(true);
  const body = table.querySelector('tbody');
  const rows = candidates || [];
  const cell = v => (v === null || v === undefined) ? '·' : v;
  for (const c of rows) {
    const tr = document.createElement('tr');
    tr.className = 'retrieval-row ' + (c.gold ? 'retrieval-row--gold' : 'retrieval-row--plain');
    tr.innerHTML = chunkCell(c) + ladder(c, rows.length)
      + `<td class="num">${cell(c.dense_rank)}</td>
      <td class="num">${cell(c.bm25_rank)}</td>
      <td class="num">${cell(c.fused_rank)}</td>
      <td class="num">${cell(c.rerank_score)}</td>
      <td class="num">${cell(c.grade_score)}</td>
      <!-- The two mark columns sort on 1/0 rather than on their glyph: an empty
           cell would read as "never measured" and pin itself to the bottom in
           both directions, so clicking twice would not flip them. -->
      <td data-sort="${c.kept ? 1 : 0}">${c.kept ? '✓' : '✗'}</td>
      <td data-sort="${c.gold ? 1 : 0}">${c.gold ? '●' : ''}</td>`;
    body.appendChild(tr);
  }
  // Sortable once the rows are in it: nine columns is nine questions you might be
  // asking of one question's candidates — what dense alone would have returned,
  // what the reranker promoted, which gold chunk the gate dropped.
  SortTable.make(table);
  return table;
}

// The question restated above its own rows, with the facts a right answer would
// have contained. `id` is enough to find both in the fixture.
function questionHead(questionId, fallbackFa) {
  const q = GT.get(questionId) || {};
  const facts = (q.key_facts || []).map(f => `<li>${escapeHtml(f)}</li>`).join('');
  return `<div class="question-head">`
    + `<div class="qh-fa" dir="${CORPUS_DIR}">${escapeHtml(q.question_fa || fallbackFa || '')}</div>`
    + `<div class="qh-en">${escapeHtml(q.question_en || '')}</div>`
    + (facts ? `<div class="qh-label">what a right answer contains</div>`
             + `<ol class="qh-facts">${facts}</ol>` : '')
    + `</div>`;
}

// The one-line summary a collapsed question shows. Same shape in both views, so
// a reader scanning either list is reading the same sentence.
function questionSummary(id, type, difficulty, tally) {
  return `<summary><span class="q-id">${escapeHtml(id)}</span> `
    + `<span class="q-tally">${escapeHtml(type || '')} · ${escapeHtml(difficulty || '')}`
    + `${tally ? ' — ' + escapeHtml(tally) : ''}</span></summary>`;
}

function renderRetrievalRows(candidates) {
  const host = document.getElementById('retrieval-body');
  host.innerHTML = '';
  host.appendChild(scrollable(retrievalTable(candidates),
    'candidates for the one-off query'));
}

// One collapsible table per question, collapsed by default, shared by both the
// followed list and questions you add — so both come from the same code.
// The agent's per-node ladder, when a scope produced one. Read vertically: the
// same node three times over is a loop that never settled, distinct from
// having simply run out of hops.
function agentLadder(visits) {
  const box = document.createElement('div');
  box.className = 'agent-ladder';
  const hops = visits.reduce((n, v) => Math.max(n, v.hop || 0), 0);
  box.innerHTML = `<h4>the loop · ${visits.length} steps · ${hops} hop`
    + `${hops === 1 ? '' : 's'}</h4><table><thead><tr><th>#</th><th>node</th>`
    + '<th>hop</th><th>what it decided</th></tr></thead><tbody>'
    + visits.map((v, i) =>
        `<tr><td class="num">${i + 1}</td><td class="node">${escapeHtml(v.node)}</td>`
        + `<td class="num">${v.hop || ''}</td>`
        + `<td class="detail" dir="auto">${escapeHtml(v.detail || '')}</td></tr>`)
      .join('')
    + '</tbody></table>';
  // Wired, like every other table on either surface. It was the one built by a
  // path that never reached the sorter — and the two questions a reader brings
  // to a loop trace are both column questions: sort by node and a node that
  // appears three times collects itself, sort by hop and you see what each hop
  // cost. The third click restores the order it was served in, which for this
  // table is the sequence itself.
  //
  // Deliberately not inside a `.table-scroll`: four columns, one of which
  // wraps, so it has nothing to overflow. A bounded, bordered region around a
  // table that never scrolls is the component worn as costume.
  SortTable.make(box.querySelector('table'));
  return box;
}

function questionBlock(q) {
  const candidates = (q.trace && q.trace.candidates) || [];
  const gold = candidates.filter(c => c.gold).length;
  const kept = candidates.filter(c => c.kept).length;
  const visits = (q.trace && q.trace.agent) || [];
  // Falls back to a bare count when `gold_available` is absent (an older run),
  // rather than implying a total nobody measured.
  const goldTally = q.gold_available === null || q.gold_available === undefined
    ? `${gold} gold` : `${gold} of ${q.gold_available} gold found`;
  const det = document.createElement('details');
  det.className = 'retrieval-question';
  det.innerHTML = questionSummary(q.question_id, q.type, q.difficulty,
      `${candidates.length} candidates · ${kept} kept · ${goldTally}`)
    + questionHead(q.question_id, q.question_fa);
  // Above the candidate table, because the ladder is how the candidates came to
  // be: the table answers "what came back", the ladder answers "after what".
  if (visits.length) det.appendChild(agentLadder(visits));
  det.appendChild(scrollable(retrievalTable(candidates),
    `candidates for ${q.question_id}`));
  return det;
}

function renderQuestionTables(questions) {
  const host = document.getElementById('retrieval-questions');
  host.innerHTML = '';
  for (const q of questions) host.appendChild(questionBlock(q));
}

// --- Adding a question ------------------------------------------------------
// Follows the page's active config so an added row is measured the same way as
// the rows beside it (falling back to the chosen config if the lab is down).
// The corpus is overlaid rather than read from `FOLLOWED_CONFIG` itself, since
// that config can name a different corpus than the one the picker is showing —
// otherwise a question run against the wrong one 404s for an id just offered.
function activeConfig() {
  const cfg = FOLLOWED_CONFIG || CHOSEN;
  return { ...cfg, index: { ...cfg.index, dataset: FOLLOWED_DATASET } };
}
let FOLLOWED_CONFIG = null;

// Every question you have added, in the order you added them. Kept here rather
// than read back off the DOM, because the followed views re-render whenever the
// lab finishes a new job and would otherwise wipe them.
const ADDED = new Map();

const picker = document.getElementById('question-picker');
const pickerButton = document.getElementById('add-question');
const pickerList = document.getElementById('question-picker-list');
const pickerFilter = document.getElementById('question-picker-filter');

function openPicker(open) {
  picker.hidden = !open;
  pickerButton.setAttribute('aria-expanded', String(open));
  if (open) { pickerFilter.value = ''; renderPicker(''); pickerFilter.focus(); }
  else pickerButton.focus();
}

// Difficulty is the one thing worth seeing before you read a word of the
// question: it is what makes a miss interesting or expected. Carried as colour
// here and nowhere else — in the tables colour already means a pipeline step.
function renderPicker(filter) {
  const needle = filter.trim().toLowerCase();
  const matches = [...GT.values()].filter(q => !needle
    || q.id.toLowerCase().includes(needle)
    || (q.question_fa || '').toLowerCase().includes(needle)
    || (q.question_en || '').toLowerCase().includes(needle));
  pickerList.innerHTML = matches.length ? '' : '<div class="q-empty">'
    + 'No question matches that. Clear the filter to see all of them.</div>';
  for (const q of matches) {
    const option = document.createElement('div');
    option.className = `q-option q-option--${q.difficulty || 'easy'}`;
    option.setAttribute('role', 'option');
    option.setAttribute('aria-selected', String(ADDED.has(q.id)));
    option.tabIndex = -1;
    option.dataset.id = q.id;
    const quotes = (q.evidence || []).map(ev =>
      `<div class="gt-quote" dir="${CORPUS_DIR}">${escapeHtml(ev.quote)}</div>`).join('');
    option.innerHTML =
      `<span class="q-option-id">${escapeHtml(q.id)}</span>`
      + `<span class="q-chip q-chip--${escapeHtml(q.difficulty || 'easy')}">`
      + `${escapeHtml(q.difficulty || '')}</span>`
      + `<span class="q-option-en">${escapeHtml(q.question_en || q.question_fa)}</span>`
      + (ADDED.has(q.id) ? '<span class="q-option-added">added</span>' : '')
      // The detail is in the DOM from the start rather than built on hover, so
      // it opens with no delay and reads the same to a screen reader.
      + `<div class="q-option-detail">`
      + `<div dir="${CORPUS_DIR}" class="q-option-fa">${escapeHtml(q.question_fa)}</div>`
      + `<div class="qh-label">expected answer</div>`
      + `<div dir="${CORPUS_DIR}">${escapeHtml(q.answer_fa || '—')}</div>`
      + (quotes ? `<div class="qh-label">evidence quoted from the diary</div>${quotes}` : '')
      + `</div>`;
    pickerList.appendChild(option);
  }
}

pickerButton.addEventListener('click', () => openPicker(picker.hidden));
pickerFilter.addEventListener('input', () => renderPicker(pickerFilter.value));
pickerList.addEventListener('click', event => {
  const option = event.target.closest('.q-option');
  if (option) addQuestion(option.dataset.id);
});

// A listbox has to be usable from the keyboard, or the difficulty colours and
// the hover detail are both only available to a mouse.
picker.addEventListener('keydown', event => {
  const options = [...pickerList.querySelectorAll('.q-option')];
  const at = options.indexOf(document.activeElement);
  if (event.key === 'Escape') { openPicker(false); return; }
  if (event.key === 'Enter' && at >= 0) {
    event.preventDefault(); addQuestion(options[at].dataset.id); return;
  }
  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
  event.preventDefault();
  const next = event.key === 'ArrowDown'
    ? Math.min(options.length - 1, at + 1) : Math.max(0, at - 1);
  if (options[next]) options[next].focus();
});

async function addQuestion(questionId) {
  const status = document.getElementById('retrieval-status');
  openPicker(false);
  try {
    status.textContent = `running ${questionId}…`;
    await chosenReady;
    const response = await fetch('/api/questions',
      { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ ...activeConfig(), question_id: questionId }) });
    const result = await pollJob(await startedJob(response));
    ADDED.set(questionId, result);
    renderAdded();
    status.textContent = `${questionId} added — it is in both tabs`;
  } catch (error) {
    status.textContent = `${questionId}: ${error.message}`;
  }
}

function renderAdded() {
  const retrievalHost = document.getElementById('retrieval-added');
  const generationHost = document.getElementById('generation-added');
  retrievalHost.innerHTML = '';
  generationHost.innerHTML = '';
  if (!ADDED.size) return;
  const heading = () => {
    const label = document.createElement('div');
    label.className = 'qh-label added-label';
    label.textContent = `added by you — scored the same way, but not part of `
      + `the run's own sample`;
    return label;
  };
  retrievalHost.appendChild(heading());
  generationHost.appendChild(heading());
  for (const [, result] of ADDED) {
    retrievalHost.appendChild(questionBlock(result.retrieval));
    generationHost.appendChild(generationBlock(result.generation, result.retrieval.trace));
  }
}

// --- Generation: the ideal answer, the written one, and the scores ----------
// Per-question deterministic scores, in pipeline order. RAGAS's judged metrics
// are per *run*, not per question, so they are rendered once above instead.
const GEN_METRICS = ['answer_similarity', 'answer_token_f1', 'key_fact_coverage',
                     'abstained_correctly', 'false_abstention'];

const fmt = v => (v === null || v === undefined) ? '·'
  : (typeof v === 'number' ? (Math.round(v * 1000) / 1000).toString() : String(v));

function metricLine(row, keys) {
  const present = keys.filter(k => row[k] !== undefined && row[k] !== null);
  if (!present.length) return '<div class="gen-metrics">no scores for this one</div>';
  return '<div class="gen-metrics">' + present.map(k =>
    `<span class="gen-metric">${escapeHtml(measureOf(k).label || k)}: `
    + `<b>${fmt(row[k])}</b>${whyMark(k)}</span>`).join('') + '</div>';
}

// `traces` is keyed by question id and only passed when the retrieval on screen
// came from this same evaluation — showing another run's ranks under this run's
// answer would invent a pipeline that never existed.
function renderGeneration(view, traces) {
  const ragasHost = document.getElementById('generation-ragas');
  const host = document.getElementById('generation-questions');
  const ragas = (view.ragas && view.ragas.metrics) || {};
  const keys = Object.keys(ragas);
  ragasHost.innerHTML = keys.length
    ? '<div class="gen-ragas"><h4>judged over the whole run</h4>'
      + metricLine(ragas, keys)
      + (view.ragas.decision !== null && view.ragas.decision !== undefined
         ? `<div class="gen-metrics"><span class="gen-metric">decision score `
           + `<b>${fmt(view.ragas.decision)}</b>${whyMark('ragas_decision')}</span></div>` : '')
      + '</div>'
    : '<div class="inspector-active-config">no RAGAS judging on this run '
      + '(ragas_mode=off) — the deterministic scores below are what exists</div>';

  host.innerHTML = '';
  for (const row of view.rows || []) {
    host.appendChild(generationBlock(row, traces && traces.get(row.id)));
  }
}

// One question's generation block. Shared with the added questions for the same
// reason `questionBlock` is: the comparison is only honest if both went through
// the same renderer.
function generationBlock(row, trace) {
  const gt = GT.get(row.id) || {};
  const det = document.createElement('details');
  det.className = 'gen-question';
  const scored = GEN_METRICS.filter(k => row[k] !== undefined && row[k] !== null);
  const tally = row.abstained ? 'abstained'
    : `${scored.length} score${scored.length === 1 ? '' : 's'}`;
  det.innerHTML = questionSummary(row.id, row.type, row.difficulty, tally)
    + questionHead(row.id, '')
    + '<div class="gen-answers">'
    + '<div class="gen-answer gen-answer--ideal"><h4>what the diary says</h4>'
    + `<div dir="${CORPUS_DIR}">${escapeHtml(gt.answer_fa || '—')}</div></div>`
    + '<div class="gen-answer gen-answer--actual">'
    + `<h4>what this run wrote${row.abstained ? ' — it refused' : ''}</h4>`
    + `<div dir="${CORPUS_DIR}">${escapeHtml(row.answer || '—')}</div></div>`
    + '</div>'
    + metricLine(row, GEN_METRICS);
  if (trace) {
    const inner = document.createElement('details');
    inner.className = 'gen-trace';
    inner.innerHTML = '<summary>the retrieval this answer was written from</summary>';
    if ((trace.agent || []).length) inner.appendChild(agentLadder(trace.agent));
    inner.appendChild(scrollable(retrievalTable(trace.candidates || [])));
    det.appendChild(inner);
  }
  return det;
}

// --- Auto-follow: poll the lab (:9002) through our own /api/follow every ~2s,
// and only touch the DOM when the followed job actually changed — a tab
// re-rendering on every tick would collapse the user's expanded <details>. ---

function showLabDown(el) {
  el.textContent = 'Nothing to show until the lab is running. Start it with '
    + '`npm run raglab`.';
}

// One line in the header, so an empty view never leaves the reader guessing
// whether nothing ran or nothing is listening.
function setFollowState(body) {
  const el = document.getElementById('follow-state');
  el.dataset.lab = body.lab;
  el.textContent = body.lab === 'up'
    ? `following the lab at ${body.lab_url}`
    : `cannot reach the lab at ${body.lab_url}`;
}

let activeArchiveId = null;
let archiveLoadingId = null;
let liveDatasetBeforeArchive = '';

// The other read-only mode: one row of the lab's ledger, opened from the board.
// Declared beside the archive's own state because the two share this page's
// read-only chrome and only one of them can be on screen.
let activeExperimentId = null;
let recordLoadingId = null;
// Why a deep-linked record is *not* on screen, when that is the answer. Held
// rather than written straight into the state line, because the page is still
// following the lab in that case and the follow loop rewrites that line.
let recordProblem = '';

// `label` is what kind of recorded thing the page is pinned to, because there
// are two — an imported archive and one row of the lab's ledger — and they are
// read-only for different reasons. The chrome is deliberately the same: this
// line and the control beside it already mean "you are pinned to something
// recorded, and here is the way back", which is exactly both states.
function setArchiveState(runId, label = 'Imported archive') {
  const state = document.getElementById('archive-state');
  const button = document.getElementById('archive-return-live');
  const readOnly = runId !== null;
  // A record the page could not produce leaves it on live and says why here,
  // so this line stays up in that one case — the follow loop calls this every
  // couple of seconds and would otherwise wipe the reason immediately.
  state.hidden = !readOnly && !recordProblem;
  // And the control stays with it, because a stated failure a reader cannot
  // dismiss is a banner and a stale `?experiment=` in the URL with no way out
  // of either. It says what it does in each case: leaving a record is a return
  // to live, while a record that never loaded was never left.
  button.hidden = !readOnly && !recordProblem;
  button.textContent = readOnly ? 'Return to live' : 'Dismiss';
  document.getElementById('build-chunks').disabled = readOnly;
  document.getElementById('add-question').disabled = readOnly;
  if (readOnly) state.textContent = `${label} · read-only · ${runId}`;
  else state.textContent = recordProblem;
}

function renderImportedArchive(archive) {
  const evaluation = archive.evaluation;
  const evidence = evaluation.inspector;
  const result = evaluation.result;
  if (activeArchiveId === null) liveDatasetBeforeArchive = FOLLOWED_DATASET;
  FOLLOWED_CONFIG = result.config;
  ADDED.clear();
  renderAdded();
  renderGroundTruth({ dataset: evidence.dataset.id,
                      meta: evidence.dataset.ground_truth.meta,
                      questions: evidence.dataset.ground_truth.questions,
                      // The archive carries the corpus it was run against, and
                      // the corpus half is where a language lives — so an
                      // archive is read in its own direction rather than in
                      // whatever the live corpus happened to be. An archive
                      // written without one falls through to `auto`, which is
                      // the honest answer for text whose language is unstated.
                      language: ((evidence.dataset.corpus || {}).meta || {}).language,
                      datasets: [] });
  renderChunkGroups(document.getElementById('chunks-body'),
                    evidence.chunks_by_session);
  setChunkSummaries(evidence.summaries);
  showChunkMode(document.getElementById('summaries-body').hidden
    ? 'chunks' : 'summaries');
  renderQuestionTables(evidence.traces);
  const traces = new Map(evidence.traces.map(row => [row.question_id, row.trace]));
  renderGeneration({ job_id: result.run_id, config: result.config,
                     rows: result.rows, summary: result.summary,
                     ragas: result.ragas }, traces);
  for (const id of ['chunks-active-config', 'retrieval-active-config',
                    'retrieval-set-config', 'generation-active-config']) {
    document.getElementById(id).textContent = formatConfig(result.config);
  }
  document.getElementById('retrieval-body').innerHTML = '';
  document.getElementById('retrieval-answer').textContent = '';
  setArchiveState(result.run_id);
}

async function archiveRequest(path, method = 'GET') {
  const response = await fetch(path, { method });
  const body = await response.json().catch(() => ({ detail: response.statusText }));
  if (!response.ok) throw new Error(body.detail || response.statusText);
  return body;
}

async function followImportedArchive(archiveId) {
  if (archiveId === activeArchiveId) return;
  archiveLoadingId = archiveId;
  try {
    const archive = await archiveRequest(
      '/api/imported-archives/' + encodeURIComponent(archiveId));
    renderImportedArchive(archive);
    activeArchiveId = archiveId;
  } finally {
    archiveLoadingId = null;
  }
}

// Back to following the lab, from whichever read-only mode was pinned: forget
// every rendered job id so the next tick redraws all four views from live jobs
// rather than deciding nothing changed. Shared by both modes because leaving
// them is the same work — only what has to be un-pinned first differs.
async function returnToLive(dataset) {
  FOLLOWED_CONFIG = null;
  followed.indexJobId = null;
  followed.queryJobId = null;
  followed.retrievalJobId = null;
  followed.generationJobId = null;
  // Leave the live dataset dirty until its ground truth arrives. If that
  // request fails, the next follow tick sees a change and retries it.
  followed.dataset = null;
  setArchiveState(null);
  await loadGroundTruth(dataset);
  followed.dataset = dataset;
}

async function leaveArchiveMode() {
  const dataset = liveDatasetBeforeArchive;
  activeArchiveId = null;
  await returnToLive(dataset);
}

document.getElementById('archive-return-live').addEventListener('click', async () => {
  // One control, three states. Only an imported archive is something the *lab*
  // is holding open, so only that one has anything to clear over there — a
  // recorded experiment was never more than this page reading a record, and a
  // record that failed to load leaves only a message and a URL to clear.
  if (activeExperimentId !== null || recordProblem) {
    await leaveRecordMode();
    return;
  }
  const state = document.getElementById('archive-state');
  let cleared = false;
  try {
    state.textContent = 'Returning to live…';
    await archiveRequest('/api/imported-archives/active', 'DELETE');
    cleared = true;
    await leaveArchiveMode();
  } catch (error) {
    state.hidden = false;
    state.textContent = cleared
      ? `Returning to live · ${error.message} · retrying…`
      : `Imported archive · read-only · ${error.message}`;
  }
});

// --- a recorded experiment, opened from the board ---------------------------
// The leaderboard's frozen right column links to `?experiment=<id>`. That is a
// third state for this page — not live, not an imported archive, but one row of
// the lab's ledger read back — and it wears archive mode's chrome deliberately:
// the state line and the return-to-live control already mean "you are pinned to
// something recorded, read-only, and here is the way back", which is exactly
// this. A second read-only mode with its own controls would be two things doing
// one job.

async function followRecordedExperiment(experimentId) {
  if (experimentId === activeExperimentId) return;
  recordLoadingId = experimentId;
  let record = null;
  let failure = '';
  try {
    record = await archiveRequest('/api/experiments/'
                                  + encodeURIComponent(experimentId));
  } catch (error) {
    failure = error.message;
  } finally {
    // Cleared before anything below asks for the live fixture, since the guard
    // that holds a live fetch off while a record is in flight reads this.
    recordLoadingId = null;
  }
  if (failure) {
    // Left on live rather than pinned to nothing: a read-only view with no
    // evidence in it reads as an experiment that recorded none. A row deleted
    // from the ledger, a mistyped id and a lab that is down all land here.
    recordProblem = `${experimentId} could not be read from the lab `
      + `(${failure}) — showing live instead`;
    setArchiveState(null);
    // And the live fixture is asked for again: the page's own boot fetch was
    // turned away while this record was in flight, and the follow loop reloads
    // only on a corpus *change* — so the ground-truth tab would otherwise stay
    // empty for want of a request nobody was going to make.
    await loadGroundTruth(followed.dataset || '');
    return;
  }
  recordProblem = '';
  // Captured before anything is pinned, so returning to live goes back to the
  // corpus this page was following rather than to the record's own.
  if (activeArchiveId === null) liveDatasetBeforeArchive = FOLLOWED_DATASET;
  // A record and an archive preview cannot both be on screen, and the deep link
  // is the more recent request; dropping the id here is also what lets the
  // follow loop re-enter archive mode once the reader goes back to live.
  activeArchiveId = null;
  activeExperimentId = experimentId;
  await renderRecordedExperiment(record);
}

// Which of the ledger's shapes a record is, told from the two columns the row
// already carries. It decides what each view can show and, where a view is
// empty, why — and deciding that from how many rows the detail happened to hold
// was one test doing five jobs: a cancelled run was told it had measured no
// questions, a one-off query was told nothing was retrieved while its own
// record held the trace and the answer, and an errored retrieval was handed an
// index build's explanation.
function recordShape(record) {
  const detail = record.detail || {};
  // An imported archive is preserved verbatim, so its detail is the archive
  // payload and not a job's result: no `config`, no `rows`, no `traces`.
  if (detail.format === 'raglab-experiment' && detail.evaluation) return 'archive';
  // A job that stopped holds its config and nothing after it, whatever kind of
  // job it was — so the reason every view is empty is the row's own state, not
  // the shape of a result it never produced.
  if ((record.state || 'done') !== 'done') return 'unfinished';
  return record.kind || 'run';
}

// The state line: the kind, and the state as well when the state is the point.
// A read-only view of a cancelled experiment that never says "cancelled" is a
// row not carrying the reason it is degraded — the rule the board's own state
// column keeps, said here in the one line this page has for it.
function recordLabel(record) {
  const kind = record.kind || 'kind unrecorded';
  const state = record.state || '';
  return `Recorded experiment (${state && state !== 'done'
    ? `${kind} · ${state}` : kind})`;
}

// Why a stopped job has no evidence, said per view: the job stopped. The reason
// the row recorded travels with it, because a page showing one experiment is
// the only place a reader can read that reason at all.
function unfinishedNote(record, missing) {
  return '<p class="empty-note">This experiment was <b>'
    + `${escapeHtml(record.state || 'never finished')}</b>, so ${missing}: a job `
    + 'that stopped recorded its config and nothing after it. '
    + (record.error
       ? `The reason it recorded — ${escapeHtml(record.error)}`
       : 'It recorded no reason beyond the state itself.')
    + '</p>';
}

// Three of the four tabs come off the record: the ledger strips only chunk
// text, so the config, the per-question traces and the answered rows all
// survive into it. What is missing is stated where it is missing, in the terms
// of the shape this record actually is.
async function renderRecordedExperiment(record) {
  const detail = record.detail || {};
  const shape = recordShape(record);
  const config = detail.config || {};
  FOLLOWED_CONFIG = config;
  // Questions added by hand were run against the live corpus under the live
  // config, so they are not evidence about this record.
  ADDED.clear();
  renderAdded();

  // An archive's payload already has a renderer on this page — the one the
  // archive control uses, which reads every field of it — so the record hands
  // over rather than describing that shape a second way. Read as a job result
  // it emptied all four tabs for a row whose decision score the board can show,
  // and offered a rebuild that would have fallen back to the *live* config
  // under a note promising this experiment's.
  if (shape === 'archive') {
    renderImportedArchive(detail);
    setArchiveState(record.experiment_id, recordLabel(record));
    return;
  }

  // The record names its own corpus, and a corpus is read in the direction its
  // own language reads — so the fixture is fetched for *that* dataset rather
  // than left showing whatever the lab happens to be working on.
  try {
    renderGroundTruth(await fetchGroundTruth(record.dataset || ''));
  } catch (error) {
    // A corpus this installation no longer has is a stated gap, not an empty
    // question list: the rows below still came from one.
    GT.clear();
    // And its direction is unstated, not the live corpus's: `auto` is the
    // honest reading for text whose language nothing here can report, and is
    // what an archive written without one falls through to as well.
    setCorpusDir('');
    document.getElementById('view-groundtruth').innerHTML =
      '<p class="empty-note">This experiment ran against '
      + `<b>${escapeHtml(record.dataset || 'the built-in corpus')}</b>, which `
      + `this installation cannot load (${escapeHtml(error.message)}). Its `
      + 'questions below are named by id only, with no ideal answer to '
      + 'compare against.</p>';
  }

  // `traces` on an evaluation, `questions` on a retrieval: two names for one
  // shape, normalised here the way `/api/follow` normalises the live pair.
  const traces = detail.traces || detail.questions || [];
  const rows = detail.rows || [];
  const questions = document.getElementById('retrieval-questions');
  const generation = document.getElementById('generation-questions');
  const answer = document.getElementById('retrieval-answer');
  // The one-off query boxes follow a live job, and a ledger row is not one —
  // so they are cleared for every shape but the one that actually recorded a
  // single traced question, whose evidence goes in exactly here.
  document.getElementById('retrieval-body').innerHTML = '';
  answer.textContent = '';
  document.getElementById('generation-ragas').innerHTML = '';
  renderQuestionTables(shape === 'query' ? [] : traces);

  if (shape === 'unfinished') {
    questions.innerHTML = unfinishedNote(record, 'nothing was retrieved');
    generation.innerHTML = unfinishedNote(record, 'no answer was written');
  } else if (shape === 'query') {
    // A one-off query is one question traced once — the single-question shape
    // this page already draws for a live query, in the same two boxes, from the
    // record's own trace and answer instead of from a job that is running. Read
    // as an evaluation it was told nothing had been retrieved and nothing
    // answered, while its record held both.
    questions.innerHTML = '<p class="empty-note">This experiment is a one-off '
      + '<b>query</b>: one question, traced once, with no selected question '
      + 'set — so there is no per-question list here. Its candidates are in the '
      + 'single table below, and the answer it wrote is under them.</p>';
    renderRetrievalRows(((detail.trace || {}).candidates) || []);
    // An answer is written in the corpus's language, so it reads in the
    // corpus's direction — settled after the fixture above, never in markup.
    answer.dir = CORPUS_DIR;
    answer.textContent = detail.answer || '';
    generation.innerHTML = '<p class="empty-note">A one-off query is not '
      + 'judged: it writes no run file and no decision score, so there is '
      + 'nothing scored to show here. Its question, its candidates and its '
      + 'answer are on the Retrieval tab.'
      + (detail.abstained
         ? ' This query <b>abstained</b> — the pipeline declined to answer '
           + 'from what it retrieved rather than answering anyway.' : '')
      + '</p>';
  } else {
    if (!traces.length) {
      // Two different reasons a build or an evaluation has no candidate tables,
      // and they must not be told in one sentence. An experiment that answered
      // questions ran a retrieval and it was simply never written down — chunk
      // text, traces and summaries do not reach a run file — while an index
      // build measured no questions at all.
      questions.innerHTML = rows.length
        ? '<p class="empty-note">The per-question retrieval of this experiment '
          + 'was <b>not recorded</b> — its answers and scores were written to a '
          + 'run file, and candidate rankings never travel there. The answers '
          + 'themselves are under Generation; what was retrieved to write them '
          + 'is not recoverable from this record.</p>'
        : '<p class="empty-note">This experiment recorded no per-question '
          + 'retrieval — an index build measures no questions, so there are no '
          + 'candidate tables to read here.</p>';
    }
    if (rows.length) {
      // Keyed by question id, and handed over only because these ranks and
      // these answers are the same experiment's — the live path's own rule.
      renderGeneration({ job_id: record.experiment_id, config: config,
                         rows: rows, summary: detail.summary,
                         ragas: detail.ragas },
                       new Map(traces.map(row => [row.question_id, row.trace])));
    } else {
      // Each kind in its own terms. "A build or a retrieval" covered both in
      // one sentence, which left the reader of either to work out which half
      // was about the row in front of them.
      generation.innerHTML = '<p class="empty-note">This experiment wrote no '
        + 'answers — ' + ({
          index: 'an index build chunks and embeds a corpus and stops there',
          retrieve: 'a retrieval ranks candidates and stops there, and only an '
            + 'evaluation goes on to generate',
        }[record.kind] || 'only an evaluation generates')
        + ', so there is nothing to show here and nothing was scored.</p>';
    }
  }
  showRecordedChunks(record, shape);

  // All four, so no view is left describing the live pipeline beside recorded
  // rows. After `showRecordedChunks`, which is why the chunks note carries its
  // own sentence about the rebuild rather than leaning on this line.
  const shown = formatConfig(recordedStages(record));
  for (const id of ['chunks-active-config', 'retrieval-active-config',
                    'retrieval-set-config', 'generation-active-config']) {
    document.getElementById(id).textContent = shown;
  }
  setArchiveState(record.experiment_id, recordLabel(record));
}

// Only the stages this experiment actually ran. A build stores a whole config
// — retrieval and generation defaults included — and no part of a build reads
// them; the record's own columns are blank there for that reason, so printing
// `bm25 k=3 · grader=none` under a build would have this page and the board
// telling two stories about one experiment.
function recordedStages(record) {
  const config = (record.detail || {}).config || {};
  if (record.kind !== 'index') return config;
  return { index: config.index };
}

// The one tab a record cannot fill. A record keeps no chunk text by design
// — the text belongs to the index a config produces, and copying it per row
// would put the corpus in the ledger — so this says so and offers the rebuild
// rather than taking it: a rebuild is today's index, and today's chunks shown
// under an old experiment's id would be a row lying about what produced it.
function showRecordedChunks(record, shape) {
  const detail = record.detail || {};
  const config = detail.config;
  // Offered only where the record actually named one. `buildChunks(null)` falls
  // back to the live config on purpose, for the button beside the tab that has
  // no other config to use — and taking that fallback here would rebuild under
  // a config this experiment never ran, under a note promising this
  // experiment's, which is the one substitution this tab exists to refuse.
  const rebuildable = !!(config && Object.keys(config).length);
  const body = document.getElementById('chunks-body');
  body.innerHTML = (shape === 'unfinished'
    ? unfinishedNote(record, 'no chunk text was recorded')
    : '<p class="empty-note">The chunk text of this experiment was <b>not '
      + 'recorded</b> — a record keeps the config that produces an index, never '
      + 'a copy of the corpus.</p>')
    + (rebuildable
       ? '<p class="empty-note">A rebuild below is today\'s index under this '
         + 'experiment\'s recorded config, which is not the same claim: the '
         + 'corpus or the chunker may have changed since it ran.</p>'
       : '<p class="empty-note">Its config was not recorded either, so there '
         + 'is nothing here to rebuild it from.</p>');
  if (rebuildable) {
    const rebuild = document.createElement('button');
    rebuild.type = 'button';
    rebuild.id = 'rebuild-recorded-chunks';
    rebuild.textContent = 'Rebuild from this config';
    // The one request path, the same the button beside the tab uses — so the
    // status line, the config line and the summaries toggle all follow.
    rebuild.addEventListener('click', () => buildChunks(config));
    // In the page's own control row rather than loose in the view: a bare
    // <button> here inherits the page's light ink onto the browser's own light
    // chassis, and the label all but disappears.
    const row = document.createElement('div');
    row.className = 'inspector-controls';
    row.appendChild(rebuild);
    body.appendChild(row);
  }
  // Summaries are the toggle's other half and a record does keep those: they
  // are text a build wrote, not the corpus's own. A record that reported none
  // at all is a third state, and says so rather than claiming a flat index.
  setChunkSummaries(detail.summaries,
    Array.isArray(detail.summaries) ? '' :
      '<p class="empty-note">This experiment recorded no summaries either way '
      + '— whether its index was flat or grouped is not in the row.</p>');
  showChunkMode('chunks');
}

async function leaveRecordMode() {
  // Which corpus to go back to: the one this page was following before a record
  // pinned it, or — when nothing was ever pinned, because the record failed to
  // load and only its message is being dismissed — the one already on screen.
  const dataset = activeExperimentId !== null
    ? liveDatasetBeforeArchive : FOLLOWED_DATASET;
  activeExperimentId = null;
  recordProblem = '';
  // The record's own note about its missing chunk text goes when the record
  // does — but only if it is still what the view holds. A rebuild the reader
  // asked for replaced it with chunks that are genuinely on screen, and those
  // are not this mode's to throw away.
  if (document.getElementById('rebuild-recorded-chunks')) {
    document.getElementById('chunks-body').innerHTML = '';
    setChunkSummaries([]);
  }
  // Off the URL as well as out of the variable: a page still carrying
  // `?experiment=` would pin itself again on the next reload, which is not what
  // pressing "return to live" asked for.
  const url = new URL(window.location.href);
  url.searchParams.delete('experiment');
  history.replaceState(null, '', url);
  await returnToLive(dataset);
}

async function renderFollow(body) {
  // Before the early-returning archive branch: a page opened while an archive
  // is already active must still say whether the lab is reachable.
  setFollowState(body);
  // A pinned record outranks both the lab's live jobs and its archive preview:
  // the reader followed a link to that experiment, not to whatever is running
  // now. Below `setFollowState` for the same reason the archive branch is.
  if (activeExperimentId !== null || recordLoadingId !== null) return;
  // A stated failure to read a deep-linked experiment holds this page on live
  // too. An archive the lab happens to be holding open is not an answer to the
  // link the reader followed, and letting it in overwrote the message with an
  // unrelated one — leaving the reader on an archive they never asked for, told
  // nothing, with a "Return to live" that routed to the record's own exit and
  // so never cleared the lab's archive. Dismissing the message clears
  // `recordProblem`, and the next tick enters archive mode properly.
  if (body.archive_id && !recordProblem) {
    await followImportedArchive(body.archive_id);
    return;
  }
  if (activeArchiveId !== null) await leaveArchiveMode();
  else setArchiveState(null);

  const chunksCfg = document.getElementById('chunks-active-config');
  const retrievalCfg = document.getElementById('retrieval-active-config');
  const setCfg = document.getElementById('retrieval-set-config');
  const genCfg = document.getElementById('generation-active-config');

  // Guarded on a change, not reloaded every tick: this polls every ~2s, and
  // re-rendering would collapse the reader's scroll and drop added questions.
  if (body.lab === 'up' && (body.dataset || '') !== followed.dataset) {
    const previousDataset = followed.dataset;
    followed.dataset = body.dataset || '';
    try {
      await followDataset(followed.dataset);
    } catch (error) {
      followed.dataset = previousDataset;
      throw error;
    }
  }

  if (body.lab === 'down') {
    showLabDown(chunksCfg);
    showLabDown(retrievalCfg);
    showLabDown(setCfg);
    showLabDown(genCfg);
    return;
  }

  if (body.generation) {
    if (body.generation.job_id !== followed.generationJobId) {
      followed.generationJobId = body.generation.job_id;
      const n = (body.generation.rows || []).length;
      // Only an evaluation generates, and the last one may be older than the
      // retrieval tables next door. Say which run this is, and hand the tables
      // over only when they belong to this same run.
      const sameRun = body.retrieval && body.retrieval.job_id === body.generation.job_id;
      const traces = sameRun
        ? new Map(body.retrieval.questions.map(q => [q.question_id, q.trace]))
        : null;
      genCfg.textContent = `${n} question${n === 1 ? '' : 's'} answered by the last `
        + `evaluation — ${formatConfig(body.generation.config)}`
        + (sameRun ? '' : ' (the Retrieval tab is showing a different, newer run)');
      renderGeneration(body.generation, traces);
    }
  } else {
    // An empty view is a place to say what to do next, not to report a state
    // the header already reports.
    genCfg.textContent = 'Run an evaluation from "3 · Generation & scoring" on '
      + 'the lab to see answers here. Retrieve on its own does not write any.';
    document.getElementById('generation-questions').innerHTML = '';
    document.getElementById('generation-ragas').innerHTML = '';
  }

  if (body.retrieval) {
    if (body.retrieval.job_id !== followed.retrievalJobId) {
      followed.retrievalJobId = body.retrieval.job_id;
      const n = body.retrieval.questions.length;
      const source = body.retrieval.kind === 'run'
        ? 'evaluation' : 'retrieval run';
      setCfg.textContent = `${n} selected question${n === 1 ? '' : 's'} from the `
        + `last ${source} — ${formatConfig(body.retrieval.config)}`;
      FOLLOWED_CONFIG = body.retrieval.config || FOLLOWED_CONFIG;
      renderQuestionTables(body.retrieval.questions);
    }
  } else {
    setCfg.textContent = 'Press Retrieve on the lab, or run an evaluation, to '
      + 'get one table per selected question.';
    // Emptied as well, the way the generation view next door is: a lab with no
    // retrieval to show must not leave the tables of whatever was pinned before
    // sitting under a line that says there is nothing to show.
    document.getElementById('retrieval-questions').innerHTML = '';
  }

  if (body.index) {
    if (body.index.job_id !== followed.indexJobId) {
      followed.indexJobId = body.index.job_id;
      const groups = body.index.chunks_by_session || [];
      const total = groups.reduce((n, g) => n + g.chunks.length, 0);
      // Which run built these, because an experiment builds its own index: "the
      // chunks the evaluation used" and "the chunks you last pressed Build on"
      // are different claims and the reader has to be able to tell them apart.
      const source = { run: 'the last evaluation', retrieve: 'the last retrieval run' }[body.index.kind]
        || 'the last index build';
      const summaries = body.index.summaries || [];
      // The summary count belongs in the same sentence as the chunk count: the
      // two together are what say whether a grouping ran, and the config line is
      // the only place on this view that describes the build as a whole.
      chunksCfg.textContent = `${total} chunks in ${groups.length} sessions`
        + (summaries.length ? ` · ${summaries.length} summar`
            + `${summaries.length === 1 ? 'y' : 'ies'}` : '')
        + `, from ${source} — ${formatConfig(body.index.config)}`;
      renderChunkGroups(document.getElementById('chunks-body'), groups);
      setChunkSummaries(summaries);
      // Redraw whichever half is on screen, so a new run does not leave the
      // previous build's summaries showing under this build's config line.
      showChunkMode(document.getElementById('summaries-body').hidden
        ? 'chunks' : 'summaries');
    }
  } else {
    chunksCfg.textContent = 'Build an index on the lab to read its chunks here.';
  }

  if (body.query) {
    if (body.query.job_id !== followed.queryJobId) {
      followed.queryJobId = body.query.job_id;
      retrievalCfg.textContent = formatConfig(body.query.config);
      renderRetrievalRows(body.query.trace.candidates);
      // An answer is written in the corpus's language, so it reads in the
      // corpus's direction. The box used to carry a fixed right-to-left in the
      // markup, which is a claim markup cannot make: it is settled at page load
      // and the corpus is not.
      const answer = document.getElementById('retrieval-answer');
      answer.dir = CORPUS_DIR;
      answer.textContent = body.query.answer || '';
    }
  } else {
    // This used to send the reader to a control on the lab that has not
    // existed for a while, which is worse than saying nothing: they go
    // looking. The view follows a one-off query job, which only the lab's own
    // route still starts — so it says that, and points at the control on this
    // page that traces a question the same way.
    retrievalCfg.textContent = 'No one-off query traced. The lab starts one '
      + 'through POST /api/queries; every question of its last retrieval or '
      + 'evaluation is listed above, and Add a question traces any other one '
      + 'under the same config.';
  }
}

async function pollFollow() {
  try {
    await renderFollow(await (await fetch('/api/follow')).json());
  } catch (error) {
    // A hiccup fetching our own origin — try again next tick rather than
    // treating a transient failure as "the lab is down".
  } finally {
    setTimeout(pollFollow, 2000);
  }
}
pollFollow();

// A deep link wins over live: someone who followed the board's `↗` asked for
// that experiment, not for whatever the lab is doing now. Started here rather
// than earlier so it is set up before the first follow tick can render — the
// loading id is claimed synchronously, which is what keeps the boot fetch of
// the live fixture from landing on top of the record.
const deepLinkedExperiment =
  new URLSearchParams(window.location.search).get('experiment');
if (deepLinkedExperiment) followRecordedExperiment(deepLinkedExperiment);
