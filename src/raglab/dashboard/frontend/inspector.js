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
  return `<button type="button" class="inspector-why" data-why="${escapeHtml(key)}"`
    + ` aria-label="What is ${label}?">!</button>`;
}

// One listener for the page: the marks are re-rendered on every poll tick, and
// a listener per button would leak one per render.
document.addEventListener('click', event => {
  const button = event.target.closest('.inspector-why');
  if (!button) return;
  const open = button.nextElementSibling;
  if (open && open.classList.contains('inspector-why-text')) { open.remove(); return; }
  const note = document.createElement('span');
  note.className = 'inspector-why-text';
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

function renderGroundTruth(body) {
  FOLLOWED_DATASET = body.dataset || '';
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
    const field = (label, text, rtl) => text
      ? `<div class="gt-field"><div class="qh-label">${label}</div>`
        + `<div${rtl ? ' dir="rtl"' : ''}>${escapeHtml(text)}</div></div>` : '';
    const quotes = (q.evidence || []).map(e => e.quote);
    row.innerHTML = `<div class="gt-head"><span class="q-id">${escapeHtml(q.id)}</span> `
      + `<span class="q-tally">${escapeHtml(q.type)} · ${escapeHtml(q.difficulty)}`
      + `${q.answerable ? '' : ' · unanswerable'}</span></div>`
      + `<div class="gt-q" dir="rtl">${escapeHtml(q.question_fa)}</div>`
      + `<div class="gt-en">${escapeHtml(q.question_en || '')}</div>`
      + field('answer', q.answer_fa, true)
      + (quotes.length
         ? `<div class="gt-field"><div class="qh-label">evidence quoted from the diary</div>`
           + quotes.map(quote =>
               `<div dir="rtl" class="gt-quote">${escapeHtml(quote)}</div>`).join('')
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

async function loadGroundTruth(dataset) {
  const requestedDataset = dataset || '';
  // Named on every request, never assumed, so this view can't show one corpus
  // while another view on the page shows a different one.
  const response = await fetch('/api/groundtruth?dataset='
                               + encodeURIComponent(requestedDataset));
  const body = await response.json().catch(() => ({ detail: response.statusText }));
  if (!response.ok) throw new Error(body.detail || response.statusText);
  // A live fetch that began before archive mode must not land afterwards and
  // replace the imported ground truth with a different corpus.
  if (activeArchiveId !== null || archiveLoadingId !== null) return;
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
        + `<div dir="rtl">${escapeHtml(c.text)}</div>`;
      det.appendChild(line);
    });
    container.appendChild(det);
  }
}

// Each card leads with what its text cannot say — group, level, chunk count,
// and session count. A group spanning several sessions carries no session id
// at all, which is why the session count must be stated rather than shown.
function renderSummaries(container, summaries) {
  container.innerHTML = '';
  if (!summaries.length) {
    // "No hierarchy" and "a hierarchy that found nothing" are different facts, and
    // this view must not read as the second when it is the first.
    container.innerHTML = '<p class="empty-note">This index is flat — no grouping was '
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
      + `<div dir="rtl">${escapeHtml(s.text)}</div>`;
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
const chunkView = { summaries: [] };

function showChunkMode(mode) {
  document.getElementById('chunks-body').hidden = mode !== 'chunks';
  document.getElementById('summaries-body').hidden = mode !== 'summaries';
  for (const button of document.querySelectorAll('#chunks-mode button')) {
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  }
  if (mode === 'summaries') {
    renderSummaries(document.getElementById('summaries-body'),
                    chunkView.summaries);
  }
}

for (const button of document.querySelectorAll('#chunks-mode button')) {
  button.addEventListener('click', () => showChunkMode(button.dataset.mode));
}

document.getElementById('build-chunks').addEventListener('click', async () => {
  const status = document.getElementById('chunks-status');
  try {
    status.textContent = 'building…';
    await chosenReady;
    const response = await fetch('/api/chunks',
      { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(CHOSEN) });
    const result = await pollJob(await startedJob(response));
    // Both counts, because "167 chunks" over an index of 174 rows is the exact
    // statement that made seven summaries invisible.
    status.textContent = `${result.total} chunks · `
      + `${result.total_summaries || 0} summaries`;
    document.getElementById('chunks-active-config').textContent = formatConfig(CHOSEN);
    renderChunkGroups(document.getElementById('chunks-body'), result.chunks_by_session);
    chunkView.summaries = result.summaries || [];
    showChunkMode(document.getElementById('summaries-body').hidden
      ? 'chunks' : 'summaries');
  } catch (error) {
    status.textContent = error.message;
  }
});

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
    + `<span class="chunk-preview" dir="rtl" tabindex="0">`
    + `${escapeHtml(preview.slice(0, 60))}${preview.length > 60 ? '…' : ''}</span>`
    + `<div class="chunk-reveal" dir="rtl">${highlighted(text, spans)}`
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

// Where an open reveal goes. It is `position: fixed`, so it must be told; and
// it must be told at the moment it opens, because the row it belongs to may
// have been scrolled anywhere inside its region since the table was built.
// Below the cell when there is room below, above it when there is not, and
// never off either side.
function placeReveal(cell) {
  const reveal = cell && cell.querySelector('.chunk-reveal');
  // `position: static` is the narrow-width variant, which opens in place and
  // wants no insets at all.
  if (!reveal || getComputedStyle(reveal).position === 'static') return;
  const box = cell.getBoundingClientRect();
  const gap = 4;
  const below = box.bottom + gap;
  const height = reveal.offsetHeight;
  reveal.style.top = `${below + height <= window.innerHeight
    ? below : Math.max(gap, box.top - gap - height)}px`;
  reveal.style.left = `${Math.max(gap,
    Math.min(box.left, window.innerWidth - reveal.offsetWidth - gap))}px`;
}

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
document.addEventListener('pointerover', event => placeReveal(revealCell(event.target)));
document.addEventListener('focusin', event => placeReveal(revealCell(event.target)));
document.addEventListener('scroll', () => {
  for (const cell of document.querySelectorAll(
    '.retrieval-row:hover .chunk-cell, .chunk-cell:focus-within')) placeReveal(cell);
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
    + `<div class="qh-fa" dir="rtl">${escapeHtml(q.question_fa || fallbackFa || '')}</div>`
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
      `<div class="gt-quote" dir="rtl">${escapeHtml(ev.quote)}</div>`).join('');
    option.innerHTML =
      `<span class="q-option-id">${escapeHtml(q.id)}</span>`
      + `<span class="q-chip q-chip--${escapeHtml(q.difficulty || 'easy')}">`
      + `${escapeHtml(q.difficulty || '')}</span>`
      + `<span class="q-option-en">${escapeHtml(q.question_en || q.question_fa)}</span>`
      + (ADDED.has(q.id) ? '<span class="q-option-added">added</span>' : '')
      // The detail is in the DOM from the start rather than built on hover, so
      // it opens with no delay and reads the same to a screen reader.
      + `<div class="q-option-detail">`
      + `<div dir="rtl" class="q-option-fa">${escapeHtml(q.question_fa)}</div>`
      + `<div class="qh-label">expected answer</div>`
      + `<div dir="rtl">${escapeHtml(q.answer_fa || '—')}</div>`
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
    + `<div dir="rtl">${escapeHtml(gt.answer_fa || '—')}</div></div>`
    + '<div class="gen-answer gen-answer--actual">'
    + `<h4>what this run wrote${row.abstained ? ' — it refused' : ''}</h4>`
    + `<div dir="rtl">${escapeHtml(row.answer || '—')}</div></div>`
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

function setArchiveState(runId) {
  const state = document.getElementById('archive-state');
  const button = document.getElementById('archive-return-live');
  const readOnly = runId !== null;
  state.hidden = !readOnly;
  button.hidden = !readOnly;
  document.getElementById('build-chunks').disabled = readOnly;
  document.getElementById('add-question').disabled = readOnly;
  if (readOnly) state.textContent = `Imported archive · read-only · ${runId}`;
  else state.textContent = '';
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
                      datasets: [] });
  renderChunkGroups(document.getElementById('chunks-body'),
                    evidence.chunks_by_session);
  chunkView.summaries = evidence.summaries;
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

async function leaveArchiveMode() {
  const dataset = liveDatasetBeforeArchive;
  activeArchiveId = null;
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

document.getElementById('archive-return-live').addEventListener('click', async () => {
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

async function renderFollow(body) {
  // Before the early-returning archive branch: a page opened while an archive
  // is already active must still say whether the lab is reachable.
  setFollowState(body);
  if (body.archive_id) {
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
      chunkView.summaries = summaries;
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
      document.getElementById('retrieval-answer').textContent = body.query.answer || '';
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
