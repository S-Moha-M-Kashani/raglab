// The leaderboard surface. Everything it renders comes from
// `GET /api/leaderboard`, which serialises `evaluation.leaderboard` — the same
// module `raglab-leaderboard` prints from. Nothing here re-derives a rank, a
// tie or a winner: a page that recomputed them could name a different winner
// than the command line from the same runs.

const $ = (id) => document.getElementById(id);

// Columns, in the order they are read. `when` is first because it is the row's
// identity: every row already carried `started_at`, and the old board threw it
// away and showed a 22-character run id instead — a date, a time and an index
// fingerprint mashed together, readable as none of the three. The run id is
// still how you open a run; it just is not a column any more.
const COLUMNS = [
  { key: 'rank', label: '#', title: 'rank within this group' },
  { key: 'when', label: 'when', text: true, title: 'when the run started' },
  { key: 'label', label: 'candidate', text: true, title: 'the run label' },
  { key: 'decision', label: 'decision', title: 'unweighted mean of the four judged metrics' },
  { key: 'spread', label: '±', title: 'standard error of the decision score' },
  { key: 'faith', label: 'faith', title: 'faithfulness' },
  { key: 'ansrel', label: 'ans rel', title: 'answer relevancy' },
  { key: 'ctxprec', label: 'ctx prec', title: 'LLM context precision' },
  { key: 'ctxrec', label: 'ctx recall', title: 'context recall' },
  { key: 'chunker', label: 'chunker', text: true },
  { key: 'embedder', label: 'embedder', text: true },
  { key: 'retriever', label: 'retriever', text: true },
  { key: 'reranker', label: 'reranker', text: true },
  { key: 'questions', label: 'questions', title: 'how many questions were scored' },
  { key: 'seconds', label: 'seconds', title: 'wall clock for the run' },
];

const fmt = (value, digits = 3) =>
  value === null || value === undefined ? '—' : Number(value).toFixed(digits);

// `started_at` is already '%Y-%m-%d %H:%M:%S'. Seconds do not help anyone
// comparing runs, so they are dropped rather than reformatted.
const when = (row) => (row.started_at || '').slice(0, 16) || '—';

function cell(row, key, rank) {
  const ragas = row.ragas || {};
  // The knobs live under `config`, not on the row: a run records the whole
  // config it ran, and the leaderboard shows the four that a sweep moves.
  const index = (row.config || {}).index || {};
  const retrieval = (row.config || {}).retrieval || {};
  switch (key) {
    case 'rank': return rank;
    case 'when': return when(row);
    case 'label': return row.label || '';
    case 'decision': return fmt(row.ragas_decision, 4);
    case 'spread': return fmt(row.ragas_decision_stderr, 3);
    case 'faith': return fmt(ragas.faithfulness);
    case 'ansrel': return fmt(ragas.answer_relevancy);
    // The `_with_reference` variant is the one the runs actually record; the
    // bare name would read '—' on every row that has the metric.
    case 'ctxprec': return fmt(ragas.llm_context_precision_with_reference);
    case 'ctxrec': return fmt(ragas.context_recall);
    case 'chunker':
      return (index.chunker || '') + (index.contextual ? '+ctx' : '');
    // The model, not just the kind: two `fastembed` rows can be two entirely
    // different representations, and the row has to say which one it was.
    case 'embedder':
      return (index.embedder || '')
        + (index.embed_model ? '·' + index.embed_model.split('/').pop() : '');
    case 'retriever': return retrieval.retriever || '';
    case 'reranker': return retrieval.reranker || '';
    case 'questions': return row.n_questions ?? 0;
    case 'seconds': return Math.round(row.seconds || 0);
    default: return '';
  }
}

// The verdict in the group's own words. The wording matches the markdown the
// command line prints, because the two are the same claim about the same rows.
const VERDICTS = {
  tie: ['No winner', 'the lead is inside the combined error of the top two '
        + 'rows, so these rows do not separate.'],
  unranked: ['One judged row only', 'there is nothing to compare it against.'],
};

function verdictLine(group) {
  if (group.verdict === 'unknown' && !group.ranked) {
    return ['Not comparable', 'these runs recorded how many questions they '
            + 'scored but not which ones. Equal counts are not a shared '
            + 'sample, so the order below is a listing, not a ranking.'];
  }
  if (group.verdict === 'unknown') {
    return ['No winner', 'these rows carry no measured error, so a lead cannot '
            + 'be told from noise.'];
  }
  return VERDICTS[group.verdict]
    || [`Winner: ${group.verdict}`, 'by more than the combined error of the '
        + 'top two rows.'];
}

function renderGroup(group, index) {
  const [call, why] = verdictLine(group);
  const heading = `${group.dataset} · ${group.sample} · judged by ${group.judge}`;
  const rows = group.rows.map((row, i) => {
    // A numbered row is a rank claim. A row with no decision score, or a group
    // whose sample was never recorded, gets no number rather than a misleading
    // one — the same rule the module applies when it prints.
    const rank = (row.ragas_decision !== null && row.ragas_decision !== undefined
                  && group.ranked) ? String(i + 1) : '—';
    const cells = COLUMNS.map((col, c) => {
      const freeze = c === 0 ? ' freeze-1' : (c === 1 ? ' freeze-2' : '');
      const kind = col.text ? ' text' : '';
      return `<td class="${(kind + freeze).trim()}">${escapeHtml(String(cell(row, col.key, rank)))}</td>`;
    }).join('');
    return `<tr title="${escapeHtml(row.run_id || '')}">${cells}</tr>`;
  }).join('');

  const head = COLUMNS.map((col, c) => {
    const freeze = c === 0 ? ' freeze-1' : (c === 1 ? ' freeze-2' : '');
    const kind = col.text ? ' text' : '';
    const tip = col.title ? ` title="${escapeHtml(col.title)}"` : '';
    return `<th scope="col" class="${(kind + freeze).trim()}"${tip}>`
      + `${escapeHtml(col.label)}</th>`;
  }).join('');

  return `
    <section class="card">
      <div class="card-head"><h2>${escapeHtml(group.dataset)}</h2>
        <span class="section-meta right">${escapeHtml(group.sample)} · judged by ${escapeHtml(group.judge)}</span>
      </div>
      <p class="verdict"><b>${escapeHtml(call)}</b> — ${escapeHtml(why)}</p>
      <div class="table-scroll" tabindex="0" role="region"
           aria-label="${escapeHtml(heading)}">
        <table class="data-table">
          <caption>${escapeHtml(heading)}</caption>
          <thead><tr>${head}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <p class="table-hint">Ordered by <b>decision</b> — the unweighted mean of
        faithfulness, answer relevancy, context precision and context recall.
        Every other column is reported and none of them votes. Click into the
        table and use the arrow keys, PageUp/PageDown, Home and End to read
        across it; hover a row for its run id.</p>
    </section>`;
}

async function loadBoard() {
  const box = $('board');
  box.innerHTML = '<div class="card"><p class="prose">Reading the run '
    + 'records…</p></div>';
  try {
    const res = await fetch('/api/leaderboard');
    if (!res.ok) throw new Error(`the lab answered ${res.status}`);
    const groups = (await res.json()).groups || [];
    if (!groups.length) {
      box.innerHTML = '<div class="card"><div class="card-head">'
        + '<h2>No runs yet</h2></div><p class="prose">Nothing has been '
        + 'evaluated on this machine. Open the <a href="/">Laboratory</a>, '
        + 'pick a dataset and press <b>Run evaluation</b>; the run lands in '
        + '<code>.runs/</code> and appears here.</p></div>';
      return;
    }
    box.innerHTML = groups.map(renderGroup).join('');
  } catch (e) {
    // A failing read says so. The old board caught nothing, so a failed fetch
    // left an empty div and no way to tell "no runs" from "the lab is down".
    box.innerHTML = '<div class="card"><div class="card-head">'
      + '<h2>Could not read the leaderboard</h2></div>'
      + `<p class="prose">${escapeHtml(e.message)}. The lab on :9002 serves `
      + 'this page and the run records behind it, so if it stopped, this is '
      + 'what that looks like.</p></div>';
  }
}

loadBoard();
