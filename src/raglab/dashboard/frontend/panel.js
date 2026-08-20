const $ = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (typeof v === 'number' ? v.toFixed(d) : String(v));
const pct = (v) => v === null || v === undefined ? '—' : Math.round(v * 100) + '%';
let OPTIONS = null;

const CHUNKER_HELP = {
  'fixed': "the brain's current 500-char packing — baseline",
  'fixed-overlap': 'sliding window, so a split thought survives whole in one window',
  'message': 'one chunk per turn — precise, but short turns lose context',
  'turn-pair': 'user turn + coach reply together',
  'session': 'whole session — max fidelity, low precision',
  'semantic-drift': 'topic segmentation: cut where the subject changes',
};

function fill(select, values, help) {
  select.innerHTML = values.map((v) =>
    `<option value="${v}">${v}${help && help[v] ? ' — ' + help[v] : ''}</option>`).join('');
}

// One line per grouping, naming which family it belongs to. Kept beside the
// dropdown, like CHUNKER_HELP, since config.HELP explains the knob, not its options.
const HIERARCHY_HELP = {
  '': 'flat, no summaries (default)',
  'louvain': 'graph · modularity communities',
  'leiden': 'graph · Louvain, refined for connectedness',
  'label-prop': 'graph · label propagation, no granularity knob',
  'raptor': 'vectors · GMM, soft, recursive',
  'agglomerative': 'vectors · Ward, a real dendrogram',
  'kmeans': 'vectors · the naive control',
  'metadata': "the corpus's own storylines — the 2026-07-31 control",
};
const GRAPH_SOURCE_HELP = {
  'hybrid': 'cosine neighbours + shared rare words',
  'knn': 'cosine neighbours only',
  'lexical': 'shared rare words only',
  'bipartite-terms': 'terms are nodes too — the offline analogue of an entity graph',
};
const SUMMARIZER_HELP = {
  'centroid': 'members nearest the group centre',
  'lead-idf': 'sentences covering the most rare words',
  'mmr': 'members picked for coverage, not repetition',
  'card': 'no prose — terms, span, count, sessions',
};
const SUMMARY_SCOPE_HELP = {
  'mixed': 'summaries and leaves in one pool (default)',
  'leaves': 'ignore the summaries — the control',
  'summaries': 'summaries only',
  'drill-down': 'retrieve summaries, then expand to their members',
};

// A grouping whose library is missing is offered as NA and refused by the
// service if picked — never quietly served by a different partition.
// One line per scope, naming the stage it owns; the four read as a 2x2.
const SCOPE_HELP = {
  '': 'the fixed pipeline (default)',
  'retrieve': 'retrieval only · plan, retrieve, judge the evidence, rewrite, again',
  'generate': 'generation only · draft, critique it, revise — retrieval held fixed',
  'full': 'both · and the one edge neither has: a bad critique can re-retrieve',
};

const CRITIC_HELP = {
  'grounded': 'is every claim supported by the retrieved text?',
  'both': 'that, plus: does the draft answer the question?',
  'none': 'ship the first draft — the control',
};

// An agent scope this installation cannot run is offered as NA and refused by
// the service if picked, never quietly served by the fixed pipeline.
function fillScopes() {
  const support = OPTIONS.agent_support || {};
  $('scope').innerHTML = (OPTIONS.scopes || []).map((v) => {
    const state = support[v];
    const na = (state && state.available === false)
      ? ` — NA, needs \`${state.install}\`` : '';
    const help = SCOPE_HELP[v] ? ' — ' + SCOPE_HELP[v] : '';
    return `<option value="${escapeHtml(v)}"${na ? ' disabled' : ''}>`
      + `${escapeHtml(v || 'off')}${escapeHtml(help)}${escapeHtml(na)}</option>`;
  }).join('');
}

function fillHierarchies() {
  const support = OPTIONS.hierarchy_support || {};
  $('hierarchy').innerHTML = (OPTIONS.hierarchies || []).map((v) => {
    const state = support[v];
    const na = (state && state.available === false)
      ? ` — NA, needs \`${state.install}\`` : '';
    const help = HIERARCHY_HELP[v] ? ' — ' + HIERARCHY_HELP[v] : '';
    const label = v || '(none)';
    return `<option value="${escapeHtml(v)}"${na ? ' disabled' : ''}>`
      + `${escapeHtml(label)}${escapeHtml(help)}${escapeHtml(na)}</option>`;
  }).join('');
}

// A model's label always carries where its weights stand, and a model this lab
// has not run is offered as NA rather than dropped from the list.
function modelLabel(m) {
  const licence = m.source === 'default' ? '' : m.source === 'open' ? ' (open source)'
    : m.source === 'closed' ? ' (closed source)' : ' (licence not recorded)';
  const na = m.available ? '' : ' — NA, not available here but worth checking';
  return `${m.label}${licence}${na}`;
}

// Language coverage travels in the option text: on a Farsi corpus an
// English-only embedder returns a full set of confident numbers that measure
// nothing, and that has to be visible while picking, not after the run.
function fillEmbedders() {
  const hints = OPTIONS.embedder_hints || [];
  $('embedder').innerHTML = hints.map((h) =>
    `<option value="${escapeHtml(h.kind)}">${escapeHtml(h.kind)} — `
    + `${escapeHtml(h.languages)}`
    + `${h.available === false ? ' — NA, not installed or no key here' : ''}`
    + `</option>`).join('');
  $('embed_model').innerHTML = (OPTIONS.embed_models || []).map((m) => {
    const licence = m.source === 'default' ? '' : m.source === 'open' ? ' (open source)'
      : m.source === 'closed' ? ' (closed source)' : ' (licence not recorded)';
    const na = m.available ? '' : ' — NA, not served here but worth checking';
    const dim = m.dim ? ` · ${m.dim}d` : '';
    // The backend is what picking it costs: an ONNX download, a multi-gigabyte
    // checkpoint, or an API bill.
    const via = m.backend ? ` · via ${escapeHtml(m.backend)}` : '';
    const tag = m.tag ? ` (${escapeHtml(m.tag)})` : '';
    return `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}${tag} — `
      + `${escapeHtml(m.languages)}${licence}${dim}${via}${na}</option>`;
  }).join('');
}

// Grouped by the step each model serves, into the host carrying that step's
// ink; a role with no matching group still renders, in the catch-all under them.
// The chat-model list for the backend that will actually run: the picked mode's
// own catalogue, or the boot catalogue when no mode is picked — a slug only
// means something to the backend that serves it.
function modelList() {
  const mode = (OPTIONS.modes || []).find((m) => m.key === $('mode').value);
  return (mode && mode.models) || OPTIONS.models || [];
}

function fillModels() {
  const options = modelList().map((m) =>
    `<option value="${escapeHtml(m.id)}">${escapeHtml(modelLabel(m))}</option>`).join('');
  // Each role in its own .field wrapper: `applyDependencies` dims the control's
  // nearest div, and with the whole group in one box a single inactive model
  // greyed out every other model of that step beside it.
  const row = (role) =>
    `<div class="field"><label>${escapeHtml(role.label)} `
    + `<span class="muted">· ${escapeHtml(role.only_when)}</span>`
    + `<button type="button" class="why" data-topic="model.${role.key}" `
    + `aria-label="What is ${escapeHtml(role.label)}?">!</button></label>`
    + `<select class="rag-model" data-role="${role.key}" data-field="${role.field}">`
    + `${options}</select></div>`;
  const groups = {};
  for (const role of OPTIONS.model_roles || []) {
    const step = role.step || role.field.split('.')[0];
    (groups[step] = groups[step] || []).push(role);
  }
  const spare = [];
  for (const host of document.querySelectorAll('[id^="modelRoles-"]')) {
    const step = host.id.slice('modelRoles-'.length);
    host.innerHTML = (groups[step] || []).map(row).join('');
    delete groups[step];
  }
  for (const roles of Object.values(groups)) spare.push(...roles);
  $('modelRoles').innerHTML = spare.map(row).join('');
}

// The corpora this lab can be pointed at. The built-in one is offered as '' —
// the value every run already recorded carries, and the one a fingerprint is
// computed without, so choosing it changes nothing about an index that exists.
function fillDatasets() {
  const found = OPTIONS.datasets || [];
  $('dataset').innerHTML = found.map((d) =>
    `<option value="${escapeHtml(d.source === 'builtin' ? '' : d.id)}">`
    + `${escapeHtml(d.name)} — ${escapeHtml(d.language || '?')} · ${d.sessions} `
    + `sessions · ${d.questions} questions`
    + `${d.source === 'imported' ? ' · imported' : ''}</option>`).join('');
  $('dataset').onchange = () => {
    describeDataset();
    syncArchiveViewOnlyFromDataset();
  };
}

const datasetOf = (id) => (OPTIONS.datasets || []).find(
  (d) => (d.source === 'builtin' ? '' : d.id) === id);

// One line about the corpus in force, in the header where the corpus has always
// been described — switching datasets has to move that line, or the page says
// 167 sessions while measuring twenty.
function describeDataset() {
  const found = datasetOf($('dataset').value);
  if (!found) return;
  const period = found.period && found.period.from
    ? `${found.period.from} → ${found.period.to} · ` : '';
  $('corpus').textContent =
    `${found.sessions} sessions · ${found.messages} messages · ${period}`
    + `${found.questions} questions`
    + `${found.query_date ? ' · asked as of ' + found.query_date : ''}`;
  $('datasetInfo').textContent = found.description;
}

// Import: the file is read here and posted as JSON, so the service checks it
// against the contract rather than the browser guessing. Every problem comes
// back at once — fixing a corpus is a slow loop if each attempt reports one
// broken quote out of nine.
$('dataset-file').onchange = async () => {
  const file = $('dataset-file').files[0];
  if (!file) return;
  $('importInfo').textContent = `reading ${file.name}…`;
  try {
    const payload = JSON.parse(await file.text());
    const added = await api('/api/datasets', payload);
    await refreshOptions();
    $('dataset').value = added.id;
    describeDataset();
    $('importInfo').innerHTML =
      `<b>${escapeHtml(added.name)}</b> imported · ${added.sessions} sessions, `
      + `${added.questions} questions. Build the index to measure against it.`;
  } catch (e) {
    $('importInfo').innerHTML =
      `<div class="note">${escapeHtml(e.message)}</div>`;
  } finally {
    $('dataset-file').value = '';
  }
};

// The mode dropdown: '' follows whatever backend the lab booted with; a mode
// applies its served preset onto the controls (the choices stay editable) and
// its provider rides along on every run.
function fillModes() {
  const caps = OPTIONS.capabilities || {};
  $('mode').innerHTML =
    `<option value="">lab boot (${escapeHtml(caps.llm_provider || 'unknown')})</option>`
    + (OPTIONS.modes || []).map((m) =>
      `<option value="${escapeHtml(m.key)}">${escapeHtml(m.label)}</option>`).join('');
  $('mode').onchange = () => {
    const mode = (OPTIONS.modes || []).find((m) => m.key === $('mode').value);
    // Read the controls before the model dropdowns are rebuilt for the new
    // backend's catalogue, then write the merged config back over them.
    const cfg = readConfig();
    if (mode && mode.config) {
      for (const group of Object.keys(mode.config)) {
        Object.assign(cfg[group], mode.config[group]);
      }
    }
    fillModels();
    applyDefaults(cfg);
    applyDependencies();
  };
}

// The backend the picked mode runs on, spread into a run's payload; {} when no
// mode is picked and the run should follow the lab's boot provider.
function pickedProvider() {
  const mode = (OPTIONS.modes || []).find((m) => m.key === $('mode').value);
  return mode ? { provider: mode.provider } : {};
}

// The three cards title themselves from the served steps (config.STEPS), so what
// a stage is called and what it costs is a fact about the pipeline rather than a
// sentence typed into this file.
function titleSteps() {
  for (const step of OPTIONS.steps || []) {
    const head = $('head-' + step.key), note = $('note-' + step.key);
    if (head) head.textContent = step.label;
    if (note) note.textContent = step.note;
    const seg = document.querySelector(`.spine-seg[data-step="${step.key}"] b`);
    if (seg) seg.textContent = step.short;
  }
}

// Every knob explains itself. The text comes from the service (config.HELP and
// models.ROLES), so a knob added there is explained here without editing this
// file — and the id of each control is the field it sets.
function decorateExplainers() {
  const byField = {};
  for (const [topic, text] of Object.entries(OPTIONS.help || {})) {
    byField[topic.split('.').pop()] = { topic, text };
  }
  for (const control of document.querySelectorAll('main [id]')) {
    const found = byField[control.id];
    // A checkbox lives *inside* its own label, so it explains itself there; every
    // other control is preceded by one. Walking up blindly would hang a
    // checkbox's explainer on the previous checkbox's label.
    const label = control.type === 'checkbox' ? control.closest('label')
      : control.previousElementSibling;
    if (!found || !label || label.tagName !== 'LABEL' || label.querySelector('.why')) continue;
    label.insertAdjacentHTML('beforeend',
      ` <button type="button" class="why" data-topic="${found.topic}" `
      + `aria-label="What is this?">!</button>`);
  }
  document.addEventListener('click', (event) => {
    const btn = event.target.closest('.why');
    if (!btn) return;
    const open = btn.parentElement.nextElementSibling;
    if (open && open.classList.contains('explain')) { open.remove(); return; }
    const text = btn.dataset.help || (OPTIONS.help || {})[btn.dataset.topic] || '';
    btn.parentElement.insertAdjacentHTML('afterend',
      `<p class="explain">${escapeHtml(text)}</p>`);
  });
}

function checkboxes(host, values, checked) {
  host.innerHTML = values.map((v) =>
    `<label><input type="checkbox" value="${v}"${checked.includes(v) ? ' checked' : ''}> ${v}</label>`
  ).join('');
}

function selected(host) {
  return [...host.querySelectorAll('input:checked')].map((el) => el.value);
}

async function api(path, body, method = body ? 'POST' : 'GET') {
  const options = method === 'GET' ? undefined : {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  };
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({ detail: res.statusText }));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

// Config fields this panel has no control for (`rrf_k`, `agentic_weights`,
// `max_context_chars`) — carried through so applying a preset that sets them
// can't silently drop them to the lab's own defaults. A real control always wins.
let UNSHOWN = { index: {}, retrieval: {}, generation: {}, agent: {} };
let CURRENT_ARCHIVE = null;
let ARCHIVE_VIEW_ONLY = false;
let ARCHIVE_EVENTS_BOUND = false;

// Remember the parts of a config the controls cannot express, having applied it.
function keepUnshown(applied) {
  const shown = readShownConfig();
  UNSHOWN = { index: {}, retrieval: {}, generation: {}, agent: {} };
  for (const group of ['index', 'retrieval', 'generation', 'agent']) {
    for (const [key, value] of Object.entries(applied[group] || {})) {
      if (!(key in shown[group])) UNSHOWN[group][key] = value;
    }
  }
}

function readShownConfig() {
  const cfg = {
    label: $('label').value,
    index: {
      dataset: $('dataset').value,
      chunker: $('chunker').value, chunk_chars: +$('chunk_chars').value,
      overlap: +$('overlap').value, contextual: $('contextual').checked,
      embedder: $('embedder').value, embed_model: $('embed_model').value,
      hierarchy: $('hierarchy').value, graph_source: $('graph_source').value,
      graph_knn: +$('graph_knn').value, granularity: +$('granularity').value,
      hierarchy_levels: +$('hierarchy_levels').value,
      min_group: +$('min_group').value, summarizer: $('summarizer').value,
    },
    retrieval: {
      retriever: $('retriever').value, k: +$('k').value,
      candidates: +$('candidates').value, time_filter: $('time_filter').checked,
      multi_query: $('multi_query').checked, hyde: $('hyde').checked,
      mmr_lambda: +$('mmr_lambda').value, reranker: $('reranker').value,
      rerank_depth: +$('rerank_depth').value,
      recency_half_life_days: +$('recency_half_life_days').value,
      grader: $('grader').value, grade_threshold: +$('grade_threshold').value,
      summary_scope: $('summary_scope').value,
      summary_boost: +$('summary_boost').value,
      summary_levels: $('summary_levels').value,
    },
    generation: {
      answerer: $('answerer').value, key_facts_judge: $('key_facts_judge').checked,
    },
    agent: {
      scope: $('scope').value, max_hops: +$('max_hops').value,
      rewrite: $('rewrite').checked,
      evidence_threshold: +$('evidence_threshold').value,
      max_revisions: +$('max_revisions').value, critic: $('critic').value,
      max_llm_calls: +$('max_llm_calls').value,
    },
  };
  // Each role writes into the config group whose stage uses it, so the index
  // fingerprint keeps describing exactly what was stored.
  for (const select of document.querySelectorAll('.rag-model')) {
    const [group, field] = select.dataset.field.split('.');
    cfg[group][field] = select.value;
  }
  return cfg;
}

function readConfig() {
  const cfg = readShownConfig();
  // Under the controls, never over them: a value you can see and change is
  // always the one that runs.
  for (const group of ['index', 'retrieval', 'generation', 'agent']) {
    cfg[group] = Object.assign({}, UNSHOWN[group], cfg[group]);
  }
  return cfg;
}

// Rules are served (/api/options -> dependencies) rather than written here, so
// the panel and the Inspector grey out the same knobs for the same reason.
// Mirrors `config.dependency_state`, resolved here too since disabling has to
// happen per keystroke without a round trip; the two copies must agree — pinned
// by `test_the_panel_resolves_a_dependency_chain_the_way_the_service_does` —
// including transitively, so a control whose owner is itself dead is dead.
function dependencyState(rules, cfg) {
  const state = {};
  const resolve = (path, seen) => {
    if (state[path]) return state[path];
    const rule = rules[path];
    const [group, name] = rule.field.split('.');
    const current = (cfg[group] || {})[name];
    let enabled = rule.on_true ? Boolean(current)
      : (rule.on || []).indexOf(current) !== -1;
    let reason = enabled ? '' : rule.reason;
    if (enabled && rules[rule.field] && seen.indexOf(path) === -1) {
      const above = resolve(rule.field, seen.concat([path]));
      if (!above.enabled) { enabled = false; reason = above.reason; }
    }
    state[path] = { enabled, reason };
    return state[path];
  };
  for (const path of Object.keys(rules)) resolve(path, []);
  return state;
}

function applyDependencies() {
  const rules = OPTIONS.dependencies || {};
  const cfg = readConfig();
  const state = dependencyState(rules, cfg);
  for (const path of Object.keys(rules)) {
    const el = controlFor(path);
    if (!el) continue;
    const { enabled, reason } = state[path];
    el.disabled = !enabled;
    const holder = el.closest('div') || el.parentElement;
    if (!holder) continue;
    holder.classList.toggle('rag-field-off', !enabled);
    holder.title = enabled ? '' : 'Disabled because ' + reason;
    let note = holder.querySelector('.rag-when-dep');
    if (!enabled) {
      if (!note) {
        note = document.createElement('span');
        note.className = 'rag-when rag-when-dep';
        holder.appendChild(note);
      }
      note.textContent = reason;
    } else if (note) {
      note.remove();
    }
  }
}

// A dependent field is either a plain control whose id is its field name, or one
// of the model dropdowns, which carry the full dotted path in data-field.
function controlFor(path) {
  const name = path.split('.')[1];
  return document.getElementById(name)
      || document.querySelector('.rag-model[data-field="' + path + '"]');
}

function applyDefaults(d) {
  $('label').value = d.label || '';
  $('dataset').value = d.index.dataset || '';
  $('chunker').value = d.index.chunker; $('chunk_chars').value = d.index.chunk_chars;
  $('overlap').value = d.index.overlap; $('contextual').checked = d.index.contextual;
  $('embedder').value = d.index.embedder;
  $('embed_model').value = d.index.embed_model || '';
  $('hierarchy').value = d.index.hierarchy || '';
  $('graph_source').value = d.index.graph_source;
  $('graph_knn').value = d.index.graph_knn;
  $('granularity').value = d.index.granularity;
  $('hierarchy_levels').value = d.index.hierarchy_levels;
  $('min_group').value = d.index.min_group;
  $('summarizer').value = d.index.summarizer;
  $('summary_scope').value = d.retrieval.summary_scope;
  $('summary_boost').value = d.retrieval.summary_boost;
  $('summary_levels').value = d.retrieval.summary_levels || '';
  $('retriever').value = d.retrieval.retriever; $('k').value = d.retrieval.k;
  $('candidates').value = d.retrieval.candidates;
  $('time_filter').checked = d.retrieval.time_filter;
  $('multi_query').checked = d.retrieval.multi_query;
  $('hyde').checked = d.retrieval.hyde;
  $('mmr_lambda').value = d.retrieval.mmr_lambda;
  $('reranker').value = d.retrieval.reranker;
  $('rerank_depth').value = d.retrieval.rerank_depth;
  $('recency_half_life_days').value = d.retrieval.recency_half_life_days;
  $('grader').value = d.retrieval.grader;
  $('grade_threshold').value = d.retrieval.grade_threshold;
  $('answerer').value = d.generation.answerer;
  $('key_facts_judge').checked = d.generation.key_facts_judge;
  const a = d.agent || {};
  $('scope').value = a.scope || '';
  $('max_hops').value = a.max_hops;
  $('rewrite').checked = !!a.rewrite;
  $('evidence_threshold').value = a.evidence_threshold;
  $('max_revisions').value = a.max_revisions;
  $('critic').value = a.critic;
  $('max_llm_calls').value = a.max_llm_calls;
  for (const select of document.querySelectorAll('.rag-model')) {
    const [group, field] = select.dataset.field.split('.');
    select.value = (d[group] || {})[field] || '';
  }
}

// --- what survives a reload -------------------------------------------------
// The readings card is only ever filled by a finishing job or a leaderboard
// click, so without this a reload leaves it blank and every control back at
// its served default. Remembered under the board's own `lodestar:` prefix.
const SAVED_CONFIG = 'lodestar:raglab-config';
const SAVED_RUN = 'lodestar:raglab-last-run';

function saved(key) {
  try {
    return JSON.parse(localStorage.getItem(key) || 'null');
  } catch (e) {
    // A key someone hand-edited, or written by an older shape of this page.
    // Forgetting it is right; refusing to boot over it is not.
    return null;
  }
}

function remember(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch (e) { /* private browsing, a full quota: not worth a broken panel */ }
}

// Merged over the served defaults per group, so a config saved before a knob
// existed still boots with that knob at its default rather than undefined.
// Groups come from the *served* defaults, never a list written here — a saved
// config from before a whole group existed would otherwise merge to a group
// missing entirely, and every blank control in it would read as 0, which
// validation then refuses. Same class of bug as `UNSHOWN`, same fix: the page
// must not hold its own idea of what a config contains.
function startingConfig(defaults) {
  const kept = saved(SAVED_CONFIG);
  if (!kept) return defaults;
  const merged = { label: kept.label || defaults.label || '' };
  for (const group of Object.keys(defaults)) {
    if (group === 'label') continue;
    merged[group] = Object.assign({}, defaults[group], kept[group] || {});
  }
  return merged;
}

// Re-read by id rather than stored whole: a run file can be deleted between two
// visits, and a page rendering a copy of something that is gone is worse than one
// that has forgotten it. A 404 clears the memory instead of showing an error —
// the run is missing, which is not a fault of this page.
async function restoreLastRun() {
  const runId = saved(SAVED_RUN);
  if (!runId) return;
  try {
    renderResult(await api('/api/evaluations/' + encodeURIComponent(runId)));
  } catch (e) {
    remember(SAVED_RUN, null);
  }
}

async function boot() {
  OPTIONS = await api('/api/options');
  const o = OPTIONS;
  titleSteps();
  fillDatasets();
  fill($('chunker'), o.chunkers, CHUNKER_HELP);
  fillEmbedders();
  fillHierarchies();
  fill($('graph_source'), o.graph_sources, GRAPH_SOURCE_HELP);
  fill($('summarizer'), o.summarizers, SUMMARIZER_HELP);
  fill($('summary_scope'), o.summary_scopes, SUMMARY_SCOPE_HELP);
  fill($('retriever'), o.retrievers);
  fill($('reranker'), o.rerankers);
  fill($('grader'), o.graders);
  fill($('answerer'), o.answerers);
  fillScopes();
  fill($('critic'), o.critics, CRITIC_HELP);
  checkboxes($('types'), o.question_types, []);
  document.addEventListener('change', applyDependencies);
  fillModels();
  fillModes();
  applyDefaults(startingConfig(o.defaults));
  keepUnshown(startingConfig(o.defaults));
  applyDependencies();
  describeDataset();
  decorateExplainers();
  // One listener for the whole panel: a knob added to the markup is remembered
  // by being a control, not by being registered here. Both events, because
  // `change` alone fires only on blur — typing k=11 and running without
  // clicking away would remember the stale k=8.
  for (const event of ['change', 'input']) {
    document.addEventListener(event, () => remember(SAVED_CONFIG, readConfig()));
  }
  if (!ARCHIVE_EVENTS_BOUND) {
    const experimentControls = document.querySelector('main');
    for (const event of ['change', 'input']) {
      experimentControls.addEventListener(event, (change) => {
        if (!CURRENT_ARCHIVE || !change.target.matches('input, select, textarea')
            || change.target.id === 'dataset-file') return;
        CURRENT_ARCHIVE = null;
        setArchiveStatus(
          'Readings belong to the previous settings; export will contain settings only.',
          'warning');
      });
    }
    ARCHIVE_EVENTS_BOUND = true;
  }

  renderCapabilities();

  renderIndexes(o.indexes);
  loadBoard();
  loadExperiments();
  restoreLastRun();
}

// Re-read the options and re-render everything they decide, keeping the config
// on screen — entering a key changes which models are offerable, and a panel
// that only learns that on reload is a panel you reload.
async function refreshOptions() {
  const cfg = readConfig();
  const mode = $('mode').value;
  const archivedOption = $('dataset').querySelector('option[data-archive-view-only]');
  const archivedText = archivedOption && archivedOption.textContent;
  const wasViewOnly = ARCHIVE_VIEW_ONLY;
  OPTIONS = await api('/api/options');
  fillEmbedders();
  fillDatasets();
  if (wasViewOnly) addArchivedDatasetOption(cfg.index.dataset, archivedText);
  fillModels();
  fillModes();
  $('mode').value = mode;
  applyDefaults(cfg);
  applyDependencies();
  describeDataset();
  setArchiveViewOnly(wasViewOnly);
  renderCapabilities();
}

function renderCapabilities() {
  const caps = OPTIONS.capabilities;
  const chip = (on, text) => `<span class="chip ${on ? 'on' : 'off'}">${escapeHtml(text)}</span>`;
  renderKeyState(caps.openrouter_key || { set: false, source: '', hint: '' });
  $('caps').innerHTML = [
    chip(caps.fastembed, `fastembed ${caps.fastembed ? 'ready' : 'missing'}`),
    chip(caps.cross_encoder, `cross-encoder ${caps.cross_encoder ? 'ready' : 'missing'}`),
    // The backend is on the chip, not just the model: the same slug costs
    // money on openrouter, nothing on ollama, and measures nothing on fake.
    chip(caps.llm, caps.llm ? `LLM ${caps.llm_provider} · ${caps.llm_model}`
                            : 'no LLM backend — pick one under Models · Backend, or set RAGLAB_LLM to ollama, claude or codex (the two CLIs need no key), or an OPENROUTER_API_KEY for openrouter'),
    chip(caps.ragas.installed, `ragas ${caps.ragas.installed ? caps.ragas.version : 'missing'}`),
    chip(caps.ragas.llm_ready, `ragas LLM metrics ${caps.ragas.llm_ready ? 'ready' : 'off'}`),
    // Not a service chip: the index is process memory and dies with the lab,
    // so what matters is where the run JSON and experiment record land instead.
    `<span class="chip">index in memory · runs → ${escapeHtml(caps.storage.runs || '')}`
      + ` · experiments → ${escapeHtml(caps.storage.experiments || '')}</span>`,
  ].join('');
  if (caps.ragas.notes.length) {
    $('notes').innerHTML = caps.ragas.notes.map((n) => `<div class="note">${escapeHtml(n)}</div>`).join('');
  }
}

// Three states, and they differ in what you can do about them. A key from the
// shell is not one this panel put there, so Clear cannot take it away and says
// so rather than failing quietly.
function renderKeyState(state) {
  const box = $('keyState');
  $('clear-key').disabled = !(state.set && state.source === 'panel');
  if (!state.set) {
    box.textContent = 'No key. OpenRouter models and the judged metrics need '
      + 'one; a model on this machine (Ollama) needs none.';
    return;
  }
  box.textContent = state.source === 'environment'
    ? `Key from the environment · ${state.hint} — set in your shell or .env, `
      + 'so Clear will not remove it.'
    : `Key set · ${state.hint} — held in this lab process and forgotten when it `
      + 'stops. Nothing writes it down.';
}

$('save-key').onclick = async () => {
  const box = $('keyField');
  try {
    // The value leaves the field the moment the service has it: a password box
    // holding a credential for the rest of the afternoon is a credential on
    // screen for the rest of the afternoon.
    renderKeyState(await api('/api/credentials', { api_key: $('openrouter_key').value }));
    $('openrouter_key').value = '';
    await refreshOptions();
  } catch (e) {
    box.insertAdjacentHTML('beforeend', `<div class="note">${escapeHtml(e.message)}</div>`);
    setTimeout(() => box.querySelector('.note') && box.querySelector('.note').remove(), 6000);
  }
};

$('clear-key').onclick = async () => {
  const res = await fetch('/api/credentials', { method: 'DELETE' });
  renderKeyState(await res.json());
  $('openrouter_key').value = '';
  await refreshOptions();
};

// Each spine segment scrolls to the controls it names.
for (const seg of document.querySelectorAll('.spine-seg')) {
  seg.onclick = () => $(seg.dataset.target)
    .scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Which step a stage belongs to, read off the stage the service reports rather
// than guessed from the job kind. An unknown stage lights nothing, since a
// wrong segment is worse than none.
const STAGE_STEP = {
  chunking: 'index', embedding: 'index',
  retrieving: 'retrieval',
  answering: 'generation', scoring: 'generation', ragas: 'generation',
};

function renderSpine(job) {
  const step = job ? STAGE_STEP[job.stage] || '' : '';
  const track = $('spineTrack');
  const caption = $('spineCaption');
  track.dataset.step = step;
  caption.dataset.step = step;
  track.firstElementChild.style.width =
    ((job ? job.progress : 0) * 100).toFixed(1) + '%';
  for (const seg of document.querySelectorAll('.spine-seg')) {
    if (seg.dataset.step === step) seg.setAttribute('aria-current', 'step');
    else seg.removeAttribute('aria-current');
  }
  if (!job) { caption.innerHTML = '<span>nothing running</span>'; return; }
  // The detail ("question 16/30 · hard", "judge call 137 of ~420") is the part
  // that distinguishes a slow stage from a stuck one, and on a local judge one
  // stage is hours.
  caption.innerHTML = `<b>${escapeHtml(job.kind)} · ${escapeHtml(job.stage)}</b>`
    + `<span>${Math.round(job.progress * 100)}%</span>`
    + (job.detail ? `<span>${escapeHtml(job.detail)}</span>` : '')
    + `<span class="right">${escapeHtml(job.id || '')}</span>`;
}

function renderIndexes(indexes) {
  if (!indexes.length) { $('indexInfo').textContent = 'no index built in this session yet'; return; }
  $('indexInfo').innerHTML = indexes.map((i) =>
    `${escapeHtml(i.collection)}: ${i.chunks} chunks`
  ).join('<br>');
}

// What the grouping did — "the index built" is not a result. The quality
// number leads, since a low-modularity partition makes every score under it
// uninformative, and that's worth knowing before reading them.
function hierarchyReport(h) {
  if (!h) return '';
  const rows = [];
  const quality = h.modularity !== null && h.modularity !== undefined
    ? `modularity ${h.modularity}`
    : (h.silhouette !== null && h.silhouette !== undefined
        ? `silhouette ${h.silhouette}` : 'no partition quality measured');
  rows.push(`<b>${escapeHtml(h.hierarchy)}</b>`
    + (h.graph_source ? ` · ${escapeHtml(h.graph_source)} edges` : '')
    + ` · ${escapeHtml(h.summarizer)} — <b>${quality}</b>`);
  rows.push(`${h.summaries} summaries over ${h.groups} groups, `
    + `${h.levels} level${h.levels === 1 ? '' : 's'}, `
    + `coverage ${Math.round((h.coverage || 0) * 100)}% of leaves, `
    + `avg ${h.avg_summary_chars} chars, ${h.seconds}s`);
  if (h.nodes) {
    rows.push(`graph: ${h.nodes} nodes, ${h.edges} edges, `
      + `density ${h.density}, ${h.components} component`
      + `${h.components === 1 ? '' : 's'}`);
  }
  for (const level of h.per_level || []) {
    rows.push(`level ${level.level}: ${level.groups} groups · `
      + `size ${level.min}/${level.median}/${level.max} (min/median/max)`
      + (level.singletons ? ` · ${level.singletons} singleton` : ''));
  }
  return `<div class="note">${rows.join('<br>')}</div>`;
}

async function poll(jobId, onDone) {
  const box = $('jobBox');
  $('cancel').dataset.jobId = jobId;
  $('cancel').disabled = false;
  const tick = async () => {
    const job = await api('/api/jobs/' + jobId);
    renderSpine(job);
    box.innerHTML = '';
    if (job.state === 'running' || job.state === 'cancelling') { setTimeout(tick, 700); return; }
    $('cancel').disabled = true;
    delete $('cancel').dataset.jobId;
    // Every job that stops running is a row in the ledger — including the ones
    // that failed or were cancelled, which are the rows you most want to find
    // later. Refreshed here, once, rather than per success path.
    loadExperiments();
    if (job.ledger_error) {
      // The experiment survived; only its record failed. Said out loud, because
      // a table quietly missing the run you just watched is worse than an error.
      box.insertAdjacentHTML('beforeend',
        `<div class="note">the experiment ran, but raglab.db could not be `
        + `written: ${escapeHtml(job.ledger_error)}</div>`);
    }
    if (job.state === 'cancelled') {
      box.insertAdjacentHTML('beforeend',
        '<div class="note">experiment stopped; no further model calls were started</div>');
      renderSpine(null);
      return;
    }
    if (job.state === 'error') {
      box.insertAdjacentHTML('beforeend',
        `<div class="note"><b>${escapeHtml(job.error)}</b>`
        + `<pre>${escapeHtml(job.traceback || '')}</pre></div>`);
      renderSpine(null);
      return;
    }
    onDone(job.result);
  };
  tick();
}

$('build').onclick = () => doBuild(false);
$('rebuild').onclick = () => doBuild(true);
async function doBuild(force) {
  try {
    const body = Object.assign(readConfig(), { force });
    const { job_id } = await api('/api/indexes', body);
    poll(job_id, (result) => {
      $('indexInfo').innerHTML =
        `<b>${escapeHtml(result.collection)}</b> — ${result.chunks} chunks, avg ${result.avg_chars} chars, ` +
        `p95 ${result.p95_chars}, dim ${result.embed_dim}, ${result.build_seconds}s` +
        (result.reused ? ' (reused)' : '') +
        hierarchyReport(result.hierarchy) +
        (result.notes || []).map((n) => `<div class="note">${escapeHtml(n)}</div>`).join('');
    });
  } catch (e) { $('jobBox').innerHTML = `<div class="note">${escapeHtml(e.message)}</div>`; }
}

$('run').onclick = async () => {
  const requested = archiveSettings();
  const body = Object.assign({}, requested.settings.config, pickedProvider(), {
    ragas_mode: requested.settings.ui.ragas_mode,
    limit: requested.settings.ui.limit || null,
    ragas_limit: requested.settings.ui.ragas_limit || null,
    types: requested.settings.ui.types,
  });
  try {
    const { job_id } = await api('/api/evaluations', body);
    poll(job_id, (result) => {
      const evidence = result.archive_evidence;
      const canonical = Object.assign({}, result);
      delete canonical.archive_evidence;
      const completed = ArchiveIO.completed(
        requested.settings.config, requested.settings.ui, canonical, evidence);
      CURRENT_ARCHIVE = ArchiveIO.equal(requested, archiveSettings())
        ? completed : null;
      renderResult(canonical);
      if (CURRENT_ARCHIVE) {
        setArchiveStatus(
          'Private evidence included: corpus, ground truth, answers, chunks, and traces.',
          'warning');
      } else {
        setArchiveStatus(
          'Readings belong to the previous settings; export will contain settings only.',
          'warning');
      }
      loadBoard();
    });
  } catch (e) { $('jobBox').innerHTML = `<div class="note">${escapeHtml(e.message)}</div>`; }
};

// Exactly the questions "Run evaluation" would score — the same limit and the
// same type filter, read from the same controls. A different selection here
// would put questions in the Inspector's retrieval window that no score was
// ever about, which is the one thing this view must not do.
function selectionBody() {
  return { limit: +$('limit').value || null, types: selected($('types')) };
}

// The same job `poll` already drives, awaited, so one click can chain a build
// into a retrieval. A failed or cancelled job never resolves — deliberately:
// `poll` has put the error on screen, and the step after it must not run.
function runJob(path, body) {
  return new Promise((resolve, reject) => {
    api(path, body).then(({ job_id }) => poll(job_id, resolve), reject);
  });
}

async function doRetrieve() {
  const body = Object.assign(readConfig(), pickedProvider(), selectionBody());
  const result = await runJob('/api/retrievals', body);
  const questions = result.questions || [];
  const gold = questions.reduce((sum, q) =>
    sum + (q.trace.candidates || []).filter((c) => c.gold).length, 0);
  $('retrieveInfo').innerHTML =
    `retrieved for <b>${questions.length}</b> question` +
    `${questions.length === 1 ? '' : 's'} · ${gold} gold chunk` +
    `${gold === 1 ? '' : 's'} among the candidates — open the Inspector ` +
    '(:9003) to read the per-question tables';
}

$('retrieve-selected').onclick = async () => {
  try { await doRetrieve(); }
  catch (e) { $('jobBox').innerHTML = `<div class="note">${escapeHtml(e.message)}</div>`; }
};

$('cancel').onclick = async () => {
  const jobId = $('cancel').dataset.jobId;
  if (!jobId) return;
  $('cancel').disabled = true;
  try { await api('/api/jobs/' + jobId + '/cancel', undefined, 'POST'); }
  catch (e) { $('jobBox').innerHTML = `<div class="note">${escapeHtml(e.message)}</div>`; }
};

$('stopPoll').onclick = () => { boot(); };

// The labels, the step each score grades and the text behind its '!' all come
// from the service (metrics.MEASURES and ragas_eval.RAGAS_MEASURES). Nothing
// about a metric is written twice, so no number on this page can end up with a
// name its definition does not carry.
const measures = () => OPTIONS.metrics || [];
const measureOf = (key, catalogue = measures()) => catalogue.find((m) => m.key === key)
  || { key, label: key, short: '', step: '' };
const measureWhy = (key, catalogue = measures()) => {
  const metric = measureOf(key, catalogue);
  const topic = `metric.${key}`;
  return `<button type="button" class="why" data-topic="${escapeHtml(topic)}" `
    + `data-help="${escapeHtml(metric.help || '')}" `
    + `aria-label="What is ${escapeHtml(metric.label)}?">!</button>`;
};

// --- portable experiment exchange -----------------------------------------

function archiveSettings() {
  return ArchiveIO.settings(readConfig(), {
    mode: $('mode').value,
    ragas_mode: $('ragas_mode').value,
    limit: +$('limit').value,
    ragas_limit: +$('ragas_limit').value,
    types: selected($('types')),
  });
}

const configValue = (config, path) => {
  const [group, field] = path.split('.');
  return config[group][field];
};

function validateAgainstPanelOptions(imported) {
  const config = imported.settings.config;
  const ui = imported.settings.ui;
  ArchiveIO.datasetDisposition(imported, OPTIONS.datasets.map((row) => row.id));

  const choices = [
    ['index.chunker', OPTIONS.chunkers],
    ['index.embedder', (OPTIONS.embedder_hints || []).map((row) => row.kind)],
    ['index.hierarchy', OPTIONS.hierarchies],
    ['index.graph_source', OPTIONS.graph_sources],
    ['index.summarizer', OPTIONS.summarizers],
    ['retrieval.retriever', OPTIONS.retrievers],
    ['retrieval.reranker', OPTIONS.rerankers],
    ['retrieval.grader', OPTIONS.graders],
    ['retrieval.summary_scope', OPTIONS.summary_scopes],
    ['generation.answerer', OPTIONS.answerers],
    ['agent.scope', OPTIONS.scopes],
    ['agent.critic', OPTIONS.critics],
  ];
  for (const [path, values] of choices) {
    const value = configValue(config, path);
    if (!(values || []).includes(value)) {
      throw new Error(`settings.config.${path}: ${value} is not served by this lab`);
    }
  }

  const modes = ['', ...(OPTIONS.modes || []).map((row) => row.key)];
  if (!modes.includes(ui.mode)) {
    throw new Error(`settings.ui.mode: ${ui.mode} is not served by this lab`);
  }
  if (![...$('ragas_mode').options].some((option) => option.value === ui.ragas_mode)) {
    throw new Error(`settings.ui.ragas_mode: ${ui.ragas_mode} is not available`);
  }
  const questionTypes = new Set(OPTIONS.question_types || []);
  for (const type of ui.types) {
    if (!questionTypes.has(type)) {
      throw new Error(`settings.ui.types: ${type} is not served by this lab`);
    }
  }

  const embedModels = new Set((OPTIONS.embed_models || []).map((row) => row.id));
  if (config.index.embed_model && !embedModels.has(config.index.embed_model)) {
    throw new Error(`settings.config.index.embed_model: ${config.index.embed_model} is not served by this lab`);
  }
  const mode = (OPTIONS.modes || []).find((row) => row.key === ui.mode);
  const modelIds = new Set(((mode && mode.models) || OPTIONS.models || [])
    .map((row) => row.id));
  for (const role of OPTIONS.model_roles || []) {
    const value = configValue(config, role.field);
    if (value && !modelIds.has(value)) {
      throw new Error(`settings.config.${role.field}: ${value} is not served in ${ui.mode || 'boot'} mode`);
    }
  }

  const numericControls = [
    ['chunk_chars', 'index.chunk_chars'], ['overlap', 'index.overlap'],
    ['graph_knn', 'index.graph_knn'], ['granularity', 'index.granularity'],
    ['hierarchy_levels', 'index.hierarchy_levels'], ['min_group', 'index.min_group'],
    ['k', 'retrieval.k'], ['candidates', 'retrieval.candidates'],
    ['rerank_depth', 'retrieval.rerank_depth'],
    ['recency_half_life_days', 'retrieval.recency_half_life_days'],
    ['mmr_lambda', 'retrieval.mmr_lambda'],
    ['grade_threshold', 'retrieval.grade_threshold'],
    ['summary_boost', 'retrieval.summary_boost'],
    ['max_hops', 'agent.max_hops'],
    ['evidence_threshold', 'agent.evidence_threshold'],
    ['max_revisions', 'agent.max_revisions'],
    ['max_llm_calls', 'agent.max_llm_calls'],
    ['limit', 'ui.limit'], ['ragas_limit', 'ui.ragas_limit'],
  ];
  for (const [id, path] of numericControls) {
    const control = $(id);
    const value = path.startsWith('ui.') ? ui[path.slice(3)] : configValue(config, path);
    const minimum = control.min === '' ? null : Number(control.min);
    const maximum = control.max === '' ? null : Number(control.max);
    if ((minimum !== null && value < minimum) || (maximum !== null && value > maximum)) {
      throw new Error(`settings.${path}: ${value} is outside the panel range`);
    }
  }
}

function removeArchivedDatasetOptions() {
  for (const option of $('dataset').querySelectorAll('option[data-archive-view-only]')) {
    option.remove();
  }
}

function addArchivedDatasetOption(value, text) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = text || `${value} — archived · view-only`;
  option.dataset.archiveViewOnly = 'true';
  $('dataset').appendChild(option);
  return option;
}

function setArchiveViewOnly(active) {
  if (active) ARCHIVE_VIEW_ONLY = true;
  else ARCHIVE_VIEW_ONLY = false;
  $('build').disabled = ARCHIVE_VIEW_ONLY;
  $('rebuild').disabled = ARCHIVE_VIEW_ONLY;
  $('retrieve-selected').disabled = ARCHIVE_VIEW_ONLY;
  $('run').disabled = ARCHIVE_VIEW_ONLY;
  if (ARCHIVE_VIEW_ONLY) {
    $('corpus').textContent = `${$('dataset').value} · archived evidence`;
    $('datasetInfo').textContent =
      'This dataset is embedded in the archive but is not installed here. '
      + 'The completed evidence is view-only; import the dataset separately to run it.';
  }
}

function syncArchiveViewOnlyFromDataset() {
  const option = $('dataset').selectedOptions[0];
  const viewOnly = Boolean(option && option.dataset.archiveViewOnly === 'true');
  setArchiveViewOnly(viewOnly);
  if (!viewOnly) removeArchivedDatasetOptions();
}

function writeArchiveSettings(imported, restoration = null) {
  const value = ArchiveIO.normalize(imported);
  const config = value.settings.config;
  const ui = value.settings.ui;
  const disposition = restoration || ArchiveIO.datasetDisposition(
    value, OPTIONS.datasets.map((row) => row.id));

  removeArchivedDatasetOptions();
  if (disposition.viewOnly) {
    addArchivedDatasetOption(config.index.dataset,
      restoration && restoration.optionText
        ? restoration.optionText : `${disposition.dataset} — archived · view-only`);
  }
  $('mode').value = ui.mode;
  fillModels();
  applyDefaults(config);
  keepUnshown(config);
  $('ragas_mode').value = ui.ragas_mode;
  $('limit').value = ui.limit;
  $('ragas_limit').value = ui.ragas_limit;
  const wantedTypes = new Set(ui.types);
  for (const input of $('types').querySelectorAll('input')) {
    input.checked = wantedTypes.has(input.value);
  }
  applyDependencies();
  setArchiveViewOnly(disposition.viewOnly);
  if (!disposition.viewOnly) describeDataset();

  const expected = ArchiveIO.settings(config, ui);
  if (!ArchiveIO.equal(archiveSettings(), expected)) {
    throw new Error('Imported settings could not be represented exactly by this panel');
  }
}

function snapshotDashboard() {
  const archivedOption = $('dataset').querySelector('option[data-archive-view-only]');
  return {
    settings: archiveSettings(),
    currentArchive: CURRENT_ARCHIVE,
    resultHtml: $('resultCard').innerHTML,
    archiveViewOnly: ARCHIVE_VIEW_ONLY,
    archivedOption: archivedOption ? {
      value: archivedOption.value, text: archivedOption.textContent,
    } : null,
    statusText: $('archive-status').textContent,
    statusClass: $('archive-status').className,
  };
}

function restoreDashboard(before) {
  writeArchiveSettings(before.settings, {
    dataset: before.settings.settings.config.index.dataset || 'diary-fa',
    viewOnly: before.archiveViewOnly,
    optionText: before.archivedOption && before.archivedOption.text,
  });
  CURRENT_ARCHIVE = before.currentArchive;
  $('resultCard').innerHTML = before.resultHtml;
  for (const tableElement of $('resultCard').querySelectorAll('table')) {
    SortTable.make(tableElement);
  }
  $('archive-status').textContent = before.statusText;
  $('archive-status').className = before.statusClass;
}

function setArchiveStatus(message, tone = '') {
  $('archive-status').textContent = message;
  $('archive-status').className = `archive-status${tone ? ' ' + tone : ''}`;
}

function databaseMessage(disposition) {
  return disposition === 'created'
    ? 'Imported archive saved in Every experiment; leaderboard unchanged.'
    : 'Database already contained this id; existing record unchanged. '
      + 'Inspector preview shows the selected file.';
}

async function importArchiveFile(file) {
  try {
    // Keep the policy visible at the integration boundary as well as in the
    // codec: reading an oversized file is already too late.
    const archiveByteLimit = 32 * 1024 * 1024;
    if (ArchiveIO.MAX_BYTES !== archiveByteLimit) {
      throw new Error('Archive size policy mismatch');
    }
    ArchiveIO.assertFileSize(file.size);
    const imported = ArchiveIO.parse(await file.text());
    const before = snapshotDashboard();
    let savedArchive = null;
    try {
      ArchiveIO.transact(imported, {
        read: archiveSettings,
        validate: validateAgainstPanelOptions,
        write: (next) => writeArchiveSettings(next,
          ArchiveIO.equal(next, before.settings) ? {
            dataset: before.settings.settings.config.index.dataset || 'diary-fa',
            viewOnly: before.archiveViewOnly,
            optionText: before.archivedOption && before.archivedOption.text,
          } : null),
      });
      if (imported.evaluation) {
        renderResult(imported.evaluation.result,
          { remember: false, imported: true,
            metric_catalogue: imported.evaluation.metric_catalogue });
        savedArchive = await api('/api/imported-archives', imported);
      } else {
        await api('/api/imported-archives/active', undefined, 'DELETE');
      }
    } catch (error) {
      restoreDashboard(before);
      throw error;
    }
    CURRENT_ARCHIVE = imported.evaluation ? imported : null;
    remember(SAVED_CONFIG, readConfig());
    if (savedArchive) {
      let message = databaseMessage(savedArchive.database)
        + ' Private evidence included: corpus, ground truth, answers, chunks, and traces.';
      if (ARCHIVE_VIEW_ONLY) {
        message += ' Dataset unavailable here; completed evidence is view-only.';
      }
      setArchiveStatus(message, ARCHIVE_VIEW_ONLY ? 'warning' : 'success');
      loadExperiments(true).catch((error) => setArchiveStatus(
        `${databaseMessage(savedArchive.database)}; list refresh failed: ${error.message}`,
        'warning'));
    } else {
      setArchiveStatus('Settings imported; no evaluation was run.', 'success');
    }
  } finally {
    $('archive-file').value = '';
  }
}

function exportArchive() {
  const settings = archiveSettings();
  const includesEvidence = Boolean(CURRENT_ARCHIVE
    && ArchiveIO.equal(CURRENT_ARCHIVE.settings, settings.settings));
  const exported = includesEvidence ? CURRENT_ARCHIVE : settings;
  const encoded = ArchiveIO.stringify(exported);
  ArchiveIO.assertFileSize(new TextEncoder().encode(encoded).byteLength);
  const url = URL.createObjectURL(new Blob([encoded], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'raglab-experiment.json';
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setArchiveStatus(includesEvidence
    ? 'Private evidence included: corpus, ground truth, answers, chunks, and traces.'
    : 'Settings-only experiment exported; no corpus, ground truth, answers, chunks, or traces included.',
  includesEvidence ? 'warning' : 'success');
}

$('archive-file').onchange = async () => {
  const file = $('archive-file').files[0];
  if (!file) return;
  try { await importArchiveFile(file); }
  catch (error) { setArchiveStatus(error.message, 'error'); }
};
$('archive-import').onkeydown = (event) => {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  event.preventDefault();
  $('archive-file').click();
};
$('archive-export').onclick = () => {
  try { exportArchive(); }
  catch (error) { setArchiveStatus(error.message, 'error'); }
};

function renderResult(result, options = {}) {
  const safe = (value) => escapeHtml(String(value ?? ''));
  const metricCatalogue = options.metric_catalogue || measures();
  // Held evidence follows the run on display: an older leaderboard or ledger
  // row changes no control, so the settings-change invalidation never fires —
  // but exporting another run's evidence under this card would let the file
  // disagree with the screen.
  if (CURRENT_ARCHIVE
      && CURRENT_ARCHIVE.evaluation.result.run_id !== result.run_id) {
    CURRENT_ARCHIVE = null;
    setArchiveStatus(
      'Readings belong to a different run; export will contain settings only.',
      'warning');
  }
  // Remembered here rather than at each caller: a run that finished, a
  // leaderboard click and a ledger click are three ways to be looking at the same
  // experiment, and all three should still be there after a refresh.
  if (options.remember !== false && result.run_id) remember(SAVED_RUN, result.run_id);
  $('resultEmpty').hidden = true;
  $('resultBody').hidden = false;
  const s = result.summary.overall;
  $('resultMeta').textContent =
    `${result.label} · ${result.summary.n_questions} questions · ${result.seconds}s · ` +
    `${result.index.chunks} chunks (${result.index.collection})` +
    (result.index.reused ? ' reused' : '')
    + (options.imported ? ' · imported · read-only' : '');
  $('notes').innerHTML = (result.notes || []).map((n) => `<div class="note">${escapeHtml(n)}</div>`).join('');
  $('scores').innerHTML = metricCatalogue.filter((m) =>
    s[m.key] !== null && s[m.key] !== undefined)
    .map((m) => {
      const v = s[m.key];
      const isRate = m.key !== 'latency_ms';
      const bar = isRate ? `<div class="bar"><i style="width:${Math.max(0, Math.min(1, v)) * 100}%"></i></div>` : '';
      return `<div class="score"${m.step ? ` data-step="${m.step}"` : ''}>
        <span>${escapeHtml(m.label)}${measureWhy(m.key, metricCatalogue)}</span>
        <b>${isRate ? safe(fmt(v)) : safe(Math.round(v))}</b>
        <span class="muted">${escapeHtml(m.short)}</span>${bar}</div>`;
    }).join('');

  const t = result.summary.by_type;
  renderTable('byType',
    ['type', 'n', 'recall', 'quote', 'nDCG', 'hit', 'abstain ok', 'false ref'],
    Object.entries(t).map(([name, row]) => [safe(name), safe(row.n), safe(fmt(row.recall)),
      safe(fmt(row.quote_recall)), safe(fmt(row.ndcg)), safe(fmt(row.hit)),
      safe(fmt(row.abstained_correctly)), safe(fmt(row.false_abstention))]));

  const r = result.ragas || {};
  const metrics = r.metrics || {};
  $('ragas').innerHTML = Object.keys(metrics).length
    // Wrapped in a span on purpose: the explainer is inserted after the button's
    // parent, and a <p> placed directly after a <td> would be hoisted out of the
    // table by the parser.
    ? table(['metric', 'score'], Object.entries(metrics).map(([k, v]) =>
      [`<span class="measure">${escapeHtml(measureOf(k, metricCatalogue).label)}`
       + `${measureWhy(k, metricCatalogue)}</span>`, safe(fmt(v))])) +
      `<div class="muted" style="font-size:.7rem">mode ${escapeHtml(r.mode || '')} · ${safe(r.n_samples)} samples · ${safe(r.skipped)} skipped</div>` +
      (r.notes || []).map((n) => `<div class="note">${escapeHtml(n)}</div>`).join('')
    : `<div class="muted">no RAGAS scores${(r.notes || []).length ? ': ' + escapeHtml(r.notes.join('; ')) : ''}</div>`;

  $('extras').innerHTML =
    table(['difficulty', 'n', 'recall'], Object.entries(result.summary.by_difficulty)
      .map(([k, v]) => [safe(k), safe(v.n), safe(fmt(v.recall))]));

  renderTable('rows',
    ['id', 'type', 'diff', 'recall', 'quote', 'ndcg', 'ctx', 'abst', 'ms'],
    result.rows.map((row) => [safe(row.id), safe(row.type), safe(row.difficulty),
      safe(fmt(row.recall, 2)), safe(fmt(row.quote_recall, 2)), safe(fmt(row.ndcg, 2)),
      safe(row.n_contexts), row.abstained ? 'yes' : '',
      safe(Math.round(row.latency_ms))]));
}

function table(head, rows) {
  return `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${r.map((c) => `<td>${c === null || c === undefined ? '—' : c}</td>`).join('')}</tr>`).join('') +
    '</tbody></table>';
}

// Write a table into an element and make its columns sortable. Every table on
// this page goes through here, so a table added later is sortable by having been
// rendered rather than by someone remembering to wire it — and the wiring
// happens after insertion, because it needs the real rows.
function renderTable(id, head, rows) {
  const host = typeof id === 'string' ? $(id) : id;
  host.innerHTML = table(head, rows);
  SortTable.make(host.querySelector('table'));
  return host;
}

// How many runs the leaderboard asks for. Stated rather than left to the
// service's default, and reported beside the table, so a truncated board says so.
const BOARD_LIMIT = 200;

async function loadBoard() {
  const { runs, total } = await api('/api/evaluations?limit=' + BOARD_LIMIT);
  // Two different reasons the table can be shorter than the directory: a limit
  // hides runs that exist, while a file that will not parse as a run is not a
  // run — and must not be reported as one truncated away.
  $('boardMeta').textContent = total > runs.length
    ? (runs.length >= BOARD_LIMIT
      ? `newest ${runs.length} of ${total} run files`
      : `${runs.length} of ${total} files in .runs/ are readable runs`)
    : `${runs.length} run${runs.length === 1 ? '' : 's'}`;
  if (!runs.length) { $('board').innerHTML = '<div class="muted">no runs yet</div>'; return; }
  // Ranked by the RAGAS decision score, because that is the number the
  // architecture is chosen by. Rows it could not be measured on — offline runs,
  // runs recorded before the score existed — sort to the bottom rather than
  // being dropped: an unranked run is still a measurement.
  const ranked = runs.slice().sort((a, b) => {
    const x = a.ragas_decision, y = b.ragas_decision;
    if (x === y || (x == null && y == null)) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return y - x;
  });
  // One table per dataset, never one ranking across them — a decision score is
  // a mean over one corpus's questions, and comparing across corpora would call
  // a different question a result. `leaderboard.py` groups the same way.
  const byDataset = new Map();
  for (const r of ranked) {
    const key = r.dataset || 'diary-fa';
    if (!byDataset.has(key)) byDataset.set(key, []);
    byDataset.get(key).push(r);
  }
  const head = ['run', 'label', 'chunker', 'embedder', 'retriever', 'reranker',
      // No arrow on the header: an indicator that cannot move becomes a lie the
      // moment you sort by another column. Which column the ranking is on is
      // stated in prose under the heading, where it stays true.
      'n', 'ragas_decision', 'faith', 'ans rel', 'ctx prec', 'ctx recall',
      'headline', 'recall', 'quote', 'nDCG', 'abstain', 's'];
  const cells = (rows) => rows.map((r) => {
      const i = r.config.index, q = r.config.retrieval, s = r.summary.overall || {};
      const g = r.ragas || {};
      return [`<a href="#" onclick="showRun('${r.run_id}');return false">${r.run_id}</a>`,
        escapeHtml(r.label), i.chunker + (i.contextual ? '+ctx' : ''),
        // The model, not just the kind: two "fastembed" rows can be two entirely
        // different representations, and the row has to say which one it was.
        i.embedder + (i.embed_model ? '·' + i.embed_model.split('/').pop() : ''),
        q.retriever, q.reranker, r.n_questions,
        // The deciding score first, then its four constituents, so a row can be
        // checked rather than trusted.
        // With its standard error, where there is one. Candidates in a sweep can
        // sit inside each other's error bars, and a bare mean cannot say so.
        `<b>${fmt(r.ragas_decision)}</b>` + (r.ragas_decision_stderr != null
          ? ` <span class="stderr">± ${fmt(r.ragas_decision_stderr)}</span>` : ''),
        fmt(g.faithfulness), fmt(g.answer_relevancy),
        fmt(g.llm_context_precision_with_reference), fmt(g.context_recall),
        fmt(s.headline), fmt(s.recall), fmt(s.quote_recall), fmt(s.ndcg),
        fmt(s.abstained_correctly),
        Math.round(r.seconds)];
    });
  $('board').innerHTML = '';
  for (const [dataset, rows] of byDataset) {
    const known = (OPTIONS.datasets || []).find((d) => d.id === dataset);
    const section = document.createElement('div');
    section.className = 'board-group';
    section.innerHTML =
      `<h3>${escapeHtml(known ? known.name : dataset)} `
      + `<span class="muted">· ${rows.length} run${rows.length === 1 ? '' : 's'}`
      + `${known ? ' · ' + escapeHtml(known.language || '') : ''}</span></h3>`
      + table(head, cells(rows));
    $('board').appendChild(section);
    SortTable.make(section.querySelector('table'));
  }
}

// Every experiment this lab has finished, from raglab.db. Deliberately not
// ranked: a build or a retrieval measures nothing, so numbering these rows
// would claim an ordering the work does not support.
async function loadExperiments(throwOnError = false) {
  const safe = (value) => escapeHtml(String(value ?? ''));
  let rows = [];
  try {
    rows = (await api('/api/experiments')).experiments;
  } catch (e) {
    $('experiments').innerHTML = `<div class="note">${escapeHtml(e.message)}</div>`;
    if (throwOnError) throw e;
    return;
  }
  const kinds = {};
  for (const r of rows) kinds[r.kind] = (kinds[r.kind] || 0) + 1;
  $('experimentsMeta').textContent = rows.length
    ? `${rows.length} recorded · ` + Object.entries(kinds)
      .map(([kind, n]) => `${n} ${kind}`).join(', ')
    : '';
  if (!rows.length) {
    $('experiments').innerHTML =
      '<div class="muted">nothing recorded yet — build an index or run '
      + 'something, and it lands here</div>';
    return;
  }
  const host = renderTable('experiments',
    ['when', 'kind', 'id', 'label', 'backend', 'chunker', 'embedder',
      'retriever', 'reranker', 'grader', 'answerer', 'n', 'decision', 'state', 's'],
    rows.map((r) => [
      safe(r.started_at),
      `<span class="pill">${safe(r.kind)}</span>`,
      safe(r.experiment_id),
      safe(r.label || ''),
      // Marked rather than merely printed: on `fake` every LLM number on the
      // row came from a stub that cannot fail.
      r.provider === 'fake'
        ? '<b title="a stub answered and judged every call: these numbers measure '
          + 'nothing">fake</b>'
        : safe(r.provider || '—'),
      safe(r.chunker || '—'), safe(r.embedder || '—'),
      safe(r.retriever || '—'), safe(r.reranker || '—'),
      safe(r.grader || '—'), safe(r.answerer || '—'),
      safe(r.n_questions || '—'),
      // Blank, never 0, when nothing was judged — the leaderboard's own rule.
      r.decision == null ? '—' : `<b>${safe(fmt(r.decision))}</b>`
        + (r.decision_stderr != null
          ? ` <span class="stderr">± ${safe(fmt(r.decision_stderr))}</span>` : ''),
      r.state === 'done' ? 'done'
        : `<b title="${safe(r.error || '')}">${safe(r.state)}</b>`,
      safe(Math.round(r.seconds)),
    ]));
  const tableRows = host.querySelectorAll('tbody tr');
  rows.forEach((r, index) => {
    const cell = tableRows[index].children[2];
    const link = document.createElement('a');
    link.href = '#';
    link.textContent = String(r.experiment_id ?? '');
    link.addEventListener('click', (event) => {
      event.preventDefault();
      window.showExperiment(r.experiment_id);
    });
    cell.replaceChildren(link);
  });
}

// The whole stored payload for one experiment — config, per-question rows,
// traced ranks. Chunk text is deliberately not in there: it belongs to the
// index fingerprint, and rebuilding reproduces it exactly.
window.showExperiment = async (id) => {
  const box = $('experimentDetail');
  box.innerHTML = '<div class="muted">reading the ledger…</div>';
  try {
    const found = await api('/api/experiments/' + encodeURIComponent(id));
    box.innerHTML = `<details style="margin-top:.7rem">`
      + `<summary>the stored detail · ${escapeHtml(found.kind)} · `
      + `${escapeHtml(found.experiment_id)}`
      + `${found.label ? ' · ' + escapeHtml(found.label) : ''}</summary>`
      + `<pre>${escapeHtml(JSON.stringify(found.detail, null, 1))}</pre></details>`;
    // An evaluation's stored detail is exactly what the readings card renders,
    // so it renders there too, rather than only as JSON to read by hand.
    if (found.kind === 'run' && found.detail && found.detail.summary) {
      renderResult(found.detail);
      $('resultCard').scrollIntoView({ behavior: 'smooth' });
    }
  } catch (e) {
    box.innerHTML = `<div class="note">${escapeHtml(e.message)}</div>`;
  }
};

window.showRun = async (runId) => {
  renderResult(await api('/api/evaluations/' + runId));
  $('resultCard').scrollIntoView({ behavior: 'smooth' });
};

boot();

// --- the LLM widget: one module behind /api/widget, a window in the corner --
// Replies are model output rendered into the page, so they pass through the
// shared escapeHtml like every other untrusted string.

function widgetSay(kind, text) {
  const log = $('widget-log');
  log.insertAdjacentHTML('beforeend',
    `<div class="widget-msg ${kind}">${escapeHtml(text)}</div>`);
  log.scrollTop = log.scrollHeight;
}

// One conversation per page: the id is minted when the script loads and sent
// with every ask, so a follow-up lands in the same thread and a reloaded page
// starts clean — nothing persisted, nothing shared between tabs.
const widgetSession = crypto.randomUUID();

async function widgetAsk(message) {
  widgetSay('you', message);
  $('widget-send').disabled = true;
  try {
    const data = await api('/api/widget',
      { message, model: $('widget-model').value, session: widgetSession });
    widgetSay('bot', data.reply);
    // The token account, when the backend reported one — an unreported
    // account renders nothing rather than a made-up zero.
    if (data.input_tokens != null) {
      widgetSay('meta', `${data.input_tokens} in / ${data.output_tokens} out tokens`);
    }
  } catch (error) {
    widgetSay('err', error.message);
  } finally {
    $('widget-send').disabled = false;
    $('widget-input').focus();
  }
}

// The model list is served, not kept here — fetched once, on the first open.
let widgetModelsLoaded = false;
async function widgetLoadModels() {
  if (widgetModelsLoaded) return;
  try {
    const data = await api('/api/widget');
    $('widget-model').innerHTML = data.models.map((m) =>
      `<option value="${escapeHtml(m.value)}"${m.value === data.default ? ' selected' : ''}>`
      + `${escapeHtml(m.label)}</option>`).join('');
    widgetModelsLoaded = true;
  } catch (error) {
    widgetSay('err', error.message);
  }
}

$('widget-launch').addEventListener('click', () => {
  const win = $('widget-window');
  win.hidden = !win.hidden;
  if (!win.hidden) { widgetLoadModels(); $('widget-input').focus(); }
});

$('widget-settings').addEventListener('click', () => {
  const row = $('widget-config');
  row.hidden = !row.hidden;
});

$('widget-close').addEventListener('click', () => { $('widget-window').hidden = true; });

$('widget-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const message = $('widget-input').value.trim();
  if (!message) return;
  $('widget-input').value = '';
  widgetAsk(message);
});
