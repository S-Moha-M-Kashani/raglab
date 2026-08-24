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

// How this panel spells a corpus. The built-in one is offered as '' — the
// value every run already recorded carries, and the one a fingerprint is
// computed without (`IndexConfig.fingerprint()` drops `dataset=''`), so
// choosing it changes nothing about an index that exists and spelling it
// `diary-fa` instead would rename every collection already built under it.
// Three readers have to agree on that: the option values, the lookup that
// reads one back, and the catalogue `servedKnobs()` calls served. Written out
// three times they did not, and an experiment opened from the board announced
// this lab's own default corpus as one it does not have.
//
// Which corpus is the built-in one is asked by *id*, against the one constant
// both codecs read (`ArchiveIO.BUILTIN_DATASET`). It used to be asked of a
// `source` field reading `'builtin'` — a value the service stopped sending
// when the diary became an ordinary bundled pair (D3), so the test was dead
// and every fresh selection of the diary sent the explicit id instead, which
// fingerprints away from every collection already recorded under `''`.
// The id is what a dataset actually has; a source label was a second name
// for it that could rot without a single reader noticing.
const datasetValue = (d) => (d.id === ArchiveIO.BUILTIN_DATASET ? '' : d.id);
const datasetValues = () => (OPTIONS.datasets || []).map(datasetValue);

function fillDatasets() {
  const found = OPTIONS.datasets || [];
  $('dataset').innerHTML = found.map((d) =>
    `<option value="${escapeHtml(datasetValue(d))}">`
    + `${escapeHtml(d.name)} — ${escapeHtml(d.language || '?')} · ${d.documents} `
    + `documents · ${d.questions} questions`
    + `${d.source === 'imported' ? ' · imported' : ''}</option>`).join('');
  $('dataset').onchange = () => {
    describeDataset();
    syncArchiveViewOnlyFromDataset();
  };
}

const datasetOf = (id) => (OPTIONS.datasets || []).find(
  (d) => datasetValue(d) === id);

// The declaration table (D4): one line per label the dataset's corpus and
// ground truth declare — name, type, its closed set of levels, whether a
// model extracted it, and the confidence rater that scores it, if any. Read
// straight off the loaded files, never hardcoded, so a sparse corpus just
// shows fewer rows rather than a placeholder for a label it lacks.
function renderDatasetLabels(found) {
  const rows = (found.label_declarations || []).map((row) => [row, 'corpus'])
    .concat((found.question_label_declarations || [])
      .map((row) => [row, 'question']));
  renderTable('datasetLabels',
    ['name', 'declared on', 'type', 'levels', 'extracted', 'confidence rater'],
    rows.map(([row, scope]) => [safe(row.name), scope, safe(row.type),
      safe((row.levels || []).join(', ')), row.extracted ? 'yes' : '',
      safe(row.confidence_for)]),
    { label: 'Declared labels', text: [0, 1, 2, 3, 5] });
}

// One switch-group per question label the dataset declares with a closed
// set of values or a glossary (D7) — data-driven, since the labels are the
// dataset's own, not a fixed vocabulary every corpus must share. `balance`
// lists the same labels plus "even spread" for a plain stride. Rebuilt on
// every dataset switch, so a stale label from the previous corpus cannot
// linger on screen offering values the new one has never declared.
function fillLabelFilters(labels) {
  const names = Object.keys(labels).sort();
  $('labelFilters').innerHTML = names.map((name) =>
    `<div class="switches" data-label="${escapeHtml(name)}">`
    + `<span class="muted">${escapeHtml(name)}</span>`
    + labels[name].map((value) =>
      `<label><input type="checkbox" value="${escapeHtml(value)}"> `
      + `${escapeHtml(value)}</label>`).join('') + '</div>').join('');
  $('balance').innerHTML = '<option value="">even spread</option>'
    + names.map((name) =>
      `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
}

function selectedLabels() {
  const labels = {};
  for (const group of $('labelFilters').querySelectorAll('[data-label]')) {
    const chosen = selected(group);
    if (chosen.length) labels[group.dataset.label] = chosen;
  }
  return labels;
}

function applyWantedLabels(wanted) {
  for (const group of $('labelFilters').querySelectorAll('[data-label]')) {
    const values = new Set((wanted || {})[group.dataset.label] || []);
    for (const input of group.querySelectorAll('input')) {
      input.checked = values.has(input.value);
    }
  }
}

// The label vocabulary a run's `labels`/`balance` are checked against. The
// installed catalogue answers this for any dataset served here; a view-only
// archive names one that is not, and is only ever view-only because it
// carries the completed evidence itself (`datasetDisposition` refuses
// otherwise) — so its own embedded ground truth is read instead, the same
// filtering `_question_vocab` applies server-side.
function labelVocabFor(config, archive) {
  const found = datasetOf(config.index.dataset);
  if (found) return found.question_labels || {};
  const groundTruth = ((((archive || {}).evaluation || {}).inspector || {})
    .dataset || {}).ground_truth || {};
  const fields = (groundTruth.groundtruth_dataset_metadata || {})
    .question_metadata_fields || {};
  const labels = {};
  for (const [name, declaration] of Object.entries(fields)) {
    const values = declaration.values
      || (declaration.glossary ? Object.keys(declaration.glossary) : null);
    if (values) labels[name] = values;
  }
  return labels;
}

// One line about the corpus in force, in the header where the corpus has always
// been described — switching datasets has to move that line, or the page says
// 167 documents while measuring twenty. Also rebuilds the declaration table
// and the label filters below, since both are this dataset's own.
function describeDataset() {
  const found = datasetOf($('dataset').value);
  if (!found) return;
  const period = found.period && found.period.from
    ? `${found.period.from} → ${found.period.to} · ` : '';
  // The name stays on screen because it changes what every number below means;
  // the census sits in the popover, because nobody reads a stats strip twice.
  $('corpusName').textContent = found.id || $('dataset').value;
  $('corpus').textContent =
    `${found.documents} documents · ${found.parts} parts · ${period}`
    + `${found.questions} questions`
    + `${found.query_date ? ' · asked as of ' + found.query_date : ''}`;
  $('datasetInfo').textContent = found.description;
  renderDatasetLabels(found);
  fillLabelFilters(found.question_labels || {});
}

// Import: a dataset is two files, paired by id (D1) — read here and posted as
// JSON, so the service checks the pair against the contract rather than the
// browser guessing. Every problem comes back at once — fixing a corpus is a
// slow loop if each attempt reports one broken quote out of nine.
$('dataset-import').onclick = async () => {
  const corpusFile = $('dataset-corpus-file').files[0];
  const groundTruthFile = $('dataset-groundtruth-file').files[0];
  if (!corpusFile || !groundTruthFile) {
    $('importInfo').innerHTML = '<div class="note">a dataset is two files: '
      + 'pick both the corpus and its ground truth</div>';
    return;
  }
  $('importInfo').textContent =
    `reading ${corpusFile.name} and ${groundTruthFile.name}…`;
  try {
    const corpus = JSON.parse(await corpusFile.text());
    const ground_truth = JSON.parse(await groundTruthFile.text());
    const added = await api('/api/datasets', { corpus, ground_truth });
    await refreshOptions();
    $('dataset').value = added.id;
    describeDataset();
    $('importInfo').innerHTML =
      `<b>${escapeHtml(added.name)}</b> imported · ${added.documents} documents, `
      + `${added.questions} questions. Build the index to measure against it.`;
  } catch (e) {
    $('importInfo').innerHTML =
      `<div class="note">${escapeHtml(e.message)}</div>`;
  } finally {
    $('dataset-corpus-file').value = '';
    $('dataset-groundtruth-file').value = '';
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

// Escaped and never null/undefined — the one guard every table cell and
// declaration row goes through, so a table added later cannot forget it.
function safe(value) {
  return escapeHtml(String(value ?? ''));
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
let UNSHOWN = { index: {}, retrieval: {}, generation: {} };
let CURRENT_ARCHIVE = null;
let ARCHIVE_VIEW_ONLY = false;
let ARCHIVE_EVENTS_BOUND = false;

// Remember the parts of a config the controls cannot express, having applied it.
//
// `applied` is not always this installation's own config on its way back in
// — an open board row hands this whatever it reconciled, and a config saved
// to `localStorage` before this schema renamed a field can still be sitting
// there. Either way, a name this lab's served defaults do not have at all
// (`generation.key_facts_judge`, before `fact_judge` existed) is not an
// unshown *knob* to carry through — it is nowhere on the schema this
// installation runs, and keeping it here would let it reappear on every
// `readConfig()` from now on, including the one the strict archive codec
// validates on the next write. Filtered against `OPTIONS.defaults` — the
// server's own current shape — rather than skipped when `OPTIONS` has not
// loaded yet (`validNames.size` guards that one boot-ordering case only).
function keepUnshown(applied) {
  const shown = readShownConfig();
  const defaults = (OPTIONS && OPTIONS.defaults) || {};
  UNSHOWN = { index: {}, retrieval: {}, generation: {} };
  for (const group of ['index', 'retrieval', 'generation']) {
    const validNames = new Set(Object.keys(defaults[group] || {}));
    for (const [key, value] of Object.entries(applied[group] || {})) {
      if (key in shown[group]) continue;
      if (validNames.size && !validNames.has(key)) continue;
      UNSHOWN[group][key] = value;
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
      answerer: $('answerer').value, fact_judge: $('fact_judge').checked,
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
  for (const group of ['index', 'retrieval', 'generation']) {
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
function checkDependencyCondition(condition, cfg) {
  const [group, name] = condition.field.split('.');
  const current = (cfg[group] || {})[name];
  const enabled = condition.on_true ? Boolean(current)
    : (condition.on || []).indexOf(current) !== -1;
  return { enabled, reason: enabled ? '' : condition.reason };
}

function dependencyState(rules, cfg) {
  const state = {};
  const resolve = (path, seen) => {
    if (state[path]) return state[path];
    const rule = rules[path];
    let { enabled, reason } = checkDependencyCondition(rule, cfg);
    if (enabled && rules[rule.field] && seen.indexOf(path) === -1) {
      const above = resolve(rule.field, seen.concat([path]));
      if (!above.enabled) { enabled = false; reason = above.reason; }
    }
    // `also`: a second, independent condition of the same shape a rule may
    // carry (D5/D6, composed with the reranker-based rule rather than
    // replacing it) — both must hold, checked after the primary condition
    // and its chain, so a control killed by its owner keeps that reason.
    if (enabled && rule.also) {
      const also = checkDependencyCondition(rule.also, cfg);
      if (!also.enabled) { enabled = false; reason = also.reason; }
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
  // The new source D5/D6 gave this mechanism: a dataset fact rather than
  // another knob, read off the served catalogue under a synthetic `dataset`
  // group beside the three real config groups — resolved by the exact same
  // rule shape every other entry uses.
  const dataset = datasetOf(cfg.index.dataset) || {};
  cfg.dataset = { date_label: dataset.date_label || '',
                 ranks_label: dataset.ranks_label || '' };
  const state = dependencyState(rules, cfg);
  for (const path of Object.keys(rules)) {
    const el = controlFor(path);
    if (!el) continue;
    const { enabled, reason } = state[path];
    el.disabled = !enabled;
    const holder = el.closest('div') || el.parentElement;
    if (!holder) continue;
    holder.classList.toggle('rag-field-off', !enabled);
    // No `title` here. The same sentence is written into the visible note
    // below, so the tooltip was a second copy reachable only by hovering —
    // and a reason worth giving is worth giving on the page.
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
  $('fact_judge').checked = d.generation.fact_judge;
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
//
// And the opposite edge of the same rule: a saved config may carry a knob
// this lab's schema *retired* since it was written (`generation.
// key_facts_judge`, before `fact_judge` existed). Merged in unfiltered, that
// name would sit on the config every control reads and writes from boot
// onward — including the one the strict archive codec validates the moment
// anything tries to save again — so only the keys `defaults[group]` still
// has are kept from what was saved; everything else in it is left behind
// exactly as an unservable knob from an opened archive is.
function startingConfig(defaults) {
  const kept = saved(SAVED_CONFIG);
  if (!kept) return defaults;
  const merged = { label: kept.label || defaults.label || '' };
  for (const group of Object.keys(defaults)) {
    if (group === 'label') continue;
    const validNames = new Set(Object.keys(defaults[group] || {}));
    const keptGroup = {};
    for (const [key, value] of Object.entries(kept[group] || {})) {
      if (validNames.has(key)) keptGroup[key] = value;
    }
    merged[group] = Object.assign({}, defaults[group], keptGroup);
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
            || change.target.id === 'dataset-corpus-file'
            || change.target.id === 'dataset-groundtruth-file') return;
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
  restoreLastRun();
  // Last, so an experiment handed over by the board writes its knobs over a
  // panel that is already fully built rather than one still filling its selects.
  takeHandedExperiment();
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
  renderStatusPill(caps);
  if (caps.ragas.notes.length) {
    $('notes').innerHTML = caps.ragas.notes.map((n) => `<div class="note">${escapeHtml(n)}</div>`).join('');
  }
}

// One indicator taking the worst of the checks, because six permanently-green
// badges is how the header got crowded and constant green stops being read. The
// count is in the label, so the meaning never rests on the dot's colour alone —
// and the popover still lists every check, so nothing became unreachable.
function renderStatusPill(caps) {
  const checks = [caps.fastembed, caps.cross_encoder, caps.llm,
                  caps.ragas.installed, caps.ragas.llm_ready];
  const missing = checks.filter((ok) => !ok).length;
  const pill = $('statusPill');
  pill.classList.toggle('warn', missing > 0);
  $('statusPillText').textContent = missing
    ? `${missing} of ${checks.length} not ready`
    : 'all systems ready';
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

// The four step strips that used to scroll to a stage's controls are gone; the
// cards they pointed at are one scroll away and carry the same step ink and the
// same served titles, so the strips were chrome duplicating content.

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
  const track = $('chromeProgress');
  const caption = $('spineCaption');
  track.dataset.step = step;
  caption.dataset.step = step;
  track.firstElementChild.style.width =
    ((job ? job.progress : 0) * 100).toFixed(1) + '%';
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
    labels: requested.settings.ui.labels,
    balance: requested.settings.ui.balance,
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
    });
  } catch (e) { $('jobBox').innerHTML = `<div class="note">${escapeHtml(e.message)}</div>`; }
};

// Exactly the questions "Run evaluation" would score — the same limit and the
// same label filters, read from the same controls. A different selection here
// would put questions in the Inspector's retrieval window that no score was
// ever about, which is the one thing this view must not do.
function selectionBody() {
  return { limit: +$('limit').value || null, labels: selectedLabels(),
           balance: $('balance').value };
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
    'to read the per-question tables';
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
    labels: selectedLabels(),
    balance: $('balance').value,
  });
}

// --- what this installation serves ------------------------------------------
// One assembly of every constraint this panel puts on a config, in the shape
// `ExperimentHandoff.reconcile` reads. Two readers, one statement: an imported
// archive is refused on the first knob in here it cannot serve, and an
// experiment opened on the board applies every knob it can and names the rest.
// Written out twice, the two would have drifted into disagreeing about what
// this lab serves — and the disagreement would surface as a config that
// imports but cannot be opened, or the reverse.

// The numeric knobs take their bounds from the controls themselves, so a range
// changed in the markup cannot leave a validator behind believing the old one.
const NUMERIC_KNOBS = [
  ['chunk_chars', 'index.chunk_chars'], ['overlap', 'index.overlap'],
  ['graph_knn', 'index.graph_knn'], ['granularity', 'index.granularity'],
  ['hierarchy_levels', 'index.hierarchy_levels'], ['min_group', 'index.min_group'],
  ['k', 'retrieval.k'], ['candidates', 'retrieval.candidates'],
  ['rerank_depth', 'retrieval.rerank_depth'],
  ['recency_half_life_days', 'retrieval.recency_half_life_days'],
  ['mmr_lambda', 'retrieval.mmr_lambda'],
  ['grade_threshold', 'retrieval.grade_threshold'],
  ['summary_boost', 'retrieval.summary_boost'],
];

function bounds(id) {
  const control = $(id);
  return {
    min: control.min === '' ? null : Number(control.min),
    max: control.max === '' ? null : Number(control.max),
  };
}

// `mode` is a parameter rather than a read of the select, because the archive
// path validates a config against the mode the *archive* declares — that is the
// mode `writeArchiveSettings` is about to put the panel into, and validating
// against the one still on screen would refuse models the import would then
// have served perfectly well.
function servedKnobs(mode = $('mode').value) {
  const offered = (OPTIONS.modes || []).find((row) => row.key === mode);
  const chatModels = ((offered && offered.models) || OPTIONS.models || [])
    .map((row) => row.id);
  // Each catalogue carries the reason it would refuse. The chat models are what
  // *this backend mode* offers; the embedding models are what is installed here
  // at all. One reason for both sends half the readers to change the wrong
  // thing — a mode switch will never install a sentence-transformer.
  const models = {
    'index.embed_model': {
      ids: (OPTIONS.embed_models || []).map((row) => row.id),
      reason: 'not served by this lab',
    },
  };
  for (const role of OPTIONS.model_roles || []) {
    models[role.field] = {
      ids: chatModels, reason: `not served in ${mode || 'boot'} mode`,
    };
  }
  const ranges = {};
  for (const [id, path] of NUMERIC_KNOBS) ranges[path] = bounds(id);
  return {
    mode: mode || 'boot',
    datasets: datasetValues(),
    choices: {
      'index.chunker': OPTIONS.chunkers || [],
      'index.embedder': (OPTIONS.embedder_hints || []).map((row) => row.kind),
      'index.hierarchy': OPTIONS.hierarchies || [],
      'index.graph_source': OPTIONS.graph_sources || [],
      'index.summarizer': OPTIONS.summarizers || [],
      'retrieval.retriever': OPTIONS.retrievers || [],
      'retrieval.reranker': OPTIONS.rerankers || [],
      'retrieval.grader': OPTIONS.graders || [],
      'retrieval.summary_scope': OPTIONS.summary_scopes || [],
      'generation.answerer': OPTIONS.answerers || [],
    },
    models,
    ranges,
  };
}

function validateAgainstPanelOptions(imported) {
  const config = imported.settings.config;
  const ui = imported.settings.ui;
  // The corpus has its own disposition rather than the rule below: an archive
  // carrying completed evidence for a dataset this installation lacks is
  // allowed in, view-only, which is why `index.dataset` is filtered out there.
  ArchiveIO.datasetDisposition(imported, OPTIONS.datasets.map((row) => row.id));

  // Before the config, because the config's models are read against it.
  const modes = ['', ...(OPTIONS.modes || []).map((row) => row.key)];
  if (!modes.includes(ui.mode)) {
    throw new Error(`settings.ui.mode: ${ui.mode} is not served by this lab`);
  }

  // The same rule the board's handoff reads, asked in the strict direction. An
  // imported file either arrives intact or it does not arrive at all, so the
  // first knob this lab cannot serve refuses the whole config. Opening a row of
  // this lab's own board is the other case, and applies what it can.
  const refused = ExperimentHandoff
    .reconcile(config, config, servedKnobs(ui.mode)).unserved
    .filter((row) => row.path !== 'index.dataset');
  if (refused.length) {
    throw new Error(`settings.config.${refused[0].path}: `
      + `${refused[0].value} is ${refused[0].reason}`);
  }

  if (![...$('ragas_mode').options].some((option) => option.value === ui.ragas_mode)) {
    throw new Error(`settings.ui.ragas_mode: ${ui.ragas_mode} is not available`);
  }
  // D7: a question filter is one switch-group per label the *dataset*
  // declares, not a fixed vocabulary every corpus shares — so `labels` and
  // `balance` are checked against that dataset's own declaration rather
  // than a served list. A view-only archive names a dataset this
  // installation does not have installed at all; `labelVocabFor` falls back
  // to the archive's own embedded ground truth for exactly that case.
  const declared = labelVocabFor(config, imported);
  if (ui.balance && !(ui.balance in declared)) {
    throw new Error(`settings.ui.balance: ${ui.balance} is not a label this `
      + 'dataset declares');
  }
  for (const [name, values] of Object.entries(ui.labels || {})) {
    if (!(name in declared)) {
      throw new Error(`settings.ui.labels: ${name} is not a label this `
        + 'dataset declares');
    }
    const allowed = new Set(declared[name]);
    for (const value of values) {
      if (!allowed.has(value)) {
        throw new Error(`settings.ui.labels.${name}: ${value} is not a `
          + 'value this dataset declares');
      }
    }
  }
  // The two numbers that are the run's and not the config's, so they are not in
  // the shared rule: how many questions to put, and how many to judge.
  for (const key of ['limit', 'ragas_limit']) {
    const range = bounds(key);
    const value = ui[key];
    if ((range.min !== null && value < range.min)
        || (range.max !== null && value > range.max)) {
      throw new Error(`settings.ui.${key}: ${value} is outside the panel range`);
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
  // A dataset this installation serves gets its own declaration and label
  // filters rebuilt by `describeDataset`; a view-only one has no files to
  // read here, so its label vocabulary comes off the archive's own embedded
  // ground truth instead (`labelVocabFor`) — either way the filters exist
  // before the checkboxes below are set.
  if (!disposition.viewOnly) describeDataset();
  else fillLabelFilters(labelVocabFor(config, value));
  $('ragas_mode').value = ui.ragas_mode;
  $('limit').value = ui.limit;
  $('ragas_limit').value = ui.ragas_limit;
  $('balance').value = ui.balance || '';
  applyWantedLabels(ui.labels);
  applyDependencies();
  setArchiveViewOnly(disposition.viewOnly);

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
    // Same one mapping the codec and the handoff read, never a fourth copy of
    // the string: an absent dataset names the built-in corpus.
    dataset: before.settings.settings.config.index.dataset
      || ArchiveIO.BUILTIN_DATASET,
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
  $('archive-status').hidden = !before.statusText;
}

// The banner under the bar, not a permanent strip inside it: a caution that is
// true only sometimes reads as decoration within a day, so it is hidden whenever
// it has nothing to say.
function setArchiveStatus(message, tone = '') {
  const box = $('archive-status');
  box.textContent = message;
  box.className = `banner archive-status${tone ? ' ' + tone : ''}`;
  box.hidden = !message;
}

function databaseMessage(disposition) {
  return disposition === 'created'
    ? 'Imported archive recorded. It appears on the leaderboard under its dataset.'
    : 'Database already contained this id; existing record unchanged. '
      + 'Inspector preview shows the selected file.';
}

// --- one way in -------------------------------------------------------------
// Every archive that reaches this dashboard arrives through here, whether the
// reader chose a file or clicked open on the board. That is the whole change:
// an experiment opened from the board *is* its exported file, so the two cannot
// drift into disagreeing about what arriving means. The transaction is what
// makes "or it does not arrive" true — a failure anywhere below puts the
// dashboard back exactly as it was, rather than leaving it half one experiment.
// What an archive is actually carrying, named rather than assumed. The line
// used to be one fixed string ending "and traces", which was a claim the
// archive itself often contradicted: an evaluation whose per-question traces
// were never recorded carries rows and judged metrics and no recording at all,
// and the banner announced traces it did not have. Same rule as a row's, one
// layer up — the page may not describe evidence that is not there.
function evidenceCarried(value) {
  const inspector = (value.evaluation || {}).inspector || {};
  const result = (value.evaluation || {}).result || {};
  const held = [];
  if ((inspector.dataset || {}).corpus) held.push('corpus');
  if ((inspector.dataset || {}).ground_truth) held.push('ground truth');
  if ((result.rows || []).some((row) => row.answer)) held.push('answers');
  if ((inspector.chunks_by_session || []).length) held.push('chunks');
  if ((inspector.summaries || []).length) held.push('summaries');
  if ((inspector.traces || []).length) held.push('traces');
  if (!held.length) return 'No private evidence included.';
  const last = held.pop();
  return `Private evidence included: ${held.length
    ? `${held.join(', ')}, and ${last}` : last}.`;
}

async function adoptArchive(imported) {
  {
    const before = snapshotDashboard();
    let savedArchive = null;
    try {
      ArchiveIO.transact(imported, {
        read: archiveSettings,
        validate: validateAgainstPanelOptions,
        write: (next) => writeArchiveSettings(next,
          ArchiveIO.equal(next, before.settings) ? {
            dataset: before.settings.settings.config.index.dataset
              || ArchiveIO.BUILTIN_DATASET,
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
      let message = `${databaseMessage(savedArchive.database)} `
        + evidenceCarried(imported);
      if (ARCHIVE_VIEW_ONLY) {
        message += ' Dataset unavailable here; completed evidence is view-only.';
      }
      setArchiveStatus(message, ARCHIVE_VIEW_ONLY ? 'warning' : 'success');
    } else {
      setArchiveStatus('Settings imported; no evaluation was run.', 'success');
    }
  }
}

async function importArchiveFile(file) {
  try {
    // Keep the policy visible at the integration boundary as well as in the
    // codec: reading an oversized file is already too late. A file is the only
    // way in that has a size before it has a shape, so this stays here rather
    // than in `adoptArchive`.
    const archiveByteLimit = 32 * 1024 * 1024;
    if (ArchiveIO.MAX_BYTES !== archiveByteLimit) {
      throw new Error('Archive size policy mismatch');
    }
    ArchiveIO.assertFileSize(file.size);
    await adoptArchive(ArchiveIO.parse(await file.text()));
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
    ? evidenceCarried(exported)
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

// --- an experiment opened on the board --------------------------------------
// The board's open button pins the Inspector to one recorded experiment and
// hands the same experiment here, so the knobs on this page become that
// experiment's. The board cannot write them itself — only this page holds
// `/api/options` — so what crosses is an id in one slot, and `servedKnobs()`
// above decides which of the recorded knobs this installation can honour.
//
// Two ways it arrives, because both happen: this page loading with a slot
// already written, and a `storage` event, which is the only thing that reaches
// a Laboratory already open in another tab. That second case is the ordinary
// one — the board opens the Inspector in a new tab, so the reader who has both
// surfaces up is the reader this is for — and without it the button would set
// the settings on the *next* reload, which is not what it said it did.
//
// This is deliberately *not* the archive-import path, even though it used to
// share it. An imported file either arrives intact or it does not — that is
// right for a file someone chose to hand this lab, which was never here
// before. An experiment opened from this lab's own board is already a record
// here: it has a job, a ledger row, everything `ImportedArchiveStore` exists
// to give something that has none of those. So nothing here is re-posted to
// `/api/imported-archives`, and — the rule CLAUDE.md and this plan's own
// handoff design both state — a knob this installation cannot serve, or a
// name this schema no longer has at all (every experiment recorded before
// this branch carries `generation.key_facts_judge`, not `fact_judge`), is
// left at the panel's own value and *named*, never made to refuse the whole
// handoff the way the strict codec used to.

// The `ui` block's own lenient reading, the same rule `ExperimentHandoff.
// reconcile` applies to `config`, in miniature: a name this panel's own `ui`
// no longer has (`types`, before `labels`/`balance` existed) is dropped and
// named; a name it still has is applied when the value is one this panel can
// actually represent — `applyWantedLabels`/`fillLabelFilters` already ignore
// a label this dataset does not declare on their own, so only `mode`,
// `ragas_mode` and `balance` need a check here, or an unmatched `<select>`
// value would read back as `''` with nothing ever having said so.
function reconcileUi(recorded, current, config, archived) {
  const ui = Object.assign({}, current);
  const unserved = [];
  const set = [];
  const modes = ['', ...(OPTIONS.modes || []).map((row) => row.key)];
  const ragasModes = [...$('ragas_mode').options].map((option) => option.value);
  const declared = labelVocabFor(config, archived);
  for (const [knob, value] of Object.entries(recorded || {})) {
    const path = `ui.${knob}`;
    if (!(knob in current)) {
      unserved.push({ path, value, reason: 'not a knob this lab reads any more' });
      continue;
    }
    if (knob === 'mode' && !modes.includes(value)) {
      unserved.push({ path, value, reason: 'not served by this lab' });
      continue;
    }
    if (knob === 'ragas_mode' && !ragasModes.includes(value)) {
      unserved.push({ path, value, reason: 'not available' });
      continue;
    }
    if (knob === 'balance' && value && !(value in declared)) {
      unserved.push({ path, value, reason: 'not a label this dataset declares' });
      continue;
    }
    if (knob === 'labels') {
      // Filtered per label rather than refused whole: one label this dataset
      // does not declare must not cost every other filter the record named.
      const kept = {};
      const dropped = [];
      for (const [label, values] of Object.entries(value || {})) {
        if (label in declared) kept[label] = values; else dropped.push(label);
      }
      ui.labels = kept;
      set.push(path);
      if (dropped.length) {
        unserved.push({ path: 'ui.labels', value: dropped,
                        reason: 'not a label this dataset declares' });
      }
      continue;
    }
    ui[knob] = value;
    set.push(path);
  }
  return { ui, unserved, set };
}

// A knob named `ui.*` has no stage `ExperimentHandoff.notice`'s own grouping
// knows about, so it is named in its own sentence rather than silently
// dropped from the "To set" list the way an unknown group would leave it.
function uiUnservedNote(unserved) {
  if (!unserved.length) return '';
  const said = unserved.map((row) => `${row.path.slice(3)} = `
    + `${JSON.stringify(row.value)} — ${row.reason}`).join('; ');
  return ` Also left at the panel’s own value: ${said}.`;
}

// The lenient half of the handoff: every config and `ui` knob this
// installation can serve is applied, and every one it cannot — including a
// name this schema no longer has at all — is left at the panel's own value
// and returned for the notice to name. Throws only for what genuinely leaves
// nothing to open with (a representation `writeArchiveSettings` cannot square
// with the panel after applying only what passed that check), never for a
// knob this function itself already decided not to apply.
function adoptHandedSettings(archived) {
  const settingsBlock = (archived || {}).settings || {};
  const recordedConfig = settingsBlock.config || {};
  const recordedUi = settingsBlock.ui || {};
  const currentSettings = archiveSettings().settings;
  const served = servedKnobs(recordedUi.mode);
  const configOut = ExperimentHandoff.reconcile(
    recordedConfig, currentSettings.config, served);
  const uiOut = reconcileUi(
    recordedUi, currentSettings.ui, configOut.config, archived);
  writeArchiveSettings(ArchiveIO.settings(configOut.config, uiOut.ui));
  if (archived.evaluation) {
    renderResult(archived.evaluation.result,
      { remember: false, imported: true,
        metric_catalogue: archived.evaluation.metric_catalogue });
  }
  // Never the archive this experiment recorded: `settings.config`/`ui` may
  // now differ from `archived.evaluation.result.config` wherever a knob was
  // left at the panel's own value, and claiming this screen re-exports that
  // record byte for byte would be exactly the row-lying-about-what-produced-
  // it CLAUDE.md forbids. A later export is settings-only, which is honest.
  CURRENT_ARCHIVE = null;
  remember(SAVED_CONFIG, readConfig());
  // Kept apart rather than merged: `ExperimentHandoff.notice`'s own count of
  // "N knobs could not be set" has to match the detail list it prints right
  // beside it, and that list only knows the three config stages — folding the
  // `ui` knobs into the same count without a place in that list would make
  // the sentence claim more detail than it goes on to give.
  return { config: configOut, ui: uiOut };
}

async function openHandedExperiment(experimentId) {
  let archived;
  try {
    archived = await api(`/api/experiments/${encodeURIComponent(experimentId)}/archive`);
  } catch (error) {
    // A row with no complete archive, a corpus that has moved under its id, a
    // lab that stopped between the click and the read. Said, and not one knob
    // touched: settings half applied under a notice that never arrived is the
    // only outcome worse than nothing happening. The route distinguishes these
    // (404 incomplete, 409 moved corpus) and its message is the one worth
    // repeating, so this adds the id rather than inventing a reason of its own.
    Widget.note(`Could not open experiment ${experimentId} `
      + `(${error.message}). No knob was changed.`);
    return;
  }
  // Snapshotted before anything is written, and restored on any failure below
  // — a refused open must leave every control exactly as it was, never half
  // one experiment's settings under a notice that never arrived.
  const before = snapshotDashboard();
  let out;
  try {
    out = adoptHandedSettings(archived);
  } catch (error) {
    restoreDashboard(before);
    Widget.note(`Experiment ${experimentId} was not opened `
      + `(${error.message}). The panel is as it was.`);
    return;
  }
  const result = archived.evaluation ? archived.evaluation.result : {};
  const record = { experiment_id: experimentId, kind: 'run',
                   label: result.label || '', started_at: result.started_at || '',
                   dataset: result.dataset || '', source: 'both' };
  Widget.note(ExperimentHandoff.notice(record, out.config)
    + uiUnservedNote(out.ui.unserved));
  // The settings are this experiment's now, and so is the conversation: the
  // widget switches to the thread it kept for this id. Coming back to another
  // experiment brings that one's conversation back with it.
  Widget.about(experimentId);
}

window.addEventListener('storage', (event) => {
  if (event.key !== ExperimentHandoff.KEY || !event.newValue) return;
  // Read from the event rather than from the slot: with two Laboratory tabs
  // open both hear this, and a slot consumed by whichever ran first would leave
  // the other sitting on the reader's old settings under no notice at all.
  let offered = null;
  try { offered = JSON.parse(event.newValue); } catch (e) { return; }
  // Cleared all the same, so a later reload does not re-announce an experiment
  // the reader opened days ago. Clearing is idempotent, and the null it writes
  // is filtered out by the guard above rather than heard as a second handoff.
  ExperimentHandoff.taken(localStorage);
  if (offered && offered.experiment_id) {
    handOver(offered.experiment_id);
  }
});

// Both callers go through here, because neither may drop the failure. The slot
// is consumed before the work starts — it has to be, or the next reload would
// re-announce it — so a rejection nobody catches leaves the reader on knobs
// that are partly one experiment, with the promise of a notice and no notice.
// Silence is the one outcome this handoff must never have: it is indistinguish-
// able from the button doing nothing at all, which is how it read.
function handOver(experimentId) {
  openHandedExperiment(experimentId).catch((error) => {
    // Not "may be part one and part the other" any more: the archive arrives
    // inside `ArchiveIO.transact`, which puts the dashboard back on any
    // failure. What is left to say is that nothing happened — and to say it,
    // because silence is indistinguishable from the button doing nothing.
    Widget.note(`Opening experiment ${experimentId} failed `
      + `(${error && error.message ? error.message : error}). `
      + 'Nothing was changed.');
  });
}

// Two ways one experiment arrives here, and the address is the reliable one now
// that the board's link lands on this page. A slot is written by a click and
// read once; it cannot survive a reload, a bookmark, a copied link, or a click
// whose new tab boots before the writing page has finished — and every one of
// those looks to the reader like the button doing nothing. The slot is still
// read, and still consumed either way so it cannot re-announce itself days
// later, because it remains the only thing that reaches a Laboratory that was
// already open in another tab: that one hears a `storage` event, not a URL.
function takeHandedExperiment() {
  const asked = new URLSearchParams(window.location.search).get('experiment');
  const offered = ExperimentHandoff.taken(localStorage);
  const wanted = asked || (offered && offered.experiment_id);
  if (wanted) handOver(wanted);
}

function renderResult(result, options = {}) {
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

  // `behavior` is the one question-classification field every ground truth
  // carries (D2/D7 retired the fixed `type`/`difficulty` vocabularies every
  // corpus used to share) — grouped here, client-side, rather than added
  // back to `metrics.aggregate`, which no longer breaks scores down by any
  // question label since the set of labels is now the dataset's own.
  const byBehavior = {};
  for (const row of result.rows) {
    (byBehavior[row.behavior || ''] = byBehavior[row.behavior || ''] || []).push(row);
  }
  const meanOf = (rows, field) => {
    const values = rows.map((r) => r[field]).filter((v) => v !== null && v !== undefined);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  };
  renderTable('byType',
    ['behavior', 'n', 'recall', 'quote', 'nDCG', 'hit', 'abstain ok', 'false ref'],
    Object.entries(byBehavior).map(([name, rows]) => [safe(name), safe(rows.length),
      safe(fmt(meanOf(rows, 'recall'))), safe(fmt(meanOf(rows, 'quote_recall'))),
      safe(fmt(meanOf(rows, 'ndcg'))), safe(fmt(meanOf(rows, 'hit'))),
      safe(fmt(meanOf(rows, 'abstained_correctly'))), safe(fmt(meanOf(rows, 'false_abstention')))]),
    { label: 'Scores by behavior', text: [0] });

  const r = result.ragas || {};
  const metrics = r.metrics || {};
  if (Object.keys(metrics).length) {
    // renderTable, not innerHTML: this table and the difficulty one below were
    // the two built by hand, so they took the sortable styling from the
    // stylesheet and none of the listeners — headers with a pointer cursor and
    // an arrow that did nothing when clicked.
    renderTable('ragas', ['metric', 'score'],
      // Wrapped in a span on purpose: the explainer is inserted after the
      // button's parent, and a <p> placed directly after a <td> would be
      // hoisted out of the table by the parser.
      Object.entries(metrics).map(([k, v]) =>
        [`<span class="measure">${escapeHtml(measureOf(k, metricCatalogue).label)}`
         + `${measureWhy(k, metricCatalogue)}</span>`, safe(fmt(v))]),
      { label: 'RAGAS judged metrics',
        text: [0],
        after: `<div class="table-hint">mode ${escapeHtml(r.mode || '')} · ${safe(r.n_samples)} samples · ${safe(r.skipped)} skipped</div>`
          + (r.notes || []).map((n) => `<div class="note">${escapeHtml(n)}</div>`).join('') });
  } else {
    $('ragas').innerHTML =
      `<div class="muted">no RAGAS scores${(r.notes || []).length ? ': ' + escapeHtml(r.notes.join('; ')) : ''}</div>`;
  }

  // What this run actually sampled: `by_<balance>` only exists when a run
  // was balanced on a question label (D7); a plain stride reports the raw
  // selection instead, since there is no band to report a breakdown by.
  const selection = result.selection || {};
  const balance = selection.balance || '';
  const bandCounts = balance ? selection[`by_${balance}`] : null;
  renderTable('extras', bandCounts ? [balance, 'n'] : ['selection', 'value'],
    bandCounts ? Object.entries(bandCounts).map(([k, v]) => [safe(k), safe(v)])
      : [['limit', safe(selection.limit ?? 'all')], ['n', safe(selection.n)],
         ['balance', 'stride (no label balance)']],
    { label: balance ? `Selection by ${balance}` : 'Selection', text: [0] });

  // The run on screen becomes a question the helper can be asked about it. Built
  // here rather than read out of the DOM later, because this is the one place
  // that holds the run itself, and handed over rather than written into the
  // widget: `Widget.offer` is the whole of what a page may say to the helper.
  Widget.offer(widgetRunAsk(result, metricCatalogue));

  renderTable('rows',
    ['id', 'behavior', 'recall', 'quote', 'ndcg', 'ctx', 'abst', 'ms'],
    result.rows.map((row) => [safe(row.id), safe(row.behavior),
      safe(fmt(row.recall, 2)), safe(fmt(row.quote_recall, 2)), safe(fmt(row.ndcg, 2)),
      safe(row.n_contexts), row.abstained ? 'yes' : '',
      safe(Math.round(row.latency_ms))]),
    { label: 'Every question, one row each', text: [0, 1] });
}

// One table inside the shared scroll region (chrome.css), which both surfaces
// use so a wide table behaves the same on either port: the region takes focus
// and scrolls by keyboard, and its bounded height is what gives the sticky
// header something to stick against. The old markup gave each host
// `overflow-x: auto` and nothing else — which makes the host a scroll container
// on *both* axes with no height limit, so `top: 0` resolved against a box that
// never scrolled and the column names left the screen on a fifty-row ledger.
//
// `options.text` names the columns that hold words rather than figures.
// `.data-table` reads numbers right and identifiers left, so a prose column
// left unnamed would sit ragged-right against its own heading. `options.label`
// names the region and the table for a screen reader; it is not shown, because
// every one of these tables already carries the same words in a heading
// directly above it.
function table(head, rows, options = {}) {
  const prose = new Set(options.text || []);
  const kind = (at) => (prose.has(at) ? ' class="text"' : '');
  const label = escapeHtml(options.label || '');
  return `<div class="table-scroll" tabindex="0" role="region" aria-label="${label}">`
    + `<table class="data-table"><caption>${label}</caption><thead><tr>`
    + head.map((h, at) => `<th scope="col"${kind(at)}>${h}</th>`).join('')
    + '</tr></thead><tbody>'
    + rows.map((r) => `<tr>${r.map((c, at) => `<td${kind(at)}>${c === null || c === undefined ? '—' : c}</td>`).join('')}</tr>`).join('')
    + '</tbody></table></div>';
}

// Write a table into an element and make its columns sortable. Every table on
// this page goes through here, so a table added later is sortable by having been
// rendered rather than by someone remembering to wire it — and the wiring
// happens after insertion, because it needs the real rows. `options.after` is
// markup that belongs under the table but outside its scroll region: a footnote
// dragged sideways with the data is a footnote nobody finds.
function renderTable(id, head, rows, options = {}) {
  const host = typeof id === 'string' ? $(id) : id;
  host.innerHTML = table(head, rows, options) + (options.after || '');
  SortTable.make(host.querySelector('table'));
  return host;
}

// The leaderboard moved to its own surface (/leaderboard), served from
// `evaluation.leaderboard` — the same module `raglab-leaderboard` prints from.
// The board that used to live here grouped by dataset only, so one table could
// rank rows scored on different questions by different judges against each
// other. Keeping a second, looser implementation beside the real one is how two
// surfaces come to name different winners.
//
// The full experiment ledger this page used to render in its own card, with
// its own loader and its own click-to-expand handler, moved with it: the
// leaderboard's unfiltered "every experiment" view reads the same ledger rows
// this page once rendered by hand, so reading across every run this
// installation has finished has one home instead of two that could drift.

window.showRun = async (runId) => {
  renderResult(await api('/api/evaluations/' + runId));
  $('resultCard').scrollIntoView({ behavior: 'smooth' });
};

boot();

// --- the one question the Laboratory asks on the helper's behalf -----------
// The widget itself is widget.js, on every surface. This much stays here
// because it reads a *run*: the Readings card is the one place that holds the
// result, so the chip is built where the result already is rather than read
// back off the DOM — and neither the Inspector nor the board has a run of its
// own to offer. It reaches the helper the same way any page does, through
// `Widget.offer`.

// The four judged metrics, in the order the leaderboard reads them, so the chip
// names the same thing a ranking would.
const DECISION_KEYS = ['faithfulness', 'answer_relevancy',
                       'llm_context_precision_with_reference', 'context_recall'];

function widgetRunAsk(result, catalogue) {
  // Same slice the leaderboard's `when` column takes: seconds do not help
  // anyone identifying a run.
  const when = String(result.started_at || '').slice(0, 16);
  if (!when) return null;
  const metrics = (result.ragas || {}).metrics || {};
  const key = DECISION_KEYS.find((k) => metrics[k] !== null && metrics[k] !== undefined);
  if (!key) {
    // No judged metric means no decision score, and why there is none is the
    // question worth asking about that run.
    return `The run from ${when} has no decision score — what is missing?`;
  }
  return `Why did the run from ${when} score ${fmt(metrics[key])} on `
    + `${measureOf(key, catalogue).label.toLowerCase()}?`;
}
