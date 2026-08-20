// Portable experiment archive codec. Deliberately has no DOM or network access.
const ArchiveIO = (() => {
  const FORMAT = 'raglab-experiment';
  const VERSION = 1;
  const MAX_BYTES = 32 * 1024 * 1024;
  const LIMITS = Object.freeze({
    depth: 32, questions: 5000, chunks: 50000, traces: 5000,
    candidates_per_trace: 1000, list_items: 100000, string_chars: 2000000,
  });
  const TYPES = Object.freeze([
    'single-hop', 'temporal', 'multi-hop', 'aggregation', 'knowledge-update',
    'commitment', 'entity', 'pattern', 'habit', 'abstention', 'adversarial',
  ]);
  const DIFFICULTIES = Object.freeze(['easy', 'medium', 'hard']);
  const STAGES = Object.freeze(['index', 'retrieval', 'generation', 'agent', 'overall']);
  const UNSAFE_METRIC_KEYS = Object.freeze(['__proto__', 'prototype', 'constructor']);
  const RESULT_KEYS = Object.freeze([
    'run_id', 'label', 'config', 'dataset', 'index', 'summary', 'rows',
    'ragas', 'seconds', 'started_at', 'notes', 'selection',
  ]);
  const CONFIG_TEMPLATE = Object.freeze({
    index: {
      dataset: '', chunker: 'semantic-drift', chunk_chars: 500, overlap: 100,
      contextual: true, embedder: 'sentence-transformers', embed_model: '',
      hierarchy: '', graph_source: 'hybrid', graph_knn: 8, granularity: 1,
      hierarchy_levels: 1, min_group: 3, summarizer: 'centroid',
    },
    retrieval: {
      retriever: 'hybrid-rrf', k: 8, candidates: 40, rrf_k: 60,
      time_filter: true, multi_query: true, hyde: false, expansion_model: '',
      mmr_lambda: 1, reranker: 'lexical', rerank_depth: 20,
      reranker_model: '', recency_half_life_days: 180,
      agentic_weights: [1, 0.3, 0.2], grader: 'none', grade_threshold: 0,
      grader_model: '', max_context_chars: 6000, summary_scope: 'mixed',
      summary_boost: 1, summary_levels: '',
    },
    generation: {
      answerer: 'extractive', model: '', key_facts_judge: false,
      judge_model: '', ragas_model: '',
    },
    agent: {
      scope: '', max_hops: 3, rewrite: true, evidence_threshold: 0.5,
      max_revisions: 1, critic: 'grounded', max_llm_calls: 12,
      plan_model: '', critic_model: '',
    },
    label: '',
  });
  const VOCABULARIES = Object.freeze({
    'index.chunker': ['semantic-drift', 'fixed', 'fixed-overlap', 'message', 'turn-pair', 'session'],
    'index.embedder': ['sentence-transformers', 'fastembed', 'ascii-hash', 'token-hash', 'char-hash'],
    'index.hierarchy': ['', 'louvain', 'leiden', 'label-prop', 'raptor', 'agglomerative', 'kmeans', 'metadata'],
    'index.graph_source': ['hybrid', 'knn', 'lexical', 'bipartite-terms'],
    'index.summarizer': ['centroid', 'lead-idf', 'mmr', 'card'],
    'retrieval.retriever': ['hybrid-rrf', 'dense', 'bm25'],
    'retrieval.reranker': ['lexical', 'none', 'recency', 'agentic', 'cross-encoder', 'llm'],
    'retrieval.grader': ['none', 'lexical', 'llm'],
    'retrieval.summary_scope': ['mixed', 'leaves', 'summaries', 'drill-down'],
    'generation.answerer': ['extractive', 'none', 'llm'],
    'agent.scope': ['', 'retrieve', 'generate', 'full'],
    'agent.critic': ['grounded', 'both', 'none'],
  });
  const MODEL_FIELDS = Object.freeze([
    ['index', 'embed_model'],
    ['retrieval', 'expansion_model'], ['retrieval', 'reranker_model'],
    ['retrieval', 'grader_model'], ['generation', 'model'],
    ['generation', 'judge_model'], ['generation', 'ragas_model'],
    ['agent', 'plan_model'], ['agent', 'critic_model'],
  ]);

  const copy = value => JSON.parse(JSON.stringify(value));
  const fail = message => { throw new Error(message); };
  const object = (value, path) => {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
      fail(`${path}: object required`);
    }
    return value;
  };
  const array = (value, path) => {
    if (!Array.isArray(value)) fail(`${path}: list required`);
    return value;
  };
  const string = (value, path, nonempty = false) => {
    if (typeof value !== 'string' || (nonempty && value.length === 0)) {
      fail(`${path}: ${nonempty ? 'non-empty ' : ''}string required`);
    }
    return value;
  };
  const integer = (value, path, minimum = null, maximum = null) => {
    if (!Number.isInteger(value)) fail(`${path}: integer required`);
    if (minimum !== null && value < minimum) fail(`${path}: must be at least ${minimum}`);
    if (maximum !== null && value > maximum) fail(`${path}: must not exceed ${maximum}`);
    return value;
  };
  const number = (value, path, nullable = false) => {
    if (nullable && value === null) return;
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      fail(`${path}: finite number${nullable ? ' or null' : ''} required`);
    }
  };
  const boolean = (value, path) => {
    if (typeof value !== 'boolean') fail(`${path}: boolean required`);
  };
  const keys = (value, expected, path) => {
    const wanted = new Set(expected);
    const actual = Object.keys(value);
    const missing = expected.filter(key => !Object.prototype.hasOwnProperty.call(value, key));
    const extra = actual.filter(key => !wanted.has(key));
    if (missing.length) fail(`${path}: missing keys ${missing.sort().join(', ')}`);
    if (extra.length) fail(`${path}: unexpected keys ${extra.sort().join(', ')}`);
  };
  const foldedKey = key => key.toLowerCase().replace(/[^a-z0-9]/g, '');
  const secretKey = key => key.endsWith('apikey') || key.endsWith('accesstoken')
    || key === 'openrouterkey' || key === 'authorization'
    || key.includes('password') || key.includes('secret');
  const walk = (value, path = 'archive', depth = 0) => {
    if (depth > LIMITS.depth) fail(`${path}: nesting exceeds ${LIMITS.depth}`);
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      for (const [key, child] of Object.entries(value)) {
        if (secretKey(foldedKey(key))) fail(`${path}.${key}: credential fields are forbidden`);
        walk(child, `${path}.${key}`, depth + 1);
      }
    } else if (Array.isArray(value)) {
      if (value.length > LIMITS.list_items) {
        fail(`${path}: list exceeds ${LIMITS.list_items}`);
      }
      value.forEach((child, index) => walk(child, `${path}[${index}]`, depth + 1));
    } else if (typeof value === 'string') {
      if (value.length > LIMITS.string_chars) {
        fail(`${path}: string exceeds ${LIMITS.string_chars} chars`);
      }
    } else if (typeof value === 'number' && !Number.isFinite(value)) {
      fail(`${path}: number must be finite or null`);
    } else if (value !== null && !['number', 'boolean'].includes(typeof value)) {
      fail(`${path}: value is not JSON-compatible`);
    }
  };
  const shape = (value, expected, path) => {
    if (expected !== null && typeof expected === 'object' && !Array.isArray(expected)) {
      object(value, path);
      keys(value, Object.keys(expected), path);
      Object.entries(expected).forEach(([key, child]) => shape(value[key], child, `${path}.${key}`));
    } else if (Array.isArray(expected)) {
      array(value, path);
      if (expected.length) value.forEach((child, index) => shape(child, expected[0], `${path}[${index}]`));
    } else if (typeof expected === 'string') string(value, path);
    else if (typeof expected === 'boolean') boolean(value, path);
    else if (typeof expected === 'number') number(value, path);
    else if (value !== null) fail(`${path}: null required`);
  };

  const finite = (value, path) => {
    if (Array.isArray(value)) {
      value.forEach((child, index) => finite(child, `${path}[${index}]`));
    } else if (value !== null && typeof value === 'object') {
      Object.entries(value).forEach(([key, child]) => finite(child, `${path}.${key}`));
    } else if (typeof value === 'number' && !Number.isFinite(value)) {
      fail(`${path}: number must be finite or null`);
    }
  };
  const metricKey = (value, path) => {
    string(value, path, true);
    if (UNSAFE_METRIC_KEYS.includes(value)) fail(`${path}: unsafe metric key ${value}`);
    return value;
  };

  const validateConfig = config => {
    shape(config, CONFIG_TEMPLATE, 'settings.config');
    Object.entries(VOCABULARIES).forEach(([field, allowed]) => {
      const [group, name] = field.split('.');
      if (!allowed.includes(config[group][name])) {
        fail(`settings.config.${field}: unsupported value ${JSON.stringify(config[group][name])}`);
      }
    });
    MODEL_FIELDS.forEach(([group, field]) => string(
      config[group][field], `settings.config.${group}.${field}`));
    if (!['fastembed', 'sentence-transformers'].includes(config.index.embedder)
        && config.index.embed_model !== '') {
      fail('settings.config.index.embed_model: must be empty for an embedder without a model');
    }
    integer(config.retrieval.k, 'settings.config.retrieval.k', 1);
    if (config.index.hierarchy) {
      integer(config.index.hierarchy_levels, 'settings.config.index.hierarchy_levels', 1);
      integer(config.index.min_group, 'settings.config.index.min_group', 2);
      if (['hybrid', 'knn'].includes(config.index.graph_source)) {
        integer(config.index.graph_knn, 'settings.config.index.graph_knn', 1);
      }
    }
    if (config.agent.scope) {
      integer(config.agent.max_hops, 'settings.config.agent.max_hops', 1);
      integer(config.agent.max_revisions, 'settings.config.agent.max_revisions', 0);
      integer(config.agent.max_llm_calls, 'settings.config.agent.max_llm_calls', 1);
      if (['generate', 'full'].includes(config.agent.scope)
          && config.generation.answerer !== 'llm') {
        fail('settings.config.agent.scope: generation ownership requires answerer llm');
      }
    }
  };

  const validateSettings = body => {
    object(body, 'settings');
    keys(body, ['config', 'ui'], 'settings');
    object(body.config, 'settings.config');
    validateConfig(body.config);
    const ui = object(body.ui, 'settings.ui');
    keys(ui, ['mode', 'ragas_mode', 'limit', 'ragas_limit', 'types'], 'settings.ui');
    string(ui.mode, 'settings.ui.mode');
    if (!['', 'local', 'openrouter', 'claude', 'codex'].includes(ui.mode)) {
      fail(`settings.ui.mode: unsupported mode ${JSON.stringify(ui.mode)}`);
    }
    string(ui.ragas_mode, 'settings.ui.ragas_mode');
    if (!['off', 'offline', 'llm'].includes(ui.ragas_mode)) {
      fail(`settings.ui.ragas_mode: unsupported mode ${JSON.stringify(ui.ragas_mode)}`);
    }
    integer(ui.limit, 'settings.ui.limit', 0, 200);
    integer(ui.ragas_limit, 'settings.ui.ragas_limit', 0, 200);
    array(ui.types, 'settings.ui.types').forEach((type, index) => {
      string(type, `settings.ui.types[${index}]`);
      if (!TYPES.includes(type)) fail(`settings.ui.types[${index}]: unknown question type ${type}`);
    });
  };

  const unique = (rows, key, path) => {
    const found = new Map();
    array(rows, path).forEach((row, index) => {
      object(row, `${path}[${index}]`);
      const value = string(row[key], `${path}[${index}].${key}`, true);
      if (found.has(value)) fail(`${path}: duplicate ${key} ${value}`);
      found.set(value, row);
    });
    return found;
  };
  const ids = (values, path) => {
    const found = new Set();
    array(values, path).forEach((value, index) => {
      string(value, `${path}[${index}]`, true);
      if (found.has(value)) fail(`${path}: duplicate id ${value}`);
      found.add(value);
    });
    return values;
  };
  const span = (value, text, path) => {
    if (!Array.isArray(value) || value.length !== 2
        || !value.every(Number.isInteger)
        || value[0] < 0 || value[0] > value[1] || value[1] > text.length) {
      fail(`${path}: expected integer bounds within candidate text`);
    }
  };

  const validateDataset = dataset => {
    object(dataset, 'evaluation.inspector.dataset');
    keys(dataset, ['id', 'corpus', 'ground_truth'], 'evaluation.inspector.dataset');
    const datasetId = string(dataset.id, 'evaluation.inspector.dataset.id', true);
    if (!/^[a-z0-9][a-z0-9-]{1,39}$/.test(datasetId)) {
      fail(`evaluation.inspector.dataset.id ${datasetId}: invalid dataset id`);
    }
    const corpus = object(dataset.corpus, 'evaluation.inspector.dataset.corpus');
    const meta = object(corpus.meta, 'evaluation.inspector.dataset.corpus.meta');
    string(meta.language, 'evaluation.inspector.dataset.corpus.meta.language', true);
    const sessions = array(corpus.sessions, 'evaluation.inspector.dataset.corpus.sessions');
    if (!sessions.length) fail('evaluation.inspector.dataset.corpus.sessions: non-empty list required');
    const sessionById = unique(sessions, 'session_id', 'evaluation.inspector.dataset.corpus.sessions');
    sessionById.forEach((session, sessionId) => {
      const datePath = `evaluation.inspector.dataset.corpus.sessions.${sessionId}.date`;
      const date = string(session.date, datePath, true);
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) fail(`${datePath}: expected YYYY-MM-DD`);
      const messages = array(session.messages,
        `evaluation.inspector.dataset.corpus.sessions.${sessionId}.messages`);
      if (!messages.length) fail(`evaluation.inspector.dataset.corpus.sessions.${sessionId}.messages: non-empty list required`);
      messages.forEach((message, index) => {
        object(message, `evaluation.inspector.dataset.corpus.sessions.${sessionId}.messages[${index}]`);
        if (!['user', 'assistant'].includes(message.role)) {
          fail(`evaluation.inspector.dataset.corpus.sessions.${sessionId}.messages[${index}].role: invalid role`);
        }
        string(message.content,
          `evaluation.inspector.dataset.corpus.sessions.${sessionId}.messages[${index}].content`, true);
      });
    });
    const truth = object(dataset.ground_truth, 'evaluation.inspector.dataset.ground_truth');
    const questions = array(truth.questions,
      'evaluation.inspector.dataset.ground_truth.questions');
    if (!questions.length) fail('evaluation.inspector.dataset.ground_truth.questions: non-empty list required');
    if (questions.length > LIMITS.questions) {
      fail(`evaluation.inspector.dataset.ground_truth.questions: questions exceed ${LIMITS.questions}`);
    }
    const questionById = unique(questions, 'id',
      'evaluation.inspector.dataset.ground_truth.questions');
    questionById.forEach((question, questionId) => {
      const path = `evaluation.inspector.dataset.ground_truth.questions.${questionId}`;
      const questionText = question.question || question.question_fa;
      string(questionText, `${path}.question`, true);
      if (!TYPES.includes(question.type)) fail(`${path}.type: unknown question type ${question.type}`);
      if (!DIFFICULTIES.includes(question.difficulty)) fail(`${path}.difficulty: invalid difficulty`);
      const evidence = array(question.evidence || [], `${path}.evidence`);
      if (question.answerable !== false && !evidence.length) {
        fail(`${path}.evidence: an answerable question needs evidence`);
      }
      if (question.answerable !== false) {
        string(question.answer || question.answer_fa, `${path}.answer`, true);
      }
      evidence.forEach((item, index) => {
        const where = `${path}.evidence[${index}]`;
        object(item, where);
        const sessionId = string(item.session_id, `${where}.session_id`, true);
        const session = sessionById.get(sessionId);
        if (!session) fail(`${where}.session_id ${sessionId} is not archived`);
        const indices = array(item.message_indices, `${where}.message_indices`);
        if (!indices.length) fail(`${where}.message_indices: non-empty list required`);
        indices.forEach(messageIndex => {
          if (!Number.isInteger(messageIndex) || messageIndex < 0
              || messageIndex >= session.messages.length) {
            fail(`${where}.message_indices ${messageIndex} is outside session ${sessionId}`);
          }
        });
        const quote = string(item.quote, `${where}.quote`, true);
        const cited = indices.map(messageIndex => session.messages[messageIndex].content).join(' \n');
        if (!cited.includes(quote)) fail(`${where}.quote: text is not in cited messages`);
      });
    });
    return questionById;
  };

  const validateCatalogue = catalogue => {
    const found = new Set();
    array(catalogue, 'evaluation.metric_catalogue').forEach((row, index) => {
      const path = `evaluation.metric_catalogue[${index}]`;
      object(row, path);
      const key = metricKey(row.key, `${path}.key`);
      if (found.has(key)) fail(`evaluation.metric_catalogue: duplicate key ${key}`);
      found.add(key);
      const step = row.step || 'overall';
      if (!STAGES.includes(step)) fail(`${path}.step: unknown stage ${step}`);
      ['label', 'short', 'formula', 'library', 'help'].forEach(field => {
        if (Object.prototype.hasOwnProperty.call(row, field)) string(row[field], `${path}.${field}`);
      });
    });
  };

  const stageResults = (result, metricCatalogue) => {
    const groups = {
      retrieval: { metrics: {} }, generation: { metrics: {} },
      agent: { metrics: {} }, overall: { metrics: {} },
    };
    groups.index = { statistics: copy(result.index), metrics: {} };
    const steps = Object.create(null);
    metricCatalogue.forEach((row, index) => {
      const key = metricKey(row.key, `metric_catalogue[${index}].key`);
      const step = row.step || 'overall';
      if (!STAGES.includes(step)) {
        fail(`metric_catalogue[${index}].step: unknown stage ${step}`);
      }
      steps[key] = step;
    });
    const addValues = (source, path) => {
      object(source, path);
      Object.entries(source).forEach(([key, value]) => {
        metricKey(key, `${path}.${key}`);
        groups[steps[key] || 'overall'].metrics[key] = value;
      });
    };
    addValues(result.summary.overall || {}, 'result.summary.overall');
    addValues((result.ragas || {}).metrics || {}, 'result.ragas.metrics');
    return groups;
  };

  const validateCompleted = (evaluation, body) => {
    object(evaluation, 'evaluation');
    keys(evaluation, ['execution', 'metric_catalogue', 'stage_results', 'result', 'inspector'],
      'evaluation');
    const execution = object(evaluation.execution, 'evaluation.execution');
    string(execution.provider, 'evaluation.execution.provider');
    const models = object(execution.models, 'evaluation.execution.models');
    Object.entries(models).forEach(([key, value]) => string(value, `evaluation.execution.models.${key}`));
    validateCatalogue(evaluation.metric_catalogue);

    const result = object(evaluation.result, 'evaluation.result');
    keys(result, RESULT_KEYS, 'evaluation.result');
    string(result.run_id, 'evaluation.result.run_id', true);
    string(result.label, 'evaluation.result.label');
    string(result.dataset, 'evaluation.result.dataset', true);
    string(result.started_at, 'evaluation.result.started_at');
    object(result.config, 'evaluation.result.config');
    if (stable(result.config) !== stable(body.config)) {
      fail('evaluation.result.config: must equal settings.config');
    }
    object(result.index, 'evaluation.result.index');
    object(result.summary, 'evaluation.result.summary');
    object(result.ragas, 'evaluation.result.ragas');
    const rows = array(result.rows, 'evaluation.result.rows');
    array(result.notes, 'evaluation.result.notes').forEach((note, index) =>
      string(note, `evaluation.result.notes[${index}]`));
    number(result.seconds, 'evaluation.result.seconds');
    finite(result.summary, 'evaluation.result.summary');
    finite(rows, 'evaluation.result.rows');
    finite(result.ragas, 'evaluation.result.ragas');
    const selection = object(result.selection, 'evaluation.result.selection');

    const inspector = object(evaluation.inspector, 'evaluation.inspector');
    keys(inspector, ['dataset', 'chunks_by_session', 'summaries', 'traces'],
      'evaluation.inspector');
    const questionById = validateDataset(inspector.dataset);
    const chunks = [];
    array(inspector.chunks_by_session, 'evaluation.inspector.chunks_by_session')
      .forEach((group, index) => {
        object(group, `evaluation.inspector.chunks_by_session[${index}]`);
        chunks.push(...array(group.chunks,
          `evaluation.inspector.chunks_by_session[${index}].chunks`));
      });
    if (chunks.length > LIMITS.chunks) {
      fail(`evaluation.inspector.chunks_by_session: chunks exceed ${LIMITS.chunks}`);
    }
    const chunkById = unique(chunks, 'id', 'evaluation.inspector.chunks_by_session.chunks');
    chunkById.forEach((chunk, chunkId) => string(chunk.text,
      `evaluation.inspector.chunks_by_session.chunks.${chunkId}.text`));
    const summaries = array(inspector.summaries, 'evaluation.inspector.summaries');
    if (chunks.length + summaries.length > LIMITS.chunks) {
      fail(`evaluation.inspector: chunks exceed ${LIMITS.chunks}`);
    }
    const summaryById = unique(summaries, 'id', 'evaluation.inspector.summaries');
    summaryById.forEach((summary, summaryId) => string(summary.text,
      `evaluation.inspector.summaries.${summaryId}.text`));
    summaryById.forEach((_summary, summaryId) => {
      if (chunkById.has(summaryId)) fail(`evaluation.inspector: duplicate chunk id ${summaryId}`);
    });
    const sources = new Map([...chunkById, ...summaryById]);

    const rowById = unique(rows, 'id', 'evaluation.result.rows');
    Array.from(rowById.keys()).forEach((rowId, index) => {
      if (!questionById.has(rowId)) {
        fail(`evaluation.result.rows[${index}].id ${rowId} is outside archived ground truth`);
      }
    });
    const traces = array(inspector.traces, 'evaluation.inspector.traces');
    if (traces.length > LIMITS.traces) {
      fail(`evaluation.inspector.traces: traces exceed ${LIMITS.traces}`);
    }
    const traceById = unique(traces, 'question_id', 'evaluation.inspector.traces');
    Array.from(traceById.entries()).forEach(([questionId, traceRow], traceIndex) => {
      const tracePath = `evaluation.inspector.traces[${traceIndex}]`;
      if (!questionById.has(questionId)) {
        fail(`${tracePath}.question_id ${questionId} is outside archived ground truth`);
      }
      const trace = object(traceRow.trace, `${tracePath}.trace`);
      const candidates = array(trace.candidates, `${tracePath}.trace.candidates`);
      if (candidates.length > LIMITS.candidates_per_trace) {
        fail(`${tracePath}.trace.candidates: candidates exceed ${LIMITS.candidates_per_trace}`);
      }
      candidates.forEach((candidate, candidateIndex) => {
        const path = `${tracePath}.trace.candidates[${candidateIndex}]`;
        object(candidate, path);
        const chunkId = string(candidate.chunk_id, `${path}.chunk_id`, true);
        const source = sources.get(chunkId);
        if (!source) fail(`${path}.chunk_id ${chunkId} is not archived`);
        const text = string(candidate.text, `${path}.text`);
        if (text !== source.text) fail(`${path}.text: differs from archived chunk ${chunkId}`);
        array(candidate.gold_spans, `${path}.gold_spans`).forEach((item, index) =>
          span(item, text, `${path}.gold_spans[${index}]`));
        Object.entries(candidate).forEach(([key, value]) => {
          const folded = foldedKey(key);
          if (folded.endsWith('rank') || folded.endsWith('score')) {
            number(value, `${path}.${key}`, true);
          }
        });
        finite(candidate, path);
      });
    });

    const questionIds = ids(selection.question_ids, 'evaluation.result.selection.question_ids');
    const rowIds = Array.from(rowById.keys());
    const traceIds = Array.from(traceById.keys());
    if (stable(questionIds) !== stable(rowIds)) {
      fail(`evaluation.result.rows ids ${JSON.stringify(rowIds)} must equal ordered selection.question_ids ${JSON.stringify(questionIds)}`);
    }
    if (stable(questionIds) !== stable(traceIds)) {
      fail(`evaluation.inspector.traces question ids ${JSON.stringify(traceIds)} must equal ordered selection.question_ids ${JSON.stringify(questionIds)}`);
    }
    const count = questionIds.length;
    if (integer(selection.n, 'evaluation.result.selection.n', 0) !== count) {
      fail(`evaluation.result.selection.n: expected ${count}`);
    }
    if (integer(result.summary.n_questions, 'evaluation.result.summary.n_questions', 0) !== count) {
      fail(`evaluation.result.summary.n_questions: expected ${count}`);
    }
    const datasetId = body.config.index.dataset || 'diary-fa';
    if (result.dataset !== datasetId) {
      fail(`evaluation.result.dataset ${result.dataset}: expected settings dataset ${datasetId}`);
    }
    if (inspector.dataset.id !== datasetId) {
      fail(`evaluation.inspector.dataset.id ${inspector.dataset.id}: expected settings dataset ${datasetId}`);
    }
    if (stable(evaluation.stage_results) !== stable(stageResults(result, evaluation.metric_catalogue))) {
      fail('evaluation.stage_results: must equal the derived metric projection');
    }
  };

  function normalize(source) {
    object(source, 'archive');
    walk(source);
    const value = copy(source);
    keys(value, Object.prototype.hasOwnProperty.call(value, 'evaluation')
      ? ['format', 'version', 'settings', 'evaluation']
      : ['format', 'version', 'settings'], 'archive');
    if (value.format !== FORMAT) {
      fail(`archive.format ${JSON.stringify(value.format)}: expected ${JSON.stringify(FORMAT)}`);
    }
    if (value.version !== VERSION) {
      fail(`archive.version ${JSON.stringify(value.version)}: expected version ${VERSION}`);
    }
    object(value.settings, 'settings');
    if (value.settings.ui && Array.isArray(value.settings.ui.types)) {
      value.settings.ui.types = Array.from(new Set(value.settings.ui.types)).sort();
    }
    validateSettings(value.settings);
    if (Object.prototype.hasOwnProperty.call(value, 'evaluation')) {
      validateCompleted(value.evaluation, value.settings);
    }
    return value;
  }

  const pickResult = result => {
    object(result, 'result');
    const picked = {};
    RESULT_KEYS.forEach(key => {
      if (!Object.prototype.hasOwnProperty.call(result, key)) {
        fail(`result.${key}: required`);
      }
      picked[key] = copy(result[key]);
    });
    return picked;
  };
  const settings = (config, ui) => normalize({
    format: FORMAT, version: VERSION,
    settings: { config: copy(config), ui: copy(ui) },
  });
  const completed = (config, ui, result, evidence) => {
    const value = settings(config, ui);
    value.evaluation = {
      execution: copy(evidence.execution),
      metric_catalogue: copy(evidence.metric_catalogue),
      result: pickResult(result),
      inspector: copy(evidence.inspector),
    };
    value.evaluation.stage_results = stageResults(
      value.evaluation.result, value.evaluation.metric_catalogue);
    return normalize(value);
  };
  const parse = source => normalize(JSON.parse(source));
  const stringify = value => `${JSON.stringify(normalize(value), null, 2)}\n`;
  const stableValue = value => {
    if (Array.isArray(value)) return value.map(stableValue);
    if (value !== null && typeof value === 'object') {
      const out = Object.create(null);
      Object.keys(value).sort().forEach(key => { out[key] = stableValue(value[key]); });
      return out;
    }
    return value;
  };
  function stable(value) { return JSON.stringify(stableValue(value)); }
  const equal = (left, right) => stable(left) === stable(right);
  const assertFileSize = bytes => {
    if (!Number.isInteger(bytes) || bytes < 0 || bytes > MAX_BYTES) {
      throw new Error('Archive must not exceed 32 MiB');
    }
  };
  const datasetDisposition = (value, servedIds) => {
    const dataset = value.settings.config.index.dataset || 'diary-fa';
    const viewOnly = !servedIds.includes(dataset);
    if (viewOnly && !value.evaluation) {
      throw new Error(`settings.config.index.dataset ${dataset} is not available`);
    }
    return { dataset, viewOnly };
  };
  const transact = (next, adapter) => {
    const before = copy(adapter.read());
    adapter.validate(next);
    try { adapter.write(copy(next)); }
    catch (error) { adapter.write(before); throw error; }
    return next;
  };
  return Object.freeze({ FORMAT, VERSION, MAX_BYTES, settings, completed,
    normalize, parse, stringify, equal, assertFileSize, stageResults,
    datasetDisposition, transact });
})();
