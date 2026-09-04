// Portable experiment archive codec. Deliberately has no DOM or network access.
const ArchiveIO = (() => {
  const FORMAT = 'raglab-experiment';
  const VERSION = 1;
  const MAX_BYTES = 32 * 1024 * 1024;
  // D3: `IndexConfig.dataset=''` still *means* the built-in diary, and is
  // still dropped from the fingerprint payload so every collection name
  // already recorded stays reproducible — the one dataset id this codec (and
  // `ExperimentHandoff.reconcile`, which reads this same constant) ever
  // fills in for an absent one, stated once rather than typed at each site.
  const BUILTIN_DATASET = 'diary-fa';
  const LIMITS = Object.freeze({
    depth: 32, questions: 5000, chunks: 50000, traces: 5000,
    candidates_per_trace: 1000, list_items: 100000, string_chars: 2000000,
  });
  const EVIDENCE_FIDELITIES = Object.freeze(['verbatim', 'paraphrase', 'computed']);
  const BEHAVIORS = Object.freeze(['answer', 'abstain', 'correct_premise']);
  const STAGES = Object.freeze(['index', 'retrieval', 'generation', 'overall']);
  const UNSAFE_METRIC_KEYS = Object.freeze(['__proto__', 'prototype', 'constructor']);
  const RESULT_KEYS = Object.freeze([
    'run_id', 'label', 'config', 'dataset', 'index', 'summary', 'rows',
    'ragas', 'seconds', 'started_at', 'notes', 'selection',
  ]);
  const CONFIG_TEMPLATE = Object.freeze({
    index: {
      // The plan's stages are objects of several shapes, which a template of
      // one cannot say; the list is checked here and each stage by `plan`.
      dataset: '', split_plan: [], chunk_chars: 500, chunk_unit: 'characters',
      overlap: 0, part_join: '\n', part_prefix: '', normalizer: '',
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
      answerer: 'extractive', model: '', fact_judge: false,
      judge_model: '', ragas_model: '',
    },
    label: '',
  });
  const VOCABULARIES = Object.freeze({
    'index.chunk_unit': ['characters', 'tokens'],
    'index.normalizer': ['', 'persian', 'neutral'],
    'index.embedder': ['sentence-transformers', 'fastembed', 'ascii-hash', 'token-hash', 'char-hash'],
    'index.hierarchy': ['', 'louvain', 'leiden', 'label-prop', 'raptor', 'agglomerative', 'kmeans', 'metadata'],
    'index.graph_source': ['hybrid', 'knn', 'lexical', 'bipartite-terms'],
    'index.summarizer': ['centroid', 'lead-idf', 'mmr', 'card'],
    'retrieval.retriever': ['hybrid-rrf', 'dense', 'bm25'],
    'retrieval.reranker': ['lexical', 'none', 'recency', 'agentic', 'cross-encoder', 'llm'],
    'retrieval.grader': ['none', 'lexical', 'llm'],
    'retrieval.summary_scope': ['mixed', 'leaves', 'summaries', 'drill-down'],
    'generation.answerer': ['extractive', 'none', 'llm'],
  });
  const MODEL_FIELDS = Object.freeze([
    ['index', 'embed_model'],
    ['retrieval', 'expansion_model'], ['retrieval', 'reranker_model'],
    ['retrieval', 'grader_model'], ['generation', 'model'],
    ['generation', 'judge_model'], ['generation', 'ragas_model'],
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

  const STAGE_KINDS = Object.freeze(['document', 'part', 'label', 'separator', 'drift']);
  // The plan's shape: the document first, every stage a known kind with its
  // lists as lists. Which stages may follow which, and whether a label is one
  // the corpus declares, is the server's `validate()` — one rule, one place.
  const plan = (stages, path) => {
    array(stages, path);
    if (!stages.length || !stages[0] || stages[0].kind !== 'document') {
      fail(`${path}: must begin with the document stage`);
    }
    stages.forEach((stage, index) => {
      object(stage, `${path}[${index}]`);
      if (!STAGE_KINDS.includes(stage.kind)) {
        fail(`${path}[${index}].kind: unsupported stage ${JSON.stringify(stage.kind)}`);
      }
      if ('atoms' in stage) array(stage.atoms, `${path}[${index}].atoms`);
      if ('markers' in stage) array(stage.markers, `${path}[${index}].markers`);
    });
  };

  const validateConfig = config => {
    shape(config, CONFIG_TEMPLATE, 'settings.config');
    plan(config.index.split_plan, 'settings.config.index.split_plan');
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
  };

  const validateSettings = body => {
    object(body, 'settings');
    keys(body, ['config', 'ui'], 'settings');
    object(body.config, 'settings.config');
    validateConfig(body.config);
    const ui = object(body.ui, 'settings.ui');
    keys(ui, ['mode', 'ragas_mode', 'limit', 'ragas_limit', 'labels', 'balance'],
      'settings.ui');
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
    string(ui.balance, 'settings.ui.balance');
    // D7: a question filter is one switch-group per label the *dataset*
    // declares — an open vocabulary, so there is no fixed list to check
    // `labels`/`balance` against here. The panel checks them against the
    // dataset the config names, which is where that vocabulary lives.
    const labels = object(ui.labels, 'settings.ui.labels');
    Object.entries(labels).forEach(([label, values]) => {
      array(values, `settings.ui.labels.${label}`).forEach((value, index) => {
        string(value, `settings.ui.labels.${label}[${index}]`, true);
      });
    });
  };

  // `isInt` is true for a question/row/trace id — the schema's own
  // `groundtruth_question_id`, an integer — and false for a chunk/summary id,
  // an arbitrary string assigned at index time and unrelated to the schema.
  const unique = (rows, key, path, isInt = false) => {
    const found = new Map();
    array(rows, path).forEach((row, index) => {
      object(row, `${path}[${index}]`);
      const value = isInt
        ? integer(row[key], `${path}[${index}].${key}`)
        : string(row[key], `${path}[${index}].${key}`, true);
      if (found.has(value)) fail(`${path}: duplicate ${key} ${value}`);
      found.set(value, row);
    });
    return found;
  };
  const ids = (values, path, isInt = false) => {
    const found = new Set();
    array(values, path).forEach((value, index) => {
      if (isInt) integer(value, `${path}[${index}]`);
      else string(value, `${path}[${index}]`, true);
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

  // The corpus/ground-truth pair, held to the schema's own contract (D4/D9):
  // the pair join, every cited document real, every verbatim quote findable
  // character for character. This is a hand-written mirror of the essential
  // cross-file rules `dataset_import_contract.validate()` runs server-side —
  // the browser has no `jsonschema` and cannot load the JSON Schema files, so
  // it cannot re-run the *full* structural contract (label vocabularies,
  // confidence/ranks legality); those were already enforced once, at import,
  // and re-deciding them here would be a second validator drifting from the
  // first rather than following it.
  const validateDataset = dataset => {
    object(dataset, 'evaluation.inspector.dataset');
    keys(dataset, ['id', 'corpus', 'ground_truth'], 'evaluation.inspector.dataset');
    const datasetId = string(dataset.id, 'evaluation.inspector.dataset.id', true);
    if (!/^[a-z0-9][a-z0-9-]{1,39}$/.test(datasetId)) {
      fail(`evaluation.inspector.dataset.id ${datasetId}: invalid dataset id`);
    }
    const corpus = object(dataset.corpus, 'evaluation.inspector.dataset.corpus');
    const corpusMeta = object(corpus.corpus_dataset_metadata,
      'evaluation.inspector.dataset.corpus.corpus_dataset_metadata');
    string(corpusMeta.dataset,
      'evaluation.inspector.dataset.corpus.corpus_dataset_metadata.dataset', true);
    string(corpusMeta.name,
      'evaluation.inspector.dataset.corpus.corpus_dataset_metadata.name', true);
    string(corpusMeta.language,
      'evaluation.inspector.dataset.corpus.corpus_dataset_metadata.language', true);
    if (corpusMeta.dataset !== datasetId) {
      fail('evaluation.inspector.dataset.corpus.corpus_dataset_metadata.dataset '
        + `${JSON.stringify(corpusMeta.dataset)}: expected archived dataset `
        + `${JSON.stringify(datasetId)}`);
    }
    const documents = array(corpus.corpus_documents,
      'evaluation.inspector.dataset.corpus.corpus_documents');
    if (!documents.length) fail('evaluation.inspector.dataset.corpus.corpus_documents: non-empty list required');
    const documentById = unique(documents, 'corpus_document_id',
      'evaluation.inspector.dataset.corpus.corpus_documents', true);
    const documentText = new Map();
    documentById.forEach((document, documentId) => {
      const path = `evaluation.inspector.dataset.corpus.corpus_documents.${documentId}`;
      const parts = array(document.document_content, `${path}.document_content`);
      if (!parts.length) fail(`${path}.document_content: non-empty list required`);
      const texts = parts.map((part, index) => string(part.text, `${path}.document_content[${index}].text`, true));
      documentText.set(documentId, texts.join(' \n'));
    });

    const truth = object(dataset.ground_truth, 'evaluation.inspector.dataset.ground_truth');
    const truthMeta = object(truth.groundtruth_dataset_metadata,
      'evaluation.inspector.dataset.ground_truth.groundtruth_dataset_metadata');
    string(truthMeta.name,
      'evaluation.inspector.dataset.ground_truth.groundtruth_dataset_metadata.name', true);
    const corpusRef = object(truthMeta.corpus_ref,
      'evaluation.inspector.dataset.ground_truth.groundtruth_dataset_metadata.corpus_ref');
    if (corpusRef.dataset !== corpusMeta.dataset) {
      fail(`${JSON.stringify(corpusRef.dataset)}: corpus_ref.dataset does not `
        + `match the corpus it is paired with (${JSON.stringify(corpusMeta.dataset)})`);
    }
    const questions = array(truth.groundtruth_dataset,
      'evaluation.inspector.dataset.ground_truth.groundtruth_dataset');
    if (!questions.length) fail('evaluation.inspector.dataset.ground_truth.groundtruth_dataset: non-empty list required');
    if (questions.length > LIMITS.questions) {
      fail(`evaluation.inspector.dataset.ground_truth.groundtruth_dataset: questions exceed ${LIMITS.questions}`);
    }
    const questionById = unique(questions, 'groundtruth_question_id',
      'evaluation.inspector.dataset.ground_truth.groundtruth_dataset', true);
    questionById.forEach((question, questionId) => {
      const path = `evaluation.inspector.dataset.ground_truth.groundtruth_dataset.${questionId}`;
      string(question.question, `${path}.question`, true);
      const expected = object(question.expected_answer, `${path}.expected_answer`);
      const behavior = expected.behavior;
      if (!BEHAVIORS.includes(behavior)) fail(`${path}.expected_answer.behavior: unknown behavior ${behavior}`);
      if (behavior !== 'abstain') string(expected.text, `${path}.expected_answer.text`, true);
      const relevant = array(question.relevant_corpus_documents || [], `${path}.relevant_corpus_documents`);
      relevant.forEach((entry, index) => {
        const where = `${path}.relevant_corpus_documents[${index}]`;
        object(entry, where);
        const documentId = entry.corpus_document_id;
        if (!documentById.has(documentId)) {
          fail(`${where}.corpus_document_id ${documentId}: cites a document not in this corpus`);
        }
        const evidence = array(entry.evidence, `${where}.evidence`);
        if (!evidence.length) fail(`${where}.evidence: non-empty list required`);
        evidence.forEach((item, evidenceIndex) => {
          const evidencePath = `${where}.evidence[${evidenceIndex}]`;
          object(item, evidencePath);
          const text = string(item.text, `${evidencePath}.text`, true);
          const fidelity = item.fidelity;
          if (!EVIDENCE_FIDELITIES.includes(fidelity)) {
            fail(`${evidencePath}.fidelity: unknown fidelity ${fidelity}`);
          }
          if (fidelity === 'verbatim' && !documentText.get(documentId).includes(text)) {
            fail(`${evidencePath}.text: not findable character for character `
              + `in document ${documentId} — a verbatim quote must appear exactly`);
          }
        });
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
      overall: { metrics: {} },
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

    const rowById = unique(rows, 'id', 'evaluation.result.rows', true);
    Array.from(rowById.keys()).forEach((rowId, index) => {
      if (!questionById.has(rowId)) {
        fail(`evaluation.result.rows[${index}].id ${rowId} is outside archived ground truth`);
      }
    });
    const traces = array(inspector.traces, 'evaluation.inspector.traces');
    if (traces.length > LIMITS.traces) {
      fail(`evaluation.inspector.traces: traces exceed ${LIMITS.traces}`);
    }
    const traceById = unique(traces, 'question_id', 'evaluation.inspector.traces', true);
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

    const questionIds = ids(selection.question_ids, 'evaluation.result.selection.question_ids', true);
    const rowIds = Array.from(rowById.keys());
    const traceIds = Array.from(traceById.keys());
    if (stable(questionIds) !== stable(rowIds)) {
      fail(`evaluation.result.rows ids ${JSON.stringify(rowIds)} must equal ordered selection.question_ids ${JSON.stringify(questionIds)}`);
    }
    // Rows are the measurement; a trace is a recording of how retrieval reached
    // one. Evidence may legitimately be absent — an evaluation recorded before
    // the export route existed kept its rows, its judged metrics and its
    // selection, but never stored a trace or a chunk — while a measurement may
    // never be invented, which is why `questionIds === rowIds` above stays
    // exact. So the traces are a *subset* of the rows: an archive with rows and
    // no traces reads "scored, trace not retained", which is the truth about
    // those runs. The relaxation goes no further than absence. A trace that is
    // present is held to everything it was held to before — its question is one
    // the run selected, every candidate resolves to an archived chunk or
    // summary and is byte-equal to it (checked above) — and the traces that are
    // present must still run in the selection's order, so a trace list cannot
    // silently reorder the run it recorded. The Python codec's
    // `_validate_completed` reads it exactly this way; the two must not drift.
    const traced = new Set(traceIds);
    const rowIdSet = new Set(rowIds);
    const outside = traceIds.filter((traceId) => !rowIdSet.has(traceId));
    if (outside.length) {
      fail(`evaluation.inspector.traces question ids ${JSON.stringify(outside)} are outside ordered selection.question_ids ${JSON.stringify(questionIds)}`);
    }
    if (stable(traceIds) !== stable(questionIds.filter((questionId) => traced.has(questionId)))) {
      fail(`evaluation.inspector.traces question ids ${JSON.stringify(traceIds)} must follow ordered selection.question_ids ${JSON.stringify(questionIds)}`);
    }
    const count = questionIds.length;
    if (integer(selection.n, 'evaluation.result.selection.n', 0) !== count) {
      fail(`evaluation.result.selection.n: expected ${count}`);
    }
    if (integer(result.summary.n_questions, 'evaluation.result.summary.n_questions', 0) !== count) {
      fail(`evaluation.result.summary.n_questions: expected ${count}`);
    }
    const datasetId = body.config.index.dataset || BUILTIN_DATASET;
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
    // Canonical order before the equality check `ArchiveIO.equal` runs
    // elsewhere: object keys sort themselves there, but an array's order is
    // part of the comparison, and which checkbox a browser reports first
    // among several checked ones is not a fact this archive should carry.
    if (value.settings.ui && value.settings.ui.labels
        && typeof value.settings.ui.labels === 'object') {
      const labels = value.settings.ui.labels;
      Object.keys(labels).forEach(name => {
        if (Array.isArray(labels[name])) {
          labels[name] = Array.from(new Set(labels[name])).sort();
        }
      });
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
    const dataset = value.settings.config.index.dataset || BUILTIN_DATASET;
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
  return Object.freeze({ FORMAT, VERSION, MAX_BYTES, BUILTIN_DATASET, settings,
    completed, normalize, parse, stringify, equal, assertFileSize, stageResults,
    datasetDisposition, transact });
})();
