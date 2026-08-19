// Browser contracts for the portable experiment archive codec.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE = readFileSync(join(HERE, '../frontend/archive_io.js'), 'utf8');
const ArchiveIO = runInNewContext(SOURCE + '\n;ArchiveIO', {});
const plain = (value) => JSON.parse(JSON.stringify(value));

const CONFIG = {
  label: 'imported experiment',
  index: {
    dataset: 'smoke-mini', chunker: 'session', chunk_chars: 500, overlap: 100,
    contextual: true, embedder: 'token-hash', embed_model: '', hierarchy: '',
    graph_source: 'hybrid', graph_knn: 8, granularity: 1, hierarchy_levels: 1,
    min_group: 3, summarizer: 'centroid',
  },
  retrieval: {
    retriever: 'hybrid-rrf', k: 8, candidates: 40, rrf_k: 60,
    time_filter: true, multi_query: true, hyde: false,
    expansion_model: 'expand-model', mmr_lambda: 1, reranker: 'lexical',
    rerank_depth: 20, reranker_model: '', recency_half_life_days: 180,
    agentic_weights: [1, 0.3, 0.2], grader: 'none', grade_threshold: 0,
    grader_model: '', max_context_chars: 6000, summary_scope: 'mixed',
    summary_boost: 1, summary_levels: '',
  },
  generation: {
    answerer: 'extractive', model: 'answer-model', key_facts_judge: false,
    judge_model: '', ragas_model: '',
  },
  agent: {
    scope: '', max_hops: 3, rewrite: true, evidence_threshold: 0.5,
    max_revisions: 1, critic: 'grounded', max_llm_calls: 12,
    plan_model: 'plan-model', critic_model: 'critic-model',
  },
};
const UI = { mode: '', ragas_mode: 'offline', limit: 1, ragas_limit: 0,
  types: ['single-hop'] };
const INDEX = { collection: 'raglab-test', chunks: 1, leaves: 1,
  avg_chars: 8, p95_chars: 8, embed_dim: 8, build_seconds: 0, reused: false };
const METRICS = [{ key: 'recall', label: 'Recall@k', short: 'evidence found',
  step: 'retrieval', formula: 'gold found / gold', library: 'test fixture',
  help: 'retrieval coverage' }];
const RESULT = {
  run_id: 'imported-run-001', label: 'imported experiment', config: CONFIG,
  dataset: 'smoke-mini', index: INDEX,
  summary: { overall: { recall: 1 },
    by_type: { 'single-hop': { n: 1, recall: 1 } },
    by_difficulty: { easy: { n: 1, recall: 1 } }, n_questions: 1 },
  rows: [{ id: 'q1', type: 'single-hop', difficulty: 'easy', recall: 1,
    latency_ms: 12, n_contexts: 1, abstained: false }],
  ragas: {}, seconds: 0.2, started_at: '2026-08-19 12:00:00', notes: [],
  selection: { balance: 'stride', limit: 1, n: 1,
    by_difficulty: { easy: 1 }, question_ids: ['q1'] },
};
const EVIDENCE = {
  execution: { provider: 'fake', models: {} }, metric_catalogue: METRICS,
  inspector: {
    dataset: { id: 'smoke-mini',
      corpus: { meta: { language: 'en' }, persona: {}, threads: [], habits: {},
        sessions: [{ session_id: 's1', date: '2026-08-19',
          messages: [{ role: 'user', content: 'evidence' }] }] },
      ground_truth: { meta: { corpus: 'smoke-mini', query_date: '2026-08-19' },
        questions: [{ id: 'q1', type: 'single-hop', difficulty: 'easy',
          answerable: true, question_fa: 'question', question_en: 'question',
          answer_fa: 'answer', key_facts: ['fact'],
          evidence: [{ session_id: 's1', message_indices: [0], quote: 'evidence' }] }] } },
    chunks_by_session: [{ session_id: 's1', date: '2026-08-19',
      chunks: [{ id: 'c1', text: 'evidence' }] }],
    summaries: [],
    traces: [{ question_id: 'q1', question_fa: 'question', question_en: 'question',
      type: 'single-hop', difficulty: 'easy', answerable: true, gold_available: 1,
      trace: { candidates: [{ chunk_id: 'c1', text: 'evidence', session_id: 's1',
        date: '2026-08-19', layer: 'leaf', level: 0, group_id: '', members: 0,
        dense_rank: 1, bm25_rank: 1, fused_rank: 1, retrieval_score: 1,
        rerank_score: 1, grade_score: null, kept: true, gold: true,
        gold_spans: [[0, 8]] }] } }],
  },
};
const FULL = ArchiveIO.completed(CONFIG, UI, RESULT, EVIDENCE);

test('settings-only archives omit evaluation and round-trip', () => {
  const value = ArchiveIO.settings(CONFIG, UI);
  assert.equal(value.format, 'raglab-experiment');
  assert.equal(value.version, 1);
  assert.equal('evaluation' in value, false);
  assert.deepEqual(plain(ArchiveIO.parse(ArchiveIO.stringify(value))), plain(value));
  assert.equal(ArchiveIO.MAX_BYTES, 32 * 1024 * 1024);
});

test('settings capture keeps hidden config, every model role, and all UI controls', () => {
  const value = ArchiveIO.settings(CONFIG, {
    mode: 'openrouter', ragas_mode: 'llm', limit: 7,
    ragas_limit: 3, types: ['multi-hop', 'single-hop'],
  });
  assert.deepEqual(plain(value.settings.config), plain(CONFIG));
  assert.equal(value.settings.config.retrieval.expansion_model, 'expand-model');
  assert.equal(value.settings.config.agent.critic_model, 'critic-model');
  assert.deepEqual(plain(value.settings.ui), {
    mode: 'openrouter', ragas_mode: 'llm', limit: 7, ragas_limit: 3,
    types: ['multi-hop', 'single-hop'],
  });
});

test('completed archives preserve every Inspector section and derive stages', () => {
  const value = ArchiveIO.completed(CONFIG, UI, RESULT, EVIDENCE);
  assert.deepEqual(Object.keys(value.evaluation.inspector).sort(),
    ['chunks_by_session', 'dataset', 'summaries', 'traces']);
  assert.deepEqual(plain(value.evaluation.stage_results.retrieval.metrics),
    { recall: 1 });
  assert.equal(value.evaluation.result.rows[0].id, 'q1');
});

test('credentials, bad references, wrong versions, and oversized files fail', () => {
  const secret = plain(FULL);
  secret.evaluation.inspector.dataset.corpus.api_key = 'no';
  assert.throws(() => ArchiveIO.normalize(secret), /api_key/);
  const missing = plain(FULL);
  missing.evaluation.inspector.traces[0].trace.candidates[0].chunk_id = 'missing';
  assert.throws(() => ArchiveIO.normalize(missing), /chunk_id.*missing/);
  assert.throws(() => ArchiveIO.parse(JSON.stringify({ ...FULL, version: 2 })),
    /version 2.*version 1/i);
  assert.doesNotThrow(() => ArchiveIO.assertFileSize(ArchiveIO.MAX_BYTES));
  assert.throws(() => ArchiveIO.assertFileSize(ArchiveIO.MAX_BYTES + 1), /32 MiB/);
});

test('malicious labels remain data and never become archive markup', () => {
  const value = plain(FULL);
  value.settings.config.label = '<img src=x onerror=alert(1)>';
  value.evaluation.result.config.label = '<img src=x onerror=alert(1)>';
  value.evaluation.result.label = '<img src=x onerror=alert(1)>';
  const normalized = ArchiveIO.normalize(value);
  assert.equal(normalized.evaluation.result.label,
    '<img src=x onerror=alert(1)>');
  assert.equal(typeof normalized.evaluation.result.label, 'string');
});

test('transaction restores the previous dashboard state after a partial write', () => {
  let state = ArchiveIO.settings(CONFIG, UI);
  const before = state;
  assert.throws(() => ArchiveIO.transact(FULL, {
    read: () => state,
    validate: () => {},
    write: (next) => { state = next; if (next.evaluation) throw new Error('stopped'); },
  }), /stopped/);
  assert.ok(ArchiveIO.equal(state, before));
});

test('only a completed archive may preview an unavailable dataset', () => {
  assert.deepEqual(plain(ArchiveIO.datasetDisposition(FULL, ['diary-fa'])),
    { dataset: 'smoke-mini', viewOnly: true });
  const settingsOnly = ArchiveIO.settings(CONFIG, UI);
  assert.throws(() => ArchiveIO.datasetDisposition(settingsOnly, ['diary-fa']),
    /settings.config.index.dataset.*smoke-mini.*not available/i);
  assert.deepEqual(plain(ArchiveIO.datasetDisposition(FULL, ['smoke-mini'])),
    { dataset: 'smoke-mini', viewOnly: false });
});
