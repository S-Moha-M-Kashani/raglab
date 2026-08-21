// Browser contracts for the leaderboard → Laboratory handoff.
//
// The board's open button pins the Inspector to one recorded experiment and
// hands the same experiment to the Laboratory, whose knobs are then that
// experiment's. Two halves live here, and neither touches the DOM:
//
//   the slot   — what the board writes and the panel takes, once
//   reconcile  — which of a recorded config's knobs this installation can
//                actually serve, and what to say about the rest
//
// `reconcile` is the half worth testing hardest. A knob this lab cannot serve
// must never land on the panel: a select with no matching option reads back as
// '' and the lab would then run a config that is neither the experiment's nor
// the reader's. So the rule is applied-what-it-can and *named* the rest — and
// the naming is the contract, not a nicety, because it is the only thing
// standing between "these are experiment X's settings" and a quiet lie.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE = readFileSync(
  join(HERE, '../frontend/experiment_handoff.js'), 'utf8');
const Handoff = runInNewContext(SOURCE + '\n;ExperimentHandoff', {});

// The smallest fake of `localStorage` these two functions actually touch.
function store(initial = {}) {
  const kept = Object.assign({}, initial);
  return {
    getItem: (key) => (key in kept ? kept[key] : null),
    setItem: (key, value) => { kept[key] = String(value); },
    removeItem: (key) => { delete kept[key]; },
    all: () => kept,
  };
}

// What the panel currently has on screen: every knob at some value, because
// that is what an unserved knob has to be left at.
const CURRENT = {
  label: 'what the reader was doing',
  index: {
    dataset: 'diary-fa', chunker: 'semantic-drift', chunk_chars: 500,
    overlap: 100, contextual: true, embedder: 'token-hash', embed_model: '',
    hierarchy: '', graph_source: 'hybrid', graph_knn: 8, granularity: 1,
    hierarchy_levels: 1, min_group: 3, summarizer: 'centroid',
  },
  retrieval: {
    retriever: 'hybrid-rrf', k: 8, candidates: 40, time_filter: true,
    multi_query: true, hyde: false, mmr_lambda: 1, reranker: 'lexical',
    rerank_depth: 20, recency_half_life_days: 180, grader: 'none',
    grade_threshold: 0, summary_scope: 'mixed', summary_boost: 1,
    summary_levels: '',
  },
  generation: { answerer: 'extractive', model: '', key_facts_judge: false },
};

// What this installation serves, in the shape the panel assembles from
// /api/options and its own numeric controls.
const SERVED = {
  mode: 'openrouter',
  datasets: ['diary-fa', 'smoke-mini'],
  choices: {
    'index.chunker': ['semantic-drift', 'session', 'fixed'],
    'index.embedder': ['token-hash', 'sentence-transformers'],
    'index.hierarchy': ['', 'louvain'],
    'index.graph_source': ['hybrid', 'knn'],
    'index.summarizer': ['centroid', 'mmr'],
    'retrieval.retriever': ['hybrid-rrf', 'dense', 'bm25'],
    'retrieval.reranker': ['lexical', 'none'],
    'retrieval.grader': ['none', 'lexical'],
    'retrieval.summary_scope': ['mixed', 'leaves'],
    'generation.answerer': ['extractive', 'llm'],
  },
  // A catalogue carries the reason it would refuse, because the two catalogues
  // refuse for different reasons: the chat models are what *this backend mode*
  // offers, while the embedding models are what is installed here at all, and a
  // reader told "not served in openrouter mode" about a sentence-transformer
  // would go and change the mode.
  models: {
    'index.embed_model': { ids: ['bge-small'], reason: 'not served by this lab' },
    'generation.model': {
      ids: ['sonnet', 'haiku'], reason: 'not served in openrouter mode',
    },
  },
  ranges: {
    'index.chunk_chars': { min: 100, max: 4000 },
    'retrieval.k': { min: 1, max: 50 },
  },
};

const copy = (value) => JSON.parse(JSON.stringify(value));
// The module is evaluated in a `vm` context, so the arrays and objects it
// returns carry that realm's prototypes and are never reference-equal to this
// file's. `archive_io.test.js` crosses the same seam the same way.
const plain = (value) => JSON.parse(JSON.stringify(value));

// --- the slot ---------------------------------------------------------------

test('the board offers an experiment and the panel takes it exactly once', () => {
  const kept = store();
  Handoff.offer(kept, 'exp-1', 1000);
  assert.deepEqual(plain(Handoff.taken(kept)),
    { experiment_id: 'exp-1', at: 1000 });
  assert.equal(Handoff.taken(kept), null,
    'taking the slot clears it: a reload must not re-announce the experiment');
});

test('offering the same experiment twice writes a different value each time', () => {
  // `storage` events fire on a *change*, so a slot that read identically on the
  // second click would leave an already-open Laboratory tab silent — which is
  // the case this whole handoff exists for.
  // Twice in the same millisecond, which is what a double-click is: a clock is
  // not fine-grained enough to be the difference on its own.
  const kept = store();
  Handoff.offer(kept, 'exp-1', 1000);
  const first = kept.all()[Handoff.KEY];
  Handoff.offer(kept, 'exp-1', 1000);
  assert.notEqual(kept.all()[Handoff.KEY], first);
  assert.equal(Handoff.taken(kept).experiment_id, 'exp-1',
    'and it is still the experiment that was clicked');
});

test('an empty or hand-edited slot is nothing to take, not a broken panel', () => {
  assert.equal(Handoff.taken(store()), null);
  assert.equal(Handoff.taken(store({ [Handoff.KEY]: '{not json' })), null);
  assert.equal(Handoff.taken(store({ [Handoff.KEY]: '{"at":7}' })), null,
    'a slot with no experiment id names no experiment');
});

// --- reconcile --------------------------------------------------------------

test('a config this lab serves entirely arrives entirely, with nothing to report', () => {
  const recorded = copy(CURRENT);
  recorded.index.chunker = 'session';
  recorded.retrieval.k = 12;
  recorded.generation.answerer = 'llm';
  const out = Handoff.reconcile(recorded, CURRENT, SERVED);
  assert.deepEqual(plain(out.unserved), []);
  assert.equal(out.config.index.chunker, 'session');
  assert.equal(out.config.retrieval.k, 12);
  assert.equal(out.config.generation.answerer, 'llm');
});

test('a knob this lab does not serve stays where it was and is named', () => {
  const recorded = copy(CURRENT);
  recorded.index.chunker = 'turn-pair';
  const out = Handoff.reconcile(recorded, CURRENT, SERVED);
  assert.equal(out.config.index.chunker, 'semantic-drift',
    'the unserved value must not reach a select that has no such option');
  assert.deepEqual(plain(out.unserved), [{
    path: 'index.chunker', value: 'turn-pair', reason: 'not served by this lab',
  }]);
});

test('a corpus this installation no longer has is named as uninstalled', () => {
  const recorded = copy(CURRENT);
  recorded.index.dataset = 'gone-fa';
  const out = Handoff.reconcile(recorded, CURRENT, SERVED);
  assert.equal(out.config.index.dataset, 'diary-fa');
  assert.deepEqual(plain(out.unserved), [{
    path: 'index.dataset', value: 'gone-fa', reason: 'not installed here',
  }]);
});

test('a model the current backend mode does not offer says which mode', () => {
  const recorded = copy(CURRENT);
  recorded.generation.model = 'gpt-4o';
  const out = Handoff.reconcile(recorded, CURRENT, SERVED);
  assert.equal(out.config.generation.model, '');
  assert.deepEqual(plain(out.unserved), [{
    path: 'generation.model', value: 'gpt-4o',
    reason: 'not served in openrouter mode',
  }]);
});

test('an embedding model that is not installed does not blame the backend mode', () => {
  // Two catalogues, two reasons. A reader told the wrong one goes and changes
  // the wrong thing.
  const recorded = copy(CURRENT);
  recorded.index.embed_model = 'bge-m3';
  assert.deepEqual(plain(Handoff.reconcile(recorded, CURRENT, SERVED).unserved), [{
    path: 'index.embed_model', value: 'bge-m3',
    reason: 'not served by this lab',
  }]);
});

test('an empty model is no model, not a model this lab fails to serve', () => {
  const recorded = copy(CURRENT);
  recorded.generation.model = '';
  recorded.index.embed_model = '';
  assert.deepEqual(plain(Handoff.reconcile(recorded, CURRENT, SERVED).unserved), []);
});

test('a number outside the panel control says the range it is outside', () => {
  const recorded = copy(CURRENT);
  recorded.retrieval.k = 500;
  const out = Handoff.reconcile(recorded, CURRENT, SERVED);
  assert.equal(out.config.retrieval.k, 8);
  assert.deepEqual(plain(out.unserved), [{
    path: 'retrieval.k', value: 500,
    reason: 'outside this panel’s range, 1 to 50',
  }]);
});

test('a knob no control shows is carried through rather than reported', () => {
  // `rrf_k`, `agentic_weights`, `max_context_chars`: the panel has no control
  // for them, and dropping them would silently re-run the experiment under this
  // lab's defaults instead of the ones it recorded.
  const recorded = copy(CURRENT);
  recorded.retrieval.rrf_k = 42;
  const out = Handoff.reconcile(recorded, CURRENT, SERVED);
  assert.equal(out.config.retrieval.rrf_k, 42);
  assert.deepEqual(plain(out.unserved), []);
});

test('a ledger-only config sets the knobs it names and no others', () => {
  // An index build, a retrieval, an imported archive or any run whose run file
  // is gone: `leaderboard.ledger_config` can name six knobs and nothing else.
  const recorded = {
    index: { chunker: 'session', embedder: 'sentence-transformers' },
    retrieval: { retriever: 'dense', reranker: 'none', grader: 'lexical' },
    generation: { answerer: 'llm' },
  };
  const out = Handoff.reconcile(recorded, CURRENT, SERVED);
  assert.deepEqual(plain(out.unserved), []);
  assert.equal(out.config.index.chunker, 'session');
  assert.equal(out.config.retrieval.retriever, 'dense');
  assert.equal(out.config.index.chunk_chars, 500,
    'a knob the record never named is the reader’s, left alone');
  assert.deepEqual(plain(out.set).sort(), [
    'generation.answerer', 'index.chunker', 'index.embedder',
    'retrieval.grader', 'retrieval.reranker', 'retrieval.retriever',
  ], 'what was set is counted so the notice can say how much of the panel moved');
});

test('the label travels with the settings it labelled', () => {
  const recorded = copy(CURRENT);
  recorded.label = 'hybrid vs dense';
  assert.equal(Handoff.reconcile(recorded, CURRENT, SERVED).config.label,
    'hybrid vs dense');
});

test('reconciling changes neither the record nor what is on screen', () => {
  const recorded = copy(CURRENT);
  recorded.index.chunker = 'turn-pair';
  const before = { recorded: copy(recorded), current: copy(CURRENT) };
  Handoff.reconcile(recorded, CURRENT, SERVED);
  assert.deepEqual(plain(recorded), before.recorded);
  assert.deepEqual(CURRENT, before.current);
});

// --- the notice -------------------------------------------------------------
// What the reader is told, in the widget's log. This is the whole difference
// between "the Laboratory is now experiment X" and a quiet substitution, so it
// is pinned as hard as the reconciliation it describes.

const RECORD = {
  experiment_id: 'exp-9f3a', kind: 'run', started_at: '2026-08-19 14:02:51',
  dataset: 'diary-fa', source: 'both', label: '',
};

test('the notice names the experiment, its corpus and how much moved', () => {
  const out = Handoff.reconcile(
    { index: { chunker: 'session' }, retrieval: { k: 12 } }, CURRENT, SERVED);
  const said = Handoff.notice(RECORD, out);
  assert.match(said, /exp-9f3a/);
  assert.match(said, /diary-fa/);
  assert.match(said, /2026-08-19 14:02/);
  assert.match(said, /\brun\b/);
  assert.match(said, /2 knobs/, 'how much of the panel moved');
});

test('the notice names every knob it could not set, with its value and why', () => {
  const recorded = copy(CURRENT);
  recorded.index.chunker = 'turn-pair';
  recorded.generation.model = 'gpt-4o';
  const said = Handoff.notice(RECORD,
    Handoff.reconcile(recorded, CURRENT, SERVED));
  assert.match(said, /index\.chunker = turn-pair \(not served by this lab\)/);
  assert.match(said, /generation\.model = gpt-4o \(not served in openrouter mode\)/);
  assert.match(said, /unchanged/,
    'a knob that could not be set is a knob left where the reader had it');
});

test('a notice with nothing to report does not manufacture a caveat', () => {
  const said = Handoff.notice(RECORD,
    Handoff.reconcile({ index: { chunker: 'session' } }, CURRENT, SERVED));
  assert.doesNotMatch(said, /could not/);
  assert.doesNotMatch(said, /unchanged/);
});

test('a ledger-only row says the record itself is partial, not this lab', () => {
  // Six knobs out of forty is not "the Laboratory is now this experiment". The
  // reason is the record, not the installation, and the two must not be told in
  // one sentence — an index build has no run file to have recorded more.
  const said = Handoff.notice(
    Object.assign({}, RECORD, { kind: 'index', source: 'ledger' }),
    Handoff.reconcile({ index: { chunker: 'session', embedder: 'token-hash' } },
      CURRENT, SERVED));
  assert.match(said, /no run file/);
  assert.match(said, /every other knob is unchanged/);
});

test('a record that carries no config at all says so rather than claiming two knobs', () => {
  const said = Handoff.notice(RECORD, Handoff.reconcile({}, CURRENT, SERVED));
  assert.match(said, /no settings/);
  assert.doesNotMatch(said, /0 knobs/, 'a count of nothing reads as an accident');
});

test('the notice puts the corpus first among what it could not set', () => {
  // A config applied against the wrong corpus is not that experiment at all,
  // whatever else survived, so the reader must not have to find that line.
  const recorded = copy(CURRENT);
  recorded.index.chunker = 'turn-pair';
  recorded.index.dataset = 'gone-fa';
  const said = Handoff.notice(RECORD,
    Handoff.reconcile(recorded, CURRENT, SERVED));
  assert.ok(said.indexOf('index.dataset') < said.indexOf('index.chunker'),
    'the corpus is named before the knobs measured against it');
});

test('every unserved knob is reported, not just the first', () => {
  // The archive import refuses on the first one it meets. This path applies
  // what it can, so a reader who is told about one of three unserved knobs is
  // being told the panel is closer to the experiment than it is.
  const recorded = copy(CURRENT);
  recorded.index.chunker = 'turn-pair';
  recorded.index.dataset = 'gone-fa';
  recorded.retrieval.k = 500;
  const out = Handoff.reconcile(recorded, CURRENT, SERVED);
  assert.deepEqual(plain(out.unserved).map((row) => row.path).sort(),
    ['index.chunker', 'index.dataset', 'retrieval.k']);
});
