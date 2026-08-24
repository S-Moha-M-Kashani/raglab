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
// that is what an unserved knob has to be left at. Every field `LabConfig`
// actually has, shown or not — a fixture that quietly dropped `rrf_k` would
// make its own "carried through unshown" test pass for the wrong reason, and
// one that still said `key_facts_judge` would make "an unknown key is dropped
// and named" untestable, since that is the exact field this schema renamed.
const CURRENT = {
  label: 'what the reader was doing',
  index: {
    dataset: 'diary-fa', chunker: 'semantic-drift', chunk_chars: 500,
    overlap: 100, contextual: true, embedder: 'token-hash', embed_model: '',
    hierarchy: '', graph_source: 'hybrid', graph_knn: 8, granularity: 1,
    hierarchy_levels: 1, min_group: 3, summarizer: 'centroid',
  },
  retrieval: {
    retriever: 'hybrid-rrf', k: 8, candidates: 40, rrf_k: 60,
    time_filter: true, multi_query: true, hyde: false, expansion_model: '',
    mmr_lambda: 1, reranker: 'lexical', rerank_depth: 20, reranker_model: '',
    recency_half_life_days: 180, agentic_weights: [1, 0.3, 0.2],
    grader: 'none', grade_threshold: 0, grader_model: '',
    max_context_chars: 6000, summary_scope: 'mixed', summary_boost: 1,
    summary_levels: '',
  },
  generation: { answerer: 'extractive', model: '', fact_judge: false,
                judge_model: '', ragas_model: '' },
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
  assert.match(said, /chunker = turn-pair — not served by this lab/);
  assert.match(said, /model = gpt-4o — not served in openrouter mode/);
  assert.match(said, /at their default/,
    'a knob this lab cannot serve is left at the default, not at a value from '
    + 'whatever the reader happened to be looking at');
});

test('what could not be set is grouped by the stage that would run it', () => {
  // One line the reader can act on, in the order the pipeline runs: a list of
  // dotted paths makes them work out for themselves which stage each belongs to.
  const recorded = copy(CURRENT);
  recorded.index.chunker = 'turn-pair';
  recorded.retrieval.k = 500;
  recorded.generation.model = 'gpt-4o';
  const said = Handoff.notice(RECORD,
    Handoff.reconcile(recorded, CURRENT, SERVED));
  assert.match(said, /To set: Index \([^)]*chunker[^)]*\), Retrieve \([^)]*k[^)]*\), Generation \([^)]*model[^)]*\)/);
});

test('a stage with nothing to set is left out of the list, not shown empty', () => {
  const recorded = copy(CURRENT);
  recorded.generation.model = 'gpt-4o';
  const said = Handoff.notice(RECORD,
    Handoff.reconcile(recorded, CURRENT, SERVED));
  assert.match(said, /To set: Generation \(/);
  assert.doesNotMatch(said, /Index \(/);
  assert.doesNotMatch(said, /Retrieve \(/);
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
  assert.match(said, /every other knob is at its default/);
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
  assert.ok(said.indexOf('dataset') < said.indexOf('chunker'),
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

// --- opening a pre-branch archive: a retired key name is not a refusal -----
// The exact regression this covers: every experiment recorded before the
// generic-dataset-schema branch carries `generation.key_facts_judge` instead
// of `generation.fact_judge` — the field was renamed, not merely moved — and
// the open path used to hand the recorded config straight to the strict
// archive codec, whose exact-key-set check refused the whole handoff over
// one renamed field. `reconcile` is where that has to be absorbed instead:
// a name this lab's config no longer has is unservable by definition, so it
// is dropped and named exactly like any other knob this lab cannot serve,
// while the field it was renamed to — never named by the record, since the
// record predates it — simply stays at whatever the panel already has,
// exactly as any other knob the record does not mention does.

test('opening a pre-branch archive drops its retired key and leaves the '
  + 'renamed one at the panel value, naming the drop', () => {
  const recorded = copy(CURRENT);
  delete recorded.generation.fact_judge;
  recorded.generation.key_facts_judge = true;
  const out = Handoff.reconcile(recorded, CURRENT, SERVED);

  // The open proceeds: no exception, a config comes back either way — this
  // is the whole point, contrasted with the archive import's strict refusal.
  assert.ok(out.config);

  // The retired key never reaches the config this lab is about to run.
  assert.equal('key_facts_judge' in out.config.generation, false,
    'a name this schema no longer has must not land on the config that runs');

  // The renamed field was never named by this record, so it is the reader's
  // — untouched, exactly like `chunk_chars` in the ledger-only case above.
  assert.equal(out.config.generation.fact_judge, CURRENT.generation.fact_judge);

  // And the drop is named, not silent — the same contract every other
  // unserved knob gets.
  assert.deepEqual(plain(out.unserved), [{
    path: 'generation.key_facts_judge', value: true,
    reason: 'not a knob this lab reads any more',
  }]);
  const said = Handoff.notice(RECORD, out);
  assert.match(said, /key_facts_judge = true — not a knob this lab reads any more/);
});
