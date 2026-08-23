"""What is left of the original monolith after the topic sections moved into
their own files, and after step 4 of the test-plan merged what remained: the
evaluation harness (run against the five-session smoke corpus, not the
167-session diary), the RAGAS bridge, the four metrics that decide the
architecture, the sweep that produces the leaderboard, what every number on
the dashboard means, the three pipeline steps, the HTTP job surface the one
end-to-end test does not cover, the panel's dependency rules, the project's
production preset, and two regression reproductions.

The full HTTP journey — index job → evaluation job → run file → ledger row →
leaderboard group — lives in `tests/test_e2e.py`; what is left here are the
route contracts that journey does not make."""
import ast
import json
import os
import threading
from dataclasses import replace

import pytest

from raglab.evaluation import production_baseline_snapshot as baseline
from raglab.configuration import lab_config as config
from raglab.corpora import corpus_reading as corpus
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import run_evaluation as evaluate
from raglab.configuration import explainer_assembly as explain
from raglab.evaluation import deterministic_metrics as metrics
from raglab.llm_backends import model_role_catalogue as models
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.evaluation import ragas_judged_metrics as ragas_eval
from raglab.rag_components.retrieval import (
    retrieve_fuse_rerank_grade as retrieval)
from raglab.rag_components.indexing import (
    index_builder_registry as index_module)
from raglab.agents.extra_tools import sweep
from raglab.configuration.lab_config import (
    GenerationConfig,
    IndexConfig,
    LabConfig,
    RetrievalConfig)
from raglab.rag_components.indexing.index_builder_registry import IndexRegistry

from raglab.conftest import (
    LAB_SETTINGS,
    RAGLAB_DIR,
    SMOKE_INDEX,
    _finished,
    drain_jobs)


@pytest.fixture(scope='module')
def smoke_lab():
    """A registry and the ground truth for the smoke corpus (5 sessions, 6
    questions, `token-hash` — no model download), which is what the harness
    tests run against: nothing they claim needs the 167-session diary, and
    the diary build is the single most expensive thing this file could do.
    Module-scoped, so the runs below share one build."""
    _, truth = datasets.load('smoke-mini')
    return IndexRegistry(LAB_SETTINGS), truth


# --- the evaluation harness -------------------------------------------------

def test_run_eval_scores_a_slice_end_to_end(smoke_lab, tmp_path, monkeypatch):
    # this is an integration test
    """One run, and everything a run has to get right about itself: a row per
    selected question with an answer on it, an overall summary, a
    `started_at` that agrees with the run id (a field named for the start
    that holds the finish turns a run into a timeline nobody can
    reconstruct), and exactly one strict-JSON file left behind — `rglob`
    rather than `glob`, so a stray subdirectory beside the runs cannot hide.
    The index, the contexts and the answers die with the process; the file is
    the only thing that outlives it.

    No `by_type` assertion here (dropped, not renamed): `type` is no longer a
    guaranteed field on a question — a corpus declares whatever question
    labels it likes (D7) — so `aggregate()` no longer has a fixed vocabulary
    to break down by; see its own docstring."""
    registry, truth = smoke_lab
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    cfg = LabConfig(index=IndexConfig(**SMOKE_INDEX),
                    retrieval=RetrievalConfig(k=3, reranker='lexical'),
                    generation=GenerationConfig(answerer='extractive'),
                    label='test-slice')
    result = evaluate.run_eval(registry, truth, cfg, LAB_SETTINGS,
                               limit=4, ragas_mode='off')

    assert len(result.rows) == 4
    assert all('answer' in row for row in result.rows)
    assert result.summary['overall']['headline'] is not None
    assert 'by_type' not in result.summary

    stamp = result.run_id.split('-')[1]                      # HHMMSS
    assert result.started_at.endswith(f'{stamp[:2]}:{stamp[2:4]}:{stamp[4:]}')

    assert [p.name for p in tmp_path.rglob('*')] == [f'{result.run_id}.json']
    saved = json.loads((tmp_path / f'{result.run_id}.json').read_text(
        encoding='utf-8'), parse_constant=lambda literal: pytest.fail(
            f'{literal} is not JSON a strict parser accepts'))
    assert saved['run_id'] == result.run_id
    assert saved['config'] and saved['summary'] and saved['rows']


def test_a_traced_evaluation_scores_identically_and_leaves_traces_off_disk(
        smoke_lab, tmp_path, monkeypatch):
    # this is an integration test
    """Two things must stay true: the scores may not move, since tracing is a
    recording of the same retrieval and not a different one; and the traces
    may not reach `.runs/`, the leaderboard's durable artifact, with data no
    score is computed from. Both halves are checked by calling `run_eval`
    twice — once traced, once not — rather than through HTTP, which proved
    the same thing an index build and a job poll more expensively."""
    registry, truth = smoke_lab
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    cfg = LabConfig(index=IndexConfig(**SMOKE_INDEX),
                    retrieval=RetrievalConfig(retriever='hybrid-rrf',
                                              reranker='none', grader='none',
                                              k=3, rerank_depth=20,
                                              time_filter=False,
                                              multi_query=False),
                    generation=GenerationConfig(answerer='extractive'))
    traced = evaluate.run_eval(registry, truth, cfg, LAB_SETTINGS, limit=2,
                               ragas_mode='off', trace=True)

    assert len(traced.rows) == 2
    assert [t['question_id'] for t in traced.traces] == \
        [row['id'] for row in traced.rows]
    assert all(t['trace']['candidates'] for t in traced.traces)

    # The chunks this run retrieved from, for the same reason `/api/retrievals`
    # carries them: an evaluation builds its index implicitly and creates no
    # index job, so this is the only way the Inspector can show the chunks the
    # scores were actually computed over instead of an unrelated earlier build.
    groups = traced.chunks_by_session
    assert sum(len(g['chunks']) for g in groups) == traced.index['chunks']

    saved = json.loads((tmp_path / f'{traced.run_id}.json').read_text(
        encoding='utf-8'))
    assert 'traces' not in saved, 'traces must not reach the run file'
    assert 'chunks_by_session' not in saved, 'chunk text must not reach the run file'
    assert saved['rows'] == evaluate.json_safe(traced.rows)

    # The load-bearing half: the same config, untraced, produces the same rows
    # and the same summary. Latency is dropped at every depth — it measures the
    # machine, not the pipeline, so two runs of identical code never agree on it
    # and comparing it would make this test flaky rather than strict.
    def scores(value):
        if isinstance(value, dict):
            return {k: scores(v) for k, v in value.items() if 'latency' not in k}
        if isinstance(value, list):
            return [scores(v) for v in value]
        return value

    untraced = evaluate.run_eval(registry, truth, cfg, LAB_SETTINGS, limit=2,
                                 ragas_mode='off')
    assert not untraced.traces, 'trace=False must record nothing'
    assert scores(untraced.rows) == scores(traced.rows)
    assert scores(untraced.summary) == scores(traced.summary)


def test_config_round_trips_through_the_panel_payload():
    # this is a unit test
    cfg = LabConfig.from_dict({'index': {'chunker': 'session', 'unknown': 1},
                               'retrieval': {'k': 3},
                               'generation': {'answerer': 'none'},
                               'label': 'x'})
    assert cfg.index.chunker == 'session' and cfg.retrieval.k == 3
    assert cfg.validate() == []
    assert LabConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


# --- RAGAS bridge ----------------------------------------------------------

def test_ragas_telemetry_is_disabled_on_import():
    # this is a unit test
    """RAGAS's usage ping blocks for minutes per `evaluate()` call when its
    endpoint is unreachable; importing the bridge must be enough to prevent
    that."""
    assert os.environ.get('RAGAS_DO_NOT_TRACK') == 'true'


def test_ragas_availability_reports_missing_pieces_instead_of_raising():
    # this is a unit test
    status = ragas_eval.availability(LAB_SETTINGS)
    assert isinstance(status.installed, bool)
    if status.installed:
        assert not status.llm_ready   # no key in LAB_SETTINGS
    assert 'ragas' in status.as_dict()['install_hint']


def test_evidence_texts_are_every_relevant_documents_own_evidence_text(
        diary, ground_truth):
    # this is a unit test
    """`corpus_reading.evidence_texts` reads the schema's own vocabulary
    directly (D4): every relevant document's evidence entries, by their own
    `text`, in order. Rewritten from a test pinning deleted behaviour — the
    old dialect's `evidence_texts` expanded a short `quote` out to the whole
    cited message and fell back to the quote when the cited session was
    missing; the new schema's evidence entries already carry their own full
    text, so neither expansion nor fallback exists any more (the `documents`
    parameter survives only so a caller already holding `documents_by_id`'s
    result need not change its call)."""
    documents = corpus.documents_by_id(diary)
    question = next(q for q in ground_truth['groundtruth_dataset']
                    if q['expected_answer']['behavior'] == 'answer')
    texts = corpus.evidence_texts(documents, question)
    assert texts
    expected = [ev['text'] for relevant in question['relevant_corpus_documents']
               for ev in relevant['evidence']]
    assert texts == expected

    # No relevant documents (an 'abstain' question) is no evidence text —
    # never a fabricated one.
    assert corpus.evidence_texts({}, {'relevant_corpus_documents': []}) == []


def test_json_safe_replaces_undefined_metrics_with_null():
    # this is a unit test
    assert evaluate.json_safe({'a': float('nan'), 'b': [1.0, float('nan')]}) == \
        {'a': None, 'b': [1.0, None]}


def test_an_offline_ragas_run_scores_a_retrieval_and_reports_no_decision(
        smoke_index):
    # this is an integration test
    """The offline pair really scores what was retrieved — and, because it
    cannot measure any of the four deciding metrics, says so with `None`
    rather than a number that looks comparable to a judged run's."""
    pytest.importorskip('ragas')
    pytest.importorskip('rapidfuzz')
    _, truth = datasets.load('smoke-mini')
    query_date = truth['groundtruth_dataset_metadata'][
        'default_question_asked_at'][:10]
    questions = [q for q in truth['groundtruth_dataset']
                if q['expected_answer']['behavior'] != 'abstain'][:3]
    pairs = [(q, pipeline.retrieve(smoke_index.index, RetrievalConfig(k=5),
                                   q['question'], query_date))
             for q in questions]
    report = ragas_eval.run(pairs, LAB_SETTINGS, smoke_index.index.embedder,
                            mode='offline')
    assert report['n_samples'] == 3, report['notes']
    assert 'non_llm_context_recall' in report['metrics']
    assert 0.0 <= report['metrics']['non_llm_context_recall'] <= 1.0
    assert report['decision'] is None
    assert report['decision_metrics'] == list(ragas_eval.DECISION_METRICS)


# --- the four metrics that decide the architecture -------------------------
# Everything the lab measures is reported, but only four RAGAS metrics vote:
# between them they cover retrieval (context precision, recall) and
# generation (faithfulness, answer relevancy) failing to use what it fetched.

def test_the_deciding_metrics_are_exactly_the_four_chosen_ones():
    # this is a convention test
    assert ragas_eval.DECISION_METRICS == (
        'faithfulness', 'answer_relevancy',
        'llm_context_precision_with_reference', 'context_recall')
    # Everything else stays measured and reported; it simply does not vote.
    assert set(ragas_eval.DECISION_METRICS) < set(ragas_eval.LLM_METRICS)
    assert 'factual_correctness(mode=f1)' not in ragas_eval.DECISION_METRICS


def test_the_decision_score_is_the_unweighted_mean_of_those_four():
    # this is a unit test
    """Unweighted on purpose: any weighting would be a claim about relative
    importance this fixture cannot support."""
    score = ragas_eval.decision_score({
        'faithfulness': 1.0, 'answer_relevancy': 0.6,
        'llm_context_precision_with_reference': 0.4, 'context_recall': 0.0,
        # Present, reported, and deliberately ignored by the arithmetic.
        'factual_correctness(mode=f1)': 0.0, 'non_llm_context_recall': 1.0,
    })
    assert score == 0.5


def test_the_decision_score_is_undefined_unless_all_four_are_present():
    # this is a unit test
    """A mean over whichever metrics happened to succeed is not comparable
    between runs: an offline run would score on two metrics and outrank a judged
    run scored on four."""
    assert ragas_eval.decision_score({'faithfulness': 1.0}) is None
    assert ragas_eval.decision_score({}) is None
    assert ragas_eval.decision_score(
        {'faithfulness': 1.0, 'answer_relevancy': 1.0,
         'llm_context_precision_with_reference': 1.0}) is None


def test_the_decision_score_carries_its_own_uncertainty():
    # this is a unit test
    """A ranking of means with no spread cannot say whether it ranked
    anything, so the score ships with the standard error of the
    per-question composite beside it. Computed per question and then across
    questions, not per metric: the four are measured on the same answers and
    are correlated, so averaging four independent standard errors would
    understate the real spread."""
    rows = [{'faithfulness': 1.0, 'answer_relevancy': 1.0,
             'llm_context_precision_with_reference': 1.0, 'context_recall': 1.0},
            {'faithfulness': 0.0, 'answer_relevancy': 0.0,
             'llm_context_precision_with_reference': 0.0, 'context_recall': 0.0}]
    spread = ragas_eval.decision_spread(rows)
    assert spread['n'] == 2
    assert spread['mean'] == 0.5
    # Two composites at 0 and 1: sd = 0.7071, so SE = sd/sqrt(2) = 0.5.
    assert spread['stderr'] == 0.5

    # A question missing one of the four has no composite, so it cannot be one.
    partial = ragas_eval.decision_spread(
        rows + [{'faithfulness': 1.0, 'answer_relevancy': 1.0}])
    assert partial['n'] == 2, 'a partial composite is not a sample'

    # One question cannot have a standard error, and must not claim zero.
    assert ragas_eval.decision_spread(rows[:1])['stderr'] is None
    assert ragas_eval.decision_spread([])['n'] == 0


def test_every_ragas_report_carries_a_spread_even_when_it_measured_nothing():
    # this is a unit test
    """A run that could not measure the four reports `n=0` rather than
    omitting the field — a missing key would make the frontend fall back to
    printing the bare mean."""
    report = ragas_eval.run([], LAB_SETTINGS, None, mode='off')
    assert report['decision_spread'] == {'n': 0, 'mean': None, 'stderr': None}


def test_the_leaderboard_row_carries_the_deciding_score_and_its_error(
        tmp_path, monkeypatch):
    # this is an integration test
    """A leaderboard that ranks on a number it does not carry cannot be
    checked against the run it came from, and the error has to travel with
    the mean or the row cannot say whether it beat the row below it. An
    absent error is reported as absent, never as `± 0` — which would claim
    the oldest rows were measured the most precisely."""
    result = evaluate.RunResult(
        run_id='x', label='y', config={}, index={},
        summary={'overall': {}, 'n_questions': 0},
        ragas={'mode': 'llm', 'metrics': {'faithfulness': 0.8},
               'decision': 0.75, 'decision_metrics': [],
               'decision_spread': {'n': 24, 'mean': 0.75, 'stderr': 0.05}})
    assert result.brief()['ragas_decision'] == 0.75
    assert result.brief()['ragas_decision_stderr'] == 0.05

    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    (tmp_path / '20260101-000000-abcdef.json').write_text(json.dumps({
        'run_id': '20260101-000000-abcdef', 'label': 'old',
        'summary': {'n_questions': 24, 'overall': {}},
        'ragas': {'metrics': {}, 'decision': 0.6}}), encoding='utf-8')
    row = evaluate.list_runs()[0]
    assert row['ragas_decision'] == 0.6
    assert row['ragas_decision_stderr'] is None


# --- the sweep that produces the leaderboard --------------------------------
# The sweep spends hours of judged model calls, so everything it can get wrong
# silently is worth an assertion: a row that is not comparable to the others, a
# row that cannot be ranked at all, and a row that raises after 40 minutes.

def test_every_candidate_is_selected_by_a_unique_letter():
    # this is a unit test
    """`--only A F` and `--final G` select on `label.split()[0]`, so two
    candidates sharing a letter would run silently under the wrong label."""
    letters = [c.label.split()[0] for c in sweep.candidates()]
    assert len(letters) == len(set(letters)), letters
    assert all(len(letter) == 1 for letter in letters), letters


def test_every_candidate_holds_the_embedder_and_both_models_fixed():
    # this is a unit test
    """The sweep's claim is that each row changes one thing — a row that
    moved the embedder or either model would be incomparable to every other
    row while looking like a knob result."""
    for cfg in sweep.candidates():
        assert cfg.index.embedder == sweep.EMBEDDER, cfg.label
        assert cfg.index.embed_model == sweep.EMBED_MODEL, cfg.label
        assert cfg.generation.model == sweep.ANSWER_MODEL, cfg.label
        assert cfg.generation.ragas_model == sweep.JUDGE_MODEL, cfg.label
        assert cfg.generation.model != cfg.generation.ragas_model, (
            'a model grading its own answer is not evidence')


def test_every_candidate_generates_an_answer_so_it_can_be_ranked():
    # this is a unit test
    """All four deciding metrics need a response. A candidate that retrieved
    without answering would score `None`, drop to the bottom of the ranking as
    if it had lost, and cost a full run to say nothing."""
    assert all(c.generation.answerer == 'llm' for c in sweep.candidates())


def test_every_candidate_validates_before_the_sweep_starts():
    # this is a unit test
    """`run_eval` raises on an invalid config. Candidate H is the eighth row, so
    a typo there would surface after an hour of paid judging."""
    for cfg in sweep.candidates():
        assert cfg.validate() == [], cfg.label


def test_no_two_candidates_are_the_same_configuration():
    # this is a unit test
    """A duplicated row costs ten minutes and reads as reproducibility."""
    seen = {}
    for cfg in sweep.candidates():
        key = json.dumps(replace(cfg, label='').to_dict(), sort_keys=True)
        assert key not in seen, f'{cfg.label} duplicates {seen.get(key)}'
        seen[key] = cfg.label


def test_the_final_run_refuses_to_start_without_a_judge(monkeypatch, tmp_path):
    # this is a unit test
    """Without a key the LLM stages fall back to the offline fake, which
    would produce a full leaderboard row of meaningless scores under the
    winner's name. `RUNS_DIR` is redirected too, since an unguarded
    `final()` can drop a fake-provider run into the real `.runs/`."""
    monkeypatch.setattr(sweep, 'load_lab_settings', lambda: LAB_SETTINGS)
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    with pytest.raises(SystemExit):
        sweep.final(None, 1, 'A')
    assert not list(tmp_path.iterdir()), 'refusing must happen before any run'


# --- what each number on the dashboard actually means -----------------------
# A claim nobody can check is worse than no claim, so each metric carries the
# same facts (label, step, formula, library, help) from one registry, and the
# panel's step inks are that registry's own list rather than a palette the
# frontend invents. Three parametrized pins instead of the thirteen separate
# ones this section used to hold.

# key → the exact fragments its own definition must carry, and (under the
# second table) the ones it must not. A measure may ship without a row here;
# a row that stops matching is a definition that has drifted from the code it
# names. Fragments are matched case-insensitively, as the originals were.
PINNED = {
    # The headline is a weighted sum invented here, so its weights are the
    # formula.
    'headline': {'formula': ('0.4', '0.3', '0.2', '0.1')},
    'recall': {'formula': ('|gold ∩ top-k| / |gold|',),
               'library': ('metrics.recall_at_k',)},
    'mrr': {'formula': ('1 / rank',)},
    'ndcg': {'formula': ('log2',)},
    'quote_recall': {'formula': ('0.9',),         # the fuzzy fallback
                     'library': ('difflib',)},
    'answer_similarity': {'library': ('difflib',)},
    # A deterministic metric must not claim to be a model, and vice versa.
    'fact_coverage': {'library': ('llm',)},
    # "Faithfulness" is RAGAS's word, not ours, so the panel says whose
    # definition it is showing and which class computed it.
    'faithfulness': {'library': ('Faithfulness', 'ragas'),
                     'formula': ('supported claims', '/'),
                     'help': ('claims',)},
    'answer_relevancy': {'library': ('ResponseRelevancy',),
                         'formula': ('cosine',)},
    'factual_correctness(mode=f1)': {'library': ('FactualCorrectness',),
                                     'formula': ('F1',)},
    # The offline pair is string distance, not a model — and says so.
    'non_llm_context_recall': {'library': ('NonLLMContextRecall', 'rapidfuzz')},
    # The deciding score is the number the architecture was chosen by, so of
    # everything on the screen it is the one that must not be a bare figure.
    'ragas_decision': {'help': ('faithfulness', 'answer relevancy',
                                'context precision', 'context recall')},
}

FORBIDDEN = {'recall': {'library': ('llm',)},
             'non_llm_context_recall': {'formula': ('llm',)}}

# Same three inks as the panels: a number about retrieval is green wherever it
# appears, so the dashboard means one thing by a colour. Whole-pipeline numbers
# carry '' rather than claiming a stage.
PINNED_STEPS = {'recall': 'retrieval', 'ndcg': 'retrieval',
                'answer_similarity': 'generation', 'latency_ms': ''}


@pytest.mark.parametrize('registry', ['ours', 'ragas'])
def test_every_measure_defines_itself(registry):
    # this is a convention test
    """Every number the panel can print carries the same facts from one
    place: a label, a short caption, the arithmetic (not prose about it), the
    module or RAGAS class that computed it *and* whether a model was
    involved, the step whose ink it wears, and a paragraph in the one help
    registry under `metric.<key>`."""
    measures = (metrics.MEASURES if registry == 'ours'
                else ragas_eval.RAGAS_MEASURES)
    steps = {step.key for step in config.STEPS} | {''}
    topics = explain.topics()
    judged = set(ragas_eval.LLM_METRICS) | {'ragas_decision'}

    for measure in measures:
        assert measure.label and measure.short, measure.key
        assert measure.formula and measure.library and measure.help, measure.key
        assert measure.step in steps, measure.key
        assert topics.get(f'metric.{measure.key}'), measure.key
        if measure.key in PINNED_STEPS:
            assert measure.step == PINNED_STEPS[measure.key], measure.key
        for field, fragments in PINNED.get(measure.key, {}).items():
            for fragment in fragments:
                assert fragment.lower() in getattr(measure, field).lower(), \
                    (measure.key, field, fragment)
        for field, fragments in FORBIDDEN.get(measure.key, {}).items():
            for fragment in fragments:
                assert fragment.lower() not in getattr(measure, field).lower(), \
                    (measure.key, field, fragment)
        if registry == 'ragas':
            # A number produced by a model is a number with variance, and the
            # reader has to know which model. The decision score is judged
            # too, being a mean of four judged metrics — a composite must not
            # launder its inputs' variance by being an average.
            if measure.key in judged:
                assert 'RAGAS judge' in measure.library, measure.key
            else:
                assert 'no model' in measure.library.lower(), measure.key


def test_the_registries_line_up():
    # this is a convention test
    """The step list is a registry the lab owns: it is the config groups in
    pipeline order, every model role names one of them, every one of them
    owns at least one model, and every key a run can report is defined. Any
    of these drifting puts a control, a colour or a number somewhere nothing
    explains it."""
    assert [step.key for step in config.STEPS] == ['index', 'retrieval',
                                                   'generation']
    # Two names on purpose: the long one titles a panel, the short one tags a
    # group of models inside another panel, where a whole sentence would not fit.
    assert all(step.label and step.short and step.note for step in config.STEPS)
    assert [step.short for step in config.STEPS] == ['Index', 'Retrieval',
                                                     'Generation']
    # A step is a config group with a colour, so the two lists cannot drift: a
    # new group would otherwise render in a panel nobody colours.
    steps = {step.key for step in config.STEPS}
    assert steps == {group for group, _ in explain.GROUPS}

    # The colour cannot disagree with where the value is stored: a role's step
    # is the group of the field its dropdown writes to.
    assert all(role.step in steps for role in models.ROLES)
    assert all(role.step == role.field.split('.')[0] for role in models.ROLES)
    # A step with no model in it is a legend entry pointing at nothing. The
    # index step owns the *embedder* — not a chat role, but a model all the same.
    assert {role.step for role in models.ROLES} | {'index'} == steps
    # And the step travels with the role to the panel.
    grade = next(role for role in models.ROLES if role.key == 'grade')
    assert grade.as_dict()['step'] == 'retrieval'

    # `aggregate()` can report these keys, so the panel can show them, so every
    # one of them has to be explainable — on both sides of the dashboard.
    defined = {measure.key for measure in metrics.MEASURES}
    reported = set(metrics.AGGREGATED) | {'headline'}
    assert reported <= defined, reported - defined
    ragas_defined = {measure.key for measure in ragas_eval.RAGAS_MEASURES}
    ragas_reported = (set(ragas_eval.OFFLINE_METRICS)
                      | set(ragas_eval.LLM_METRICS))
    assert ragas_reported <= ragas_defined, ragas_reported - ragas_defined

    # And every key the tables above pin still names a measure: the pins are
    # applied by walking the registries, so a renamed metric would take its
    # own pin out of the suite silently rather than failing.
    pinned = set(PINNED) | set(FORBIDDEN) | set(PINNED_STEPS)
    assert pinned <= defined | ragas_defined, pinned - (defined | ragas_defined)


# ---------------------------------------------------------------------------
# The HTTP surface — resource collections rather than action verbs. The full
# journey (index job → evaluation job → run file → ledger row → leaderboard)
# is tests/test_e2e.py; what is here is what that journey does not assert.
# ---------------------------------------------------------------------------

def test_starting_work_creates_a_job_and_says_where_to_watch_it(client,
                                                                monkeypatch):
    # this is an integration test
    """202 rather than 200: the work has been accepted, not done — the response
    body is a receipt, not a result. `Location` points at the job so a caller
    never has to build the polling url by string concatenation. A query is one
    of these too, for the same reason its siblings are: the index it builds
    implicitly can outwait anything a spinner honestly promises.

    The spy is here rather than in a second TestClient: if the query job does
    not pass its reporter down to the registry, the bar sits on 'starting 0%'
    for the whole implicit build. Recorded per route, not into one slot — a
    single slot the other two legs also write to would still hold a callable
    if the query route stopped building an index at all, which is exactly the
    case the second app used to isolate."""
    seen = {}
    original = IndexRegistry.get
    building = {}

    def spy(self, cfg, progress=None, force=False):
        seen[building['path']] = progress
        return original(self, cfg, progress=progress, force=force)

    monkeypatch.setattr(IndexRegistry, 'get', spy)

    for path, payload in (
            ('/api/indexes', {'index': dict(SMOKE_INDEX)}),
            ('/api/evaluations', {'index': dict(SMOKE_INDEX),
                                  'generation': {'answerer': 'none'},
                                  'limit': 1, 'ragas_mode': 'off'}),
            ('/api/queries', {'index': dict(SMOKE_INDEX),
                              'retrieval': {'retriever': 'dense', 'k': 2},
                              'generation': {'answerer': 'none'},
                              'question': 'What broke in the kitchen?'})):
        # Safe to key on: one job runs at a time and each is drained below
        # before the next is posted.
        building['path'] = path
        res = client.post(path, json=payload)
        assert res.status_code == 202, f'{path} -> {res.status_code}'
        job_id = res.json()['job_id']
        assert job_id
        assert res.headers['Location'] == f'/api/jobs/{job_id}'
        # And the url it points at is real.
        assert client.get(res.headers['Location']).status_code == 200
        # Drained before the next iteration — the client is module-scoped and
        # only one job runs at a time, so a leaked job here would hand the
        # next post (or the next test) a spurious 409.
        job = _finished(client, job_id)
        assert job['state'] == 'done', job.get('error')

    # `job` is the last leg, the query: the one whose result is every stage.
    assert job['kind'] == 'query'
    assert 'contexts' in job['result'] and 'diagnostics' in job['result']
    assert callable(seen.get('/api/queries')), \
        'the query job built its index without handing down its reporter'
    # The preconditions still refuse synchronously, and still say which one:
    # a bad payload is a 400 the panel shows at once, never a job that dies.
    assert client.post('/api/queries', json={}).status_code == 400


def test_a_second_job_is_refused_in_readable_english(client, monkeypatch):
    # this is an integration test
    """The refusal read 'a index job is still stopping' — wrong article, and
    'stopping' for a job that is running. A message describing the wrong state
    sends the reader looking for a bug that is not there.

    The race this pins used to be conditional on scheduling luck: the first
    job could finish before the second post landed, and the test would pass
    having exercised nothing. Held open on a `threading.Event`-blocked
    chunker instead, so the first job is provably still running when the
    second is posted."""
    entered = threading.Event()
    release = threading.Event()
    original = index_module.chunk_document

    def held(document, cfg, embedder, label_fields, language):
        entered.set()
        release.wait(timeout=5)
        return original(document, cfg, embedder, label_fields, language)

    monkeypatch.setattr(index_module, 'chunk_document', held)
    try:
        first = client.post('/api/indexes', json={
            'index': {**SMOKE_INDEX, 'chunker': 'message'}})
        assert first.status_code == 202
        assert entered.wait(timeout=5), 'the first job never reached the chunker'

        second = client.post('/api/indexes', json={
            'index': {**SMOKE_INDEX, 'chunker': 'turn-pair'}})
        assert second.status_code == 409
        detail = second.json()['detail']
        assert 'a index' not in detail
        assert 'an index job is already running' in detail
    finally:
        # Unblock the held job and drain it, since the client is module-scoped:
        # leaving a job running here would hand the next test a spurious 409.
        release.set()
        drain_jobs(client)


def test_the_index_job_reports_every_row_it_wrote_and_lists_what_it_ran(client):
    # this is an integration test
    """The Inspector follows the lab by polling `/api/jobs`, so a listed job
    has to carry the config it actually ran — `LabConfig`'s own normalised
    form, not the raw posted body — and nothing heavier than
    id/kind/state/config.

    And it holds no index of its own: it renders what the job returned, so a
    build that wrote summary rows and reported only its leaves would make
    them unreachable. `chunks` counts every row in the index, so leaves plus
    summaries have to sum back to it — while a flat build says "none" with an
    empty list rather than by leaving the key out, since "no hierarchy" and
    "a hierarchy that found nothing" are different facts."""
    # `louvain` rather than `metadata`, which groups by the storylines a
    # corpus declares and smoke-mini declares none of, so it would write no
    # summary at all. Over five densely-connected chunks louvain returns a
    # single community of five — above the default `min_group=3`, so the
    # build is guaranteed a group — and it is deterministic since the
    # 2026-08-13 IDF tie-break fix. It also needs no scikit-learn import,
    # which is what makes it ~6× faster here than `kmeans`.
    posted = client.post('/api/indexes', json={
        'index': {**SMOKE_INDEX, 'hierarchy': 'louvain',
                  'summarizer': 'centroid'}})
    assert posted.status_code == 202
    job = _finished(client, posted.json()['job_id'], timeout=120.0)
    assert job['state'] == 'done', job.get('error')
    result = job['result']

    entries = client.get('/api/jobs').json()['jobs']
    assert entries, 'expected at least one job listed'
    newest = entries[0]
    assert newest['id'] == job['id']
    assert newest['kind'] == 'index'
    assert newest['config']['index']['chunker'] == SMOKE_INDEX['chunker']
    assert newest['config']['index']['embedder'] == SMOKE_INDEX['embedder']
    assert newest['config']['index']['hierarchy'] == 'louvain'
    assert 'result' not in newest and '_cancel' not in newest

    leaves = sum(len(g['chunks']) for g in result['chunks_by_session'])
    summaries = result['summaries']
    assert summaries, 'this grouping over this corpus has groups to summarise'
    assert leaves + len(summaries) == result['chunks'], (
        'every row in the index must be reachable from one of the two views')
    assert leaves == result['leaves'], \
        'the chunk view holds the chunker output and nothing the summariser wrote'

    # each row carries what the summaries view labels it with
    for summary in summaries:
        for key in ('id', 'text', 'group_id', 'level', 'members', 'member_ids',
                    'sessions', 'chars'):
            assert key in summary, f'missing {key}'
        assert summary['members'] == len(summary['member_ids'])

    flat = client.post('/api/indexes', json={'index': dict(SMOKE_INDEX)})
    flat_job = _finished(client, flat.json()['job_id'], timeout=120.0)
    assert flat_job['state'] == 'done', flat_job.get('error')
    assert flat_job['result']['summaries'] == []
    assert sum(len(g['chunks']) for g in flat_job['result']['chunks_by_session']) \
        == flat_job['result']['chunks']


def test_both_run_routes_screen_the_models_the_backend_serves(client, monkeypatch):
    # this is an integration test
    """`/api/evaluations` refused a model the active backend does not serve;
    `/api/queries` ran it. Two routes over the same pipeline disagreeing about
    which configs are legal is a bug on its own, and it got worse the moment a
    dead grade stage started raising: the panel's fastest feedback loop would
    answer a bare 500 where the slow one answers a 400 naming the model."""
    monkeypatch.setattr(models, 'provider_problems',
                        lambda cfg, settings: ['model "qwen3.5:2b" is not served'])
    payload = {'index': dict(SMOKE_INDEX), 'generation': {'answerer': 'none'},
               'question': 'What broke in the kitchen?'}
    for path in ('/api/queries', '/api/evaluations'):
        res = client.post(path, json=payload)
        assert res.status_code == 400, f'{path} -> {res.status_code}'
        assert 'qwen3.5:2b' in res.json()['detail'], path


def test_a_query_whose_gate_cannot_reach_its_model_says_so(client, monkeypatch):
    # this is an integration test
    """The route half of the `GradeUnavailable` rule (the refusal itself is
    asserted directly below): refusing to score is only an improvement if the
    refusal reaches the caller readably, so the job's error names the stage
    that went missing — otherwise the reader blames retrieval for what the
    grader did."""
    def unreachable(*args, **kwargs):
        raise ConnectionError('the model daemon is not running')

    monkeypatch.setattr(retrieval, 'lab_chat', unreachable)
    res = client.post('/api/queries', json={
        'question': 'What broke in the kitchen?',
        'index': dict(SMOKE_INDEX),
        'retrieval': {'k': 4, 'grader': 'llm', 'grade_threshold': 0.4},
        'generation': {'answerer': 'none'}})
    assert res.status_code == 202, res.status_code
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'error'
    assert 'grade' in job['error'].lower() and 'not running' in job['error']


def test_retrieval_only_covers_exactly_the_experiment_questions(client):
    # this is an integration test
    """Retrieve, for the questions the eval card has selected, and nothing
    more: no answering, no judging, no run file. The selection has to be the
    *same* selection `/api/evaluations` would score, or the Inspector shows
    retrieval for questions the numbers were never about."""
    _, truth = datasets.load('smoke-mini')
    picked_type = truth['groundtruth_dataset'][0]['question_metadata']['question_type']
    payload = {'index': dict(SMOKE_INDEX),
               'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'none',
                             'grader': 'none', 'k': 3, 'rerank_depth': 20,
                             'time_filter': False, 'multi_query': False},
               'labels': {'question_type': [picked_type]}, 'limit': 2,
               'balance': ''}
    res = client.post('/api/retrievals', json=payload)
    assert res.status_code == 202, res.text
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    assert job['kind'] == 'retrieve'

    result = job['result']
    expected = evaluate.select_questions(
        truth, limit=2, labels={'question_type': [picked_type]}, balance='')
    assert [q['question_id'] for q in result['questions']] == \
        [q['groundtruth_question_id'] for q in expected]
    assert result['selection']['n'] == 2

    # The chunks it retrieved *from* travel with it. A run builds its index
    # implicitly, so without this the Inspector's chunks window would keep
    # showing whatever index job was last pressed — a different chunker than the
    # one that produced these rows, with nothing on screen saying so.
    groups = result['chunks_by_session']
    assert sum(len(g['chunks']) for g in groups) == result['index']['chunks']

    first = result['questions'][0]
    assert first['question'] == expected[0]['question']
    # retrieval only: the generation step never ran, so there is no answer to
    # show and no run file to leave behind.
    assert 'answer' not in first
    candidates = first['trace']['candidates']
    assert candidates
    for key in ('dense_rank', 'bm25_rank', 'fused_rank', 'rerank_score',
                'grade_score', 'kept'):
        assert key in candidates[0], f'missing {key}'
    # gold is marked per question, against that question's own evidence
    assert all(isinstance(c['gold'], bool) for c in candidates)


# ---------------------------------------------------------------------------
# The panel's two usability guarantees, held by the served data rather than by
# either frontend — a rule copied into two panels is a rule that will disagree.
# ---------------------------------------------------------------------------

def test_every_option_list_leads_with_the_default():
    # this is a unit test
    """A default buried sixth reads as an exotic choice; this fails if a
    default moves without its list."""
    cfg = LabConfig()
    for name, options, default in (
            ('chunkers', config.CHUNKERS, cfg.index.chunker),
            ('embedders', config.EMBEDDERS, cfg.index.embedder),
            ('retrievers', config.RETRIEVERS, cfg.retrieval.retriever),
            ('rerankers', config.RERANKERS, cfg.retrieval.reranker),
            ('graders', config.GRADERS, cfg.retrieval.grader),
            ('answerers', config.ANSWERERS, cfg.generation.answerer),
            ('hierarchies', config.HIERARCHIES, cfg.index.hierarchy),
            ('graph_sources', config.GRAPH_SOURCES, cfg.index.graph_source),
            ('summarizers', config.SUMMARIZERS, cfg.index.summarizer),
            ('summary_scopes', config.SUMMARY_SCOPES,
             cfg.retrieval.summary_scope)):
        assert options[0] == default, (
            f'{name} leads with {options[0]!r} but the default is {default!r}')


def test_a_dependent_control_is_live_only_when_its_owner_makes_it_mean_something():
    # this is a unit test
    """Each case is a knob the pipeline would ignore, so leaving it editable
    invites tuning a number that does nothing. `semantic-drift` is
    deliberately in the *enabled* set for chunk_chars: unlike
    message/turn-pair/session it genuinely reads it, as a max_chars cap."""
    def state(cfg):
        return config.dependency_state(cfg.to_dict())

    drift = state(LabConfig(index=IndexConfig(chunker='semantic-drift')))
    assert drift['index.chunk_chars']['enabled']
    assert not drift['index.overlap']['enabled']
    # Both directions, because only one of them is a *disabled* control: a
    # rule that emptied `index.overlap`'s owner list would grey the knob out
    # for every chunker, including the one whose whole point is the overlap.
    assert state(LabConfig(index=IndexConfig(chunker='fixed-overlap'))
                 )['index.overlap']['enabled']

    per_message = state(LabConfig(index=IndexConfig(chunker='message')))
    assert not per_message['index.chunk_chars']['enabled']
    assert 'structure' in per_message['index.chunk_chars']['reason']

    hashed = state(LabConfig(index=IndexConfig(embedder='char-hash')))
    assert not hashed['index.embed_model']['enabled']
    real = state(LabConfig(index=IndexConfig(embedder='sentence-transformers')))
    assert real['index.embed_model']['enabled']

    ungated = state(LabConfig(retrieval=RetrievalConfig(grader='none')))
    assert not ungated['retrieval.grade_threshold']['enabled']
    assert not ungated['retrieval.grader_model']['enabled']
    lexical_gate = state(LabConfig(retrieval=RetrievalConfig(grader='lexical')))
    assert lexical_gate['retrieval.grade_threshold']['enabled']
    assert not lexical_gate['retrieval.grader_model']['enabled']   # no model involved

    no_hyde = state(LabConfig(retrieval=RetrievalConfig(hyde=False)))
    assert not no_hyde['retrieval.expansion_model']['enabled']
    assert state(LabConfig(retrieval=RetrievalConfig(hyde=True))
                 )['retrieval.expansion_model']['enabled']

    extractive = state(LabConfig(generation=GenerationConfig(answerer='extractive')))
    assert not extractive['generation.model']['enabled']
    assert state(LabConfig(generation=GenerationConfig(answerer='llm'))
                 )['generation.model']['enabled']


def test_every_disabled_control_says_why():
    # this is a convention test
    """A greyed-out control with no reason is indistinguishable from a broken
    one. Every rule carries the sentence the panel shows."""
    for key, rule in config.DEPENDENCIES.items():
        assert rule['reason'], f'{key} has no reason'
        assert not rule['reason'].endswith('.'), (
            f'{key}: the panel completes "disabled because …", so no full stop')


# That both frontends are *served* these rules rather than each keeping a copy
# is asserted where the rest of `/api/options` is —
# test_server.py::test_options_describe_the_corpus_the_knobs_and_the_metrics.


def test_the_embedder_hints_describe_the_embedders_in_their_own_order():
    # this is a convention test
    """The standalone panel builds its embedder dropdown from
    EMBEDDER_HINTS, not from EMBEDDERS — two lists describing one set of
    choices have to agree on order or the panel disagrees with itself about
    what is recommended. And `hash` is retired in production *by name*, so a
    hint calling `ascii-hash` "the brain default" would describe a
    configuration that now raises at boot."""
    from raglab.rag_components.indexing.embedding_backends import EMBEDDER_HINTS

    assert [hint.kind for hint in EMBEDDER_HINTS] == list(config.EMBEDDERS)
    for hint in EMBEDDER_HINTS:
        if hint.kind.endswith('-hash'):
            assert 'brain default' not in hint.label, hint.label


# ---------------------------------------------------------------------------
# Two regression reproductions: they encode the correct behaviour and fail
# against the code as it stood before the fix.
# ---------------------------------------------------------------------------

def test_a_gate_whose_model_call_fails_does_not_silently_pass_everything():
    # this is a unit test
    """0.5 clears the default 0.4 threshold, so an unreachable model must
    not turn `grader='llm'` into a silent no-op — a row labelled
    `grader=llm` that was measured ungated is the one artefact this lab
    must never produce. It raises `GradeUnavailable` and names both the
    stage and the model, so the reader is not left guessing which stage went
    missing (the route half of the same rule is asserted above). The parse
    fallback is a different thing and must survive: a line the model wrote
    that we could not read is genuinely 'no opinion'."""
    class Unreachable:
        def invoke(self, messages, **kwargs):
            raise ConnectionError('the model daemon is not running')

    with pytest.raises(retrieval.GradeUnavailable) as caught:
        # A model name with no 'grade' in it, so the stage assertion below
        # stands on its own rather than being entailed by the model name the
        # message interpolates.
        retrieval.llm_scores(Unreachable(), 'qwen3.5:2b', 'q', ['a', 'b', 'c'])
    assert 'grade' in str(caught.value).lower()
    assert 'qwen3.5:2b' in str(caught.value)
    assert 'not running' in str(caught.value)

    # Unchanged: a reply that arrives but cannot be parsed is still no opinion.
    class Unparseable:
        def invoke(self, messages, **kwargs):
            return type('Reply', (), {'content': 'I think they all look fine!'})()

    scores = retrieval.llm_scores(Unparseable(), 'm', 'q', ['a', 'b'])
    assert list(scores) == [pytest.approx(0.5), pytest.approx(0.5)]


def test_no_test_in_this_suite_can_reach_the_real_runs_directory(smoke_lab):
    # this is an integration test
    """Both halves of the guard the autouse fixture in conftest.py exists
    for. The structural half: whatever redirects `RUNS_DIR` has to apply to
    every test, not to the ones whose author remembered. The behavioural
    half: an actual `run_eval` — which ends in `save_run`, writing to the
    module-level `RUNS_DIR` — deliberately does *not* redirect it here, and
    the repository's real `.runs/` stays untouched anyway."""
    assert evaluate.RUNS_DIR != config.RUNS_DIR, (
        'evaluate.RUNS_DIR still points at the repository .runs/ during tests')

    real = config.RUNS_DIR
    before = {p.name for p in real.glob('*.json')} if real.exists() else set()

    registry, truth = smoke_lab
    cfg = LabConfig(index=IndexConfig(**SMOKE_INDEX),
                    generation=GenerationConfig(answerer='none'))
    evaluate.run_eval(registry, truth, cfg, LAB_SETTINGS,
                      limit=2, balance='difficulty', ragas_mode='off')

    after = {p.name for p in real.glob('*.json')} if real.exists() else set()
    assert after == before, (
        f'the suite wrote {sorted(after - before)} into the real .runs/')


# --- the project's own RAG settings, in one click --------------------------

def test_the_production_preset_is_a_declared_snapshot():
    # this is a convention test
    """The lab no longer shares a repository with the brain, so these values
    are literals in `production_baseline_snapshot.py` and this test pins them — including the
    two deliberate differences a careless re-snapshot would "fix"."""
    preset = config.PRODUCTION_CONFIG
    index, ret = preset['index'], preset['retrieval']
    # chunking: the brain splits at 500 with 100 of overlap, so the lab's
    # honest mirror is fixed-overlap at those exact sizes — not semantic-drift,
    # which the sweep preferred but the brain does not ship.
    assert index['chunker'] == 'fixed-overlap'
    assert index['chunk_chars'] == 500          # retrieval.CHUNK_SIZE
    assert index['overlap'] == 100              # retrieval.CHUNK_OVERLAP
    # and it prepends no situating header
    assert index['contextual'] is False
    assert index['embedder'] == 'sentence-transformers'   # Settings.embedder
    # retrieval: every depth the shipped pipeline uses
    assert ret['k'] == 8                        # retrieval.TOP_K
    assert ret['candidates'] == 40              # retrieval.CANDIDATES
    assert ret['rerank_depth'] == 20            # retrieval.RERANK_DEPTH
    assert ret['grade_threshold'] == 0.4        # retrieval.GRADE_THRESHOLD
    assert ret['grader'] == 'llm'               # Settings.grader
    # and the shape of the pipeline itself: hybrid + RRF, lexical rerank,
    # the Farsi time filter and query expansion on, HyDE and MMR off.
    assert ret['retriever'] == 'hybrid-rrf' and ret['reranker'] == 'lexical'
    assert ret['time_filter'] is True and ret['multi_query'] is True
    assert ret['hyde'] is False and ret['mmr_lambda'] == 1.0
    # a snapshot that does not say when it was taken is a claim about now
    assert baseline.SNAPSHOT_DATE in preset['label']
    # a preset the lab would refuse to run is not a preset
    assert LabConfig.from_dict(preset).validate() == []


def test_options_do_not_offer_a_retired_production_preset(client):
    # this is an integration test
    assert 'production' not in client.get('/api/options').json()


def test_the_two_runners_that_refuse_an_unbacked_run_name_every_backend_too():
    # this is a convention test
    """The panel's hint is one of three places this sentence is written;
    the other two are the entry points a sweep and a judge screen actually
    stop at. Read out of the source rather than triggered: `judged_settings`
    and `screen` answer with `sys.exit`, which would end the test itself."""
    for name in ('sweep.py', 'judgescreen.py'):
        tree = ast.parse((RAGLAB_DIR / 'agents' / 'extra_tools' / name).read_text(encoding='utf-8'))
        refusals = [node.args[0].value for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and getattr(node.func, 'attr', '') == 'exit'
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and 'no LLM backend' in str(node.args[0].value)]
        assert len(refusals) == 1, (name, refusals)
        for provider in config.LLM_PROVIDERS:
            if provider and provider != 'fake':
                assert provider in refusals[0], (name, provider)
