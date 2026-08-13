"""Tests for the RAG lab."""
import ast
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from raglab import (baseline, config, corpus,
                    evaluate, explain, metrics, models, pipeline,
                    ragas_eval, retrieval, sweep, textnorm)
from raglab.config import (EMBEDDERS, RERANKERS, GenerationConfig, IndexConfig,
                            LabConfig, LabSettings, RetrievalConfig)
from raglab.index import IndexRegistry

from conftest import LAB_SETTINGS, RAGLAB_DIR, _finished


# --- evaluation harness ----------------------------------------------------

def test_a_run_writes_one_json_file_and_nothing_else(registry, ground_truth,
                                                     tmp_path, monkeypatch):
    """A run's index, contexts and answers die with the process; the one
    thing that outlives it is a single strict-JSON file. `rglob` rather than
    `glob`, so a stray subdirectory beside the runs cannot hide."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    cfg = LabConfig(index=IndexConfig(chunker='session', embedder='char-hash'),
                    retrieval=RetrievalConfig(k=4),
                    generation=GenerationConfig(answerer='extractive'))
    result = evaluate.run_eval(registry, ground_truth, cfg, LAB_SETTINGS,
                               limit=2, ragas_mode='off')
    assert [p.name for p in tmp_path.rglob('*')] == [f'{result.run_id}.json']
    saved = json.loads((tmp_path / f'{result.run_id}.json').read_text(
        encoding='utf-8'), parse_constant=lambda literal: pytest.fail(
            f'{literal} is not JSON a strict parser accepts'))
    assert saved['run_id'] == result.run_id
    assert saved['config'] and saved['summary'] and saved['rows']


def test_run_eval_scores_a_slice_end_to_end(registry, ground_truth, tmp_path,
                                            monkeypatch):
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    cfg = LabConfig(index=IndexConfig(chunker='message', embedder='char-hash',
                                      contextual=True),
                    retrieval=RetrievalConfig(k=6, reranker='lexical'),
                    generation=GenerationConfig(answerer='extractive'),
                    label='test-slice')
    result = evaluate.run_eval(registry, ground_truth, cfg, LAB_SETTINGS,
                               limit=12, ragas_mode='off')
    assert len(result.rows) == 12
    assert result.summary['overall']['headline'] is not None
    assert result.summary['by_type']
    assert (tmp_path / f'{result.run_id}.json').exists()
    assert all('answer' in row for row in result.rows)


def test_started_at_is_when_the_run_started(registry, ground_truth, tmp_path,
                                            monkeypatch):
    """`started_at` must agree with the run id, stamped at the start — a
    field named for the start that actually holds the finish turns a run
    into a timeline nobody can reconstruct."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    cfg = LabConfig(index=IndexConfig(chunker='session', embedder='char-hash'),
                    retrieval=RetrievalConfig(k=4),
                    generation=GenerationConfig(answerer='extractive'))
    result = evaluate.run_eval(registry, ground_truth, cfg, LAB_SETTINGS,
                               limit=2, ragas_mode='off')
    stamp = result.run_id.split('-')[1]                      # HHMMSS
    assert result.started_at.endswith(f'{stamp[:2]}:{stamp[2:4]}:{stamp[4:]}')


def test_select_questions_strides_across_types(ground_truth):
    picked = evaluate.select_questions(ground_truth, limit=10)
    assert len(picked) == 10
    assert len({q['type'] for q in picked}) > 1, 'a limited run must stay diverse'


def test_a_limited_run_reaches_the_end_of_the_question_set(ground_truth):
    """Striding with `questions[::step][:limit]` silently drops a tail
    whenever the count is not a multiple of the limit. Not a rounding
    detail: the set is grouped by type with the newest appended last, so
    the dropped tail is always the most recently added question type."""
    questions = ground_truth['questions']
    for limit in (5, 10, 20, 25, 40):
        picked = evaluate.select_questions(ground_truth, limit=limit)
        assert len(picked) == limit, limit
        ids = [q['id'] for q in picked]
        assert len(set(ids)) == limit, f'{limit} produced duplicates'
        assert ids[0] == questions[0]['id'], limit
        # The last pick is within one stride of the end, not 16 short of it.
        stride = -(-len(questions) // limit)          # ceil
        assert questions.index(picked[-1]) >= len(questions) - stride, limit


def test_a_limited_run_covers_the_newest_question_type(ground_truth):
    """Habit questions are last in the file, so a limit that cannot reach
    the end cannot measure habit retrieval at all."""
    picked = evaluate.select_questions(ground_truth, limit=20)
    assert any(q['type'] == 'habit' for q in picked)


def test_config_round_trips_through_the_panel_payload():
    cfg = LabConfig.from_dict({'index': {'chunker': 'session', 'unknown': 1},
                               'retrieval': {'k': 3},
                               'generation': {'answerer': 'none'},
                               'label': 'x'})
    assert cfg.index.chunker == 'session' and cfg.retrieval.k == 3
    assert cfg.validate() == []
    assert LabConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_the_lab_names_no_vector_database_at_all():
    """A database the lab cannot name is one it cannot be pointed at by a
    typo, an old shell, or a copied command."""
    settings = LabSettings()
    assert [f for f in vars(settings) if 'chroma' in f or 'database' in f] == []
    with pytest.raises(TypeError):
        LabSettings(chroma_database='lodestar')


def test_the_lab_ignores_a_leftover_chroma_environment(monkeypatch):
    """The board's Chroma stack runs whenever a board does, and a shell that
    ran the old lab commands still exports these; neither may reach the lab."""
    monkeypatch.setenv('RAGLAB_CHROMA_DATABASE', 'lodestar')
    monkeypatch.setenv('BRAIN_CHROMA_URL', 'http://localhost:8001')
    assert 'lodestar' not in repr(config.load_lab_settings())


# --- RAGAS bridge ----------------------------------------------------------

def test_ragas_telemetry_is_disabled_on_import():
    """RAGAS's usage ping blocks for minutes per `evaluate()` call when its
    endpoint is unreachable; importing the bridge must be enough to prevent
    that."""
    from raglab import ragas_eval  # noqa: F401
    assert os.environ.get('RAGAS_DO_NOT_TRACK') == 'true'


def test_ragas_availability_reports_missing_pieces_instead_of_raising():
    from raglab import ragas_eval
    status = ragas_eval.availability(LAB_SETTINGS)
    assert isinstance(status.installed, bool)
    if status.installed:
        assert not status.llm_ready   # no key in LAB_SETTINGS
    assert 'ragas' in status.as_dict()['install_hint']


def test_evidence_texts_are_the_cited_messages_not_the_short_quotes(diary,
                                                                   ground_truth):
    """String-similarity metrics need comparable units, so RAGAS is given
    the whole cited message, which must still contain the quote."""
    sessions = corpus.sessions_by_id(diary)
    question = next(q for q in ground_truth['questions'] if q['answerable'])
    texts = corpus.evidence_texts(sessions, question)
    assert texts
    quote = question['evidence'][0]['quote']
    assert any(quote in text for text in texts)
    assert sum(map(len, texts)) > len(quote)


def test_evidence_texts_fall_back_to_quotes_for_unknown_sessions():
    question = {'evidence': [{'session_id': 'nope', 'message_indices': [0],
                              'quote': 'یه چیزی'}]}
    assert corpus.evidence_texts({}, question) == ['یه چیزی']


def test_json_safe_replaces_undefined_metrics_with_null():
    assert evaluate.json_safe({'a': float('nan'), 'b': [1.0, float('nan')]}) == \
        {'a': None, 'b': [1.0, None]}


def test_ragas_offline_metrics_score_a_retrieval(index, ground_truth):
    pytest.importorskip('ragas')
    pytest.importorskip('rapidfuzz')
    from raglab import ragas_eval
    questions = [q for q in ground_truth['questions'] if q['answerable']][:3]
    pairs = [(q, pipeline.retrieve(index, RetrievalConfig(k=5), q['question_fa'],
                                   q['query_date'])) for q in questions]
    report = ragas_eval.run(pairs, LAB_SETTINGS, index.embedder, mode='offline')
    assert report['n_samples'] == 3, report['notes']
    assert 'non_llm_context_recall' in report['metrics']
    assert 0.0 <= report['metrics']['non_llm_context_recall'] <= 1.0


# --- the four metrics that decide the architecture -------------------------
# Everything the lab measures is reported, but only four RAGAS metrics vote:
# between them they cover retrieval (context precision, recall) and
# generation (faithfulness, answer relevancy) failing to use what it fetched.

def test_the_deciding_metrics_are_exactly_the_four_chosen_ones():
    from raglab import ragas_eval
    assert ragas_eval.DECISION_METRICS == (
        'faithfulness', 'answer_relevancy',
        'llm_context_precision_with_reference', 'context_recall')
    # Everything else stays measured and reported; it simply does not vote.
    assert set(ragas_eval.DECISION_METRICS) < set(ragas_eval.LLM_METRICS)
    assert 'factual_correctness(mode=f1)' not in ragas_eval.DECISION_METRICS


def test_the_decision_score_is_the_unweighted_mean_of_those_four():
    """Unweighted on purpose: any weighting would be a claim about relative
    importance this fixture cannot support."""
    from raglab import ragas_eval
    score = ragas_eval.decision_score({
        'faithfulness': 1.0, 'answer_relevancy': 0.6,
        'llm_context_precision_with_reference': 0.4, 'context_recall': 0.0,
        # Present, reported, and deliberately ignored by the arithmetic.
        'factual_correctness(mode=f1)': 0.0, 'non_llm_context_recall': 1.0,
    })
    assert score == 0.5


def test_the_decision_score_is_undefined_unless_all_four_are_present():
    """A mean over whichever metrics happened to succeed is not comparable
    between runs: an offline run would score on two metrics and outrank a judged
    run scored on four."""
    from raglab import ragas_eval
    assert ragas_eval.decision_score({'faithfulness': 1.0}) is None
    assert ragas_eval.decision_score({}) is None
    assert ragas_eval.decision_score(
        {'faithfulness': 1.0, 'answer_relevancy': 1.0,
         'llm_context_precision_with_reference': 1.0}) is None


def test_the_decision_score_carries_its_own_uncertainty():
    """A ranking of means with no spread cannot say whether it ranked
    anything, so the score ships with the standard error of the
    per-question composite beside it. Computed per question and then across
    questions, not per metric: the four are measured on the same answers and
    are correlated, so averaging four independent standard errors would
    understate the real spread."""
    from raglab import ragas_eval
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
    """A run that could not measure the four reports `n=0` rather than
    omitting the field — a missing key would make the frontend fall back to
    printing the bare mean."""
    from raglab import ragas_eval
    report = ragas_eval.run([], LAB_SETTINGS, None, mode='off')
    assert report['decision_spread'] == {'n': 0, 'mean': None, 'stderr': None}


def test_an_offline_ragas_run_reports_no_decision_score(index, ground_truth):
    """The offline mode cannot measure any of the four, so it must say so
    rather than produce a number that looks comparable to a judged run's."""
    pytest.importorskip('ragas')
    pytest.importorskip('rapidfuzz')
    from raglab import ragas_eval
    questions = [q for q in ground_truth['questions'] if q['answerable']][:2]
    pairs = [(q, pipeline.retrieve(index, RetrievalConfig(k=5), q['question_fa'],
                                   q['query_date'])) for q in questions]
    report = ragas_eval.run(pairs, LAB_SETTINGS, index.embedder, mode='offline')
    assert report['decision'] is None
    assert report['decision_metrics'] == list(ragas_eval.DECISION_METRICS)


def test_the_decision_score_explains_itself_like_every_other_number():
    """It is the number the architecture was chosen by, so of everything on the
    screen it is the one that must not be a bare figure."""
    keys = {measure['key']: measure for measure in explain.measures()}
    decision = keys.get('ragas_decision')
    assert decision, 'the deciding score has no definition'
    assert decision['formula'] and decision['library'] and decision['help']
    for name in ('faithfulness', 'answer relevancy', 'context precision',
                 'context recall'):
        assert name in decision['help'].lower(), name
    assert explain.topics()['metric.ragas_decision']


def test_the_leaderboard_row_carries_the_deciding_score(index, ground_truth):
    """A leaderboard that ranks on a number it does not carry cannot be
    checked against the run it came from."""
    result = evaluate.RunResult(
        run_id='x', label='y', config={}, index={},
        summary={'overall': {}, 'n_questions': 0},
        ragas={'mode': 'llm', 'metrics': {'faithfulness': 0.8},
               'decision': 0.75, 'decision_metrics': [],
               'decision_spread': {'n': 24, 'mean': 0.75, 'stderr': 0.05}})
    assert result.brief()['ragas_decision'] == 0.75
    # The error travels with the mean, or the row it lands in cannot say whether
    # it beat the row below it.
    assert result.brief()['ragas_decision_stderr'] == 0.05


def test_a_row_recorded_before_the_spread_existed_reports_no_error(tmp_path,
                                                                  monkeypatch):
    """An absent error must not be rendered as `± 0`, which would claim the
    run was measured more precisely than ones that carry a real number."""
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
    """`--only A F` and `--final G` select on `label.split()[0]`, so two
    candidates sharing a letter would run silently under the wrong label."""
    letters = [c.label.split()[0] for c in sweep.candidates()]
    assert len(letters) == len(set(letters)), letters
    assert all(len(letter) == 1 for letter in letters), letters


def test_every_candidate_holds_the_embedder_and_both_models_fixed():
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
    """All four deciding metrics need a response. A candidate that retrieved
    without answering would score `None`, drop to the bottom of the ranking as
    if it had lost, and cost a full run to say nothing."""
    assert all(c.generation.answerer == 'llm' for c in sweep.candidates())


def test_every_candidate_validates_before_the_sweep_starts():
    """`run_eval` raises on an invalid config. Candidate H is the eighth row, so
    a typo there would surface after an hour of paid judging."""
    for cfg in sweep.candidates():
        assert cfg.validate() == [], cfg.label


def test_no_two_candidates_are_the_same_configuration():
    """A duplicated row costs ten minutes and reads as reproducibility."""
    seen = {}
    for cfg in sweep.candidates():
        key = json.dumps(replace(cfg, label='').to_dict(), sort_keys=True)
        assert key not in seen, f'{cfg.label} duplicates {seen.get(key)}'
        seen[key] = cfg.label


def test_the_final_run_refuses_to_start_without_a_judge(monkeypatch, tmp_path):
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
# A claim nobody can check is worse than no claim, so each metric carries
# the same four facts (label, formula, library, help) from one registry.

def test_every_reported_metric_has_a_definition():
    """The gate: `aggregate()` can report these keys, so the panel can show them,
    so every one of them has to be explainable."""
    defined = {measure.key for measure in metrics.MEASURES}
    reported = set(metrics.AGGREGATED) | {'headline'}
    assert reported <= defined, reported - defined
    for measure in metrics.MEASURES:
        assert measure.label and measure.short, measure.key
        assert measure.formula and measure.library and measure.help, measure.key


def test_a_metric_states_the_exact_formula_it_computes():
    """Not prose about the idea — the arithmetic, matching the code above it."""
    by_key = {measure.key: measure for measure in metrics.MEASURES}
    assert '|gold ∩ top-k| / |gold|' in by_key['recall'].formula
    assert '1 / rank' in by_key['mrr'].formula
    assert 'log2' in by_key['ndcg'].formula
    # The headline is a weighted sum invented here, so its weights are the formula.
    headline = by_key['headline'].formula
    for weight in ('0.4', '0.3', '0.2', '0.1'):
        assert weight in headline, weight
    assert '0.9' in by_key['quote_recall'].formula      # the fuzzy fallback


def test_every_metric_names_the_library_that_computes_it():
    by_key = {measure.key: measure for measure in metrics.MEASURES}
    assert 'metrics.recall_at_k' in by_key['recall'].library
    assert 'difflib' in by_key['quote_recall'].library
    assert 'difflib' in by_key['answer_similarity'].library
    # A deterministic metric must not claim to be a model, and vice versa.
    assert 'llm' not in by_key['recall'].library.lower()
    assert 'llm' in by_key['key_fact_coverage'].library.lower()


def test_every_metric_says_which_step_it_grades():
    """Same three inks as the panels: a number about retrieval is green wherever
    it appears, so the dashboard means one thing by a colour."""
    steps = {step.key for step in config.STEPS} | {''}
    assert all(measure.step in steps for measure in metrics.MEASURES)
    by_key = {measure.key: measure for measure in metrics.MEASURES}
    assert by_key['recall'].step == 'retrieval'
    assert by_key['ndcg'].step == 'retrieval'
    assert by_key['answer_similarity'].step == 'generation'
    assert by_key['latency_ms'].step == ''      # whole pipeline, no single step


def test_the_ragas_definitions_cover_every_metric_ragas_can_report():
    from raglab import ragas_eval
    defined = {measure.key for measure in ragas_eval.RAGAS_MEASURES}
    reported = set(ragas_eval.OFFLINE_METRICS) | set(ragas_eval.LLM_METRICS)
    assert reported <= defined, reported - defined


def test_a_ragas_metric_carries_ragas_own_class_definition_and_formula():
    """"Faithfulness" is RAGAS's word, not ours, so the panel says whose
    definition it is showing and which class computed it."""
    from raglab import ragas_eval
    by_key = {m.key: m for m in ragas_eval.RAGAS_MEASURES}
    faith = by_key['faithfulness']
    assert 'Faithfulness' in faith.library and 'ragas' in faith.library.lower()
    assert 'claims' in faith.help.lower()
    assert 'supported claims' in faith.formula and '/' in faith.formula
    relevancy = by_key['answer_relevancy']
    assert 'ResponseRelevancy' in relevancy.library
    assert 'cosine' in relevancy.formula.lower()
    f1 = by_key['factual_correctness(mode=f1)']
    assert 'FactualCorrectness' in f1.library and 'F1' in f1.formula
    offline = by_key['non_llm_context_recall']
    assert 'NonLLMContextRecall' in offline.library
    # The offline pair is string distance, not a model — and says so.
    assert 'rapidfuzz' in offline.library and 'llm' not in offline.formula.lower()


def test_a_judged_metric_says_which_model_judged_it():
    """A number produced by a model is a number with variance, and the
    reader has to know which model. The decision score is judged too, being
    a mean of four judged metrics — a composite must not launder its
    inputs' variance by being an average."""
    from raglab import ragas_eval
    judged = set(ragas_eval.LLM_METRICS) | {'ragas_decision'}
    for measure in ragas_eval.RAGAS_MEASURES:
        if measure.key in judged:
            assert 'RAGAS judge' in measure.library, measure.key
        else:
            assert 'no model' in measure.library.lower(), measure.key


def test_no_metric_ships_without_an_explainer():
    """The counterpart of explain.missing() for the knobs: a metric added to
    AGGREGATED or to the RAGAS list without a definition fails here."""
    assert explain.missing_metrics() == []


def test_metric_definitions_join_the_one_help_registry():
    """Homogeneous by construction: the panel has one explainer mechanism, so a
    metric's text lives with the knobs' text under 'metric.<key>'."""
    topics = explain.topics()
    for key in ('metric.recall', 'metric.quote_recall', 'metric.headline',
                'metric.faithfulness', 'metric.non_llm_context_recall'):
        assert topics.get(key), key


# --- the three pipeline steps ----------------------------------------------
# The panel groups and colours every control by the step it belongs to, so
# the step list is a registry the lab owns, not a palette the frontend
# invents. Colours stay in CSS; which step each knob serves is single-sourced
# here.

def test_the_pipeline_steps_are_named_once_in_pipeline_order():
    assert [step.key for step in config.STEPS] == ['index', 'retrieval',
                                                   'generation', 'agent']
    # Two names on purpose: the long one titles a panel, the short one tags a
    # group of models inside another panel, where a whole sentence would not fit.
    assert all(step.label and step.short and step.note for step in config.STEPS)
    assert [step.short for step in config.STEPS] == ['Index', 'Retrieval',
                                                     'Generation', 'Agent']


def test_the_steps_are_exactly_the_config_groups():
    """A step is a config group with a colour, so the two lists cannot drift:
    a fourth group would otherwise render in a panel nobody colours."""
    assert {step.key for step in config.STEPS} == {group for group, _
                                                   in explain.GROUPS}


def test_every_model_role_says_which_step_it_serves():
    steps = {step.key for step in config.STEPS}
    assert all(role.step in steps for role in models.ROLES)
    # The colour cannot disagree with where the value is stored: the step is the
    # group of the field the dropdown writes to.
    assert all(role.step == role.field.split('.')[0] for role in models.ROLES)


def test_every_step_owns_at_least_one_model():
    """A step with no model in it is a legend entry pointing at nothing.
    The index step owns the *embedder* — not a chat role, but a model all
    the same."""
    served = {role.step for role in models.ROLES} | {'index'}
    assert served == {step.key for step in config.STEPS}


def test_a_model_role_is_serialised_with_its_step():
    role = next(r for r in models.ROLES if r.key == 'grade')
    assert role.as_dict()['step'] == 'retrieval'


# --- exporting a run for reading ------------------------------------------
# The leaderboard cannot show what the pipeline did to any one question; the
# export writes that out from a finished run, only from what it stored.

RUN_FIXTURE = {
    'run_id': '20260101-010101-abc123', 'label': 'D wider context k=12',
    'seconds': 671.68, 'started_at': '2026-01-01 01:01:01',
    'config': {'index': {'chunker': 'semantic-drift', 'embedder': 'x',
                         'embed_model': 'y', 'layers': ['chunk', 'habit']},
               'retrieval': {'k': 12, 'retriever': 'hybrid-rrf',
                             'reranker': 'lexical'},
               'generation': {'answerer': 'llm', 'model': 'm'}},
    'index': {'chunks': 732, 'by_layer': {'chunk': 700, 'habit': 5},
              'embed_dim': 1024},
    'summary': {'n_questions': 3, 'overall': {}, 'by_type': {}},
    'ragas': {'mode': 'llm', 'metrics': {'faithfulness': 0.77},
              'decision': 0.6501, 'n_samples': 3,
              'decision_spread': {'n': 3, 'mean': 0.65, 'stderr': 0.04}},
    'rows': [
        {'id': 'q-hb-001', 'type': 'habit', 'difficulty': 'medium',
         'answerable': True, 'abstained': False, 'hit': 1.0, 'recall': 1.0,
         'quote_recall': 1.0, 'ndcg': 0.63, 'precision': 0.2, 'mrr': 0.5,
         'n_contexts': 8, 'context_chars': 4689, 'latency_ms': 21140.4,
         'layers': ['chunk', 'habit'], 'time_scope': None,
         'retrieved_sessions': ['2026-05-16-a', '2025-11-08-a'],
         'answer': 'هفته‌ای سه بار.', 'answer_similarity': 0.31,
         'answer_token_f1': 0.36},
        {'id': 'q-ab-001', 'type': 'abstention', 'difficulty': 'hard',
         'answerable': False, 'abstained': True, 'hit': None, 'recall': None,
         'quote_recall': None, 'n_contexts': 8, 'context_chars': 100,
         'latency_ms': 900.0, 'layers': ['chunk'], 'time_scope': None,
         'retrieved_sessions': [], 'answer': 'پیدا نکردم.'},
        {'id': 'q-sh-001', 'type': 'single-hop', 'difficulty': 'easy',
         'answerable': True, 'abstained': False, 'hit': 0.0, 'recall': 0.0,
         'quote_recall': 0.0, 'n_contexts': 8, 'context_chars': 300,
         'latency_ms': 800.0, 'layers': ['chunk'], 'time_scope': 'تیر',
         'retrieved_sessions': ['2025-01-01-a'], 'answer': 'نمی‌دانم.'},
    ],
}


def test_the_difficulty_table_counts_answers_not_just_retrieval():
    """The run files store no judged grade per question, so "correct" is
    evidence-based: an answerable question counts when the pipeline did not
    refuse *and* reached a gold session; an unanswerable one counts when it
    did refuse."""
    from raglab import export
    table = export.difficulty_rates(RUN_FIXTURE['rows'])
    assert [row['difficulty'] for row in table] == ['easy', 'medium', 'hard']
    easy, medium, hard = table
    assert easy['n'] == 1 and easy['answered'] == 0.0     # retrieved nothing gold
    assert medium['n'] == 1 and medium['answered'] == 1.0
    assert hard['n'] == 1 and hard['answered'] == 1.0     # correctly refused
    # The share is a share of that difficulty, so it needs the count beside it:
    # one hard question at 100% is not a finding.
    assert all('n' in row for row in table)


def test_the_difficulty_table_reports_evidence_separately_from_answers():
    """Retrieval reaching the evidence and the answer using it are different
    failures, and collapsing them hides which half to fix."""
    from raglab import export
    rows = export.difficulty_rates(RUN_FIXTURE['rows'])
    easy = rows[0]
    assert easy['evidence_found'] == 0.0
    assert easy['quotes_in_context'] == 0.0
    # An unanswerable question has no evidence to find, so it must not be
    # averaged in as a miss.
    assert rows[2]['evidence_found'] is None


def test_a_question_page_shows_reference_retrieval_response_and_grades(ground_truth):
    """The four things you need to judge one question, in one file."""
    from raglab import export
    question = next(q for q in ground_truth['questions'] if q['id'] == 'q-sh-001')
    row = next(r for r in RUN_FIXTURE['rows'] if r['id'] == 'q-sh-001')
    page = export.question_page(RUN_FIXTURE, question, row)
    assert question['question_fa'] in page
    assert question['question_en'] in page
    assert question['answer_fa'] in page                  # the reference
    assert question['evidence'][0]['quote'] in page       # and its quote
    assert question['evidence'][0]['session_id'] in page
    assert row['answer'] in page                          # what it replied
    assert '2025-01-01-a' in page                         # what it retrieved
    # Every grade names its own arithmetic, the same rule the dashboard follows.
    assert '|gold ∩ top-k| / |gold|' in page
    assert 'Recall@k' in page
    # And the run it came from, or the page cannot be traced back.
    assert RUN_FIXTURE['run_id'] in page and RUN_FIXTURE['label'] in page


def test_a_question_page_says_which_grades_are_not_per_question(ground_truth):
    """The four deciding metrics are stored as run means only; printed
    unlabelled they would read as that question's own faithfulness."""
    from raglab import export
    question = next(q for q in ground_truth['questions'] if q['id'] == 'q-sh-001')
    row = next(r for r in RUN_FIXTURE['rows'] if r['id'] == 'q-sh-001')
    page = export.question_page(RUN_FIXTURE, question, row)
    assert 'run mean' in page.lower()
    assert 'not per question' in page.lower()


def test_the_export_writes_one_file_per_question_plus_an_index(ground_truth,
                                                              tmp_path):
    from raglab import export
    written = export.write_run(RUN_FIXTURE, ground_truth, tmp_path)
    names = sorted(path.name for path in written)
    assert names == ['README.md', 'q-ab-001.md', 'q-hb-001.md', 'q-sh-001.md']
    index = (tmp_path / 'README.md').read_text(encoding='utf-8')
    assert 'easy' in index and 'medium' in index and 'hard' in index
    assert RUN_FIXTURE['run_id'] in index
    # The index links the files, or a folder of 24 pages is unnavigable.
    assert '(q-sh-001.md)' in index


def test_the_export_never_invents_the_context_text(ground_truth, tmp_path):
    """Runs store the retrieved session ids, not the chunk text, so the page
    says what it has rather than reconstructing chunks by re-running
    retrieval — which would document a different retrieval than the one
    that was graded."""
    from raglab import export
    export.write_run(RUN_FIXTURE, ground_truth, tmp_path)
    page = (tmp_path / 'q-sh-001.md').read_text(encoding='utf-8')
    assert 'chunk text is not stored' in page.lower()



# --- the 49-question sample, balanced across difficulty --------------------
# The four deciding metrics are means over questions, so which questions a run
# scored is part of the measurement. The natural distribution is 29 easy / 57
# medium / 26 hard, and a plain stride hands medium about half of any sample —
# which measures the medium pipeline and reports it as the pipeline.

def test_a_balanced_sample_splits_the_difficulty_bands_as_evenly_as_it_can(
        ground_truth):
    picked = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    assert len(picked) == 49
    counts = {name: sum(1 for q in picked if q['difficulty'] == name)
              for name in config.DIFFICULTIES}
    # 49 does not divide by three; the remainder goes to the earlier bands in
    # DIFFICULTIES order, so the split is 17/16/16 and not "whatever came out".
    assert counts == {'easy': 17, 'medium': 16, 'hard': 16}, counts


def test_a_balanced_sample_that_divides_evenly_is_exactly_equal(ground_truth):
    picked = evaluate.select_questions(ground_truth, limit=51,
                                       balance='difficulty')
    counts = {name: sum(1 for q in picked if q['difficulty'] == name)
              for name in config.DIFFICULTIES}
    assert counts == {'easy': 17, 'medium': 17, 'hard': 17}, counts


def test_a_balanced_sample_is_the_same_questions_every_time(ground_truth):
    """Two candidates are only comparable if they scored the same questions, so
    the selection has to be deterministic rather than merely proportionate."""
    first = evaluate.select_questions(ground_truth, limit=49,
                                      balance='difficulty')
    second = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    assert [q['id'] for q in first] == [q['id'] for q in second]


def test_a_balanced_sample_still_spreads_across_the_question_types(ground_truth):
    """Balancing difficulty must not cost type coverage — habit questions are
    last in the file and were the type a bad stride used to lose entirely."""
    picked = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    types = {q['type'] for q in picked}
    assert len(types) >= 9, types
    assert 'habit' in types


def test_a_balanced_sample_keeps_the_fixture_order(ground_truth):
    """Band-by-band output would make two runs undiffable line by line for no
    reason."""
    picked = evaluate.select_questions(ground_truth, limit=49,
                                       balance='difficulty')
    order = [q['id'] for q in ground_truth['questions']]
    assert [q['id'] for q in picked] == [i for i in order
                                        if i in {q['id'] for q in picked}]


def test_a_band_too_small_for_its_share_does_not_shrink_the_sample():
    """A run asked for N questions must produce N whenever the set holds that
    many; what a small band cannot supply is offered to the others."""
    questions = ([{'id': f'e{i}', 'difficulty': 'easy', 'type': 't'}
                  for i in range(2)]
                 + [{'id': f'm{i}', 'difficulty': 'medium', 'type': 't'}
                    for i in range(20)]
                 + [{'id': f'h{i}', 'difficulty': 'hard', 'type': 't'}
                    for i in range(20)])
    picked = evaluate.select_questions({'questions': questions}, limit=12,
                                       balance='difficulty')
    assert len(picked) == 12
    assert sum(1 for q in picked if q['difficulty'] == 'easy') == 2


def test_the_default_sampling_rule_is_unchanged(ground_truth):
    """The twelve runs already in `.runs/` were strided. Changing the default
    underneath the leaderboard would make those rows incomparable rather than
    merely old — so 'stride' stays the default and the sweep opts in."""
    strided = evaluate.select_questions(ground_truth, limit=24)
    explicit = evaluate.select_questions(ground_truth, limit=24,
                                         balance='stride')
    assert [q['id'] for q in strided] == [q['id'] for q in explicit]


def test_an_unknown_balance_raises_rather_than_silently_striding(ground_truth):
    with pytest.raises(ValueError, match='balance'):
        evaluate.select_questions(ground_truth, limit=10, balance='difficlty')


def test_an_unknown_balance_raises_even_when_there_is_no_limit(ground_truth):
    """Checked after the early return, the validation passed silently on any run
    without a limit — so a typo would only raise on the runs where it happened to
    change something, which is the worst possible place to find it."""
    with pytest.raises(ValueError, match='balance'):
        evaluate.select_questions(ground_truth, balance='difficlty')


def test_a_run_saves_the_questions_it_was_measured_on(registry, ground_truth):
    """Neither the config nor the metric means say which questions produced them,
    so the ids travel with the row. Losing them is how two rows get compared
    across two different samples with nothing to reveal it."""
    cfg = LabConfig(index=IndexConfig(chunker='semantic-drift',
                                      embedder='char-hash', contextual=True),
                    generation=GenerationConfig(answerer='none'))
    result = evaluate.run_eval(registry, ground_truth, cfg, LAB_SETTINGS,
                               limit=9, balance='difficulty', ragas_mode='off')
    selection = result.selection
    assert selection['balance'] == 'difficulty' and selection['limit'] == 9
    assert selection['n'] == 9
    assert len(selection['question_ids']) == 9
    assert selection['by_difficulty'] == {'easy': 3, 'medium': 3, 'hard': 3}
    assert result.as_dict()['selection'] == selection
    # And on the leaderboard row — minus the ids, which would swamp it.
    assert result.brief()['selection']['balance'] == 'difficulty'
    assert 'question_ids' not in result.brief()['selection']


def test_the_sweep_measures_every_candidate_on_the_same_balanced_30():
    """The sample is a property of the sweep, not of the invocation: a row
    measured on a different sample is a different measurement."""
    assert sweep.SWEEP_LIMIT == 30
    assert sweep.SWEEP_BALANCE == 'difficulty'
    assert sweep.SWEEP_BALANCE in config.BALANCES


def test_the_sweep_sample_is_exactly_ten_of_each_band(ground_truth):
    """30 divides by three, so this sample needs no remainder rule at all — the
    bands are equal rather than merely as-equal-as-possible."""
    picked = evaluate.select_questions(ground_truth, limit=sweep.SWEEP_LIMIT,
                                       balance=sweep.SWEEP_BALANCE)
    counts = {name: sum(1 for q in picked if q['difficulty'] == name)
              for name in config.DIFFICULTIES}
    assert counts == {'easy': 10, 'medium': 10, 'hard': 10}, counts


# --- progress: a run that reports nothing is indistinguishable from a hang ---
# With a local judge the judged phase is hours, not minutes, so every phase has
# to say where it is. The callback carries a human detail beside the fraction
# because "0.92" for two hours tells the reader nothing about what is happening.

PROGRESS_CFG = LabConfig(index=IndexConfig(chunker='message', embedder='char-hash'),
                         generation=GenerationConfig(answerer='extractive'),
                         label='progress')


def test_progress_reports_which_question_it_is_on(registry, ground_truth,
                                                  tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    seen = []
    evaluate.run_eval(registry, ground_truth, PROGRESS_CFG, LAB_SETTINGS,
                      limit=4, balance='difficulty', ragas_mode='off',
                      progress=lambda stage, fraction, detail='': seen.append(
                          (stage, round(fraction, 3), detail)))
    scoring = [row for row in seen if row[0] == 'scoring']
    assert len(scoring) == 4, seen
    # The count is the point: "question 3/4" is checkable against the sample the
    # row itself records, where a bare fraction is not.
    assert scoring[2][2].startswith('question 3/4'), scoring
    assert scoring[-1][2].startswith('question 4/4'), scoring
    # And the band, because a slow phase on hard questions is a different fact
    # from a slow phase overall.
    assert any(band in scoring[0][2] for band in config.DIFFICULTIES), scoring


def test_a_two_argument_progress_callback_still_works(registry, ground_truth,
                                                      tmp_path, monkeypatch):
    """The detail is additive. The panel's reporter predates it, and a run must
    not fail because its caller does not want the third argument."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    seen = []
    evaluate.run_eval(registry, ground_truth, PROGRESS_CFG, LAB_SETTINGS,
                      limit=2, ragas_mode='off',
                      progress=lambda stage, fraction: seen.append(stage))
    assert 'scoring' in seen and 'done' in seen


def test_the_judged_phase_reports_calls_as_they_land():
    """The judged phase is the whole wall clock on a local judge. RAGAS scores a
    batch, so without a per-call hook the bar sits at one number for hours."""
    watch = ragas_eval.JudgeWatch(total=6)
    seen = []
    watch.progress = lambda stage, fraction, detail='': seen.append(detail)
    watch.on_llm_end(None)
    watch.on_llm_end(None)
    assert 'judge call 2' in seen[-1]
    assert '~6' in seen[-1], 'the estimate is marked as one, not stated as fact'
    # A judge that makes more calls than estimated must not report >100%.
    for _ in range(20):
        watch.on_llm_end(None)
    assert watch.fraction() <= 1.0


def test_the_job_carries_the_detail_to_whoever_is_polling():
    """The panel polls a job dict, so the detail has to be a field on it — a
    progress line only the terminal sees leaves the two UIs looking hung."""
    from raglab.server import Jobs
    jobs = Jobs()
    captured = {}

    def target(report):
        report('scoring', 0.5, 'question 16/30 · hard')
        captured['snapshot'] = dict(jobs.jobs[jobs.current])
        return {'ok': True}

    job_id = jobs.start('run', target)
    while jobs.jobs[job_id]['state'] == 'running':
        time.sleep(0.01)
    assert captured['snapshot']['detail'] == 'question 16/30 · hard'
    assert captured['snapshot']['stage'] == 'scoring'
    # Present from the start, so a poll landing before the first report reads a
    # blank rather than undefined.
    assert 'detail' in jobs.jobs[job_id]


def test_a_running_job_can_be_cancelled_before_its_next_call():
    """Stopping a run must prevent its next unit of work, not just its polling."""
    from raglab.server import Jobs
    jobs = Jobs()
    started = threading.Event()

    def target(report, cancelled):
        started.set()
        while not cancelled():
            time.sleep(0.001)
        # The normal checkpoint used by indexing and evaluation raises the
        # cooperative cancellation exception as soon as the active call ends.
        report('scoring', 0.5, 'would have made another model call')

    job_id = jobs.start('run', target)
    assert started.wait(timeout=1)
    stopped = jobs.cancel(job_id)
    assert stopped['state'] == 'cancelling'
    for _ in range(100):
        if jobs.get(job_id)['state'] == 'cancelled':
            break
        time.sleep(0.01)
    job = jobs.get(job_id)
    assert job['state'] == 'cancelled'
    assert job['cancel_requested'] is True
    assert '_cancel' not in job


def test_the_terminal_bar_says_stage_fraction_elapsed_and_detail():
    line = sweep.bar('Stage F', 'scoring', 0.5, 'question 16/30 · hard',
                     time.time() - 63)
    assert line.startswith('\r'), 'the line is rewritten in place, not appended'
    assert 'Stage F' in line
    assert '50.0%' in line
    assert '1m03s' in line, line          # elapsed, because a fraction alone
    assert 'question 16/30 · hard' in line   # cannot tell slow from stuck
    filled = line.count('█')
    assert filled == sweep.BAR_WIDTH // 2, filled


def test_a_shorter_detail_cannot_leave_the_tail_of_a_longer_one_behind():
    """Without the padding a redraw leaves stale characters on the line, which
    reads as a stale *number* rather than as a drawing artefact."""
    written = []
    report = sweep.live('Stage A', time.time(),
                        stream=type('S', (), {'write': written.append,
                                              'flush': lambda self: None})())
    report('ragas', 0.94, 'judge call 137 of ~420')
    report('done', 1.0, '')
    assert len(written[0]) == len(written[1])
    assert '137' not in written[1]


def test_the_expected_judge_call_count_scales_with_k():
    """Context precision asks one verdict per retrieved chunk, so k is what
    drives the bill — the estimate has to know that or it is decoration."""
    at_k5 = ragas_eval.expected_judge_calls(n_samples=10, k=5)
    at_k12 = ragas_eval.expected_judge_calls(n_samples=10, k=12)
    assert at_k12 > at_k5
    assert at_k12 - at_k5 == 10 * 7, (at_k5, at_k12)


def test_the_balance_control_is_explained_like_every_other_knob():
    """`explain.missing()` covers config fields; a run-level control has to be
    added to the same registry by hand or it reaches the panel unexplained."""
    assert 'run.balance' in explain.topics()
    assert 'run.difficulty' in explain.topics()


# --- screening a judge before it is allowed to grade ------------------------
# A weak judge does not produce noisy rankings — it produces confident wrong
# ones, so the judge is screened before it is trusted to grade anything.

def test_the_screen_pairs_a_verified_answer_with_one_fabricated_number(
        diary, ground_truth):
    """Built from the ground truth, not hand-authored: a supported claim is a
    question's own verified answer, and its partner is that answer with one
    numeral changed to one the context never states."""
    from raglab import judgescreen
    sessions = corpus.sessions_by_id(diary)
    items = judgescreen.build_items(ground_truth, sessions, pairs=4)
    assert len(items) == 8
    yes = [i for i in items if i.supported]
    no = [i for i in items if not i.supported]
    assert len(yes) == len(no) == 4, 'an unbalanced screen flatters a constant judge'
    for supported, fabricated in zip(yes, no):
        assert supported.question_id == fabricated.question_id
        assert supported.claim != fabricated.claim
        # Word-for-word identical apart from digits: that is what removes the
        # lexical shortcut.
        strip = lambda text: ''.join(c for c in text if not c.isdigit()
                                     and c not in '۰۱۲۳۴۵۶۷۸۹')
        assert strip(supported.claim) == strip(fabricated.claim)


def test_the_screen_measures_how_much_word_overlap_could_explain(diary,
                                                                ground_truth):
    """Reported, not assumed — and it is not zero, which is a deliberate
    trade: correct labels matter more than a screen a word-counter could
    partly game. The check that actually decides is degeneracy, which no
    lexical shortcut can pass."""
    from raglab import judgescreen
    items = judgescreen.build_items(ground_truth, corpus.sessions_by_id(diary),
                                    pairs=6)
    signal = judgescreen.lexical_signal(items)
    assert signal['difference'] is not None
    assert 'blind' in signal
    # Small enough that overlap cannot be the whole story: the fabricated claims
    # still share almost all their vocabulary with the context.
    assert abs(signal['difference']) <= 0.15, signal


def test_the_screen_dates_its_context_the_way_the_pipeline_does(diary,
                                                               ground_truth):
    """Diary messages are spoken and almost never state a date; the date is
    session metadata, so a judge shown bare message text would refuse a
    true claim *for the right reason*. The pipeline under test has the same
    problem and solves it the same way (`IndexConfig.contextual`)."""
    from raglab import judgescreen
    sessions = corpus.sessions_by_id(diary)
    items = judgescreen.build_items(ground_truth, sessions, pairs=6)
    for item in items:
        for line in item.context:
            assert re.match(r'^\[\d{4}-\d{2}-\d{2}\]', line), line


def test_a_screened_claim_is_one_sentence_not_a_whole_answer(diary,
                                                            ground_truth):
    """A reference answer spans several clauses and sessions, so a judge
    asked to entail all of it against one evidence set is right to refuse.
    RAGAS's own faithfulness decomposes a response into atomic statements
    before judging, so an undecomposed paragraph would not resemble the
    real task either."""
    from raglab import judgescreen
    sessions = corpus.sessions_by_id(diary)
    items = judgescreen.build_items(ground_truth, sessions, pairs=6)
    answers = {q['id']: q['answer_fa'] for q in ground_truth['questions']}
    for item in items:
        # One sentence. Not "shorter than the answer": a single-sentence answer
        # legitimately yields a claim of the same length, and asserting length
        # would be testing the fixture's prose rather than the decomposition.
        assert len(textnorm.sentences(item.claim)) == 1, item.id
        assert len(item.claim) <= len(answers[item.question_id]), item.id
        # And it is anchored: it states a number the context also states, so the
        # context can actually settle it either way.
        anchored = [n for n in judgescreen.NUMERAL.findall(item.claim)
                    if textnorm.normalize(n)
                    in textnorm.normalize(' '.join(item.context))]
        assert anchored or not item.supported, item.id


def test_the_fabricated_number_is_one_the_context_never_states(diary,
                                                              ground_truth):
    """Otherwise the claim is labelled unsupported while being arguably
    supported, and the screen would disqualify the judge that got it right."""
    from raglab import judgescreen
    sessions = corpus.sessions_by_id(diary)
    items = judgescreen.build_items(ground_truth, sessions, pairs=6)
    for item in (i for i in items if not i.supported):
        context = textnorm.normalize(' '.join(item.context))
        original = next(i for i in items
                        if i.question_id == item.question_id and i.supported)
        changed = [n for n in judgescreen.NUMERAL.findall(item.claim)
                   if n not in judgescreen.NUMERAL.findall(original.claim)]
        assert changed, item.id
        for numeral in changed:
            assert textnorm.normalize(numeral) not in context, (item.id, numeral)


def test_a_question_that_cannot_be_mutated_cleanly_is_skipped(diary):
    """No mutation is better than a mislabelled one."""
    from raglab import judgescreen
    sessions = corpus.sessions_by_id(diary)
    # No numerals at all, so nothing can be fabricated.
    ground_truth = {'questions': [
        {'id': 'q-x', 'answerable': True, 'answer_fa': 'هیچ عددی اینجا نیست',
         'evidence': [{'quote': 'متن بدون عدد', 'session_id': 'nope',
                       'message_indices': []}]}]}
    assert judgescreen.build_items(ground_truth, sessions, pairs=4) == []


def test_the_screen_reads_a_ragas_shaped_reply_and_nothing_looser():
    """RAGAS asks for nested JSON and retries on malformed output, so a model that
    judges well but writes prose spends its speed advantage on retries. Counting
    a bare 'yes' as an answer here would hide exactly that cost."""
    from raglab import judgescreen
    good = '{"statements": [{"statement": "x", "verdict": 1, "reason": "y"}]}'
    assert judgescreen._verdict(good) == 1
    # A fenced block is a formatting habit, not a failure to answer.
    assert judgescreen._verdict('```json\n{"statements":[{"verdict":0}]}\n```') == 0
    assert judgescreen._verdict('Yes, it is supported.') is None
    assert judgescreen._verdict('{"statements": []}') is None
    assert judgescreen._verdict('{"verdict": 1}') is None
    assert judgescreen._verdict('') is None


def test_a_constant_judge_is_reported_as_degenerate_not_as_fifty_percent():
    """The field that decides. A model answering the same way every time is
    unusable at any accuracy, because it cannot separate two candidates — and on
    a balanced set it posts 0.5, which reads like a merely weak judge."""
    from raglab.judgescreen import Call, score
    calls = [Call(item_id=f'i{i}', supported=i % 2 == 0, verdict=1, parsed=True,
                  seconds=1.0, prompt='p', reply='r') for i in range(8)]
    result = score(calls)
    assert result['degenerate'] is True
    assert result['accuracy'] == 0.5
    assert result['recall_supported'] == 1.0
    assert result['recall_unsupported'] == 0.0


def test_a_judge_that_tracks_the_claim_is_not_flagged_degenerate():
    from raglab.judgescreen import Call, score
    calls = [Call(item_id=f'i{i}', supported=i % 2 == 0,
                  verdict=int(i % 2 == 0), parsed=True, seconds=1.0,
                  prompt='p', reply='r') for i in range(8)]
    result = score(calls)
    assert result['degenerate'] is False and result['accuracy'] == 1.0


def test_unparseable_replies_are_counted_separately_from_wrong_ones():
    """Two different problems with two different fixes: a prompt/format issue and
    a comprehension issue. Folding them together would send you tuning the wrong
    one."""
    from raglab.judgescreen import Call, score
    calls = [Call(item_id='a', supported=True, verdict=1, parsed=True,
                  seconds=1.0, prompt='p', reply='r'),
             Call(item_id='b', supported=False, verdict=None, parsed=False,
                  seconds=1.0, prompt='p', reply='I think maybe')]
    result = score(calls)
    assert result['schema_failures'] == 1
    assert result['n_parsed'] == 1
    assert result['accuracy'] == 1.0, 'accuracy is over what could be graded'


def test_the_screen_refuses_to_run_without_a_backend(monkeypatch):
    """The same guard as the sweep, for the same reason: the fake provider judges
    every claim without failing, and a screen it passed would be a licence."""
    from raglab import judgescreen
    monkeypatch.setattr(judgescreen, 'load_lab_settings', lambda: LAB_SETTINGS)
    with pytest.raises(SystemExit, match='no LLM backend'):
        judgescreen.screen(['whatever:1b'], pairs=1)


def test_the_screen_keeps_every_prompt_and_reply_it_sent():
    """A screen that reported only an accuracy could not be re-read to see
    *how* a model failed — "it was a constant predictor" is a conclusion
    nobody can check from a bare number."""
    from dataclasses import fields

    from raglab.judgescreen import Call
    names = {f.name for f in fields(Call)}
    assert {'prompt', 'reply', 'verdict', 'parsed', 'seconds', 'usage'} <= names


def test_a_remote_slug_is_never_refused_on_the_strength_of_a_listing(monkeypatch):
    """OpenRouter's list is authoritative in one direction only: everything on it
    works, but a slug missing from it may still be valid — the routing suffixes
    (`:free`, `:floor`) do not appear as ids. Blocking runs that used to work is a
    worse failure than the mislabelled row this guard exists to prevent, so the
    refusal is scoped to the local backend, whose tag list *is* authoritative both
    ways."""
    keyed = replace(LAB_SETTINGS, openrouter_api_key='sk-x')
    monkeypatch.setattr(models, 'openrouter_ids',
                        lambda settings: frozenset({'openai/gpt-5-nano'}))
    cfg = LabConfig(generation=GenerationConfig(ragas_model='openai/gpt-5-mini:floor'))
    assert models.provider_problems(cfg, keyed) == []


# ---------------------------------------------------------------------------
# The HTTP surface — resource collections rather than action verbs.
# ---------------------------------------------------------------------------

def test_the_run_and_runs_collision_is_gone(client):
    """The old singular/plural split meant two unrelated things at one
    character apart. The new names are gone rather than aliased, since a
    second name for one thing is the thing this rename was fixing."""
    assert client.post('/api/run', json={}).status_code == 404
    assert client.get('/api/runs').status_code == 404
    assert client.post('/api/index', json={}).status_code == 404
    assert client.post('/api/query', json={'question': 'x'}).status_code == 404


def test_starting_work_creates_a_job_and_says_where_to_watch_it(client):
    """202 rather than 200: the work has been accepted, not done — the response
    body is a receipt, not a result. `Location` points at the job so a caller
    never has to build the polling url by string concatenation."""
    for path, payload in (
            ('/api/indexes', {'index': {'chunker': 'session',
                                        'embedder': 'ascii-hash'}}),
            ('/api/evaluations', {'index': {'chunker': 'session',
                                            'embedder': 'ascii-hash'},
                                  'generation': {'answerer': 'none'},
                                  'limit': 1, 'ragas_mode': 'off'})):
        res = client.post(path, json=payload)
        assert res.status_code == 202, f'{path} -> {res.status_code}'
        job_id = res.json()['job_id']
        assert job_id
        assert res.headers['Location'] == f'/api/jobs/{job_id}'
        # And the url it points at is real.
        assert client.get(res.headers['Location']).status_code == 200


def test_evaluations_lists_and_fetches_the_same_resource(client):
    """One noun, three operations, no second spelling for any of them."""
    assert 'runs' in client.get('/api/evaluations').json()
    assert client.get('/api/evaluations/no-such-run').status_code == 404


def test_a_query_is_a_job_like_its_sibling_collections(client):
    """A query is accepted as a job — 202, a Location to poll,
    stage/fraction/detail while it runs — like /api/indexes and
    /api/evaluations, for the same reason: the work can outlive anything a
    spinner honestly promises."""
    res = client.post('/api/queries', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
        'retrieval': {'retriever': 'dense', 'k': 2},
        'generation': {'answerer': 'none'},
        'question': 'وام مسکن'})
    assert res.status_code == 202
    job_id = res.json()['job_id']
    assert res.headers['Location'] == f'/api/jobs/{job_id}'
    job = _finished(client, job_id)
    assert job['kind'] == 'query'
    assert job['state'] == 'done', job.get('error')
    assert 'contexts' in job['result'] and 'diagnostics' in job['result']
    # The preconditions still refuse synchronously, and still say which one:
    # a bad payload is a 400 the panel shows at once, never a job that dies.
    assert client.post('/api/queries', json={}).status_code == 400


def test_the_query_job_hands_its_reporter_to_the_index_build(monkeypatch):
    """If the job does not pass its reporter down to the registry, the bar
    sits on 'starting 0%' for the whole implicit index build."""
    from fastapi.testclient import TestClient

    from raglab.server import create_app
    seen = {}
    original = IndexRegistry.get

    def spy(self, cfg, progress=None, force=False):
        seen['progress'] = progress
        return original(self, cfg, progress=progress, force=force)

    monkeypatch.setattr(IndexRegistry, 'get', spy)
    fresh = TestClient(create_app())
    res = fresh.post('/api/queries', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
        'retrieval': {'retriever': 'bm25', 'k': 2},
        'generation': {'answerer': 'none'},
        'question': 'وام مسکن'})
    assert res.status_code == 202
    job = _finished(fresh, res.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    assert callable(seen.get('progress'))


def test_a_second_job_is_refused_in_readable_english(client):
    """The refusal read 'a index job is still stopping' — wrong article, and
    'stopping' for a job that is running. A message describing the wrong state
    sends the reader looking for a bug that is not there."""
    first = client.post('/api/indexes', json={
        'index': {'chunker': 'message', 'embedder': 'token-hash'}})
    assert first.status_code == 202
    second = client.post('/api/indexes', json={
        'index': {'chunker': 'turn-pair', 'embedder': 'token-hash'}})
    if second.status_code == 409:
        detail = second.json()['detail']
        assert 'a index' not in detail
        assert 'an index job is already running' in detail
    # Drained before returning, since the client is module-scoped: leaving a
    # job running here would hand the next test a spurious 409.
    for res in (first, second):
        if res.status_code == 202:
            _finished(client, res.json()['job_id'])


def test_jobs_index_lists_runs_with_their_config(client):
    """The Inspector follows the lab by polling this index, so it has to
    carry the config the job actually ran — `LabConfig`'s own normalised
    form, not the raw posted body — and nothing heavier than
    id/kind/state/config."""
    posted = client.post('/api/indexes', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash'}})
    assert posted.status_code == 202
    job = _finished(client, posted.json()['job_id'])
    assert job['state'] == 'done', job.get('error')

    entries = client.get('/api/jobs').json()['jobs']
    assert entries, 'expected at least one job listed'
    newest = entries[0]
    assert newest['id'] == job['id']
    assert newest['kind'] == 'index'
    assert newest['config']['index']['chunker'] == 'session'
    assert newest['config']['index']['embedder'] == 'ascii-hash'
    assert 'result' not in newest and '_cancel' not in newest

    chunks_total = sum(len(g['chunks'])
                       for g in job['result']['chunks_by_session'])
    assert chunks_total == job['result']['chunks']


def test_a_hierarchical_build_reports_the_summaries_it_wrote(client):
    """The Inspector holds no index of its own — it renders what a job
    returned — so a build that wrote summary rows and reported only its
    leaves would make them unreachable. `chunks` counts every row in the
    index, so leaves plus summaries have to sum back to it."""
    posted = client.post('/api/indexes', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash',
                  'hierarchy': 'metadata', 'summarizer': 'centroid'}})
    assert posted.status_code == 202
    job = _finished(client, posted.json()['job_id'], timeout=120.0)
    assert job['state'] == 'done', job.get('error')
    result = job['result']

    leaves = sum(len(g['chunks']) for g in result['chunks_by_session'])
    summaries = result['summaries']
    assert summaries, 'a metadata hierarchy over this corpus has groups to summarise'
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

    # a flat build says "none", rather than leaving the key out and making the
    # Inspector guess whether this lab is simply older than the feature
    flat = client.post('/api/indexes', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash'}})
    flat_job = _finished(client, flat.json()['job_id'], timeout=120.0)
    assert flat_job['state'] == 'done', flat_job.get('error')
    assert flat_job['result']['summaries'] == []


# ---------------------------------------------------------------------------
# The panel's two usability guarantees, held by the served data rather than by
# either frontend — a rule copied into two panels is a rule that will disagree.
# ---------------------------------------------------------------------------

def test_every_option_list_leads_with_the_default():
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
    """Each case is a knob the pipeline would ignore, so leaving it editable
    invites tuning a number that does nothing. `semantic-drift` is
    deliberately in the *enabled* set for chunk_chars: unlike
    message/turn-pair/session it genuinely reads it, as a max_chars cap."""
    def state(cfg):
        return config.dependency_state(cfg.to_dict())

    drift = state(LabConfig(index=IndexConfig(chunker='semantic-drift')))
    assert drift['index.chunk_chars']['enabled']
    assert not drift['index.overlap']['enabled']

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
    """A greyed-out control with no reason is indistinguishable from a broken
    one. Every rule carries the sentence the panel shows."""
    for key, rule in config.DEPENDENCIES.items():
        assert rule['reason'], f'{key} has no reason'
        assert not rule['reason'].endswith('.'), (
            f'{key}: the panel completes "disabled because …", so no full stop')


def test_the_panel_is_served_the_dependency_rules(client):
    """Both frontends read this rather than each keeping a copy."""
    served = client.get('/api/options').json()['dependencies']
    assert served['index.overlap']['on'] == ['fixed-overlap']
    assert 'semantic-drift' in served['index.chunk_chars']['on']


def test_the_embedder_hints_render_in_the_same_order_as_the_embedders():
    """The standalone panel builds its embedder dropdown from
    EMBEDDER_HINTS, not from EMBEDDERS — two lists describing one set of
    choices have to agree on order or the panel disagrees with itself about
    what is recommended."""
    from raglab.embedding import EMBEDDER_HINTS

    assert [hint.kind for hint in EMBEDDER_HINTS] == list(config.EMBEDDERS)


def test_no_hint_still_calls_a_hash_embedder_the_brain_default():
    """`hash` is retired in production *by name*, so a hint calling
    `ascii-hash` "the brain default" would describe a configuration that
    now raises at boot."""
    from raglab.embedding import EMBEDDER_HINTS

    for hint in EMBEDDER_HINTS:
        if hint.kind.endswith('-hash'):
            assert 'brain default' not in hint.label, hint.label


# ---------------------------------------------------------------------------
# Two regression reproductions: they encode the correct behaviour and fail
# against the code as it stood before the fix.
# ---------------------------------------------------------------------------

def test_a_gate_whose_model_call_fails_does_not_silently_pass_everything():
    """0.5 clears the default 0.4 threshold, so an unreachable model must
    not turn `grader='llm'` into a silent no-op — a row labelled
    `grader=llm` that was measured ungated is the one artefact this lab
    must never produce. The parse fallback is a different thing and must
    survive: a line the model wrote that we could not read is genuinely
    'no opinion'."""
    class Unreachable:
        def invoke(self, messages, **kwargs):
            raise ConnectionError('the model daemon is not running')

    with pytest.raises(Exception) as caught:
        retrieval.llm_scores(Unreachable(), 'm', 'q', ['a', 'b', 'c'])
    # And it names the cause, so the reader is not left guessing which stage
    # went missing.
    assert 'not running' in str(caught.value) or 'grade' in str(caught.value).lower()

    # Unchanged: a reply that arrives but cannot be parsed is still no opinion.
    class Unparseable:
        def invoke(self, messages, **kwargs):
            return type('Reply', (), {'content': 'I think they all look fine!'})()

    scores = retrieval.llm_scores(Unparseable(), 'm', 'q', ['a', 'b'])
    assert list(scores) == [pytest.approx(0.5), pytest.approx(0.5)]


def test_running_an_evaluation_leaves_the_repositorys_runs_directory_alone(
        registry, ground_truth):
    """`run_eval` ends in `save_run`, which writes to the module-level
    RUNS_DIR. This test deliberately does *not* redirect it itself — that
    the real directory stays untouched anyway is exactly what the autouse
    fixture in conftest.py is under test for."""
    real = config.RUNS_DIR
    before = {p.name for p in real.glob('*.json')} if real.exists() else set()

    cfg = LabConfig(index=IndexConfig(chunker='session', embedder='ascii-hash'),
                    generation=GenerationConfig(answerer='none'))
    evaluate.run_eval(registry, ground_truth, cfg, LAB_SETTINGS,
                      limit=2, balance='difficulty', ragas_mode='off')

    after = {p.name for p in real.glob('*.json')} if real.exists() else set()
    assert after == before, (
        f'the suite wrote {sorted(after - before)} into the real .runs/')


def test_no_test_in_this_suite_can_reach_the_real_runs_directory():
    """The structural half. Whatever redirects RUNS_DIR has to apply to every
    test, not to the ones whose author remembered — otherwise this returns the
    next time someone adds a case that calls run_eval."""
    assert evaluate.RUNS_DIR != config.RUNS_DIR, (
        'evaluate.RUNS_DIR still points at the repository .runs/ during tests')


def test_both_run_routes_screen_the_models_the_backend_serves(client, monkeypatch):
    """`/api/evaluations` refused a model the active backend does not serve;
    `/api/queries` ran it. Two routes over the same pipeline disagreeing about
    which configs are legal is a bug on its own, and it got worse the moment a
    dead grade stage started raising: the panel's fastest feedback loop would
    answer a bare 500 where the slow one answers a 400 naming the model."""
    monkeypatch.setattr(models, 'provider_problems',
                        lambda cfg, settings: ['model "qwen3.5:2b" is not served'])
    payload = {'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
               'generation': {'answerer': 'none'}, 'question': 'چه خبر؟'}
    for path in ('/api/queries', '/api/evaluations'):
        res = client.post(path, json=payload)
        assert res.status_code == 400, f'{path} -> {res.status_code}'
        assert 'qwen3.5:2b' in res.json()['detail'], path


def test_a_query_whose_gate_cannot_reach_its_model_says_so(client, monkeypatch):
    """Refusing to score is only an improvement if the refusal reaches the
    caller readably: the job's error must name the stage that went missing,
    or the reader blames retrieval for what the grader did."""
    def unreachable(*args, **kwargs):
        raise ConnectionError('the model daemon is not running')

    monkeypatch.setattr(retrieval, 'lab_chat', unreachable)
    res = client.post('/api/queries', json={
        'question': 'چه خبر؟',
        'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
        'retrieval': {'k': 4, 'grader': 'llm', 'grade_threshold': 0.4},
        'generation': {'answerer': 'none'}})
    assert res.status_code == 202, res.status_code
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'error'
    assert 'grade' in job['error'].lower() and 'not running' in job['error']


# --- retrieval on its own, and the shipped assistant's own settings ---------
# The panel could build an index and score a full judged run, but it had no way
# to do the middle step alone: retrieve for the questions an experiment is
# about, look at what came back, change one knob, look again. That is the loop
# the Inspector (:9003) exists to serve, and it needs the lab to offer both a
# retrieval-only run over the *selected* questions and a one-click preset that
# is the shipped assistant rather than a taste.

def test_the_production_preset_is_a_declared_snapshot():
    """The lab no longer shares a repository with the brain, so these values
    are literals in `baseline.py` and this test pins them — including the
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


def test_the_panel_serves_the_production_preset_for_its_button(client):
    """Served rather than written into the frontend, for the reason the mode
    dropdown is: a preset kept in a browser is a preset that will drift from
    the brain it claims to mirror."""
    assert client.get('/api/options').json()['production'] == config.PRODUCTION_CONFIG


def test_retrieval_only_covers_exactly_the_experiment_questions(client,
                                                                ground_truth):
    """Retrieve, for the questions the eval card has selected, and nothing
    more: no answering, no judging, no run file. The selection has to be the
    *same* selection `/api/evaluations` would score, or the Inspector shows
    retrieval for questions the numbers were never about."""
    picked_type = ground_truth['questions'][0]['type']
    payload = {'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
               'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'none',
                             'grader': 'none', 'k': 3, 'rerank_depth': 20,
                             'time_filter': False, 'multi_query': False},
               'types': [picked_type], 'limit': 2, 'balance': 'stride'}
    res = client.post('/api/retrievals', json=payload)
    assert res.status_code == 202, res.text
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    assert job['kind'] == 'retrieve'

    result = job['result']
    expected = evaluate.select_questions(ground_truth, [picked_type], 2,
                                        None, 'stride')
    assert [q['question_id'] for q in result['questions']] == \
        [q['id'] for q in expected]
    assert result['selection']['n'] == 2

    # The chunks it retrieved *from* travel with it. A run builds its index
    # implicitly, so without this the Inspector's chunks window would keep
    # showing whatever index job was last pressed — a different chunker than the
    # one that produced these rows, with nothing on screen saying so.
    groups = result['chunks_by_session']
    assert sum(len(g['chunks']) for g in groups) == result['index']['chunks']

    first = result['questions'][0]
    assert first['question_fa'] == expected[0]['question_fa']
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


def test_a_traced_evaluation_scores_identically_and_leaves_traces_off_disk(
        client, monkeypatch, tmp_path, registry, ground_truth):
    """Two things must stay true: the scores may not move, since tracing is
    a recording of the same retrieval, not a different one; and the traces
    may not reach `.runs/`, the leaderboard's durable artifact, with data
    no score is computed from."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    payload = {'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
               'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'none',
                             'grader': 'none', 'k': 3, 'rerank_depth': 20,
                             'time_filter': False, 'multi_query': False},
               'generation': {'answerer': 'extractive'},
               'limit': 2, 'balance': 'stride', 'ragas_mode': 'off'}
    res = client.post('/api/evaluations', json=payload)
    assert res.status_code == 202, res.text
    job = _finished(client, res.json()['job_id'], timeout=120.0)
    assert job['state'] == 'done', job.get('error')

    result = job['result']
    assert len(result['rows']) == 2
    traces = result['traces']
    assert [t['question_id'] for t in traces] == [row['id'] for row in result['rows']]
    assert all(t['trace']['candidates'] for t in traces)

    # The chunks this run retrieved from, for the same reason `/api/retrievals`
    # carries them: an evaluation builds its index implicitly and creates no
    # index job, so this is the only way the Inspector can show the chunks the
    # scores were actually computed over instead of an unrelated earlier build.
    groups = result['chunks_by_session']
    assert sum(len(g['chunks']) for g in groups) == result['index']['chunks']

    saved = json.loads((tmp_path / f"{result['run_id']}.json").read_text(
        encoding='utf-8'))
    assert 'traces' not in saved, 'traces must not reach the run file'
    assert 'chunks_by_session' not in saved, 'chunk text must not reach the run file'
    assert saved['rows'] == result['rows']

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

    cfg = LabConfig.from_dict(payload)
    untraced = evaluate.run_eval(registry, ground_truth, cfg, LAB_SETTINGS,
                                 limit=2, balance='stride', ragas_mode='off')
    assert not untraced.traces, 'trace=False must record nothing'
    assert scores(untraced.rows) == scores(result['rows'])
    assert scores(untraced.summary) == scores(result['summary'])


# --- the experiment ledger (raglab.db) -------------------------------------

# A real SQLite file on a temp path.
def test_every_experiment_the_lab_runs_lands_in_the_ledger(client, tmp_path,
                                                           monkeypatch):
    """Three experiments, three rows: the ledger records every job the lab
    *finishes*, which is what makes "what have I already tried?" a
    question with an answer after the process that tried it is gone."""
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    index = {'chunker': 'session', 'embedder': 'ascii-hash'}
    retrieval_cfg = {'retriever': 'hybrid-rrf', 'reranker': 'none',
                     'grader': 'none', 'k': 3, 'rerank_depth': 20,
                     'time_filter': False, 'multi_query': False}

    built = client.post('/api/indexes', json={'index': index})
    assert _finished(client, built.json()['job_id'])['state'] == 'done'
    got = client.post('/api/retrievals', json={
        'index': index, 'retrieval': retrieval_cfg, 'limit': 2,
        'balance': 'stride'})
    assert _finished(client, got.json()['job_id'])['state'] == 'done'
    ran = client.post('/api/evaluations', json={
        'index': index, 'retrieval': retrieval_cfg,
        'generation': {'answerer': 'extractive'}, 'label': 'the ledger',
        'limit': 2, 'balance': 'stride', 'ragas_mode': 'off'})
    run_job = _finished(client, ran.json()['job_id'], timeout=120.0)
    assert run_job['state'] == 'done', run_job.get('error')

    rows = client.get('/api/experiments').json()['experiments']
    # Newest first, like every other listing the lab serves.
    assert [row['kind'] for row in rows[:3]] == ['run', 'retrieve', 'index']
    evaluation, retrieved, build = rows[0], rows[1], rows[2]

    # An evaluation is identified by its run id, never by its job id: the ledger
    # row and the JSON file the leaderboard reads are then the same measurement,
    # each checkable against the other.
    assert evaluation['experiment_id'] == run_job['result']['run_id']
    assert evaluation['label'] == 'the ledger'
    assert evaluation['n_questions'] == 2 and evaluation['state'] == 'done'
    assert evaluation['seconds'] > 0
    # Recorded before the job goes terminal, so a follower that sees 'done' can
    # never look for the row and miss it.
    assert evaluation['started_at']
    # `ragas_mode='off'` judged nothing, and an unjudged row carries no score
    # rather than a zero — the rule the leaderboard already keeps, because a
    # fabricated 0.0 would rank below every real row and read as a measurement.
    assert evaluation['decision'] is None
    assert evaluation['decision_stderr'] is None

    # A retrieval scored nothing either, but it did choose a sample, and which
    # questions it covered is the whole point of having run it.
    assert retrieved['n_questions'] == 2 and retrieved['decision'] is None
    # An index build has no sample at all: it is a fact about the corpus.
    assert build['n_questions'] == 0 and build['decision'] is None

    # Every row says which index it was over, so the panel's table needs no
    # per-kind branch to render one.
    for row in rows[:3]:
        assert row['chunker'] == 'session'
        assert row['embedder'] == 'ascii-hash'
        assert row['experiment_id']

    # But a build's row stops there. Its job config carries a whole LabConfig, so
    # the retrieval group is populated with defaults the panel happened to be
    # showing and no part of a build reads — recorded, they would put a reranker
    # on a row that never retrieved anything, and a reader comparing rows would
    # attribute a chunk count to it. Same reason `provider` is blank: no chat
    # model is involved in chunking, not even for contextual headers.
    assert build['retriever'] == '' and build['reranker'] == ''
    assert build['grader'] == '' and build['answerer'] == ''
    assert build['provider'] == ''
    # The two that did retrieve say so, and say where the calls went — the one
    # field that separates a measurement from a rehearsal.
    assert retrieved['retriever'] == 'hybrid-rrf'
    assert evaluation['answerer'] == 'extractive'
    assert evaluation['provider'] == 'fake', 'the resolved backend, not the ask'

    assert (tmp_path / 'raglab.db').exists(), 'the ledger is one SQLite file'


def test_the_ledger_explains_a_row_without_storing_the_corpus(client, tmp_path,
                                                              monkeypatch):
    """"With all the details" means the details of the *experiment*. The
    chunk text is not one: it is byte-identical across every experiment
    sharing a fingerprint and rebuilt exactly by re-running the build, so
    storing it per row would store the whole corpus once per experiment."""
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    index = {'chunker': 'session', 'embedder': 'ascii-hash'}
    retrieval_cfg = {'retriever': 'hybrid-rrf', 'reranker': 'none',
                     'grader': 'none', 'k': 3, 'rerank_depth': 20,
                     'time_filter': False, 'multi_query': False}
    ran = client.post('/api/evaluations', json={
        'index': index, 'retrieval': retrieval_cfg,
        'generation': {'answerer': 'extractive'}, 'limit': 2,
        'balance': 'stride', 'ragas_mode': 'off'})
    job = _finished(client, ran.json()['job_id'], timeout=120.0)
    assert job['state'] == 'done', job.get('error')
    run_id = job['result']['run_id']

    stored = client.get(f'/api/experiments/{run_id}').json()
    assert stored['experiment_id'] == run_id
    detail = stored['detail']
    assert detail['config']['index']['chunker'] == 'session'
    assert detail['config']['retrieval']['k'] == 3
    assert detail['summary'] == job['result']['summary']
    assert [row['id'] for row in detail['rows']] == \
        [row['id'] for row in job['result']['rows']]
    assert detail['selection']['n'] == 2
    assert 'chunks_by_session' not in detail

    # A retrieval's detail is its traces: the ranks at every step are the only
    # thing it produced, so dropping them would leave a row that records that
    # something ran and nothing about what it found.
    got = client.post('/api/retrievals', json={
        'index': index, 'retrieval': retrieval_cfg, 'limit': 2,
        'balance': 'stride'})
    retrieval_job = _finished(client, got.json()['job_id'])
    assert retrieval_job['state'] == 'done', retrieval_job.get('error')
    newest = client.get('/api/experiments').json()['experiments'][0]
    kept = client.get(f"/api/experiments/{newest['experiment_id']}").json()['detail']
    assert kept['questions'][0]['trace']['candidates']
    assert 'chunks_by_session' not in kept

    assert client.get('/api/experiments/no-such-experiment').status_code == 404


def test_a_ledger_that_cannot_be_written_does_not_lose_the_experiment(
        client, monkeypatch):
    """A judged run costs hours, and an unwritable database must not be
    able to turn one into an error the panel reports over a result nobody
    can read — the same call `ragas_eval.JudgeWatch` makes about its
    progress counter."""
    from raglab import ledger

    def refuse(*_args, **_kwargs):
        raise sqlite3.OperationalError('unable to open database file')

    monkeypatch.setattr(ledger, 'connect', refuse)
    res = client.post('/api/indexes', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash'}})
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    assert job['result']['chunks'] > 0


def test_the_ledger_is_not_kept_beside_the_code_that_writes_it():
    """Where a `.db` goes is a settled question, and the answer is not
    "next to the code that writes it" — a durable record inside `src/`
    reads as build output and is the first thing a clean-up deletes."""
    from raglab import ledger

    default = ledger.db_path(env={})
    assert default == config.ROOT / 'databases' / 'raglab.db'
    assert 'src' not in default.parts
    # Overridable, which is what lets the suite guard itself in conftest.
    assert ledger.db_path(env={'RAGLAB_DB': '/tmp/x.db'}) == Path('/tmp/x.db')


# --- the project's own RAG settings, in one click --------------------------

def test_the_two_runners_that_refuse_an_unbacked_run_name_every_backend_too():
    """The panel's hint is one of three places this sentence is written;
    the other two are the entry points a sweep and a judge screen actually
    stop at. Read out of the source rather than triggered: `judged_settings`
    and `screen` answer with `sys.exit`, which would end the test itself."""
    for name in ('sweep.py', 'judgescreen.py'):
        tree = ast.parse((RAGLAB_DIR / name).read_text(encoding='utf-8'))
        refusals = [node.args[0].value for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and getattr(node.func, 'attr', '') == 'exit'
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and 'no LLM backend' in str(node.args[0].value)]
        assert len(refusals) == 1, (name, refusals)
        for provider in config.LLM_PROVIDERS:
            if provider and provider != 'fake':
                assert provider in refusals[0], (name, provider)
