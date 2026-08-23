"""Archive fixtures: one small complete archive, and the ladder.

`completed_archive` is the single example the validation contracts mutate.
The ladder below is a different instrument: archives that each add one
stage's evidence to the one before it — knobs, then what was indexed, then
what was retrieved, then what was generated — so export and import are
exercised at every shape the format actually takes, not only the full one.

Four of those rungs stack (`SPINE`). The fifth hangs off the fourth rather
than above it: `scored-without-traces` is the generated rung with its traces
removed, which is the shape of every evaluation recorded before the export
route existed — rows, judged metrics and a selection, no trace kept. It is a
real shape of the format, not a degenerate one, so it belongs on the ladder;
it is simply not a superset of the rung below, which is why the monotonicity
test walks `SPINE` and checks this rung against `generated` by name.
"""
import copy
import json

from raglab.configuration.lab_config import (
    GenerationConfig,
    IndexConfig,
    LabConfig,
    RetrievalConfig)


def completed_archive(run_id: str = 'imported-run-001') -> dict:
    config = json.loads(json.dumps(LabConfig(
        index=IndexConfig(dataset='smoke-mini', chunker='session',
                          embedder='token-hash'),
        retrieval=RetrievalConfig(grader='none'),
        generation=GenerationConfig(answerer='extractive')).to_dict()))
    index_stats = {
        'collection': 'raglab-test', 'chunks': 1, 'leaves': 1,
        'avg_chars': 8, 'p95_chars': 8, 'embed_dim': 8,
        'build_seconds': 0.0, 'reused': False,
    }
    return {
        'format': 'raglab-experiment',
        'version': 1,
        'settings': {
            'config': config,
            'ui': {'mode': '', 'ragas_mode': 'offline', 'limit': 1,
                   'ragas_limit': 0, 'types': ['single-hop']},
        },
        'evaluation': {
            'execution': {'provider': 'fake', 'models': {}},
            'metric_catalogue': [{
                'key': 'recall', 'label': 'Recall@k',
                'short': 'evidence found', 'step': 'retrieval',
                'formula': 'gold found / gold', 'library': 'test fixture',
                'help': 'retrieval coverage',
            }],
            'stage_results': {
                'index': {'statistics': index_stats, 'metrics': {}},
                'retrieval': {'metrics': {'recall': 1.0}},
                'generation': {'metrics': {}},
                'overall': {'metrics': {}},
            },
            'result': {
                'run_id': run_id, 'label': 'imported experiment',
                'config': copy.deepcopy(config), 'dataset': 'smoke-mini',
                'index': copy.deepcopy(index_stats),
                'summary': {
                    'overall': {'recall': 1.0},
                    'by_type': {'single-hop': {'n': 1, 'recall': 1.0}},
                    'by_difficulty': {'easy': {'n': 1, 'recall': 1.0}},
                    'n_questions': 1,
                },
                'rows': [{'id': 'q1', 'type': 'single-hop', 'difficulty': 'easy',
                          'recall': 1.0, 'latency_ms': 12.0, 'n_contexts': 1,
                          'abstained': False}],
                'ragas': {}, 'seconds': 0.2,
                'started_at': '2026-08-19 12:00:00', 'notes': [],
                'selection': {'balance': 'stride', 'limit': 1, 'n': 1,
                              'by_difficulty': {'easy': 1},
                              'question_ids': ['q1']},
            },
            'inspector': {
                'dataset': {
                    'id': 'smoke-mini',
                    'corpus': {'meta': {'language': 'en'}, 'persona': {},
                               'threads': [], 'habits': {},
                               'sessions': [{'session_id': 's1',
                                             'date': '2026-08-19',
                                             'messages': [{'role': 'user',
                                                           'content': 'evidence'}]}]},
                    'ground_truth': {
                        'meta': {'corpus': 'smoke-mini',
                                 'query_date': '2026-08-19'},
                        'questions': [{'id': 'q1', 'type': 'single-hop',
                                       'difficulty': 'easy', 'answerable': True,
                                       'question_fa': 'question',
                                       'question_en': 'question',
                                       'answer_fa': 'answer',
                                       'key_facts': ['fact'],
                                       'evidence': [{'session_id': 's1',
                                                     'message_indices': [0],
                                                     'quote': 'evidence'}]}]},
                },
                'chunks_by_session': [{'session_id': 's1', 'date': '2026-08-19',
                                       'chunks': [{'id': 'c1', 'text': 'evidence'}]}],
                'summaries': [],
                'traces': [{'question_id': 'q1', 'question_fa': 'question',
                            'question_en': 'question', 'type': 'single-hop',
                            'difficulty': 'easy', 'answerable': True,
                            'gold_available': 1,
                            'trace': {'candidates': [{
                                'chunk_id': 'c1', 'text': 'evidence',
                                'session_id': 's1', 'date': '2026-08-19',
                                'layer': 'leaf', 'level': 0, 'group_id': '',
                                'members': 0, 'dense_rank': 1, 'bm25_rank': 1,
                                'fused_rank': 1, 'retrieval_score': 1.0,
                                'rerank_score': 1.0, 'grade_score': None,
                                'kept': True, 'gold': True,
                                'gold_spans': [[0, 8]],
                            }]}}],
            },
        },
    }


# --- the four-rung ladder ---------------------------------------------------
# Every knob moved off its default, deliberately and by hand. A round trip over
# a config left at its defaults proves nothing: a knob the codec drops entirely
# still reads back correct, because the value it fell back to is the value it
# started at. Only a config where nothing is default can tell "carried" from
# "reconstructed", which is why `test_archive_round_trip.py` asserts against the
# dataclasses that this names every knob and moves every one of them.
#
# Choices that are not free: `dataset` must match the archived corpus id and the
# id grammar; `embedder` must stay model-backed or `embed_model` is blanked by
# `IndexConfig.normalized()` and the archive stops being its own fixed point;
# `hierarchy` is `metadata` because it is the one grouping that needs no package,
# so this fixture cannot fail on a leaner installation than the author's.
SHIFTED_CONFIG = {
    'label': 'every knob moved off its default',
    'index': {
        'dataset': 'smoke-mini', 'chunker': 'fixed-overlap', 'chunk_chars': 384,
        'overlap': 64, 'contextual': False, 'embedder': 'fastembed',
        'embed_model': 'shifted-embed-model', 'hierarchy': 'metadata',
        'graph_source': 'knn', 'graph_knn': 5, 'granularity': 1.5,
        'hierarchy_levels': 2, 'min_group': 4, 'summarizer': 'lead-idf',
    },
    'retrieval': {
        'retriever': 'dense', 'k': 6, 'candidates': 24, 'rrf_k': 45,
        'time_filter': False, 'multi_query': False, 'hyde': True,
        'expansion_model': 'shifted-expansion-model', 'mmr_lambda': 0.7,
        'reranker': 'llm', 'rerank_depth': 12,
        'reranker_model': 'shifted-reranker-model',
        'recency_half_life_days': 90.0, 'agentic_weights': [0.5, 0.25, 0.75],
        'grader': 'llm', 'grade_threshold': 0.35,
        'grader_model': 'shifted-grader-model', 'max_context_chars': 4200,
        'summary_scope': 'drill-down', 'summary_boost': 1.4,
        'summary_levels': '1,2',
    },
    'generation': {
        'answerer': 'llm', 'model': 'shifted-answer-model',
        'key_facts_judge': True, 'judge_model': 'shifted-judge-model',
        'ragas_model': 'shifted-ragas-model',
    },
}

# Every UI control likewise off its default, for the same reason.
SHIFTED_UI = {'mode': 'openrouter', 'ragas_mode': 'llm', 'limit': 5,
              'ragas_limit': 3, 'types': ['knowledge-update', 'single-hop']}

DATASET_ID = 'smoke-mini'

# Two sessions and two questions, not one of each: a single row cannot catch an
# export that keeps only the first, and the second question is the one whose
# answer contradicts the first, so a dropped session is visible in the answer.
_SHELF = 'the amber notebook is on the third shelf'
_DESK = 'i moved the amber notebook to the desk'

CORPUS = {
    'meta': {'language': 'en'}, 'persona': {}, 'threads': [], 'habits': {},
    'sessions': [
        {'session_id': 's1', 'date': '2026-08-01',
         'messages': [{'role': 'user', 'content': _SHELF},
                      {'role': 'assistant', 'content': 'noted, the third shelf'}]},
        {'session_id': 's2', 'date': '2026-08-02',
         'messages': [{'role': 'user', 'content': _DESK},
                      {'role': 'assistant', 'content': 'noted, the desk'}]},
    ],
}

GROUND_TRUTH = {
    'meta': {'corpus': DATASET_ID, 'query_date': '2026-08-03'},
    'questions': [
        {'id': 'q1', 'type': 'single-hop', 'difficulty': 'easy',
         'answerable': True, 'question_fa': 'دفترچه کجا بود؟',
         'question_en': 'where was the notebook?',
         'answer_fa': 'طبقه سوم', 'key_facts': ['third shelf'],
         'evidence': [{'session_id': 's1', 'message_indices': [0],
                       'quote': _SHELF}]},
        {'id': 'q2', 'type': 'knowledge-update', 'difficulty': 'hard',
         'answerable': True, 'question_fa': 'دفترچه الان کجاست؟',
         'question_en': 'where is the notebook now?',
         'answer_fa': 'روی میز', 'key_facts': ['the desk'],
         'evidence': [{'session_id': 's2', 'message_indices': [0],
                       'quote': _DESK}]},
    ],
}

CHUNKS_BY_SESSION = [
    {'session_id': 's1', 'date': '2026-08-01',
     'chunks': [{'id': 'c1', 'text': _SHELF}]},
    {'session_id': 's2', 'date': '2026-08-02',
     'chunks': [{'id': 'c2', 'text': _DESK}]},
]

# `hierarchy` is set, so the archive must carry the rows it wrote. A summary
# with the same id as a chunk is what the codec refuses, so these are prefixed.
_SUMMARY_TEXT = 'the amber notebook moved from the third shelf to the desk'
SUMMARIES = [{
    'id': 'g1-summary', 'text': _SUMMARY_TEXT, 'group_id': 'g1', 'level': 1,
    'members': 2, 'member_ids': ['c1', 'c2'], 'sessions': 2,
    'date': '2026-08-02', 'chars': len(_SUMMARY_TEXT),
}]

INDEX_STATISTICS = {
    'collection': 'raglab-shifted', 'chunks': 3, 'leaves': 2,
    'avg_chars': 45, 'p95_chars': len(_SUMMARY_TEXT), 'embed_dim': 16,
    'build_seconds': 0.4, 'reused': False,
}

EXECUTION = {'provider': 'openrouter',
             'models': {'answerer': 'shifted-answer-model',
                        'ragas': 'shifted-ragas-model'}}


def _candidate(chunk_id: str, text: str, *, session_id: str, date: str,
               layer: str, gold: bool) -> dict:
    """One retrieved candidate, with every rank and score the codec type-checks."""
    return {
        'chunk_id': chunk_id, 'text': text, 'session_id': session_id,
        'date': date, 'layer': layer, 'level': 1 if layer == 'summary' else 0,
        'group_id': 'g1' if layer == 'summary' else '',
        'members': 2 if layer == 'summary' else 0,
        'dense_rank': 1, 'bm25_rank': 2, 'fused_rank': 1,
        'retrieval_score': 0.91, 'rerank_score': 0.87, 'grade_score': 0.62,
        'kept': True, 'gold': gold,
        # Whole-candidate spans, computed rather than typed: a hand-written
        # bound that outruns the text is refused, and would read as a codec bug.
        'gold_spans': [[0, len(text)]] if gold else [],
    }


TRACES = [
    {'question_id': 'q1', 'question_fa': 'دفترچه کجا بود؟',
     'question_en': 'where was the notebook?', 'type': 'single-hop',
     'difficulty': 'easy', 'answerable': True, 'gold_available': 1,
     'trace': {'candidates': [
         _candidate('c1', _SHELF, session_id='s1', date='2026-08-01',
                    layer='leaf', gold=True),
         _candidate('g1-summary', _SUMMARY_TEXT, session_id='', date='2026-08-02',
                    layer='summary', gold=False),
     ]}},
    {'question_id': 'q2', 'question_fa': 'دفترچه الان کجاست؟',
     'question_en': 'where is the notebook now?', 'type': 'knowledge-update',
     'difficulty': 'hard', 'answerable': True, 'gold_available': 1,
     'trace': {'candidates': [
         _candidate('c2', _DESK, session_id='s2', date='2026-08-02',
                    layer='leaf', gold=True),
     ]}},
]

# Each entry names the stage that owns it, which is the only thing
# `stage_results` reads it for — so the ladder's projection changes rung by rung.
_METRIC_STEPS = {
    'recall': 'retrieval', 'mrr': 'retrieval',
    'llm_context_precision_with_reference': 'retrieval',
    'context_recall': 'retrieval', 'key_facts_hit': 'generation',
    'faithfulness': 'generation', 'answer_relevancy': 'generation',
    'abstention_rate': 'overall',
}


def _catalogue(*keys: str) -> list[dict]:
    return [{'key': key, 'label': key.replace('_', ' '), 'short': key,
             'step': _METRIC_STEPS[key], 'formula': f'{key} formula',
             'library': 'test fixture', 'help': f'what {key} means'}
            for key in keys]


def _result(*, rows: list, question_ids: list, summary_overall: dict,
            ragas: dict) -> dict:
    """The canonical result block, whose selection, rows and traces must agree."""
    return {
        'run_id': 'ladder-run-001', 'label': SHIFTED_CONFIG['label'],
        'config': copy.deepcopy(SHIFTED_CONFIG), 'dataset': DATASET_ID,
        'index': copy.deepcopy(INDEX_STATISTICS),
        'summary': {
            'overall': copy.deepcopy(summary_overall),
            'by_type': {}, 'by_difficulty': {},
            'n_questions': len(question_ids),
        },
        'rows': copy.deepcopy(rows), 'ragas': copy.deepcopy(ragas),
        'seconds': 3.5, 'started_at': '2026-08-20 09:30:00',
        'notes': ['exported by the ladder fixture'],
        'selection': {'balance': 'stride', 'limit': SHIFTED_UI['limit'],
                      'n': len(question_ids), 'by_difficulty': {},
                      'question_ids': list(question_ids)},
    }


def _inspector(*, traces: list) -> dict:
    return {
        'dataset': {'id': DATASET_ID, 'corpus': copy.deepcopy(CORPUS),
                    'ground_truth': copy.deepcopy(GROUND_TRUTH)},
        'chunks_by_session': copy.deepcopy(CHUNKS_BY_SESSION),
        'summaries': copy.deepcopy(SUMMARIES),
        'traces': copy.deepcopy(traces),
    }


def _stages(*, retrieval: dict, generation: dict, overall: dict) -> dict:
    return {'index': {'statistics': copy.deepcopy(INDEX_STATISTICS),
                      'metrics': {}},
            'retrieval': {'metrics': dict(retrieval)},
            'generation': {'metrics': dict(generation)},
            'overall': {'metrics': dict(overall)}}


_RETRIEVAL_ROWS = [
    {'id': 'q1', 'type': 'single-hop', 'difficulty': 'easy', 'recall': 1.0,
     'mrr': 1.0, 'n_contexts': 2, 'gold_available': 1, 'latency_ms': 21.0},
    {'id': 'q2', 'type': 'knowledge-update', 'difficulty': 'hard',
     'recall': 1.0, 'mrr': 0.5, 'n_contexts': 1, 'gold_available': 1,
     'latency_ms': 18.0},
]

# The same rows once an answerer has run: the retrieval columns are untouched,
# and everything a generation stage adds sits beside them.
_GENERATION_ROWS = [
    dict(_RETRIEVAL_ROWS[0], answer='طبقه سوم', answer_error='',
         abstained=False, key_facts_hit=1.0),
    dict(_RETRIEVAL_ROWS[1], answer='روی میز', answer_error='',
         abstained=False, key_facts_hit=0.8),
]

RAGAS = {
    'mode': 'llm', 'n': 2,
    'metrics': {'faithfulness': 0.94, 'answer_relevancy': 0.91,
                'llm_context_precision_with_reference': 0.82,
                'context_recall': 0.88},
    'decision': 0.8875,
    'decision_spread': {'stderr': 0.021, 'spread': 0.12},
    'decision_metrics': ['faithfulness', 'answer_relevancy',
                         'llm_context_precision_with_reference',
                         'context_recall'],
    'judge': {'model': 'shifted-ragas-model', 'provider': 'openrouter'},
    'notes': [],
}


def _rung(name: str, description: str, *, result=None, evidence=None,
          stage_results=None, catalogue=None) -> dict:
    """One rung: the inputs an export is given, and the archive it must produce."""
    value = {
        'format': 'raglab-experiment', 'version': 1,
        'settings': {'config': copy.deepcopy(SHIFTED_CONFIG),
                     'ui': copy.deepcopy(SHIFTED_UI)},
    }
    if result is not None:
        value['evaluation'] = {
            'execution': copy.deepcopy(EXECUTION),
            'metric_catalogue': copy.deepcopy(catalogue),
            'stage_results': copy.deepcopy(stage_results),
            'result': copy.deepcopy(result),
            'inspector': copy.deepcopy(evidence),
        }
    return {
        'name': name, 'description': description,
        'config': copy.deepcopy(SHIFTED_CONFIG), 'ui': copy.deepcopy(SHIFTED_UI),
        'result': copy.deepcopy(result),
        'evidence': None if result is None else {
            'execution': copy.deepcopy(EXECUTION),
            'metric_catalogue': copy.deepcopy(catalogue),
            'inspector': copy.deepcopy(evidence),
        },
        'archive': value,
    }


def settings_rung() -> dict:
    """Rung 1 — the knob surface alone, which is what an unrun experiment is."""
    return _rung('settings', 'every knob and UI control, no evaluation')


def indexed_rung() -> dict:
    """Rung 2 — plus what was indexed: the corpus, its chunks and its summaries.

    Nothing was asked yet, so rows, traces and the selection are empty together
    — the codec ties all three to one count, and zero is a value it must accept.
    """
    return _rung(
        'indexed', 'plus corpus, ground truth, chunks and summaries',
        result=_result(rows=[], question_ids=[], summary_overall={}, ragas={}),
        evidence=_inspector(traces=[]), catalogue=[],
        stage_results=_stages(retrieval={}, generation={}, overall={}))


def retrieved_rung() -> dict:
    """Rung 3 — plus what retrieval found: a trace per question, and its metrics.

    The shape `run_retrieval` produces: candidates ranked and graded, no answer
    written and nothing judged, so the generation and overall stages stay empty.
    """
    overall = {'recall': 1.0, 'mrr': 0.75}
    return _rung(
        'retrieved', 'plus per-question traces, candidates and retrieval metrics',
        result=_result(rows=_RETRIEVAL_ROWS, question_ids=['q1', 'q2'],
                       summary_overall=overall, ragas={}),
        evidence=_inspector(traces=TRACES),
        catalogue=_catalogue('recall', 'mrr'),
        stage_results=_stages(retrieval=overall, generation={}, overall={}))


def _judged(name: str, description: str, *, traces: list) -> dict:
    """A judged rung: answers, key-facts scores and the four judged metrics.

    Written once and built twice, because `generated` and
    `scored-without-traces` must differ in their traces and in *nothing else* —
    two rungs that also drifted in a metric would not isolate what the absent
    evidence costs.
    """
    overall = {'recall': 1.0, 'mrr': 0.75, 'key_facts_hit': 0.9,
               'abstention_rate': 0.0}
    return _rung(
        name, description,
        result=_result(rows=_GENERATION_ROWS, question_ids=['q1', 'q2'],
                       summary_overall=overall, ragas=RAGAS),
        evidence=_inspector(traces=traces),
        catalogue=_catalogue('recall', 'mrr', 'key_facts_hit',
                             'abstention_rate', 'faithfulness',
                             'answer_relevancy',
                             'llm_context_precision_with_reference',
                             'context_recall'),
        stage_results=_stages(
            retrieval={'recall': 1.0, 'mrr': 0.75,
                       'llm_context_precision_with_reference': 0.82,
                       'context_recall': 0.88},
            generation={'key_facts_hit': 0.9, 'faithfulness': 0.94,
                        'answer_relevancy': 0.91},
            overall={'abstention_rate': 0.0}))


def generated_rung() -> dict:
    """Rung 4 — plus what generation produced: answers, and the judged metrics.

    The fullest shape, and — with the rung below — one of the two carrying
    `ragas`, so its projection splits one metric block across three stages.
    """
    return _judged(
        'generated',
        'plus answers, key-facts scores and the four judged metrics',
        traces=TRACES)


def scored_without_traces_rung() -> dict:
    """Off rung 4 — the same judged experiment with no trace kept.

    The shape of every evaluation this lab recorded before an experiment could
    be exported: a `.runs/` file holds the config, the index statistics, the
    rows, the summary, the selection and the `ragas` block, and has no place at
    all for a trace or a chunk. The corpus, the chunks and the summaries are
    still here because a rebuild recovers them without inventing anything —
    the corpus loads back by id and the chunker is deterministic — whereas
    what retrieval ranked, once nothing wrote it down, is gone for good.

    So the rows and the four judged metrics are exactly the `generated` rung's:
    the measurement is untouched, and only the recording of how retrieval got
    there is missing. That is the one direction the codecs relax in — traces
    are a subset of the rows, never a superset, and never absent rows.
    """
    return _judged(
        'scored-without-traces',
        'the judged rung with no traces: scored, trace never retained',
        traces=[])


# The rungs that stack, in order. `scored-without-traces` is deliberately not
# here: it removes evidence from the rung above rather than adding any, so it
# would break the monotonic reading this tuple exists to support.
SPINE = ('settings', 'indexed', 'retrieved', 'generated')

LADDER_BUILDERS = (settings_rung, indexed_rung, retrieved_rung, generated_rung,
                   scored_without_traces_rung)


def ladder() -> list[dict]:
    """Every shape the format takes: the four stacked rungs, then the branch."""
    return [build() for build in LADDER_BUILDERS]


# What each rung must be carrying once it has crossed a codec, written out
# rather than derived from the fixture above: a fixture that lost its summaries
# would otherwise agree with a reading of itself that had lost them too. Kept
# here beside the ladder because both codecs are held to it — Python's in
# `test_archive_round_trip.py`, the browser's in `dashboard/tests`.
CARRIED = {
    'settings': {'evaluation': False, 'sessions': 0, 'questions': 0,
                 'chunks': 0, 'summaries': 0, 'traces': 0, 'candidates': 0,
                 'rows': 0, 'answers': 0, 'judged': False},
    'indexed': {'evaluation': True, 'sessions': 2, 'questions': 2,
                'chunks': 2, 'summaries': 1, 'traces': 0, 'candidates': 0,
                'rows': 0, 'answers': 0, 'judged': False},
    'retrieved': {'evaluation': True, 'sessions': 2, 'questions': 2,
                  'chunks': 2, 'summaries': 1, 'traces': 2, 'candidates': 3,
                  'rows': 2, 'answers': 0, 'judged': False},
    'generated': {'evaluation': True, 'sessions': 2, 'questions': 2,
                  'chunks': 2, 'summaries': 1, 'traces': 2, 'candidates': 3,
                  'rows': 2, 'answers': 2, 'judged': True},
    # The same numbers as `generated` but for the two counts that recorded
    # retrieval: rows and judged metrics survive with no trace behind them.
    'scored-without-traces': {'evaluation': True, 'sessions': 2, 'questions': 2,
                              'chunks': 2, 'summaries': 1, 'traces': 0,
                              'candidates': 0, 'rows': 2, 'answers': 2,
                              'judged': True},
}


def contents(value: dict) -> dict:
    """Count what an archive is actually carrying, in the terms `CARRIED` uses."""
    evaluation = value.get('evaluation')
    if evaluation is None:
        return dict(CARRIED['settings'])
    inspector = evaluation['inspector']
    dataset = inspector['dataset']
    traces = inspector['traces']
    rows = evaluation['result']['rows']
    return {
        'evaluation': True,
        'sessions': len(dataset['corpus']['sessions']),
        'questions': len(dataset['ground_truth']['questions']),
        'chunks': sum(len(group['chunks'])
                      for group in inspector['chunks_by_session']),
        'summaries': len(inspector['summaries']),
        'traces': len(traces),
        'candidates': sum(len(trace['trace']['candidates']) for trace in traces),
        'rows': len(rows),
        'answers': sum(1 for row in rows if row.get('answer')),
        'judged': bool((evaluation['result']['ragas'] or {}).get('metrics')),
    }
