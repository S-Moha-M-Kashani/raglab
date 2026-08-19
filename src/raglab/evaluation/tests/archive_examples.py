"""One small, complete archive used by every import contract."""
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
                'generation': {'metrics': {}}, 'agent': {'metrics': {}},
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
