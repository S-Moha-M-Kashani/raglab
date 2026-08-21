"""Validation and construction for portable RAG Lab experiment archives."""

from __future__ import annotations

import copy
import json
import math

from raglab.configuration.lab_config import LabConfig
from raglab.corpora import dataset_import_contract as datasets


FORMAT = 'raglab-experiment'
VERSION = 1
MAX_BYTES = 32 * 1024 * 1024
DEFAULT_LIMITS = {
    'depth': 32, 'questions': 5000, 'chunks': 50000,
    'traces': 5000, 'candidates_per_trace': 1000,
    'list_items': 100000, 'string_chars': 2_000_000,
}

CANONICAL_RESULT_KEYS = (
    'run_id', 'label', 'config', 'dataset', 'index', 'summary', 'rows',
    'ragas', 'seconds', 'started_at', 'notes', 'selection')

_MODEL_FIELDS = (
    ('index', 'embed_model'),
    ('retrieval', 'expansion_model'),
    ('retrieval', 'reranker_model'),
    ('retrieval', 'grader_model'),
    ('generation', 'model'),
    ('generation', 'judge_model'),
    ('generation', 'ragas_model'),
)
_STAGES = ('index', 'retrieval', 'generation', 'overall')
_UNSAFE_METRIC_KEYS = frozenset(('__proto__', 'prototype', 'constructor'))


class ArchiveError(ValueError):
    pass


def _secret_key(folded: str) -> bool:
    return (folded.endswith('apikey') or folded.endswith('accesstoken')
            or folded in ('openrouterkey', 'authorization')
            or 'password' in folded or 'secret' in folded)


def _walk(value, path: str, depth: int, limits: dict) -> None:
    if depth > limits['depth']:
        raise ArchiveError(f'{path}: nesting depth exceeds {limits["depth"]}')
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArchiveError(f'{path}: object keys must be strings')
            folded = ''.join(char for char in key.lower() if char.isalnum())
            if _secret_key(folded):
                raise ArchiveError(f'{path}.{key}: credential fields are forbidden')
            _walk(child, f'{path}.{key}', depth + 1, limits)
    elif isinstance(value, list):
        if len(value) > limits['list_items']:
            raise ArchiveError(f'{path}: list exceeds {limits["list_items"]}')
        for index, child in enumerate(value):
            _walk(child, f'{path}[{index}]', depth + 1, limits)
    elif isinstance(value, str) and len(value) > limits['string_chars']:
        raise ArchiveError(f'{path}: string exceeds {limits["string_chars"]} chars')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ArchiveError(f'{path}: number must be finite or null')
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ArchiveError(f'{path}: value is not JSON-compatible')


def _limits(overrides: dict | None) -> dict:
    if overrides is not None and not isinstance(overrides, dict):
        raise ArchiveError('limits: object required')
    value = dict(DEFAULT_LIMITS)
    for key, limit in (overrides or {}).items():
        if key not in value:
            raise ArchiveError(f'limits.{key}: unknown structural limit')
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ArchiveError(f'limits.{key}: non-negative integer required')
        value[key] = limit
    return value


def _dict(value, path: str) -> dict:
    if not isinstance(value, dict):
        raise ArchiveError(f'{path}: object required')
    return value


def _list(value, path: str) -> list:
    if not isinstance(value, list):
        raise ArchiveError(f'{path}: list required')
    return value


def _string(value, path: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = 'non-empty ' if nonempty else ''
        raise ArchiveError(f'{path}: {qualifier}string required')
    return value


def _integer(value, path: str, *, minimum: int | None = None,
             maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArchiveError(f'{path}: integer required')
    if minimum is not None and value < minimum:
        raise ArchiveError(f'{path}: must be at least {minimum}')
    if maximum is not None and value > maximum:
        raise ArchiveError(f'{path}: must not exceed {maximum}')
    return value


def _number(value, path: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        suffix = ' or null' if nullable else ''
        raise ArchiveError(f'{path}: finite number{suffix} required')
    if not math.isfinite(value):
        raise ArchiveError(f'{path}: number must be finite or null')


def _bool(value, path: str) -> None:
    if not isinstance(value, bool):
        raise ArchiveError(f'{path}: boolean required')


def _keys(value: dict, expected, path: str) -> None:
    expected = set(expected)
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        raise ArchiveError(f'{path}: missing keys {", ".join(missing)}')
    if extra:
        raise ArchiveError(f'{path}: unexpected keys {", ".join(extra)}')


def _json_value(value):
    """Return the JSON representation used when dataclass tuples are encoded."""
    return json.loads(json.dumps(value, allow_nan=False))


def _json_equal(left, right) -> bool:
    """Compare JSON values without Python's bool/number equality coercion."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return (isinstance(left, (int, float))
                and not isinstance(left, bool)
                and isinstance(right, (int, float))
                and not isinstance(right, bool)
                and left == right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_json_equal(left[key], right[key]) for key in left))
    if isinstance(left, list) or isinstance(right, list):
        return (isinstance(left, list) and isinstance(right, list)
                and len(left) == len(right)
                and all(_json_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _shape(value, expected, path: str) -> None:
    if isinstance(expected, dict):
        value = _dict(value, path)
        _keys(value, expected, path)
        for key, child in expected.items():
            _shape(value[key], child, f'{path}.{key}')
    elif isinstance(expected, list):
        value = _list(value, path)
        if expected:
            for index, child in enumerate(value):
                _shape(child, expected[0], f'{path}[{index}]')
    elif isinstance(expected, bool):
        _bool(value, path)
    elif isinstance(expected, int):
        _integer(value, path)
    elif isinstance(expected, float):
        _number(value, path)
    elif isinstance(expected, str):
        _string(value, path)
    elif value is not None:
        raise ArchiveError(f'{path}: null required')


def _validate_settings(value, limits: dict) -> dict:
    settings = _dict(value, 'settings')
    _keys(settings, ('config', 'ui'), 'settings')
    config = _dict(settings['config'], 'settings.config')
    _shape(config, _json_value(LabConfig().to_dict()), 'settings.config')
    try:
        cfg = LabConfig.from_dict(config)
        canonical = _json_value(cfg.to_dict())
    except (TypeError, ValueError, OverflowError) as error:
        raise ArchiveError(f'settings.config: {error}') from error
    if canonical != config:
        raise ArchiveError('settings.config: complete normalized LabConfig required')
    problems = cfg.validate()
    if problems:
        raise ArchiveError('settings.config: ' + '; '.join(problems))
    for group, field in _MODEL_FIELDS:
        _string(config[group][field], f'settings.config.{group}.{field}')

    ui = _dict(settings['ui'], 'settings.ui')
    _keys(ui, ('mode', 'ragas_mode', 'limit', 'ragas_limit', 'types'),
          'settings.ui')
    mode = _string(ui['mode'], 'settings.ui.mode')
    if mode not in ('', 'local', 'openrouter', 'claude', 'codex'):
        raise ArchiveError(f'settings.ui.mode: unsupported mode {mode!r}')
    ragas_mode = _string(ui['ragas_mode'], 'settings.ui.ragas_mode')
    if ragas_mode not in ('off', 'offline', 'llm'):
        raise ArchiveError(
            f'settings.ui.ragas_mode: unsupported mode {ragas_mode!r}')
    _integer(ui['limit'], 'settings.ui.limit', minimum=0, maximum=200)
    _integer(ui['ragas_limit'], 'settings.ui.ragas_limit',
             minimum=0, maximum=200)
    types = _list(ui['types'], 'settings.ui.types')
    if len(types) != len(set(types)):
        raise ArchiveError('settings.ui.types: duplicate question types')
    for index, question_type in enumerate(types):
        _string(question_type, f'settings.ui.types[{index}]')
        if question_type not in datasets.TYPES:
            raise ArchiveError(
                f'settings.ui.types[{index}]: unknown question type '
                f'{question_type!r}')
    return settings


def _unique(rows: list[dict], key: str, path: str) -> dict[str, dict]:
    found = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ArchiveError(f'{path}[{index}]: object required')
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ArchiveError(f'{path}[{index}].{key}: non-empty string required')
        if value in found:
            raise ArchiveError(f'{path}: duplicate {key} {value}')
        found[value] = row
    return found


def _span(span, text: str, path: str) -> None:
    invalid = (not isinstance(span, list) or len(span) != 2
               or any(isinstance(n, bool) or not isinstance(n, int)
                      for n in span)
               or not 0 <= span[0] <= span[1] <= len(text))
    if invalid:
        raise ArchiveError(f'{path}: expected integer bounds within candidate text')


def _finite(value, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f'{path}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f'{path}[{index}]')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ArchiveError(f'{path}: number must be finite or null')


def _validate_metric_catalogue(value) -> list[dict]:
    rows = _list(value, 'evaluation.metric_catalogue')
    found = set()
    for index, row in enumerate(rows):
        path = f'evaluation.metric_catalogue[{index}]'
        row = _dict(row, path)
        key = _metric_key(row.get('key'), f'{path}.key')
        if key in found:
            raise ArchiveError(f'evaluation.metric_catalogue: duplicate key {key}')
        found.add(key)
        step = row.get('step') or 'overall'
        if not isinstance(step, str) or step not in _STAGES:
            raise ArchiveError(f'{path}.step: unknown stage {step!r}')
        for field in ('label', 'short', 'formula', 'library', 'help'):
            if field in row:
                _string(row[field], f'{path}.{field}')
    return rows


def _metric_key(value, path: str) -> str:
    value = _string(value, path, nonempty=True)
    if value in _UNSAFE_METRIC_KEYS:
        raise ArchiveError(f'{path}: unsafe metric key {value}')
    return value


def _metric_items(value, path: str):
    value = _dict(value, path)
    for key, metric in value.items():
        yield _metric_key(key, f'{path}.{key}'), metric


def _validate_dataset(inspector_dataset: dict, limits: dict) -> dict[str, dict]:
    inspector_dataset = _dict(inspector_dataset,
                              'evaluation.inspector.dataset')
    _keys(inspector_dataset, ('id', 'corpus', 'ground_truth'),
          'evaluation.inspector.dataset')
    dataset_id = _string(inspector_dataset['id'],
                         'evaluation.inspector.dataset.id', nonempty=True)
    corpus = _dict(inspector_dataset['corpus'],
                   'evaluation.inspector.dataset.corpus')
    ground_truth = _dict(inspector_dataset['ground_truth'],
                         'evaluation.inspector.dataset.ground_truth')
    questions = _list(ground_truth.get('questions'),
                      'evaluation.inspector.dataset.ground_truth.questions')
    if len(questions) > limits['questions']:
        raise ArchiveError(
            'evaluation.inspector.dataset.ground_truth.questions: '
            f'questions exceed {limits["questions"]}')
    question_by_id = _unique(
        questions, 'id',
        'evaluation.inspector.dataset.ground_truth.questions')
    sessions = _list(corpus.get('sessions'),
                     'evaluation.inspector.dataset.corpus.sessions')
    session_by_id = _unique(
        sessions, 'session_id', 'evaluation.inspector.dataset.corpus.sessions')
    for question_index, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        evidence = question.get('evidence')
        if not isinstance(evidence, list):
            continue
        for evidence_index, item in enumerate(evidence):
            if not isinstance(item, dict):
                continue
            path = ('evaluation.inspector.dataset.ground_truth.questions'
                    f'[{question_index}].evidence[{evidence_index}]')
            session_id = item.get('session_id')
            if not isinstance(session_id, str) or not session_id:
                raise ArchiveError(f'{path}.session_id: non-empty string required')
            session = session_by_id.get(session_id)
            if session is None:
                raise ArchiveError(
                    f'{path}.session_id {session_id}: session is not archived')
            indices = item.get('message_indices')
            if not isinstance(indices, list):
                continue
            messages = session.get('messages')
            message_count = len(messages) if isinstance(messages, list) else 0
            for message_index in indices:
                if (isinstance(message_index, bool)
                        or not isinstance(message_index, int)
                        or not 0 <= message_index < message_count):
                    raise ArchiveError(
                        f'{path}.message_indices {message_index}: outside session '
                        f'{session_id} with {message_count} messages')
    dataset_payload = {
        'dataset': {
            'id': dataset_id,
            'name': f'Archived {dataset_id}',
            'language': corpus.get('meta', {}).get('language')
            if isinstance(corpus.get('meta'), dict) else None,
        },
        'sessions': sessions,
        'questions': questions,
    }
    problems = datasets.validate(dataset_payload)
    if problems:
        raise ArchiveError('evaluation.inspector.dataset: ' + '; '.join(problems))
    return question_by_id


def _validate_chunks(inspector: dict, limits: dict) -> dict[str, dict]:
    groups = _list(inspector.get('chunks_by_session'),
                   'evaluation.inspector.chunks_by_session')
    chunks = []
    for index, group in enumerate(groups):
        group = _dict(group, f'evaluation.inspector.chunks_by_session[{index}]')
        chunks.extend(_list(
            group.get('chunks'),
            f'evaluation.inspector.chunks_by_session[{index}].chunks'))
    if len(chunks) > limits['chunks']:
        raise ArchiveError(
            f'evaluation.inspector.chunks_by_session: chunks exceed '
            f'{limits["chunks"]}')
    found = _unique(chunks, 'id', 'evaluation.inspector.chunks_by_session.chunks')
    for chunk_id, chunk in found.items():
        _string(chunk.get('text'),
                f'evaluation.inspector.chunks_by_session.chunks[{chunk_id}].text')
    return found


def _ids(values, path: str) -> list[str]:
    values = _list(values, path)
    found = set()
    for index, value in enumerate(values):
        _string(value, f'{path}[{index}]', nonempty=True)
        if value in found:
            raise ArchiveError(f'{path}: duplicate id {value}')
        found.add(value)
    return values


def _validate_completed(evaluation, settings: dict, limits: dict) -> None:
    evaluation = _dict(evaluation, 'evaluation')
    _keys(evaluation, ('execution', 'metric_catalogue', 'stage_results',
                       'result', 'inspector'), 'evaluation')

    execution = _dict(evaluation['execution'], 'evaluation.execution')
    _string(execution.get('provider'), 'evaluation.execution.provider')
    models = _dict(execution.get('models'), 'evaluation.execution.models')
    for key, value in models.items():
        _string(value, f'evaluation.execution.models.{key}')
    catalogue = _validate_metric_catalogue(evaluation['metric_catalogue'])

    result = _dict(evaluation['result'], 'evaluation.result')
    _keys(result, CANONICAL_RESULT_KEYS, 'evaluation.result')
    for field in ('run_id', 'label', 'dataset', 'started_at'):
        _string(result[field], f'evaluation.result.{field}',
                nonempty=field in ('run_id', 'dataset'))
    if not _json_equal(result['config'], settings['config']):
        raise ArchiveError(
            'evaluation.result.config: must equal settings.config')
    for field in ('index', 'summary', 'ragas', 'selection'):
        _dict(result[field], f'evaluation.result.{field}')
    rows = _list(result['rows'], 'evaluation.result.rows')
    notes = _list(result['notes'], 'evaluation.result.notes')
    for index, note in enumerate(notes):
        _string(note, f'evaluation.result.notes[{index}]')
    _number(result['seconds'], 'evaluation.result.seconds')
    _finite(result['summary'], 'evaluation.result.summary')
    _finite(rows, 'evaluation.result.rows')
    _finite(result['ragas'], 'evaluation.result.ragas')

    inspector = _dict(evaluation['inspector'], 'evaluation.inspector')
    _keys(inspector, ('dataset', 'chunks_by_session', 'summaries', 'traces'),
          'evaluation.inspector')
    question_by_id = _validate_dataset(inspector['dataset'], limits)
    chunk_by_id = _validate_chunks(inspector, limits)
    summaries = _list(inspector['summaries'], 'evaluation.inspector.summaries')
    if len(chunk_by_id) + len(summaries) > limits['chunks']:
        raise ArchiveError(
            f'evaluation.inspector: chunks exceed {limits["chunks"]}')
    summary_by_id = _unique(summaries, 'id', 'evaluation.inspector.summaries')
    for summary_id, summary in summary_by_id.items():
        _string(summary.get('text'),
                f'evaluation.inspector.summaries[{summary_id}].text')
    duplicate_source_ids = set(chunk_by_id) & set(summary_by_id)
    if duplicate_source_ids:
        duplicate = sorted(duplicate_source_ids)[0]
        raise ArchiveError(f'evaluation.inspector: duplicate chunk id {duplicate}')

    row_by_id = _unique(rows, 'id', 'evaluation.result.rows')
    for row_index, row_id in enumerate(row_by_id):
        if row_id not in question_by_id:
            raise ArchiveError(
                f'evaluation.result.rows[{row_index}].id {row_id} is outside '
                'archived ground truth')

    traces = _list(inspector['traces'], 'evaluation.inspector.traces')
    if len(traces) > limits['traces']:
        raise ArchiveError(
            f'evaluation.inspector.traces: traces exceed {limits["traces"]}')
    trace_by_id = _unique(traces, 'question_id',
                          'evaluation.inspector.traces')
    sources = chunk_by_id | summary_by_id
    for trace_index, (question_id, trace_row) in enumerate(trace_by_id.items()):
        trace_path = f'evaluation.inspector.traces[{trace_index}]'
        if question_id not in question_by_id:
            raise ArchiveError(
                f'{trace_path}.question_id {question_id} is outside archived '
                'ground truth')
        trace = _dict(trace_row.get('trace'), f'{trace_path}.trace')
        candidates = _list(trace.get('candidates'),
                           f'{trace_path}.trace.candidates')
        if len(candidates) > limits['candidates_per_trace']:
            raise ArchiveError(
                f'{trace_path}.trace.candidates: candidates exceed '
                f'{limits["candidates_per_trace"]}')
        for candidate_index, candidate in enumerate(candidates):
            candidate_path = f'{trace_path}.trace.candidates[{candidate_index}]'
            candidate = _dict(candidate, candidate_path)
            chunk_id = _string(candidate.get('chunk_id'),
                               f'{candidate_path}.chunk_id', nonempty=True)
            source = sources.get(chunk_id)
            if source is None:
                raise ArchiveError(
                    f'{candidate_path}.chunk_id {chunk_id} is not archived')
            text = _string(candidate.get('text'), f'{candidate_path}.text')
            if text != source.get('text'):
                raise ArchiveError(
                    f'{candidate_path}.text: differs from archived chunk '
                    f'{chunk_id}')
            spans = _list(candidate.get('gold_spans'),
                          f'{candidate_path}.gold_spans')
            for span_index, span in enumerate(spans):
                _span(span, text, f'{candidate_path}.gold_spans[{span_index}]')
            for key, value in candidate.items():
                folded = ''.join(char for char in key.lower() if char.isalnum())
                if folded.endswith(('rank', 'score')):
                    _number(value, f'{candidate_path}.{key}', nullable=True)
            _finite(candidate, candidate_path)

    selection = result['selection']
    question_ids = _ids(selection.get('question_ids'),
                        'evaluation.result.selection.question_ids')
    row_ids = list(row_by_id)
    trace_ids = list(trace_by_id)
    if question_ids != row_ids:
        raise ArchiveError(
            f'evaluation.result.rows ids {row_ids!r} must equal ordered '
            f'selection.question_ids {question_ids!r}')
    if question_ids != trace_ids:
        raise ArchiveError(
            f'evaluation.inspector.traces question ids {trace_ids!r} must '
            f'equal ordered selection.question_ids {question_ids!r}')
    count = len(question_ids)
    if _integer(selection.get('n'), 'evaluation.result.selection.n',
                minimum=0) != count:
        raise ArchiveError(
            f'evaluation.result.selection.n: expected {count}')
    if _integer(result['summary'].get('n_questions'),
                'evaluation.result.summary.n_questions', minimum=0) != count:
        raise ArchiveError(
            f'evaluation.result.summary.n_questions: expected {count}')

    dataset_id = settings['config']['index']['dataset'] or datasets.BUILTIN
    if result['dataset'] != dataset_id:
        raise ArchiveError(
            f'evaluation.result.dataset {result["dataset"]}: expected '
            f'settings dataset {dataset_id}')
    inspector_id = inspector['dataset']['id']
    if inspector_id != dataset_id:
        raise ArchiveError(
            f'evaluation.inspector.dataset.id {inspector_id}: expected '
            f'settings dataset {dataset_id}')

    projected = stage_results(result, catalogue)
    if not _json_equal(evaluation['stage_results'], projected):
        raise ArchiveError(
            'evaluation.stage_results: must equal the derived metric projection')


def validate_archive(payload, *, encoded_size=None, limits=None) -> dict:
    """Validate an archive at the server trust boundary and return it unchanged."""
    if encoded_size is not None:
        if (isinstance(encoded_size, bool) or not isinstance(encoded_size, int)
                or encoded_size < 0):
            raise ArchiveError('archive: encoded_size must be a non-negative integer')
        if encoded_size > MAX_BYTES:
            raise ArchiveError('archive: encoded file must not exceed 32 MiB')
    bounds = _limits(limits)
    if not isinstance(payload, dict):
        raise ArchiveError('archive: object required')
    _walk(payload, 'archive', 0, bounds)
    if encoded_size is None:
        try:
            encoded_size = len(json.dumps(
                payload, ensure_ascii=False, allow_nan=False).encode('utf-8'))
        except (TypeError, ValueError, OverflowError) as error:
            raise ArchiveError(
                f'archive: could not serialize encoded size: {error}') from error
        if encoded_size > MAX_BYTES:
            raise ArchiveError('archive: encoded file must not exceed 32 MiB')
    _keys(payload, ('format', 'version', 'settings')
          if 'evaluation' not in payload
          else ('format', 'version', 'settings', 'evaluation'), 'archive')
    if payload.get('format') != FORMAT:
        raise ArchiveError(
            f'archive.format {payload.get("format")!r}: expected {FORMAT!r}')
    version = payload.get('version')
    if type(version) is not int or version != VERSION:
        raise ArchiveError(
            f'archive.version {version!r}: expected integer version {VERSION}')
    settings = _validate_settings(payload.get('settings'), bounds)
    if 'evaluation' in payload:
        _validate_completed(payload['evaluation'], settings, bounds)
    return payload


def stage_results(result: dict, metric_catalogue: list[dict]) -> dict:
    groups = {step: {'metrics': {}}
              for step in ('retrieval', 'generation', 'overall')}
    groups['index'] = {'statistics': copy.deepcopy(result['index']),
                       'metrics': {}}
    steps = {}
    for index, row in enumerate(metric_catalogue):
        key = _metric_key(row.get('key'),
                          f'metric_catalogue[{index}].key')
        step = row.get('step') or 'overall'
        if step not in _STAGES:
            raise ArchiveError(
                f'metric_catalogue[{index}].step: unknown stage {step!r}')
        steps[key] = step
    values = list(_metric_items(
        result['summary'].get('overall') or {}, 'result.summary.overall'))
    values.extend(_metric_items(
        (result.get('ragas') or {}).get('metrics') or {},
        'result.ragas.metrics'))
    for key, value in values:
        groups[steps.get(key, 'overall')]['metrics'][key] = value
    return groups


def build_completed(settings: dict, result: dict, evidence: dict) -> dict:
    body = copy.deepcopy(settings.get('settings', settings))
    canonical = {key: copy.deepcopy(result[key])
                 for key in CANONICAL_RESULT_KEYS}
    evaluation = {
        'execution': copy.deepcopy(evidence['execution']),
        'metric_catalogue': copy.deepcopy(evidence['metric_catalogue']),
        'result': canonical,
        'inspector': copy.deepcopy(evidence['inspector']),
    }
    evaluation['stage_results'] = stage_results(
        canonical, evaluation['metric_catalogue'])
    return validate_archive({'format': FORMAT, 'version': VERSION,
                             'settings': body, 'evaluation': evaluation})
