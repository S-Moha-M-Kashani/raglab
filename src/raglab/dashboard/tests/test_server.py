"""The service — the lab's own FastAPI app, served at /api/options and the
ad hoc query endpoint.

`/api/options` is exercised by three grouped-assert tests, one per topic
rather than one per field:
`test_options_describe_the_corpus_the_knobs_and_the_metrics` (corpus counts,
question types, knob help text, the metric definitions block, the
steps/colour map, dependency rules), `test_options_offer_models_embedders_and_datasets_per_backend`
(model roles, embedder hints and languages, the embed-model catalogue,
provider modes, the dataset catalogue) and `test_options_advertise_no_vector_database`
(the storage/capabilities claim, merged with the health check). The rest of
the file is the query endpoint (a smoke-index round trip plus a direct
`LabConfig.from_dict` unit test for its optional model fields), the ragas
judge-model plumbing, and the route-contract group — two of those moved in
from `test_raglab.py`."""
import pytest

from raglab.llm_backends import model_role_catalogue as models
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.configuration.lab_config import LabConfig, RetrievalConfig

from raglab.conftest import (
    LAB_SETTINGS,
    OLLAMA_SETTINGS,
    REQUESTED_MODELS,
    _finished)


# --- the service -----------------------------------------------------------

def _ask(client, payload: dict) -> dict:
    """POST one question and wait for its job — the panel's ask flow."""
    res = client.post('/api/queries', json=payload)
    assert res.status_code == 202, res.text
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    return job['result']


def test_a_build_starts_without_any_service_running(client):
    # this is an integration test
    """With the index in process memory there is nothing that can be down to
    refuse a build. The structural half of this claim — that the app exposes
    no vector-store gate at all — is a static, no-server-needed check and
    moved to test_conventions.py
    (test_a_build_exposes_no_vector_store_gate); this half needs the real
    `client` fixture and a real build, so it stays here."""
    body = client.post('/api/indexes', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash',
                  'layers': ['session']}}).json()
    assert body['job_id']


def test_options_describe_the_corpus_the_knobs_and_the_metrics(client):
    # this is an integration test
    """Corpus counts, question types incl. habit, help for every knob key, the
    metric definitions block, the steps/colour map, dependency rules — one
    GET, one dict, grouped asserts."""
    body = client.get('/api/options').json()

    # --- the corpus: pinned rather than derived, so a change to its size has
    # to be stated here on purpose. The habit ledger is only as good as the
    # habits behind it, and the per-type breakdown is where habit retrieval
    # either shows up or hides inside the aggregation bucket.
    corpus = body['corpus']
    assert corpus['sessions'] == 167
    assert corpus['questions'] == 112
    assert corpus['habits'] == 5
    assert 'habit' in body['question_types']
    assert 'semantic-drift' in body['chunkers']

    # --- help text for every knob key, and the new metadata/deciding-score
    # metric's own text.
    for key in ('index.chunker', 'retrieval.reranker', 'retrieval.grade_threshold',
                'generation.answerer', 'model.answer', 'metric.recall',
                'metric.ragas_decision'):
        assert body['help'].get(key), key

    # --- the metric definitions block: deterministic and judged metrics
    # arrive through the same shape, so the dashboard renders one concept
    # rather than two.
    by_measure = {measure['key']: measure for measure in body['metrics']}
    for key in ('recall', 'quote_recall', 'headline', 'faithfulness',
                'non_llm_context_recall'):
        measure = by_measure.get(key)
        assert measure, key
        assert measure['label'] and measure['short'], key
        assert measure['formula'] and measure['library'] and measure['help'], key
        assert 'step' in measure, key
    assert 'ragas' in by_measure['faithfulness']['library'].lower()
    assert by_measure['ragas_decision']['step'] == ''

    # --- steps/colour map: which step a control belongs to is a fact about
    # the pipeline, served with everything else — the panel cannot invent it.
    assert [step['key'] for step in body['steps']] == ['index', 'retrieval',
                                                        'generation']
    assert all(step['label'] and step['short'] and step['note']
               for step in body['steps'])
    step_keys = {step['key'] for step in body['steps']}
    assert all(role['step'] in step_keys for role in body['model_roles'])
    step_by_role = {role['key']: role['step'] for role in body['model_roles']}
    assert step_by_role['rerank'] == 'retrieval'
    assert step_by_role['answer'] == 'generation'

    # --- dependency rules: served once so both panels grey out the same
    # knobs for the same stated reason, rather than duplicating the table.
    deps = body['dependencies']
    assert deps['index.embed_model']['field'] == 'index.embedder'
    assert deps['retrieval.grader_model']['on'] == ['llm']


def test_options_offer_models_embedders_and_datasets_per_backend(client):
    # this is an integration test
    """Model roles, embedder hints and languages, the embed-model catalogue,
    and provider modes — every model lives in one place, and a slug only
    means something to the backend that serves it."""
    body = client.get('/api/options').json()

    # --- a model role for every LLM-backed stage, every one fully labelled.
    roles = {role['key']: role for role in body['model_roles']}
    assert set(roles) == {'expand', 'rerank', 'grade', 'answer',
                          'judge', 'ragas'}
    assert all(role['help'] and role['label'] and role['field']
               for role in roles.values())
    ids = [m['id'] for m in body['models']]
    # The backend's own default leads, because a slug only means something to
    # the backend serving it.
    assert ids[0] == ''
    # The *list* follows the configured backend: a fake backend serves no
    # local slugs, checked directly rather than against a daemon.
    assert '4skl/gemma4-e2b-mtp' not in ids, 'a fake backend serves no local slugs'
    assert '4skl/gemma4-e2b-mtp' in [m.id for m in models.known_models(
        OLLAMA_SETTINGS)]
    # 'open' is not guaranteed here: the remote list kept only what this
    # account can reach; open weights are the local list's business.
    assert {m['source'] for m in body['models']} >= {'default', 'closed'}

    # --- the panel merges saved settings over these, so a field missing here
    # is a dropdown that renders as undefined on an old browser tab.
    defaults = body['defaults']
    assert defaults['retrieval']['reranker_model'] == ''
    assert defaults['retrieval']['grader_model'] == ''
    assert defaults['retrieval']['expansion_model'] == ''
    assert defaults['generation']['judge_model'] == ''
    assert defaults['generation']['ragas_model'] == ''
    assert defaults['index']['embed_model'] == ''

    # --- embedder hints: which language(s) each embedder covers, so an
    # English-only embedder cannot silently be picked for Farsi.
    hints = {hint['kind']: hint for hint in body['embedder_hints']}
    assert set(hints) == set(body['embedders'])
    assert all(hint['languages'] for hint in hints.values())
    assert hints['ascii-hash']['farsi'] is False

    # --- the embed-model catalogue: only what is actually installed, Farsi
    # coverage stated rather than guessed.
    assert body['embed_models'][0]['id'] == ''
    by_id = {entry['id']: entry for entry in body['embed_models']}
    assert by_id['intfloat/multilingual-e5-small']['farsi'] is True
    assert by_id['BAAI/bge-small-en-v1.5']['farsi'] is False
    assert body['help']['index.embed_model']
    assert 'sentence-transformers' in set(body['embedders'])
    assert 'openai' not in set(body['embedders'])
    for model_id, (backend, dim, _) in REQUESTED_MODELS.items():
        assert model_id in by_id, model_id
        assert by_id[model_id]['backend'] == backend
        assert by_id[model_id]['dim'] == dim
    # The panel reports what is installed, so a dropdown never promises a
    # download or an API call that cannot happen.
    caps = body['capabilities']
    assert isinstance(caps['sentence_transformers'], bool)
    assert 'openai_embeddings' not in caps

    # --- provider modes: each backend carries its own model catalogue,
    # never one list shared by all four.
    modes = {mode['key']: mode for mode in body['modes']}
    assert set(modes) == {'local', 'openrouter', 'claude', 'codex'}
    assert modes['local']['provider'] == 'ollama'
    assert all(isinstance(mode['models'], list) for mode in modes.values())

    # --- datasets: the built-in corpus leads, and the bundled samples
    # (the smoke set among them) are offered beside it.
    dataset_ids = [d['id'] for d in body['datasets']]
    assert dataset_ids[0] == 'diary-fa'
    assert 'smoke-mini' in dataset_ids


def test_options_advertise_no_vector_database(client):
    # this is an integration test
    """A positive statement rather than an absence: the panel has to say
    where an experiment's vectors live and where its durable artifacts land
    — merged with the health check, which makes the same claim its own way."""
    body = client.get('/api/options').json()
    caps = body['capabilities']
    assert [key for key in caps if 'chroma' in key] == []
    assert caps['storage']['index'] == 'memory'
    assert caps['storage']['runs'] == '.runs'
    assert caps['storage']['experiments'].endswith('raglab.db')
    assert 'ragas' in caps

    health = client.get('/api/health').json()
    assert health['ok'] and health['storage'] == 'memory'
    assert [key for key in health if 'chroma' in key or key == 'database'] == []


def test_query_rejects_an_unknown_strategy(client):
    # this is an integration test
    res = client.post('/api/queries', json={'question': 'x',
                                          'index': {'chunker': 'nope'}})
    assert res.status_code == 400
    assert 'unknown chunker' in res.json()['detail']


def test_questions_endpoint_hides_the_answers(client):
    # this is an integration test
    body = client.get('/api/questions?limit=5').json()
    assert len(body['questions']) == 5
    assert 'answer_fa' not in body['questions'][0]
    assert body['questions'][0]['evidence_sessions']


def test_evaluations_lists_and_fetches_the_same_resource(client):
    # this is an integration test
    """One noun, three operations, no second spelling for any of them.
    Moved from test_raglab.py — the same route-contract group as the
    run/runs collision test below."""
    assert 'runs' in client.get('/api/evaluations').json()
    assert client.get('/api/evaluations/no-such-run').status_code == 404


def test_an_experiment_resolves_from_the_run_file_when_the_ledger_has_none(client):
    # this is an integration test
    """The leaderboard is a union of two records — the ledger, and the run
    files in `.runs/` — because the ledger is written in `Jobs.run` and every
    evaluation that finished before it existed has a file and no row. Its
    per-row link resolves an experiment by id, so this route has to read the
    same union: answered from the ledger alone it failed on precisely the rows
    worth opening, since the older evaluations are the ones carrying a score.

    The projection is checked field by field against the ledger's own column
    list, so a run-file experiment and a ledger experiment are the same shape
    to every reader of this route."""
    import json

    from raglab.evaluation import run_evaluation as evaluate
    from raglab.evaluation import service_experiment_ledger as ledger

    run_id = '20260101-000000-runonly'
    payload = {
        'run_id': run_id, 'label': 'older than the ledger',
        'started_at': '2026-01-01 00:00:00', 'seconds': 12.5,
        'dataset': 'smoke-mini',
        'config': {'provider': 'fake',
                   'index': {'chunker': 'session', 'embedder': 'token-hash'},
                   'retrieval': {'retriever': 'bm25', 'reranker': 'none',
                                 'grader': 'none'},
                   'generation': {'answerer': 'llm'}},
        'summary': {'n_questions': 2},
        'ragas': {'decision': 0.5, 'decision_spread': {'stderr': 0.1}},
        'rows': [{'id': 'q-1', 'answer': 'an answer'}],
        'selection': {'n': 2},
    }
    evaluate.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (evaluate.RUNS_DIR / f'{run_id}.json').write_text(json.dumps(payload),
                                                      encoding='utf-8')
    try:
        assert ledger.experiment(run_id) is None, 'the ledger must not know it'
        found = client.get(f'/api/experiments/{run_id}')
        assert found.status_code == 200, found.text
        body = found.json()
        assert set(body) == set(ledger.COLUMNS) | {'detail'}
        assert body['experiment_id'] == run_id
        assert body['kind'] == 'run' and body['state'] == 'done'
        assert body['dataset'] == 'smoke-mini' and body['n_questions'] == 2
        assert body['retriever'] == 'bm25' and body['answerer'] == 'llm'
        assert body['decision'] == 0.5 and body['decision_stderr'] == 0.1
        # The rows travel; the traces, chunk text and summaries never reached
        # a run file, so the payload must not pretend to carry them.
        assert body['detail']['rows'] == payload['rows']
        for absent in ('traces', 'chunks_by_session', 'summaries'):
            assert absent not in body['detail']
        # And an id in neither record is still a 404, not an empty experiment.
        assert client.get('/api/experiments/in-neither-record').status_code == 404
    finally:
        (evaluate.RUNS_DIR / f'{run_id}.json').unlink()


def test_a_run_that_judged_nothing_reports_no_decision_rather_than_zero(client):
    # this is an integration test
    """`decision` is NULL, never 0.0, on every experiment that judged nothing —
    the ledger's own column says so in a comment. The run-file projection has
    to keep the same rule, or a run with ragas off resolves as a measured
    refusal and sorts below rows that were actually scored."""
    import json

    from raglab.evaluation import run_evaluation as evaluate

    run_id = '20260101-000001-nojudge'
    evaluate.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (evaluate.RUNS_DIR / f'{run_id}.json').write_text(
        json.dumps({'run_id': run_id, 'summary': {}, 'ragas': {}}),
        encoding='utf-8')
    try:
        body = client.get(f'/api/experiments/{run_id}').json()
        assert body['decision'] is None
        assert body['decision_stderr'] is None
        assert body['n_questions'] == 0
    finally:
        (evaluate.RUNS_DIR / f'{run_id}.json').unlink()


def test_the_run_and_runs_collision_is_gone(client):
    # this is an integration test
    """The old singular/plural split meant two unrelated things at one
    character apart. The new names are gone rather than aliased, since a
    second name for one thing is the thing this rename was fixing. Moved
    from test_raglab.py."""
    assert client.post('/api/run', json={}).status_code == 404
    assert client.get('/api/runs').status_code == 404
    assert client.post('/api/index', json={}).status_code == 404
    assert client.post('/api/query', json={'question': 'x'}).status_code == 404


def test_ad_hoc_query_returns_stages_and_contexts(client, smoke_index):
    # this is an integration test
    """The one kept round trip through `/api/queries`: the smoke set answers
    in milliseconds, so what this proves is the job's wiring, not a full
    corpus build. The Farsi time-scope label this test used to check against
    the diary now has its own direct unit assertion in
    `test_primitives.py::test_time_scopes_resolve_to_the_right_window` —
    a cheaper home for it than an HTTP round trip. `resolve_time_scope`
    matches only Farsi Jalali month/season/holiday names, and this question
    is English, so `time_scope` is expected to be `None` here rather than
    merely present — asserting the key merely exists would pass whether or
    not resolution ever ran, which is not a real check."""
    body = _ask(client, {
        'question': 'What broke in the kitchen?',
        'index': smoke_index.config,
        'retrieval': {'k': 3},
        'generation': {'answerer': 'extractive'}})
    assert body['contexts'] and body['answer']
    assert 'retrieve_ms' in body['timings']
    assert body['time_scope'] is None


def test_lab_config_accepts_the_embed_model_and_per_task_model_fields():
    # this is a unit test
    """The three query-round-trip jobs that used to prove `grader_model`,
    `judge_model` and `embed_model` survive a panel round trip did it by
    posting a whole job and waiting for it to finish. What they were
    actually proving lives in `LabConfig.from_dict` — called directly here,
    no job and no index build.

    `grader_model`/`judge_model` are retrieval/generation fields with no gate
    of their own, so they land on the config exactly as posted regardless of
    `grader`/`answerer`. `embed_model` is different: `IndexConfig.normalized()`
    (called from `from_dict`) blanks it for a non-model embedder so an unused
    value cannot move the fingerprint — that is the "survives even when the
    running embedder ignores it" the original test's docstring meant, and it
    is asserted both ways below."""
    cfg = LabConfig.from_dict({
        'index': {'chunker': 'message', 'embedder': 'sentence-transformers',
                  'embed_model': 'intfloat/multilingual-e5-small'},
        'retrieval': {'k': 4, 'grader_model': 'anthropic/claude-haiku-4.5'},
        'generation': {'answerer': 'extractive',
                       'judge_model': 'openai/gpt-5-mini'}})
    assert cfg.index.embed_model == 'intfloat/multilingual-e5-small'
    assert cfg.retrieval.grader_model == 'anthropic/claude-haiku-4.5'
    assert cfg.generation.judge_model == 'openai/gpt-5-mini'

    # A hash embedder ignores embed_model rather than erroring on it — a
    # stale browser tab posting both must not break the query.
    ignored = LabConfig.from_dict({
        'index': {'chunker': 'message', 'embedder': 'char-hash',
                  'embed_model': 'intfloat/multilingual-e5-small'}})
    assert ignored.index.embed_model == ''


def test_ragas_takes_its_own_judge_model(smoke_index):
    # this is an integration test
    """The judge is a config field of its own, not the answerer's model —
    proven against the smoke index rather than the 167-session diary build,
    since nothing about this claim needs the big corpus."""
    pytest.importorskip('ragas')
    pytest.importorskip('rapidfuzz')
    from raglab.corpora import dataset_import_contract as datasets
    from raglab.evaluation import ragas_judged_metrics as ragas_eval
    _, ground_truth = datasets.load('smoke-mini')
    question = next(q for q in ground_truth['questions'] if q['answerable'])
    pairs = [(question, pipeline.retrieve(smoke_index.index, RetrievalConfig(k=3),
                                          question['question_fa'],
                                          question['query_date']))]
    report = ragas_eval.run(pairs, LAB_SETTINGS, smoke_index.index.embedder,
                            mode='offline', judge_model='judge/model')
    assert report['n_samples'] == 1, report['notes']
