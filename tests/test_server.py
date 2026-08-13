"""The service — the lab's own FastAPI app, served at /api/options and the
ad hoc query endpoint."""
import pytest

from raglab import models, pipeline
from raglab.config import RetrievalConfig

from conftest import LAB_SETTINGS, OLLAMA_SETTINGS, REQUESTED_MODELS, _finished


# --- the service -----------------------------------------------------------

def _ask(client, payload: dict) -> dict:
    """POST one question and wait for its job — the panel's ask flow."""
    res = client.post('/api/queries', json=payload)
    assert res.status_code == 202, res.text
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    return job['result']


def test_options_describes_the_corpus_and_capabilities(client):
    body = client.get('/api/options').json()
    # Pinned rather than derived: the corpus is the measuring instrument, so a
    # change to its size should have to be stated here on purpose.
    assert body['corpus']['sessions'] == 167
    assert body['corpus']['questions'] == 112
    assert 'semantic-drift' in body['chunkers']
    assert 'ragas' in body['capabilities']


def test_options_advertises_no_vector_database(client):
    """A positive statement rather than an absence: the panel has to say
    where an experiment's vectors live and where its durable artifacts
    land."""
    caps = client.get('/api/options').json()['capabilities']
    assert [key for key in caps if 'chroma' in key] == []
    assert caps['storage']['index'] == 'memory'
    assert caps['storage']['runs'] == '.runs'
    assert caps['storage']['experiments'].endswith('raglab.db')


def test_health_says_the_lab_depends_on_no_service(client):
    body = client.get('/api/health').json()
    assert body['ok'] and body['storage'] == 'memory'
    assert [key for key in body if 'chroma' in key or key == 'database'] == []


def test_a_build_starts_without_any_service_running(client):
    """With the index in process memory there is nothing that can be down,
    so the gate that could refuse a build is gone rather than passing."""
    from raglab import server as lab_server

    assert not hasattr(lab_server, 'require_chroma')
    body = client.post('/api/indexes', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash',
                  'layers': ['session']}}).json()
    assert body['job_id']


def test_options_counts_the_habits_the_corpus_tracks(client):
    """The habit ledger is only as good as the habits behind it, so how many the
    fixture declares is part of describing the corpus."""
    corpus_facts = client.get('/api/options').json()['corpus']
    assert corpus_facts['habits'] == 5


def test_options_names_habit_as_a_question_type(client):
    """The per-type breakdown is where habit retrieval either shows up or hides
    inside the aggregation bucket."""
    assert 'habit' in client.get('/api/options').json()['question_types']


def test_options_explains_the_new_metadata_and_the_deciding_score(client):
    body = client.get('/api/options').json()
    assert body['help']['metric.ragas_decision']
    by_key = {measure['key']: measure for measure in body['metrics']}
    assert by_key['ragas_decision']['step'] == ''


def test_ad_hoc_query_returns_stages_and_contexts(client):
    body = _ask(client, {
        'question': 'آذر چه خبر بود؟',
        'index': {'chunker': 'message', 'embedder': 'char-hash',
                  'layers': ['chunk']},
        'retrieval': {'k': 4},
        'generation': {'answerer': 'extractive'}})
    assert body['contexts'] and body['answer']
    assert body['time_scope']['label'] == 'آذر'
    assert 'retrieve_ms' in body['timings']


def test_query_rejects_an_unknown_strategy(client):
    res = client.post('/api/queries', json={'question': 'x',
                                          'index': {'chunker': 'nope'}})
    assert res.status_code == 400
    assert 'unknown chunker' in res.json()['detail']


def test_questions_endpoint_hides_the_answers(client):
    body = client.get('/api/questions?limit=5').json()
    assert len(body['questions']) == 5
    assert 'answer_fa' not in body['questions'][0]
    assert body['questions'][0]['evidence_sessions']


def test_options_offers_a_model_choice_for_every_llm_task(client):
    body = client.get('/api/options').json()
    roles = {role['key']: role for role in body['model_roles']}
    assert set(roles) == {'expand', 'rerank', 'grade', 'answer',
                          'judge', 'ragas', 'plan', 'critic'}
    assert all(role['help'] and role['label'] and role['field']
               for role in roles.values())
    ids = [m['id'] for m in body['models']]
    # The backend's own default leads, because a slug only means something to the
    # backend serving it.
    assert ids[0] == ''
    # The *list* follows the configured backend: a fake backend serves no
    # local slugs, checked directly rather than against a daemon.
    assert '4skl/gemma4-e2b-mtp' not in ids, 'a fake backend serves no local slugs'
    assert '4skl/gemma4-e2b-mtp' in [m.id for m in models.known_models(
        OLLAMA_SETTINGS)]
    # 'open' is not guaranteed here: the remote list kept only what this
    # account can reach; open weights are the local list's business.
    assert {m['source'] for m in body['models']} >= {'default', 'closed'}


def test_options_explains_every_knob(client):
    body = client.get('/api/options').json()
    for key in ('index.chunker', 'retrieval.reranker',
                'retrieval.grade_threshold', 'generation.answerer', 'model.answer',
                'model.answer'):
        assert body['help'].get(key), key


def test_defaults_carry_the_per_task_model_fields(client):
    """The panel merges saved settings over these, so a field missing here is a
    dropdown that renders as undefined on an old browser tab."""
    defaults = client.get('/api/options').json()['defaults']
    assert defaults['retrieval']['reranker_model'] == ''
    assert defaults['retrieval']['grader_model'] == ''
    assert defaults['retrieval']['expansion_model'] == ''
    assert defaults['generation']['judge_model'] == ''
    assert defaults['generation']['ragas_model'] == ''


def test_a_per_task_model_is_accepted_by_the_query_endpoint(client):
    body = _ask(client, {
        'question': 'آذر چه خبر بود؟',
        'index': {'chunker': 'message', 'embedder': 'char-hash', 'layers': ['chunk']},
        'retrieval': {'k': 4,
                      'grader_model': 'anthropic/claude-haiku-4.5'},
        'generation': {'answerer': 'extractive',
                       'judge_model': 'openai/gpt-5-mini'}})
    assert body['contexts']


def test_ragas_takes_its_own_judge_model(index, ground_truth):
    pytest.importorskip('ragas')
    pytest.importorskip('rapidfuzz')
    from raglab import ragas_eval
    question = next(q for q in ground_truth['questions'] if q['answerable'])
    pairs = [(question, pipeline.retrieve(index, RetrievalConfig(k=5),
                                          question['question_fa'],
                                          question['query_date']))]
    report = ragas_eval.run(pairs, LAB_SETTINGS, index.embedder, mode='offline',
                            judge_model='judge/model')
    assert report['n_samples'] == 1, report['notes']


def test_options_say_which_languages_each_embedder_covers(client):
    body = client.get('/api/options').json()
    hints = {hint['kind']: hint for hint in body['embedder_hints']}
    assert set(hints) == set(body['embedders'])
    assert all(hint['languages'] for hint in hints.values())
    assert hints['ascii-hash']['farsi'] is False


def test_options_offer_farsi_capable_embedding_models(client):
    body = client.get('/api/options').json()
    assert body['embed_models'][0]['id'] == ''
    by_id = {entry['id']: entry for entry in body['embed_models']}
    assert by_id['intfloat/multilingual-e5-small']['farsi'] is True
    assert by_id['BAAI/bge-small-en-v1.5']['farsi'] is False
    assert body['defaults']['index']['embed_model'] == ''
    assert body['help']['index.embed_model']


def test_an_embedding_model_is_accepted_by_the_query_endpoint(client):
    """The field has to survive the panel round trip even when the running
    embedder ignores it, or a stale tab breaks a query."""
    body = _ask(client, {
        'question': 'آذر چه خبر بود؟',
        'index': {'chunker': 'message', 'embedder': 'char-hash',
                  'embed_model': 'intfloat/multilingual-e5-small',
                  'layers': ['chunk']},
        'retrieval': {'k': 4},
        'generation': {'answerer': 'extractive'}})
    assert body['contexts']


def test_options_define_every_metric_the_panel_can_show(client):
    body = client.get('/api/options').json()
    by_key = {measure['key']: measure for measure in body['metrics']}
    # Deterministic and judged metrics arrive through the same shape, so the
    # dashboard renders one concept rather than two.
    for key in ('recall', 'quote_recall', 'headline', 'faithfulness',
                'non_llm_context_recall'):
        measure = by_key.get(key)
        assert measure, key
        assert measure['label'] and measure['short'], key
        assert measure['formula'] and measure['library'] and measure['help'], key
        assert 'step' in measure, key
    assert body['help']['metric.recall']
    assert 'ragas' in by_key['faithfulness']['library'].lower()


def test_options_colour_code_the_pipeline_steps(client):
    """The panel cannot invent the grouping: which step a control belongs to is
    a fact about the pipeline, served with everything else."""
    body = client.get('/api/options').json()
    assert [step['key'] for step in body['steps']] == ['index', 'retrieval',
                                                       'generation', 'agent']
    assert all(step['label'] and step['short'] and step['note']
               for step in body['steps'])
    steps = {step['key'] for step in body['steps']}
    assert all(role['step'] in steps for role in body['model_roles'])
    by_key = {role['key']: role['step'] for role in body['model_roles']}
    assert by_key['rerank'] == 'retrieval'
    assert by_key['answer'] == 'generation'


def test_options_offer_the_local_backend_and_its_models(client):
    body = client.get('/api/options').json()
    assert 'sentence-transformers' in set(body['embedders'])
    assert 'openai' not in set(body['embedders'])
    by_id = {entry['id']: entry for entry in body['embed_models']}
    for model_id, (backend, dim, _) in REQUESTED_MODELS.items():
        assert model_id in by_id, model_id
        assert by_id[model_id]['backend'] == backend
        assert by_id[model_id]['dim'] == dim
    # The panel reports what is installed, so a dropdown never promises a
    # download or an API call that cannot happen.
    caps = body['capabilities']
    assert isinstance(caps['sentence_transformers'], bool)
    assert 'openai_embeddings' not in caps
