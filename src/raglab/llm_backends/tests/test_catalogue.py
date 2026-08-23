"""Picking a model per task, picking an embedder that can read the corpus,
and the two catalogues (chat models, embedding models) that offer only what
has actually run on this machine."""
from dataclasses import replace

import numpy as np
import pytest

from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.evaluation import run_evaluation as evaluate
from raglab.configuration import explainer_assembly as explain
from raglab.llm_backends import model_role_catalogue as models
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.configuration.lab_config import (
    EMBEDDERS,
    RERANKERS,
    GenerationConfig,
    IndexConfig,
    LabConfig,
    LabSettings,
    RetrievalConfig)
from raglab.rag_components.indexing.index_builder_registry import LabIndex

from raglab.conftest import LAB_SETTINGS, REQUESTED_MODELS


# --- picking a model per task ----------------------------------------------
# Every stage that can call a language model carries its own choice, since
# a judge and a cheap reranker want very different things from one model.

class Recorder:
    """A provider that remembers which model each stage asked for.

    The reply is shaped like an llm_scores answer so the reranking and grading
    stages parse it and carry on; the answer stage just repeats it back."""

    def __init__(self, reply: str = '1: 9\n2: 9\n3: 9\n4: 9'):
        self.reply = reply
        self.calls: list[str] = []

    def invoke(self, messages, model='', **kwargs):
        # '' is the default because lab_chat omits the kwarg entirely for a
        # stage with no model choice — which is what "leave it to the provider"
        # has to look like on the wire.
        self.calls.append(model)
        return type('Turn', (), {'content': self.reply, 'tool_calls': []})()


def test_every_llm_stage_has_a_role_in_the_registry():
    # this is a convention test
    # Every stage that calls a model carries its own, rather than borrowing the
    # answerer's: a gate runs k times a question and wants cheap, a judge runs
    # once and wants strong.
    assert {role.key for role in models.ROLES} == {
        'expand', 'rerank', 'grade', 'answer', 'judge', 'ragas'}


def test_every_model_role_points_at_a_real_config_field():
    # this is a convention test
    cfg = LabConfig()
    for role in models.ROLES:
        group, _, field = role.field.partition('.')
        assert field in getattr(cfg, group).__dataclass_fields__, role.key


def test_every_model_in_the_catalogue_declares_where_its_weights_stand():
    # this is a unit test
    entries = models.catalogue(LAB_SETTINGS)
    assert entries[0]['id'] == ''          # the lab default stays the first choice
    assert all(e['source'] in ('default', 'open', 'closed') for e in entries)
    assert any(e['source'] == 'closed' for e in entries)
    # 'open' lives on the local Ollama list — still declared even though the
    # remote list's open-weight options all answered 404 on this account.
    assert all(option.source == 'open' for option in models.OLLAMA_MODELS)
    assert all(e['label'] for e in entries)


def test_the_chat_catalogue_never_drops_a_model_it_cannot_verify():
    # this is a unit test
    """A model this lab has not actually run is still worth trying, so it
    stays in the list marked NA — whether it is a known entry nobody could
    reach (no key to probe with) or a model named only by RAGLAB_MODEL and
    not in the registry at all. Silently omitting either would hide the
    option."""
    entries = models.catalogue(LAB_SETTINGS)     # no API key: nothing to probe
    assert any(not e['available'] for e in entries)
    by_id = {e['id']: e for e in entries}
    assert by_id[LAB_SETTINGS.llm_model]['available']

    settings = replace(LAB_SETTINGS, llm_model='someone/custom-7b')
    custom = models.catalogue(settings)
    assert 'someone/custom-7b' in [e['id'] for e in custom]
    assert custom[0]['label'].endswith('someone/custom-7b)')


def test_a_blank_role_falls_back_to_the_lab_default_model():
    # this is a unit test
    settings = replace(LAB_SETTINGS, llm_model='lab/default')
    roles = models.resolve(LabConfig(), settings)
    assert roles.answer == 'lab/default' and roles.grade == 'lab/default'
    assert roles.ragas == 'lab/default' and roles.judge == 'lab/default'


def test_each_role_round_trips_from_the_panels_json():
    # this is a unit test
    cfg = LabConfig.from_dict({
        'retrieval': {'reranker_model': 'rerank/model', 'grader_model': 'grade/model',
                      'expansion_model': 'hyde/model'},
        'generation': {'model': 'answer/model', 'judge_model': 'judge/model',
                       'ragas_model': 'ragas/model'}})
    roles = models.resolve(cfg, LAB_SETTINGS)
    assert (roles.rerank, roles.grade, roles.expand, roles.answer,
            roles.judge, roles.ragas) == (
        'rerank/model', 'grade/model', 'hyde/model', 'answer/model',
        'judge/model', 'ragas/model')


def test_each_stage_calls_the_model_chosen_for_its_own_role(index):
    # this is an integration test
    """The point of per-task models: a cheap reranker and an expensive answerer
    in the same run. One model for everything makes that impossible to measure."""
    cfg = LabConfig(
        retrieval=RetrievalConfig(k=3, rerank_depth=3, reranker='llm',
                                  reranker_model='rerank/model', grader='llm',
                                  grader_model='grade/model', hyde=True,
                                  expansion_model='hyde/model'),
        generation=GenerationConfig(answerer='llm', model='answer/model'))
    roles = models.resolve(cfg, LAB_SETTINGS)
    provider = Recorder()
    outcome = pipeline.retrieve(index, cfg.retrieval, 'قرار بود چی کار کنم؟',
                                '2026-07-28', llm=provider, models=roles)
    pipeline.answer(outcome, cfg.generation, llm=provider, models=roles)
    assert provider.calls == ['hyde/model', 'rerank/model', 'grade/model',
                             'answer/model']


def test_a_stage_with_no_model_choice_leaves_it_to_the_provider(index):
    # this is an integration test
    """'' rather than a guess: the provider already knows its default model, and
    a lab that hard-codes one here would silently ignore RAGLAB_MODEL."""
    provider = Recorder()
    pipeline.retrieve(index, RetrievalConfig(k=3, rerank_depth=3, reranker='llm'),
                      'قول باشگاه', '2026-07-28', llm=provider)
    assert provider.calls == ['']


def test_the_fact_judge_uses_the_judge_model():
    # this is a unit test
    provider = Recorder(reply='1: yes')
    question = {'expected_answer': {'derived_facts': [
        {'derived_fact_id': 1, 'fact': 'he went to the gym'}]}}
    score = evaluate.judge_derived_facts(provider, 'judge/model', question, 'رفتم')
    assert provider.calls == ['judge/model']
    assert score == pytest.approx(1.0)


def test_the_explainers_cover_the_model_roles_too():
    # this is a convention test
    """`explain.missing() == []` (test_conventions.py) is the completeness gate;
    this pins the content it is gating for the six chat-model roles."""
    topics = explain.topics()
    for role in models.ROLES:
        assert topics[f'model.{role.key}'], role.key
    assert 'model.answer' in topics and len(topics['model.answer']) > 20


# --- picking an embedder that can read the corpus --------------------------
# The embedder decides everything downstream, and on a Farsi diary most of the
# well-known ones cannot represent the text at all: the brain's default tokenises
# [a-z0-9]+, and the fastembed default the brain hardwires (bge-small-**en**) is
# an English model. A dropdown that does not say which languages an option covers
# is how a run ends up measuring nothing — so language coverage is part of every
# entry, and the Farsi-capable models are offered by name.

FARSI_MODELS = ('heydariAI/persian-embeddings',
                'intfloat/multilingual-e5-small',
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


class FakeTextEmbedding:
    """Stands in for fastembed's TextEmbedding: records exactly what it was
    asked to encode, so the E5 prefixes can be asserted with no model download."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.seen: list[str] = []

    def embed(self, texts, batch_size=None):
        for text in list(texts):
            self.seen.append(text)
            vector = np.zeros(self.dim, dtype=np.float32)
            vector[len(text) % self.dim] = 1.0
            yield vector


class FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer, recording every
    text it was asked to encode so the prefix behaviour can be asserted."""

    def __init__(self, name: str, dim: int = 8):
        self.name = name
        self.dim = dim
        self.seen: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        return np.ones((len(list(texts)), self.dim), dtype=np.float32)


class FakeSentenceTransformerModern(FakeSentenceTransformer):
    """The sentence-transformers 5 shape — `get_embedding_dimension` rather
    than `get_sentence_embedding_dimension` — so the wrapper's `getattr(...)
    or ...` fallback has a modern method to prefer, not just a legacy one to
    fall back to. The legacy method raises if called, so a regression that
    stops preferring the modern one is caught rather than passing by
    coincidence (both return the same `dim`)."""

    def get_embedding_dimension(self) -> int:
        return self.dim

    def get_sentence_embedding_dimension(self) -> int:
        raise AssertionError('the modern get_embedding_dimension must be '
                              'preferred over the legacy method')


class _Asymmetric:
    """An embedder whose query and passage sides are genuinely different, so
    `query_vectors` routing to `embed_queries` is a real distinction and not a
    coincidence of two identical calls."""
    dim = 2
    name = 'asymmetric'

    def __init__(self):
        self.as_query: list[str] = []

    def embed(self, texts):
        return np.zeros((len(list(texts)), 2), dtype=np.float32)

    def embed_queries(self, texts):
        self.as_query.extend(texts)
        return np.ones((len(list(texts)), 2), dtype=np.float32)


def test_every_embedder_hint_says_which_languages_it_covers():
    # this is a unit test
    """A hint per option, covering the whole registry: an embedder the panel
    offers without saying what it can read is a run nobody can interpret.
    ascii-hash tokenises [a-z0-9]+, so it is Latin-only and has to say so
    rather than let that be discovered by a run; the three Farsi-capable
    kinds have to say the opposite."""
    hints = {hint['kind']: hint for hint in embedding.embedder_hints()}
    assert set(hints) == set(EMBEDDERS)
    assert all(h['languages'] and h['label'] and h['note'] for h in hints.values())
    assert hints['ascii-hash']['farsi'] is False
    assert 'latin' in hints['ascii-hash']['languages'].lower()
    for kind in ('token-hash', 'char-hash', 'fastembed', 'sentence-transformers'):
        assert hints[kind]['farsi'] is True, kind


def test_the_embedding_model_catalogue_offers_models_that_speak_farsi():
    # this is a unit test
    entries = embedding.embed_model_catalogue(LAB_SETTINGS)
    assert entries[0]['id'] == ''            # the lab default stays first
    by_id = {entry['id']: entry for entry in entries}
    for model in FARSI_MODELS:
        assert by_id[model]['farsi'] is True, model
        assert 'farsi' in by_id[model]['languages'].lower(), model
    assert LAB_SETTINGS.fastembed_model in by_id
    assert all(e['languages'] and e['label'] and e['note'] for e in entries)
    assert all(e['source'] in ('default', 'open', 'closed', 'unknown')
               for e in entries)


def test_an_english_only_model_is_offered_but_says_so(monkeypatch):
    # this is a unit test
    """The brain hardwires bge-small-en today. The lab must be able to measure
    that choice, and must never let it be picked by accident."""
    monkeypatch.setattr(embedding, 'fastembed_available', lambda: True)
    monkeypatch.setattr(embedding, 'fastembed_models',
                        lambda: frozenset(embedding.MODEL_IDS))
    by_id = {e['id']: e for e in embedding.embed_model_catalogue(LAB_SETTINGS)}
    english = by_id['BAAI/bge-small-en-v1.5']
    assert english['farsi'] is False
    assert 'english' in english['languages'].lower()
    assert english['available'] is True      # installable, just wrong for Farsi


# NA means one thing only: *this installation* cannot load it. For fastembed
# that is two independent gates — the import succeeding and fastembed's own
# served-model list naming the model — an older fastembed serves a shorter
# list than the registry promises. sentence-transformers has no served-list
# concept, so an import check alone decides.
@pytest.mark.parametrize(
    'backend,avail,served,model_id,expected,check_sort', [
        ('fastembed', True,
         {'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'},
         'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
         True, True),
        ('fastembed', True,
         {'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'},
         'BAAI/bge-small-en-v1.5', False, False),
        ('fastembed', False, frozenset(embedding.MODEL_IDS),
         'BAAI/bge-small-en-v1.5', False, False),
        ('fastembed', False, frozenset(embedding.MODEL_IDS),
         'sentence-transformers/all-MiniLM-L6-v2', False, False),
        ('fastembed', True, frozenset(embedding.MODEL_IDS),
         'BAAI/bge-small-en-v1.5', True, False),
        ('sentence-transformers', False, None,
         'intfloat/multilingual-e5-small', False, False),
        ('sentence-transformers', False, None,
         'heydariAI/persian-embeddings', False, False),
        ('sentence-transformers', True, None,
         'intfloat/multilingual-e5-small', True, False),
    ])
def test_a_model_reads_NA_only_when_this_installation_cannot_load_it(
        monkeypatch, backend, avail, served, model_id, expected, check_sort):
    # this is a unit test
    if backend == 'fastembed':
        monkeypatch.setattr(embedding, 'fastembed_available', lambda: avail)
        monkeypatch.setattr(embedding, 'fastembed_models',
                            lambda: frozenset(served))
    else:
        monkeypatch.setattr(embedding, 'sentence_transformers_available',
                            lambda: avail)
    entries = embedding.embed_model_catalogue(LAB_SETTINGS)
    by_id = {entry['id']: entry for entry in entries}
    assert by_id[model_id]['available'] is expected
    if check_sort:
        flags = [entry['available'] for entry in entries]
        assert flags == sorted(flags, reverse=True), 'usable models come first'


@pytest.mark.parametrize('model_id,backend,expected', [
    ('intfloat/multilingual-e5-small', 'sentence-transformers',
     ('query: ', 'passage: ')),
    ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
     'fastembed', ('', '')),
])
def test_the_catalogue_declares_the_prefixes_a_model_was_trained_with(
        model_id, backend, expected):
    # this is a unit test
    """E5 was trained with "query: " / "passage: "; the plain multilingual
    model gets neither. Both the prefix and the backend that serves a model
    belong to the catalogue entry, not to whichever caller happens to load
    it."""
    by_id = {e['id']: e for e in embedding.embed_model_catalogue(LAB_SETTINGS)}
    entry = by_id[model_id]
    assert entry['backend'] == backend
    assert (entry['query_prefix'], entry['passage_prefix']) == expected


@pytest.mark.parametrize('kind,wrapper_name,fake_factory,model_id', [
    ('fastembed', 'FastEmbedMultilingual', lambda: FakeTextEmbedding(),
     'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
    ('sentence-transformers', 'SentenceTransformerEmbedder',
     lambda: FakeSentenceTransformer('intfloat/multilingual-e5-small'),
     'intfloat/multilingual-e5-small'),
    ('sentence-transformers', 'SentenceTransformerEmbedder',
     lambda: FakeSentenceTransformerModern('intfloat/multilingual-e5-small'),
     'intfloat/multilingual-e5-small'),
])
def test_a_prefixed_wrapper_marks_queries_and_passages_apart(
        monkeypatch, kind, wrapper_name, fake_factory, model_id):
    # this is a unit test
    """Declaring a prefix is worthless if the wrapper that actually calls the
    model does not apply it — both backends' classes must mark queries and
    passages apart identically, and the vectors that come back must still be
    the width the model actually reports (SentenceTransformerEmbedder reads
    that off `get_embedding_dimension`/`get_sentence_embedding_dimension`,
    whichever the installed version carries, and `_normalize` must not change
    it)."""
    fake = fake_factory()
    wrapper = getattr(embedding, wrapper_name)
    embedder = wrapper(model_id, query_prefix='query: ', passage_prefix='passage: ',
                       factory=lambda name: fake)
    fake.seen.clear()                      # drop anything the constructor's own probe encoded
    passages = embedder.embed(['امروز جلسه داشتم'])
    queries = embedder.embed_queries(['جلسه کی بود؟'])
    assert fake.seen == ['passage: امروز جلسه داشتم', 'query: جلسه کی بود؟']
    assert embedder.dim == fake.dim == passages.shape[1] == queries.shape[1]
    assert model_id in embedder.name
    if kind == 'sentence-transformers':
        # And make_embedder must resolve that same prefix from the catalogue
        # on its own, not merely support a caller who passes it explicitly —
        # otherwise the declaration above is a dict entry nothing reads.
        seen: dict = {}
        monkeypatch.setattr(embedding, wrapper_name,
                            lambda name, **kw: seen.update({'model': name} | kw))
        embedding.make_embedder(kind, LAB_SETTINGS, model_id)
        assert seen['model'] == model_id
        assert seen['query_prefix'] == 'query: '
        assert seen['passage_prefix'] == 'passage: '


@pytest.mark.parametrize('symmetric', [False, True])
def test_query_vectors_uses_the_query_method_only_when_the_model_has_one(
        symmetric):
    # this is a unit test
    """Every hash embedder embeds both sides the same way and must keep
    working without knowing this distinction exists; an asymmetric model must
    be routed through its own query method rather than treated like a
    passage."""
    if symmetric:
        vectors = embedding.query_vectors(embedding.make_embedder('char-hash'),
                                          ['سلام'])
        assert vectors.shape[0] == 1 and np.any(vectors)
    else:
        embedder = _Asymmetric()
        vectors = embedding.query_vectors(embedder, ['سلام'])
        assert embedder.as_query == ['سلام']
        assert vectors.shape == (1, 2) and vectors.any()


def test_dense_retrieval_embeds_the_question_as_a_query(index, monkeypatch):
    # this is an integration test
    """The prefix is worthless if retrieval bypasses it, so the pipeline must go
    through the query seam rather than calling embed() itself."""
    real = embedding.query_vectors
    seen: list[list[str]] = []

    def spy(embedder, texts):
        seen.append(list(texts))
        return real(embedder, texts)

    monkeypatch.setattr(pipeline.embedding, 'query_vectors', spy)
    pipeline.retrieve(index, RetrievalConfig(retriever='dense', k=3,
                                            multi_query=False, time_filter=False),
                      'قول باشگاه', '2026-07-28')
    assert seen and seen[0] == ['قول باشگاه']


def test_the_embedding_model_names_the_collection_only_when_it_is_used():
    # this is a unit test
    """A model nobody loads must not invalidate an index and cost a rebuild;
    a real backend's model choice must, or two different encoders would share
    one collection name."""
    hashed = IndexConfig(embedder='char-hash')
    assert hashed.fingerprint() == \
        replace(hashed, embed_model='BAAI/bge-small-en-v1.5').fingerprint()
    real = IndexConfig(embedder='fastembed')
    assert real.fingerprint() != \
        replace(real, embed_model='BAAI/bge-small-en-v1.5').fingerprint()
    for kind in ('fastembed', 'sentence-transformers'):
        kept = IndexConfig(embedder=kind, embed_model='some/model').normalized()
        assert kept.embed_model == 'some/model', kind
    dropped = IndexConfig(embedder='char-hash', embed_model='some/model').normalized()
    assert dropped.embed_model == ''
    a = IndexConfig(embedder='sentence-transformers',
                    embed_model='heydariAI/persian-embeddings')
    b = IndexConfig(embedder='sentence-transformers',
                    embed_model='intfloat/multilingual-e5-small')
    assert a.fingerprint() != b.fingerprint()


_BLANK_DEFAULT_FACTORIES = {
    'fastembed': ('_text_embedding', lambda name: FakeTextEmbedding()),
    'sentence-transformers': ('_sentence_transformer',
                              lambda name: FakeSentenceTransformer(name)),
}


@pytest.mark.parametrize('kind,named_model,default_fragment', [
    ('fastembed', 'some/fastembed-model', LAB_SETTINGS.fastembed_model),
    ('sentence-transformers', 'intfloat/multilingual-e5-small',
     'heydariAI/persian-embeddings'),
])
def test_a_blank_embedding_model_keeps_following_the_backends_own_default(
        monkeypatch, kind, named_model, default_fragment):
    # this is a unit test
    """'' means "that backend's default" — RAGLAB_FASTEMBED_MODEL for
    fastembed, the lab's Persian encoder for sentence-transformers — the same
    rule '' plays for every chat-model role, and the lab never hard-codes one
    of its own."""
    factory_attr, fake_factory = _BLANK_DEFAULT_FACTORIES[kind]
    monkeypatch.setattr(embedding, factory_attr, fake_factory)
    named = embedding.make_embedder(kind, LAB_SETTINGS, named_model)
    assert named_model in named.name
    default = embedding.make_embedder(kind, LAB_SETTINGS, '')
    assert default_fragment in default.name


def test_the_index_builds_with_the_embedding_model_from_its_config(monkeypatch,
                                                                  diary):
    # this is an integration test
    from raglab.rag_components.indexing import (
    index_builder_registry as index_module)
    seen: list[tuple] = []

    def spy(kind, settings=None, model=''):
        seen.append((kind, model))
        return embedding.CharHashEmbedder()   # anything that embeds, offline

    monkeypatch.setattr(index_module.embedding, 'make_embedder', spy)
    cfg = IndexConfig(chunker='session', embedder='fastembed',
                      embed_model='BAAI/bge-small-en-v1.5')
    LabIndex.build(cfg, {'sessions': diary['sessions'][:2], 'threads': {}},
                   LAB_SETTINGS)
    assert seen == [('fastembed', 'BAAI/bge-small-en-v1.5')]


def test_the_language_note_names_the_model_that_was_actually_used():
    # this is a unit test
    note = embedding.language_note('fastembed', 'BAAI/bge-small-en-v1.5')
    assert 'bge-small-en' in note and 'english' in note.lower()
    assert 'ascii-hash' in embedding.language_note('ascii-hash', '')


def test_a_run_records_which_languages_its_embedder_can_represent(
        registry, ground_truth, tmp_path, monkeypatch):
    # this is an integration test
    """A leaderboard row whose embedder could not read the corpus is not a
    result, and three days later nothing on the row says so."""
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    cfg = LabConfig(index=IndexConfig(chunker='fixed', embedder='ascii-hash',
                                      contextual=False),
                    retrieval=RetrievalConfig(k=4),
                    generation=GenerationConfig(answerer='none'))
    result = evaluate.run_eval(registry, ground_truth, cfg, LAB_SETTINGS,
                               limit=2, ragas_mode='off')
    notes = ' '.join(result.notes).lower()
    assert 'ascii-hash' in notes and 'latin' in notes


def test_the_embedding_model_knob_explains_itself():
    # this is a convention test
    topics = explain.topics()
    assert 'farsi' in topics['index.embed_model'].lower()
    assert 'farsi' in topics['index.embedder'].lower()


# --- the catalogue offers only what has run on this machine -----------------
#
# NA means "this installation cannot load it", not "unmeasured" — a remote
# or embedding entry lists only what actually answered here. The local Ollama
# list keeps the old rule instead, since there NA is honest: a tag not yet
# pulled is one `ollama pull` away, and the daemon is asked directly.

REACHABLE_CHAT = ('openai/gpt-5-nano', 'openai/gpt-5-mini',
                  'anthropic/claude-haiku-4.5', 'google/gemini-2.5-flash')

# Every open-weight option the remote list had; all answered 404 on this
# account.
UNREACHABLE_CHAT = ('openai/gpt-5', 'meta-llama/llama-3.3-70b-instruct',
                    'qwen/qwen-2.5-72b-instruct', 'google/gemma-3-27b-it',
                    'mistralai/mistral-nemo', 'deepseek/deepseek-chat')

# Each of these embedded a Farsi sentence here, through the backend it names.
VERIFIED_EMBED = {
    'heydariAI/persian-embeddings': 'sentence-transformers',
    'intfloat/multilingual-e5-small': 'sentence-transformers',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 'fastembed',
    'BAAI/bge-small-en-v1.5': 'fastembed',
    'sentence-transformers/all-MiniLM-L6-v2': 'fastembed',
}


def test_the_remote_catalogue_offers_only_models_this_account_can_reach():
    # this is a convention test
    ids = {option.id for option in models.CHAT_MODELS}
    assert ids == set(REACHABLE_CHAT)
    assert not ids & set(UNREACHABLE_CHAT)
    # And the local list is deliberately untouched: llama3.1:8b is not installed,
    # reads NA, and stays, because pulling it is a one-line fix by the user.
    assert 'llama3.1:8b' in {option.id for option in models.OLLAMA_MODELS}


def test_the_embedding_catalogue_offers_only_models_that_loaded_here():
    # this is a convention test
    assert {m.id: m.backend for m in embedding.EMBED_MODELS} == VERIFIED_EMBED


@pytest.mark.parametrize('backend', ['openai', 'openrouter'])
def test_a_backend_whose_catalogue_is_gone_is_not_offered_as_one(backend):
    # this is a convention test
    """Availability is verified, never guessed: OpenRouter's catalogue has not
    one embedding or rerank entry, though the gateway answers 401 rather than
    404 on those routes — a route with no servable model is not a backend.
    The openai backend's whole catalogue left with it on 2026-08-02. Either
    way, a backend whose every model is gone is still selectable in
    principle, and would build an embedder with no model and dim 0."""
    assert backend not in EMBEDDERS
    assert backend not in embedding.BACKENDS
    assert backend not in embedding.BACKEND_DEFAULTS
    if backend == 'openai':
        assert not hasattr(embedding, 'OpenAIEmbedder')
        assert backend not in {hint['kind'] for hint in embedding.embedder_hints()}
        with pytest.raises(ValueError):
            embedding.make_embedder('openai', LAB_SETTINGS)
        # The key goes too, rather than sitting in the settings advertising a
        # backend that is not there.
        assert not hasattr(LabSettings(), 'openai_api_key')
    else:
        # An unservable *reranker* is the worse half: `pipeline._rerank`
        # swallows every exception and returns the pre-rerank order, so such
        # a candidate would report itself as reranked while doing nothing.
        assert 'rerank-4-fast' not in RERANKERS
        # And neither is reachable as a default, which is how they arrived.
        assert IndexConfig().embedder == 'sentence-transformers'
        assert RetrievalConfig().reranker == 'lexical'


def test_the_catalogue_offers_every_requested_model_with_its_backend():
    # this is a convention test
    """Audited 2026-08-02: every entry the catalogue offers actually loaded
    here, through the backend it names, and speaks Farsi — and no entry in
    the wider catalogue names a backend the lab does not have."""
    by_id = {model.id: model for model in embedding.EMBED_MODELS}
    for model_id, (backend, dim, source) in REQUESTED_MODELS.items():
        entry = by_id.get(model_id)
        assert entry is not None, model_id
        assert (entry.backend, entry.dim, entry.source) == (backend, dim, source)
        assert entry.farsi and entry.note, model_id
    assert all(model.backend in embedding.BACKENDS
               for model in embedding.EMBED_MODELS)
    assert set(embedding.BACKENDS) <= set(EMBEDDERS)


def test_the_persian_tuned_model_is_the_default():
    # this is a unit test
    """A Farsi corpus deserves a Persian-tuned encoder, and it is the
    cheapest real encoder verified here."""
    assert IndexConfig().embedder == 'sentence-transformers'
    assert IndexConfig().embed_model == ''      # '' = the backend's default
    assert embedding.BACKEND_DEFAULTS['sentence-transformers'] == \
        'heydariAI/persian-embeddings'
    assert embedding.resolve_model('sentence-transformers', LAB_SETTINGS, '') == \
        'heydariAI/persian-embeddings'
    by_id = {m.id: m for m in embedding.EMBED_MODELS}
    entry = by_id['heydariAI/persian-embeddings']
    # Visible in the option itself, not only behind the explainer: the standing is
    # what you are looking for while the dropdown is open.
    assert entry.tag == 'lab default'
    assert 'persian' in entry.languages.lower() or 'farsi' in entry.languages.lower()
    # RAGLAB_FASTEMBED_MODEL still drives the fastembed backend, untouched.
    assert embedding.resolve_model('fastembed', LAB_SETTINGS, '') == \
        LabSettings().fastembed_model


def test_a_model_from_the_wrong_backend_is_refused_before_the_run():
    # this is a unit test
    """Picking a HuggingFace checkpoint while the embedder is fastembed must
    not silently fall back to the default — a run labelled with one model
    that measured another."""
    problems = LabConfig(index=IndexConfig(
        embedder='fastembed',
        embed_model='heydariAI/persian-embeddings')).validate()
    assert any('sentence-transformers' in problem for problem in problems)
    assert LabConfig(index=IndexConfig(
        embedder='sentence-transformers',
        embed_model='heydariAI/persian-embeddings')).validate() == []


def test_the_embedder_explainer_says_how_to_reach_a_model_it_cannot_download():
    # this is a convention test
    """Two backends is a choice nobody can make from the kind names alone."""
    text = explain.topics()['index.embedder'].lower()
    assert 'sentence-transformers' in text and 'fastembed' in text
