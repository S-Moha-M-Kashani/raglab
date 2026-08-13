"""Picking a model per task, picking an embedder that can read the corpus,
and the two catalogues (chat models, embedding models) that offer only what
has actually run on this machine."""
from dataclasses import replace

import numpy as np
import pytest

from raglab import embedding, evaluate, explain, models, pipeline
from raglab.config import (EMBEDDERS, RERANKERS, GenerationConfig, IndexConfig,
                            LabConfig, LabSettings, RetrievalConfig)
from raglab.index import LabIndex

from conftest import LAB_SETTINGS, REQUESTED_MODELS


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
    # The agent's planner and critic are LLM stages like any other, so they
    # carry their own model rather than borrowing the answerer's.
    assert {role.key for role in models.ROLES} == {
        'expand', 'rerank', 'grade', 'answer', 'judge', 'ragas', 'plan',
        'critic'}


def test_every_model_role_points_at_a_real_config_field():
    cfg = LabConfig()
    for role in models.ROLES:
        group, _, field = role.field.partition('.')
        assert field in getattr(cfg, group).__dataclass_fields__, role.key


def test_every_model_in_the_catalogue_declares_where_its_weights_stand():
    entries = models.catalogue(LAB_SETTINGS)
    assert entries[0]['id'] == ''          # the lab default stays the first choice
    assert all(e['source'] in ('default', 'open', 'closed') for e in entries)
    assert any(e['source'] == 'closed' for e in entries)
    # 'open' lives on the local Ollama list — still declared even though the
    # remote list's open-weight options all answered 404 on this account.
    assert all(option.source == 'open' for option in models.OLLAMA_MODELS)
    assert all(e['label'] for e in entries)


def test_an_unverified_model_is_offered_as_unavailable_rather_than_dropped():
    """A model this lab has not actually run is still worth trying, so it stays
    in the list marked NA. Silently omitting it would hide the option."""
    entries = models.catalogue(LAB_SETTINGS)     # no API key: nothing to probe
    assert any(not e['available'] for e in entries)
    by_id = {e['id']: e for e in entries}
    assert by_id[LAB_SETTINGS.llm_model]['available']


def test_the_configured_model_is_always_offered_even_if_it_is_not_in_the_registry():
    settings = replace(LAB_SETTINGS, llm_model='someone/custom-7b')
    entries = models.catalogue(settings)
    assert 'someone/custom-7b' in [e['id'] for e in entries]
    assert entries[0]['label'].endswith('someone/custom-7b)')


def test_a_blank_role_falls_back_to_the_lab_default_model():
    settings = replace(LAB_SETTINGS, llm_model='lab/default')
    roles = models.resolve(LabConfig(), settings)
    assert roles.answer == 'lab/default' and roles.grade == 'lab/default'
    assert roles.ragas == 'lab/default' and roles.judge == 'lab/default'


def test_each_role_round_trips_from_the_panels_json():
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
    """'' rather than a guess: the provider already knows its default model, and
    a lab that hard-codes one here would silently ignore RAGLAB_MODEL."""
    provider = Recorder()
    pipeline.retrieve(index, RetrievalConfig(k=3, rerank_depth=3, reranker='llm'),
                      'قول باشگاه', '2026-07-28', llm=provider)
    assert provider.calls == ['']


def test_the_key_facts_judge_uses_the_judge_model():
    provider = Recorder(reply='1: yes')
    score = evaluate.judge_key_facts(provider, 'judge/model',
                                     {'key_facts': ['he went to the gym']}, 'رفتم')
    assert provider.calls == ['judge/model']
    assert score == pytest.approx(1.0)


def test_every_configuration_factor_has_an_explainer():
    """An unexplained knob is a knob nobody can make a real decision about."""
    assert explain.missing() == []


def test_the_explainers_cover_the_model_roles_too():
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


def test_every_embedder_says_which_languages_it_covers():
    """A hint per option, covering the whole registry: an embedder the panel
    offers without saying what it can read is a run nobody can interpret."""
    hints = {hint['kind']: hint for hint in embedding.embedder_hints()}
    assert set(hints) == set(EMBEDDERS)
    assert all(h['languages'] and h['label'] and h['note'] for h in hints.values())


def test_the_production_default_is_labelled_as_latin_only():
    """ascii-hash scores near chance on this corpus for one reason, and the
    dropdown has to say it out loud rather than leave it to be discovered by
    a run."""
    hints = {hint['kind']: hint for hint in embedding.embedder_hints()}
    assert hints['ascii-hash']['farsi'] is False
    assert 'latin' in hints['ascii-hash']['languages'].lower()
    for kind in ('token-hash', 'char-hash', 'fastembed'):
        assert hints[kind]['farsi'] is True, kind


def test_the_embedding_model_catalogue_offers_models_that_speak_farsi():
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


def _fastembed_serving(monkeypatch, ids):
    """Pretend fastembed is installed and serves exactly `ids`. Both halves
    have to be stubbed: availability is a separate import check from the
    served list, so patching only the list would assert on whether the
    `semantic` extra happens to be installed here."""
    monkeypatch.setattr(embedding, 'fastembed_available', lambda: True)
    monkeypatch.setattr(embedding, 'fastembed_models', lambda: frozenset(ids))


def test_an_english_only_model_is_offered_but_says_so(monkeypatch):
    """The brain hardwires bge-small-en today. The lab must be able to measure
    that choice, and must never let it be picked by accident."""
    _fastembed_serving(monkeypatch, embedding.MODEL_IDS)
    by_id = {e['id']: e for e in embedding.embed_model_catalogue(LAB_SETTINGS)}
    english = by_id['BAAI/bge-small-en-v1.5']
    assert english['farsi'] is False
    assert 'english' in english['languages'].lower()
    assert english['available'] is True      # installable, just wrong for Farsi


def test_a_model_this_fastembed_cannot_serve_reads_NA(monkeypatch):
    """NA now means one thing only: *this installation* cannot load it. An older
    fastembed serves a shorter list, and the panel has to say so rather than
    promise a wheel that is not there."""
    _fastembed_serving(
        monkeypatch,
        {'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'})
    entries = embedding.embed_model_catalogue(LAB_SETTINGS)
    by_id = {entry['id']: entry for entry in entries}
    assert by_id[
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'][
            'available'] is True
    assert by_id['BAAI/bge-small-en-v1.5']['available'] is False
    flags = [entry['available'] for entry in entries]
    assert flags == sorted(flags, reverse=True), 'usable models come first'


def test_fastembed_models_are_NA_until_the_semantic_extra_is_installed(monkeypatch):
    """With the extra missing, every fastembed model must read NA rather
    than promise a wheel that is not there — the import check alone
    decides, regardless of how generous the served list is."""
    monkeypatch.setattr(embedding, 'fastembed_models',
                        lambda: frozenset(embedding.MODEL_IDS))
    monkeypatch.setattr(embedding, 'fastembed_available', lambda: False)
    absent = {e['id']: e for e in embedding.embed_model_catalogue(LAB_SETTINGS)}
    assert absent['sentence-transformers/all-MiniLM-L6-v2']['available'] is False
    assert absent['BAAI/bge-small-en-v1.5']['available'] is False
    monkeypatch.setattr(embedding, 'fastembed_available', lambda: True)
    present = {e['id']: e for e in embedding.embed_model_catalogue(LAB_SETTINGS)}
    assert present['BAAI/bge-small-en-v1.5']['available'] is True


def test_e5_models_carry_the_prefixes_they_were_trained_with():
    """E5 was trained with "query: " / "passage: ". Dropping the prefixes is a
    silent quality loss, so they belong to the model entry, not to a caller."""
    by_id = {e['id']: e for e in embedding.embed_model_catalogue(LAB_SETTINGS)}
    e5 = by_id['intfloat/multilingual-e5-small']
    assert (e5['query_prefix'], e5['passage_prefix']) == ('query: ', 'passage: ')
    plain = by_id['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2']
    assert (plain['query_prefix'], plain['passage_prefix']) == ('', '')


def test_a_prefixed_embedder_marks_queries_and_passages_apart():
    fake = FakeTextEmbedding()
    embedder = embedding.FastEmbedMultilingual(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        query_prefix='query: ', passage_prefix='passage: ',
        factory=lambda name: fake)
    embedder.embed(['دعوا با پریا سر خونه'])
    embedder.embed_queries(['دعوا با پریا'])
    assert 'passage: دعوا با پریا سر خونه' in fake.seen
    assert 'query: دعوا با پریا' in fake.seen


def test_a_query_is_embedded_as_a_query_when_the_model_distinguishes_them():
    class Asymmetric:
        dim = 2
        name = 'asymmetric'

        def __init__(self):
            self.as_query: list[str] = []

        def embed(self, texts):
            return np.zeros((len(list(texts)), 2), dtype=np.float32)

        def embed_queries(self, texts):
            self.as_query.extend(texts)
            return np.ones((len(list(texts)), 2), dtype=np.float32)

    embedder = Asymmetric()
    vectors = embedding.query_vectors(embedder, ['سلام'])
    assert embedder.as_query == ['سلام']
    assert vectors.shape == (1, 2) and vectors.any()


def test_a_symmetric_embedder_needs_no_query_method():
    """Every hash embedder embeds both sides the same way, and must keep
    working without knowing this distinction exists."""
    vectors = embedding.query_vectors(embedding.make_embedder('char-hash'),
                                      ['سلام'])
    assert vectors.shape[0] == 1 and np.any(vectors)


def test_dense_retrieval_embeds_the_question_as_a_query(index, monkeypatch):
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
    """A model nobody loads must not invalidate an index and cost a
    rebuild."""
    hashed = IndexConfig(embedder='char-hash')
    assert hashed.fingerprint() == \
        replace(hashed, embed_model='BAAI/bge-small-en-v1.5').fingerprint()
    real = IndexConfig(embedder='fastembed')
    assert real.fingerprint() != \
        replace(real, embed_model='BAAI/bge-small-en-v1.5').fingerprint()


def test_the_chosen_embedding_model_is_the_one_that_gets_loaded(monkeypatch):
    seen: dict = {}

    def spy(model_name, **kwargs):
        seen.update({'model': model_name} | kwargs)
        return object()

    monkeypatch.setattr(embedding, 'SentenceTransformerEmbedder', spy)
    embedding.make_embedder('sentence-transformers', LAB_SETTINGS,
                            model='intfloat/multilingual-e5-small')
    assert seen['model'] == 'intfloat/multilingual-e5-small'
    assert seen['query_prefix'] == 'query: '
    assert seen['passage_prefix'] == 'passage: '


def test_a_blank_embedding_model_keeps_following_the_lab_default(monkeypatch):
    """'' means RAGLAB_FASTEMBED_MODEL, exactly as it means RAGLAB_MODEL for
    the chat roles — the lab never hard-codes a model of its own."""
    seen: dict = {}
    monkeypatch.setattr(embedding, 'FastEmbedMultilingual',
                        lambda model_name, **kw: seen.update(model=model_name))
    embedding.make_embedder('fastembed', LAB_SETTINGS)
    assert seen['model'] == LAB_SETTINGS.fastembed_model


def test_the_index_builds_with_the_embedding_model_from_its_config(monkeypatch,
                                                                  diary):
    from raglab import index as index_module
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
    note = embedding.language_note('fastembed', 'BAAI/bge-small-en-v1.5')
    assert 'bge-small-en' in note and 'english' in note.lower()
    assert 'ascii-hash' in embedding.language_note('ascii-hash', '')


def test_a_run_records_which_languages_its_embedder_can_represent(
        registry, ground_truth, tmp_path, monkeypatch):
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
    ids = {option.id for option in models.CHAT_MODELS}
    assert ids == set(REACHABLE_CHAT)
    assert not ids & set(UNREACHABLE_CHAT)
    # And the local list is deliberately untouched: llama3.1:8b is not installed,
    # reads NA, and stays, because pulling it is a one-line fix by the user.
    assert 'llama3.1:8b' in {option.id for option in models.OLLAMA_MODELS}


def test_the_embedding_catalogue_offers_only_models_that_loaded_here():
    assert {m.id: m.backend for m in embedding.EMBED_MODELS} == VERIFIED_EMBED


def test_the_lab_has_no_openai_embedding_backend():
    """A backend whose whole catalogue is gone is still selectable in
    principle, and would build an embedder with no model and dim 0."""
    assert 'openai' not in EMBEDDERS
    assert 'openai' not in embedding.BACKENDS
    assert 'openai' not in embedding.BACKEND_DEFAULTS
    assert not hasattr(embedding, 'OpenAIEmbedder')
    assert 'openai' not in {hint['kind'] for hint in embedding.embedder_hints()}
    with pytest.raises(ValueError):
        embedding.make_embedder('openai', LAB_SETTINGS)
    # The key goes too, rather than sitting in the settings advertising a backend
    # that is not there.
    assert not hasattr(LabSettings(), 'openai_api_key')


def test_e5_small_is_offered_through_the_backend_that_can_load_it():
    """The prefixes come along with the backend: they belong to the model,
    not to whichever backend happens to load it."""
    entry = {m.id: m for m in embedding.EMBED_MODELS}[
        'intfloat/multilingual-e5-small']
    assert entry.backend == 'sentence-transformers'
    assert (entry.query_prefix, entry.passage_prefix) == ('query: ', 'passage: ')
    assert LabConfig(index=IndexConfig(
        embedder='sentence-transformers',
        embed_model='intfloat/multilingual-e5-small')).validate() == []
    assert LabConfig(index=IndexConfig(
        embedder='fastembed',
        embed_model='intfloat/multilingual-e5-small')).validate()


# --- models fastembed cannot serve -----------------------------------------
#
# The Persian-tuned encoder is not in fastembed's list — it is a HuggingFace
# checkpoint — so each model names the backend that serves it and the lab grows a
# second one. Everything here runs offline: the local backend is exercised
# through an injected factory, because a test that needs a 2 GB download is a
# test nobody runs.

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


def test_the_catalogue_offers_every_requested_model_with_its_backend():
    by_id = {model.id: model for model in embedding.EMBED_MODELS}
    for model_id, (backend, dim, source) in REQUESTED_MODELS.items():
        entry = by_id.get(model_id)
        assert entry is not None, model_id
        assert (entry.backend, entry.dim, entry.source) == (backend, dim, source)
        assert entry.farsi and entry.note, model_id


def test_no_knob_offers_an_openrouter_embedding_or_rerank_model():
    """Availability is verified here, never guessed: OpenRouter's catalogue
    has not one embedding or rerank entry, though the gateway answers 401
    rather than 404 on those routes — a route with no servable model is not
    a backend. An unservable *reranker* is the worse half: `pipeline._rerank`
    swallows every exception and returns the pre-rerank order, so such a
    candidate would report itself as reranked while doing nothing."""
    assert 'openrouter' not in EMBEDDERS
    assert 'openrouter' not in embedding.BACKENDS
    assert 'openrouter' not in embedding.BACKEND_DEFAULTS
    assert 'rerank-4-fast' not in RERANKERS
    # And neither is reachable as a default, which is how they arrived.
    assert IndexConfig().embedder == 'sentence-transformers'
    assert RetrievalConfig().reranker == 'lexical'


def test_every_model_names_a_backend_the_lab_actually_has():
    assert all(model.backend in embedding.BACKENDS
               for model in embedding.EMBED_MODELS)
    assert set(embedding.BACKENDS) <= set(EMBEDDERS)


def test_the_persian_tuned_model_is_the_default():
    """A Farsi corpus deserves a Persian-tuned encoder, and it is the
    cheapest real encoder verified here."""
    assert IndexConfig().embedder == 'sentence-transformers'
    assert IndexConfig().embed_model == ''      # '' = the backend's default
    assert embedding.BACKEND_DEFAULTS['sentence-transformers'] == \
        'heydariAI/persian-embeddings'
    assert embedding.resolve_model('sentence-transformers', LAB_SETTINGS, '') == \
        'heydariAI/persian-embeddings'
    by_id = {m.id: m for m in embedding.EMBED_MODELS}
    # Visible in the option itself, not only behind the explainer: the standing is
    # what you are looking for while the dropdown is open.
    assert by_id['heydariAI/persian-embeddings'].tag == 'lab default'
    # RAGLAB_FASTEMBED_MODEL still drives the fastembed backend, untouched.
    assert embedding.resolve_model('fastembed', LAB_SETTINGS, '') == \
        LabSettings().fastembed_model


def test_the_persian_tuned_model_says_which_language_it_was_tuned_for():
    entry = {m.id: m for m in embedding.EMBED_MODELS}['heydariAI/persian-embeddings']
    assert 'persian' in entry.languages.lower() or 'farsi' in entry.languages.lower()


def test_the_local_backend_is_offered_as_an_embedder_with_its_coverage():
    assert 'sentence-transformers' in EMBEDDERS
    hints = {hint['kind']: hint for hint in embedding.embedder_hints()}
    assert set(hints) == set(EMBEDDERS)
    for kind in ('sentence-transformers', 'fastembed'):
        assert hints[kind]['farsi'] is True
        assert hints[kind]['languages'] and hints[kind]['note']


def test_a_local_model_is_offered_as_NA_until_its_library_is_installed(monkeypatch):
    monkeypatch.setattr(embedding, 'sentence_transformers_available', lambda: False)
    absent = {e['id']: e for e in embedding.embed_model_catalogue(LAB_SETTINGS)}
    assert absent['intfloat/multilingual-e5-small']['available'] is False
    assert absent['heydariAI/persian-embeddings']['available'] is False
    monkeypatch.setattr(embedding, 'sentence_transformers_available', lambda: True)
    present = {e['id']: e for e in embedding.embed_model_catalogue(LAB_SETTINGS)}
    assert present['intfloat/multilingual-e5-small']['available'] is True


def test_the_local_backend_applies_the_prefixes_the_model_was_trained_with():
    """The same guarantee the fastembed side keeps, on the backend that now
    serves the E5 model: query and passage are marked apart, and getting that
    backwards is a silent accuracy loss."""
    fake = FakeSentenceTransformer('intfloat/multilingual-e5-small')
    embedder = embedding.SentenceTransformerEmbedder(
        'intfloat/multilingual-e5-small', query_prefix='query: ',
        passage_prefix='passage: ', factory=lambda name: fake)
    fake.seen.clear()                      # drop anything the probe encoded
    passages = embedder.embed(['امروز جلسه داشتم'])
    queries = embedder.embed_queries(['جلسه کی بود؟'])
    assert fake.seen == ['passage: امروز جلسه داشتم', 'query: جلسه کی بود؟']
    assert embedder.dim == fake.dim == passages.shape[1] == queries.shape[1]
    assert 'intfloat/multilingual-e5-small' in embedder.name


def test_make_embedder_builds_the_local_backend(monkeypatch):
    monkeypatch.setattr(embedding, '_sentence_transformer',
                        lambda name: FakeSentenceTransformer(name))
    local = embedding.make_embedder('sentence-transformers', LAB_SETTINGS,
                                    'intfloat/multilingual-e5-small')
    assert 'intfloat/multilingual-e5-small' in local.name
    # Blank means "the default model for the backend you chose", the same rule as
    # '' meaning RAGLAB_FASTEMBED_MODEL for fastembed.
    default = embedding.make_embedder('sentence-transformers', LAB_SETTINGS, '')
    assert 'heydariAI/persian-embeddings' in default.name


def test_the_chosen_model_survives_the_fingerprint_for_every_model_backend():
    """The model is part of what got stored, so it has to reach the collection
    name — for all three backends, not just the first one the lab had."""
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


def test_a_model_from_the_wrong_backend_is_refused_before_the_run():
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
    """Two backends is a choice nobody can make from the kind names alone."""
    text = explain.topics()['index.embedder'].lower()
    assert 'sentence-transformers' in text and 'fastembed' in text
