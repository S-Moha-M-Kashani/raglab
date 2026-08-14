"""The in-memory vector store and the index/pipeline built on top of it —
integration tests running entirely in process memory."""
import socket

import numpy as np
import pytest

from raglab import corpus, metrics, pipeline, query, store
from raglab.config import GenerationConfig, IndexConfig, RetrievalConfig
from raglab.index import IndexRegistry

from conftest import LAB_SETTINGS, RAGLAB_DIR


# --- the ephemeral vector store --------------------------------------------
# An experiment's vectors, contexts and answers live for the process and no
# longer: the lab owns a store in memory instead of a Chroma database.

def test_memory_store_ranks_by_cosine_distance():
    """Chroma's contract, which `LabIndex.dense` reads: `distances`, not
    similarities, so the caller's `1 - d` keeps meaning what it meant."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['near', 'orthogonal', 'opposite'],
                   documents=['a', 'b', 'c'],
                   embeddings=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                   metadatas=[{'layer': 'chunk'}] * 3)
    res = vectors.query(query_embeddings=[[1.0, 0.0]], n_results=3)
    assert res['ids'][0] == ['near', 'orthogonal', 'opposite']
    assert res['distances'][0] == pytest.approx([0.0, 1.0, 2.0], abs=1e-6)
    assert res['documents'][0][0] == 'a'
    assert res['metadatas'][0][0] == {'layer': 'chunk'}


def test_memory_store_answers_several_query_vectors_at_once():
    """A store that silently answered only the first query vector would
    score multi-query expansion as doing nothing."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['x', 'y'], documents=['a', 'b'],
                   embeddings=[[1.0, 0.0], [0.0, 1.0]], metadatas=[{}, {}])
    res = vectors.query(query_embeddings=[[1.0, 0.0], [0.0, 1.0]], n_results=1)
    assert res['ids'] == [['x'], ['y']]


def test_memory_store_upsert_replaces_a_record_instead_of_duplicating_it():
    """Chunk ids are deterministic, so a rebuild writes the same ids again."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['a'], documents=['first'], embeddings=[[1.0, 0.0]],
                   metadatas=[{'layer': 'chunk'}])
    vectors.upsert(ids=['a'], documents=['second'], embeddings=[[0.0, 1.0]],
                   metadatas=[{'layer': 'session'}])
    assert vectors.count() == 1
    res = vectors.query(query_embeddings=[[0.0, 1.0]], n_results=1)
    assert res['documents'][0] == ['second']
    assert res['metadatas'][0] == [{'layer': 'session'}]


def test_memory_store_applies_the_where_clause_the_lab_actually_builds():
    """Asserted against `query.where_clause` itself, not a hand-written
    dict, since a hand-rolled store could quietly differ from Chroma here."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(
        ids=['in-scope', 'too-early', 'too-late'],
        documents=['keep', 'drop', 'drop'],
        embeddings=[[1.0, 0.0]] * 3,
        metadatas=[{'span_from': 20251201, 'span_to': 20251201},
                   {'span_from': 20250101, 'span_to': 20250101},
                   {'span_from': 20260301, 'span_to': 20260301}])
    where = query.where_clause(query.TimeScope(20251122, 20251221, 'آذر', 'jalali'))
    res = vectors.query(query_embeddings=[[1.0, 0.0]], n_results=3, where=where)
    assert res['ids'][0] == ['in-scope']


def test_memory_store_keeps_a_chunk_that_merely_overlaps_the_scope():
    """Containment would drop exactly the evidence a scoped question is
    reaching for."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['thread'], documents=['a year of it'],
                   embeddings=[[1.0, 0.0]],
                   metadatas=[{'span_from': 20250801, 'span_to': 20260720}])
    where = query.where_clause(query.TimeScope(20251122, 20251221, 'آذر', 'jalali'))
    res = vectors.query(query_embeddings=[[1.0, 0.0]], n_results=3, where=where)
    assert res['ids'][0] == ['thread']


def test_memory_store_does_not_match_a_metadata_key_a_record_lacks():
    """Chroma's semantics: a record missing the filtered key is excluded,
    never kept — the reason `Chunk.metadata()` carries `habit` on every
    chunk."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['has', 'lacks'], documents=['a', 'b'],
                   embeddings=[[1.0, 0.0], [1.0, 0.0]],
                   metadatas=[{'habit': 'gym'}, {}])
    res = vectors.query(query_embeddings=[[1.0, 0.0]], n_results=2,
                        where={'habit': 'gym'})
    assert res['ids'][0] == ['has']


def test_memory_store_never_returns_more_than_it_holds():
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['a'], documents=['one'], embeddings=[[1.0, 0.0]],
                   metadatas=[{}])
    assert vectors.query(query_embeddings=[[1.0, 0.0]], n_results=8)['ids'] == [['a']]
    empty = store.MemoryVectors('raglab-empty')
    assert empty.count() == 0
    assert empty.query(query_embeddings=[[1.0, 0.0]], n_results=5)['ids'] == [[]]


def test_memory_store_returns_stored_vectors_in_the_order_asked_for():
    """`LabIndex.vectors_for` reads vectors back for MMR rather than
    re-embedding, and zips the result against the ids it asked for."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['a', 'b'], documents=['x', 'y'],
                   embeddings=[[1.0, 0.0], [0.0, 1.0]], metadatas=[{}, {}])
    got = vectors.get(ids=['b', 'a'], include=['embeddings'])
    assert got['ids'] == ['b', 'a']
    assert np.allclose(got['embeddings'][0], [0.0, 1.0])
    assert np.allclose(got['embeddings'][1], [1.0, 0.0])


def test_memory_store_get_skips_an_id_it_does_not_hold():
    """A silent partial result, like Chroma's: the caller pairs ids with
    vectors by name, so a placeholder row would be a wrong vector."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['a'], documents=['x'], embeddings=[[1.0, 0.0]],
                   metadatas=[{}])
    assert vectors.get(ids=['a', 'missing'], include=['embeddings'])['ids'] == ['a']


def test_the_index_holds_its_vectors_in_process_memory(index):
    """Asserted on the type, since "there is no database" is not observable
    from a query that merely succeeds."""
    assert isinstance(index.store, store.MemoryVectors)
    assert index.store.count() == index.stats.chunks


# The lab must never grow a vector-database dependency.
def test_no_lab_module_imports_a_vector_database_client():
    """chromadb is production's dependency, not the lab's. One import line
    would bring the persistence back, and it would look harmless."""
    offenders = []
    for path in sorted(RAGLAB_DIR.glob('*.py')):
        for line in path.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if not stripped.startswith(('import ', 'from ')):
                continue
            if 'chromadb' in stripped or 'ChatStore' in stripped:
                offenders.append(f'{path.name}: {stripped}')
    assert offenders == []


def test_a_fresh_lab_process_rebuilds_its_index(diary):
    """Nothing outlives the registry: a second one over the same config
    builds again rather than finding a collection waiting for it."""
    cfg = IndexConfig(chunker='session', embedder='token-hash')
    first = IndexRegistry(LAB_SETTINGS, diary).get(cfg)
    second = IndexRegistry(LAB_SETTINGS, diary).get(cfg)
    assert second is not first and second.store is not first.store
    assert not first.stats.reused and not second.stats.reused
    assert second.stats.chunks == first.stats.chunks
    assert second.store.count() == second.stats.chunks


def test_building_an_index_opens_no_socket(diary, monkeypatch):
    """The offline embedders download nothing, so a connection here could
    only be a store trying to persist."""
    def refuse(*_args, **_kwargs):
        raise AssertionError('the lab opened a network connection while building')

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket.socket, 'connect_ex', refuse)
    built = IndexRegistry(LAB_SETTINGS, diary).get(
        IndexConfig(chunker='session', embedder='ascii-hash'))
    assert built.stats.chunks == len(diary['sessions'])


# --- index and pipeline (integration, in-process memory) -------------------

def test_index_is_reused_for_the_same_fingerprint(registry):
    """`reused` reports the only form left: this process built the index
    earlier and still has it."""
    cfg = IndexConfig(chunker='turn-pair', embedder='token-hash')
    first = registry.get(cfg)
    assert not first.stats.reused
    assert registry.get(cfg) is first
    assert first.stats.reused
    assert first.stats.collection == cfg.collection()


def test_different_configs_get_different_collections():
    a = IndexConfig(chunker='fixed').collection()
    b = IndexConfig(chunker='session').collection()
    assert a != b and a.startswith('raglab-')


def test_retrieval_finds_the_evidence_session_for_a_known_question(index, ground_truth):
    """End-to-end on the real corpus: a hybrid retrieval over semantic chunks
    must surface at least one cited evidence session for most single-hop
    questions. Asserted as a rate, not per question — a single hard question
    should not be able to fail the suite."""
    questions = [q for q in ground_truth['questions']
                 if q['type'] == 'single-hop'][:10]
    cfg = RetrievalConfig(retriever='hybrid-rrf', k=8, reranker='lexical')
    hits = 0
    for question in questions:
        outcome = pipeline.retrieve(index, cfg, question['question_fa'],
                                    question['query_date'])
        gold = corpus.evidence_sessions(question)
        hits += metrics.hit_at_k(outcome.sessions, gold, cfg.k)
    assert hits >= 4, f'only {hits}/10 single-hop questions found any evidence'


def test_time_filter_narrows_the_candidate_pool(index, ground_truth):
    scoped = 'آذر چه خبر بود؟'
    with_filter = pipeline.retrieve(index, RetrievalConfig(time_filter=True),
                                    scoped, '2026-07-28')
    without = pipeline.retrieve(index, RetrievalConfig(time_filter=False),
                                scoped, '2026-07-28')
    assert with_filter.time_scope is not None
    assert (with_filter.diagnostics['candidates_in_scope']
            < without.diagnostics['candidates_in_scope'])
    dates = [corpus.date_int(c.date) for c in with_filter.contexts]
    assert dates and all(20251122 <= d <= 20251221 for d in dates), dates


def test_grader_threshold_produces_an_abstention(index):
    """A question about something the diary never mentions must be refusable —
    and only the grader can refuse it."""
    nonsense = 'قرارداد خرید کشتی در بندر عباس چی شد؟'
    ungated = pipeline.retrieve(index, RetrievalConfig(grader='none'), nonsense,
                                '2026-07-28')
    gated = pipeline.retrieve(index, RetrievalConfig(grader='lexical',
                                                    grade_threshold=0.9),
                              nonsense, '2026-07-28')
    assert not ungated.abstained and ungated.contexts
    assert gated.abstained and not gated.contexts


def test_answerer_emits_the_refusal_when_abstaining(index):
    outcome = pipeline.retrieve(index, RetrievalConfig(grader='lexical',
                                                      grade_threshold=0.99),
                                'قرارداد کشتی', '2026-07-28')
    outcome = pipeline.answer(outcome, GenerationConfig(answerer='extractive'))
    assert outcome.answer == pipeline.REFUSAL
    assert outcome.abstained


def test_quoting_the_diarist_saying_i_dont_know_is_not_an_abstention():
    """The diarist writes «نمیدونم» constantly, and counting it as a refusal
    would score answerable questions as abstentions on a pipeline with no
    gate."""
    assert not pipeline.reads_as_refusal('نمیدونم چیکار کنم [2026-01-05-a]',
                                         'extractive')
    assert not pipeline.reads_as_refusal(
        'کارت رو عوض کردی. خودت گفتی نمیدونم درست بود یا نه.', 'llm')
    assert pipeline.reads_as_refusal(pipeline.REFUSAL, 'extractive')
    assert pipeline.reads_as_refusal('چیزی در این مورد ذکر نشده.', 'llm')


def test_ascii_hash_baseline_retrieves_worse_than_char_hash(registry, ground_truth):
    """The production embedder cannot represent this corpus, so it must lose
    to a Unicode-aware one."""
    questions = [q for q in ground_truth['questions']
                 if q['type'] == 'single-hop'][:8]
    cfg = RetrievalConfig(retriever='dense', k=8, reranker='none', time_filter=False)

    def rate(embedder_name):
        index = registry.get(IndexConfig(chunker='fixed', embedder=embedder_name,
                                         contextual=False))
        total = 0.0
        for question in questions:
            outcome = pipeline.retrieve(index, cfg, question['question_fa'],
                                        question['query_date'])
            total += metrics.hit_at_k(outcome.sessions,
                                      corpus.evidence_sessions(question), cfg.k)
        return total / len(questions)

    assert rate('char-hash') > rate('ascii-hash')
