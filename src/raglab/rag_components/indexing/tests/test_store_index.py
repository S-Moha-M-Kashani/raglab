"""The in-memory vector store (unit tests, direct calls) and the index
registry / pipeline built on top of it (integration tests: an index build or
a full retrieval crosses modules)."""
import socket
import threading
from dataclasses import replace

import numpy as np
import pytest

from raglab.corpora import corpus_reading as corpus
from raglab.corpora import dataset_import_contract as datasets
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.rag_components.retrieval import query_understanding as query
from raglab.rag_components.indexing import in_memory_vector_store as store
from raglab.configuration.lab_config import (
    GenerationConfig,
    IndexConfig,
    RetrievalConfig)
from raglab.rag_components.indexing import index_builder_registry as registry_mod
from raglab.rag_components.indexing.index_builder_registry import IndexRegistry

from raglab.conftest import LAB_SETTINGS, SMOKE_INDEX


# --- the ephemeral vector store --------------------------------------------
# An experiment's vectors, contexts and answers live for the process and no
# longer: the lab owns a store in memory instead of a Chroma database.

def test_memory_store_ranks_by_cosine_distance():
    # this is a unit test
    """Chroma's contract, which `LabIndex.dense` reads: `distances`, not
    similarities, so the caller's `1 - d` keeps meaning what it meant. Asked
    for more rows than it holds (8 against 3), it still returns only the 3 —
    a store may never invent rows, and an empty one must answer empty
    rather than raising."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['near', 'orthogonal', 'opposite'],
                   documents=['a', 'b', 'c'],
                   embeddings=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                   metadatas=[{'layer': 'chunk'}] * 3)
    res = vectors.query(query_embeddings=[[1.0, 0.0]], n_results=8)
    assert res['ids'][0] == ['near', 'orthogonal', 'opposite']
    assert res['distances'][0] == pytest.approx([0.0, 1.0, 2.0], abs=1e-6)
    assert res['documents'][0][0] == 'a'
    assert res['metadatas'][0][0] == {'layer': 'chunk'}

    empty = store.MemoryVectors('raglab-empty')
    assert empty.count() == 0
    assert empty.query(query_embeddings=[[1.0, 0.0]], n_results=5)['ids'] == [[]]


def test_memory_store_answers_several_query_vectors_at_once():
    # this is a unit test
    """A store that silently answered only the first query vector would
    score multi-query expansion as doing nothing."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['x', 'y'], documents=['a', 'b'],
                   embeddings=[[1.0, 0.0], [0.0, 1.0]], metadatas=[{}, {}])
    res = vectors.query(query_embeddings=[[1.0, 0.0], [0.0, 1.0]], n_results=1)
    assert res['ids'] == [['x'], ['y']]


def test_memory_store_upsert_replaces_a_record_instead_of_duplicating_it():
    # this is a unit test
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
    # this is a unit test
    """Asserted against `query.where_clause` itself, not a hand-written
    dict, since a hand-rolled store could quietly differ from Chroma here.
    Two cases share one clause: a chunk entirely outside the window is
    dropped, one merely straddling its edge is kept — the overlap semantics
    of the clause itself are pinned in test_primitives.py
    (`test_where_clause_overlaps_rather_than_contains`); this only checks
    that the store applies whatever clause it is handed."""
    where = query.where_clause(query.TimeScope(20251122, 20251221, 'آذر', 'jalali'))
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(
        ids=['in-scope', 'too-early', 'too-late', 'straddling'],
        documents=['keep', 'drop', 'drop', 'keep too'],
        embeddings=[[1.0, 0.0]] * 4,
        metadatas=[{'span_from': 20251201, 'span_to': 20251201},
                   {'span_from': 20250101, 'span_to': 20250101},
                   {'span_from': 20260301, 'span_to': 20260301},
                   {'span_from': 20250801, 'span_to': 20260720}])
    res = vectors.query(query_embeddings=[[1.0, 0.0]], n_results=4, where=where)
    assert res['ids'][0] == ['in-scope', 'straddling']


def test_memory_store_does_not_match_a_metadata_key_a_record_lacks():
    # this is a unit test
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


def test_memory_store_get_returns_requested_ids_in_order_and_skips_missing_ones():
    # this is a unit test
    """`LabIndex.vectors_for` reads vectors back for MMR rather than
    re-embedding, and zips the result against the ids it asked for — so
    order must be preserved. And a silent partial result, like Chroma's:
    the caller pairs ids with vectors by name, so a placeholder row for an
    id never stored would be a wrong vector."""
    vectors = store.MemoryVectors('raglab-test')
    vectors.upsert(ids=['a', 'b'], documents=['x', 'y'],
                   embeddings=[[1.0, 0.0], [0.0, 1.0]], metadatas=[{}, {}])
    got = vectors.get(ids=['b', 'a'], include=['embeddings'])
    assert got['ids'] == ['b', 'a']
    assert np.allclose(got['embeddings'][0], [0.0, 1.0])
    assert np.allclose(got['embeddings'][1], [1.0, 0.0])

    partial = vectors.get(ids=['a', 'missing'], include=['embeddings'])
    assert partial['ids'] == ['a']


def test_building_an_index_opens_no_socket(monkeypatch):
    # this is an integration test
    """The offline embedders download nothing, so a connection here could
    only be a store trying to persist. Built over the smoke corpus — cheap,
    5 sessions, no model download — rather than the 167-session diary this
    guard used to build for the same claim."""
    def refuse(*_args, **_kwargs):
        raise AssertionError('the lab opened a network connection while building')

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket.socket, 'connect_ex', refuse)
    built = IndexRegistry(LAB_SETTINGS).get(IndexConfig(**SMOKE_INDEX))
    assert built.stats.chunks == 5


# --- index registry and pipeline (integration, in-process memory) ----------

def test_registry_reuses_within_a_process_and_rebuilds_across_one(diary):
    # this is an integration test
    """Three claims in one build+get sequence rather than three tests: a
    second `get` on the same registry returns the same object and reports
    `reused`; a fresh registry never inherits another's index and rebuilds
    from scratch; and the store always holds exactly as many rows as
    `stats.chunks` claims, on both the first build and the second."""
    cfg = IndexConfig(split_plan=({'kind': 'document'}, {'kind': 'label', 'atoms': ({'label': 'role', 'value': 'user'},)}), embedder='token-hash')
    reg = IndexRegistry(LAB_SETTINGS, diary)

    first = reg.get(cfg)
    assert not first.stats.reused
    assert isinstance(first.store, store.MemoryVectors)
    assert first.store.count() == first.stats.chunks

    same = reg.get(cfg)
    assert same is first and same.stats.reused
    assert same.stats.collection == cfg.collection()

    fresh = IndexRegistry(LAB_SETTINGS, diary).get(cfg)
    assert fresh is not first and fresh.store is not first.store
    assert not fresh.stats.reused
    assert fresh.stats.chunks == first.stats.chunks
    assert fresh.store.count() == fresh.stats.chunks


def _smoke_cfg(chunk_chars: int) -> IndexConfig:
    """One cheap distinct index. `session` ignores `chunk_chars`, but the
    fingerprint does not, so a number is all it takes to name another build of
    the same five sessions — the cheapest way to fill a cache with real
    indexes."""
    return IndexConfig(**SMOKE_INDEX, chunk_chars=chunk_chars)


def test_the_registry_drops_the_least_recently_used_index_past_its_ceiling():
    # this is an integration test
    """The cache is a cache, not a table that only grows: past the ceiling the
    least recently used index is dropped, and *used* means served, not built —
    touching the oldest entry moves it out of the firing line. A rebuild after
    an eviction is not reuse and says why on its own row, while a first build
    of a fingerprint nobody evicted carries no such note. Zero is the reader
    who wants the old unbounded behaviour and says so."""
    reg = IndexRegistry(replace(LAB_SETTINGS, max_indexes=3))
    first, second, third = (_smoke_cfg(n) for n in (100, 200, 300))
    for cfg in (first, second, third):
        reg.get(cfg)
    # Serving `first` again makes `second` the least recently used.
    assert reg.get(first).stats.reused
    reg.get(_smoke_cfg(400))
    resident = {row['fingerprint'] for row in reg.known()}
    assert second.fingerprint() not in resident
    assert first.fingerprint() in resident and third.fingerprint() in resident

    rebuilt = reg.get(second)
    assert not rebuilt.stats.reused
    assert any('evicted' in note for note in rebuilt.stats.notes), rebuilt.stats.notes
    assert not any('evicted' in note for note in reg.get(second).stats.notes)
    fresh = IndexRegistry(LAB_SETTINGS).get(second)
    assert not any('evicted' in note for note in fresh.stats.notes)

    unbounded = IndexRegistry(replace(LAB_SETTINGS, max_indexes=0))
    for n in (100, 200, 300, 400):
        unbounded.get(_smoke_cfg(n))
    assert len(unbounded.known()) == 4


def test_a_held_index_outlives_more_builds_than_the_ceiling_allows():
    # this is an integration test
    """Recency is not safety: a job answering questions against an index holds
    a reference the registry cannot see, so evicting the entry would free
    nothing and make the next request rebuild something already resident. A
    held fingerprint is therefore skipped by eviction — and becomes an
    ordinary candidate again the moment its holder is done. When every older
    entry is held, the index just built is the only unheld one, and the
    registry stays over the ceiling rather than throwing away the build its
    caller is still waiting for."""
    reg = IndexRegistry(replace(LAB_SETTINGS, max_indexes=2))
    reading = _smoke_cfg(100)
    held = reg.get(reading)
    with reg.hold(reading):
        for n in (200, 300, 400, 500, 600):
            reg.get(_smoke_cfg(n))
        served = reg.get(reading)
        assert served is held and served.stats.reused
    for n in (700, 800):
        reg.get(_smoke_cfg(n))
    assert reading.fingerprint() not in {row['fingerprint'] for row in reg.known()}

    one, two, fresh = (_smoke_cfg(n) for n in (900, 1000, 1100))
    reg.get(one)
    reg.get(two)
    with reg.hold(one), reg.hold(two):
        reg.get(fresh)
    assert fresh.fingerprint() in {row['fingerprint'] for row in reg.known()}


def test_one_build_per_cold_fingerprint_and_no_registry_wide_wait(monkeypatch):
    # this is an integration test
    """An index build is the most expensive thing this process does, so two
    threads arriving on one cold fingerprint must produce one build and one
    object — and the guard that arranges that must not be a registry-wide lock
    held across the build, or every other fingerprint would queue behind it."""
    real_build = registry_mod.LabIndex.build
    built: list[int] = []
    inside = threading.Event()
    release = threading.Event()

    def build(cfg, corpus, settings, progress=None):
        built.append(cfg.chunk_chars)
        if cfg.chunk_chars == 100:
            inside.set()
            assert release.wait(timeout=5)
        return real_build(cfg, corpus, settings, progress=progress)

    monkeypatch.setattr(registry_mod.LabIndex, 'build', build)
    reg = IndexRegistry(LAB_SETTINGS)
    got: dict[str, object] = {}

    def ask(name: str) -> None:
        got[name] = reg.get(_smoke_cfg(100))

    threads = [threading.Thread(target=ask, args=(name,))
               for name in ('first', 'second')]
    for thread in threads:
        thread.start()
    assert inside.wait(timeout=5)
    # The build above is still in flight; an unrelated fingerprint sails past it.
    assert reg.get(_smoke_cfg(200)).stats.chunks == 5
    release.set()
    for thread in threads:
        thread.join(timeout=10)
    assert got['first'] is got['second']
    assert built == [100, 200], 'the cold fingerprint was built exactly once'


def test_different_configs_get_different_collections():
    # this is a unit test
    a = IndexConfig(split_plan=({'kind': 'document'}, {'kind': 'part'})).collection()
    b = IndexConfig(split_plan=({'kind': 'document'},)).collection()
    assert a != b and a.startswith('raglab-')


def test_retrieval_finds_the_evidence_document_for_a_known_question(smoke_index):
    # this is an integration test
    """Replaces a statistical rate over the real corpus (>=4/10 single-hop
    questions found *any* evidence, one hard question away from flaking)
    with one deterministic pin on the 5-document smoke set: `token-hash` has
    no RNG, so a known question retrieves its single known evidence document
    at rank 1, every run, on every process. Document 4 rather than document
    1 on purpose: a broken retriever that degenerates to insertion order
    would still pass a check against the first document, and document 4 is
    exactly what catches that."""
    _, truth = datasets.load('smoke-mini')
    question = next(q for q in truth['groundtruth_dataset']
                    if q['groundtruth_question_id'] == 4)
    assert corpus.evidence_documents(question) == [4]
    query_date = truth['groundtruth_dataset_metadata'][
        'default_question_asked_at'][:10]

    outcome = pipeline.retrieve(smoke_index.index,
                                RetrievalConfig(retriever='hybrid-rrf', k=3,
                                                reranker='lexical'),
                                question['question'], query_date)
    assert outcome.sessions[0] == '4'


def test_time_filter_narrows_the_candidate_pool(index, ground_truth):
    # this is an integration test
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


def test_grader_threshold_produces_the_refusal_string(index):
    # this is an integration test
    """A question about something the diary never mentions must be
    refusable, and only the grader can refuse it: the same nonsense
    question retrieves contexts ungated but abstains once a strict lexical
    threshold is applied. That gated outcome, fed straight into `answer()`,
    must produce the refusal string rather than an invented one — the
    abstention trio's three claims in a single retrieve-then-answer."""
    nonsense = 'قرارداد خرید کشتی در بندر عباس چی شد؟'
    ungated = pipeline.retrieve(index, RetrievalConfig(grader='none'), nonsense,
                                '2026-07-28')
    assert not ungated.abstained and ungated.contexts

    gated = pipeline.retrieve(index, RetrievalConfig(grader='lexical',
                                                    grade_threshold=0.9),
                              nonsense, '2026-07-28')
    assert gated.abstained and not gated.contexts

    outcome = pipeline.answer(gated, GenerationConfig(answerer='extractive'))
    assert outcome.answer == pipeline.REFUSAL
    assert outcome.abstained


def test_quoting_the_diarist_saying_i_dont_know_is_not_an_abstention():
    # this is a unit test
    """The diarist writes «نمیدونم» constantly, and counting it as a refusal
    would score answerable questions as abstentions on a pipeline with no
    gate."""
    assert not pipeline.reads_as_refusal('نمیدونم چیکار کنم [2026-01-05-a]',
                                         'extractive')
    assert not pipeline.reads_as_refusal(
        'کارت رو عوض کردی. خودت گفتی نمیدونم درست بود یا نه.', 'llm')
    assert pipeline.reads_as_refusal(pipeline.REFUSAL, 'extractive')
    assert pipeline.reads_as_refusal('چیزی در این مورد ذکر نشده.', 'llm')
