"""Building and holding one indexed configuration.

A LabIndex pairs a vector store with a chunk table, both keyed by deterministic
chunk ids and held only in process memory — nothing here survives a restart.
"""
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from raglab.corpora.corpus_reading import date_label, ranks_label
from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.rag_components.indexing.chunking_strategies import (
    Chunk,
    chunk_document)
from raglab.configuration.lab_config import IndexConfig, LabSettings
from raglab.rag_components.indexing.in_memory_vector_store import MemoryVectors

BATCH = 200


@dataclass
class IndexStats:
    collection: str = ''
    chunks: int = 0             # every row in the index, summaries included
    leaves: int = 0             # the chunker's own output; equal to `chunks` on a flat index
    # Measured over the leaves only, so a summariser's rows don't skew the chunk-size knob.
    avg_chars: float = 0.0
    p95_chars: int = 0
    embed_dim: int = 0
    build_seconds: float = 0.0
    # None on a flat index — distinguishes "no hierarchy" from "a hierarchy that found nothing".
    hierarchy: dict | None = None
    # Set by IndexRegistry: this process already had it.
    reused: bool = False
    # '' means the corpus declares no such label (D5/D6): time filtering,
    # recency ranking and summary date ranges are inert without one, and the
    # agentic retriever's importance weight is inert without the other —
    # reported here rather than guessed, since a row must say why a knob did
    # nothing rather than leave it silent.
    date_label: str = ''
    ranks_label: str = ''
    notes: list = field(default_factory=list)


class LabIndex:
    def __init__(self, cfg: IndexConfig, embedder, store: MemoryVectors,
                 chunks: list[Chunk], stats: IndexStats):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.chunks = chunks
        self.stats = stats
        self.by_id = {c.id: c for c in chunks}
        self.by_document: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if chunk.document_id:
                self.by_document.setdefault(chunk.document_id, []).append(chunk)
        self._bm25 = None

    # --- building ---------------------------------------------------------

    @classmethod
    def build(cls, cfg: IndexConfig, corpus: dict, settings: LabSettings,
              progress=None) -> 'LabIndex':
        """Always a full build; skipping the work is the registry's decision, not this one's."""
        started = time.time()
        cfg = cfg.normalized()
        stats = IndexStats(collection=cfg.collection())
        note = stats.notes.append
        embedder = embedding.make_embedder(cfg.embedder, settings, cfg.embed_model)
        stats.embed_dim = getattr(embedder, 'dim', 0)
        store = MemoryVectors(cfg.collection())

        meta = corpus.get('corpus_dataset_metadata') or {}
        label_fields = meta.get('label_fields') or {}
        language = meta.get('language', '')
        stats.date_label = date_label(label_fields)
        stats.ranks_label = ranks_label(label_fields)
        documents = corpus.get('corpus_documents') or []
        if progress:
            progress('chunking', 0.1)
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(chunk_document(document, cfg, embedder, label_fields,
                                         language))

        lengths = np.array([len(c.text) for c in chunks]) if chunks else np.array([0])
        stats.chunks = stats.leaves = len(chunks)
        stats.avg_chars = round(float(lengths.mean()), 1)
        stats.p95_chars = int(np.percentile(lengths, 95))

        # Leaf vectors are kept, not discarded, since the grouping below reads
        # them and re-embedding to cluster would double the cost.
        leaf_vectors = cls._embed_into(store, chunks, embedder, note, progress,
                                       0.5, 0.4 if cfg.hierarchy else 0.5)

        if cfg.hierarchy:
            # Additive, always: leaves stay in the store, since replacing them
            # with summaries is what would lose information permanently.
            from raglab.rag_components.indexing import (
    summary_hierarchy_builder as hierarchy_mod)
            if progress:
                progress('grouping', 0.9)
            hierarchy_stats = hierarchy_mod.HierarchyStats()
            summaries = hierarchy_mod.build(chunks, leaf_vectors, cfg, embedder,
                                            hierarchy_stats, label_fields)
            stats.hierarchy = hierarchy_stats.as_dict()
            for message in hierarchy_stats.notes:
                note(message)
            if summaries:
                cls._embed_into(store, summaries, embedder, note, progress,
                                0.95, 0.05)
                chunks = chunks + summaries
                stats.chunks = len(chunks)
            else:
                note('the grouping produced no summary — this index is flat')

        stats.build_seconds = round(time.time() - started, 2)
        return cls(cfg, embedder, store, chunks, stats)

    @staticmethod
    def _embed_into(store, chunks, embedder, note, progress,
                    at: float, span: float) -> np.ndarray:
        """Embed and upsert one set of rows, returning their vectors.
        Shared by leaves and summaries so the two paths cannot drift apart."""
        out: list[np.ndarray] = []
        for start in range(0, len(chunks), BATCH):
            batch = chunks[start:start + BATCH]
            vectors = embedder.embed([c.text for c in batch])
            if not np.any(vectors):
                note('WARNING: this embedder produced all-zero vectors for '
                     'part of the corpus — it cannot represent this text')
            store.upsert(ids=[c.id for c in batch],
                         documents=[c.text for c in batch],
                         embeddings=list(vectors),
                         metadatas=[c.metadata() for c in batch])
            out.append(np.asarray(vectors, dtype=np.float32))
            if progress:
                done = (start + len(batch)) / max(1, len(chunks))
                progress('embedding', at + span * done)
        if not out:
            return np.zeros((0, getattr(embedder, 'dim', 1) or 1),
                            dtype=np.float32)
        return np.vstack(out)

    # --- retrieval primitives --------------------------------------------

    @property
    def bm25(self):
        if self._bm25 is None:
            from raglab.rag_components.retrieval.retrieve_fuse_rerank_grade import BM25
            self._bm25 = BM25([c.text for c in self.chunks])
        return self._bm25

    def dense(self, query_vectors: np.ndarray, k: int,
              where: dict | None = None) -> list[tuple[str, float]]:
        """Nearest chunks for one or more query vectors, merged by best score."""
        count = self.store.count()
        if not count:
            return []
        res = self.store.query(
            query_embeddings=np.atleast_2d(query_vectors),
            n_results=min(k, count), where=where or None)
        best: dict[str, float] = {}
        for ids, distances in zip(res['ids'], res['distances']):
            for chunk_id, distance in zip(ids, distances):
                score = 1.0 - float(distance)
                if score > best.get(chunk_id, -2.0):
                    best[chunk_id] = score
        return sorted(best.items(), key=lambda kv: -kv[1])

    def vectors_for(self, chunk_ids: list[str]) -> np.ndarray:
        """Stored vectors, read back rather than re-embedded, so a slow embedder is not re-run per query."""
        if not chunk_ids:
            return np.zeros((0, 1), dtype=np.float32)
        got = self.store.get(ids=chunk_ids, include=['embeddings'])
        order = {cid: i for i, cid in enumerate(got['ids'])}
        stacked = np.array(got['embeddings'], dtype=np.float32)
        return np.array([stacked[order[cid]] for cid in chunk_ids
                         if cid in order], dtype=np.float32)

    def neighbors(self, chunk: Chunk) -> list[Chunk]:
        """Chunks either side of this one inside the same document."""
        siblings = list(self.by_document.get(chunk.document_id, []))
        if chunk not in siblings:
            return []
        i = siblings.index(chunk)
        return [c for c in siblings[max(0, i - 1):i + 2] if c is not chunk]

    def drop(self) -> None:
        self.store.drop()


EVICTED_NOTE = ('a previous build of this fingerprint was evicted to stay '
                'under the index ceiling, so this one is a rebuild rather '
                'than reuse')


class IndexRegistry:
    """Bounded process-lifetime cache of built indexes, keyed by fingerprint;
    every index it built dies with the process, and it lets go of the least
    recently used one — never one a caller is holding — past
    `LabSettings.max_indexes` (0 = unbounded)."""

    def __init__(self, settings: LabSettings, corpus: dict | None = None):
        self.settings = settings
        # A caller that already holds the corpus (the suite, a booted service)
        # passes it; anything else is loaded per dataset below.
        self.corpus = corpus
        # Ordered least-recently-served first: `get` moves what it serves to
        # the end, and eviction reads from the front.
        self._indexes: OrderedDict[str, LabIndex] = OrderedDict()
        # Fingerprints this registry dropped -> the dataset each indexed, so
        # the rebuild that follows can say why it is not reuse, and a
        # re-import of that dataset can withdraw the claim. Emptied as each is
        # noted.
        self._evicted: dict[str, str] = {}
        # Fingerprint -> how many callers are inside `hold`. An entry counted
        # here is skipped by eviction.
        self._held: dict[str, int] = {}
        # `_guard` protects the three tables above and nothing else — it is
        # never held while a build runs. `_builds` hands out one lock per
        # fingerprint, which *is* held across the build, so two threads on one
        # cold fingerprint produce one build. Same one-way order as
        # `panel_server.dataset_lock`: take the guard, get the lock, release
        # the guard, then take the lock. A lock here outlives its index on
        # purpose — one is a few dozen bytes against an index's tens of
        # megabytes, and a table of them is not the leak this class bounds.
        self._guard = threading.Lock()
        self._builds: dict[str, threading.Lock] = {}

    def corpus_for(self, dataset: str = '') -> dict:
        """The corpus one config indexes. Resolved per call, since dataset is
        a field of `IndexConfig` and the registry holds several corpora at once."""
        if not dataset:
            if self.corpus is not None:
                return self.corpus
            dataset = ''
        from raglab.corpora import dataset_import_contract as datasets
        return datasets.load(dataset)[0]

    def get(self, cfg: IndexConfig, progress=None, force: bool = False) -> LabIndex:
        """Serve one fingerprint, building it if this process has not got it.

        `stats.reused` is a field on the shared index rather than on this
        request, so it only means what it says while one job runs at a time —
        which is what the panel enforces today (a second concurrent job is
        refused) and what the Inspector and the sweep do by construction.
        Making it per-request belongs with concurrent jobs, not here.
        """
        key = cfg.normalized().fingerprint()
        with self._guard:
            build_lock = self._builds.setdefault(key, threading.Lock())
        with build_lock:
            with self._guard:
                index = None if force else self._indexes.get(key)
                if index is not None:
                    self._indexes.move_to_end(key)
                    # The one form of reuse that still exists: this process
                    # already built it.
                    index.stats.reused = True
                    # The eviction note belongs to the build it explained, and
                    # the stats travel with the index rather than with this
                    # request — so reuse of that same object would otherwise
                    # carry a line calling itself a rebuild, onto the row and
                    # into the ledger. Withdraw it here.
                    index.stats.notes = [note for note in index.stats.notes
                                         if note != EVICTED_NOTE]
                    return index
            # Outside every registry lock: a build is the most expensive thing
            # this process does, and a request for another fingerprint must not
            # queue behind it.
            built = LabIndex.build(cfg, self.corpus_for(cfg.dataset),
                                   self.settings, progress=progress)
            with self._guard:
                if key in self._evicted:
                    # `reused` is already False on a fresh build; the note is
                    # what tells a reader why this build time is an outlier.
                    built.stats.notes.append(EVICTED_NOTE)
                    self._evicted.pop(key, None)
                self._indexes[key] = built
                self._indexes.move_to_end(key)
                self._evict_to_ceiling(keep=key)
            return built

    @contextmanager
    def hold(self, cfg: IndexConfig):
        """Pin one fingerprint for the duration of a caller's work.

        Recency is not safety: a job answering questions against an index holds
        a reference the registry cannot see, so dropping the entry would free
        nothing and make the next request rebuild something already resident.
        Pinning is deliberately separate from building — a caller states what it
        is about to work on, then builds inside the pin, which is why every job
        route can say it on the `with` line it already has.
        """
        key = cfg.normalized().fingerprint()
        with self._guard:
            self._held[key] = self._held.get(key, 0) + 1
        try:
            yield
        finally:
            with self._guard:
                remaining = self._held.get(key, 0) - 1
                if remaining > 0:
                    self._held[key] = remaining
                else:
                    self._held.pop(key, None)

    def _evict_to_ceiling(self, keep: str = '') -> None:
        """Drop least-recently-served unheld indexes until the ceiling is met.
        Called under `_guard`, on insert, where the cost of a build is already
        being paid. A ceiling of 0 is unbounded; a registry whose every entry is
        held stays over the ceiling rather than taking an index away from work.

        `keep` is the fingerprint just inserted. It is never a candidate: when
        every older entry is held it would be the only one, and throwing away
        the build the caller is still waiting for — then telling the next
        request it was evicted — is the one outcome worse than staying over
        the ceiling.
        """
        ceiling = self.settings.max_indexes
        while ceiling and len(self._indexes) > ceiling:
            droppable = next((key for key in self._indexes
                              if key not in self._held and key != keep), None)
            if droppable is None:
                return
            index = self._indexes.pop(droppable)
            self._evicted[droppable] = index.cfg.dataset

    def invalidate_dataset(self, dataset: str) -> int:
        """Forget cached indexes for a replaced dataset id.

        Build fingerprints intentionally identify a dataset by id, not by its
        machine-local file. Re-importing that id is therefore the one boundary
        that must evict matching process-memory indexes explicitly. Replacing
        the mapping rather than dropping each index leaves any already-running
        job's private reference intact while every later request rebuilds.

        Not an eviction in the ceiling's sense, so the rebuild that follows
        carries no eviction note: the corpus changed, which is a different
        reason for a rebuild and not one the note would describe honestly.
        That holds for a fingerprint the ceiling *had* evicted earlier too —
        its pending note is dropped here rather than being handed to a rebuild
        the re-import is the real reason for. Dropping those entries is also
        what keeps this table from growing for the life of the process.
        """
        with self._guard:
            stale = [key for key, index in self._indexes.items()
                     if index.cfg.dataset == dataset]
            if stale:
                self._indexes = OrderedDict(
                    (key, index) for key, index in self._indexes.items()
                    if key not in stale)
            self._evicted = {key: owner for key, owner in self._evicted.items()
                             if owner != dataset}
        return len(stale)

    def known(self) -> list[dict]:
        # Under the guard: a panel asking what is resident while a job's build
        # inserts or evicts must read a whole answer, not a mutating mapping.
        with self._guard:
            return [{'fingerprint': key, 'collection': ix.stats.collection,
                     'chunks': ix.stats.chunks,
                     'config': dict(ix.cfg.__dict__)}
                    for key, ix in self._indexes.items()]
