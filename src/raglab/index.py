"""Building and holding one indexed configuration.

A LabIndex pairs a vector store with a chunk table, both keyed by deterministic
chunk ids and held only in process memory — nothing here survives a restart.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from . import embedding
from .chunking import Chunk, chunk_session
from .config import IndexConfig, LabSettings
from .store import MemoryVectors

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
        self.by_session: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if chunk.session_id:
                self.by_session.setdefault(chunk.session_id, []).append(chunk)
        self._bm25 = None

    # --- building ---------------------------------------------------------

    @classmethod
    def build(cls, cfg: IndexConfig, diary: dict, settings: LabSettings,
              progress=None) -> 'LabIndex':
        """Always a full build; skipping the work is the registry's decision, not this one's."""
        started = time.time()
        cfg = cfg.normalized()
        stats = IndexStats(collection=cfg.collection())
        note = stats.notes.append
        embedder = embedding.make_embedder(cfg.embedder, settings, cfg.embed_model)
        stats.embed_dim = getattr(embedder, 'dim', 0)
        store = MemoryVectors(cfg.collection())

        sessions = diary['sessions']
        if progress:
            progress('chunking', 0.1)
        chunks: list[Chunk] = []
        for session in sessions:
            chunks.extend(chunk_session(session, cfg, embedder))

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
            from . import hierarchy as hierarchy_mod
            if progress:
                progress('grouping', 0.9)
            hierarchy_stats = hierarchy_mod.HierarchyStats()
            summaries = hierarchy_mod.build(chunks, leaf_vectors, cfg, embedder,
                                            hierarchy_stats)
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
            from .retrieval import BM25
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
        """Chunks either side of this one inside the same session."""
        siblings = list(self.by_session.get(chunk.session_id, []))
        if chunk not in siblings:
            return []
        i = siblings.index(chunk)
        return [c for c in siblings[max(0, i - 1):i + 2] if c is not chunk]

    def drop(self) -> None:
        self.store.drop()


class IndexRegistry:
    """Process-lifetime cache of built indexes, keyed by fingerprint; every
    index it built dies with the process."""

    def __init__(self, settings: LabSettings, diary: dict | None = None):
        self.settings = settings
        # A caller that already holds the corpus (the suite, a booted service)
        # passes it; anything else is loaded per dataset below.
        self.diary = diary
        self._indexes: dict[str, LabIndex] = {}

    def corpus_for(self, dataset: str = '') -> dict:
        """The sessions one config indexes. Resolved per call, since dataset is
        a field of `IndexConfig` and the registry holds several corpora at once."""
        if not dataset:
            if self.diary is not None:
                return self.diary
            dataset = ''
        from . import datasets
        return datasets.load(dataset)[0]

    def get(self, cfg: IndexConfig, progress=None, force: bool = False) -> LabIndex:
        key = cfg.normalized().fingerprint()
        if force or key not in self._indexes:
            self._indexes[key] = LabIndex.build(cfg, self.corpus_for(cfg.dataset),
                                                self.settings, progress=progress,
                                                )
        else:
            # The one form of reuse that still exists: this process already built it.
            self._indexes[key].stats.reused = True
        return self._indexes[key]

    def known(self) -> list[dict]:
        return [{'fingerprint': key, 'collection': ix.stats.collection,
                 'chunks': ix.stats.chunks,
                 'config': dict(ix.cfg.__dict__)}
                for key, ix in self._indexes.items()]
