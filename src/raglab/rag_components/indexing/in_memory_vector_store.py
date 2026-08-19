"""The lab's vector store: process memory only, discarded when the process ends.
Return shapes mimic Chroma's (nested rows, cosine *distance*) so `LabIndex.dense`'s `1 - d` keeps meaning what it meant.
Filters use Chroma's operator dicts; an absent key never matches, and an unknown operator raises rather than being ignored.
"""
import numpy as np


def _compare(value, wanted, test) -> bool:
    """Order comparisons on metadata of unknown type: a type mismatch is a filter that cannot hold, not a crash."""
    try:
        return bool(test(value, wanted))
    except TypeError:
        return False


OPERATORS = {
    '$eq': lambda value, wanted: value == wanted,
    '$ne': lambda value, wanted: value != wanted,
    '$gt': lambda value, wanted: _compare(value, wanted, lambda a, b: a > b),
    '$gte': lambda value, wanted: _compare(value, wanted, lambda a, b: a >= b),
    '$lt': lambda value, wanted: _compare(value, wanted, lambda a, b: a < b),
    '$lte': lambda value, wanted: _compare(value, wanted, lambda a, b: a <= b),
    '$in': lambda value, wanted: value in wanted,
    '$nin': lambda value, wanted: value not in wanted,
}


def matches(metadata: dict, where: dict | None) -> bool:
    """Whether one record satisfies a Chroma-style filter; no clause means every record."""
    if not where:
        return True
    for key, condition in where.items():
        if key == '$and':
            if not all(matches(metadata, clause) for clause in condition):
                return False
        elif key == '$or':
            if not any(matches(metadata, clause) for clause in condition):
                return False
        elif not _holds(metadata, key, condition):
            return False
    return True


def _holds(metadata: dict, key: str, condition) -> bool:
    if key not in metadata:
        return False
    value = metadata[key]
    if not isinstance(condition, dict):
        return value == condition
    for operator, wanted in condition.items():
        test = OPERATORS.get(operator)
        if test is None:
            raise ValueError(
                f'unsupported filter operator {operator!r}; the lab store '
                f'implements ' + ', '.join(sorted(OPERATORS)))
        if not test(value, wanted):
            return False
    return True


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Row-normalised; a zero row stays zero rather than becoming NaN, since
    `ascii-hash` embeds Farsi to the zero vector and that must score badly, not crash."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class MemoryVectors:
    """One named, in-process index; `name` comes from `IndexConfig.collection()`."""

    def __init__(self, name: str = ''):
        self.name = name
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self._rows: list[np.ndarray] = []
        self._at: dict[str, int] = {}
        self._matrix: np.ndarray | None = None   # built lazily, per query batch

    def count(self) -> int:
        return len(self.ids)

    def upsert(self, ids: list[str], documents: list[str], embeddings,
               metadatas: list[dict] | None = None) -> None:
        """Add or replace by id: deterministic chunk ids mean a rebuild must overwrite rather than duplicate."""
        metadatas = metadatas if metadatas is not None else [{}] * len(ids)
        for chunk_id, document, vector, metadata in zip(ids, documents,
                                                        embeddings, metadatas):
            row = np.asarray(vector, dtype=np.float32).reshape(-1)
            at = self._at.get(chunk_id)
            if at is None:
                self._at[chunk_id] = len(self.ids)
                self.ids.append(chunk_id)
                self.documents.append(document)
                self.metadatas.append(dict(metadata or {}))
                self._rows.append(row)
            else:
                self.documents[at] = document
                self.metadatas[at] = dict(metadata or {})
                self._rows[at] = row
        self._matrix = None

    def query(self, query_embeddings, n_results: int = 8,
              where: dict | None = None) -> dict:
        """Nearest records per query vector; one output row per vector, so multi-query expansion can merge results."""
        queries = _unit(np.atleast_2d(np.asarray(query_embeddings,
                                                 dtype=np.float32)))
        kept = [i for i, metadata in enumerate(self.metadatas)
                if matches(metadata, where)]
        out: dict[str, list] = {'ids': [], 'documents': [], 'metadatas': [],
                                'distances': [], 'embeddings': []}
        matrix = self._unit_matrix()[kept] if kept else None
        take = max(0, min(n_results, len(kept)))
        for vector in queries:
            if matrix is None or not take:
                for key in out:
                    out[key].append([])
                continue
            similarities = matrix @ vector
            # Stable, so the zero-vector baseline (everything scores identically) still returns the same order every run.
            order = np.argsort(-similarities, kind='stable')[:take]
            rows = [kept[int(i)] for i in order]
            out['ids'].append([self.ids[r] for r in rows])
            out['documents'].append([self.documents[r] for r in rows])
            out['metadatas'].append([self.metadatas[r] for r in rows])
            out['distances'].append([1.0 - float(similarities[int(i)])
                                     for i in order])
            out['embeddings'].append([self._rows[r] for r in rows])
        return out

    def get(self, ids: list[str], include=('embeddings',)) -> dict:
        """The records named, in order, silently skipping ids not held
        (Chroma's behaviour) — the caller pairs ids with vectors by position."""
        out: dict[str, list] = {'ids': [], 'documents': [], 'metadatas': []}
        if 'embeddings' in include:
            out['embeddings'] = []
        for chunk_id in ids:
            at = self._at.get(chunk_id)
            if at is None:
                continue
            out['ids'].append(chunk_id)
            out['documents'].append(self.documents[at])
            out['metadatas'].append(self.metadatas[at])
            if 'embeddings' in out:
                out['embeddings'].append(self._rows[at])
        return out

    def drop(self) -> None:
        """Forgets everything; kept for the panel's rebuild path, though letting the object go would do the same."""
        self.ids = []
        self.documents = []
        self.metadatas = []
        self._rows = []
        self._at = {}
        self._matrix = None

    def _unit_matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = _unit(np.vstack(self._rows).astype(np.float32))
        return self._matrix
