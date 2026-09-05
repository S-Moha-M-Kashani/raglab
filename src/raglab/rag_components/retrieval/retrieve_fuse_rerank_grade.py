"""Retrieval, fusion, diversification, reranking and grading — each stage
independently switchable so a run can isolate which one moved the score.
"""
import math
import re
from collections import defaultdict

import numpy as np

from raglab.rag_components.retrieval import text_normalizers
from raglab.llm_backends.chat_model_factory import lab_chat


class BM25:
    """Okapi BM25 over the tokens of the normaliser the index was built with —
    the same one for every document and every query, so a corpus in one
    language is never searched through another language's folds."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75,
                 tokenize=text_normalizers.NEUTRAL.tokens):
        self.k1, self.b = k1, b
        self.tokenize = tokenize
        self.docs = [tokenize(d) for d in documents]
        self.lengths = np.array([len(d) or 1 for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tokens in enumerate(self.docs):
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for token, count in counts.items():
                self.postings[token].append((i, count))
        n = len(self.docs) or 1
        self.idf = {token: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for token, p in self.postings.items()}

    def scores(self, query: str, allowed: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len)
        for token in set(self.tokenize(query)):
            posting = self.postings.get(token)
            if not posting:
                continue
            idf = self.idf[token]
            for doc_id, count in posting:
                out[doc_id] += idf * count * (self.k1 + 1) / (count + norm[doc_id])
        if allowed is not None:
            out = np.where(allowed, out, 0.0)
        return out

    def top(self, query: str, k: int,
            allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        scores = self.scores(query, allowed)
        if not scores.size:
            return []
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: combines by rank alone, so incomparable scores (BM25 vs cosine) never need calibration."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] += 1.0 / (k + rank)
    return dict(fused)


def mmr(candidate_vectors: np.ndarray, relevance: np.ndarray, k: int,
        lam: float) -> list[int]:
    """Maximal Marginal Relevance: picks the next chunk that is relevant *and*
    unlike what is already picked. lam=1.0 is plain relevance order; a
    candidate/relevance size mismatch falls back to relevance order."""
    if lam >= 1.0 or candidate_vectors.shape[0] != relevance.size:
        return list(np.argsort(-relevance)[:k])
    chosen: list[int] = []
    remaining = set(range(len(relevance)))
    similarity = candidate_vectors @ candidate_vectors.T
    while remaining and len(chosen) < k:
        best, best_score = None, -1e9
        for i in remaining:
            redundancy = max((similarity[i, j] for j in chosen), default=0.0)
            score = lam * relevance[i] - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)      # type: ignore[arg-type]
        remaining.discard(best)  # type: ignore[arg-type]
    return chosen


def recency_weight(span_to: int, query_date_int: int, half_life_days: float) -> float:
    """Exponential decay on age (the Generative Agents recency term); diary questions skew toward the current state, so age is signal."""
    days = max(0.0, _days_between(span_to, query_date_int))
    return float(0.5 ** (days / max(1.0, half_life_days)))


def _days_between(a: int, b: int) -> float:
    from datetime import date
    def to_date(value: int) -> date:
        return date(value // 10000, (value // 100) % 100, value % 100)
    try:
        return abs((to_date(b) - to_date(a)).days)
    except ValueError:
        return 0.0


def normalize_scores(values: np.ndarray) -> np.ndarray:
    """Min-max to [0,1], needed before mixing relevance with recency and importance, which live on their own scales."""
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values) * 0.5
    return (values - low) / (high - low)


class CrossEncoderReranker:
    """fastembed's cross-encoder: scores the (query, chunk) pair jointly rather than comparing two independent embeddings."""

    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model = TextCrossEncoder(model_name=model_name)
        self.model_name = model_name

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        return np.array(list(self.model.rerank(query, documents)), dtype=np.float32)


def cross_encoder_available(model_name: str) -> bool:
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        return any(m['model'] == model_name
                   for m in TextCrossEncoder.list_supported_models())
    except Exception:
        return False


LLM_GRADE_PROMPT = (
    'You score how useful each numbered excerpt is for answering a question '
    'about the user\'s personal diary. The diary is in Persian. Reply with one '
    'line per excerpt in the form "<number>: <score 0-10>" and nothing else. '
    '0 means irrelevant, 10 means it directly contains the answer.')


class GradeUnavailable(RuntimeError):
    """The grading model could not be reached, so nothing was scored."""


def llm_scores(llm, model: str, query: str, documents: list[str],
               max_chars: int = 700) -> np.ndarray:
    """Batches all candidates into one call; one call per candidate would multiply request count across a sweep."""
    if not documents:
        return np.zeros(0, dtype=np.float32)
    listing = '\n\n'.join(f'[{i + 1}] {d[:max_chars]}'
                          for i, d in enumerate(documents))
    try:
        turn = lab_chat(llm, [{'role': 'system', 'content': LLM_GRADE_PROMPT},
                              {'role': 'user',
                               'content': f'Question: {query}\n\n{listing}'}],
                        model)
        text = turn.content or ''
    except Exception as error:
        # Raise rather than default to 0.5: that score clears the gate's 0.4
        # threshold, which would make grader='llm' silently score ungated.
        raise GradeUnavailable(
            f'the LLM grade stage could not reach its model '
            f'({model or "the configured default"}): {error}') from error
    scores = np.full(len(documents), np.nan, dtype=np.float32)
    for line in text.splitlines():
        match = re.match(r'\s*\[?(\d+)\]?\s*[:.\-]\s*(\d+(?:\.\d+)?)', line)
        if match:
            i, value = int(match.group(1)) - 1, float(match.group(2))
            if 0 <= i < len(documents):
                scores[i] = min(10.0, value) / 10.0
    # Unparsed means "no opinion", not "irrelevant" — zero would silently empty the context.
    return np.where(np.isnan(scores), 0.5, scores)
