"""One question, end to end: understand → retrieve → fuse → rerank → grade →
assemble → answer, with every stage's output kept for the panel.
Diagnostics say *which stage* lost the evidence, so tuning is not guesswork.
"""
import re
import time
from dataclasses import dataclass, field

import numpy as np
from raglab.rag_components.retrieval import farsi_text_normalizer as textnorm

from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.rag_components.retrieval import query_understanding as query_mod
from raglab.rag_components.retrieval import (
    retrieve_fuse_rerank_grade as retrieval)
from raglab.configuration.lab_config import GenerationConfig, RetrievalConfig
from raglab.llm_backends.chat_model_factory import lab_chat
from raglab.llm_backends.model_role_catalogue import Roles

REFUSAL = 'تو دفترچه چیزی دربارهٔ این پیدا نکردم.'
REFUSALS = {
    'Persian': REFUSAL,
    'English': 'I could not find anything about this in the diary.',
    'German': 'Ich habe dazu im Tagebuch nichts gefunden.',
}
GERMAN_WORDS = {'der', 'die', 'das', 'ist', 'sind', 'wie', 'wer', 'welche',
                'welcher', 'welches', 'heute', 'gestern', 'warum', 'wann',
                'wo', 'wurde', 'nicht'}
# «نمیدونم» is deliberately absent: the diarist says it constantly in
# non-refusals, so counting it would misread quoted answers as abstentions.
REFUSAL_MARKERS = ('پیدا نکردم', 'چیزی ثبت نشده', 'اطلاعاتی ندارم',
                   'در دفترچه نیست', 'در یادداشت‌ها نیست', 'موجود نیست',
                   'اشاره‌ای نشده', 'اشاره ای نشده', 'ذکر نشده', 'ثبت نشده')


@dataclass
class Context:
    chunk_id: str
    text: str
    session_id: str    # a chunk's `document_id` — the field name is unchanged
                        # here (and in the API payload it feeds) pending the
                        # panel/evaluation surfaces that read it
    date: str
    score: float
    stages: dict = field(default_factory=dict)
    expanded_from: str = ''

    def as_dict(self) -> dict:
        return {'chunk_id': self.chunk_id,
                'session_id': self.session_id, 'date': self.date,
                'score': round(self.score, 4), 'text': self.text,
                'stages': {k: round(v, 4) for k, v in self.stages.items()},
                'expanded_from': self.expanded_from}


@dataclass
class Outcome:
    question: str
    contexts: list[Context]
    abstained: bool = False
    answer: str | None = None
    time_scope: dict | None = None
    diagnostics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def sessions(self) -> list[str]:
        """Distinct sessions represented in the context, best first — the unit
        the ground truth cites evidence in."""
        out: list[str] = []
        for context in self.contexts:
            if context.session_id and context.session_id not in out:
                out.append(context.session_id)
        return out

    def as_dict(self) -> dict:
        return {'question': self.question, 'abstained': self.abstained,
                'answer': self.answer, 'time_scope': self.time_scope,
                'contexts': [c.as_dict() for c in self.contexts],
                'sessions': self.sessions, 'diagnostics': self.diagnostics,
                'timings': self.timings}


def _allowed(index, scope, layers: tuple[str, ...] | None = None,
             levels: tuple[int, ...] | None = None) -> np.ndarray:
    """The BM25 equivalent of the store's `where` clause; kept in lockstep with
    `query.layer_clause` — if they disagree, hybrid fusion silently compares two different pools."""
    allowed = np.ones(len(index.chunks), dtype=bool)
    for i, chunk in enumerate(index.chunks):
        if scope and (chunk.span_from > scope.to_int
                      or chunk.span_to < scope.from_int):
            allowed[i] = False
        elif layers is not None and chunk.layer not in layers:
            allowed[i] = False
        elif levels and chunk.layer == 'summary' and chunk.level not in levels:
            allowed[i] = False
    return allowed


def summary_filter(cfg: RetrievalConfig) -> tuple[tuple[str, ...] | None,
                                                  tuple[int, ...] | None]:
    """Which layers and levels this scope may retrieve. `None` means no
    restriction (the flat behaviour, and what `mixed` gets), so an index with
    no summaries is searched exactly as before this setting existed."""
    levels = tuple(int(part) for part in cfg.summary_levels.split()
                   if part.isdigit()) or None
    if cfg.summary_scope == 'leaves':
        return ('',), None
    if cfg.summary_scope in ('summaries', 'drill-down'):
        return ('summary',), levels
    return None, levels


def lexical_grade(index, question: str, text: str) -> float:
    """IDF-weighted coverage of the question's informative words, bounded to
    [0,1] so it can be thresholded (unlike a raw BM25 score)."""
    tokens = [t for t in textnorm.tokens(question)
              if t not in query_mod.QUESTION_WORDS]
    if not tokens:
        return 0.0
    present = set(textnorm.tokens(text))
    weights = {t: index.bm25.idf.get(t, 1.0) for t in set(tokens)}
    total = sum(weights.values()) or 1.0
    return float(sum(w for t, w in weights.items() if t in present) / total)


def retrieve(index, cfg: RetrievalConfig, question: str, query_date: str,
             llm=None, models: Roles | None = None,
             trace: dict | None = None) -> Outcome:
    # An empty Roles means "whatever the provider defaults to".
    roles = models or Roles()
    timings: dict = {}
    diagnostics: dict = {}
    clock = time.perf_counter

    start = clock()
    scope = (query_mod.resolve_time_scope(question, query_date)
             if cfg.time_filter else None)
    queries = query_mod.expand(question) if cfg.multi_query else [question]
    if cfg.hyde and llm is not None:
        queries = queries + [query_mod.hyde(llm, roles.expand, question)]
    layers, levels = summary_filter(cfg)
    where = query_mod.layer_clause(scope, layers, levels)
    allowed = _allowed(index, scope, layers, levels)
    timings['understand_ms'] = round((clock() - start) * 1000, 1)
    diagnostics['queries'] = queries
    diagnostics['candidates_in_scope'] = int(allowed.sum())

    start = clock()
    dense_ranked: list[str] = []
    dense_scores: dict[str, float] = {}
    if cfg.retriever in ('dense', 'hybrid-rrf'):
        # A question embeds as a question: E5 models expect "query: "/"passage: " and quietly lose accuracy without them.
        vectors = embedding.query_vectors(index.embedder, queries)
        for chunk_id, score in index.dense(vectors, cfg.candidates, where):
            dense_scores[chunk_id] = score
            dense_ranked.append(chunk_id)
    bm25_ranked: list[str] = []
    bm25_scores: dict[str, float] = {}
    if cfg.retriever in ('bm25', 'hybrid-rrf'):
        merged: dict[int, float] = {}
        for text in queries:
            for doc_id, score in index.bm25.top(text, cfg.candidates, allowed):
                merged[doc_id] = max(merged.get(doc_id, 0.0), score)
        for doc_id, score in sorted(merged.items(), key=lambda kv: -kv[1]):
            chunk_id = index.chunks[doc_id].id
            bm25_scores[chunk_id] = score
            bm25_ranked.append(chunk_id)
    timings['retrieve_ms'] = round((clock() - start) * 1000, 1)
    diagnostics['dense_hits'] = len(dense_ranked)
    diagnostics['bm25_hits'] = len(bm25_ranked)

    if cfg.retriever == 'dense':
        base = {cid: dense_scores[cid] for cid in dense_ranked}
    elif cfg.retriever == 'bm25':
        base = {cid: bm25_scores[cid] for cid in bm25_ranked}
    else:
        base = retrieval.rrf([dense_ranked, bm25_ranked], cfg.rrf_k)
    # Applied before the candidate cut, never after: far more leaves than
    # summaries, so an unpromoted summary can't be boosted into the cut.
    if cfg.summary_boost != 1.0:
        boosted = 0
        for chunk_id in base:
            chunk = index.by_id.get(chunk_id)
            if chunk is not None and chunk.layer == 'summary':
                base[chunk_id] *= cfg.summary_boost
                boosted += 1
        diagnostics['summaries_boosted'] = boosted
    if not base:
        if trace is not None:
            trace.update({'dense': list(dense_ranked), 'bm25': list(bm25_ranked),
                          'fused': [], 'candidates': []})
        return Outcome(question=question, contexts=[], abstained=True,
                       time_scope=scope.as_dict() if scope else None,
                       diagnostics=diagnostics | {'reason': 'no candidates'},
                       timings=timings)

    ids = sorted(base, key=lambda cid: -base[cid])[:max(cfg.rerank_depth, cfg.k)]
    chunks = [index.by_id[cid] for cid in ids if cid in index.by_id]
    ids = [c.id for c in chunks]
    relevance = retrieval.normalize_scores(
        np.array([base[cid] for cid in ids], dtype=np.float32))
    stage_scores = {cid: {'retrieval': float(r)} for cid, r in zip(ids, relevance)}

    start = clock()
    final = _rerank(index, cfg, question, chunks, relevance, query_date,
                    stage_scores, llm, roles.rerank)
    timings['rerank_ms'] = round((clock() - start) * 1000, 1)

    start = clock()
    vectors = (index.vectors_for(ids) if cfg.mmr_lambda < 1.0
               else np.zeros((0, 1), dtype=np.float32))
    order = retrieval.mmr(vectors, final, cfg.k, cfg.mmr_lambda)
    timings['diversify_ms'] = round((clock() - start) * 1000, 1)

    contexts = []
    for i in order:
        chunk = chunks[i]
        contexts.append(Context(chunk_id=chunk.id, text=chunk.text,
                                session_id=chunk.document_id, date=chunk.date,
                                score=float(final[i]),
                                stages=stage_scores[chunk.id]))

    start = clock()
    kept, abstained = _grade(index, cfg, question, contexts, llm, roles.grade)
    timings['grade_ms'] = round((clock() - start) * 1000, 1)
    diagnostics['graded_out'] = len(contexts) - len(kept)

    if cfg.summary_scope == 'drill-down':
        kept = _drill_down(index, cfg, question, kept)
    kept = _fit_budget(kept, cfg.max_context_chars)
    # Which layers actually reached the answerer — a field on the row, not something to infer from a flat score.
    diagnostics['contexts_by_layer'] = _by_layer(index, kept)
    if trace is not None:
        trace.update(_trace_candidates(chunks, dense_ranked, bm25_ranked,
                                       base, relevance, final, contexts, kept))
    return Outcome(question=question, contexts=kept, abstained=abstained,
                   time_scope=scope.as_dict() if scope else None,
                   diagnostics=diagnostics, timings=timings)


def _trace_candidates(chunks, dense_ranked, bm25_ranked, base, relevance,
                      final, contexts, kept) -> dict:
    """The Inspector's per-candidate ladder: each candidate's rank at every
    stage plus whether it survived grading. `contexts` is pre-grade (its
    `grade` stage score, if any) and `kept` is post-grade — the two rank maps
    this needs but `retrieve` otherwise has no reason to keep side by side."""
    fused_order = sorted(base, key=lambda cid: -base[cid])
    dense_pos = {cid: r + 1 for r, cid in enumerate(dense_ranked)}
    bm25_pos = {cid: r + 1 for r, cid in enumerate(bm25_ranked)}
    fused_pos = {cid: r + 1 for r, cid in enumerate(fused_order)}
    kept_ids = {c.chunk_id for c in kept}
    grade_by_id = {c.chunk_id: c.stages.get('grade') for c in contexts}
    candidates = []
    for i, chunk in enumerate(chunks):
        cid = chunk.id
        grade = grade_by_id.get(cid)
        candidates.append({
            'chunk_id': cid, 'text': chunk.text,
            'session_id': chunk.document_id, 'date': chunk.date,
            # Layer is a different axis from rank: a summary that ranked
            # first but expanded to irrelevant leaves must be visible as that, not as a score.
            'layer': chunk.layer, 'level': chunk.level,
            'group_id': chunk.group_id,
            'members': len(chunk.member_ids),
            'dense_rank': dense_pos.get(cid), 'bm25_rank': bm25_pos.get(cid),
            'fused_rank': fused_pos.get(cid),
            'retrieval_score': round(float(relevance[i]), 4),
            'rerank_score': round(float(final[i]), 4),
            'grade_score': (round(float(grade), 4) if grade is not None else None),
            'kept': cid in kept_ids})
    return {'dense': list(dense_ranked), 'bm25': list(bm25_ranked),
            'fused': fused_order, 'candidates': candidates}


def _rerank(index, cfg, question, chunks, relevance, query_date, stage_scores,
            llm, model) -> np.ndarray:
    if cfg.reranker == 'none' or not chunks:
        return relevance
    query_int = int(query_date.replace('-', ''))
    if cfg.reranker == 'lexical':
        scores = np.array([lexical_grade(index, question, c.text) for c in chunks],
                          dtype=np.float32)
        final = 0.5 * relevance + 0.5 * retrieval.normalize_scores(scores)
        key = 'lexical'
    elif cfg.reranker == 'recency':
        weights = np.array([retrieval.recency_weight(c.span_to, query_int,
                                                     cfg.recency_half_life_days)
                            for c in chunks], dtype=np.float32)
        final = relevance * weights
        scores, key = weights, 'recency'
    elif cfg.reranker == 'agentic':
        # Generative Agents' retrieval function: relevance + recency + importance.
        wr, wt, wi = cfg.agentic_weights
        recency = np.array([retrieval.recency_weight(c.span_to, query_int,
                                                     cfg.recency_half_life_days)
                            for c in chunks], dtype=np.float32)
        importance = np.array([c.importance for c in chunks], dtype=np.float32)
        final = wr * relevance + wt * recency + wi * importance
        scores, key = final, 'agentic'
    elif cfg.reranker == 'cross-encoder':
        try:
            encoder = _cross_encoder()
            scores = encoder.score(question, [c.text for c in chunks])
            final = retrieval.normalize_scores(scores)
        except Exception:
            return relevance
        key = 'cross_encoder'
    elif cfg.reranker == 'llm':
        if llm is None:
            return relevance
        scores = retrieval.llm_scores(llm, model, question,
                                      [c.text for c in chunks])
        final = 0.3 * relevance + 0.7 * scores
        key = 'llm_grade'
    else:
        raise ValueError(f'unknown reranker: {cfg.reranker!r}')
    for chunk, score in zip(chunks, scores):
        stage_scores[chunk.id][key] = float(score)
    return final.astype(np.float32)


_ENCODERS: dict[str, retrieval.CrossEncoderReranker] = {}


def _cross_encoder():
    from raglab.configuration.lab_config import load_lab_settings
    name = load_lab_settings().cross_encoder_model
    if name not in _ENCODERS:
        _ENCODERS[name] = retrieval.CrossEncoderReranker(name)
    return _ENCODERS[name]


def _grade(index, cfg, question, contexts, llm, model):
    """Drops contexts below the bar; the second return value (whether nothing
    cleared it) is the whole abstention story — a pipeline with no gate answers every question."""
    if cfg.grader == 'none':
        return contexts, False
    if cfg.grader == 'lexical':
        grades = [lexical_grade(index, question, c.text) for c in contexts]
    else:
        grades = list(retrieval.llm_scores(llm, model, question,
                                           [c.text for c in contexts])) \
            if llm is not None else [1.0] * len(contexts)
    kept = []
    for context, grade in zip(contexts, grades):
        context.stages['grade'] = float(grade)
        if grade >= cfg.grade_threshold:
            kept.append(context)
    return kept, not kept


def _by_layer(index, contexts) -> dict:
    """`{'leaf': n, 'summary': n, 'expanded': n}` for one question's context."""
    counts = {'leaf': 0, 'summary': 0, 'expanded': 0}
    for context in contexts:
        chunk = index.by_id.get(context.chunk_id)
        if context.expanded_from:
            counts['expanded'] += 1
        elif chunk is not None and chunk.layer == 'summary':
            counts['summary'] += 1
        else:
            counts['leaf'] += 1
    return counts


def _drill_down(index, cfg, question: str, contexts):
    """Expands each retrieved summary to the members it stands for. The
    summary itself is kept first (it states the count, for a counting
    question); members are ordered by IDF coverage rather than a stored score,
    since none exists here and re-embedding would add an encoder pass per query."""
    out = []
    for context in contexts:
        out.append(context)
        chunk = index.by_id.get(context.chunk_id)
        if chunk is None or chunk.layer != 'summary':
            continue
        members = [index.by_id[cid] for cid in chunk.member_ids
                   if cid in index.by_id]
        ranked = sorted(members,
                        key=lambda c: -lexical_grade(index, question, c.text))
        for member in ranked[:max(1, cfg.k)]:
            out.append(Context(chunk_id=member.id, text=member.text,
                               session_id=member.document_id, date=member.date,
                               score=context.score,
                               stages=dict(context.stages),
                               expanded_from=chunk.id))
    return out


def _fit_budget(contexts, max_chars: int):
    """Truncates the *list*, never a chunk: half an entry reads as complete and can flip the meaning of what's quoted."""
    out, used = [], 0
    for context in contexts:
        if out and used + len(context.text) > max_chars:
            continue
        out.append(context)
        used += len(context.text)
    return out


def question_language(question: str) -> str:
    """Return the answer language using the question's text, not the corpus."""
    if re.search(r'[\u0600-\u06ff]', question):
        return 'Persian'
    if re.search(r'[äöüß]', question.lower()):
        return 'German'
    words = set(re.findall(r"[a-zA-ZÀ-ÿ]+", question.lower()))
    if words & GERMAN_WORDS:
        return 'German'
    return 'English'


def refusal_for(question: str) -> str:
    return REFUSALS[question_language(question)]


ANSWER_PROMPT = (
    'You answer questions about a diary using only the provided excerpts. '
    'answer in {language}. Be concise and specific, include dates, and say '
    'what happened last when something changed. Put the session id in [] '
    'after each claim. If the answer is not in the excerpts, write exactly: '
    '“{refusal}”. Do not invent information. Correct a false premise.')


def answer(outcome: Outcome, cfg: GenerationConfig, llm=None,
           models: Roles | None = None) -> Outcome:
    if cfg.answerer == 'none':
        return outcome
    if outcome.abstained or not outcome.contexts:
        outcome.answer = refusal_for(outcome.question)
        return outcome
    started = time.perf_counter()
    if cfg.answerer == 'extractive':
        outcome.answer = _extractive_answer(outcome)
    else:
        roles = models or Roles()
        outcome.answer = _llm_answer(outcome, llm, roles.answer or cfg.model)
    outcome.timings['answer_ms'] = round((time.perf_counter() - started) * 1000, 1)
    if reads_as_refusal(outcome.answer or '', cfg.answerer):
        outcome.abstained = True
    return outcome


def reads_as_refusal(answer: str, answerer: str) -> bool:
    """Did the model decline? Only the LLM answerer can decline in its own
    words, and only in the opening sentence — a refusal phrase deep in a long answer is the model quoting the diary."""
    if answer.strip() in REFUSALS.values():
        return True
    if answerer != 'llm':
        return False
    opening = ' '.join(textnorm.sentences(answer)[:1])
    return any(marker in opening for marker in REFUSAL_MARKERS)


def _extractive_answer(outcome: Outcome, limit: int = 3) -> str:
    """A deterministic stand-in for generation, with no key and no cost — quoting, not answering, never compared against LLM runs as the same system."""
    lines = []
    for context in outcome.contexts[:limit]:
        sentences = textnorm.sentences(context.text)
        best = max(sentences, key=len) if sentences else context.text
        tag = f' [{context.session_id or context.chunk_id}]'
        lines.append(best[:300] + tag)
    return ' '.join(lines)


def context_blocks(outcome: Outcome) -> str:
    """The retrieved evidence as the answerer sees it: labelled, dated, never
    truncated here — `_fit_budget` already dropped whole contexts to the
    character budget."""
    return '\n\n'.join(
        f'[{context.session_id or context.chunk_id} | {context.date}]\n'
        f'{context.text}' for context in outcome.contexts)


def _llm_answer(outcome: Outcome, llm, model: str) -> str:
    if llm is None:
        return refusal_for(outcome.question)
    try:
        language = question_language(outcome.question)
        prompt = ANSWER_PROMPT.format(language=language,
                                      refusal=REFUSALS[language])
        turn = lab_chat(llm, [{'role': 'system', 'content': prompt},
                              {'role': 'user', 'content':
                               f"سؤال: {outcome.question}\n\nتکه‌های دفترچه:\n"
                               + context_blocks(outcome)}], model)
        return (turn.content or '').strip() or refusal_for(outcome.question)
    except Exception as error:
        outcome.diagnostics['answer_error'] = str(error)[:200]
        return refusal_for(outcome.question)


def retrieve_traced(index, cfg: RetrievalConfig, question: str, query_date: str,
                    llm=None, models: Roles | None = None) -> tuple[Outcome, dict]:
    """`retrieve`, plus the full per-candidate step ladder for the Inspector;
    a thin wrapper so the eval path never pays the trace's cost."""
    trace: dict = {}
    outcome = retrieve(index, cfg, question, query_date,
                       llm=llm, models=models, trace=trace)
    return outcome, trace
