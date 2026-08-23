"""Deterministic scoring against the ground truth: no LLM, no run-to-run variance.

RAGAS metrics (ragas_judged_metrics.py) sit on top for the dimensions that need a model.
Quote recall is specific to this ground truth's evidence.
"""
import difflib
import math
from dataclasses import dataclass

from raglab.rag_components.retrieval import farsi_text_normalizer as textnorm
from raglab.corpora.corpus_reading import evidence_documents


@dataclass(frozen=True)
class Measure:
    """What one number on the dashboard means: label, step, formula, library, help —
    the same shape for ours and RAGAS's metrics. `step` is '' for the whole pipeline."""
    key: str          # the key it arrives under in summary.overall / ragas.metrics
    label: str        # what the score card is titled
    short: str        # the one-line caption under the number
    step: str         # 'retrieval' | 'generation' | '' (whole pipeline)
    formula: str      # the arithmetic, not a description of it
    library: str      # what computed it, and whether a model was involved
    help: str         # the paragraph behind the '!'

    def as_dict(self) -> dict:
        return {'key': self.key, 'label': self.label, 'short': self.short,
                'step': self.step, 'formula': self.formula,
                'library': self.library, 'help': self.help}


# Order matters: this is the order the score cards appear in, headline first.
NO_MODEL = ' — pure Python, no model, so it never varies between runs'

MEASURES = (
    Measure('headline', 'Composite', 'weighted retrieval score', '',
            '0.4·recall + 0.3·quote_recall + 0.2·ndcg + '
            '0.1·abstained_correctly, renormalised over whichever of the four '
            'this run produced',
            'metrics._headline' + NO_MODEL,
            'One comparable number for the leaderboard, weighted in the order '
            'that matters: did retrieval find the evidence, did the answering '
            'sentence survive chunking, was the evidence ranked first, and was '
            'an unanswerable question refused. Generation quality is '
            'deliberately excluded, so a config measured with the extractive '
            'answerer stays comparable to one measured with an LLM.'),
    Measure('recall', 'Recall@k', 'evidence sessions found',
            'retrieval', '|gold ∩ top-k| / |gold|',
            'metrics.recall_at_k' + NO_MODEL,
            'Of the sessions the ground truth marks as evidence, the share '
            'that appear in the top k retrieved. This is the ceiling on '
            'everything downstream: an answer cannot cite what retrieval never '
            'returned. Questions with no evidence (the unanswerable ones) are '
            'excluded rather than scored zero.'),
    Measure('quote_recall', 'Quote recall', 'answering sentence survived chunking',
            'retrieval',
            'matched quotes / total quotes, where a quote counts as matched if it '
            'is a substring of the normalised context or its longest common run '
            'covers >= 0.9 of it',
            'metrics.quote_recall (difflib.SequenceMatcher)' + NO_MODEL,
            'Session recall says the right session was found; this says the '
            'sentence that actually answers the question is inside the '
            'retrieved text. A splitter that cuts mid-thought scores well on '
            'the first and badly on this one, which is exactly the failure a '
            'session-level metric hides.'),
    Measure('ndcg', 'nDCG@k', 'evidence ranked first', 'retrieval',
            'DCG/IDCG with binary gains: DCG = Σ gain_i / log2(i+2), IDCG the '
            'same sum over a perfect ordering',
            'metrics.ndcg_at_k (math.log2)' + NO_MODEL,
            'Rewards putting evidence at the top rather than merely including it '
            'somewhere in k. It matters because the answerer sees a context that '
            'gets truncated, so rank 1 and rank 8 are not worth the same.'),
    Measure('mrr', 'MRR', 'rank of first evidence', 'retrieval',
            '1 / rank of the first evidence session, 0 if none was retrieved',
            'metrics.mrr' + NO_MODEL,
            'How deep you have to read before the first correct hit. Sensitive to '
            'the top of the list only — useful next to nDCG, not instead of it.'),
    Measure('precision', 'Precision@k', 'signal to noise', 'retrieval',
            '|top-k ∩ gold| / |top-k|',
            'metrics.precision_at_k' + NO_MODEL,
            'How much of what was retrieved is actually evidence. Low precision '
            'is not automatically bad — it is the cost of a wide k — but it is '
            'what dilutes the answerer\'s context and what MMR trades against.'),
    Measure('hit', 'Hit@k', 'at least one evidence session', 'retrieval',
            '1 if top-k contains any gold session, else 0',
            'metrics.hit_at_k' + NO_MODEL,
            'The most forgiving retrieval metric: did anything relevant come '
            'back at all. Useful for spotting total misses that an averaged '
            'recall softens.'),
    Measure('abstained_correctly', 'Abstention', 'unanswerable refused',
            'generation',
            'refusals / unanswerable questions',
            'metrics.score_question, reading the answerer\'s refusal flag'
            + NO_MODEL,
            'The corpus genuinely has nothing about some of these questions. '
            'Saying so is the correct answer, and this is the fraction of those '
            'where the answerer refused instead of inventing something.'),
    Measure('false_abstention', 'False refusals', 'answerable wrongly refused',
            'generation', 'refusals / answerable questions (lower is better)',
            'metrics.score_question, reading the answerer\'s refusal flag'
            + NO_MODEL,
            'The other side of abstention, and the failure a badly tuned '
            'relevance gate produces: the evidence was there and the answerer '
            'still said it had nothing. Read it together with abstention — a '
            'gate that refuses everything scores perfectly on one and terribly '
            'here.'),
    Measure('answer_similarity', 'Answer similarity', 'vs the reference answer',
            'generation',
            'difflib ratio = 2·M / T over normalised characters, where M is '
            'matched characters and T the combined length',
            'metrics.answer_similarity (difflib.SequenceMatcher)' + NO_MODEL,
            'A blunt instrument, chosen because it is a *stable* one: no '
            'model, no variance. In practice it mostly tracks whether the same '
            'names, dates and numbers appear, and it punishes a correct short '
            'answer — which is why token F1 sits beside it. It compares '
            'characters, so it is meaningless when the answer and the '
            'reference are in different languages.'),
    Measure('answer_token_f1', 'Answer token F1', 'unigram overlap', 'generation',
            'F1 = 2PR/(P+R) over content words, P = overlap/|predicted|, '
            'R = overlap/|reference|, counting duplicates once',
            'metrics.token_f1 (textnorm tokeniser)' + NO_MODEL,
            'The SQuAD-style measure. It credits a short correct answer that a '
            'character-similarity ratio penalises purely for being short.'),
    Measure('fact_coverage', 'Derived facts', 'judged fact coverage',
            'generation',
            'facts the judge marked present / derived_facts in the ground truth',
            'evaluate.judge_derived_facts — an LLM judge (the "Fact judge" '
            'model role), so this number carries that model\'s variance',
            'The ground truth breaks its reference answer into atomic '
            'derived_facts, and a judge checks each one against the produced '
            'answer. When the facts and the answer are in different languages '
            '— the bundled diary lists English facts against Farsi answers — '
            'the judge is translating as well as checking, which is why no '
            'deterministic metric replaces it, and why a weak model here '
            'produces confidently wrong scores.'),
    Measure('latency_ms', 'Latency', 'ms per question', '',
            'sum of the per-stage timings for one question, in milliseconds',
            'time.perf_counter around each pipeline stage' + NO_MODEL,
            'Whole-pipeline wall clock, so it spans every stage rather than '
            'grading one. Dominated by whatever calls a model: an LLM reranker or '
            'gate moves this by orders of magnitude, a hash embedder does not.'),
    Measure('n_contexts', 'Contexts', 'chunks handed to the answerer', 'retrieval',
            'mean number of contexts surviving rerank, gate and expansion',
            'metrics.score_question' + NO_MODEL,
            'What k actually delivered. It falls below k when the relevance gate '
            'removes chunks, which is the honest reason an answer had less to '
            'work with.'),
    Measure('context_chars', 'Context size', 'characters of context', 'retrieval',
            'mean total characters of the assembled context',
            'metrics.score_question' + NO_MODEL,
            'How much text the answerer was given. Worth watching next to '
            'precision: two configs with the same recall can differ by 3x here, '
            'and the bigger one is paying for it in tokens and in dilution.'),
    Measure('n_summaries', 'Summaries used', 'summary rows in the context', 'index',
            'mean count of contexts whose layer is summary',
            'metrics.score_question' + NO_MODEL,
            'How often the summary hierarchy was actually retrieved. This is '
            'the number that had to exist: a hierarchy scoring flat because '
            'nothing retrieved it and a hierarchy scoring flat because it did '
            'not help are different findings, and no other field on the row '
            'tells them apart. Zero here means the second reading is '
            'unavailable — which is exactly what the lab found once, having '
            'deleted five summary layers that were correct, reachable, and '
            'almost never retrieved.'),
    Measure('n_expanded', 'Drilled down', 'members reached through a summary',
            'retrieval',
            'mean count of contexts expanded from a retrieved summary',
            'metrics.score_question' + NO_MODEL,
            'Contexts that arrived because a summary was retrieved first and then '
            'expanded to the chunks it stands for — the drill-down scope. Zero '
            'under any other scope, by construction. Read beside Summaries used: '
            'the summary is what states an aggregate, and these are the leaves the '
            'answerer can actually quote.'),
)

MEASURE_HELP = {f'metric.{measure.key}':
                f'{measure.help} Formula: {measure.formula}. Computed by '
                f'{measure.library}.'
                for measure in MEASURES}


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float('nan')
    top = set(retrieved[:k])
    return len([g for g in gold if g in top]) / len(gold)


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not retrieved:
        return 0.0
    top = retrieved[:k]
    return len([r for r in top if r in set(gold)]) / len(top)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Binary-gain nDCG. Rewards putting evidence first, not merely including
    it — which matters because the answerer sees a truncated context."""
    if not gold:
        return float('nan')
    gains = [1.0 if item in set(gold) else 0.0 for item in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else float('nan')


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def verbatim_quotes(question: dict) -> list[str]:
    """Every evidence string a lexical match may be scored against — `fidelity
    == 'verbatim'` entries only. A paraphrase is the ground truth's own words
    for something the document says, and a computed fact is never in the text
    at all; a lexical match against either measures nothing, so quote recall
    reads only the entries that are checked to appear character for character
    in the document they cite."""
    return [evidence['text']
            for relevant in question.get('relevant_corpus_documents') or []
            for evidence in relevant.get('evidence') or []
            if evidence.get('fidelity') == 'verbatim']


def quote_recall(context_text: str, question: dict) -> float:
    """Verbatim-quote coverage; falls back to a >=90% similarity match for a
    quote a chunker's whitespace normalisation would otherwise miss."""
    quotes = verbatim_quotes(question)
    if not quotes:
        return float('nan')
    haystack = textnorm.normalize(context_text)
    found = 0
    for quote in quotes:
        needle = textnorm.normalize(quote)
        if needle in haystack:
            found += 1
        elif _fuzzy_contains(haystack, needle):
            found += 1
    return found / len(quotes)


def _fuzzy_contains(haystack: str, needle: str, threshold: float = 0.9) -> bool:
    if len(needle) > len(haystack):
        return False
    matcher = difflib.SequenceMatcher(None, needle, haystack, autojunk=False)
    match = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    return match.size / len(needle) >= threshold


def answer_similarity(response: str, reference: str) -> float:
    """Character-level similarity to the reference answer — no model, no variance."""
    if not response or not reference:
        return 0.0
    return difflib.SequenceMatcher(None, textnorm.normalize(response),
                                   textnorm.normalize(reference)).ratio()


def token_f1(response: str, reference: str) -> float:
    """SQuAD-style unigram F1 over content words."""
    predicted = textnorm.tokens(response)
    gold = textnorm.tokens(reference)
    if not predicted or not gold:
        return 0.0
    overlap = 0
    remaining = list(gold)
    for token in predicted:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_question(question: dict, outcome, k: int) -> dict:
    """Every per-question number the report needs, for one config."""
    gold = [str(document_id) for document_id in evidence_documents(question)]
    retrieved = outcome.sessions
    context_text = '\n'.join(c.text for c in outcome.contexts)
    behavior = question['expected_answer']['behavior']
    # 'correct_premise' must answer *and* contradict the false premise, so for
    # scoring it is answerable exactly like 'answer' — only 'abstain' carries
    # no relevant documents and has nothing to retrieve.
    answerable = behavior != 'abstain'
    row = {
        'id': question['groundtruth_question_id'], 'behavior': behavior,
        'retrieved_sessions': retrieved[:k],
        'n_contexts': len(outcome.contexts),
        # Per question, not inferred from config: 'configured' and 'retrieved'
        # are different facts.
        'n_summaries': (outcome.diagnostics.get('contexts_by_layer') or {}
                        ).get('summary', 0),
        'n_expanded': (outcome.diagnostics.get('contexts_by_layer') or {}
                       ).get('expanded', 0),
        'context_chars': len(context_text),
        'abstained': outcome.abstained,
        'time_scope': outcome.time_scope,
        'latency_ms': round(sum(outcome.timings.values()), 1),
    }
    if answerable:
        row |= {
            'recall': recall_at_k(retrieved, gold, k),
            'precision': precision_at_k(retrieved, gold, k),
            'mrr': mrr(retrieved, gold),
            'ndcg': ndcg_at_k(retrieved, gold, k),
            'hit': hit_at_k(retrieved, gold, k),
            'quote_recall': quote_recall(context_text, question),
        }
        row['false_abstention'] = float(outcome.abstained)
    else:
        # Unanswerable is correctly handled by abstaining; the answerer sets
        # `abstained` for it.
        row['abstained_correctly'] = float(outcome.abstained)
    if outcome.diagnostics.get('answer_error'):
        # `pipeline._llm_answer` swallows every failure into the canonical
        # refusal; this is the only field saying the model was never reached.
        row['answer_error'] = outcome.diagnostics['answer_error']
    if outcome.answer is not None:
        row['answer'] = outcome.answer
        reference = question['expected_answer'].get('text', '')
        if answerable and reference:
            row['answer_similarity'] = answer_similarity(outcome.answer, reference)
            row['answer_token_f1'] = token_f1(outcome.answer, reference)
    return row


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not _isnan(v)]
    return round(sum(clean) / len(clean), 4) if clean else None


def _isnan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


AGGREGATED = ('recall', 'precision', 'mrr', 'ndcg', 'hit', 'quote_recall',
              'false_abstention', 'abstained_correctly',
              'answer_similarity', 'answer_token_f1', 'fact_coverage',
              'latency_ms', 'n_contexts', 'n_summaries', 'n_expanded',
              'context_chars')


def aggregate(rows: list[dict]) -> dict:
    """Overall means over every scored question.

    A per-question-label breakdown (what `by_type`/`by_difficulty` used to be,
    fixed to two vocabularies every corpus had to share) is not reproduced
    here: `type` and `difficulty` are no longer guaranteed fields — a corpus
    declares whatever question labels it likes (D7) — so a generic
    replacement would have to be invented rather than substituted, and this
    step only substitutes. `selection_note` already reports the run's own
    `by_<balance>` breakdown for whichever label a run was balanced on."""
    overall = {name: _mean([r[name] for r in rows if name in r])
               for name in AGGREGATED}
    overall['headline'] = _headline(overall)
    return {'overall': overall, 'n_questions': len(rows)}


def _headline(overall: dict) -> float | None:
    """The weighted composite defined in MEASURES['headline']."""
    parts = [(overall.get('recall'), 0.4), (overall.get('quote_recall'), 0.3),
             (overall.get('ndcg'), 0.2), (overall.get('abstained_correctly'), 0.1)]
    usable = [(v, w) for v, w in parts if v is not None]
    if not usable:
        return None
    total = sum(w for _, w in usable)
    return round(sum(v * w for v, w in usable) / total, 4)
