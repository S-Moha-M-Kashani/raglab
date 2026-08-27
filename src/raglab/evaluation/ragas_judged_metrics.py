"""RAGAS bridge: offline metrics (string similarity, no model) and llm metrics
(Faithfulness, ResponseRelevancy, FactualCorrectness, the LLM context pair).

RAGAS is an optional dependency of the lab. Everything here is imported lazily
and every failure is reported as a note rather than raised.
"""
import os
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler

from raglab.evaluation.deterministic_metrics import Measure

# RAGAS posts a usage event per evaluate() call and blocks for minutes when that
# endpoint is unreachable. Must be set before any ragas import; every ragas
# import in this module is lazy, so this module-level line always wins.
os.environ.setdefault('RAGAS_DO_NOT_TRACK', 'true')

OFFLINE_METRICS = ('non_llm_context_precision_with_reference',
                   'non_llm_context_recall')
LLM_METRICS = ('faithfulness', 'answer_relevancy', 'factual_correctness(mode=f1)',
               'llm_context_precision_with_reference', 'context_recall')

# The four that get a vote — the rest is still computed and reported, but a
# metric only earns a vote if it grades something the others do not:
#   context precision   was what came back relevant, and ranked well?
#   context recall       did retrieval get everything the answer needed?
#   faithfulness         did the answer stay inside what was retrieved?
#   answer relevancy     did it actually address the question asked?
# Excluded: `factual_correctness` grades the fixture's phrasing as much as the
# pipeline; the offline pair are string containment over verbatim quotes, so
# they grade retrieval only and never vary; our own deterministic metrics grade
# retrieval almost exclusively. Those stay as numbers to debug with — they never vary between
# runs, which a judged score cannot promise.
DECISION_METRICS = ('faithfulness', 'answer_relevancy',
                    'llm_context_precision_with_reference', 'context_recall')

# How hard RAGAS is allowed to push the judge, per backend. RAGAS defaults to 16
# concurrent requests; a local model serves far fewer at once, so the tail waits
# behind the queue and trips the client timeout. Concurrency and timeout cannot
# change *what* a judge scores, only whether the score arrives. A CLI backend is
# throttled for a different reason than a local daemon — each call is a whole
# process rather than a socket — but the same limit applies.
JUDGE_LOAD = {'openrouter': {'max_workers': 16, 'timeout': 180},
              'ollama': {'max_workers': 3, 'timeout': 600},
              'claude': {'max_workers': 3, 'timeout': 600},
              'codex': {'max_workers': 3, 'timeout': 600},
              'fake': {'max_workers': 16, 'timeout': 60}}


def judge_load(settings) -> dict:
    """RunConfig arguments for the backend actually serving the judge."""
    return JUDGE_LOAD.get(getattr(settings, 'provider', 'openrouter'),
                          JUDGE_LOAD['openrouter'])


# Judge calls per sample, per metric — measured against the prompts RAGAS
# actually sends. Context precision is not listed here: it is one verdict per
# retrieved chunk, so it scales with k instead (see expected_judge_calls).
CALLS_PER_SAMPLE = {'faithfulness': 2, 'answer_relevancy': 1,
                    'context_recall': 1, 'factual_correctness': 2}


def expected_judge_calls(n_samples: int, k: int,
                         include_factual: bool = True) -> int:
    """Roughly how many judge calls a judged run will make — an estimate, since
    RAGAS retries malformed output, so the true figure is this ± retries."""
    per_sample = sum(count for name, count in CALLS_PER_SAMPLE.items()
                     if include_factual or name != 'factual_correctness')
    return max(0, n_samples) * (per_sample + max(0, k))


class JudgeWatch(BaseCallbackHandler):
    """Counts judge calls as they land, so a batch scored behind one `evaluate()`
    call still reports progress. A LangChain handler rather than RAGAS's own
    `_pbar`, which is private API — a progress bar must not be able to break a
    run."""

    def __init__(self, total: int, progress=None,
                 base: float = 0.92, span: float = 0.07):
        self.total = max(1, total)
        self.progress = progress
        self.base = base
        self.span = span
        self.calls = 0

    def fraction(self) -> float:
        # Clamped: a retrying judge can exceed the estimate.
        return self.base + self.span * min(1.0, self.calls / self.total)

    def detail(self) -> str:
        return f'judge call {self.calls} of ~{self.total}'

    def _tick(self) -> None:
        self.calls += 1
        if self.progress:
            self.progress('ragas', self.fraction(), self.detail())

    def on_llm_end(self, response, **kwargs) -> None:
        self._tick()

    def on_llm_error(self, error, **kwargs) -> None:
        # A failed call is still time spent, so it counts too.
        self._tick()


def decision_score(metrics: dict) -> float | None:
    """The unweighted mean of the four deciding metrics, or None unless every one
    of them is present — a partial mean would let the run that measured least win."""
    values = []
    for name in DECISION_METRICS:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or value != value:   # missing/NaN
            return None
        values.append(float(value))
    return round(sum(values) / len(values), 4)


def decision_spread(rows: list[dict]) -> dict:
    """Mean and standard error of the deciding score over per-question composites.

    Per question first, then across questions: the four metrics score the same
    answers and move together, so pooling four separate standard errors would
    understate the spread. A question missing any of the four has no composite,
    the same rule `decision_score` applies to the run as a whole."""
    composites = [value for value in (decision_score(row) for row in rows)
                  if value is not None]
    n = len(composites)
    if not n:
        return {'n': 0, 'mean': None, 'stderr': None}
    mean = sum(composites) / n
    if n < 2:
        # One sample has no spread. Reporting 0.0 would read as certainty.
        return {'n': n, 'mean': round(mean, 4), 'stderr': None}
    variance = sum((value - mean) ** 2 for value in composites) / (n - 1)
    return {'n': n, 'mean': round(mean, 4),
            'stderr': round((variance / n) ** 0.5, 4)}


INSTALL_HINT = 'uv sync  (these are locked dependencies now: ' \
    "ragas==0.4.*, langchain-community>=0.3.31,<0.4, rapidfuzz)"

# RAGAS's metrics, under RAGAS's names, in the same shape as metrics.MEASURES.
JUDGED = ', scored by the RAGAS judge model — a model\'s verdict, so it varies'
NO_JUDGE = ' with the lab\'s quote-in-chunk similarity (rapidfuzz ' \
    'partial_ratio, threshold 0.5), no model involved'

RAGAS_MEASURES = (
    Measure('ragas_decision', 'RAGAS decision score',
            'the number the architecture was chosen by', '',
            'mean(faithfulness, answer_relevancy, '
            'llm_context_precision_with_reference, context_recall) — undefined '
            'unless all four are present',
            'ragas_eval.decision_score over ragas 0.4.x metrics' + JUDGED,
            'Four judged RAGAS metrics, averaged unweighted, and the only score '
            'that picks between configurations here. They were chosen because '
            'between them they cover both halves of the pipeline in both '
            'directions: context precision asks whether what came back was '
            'relevant and well ranked, context recall whether everything the '
            'answer needed was retrieved, faithfulness whether the answer '
            'stayed inside what was retrieved, and answer relevancy whether it '
            'addressed the question at all. Everything else on this screen is '
            'reported and none of it votes — factual correctness grades the '
            'fixture\'s phrasing as much as the pipeline, the offline context '
            'pair only check that verbatim quotes were retrieved, and our own '
            'deterministic metrics grade retrieval almost exclusively, so '
            'ranking on them rewards a config that finds the evidence and then '
            'says nothing useful about it. The cost of that choice is honest: '
            'all four come from a judge, so this number carries a model\'s '
            'variance, and it is only comparable between runs scored by the '
            'same RAGAS judge model.'),
    Measure('non_llm_context_recall', 'Context recall (offline)',
            'RAGAS, string distance', 'retrieval',
            'reference quotes found inside some retrieved context / reference '
            'quotes, a quote counting as found when its best alignment inside '
            'a chunk is at least 0.5 similar',
            'ragas 0.4.x NonLLMContextRecall' + NO_JUDGE,
            'RAGAS\'s context recall with the judge removed: for each verbatim '
            'evidence quote in the ground truth it asks whether some retrieved '
            'chunk contains it, near enough. Deterministic, so it never varies '
            'between runs — but it grades retrieval only, and a quote is '
            'credited wherever it lands in a chunk, so a huge chunk that '
            'happens to contain it scores the same as a tight one. It does '
            'not vote.'),
    Measure('non_llm_context_precision_with_reference',
            'Context precision (offline)', 'RAGAS, string distance', 'retrieval',
            'mean precision@k over the retrieved contexts, a context counting as '
            'relevant when some reference quote aligns inside it at 0.5 or better',
            'ragas 0.4.x NonLLMContextPrecisionWithReference' + NO_JUDGE,
            'How many of the retrieved chunks contain a piece of the reference '
            'evidence, weighted towards the top of the ranking. Same caveat as '
            'its recall twin: containment credits a chunk however large it is.'),
    Measure('faithfulness', 'Faithfulness', 'answer supported by the context',
            'generation',
            'supported claims / total claims in the answer',
            'ragas 0.4.x Faithfulness' + JUDGED,
            'RAGAS\'s own definition: the judge breaks the generated answer into '
            'individual claims, then checks each one against the retrieved '
            'context. An answer is faithful when the context supports every '
            'claim in it — so this measures invention, not correctness. A '
            'perfectly faithful answer can still be wrong if retrieval returned '
            'the wrong passage.'),
    Measure('answer_relevancy', 'Answer relevancy', 'does it answer the question',
            'generation',
            'mean cosine similarity between the original question and n '
            'questions the judge reverse-engineers from the answer',
            'ragas 0.4.x ResponseRelevancy (judge + the lab\'s own embedder for '
            'the vectors)' + JUDGED,
            'RAGAS asks the judge to invent the questions the answer would be a '
            'reply to, embeds them, and compares them with the real question. It '
            'catches answers that are true and on-topic but do not actually '
            'address what was asked. The vectors come from the embedder under '
            'test, so no external embedding API is called.'),
    Measure('factual_correctness(mode=f1)', 'Factual correctness',
            'claims vs the reference answer', 'generation',
            'F1 = 2PR/(P+R) over claims: P = claims in the answer supported by '
            'the reference, R = claims in the reference present in the answer',
            'ragas 0.4.x FactualCorrectness(mode=f1)' + JUDGED,
            'Unlike faithfulness, this compares the answer to the *reference '
            'answer* rather than to the retrieved context — so it measures being '
            'right, not merely being grounded. Both directions are counted, which '
            'is why it is an F1: an answer that omits half the reference loses '
            'recall, one that adds unsupported detail loses precision.'),
    Measure('llm_context_precision_with_reference', 'Context precision (judged)',
            'RAGAS, judged relevance', 'retrieval',
            'mean precision@k where the judge, not a string metric, decides '
            'whether each retrieved context is relevant',
            'ragas 0.4.x LLMContextPrecisionWithReference' + JUDGED,
            'The judged twin of the offline precision metric. It judges relevance '
            'rather than checking for a verbatim quote, so it can credit a '
            'paraphrase the offline pair would miss, at the cost of a model call '
            'per context and a model\'s variance.'),
    Measure('context_recall', 'Context recall (judged)', 'RAGAS, judged coverage',
            'retrieval',
            'reference claims attributable to the retrieved context / reference '
            'claims',
            'ragas 0.4.x LLMContextRecall' + JUDGED,
            'The judge splits the reference answer into claims and asks, for each '
            'one, whether the retrieved context could support it. It answers '
            '"was everything needed retrieved?" without depending on the exact '
            'wording of a quote.'),
)

RAGAS_MEASURE_HELP = {
    f'metric.{measure.key}':
    f'{measure.help} Formula: {measure.formula}. Computed by {measure.library}.'
    for measure in RAGAS_MEASURES}


@dataclass
class Availability:
    installed: bool = False
    llm_ready: bool = False
    version: str = ''
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {'installed': self.installed, 'llm_ready': self.llm_ready,
                'version': self.version, 'notes': list(self.notes),
                'offline_metrics': list(OFFLINE_METRICS),
                'llm_metrics': list(LLM_METRICS), 'install_hint': INSTALL_HINT}


def availability(settings) -> Availability:
    notes: list[str] = []
    try:
        import ragas
    except Exception as error:
        return Availability(notes=(f'ragas not importable: {error}',))
    version = getattr(ragas, '__version__', '?')
    try:
        import rapidfuzz  # noqa: F401
    except Exception:
        notes.append('rapidfuzz is missing — the offline (non-LLM) RAGAS metrics '
                     'cannot run without it')
    llm_ready = False
    # Asks whether a real judge can be reached, not whether a key exists: a local
    # Ollama model judges fine with none. The fake provider must stay disqualified.
    if not settings.llm_ready:
        notes.append('no LLM backend for judging — set OPENROUTER_API_KEY, or '
                     'RAGLAB_LLM=ollama to judge on a local model')
    else:
        try:
            import langchain_openai  # noqa: F401
            llm_ready = True
        except Exception as error:
            notes.append(f'langchain-openai missing, LLM metrics disabled: {error}')
    return Availability(installed=True, llm_ready=llm_ready, version=version,
                        notes=tuple(notes))


class QuoteInChunkSimilarity:
    """The distance the offline context pair score with: how well the reference
    quote fits *inside* the retrieved chunk (rapidfuzz `partial_ratio`, the
    best local alignment of the shorter string within the longer, in [0, 1]).

    RAGAS's default is whole-string Levenshtein similarity, which is the right
    question when both sides are chunks of the same size — and the wrong one
    here, where the reference is a sentence-long evidence quote and the
    retrieved side is a whole chunk: a quote sitting verbatim inside a chunk
    three times its length scores ~0.3, under RAGAS's fixed 0.5 threshold, so
    both metrics reported a flat 0 on every run. RAGAS hands the chunk as
    `reference` and the quote as `response`; `partial_ratio` is symmetric in
    which is the shorter, so the order does not matter."""

    name = 'quote_in_chunk_similarity'

    def init(self, run_config) -> None:
        pass

    async def single_turn_ascore(self, sample, callbacks=None) -> float:
        from rapidfuzz import fuzz
        return fuzz.partial_ratio(sample.reference or '', sample.response or '') / 100.0

    # RAGAS's SingleTurnMetric surface, in case a caller uses the plain name.
    async def _single_turn_ascore(self, sample, callbacks=None) -> float:
        return await self.single_turn_ascore(sample, callbacks)


def offline_metrics():
    """RAGAS's two model-free context metrics, scoring by quote-in-chunk
    containment rather than whole-string distance (see QuoteInChunkSimilarity).
    Precision exposes `distance_measure` as a field; recall's setter only takes
    RAGAS's own enum of whole-string measures, so its field is set directly."""
    from ragas.metrics import (NonLLMContextPrecisionWithReference,
                               NonLLMContextRecall)
    similarity = QuoteInChunkSimilarity()
    precision = NonLLMContextPrecisionWithReference(distance_measure=similarity)
    recall = NonLLMContextRecall()
    recall._distance_measure = similarity
    return [precision, recall]


class _LabEmbeddings:
    """LangChain's Embeddings surface over the lab's embedder, so RAGAS metrics
    that need vectors use the same representation the retrieval under test used
    — and no external embedding API is called."""

    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.embedder.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed([text])[0].tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _samples(pairs, ragas_mod, include_answers: bool,
             reference_texts: dict | None = None):
    """(question, outcome) → RAGAS samples, skipping what cannot be scored."""
    sample_cls = ragas_mod.SingleTurnSample
    samples, skipped = [], 0
    for question, outcome in pairs:
        contexts = [c.text for c in outcome.contexts]
        references = (reference_texts or {}).get(
            question['groundtruth_question_id']) or [
                ev['text']
                for relevant in question.get('relevant_corpus_documents') or []
                for ev in relevant.get('evidence') or []]
        # Non-LLM context metrics compare retrieved text to reference text; with
        # either side empty the score is undefined, not zero.
        if not contexts or not references:
            skipped += 1
            continue
        payload = dict(user_input=question['question'],
                       retrieved_contexts=contexts,
                       reference_contexts=references,
                       reference=question['expected_answer'].get('text', ''))
        if include_answers:
            if not outcome.answer:
                skipped += 1
                continue
            payload['response'] = outcome.answer
        samples.append(sample_cls(**payload))
    return samples, skipped


def run(pairs, settings, embedder, mode: str = 'offline',
        sample_limit: int | None = None,
        reference_texts: dict | None = None,
        judge_model: str = '', progress=None, k: int = 0) -> dict:
    """Score a run with RAGAS. `pairs` is [(ground-truth question, Outcome)];
    `judge_model` is separate from the answerer, since a model grading its own
    output is not evidence. `k` estimates judge calls (context precision is one
    per retrieved chunk). Returns means per metric plus notes; never raises."""
    import warnings

    # `decision` stays None on every path that cannot measure all four, so a
    # partially-measured composite never outranks a fully-measured one.
    report: dict = {'mode': mode, 'metrics': {}, 'n_samples': 0, 'skipped': 0,
                    'decision': None,
                    'decision_spread': decision_spread([]),
                    'decision_metrics': list(DECISION_METRICS), 'notes': []}
    status = availability(settings)
    if not status.installed:
        report['notes'] = list(status.notes) + [f'install with: {INSTALL_HINT}']
        return report
    if mode == 'llm' and not status.llm_ready:
        report['notes'] = list(status.notes) or ['LLM metrics unavailable']
        return report

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            import ragas
            from ragas import EvaluationDataset, evaluate
            from ragas.run_config import RunConfig
        except Exception as error:
            report['notes'].append(f'ragas import failed: {error}')
            return report

        # Answer-side metrics only make sense when something was generated.
        include_answers = mode == 'llm'
        if sample_limit:
            pairs = list(pairs)[:sample_limit]
        samples, skipped = _samples(pairs, ragas, include_answers, reference_texts)
        report['skipped'] = skipped
        if not samples:
            report['notes'].append('nothing scoreable: no question produced both '
                                   'contexts and reference quotes')
            return report

        metrics = offline_metrics()
        if mode == 'llm':
            try:
                metrics += _llm_metrics(settings, embedder, judge_model)
                # Backend and model both, since the slug alone doesn't say
                # whether it ran locally, and a decision score is comparable
                # only within one judge.
                report['judge'] = {
                    'provider': getattr(settings, 'provider', ''),
                    'model': judge_model or settings.llm_model}
                report['notes'].append(
                    f"judged by {report['judge']['model']} via "
                    f"{report['judge']['provider']}")
            except Exception as error:
                report['notes'].append(f'LLM metrics unavailable: {error}')
        load = judge_load(settings)
        report['judge_load'] = dict(load)
        watch = None
        if mode == 'llm':
            report['expected_judge_calls'] = expected_judge_calls(len(samples), k)
            watch = JudgeWatch(report['expected_judge_calls'], progress)
        try:
            result = evaluate(EvaluationDataset(samples=samples), metrics=metrics,
                              show_progress=False,
                              callbacks=[watch] if watch else None,
                              run_config=RunConfig(**load))
        except Exception as error:
            report['notes'].append(f'ragas evaluate failed: {error}')
            return report

    if watch:
        report['judge_calls'] = watch.calls
    report['n_samples'] = len(samples)
    report['metrics'] = _means(result)
    report['decision'] = decision_score(report['metrics'])
    report['decision_spread'] = decision_spread(
        list(getattr(result, 'scores', []) or []))
    if report['decision'] is None:
        absent = [name for name in DECISION_METRICS
                  if name not in report['metrics']]
        report['notes'].append(
            'no decision score: this run did not measure '
            + ', '.join(absent)
            + ' — only a judged run with an answerer can be ranked')
    if mode == 'offline':
        report['notes'].append(
            'offline RAGAS context metrics check whether verbatim evidence quotes '
            'sit inside the retrieved chunks — deterministic, retrieval only, '
            'and blind to how much else a chunk carries')
    report['ragas_version'] = getattr(ragas, '__version__', '?')
    report['notes'].extend(status.notes)
    return report


def _llm_metrics(settings, embedder, judge_model: str = ''):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, FactualCorrectness, LLMContextRecall,
                               LLMContextPrecisionWithReference, ResponseRelevancy)

    from raglab.llm_backends.chat_model_factory import judge_llm

    # Through the lab's own seam so the judge follows RAGLAB_LLM like every
    # other stage, rather than always going out to a paid API.
    judge = LangchainLLMWrapper(judge_llm(settings, judge_model))
    vectors = LangchainEmbeddingsWrapper(_LabEmbeddings(embedder))
    return [Faithfulness(llm=judge),
            ResponseRelevancy(llm=judge, embeddings=vectors),
            FactualCorrectness(llm=judge),
            LLMContextPrecisionWithReference(llm=judge),
            LLMContextRecall(llm=judge)]


def _means(result) -> dict:
    """Average RAGAS's per-sample scores. `.scores` is a list of dicts in every
    0.x release; `.to_pandas()` is not used, to keep pandas out of the lab."""
    rows = list(getattr(result, 'scores', []) or [])
    if not rows:
        return {}
    keys = {key for row in rows for key in row}
    out = {}
    for key in sorted(keys):
        values = [row[key] for row in rows
                  if isinstance(row.get(key), (int, float))
                  and row[key] == row[key]]           # drop NaN
        if values:
            out[key] = round(sum(values) / len(values), 4)
    return out
