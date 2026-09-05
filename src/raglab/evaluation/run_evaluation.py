"""Run one configuration over the ground-truth set and persist the result.

Build (or reuse) the index, retrieve and optionally answer each question, score
deterministically, optionally score with RAGAS, then write a JSON run file.
"""
import inspect
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from raglab.corpora import corpus_reading as corpus
from raglab.corpora import dataset_import_contract as datasets
from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.evaluation import deterministic_metrics as metrics
from raglab.llm_backends import model_role_catalogue as models
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.evaluation import ragas_judged_metrics as ragas_eval
from raglab.configuration import split_plan
from raglab.configuration.lab_config import (
    RUNS_DIR,
    LabConfig,
    LabSettings)
from raglab.rag_components.indexing.index_builder_registry import IndexRegistry
from raglab.llm_backends.chat_model_factory import lab_chat, lab_llm
# Aliased: `RunResult.chunks_by_session` is a field of the same name (kept for
# the run files that serialise it), and the two must not shadow each other.
from raglab.dashboard.service_presentation import (
    chunks_by_session as chunks_by_session_rows,
    evidence_spans,
    gold_available,
    mark_gold,
    normalised_chunks,
    summary_rows)

DERIVED_FACTS_PROMPT = (
    'You check whether an answer contains specific facts. The answer and the '
    'facts may be in different languages; translate as you check. For each '
    'numbered fact reply on its own line with "<number>: yes" if the answer '
    'states or clearly implies it, otherwise "<number>: no". Output nothing '
    'else.')


def judge_derived_facts(llm, model: str, question: dict, answer: str) -> float:
    """Share of the ground truth's derived_facts present in the answer. The
    facts and the answer may be in different languages, so lexical overlap
    can't score this — a judge that translates as it checks is the only
    option."""
    facts = question['expected_answer'].get('derived_facts') or []
    if not facts or not answer:
        return float('nan')
    listing = '\n'.join(f'{i + 1}. {fact["fact"]}' for i, fact in enumerate(facts))
    try:
        turn = lab_chat(llm, [{'role': 'system', 'content': DERIVED_FACTS_PROMPT},
                              {'role': 'user',
                               'content': f'Answer:\n{answer}\n\nFacts:\n{listing}'}],
                        model)
        text = turn.content or ''
    except Exception:
        return float('nan')
    verdicts = {}
    for line in text.splitlines():
        match = re.match(r'\s*(\d+)\s*[:.\-]\s*(yes|no|true|false)', line.strip(),
                         re.IGNORECASE)
        if match:
            verdicts[int(match.group(1))] = match.group(2).lower() in ('yes', 'true')
    if not verdicts:
        return float('nan')
    return sum(1 for i in range(1, len(facts) + 1) if verdicts.get(i)) / len(facts)


def json_safe(value):
    """NaN → None, recursively. Metrics use NaN internally for "undefined for
    this question"; NaN is not valid JSON, so this converts at the boundary."""
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _row_shape(*, run_id: str, label: str, started_at: str, seconds: float,
              config: dict, dataset: str, summary: dict, ragas: dict,
              selection: dict, n_questions: int) -> dict:
    """The row `RunResult.brief()` and `list_runs` both build; `selection`/`n_questions` are taken pre-computed since the two callers disagree about what each strips (`brief()` drops question ids, `list_runs` keeps them)."""
    return {'run_id': run_id, 'label': label, 'started_at': started_at,
            'seconds': seconds, 'config': config, 'dataset': dataset,
            'summary': summary,
            'ragas': ragas.get('metrics', {}),
            # Absent or None both mean "not ranked", which is the truth.
            'ragas_decision': ragas.get('decision'),
            # None rather than 0: `± 0` would claim more precision than the
            # rows that actually measured a spread.
            'ragas_decision_stderr': (ragas.get('decision_spread')
                                      or {}).get('stderr'),
            'selection': selection,
            'judge': ragas.get('judge') or {},
            'n_questions': n_questions}


@dataclass
class RunResult:
    run_id: str
    label: str
    config: dict
    index: dict
    summary: dict
    # Resolved rather than left blank: the leaderboard groups by dataset first.
    dataset: str = ''
    rows: list = field(default_factory=list)
    ragas: dict = field(default_factory=dict)
    seconds: float = 0.0
    started_at: str = ''
    notes: list = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    # traces, chunks_by_session and summaries are deliberately absent from
    # as_dict/save_run, so no per-question trace or chunk text reaches a run
    # file; they travel in the job's result and die with it.
    traces: list = field(default_factory=list)
    chunks_by_session: list = field(default_factory=list)
    summaries: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {'run_id': self.run_id, 'label': self.label, 'config': self.config,
                'dataset': self.dataset,
                'index': self.index, 'summary': self.summary, 'rows': self.rows,
                'ragas': self.ragas, 'seconds': self.seconds,
                'started_at': self.started_at, 'notes': self.notes,
                'selection': self.selection}

    def brief(self) -> dict:
        """Leaderboard row — no per-question detail."""
        return _row_shape(run_id=self.run_id, label=self.label,
                          started_at=self.started_at, seconds=self.seconds,
                          config=self.config, dataset=self.dataset,
                          summary=self.summary, ragas=self.ragas,
                          # selection/judge travel too: a decision score is
                          # comparable only against rows that scored the same
                          # questions with the same model.
                          selection={k: v for k, v in self.selection.items()
                                    if k != 'question_ids'},
                          n_questions=self.summary.get('n_questions', 0))


def _stride(questions: list[dict], limit: int) -> list[dict]:
    """`limit` questions spread evenly across the list, not truncated: the set is
    grouped by type, so the first N would be one type and nothing else. Indexes
    across the whole list rather than `questions[::step][:limit]`, which drops a
    tail whenever the count is not a multiple of the limit."""
    total = len(questions)
    if limit >= total:
        return list(questions)
    return [questions[i * total // limit] for i in range(limit)]


def _label_value(question: dict, label: str):
    return (question.get('question_metadata') or {}).get(label)


def _bands(questions: list[dict], label: str, fields: dict) -> list[str]:
    """The order to walk a label's values in, so a sample is deterministic and
    a remainder always lands the same way twice. A declared `values` list or
    `glossary` (its own key order) is preferred over the data, so the order is
    the dataset's own choice rather than an accident of which value appeared
    first among these particular rows."""
    declared = fields.get(label) or {}
    order = declared.get('values') or list((declared.get('glossary') or {}))
    if order:
        return order
    seen: list[str] = []
    for question in questions:
        value = _label_value(question, label)
        if value is not None and value not in seen:
            seen.append(value)
    return seen


def _balanced(questions: list[dict], limit: int, label: str,
             fields: dict) -> list[dict]:
    """`limit` questions with `label`'s own bands as equal as the count allows —
    a plain stride hands the most common band roughly half of every sample,
    which would let one band's pipeline stand in for the four deciding
    metrics' score. The remainder when `limit` doesn't divide evenly goes to
    the earlier bands in the label's own declared order, deterministically.
    Each band is itself strided."""
    order = _bands(questions, label, fields)
    bands = [[q for q in questions if _label_value(q, label) == name]
             for name in order]
    bands.append([q for q in questions if _label_value(q, label) not in order])
    quotas = _quotas(limit, [len(band) for band in bands])
    picked_ids = {q['groundtruth_question_id']
                 for band, quota in zip(bands, quotas)
                 for q in _stride(band, quota)}
    # In the ground truth's own order, not band by band, so two runs' rows stay
    # diffable line by line.
    return [q for q in questions if q['groundtruth_question_id'] in picked_ids]


def _quotas(limit: int, sizes: list[int]) -> list[int]:
    """How many questions to take from each band. A band smaller than its share
    offers the remainder to the others, so a limit the set can meet is always met."""
    quotas = [0] * len(sizes)
    remaining = min(limit, sum(sizes))
    while remaining > 0:
        hungry = [i for i, size in enumerate(sizes) if quotas[i] < size]
        if not hungry:
            break
        for i in hungry:
            if remaining == 0:
                break
            quotas[i] += 1
            remaining -= 1
    return quotas


def select_questions(ground_truth: dict, limit: int | None = None,
                     labels: dict[str, list[str]] | None = None,
                     balance: str = '') -> list[dict]:
    """The questions one run is measured on (D7). `labels` keeps only questions
    whose named `question_metadata` value is one of the given values — the
    panel's per-label switch-groups. `balance` names a question label to
    equalise the sample across that label's own bands; '' (the default, for
    reproducibility against `.runs/`) spreads across the set as it is."""
    fields = (ground_truth.get('groundtruth_dataset_metadata') or {}
             ).get('question_metadata_fields') or {}
    # Validated unconditionally, before the limit check below, so a bad config
    # raises on every run rather than only the ones where a limit is applied.
    if balance and balance not in fields:
        raise ValueError(f'unknown balance {balance!r}: no question label '
                         'named that is declared in question_metadata_fields')
    questions = ground_truth['groundtruth_dataset']
    for name, wanted in (labels or {}).items():
        wanted = set(wanted)
        questions = [q for q in questions if _label_value(q, name) in wanted]
    if not limit or limit >= len(questions):
        return list(questions)
    if balance:
        return _balanced(questions, limit, balance, fields)
    return _stride(questions, limit)


def selection_note(questions: list[dict], limit: int | None,
                   balance: str) -> dict:
    """What a run was actually measured on. The question ids travel with the row,
    because two rows are comparable only if they scored the same questions, and
    nothing else on the row says which those were. `by_<balance>` reports the
    counts of whichever label the run was balanced on; there is nothing to
    report by when a run was strided ('')."""
    note = {'balance': balance, 'limit': limit, 'n': len(questions),
            'question_ids': [question['groundtruth_question_id']
                             for question in questions]}
    if balance:
        counts: dict[str, int] = {}
        for question in questions:
            value = _label_value(question, balance)
            if value is not None:
                counts[value] = counts.get(value, 0) + 1
        note[f'by_{balance}'] = counts
    return note


def _question_note(done: int, questions: list[dict]) -> str:
    """"question 16/30" — a plain counter; no label is guaranteed to exist on
    every question, so nothing more specific can be shown here in general."""
    return f'question {done}/{len(questions)}'


def _reporter(progress):
    """Adapt a progress callback to `(stage, fraction, detail)` whether or not it
    accepts the detail. Arity is inspected once rather than caught per call, so a
    real error raised inside the callback is not swallowed as a `TypeError`."""
    if progress is None:
        return lambda stage, fraction, detail='': None
    try:
        params = inspect.signature(progress).parameters
        takes_detail = (len(params) >= 3
                        or any(p.kind is p.VAR_POSITIONAL or p.kind is p.VAR_KEYWORD
                               for p in params.values()))
    except (TypeError, ValueError):         # a builtin or a C callable
        takes_detail = False
    if takes_detail:
        return lambda stage, fraction, detail='': progress(stage, fraction, detail)
    return lambda stage, fraction, detail='': progress(stage, fraction)


def trace_row(question: dict, trace: dict,
              gold_present: int | None = None) -> dict:
    """One question's retrieval trace, gold-marked, in the shape the Inspector
    renders a table from. Gold is the ground truth's verdict on a candidate,
    kept apart from `kept` (the pipeline's own decision), so a gold chunk that
    was dropped is visible rather than hidden. `gold_present` is a property of
    the index the caller may not hold; without one the view shows no denominator
    rather than inventing one."""
    quotes = metrics.verbatim_quotes(question)
    candidates = trace.get('candidates', [])
    for candidate, gold in zip(candidates,
                               mark_gold([c['text'] for c in candidates],
                                         quotes)):
        candidate['gold'] = gold
        # Computed for every candidate, not only gold ones, so a verbatim quote
        # can never sit in a row that was not marked gold.
        candidate['gold_spans'] = evidence_spans(candidate['text'], quotes)
    return {'question_id': question['groundtruth_question_id'],
            'question': question['question'],
            'behavior': question['expected_answer']['behavior'],
            'gold_available': gold_present,
            'trace': trace}


def _gold_trace_row(question: dict, trace: dict, index, norm_chunks: list) -> dict:
    """`trace_row`, with gold availability counted from this question's own
    evidence quotes against the index — the three lines `run_retrieval` and
    `run_eval` both repeated identically."""
    quotes = metrics.verbatim_quotes(question)
    return trace_row(question, trace,
                     gold_present=gold_available(index, quotes, norm_chunks))


@dataclass(frozen=True)
class _RunSetup:
    """What `run_retrieval` and `run_eval` both need before their question loop, built once by `_prepare_run`; `started` is read right after validation and only `run_eval` derives its run id/started_at from it."""
    started: float
    report: Callable
    check_cancelled: Callable
    index: Any
    questions: list[dict]
    selection: dict
    query_date: str
    llm: Any
    roles: Any
    norm_chunks: list


def _query_date(ground_truth: dict) -> str:
    """The 'now' a relative time expression in a question resolves against —
    the ground truth's own `default_question_asked_at`, sliced to a plain
    date. There is no per-question override in the schema: one dataset, one
    default."""
    default = (ground_truth.get('groundtruth_dataset_metadata') or {}
              ).get('default_question_asked_at', '2026-07-28T00:00:00Z')
    return default[:10]


def _prepare_run(registry: IndexRegistry, ground_truth: dict, cfg: LabConfig,
                 settings: LabSettings, *, labels: dict[str, list[str]] | None,
                 limit: int | None, balance: str,
                 progress, cancelled, need_norm_chunks: bool,
                 recheck_after_index: bool) -> _RunSetup:
    """The setup `run_retrieval` and `run_eval` both open with; `need_norm_chunks`/`recheck_after_index` are the two points where the two callers differ. `started` is read here, right after validation and before the index build, because `models.provider_problems` can be a network round trip — `run_eval`'s run id and `started_at` must be timestamped after that call resolves, not before it, or a slow or unreachable backend would stretch the recorded start time backwards."""
    problems = cfg.validate() + models.provider_problems(cfg, settings)
    if problems:
        raise ValueError('; '.join(problems))
    started = time.time()
    report = _reporter(progress)
    check_cancelled = cancelled or (lambda: None)

    check_cancelled()
    index = registry.get(cfg.index,
                         progress=lambda stage, f: report(stage, f * 0.4))
    if recheck_after_index:
        check_cancelled()
    questions = select_questions(ground_truth, limit, labels, balance)
    selection = selection_note(questions, limit, balance)
    query_date = _query_date(ground_truth)
    llm = lab_llm(settings)
    roles = models.resolve(cfg, settings)
    # Normalised once: `gold_available` counts gold over every chunk in the
    # index, and per-question would re-tokenise the corpus needlessly. Needed
    # unconditionally by `run_retrieval`; only for a traced `run_eval`.
    norm_chunks = normalised_chunks(index) if need_norm_chunks else []
    return _RunSetup(started=started, report=report,
                     check_cancelled=check_cancelled, index=index,
                     questions=questions, selection=selection,
                     query_date=query_date, llm=llm, roles=roles,
                     norm_chunks=norm_chunks)


def run_retrieval(registry: IndexRegistry, ground_truth: dict, cfg: LabConfig,
                  settings: LabSettings, *,
                  labels: dict[str, list[str]] | None = None,
                  limit: int | None = None,
                  balance: str = '', progress=None,
                  cancelled=None) -> dict:
    """Retrieval only, over the questions an experiment selected: build (or reuse)
    the index, retrieve with a full per-step trace, mark gold — and stop. Nothing
    is answered, scored or written to `.runs/`. Takes the same selection
    arguments as `run_eval`, deliberately, so what is shown is what the numbers
    were about."""
    setup = _prepare_run(registry, ground_truth, cfg, settings, labels=labels,
                         limit=limit, balance=balance,
                         progress=progress, cancelled=cancelled,
                         need_norm_chunks=True, recheck_after_index=False)
    report, check_cancelled = setup.report, setup.check_cancelled
    index, questions = setup.index, setup.questions
    query_date, llm, roles = setup.query_date, setup.llm, setup.roles
    norm_chunks = setup.norm_chunks

    rows = []
    for i, question in enumerate(questions):
        check_cancelled()
        _outcome, trace = pipeline.retrieve_traced(
            index, cfg.retrieval, question['question'],
            query_date, llm=llm, models=roles)
        rows.append(_gold_trace_row(question, trace, index, norm_chunks))
        report('retrieving', 0.4 + 0.6 * (i + 1) / len(questions),
               _question_note(i + 1, questions))
    report('done', 1.0, 'done')
    return {'selection': setup.selection,
            'dataset': cfg.index.dataset or datasets.BUILTIN,
            'index': {'collection': index.stats.collection,
                      'chunks': index.stats.chunks,
                      'reused': index.stats.reused},
            'config': cfg.to_dict(),
            'models': roles.as_dict(),
            # Reported here because this run built its index implicitly, so
            # there is no index job the Inspector could read the chunks from.
            'chunks_by_session': chunks_by_session_rows(index),
            'summaries': summary_rows(index),
            'questions': rows}


def _assemble_notes(index, cfg: LabConfig, settings: LabSettings) -> list[str]:
    """The row's free-text notes: the index's own build notes, which model ran
    which stage (so two leaderboard rows are comparable), the embedder's
    language coverage, and — only when it would
    otherwise go unsaid — that an LLM stage fell back to the offline fake
    provider and so measured nothing."""
    notes = list(index.stats.notes)
    notes.append(models.note_for(cfg, settings))
    notes.append(embedding.language_note(
        cfg.index.embedder,
        embedding.resolve_model(cfg.index.embedder, settings,
                                cfg.index.embed_model)))
    if not settings.llm_ready and (cfg.generation.answerer == 'llm'
                                            or cfg.retrieval.reranker == 'llm'
                                            or cfg.retrieval.grader == 'llm'
                                            or cfg.retrieval.hyde):
        notes.append('no OPENROUTER_API_KEY: LLM stages fell back to the offline '
                     'fake provider, so their numbers are meaningless')
    # A real behaviour change (not a rename): quote recall only ever matches a
    # `fidelity: verbatim` evidence entry — a paraphrase or a computed fact
    # was never in the text, and a lexical match against it measures nothing.
    notes.append('quote recall is measured only over verbatim evidence; '
                 'paraphrase and computed entries are excluded from it')
    return notes


def _run_questions(questions: list[dict], handle: Callable, workers: int,
                   report: Callable) -> list[tuple]:
    """Runs `handle` over every question — threaded above one worker, serial
    otherwise — and returns the results in question order regardless of which
    way they finished, reporting progress as each one lands rather than after
    all of them."""
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(handle, q): i for i, q in enumerate(questions)}
            done_count = 0
            slots: list = [None] * len(questions)
            for future in as_completed(futures):
                landed = futures[future]
                slots[landed] = future.result()
                done_count += 1
                report('scoring', 0.4 + 0.5 * done_count / len(questions),
                       _question_note(done_count, questions))
        return [row for row in slots if row is not None]
    results = []
    for i, question in enumerate(questions):
        results.append(handle(question))
        report('scoring', 0.4 + 0.5 * (i + 1) / len(questions),
               _question_note(i + 1, questions))
    return results


def _build_result(*, run_id: str, cfg: LabConfig, index, summary: dict,
                  rows: list, ragas_report: dict, started: float,
                  started_at: str, notes: list, selection: dict, traces: list,
                  trace: bool) -> RunResult:
    """The `RunResult` `run_eval` ends its work with — one call rather than one
    seventeen-field literal, so the conductor reads as "build the result" and
    not as the result's own shape."""
    return RunResult(run_id=run_id,
                     label=cfg.label or split_plan.short(cfg.index.split_plan),
                     config=cfg.to_dict(),
                     dataset=cfg.index.dataset or datasets.BUILTIN,
                     index={'collection': index.stats.collection,
                            'chunks': index.stats.chunks,
                            'avg_chars': index.stats.avg_chars,
                            'p95_chars': index.stats.p95_chars,
                            'embed_dim': index.stats.embed_dim,
                            'build_seconds': index.stats.build_seconds,
                            'reused': index.stats.reused},
                     summary=summary, rows=rows, ragas=ragas_report,
                     seconds=round(time.time() - started, 2),
                     started_at=started_at, notes=notes,
                     selection=selection, traces=traces,
                     chunks_by_session=(chunks_by_session_rows(index)
                                        if trace else []),
                     summaries=summary_rows(index) if trace else [])


def run_eval(registry: IndexRegistry, ground_truth: dict, cfg: LabConfig,
             settings: LabSettings, *,
             labels: dict[str, list[str]] | None = None,
             limit: int | None = None,
             balance: str = '',
             ragas_mode: str = 'offline', ragas_limit: int | None = None,
             workers: int = 1, trace: bool = False,
             progress=None, cancelled=None) -> RunResult:
    """`trace=True` records each question's retrieval trace via `retrieve_traced`,
    which fills a dict the plain path never reads and returns the identical
    `Outcome` — so no score can move because tracing was asked for."""
    setup = _prepare_run(registry, ground_truth, cfg, settings, labels=labels,
                         limit=limit, balance=balance,
                         progress=progress, cancelled=cancelled,
                         need_norm_chunks=trace, recheck_after_index=True)
    started = setup.started
    report, check_cancelled = setup.report, setup.check_cancelled
    index, questions, selection = setup.index, setup.questions, setup.selection
    query_date, llm, roles = setup.query_date, setup.llm, setup.roles
    norm_chunks = setup.norm_chunks
    # One clock read for both stamps, so start and finish cannot end up the same.
    clock = time.localtime(started)
    started_at = time.strftime('%Y-%m-%d %H:%M:%S', clock)
    run_id = time.strftime('%Y%m%d-%H%M%S', clock) + '-' + cfg.index.fingerprint()[:6]
    notes = _assemble_notes(index, cfg, settings)

    def handle(question: dict):
        check_cancelled()
        recorded = None
        asked = question['question']
        behavior = question['expected_answer']['behavior']
        # 'correct_premise' must answer *and* contradict the false premise, so
        # it is answerable exactly like 'answer' — only 'abstain' is not.
        answerable = behavior != 'abstain'
        if trace:
            outcome, tr = pipeline.retrieve_traced(
                index, cfg.retrieval, asked, query_date, llm=llm, models=roles)
        else:
            outcome = pipeline.retrieve(index, cfg.retrieval, asked, query_date,
                                        llm=llm, models=roles)
        if trace:
            recorded = _gold_trace_row(question, tr, index, norm_chunks)
        check_cancelled()
        outcome = pipeline.answer(outcome, cfg.generation, llm=llm,
                                  models=roles)
        check_cancelled()
        row = metrics.score_question(question, outcome, cfg.retrieval.k)
        if (cfg.generation.fact_judge and outcome.answer
                and settings.llm_ready and answerable):
            check_cancelled()
            row['fact_coverage'] = judge_derived_facts(llm, roles.judge, question,
                                                       outcome.answer)
        return question, outcome, row, recorded

    pairs, rows, traces = [], [], []
    results = _run_questions(questions, handle, workers, report)
    for question, outcome, row, recorded in results:
        pairs.append((question, outcome))
        rows.append(json_safe(row))
        if recorded is not None:
            traces.append(recorded)
    report('ragas', 0.92, 'judging')

    check_cancelled()
    summary = metrics.aggregate(rows)
    ragas_report: dict = {}
    if ragas_mode != 'off':
        documents = corpus.documents_by_id(registry.corpus_for(cfg.index.dataset))
        references = {q['groundtruth_question_id']: corpus.evidence_texts(documents, q)
                     for q in questions}
        ragas_report = ragas_eval.run(pairs, settings, index.embedder,
                                      mode=ragas_mode, sample_limit=ragas_limit,
                                      reference_texts=references,
                                      judge_model=roles.ragas,
                                      progress=report, k=cfg.retrieval.k)
    report('done', 1.0, 'done')
    result = _build_result(run_id=run_id, cfg=cfg, index=index, summary=summary,
                           rows=rows, ragas_report=ragas_report, started=started,
                           started_at=started_at, notes=notes, selection=selection,
                           traces=traces, trace=trace)
    save_run(result)
    return result


def save_run(result: RunResult) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f'{result.run_id}.json'
    # allow_nan=False: a NaN here would write a file strict JSON parsers reject.
    path.write_text(json.dumps(json_safe(result.as_dict()), ensure_ascii=False,
                               indent=1, allow_nan=False), encoding='utf-8')


def count_runs() -> int:
    """How many run files exist, whether or not they were listed — so a bounded
    listing can say what it left out."""
    if not RUNS_DIR.exists():
        return 0
    return sum(1 for _ in RUNS_DIR.glob('*.json'))


def list_runs(limit: int = 50) -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for path in sorted(RUNS_DIR.glob('*.json'), reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if 'run_id' not in data:
            continue        # not a run: never assume a directory holds only ours
        ragas = data.get('ragas') or {}
        out.append(_row_shape(
            run_id=data['run_id'], label=data.get('label', ''),
            started_at=data.get('started_at', ''),
            seconds=data.get('seconds', 0), config=data.get('config'),
            # Absent means the built-in diary: the only corpus there was
            # before a second one existed.
            dataset=data.get('dataset') or datasets.BUILTIN,
            summary=data.get('summary', {}), ragas=ragas,
            # Question ids stay in, unlike `brief()`: grouping needs them and
            # this list is read by code, never rendered.
            selection=data.get('selection') or {},
            n_questions=(data.get('summary') or {}).get('n_questions', 0)))
    return out


# How many per-question rows one read returns. The rows are the expensive layer
# of a run file, and a whole 167-question failure set would fill the context
# window of the one reader that asks for them with the tail of a list nobody
# reads past.
MAX_QUESTION_ROWS = 25


def load_runs(limit: int = 500) -> list[dict]:
    """Run files as they were saved, newest first, without their per-question rows.

    `list_runs` reads the same files and flattens them for an API response —
    `_row_shape` reduces `ragas` to its metrics dict alone. The board and the
    digest both project from a run file rather than from that flattening, so
    they need `ragas` as the block it is on disk, with the decision, its spread
    and the judge still inside it.

    `rows` is dropped because neither reader looks at it and it is the one part
    of a run file that is large: keeping five hundred runs' worth of
    per-question rows in memory to build a table that shows none of them is
    what `list_runs` was already careful not to do. Nothing else is reshaped.
    """
    if not RUNS_DIR.exists():
        return []
    out = []
    for path in sorted(RUNS_DIR.glob('*.json'), reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if 'run_id' not in data:
            continue        # not a run: never assume a directory holds only ours
        data.pop('rows', None)
        out.append(data)
    return out


def load_run(run_id: str) -> dict | None:
    path = RUNS_DIR / f'{run_id}.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def question_rows(experiment_id: str, only: str = 'missed',
                  limit: int = MAX_QUESTION_ROWS, db_path=None) -> dict:
    """The per-question rows of one evaluation, filtered, named and capped.

    `only='missed'` is the failure set worth reading: every question whose gold
    evidence retrieval did not fully find inside k. The rows themselves carry a
    question *id* and no text, so the dataset's ground truth is joined in — an
    id alone cannot tell a reader what the retriever was looking for.
    """
    run = load_run(experiment_id) or {}
    if not run:
        return {'experiment_id': experiment_id, 'rows': [], 'n_questions': 0,
                'n_matched': 0,
                'reason': 'no run file for this experiment — per-question rows '
                          'live in .runs/ only, and this experiment has a '
                          'ledger row alone (an index build, a retrieval, or a '
                          'run older than the ledger)'}
    dataset = run.get('dataset') or datasets.BUILTIN
    rows = run.get('rows') or []
    matched = [row for row in rows if _matches(row, only)]
    asked = _questions(dataset)
    return {
        'experiment_id': experiment_id, 'dataset': dataset,
        'filter': only,
        'n_questions': len(rows), 'n_matched': len(matched),
        'k': (run.get('config') or {}).get('retrieval', {}).get('k'),
        'rows': [_question_row(row, asked.get(row.get('id'), {}))
                 for row in matched[:limit]],
    }


def _matches(row: dict, only: str) -> bool:
    """Which rows one filter keeps. Unknown filter names keep everything rather
    than silently returning an empty failure set, which would read as a run
    with nothing wrong with it."""
    recall = row.get('recall')
    if only == 'missed':
        # Below 1.0 is "not all of the gold evidence was inside k"; a missing
        # recall is not a pass, so it stays in.
        return recall is None or recall < 1.0
    if only == 'abstained':
        return bool(row.get('abstained'))
    return True


def _question_row(row: dict, asked: dict) -> dict:
    """One per-question row, joined to what the question actually was."""
    question_metadata = asked.get('question_metadata') or {}
    return {
        'id': row.get('id', ''),
        'question': asked.get('question', ''),
        'type': question_metadata.get('question_type', ''),
        'difficulty': question_metadata.get('difficulty', ''),
        'behavior': row.get('behavior')
                   or (asked.get('expected_answer') or {}).get('behavior', ''),
        'recall': row.get('recall'), 'precision': row.get('precision'),
        'mrr': row.get('mrr'), 'hit': row.get('hit'),
        'n_contexts': row.get('n_contexts'),
        'retrieved_sessions': list(row.get('retrieved_sessions') or []),
        'expected_sessions': [str(relevant.get('corpus_document_id', ''))
                              for relevant
                              in asked.get('relevant_corpus_documents') or []],
        'abstained': bool(row.get('abstained')),
        'false_abstention': bool(row.get('false_abstention')),
    }


def _questions(dataset: str) -> dict:
    """The dataset's ground-truth questions by id, or nothing when the corpus
    is no longer loadable — an imported dataset can be gone while the runs it
    produced remain, and a digest of those runs is still worth reading."""
    try:
        return {q.get('groundtruth_question_id'): q
                for q in datasets.load(dataset)[1].get('groundtruth_dataset') or []}
    except (ValueError, OSError, KeyError):
        return {}
