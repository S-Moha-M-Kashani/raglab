"""Run one configuration over the ground-truth set and persist the result.

Build (or reuse) the index, retrieve and optionally answer each question, score
deterministically, optionally score with RAGAS, then write a JSON run file.
"""
import inspect
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace

from . import (agent, corpus, datasets, embedding, metrics, models, pipeline,
               ragas_eval)
from .config import (BALANCES, DIFFICULTIES, RUNS_DIR, LabConfig, LabSettings)
from .index import IndexRegistry, _lab_llm
from .llm import lab_chat
from .present import (chunks_by_session, evidence_spans, gold_available,
                      mark_gold, normalised_chunks, summary_rows)

KEY_FACTS_PROMPT = (
    'You check whether an answer contains specific facts. The answer is in '
    'Persian; the facts are in English. For each numbered fact reply on its own '
    'line with "<number>: yes" if the answer states or clearly implies it, '
    'otherwise "<number>: no". Output nothing else.')


def judge_key_facts(llm, model: str, question: dict, answer: str) -> float:
    """Share of the ground truth's key facts present in the answer. Facts are
    English, answers are Farsi, so lexical overlap can't score this — a judge
    that translates as it checks is the only option."""
    facts = question.get('key_facts') or []
    if not facts or not answer:
        return float('nan')
    listing = '\n'.join(f'{i + 1}. {fact}' for i, fact in enumerate(facts))
    try:
        turn = lab_chat(llm, [{'role': 'system', 'content': KEY_FACTS_PROMPT},
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
        return {'run_id': self.run_id, 'label': self.label,
                'started_at': self.started_at, 'seconds': self.seconds,
                'config': self.config, 'dataset': self.dataset,
                'summary': self.summary,
                'ragas': self.ragas.get('metrics', {}),
                'ragas_decision': self.ragas.get('decision'),
                'ragas_decision_stderr': (self.ragas.get('decision_spread')
                                          or {}).get('stderr'),
                # selection/judge travel too: a decision score is comparable only
                # against rows that scored the same questions with the same model.
                'selection': {k: v for k, v in self.selection.items()
                              if k != 'question_ids'},
                'judge': self.ragas.get('judge') or {},
                'n_questions': self.summary.get('n_questions', 0)}


def _stride(questions: list[dict], limit: int) -> list[dict]:
    """`limit` questions spread evenly across the list, not truncated: the set is
    grouped by type, so the first N would be one type and nothing else. Indexes
    across the whole list rather than `questions[::step][:limit]`, which drops a
    tail whenever the count is not a multiple of the limit."""
    total = len(questions)
    if limit >= total:
        return list(questions)
    return [questions[i * total // limit] for i in range(limit)]


def _balanced(questions: list[dict], limit: int) -> list[dict]:
    """`limit` questions with the difficulty bands as equal as the count allows —
    a plain stride hands medium roughly half of every sample, which would let one
    band's pipeline stand in for the four deciding metrics' score. The remainder
    when `limit` doesn't divide by three goes to the earlier bands in
    DIFFICULTIES order, deterministically. Each band is itself strided."""
    bands = [[q for q in questions if q['difficulty'] == name]
             for name in DIFFICULTIES]
    bands += [[q for q in questions
               if q['difficulty'] not in set(DIFFICULTIES)]]
    quotas = _quotas(limit, [len(band) for band in bands])
    picked_ids = {q['id'] for band, quota in zip(bands, quotas)
                  for q in _stride(band, quota)}
    # In the ground truth's own order, not band by band, so two runs' rows stay
    # diffable line by line.
    return [q for q in questions if q['id'] in picked_ids]


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


def select_questions(ground_truth: dict, types: list[str] | None = None,
                     limit: int | None = None,
                     difficulty: list[str] | None = None,
                     balance: str = 'stride') -> list[dict]:
    """The questions one run is measured on. `balance='difficulty'` equalises the
    easy/medium/hard bands; 'stride' (the default, for reproducibility against
    `.runs/`) spreads across the set as it is."""
    # Validated unconditionally, before the limit check below, so a bad config
    # raises on every run rather than only the ones where a limit is applied.
    if balance not in BALANCES:
        raise ValueError(f'unknown balance {balance!r}; expected one of '
                         + ', '.join(repr(name) for name in BALANCES))
    questions = ground_truth['questions']
    if types:
        questions = [q for q in questions if q['type'] in set(types)]
    if difficulty:
        questions = [q for q in questions if q['difficulty'] in set(difficulty)]
    if not limit or limit >= len(questions):
        return list(questions)
    if balance == 'difficulty':
        return _balanced(questions, limit)
    return _stride(questions, limit)


def selection_note(questions: list[dict], limit: int | None,
                   balance: str) -> dict:
    """What a run was actually measured on. The question ids travel with the row,
    because two rows are comparable only if they scored the same questions, and
    nothing else on the row says which those were."""
    counts: dict[str, int] = {}
    for question in questions:
        counts[question['difficulty']] = counts.get(question['difficulty'], 0) + 1
    return {'balance': balance, 'limit': limit, 'n': len(questions),
            'by_difficulty': {name: counts.get(name, 0)
                              for name in DIFFICULTIES if counts.get(name)},
            'question_ids': [question['id'] for question in questions]}


def _question_note(done: int, questions: list[dict], difficulty: str) -> str:
    """"question 16/30 · hard" — the band says whether a slow phase is slow
    throughout or only on hard questions."""
    return f'question {done}/{len(questions)} · {difficulty}'


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
              gold_available: int | None = None) -> dict:
    """One question's retrieval trace, gold-marked, in the shape the Inspector
    renders a table from. Gold is the ground truth's verdict on a candidate,
    kept apart from `kept` (the pipeline's own decision), so a gold chunk that
    was dropped is visible rather than hidden. `gold_available` is a property of
    the index the caller may not hold; without one the view shows no denominator
    rather than inventing one."""
    quotes = [ev['quote'] for ev in question.get('evidence', [])]
    candidates = trace.get('candidates', [])
    for candidate, gold in zip(candidates,
                               mark_gold([c['text'] for c in candidates],
                                         quotes)):
        candidate['gold'] = gold
        # Computed for every candidate, not only gold ones, so a verbatim quote
        # can never sit in a row that was not marked gold.
        candidate['gold_spans'] = evidence_spans(candidate['text'], quotes)
    return {'question_id': question['id'],
            'question_fa': question['question_fa'],
            'question_en': question.get('question_en', ''),
            'type': question['type'], 'difficulty': question['difficulty'],
            'answerable': bool(question.get('answerable')),
            'gold_available': gold_available,
            'trace': trace}


def run_retrieval(registry: IndexRegistry, ground_truth: dict, cfg: LabConfig,
                  settings: LabSettings, *, types: list[str] | None = None,
                  limit: int | None = None, difficulty: list[str] | None = None,
                  balance: str = 'stride', progress=None,
                  cancelled=None) -> dict:
    """Retrieval only, over the questions an experiment selected: build (or reuse)
    the index, retrieve with a full per-step trace, mark gold — and stop. Nothing
    is answered, scored or written to `.runs/`. Takes the same selection
    arguments as `run_eval`, deliberately, so what is shown is what the numbers
    were about."""
    problems = cfg.validate() + models.provider_problems(cfg, settings)
    if problems:
        raise ValueError('; '.join(problems))
    report = _reporter(progress)
    check_cancelled = cancelled or (lambda: None)

    check_cancelled()
    index = registry.get(cfg.index, progress=lambda stage, f: report(stage, f * 0.4))
    questions = select_questions(ground_truth, types, limit, difficulty, balance)
    query_date = ground_truth['meta'].get('query_date', '2026-07-28')
    llm = _lab_llm(settings)
    roles = models.resolve(cfg, settings)

    # Normalised once: `gold_available` counts gold over every chunk in the
    # index, and per-question would re-tokenise the corpus needlessly.
    norm_chunks = normalised_chunks(index)

    rows = []
    for i, question in enumerate(questions):
        check_cancelled()
        if agent.owns_retrieval(cfg.agent.scope):
            # Narrowed to the retrieval half of the scope and the answerer
            # forced off: this route retrieves and stops, and the drafting half
            # of `full` is an answering stage.
            trace = {}
            _outcome = agent.run(
                index,
                replace(cfg, agent=replace(cfg.agent, scope='retrieve'),
                        generation=replace(cfg.generation, answerer='none')),
                question['question_fa'],
                question.get('query_date', query_date), llm=llm, models=roles,
                trace=trace)
        else:
            _outcome, trace = pipeline.retrieve_traced(
                index, cfg.retrieval, question['question_fa'],
                question.get('query_date', query_date), llm=llm, models=roles)
        quotes = [ev['quote'] for ev in question.get('evidence', [])]
        rows.append(trace_row(
            question, trace,
            gold_available=gold_available(index, quotes, norm_chunks)))
        report('retrieving', 0.4 + 0.6 * (i + 1) / len(questions),
               _question_note(i + 1, questions, question['difficulty']))
    report('done', 1.0, 'done')
    return {'selection': selection_note(questions, limit, balance),
            'dataset': cfg.index.dataset or datasets.BUILTIN,
            'index': {'collection': index.stats.collection,
                      'chunks': index.stats.chunks,
                      'reused': index.stats.reused},
            'config': cfg.to_dict(),
            'models': roles.as_dict(),
            # Reported here because this run built its index implicitly, so
            # there is no index job the Inspector could read the chunks from.
            'chunks_by_session': chunks_by_session(index),
            'summaries': summary_rows(index),
            'questions': rows}


def run_eval(registry: IndexRegistry, ground_truth: dict, cfg: LabConfig,
             settings: LabSettings, *, types: list[str] | None = None,
             limit: int | None = None, difficulty: list[str] | None = None,
             balance: str = 'stride',
             ragas_mode: str = 'offline', ragas_limit: int | None = None,
             workers: int = 1, trace: bool = False,
             progress=None, cancelled=None) -> RunResult:
    """`trace=True` records each question's retrieval trace via `retrieve_traced`,
    which fills a dict the plain path never reads and returns the identical
    `Outcome` — so no score can move because tracing was asked for."""
    problems = cfg.validate() + models.provider_problems(cfg, settings)
    if problems:
        raise ValueError('; '.join(problems))
    started = time.time()
    # One clock read for both stamps, so start and finish cannot end up the same.
    clock = time.localtime(started)
    started_at = time.strftime('%Y-%m-%d %H:%M:%S', clock)
    run_id = time.strftime('%Y%m%d-%H%M%S', clock) + '-' + cfg.index.fingerprint()[:6]
    report = _reporter(progress)
    check_cancelled = cancelled or (lambda: None)

    check_cancelled()
    index = registry.get(cfg.index, progress=lambda stage, f: report(stage, f * 0.4))
    check_cancelled()
    questions = select_questions(ground_truth, types, limit, difficulty, balance)
    selection = selection_note(questions, limit, balance)
    query_date = ground_truth['meta'].get('query_date', '2026-07-28')
    llm = _lab_llm(settings)
    roles = models.resolve(cfg, settings)
    # Needed only by the traced path; normalised once rather than per question.
    norm_chunks = normalised_chunks(index) if trace else []
    notes = list(index.stats.notes)
    # Which model ran which stage, so two leaderboard rows are comparable.
    notes.append(models.note_for(cfg, settings))
    if cfg.agent.scope:
        notes.append(agent.note_for(cfg.agent))
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

    def handle(question: dict):
        check_cancelled()
        recorded = None
        asked = question['question_fa']
        when = question.get('query_date', query_date)
        if cfg.agent.scope:
            # The agent answers as well as retrieves — under `retrieve` it calls
            # `pipeline.answer` itself — so `pipeline.answer` must not run again
            # below, or the row would describe neither call.
            tr = {} if trace else None
            outcome = agent.run(index, cfg, asked, when, llm=llm, models=roles,
                                trace=tr)
        elif trace:
            outcome, tr = pipeline.retrieve_traced(
                index, cfg.retrieval, asked, when, llm=llm, models=roles)
        else:
            outcome = pipeline.retrieve(index, cfg.retrieval, asked, when,
                                        llm=llm, models=roles)
        if trace:
            quotes = [ev['quote'] for ev in question.get('evidence', [])]
            recorded = trace_row(
                question, tr,
                gold_available=gold_available(index, quotes, norm_chunks))
        check_cancelled()
        if not cfg.agent.scope:
            outcome = pipeline.answer(outcome, cfg.generation, llm=llm,
                                      models=roles)
        check_cancelled()
        row = metrics.score_question(question, outcome, cfg.retrieval.k)
        if (cfg.generation.key_facts_judge and outcome.answer
                and settings.llm_ready and question.get('answerable')):
            check_cancelled()
            row['key_fact_coverage'] = judge_key_facts(llm, roles.judge, question,
                                                       outcome.answer)
        return question, outcome, row, recorded

    pairs, rows, traces = [], [], []
    results: list = []
    if workers > 1:
        # Progress reported as each question lands, not after all of them.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(handle, q): i for i, q in enumerate(questions)}
            done_count = 0
            slots: list = [None] * len(questions)
            for future in as_completed(futures):
                landed = futures[future]
                slots[landed] = future.result()
                done_count += 1
                report('scoring', 0.4 + 0.5 * done_count / len(questions),
                       _question_note(done_count, questions,
                                      questions[landed]['difficulty']))
        results = [row for row in slots if row is not None]
    else:
        for i, question in enumerate(questions):
            results.append(handle(question))
            report('scoring', 0.4 + 0.5 * (i + 1) / len(questions),
                   _question_note(i + 1, questions, question['difficulty']))
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
        sessions = corpus.sessions_by_id(registry.corpus_for(cfg.index.dataset))
        references = {q['id']: corpus.evidence_texts(sessions, q) for q in questions}
        ragas_report = ragas_eval.run(pairs, settings, index.embedder,
                                      mode=ragas_mode, sample_limit=ragas_limit,
                                      reference_texts=references,
                                      judge_model=roles.ragas,
                                      progress=report, k=cfg.retrieval.k)
    report('done', 1.0, 'done')
    result = RunResult(run_id=run_id, label=cfg.label or cfg.index.chunker,
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
                       chunks_by_session=(chunks_by_session(index)
                                          if trace else []),
                       summaries=summary_rows(index) if trace else [])
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
        out.append({'run_id': data['run_id'], 'label': data.get('label', ''),
                    'started_at': data.get('started_at', ''),
                    # Absent means the built-in diary: the only corpus there
                    # was before a second one existed.
                    'dataset': data.get('dataset') or datasets.BUILTIN,
                    'seconds': data.get('seconds', 0), 'config': data.get('config'),
                    'summary': data.get('summary', {}),
                    'ragas': ragas.get('metrics', {}),
                    # Absent or None both mean "not ranked", which is the truth.
                    'ragas_decision': ragas.get('decision'),
                    # None rather than 0: `± 0` would claim more precision than
                    # the rows that actually measured a spread.
                    'ragas_decision_stderr': (ragas.get('decision_spread')
                                              or {}).get('stderr'),
                    # Question ids stay in, unlike `brief()`: grouping needs them
                    # and this list is read by code, never rendered.
                    'selection': data.get('selection') or {},
                    'judge': ragas.get('judge') or {},
                    'n_questions': (data.get('summary') or {}).get('n_questions', 0)})
    return out


def load_run(run_id: str) -> dict | None:
    path = RUNS_DIR / f'{run_id}.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))
