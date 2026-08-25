"""Three read-only windows onto what this lab has already measured.

The widget is a sealed leaf — it imports no evaluation module, no store and no
pipeline — so the two durable records reach it the way the OpenRouter key does:
injected. `panel_server` wires in three functions at app construction, because
it is the one module allowed to import both sides: `leaderboard.board_rows` and
`leaderboard.experiment` — the board's own rows and its projection resolved for
one id — and `run_evaluation.question_rows`. Nothing was written for this
widget: it is handed what the board is built from and what the lab's own
`/api/experiments/{id}` answers with, minus the evidence.

What the tools do with what they are handed is render it, nothing more. No mean
is computed here, no row is ranked, no blank is filled: a decision arrives with
its own error or reads as unjudged, and a widget with no reader wired in says so
rather than answering from an empty list — "no experiments" and "I cannot see
the experiments" are different sentences.
"""
from langchain_core.tools import tool

# The reader, injected. None until a panel wires one in — the `__main__`
# harness and any test that does not want records both run without one.
_READER = None

# How many experiments one listing offers, and how many per-question rows one
# call returns. The model names a filter, never a page size: a call that could
# ask for every row of a 167-question run would spend the context window on the
# tail of a list nobody reads.
MAX_LISTED = 20
# How deep the listing reads before a corpus filter is applied and the cap bites.
# The board's own default: a dataset filter that matched nothing inside it is a
# corpus with no recent experiments, which is an answer.
SCAN = 500
MAX_QUESTION_ROWS = 25

_UNWIRED = ('Experiment records are not available: this widget was started '
            'without a reader for the lab\'s ledger and .runs/ files. Say so '
            'rather than guessing what the experiments contain.')


def set_experiment_reader(reader) -> None:
    """Wire the records in, or pass None to unwire them."""
    global _READER
    _READER = reader


def _number(value, digits: int = 3) -> str:
    """A number, or an em dash — never a zero standing in for a missing one."""
    if not isinstance(value, (int, float)):
        return '—'
    return f'{float(value):.{digits}f}'


def _pipeline(row: dict) -> str:
    """The board's pipeline fragments as one sentence. A board row carries them
    as fragments because the table colours each by its step; a line of text has
    no colours, so they are joined here rather than upstream."""
    return ' | '.join(fragment['text'] for fragment in row.get('pipeline') or [])


def _decision(row: dict) -> str:
    """The four-metric mean beside its own error, or the reason there is none.

    Both or neither: `decision_score()` is never shown without
    `decision_spread()`, and an unjudged run reads as unjudged rather than as
    0.000, which would sort it below every run that measured something."""
    score = row.get('decision')
    if not isinstance(score, (int, float)):
        return 'unjudged'
    stderr = row.get('decision_stderr')
    spread = _number(stderr) if isinstance(stderr, (int, float)) else 'unknown'
    return f'decision {_number(score)} ± {spread}'


@tool
def list_experiments(dataset: str = '') -> str:
    """Recorded experiments, newest first; the model-facing prompt is
    fixtures/prompts/widget_tools.yaml's entry."""
    if _READER is None:
        return _UNWIRED
    # The board's own rows, filtered and capped here rather than by a reader
    # written for this tool: one corpus if one was named, newest first the way
    # the ledger is read, and never more than a model should be handed at once.
    rows = _READER.board_rows(limit=SCAN)
    if dataset:
        rows = [row for row in rows if row.get('dataset') == dataset]
    rows = rows[:MAX_LISTED]
    if not rows:
        said = f" for dataset '{dataset}'" if dataset else ''
        return (f'No experiments are recorded{said}. The records are the '
                'ledger (every job) and .runs/ (every evaluation); both are '
                'empty here.')
    head = (f'{len(rows)} experiment(s), newest first'
            + (f", dataset '{dataset}'" if dataset else ', every dataset'))
    lines = [head]
    for row in rows:
        state = row.get('state') or '?'
        if row.get('error'):
            state += f" ({' '.join(row['error'].split())[:80]})"
        judge_model = (row.get('judge') or {}).get('model')
        judge = f'judge {judge_model}' if judge_model else 'no judge'
        lines.append(
            f"{row.get('experiment_id', '')}  {row.get('started_at', '')}  "
            f"{row.get('dataset', '')}  {row.get('kind', '')}/{state}  "
            f"{_decision(row)}  {judge}  "
            f"{row.get('n_questions', 0)}q  "
            f"backend {row.get('provider') or 'unrecorded'}  "
            f"{_pipeline(row) or 'no pipeline recorded'}  "
            f"[{row.get('source', '')}]")
    lines.append('Ask read_experiment for one id; read_experiment_questions '
                 'for the questions it got wrong.')
    return '\n'.join(lines)


def _knobs(knobs: dict, inert: dict | None = None) -> list[str]:
    """The config as one line per pipeline step: the run's own read knobs,
    with any knob the run never read shown as `name=none` rather than the
    value still sitting in the recorded config — a value that number would be
    a lie about what produced the row (`inert`, dotted path -> reason, is
    absent from a found dict no injected reader has touched yet, in which
    case every knob renders exactly as it always has)."""
    inert = inert or {}
    out = []
    for step in ('index', 'retrieval', 'generation', 'agent'):
        values = knobs.get(step) or {}
        parts = []
        for name, value in values.items():
            if f'{step}.{name}' in inert:
                parts.append(f'{name}=none')
            elif value not in ('', None, [], {}):
                parts.append(f'{name}={value}')
        if parts:
            out.append(f'    {step}: {", ".join(parts)}')
    return out


def _pairs(values: dict) -> str:
    """`key=value` for every measure that measured something.

    A `None` is a measure that did not apply to these questions — an abstention
    rate over questions that were all answerable. Printing it spends a line
    saying nothing and reads as a recorded zero to anything skimming."""
    return ', '.join(
        f'{key}={_number(item, 3) if isinstance(item, float) else item}'
        for key, item in values.items() if item is not None)


def _flat(name: str, value) -> list[str]:
    """One summary block as lines: flat dicts on one, a dict of groups on one
    line per group.

    `by_type` and `by_difficulty` are dicts of dicts, and the whole of one on a
    single line is over a thousand characters of braces — the most useful signal
    in a run rendered as the least readable part of the answer. Per band and per
    question type is the comparison a reader actually asks for."""
    if not isinstance(value, dict):
        return [f'    {name}: {value}']
    grouped = {key: inner for key, inner in value.items()
               if isinstance(inner, dict)}
    if not grouped:
        return [f'    {name}: {_pairs(value)}']
    return [f'    {name}:'] + [f'      {key}: {_pairs(inner)}'
                               for key, inner in grouped.items()]


@tool
def read_experiment(experiment_id: str) -> str:
    """One experiment's knobs, metrics and judge; the model-facing prompt is
    fixtures/prompts/widget_tools.yaml's entry."""
    if _READER is None:
        return _UNWIRED
    found = _READER.experiment(experiment_id)
    if not found:
        return (f"No experiment '{experiment_id}' is recorded — neither the "
                'ledger nor .runs/ holds that id. Call list_experiments to see '
                'which ids exist.')
    judge = found.get('judge') or {}
    named = judge.get('model') or 'none'
    if judge.get('provider'):
        named += f" via {judge['provider']}"
    lines = [
        f"experiment {found.get('experiment_id')} — "
        f"{found.get('label') or 'no label'}",
        f"  dataset {found.get('dataset')} · {found.get('kind')}/"
        f"{found.get('state')} · started {found.get('started_at')} · "
        f"{found.get('seconds')}s · records: {found.get('source')}",
        f"  answering backend {found.get('provider') or 'unrecorded'} · "
        f"{found.get('n_questions', 0)} questions",
        f"  {_decision(found)} (judge {named})",
    ]
    if found.get('error'):
        lines.append(f"  did not finish: {' '.join(found['error'].split())}")
    metrics = found.get('metrics') or {}
    if metrics:
        lines.append('  the four judged metrics that decide:')
        lines += [f'    {name} {_number(value)}'
                  for name, value in metrics.items()]
    else:
        lines.append('  no judged metrics: this experiment scored none of the '
                     'four, so it makes no decision claim.')
    if found.get('ragas_skipped'):
        lines.append(f"  judged rows skipped: {found['ragas_skipped']}")
    knobs = _knobs(found.get('config') or {}, found.get('inert') or {})
    if knobs:
        lines.append('  knobs recorded:')
        lines += knobs
    if found.get('index'):
        lines.append(f"  index build: {_pairs(found['index'])}")
    if found.get('summary'):
        lines.append('  deterministic summary:')
    for name, value in (found.get('summary') or {}).items():
        lines += _flat(name, value)
    for note in list(found.get('notes') or []) + list(found.get('ragas_notes')
                                                      or []):
        lines.append(f'  note: {note}')
    lines.append('  per-question rows: '
                 + ('available — call read_experiment_questions'
                    if found.get('has_question_rows')
                    else 'none recorded for this experiment'))
    return '\n'.join(lines)


@tool
def read_experiment_questions(experiment_id: str, only: str = 'missed') -> str:
    """The per-question rows of one evaluation, filtered; the model-facing
    prompt is fixtures/prompts/widget_tools.yaml's entry."""
    if _READER is None:
        return _UNWIRED
    found = _READER.question_rows(experiment_id, only, MAX_QUESTION_ROWS)
    rows = found.get('rows') or []
    if not rows:
        # "No rows" and "no failures" are opposite answers, so the reason the
        # set is empty travels with it.
        reason = found.get('reason')
        if reason:
            return f'{experiment_id}: {reason}'
        return (f"{experiment_id}: no question matched the filter "
                f"'{only}' — of {found.get('n_questions', 0)} questions, none "
                'is in that set.')
    head = (f"{experiment_id} — {found.get('n_matched', 0)} of "
            f"{found.get('n_questions', 0)} questions match '{only}'"
            + (f", k={found['k']}" if found.get('k') else '')
            + f', showing {len(rows)}')
    lines = [head]
    for row in rows:
        lines.append(
            f"  {row.get('id')} [{row.get('type') or '?'}/"
            f"{row.get('difficulty') or '?'}]"
            f"{' unanswerable' if row.get('behavior') == 'abstain' else ''}"
            f" recall {_number(row.get('recall'), 2)}"
            f" precision {_number(row.get('precision'), 2)}"
            f" mrr {_number(row.get('mrr'), 2)}"
            f" contexts {row.get('n_contexts')}"
            + (' abstained' if row.get('abstained') else '')
            + (' false-abstention' if row.get('false_abstention') else ''))
        asked = row.get('question') or '(question text unavailable — the ' \
            'dataset that produced this run is no longer loadable)'
        lines.append(f'    Q: {asked}')
        lines.append(
            '    evidence expected in '
            + (', '.join(row.get('expected_sessions') or []) or 'none recorded')
            + ' · retrieved '
            + (', '.join(row.get('retrieved_sessions') or []) or 'nothing'))
    if found.get('n_matched', 0) > len(rows):
        lines.append(f"{found['n_matched'] - len(rows)} more matching question(s)"
                     ' are not shown: this listing is capped at '
                     f'{MAX_QUESTION_ROWS}.')
    return '\n'.join(lines)


EXPERIMENT_TOOLS = [list_experiments, read_experiment,
                    read_experiment_questions]
