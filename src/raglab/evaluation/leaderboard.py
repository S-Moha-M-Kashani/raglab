"""Build the leaderboard from `.runs/`, and refuse to rank rows that are not comparable — group first, rank second, since a decision score is comparable only against rows judged on the same questions by the same judge.
`uv run raglab-leaderboard` prints it; `--write <path>` writes it.
Nothing here recomputes a score: it reads what the runs stored, so a number can always be checked against the run id on its row.
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from raglab.configuration.lab_config import RUNS_DIR
from raglab.evaluation.run_evaluation import list_runs
from raglab.evaluation import ragas_judged_metrics as judged
from raglab.evaluation import service_experiment_ledger as ledger


# The board's leftmost column: one fragment per pipeline step that actually ran,
# each inked with that step's colour by whatever renders it. Assembled here, not
# in the page, for the same reason `as_dict` exists — two surfaces that each
# derived the sentence could describe one row two ways.
#
# A step that did not run is absent, not '—'. An index build's sentence is its
# index fragment and nothing else; three em-dashes beside it would draw a row
# that reads as a failed evaluation rather than a finished build.
def pipeline_fragments(config: dict) -> list[dict]:
    index = config.get('index') or {}
    retrieval = config.get('retrieval') or {}
    generation = config.get('generation') or {}
    agent = config.get('agent') or {}

    # The vendor prefix is identical on every row and costs the width the
    # sentence needs; the model name after it is the part that differs.
    model = (index.get('embed_model') or '').split('/')[-1]
    parts = {
        'index': [(index.get('chunker') or '')
                  + ('+ctx' if index.get('contextual') else ''),
                  index.get('hierarchy') or '',
                  index.get('embedder') or '', model],
        'retrieval': [retrieval.get('retriever') or '',
                      retrieval.get('reranker') or '',
                      retrieval.get('grader') or ''],
        'generation': [generation.get('answerer') or ''],
        # 'off' on nearly every row would spend the sentence saying nothing.
        'agent': [agent.get('scope') or ''],
    }
    out = []
    for step in ('index', 'retrieval', 'generation', 'agent'):
        text = '·'.join(p for p in parts[step] if p and p != 'none')
        if text:
            out.append({'step': step, 'text': text})
    return out


@dataclass
class Group:
    """Rows that are comparable to one another, and nothing else."""
    question_ids: tuple
    judge_model: str
    judge_provider: str
    dataset: str = ''
    rows: list = field(default_factory=list)

    @property
    def n_questions(self) -> int:
        # From the ids, not the row's own count, which could disagree with them.
        if not _sample_recorded(self):
            return self.rows[0].get('n_questions', 0) if self.rows else 0
        return len(self.question_ids)

    @property
    def sample(self) -> str:
        balance = (self.rows[0].get('selection') or {}).get('balance', '') \
            if self.rows else ''
        if not _sample_recorded(self):
            return (f'{self.n_questions} questions, but *which* questions was '
                    'not recorded')
        return f'{self.n_questions} questions' + (f', {balance}' if balance else '')

    @property
    def newest(self) -> str:
        """The latest `started_at` on any row. Timestamps are zero-padded, so
        lexical order is chronological order."""
        return max((r.get('started_at', '') for r in self.rows), default='')

    @property
    def judge(self) -> str:
        """Reads after "judged by", so every branch has to be a noun phrase."""
        if self.judge_model:
            return f'{self.judge_model} via {self.judge_provider or "?"}'
        # A decision score was judged by something; runs predating
        # `report['judge']` simply didn't record what.
        if any(r.get('ragas_decision') is not None for r in self.rows):
            return 'a judge that was not recorded'
        return 'no judge — nothing on these rows was judged'


def _key(row: dict) -> tuple:
    selection = row.get('selection') or {}
    judge = row.get('judge') or {}
    # Coarsest first: two corpora are never one measurement. No dataset predates
    # the field and means the built-in diary, the only corpus that existed.
    dataset = row.get('dataset') or 'diary-fa'
    ids = tuple(sorted(selection.get('question_ids') or ()))
    if not ids:
        # No ids predates `RunResult.selection`; falls back to the count so a
        # 3- and a 100-question run don't share a table. `verdict()` still
        # refuses these groups, since equal counts may not be the same 24.
        ids = ('n', row.get('n_questions', 0))
    return (dataset, ids, judge.get('model', ''), judge.get('provider', ''))


def _sample_recorded(found: 'Group') -> bool:
    return bool(found.question_ids) and found.question_ids[0] != 'n'


def group(rows: list[dict]) -> list[Group]:
    """Partition rows into comparability groups, each internally ranked.

    Rows with no decision score stay in their group and sort last: a run that
    could not measure all four deciding metrics is a fact about that run, and
    dropping it would hide an attempt rather than report it."""
    groups: dict[tuple, Group] = {}
    for row in rows:
        key = _key(row)
        dataset, ids, model, provider = key
        found = groups.get(key)
        if found is None:
            found = Group(ids, model, provider, dataset)
            groups[key] = found
        found.rows.append(row)
    for found in groups.values():
        found.rows.sort(key=lambda r: (r.get('ragas_decision') is None,
                                       -(r.get('ragas_decision') or 0.0)))
    # Rankable groups first, most recent within each class. Two stable passes
    # rather than one key: the date wants descending order, the flags ascending,
    # and a string cannot be negated.
    by_date = sorted(groups.values(), key=lambda g: g.newest, reverse=True)
    return sorted(by_date, key=lambda g: (not _sample_recorded(g),
                                          verdict(g) in ('unknown', 'unranked')))


def verdict(found: Group) -> str:
    """The label of the winning row, 'tie' if the lead is inside the *combined*
    error of the two rows, or 'unknown' if no error was measured."""
    if not _sample_recorded(found):
        # Equal counts are not a shared sample: two runs of 24 may be different 24s.
        return 'unknown'
    ranked = [r for r in found.rows if r.get('ragas_decision') is not None]
    if not ranked:
        return 'unknown'
    if len(ranked) == 1:
        # One measured row cannot beat anything.
        return 'unranked'
    best, second = ranked[0], ranked[1]
    errors = [best.get('ragas_decision_stderr'),
              second.get('ragas_decision_stderr')]
    if any(error is None for error in errors):
        return 'unknown'
    combined = (errors[0] ** 2 + errors[1] ** 2) ** 0.5
    lead = best['ragas_decision'] - second['ragas_decision']
    return best['label'] if lead > combined else 'tie'


@dataclass
class Board:
    """Every experiment that touched one corpus, in one table.

    Deliberately not a `Group`. `Group` is a comparability class — same
    questions, same judge — and it is what a *sweep* ranks, which is why
    `group()` and `verdict()` below are untouched. A `Board` mixes judges and
    question sets on purpose, so it carries no `sample`, no `judge` and no
    `verdict`: none of the three is a property of a mixed table, and all three
    are per-row columns instead. It names no winner for the same reason."""
    dataset: str
    rows: list = field(default_factory=list)

    @property
    def n_experiments(self) -> int:
        return len(self.rows)

    @property
    def newest(self) -> str:
        """Timestamps are zero-padded, so lexical order is chronological."""
        return max((r.get('started_at', '') for r in self.rows), default='')


def by_dataset(rows: list[dict]) -> list[Board]:
    """One Board per corpus, each ordered by decision score.

    The order rows are served in *is* the ranking, and the shared sorter's
    third click restores exactly it — so the ordering rule here is the same one
    `group()` applies within a group: unjudged rows last, never as a zero."""
    boards: dict[str, Board] = {}
    for row in rows:
        # No dataset predates the field and means the built-in diary, the only
        # corpus that existed then — the same fallback `_key` applies.
        dataset = row.get('dataset') or 'diary-fa'
        boards.setdefault(dataset, Board(dataset)).rows.append(row)
    for found in boards.values():
        found.rows.sort(key=lambda r: (_decision(r) is None, -(_decision(r) or 0.0)))
    return sorted(boards.values(), key=lambda b: b.newest, reverse=True)


# A `.runs/` row calls it `ragas_decision`; a board row calls it `decision`.
# `by_dataset` is fed both — the sweep's rows and the board's — so it reads
# either rather than making each caller reshape first.
def _decision(row: dict):
    value = row.get('decision')
    return row.get('ragas_decision') if value is None else value


# --- the board's population: both durable records, joined ------------------
# Two records exist and neither is sufficient. The ledger has every job — index
# builds, retrievals, evaluations, imported archives — with the knobs, the kind
# and the state, but not the four judged metrics and not which judge graded.
# `.runs/` has those, for evaluations only.
#
# The join key already exists: the ledger stores
# `experiment_id = result['run_id'] or job['id']`, so for an evaluation it *is*
# the .runs/ filename. Nothing new has to be recorded.
#
# It is a union rather than a ledger read because the ledger is written in
# `Jobs.run`: every evaluation older than the ledger has a run file and no
# ledger row, and reading one record would drop those runs off the board
# without saying so.
def _metrics(run: dict) -> dict:
    """Only the four that decide, read from the tuple that defines them so a
    board column cannot drift from the decision rule."""
    found = run.get('ragas') or {}
    return {name: found[name] for name in judged.DECISION_METRICS
            if found.get(name) is not None}


def board_rows(limit: int = 500, db_path=None) -> list[dict]:
    runs = {r['run_id']: r for r in list_runs(limit=limit)}
    out, seen = [], set()
    for row in ledger.experiments(limit=limit, path=db_path):
        found = runs.get(row['experiment_id'])
        seen.add(row['experiment_id'])
        out.append(_board_row(row, found))
    # Evaluations the ledger never saw. Ordered after the ledger's rows and then
    # re-sorted by `by_dataset`, so the seam is not visible in the table.
    for run_id, found in runs.items():
        if run_id not in seen:
            out.append(_board_row(None, found))
    return out


def _board_row(row: dict | None, run: dict | None) -> dict:
    row = row or {}
    run = run or {}
    config = run.get('config') or row.get('config') or {}
    source = 'both' if (row and run) else ('ledger' if row else 'run')
    return {
        'experiment_id': row.get('experiment_id') or run.get('run_id') or '',
        # A run file is an evaluation by definition; only the ledger records
        # builds and retrievals, so only it can say otherwise.
        'kind': row.get('kind') or ('run' if run else ''),
        # A record that exists is a job that finished. Nothing else is inferred.
        'state': row.get('state') or ('done' if run else ''),
        'error': row.get('error') or '',
        'label': run.get('label') or row.get('label') or '',
        'started_at': run.get('started_at') or row.get('started_at') or '',
        'seconds': run.get('seconds') or row.get('seconds') or 0,
        'dataset': run.get('dataset') or row.get('dataset') or '',
        'provider': row.get('provider') or '',
        'n_questions': run.get('n_questions') or row.get('n_questions') or 0,
        # The run file wins where both carry it: that file is where the number
        # was computed and the one a reader can open to check it.
        'decision': (run.get('ragas_decision') if run
                     else row.get('decision')),
        'decision_stderr': (run.get('ragas_decision_stderr') if run
                            else row.get('decision_stderr')),
        'metrics': _metrics(run),
        'judge': run.get('judge') or {},
        'pipeline': pipeline_fragments(config),
        'config': config,
        # A reader is entitled to know why a metric column is blank.
        'source': source,
    }


def _cell(value, digits: int = 3) -> str:
    return '—' if value is None else f'{value:.{digits}f}'


def markdown(boards: list['Board']) -> str:
    """The board as markdown, one table per dataset.

    No verdict line. A board mixes question sets and judges, so 'winner by more
    than the combined error of the top two rows' would compare numbers that
    never met. That claim still exists — `verdict()` above — for the sweep,
    whose candidates share a question set and a judge by construction.

    `judge` and `questions` are columns here rather than table headings, so the
    reason two rows are not comparable is on screen instead of inferred."""
    out = ['# RAG lab leaderboard',
           '',
           'Generated by `uv run raglab-leaderboard` from `.runs/` and',
           '`databases/raglab.db`. **One table per dataset** — every experiment',
           'that touched one corpus, in one table, ordered by decision score.',
           'Nothing here names a winner: rows judged by different models over',
           'different question sets share a table, so `judge` and `questions`',
           'are columns you compare on rather than a promise that two rows are',
           'comparable.',
           '']
    for found in boards:
        out.append(f'## {found.dataset} · {found.n_experiments} experiments')
        out.append('')
        out.append('| pipeline | kind | when | decision | ± | judge '
                   '| questions | seconds | id |')
        out.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
        for row in found.rows:
            judge = row.get('judge') or {}
            named = (f"{judge.get('model')} via {judge.get('provider') or '?'}"
                     if judge.get('model') else '—')
            sentence = ' '.join(f['text'] for f in row.get('pipeline') or []) or '—'
            out.append(
                f'| {sentence} | {row.get("kind", "") or "—"} '
                f'| {(row.get("started_at") or "")[:16] or "—"} '
                f'| {_cell(_decision(row), 4)} '
                f'| {_cell(row.get("decision_stderr"), 3)} '
                f'| {named} | {row.get("n_questions", 0)} '
                f'| {round(row.get("seconds", 0))} '
                f'| `{row.get("experiment_id", "")}` |')
        out.append('')
    return '\n'.join(out)


def board_dict(found: 'Board') -> dict:
    """One serialised shape, so the command line and the panel's route cannot
    come to disagree about what a board is."""
    return {'dataset': found.dataset, 'n_experiments': found.n_experiments,
            'newest': found.newest, 'rows': found.rows}


def build_board(limit: int = 500, db_path=None) -> list['Board']:
    return by_dataset(board_rows(limit=limit, db_path=db_path))


def as_dict(found: Group) -> dict:
    """One serialised shape, so the command line and the panel's route cannot
    come to disagree about what a group is. `verdict` travels with the group
    because a caller that re-derived it could reach a different answer from the
    same rows, and then two surfaces would name different winners."""
    return {'dataset': found.dataset, 'sample': found.sample,
            'judge': found.judge, 'verdict': verdict(found),
            'n_questions': found.n_questions, 'newest': found.newest,
            # A numbered row is a rank claim, so the ranks are computed here
            # rather than left to whatever renders them: a client counting from
            # one would silently promote a row whose sample was never recorded.
            'ranked': _sample_recorded(found),
            'rows': found.rows}


def build(limit: int = 500) -> list[Group]:
    return group(list_runs(limit=limit))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=500,
                        help='how many run files to read (newest first)')
    parser.add_argument('--write', metavar='PATH',
                        help='write the markdown here instead of printing it')
    parser.add_argument('--json', action='store_true',
                        help='dump the grouped rows instead of markdown')
    args = parser.parse_args()
    boards = build_board(args.limit)
    if args.json:
        print(json.dumps([board_dict(b) for b in boards],
                         ensure_ascii=False, indent=1))
        return
    text = markdown(boards)
    if args.write:
        path = Path(args.write)
        path.write_text(text + '\n', encoding='utf-8')
        print(f'wrote {path} · {len(boards)} datasets from '
              f'{sum(len(b.rows) for b in boards)} experiments in {RUNS_DIR} '
              'and the ledger')
    else:
        print(text)


if __name__ == '__main__':
    main()
