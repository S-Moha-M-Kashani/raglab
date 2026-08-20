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


def _cell(value, digits: int = 3) -> str:
    return '—' if value is None else f'{value:.{digits}f}'


def markdown(groups: list[Group]) -> str:
    """The leaderboard as markdown, one table per comparability group."""
    out = ['# RAG lab leaderboard',
           '',
           'Generated by `uv run raglab-leaderboard` from `.runs/`. **One table per',
           'comparability group** — a decision score is a mean over questions judged',
           'by a model, so it is comparable only against rows that scored the same',
           'questions, from the same corpus, with the same judge. Rows are never',
           'ranked across tables.',
           '']
    for found in groups:
        out.append(f'## {found.dataset} · {found.sample} · judged by {found.judge}')
        out.append('')
        call = verdict(found)
        if call == 'tie':
            out.append('**No winner**: the lead is inside the combined error of the '
                       'top two rows, so these rows do not separate.')
        elif call == 'unknown' and not _sample_recorded(found):
            out.append('**Not comparable**: these runs did not record *which* '
                       'questions they scored, only how many. Equal counts are not '
                       'a shared sample — the sampling rule changed during this '
                       "lab's life — so the ordering below is a listing, not a "
                       'ranking.')
        elif call == 'unknown':
            out.append('**No winner**: these rows carry no measured error, so a lead '
                       'cannot be told from noise. Runs predating `decision_spread` '
                       'cannot have one reconstructed.')
        elif call == 'unranked':
            out.append('**One judged row only** — nothing to compare it against.')
        else:
            out.append(f'**Winner: {call}**, by more than the combined error of the '
                       'top two rows.')
        out.append('')
        out.append('| # | candidate | decision | ± | questions | seconds | run |')
        out.append('| --- | --- | --- | --- | --- | --- | --- |')
        for i, row in enumerate(found.rows, start=1):
            # A numbered row is a rank claim, so an unrecorded sample gets none.
            rank = (str(i) if row.get('ragas_decision') is not None
                    and _sample_recorded(found) else '—')
            out.append(
                f'| {rank} | {row.get("label", "")} '
                f'| {_cell(row.get("ragas_decision"), 4)} '
                f'| {_cell(row.get("ragas_decision_stderr"), 3)} '
                f'| {row.get("n_questions", 0)} '
                f'| {round(row.get("seconds", 0))} '
                f'| `{row.get("run_id", "")}` |')
        out.append('')
    return '\n'.join(out)


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
    groups = build(args.limit)
    if args.json:
        print(json.dumps([as_dict(g) for g in groups],
                         ensure_ascii=False, indent=1))
        return
    text = markdown(groups)
    if args.write:
        path = Path(args.write)
        path.write_text(text + '\n', encoding='utf-8')
        print(f'wrote {path} · {len(groups)} comparability groups '
              f'from {sum(len(g.rows) for g in groups)} runs in {RUNS_DIR}')
    else:
        print(text)


if __name__ == '__main__':
    main()
