"""Build the board: every experiment that touched one corpus, in one table, read
from both durable records — `databases/raglab.db` (every job: builds, retrievals,
evaluations, imports) and `.runs/` (the four judged metrics and the judge, for
evaluations only) — joined on the id the ledger already stores.
A board mixes question sets and judges on purpose and names no winner for it;
`group()`/`verdict()` below still do that, for a sweep, whose candidates are
comparable by construction.
`uv run raglab-leaderboard` prints it; `--write <path>` writes it.
Nothing here recomputes a score: it reads what was stored, so a number can
always be checked against the experiment id on its row.
"""
import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from raglab.configuration.lab_config import RUNS_DIR
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation.run_evaluation import list_runs
from raglab.evaluation import ragas_judged_metrics as judged
from raglab.evaluation import service_experiment_ledger as ledger


# The board's leftmost column: one fragment per pipeline step that actually ran,
# each inked with that step's colour by whatever renders it. Assembled here, not
# in the page and not in the printer, for the same reason `board_dict` exists:
# two surfaces that each derived the sentence could describe one row two ways.
#
# The short form of the words the sentence is made of. The sentence is the
# board's widest column and the one column a reader cannot do without — it is
# what tells two rows apart — so it is written twice rather than once: `text` is
# every knob spelled out, `short` is the same knobs in the abbreviations the
# field already uses. Only what a reader would recognise abbreviated is here; a
# knob with no entry keeps its own name, which is what stops a knob added later
# from being drawn as a word nobody can expand.
SHORT_PARTS = {
    # chunkers
    'semantic-drift': 'sem-drift', 'fixed-overlap': 'fix-ov',
    'message': 'msg', 'turn-pair': 'pair', 'session': 'sess',
    # hierarchies
    'louvain': 'louv', 'leiden': 'leid', 'label-prop': 'lprop',
    'raptor': 'rapt', 'agglomerative': 'agglo', 'kmeans': 'kmn',
    'metadata': 'meta',
    # embedders. The hash embedders keep the '#' rather than losing the word
    # that says what they are: hashing is the whole claim of that row.
    'sentence-transformers': 'ST', 'fastembed': 'FE',
    'ascii-hash': 'ascii#', 'token-hash': 'tok#', 'char-hash': 'char#',
    # retrievers
    'hybrid-rrf': 'rrf',
    # rerankers and graders, which share a vocabulary
    'lexical': 'lex', 'cross-encoder': 'CE', 'recency': 'rec',
    'agentic': 'agt',
    # answerers
    'extractive': 'extr',
}

# What every offered embedding model's name ends with, and none of what a reader
# is comparing: the version is fixed by the checkpoint and the family is the
# same on every row that names a model at all. The *size* is never dropped —
# MiniLM-L6 and MiniLM-L12 are two different indexes.
_MODEL_VERSION = re.compile(r'-v\d+(?:\.\d+)*$')
_MODEL_FAMILY = ('paraphrase-multilingual-', 'multilingual-', 'paraphrase-',
                 'all-')


def short_part(part: str) -> str:
    """One word of the pipeline sentence, in its short form."""
    # The contextual mark rides on the chunker's name, and it is two syllables
    # already: abbreviate what it is attached to, keep the mark.
    if part.endswith('+ctx'):
        return short_part(part[:-len('+ctx')]) + '+ctx'
    if part in SHORT_PARTS:
        return SHORT_PARTS[part]
    trimmed = _MODEL_VERSION.sub('', part)
    for family in _MODEL_FAMILY:
        if trimmed.startswith(family):
            return trimmed[len(family):]
    return trimmed


# A step that did not run is absent, not '—'. An index build's sentence is its
# index fragment and nothing else; three em-dashes beside it would draw a row
# that reads as a failed evaluation rather than a finished build.
def pipeline_fragments(config: dict) -> list[dict]:
    index = config.get('index') or {}
    retrieval = config.get('retrieval') or {}
    generation = config.get('generation') or {}

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
    }
    out = []
    for step in ('index', 'retrieval', 'generation'):
        said = [p for p in parts[step] if p and p != 'none']
        if said:
            out.append({'step': step, 'text': '·'.join(said),
                        'short': '·'.join(short_part(p) for p in said)})
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


# Which corpus a row belongs to, decided in one place. No dataset predates the
# field and means the built-in corpus, the only one that existed then. Three
# callers ask — the comparability key, the board's grouping and the board's own
# row — and while the row answered it for itself, a blank landed on the built-in
# board carrying a cell that said it belonged to no corpus at all.
def _dataset(*rows: dict | None) -> str:
    for row in rows:
        found = (row or {}).get('dataset')
        if found:
            return found
    return datasets.BUILTIN


def _key(row: dict) -> tuple:
    selection = row.get('selection') or {}
    judge = row.get('judge') or {}
    # Coarsest first: two corpora are never one measurement.
    dataset = _dataset(row)
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
        dataset = _dataset(row)
        boards.setdefault(dataset, Board(dataset)).rows.append(row)
    for found in boards.values():
        found.rows = _by_decision(found.rows)
    return sorted(boards.values(), key=lambda b: b.newest, reverse=True)


def _by_decision(rows: list[dict]) -> list[dict]:
    """Judged rows best first, unjudged last — never as a zero."""
    return sorted(rows, key=lambda r: (_decision(r) is None,
                                       -(_decision(r) or 0.0)))


def every_row(boards: list['Board']) -> list[dict]:
    """Every board's rows in one list, ordered the way one board is.

    The order rows are served in *is* the ranking the page describes, so the
    unfiltered view cannot be the boards concatenated: that is ordered by
    dataset block, and a page whose own prose says the served order is the
    ranking would then be wrong about itself."""
    return _by_decision([row for board in boards for row in board.rows])


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


# `ledger.experiments()` returns the flat columns `service_experiment_ledger`
# stores — `chunker`, `embedder`, `retriever`, `reranker`, `grader`,
# `answerer` — never a nested config dict; only a run file carries one. A
# ledger-only row (every index build, every retrieval, every imported
# archive — nothing a run file ever covers) still needs a pipeline sentence,
# so this reshapes the flat columns into the nesting `pipeline_fragments`
# reads. The ledger has no `contextual`, no `hierarchy` and no `embed_model`,
# so a ledger-only sentence is necessarily shorter than a run-file one — that
# is correct and nothing here guesses to fill the gap.
#
# Second of three projections between a nested config and the flat columns a row
# has, and the inverse of the first: `ledger.row_for` writes a job's config into
# those columns, and `panel_server._experiment_from_run` writes a run file into
# the same shape. What one of the three calls a knob, all three must.
#
# A knob with no recorded value is dropped, and a step left with no knobs at all
# goes with it. Emitting the empty shell had the settings panel draw RETRIEVAL
# and GENERATION headings with blank knobs under them for every index build —
# inventing, one function later, exactly the two stages `pipeline_fragments`
# above refuses to pad. `'none'` is a recorded value and stays.
def ledger_config(row: dict) -> dict:
    steps = {
        'index': {'chunker': row.get('chunker'),
                  'embedder': row.get('embedder')},
        'retrieval': {'retriever': row.get('retriever'),
                      'reranker': row.get('reranker'),
                      'grader': row.get('grader')},
        'generation': {'answerer': row.get('answerer')},
    }
    recorded = {step: {knob: value for knob, value in knobs.items() if value}
                for step, knobs in steps.items()}
    return {step: knobs for step, knobs in recorded.items() if knobs}


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
    # A run file's config wins when there is one; a ledger-only row (no run
    # file at all) still gets a sentence, built from the ledger's own flat
    # columns rather than left blank.
    config = run.get('config') or (ledger_config(row) if row else {})
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
        # Resolved, not served raw: `by_dataset` files a blank under the
        # built-in corpus, and a row whose own cell then said it had no dataset
        # would deny the table it is sitting in.
        'dataset': _dataset(run, row),
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


def _state(row: dict) -> str:
    """A job that did not finish, and why, in one cell.

    Bold because it changes how every other cell on the row reads: a cancelled
    run's blank decision is not a run waiting to be judged. The reason is
    flattened onto one line and its pipes escaped, or it would end the cell it
    is in and shift every column after it."""
    state = row.get('state') or ''
    if state == 'done':
        return 'done'
    said = f'**{state or "?"}**'
    reason = ' '.join((row.get('error') or '').split()).replace('|', r'\|')
    return f'{said} — {reason}' if reason else said


def markdown(boards: list['Board']) -> str:
    """The board as markdown, one table per dataset.

    No verdict line. A board mixes question sets and judges, so 'winner by more
    than the combined error of the top two rows' would compare numbers that
    never met. That claim still exists — `verdict()` above — for the sweep,
    whose candidates share a question set and a judge by construction.

    `judge` and `questions` are columns here rather than table headings, so the
    reason two rows are not comparable is on screen instead of inferred.

    `state` is a column for the same kind of reason. The population is every
    job, not only the evaluations a run file exists for, so cancelled and failed
    jobs print here — and one of those read as an ordinary unjudged experiment
    while its `—` decision looked like a run nobody had judged yet. The page
    carries the same column with a '!' for the reason; a terminal has nowhere to
    put a '!', so the reason is in the cell."""
    out = ['# RAG lab leaderboard',
           '',
           'Generated by `uv run raglab-leaderboard` from `.runs/` and',
           '`databases/raglab.db`. **One table per dataset** — every experiment',
           'that touched one corpus, in one table, ordered by decision score.',
           'Nothing here names a winner. Rows judged by different models over',
           'different question sets share a table, so `judge` and `questions`',
           'are columns you compare on rather than a promise that two rows are',
           'comparable.',
           '']
    for found in boards:
        n = found.n_experiments
        out.append(f'## {found.dataset} · {n} experiment{"" if n == 1 else "s"}')
        out.append('')
        out.append('| pipeline | kind | state | when | decision | ± | judge '
                   '| questions | seconds | id |')
        out.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- '
                   '| --- |')
        for row in found.rows:
            judge = row.get('judge') or {}
            named = (f"{judge.get('model')} via {judge.get('provider') or '?'}"
                     if judge.get('model') else '—')
            # ' · ' the way the page joins them. On screen the boundary between
            # two steps is carried by their colours; in a terminal it is carried
            # by nothing, and a space left the whole sentence reading as one
            # run-on token.
            sentence = ' · '.join(f['text']
                                  for f in row.get('pipeline') or []) or '—'
            out.append(
                f'| {sentence} | {row.get("kind", "") or "—"} '
                f'| {_state(row)} '
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=500,
                        help='how many ledger rows and run files to read '
                             '(newest first)')
    parser.add_argument('--write', metavar='PATH',
                        help='write the markdown here instead of printing it')
    parser.add_argument('--json', action='store_true',
                        help='dump the boards instead of markdown')
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
