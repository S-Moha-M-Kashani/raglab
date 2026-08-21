"""Every finished experiment — index build, retrieval, or run — as one row.

    databases/raglab.db               (RAGLAB_DB overrides, `*.db` git-ignored)

Written in `Jobs.run` before the job goes terminal, so a poller looking for
`state = 'done'` never finds it missing. Never the vector store: the index stays
in process memory (in_memory_vector_store.py) and this records only what ran, not what it ran over.
"""
import json
import os
import sqlite3
import time
from pathlib import Path

from raglab.configuration.lab_config import ROOT

# Order matters: it is the column order of the table and of every row returned.
COLUMNS = (
    'experiment_id',        # a run's own id, else the job id — never both
    'kind',                 # index | retrieve | run | query
    'state',                # done | error | cancelled
    'label', 'started_at', 'seconds',
    'dataset',              # which corpus — a score means nothing without it
    'provider',             # the resolved chat backend: fake means "not a measurement"
    'chunker', 'embedder', 'retriever', 'reranker', 'grader', 'answerer',
    'n_questions',
    'decision', 'decision_stderr',
    'error',
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id   TEXT PRIMARY KEY,
  kind            TEXT NOT NULL,
  state           TEXT NOT NULL,
  label           TEXT NOT NULL DEFAULT '',
  started_at      TEXT NOT NULL DEFAULT '',
  seconds         REAL NOT NULL DEFAULT 0,
  dataset         TEXT NOT NULL DEFAULT '',
  provider        TEXT NOT NULL DEFAULT '',
  chunker         TEXT NOT NULL DEFAULT '',
  embedder        TEXT NOT NULL DEFAULT '',
  retriever       TEXT NOT NULL DEFAULT '',
  reranker        TEXT NOT NULL DEFAULT '',
  grader          TEXT NOT NULL DEFAULT '',
  answerer        TEXT NOT NULL DEFAULT '',
  n_questions     INTEGER NOT NULL DEFAULT 0,
  -- NULL, never 0.0, on every experiment that judged nothing — which is every
  -- index build, every retrieval, and every run with ragas off. A fabricated
  -- zero would sort below real rows and read as a measured refusal.
  decision        REAL,
  decision_stderr REAL,
  error           TEXT NOT NULL DEFAULT '',
  detail          TEXT NOT NULL DEFAULT '{}'
);
"""

# What never goes into a *job* row's `detail`: a new result shape carrying
# chunk text should be stripped by being added here, not by every caller
# remembering to. The one deliberate exception is `insert_archive` below —
# an imported archive is preserved verbatim, evidence included, because the
# ledger is its only home (it has no job, no run file, no leaderboard row).
HEAVY = ('chunks_by_session', 'archive_evidence')


def db_path(env: dict | None = None) -> Path:
    """Where the ledger lives. `RAGLAB_DB` overrides, which is what lets the
    suite point every test at a temp file from one autouse fixture."""
    environ = os.environ if env is None else env
    override = (environ.get('RAGLAB_DB') or '').strip()
    return Path(override) if override else ROOT / 'databases' / 'raglab.db'


def connect(path: Path | None = None) -> sqlite3.Connection:
    """An open connection with the schema in place. Resolved per call rather
    than held open for the process, so it cannot go stale between experiments."""
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target)
    db.row_factory = sqlite3.Row
    db.execute(SCHEMA)
    _migrate(db)
    return db


def _migrate(db: sqlite3.Connection) -> None:
    """Add columns this schema has gained since a ledger was created —
    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists."""
    have = {row['name'] for row in db.execute('PRAGMA table_info(experiments)')}
    for name, kind in (('dataset', "TEXT NOT NULL DEFAULT ''"),):
        if name not in have:
            db.execute(f'ALTER TABLE experiments ADD COLUMN {name} {kind}')


def stamp() -> str:
    """Local time, zero-padded, matching `RunResult.started_at`'s format."""
    return time.strftime('%Y-%m-%d %H:%M:%S')


def row_for(job: dict, state: str) -> dict:
    """One ledger row from one finished job. Derived from the job's own config
    and result rather than passed in by each route, so a route added later is
    recorded by having been run. Every field degrades to a blank or a zero.

    First of three projections between a job's nested config and the flat
    columns a row has. `leaderboard._ledger_config` reads these same columns
    back into a nested config for the board's settings panel, and
    `panel_server._experiment_from_run` writes a run file into this shape for a
    board row. All three have to mean the same thing by `chunker`, `retriever`
    and `answerer`."""
    result = job.get('result') if isinstance(job.get('result'), dict) else {}
    config = job.get('config') or {}
    index = config.get('index') or {}
    # A build's job config carries a full LabConfig, so retrieval/generation are
    # populated with defaults no part of a build reads; a build is defined by
    # its index config alone, so the rest of the row stays honestly blank.
    ran_pipeline = job.get('kind') != 'index'
    retrieval = (config.get('retrieval') or {}) if ran_pipeline else {}
    generation = (config.get('generation') or {}) if ran_pipeline else {}
    summary = result.get('summary') or {}
    ragas = result.get('ragas') or {}
    selection = result.get('selection') or {}
    return {
        # A run is identified by its run id, so this row and the JSON file the
        # leaderboard reads are the same measurement, each checkable against the
        # other. Everything else has only its job id.
        'experiment_id': result.get('run_id') or job.get('id') or '',
        'kind': job.get('kind') or '',
        'state': state,
        'label': result.get('label') or config.get('label') or '',
        'started_at': result.get('started_at') or stamp(),
        # The run's own figure where it has one, so the ledger and the run file
        # never disagree by a rounding; the job's wall clock otherwise, which is
        # the only duration a build or a retrieval has.
        'seconds': round(float(result.get('seconds')
                               or job.get('seconds') or 0.0), 2),
        'provider': config.get('provider') or '',
        'dataset': result.get('dataset') or index.get('dataset') or '',
        'chunker': index.get('chunker') or '',
        'embedder': index.get('embedder') or '',
        'retriever': retrieval.get('retriever') or '',
        'reranker': retrieval.get('reranker') or '',
        'grader': retrieval.get('grader') or '',
        'answerer': generation.get('answerer') or '',
        'n_questions': int(summary.get('n_questions')
                           or selection.get('n') or 0),
        'decision': ragas.get('decision'),
        'decision_stderr': (ragas.get('decision_spread') or {}).get('stderr'),
        'error': job.get('error') or '',
    }


def detail_for(job: dict) -> dict:
    """The stored payload: the whole result, minus the corpus."""
    result = job.get('result') if isinstance(job.get('result'), dict) else {}
    detail = {key: value for key, value in result.items() if key not in HEAVY}
    # The config travels even when the job never produced a result, because a
    # failed experiment is only worth recording if it says what was attempted.
    detail.setdefault('config', job.get('config') or {})
    return detail


def _row_for_archive(payload: dict) -> dict:
    """One completed imported archive as an ordinary, unranked experiment.

    Archives contain a canonical completed result rather than a ``Jobs``
    record, so they deliberately have their own narrow projection.  In
    particular this does not call the run-file helpers: imported evidence is
    inspectable from the ledger only and is never leaderboard input.
    """
    evaluation = payload['evaluation']
    result = evaluation['result']
    config = result['config']
    index = config['index']
    retrieval = config['retrieval']
    generation = config['generation']
    ragas = result['ragas']
    return {
        'experiment_id': result['run_id'],
        'kind': 'run',
        'state': 'done',
        'label': result['label'],
        'started_at': result['started_at'],
        'seconds': round(float(result['seconds']), 2),
        'dataset': result['dataset'],
        'provider': evaluation['execution']['provider'],
        'chunker': index['chunker'],
        'embedder': index['embedder'],
        'retriever': retrieval['retriever'],
        'reranker': retrieval['reranker'],
        'grader': retrieval['grader'],
        'answerer': generation['answerer'],
        'n_questions': int(result['summary']['n_questions']),
        'decision': ragas.get('decision'),
        'decision_stderr': (ragas.get('decision_spread') or {}).get('stderr'),
        'error': '',
    }


def insert_archive(payload: dict, path: Path | None = None) -> str:
    """Insert a completed archive once, preserving the first import verbatim."""
    row = _row_for_archive(payload)
    values = dict(row, detail=json.dumps(payload, ensure_ascii=False,
                                         allow_nan=False))
    fields = tuple(values)
    with connect(path) as db:
        cursor = db.execute(
            f'INSERT INTO experiments ({", ".join(fields)}) '
            f'VALUES ({", ".join(":" + name for name in fields)}) '
            'ON CONFLICT(experiment_id) DO NOTHING', values)
    return 'created' if cursor.rowcount == 1 else 'existing'


def load_archive(run_id: str, path: Path | None = None) -> dict | None:
    """Return a stored archive payload, never an ordinary experiment detail."""
    found = experiment(run_id, path=path)
    detail = (found or {}).get('detail')
    return detail if (isinstance(detail, dict)
                      and detail.get('format') == 'raglab-experiment') else None


def record(job: dict, state: str, path: Path | None = None) -> str:
    """Write one finished job down and return its experiment id. Raises on
    failure, deliberately: a ledger write must never be able to fail a run.
    `Jobs.run` catches it and reports it on the job."""
    row = row_for(job, state)
    values = dict(row, detail=json.dumps(detail_for(job), ensure_ascii=False,
                                         default=str))
    fields = tuple(values)
    with connect(path) as db:
        db.execute(
            f'INSERT INTO experiments ({", ".join(fields)}) '
            f'VALUES ({", ".join(":" + name for name in fields)}) '
            # Fires only on a retry of the same experiment, never two different
            # ones: a job id is unique per process, a run id carries a timestamp.
            f'ON CONFLICT(experiment_id) DO UPDATE SET '
            + ', '.join(f'{name} = excluded.{name}' for name in fields
                        if name != 'experiment_id'),
            values)
    return row['experiment_id']


def experiments(limit: int = 200, path: Path | None = None) -> list[dict]:
    """Every experiment, newest first, without its detail. Ordered by insertion
    (completion order), not `started_at`: a long run began earlier than short
    ones that finished first, and would otherwise sort as older."""
    try:
        with connect(path) as db:
            rows = db.execute(
                f'SELECT {", ".join(COLUMNS)} FROM experiments '
                'ORDER BY rowid DESC LIMIT ?', (limit,)).fetchall()
    except sqlite3.Error:
        # A ledger that cannot be read is an empty listing, never a 500.
        return []
    return [dict(row) for row in rows]


def experiment(experiment_id: str, path: Path | None = None) -> dict | None:
    """One experiment with its detail parsed, or None if the ledger has no such
    row."""
    with connect(path) as db:
        found = db.execute(
            f'SELECT {", ".join(COLUMNS)}, detail FROM experiments '
            'WHERE experiment_id = ?', (experiment_id,)).fetchone()
    if found is None:
        return None
    row = dict(found)
    try:
        row['detail'] = json.loads(row['detail'] or '{}')
    except json.JSONDecodeError:
        # Reported as unreadable, not as an empty experiment: the scalar
        # columns beside it are still true.
        row['detail'] = {'unreadable': True}
    return row
