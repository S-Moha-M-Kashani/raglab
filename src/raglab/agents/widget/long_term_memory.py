"""Selective long-term memory for the widget.

This store lives beside the LangGraph checkpointer in ``widget.db`` but owns
only its three tables.  Dataset and global summaries are compact context, not
transcripts or measurements; ``memory_updates`` records the provenance of
each accepted update so a summary never loses the experiment that produced it.

The two summary tables are not the same kind of row. `dataset_memory` is filed
under one corpus and read only by that corpus's threads, so it holds that
corpus's findings. `global_memory` is one row every dataset's thread is handed,
so the only thing it may hold is a pattern that holds *across* corpora — which
is why `cross_dataset_violation` guards both ends of it: the write refuses a
note naming one corpus, and `memory_context` holds back one already stored.
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from raglab.configuration.env_settings import ROOT

MAX_SUMMARY_CHARS = 2_000
MAX_SUBTOPIC_CHARS = 64

#: What an experiment id looks like on this board: the run's local date and
#: time, then six hex digits of the index fingerprint
#: (`run_evaluation`, e.g. ``20260828-160758-305a19``). Matched by shape and
#: not by lookup, because the shape is decisive on its own and because the
#: read side has to keep guarding when no ledger reader is wired at all.
EXPERIMENT_ID_SHAPE = re.compile(r'(?<![\w-])\d{8}-\d{6}-[0-9a-f]{6}(?![\w-])')

#: The same busy timeout `conversation_memory.BUSY_TIMEOUT_SECONDS` states in
#: full: three writers share ``widget.db``, and this one runs on the daemon
#: thread the deferred memory decision starts, which can still be inside its
#: transaction when the next turn's checkpoint or turn-log row is written.
#: Its own copy for the reason ``db_path`` is one: this module must not import
#: the checkpointer's, which pulls langchain in behind it.
BUSY_TIMEOUT_SECONDS = 5.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_memory (
  dataset_id TEXT PRIMARY KEY,
  summary TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS global_memory (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  summary TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_updates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id TEXT NOT NULL,
  experiment_id TEXT NOT NULL,
  subtopic TEXT NOT NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  decision TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

_LOCK = RLock()


def db_path(env: dict | None = None) -> Path:
    """Return the widget database path, honoring the test/runtime override."""
    environ = os.environ if env is None else env
    override = (environ.get('RAGLAB_WIDGET_DB') or '').strip()
    return Path(override) if override else ROOT / 'databases' / 'widget.db'


def _connect() -> sqlite3.Connection:
    target = db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target, timeout=BUSY_TIMEOUT_SECONDS)
    db.executescript(SCHEMA)
    return db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _bounded(summary: str) -> str:
    """Keep the newest part of a summary within the prompt-size contract."""
    return str(summary or '').strip()[-MAX_SUMMARY_CHARS:]


def _normalize_subtopic(subtopic: str) -> str:
    """Store a compact, stable label rather than model punctuation or prose."""
    words = re.findall(r'[\w]+', str(subtopic or '').casefold(), re.UNICODE)
    return '-'.join(words)[:MAX_SUBTOPIC_CHARS].strip('-')


def _names_dataset(text: str, dataset_ids) -> str:
    """The first known dataset id `text` names, or `''`.

    Bounded on both sides so `diary-fa` is not found inside `diary-fashion`
    and `smoke-import` is not found inside `smoke-import-check`: an id is
    named when it stands as a whole token, not when its letters occur.
    """
    lowered = str(text or '').casefold()
    for dataset_id in sorted(str(value).strip() for value in (dataset_ids or ())
                             if str(value).strip()):
        if re.search(rf'(?<![\w-]){re.escape(dataset_id.casefold())}(?![\w-])',
                     lowered):
            return dataset_id
    return ''


def cross_dataset_violation(summary: str, dataset_ids) -> str:
    """Why `summary` is not a cross-dataset pattern, or `''` when it is one.

    Global memory is the one row every dataset's thread reads, so the only
    thing it may hold is a pattern that holds across corpora. Two shapes say a
    line is about one corpus instead: a known dataset id and an experiment id.
    Both are refused — including the thread's own dataset id, because a fact
    about this corpus belongs in this corpus's `dataset_memory` row, where a
    thread standing on another one will never be handed it.

    The summarizer is *told* this (`hooks.summarize_memory_update`), and being
    told is not enough: on 2026-08-28 a `nosrat-fa` thread was handed
    "Last experiment details for smoke-import-check: 6 questions analyzed …"
    as standing context, which is another corpus's run stated as fact about
    this one. An instruction is a request; this is the check.

    One run's *numbers* are named in the instruction and not enforced here, on
    purpose: a genuine cross-corpus pattern can carry a count ("in three of the
    four corpora"), so a rule against digits would refuse the very lines this
    row exists for. The two id shapes are what make a line a claim about one
    corpus, and they are what can be checked without guessing.
    """
    named = _names_dataset(summary, dataset_ids)
    if named:
        return f'it names dataset {named!r}, so it is not a cross-dataset pattern'
    found = EXPERIMENT_ID_SHAPE.search(str(summary or ''))
    if found:
        return (f'it names experiment {found.group(0)}, so it is one '
                "run's detail rather than a cross-dataset pattern")
    return ''


def _recorded_dataset_ids(db) -> set[str]:
    """Every dataset id this store has itself filed a memory under.

    Not model text and not a guess: a row reaches `dataset_memory` or
    `memory_updates` only through `save_memory_update`, whose caller has
    already matched the policy's dataset against the one the validated
    experiment records gave the thread (`backends._finish_memory`). So these
    ids are the validated ones, minus any corpus this widget has never filed
    anything about.

    Why not `experiment_tools.validated_dataset_ids` here, which is the source
    the write gate uses: it costs a full board reading — the ledger plus up to
    `SCAN` run files — and this runs on the prompt path of every turn, not
    once per saved turn. And it returns an empty set when no ledger reader is
    wired, which would turn the read-time filter off in exactly the
    installation that has the least provenance to lose.
    """
    rows = db.execute('SELECT dataset_id FROM dataset_memory '
                      'UNION SELECT dataset_id FROM memory_updates').fetchall()
    return {str(row[0]).strip() for row in rows if str(row[0] or '').strip()}


def _cross_dataset_lines(summary: str, dataset_ids) -> tuple[str, int]:
    """The lines of a global summary that are patterns, and how many are not.

    Line by line rather than all-or-nothing, because the row is an aggregate:
    `_aggregate` joins each accepted global note onto the previous ones with a
    newline, so one bad note must not cost a thread the good ones beside it.
    """
    kept, withheld = [], 0
    for line in str(summary or '').splitlines():
        if line.strip() and cross_dataset_violation(line, dataset_ids):
            withheld += 1
            continue
        kept.append(line)
    return '\n'.join(kept).strip(), withheld


def _aggregate(previous: str, incoming: str) -> str:
    previous, incoming = previous.strip(), incoming.strip()
    if not previous:
        return _bounded(incoming)
    if not incoming:
        return _bounded(previous)
    return _bounded(f'{previous}\n{incoming}')


def memory_context(dataset_id: str) -> str:
    """Return applicable dataset and global summaries, or empty context.

    The global row is filtered on the way out, and it is filtered rather than
    repaired. `save_memory_update` now refuses to store a global note that
    names one corpus, but rows written before that guard existed are already
    on disk, and the two honest ways to deal with them are to hold them back
    at the read or to mark them. Holding them back is what this does, for one
    reason: the store is a record of what the widget accepted, and rewriting
    it now would erase the evidence that it once accepted this. A marking pass
    would have to write to every stored row to say something the read can work
    out for itself, and a migration that edits memory is exactly the silent
    rewrite CLAUDE.md refuses. So nothing on disk changes; what changes is what
    a thread is handed.

    What is removed is only ever another corpus's specifics — a genuine
    cross-dataset pattern names no dataset and no experiment, so it survives
    the filter untouched and still reaches every thread. The count of what was
    held back is stated rather than dropped in silence: a context that quietly
    shrank would be the prompt lying about what memory holds.
    """
    dataset_id = (dataset_id or '').strip()
    if not dataset_id:
        return ''
    with _LOCK, _connect() as db:
        dataset = db.execute(
            'SELECT summary FROM dataset_memory WHERE dataset_id = ?',
            (dataset_id,)).fetchone()
        global_row = db.execute(
            'SELECT summary FROM global_memory WHERE id = 1').fetchone()
        known = _recorded_dataset_ids(db) | {dataset_id}
    sections = []
    if dataset and dataset[0]:
        sections.append(f'Dataset memory ({dataset_id}):\n{dataset[0]}')
    if global_row and global_row[0]:
        pattern, withheld = _cross_dataset_lines(global_row[0], known)
        if pattern:
            sections.append(f'Global memory:\n{pattern}')
        if withheld:
            sections.append(
                f'(Withheld from global memory: {withheld} older '
                f'note{"s" if withheld > 1 else ""} naming one corpus\'s '
                'specifics, which is not a cross-dataset pattern.)')
    return '\n\n'.join(sections)


def save_memory_update(dataset_id: str, experiment_id: str, subtopic: str,
                       question: str, answer: str, dataset_summary: str,
                       global_summary: str = '',
                       validated_dataset_ids: set[str] | None = None) -> dict:
    """Persist one accepted update and return its stored summary state.

    An empty dataset has no valid provenance, so it is a no-op.  The caller may
    pass an empty global summary when the discussion supports no cross-dataset
    pattern; in that case the existing global memory remains unchanged.

    A global summary that is not a cross-dataset pattern
    (`cross_dataset_violation`) is refused here and refused alone: the dataset
    summary still stores, the update still records its provenance, and
    `global_refused` carries the reason back so the turn's own row can say what
    was not kept. Refusing the whole save would cost a thread a finding about
    its own corpus because the summarizer over-reached about everyone else's.
    """
    dataset_id = (dataset_id or '').strip()
    if not dataset_id:
        return {'saved': False, 'dataset_id': '', 'reason': 'empty dataset'}

    values = (dataset_id, (experiment_id or '').strip(),
              _normalize_subtopic(subtopic), (question or '').strip(),
              (answer or '').strip())
    now = _now()
    with _LOCK, _connect() as db:
        previous = db.execute(
            'SELECT summary FROM dataset_memory WHERE dataset_id = ?',
            (dataset_id,)).fetchone()
        stored_dataset = _aggregate(previous[0] if previous else '',
                                    dataset_summary)
        db.execute(
            'INSERT INTO dataset_memory(dataset_id, summary, updated_at) '
            'VALUES (?, ?, ?) ON CONFLICT(dataset_id) DO UPDATE SET '
            'summary=excluded.summary, updated_at=excluded.updated_at',
            (dataset_id, stored_dataset, now))
        update = db.execute(
            'INSERT INTO memory_updates(dataset_id, experiment_id, subtopic, '
            'question, answer, decision, created_at) VALUES (?, ?, ?, ?, ?, '
            '?, ?)', values + ('accepted', now))

        old_global = db.execute(
            'SELECT summary FROM global_memory WHERE id = 1').fetchone()
        stored_global = ''
        refused = ''
        dataset_ids = {str(value).strip() for value in
                       (validated_dataset_ids or set()) if str(value).strip()}
        offered = ((global_summary or '').strip() and len(dataset_ids) >= 2
                   and dataset_id in dataset_ids)
        if offered:
            # The ids are the caller's validated ones, which is what makes this
            # a check and not a word search: `backends._finish_memory` reads
            # them from `experiment_tools.validated_dataset_ids`, never from
            # the model's own text.
            refused = cross_dataset_violation(global_summary, dataset_ids)
        if offered and not refused:
            stored_global = _aggregate(old_global[0] if old_global else '',
                                       global_summary)
            db.execute(
                'INSERT INTO global_memory(id, summary, updated_at) VALUES '
                '(1, ?, ?) ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, '
                'updated_at=excluded.updated_at', (stored_global, now))
        else:
            stored_global = old_global[0] if old_global else ''
        update_id = update.lastrowid
    return {'saved': True, 'dataset_id': dataset_id, 'update_id': update_id,
            'dataset_summary': stored_dataset,
            'global_summary': stored_global,
            'global_refused': refused}


def clear_long_term_memory() -> None:
    """Delete long-term rows while leaving the widget checkpointer untouched."""
    with _LOCK, _connect() as db:
        db.execute('DELETE FROM memory_updates')
        db.execute('DELETE FROM dataset_memory')
        db.execute('DELETE FROM global_memory')
