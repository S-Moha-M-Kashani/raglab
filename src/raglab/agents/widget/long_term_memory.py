"""Selective long-term memory for the widget.

This store lives beside the LangGraph checkpointer in ``widget.db`` but owns
only its three tables.  Dataset and global summaries are compact context, not
transcripts or measurements; ``memory_updates`` records the provenance of
each accepted update so a summary never loses the experiment that produced it.
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
    db = sqlite3.connect(target)
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


def _aggregate(previous: str, incoming: str) -> str:
    previous, incoming = previous.strip(), incoming.strip()
    if not previous:
        return _bounded(incoming)
    if not incoming:
        return _bounded(previous)
    return _bounded(f'{previous}\n{incoming}')


def memory_context(dataset_id: str) -> str:
    """Return applicable dataset and global summaries, or empty context."""
    dataset_id = (dataset_id or '').strip()
    if not dataset_id:
        return ''
    with _LOCK, _connect() as db:
        dataset = db.execute(
            'SELECT summary FROM dataset_memory WHERE dataset_id = ?',
            (dataset_id,)).fetchone()
        global_row = db.execute(
            'SELECT summary FROM global_memory WHERE id = 1').fetchone()
    sections = []
    if dataset and dataset[0]:
        sections.append(f'Dataset memory ({dataset_id}):\n{dataset[0]}')
    if global_row and global_row[0]:
        sections.append(f'Global memory:\n{global_row[0]}')
    return '\n\n'.join(sections)


def save_memory_update(dataset_id: str, experiment_id: str, subtopic: str,
                       question: str, answer: str, dataset_summary: str,
                       global_summary: str = '',
                       validated_dataset_ids: set[str] | None = None) -> dict:
    """Persist one accepted update and return its stored summary state.

    An empty dataset has no valid provenance, so it is a no-op.  The caller may
    pass an empty global summary when the discussion supports no cross-dataset
    pattern; in that case the existing global memory remains unchanged.
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
        dataset_ids = {str(value).strip() for value in
                       (validated_dataset_ids or set()) if str(value).strip()}
        if ((global_summary or '').strip() and len(dataset_ids) >= 2
                and dataset_id in dataset_ids):
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
            'global_summary': stored_global}


def clear_long_term_memory() -> None:
    """Delete long-term rows while leaving the widget checkpointer untouched."""
    with _LOCK, _connect() as db:
        db.execute('DELETE FROM memory_updates')
        db.execute('DELETE FROM dataset_memory')
        db.execute('DELETE FROM global_memory')
