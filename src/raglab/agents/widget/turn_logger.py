"""Readable, one-row-per-question operational logging for the widget.

LangGraph checkpoints remain an internal execution format. This table is the
small human-readable account of what happened: the root question, answer,
overall usage and latency, plus a JSON list of the steps that produced it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from raglab.configuration.env_settings import ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS widget_turn_log (
  turn_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  experiment_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  user_message_id TEXT NOT NULL,
  user_message TEXT NOT NULL,
  ai_message_id TEXT,
  ai_message TEXT,
  total_input_tokens INTEGER,
  total_output_tokens INTEGER,
  total_tokens INTEGER,
  total_latency_ms INTEGER,
  steps_json TEXT NOT NULL,
  memory_update_id INTEGER,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

_LOCK = RLock()


def db_path(env: dict | None = None) -> Path:
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


def _total(input_tokens, output_tokens):
    if input_tokens is None and output_tokens is None:
        return None
    return (input_tokens or 0) + (output_tokens or 0)


def log_turn(*, thread_id: str, experiment_id: str, dataset_id: str,
             user_message_id: str, user_message: str, ai_message_id: str = '',
             ai_message: str = '', steps: list[dict] | None = None,
             total_input_tokens: int | None = None,
             total_output_tokens: int | None = None,
             total_latency_ms: int | None = None, status: str = 'answered',
             memory_update_id: int | None = None) -> str:
    """Write one readable turn and return its stable ID."""
    turn_id = str(uuid.uuid4())
    input_tokens = (int(total_input_tokens)
                    if total_input_tokens is not None else None)
    output_tokens = (int(total_output_tokens)
                     if total_output_tokens is not None else None)
    row = (turn_id, str(thread_id or ''), str(experiment_id or ''),
           str(dataset_id or ''), str(user_message_id or ''),
           str(user_message or ''), str(ai_message_id or '') or None,
           str(ai_message or '') or None, input_tokens, output_tokens,
           _total(input_tokens, output_tokens),
           int(total_latency_ms) if total_latency_ms is not None else None,
           json.dumps(steps or [], ensure_ascii=False, separators=(',', ':')),
           memory_update_id, str(status or 'answered'), _now())
    with _LOCK, _connect() as db:
        db.execute(
            'INSERT INTO widget_turn_log '
            '(turn_id, thread_id, experiment_id, dataset_id, '
            'user_message_id, user_message, ai_message_id, ai_message, '
            'total_input_tokens, total_output_tokens, total_tokens, '
            'total_latency_ms, steps_json, memory_update_id, status, '
            'created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            row)
        db.commit()
    return turn_id


def _row(row) -> dict:
    values = dict(row)
    return values


def read_turn(turn_id: str) -> dict | None:
    with _LOCK, _connect() as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT * FROM widget_turn_log WHERE turn_id = ?',
                         (turn_id,)).fetchone()
    return _row(row) if row else None


def list_turns(thread_id: str = '') -> list[dict]:
    with _LOCK, _connect() as db:
        db.row_factory = sqlite3.Row
        if thread_id:
            rows = db.execute(
                'SELECT * FROM widget_turn_log WHERE thread_id = ? '
                'ORDER BY created_at, rowid', (thread_id,)).fetchall()
        else:
            rows = db.execute(
                'SELECT * FROM widget_turn_log ORDER BY created_at, rowid'
            ).fetchall()
    return [_row(row) for row in rows]


def clear() -> None:
    with _LOCK, _connect() as db:
        db.execute('DELETE FROM widget_turn_log')
        db.commit()


def attach_memory_update(turn_id: str, memory_update_id: int | None) -> None:
    """Link a completed summary write to its already logged question."""
    if memory_update_id is None:
        return
    with _LOCK, _connect() as db:
        db.execute('UPDATE widget_turn_log SET memory_update_id = ? '
                   'WHERE turn_id = ?', (int(memory_update_id), turn_id))
        db.commit()
