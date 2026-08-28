"""Selective long-term memory for the widget.

This store lives beside the LangGraph checkpointer in ``widget.db`` but owns
only its three tables.  Dataset and global summaries are compact context, not
transcripts or measurements; ``memory_updates`` records the provenance of
each accepted update so a summary never loses the experiment that produced it.

The two summary tables are not the same kind of row. `dataset_memory` is filed
under one corpus and read only by that corpus's threads, so it holds that
corpus's findings and may name it. `global_memory` is one row every dataset's
thread is handed, so the only thing it may hold is a pattern that holds
*across* corpora and it may name none of them. `names_one_corpus` is the check
behind both, at both ends: the write refuses a note pointing at a corpus the
row may not be about, and `memory_context` holds back one already stored.
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
#:
#: The boundaries exclude a word character and *not* a hyphen, which is the
#: opposite of `_names_dataset`'s rule and deliberate: `run-20260828-160758-
#: 305a19` names an experiment just as plainly as the bare id does, and a
#: lookbehind that refused a leading hyphen missed exactly that. Widening it
#: costs no false positives, because what follows is fourteen digits in two
#: fixed groups and six hex characters — a shape nothing else in a summary
#: has — while `(?<!\w)` still stops the pattern being found halfway along a
#: longer number.
EXPERIMENT_ID_SHAPE = re.compile(r'(?<!\w)\d{8}-\d{6}-[0-9a-f]{6}(?!\w)')

#: How short a dataset id has to be before this stops looking for it in prose.
#: A one-word id is indistinguishable from the word: an installation with a
#: corpus called `index` would otherwise have "…in every corpus we index"
#: refused as naming it, which is a genuine cross-corpus pattern thrown away
#: to catch a mention nobody made. Every id this lab ships is a hyphenated
#: compound well over the bound (`diary-fa`, `smoke-import-check`), so the
#: rule costs nothing here and buys back the whole class of false refusals.
#: What a short id gives up is only *this* check — the summarizer is still
#: told not to name one, and an experiment id in the same sentence is still
#: caught by shape.
MIN_MATCHED_DATASET_ID = 6

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
    named when it stands as a whole token, not when its letters occur. Ids
    shorter than `MIN_MATCHED_DATASET_ID` are skipped for the reason stated
    there.
    """
    lowered = str(text or '').casefold()
    for dataset_id in sorted(str(value).strip() for value in (dataset_ids or ())
                             if len(str(value).strip()) >= MIN_MATCHED_DATASET_ID):
        if re.search(rf'(?<![\w-]){re.escape(dataset_id.casefold())}(?![\w-])',
                     lowered):
            return dataset_id
    return ''


def names_one_corpus(summary: str, dataset_ids) -> str:
    """What in `summary` points at one particular corpus or run, or `''`.

    A noun phrase — ``dataset 'meetings-de'``, ``experiment 20260828-160758-
    305a19`` — because the two callers frame it differently: naming another
    corpus is a different offence in the row every thread reads than it is in
    a row filed under one corpus, and each says so in its own words.

    Two shapes say a summary is about one corpus: a known dataset id and an
    experiment id. Which ids count as known is the caller's business — global
    memory passes every id including the thread's own, since a fact about one
    corpus does not belong in the row all of them read; dataset memory passes
    every id *except* its own, since a note filed under a corpus is supposed
    to be about it.

    The summarizer is *told* all of this (`hooks.SUMMARIZE_MEMORY_PROMPT`),
    and being told is not enough: on 2026-08-28 a `nosrat-fa` thread was
    handed "Last experiment details for smoke-import-check: 6 questions
    analyzed …" as standing context, which is another corpus's run stated as
    fact about this one. An instruction is a request; this is the check.

    One run's *numbers* are named in the instruction and not checked here, on
    purpose: a genuine cross-corpus pattern can carry a count ("in three of the
    four corpora"), so a rule against digits would refuse the very lines global
    memory exists for. The two id shapes are what make a summary a claim about
    one corpus, and they are what can be checked without guessing.
    """
    named = _names_dataset(summary, dataset_ids)
    if named:
        return f'dataset {named!r}'
    found = EXPERIMENT_ID_SHAPE.search(str(summary or ''))
    return f'experiment {found.group(0)}' if found else ''


#: The board's dataset ids, read once per process — see `_board_dataset_ids`.
_BOARD_DATASET_IDS: set | None = None


def forget_board_dataset_ids() -> None:
    """Drop the cached board ids, because the reader behind them changed.

    Called from `experiment_tools.set_experiment_reader`: the cache holds what
    one reader said, so wiring a different one (or none) has to forget it
    rather than let a record from the previous installation keep answering.
    """
    global _BOARD_DATASET_IDS
    _BOARD_DATASET_IDS = None


def remember_board_dataset_ids(dataset_ids) -> None:
    """Seed the cache from a board reading its caller has already taken.

    `backends._finish_memory` takes one `board_snapshot()` and hands
    `validated_dataset_ids(rows=board) | {dataset}` to the write. Those ids are
    the board's own plus the thread's validated corpus, so the read-time filter
    can have them for nothing — and without this the first saved turn after a
    reader is wired would read the board a second time, breaking the one
    reading per memory pass that `_finish_memory` was written to keep.

    A union rather than a replacement: a cache is only ever made more useful by
    one more validated id, and nothing here may narrow what an earlier reading
    already established.
    """
    global _BOARD_DATASET_IDS
    found = {str(value).strip() for value in (dataset_ids or ())
             if str(value).strip()}
    if found:
        _BOARD_DATASET_IDS = (_BOARD_DATASET_IDS or set()) | found


def _board_dataset_ids() -> set:
    """Every dataset id the validated experiment records know, cached.

    `experiment_tools.validated_dataset_ids` is the trustworthy source — it is
    what the write gate checks against — and the read side needs it too. The
    first version of this guard did without it, on the grounds that the store
    already knew which corpora it had filed memories under; that was wrong in
    a way a probe found at once. A pre-guard global note naming `meetings-de`,
    written on a thread for some other corpus, is invisible to a store that has
    never filed a `meetings-de` memory — and a widget.db carried over from
    another machine has exactly that shape. The board knows the id; the store
    does not.

    Imported inside the function, not at module scope, because
    `experiment_tools` pulls langchain_core in behind it and this module is
    deliberately the light one (see `BUSY_TIMEOUT_SECONDS` on why it does not
    import the checkpointer's).

    Cached for the life of the process because the alternative is a full board
    reading — the ledger plus up to `SCAN` run files — on the prompt path of
    every turn, where the write gate pays it once per saved turn. An empty
    reading is never cached: no reader is wired yet in that case, and the
    filter has to start working the moment one is, not at the next restart.
    """
    global _BOARD_DATASET_IDS
    if _BOARD_DATASET_IDS is not None:
        return _BOARD_DATASET_IDS
    try:
        from raglab.agents.widget import experiment_tools
        found = {str(value).strip()
                 for value in experiment_tools.validated_dataset_ids()
                 if str(value).strip()}
    except Exception:
        return set()
    if found:
        _BOARD_DATASET_IDS = found
    return found


def _known_dataset_ids(db) -> set:
    """Every dataset id either record can name, for the read-time filter.

    A union of two sources, and the union is the point. The board
    (`_board_dataset_ids`) is the trustworthy one and the one the write gate
    uses, but it is empty when no ledger reader is wired — which would switch
    this filter off in exactly the installation with the least provenance to
    lose. The store's own ids (`dataset_memory` ∪ `memory_updates`) are never
    empty once anything has been filed and are not model text either: a row
    reaches those tables only through `save_memory_update`, whose caller has
    already matched the policy's dataset against the one the validated records
    gave the thread (`backends._finish_memory`).

    So neither is a substitute for the other. The board sees corpora this
    widget has never written about; the store sees corpora on a machine whose
    board is not wired. Each covers the other's blind spot, and a filter is
    only ever made stricter by knowing one more id.
    """
    rows = db.execute('SELECT dataset_id FROM dataset_memory '
                      'UNION SELECT dataset_id FROM memory_updates').fetchall()
    recorded = {str(row[0]).strip() for row in rows if str(row[0] or '').strip()}
    return recorded | _board_dataset_ids()


def _without_foreign_notes(summary: str, dataset_ids) -> str:
    """The lines of a stored summary that do not point at one other corpus.

    Line by line rather than all-or-nothing, because both summary rows are
    aggregates: `_aggregate` joins each accepted note onto the previous ones
    with a newline, so one bad note must not cost a thread the good ones
    beside it.

    What was dropped is not reported into the prompt, and that is a decision
    rather than an omission. A bare "one note was withheld" is content the
    model can speculate about and cannot resolve, and on a thread with no
    dataset memory it would be the *entire* injected context — a prompt whose
    only content is that something is hidden. The honesty duty in CLAUDE.md
    attaches to records, and the records already carry it: the withheld note
    is still on disk exactly as it was written, its provenance is the
    `memory_updates` row that accepted it, and anything refused at the write
    is named on that turn's `widget_turn_log` row. A prompt is not a record.
    """
    return '\n'.join(
        line for line in str(summary or '').splitlines()
        if not (line.strip() and names_one_corpus(line, dataset_ids))).strip()


def _aggregate(previous: str, incoming: str) -> str:
    previous, incoming = previous.strip(), incoming.strip()
    if not previous:
        return _bounded(incoming)
    if not incoming:
        return _bounded(previous)
    return _bounded(f'{previous}\n{incoming}')


def memory_context(dataset_id: str) -> str:
    """Return applicable dataset and global summaries, or empty context.

    Both rows are filtered on the way out, and filtered rather than repaired.
    `save_memory_update` now refuses to store a note that points at another
    corpus, but rows written before that guard existed are already on disk, and
    the two honest ways to deal with them are to hold them back at the read or
    to mark them. Holding them back is what this does, for one reason: the
    store is a record of what the widget accepted, and rewriting it now would
    erase the evidence that it once accepted this. A marking pass would have to
    write to every stored row to say something the read can work out for
    itself, and a migration that edits memory is exactly the silent rewrite
    CLAUDE.md refuses. So nothing on disk changes; what changes is what a
    thread is handed.

    The two rows are filtered against different id sets, which is the whole
    difference between them. Global memory is read by every thread, so a note
    naming *any* corpus — this one included — is not the pattern that row is
    for. Dataset memory is filed under one corpus and read only by its own
    threads, so naming that corpus is exactly right and naming another is the
    same lie one thread wide.

    What is removed is only ever another corpus's specifics. A genuine
    cross-dataset pattern names no dataset and no experiment, and a dataset
    note about its own corpus may name it freely, so both survive untouched.
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
        known = _known_dataset_ids(db) | {dataset_id}
    sections = []
    if dataset and dataset[0]:
        own = _without_foreign_notes(dataset[0], known - {dataset_id})
        if own:
            sections.append(f'Dataset memory ({dataset_id}):\n{own}')
    if global_row and global_row[0]:
        pattern = _without_foreign_notes(global_row[0], known)
        if pattern:
            sections.append(f'Global memory:\n{pattern}')
    return '\n\n'.join(sections)


def save_memory_update(dataset_id: str, experiment_id: str, subtopic: str,
                       question: str, answer: str, dataset_summary: str,
                       global_summary: str = '',
                       validated_dataset_ids: set[str] | None = None) -> dict:
    """Persist one accepted update and return its stored summary state.

    An empty dataset has no valid provenance, so it is a no-op.  The caller may
    pass an empty global summary when the discussion supports no cross-dataset
    pattern; in that case the existing global memory remains unchanged.

    Each summary is checked against `names_one_corpus` before it is stored, and
    each is refused alone. A global note may name no corpus at all; a dataset
    note may name its own and no other. Refusing one does not cost the other,
    and neither costs the `memory_updates` row: that row is the provenance of
    the *turn*, and it records in its `decision` column which halves were kept
    — a turn that was accepted and then found to be about somebody else's
    corpus is exactly the thing a later reader will want to find.

    `dataset_refused` and `global_refused` carry the reasons back so the turn's
    own `widget_turn_log` row can say what was not kept (`backends.
    _record_memory_outcome`). `saved` follows the dataset summary, because that
    is the substance of a save: a turn whose only stored trace is its
    provenance row was not filed as memory, and saying otherwise would be the
    row lying about what it holds.
    """
    dataset_id = (dataset_id or '').strip()
    if not dataset_id:
        return {'saved': False, 'dataset_id': '', 'reason': 'empty dataset'}

    values = (dataset_id, (experiment_id or '').strip(),
              _normalize_subtopic(subtopic), (question or '').strip(),
              (answer or '').strip())
    now = _now()
    with _LOCK, _connect() as db:
        # The caller's validated ids (`backends._finish_memory` reads them from
        # `experiment_tools.validated_dataset_ids`, never from the model's own
        # text) widened by everything either record already knows — a check is
        # only ever made stricter by knowing one more id, and a direct caller
        # that passes none still gets one.
        dataset_ids = {str(value).strip() for value in
                       (validated_dataset_ids or set()) if str(value).strip()}
        remember_board_dataset_ids(dataset_ids)
        known = dataset_ids | _known_dataset_ids(db)

        previous = db.execute(
            'SELECT summary FROM dataset_memory WHERE dataset_id = ?',
            (dataset_id,)).fetchone()
        named = names_one_corpus(dataset_summary, known - {dataset_id})
        # Phrased to be read after a subject the caller supplies ("the dataset
        # note …"), because the reason is written on the turn's row and a row
        # a reader cannot read as a sentence is a row nobody reads.
        dataset_refused = (f'names {named}, which is a corpus other than '
                           f'{dataset_id!r}') if named else ''
        if dataset_refused:
            stored_dataset = previous[0] if previous else ''
        else:
            stored_dataset = _aggregate(previous[0] if previous else '',
                                        dataset_summary)
            db.execute(
                'INSERT INTO dataset_memory(dataset_id, summary, updated_at) '
                'VALUES (?, ?, ?) ON CONFLICT(dataset_id) DO UPDATE SET '
                'summary=excluded.summary, updated_at=excluded.updated_at',
                (dataset_id, stored_dataset, now))

        old_global = db.execute(
            'SELECT summary FROM global_memory WHERE id = 1').fetchone()
        stored_global = ''
        global_refused = ''
        # Two datasets have to be validated before a cross-dataset claim can be
        # made at all, and the thread has to stand on one of them — the rule
        # this row had before any of the naming checks existed.
        offered = ((global_summary or '').strip() and len(dataset_ids) >= 2
                   and dataset_id in dataset_ids)
        if offered:
            named = names_one_corpus(global_summary, known)
            global_refused = (f'names {named}, so it is not a pattern that '
                              'holds across corpora') if named else ''
        if offered and not global_refused:
            stored_global = _aggregate(old_global[0] if old_global else '',
                                       global_summary)
            db.execute(
                'INSERT INTO global_memory(id, summary, updated_at) VALUES '
                '(1, ?, ?) ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, '
                'updated_at=excluded.updated_at', (stored_global, now))
        else:
            stored_global = old_global[0] if old_global else ''

        decided = 'accepted'
        if dataset_refused or global_refused:
            decided = 'refused: ' + '; '.join(
                reason for reason in (dataset_refused, global_refused) if reason)
        update = db.execute(
            'INSERT INTO memory_updates(dataset_id, experiment_id, subtopic, '
            'question, answer, decision, created_at) VALUES (?, ?, ?, ?, ?, '
            '?, ?)', values + (decided, now))
        update_id = update.lastrowid
    return {'saved': not dataset_refused, 'dataset_id': dataset_id,
            'update_id': update_id,
            'dataset_summary': stored_dataset,
            'global_summary': stored_global,
            'dataset_refused': dataset_refused,
            'global_refused': global_refused}


def clear_long_term_memory() -> None:
    """Delete long-term rows while leaving the widget checkpointer untouched."""
    with _LOCK, _connect() as db:
        db.execute('DELETE FROM memory_updates')
        db.execute('DELETE FROM dataset_memory')
        db.execute('DELETE FROM global_memory')
