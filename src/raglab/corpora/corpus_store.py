"""Every corpus an experiment ran on, kept once and addressed by its content.

    databases/corpora.db              (RAGLAB_CORPORA_DB overrides, `*.db` git-ignored)

An experiment archive carries the corpus it was measured against, because a
score without the text behind it is not evidence. Carrying it *inside* every
archive means the bundled diary — 690 KB — is written once per experiment: 17
diary runs are 17 identical copies, ~11 MB of the same sentences. So the archive
keeps a reference and the text lives here, once per distinct version.

The key is a fingerprint of the content itself, so the same corpus stored twice
is one row and two versions of one dataset id are two rows. That is the point,
not a side effect: a corpus edited between runs is a *different* corpus, and the
archive of the earlier run must keep resolving to the text that run actually
saw. Keying by dataset id instead would mean the newest edit answering for every
run that ever named it — a row lying about what produced it.

This replaces re-reading `fixtures/corpus_groundtruth_datasets/*.json` at serve
time. Those files are tracked in git and meant to be edited; under the old
scheme an ordinary edit permanently broke every archive that referenced them,
because the only honest answer to "the installed corpus is not the one this
experiment ran on" is a refusal. Content-addressed, an edit adds a row and
takes nothing away.

The key is checked rather than trusted. A fingerprint is 16 hex characters of
SHA-256 — evidence, not identity — so a row found by one is compared with the
text being stored before it is reused, and a disagreement is refused rather
than aliased (`_same_or_refuse`). Two different corpora sharing one row is
exactly the lie the whole store exists to prevent.

The ground truth travels with the corpus, because an archive carries both
(`inspector.dataset.{corpus,ground_truth}`) and both have to come back exactly.
The uniqueness that produces the dedup is over *both* fingerprints: one corpus
can be graded against two different question sets over its life, and a store
that kept only the first would quietly hand the wrong questions to the second
experiment.

A row's *key*, though, is `id` — a plain autoincrement integer. That is the
foreign key `databases/raglab.db` holds against each archived experiment, and
an integer join is what a second database file can actually carry. The
fingerprints stay on the row as a unique constraint, which keeps every property
the content-addressing bought: the same content stored twice is one row, one
corpus serves however many archives reference it, and a corpus edited between
runs becomes a *new* row rather than an overwrite of the version an older
archive names. An id is local, so it never leaves this machine; the archive
keeps the fingerprint too, and that is what makes an exported file
self-describing somewhere else.

No vector database and no index: this is the text as it was, not anything built
from it. The index still lives in process memory and dies with the process.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from raglab.configuration.lab_config import ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora (
  -- The key, and the one an archive row in `databases/raglab.db` joins on: a
  -- plain integer, because that is what a reference living in another database
  -- file can carry. It is local storage identity and nothing more — it says
  -- which row, never what the row contains, and it means nothing on another
  -- machine.
  id                       INTEGER PRIMARY KEY AUTOINCREMENT,
  -- What the row contains, said in content: `fingerprint` addresses the corpus
  -- and `ground_truth_fingerprint` the questions graded against it. Unique
  -- together rather than the key, which keeps every property the
  -- content-addressing bought — the same pair stored twice is one row, and an
  -- edit to either is a new row rather than an overwrite of the version an
  -- older archive names. This pair is also the portable half: an archive
  -- carries it, so an exported file can be resolved and verified where the id
  -- means nothing.
  fingerprint              TEXT NOT NULL,
  ground_truth_fingerprint TEXT NOT NULL,
  -- Descriptive only. A dataset id is what the corpus was called, never what
  -- it is; many rows may share one, which is a corpus with a history.
  dataset                  TEXT NOT NULL DEFAULT '',
  stored_at                TEXT NOT NULL DEFAULT '',
  corpus                   TEXT NOT NULL,
  ground_truth             TEXT NOT NULL,
  UNIQUE (fingerprint, ground_truth_fingerprint)
);
"""


class CorpusStoreError(ValueError):
    pass


def canonical(value) -> str:
    """One corpus (or ground truth) as one string, keys in order.

    The encoding `fingerprint` hashes, named because two callers need it: the
    hash, and the comparison that checks a hash's answer. Sorted keys and a
    fixed float repr, so the same content written by two different code paths
    compares equal — otherwise a mere difference in key order would read as two
    different corpora.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)


def fingerprint(value: dict) -> str:
    """A stable hash of a corpus (or a ground truth) as it was.

    The hashing `experiment_archive_store.corpus_fingerprint` shipped with,
    unchanged and deliberately so: fingerprints already written into archives
    have to keep addressing the same content, and a hash that moved would
    orphan every one of them.
    """
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()[:16]


def db_path(env: dict | None = None) -> Path:
    """Where the corpora live. `RAGLAB_CORPORA_DB` overrides, which is what
    lets the suite point every test at a temp file from one autouse fixture."""
    environ = os.environ if env is None else env
    override = (environ.get('RAGLAB_CORPORA_DB') or '').strip()
    return Path(override) if override else ROOT / 'databases' / 'corpora.db'


def connect(path: Path | None = None) -> sqlite3.Connection:
    """An open connection with the schema in place. Resolved per call rather
    than held open for the process, so it cannot go stale between experiments."""
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target)
    db.row_factory = sqlite3.Row
    db.execute(SCHEMA)
    return db


def put(dataset_id: str, corpus: dict, ground_truth: dict,
        path: Path | None = None) -> int:
    """Store one corpus with its ground truth; return the row's `id`.

    That id is what a reference elsewhere holds, so this returns the existing
    row's id when the content is already stored and a new one only when it is
    not. Idempotent because the *content* decides: the seventeenth archive on
    the bundled diary stores no bytes and gets the first archive's id back.

    `DO NOTHING` rather than a replace — identical content cannot need
    overwriting, and the only thing a rewrite could change is which dataset id
    and timestamp the row remembers, neither of which any reader may treat as
    identity, and neither of which is worth burning a new id over.
    """
    corpus_key = fingerprint(corpus)
    truth_key = fingerprint(ground_truth)
    values = {
        'fingerprint': corpus_key,
        'ground_truth_fingerprint': truth_key,
        'dataset': dataset_id or '',
        'stored_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'corpus': json.dumps(corpus, ensure_ascii=False, allow_nan=False),
        'ground_truth': json.dumps(ground_truth, ensure_ascii=False,
                                   allow_nan=False),
    }
    fields = tuple(values)
    with connect(path) as db:
        # Looked up before the insert is attempted, not only after. The insert
        # is still guarded by the unique constraint — that is what makes this
        # safe rather than a check-then-act race — but `AUTOINCREMENT` burns an
        # id on every attempt it refuses, so an unconditional insert would have
        # the seventeen diary archives leave a gap of sixteen ids behind them.
        found = db.execute(
            'SELECT id, corpus, ground_truth FROM corpora WHERE '
            'fingerprint = ? AND ground_truth_fingerprint = ?',
            (corpus_key, truth_key)).fetchone()
        if found is not None:
            _same_or_refuse(found, corpus, ground_truth, corpus_key, truth_key)
            return int(found['id'])
        db.execute(
            f'INSERT INTO corpora ({", ".join(fields)}) '
            f'VALUES ({", ".join(":" + name for name in fields)}) '
            'ON CONFLICT(fingerprint, ground_truth_fingerprint) DO NOTHING',
            values)
        # Read back rather than trusting `lastrowid`, which is meaningless
        # when the insert did nothing — which is what happens when another
        # writer stored this same content between the select above and here.
        row = db.execute(
            'SELECT id FROM corpora WHERE fingerprint = ? '
            'AND ground_truth_fingerprint = ?',
            (corpus_key, truth_key)).fetchone()
    return int(row['id'])


def _same_or_refuse(row, corpus: dict, ground_truth: dict,
                    corpus_key: str, truth_key: str) -> None:
    """The check the fingerprint cannot make for itself: is the row that
    answers to this key actually holding this text?

    A fingerprint here is 64 bits of SHA-256, which is evidence and not
    identity. Treating a match as proof would mean two different corpora
    aliasing to one row the day they ever collided — and then every archive
    referencing it would claim a corpus it never ran on, which is the one thing
    this repo refuses. So the stored text is compared with the text being
    stored, canonically, and a disagreement is a refusal.

    Never a second row instead: the key says these are the same corpus, so
    writing another under it would break the promise the uniqueness makes and
    leave two rows nothing could choose between. Refused rather than aliased,
    for the reason `experiment_archive_store.swell` refuses rather than
    substitutes.
    """
    stored_corpus = json.loads(row['corpus'])
    stored_truth = json.loads(row['ground_truth'])
    mismatched = [
        name for name, stored, offered in
        (('corpus', stored_corpus, corpus),
         ('ground truth', stored_truth, ground_truth))
        if canonical(stored) != canonical(offered)]
    if mismatched:
        raise CorpusStoreError(
            f'{corpus_key}/{truth_key}: this fingerprint already names a '
            f'different {" and ".join(mismatched)} in corpora row '
            f'{int(row["id"])}. A fingerprint is evidence, not identity, so a '
            'collision is refused rather than aliased onto the stored row — a '
            'row must never lie about what produced it.')


def get(id_corpora: int, path: Path | None = None) -> tuple[dict, dict] | None:
    """`(corpus, ground_truth)` for one stored row, or None if there is no such
    row. The lookup a local reference uses — `archives.id_corpora` in
    `databases/raglab.db` holds exactly this value, and is named for the table
    it points at.

    None means "this store does not have it" and nothing else — never a
    near-miss, never the newest version of the same dataset id. An id says
    which row and not what is in it, so a caller that has a fingerprint to
    check against must still check it: see `experiment_archive_store.swell`,
    which refuses rather than splicing when the two disagree.
    """
    with connect(path) as db:
        row = db.execute(
            'SELECT corpus, ground_truth FROM corpora WHERE id = ?',
            (int(id_corpora),)).fetchone()
    if row is None:
        return None
    return json.loads(row['corpus']), json.loads(row['ground_truth'])


def find(corpus_fingerprint: str, ground_truth_fingerprint: str = '',
         path: Path | None = None) -> tuple[dict, dict] | None:
    """The same answer, addressed by content instead of by id — the portable
    lookup, for a reference that came from somewhere this machine's ids mean
    nothing (an imported archive, a file another lab exported).

    Naming the ground truth as well selects one row exactly. Without it, one
    corpus graded against two different question sets is genuinely ambiguous,
    and an ambiguity is refused rather than resolved by picking.
    """
    found = locate(corpus_fingerprint, ground_truth_fingerprint, path)
    return None if found is None else get(found, path)


def locate(corpus_fingerprint: str, ground_truth_fingerprint: str = '',
           path: Path | None = None) -> int | None:
    """The id of one stored version, by content address, or None."""
    with connect(path) as db:
        if ground_truth_fingerprint:
            rows = db.execute(
                'SELECT id FROM corpora WHERE fingerprint = ? '
                'AND ground_truth_fingerprint = ?',
                (corpus_fingerprint, ground_truth_fingerprint)).fetchall()
        else:
            rows = db.execute('SELECT id FROM corpora WHERE fingerprint = ?',
                              (corpus_fingerprint,)).fetchall()
    if not rows:
        return None
    if len(rows) > 1:
        raise CorpusStoreError(
            f'{corpus_fingerprint}: this corpus is stored against '
            f'{len(rows)} different ground truths, so a reference that names '
            'none of them cannot say which experiment read which. Refused '
            'rather than guessed.')
    return int(rows[0]['id'])


def versions(dataset_id: str, path: Path | None = None) -> list[dict]:
    """Every stored version of one dataset id, oldest first — a corpus with a
    history, which is what a dataset that was edited between runs is."""
    with connect(path) as db:
        rows = db.execute(
            'SELECT id, fingerprint, ground_truth_fingerprint, dataset, '
            'stored_at FROM corpora WHERE dataset = ? ORDER BY stored_at, id',
            (dataset_id or '',)).fetchall()
    return [dict(row) for row in rows]
