"""The corpora this lab can measure against, and the contract a new one meets.

The lab was built around one fixture — a year of Farsi diary chat and 112
ground-truth questions — and every retrieval finding it has produced is a finding
*about that corpus*. Some of them are obviously general (an embedder that cannot
represent the script scores at chance) and some are obviously not (the Farsi
time-scope filter), and until a second corpus existed there was no way to tell
which was which. So a dataset became a thing the lab can be pointed at.

Three rules hold the design up.

**A dataset is part of the index, not part of the run.** `IndexConfig.dataset`
is the field, so it lands in the fingerprint: an index built over one corpus can
never be handed to a question from another, which is the one bug this feature
could plausibly introduce and the most expensive one to notice late. `''` means
the built-in diary, which keeps every fingerprint already recorded in `.runs/`
exactly what it was.

**A score is comparable only within a dataset.** The leaderboard already refuses
to rank across question sets and judges; the dataset is the coarsest such key and
is now the first one it groups by. A run recorded before this existed carries
`''`, which *is* the built-in diary — that is not a guess, it is the only corpus
that existed — so old rows stay comparable with new ones rather than being
quarantined.

**The contract is checked, never assumed** (`docs/groundtruth-dataset-contract.md`).
`validate()` refuses a dataset whose evidence does not hold: a quote that is not
verbatim in the message it cites is the failure that matters, because every
lexical metric in this lab is computed against those quotes and a dataset that
lies about them produces confident numbers about nothing. That check is why the
four bundled samples can be trusted as reference points.

**On disk a dataset is one file, in the lab it is two objects.** The internal
shape (`diary` + `ground_truth`) predates this module and is what six other
modules speak, including the field names `question_fa`/`answer_fa` — historical,
and not worth renaming through every stored run to satisfy an English corpus.
`load()` is the one place that translates: the contract says `question`, the lab
says `question_fa`, and a dataset written either way loads.
"""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DIFFICULTIES, ROOT
from .corpus import DIARY_PATH, GROUND_TRUTH_PATH, load_diary, load_ground_truth
from .metrics import TYPES

# Shipped with the repository and read-only: these are reference points, and a
# reference point that can be edited in place is not one.
BUNDLED_DIR = ROOT / 'fixtures' / 'groundtruth_datasets'
# Imported through the panel. Git-ignored and machine-local, like `.runs/`: a
# corpus somebody uploaded is their material, not this repository's.
IMPORTED_DIR = ROOT / '.datasets'

# The built-in corpus, which is two files rather than one because it predates the
# contract. `''` rather than its name is what a config carries by default, so
# every fingerprint and every stored run keeps meaning what it meant.
BUILTIN = 'diary-fa'

ID_SHAPE = re.compile(r'^[a-z0-9][a-z0-9-]{1,39}$')


@dataclass(frozen=True)
class Dataset:
    """One corpus with its ground truth, as the panel lists it."""
    id: str
    name: str
    description: str
    language: str
    source: str                  # builtin | bundled | imported
    path: str                    # repo-relative, for a reader who wants to look
    sessions: int = 0
    messages: int = 0
    questions: int = 0
    query_date: str = ''
    period: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {'id': self.id, 'name': self.name, 'description': self.description,
                'language': self.language, 'source': self.source, 'path': self.path,
                'sessions': self.sessions, 'messages': self.messages,
                'questions': self.questions, 'query_date': self.query_date,
                'period': self.period}


def imported_dir() -> Path:
    """Where imports land. `RAGLAB_DATASETS` overrides, which is what lets the
    suite import into a temp directory without touching the real one."""
    override = (os.environ.get('RAGLAB_DATASETS') or '').strip()
    return Path(override) if override else IMPORTED_DIR


# --- reading ---------------------------------------------------------------

def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _files() -> list[tuple[str, Path]]:
    """Every dataset file, bundled then imported, newest name order. An import
    with the id of a bundled sample shadows it — it is the copy the user made
    on purpose, and refusing the name would be refusing the edit."""
    found: list[tuple[str, Path]] = []
    for source, folder in (('bundled', BUNDLED_DIR), ('imported', imported_dir())):
        if not folder.exists():
            continue
        for path in sorted(folder.glob('*.json')):
            found.append((source, path))
    return found


def _builtin() -> Dataset:
    diary = load_diary()
    ground_truth = load_ground_truth()
    return Dataset(
        id=BUILTIN, name='Farsi diary — one year',
        description=(diary['meta'].get('description') or '')[:200],
        language=diary['meta'].get('language', 'fa'), source='builtin',
        path=str(DIARY_PATH.relative_to(ROOT)),
        sessions=len(diary['sessions']),
        messages=sum(len(s['messages']) for s in diary['sessions']),
        questions=len(ground_truth['questions']),
        query_date=ground_truth['meta'].get('query_date', ''),
        period=diary['meta'].get('period', {}))


def describe(payload: dict, source: str, path: Path) -> Dataset:
    meta = payload.get('dataset') or {}
    sessions = payload.get('sessions') or []
    dates = sorted(s.get('date', '') for s in sessions if s.get('date'))
    try:
        shown = str(path.relative_to(ROOT))
    except ValueError:
        shown = str(path)
    return Dataset(
        id=meta.get('id', path.stem), name=meta.get('name', path.stem),
        description=meta.get('description', ''),
        language=meta.get('language', ''), source=source, path=shown,
        sessions=len(sessions),
        messages=sum(len(s.get('messages') or []) for s in sessions),
        questions=len(payload.get('questions') or []),
        query_date=meta.get('query_date', ''),
        period={'from': dates[0], 'to': dates[-1]} if dates else {})


def catalogue() -> list[Dataset]:
    """Every dataset this lab can be pointed at, the built-in one first.

    First because it is the default and because every finding in
    `docs/report/` is about it — a list whose head is whatever sorted first
    would put the corpus the report argues over in an arbitrary place. A file
    that will not parse is skipped rather than fatal: a broken import must not
    take the panel down with it."""
    out = [_builtin()]
    seen = {BUILTIN}
    for source, path in _files():
        try:
            payload = _read(path)
        except Exception:
            continue
        found = describe(payload, source, path)
        if found.id in seen:
            out = [d for d in out if d.id != found.id]
        seen.add(found.id)
        out.append(found)
    return out


def find(dataset_id: str) -> Dataset | None:
    wanted = dataset_id or BUILTIN
    return next((d for d in catalogue() if d.id == wanted), None)


def _path_for(dataset_id: str) -> Path | None:
    for source, path in _files():
        try:
            meta = (_read(path).get('dataset') or {})
        except Exception:
            continue
        if meta.get('id', path.stem) == dataset_id:
            return path
    return None


_CACHE: dict[str, tuple[dict, dict]] = {}


def load(dataset_id: str = '') -> tuple[dict, dict]:
    """`(diary, ground_truth)` for one dataset, in the shape the lab speaks.

    Cached per id: a corpus is read once per process and then handed to every
    build, which is the same bargain the index registry makes. Nothing here is
    written back, so the cache cannot go stale against anything but the file —
    and an import clears it."""
    wanted = dataset_id or BUILTIN
    if wanted == BUILTIN:
        return load_diary(), load_ground_truth()
    if wanted in _CACHE:
        return _CACHE[wanted]
    path = _path_for(wanted)
    if path is None:
        raise ValueError(
            f'unknown dataset {wanted!r} — known: '
            + ', '.join(d.id for d in catalogue()))
    _CACHE[wanted] = _split(_read(path))
    return _CACHE[wanted]


def forget() -> None:
    """Drop the corpus cache. Called on import, so a dataset replaced under the
    same id is the one the next build reads."""
    _CACHE.clear()


def _split(payload: dict) -> tuple[dict, dict]:
    """One dataset file → the two objects the pipeline takes.

    Every optional session field is filled here rather than defended against
    downstream: the chunkers read `mood`, `time`, `source`, `topics` and
    `recurring_threads` directly, and a corpus that has to carry five fields it
    has no use for is a corpus nobody will write. `language` rides on each
    session so the contextual header can be written in the corpus's own
    language."""
    meta = payload.get('dataset') or {}
    language = meta.get('language', 'en')
    sessions = [_session(raw, language) for raw in payload.get('sessions') or []]
    dates = sorted(s['date'] for s in sessions if s.get('date'))
    query_date = meta.get('query_date') or (dates[-1] if dates else '')
    diary = {
        'meta': {'description': meta.get('description', ''), 'language': language,
                 'period': {'from': dates[0] if dates else '',
                            'to': dates[-1] if dates else ''}},
        'persona': meta.get('persona', {}),
        'threads': meta.get('threads', []),
        'sessions': sessions,
        'habits': meta.get('habits', {}),
    }
    ground_truth = {
        'meta': {'description': meta.get('description', ''),
                 'corpus': meta.get('id', ''), 'query_date': query_date},
        'questions': [_question(raw, query_date)
                      for raw in payload.get('questions') or []],
    }
    return diary, ground_truth


def _session(raw: dict, language: str) -> dict:
    mood = raw.get('mood') or {}
    return {
        'session_id': raw.get('session_id', ''), 'date': raw.get('date', ''),
        'time': raw.get('time', '12:00'), 'source': raw.get('source', 'text'),
        'language': language,
        # Neutral rather than absent: `importance_of` is arithmetic over these,
        # and the agentic reranker would otherwise refuse a corpus for not being
        # a diary. 5.5/5.5 is the midpoint of both scales, so a corpus that does
        # not track mood contributes no importance signal in either direction.
        'mood': {'label': mood.get('label', ''),
                 'valence': mood.get('valence', 5.5),
                 'arousal': mood.get('arousal', 5.5)},
        'topics': list(raw.get('topics') or []),
        'recurring_threads': list(raw.get('threads')
                                  or raw.get('recurring_threads') or []),
        'messages': [{'role': m.get('role', 'user'),
                      'intent': m.get('intent', ''),
                      'content': m.get('content', '')}
                     for m in raw.get('messages') or []],
    }


def _question(raw: dict, query_date: str) -> dict:
    """The contract's `question`/`answer` under the names the lab reads.

    `question_fa` is what `pipeline`, `evaluate`, `metrics` and both frontends
    ask for. Renaming it would touch six modules and make every stored run
    unreadable by the leaderboard, to gain nothing a reader of this function
    does not already know: the field holds the question in the corpus's own
    language, and for the corpus it was named after that language is Farsi."""
    text = raw.get('question') or raw.get('question_fa') or ''
    answer = raw.get('answer') or raw.get('answer_fa') or ''
    return {
        'id': raw.get('id', ''), 'type': raw.get('type', ''),
        'difficulty': raw.get('difficulty', ''),
        'query_date': raw.get('query_date') or query_date,
        'question_fa': text, 'question_en': raw.get('question_en', ''),
        'answer_fa': answer, 'answer_en': raw.get('answer_en', ''),
        'time_scope': raw.get('time_scope'),
        'answerable': bool(raw.get('answerable', True)),
        'key_facts': list(raw.get('key_facts') or []),
        'evidence': [{'session_id': ev.get('session_id', ''),
                      'message_indices': list(ev.get('message_indices') or []),
                      'quote': ev.get('quote', '')}
                     for ev in raw.get('evidence') or []],
        'threads': list(raw.get('threads') or []),
    }


# --- the contract ----------------------------------------------------------

def validate(payload: dict) -> list[str]:
    """Everything wrong with a dataset, in the order a reader would fix it.

    A list rather than an exception, and *all* of the problems rather than the
    first: importing a corpus is a slow loop if each attempt reports one broken
    quote out of nine. The messages name the offending id, because "evidence
    quote not found" over 200 questions is not a message, it is a search.

    The rules are `docs/groundtruth-dataset-contract.md`; the one that earns its
    cost is the last: a quote must appear verbatim in a message it cites. Every
    lexical metric in this lab is computed against those quotes, so a dataset
    that misquotes its own corpus reports confident numbers about nothing.
    """
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ['the dataset must be a JSON object']
    meta = payload.get('dataset')
    if not isinstance(meta, dict):
        problems.append('"dataset" is missing: a corpus has to say what it is')
        meta = {}
    dataset_id = meta.get('id', '')
    if not ID_SHAPE.match(str(dataset_id)):
        problems.append(
            f'dataset.id {dataset_id!r} must be 2–40 characters of lowercase '
            'letters, digits and hyphens — it names a file and a config value')
    if not meta.get('name'):
        problems.append('dataset.name is missing: the panel lists it by name')
    if not meta.get('language'):
        problems.append('dataset.language is missing — an English-only embedder '
                        'returns confident numbers on a corpus it cannot read, '
                        'and the panel says so only if the corpus states its '
                        'language')

    sessions = payload.get('sessions')
    if not isinstance(sessions, list) or not sessions:
        problems.append('"sessions" must be a non-empty list')
        sessions = []
    by_id: dict[str, dict] = {}
    for i, session in enumerate(sessions):
        where = session.get('session_id') or f'sessions[{i}]'
        if not isinstance(session, dict):
            problems.append(f'{where}: each session must be an object')
            continue
        if not session.get('session_id'):
            problems.append(f'sessions[{i}]: session_id is required — evidence '
                            'points at it')
        elif session['session_id'] in by_id:
            problems.append(f'{where}: duplicate session_id')
        if not _is_date(session.get('date')):
            problems.append(f'{where}: date must be YYYY-MM-DD — the time filter '
                            'and every recency score read it as a number')
        messages = session.get('messages')
        if not isinstance(messages, list) or not messages:
            problems.append(f'{where}: messages must be a non-empty list')
            messages = []
        for j, message in enumerate(messages):
            if not isinstance(message, dict) or not (message.get('content') or '').strip():
                problems.append(f'{where}: messages[{j}] has no content')
            elif message.get('role') not in ('user', 'assistant'):
                problems.append(f'{where}: messages[{j}].role must be "user" or '
                                '"assistant"')
        if session.get('session_id'):
            by_id[session['session_id']] = session

    questions = payload.get('questions')
    if not isinstance(questions, list) or not questions:
        problems.append('"questions" must be a non-empty list — a corpus with no '
                        'ground truth cannot be measured against')
        questions = []
    seen: set[str] = set()
    for i, question in enumerate(questions):
        if not isinstance(question, dict):
            problems.append(f'questions[{i}]: each question must be an object')
            continue
        where = question.get('id') or f'questions[{i}]'
        if not question.get('id'):
            problems.append(f'questions[{i}]: id is required — it is what a run '
                            'records to say which questions it scored')
        elif question['id'] in seen:
            problems.append(f'{where}: duplicate question id')
        seen.add(question.get('id', ''))
        if not (question.get('question') or question.get('question_fa') or '').strip():
            problems.append(f'{where}: question is empty')
        if question.get('type') not in TYPES:
            problems.append(f'{where}: type {question.get("type")!r} is not one '
                            f'of {", ".join(TYPES)} — the panel filters a run by '
                            'type and reports a breakdown per type')
        if question.get('difficulty') not in DIFFICULTIES:
            problems.append(f'{where}: difficulty must be one of '
                            f'{", ".join(DIFFICULTIES)} — a balanced sample is '
                            'drawn from those three bands')
        answerable = bool(question.get('answerable', True))
        evidence = question.get('evidence') or []
        if answerable and not evidence:
            problems.append(f'{where}: an answerable question needs evidence — '
                            'recall is measured against it')
        if answerable and not (question.get('answer')
                               or question.get('answer_fa') or '').strip():
            problems.append(f'{where}: an answerable question needs an answer')
        problems.extend(_evidence_problems(where, evidence, by_id))
    return problems


def _evidence_problems(where: str, evidence: list, by_id: dict) -> list[str]:
    problems = []
    for k, ev in enumerate(evidence):
        if not isinstance(ev, dict):
            problems.append(f'{where}: evidence[{k}] must be an object')
            continue
        session = by_id.get(ev.get('session_id', ''))
        if session is None:
            problems.append(f'{where}: evidence[{k}] cites session '
                            f'{ev.get("session_id")!r}, which this corpus does '
                            'not contain')
            continue
        messages = session.get('messages') or []
        indices = ev.get('message_indices') or []
        if not indices:
            problems.append(f'{where}: evidence[{k}] names no message_indices')
        bad = [n for n in indices
               if not isinstance(n, int) or not 0 <= n < len(messages)]
        if bad:
            problems.append(f'{where}: evidence[{k}] message_indices {bad} are '
                            f'outside session {ev.get("session_id")!r}, which has '
                            f'{len(messages)} messages')
        quote = (ev.get('quote') or '').strip()
        if not quote:
            problems.append(f'{where}: evidence[{k}] has no quote — the quote is '
                            'what quote recall and every span highlight are '
                            'measured against')
            continue
        cited = ' \n'.join((messages[n].get('content') or '')
                           for n in indices
                           if isinstance(n, int) and 0 <= n < len(messages))
        if quote not in cited:
            problems.append(
                f'{where}: evidence[{k}] quotes text that is not in the messages '
                'it cites — a quote must be verbatim, or every lexical score in '
                'this lab is measured against something the corpus never said')
    return problems


def _is_date(value) -> bool:
    return bool(isinstance(value, str)
                and re.match(r'^\d{4}-\d{2}-\d{2}$', value))


def import_dataset(payload: dict) -> Dataset:
    """Validate and write one dataset into the imported directory.

    Refuses rather than repairs. A corpus is somebody's material and the lab's
    job is to say precisely what is wrong with it, not to guess what was meant —
    a silently repaired dataset measures something nobody described."""
    problems = validate(payload)
    if problems:
        raise ValueError('; '.join(problems))
    dataset_id = payload['dataset']['id']
    if dataset_id == BUILTIN:
        raise ValueError(
            f'{BUILTIN!r} is the built-in corpus and cannot be replaced — every '
            'run already recorded was measured against it. Give this one its '
            'own id.')
    folder = imported_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{dataset_id}.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding='utf-8')
    forget()
    return describe(payload, 'imported', path)
