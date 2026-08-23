"""Rebuild the best archive each recorded experiment's own evidence supports —
and say, per row, exactly what could not be carried.

Nothing finished may be lost. An experiment that ran to `done` becomes an
archive; the only question is which rung of the format's ladder
(`tests/archive_examples`) its evidence reaches.

Two kinds of row are excluded rather than archived, and neither is a failure.
Unfinished work — `cancelled` or `error` — has no finding in it to preserve. A
row whose recorded knobs this lab's vocabulary no longer serves is left alone
by policy: the project's rule is that a knob this installation cannot serve is
a refusal, never a substitution, and settling a retired value to today's
default would put a knob surface on the archive that the row never had. Which
values count as retired is read off `option_vocabularies` itself
(`DEAD_KNOB_CHECKS`), so a knob retired next year excludes its rows by having
been retired.

An archive written today is assembled while the run is still in memory
(`panel_server.start_evaluation` hands `archive_evidence` to the browser), so it
carries the four things the format calls evidence: the corpus with its ground
truth, the chunks the chunker produced, the summaries a hierarchy wrote, and one
retrieval trace per question. What was recorded before that route existed varies
by row, and the rung follows the evidence rather than the job's kind:

* an evaluation's ledger `detail` holds the canonical result *and its traces* —
  and, since `HEAVY` strips it, never `chunks_by_session`;
* a `.runs/` file holds the canonical result and nothing else: no traces, no
  chunks, and on the oldest files no selection either;
* an index build recorded its *statistics* — the collection, the chunk count,
  the character percentiles, the embedding width — which are the finding of an
  index build and are not thrown away for want of rows;
* a retrieval recorded one trace per question it was given, and a single query
  recorded one trace and the answer it wrote;
* no record ever stored the corpus.

The corpus is loaded back by id (`dataset_import_contract.load`), the summaries
come off the ledger row, and the chunks are *replayed*: the recorded index config
is handed to the lab's own `chunk_session`, which is deterministic, so the ids
and the text it produces are the ids and the text the run saw — or they are not,
and then the traces come off the archive.

That is the whole discipline here, and it is a demotion rule, never a repair
rule. Every chunk id a stored trace candidate names must resolve to a rebuilt
chunk (or to a stored summary) with byte-equal text. A corpus edited since the
run, a chunker whose embedder this installation cannot load, a hierarchy whose
summaries were never stored — each makes the replay disagree with the recording,
and a row that disagrees keeps its rows and its judged metrics and loses its
traces, with the disagreement named. Evidence is dropped downwards; it is never
invented upwards. Nothing here writes to the ledger, to `.runs/`, or to a
dataset file; it reads and returns.

`build(experiment_id)` returns the highest-rung archive that experiment's
evidence supports, or `None` when the experiment is unknown, excluded, or
unreadable; `reason()` says what was lost on the way, or why there is no
archive; `survey()` walks every board row and reports the pile and rung
breakdown.
"""
from __future__ import annotations

import copy
import json

from raglab.configuration import option_vocabularies as vocabularies
from raglab.configuration.lab_config import IndexConfig, LabConfig
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import experiment_archive as archive
from raglab.evaluation import leaderboard
from raglab.evaluation import run_evaluation as runs
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.llm_backends import model_role_catalogue as models
from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.rag_components.indexing.chunking_strategies import chunk_session


# The one chunker whose output depends on a model: `_semantic_segments` embeds
# the messages of a session to find where the subject changes. Every other
# strategy is pure text, so the replay below builds an embedder only when this
# is the recorded chunker — a rebuild must not load a 2 GB checkpoint to cut a
# session into whole-session chunks.
EMBEDDING_CHUNKERS = ('semantic-drift',)

# The fields that decide what the chunker emits, and therefore the cache key for
# a replay. Deliberately not `IndexConfig.fingerprint()`: that hash also covers
# the hierarchy, which this module never replays — it reads the summaries the
# run stored — so two experiments differing only in their grouping share one
# chunking and must share one replay. The embedder pair is in the key only when
# the chunker actually consults it (`_leaf_key`), for the same reason: two
# `session` builds on different embedders cut identically, and replaying the
# diary twice to prove it is minutes spent on a foregone conclusion.
LEAF_FIELDS = ('dataset', 'chunker', 'chunk_chars', 'overlap', 'contextual',
               'embedder', 'embed_model')

# The panel's own control state was never recorded on a row, and this module
# does not guess it. `mode` is the backend *override* dropdown and '' is its
# "whatever the lab is configured for" value — the honest reading of a row that
# never stored one; what actually answered is recorded, from the ledger's own
# `provider` column, in `evaluation.execution.provider` where the format puts
# it. The other three are read back off the result: the judging mode and the
# sample size are properties of the run, not of the panel.
RAGAS_MODES = ('off', 'offline', 'llm')

# A job that did not finish has no finding to preserve. These are excluded and
# counted, never reported as archives that failed.
UNFINISHED = ('cancelled', 'error')

# What an index build wrote down about itself. `Jobs.run` stores a build's
# result flat on the detail rather than under an `index` key, so these are read
# by name; every one of them is a measurement of the build and belongs in
# `evaluation.result.index`, which is what `stage_results` projects into the
# index stage's `statistics`.
INDEX_STATISTIC_FIELDS = ('collection', 'chunks', 'leaves', 'avg_chars',
                          'p95_chars', 'embed_dim', 'build_seconds', 'reused',
                          'hierarchy')

# The rungs, weakest first along the spine, with the branch last — the names
# `archive_examples` gives them, so a survey and the ladder describe one shape
# with one word.
RUNGS = ('settings', 'indexed', 'retrieved', 'generated',
         'scored-without-traces')

# The ten knobs whose value has to come out of a vocabulary, and the vocabulary
# each answers to — the same ten `LabConfig.validate()` checks, read from the
# module that defines them. A recorded value outside its vocabulary is a knob
# this lab has retired, and the row that names it is excluded (`_dead_knobs`).
# Listed against `option_vocabularies` rather than against the two retired
# values this board happens to carry, so retiring a knob is what excludes its
# rows: nothing here has to be edited when a vocabulary next moves on.
DEAD_KNOB_CHECKS = (
    ('index', 'chunker', vocabularies.CHUNKERS),
    ('index', 'embedder', vocabularies.EMBEDDERS),
    ('index', 'hierarchy', vocabularies.HIERARCHIES),
    ('index', 'graph_source', vocabularies.GRAPH_SOURCES),
    ('index', 'summarizer', vocabularies.SUMMARIZERS),
    ('retrieval', 'summary_scope', vocabularies.SUMMARY_SCOPES),
    ('retrieval', 'retriever', vocabularies.RETRIEVERS),
    ('retrieval', 'reranker', vocabularies.RERANKERS),
    ('retrieval', 'grader', vocabularies.GRADERS),
    ('generation', 'answerer', vocabularies.ANSWERERS),
)


class _Refused(Exception):
    """One reason an experiment cannot be archived at all, raised where found."""


class _Unfinished(Exception):
    """The job never finished, so there is no finding to preserve."""


class _DeadKnob(Exception):
    """The row names a knob value this lab's vocabulary no longer serves."""


def _dead_knobs(cfg: LabConfig) -> list[str]:
    """Every recorded knob whose value is outside the vocabulary that owns it.

    Read from `option_vocabularies` rather than from a list of the values this
    board happens to carry, and from the same ten vocabulary-backed fields
    `LabConfig.validate()` checks — so a knob retired next year excludes its
    rows by having been retired, not by anyone remembering to add it here.

    Each is returned spelled `group.field='value'`, which is what the exclusion
    reason is built from: a reader has to be able to see which knob, and which
    value, took the row off the board.
    """
    found = []
    for group, field, vocabulary in DEAD_KNOB_CHECKS:
        value = getattr(getattr(cfg, group), field)
        if value not in vocabulary:
            found.append(f'{group}.{field}={value!r}')
    return found


def _config(recorded: dict) -> tuple[LabConfig, dict]:
    """The recorded config as the dataclasses read it, and its canonical JSON.

    `validate_archive` demands a complete, normalized `LabConfig` — the archive
    has to be its own fixed point — so a config recorded before a knob existed
    is rebuilt through `LabConfig.from_dict`, which fills the missing knob with
    the default it has always had. That is a widening, not an edit: no recorded
    value is changed, and `settings.config` and `result.config` are then the
    same object, which is the format's own requirement.

    A knob whose recorded value this lab's vocabulary no longer serves ends the
    experiment here, and it is an *exclusion* rather than a repair or a failure.
    The rule it follows is the project's own: a knob this installation cannot
    serve is a refusal, never a substitution. Settling such a value to today's
    default — even one that named a stage the run never built, so that nothing
    read it — is a mild substitution, and it would put a value on the knob
    surface that the row never had. There is no honest archive at any rung for
    a row whose knobs cannot be stated, so the row is excluded by policy and
    named, and nothing about it is rewritten.
    """
    try:
        cfg = LabConfig.from_dict(recorded or {})
    except (TypeError, ValueError, OverflowError) as error:
        raise _Refused(f'the recorded config will not rebuild: {error}')
    dead = _dead_knobs(cfg)
    if dead:
        raise _DeadKnob(
            'this lab no longer serves the recorded ' + ', '.join(dead)
            + ' — a knob this installation cannot serve is a refusal, never a '
            'substitution, so the row is excluded rather than rewritten')
    problems = cfg.validate()
    if problems:
        raise _Refused('this installation refuses the recorded config: '
                       + '; '.join(problems))
    try:
        canonical = json.loads(json.dumps(cfg.to_dict(), allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise _Refused(f'the recorded config will not serialize: {error}')
    return cfg, canonical


def _leaf_key(index: IndexConfig) -> tuple:
    normalized = index.normalized()
    key = {field: getattr(normalized, field) for field in LEAF_FIELDS}
    if normalized.chunker not in EMBEDDING_CHUNKERS:
        key['embedder'] = key['embed_model'] = ''
    return tuple(key[field] for field in LEAF_FIELDS)


def _embedder(index: IndexConfig, factory):
    """The embedder the recorded chunker needs, or none if it needs none."""
    if index.chunker not in EMBEDDING_CHUNKERS:
        return None
    make = factory or embedding.make_embedder
    try:
        return make(index.embedder, None, index.embed_model)
    except Exception as error:              # a missing extra, absent weights
        raise _Refused(
            f'chunker {index.chunker!r} needs the {index.embedder!r} embedder '
            f'to reproduce its cuts, and this installation cannot load it: '
            f'{type(error).__name__}: {error}')


def replay_chunks(index: IndexConfig, corpus: dict, *,
                  embedder_factory=None) -> list[dict]:
    """The chunks this index config produced, grouped by session.

    The lab's own chunker over the lab's own corpus, in the shape
    `service_presentation.chunks_by_session` writes — one group per session that
    produced a leaf, in corpus order, each chunk as `{'id', 'text'}`. Summaries
    are not replayed: a grouping is not deterministic in the way a chunker is,
    and the run recorded its summary rows, so those are read rather than redone.
    """
    embedder = _embedder(index, embedder_factory)
    groups = []
    for session in corpus.get('sessions') or []:
        try:
            leaves = [chunk for chunk in chunk_session(session, index, embedder)
                      if chunk.layer != 'summary']
        except Exception as error:
            raise _Refused(
                f'replaying the chunker over session '
                f'{session.get("session_id", "?")!r} failed: '
                f'{type(error).__name__}: {error}')
        if not leaves:
            continue
        groups.append({'session_id': leaves[0].session_id,
                       'date': leaves[0].date,
                       'chunks': [{'id': chunk.id, 'text': chunk.text}
                                  for chunk in leaves]})
    return groups


def _sources(chunks_by_session: list[dict], summaries: list[dict]) -> dict:
    found = {}
    for group in chunks_by_session:
        for chunk in group['chunks']:
            found[chunk['id']] = chunk['text']
    for summary in summaries:
        if isinstance(summary, dict) and isinstance(summary.get('id'), str):
            found.setdefault(summary['id'], summary.get('text'))
    return found


def _check_traces(traces: list, sources: dict) -> None:
    """Every candidate a trace recorded must be a row the rebuild can show.

    `validate_archive` enforces this too, and refusing here as well is not
    duplication: this is the check the whole exercise turns on, and a reason
    naming the chunk that disagreed is what tells "the corpus moved" apart from
    "the archive is malformed". Failing it costs the traces, never the rows.
    """
    for trace_row in traces:
        candidates = ((trace_row or {}).get('trace') or {}).get('candidates')
        for candidate in candidates or []:
            chunk_id = (candidate or {}).get('chunk_id')
            if chunk_id not in sources:
                raise _Refused(
                    f'chunk {chunk_id!r}, retrieved for question '
                    f'{(trace_row or {}).get("question_id")!r}, is not in the '
                    'rebuilt index — the corpus or the chunker moved since the '
                    'run')
            if candidate.get('text') != sources[chunk_id]:
                raise _Refused(
                    f'chunk {chunk_id!r} was rebuilt with different text than '
                    'the run recorded — the corpus moved since the run')


# --- reading what each kind of record actually wrote down -------------------
# Read by evidence, not by the job's kind. A row's `kind` column says what was
# asked for; these say what came back, and it is what came back that decides
# how high the archive goes.

def _evaluation_source(detail: dict, run: dict) -> dict:
    """The record holding the canonical result: the ledger's, else the run file.

    The ledger's detail wins where both exist, because it is the only one that
    ever carried traces — a run file has no place for one. A `.runs/` file with
    no ledger row behind it is the other 166 experiments on this board, and it
    is a complete canonical result minus its evidence. A job that produced no
    result at all — a build, a retrieval, a single query — falls back to its own
    detail, which is where everything it *did* write down lives.
    """
    if isinstance(detail.get('rows'), list):
        return detail
    if isinstance(run.get('rows'), list):
        return run
    return detail or run


def _recorded_traces(detail: dict) -> list[dict]:
    """One trace per question, in the shape the format holds them, or none.

    Three recordings, one shape. An evaluation wrote `traces` already in it. A
    retrieval wrote `questions`, whose entries are that same row — a
    `question_id` beside a `trace` with its ranked `candidates` — under another
    name. A single query wrote one trace and the id of the question it was
    asked, which is a list of one. Nothing is reshaped beyond the renaming;
    a candidate is carried exactly as it was stored, and if what it stored is
    not what the format holds, `validate_archive` refuses it and the traces come
    off (`_archive`).
    """
    traces = detail.get('traces')
    if isinstance(traces, list) and traces:
        return copy.deepcopy(traces)
    questions = detail.get('questions')
    if isinstance(questions, list) and questions:
        return [copy.deepcopy(question) for question in questions
                if isinstance(question, dict) and question.get('question_id')
                and isinstance(question.get('trace'), dict)]
    trace = detail.get('trace')
    if isinstance(trace, dict) and detail.get('question_id'):
        return [{'question_id': detail['question_id'],
                 'trace': copy.deepcopy(trace)}]
    return []


# The per-question facts a retrieval wrote beside each trace. Identity, not
# measurement: the question's id, what kind it is, how hard it is, whether it is
# answerable, and how much gold evidence existed for it. Every one is copied
# from the record; a retrieval computed no metric, so the row it becomes carries
# none, and a reader sees an unmeasured question rather than a scored one.
_RETRIEVAL_ROW_FIELDS = ('type', 'difficulty', 'answerable', 'gold_available')

# What a single query wrote about the one question it answered.
_QUERY_ROW_FIELDS = ('answer', 'abstained')


def _recorded_rows(source: dict, detail: dict) -> list[dict]:
    """The per-question rows, read from whichever record holds them."""
    rows = source.get('rows')
    if isinstance(rows, list):
        return copy.deepcopy(rows)
    questions = detail.get('questions')
    if isinstance(questions, list) and questions:
        out = []
        for question in questions:
            if not isinstance(question, dict) or not question.get('question_id'):
                continue
            row = {'id': question['question_id']}
            row.update({field: question[field]
                        for field in _RETRIEVAL_ROW_FIELDS if field in question})
            out.append(row)
        return out
    if detail.get('question_id') and isinstance(detail.get('trace'), dict):
        row = {'id': detail['question_id']}
        row.update({field: detail[field]
                    for field in _QUERY_ROW_FIELDS if field in detail})
        return [row]
    return []


def _index_statistics(source: dict, detail: dict) -> dict:
    """What the build measured about the index, wherever the record put it."""
    for holder in (source, detail):
        found = holder.get('index')
        if isinstance(found, dict) and found:
            return copy.deepcopy(found)
    stats = {field: detail[field] for field in INDEX_STATISTIC_FIELDS
             if field in detail}
    return copy.deepcopy(stats)


def _selection(recorded, row_ids: list[str]) -> tuple[dict, list[str]]:
    """The selection block, reconciled against the rows it must agree with.

    `question_ids == row_ids` is the one equality the format keeps exact, so
    three cases and no fudge. Recorded ids that match are carried whole.
    Recorded ids that disagree are a record contradicting itself, and the rows
    come off rather than either side being edited. No recorded ids at all —
    every `.runs/` file older than `RunResult.selection` — means the ids are
    read off the rows, which is a restatement of the rows and not a claim about
    sampling: nothing else about the selection is invented, so a reader sees the
    questions that were measured and no balance or limit that was never stored.
    """
    recorded = copy.deepcopy(recorded) if isinstance(recorded, dict) else {}
    ids = recorded.get('question_ids')
    if isinstance(ids, list) and ids:
        if list(ids) != row_ids:
            return {}, [f'the stored rows and the stored selection name '
                        f'different questions ({row_ids!r} against {ids!r}), '
                        'so neither is carried']
        count = recorded.get('n')
        if isinstance(count, int) and not isinstance(count, bool) \
                and count != len(row_ids):
            return {}, [f'the stored selection counts {count} questions but '
                        f'names {len(row_ids)}, so neither is carried']
        recorded['n'] = len(row_ids)
        return recorded, []
    note = ([] if not row_ids else
            ['backfill: no selection was recorded — this experiment predates '
             '`RunResult.selection`, or never had one — so the archived '
             'question ids are read off its own rows and no balance or limit '
             'is claimed'])
    recorded['question_ids'] = list(row_ids)
    recorded['n'] = len(row_ids)
    return recorded, note


def _summary(recorded, count: int) -> tuple[dict, list[str]]:
    """The summary block, tied to the same question count as everything else."""
    recorded = copy.deepcopy(recorded) if isinstance(recorded, dict) else {}
    found = recorded.get('n_questions')
    if isinstance(found, int) and not isinstance(found, bool) and found != count:
        return {}, [f'the stored summary counts {found} questions but the rows '
                    f'name {count}, so the rows are not carried']
    recorded['n_questions'] = count
    return recorded, []


def _result(record: dict, source: dict, detail: dict, canonical: dict,
            dataset_id: str, *, rows: list, selection: dict,
            summary: dict, notes: list[str]) -> dict:
    """The canonical result block, assembled from the record and nothing else.

    Every one of the twelve canonical keys comes from what was written down.
    The exceptions are named where they are made: `dataset` predates the field
    on the oldest rows, and a blank there has exactly one meaning, here and in
    `leaderboard._dataset` alike — the built-in corpus, the only one that
    existed then; and `run_id` falls back to the job id, which is the ledger's
    own spelling of an experiment that produced no run file
    (`service_experiment_ledger.row_for`).
    """
    stored = [note for note in (source.get('notes') or detail.get('notes') or [])
              if isinstance(note, str)]
    return {
        'run_id': (source.get('run_id') or record.get('experiment_id') or ''),
        'label': (source.get('label') or record.get('label')
                  or canonical.get('label') or ''),
        'config': copy.deepcopy(canonical),
        'dataset': dataset_id,
        'index': _index_statistics(source, detail),
        'summary': summary,
        'rows': rows,
        'ragas': copy.deepcopy(source.get('ragas') or {}),
        'seconds': source.get('seconds') or record.get('seconds') or 0.0,
        'started_at': (source.get('started_at') or record.get('started_at')
                       or ''),
        'notes': stored + list(notes),
        'selection': selection,
    }


def _ui(source: dict, rows: list) -> dict:
    """The panel controls, read off the run rather than guessed.

    Only `ragas_mode`, `limit`, `ragas_limit` and `types` are recoverable, and
    each is read from the place the run wrote it. `mode` is left at '' — the
    dropdown's own "use the configured backend" value — because no row ever
    stored the override, and the backend that actually answered is recorded in
    `evaluation.execution.provider` instead of being invented here.
    """
    ragas = source.get('ragas') or {}
    mode = ragas.get('mode') if ragas.get('mode') in RAGAS_MODES else 'off'
    selection = source.get('selection') or {}
    limit = selection.get('limit')
    judged = ragas.get('n_samples')
    types = sorted({row.get('type') for row in rows
                    if row.get('type') in datasets.TYPES})
    return {
        'mode': '',
        'ragas_mode': mode,
        'limit': int(limit) if isinstance(limit, int) and 0 <= limit <= 200 else 0,
        'ragas_limit': (int(judged) if isinstance(judged, int)
                        and 0 <= judged <= 200 else 0),
        'types': types,
    }


def _execution(record: dict, cfg: LabConfig, source: dict,
               detail: dict) -> dict:
    """Who ran it and with which models — recorded values only.

    A retrieval and a single query wrote the resolved role table down
    (`detail['models']`), which is better evidence than the config: it is what
    the lab actually asked for, blanks already filled in. An evaluation did not,
    so its roles are read off its own config — and a blank role is *dropped*
    rather than filled with today's default, because the setting that would have
    filled it was never stored and a row must not name a model it cannot show
    was used. The one role recoverable beyond the config either way is the RAGAS
    judge, which the run itself wrote down.
    """
    recorded = detail.get('models')
    if isinstance(recorded, dict) and recorded:
        named = {key: value for key, value in recorded.items()
                 if isinstance(key, str) and isinstance(value, str) and value}
    else:
        picked = {role.key: models.chosen(cfg, role) for role in models.ROLES}
        named = {key: value for key, value in picked.items() if value}
    judge = (source.get('ragas') or {}).get('judge') or {}
    if judge.get('model'):
        named['ragas'] = judge['model']
    return {'provider': record.get('provider') or judge.get('provider') or '',
            'models': named}


def _metric_catalogue() -> list[dict]:
    """Every metric the panel can print, which is what the live export stores.

    Imported here rather than at module scope: `explainer_assembly` reaches
    `deterministic_metrics`, which reaches `raglab.corpora`, which reaches
    `deterministic_metrics` again — a cycle that only resolves when something
    else has imported the corpora package first. A module-level import would
    make this file the one that trips it.
    """
    from raglab.configuration import explainer_assembly as explain
    return explain.measures()


def _corpus(dataset_id: str) -> tuple[dict, dict]:
    try:
        return datasets.load(dataset_id)
    except Exception as error:
        raise _Refused(f'corpus {dataset_id!r} is not installed here: {error}')


def rung(value: dict) -> str:
    """Which rung of the ladder an archive is standing on, read off its contents.

    The same reading `archive_examples.contents` takes, in one word: no
    evaluation is `settings`; an evaluation with no rows is `indexed`, whatever
    else it carries; rows with no trace behind them are `scored-without-traces`;
    rows with traces are `generated` once a judge has graded them and
    `retrieved` before that.
    """
    evaluation = value.get('evaluation')
    if not isinstance(evaluation, dict):
        return 'settings'
    result = evaluation.get('result') or {}
    rows = result.get('rows') or []
    traces = (evaluation.get('inspector') or {}).get('traces') or []
    if not rows:
        return 'indexed'
    if not traces:
        return 'scored-without-traces'
    return 'generated' if (result.get('ragas') or {}).get('metrics') \
        else 'retrieved'


def _settings_archive(canonical: dict, ui: dict) -> dict:
    return archive.validate_archive({
        'format': archive.FORMAT, 'version': archive.VERSION,
        'settings': {'config': copy.deepcopy(canonical), 'ui': copy.deepcopy(ui)},
    })


def _archive(record: dict, row: dict | None, run: dict | None, *,
             chunk_cache: dict | None = None, embedder_factory=None) -> dict:
    """One recorded experiment → the highest-rung archive its evidence supports.

    Returns `{'experiment_id', 'kind', 'state', 'rung', 'archive', 'notes'}`.
    `notes` is what could not be carried and why — empty when nothing was lost.
    Raises `_Unfinished` for work that never finished and `_DeadKnob` for a row
    naming a knob value this lab has retired — the two deliberate exclusions —
    and `_Refused` for a record this installation cannot read at all, which is
    the only one of the three that is a finding rather than a policy.
    """
    row = row or {}
    run = run or {}
    state = record.get('state') or row.get('state') or ''
    if state in UNFINISHED:
        raise _Unfinished(f'the job did not finish: state {state}')

    detail = row.get('detail')
    if not isinstance(detail, dict) or detail.get('unreadable'):
        detail = {}
    if detail.get('format') == archive.FORMAT:
        # An imported archive is its own record, kept verbatim: it is whatever
        # it was when it arrived, and nothing here rebuilds any part of it.
        kept = archive.validate_archive(copy.deepcopy(detail))
        return {'experiment_id': record.get('experiment_id') or '',
                'kind': record.get('kind') or '', 'state': state,
                'rung': rung(kept), 'archive': kept, 'notes': []}

    source = _evaluation_source(detail, run)
    cfg, canonical = _config(
        detail.get('config') or source.get('config') or run.get('config') or {})
    notes: list[str] = []
    dataset_id = (cfg.index.dataset or source.get('dataset')
                  or detail.get('dataset') or record.get('dataset')
                  or datasets.BUILTIN)

    rows = _recorded_rows(source, detail)
    row_ids = [item.get('id') for item in rows]
    # An empty block back from either reconciler is the record contradicting
    # itself, and the rows come off; a note beside a block that *is* there is
    # provenance about how it was read, and costs nothing.
    selection, why = _selection(source.get('selection'), row_ids)
    notes.extend(why)
    if not selection:
        rows, row_ids = [], []
        selection, _ = _selection({}, [])
    summary, why = _summary(source.get('summary'), len(rows))
    notes.extend(why)
    if not summary:
        rows, row_ids = [], []
        selection, _ = _selection({}, [])
        summary, _ = _summary(None, 0)

    ui = _ui(source, rows)

    try:
        corpus, ground_truth = _corpus(dataset_id)
    except _Refused as refusal:
        # The corpus is the frame every other piece of evidence is checked
        # against — a row id has to be a question in it, a chunk id a chunk of
        # it — so without it the knob surface is all that can honestly stand.
        notes.append(str(refusal))
        notes.append('without the corpus it ran on, nothing but the knob '
                     'surface can be archived')
        return {'experiment_id': record.get('experiment_id') or '',
                'kind': record.get('kind') or '', 'state': state,
                'rung': 'settings', 'archive': _settings_archive(canonical, ui),
                'notes': notes}

    cache = chunk_cache if chunk_cache is not None else {}
    key = _leaf_key(cfg.index)
    if key not in cache:
        try:
            cache[key] = replay_chunks(cfg.index.normalized(), corpus,
                                       embedder_factory=embedder_factory)
        except _Refused as refusal:
            cache[key] = refusal
    chunks_by_session = cache[key]
    if isinstance(chunks_by_session, _Refused):
        notes.append(str(chunks_by_session))
        chunks_by_session = []

    summaries = detail.get('summaries')
    if not isinstance(summaries, list):
        if cfg.index.hierarchy:
            notes.append(
                f'hierarchy {cfg.index.hierarchy!r} was built but its summary '
                'rows were never stored, so the archive shows the leaves only')
        summaries = []

    traces = _recorded_traces(detail)
    if traces:
        try:
            _check_traces(traces, _sources(chunks_by_session, summaries))
        except _Refused as refusal:
            notes.append(str(refusal))
            traces = []
    if traces and not rows:
        # The format holds traces as a subset of the rows, so a trace with no
        # row under it could only be carried by inventing the row — which is
        # the measurement, and the one thing that is never invented.
        notes.append('the recorded traces have no rows under them, and a row '
                     'is a measurement rather than something to invent, so the '
                     'traces cannot be carried')
        traces = []

    evidence = {
        'execution': _execution(record, cfg, source, detail),
        'metric_catalogue': _metric_catalogue(),
        'inspector': {
            'dataset': {'id': dataset_id, 'corpus': corpus,
                        'ground_truth': ground_truth},
            'chunks_by_session': copy.deepcopy(chunks_by_session),
            'summaries': copy.deepcopy(summaries),
            'traces': traces,
        },
    }
    settings = {'config': copy.deepcopy(canonical), 'ui': ui}

    # The demotion ladder, walked once. Each step drops the evidence the step
    # above could not stand on — never repairs it — and names what it dropped.
    attempts = [(rows, selection, summary, traces, '')]
    if traces:
        attempts.append((rows, selection, summary, [],
                         'the archive does not validate with its traces'))
    if rows:
        empty_selection, _ = _selection({}, [])
        empty_summary, _ = _summary(None, 0)
        attempts.append(([], empty_selection, empty_summary, [],
                         'the archive does not validate with its rows'))
    for attempt_rows, attempt_selection, attempt_summary, attempt_traces, note \
            in attempts:
        evidence['inspector']['traces'] = attempt_traces
        result = _result(record, source, detail, canonical, dataset_id,
                         rows=attempt_rows, selection=attempt_selection,
                         summary=attempt_summary, notes=notes)
        try:
            built = archive.build_completed(settings, result, evidence)
        except archive.ArchiveError as error:
            if note:
                notes.append(f'{note}: {error}')
            else:
                notes.append(f'the rebuilt archive does not validate: {error}')
            continue
        return {'experiment_id': record.get('experiment_id') or '',
                'kind': record.get('kind') or '', 'state': state,
                'rung': rung(built), 'archive': built, 'notes': notes}

    return {'experiment_id': record.get('experiment_id') or '',
            'kind': record.get('kind') or '', 'state': state,
            'rung': 'settings', 'archive': _settings_archive(canonical, ui),
            'notes': notes}


def _record(experiment_id: str, *, db_path=None) -> tuple:
    """One experiment as both durable records describe it, plus each record."""
    row = ledger.experiment(experiment_id, path=db_path)
    run = runs.load_run(experiment_id)
    if row is None and run is None:
        return None, None, None
    return leaderboard.experiment_record(row, run), row, run


# Which pile an experiment lands in, named once so the single-row reader and
# the survey cannot come to disagree. Two of the four are exclusions and one is
# a finding, which is the distinction worth keeping in the vocabulary: work that
# never finished has nothing to preserve, a row on a retired knob is left alone
# by policy, and a record this installation cannot read is a failure that wants
# looking at.
PILES = ('archived', 'excluded-unfinished', 'excluded-dead-knob', 'failed')


def describe(experiment_id: str, *, db_path=None) -> dict | None:
    """One experiment's archive, its rung and its pile — or `None`.

    `None` only when nothing on this board answers to that id. Everything else
    comes back with a `pile` off `PILES`: `archived` carries the archive and
    the notes saying what it could not hold; the two exclusions and the failure
    carry `reason` instead and no archive.
    """
    record, row, run = _record(experiment_id, db_path=db_path)
    if record is None:
        return None
    empty = {'experiment_id': experiment_id, 'kind': record.get('kind') or '',
             'state': record.get('state') or '', 'rung': '', 'archive': None,
             'notes': []}
    try:
        found = _archive(record, row, run)
    except _Unfinished as unfinished:
        return empty | {'pile': 'excluded-unfinished', 'reason': str(unfinished)}
    except _DeadKnob as retired:
        return empty | {'pile': 'excluded-dead-knob', 'reason': str(retired)}
    except (_Refused, archive.ArchiveError) as refusal:
        return empty | {'pile': 'failed', 'reason': str(refusal)}
    return found | {'pile': 'archived', 'reason': ''}


def build(experiment_id: str, *, db_path=None) -> dict | None:
    """The highest-rung archive this experiment's evidence supports.

    `None` for an id nothing on the board answers to, for work that never
    finished, for a row naming a knob this lab has retired, and for a record
    this installation cannot read — the four cases where there is no honest
    archive to return. Everything else comes back as an archive, at whatever
    rung its evidence reaches; `reason()` says what that evidence could not
    carry, or why there is no archive.
    """
    found = describe(experiment_id, db_path=db_path)
    return None if found is None else found.get('archive')


def reason(experiment_id: str, *, db_path=None) -> str:
    """What this experiment's archive could not carry, or '' when nothing was lost."""
    found = describe(experiment_id, db_path=db_path)
    if found is None:
        return 'no ledger row and no `.runs/` file answers to that id'
    if found['pile'] != 'archived':
        return found['reason']
    return '; '.join(found['notes'])


def _encoded_bytes(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False,
                          allow_nan=False).encode('utf-8'))


def survey(*, db_path=None, limit: int = 500) -> dict:
    """Walk every board row and report which pile — and which rung — it lands in.

    The board is the population on purpose: it is what a reader sees, and it is
    the union of both durable records, so an evaluation the ledger never saw is
    surveyed as the experiment it is rather than skipped as a row that does not
    exist. One replay is shared by every row that chunked the same way, which is
    what keeps a survey over a model-backed chunker affordable.

    Four piles and no fifth, and the difference between them is the point.
    `archived` carries a rung off the ladder and a size. The two exclusions are
    deliberate and are not failures: `excluded_unfinished` is work that never
    reached a terminal `done`, so there is no finding in it, and
    `excluded_dead_knob` is a row whose recorded knobs this lab's vocabulary no
    longer serves, each naming the knob and the value that excluded it.
    `failed` is what is left — a record this installation could not read at all
    — and it is a finding, so every entry names why.
    """
    rows = leaderboard.board_rows(limit=limit, db_path=db_path)
    ledger_rows = {row['experiment_id']: row
                   for row in ledger.experiments(limit=limit, path=db_path)}
    cache: dict = {}
    archived: list[dict] = []
    unfinished: list[dict] = []
    retired: list[dict] = []
    failed: list[dict] = []
    total_bytes = 0
    for record in rows:
        experiment_id = record.get('experiment_id') or ''
        ledger_row = (ledger.experiment(experiment_id, path=db_path)
                      if experiment_id in ledger_rows else None)
        run = runs.load_run(experiment_id)
        excluded = {'experiment_id': experiment_id,
                    'kind': record.get('kind') or '',
                    'state': record.get('state') or ''}
        try:
            found = _archive(record, ledger_row, run, chunk_cache=cache)
        except _Unfinished as reason_given:
            unfinished.append(excluded | {'reason': str(reason_given)})
            continue
        except _DeadKnob as reason_given:
            retired.append(excluded | {'reason': str(reason_given),
                                       'knobs': _recorded_dead_knobs(
                                           ledger_row, run)})
            continue
        except (_Refused, archive.ArchiveError) as reason_given:
            failed.append(excluded | {'reason': str(reason_given)})
            continue
        size = _encoded_bytes(found['archive'])
        total_bytes += size
        archived.append({'experiment_id': experiment_id,
                         'kind': found['kind'], 'rung': found['rung'],
                         'bytes': size, 'notes': found['notes']})
    rungs = {name: sum(1 for item in archived if item['rung'] == name)
             for name in RUNGS}
    return {
        'archived': archived,
        'excluded_unfinished': unfinished,
        'excluded_dead_knob': retired,
        'failed': failed,
        'rungs': rungs,
        'counts': {'board_rows': len(rows), 'archived': len(archived),
                   'excluded_unfinished': len(unfinished),
                   'excluded_dead_knob': len(retired), 'failed': len(failed),
                   'bytes': total_bytes},
    }


def _recorded_dead_knobs(row: dict | None, run: dict | None) -> list[str]:
    """The knobs that excluded one row, listed for a reader who has to decide
    what to do about them — restore the vocabulary entry, or delete the row."""
    detail = (row or {}).get('detail')
    detail = detail if isinstance(detail, dict) else {}
    recorded = detail.get('config') or (run or {}).get('config') or {}
    try:
        return _dead_knobs(LabConfig.from_dict(recorded))
    except (TypeError, ValueError, OverflowError):
        return []
