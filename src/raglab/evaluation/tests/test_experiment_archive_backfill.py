"""Backfilling an archive recovers the evidence, or demotes — never invents it.

Nothing finished may be lost, so the question a backfill answers is not "can
this experiment be exported whole" but "which rung of the ladder does its own
evidence reach". Every claim below is one half of that: what each kind of
record actually recorded is carried up as far as it goes, and every piece the
rebuild cannot stand behind is dropped downwards with a reason, never filled in.

The honesty gate is the middle of it. A historical row stores its traces but
never its chunks, so the chunks have to be replayed from the corpus; a replay
that disagrees with what the traces recorded means the corpus or the chunker
moved since the run, and the archive then keeps the rows it measured and loses
the traces it cannot show. An archive quietly finished with a chunk nobody can
show was retrieved is a row lying about what produced it.

The fixtures are the ladder's (`archive_examples`): its shifted config, its
summary rows, its candidate shape. What they are *pointed at* is the real
bundled `smoke-mini` corpus rather than the ladder's own two-session fixture,
because the chunk ids and chunk text under test are the ones the lab's own
chunker produces — a hand-written corpus would exercise the assembly and skip
the replay, which is the half that can be wrong.
"""
import copy
import json

import pytest

from raglab.configuration import option_vocabularies as vocabularies
from raglab.configuration.lab_config import LabConfig
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import experiment_archive as archive
from raglab.evaluation import experiment_archive_backfill as backfill
from raglab.evaluation import leaderboard
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.evaluation.tests import archive_examples as examples


DATASET = examples.DATASET_ID          # 'smoke-mini', the bundled control corpus


def _config() -> dict:
    """The ladder's shifted config, kept as-is: no knob is at its default, and
    its chunker (`fixed-overlap`) reads no model, so the replay stays offline."""
    return copy.deepcopy(examples.SHIFTED_CONFIG)


def _flat_config() -> dict:
    """The same config with no hierarchy, for the records that never built one.

    An index build with a grouping is a different claim from one without, and
    the tests that are about statistics and chunks should not also be about
    summary rows nobody stored.
    """
    config = _config()
    config['index'] = dict(config['index'], hierarchy='')
    return config


def _replayed(config: dict | None = None) -> list[dict]:
    corpus, _ = datasets.load(DATASET)
    index = LabConfig.from_dict(config or _config()).index
    return backfill.replay_chunks(index, corpus)


def _question(offset: int = 0) -> dict:
    _, truth = datasets.load(DATASET)
    return truth['questions'][offset]


def _recorded(run_id: str, *, candidates, ragas: dict | None = None) -> dict:
    """One historical ledger row, written the way `Jobs.run` wrote them.

    Through `ledger.record` rather than a hand-built INSERT, so the detail blob
    under test is the blob the real path produces — `HEAVY` strips the chunks
    here exactly as it stripped them then.
    """
    question = _question()
    result = {
        'run_id': run_id, 'label': 'a run recorded before export existed',
        'config': _config(), 'dataset': DATASET,
        'index': {'collection': 'raglab-backfill', 'chunks': 2, 'leaves': 1},
        'summary': {'overall': {'recall': 1.0}, 'by_type': {},
                    'by_difficulty': {}, 'n_questions': 1},
        'rows': [{'id': question['id'], 'type': question['type'],
                  'difficulty': question['difficulty'], 'recall': 1.0}],
        'ragas': copy.deepcopy(examples.RAGAS) if ragas is None else ragas,
        'seconds': 1.5, 'started_at': '2026-08-14 09:00:00',
        'notes': [], 'selection': {'balance': 'stride', 'limit': 1, 'n': 1,
                                   'by_difficulty': {},
                                   'question_ids': [question['id']]},
        'traces': [{'question_id': question['id'],
                    'question_fa': question['question_fa'],
                    'question_en': question.get('question_en', ''),
                    'type': question['type'],
                    'difficulty': question['difficulty'],
                    'answerable': bool(question.get('answerable')),
                    'gold_available': 1,
                    'trace': {'candidates': list(candidates)}}],
        'summaries': copy.deepcopy(examples.SUMMARIES),
    }
    ledger.record({'id': run_id, 'kind': 'run',
                   'config': _config() | {'provider': 'fake'},
                   'result': result}, 'done')
    return result


def _leaf_candidate(chunk: dict):
    """One retrieved leaf, in the ladder's candidate shape."""
    return examples._candidate(chunk['id'], chunk['text'], session_id='',
                               date='', layer='leaf', gold=False)


def _summary_candidate():
    summary = examples.SUMMARIES[0]
    return examples._candidate(summary['id'], summary['text'], session_id='',
                               date=summary['date'], layer='summary',
                               gold=False)


def _orphan_run(run_id: str, *, config: dict | None = None,
                selection: dict | None = None, ragas: dict | None = None,
                rows: list | None = None) -> None:
    """One `.runs/` file with no ledger row behind it — 166 of the real board.

    A run file has no place at all for a trace or a chunk, which is exactly the
    shape `scored-without-traces` exists for.
    """
    question = _question()
    rows = [{'id': question['id'], 'type': question['type'],
             'difficulty': question['difficulty'], 'recall': 1.0,
             'answer': 'an answer the run wrote', 'abstained': False}] \
        if rows is None else rows
    payload = {
        'run_id': run_id, 'label': 'an orphan run file', 'dataset': DATASET,
        'config': config or _config(),
        'index': {'collection': 'raglab-orphan', 'chunks': 2, 'leaves': 2},
        'summary': {'overall': {'recall': 1.0}, 'by_type': {},
                    'by_difficulty': {}, 'n_questions': len(rows)},
        'rows': rows,
        'ragas': copy.deepcopy(examples.RAGAS) if ragas is None else ragas,
        'seconds': 2.0, 'started_at': '2026-08-15 09:00:00', 'notes': [],
    }
    if selection is not None:
        payload['selection'] = selection
    evaluate.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (evaluate.RUNS_DIR / f'{run_id}.json').write_text(
        json.dumps(payload, ensure_ascii=False), encoding='utf-8')


def test_a_backfilled_archive_carries_every_stage_and_validates():
    # this is an integration test
    """The whole point: an archive rebuilt from a row that never stored its
    chunks passes the same trust boundary an exported one does, with the corpus,
    the chunks, the summaries and the traces all present and agreeing."""
    groups = _replayed()
    chunk = groups[0]['chunks'][0]
    _recorded('backfill-complete-001',
              candidates=[_leaf_candidate(chunk), _summary_candidate()])

    built = backfill.build('backfill-complete-001')
    assert built is not None, backfill.reason('backfill-complete-001')
    assert backfill.reason('backfill-complete-001') == ''
    assert backfill.rung(built) == 'generated'

    # Validated again from a copy: `build` returns what `validate_archive`
    # already accepted, and this says so independently rather than trusting it.
    archive.validate_archive(copy.deepcopy(built))

    carried = examples.contents(built)
    corpus, truth = datasets.load(DATASET)
    assert carried['sessions'] == len(corpus['sessions'])
    assert carried['questions'] == len(truth['questions'])
    assert carried['chunks'] == sum(len(g['chunks']) for g in groups)
    assert carried['summaries'] == len(examples.SUMMARIES)
    assert carried['traces'] == 1 and carried['candidates'] == 2

    # The knobs came back whole, and the backend that answered is recorded
    # where the format puts it rather than guessed at in the UI block.
    assert built['settings']['config'] == json.loads(json.dumps(
        LabConfig.from_dict(_config()).to_dict()))
    assert built['evaluation']['execution']['provider'] == 'fake'
    assert built['settings']['ui']['mode'] == ''


@pytest.mark.parametrize('broken,expected', [
    ('unknown-id', 'is not in the rebuilt index'),
    ('changed-text', 'different text than the run recorded'),
])
def test_a_trace_the_replay_cannot_reproduce_demotes_rather_than_fudging(
        broken, expected):
    # this is an integration test
    """A candidate the rebuilt index cannot show costs the traces, not the run.

    Both halves of the check, because they fail for different reasons a reader
    cares about: an id the chunker no longer emits means the *chunking* moved,
    and matching id with different text means the *corpus* did. Neither may be
    answered by showing a chunk nobody can prove was retrieved — so the archive
    drops to the rung below, keeping the rows that were actually measured, and
    names on the way down which chunk disagreed.
    """
    groups = _replayed()
    chunk = dict(groups[0]['chunks'][0])
    if broken == 'unknown-id':
        chunk['id'] = 'no-such-session:c99'
    else:
        chunk['text'] = chunk['text'] + ' — a word the corpus never said'
    run_id = f'backfill-broken-{broken}'
    recorded = _recorded(run_id, candidates=[_leaf_candidate(chunk)])

    built = backfill.build(run_id)
    assert built is not None
    assert backfill.rung(built) == 'scored-without-traces'
    assert expected in backfill.reason(run_id)

    # The measurement is untouched — only the recording of how retrieval
    # reached it is gone, and the archive says so on its own row.
    carried = examples.contents(built)
    assert carried['traces'] == 0 and carried['candidates'] == 0
    assert built['evaluation']['result']['rows'] == recorded['rows']
    assert any(expected in note
               for note in built['evaluation']['result']['notes'])
    archive.validate_archive(copy.deepcopy(built))


def test_an_index_build_archives_its_statistics_and_its_chunks():
    # this is an integration test
    """A build asked no question, and its statistics are still its finding.

    An index build produces no rows and no traces, so the three that count one
    another — rows, traces, selection — are empty together. What it did produce
    is the collection it wrote, the chunk count, the character percentiles and
    the embedding width, and those are carried into the index stage's
    `statistics` exactly as recorded, beside the chunks the replay recovers.
    """
    statistics = {'collection': 'raglab-backfill-index', 'chunks': 4,
                  'leaves': 4, 'avg_chars': 41.0, 'p95_chars': 58,
                  'embed_dim': 16, 'build_seconds': 0.42, 'reused': False}
    ledger.record({'id': 'backfill-index-001', 'kind': 'index',
                   'config': _flat_config(),
                   'result': dict(statistics, notes=[])}, 'done')

    built = backfill.build('backfill-index-001')
    assert built is not None, backfill.reason('backfill-index-001')
    assert backfill.rung(built) == 'indexed'
    archive.validate_archive(copy.deepcopy(built))

    stages = built['evaluation']['stage_results']
    assert stages['index']['statistics'] == statistics
    assert built['evaluation']['result']['index'] == statistics

    carried = examples.contents(built)
    assert carried['rows'] == 0 and carried['traces'] == 0
    assert carried['chunks'] == sum(len(group['chunks'])
                                    for group in _replayed(_flat_config()))
    assert carried['sessions'] == len(datasets.load(DATASET)[0]['sessions'])
    assert built['evaluation']['result']['selection']['question_ids'] == []
    assert built['evaluation']['result']['summary']['n_questions'] == 0


def test_an_orphan_run_file_archives_its_rows_and_its_judged_metrics():
    # this is an integration test
    """A `.runs/` file with no ledger row is still a finished experiment.

    It stores the config, the index statistics, the rows, the summary and the
    `ragas` block, and has no place for a trace or a chunk — which is exactly
    the `scored-without-traces` rung. Its selection predates `RunResult`, so
    the archived question ids are read off its own rows and the archive says
    so rather than inventing a balance nobody recorded.
    """
    _orphan_run('backfill-orphan-001', selection=None)
    assert ledger.experiment('backfill-orphan-001') is None

    built = backfill.build('backfill-orphan-001')
    assert built is not None, backfill.reason('backfill-orphan-001')
    assert backfill.rung(built) == 'scored-without-traces'
    archive.validate_archive(copy.deepcopy(built))

    result = built['evaluation']['result']
    carried = examples.contents(built)
    assert carried['rows'] == 1 and carried['answers'] == 1
    assert carried['traces'] == 0 and carried['candidates'] == 0
    assert carried['judged'] is True
    assert result['ragas']['metrics'] == examples.RAGAS['metrics']
    assert result['selection']['question_ids'] == [_question()['id']]
    assert 'balance' not in result['selection']
    assert any('read off its own rows' in note for note in result['notes'])
    # Chunks are recoverable even here: the corpus loads back by id and the
    # chunker is deterministic. Only what retrieval ranked is gone for good.
    assert carried['chunks'] == sum(len(group['chunks'])
                                    for group in _replayed())


def test_unfinished_work_is_excluded_rather_than_archived():
    # this is an integration test
    """A cancelled job has no finding in it, so it gets no archive and no blame.

    Excluded is a third answer, not a failure: `build` returns nothing, the
    reason names the state, and `survey` files it apart from the rows that
    could not be archived — which are a finding and would otherwise be buried
    under work nobody expected an archive from.
    """
    ledger.record({'id': 'backfill-cancelled-001', 'kind': 'run',
                   'config': _config()}, 'cancelled')

    assert backfill.build('backfill-cancelled-001') is None
    assert 'cancelled' in backfill.reason('backfill-cancelled-001')

    found = backfill.survey()
    excluded = {row['experiment_id'] for row in found['excluded_unfinished']}
    others = {row['experiment_id']
              for pile in ('archived', 'excluded_dead_knob', 'failed')
              for row in found[pile]}
    assert 'backfill-cancelled-001' in excluded
    assert 'backfill-cancelled-001' not in others


@pytest.mark.parametrize('group,field,retired', [
    ('index', 'summarizer', 'extractive'),
    ('retrieval', 'reranker', 'rerank-4-fast'),
    ('index', 'chunker', 'a-chunker-this-lab-retired'),
    ('generation', 'answerer', 'an-answerer-this-lab-retired'),
])
def test_a_retired_knob_excludes_the_row_and_names_the_knob(group, field,
                                                            retired):
    # this is an integration test
    """A value the vocabulary no longer serves takes the row off the board.

    Not settled to today's default and noted — that is a substitution, and the
    project's rule has no exception for a knob that happens to have run
    nothing: a knob this installation cannot serve is a refusal. So there is no
    archive at any rung, and the exclusion names the knob and the value, which
    is what lets a reader decide between restoring the vocabulary entry and
    deleting the row.

    Four knobs across three config groups, and two of them (`summarizer`,
    `reranker`) are the values this lab's own board actually carries. The other
    two are invented on purpose: the rule is read off `option_vocabularies`
    rather than off a list of the values we happened to find, so a knob retired
    later must exclude its rows without anyone editing the backfill.
    """
    config = _flat_config()
    config[group] = dict(config[group], **{field: retired})
    run_id = f'backfill-retired-{group}-{field}'
    _orphan_run(run_id, config=config)

    assert backfill.build(run_id) is None
    refusal = backfill.reason(run_id)
    assert f'{group}.{field}' in refusal and retired in refusal
    assert 'never a substitution' in refusal

    found = backfill.describe(run_id)
    assert found['pile'] == 'excluded-dead-knob'
    assert found['archive'] is None

    surveyed = {row['experiment_id']: row
                for row in backfill.survey()['excluded_dead_knob']}
    assert f'{group}.{field}={retired!r}' in surveyed[run_id]['knobs']
    assert run_id not in {row['experiment_id']
                          for row in backfill.survey()['failed']}


def test_every_vocabulary_backed_knob_is_checked_for_retirement():
    # this is a convention test
    """The retirement check covers the same ten knobs `LabConfig.validate` does.

    A vocabulary this table forgets is one whose retirement would silently do
    nothing — the row would archive carrying a knob value no control here can
    hold. So the pairs are pinned by name against the module that defines them,
    and a knob added to the config without being added here is a failure rather
    than a hole.
    """
    expected = {
        ('index', 'chunker'): vocabularies.CHUNKERS,
        ('index', 'embedder'): vocabularies.EMBEDDERS,
        ('index', 'hierarchy'): vocabularies.HIERARCHIES,
        ('index', 'graph_source'): vocabularies.GRAPH_SOURCES,
        ('index', 'summarizer'): vocabularies.SUMMARIZERS,
        ('retrieval', 'summary_scope'): vocabularies.SUMMARY_SCOPES,
        ('retrieval', 'retriever'): vocabularies.RETRIEVERS,
        ('retrieval', 'reranker'): vocabularies.RERANKERS,
        ('retrieval', 'grader'): vocabularies.GRADERS,
        ('generation', 'answerer'): vocabularies.ANSWERERS,
    }
    assert {(group, field): vocabulary
            for group, field, vocabulary in backfill.DEAD_KNOB_CHECKS} == expected
    # Every default is inside its own vocabulary, which is what makes a value
    # outside one a retirement rather than a config nobody could have written.
    for (group, field), vocabulary in expected.items():
        assert getattr(getattr(LabConfig(), group), field) in vocabulary


def test_the_survey_accounts_for_every_row_on_the_board():
    # this is an integration test
    """Every board row lands in exactly one of four piles, and no fifth.

    The population is the board, so it is the union of both durable records: a
    run file with no ledger row behind it is an experiment the survey has to
    account for, and so is a build that produced no rows. Archived rows each
    carry a rung off the ladder; unfinished work and rows on a retired knob are
    the two exclusions; anything left is a failure, named — and a failure is a
    finding rather than a resting place, which is why the three are counted
    apart instead of being pooled as "not archived".
    """
    groups = _replayed()
    _recorded('backfill-survey-good', candidates=[
        _leaf_candidate(groups[0]['chunks'][0])])
    _recorded('backfill-survey-bad', candidates=[
        _leaf_candidate(dict(groups[0]['chunks'][0], id='no-such:c0'))])
    ledger.record({'id': 'backfill-survey-index', 'kind': 'index',
                   'config': _flat_config(),
                   'result': {'collection': 'raglab-survey', 'chunks': 4,
                              'notes': []}}, 'done')
    _orphan_run('backfill-survey-orphan')

    found = backfill.survey()
    counts = found['counts']
    piles = ('archived', 'excluded_unfinished', 'excluded_dead_knob', 'failed')
    assert counts['board_rows'] == len(leaderboard.board_rows())
    for pile in piles:
        assert counts[pile] == len(found[pile])
    assert sum(counts[pile] for pile in piles) == counts['board_rows']
    # And no row is in two piles: the four partition the board rather than
    # merely covering it.
    landed = [row['experiment_id'] for pile in piles for row in found[pile]]
    assert len(landed) == len(set(landed)) == counts['board_rows']
    assert sum(found['rungs'].values()) == counts['archived']
    assert set(found['rungs']) == set(backfill.RUNGS)
    assert counts['bytes'] > 0

    rungs = {row['experiment_id']: row['rung'] for row in found['archived']}
    assert rungs['backfill-survey-good'] == 'generated'
    assert rungs['backfill-survey-bad'] == 'scored-without-traces'
    assert rungs['backfill-survey-index'] == 'indexed'
    assert rungs['backfill-survey-orphan'] == 'scored-without-traces'

    notes = {row['experiment_id']: row['notes'] for row in found['archived']}
    assert any('is not in the rebuilt index' in note
               for note in notes['backfill-survey-bad'])
    assert notes['backfill-survey-good'] == []
    for pile in ('excluded_unfinished', 'excluded_dead_knob', 'failed'):
        assert all(row['reason'] for row in found[pile]), (
            f'a {pile} row with no reason is the fudge this module refuses')
    assert all(row['bytes'] > 0 for row in found['archived'])
