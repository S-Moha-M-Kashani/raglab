"""The widget's three read-only windows onto recorded experiments.

The widget is a sealed leaf: it imports no evaluation module, so the records
reach it by injection (`set_experiment_reader`, the same shape as the
OpenRouter key resolver). These tests pin both halves — that an un-wired widget
refuses instead of inventing, and that a wired one renders the board's own
rows without adding a number of its own.
"""
import pytest

from raglab.agents import widget
from raglab.agents.widget import experiment_tools as tools


class _Reader:
    """The three injected functions, with no lab behind them.

    The same names `panel_server` wires in — `leaderboard.board_rows`,
    `leaderboard.experiment`, `run_evaluation.question_rows` — so a rename on
    the lab side fails here rather than at the model's next question."""

    def __init__(self, board_rows=None, experiment=None, question_rows=None):
        self._rows_out = board_rows or []
        self._experiment = experiment
        self._rows = question_rows or {'rows': [], 'n_questions': 0,
                                       'n_matched': 0}
        self.asked = []

    def board_rows(self, limit=500):
        self.asked.append(('board_rows', limit))
        return list(self._rows_out)

    def experiment(self, experiment_id):
        self.asked.append(('experiment', experiment_id))
        return self._experiment

    def question_rows(self, experiment_id, only='missed', limit=25):
        self.asked.append(('question_rows', experiment_id, only, limit))
        return self._rows


@pytest.fixture(autouse=True)
def _no_reader():
    """Every test starts unwired and leaves nothing wired behind it.

    Both ends matter, and the first one is the one that bites: the injection is
    module state on a package the whole suite imports, so a test that built the
    panel app (which wires the real reader in) leaves this widget reading the
    developer's own records. Unwiring only on the way out passes alone and
    fails in a full run."""
    tools.set_experiment_reader(None)
    yield
    tools.set_experiment_reader(None)


ENTRY = {'experiment_id': 'r1', 'kind': 'run', 'state': 'done',
         'error': '', 'label': 'baseline', 'dataset': 'smoke-mini',
         'started_at': '2026-08-01 10:00:00', 'seconds': 12,
         'provider': 'openrouter', 'n_questions': 6, 'decision': 0.7125,
         'decision_stderr': 0.01,
         'judge': {'model': 'sonnet-4', 'provider': 'openrouter'},
         'pipeline': [{'text': 'session·token-hash'}, {'text': 'bm25'},
                      {'text': 'llm'}],
         'source': 'both'}

DIGEST = {'experiment_id': 'r1', 'kind': 'run', 'state': 'done', 'error': '',
          'label': 'baseline', 'dataset': 'smoke-mini',
          'started_at': '2026-08-01 10:00:00', 'seconds': 12,
          'provider': 'openrouter', 'n_questions': 6, 'decision': 0.7125,
          'decision_stderr': 0.01,
          'judge': {'model': 'sonnet-4', 'provider': 'openrouter'},
          'metrics': {'faithfulness': 0.8, 'answer_relevancy': 0.7,
                      'llm_context_precision_with_reference': 0.6,
                      'context_recall': 0.75},
          'ragas_notes': [], 'ragas_skipped': 0,
          'config': {'index': {'split_plan': [{'kind': 'document'}]},
                    'retrieval': {'retriever': 'bm25', 'k': 3},
                    'generation': {'answerer': 'llm'}},
          'index': {'chunks': 15}, 'summary': {'overall': {'recall': 0.5}},
          'notes': [], 'has_question_rows': True, 'source': 'both'}


# --- an un-wired widget refuses -------------------------------------------

@pytest.mark.parametrize('call', [
    lambda: widget.list_experiments.invoke({}),
    lambda: widget.read_experiment.invoke({'experiment_id': 'r1'}),
    lambda: widget.read_experiment_questions.invoke({'experiment_id': 'r1'}),
])
def test_every_tool_says_so_when_no_records_are_wired_in(call):
    # this is a unit test
    """A widget served without a reader — the `__main__` harness, a panel that
    did not inject one — has no records at all. Saying so is the only honest
    answer; a plausible-looking empty listing would read as "no experiments".
    """
    reply = call()
    assert 'not available' in reply.lower()


# --- the listing -----------------------------------------------------------

def test_the_listing_names_the_decision_beside_its_own_error():
    # this is a unit test
    """The four-metric mean is never shown without its spread, so the tool
    that shows it to a model shows both or neither."""
    tools.set_experiment_reader(_Reader(board_rows=[ENTRY]))
    reply = widget.list_experiments.invoke({})
    assert 'r1' in reply
    assert '0.71' in reply and '0.01' in reply
    assert 'sonnet-4' in reply


def test_the_listing_keeps_only_the_corpus_that_was_asked_about():
    # this is a unit test
    """Scores do not cross corpora, so a question about one dataset must be
    answered from that dataset's rows alone.

    The filter is applied here rather than asked of the reader, because what the
    reader hands over is the board's own rows — the same object the leaderboard
    is built from, unfiltered, exactly as the board serves it. Which corpus a
    question is about is this tool's business, not the record's."""
    reader = _Reader(board_rows=[ENTRY,
                                dict(ENTRY, experiment_id='other',
                                     dataset='meetings-de')])
    tools.set_experiment_reader(reader)
    reply = widget.list_experiments.invoke({'dataset': 'smoke-mini'})
    assert 'r1' in reply and 'other' not in reply
    assert reader.asked == [('board_rows', tools.SCAN)]


def test_an_empty_listing_says_the_records_are_empty():
    # this is a unit test
    tools.set_experiment_reader(_Reader(board_rows=[]))
    reply = widget.list_experiments.invoke({'dataset': 'nope'})
    assert 'no experiment' in reply.lower()


def test_a_run_that_measured_nothing_reads_as_unjudged_not_as_zero():
    # this is a unit test
    """A blank decision is a run that judged nothing. Rendering it as 0.000
    would make it the worst experiment on the board instead of an unmeasured
    one."""
    unjudged = dict(ENTRY, decision=None, decision_stderr=None, judge={})
    tools.set_experiment_reader(_Reader(board_rows=[unjudged]))
    reply = widget.list_experiments.invoke({})
    assert '0.000' not in reply
    assert 'unjudged' in reply.lower()


# --- one experiment --------------------------------------------------------

def test_reading_one_experiment_names_its_knobs_and_its_four_metrics():
    # this is a unit test
    tools.set_experiment_reader(_Reader(experiment=DIGEST))
    reply = widget.read_experiment.invoke({'experiment_id': 'r1'})
    assert 'faithfulness' in reply and '0.800' in reply
    assert 'bm25' in reply and 'split_plan' in reply
    assert 'sonnet-4' in reply


def test_reading_an_experiment_names_the_backend_that_answered():
    # this is a unit test
    """`fake` is a rehearsal, not a measurement, and a reader asked to compare
    two experiments must be able to see that one of them measured nothing."""
    tools.set_experiment_reader(_Reader(experiment=dict(DIGEST, provider='fake')))
    reply = widget.read_experiment.invoke({'experiment_id': 'r1'})
    assert 'fake' in reply


def test_an_unknown_experiment_id_is_reported_as_unknown():
    # this is a unit test
    """Neither record holds it. The tool says so and lists nothing — inventing
    a digest is the one failure that would put a made-up number in an answer.
    """
    tools.set_experiment_reader(_Reader(experiment=None))
    reply = widget.read_experiment.invoke({'experiment_id': 'ghost'})
    assert 'ghost' in reply and 'no experiment' in reply.lower()


# --- the failure set -------------------------------------------------------

MISSED = {
    'experiment_id': 'r1', 'dataset': 'smoke-mini', 'filter': 'missed',
    'n_questions': 6, 'n_matched': 2, 'k': 3,
    'rows': [{'id': 'mini-002', 'question': 'What broke in the kitchen?',
              'type': 'single-hop', 'difficulty': 'easy', 'behavior': 'answer',
              'recall': 0.5, 'precision': 0.33, 'mrr': 0.5, 'hit': 1.0,
              'n_contexts': 3, 'retrieved_sessions': ['mini-01'],
              'expected_sessions': ['mini-01', 'mini-04'], 'abstained': False,
              'false_abstention': False}]}


def test_the_failure_set_names_each_question_and_what_was_missed():
    # this is a unit test
    """The question's own text, its recall, and which evidence sessions were
    expected against which were retrieved — the three things an answer about
    what to change has to rest on."""
    tools.set_experiment_reader(_Reader(question_rows=MISSED))
    reply = widget.read_experiment_questions.invoke({'experiment_id': 'r1'})
    assert 'What broke in the kitchen?' in reply
    assert 'mini-04' in reply
    assert '0.5' in reply


def test_an_abstain_question_is_tagged_unanswerable():
    # this is a unit test
    """`behavior` is the row's own field (task-5 renamed `answerable` off it);
    a row that abstains must read as unanswerable rather than silently
    defaulting to answerable, which is exactly the regression a stale
    `row.get('answerable', True)` produced — every row looked answerable
    regardless of its real `behavior`."""
    abstained = dict(MISSED, rows=[dict(MISSED['rows'][0], id='mini-006',
                                        behavior='abstain', abstained=True)])
    tools.set_experiment_reader(_Reader(question_rows=abstained))
    reply = widget.read_experiment_questions.invoke({'experiment_id': 'r1'})
    assert 'unanswerable' in reply


def test_the_failure_set_says_how_many_it_left_out():
    # this is a unit test
    """A capped listing that read as a census would send a recommendation off
    a sample the reader took for the whole failure set."""
    many = dict(MISSED, n_matched=40)
    tools.set_experiment_reader(_Reader(question_rows=many))
    reply = widget.read_experiment_questions.invoke({'experiment_id': 'r1'})
    assert '40' in reply


def test_a_ledger_only_experiment_says_why_it_has_no_question_rows():
    # this is a unit test
    """No run file, so no per-question rows exist. The reason travels, because
    "no rows" and "no failures" are opposite answers."""
    tools.set_experiment_reader(_Reader(question_rows={
        'experiment_id': 'build-1', 'rows': [], 'n_questions': 0,
        'n_matched': 0, 'reason': 'no run file for this experiment'}))
    reply = widget.read_experiment_questions.invoke({'experiment_id': 'build-1'})
    assert 'no run file' in reply


def test_the_row_cap_the_tool_asks_for_is_its_own_not_the_models():
    # this is a unit test
    """The model names a filter, never a page size: a call that could ask for
    every row of a 167-question run would fill the context window with the
    tail of a list."""
    reader = _Reader(question_rows=MISSED)
    tools.set_experiment_reader(reader)
    widget.read_experiment_questions.invoke({'experiment_id': 'r1',
                                            'only': 'abstained'})
    assert reader.asked == [('question_rows', 'r1', 'abstained',
                             tools.MAX_QUESTION_ROWS)]


# --- the tools are tools ---------------------------------------------------

def test_the_three_tools_are_in_the_registry_with_fixture_descriptions():
    # this is a convention test
    """Same rule as the other five: the model-facing description comes from
    fixtures/prompts/widget_tools.yaml, and the registry the agent is handed
    holds these three."""
    names = {t.name for t in widget.TOOLS}
    assert {'list_experiments', 'read_experiment',
            'read_experiment_questions'} <= names
    for tool in widget.TOOLS:
        assert tool.description and tool.description.strip()


# --- the digest stays readable ---------------------------------------------

BY_TYPE = {'single-hop': {'n': 20, 'recall': 0.85, 'hit': 0.85,
                          'abstained_correctly': None},
           'pattern': {'n': 7, 'recall': 0.4929, 'hit': 0.8571,
                       'abstained_correctly': None}}


def test_a_summary_broken_down_by_group_is_one_line_per_group():
    # this is a unit test
    """`by_type` and `by_difficulty` are dicts of dicts, and the whole of one
    printed as a single line is 1,500 characters of `{...}` — the most useful
    signal in a run rendered as the least readable thing in the answer. Per
    difficulty band or question type is exactly the comparison a reader asks
    for, so each group gets its own line."""
    tools.set_experiment_reader(_Reader(experiment=dict(
        DIGEST, summary={'by_type': BY_TYPE})))
    reply = widget.read_experiment.invoke({'experiment_id': 'r1'})
    lines = [line for line in reply.splitlines() if 'pattern' in line]
    assert len(lines) == 1
    assert '{' not in reply
    assert 'recall=0.493' in lines[0]


def test_a_measure_that_measured_nothing_is_left_out_of_the_summary():
    # this is a unit test
    """An abstention rate of None is a measure that did not apply to those
    questions. Printing `abstained_correctly=None` spends a line saying
    nothing, and reads as a recorded zero to anything skimming it."""
    tools.set_experiment_reader(_Reader(experiment=dict(
        DIGEST, summary={'by_type': BY_TYPE})))
    reply = widget.read_experiment.invoke({'experiment_id': 'r1'})
    assert 'None' not in reply
    assert 'abstained_correctly' not in reply


def test_an_unjudged_row_does_not_read_as_judge_no_judge():
    # this is a unit test
    tools.set_experiment_reader(_Reader(board_rows=[
        dict(ENTRY, decision=None, decision_stderr=None, judge={})]))
    reply = widget.list_experiments.invoke({})
    assert 'judge no judge' not in reply
    assert 'no judge' in reply


INERT_DIGEST = dict(DIGEST, config={
    'index': {'hierarchy': '', 'chunk_chars': 500, 'min_group': 3},
}, inert={'index.min_group': 'nothing is grouped'})


def test_a_knob_the_run_never_read_renders_as_none_not_its_leftover_value():
    # this is a unit test
    """`min_group=3` on a flat index that grouped nothing is a lie about what
    produced the row. The inert map (tasks 1-3, injected on the found dict) says
    which dotted knob paths the run never read; this tool must render exactly
    those as the literal word `none`, never the number still sitting in the
    recorded config."""
    tools.set_experiment_reader(_Reader(experiment=INERT_DIGEST))
    reply = widget.read_experiment.invoke({'experiment_id': 'r1'})
    assert 'chunk_chars=500' in reply
    assert 'min_group=none' in reply
    assert 'min_group=3' not in reply


def test_a_found_dict_without_inert_renders_knobs_as_before():
    # this is a unit test
    """Backwards safety: a reader that has not been touched by tasks 1-3 hands
    over a found dict with no `inert` key at all, and this tool must render
    exactly as it did before this change."""
    without_inert = dict(INERT_DIGEST)
    del without_inert['inert']
    tools.set_experiment_reader(_Reader(experiment=without_inert))
    reply = widget.read_experiment.invoke({'experiment_id': 'r1'})
    assert 'min_group=3' in reply
    assert 'overlap=none' not in reply


def test_the_index_build_line_names_its_own_measures_and_nothing_else():
    # this is a unit test
    """The build's stats are one flat block, so they belong on the "index
    build" line itself — a sub-label invented to hold them would read as a
    field the run recorded."""
    tools.set_experiment_reader(_Reader(experiment=dict(
        DIGEST, index={'chunks': 533, 'embed_dim': 1024})))
    reply = widget.read_experiment.invoke({'experiment_id': 'r1'})
    assert 'index build: chunks=533, embed_dim=1024' in reply
