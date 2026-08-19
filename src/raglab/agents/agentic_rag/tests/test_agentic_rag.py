"""The scoped RAG agent: what each scope owns, what it refuses, and what every
row it produces has to say.

Offline throughout, via a stub for `loop._ask` — the seam every node calls
and none bypasses. Two tests skip the stub and run against the real
`FakeChat`, because the conservative reading of an *unparsable* reply is the
safety property under test there.

Uses `token-hash`, not `ascii-hash`: the corpus is Farsi and ascii-hash embeds
it to the zero vector."""
import pytest

from raglab.agents import agentic_rag
from raglab.configuration import lab_config as config
from raglab.evaluation import run_evaluation as evaluate
from raglab.configuration import explainer_assembly as explain
from raglab.evaluation import deterministic_metrics as metrics
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.agents.agentic_rag import loop as agent_loop
from raglab.llm_backends import model_role_catalogue as models_mod
from raglab.configuration.lab_config import (
    AgentConfig,
    CRITICS,
    GenerationConfig,
    IndexConfig,
    LabConfig,
    LabSettings,
    RetrievalConfig,
    SCOPES,
    dependency_state)
from raglab.corpora.diary_corpus_loader import load_diary, load_ground_truth
from raglab.rag_components.indexing.index_builder_registry import IndexRegistry
from raglab.llm_backends.chat_model_factory import FakeChat

from raglab.conftest import SMOKE_INDEX, _finished

LAB_SETTINGS = LabSettings(openrouter_api_key='', llm_provider='fake')

LEAVES = dict(chunker='session', embedder='token-hash', contextual=False)


@pytest.fixture(scope='module')
def ground_truth():
    """The five-session smoke corpus's own ground truth, not the diary's —
    every loop test below scripts `loop._ask`, so the corpus only has to
    exist, never carry the specific evidence one diary question needs."""
    from raglab.corpora import dataset_import_contract as datasets
    _, gt = datasets.load('smoke-mini')
    return gt


@pytest.fixture(scope='module')
def index(smoke_index):
    return smoke_index.index


@pytest.fixture
def question(ground_truth):
    return ground_truth['questions'][0]['question_fa']


@pytest.fixture
def query_date(ground_truth):
    return ground_truth['meta']['query_date']


def agent_cfg(**kwargs) -> LabConfig:
    """A config whose agent is on, with the llm answerer the writing scopes
    require. Retrieval is kept small so a hop is fast."""
    generation = kwargs.pop('generation', {'answerer': 'llm'})
    retrieval = kwargs.pop('retrieval', {'k': 3, 'candidates': 12,
                                         'rerank_depth': 6})
    return LabConfig.from_dict({'index': LEAVES, 'retrieval': retrieval,
                                'generation': generation, 'agent': kwargs})


class Stub:
    """A scripted `loop._ask`: canned text per node, and a record of the order.

    Answers are per *node*, not per call index, so a test states "assess says
    0.9" without also having to know how many times the graph asks.
    """

    def __init__(self, **answers):
        self.answers = answers
        self.calls: list[str] = []

    def __call__(self, llm, model, node, system, user):
        self.calls.append(node)
        answer = self.answers.get(node, '')
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, list):
            # A node that must answer differently on successive hops.
            return answer[min(self.calls.count(node), len(answer)) - 1]
        return answer

    def count(self, node: str) -> int:
        return self.calls.count(node)


# --- the settings: off by default, outside the fingerprint ------------------

def test_the_agent_is_off_by_default_and_changes_nothing_until_asked():
    # this is a unit test
    """The `summary_scope` rule applied to a loop: shipping the agent must move
    no number in a lab nobody has reconfigured, or the four scopes stop being a
    factorial and become a confound in every earlier row."""
    assert AgentConfig().scope == ''
    assert LabConfig().agent == AgentConfig()
    assert LabConfig().validate() == []


def test_no_agent_knob_can_cost_an_index_rebuild():
    # this is a unit test
    """The agent is not an index field, so all four scopes sweep free against a
    single build — the property that makes this affordable to measure. A scope in
    the fingerprint would mean four 167-session builds to fill one 2x2 table."""
    flat = IndexConfig(**LEAVES).fingerprint()
    for scope in SCOPES:
        cfg = agent_cfg(scope=scope, max_hops=7, critic='both')
        assert cfg.index.fingerprint() == flat


def test_the_four_scopes_are_the_two_by_two_and_nothing_else():
    # this is a unit test
    """`retrieve` and `generate` own one stage each; `full` owns both. A fifth
    value would be a mechanism with no cell in the table."""
    assert SCOPES == ('', 'retrieve', 'generate', 'full')
    assert agentic_rag.owns_retrieval('retrieve') and agentic_rag.owns_retrieval('full')
    assert not agentic_rag.owns_retrieval('generate') and not agentic_rag.owns_retrieval('')
    assert agentic_rag.owns_generation('generate') and agentic_rag.owns_generation('full')
    assert not agentic_rag.owns_generation('retrieve') and not agentic_rag.owns_generation('')


# --- what is refused -------------------------------------------------------

def test_a_scope_this_installation_cannot_run_is_refused_never_substituted(
        monkeypatch):
    # this is a unit test
    """The `leiden` rule. A row labelled `scope=full` that was actually served
    by the fixed pipeline is the worst artefact this lab can produce, because no
    other field on it disagrees."""
    monkeypatch.setattr(agentic_rag, 'agent_available', lambda: False)
    problems = agent_cfg(scope='retrieve').validate()
    assert len(problems) == 1
    assert 'uv sync --extra agent' in problems[0]
    assert 'refused' in problems[0]
    # ...and the control is unaffected: no agent, nothing to install.
    assert LabConfig().validate() == []


def test_the_writing_scopes_require_the_llm_answerer():
    # this is a unit test
    """Under `extractive` the answer is quoted from the corpus, so there is
    nothing for a critic to critique and nothing for a revision to change. A
    validation error, never a silent promotion of the answerer."""
    for scope in ('generate', 'full'):
        problems = agent_cfg(scope=scope,
                             generation={'answerer': 'extractive'}).validate()
        assert any('answerer' in p and scope in p for p in problems), problems
        assert agent_cfg(scope=scope,
                         generation={'answerer': 'llm'}).validate() == []
    # `retrieve` owns no generation, so it constrains the answerer not at all.
    assert agent_cfg(scope='retrieve',
                     generation={'answerer': 'extractive'}).validate() == []


@pytest.mark.parametrize('knob,value', [('max_hops', 0), ('max_revisions', -1),
                                        ('max_llm_calls', 0)])
def test_a_loop_bound_below_its_floor_is_refused(knob, value):
    # this is a unit test
    problems = agent_cfg(scope='full', **{knob: value}).validate()
    assert any(knob in p for p in problems), problems


def test_an_unknown_scope_or_critic_is_refused_with_the_list():
    # this is a unit test
    assert any('scope' in p for p in agent_cfg(scope='wander').validate())
    assert any('critic' in p
               for p in agent_cfg(scope='generate', critic='vibes').validate())
    assert CRITICS == ('grounded', 'both', 'none')


# --- what the panel shows --------------------------------------------------

def test_every_agent_knob_explains_itself():
    # this is a unit test
    """`explain.missing() == []` is what stops a knob shipping unexplained, and
    the agent's group has to be inside that gate rather than beside it."""
    covered = set(explain.topics()) | explain.model_fields()
    for name in AgentConfig.__dataclass_fields__:
        assert f'agent.{name}' in covered, name


def test_the_agent_is_the_fourth_step_and_its_models_wear_its_ink():
    # this is a unit test
    """`ModelRole.step` is derived from the field a role writes to, so an ink
    cannot disagree with where the value is stored."""
    keys = [step.key for step in config.STEPS]
    assert keys == ['index', 'retrieval', 'generation', 'agent']
    roles = {role.key: role for role in models_mod.ROLES}
    assert roles['plan'].step == 'agent'
    assert roles['critic'].step == 'agent'
    # The answer is still written by the answerer, so no third dropdown appears.
    assert roles['answer'].field == 'generation.model'


def test_each_agent_knob_is_greyed_out_by_the_scope_that_never_reads_it():
    # this is a unit test
    """Per scope, and transitively: a critic model is dead when the critic is
    off, which is itself dead when the scope owns no generation."""
    def state(**kwargs):
        return dependency_state(agent_cfg(**kwargs).to_dict())

    off = state(scope='')
    for knob in ('max_hops', 'rewrite', 'evidence_threshold', 'max_revisions',
                 'critic', 'max_llm_calls', 'plan_model', 'critic_model'):
        assert not off[f'agent.{knob}']['enabled'], knob
        assert off[f'agent.{knob}']['reason']

    retrieving = state(scope='retrieve')
    assert retrieving['agent.max_hops']['enabled']
    assert retrieving['agent.rewrite']['enabled']
    assert retrieving['agent.plan_model']['enabled']
    assert not retrieving['agent.max_revisions']['enabled']
    assert not retrieving['agent.critic']['enabled']

    writing = state(scope='generate')
    assert writing['agent.max_revisions']['enabled']
    assert writing['agent.critic']['enabled']
    assert writing['agent.critic_model']['enabled']
    assert not writing['agent.max_hops']['enabled']

    both = state(scope='full')
    assert both['agent.max_hops']['enabled'] and both['agent.critic']['enabled']
    # Transitive: no critic, no critic model — and it reports the owner's reason.
    silent = state(scope='full', critic='none')
    assert not silent['agent.critic_model']['enabled']
    assert silent['agent.critic_model']['reason']


# --- the loop: retrieval scope ---------------------------------------------

def test_a_sufficient_first_hop_stops_and_names_why(index, question, query_date,
                                                    monkeypatch):
    # this is an integration test
    stub = Stub(plan='the diary entries about this', assess='SCORE: 0.9')
    monkeypatch.setattr(agent_loop, '_ask', stub)
    outcome = agentic_rag.run(index, agent_cfg(scope='retrieve', max_hops=3),
                        question, query_date, llm=FakeChat())
    assert outcome.diagnostics['agent_stop'] == 'evidence-sufficient'
    assert outcome.diagnostics['agent_hops'] == 1
    assert outcome.diagnostics['agent_rewrites'] == 0
    assert outcome.contexts, 'a sufficient hop still has to return its evidence'
    assert stub.count('rewrite') == 0


# Four ways a retrieval-scope loop can end, each a real cause rather than a
# label derived from the config passed in — `agent_stop` is the row's whole
# excuse for why a run refused, so the values below are hand-picked per case,
# never computed from `config_delta`. `threshold` needs a contrast (same
# score, two thresholds) to show the threshold decides anything at all, so it
# is two rows rather than one. On the five-session smoke index rather than
# the 167-session diary: every case scripts `loop._ask`, so the corpus only
# has to exist, never carry any one question's specific evidence.
LOOP_BOUND_CASES = [
    dict(id='hop-cap', config_delta=dict(max_hops=3),
         answers=dict(plan='...', assess='SCORE: 0.1',
                      rewrite='خواب و بیخوابی'),
         diagnostics=dict(agent_stop='hop-cap', agent_hops=3, agent_rewrites=2)),
    dict(id='rewrite-off', config_delta=dict(max_hops=2, rewrite=False),
         answers=dict(plan='...', assess='SCORE: 0.1'),
         diagnostics=dict(agent_stop='hop-cap', agent_hops=2, agent_rewrites=0),
         stub_counts=dict(rewrite=0)),
    dict(id='threshold-lenient', config_delta=dict(evidence_threshold=0.5),
         answers=dict(plan='...', assess='SCORE: 0.6'),
         diagnostics=dict(agent_stop='evidence-sufficient', agent_hops=1)),
    dict(id='threshold-strict',
         config_delta=dict(max_hops=2, evidence_threshold=0.8),
         answers=dict(plan='...', assess='SCORE: 0.6'),
         diagnostics=dict(agent_stop='hop-cap', agent_hops=2)),
    dict(id='call-cap', config_delta=dict(max_hops=9, max_llm_calls=3),
         answers=dict(plan='...', assess='SCORE: 0.1', rewrite='...'),
         diagnostics=dict(agent_stop='call-cap', agent_calls=3),
         stub_call_total=3),
]


@pytest.mark.parametrize('case', LOOP_BOUND_CASES,
                         ids=[c['id'] for c in LOOP_BOUND_CASES])
def test_the_retrieval_loop_stops_for_a_different_reason_per_config(
        index, question, query_date, monkeypatch, case):
    # this is an integration test
    """`test_an_insufficient_verdict_rewrites_and_hops_again_up_to_the_cap`,
    `test_rewriting_off_still_hops_and_is_the_control_for_the_rewrite`,
    `test_the_threshold_is_what_decides_sufficiency` and
    `test_the_cost_cap_ends_the_loop_and_names_itself`, folded into one table:
    hop-cap, the rewrite control, the threshold (both directions) and the
    call-cap, each asserted on the exact diagnostics field the original test
    checked."""
    stub = Stub(**case['answers'])
    monkeypatch.setattr(agent_loop, '_ask', stub)
    outcome = agentic_rag.run(index, agent_cfg(scope='retrieve', **case['config_delta']),
                        question, query_date, llm=FakeChat())
    for key, expected in case['diagnostics'].items():
        assert outcome.diagnostics[key] == expected, key
    for node, expected in case.get('stub_counts', {}).items():
        assert stub.count(node) == expected, node
    if 'stub_call_total' in case:
        assert len(stub.calls) == case['stub_call_total']


def test_every_hop_retrieves_through_the_measured_pipeline(
        index, question, query_date, monkeypatch):
    # this is an integration test
    """The agent loops *around* `pipeline.retrieve`, never past it: every
    existing retrieval knob still applies on every hop, so the row is a loop
    over a pipeline the lab has already swept rather than a second one it has
    not."""
    seen: list[RetrievalConfig] = []
    real = pipeline.retrieve

    def spy(idx, cfg, *args, **kwargs):
        seen.append(cfg)
        return real(idx, cfg, *args, **kwargs)

    monkeypatch.setattr(pipeline, 'retrieve', spy)
    monkeypatch.setattr(agent_loop, '_ask', Stub(plan='...', assess='SCORE: 0.1'))
    outcome = agentic_rag.run(index, agent_cfg(scope='retrieve', max_hops=2,
                                         retrieval={'k': 4, 'candidates': 10}),
                        question, query_date, llm=FakeChat())
    assert len(seen) == 2, 'one retrieval per hop'
    assert all(cfg.k == 4 and cfg.candidates == 10 for cfg in seen)
    assert len(outcome.contexts) <= 4


# --- the loop: generation scope --------------------------------------------

def test_a_grounded_draft_ships_without_revision(index, question, query_date,
                                                 monkeypatch):
    # this is an integration test
    stub = Stub(draft='جواب فارسی [s001]', critique='SCORE: 0.9')
    monkeypatch.setattr(agent_loop, '_ask', stub)
    outcome = agentic_rag.run(index, agent_cfg(scope='generate'), question,
                        query_date, llm=FakeChat())
    assert outcome.answer == 'جواب فارسی [s001]'
    assert outcome.diagnostics['agent_revisions'] == 0
    assert outcome.diagnostics['agent_stop'] == 'grounded'
    assert outcome.diagnostics['agent_hops'] == 1, 'retrieval stays fixed'


def test_a_refused_draft_is_revised_up_to_the_cap(index, question, query_date,
                                                  monkeypatch):
    # this is an integration test
    stub = Stub(draft=['first', 'second'], critique='SCORE: 0.1')
    monkeypatch.setattr(agent_loop, '_ask', stub)
    outcome = agentic_rag.run(index, agent_cfg(scope='generate', max_revisions=1),
                        question, query_date, llm=FakeChat())
    assert outcome.diagnostics['agent_revisions'] == 1
    assert outcome.diagnostics['agent_stop'] == 'revision-cap'
    assert stub.count('draft') == 2


def test_the_critic_off_ships_the_draft_and_calls_no_critic(
        index, question, query_date, monkeypatch):
    # this is an integration test
    """The control for the critic: without it the scope is one drafting call, so
    a `generate` row that beats `critic='none'` beat the critique, not the
    prompt."""
    stub = Stub(draft='جواب', critique='SCORE: 0.0')
    monkeypatch.setattr(agent_loop, '_ask', stub)
    outcome = agentic_rag.run(index, agent_cfg(scope='generate', critic='none'),
                        question, query_date, llm=FakeChat())
    assert stub.count('critique') == 0
    assert outcome.answer == 'جواب'
    assert outcome.diagnostics['agent_stop'] == 'drafted'


def test_the_both_critic_also_asks_whether_the_answer_answers_the_question(
        index, question, query_date, monkeypatch):
    # this is an integration test
    stub = Stub(draft='جواب', critique='SCORE: 0.9', completeness='SCORE: 0.9')
    monkeypatch.setattr(agent_loop, '_ask', stub)
    agentic_rag.run(index, agent_cfg(scope='generate', critic='both'), question,
              query_date, llm=FakeChat())
    assert stub.count('critique') == 1
    assert stub.count('completeness') == 1
    # ...and 'grounded' asks only the first question.
    stub = Stub(draft='جواب', critique='SCORE: 0.9', completeness='SCORE: 0.9')
    monkeypatch.setattr(agent_loop, '_ask', stub)
    agentic_rag.run(index, agent_cfg(scope='generate', critic='grounded'), question,
              query_date, llm=FakeChat())
    assert stub.count('completeness') == 0


# --- the loop: full, and the edge only it has ------------------------------

def test_only_the_full_scope_retrieves_again_after_a_bad_critique(
        index, question, query_date, monkeypatch):
    # this is an integration test
    """The interaction term of the factorial. `generate` can only rewrite the
    answer it has; `full` can go back for different evidence, which is the one
    mechanism neither middle row holds."""
    def hops(scope):
        calls: list[str] = []
        real = pipeline.retrieve
        monkeypatch.setattr(pipeline, 'retrieve',
                            lambda *a, **k: (calls.append('r'), real(*a, **k))[1])
        monkeypatch.setattr(agent_loop, '_ask',
                            Stub(plan='...', assess='SCORE: 0.9',
                                 draft='جواب', critique='SCORE: 0.1',
                                 rewrite='دوباره'))
        outcome = agentic_rag.run(index, agent_cfg(scope=scope, max_hops=2,
                                             max_revisions=1),
                            question, query_date, llm=FakeChat())
        return len(calls), outcome

    fixed, _ = hops('generate')
    looping, outcome = hops('full')
    assert fixed == 1
    assert looping == 2
    assert outcome.diagnostics['agent_hops'] == 2


# --- the conservative reading of an unreadable reply -----------------------

def test_an_unreadable_verdict_keeps_working_rather_than_declaring_success(
        index, question, query_date):
    # this is an integration test
    """No stub: `FakeChat` echoes its prompt, so every verdict is unparsable.
    An unreadable sufficiency verdict must mean *insufficient*, never a
    number that clears the threshold, or an unreachable model turns the loop
    into a silent no-op."""
    outcome = agentic_rag.run(index, agent_cfg(scope='full', max_hops=2,
                                         max_revisions=1),
                        question, query_date, llm=FakeChat())
    assert outcome.diagnostics['agent_hops'] == 2, 'never stopped early'
    assert outcome.diagnostics['agent_stop'] in ('hop-cap', 'revision-cap')
    assert outcome.diagnostics['agent_unparsed'] > 0, 'and it is counted'


def test_the_verdict_parser_reads_a_score_and_refuses_to_invent_one():
    # this is a unit test
    assert agentic_rag.verdict('SCORE: 0.8') == pytest.approx(0.8)
    assert agentic_rag.verdict('score 8/10') == pytest.approx(0.8)
    assert agentic_rag.verdict('YES') == 1.0
    assert agentic_rag.verdict('NO') == 0.0
    # Prose, an echo, and an empty reply are all *no opinion* — never a value.
    assert agentic_rag.verdict('FAKE: سؤال: چند بار ...') is None
    assert agentic_rag.verdict('') is None
    assert agentic_rag.verdict('it depends on what you mean by 0.9') is None


# --- cost, failure, and what the row says ----------------------------------
#
# The cost cap (call-cap) itself is now one row of `LOOP_BOUND_CASES` above.

def test_a_model_the_agent_cannot_reach_abstains_and_says_why(
        index, question, query_date, monkeypatch):
    # this is an integration test
    """`_llm_answer`'s call one level up: one unreachable question must not end
    a run that has paid for twenty-nine others, and it must never quietly fall
    back to the fixed pipeline — that produces a row labelled with an agent that
    never ran."""
    monkeypatch.setattr(agent_loop, '_ask',
                        Stub(plan=RuntimeError('daemon is down')))
    outcome = agentic_rag.run(index, agent_cfg(scope='full'), question, query_date,
                        llm=FakeChat())
    assert outcome.abstained
    assert outcome.answer == pipeline.REFUSAL
    assert 'daemon is down' in outcome.diagnostics['agent_error']
    assert outcome.diagnostics['agent_stop'] == 'error'


def test_agent_columns_join_the_runs_aggregate_and_its_notes_name_the_scope(
        index, ground_truth, query_date, monkeypatch):
    # this is an integration test
    """Three distinct claims, folded into one test since the second used to
    be a strict superset of a since-deleted
    `test_the_agent_returns_the_outcome_the_rest_of_the_lab_already_scores`
    (its only additions were the `isinstance`/`n_contexts`/`latency_ms`
    checks now inlined below): the agent fills the same `Outcome` so scoring
    needs no second idea of what a result is; the run's notes name the scope
    and its caps (`agentic_rag.note_for` on its own); and `n_hops`/`n_agent_calls`
    ride the same `AGGREGATED` tuple every other per-question number does, so
    they reach the run's overall means with no special case. No HTTP, no job,
    no ragas; the route-narrowing claim these used to sit beside is
    `test_the_retrieval_route_shows_the_loop_and_never_answers`, kept."""
    retrieving = agentic_rag.note_for(AgentConfig(scope='retrieve', max_hops=5))
    assert 'agent scope=retrieve' in retrieving and 'max_hops=5' in retrieving
    assert 'critic=' not in retrieving, 'retrieve owns no generation'
    full_note = agentic_rag.note_for(AgentConfig(scope='full', critic='both',
                                           max_revisions=2))
    assert 'agent scope=full' in full_note
    assert 'critic=both' in full_note and 'max_revisions=2' in full_note

    monkeypatch.setattr(agent_loop, '_ask',
                        Stub(plan='...', assess='SCORE: 0.9', draft='جواب [s001]',
                             critique='SCORE: 0.9'))
    rows = []
    for asked in ground_truth['questions'][:2]:
        outcome = agentic_rag.run(index, agent_cfg(scope='full'), asked['question_fa'],
                            query_date, llm=FakeChat())
        assert isinstance(outcome, pipeline.Outcome)
        row = metrics.score_question(asked, outcome, k=3)
        assert 'n_contexts' in row and 'latency_ms' in row
        rows.append(row)
    assert all(row['agent_scope'] == 'full' for row in rows)
    assert all(row['n_hops'] >= 1 and row['n_agent_calls'] >= 2 for row in rows)
    assert all(row['agent_stop'] for row in rows)
    assert metrics.aggregate(rows)['overall']['n_hops'] >= 1


# Direct in-memory index (`agent_cfg`'s `LEAVES`/module `index` fixture) does
# not fit here: `run_eval` builds its own index from `cfg.index`, so this one
# needs its own smoke-corpus config instead.
def test_an_agent_scoped_run_eval_traces_the_loop_and_notes_its_scope(
        ground_truth, monkeypatch):
    # this is an integration test
    """`run_eval`'s agent branch (`if cfg.agent.scope: ... agentic_rag.run(...,
    trace=tr)`) and `_assemble_notes`'s `if cfg.agent.scope:
    notes.append(agentic_rag.note_for(cfg.agent))` are reachable only from an
    agent-scoped `run_eval` call — `run_retrieval` calls neither, and the one
    caller that used to exercise both, an HTTP `/api/evaluations` job, is
    gone along with the deleted `test_an_evaluation_with_a_scope_scores_
    records_and_names_the_loop`. Without a test calling `run_eval` itself with
    `cfg.agent.scope` set, deleting that branch or its `trace=tr` argument
    passes this file's suite. Direct: no HTTP, no job, no ragas."""
    monkeypatch.setattr(agent_loop, '_ask',
                        Stub(plan='...', assess='SCORE: 0.9', draft='جواب [s001]',
                             critique='SCORE: 0.9'))
    registry = IndexRegistry(LAB_SETTINGS)
    cfg = LabConfig(index=IndexConfig(**SMOKE_INDEX),
                    retrieval=RetrievalConfig(k=3, candidates=12, rerank_depth=6),
                    generation=GenerationConfig(answerer='llm'),
                    agent=AgentConfig(scope='full', max_hops=2))
    result = evaluate.run_eval(registry, ground_truth, cfg, LAB_SETTINGS,
                               limit=1, ragas_mode='off', trace=True)
    assert result.rows[0]['agent_scope'] == 'full'
    assert any('agent scope=full' in note for note in result.notes)
    assert result.traces[0]['trace']['agent'][0]['node'] == 'plan'


def test_the_loop_counters_are_explained_measures_not_bare_numbers():
    # this is a unit test
    """`explain.missing_metrics() == []` is the gate; these two join it rather
    than arriving on the dashboard as unlabelled integers."""
    defined = {m['key']: m for m in explain.measures()}
    for key in ('n_hops', 'n_agent_calls'):
        assert key in defined, key
        assert defined[key]['step'] == 'agent'
        assert defined[key]['formula']
    assert 'n_hops' in metrics.AGGREGATED
    assert 'n_agent_calls' in metrics.AGGREGATED


def test_a_traced_agent_records_every_node_and_moves_no_number(
        index, question, query_date, monkeypatch):
    # this is an integration test
    """The Inspector's ladder — "refused because the diary is silent" and
    "refused after two hops found nothing" are different findings, and the
    trace is where the second one is legible — and `retrieve_traced`'s
    guarantee one level up: tracing records the same run, so asking for it
    can never change a score. One scripted run proves both, run twice (plain,
    then traced) from a fresh stub each time."""
    def scripted() -> Stub:
        return Stub(plan='...', assess='SCORE: 0.1', rewrite='...',
                    draft='جواب', critique='SCORE: 0.9')

    monkeypatch.setattr(agent_loop, '_ask', scripted())
    plain = agentic_rag.run(index, agent_cfg(scope='full', max_hops=2), question,
                      query_date, llm=FakeChat())

    trace: dict = {}
    monkeypatch.setattr(agent_loop, '_ask', scripted())
    traced = agentic_rag.run(index, agent_cfg(scope='full', max_hops=2), question,
                       query_date, llm=FakeChat(), trace=trace)

    nodes = [visit['node'] for visit in trace['agent']]
    assert nodes[0] == 'plan'
    assert nodes.count('retrieve') == 2
    assert 'draft' in nodes and 'critique' in nodes
    assert all('hop' in visit for visit in trace['agent'])
    # The per-candidate ladder the Inspector already renders is still filled, so
    # the agent's traced run is not a second, poorer kind of trace.
    assert trace['candidates']

    # and the numbers agree, node-recording included: tracing changed nothing
    assert [c.chunk_id for c in plain.contexts] == [c.chunk_id
                                                    for c in traced.contexts]
    assert plain.answer == traced.answer
    assert plain.diagnostics['agent_hops'] == traced.diagnostics['agent_hops']


# --- the graph itself ------------------------------------------------------

def test_the_loop_is_a_compiled_langgraph_with_the_edge_full_alone_has():
    # this is a unit test
    """Not an incidental while-loop: the scopes differ by which nodes and edges
    the graph has, which is the thing a reader can check against the design."""
    pytest.importorskip('langgraph')
    for scope in ('retrieve', 'generate', 'full'):
        nodes = agentic_rag.graph_nodes(agent_cfg(scope=scope).agent)
        assert ('retrieve' in nodes) == agentic_rag.owns_retrieval(scope)
        assert ('critique' in nodes) == agentic_rag.owns_generation(scope)
    edges = agentic_rag.graph_edges(agent_cfg(scope='full').agent)
    assert ('critique', 'retrieve') in edges
    assert ('critique', 'retrieve') not in agentic_rag.graph_edges(
        agent_cfg(scope='generate').agent)


# --- the routes ------------------------------------------------------------

@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient

    from raglab.dashboard.panel_server import create_app
    return TestClient(create_app())


def test_the_panel_offers_every_scope_and_says_which_can_run(client):
    # this is an integration test
    """Served, never listed in the frontend: a panel with its own list is a
    panel that will offer a scope the service refuses."""
    body = client.get('/api/options').json()
    assert body['scopes'] == list(SCOPES)
    assert body['critics'] == list(CRITICS)
    support = body['agent_support']
    assert support['']['available'], 'no agent is always available'
    assert set(support) == set(SCOPES)
    assert all(support[s]['install'] for s in SCOPES if s)
    # The fourth step and its two model roles arrive through the same lists the
    # other three do, so the panel cannot colour them by guessing.
    assert 'agent' in [step['key'] for step in body['steps']]
    assert {'plan', 'critic'} <= {r['key'] for r in body['model_roles']}
    assert all(r['step'] == 'agent'
               for r in body['model_roles'] if r['key'] in ('plan', 'critic'))
    # And the dependency rules for the agent's knobs are served with the rest.
    assert any(key.startswith('agent.') for key in body['dependencies'])
    assert 'agent.scope' in body['help']


def test_the_retrieval_route_shows_the_loop_and_never_answers(client,
                                                             monkeypatch):
    # this is an integration test
    """`/api/retrievals` retrieves and stops. An agent that owns retrieval is
    part of what there is to show; the drafting half of `full` is an answering
    stage, so it must not run here however the scope is set.

    (`test_an_evaluation_with_a_scope_scores_records_and_names_the_loop` used
    to sit beside this, driving the same loop through `/api/evaluations`
    instead — deleted, since this route already pins that the loop actually
    runs through an HTTP job and traces come back; its one distinct claim,
    that the agent's columns join the run's aggregate, now has its own direct
    test beside `test_the_loop_counters_are_explained_measures_not_bare_
    numbers`, and its notes claim has its own direct test on `agentic_rag.note_for`.)"""
    stub = Stub(plan='...', assess='SCORE: 0.1', rewrite='دوباره',
                draft='این نباید اجرا شود')
    monkeypatch.setattr(agent_loop, '_ask', stub)
    payload = agent_cfg(scope='full', max_hops=2).to_dict() | {'limit': 1}
    res = client.post('/api/retrievals', json=payload)
    assert res.status_code == 202, res.text
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    questions = job['result']['questions']
    assert len(questions) == 1
    nodes = [v['node'] for v in questions[0]['trace']['agent']]
    assert nodes.count('retrieve') == 2, 'the loop ran'
    assert 'draft' not in nodes and stub.count('draft') == 0


def test_a_scope_the_backend_cannot_run_is_a_400_on_both_run_routes(
        client, monkeypatch):
    # this is an integration test
    """Both run routes apply the same screen — the rule `/api/queries` and
    `/api/evaluations` already share about models, applied to the agentic_rag."""
    monkeypatch.setattr(agentic_rag, 'agent_available', lambda: False)
    payload = agent_cfg(scope='retrieve').to_dict() | {'limit': 1}
    for route in ('/api/evaluations', '/api/retrievals', '/api/queries'):
        body = payload | ({'question': 'چطور بودم؟'} if 'queries' in route else {})
        res = client.post(route, json=body)
        assert res.status_code == 400, (route, res.status_code)
        assert '--extra agent' in res.json()['detail'], route


# --- the two pages ---------------------------------------------------------

@pytest.fixture(scope='module')
def agent_page_texts(client):
    """Every named text the convention table below checks, fetched the one
    way a browser actually reaches it (`client.get`) rather than a second
    disk read of the same file. The Inspector's half comes from its own
    `TestClient`, since :9002 and :9003 are separate services serving
    separate copies of `tokens.css`/`panel.css`'s counterpart."""
    from fastapi.testclient import TestClient

    from raglab.dashboard import inspector_server as inspector_mod

    insp = TestClient(inspector_mod.create_inspector_app())
    return {
        'tokens.css': client.get('/tokens.css').text,
        'index.html': client.get('/').text,
        'panel.css': client.get('/panel.css').text,
        'panel.js': client.get('/panel.js').text,
        'inspector.css': insp.get('/inspector.css').text,
        'inspector.js': insp.get('/inspector.js').text,
    }


# (file, must_contain, must_not_contain, reason) — one row per retired
# single-substring pin test, each carrying the one line that used to be its
# docstring so a failure names the rule rather than printing a bare
# "assert 'x' in text".
AGENT_PAGE_CONVENTIONS = [
    ('tokens.css', '--step-agent:', None,
     'the fourth step ink must be defined in the one tokens sheet both pages '
     'share, or a step whose colour exists on one page only is a legend that '
     'lies on the other'),
    ('tokens.css', '--step-agent-lit:', None,
     'the lit variant for dark surfaces must be defined beside the base ink'),
    ('tokens.css', 'oklch(0.48 0.16 318)', None,
     'both pages must read the one same ink value, not two tokens of the '
     'same name that happen to differ'),
    ('panel.css', 'var(--step-agent)', None,
     'the panel must actually use the agent ink somewhere, not just define it'),
    ('inspector.css', 'var(--step-agent)', None,
     'the Inspector must actually use the agent ink too'),
    ('index.html', 'data-step="agent"', None,
     'the agent card must be tagged with its step so the ink and the stage '
     'cannot disagree'),
    ('inspector.js', 'function agentLadder', None,
     'the ladder is what makes an agent row readable, and it must exist'),
    ('inspector.js', 'trace.agent', None,
     'the ladder must read the trace the run actually produced'),
    ('inspector.css', 'agent-ladder', None,
     'the ladder needs its own style hook'),
    ('panel.js', 'for (const group of Object.keys(defaults))', None,
     'the group list must come from the served defaults, never hard-coded — '
     'a browser holding a config from before the agent group existed must '
     'not come up with blank agent controls'),
    ('panel.js', None,
     "for (const group of ['index', 'retrieval', 'generation']) {\n    merged",
     'the hard-coded group list must not come back to startingConfig'),
    ('panel.css', '.bench > .rag-models { grid-column: -2 / -1; }', None,
     'every model must stay in the one right-hand column, pinned rather than '
     'left to the grid\'s own auto-placement'),
    ('panel.css', 'repeat(4, minmax(0, 1fr)) minmax(0, 300px)', None,
     "a four-slot grid's auto-placement fills a row before wrapping, which "
     'would put the models card on a second row'),
]


@pytest.mark.parametrize('file, must_contain, must_not_contain, reason',
                         AGENT_PAGE_CONVENTIONS)
def test_the_served_pages_keep_their_agent_conventions(
        agent_page_texts, file, must_contain, must_not_contain, reason):
    # this is a convention test
    """Three of the five agent-frontend pin tests, folded into one table:
    the fourth step ink both pages must share, and the Inspector's ladder.
    Each row is a claim a served asset makes about itself, and the reason
    string is what a failure prints instead of a bare `assert 'x' in text`."""
    text = agent_page_texts[file]
    if must_contain is not None:
        assert must_contain in text, reason
    if must_not_contain is not None:
        assert must_not_contain not in text, reason


def test_the_panel_has_a_control_for_every_agent_knob(client):
    # this is a convention test
    """`explain.missing()` stops a knob shipping unexplained; this stops one
    shipping unreachable. A field with no control is a field the panel
    silently posts at its default, which is how a preset comes to lie. Kept
    as its own test rather than a table row: the claim is checked once per
    `AgentConfig` field, not against one fixed string."""
    panel = client.get('/').text
    js = client.get('/panel.js').text
    models = {role.field for role in models_mod.ROLES}
    for name in AgentConfig.__dataclass_fields__:
        if f'agent.{name}' in models:
            # Model roles render from the served list; what must exist here
            # is the column the agent's group renders into.
            assert 'id="modelRoles-agent"' in panel
            continue
        assert f"$('{name}')" in js, name
        assert f'id="{name}"' in panel, name


def test_every_agent_node_reads_the_evidence_the_answerer_reads(monkeypatch):
    # this is an integration test
    """The `generate` scope asks whether a critique loop writes better
    answers from the *same* evidence, so a draft node holding less of it
    makes the scope partly a measurement of truncation rather than of
    critique. Deliberately not the module `index`/`question`/`query_date`
    fixtures: this is the one test in the file that needs a corpus large
    enough to show the fault (>900 handed characters), which the five-session
    smoke corpus (630 raw content characters, before any retrieval narrows it
    further) cannot supply — so it builds the full diary directly."""
    diary_index = IndexRegistry(LAB_SETTINGS, load_diary()).get(IndexConfig(**LEAVES))
    gt = load_ground_truth()
    question = gt['questions'][0]['question_fa']
    query_date = gt['meta']['query_date']
    seen: dict[str, str] = {}

    def spy(llm, model, node, system, user):
        seen[node] = user
        return 'SCORE: 0.9'

    monkeypatch.setattr(agent_loop, '_ask', spy)
    outcome = agentic_rag.run(diary_index, agent_cfg(scope='full', critic='both'),
                        question, query_date, llm=FakeChat())
    handed = pipeline.context_blocks(outcome)
    assert len(handed) > 900, 'a shorter corpus than this cannot show the fault'
    for node in ('assess', 'draft', 'critique', 'completeness'):
        assert handed in seen[node], node
