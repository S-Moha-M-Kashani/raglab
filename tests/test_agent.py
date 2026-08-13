"""The scoped RAG agent: what each scope owns, what it refuses, and what every
row it produces has to say.

Offline throughout, via a stub for `agent._ask` — the seam every node calls
and none bypasses. Two tests skip the stub and run against the real
`FakeChat`, because the conservative reading of an *unparsable* reply is the
safety property under test there.

Uses `token-hash`, not `ascii-hash`: the corpus is Farsi and ascii-hash embeds
it to the zero vector."""
import time

import pytest

from raglab import agent, config, explain, metrics, pipeline
from raglab import models as models_mod
from raglab.config import (AgentConfig, CRITICS, IndexConfig, LabConfig,
                           LabSettings, RetrievalConfig, SCOPES,
                           dependency_state)
from raglab.corpus import load_diary, load_ground_truth
from raglab.index import IndexRegistry
from raglab.llm import FakeChat

LAB_SETTINGS = LabSettings(openrouter_api_key='', llm_provider='fake')

LEAVES = dict(chunker='session', embedder='token-hash', contextual=False)


@pytest.fixture(scope='module')
def diary():
    return load_diary()


@pytest.fixture(scope='module')
def ground_truth():
    return load_ground_truth()


@pytest.fixture(scope='module')
def registry(diary):
    return IndexRegistry(LAB_SETTINGS, diary)


@pytest.fixture(scope='module')
def index(registry):
    return registry.get(IndexConfig(**LEAVES))


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
    """A scripted `agent._ask`: canned text per node, and a record of the order.

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
    """The `summary_scope` rule applied to a loop: shipping the agent must move
    no number in a lab nobody has reconfigured, or the four scopes stop being a
    factorial and become a confound in every earlier row."""
    assert AgentConfig().scope == ''
    assert LabConfig().agent == AgentConfig()
    assert LabConfig().validate() == []


def test_no_agent_knob_can_cost_an_index_rebuild():
    """The agent is not an index field, so all four scopes sweep free against a
    single build — the property that makes this affordable to measure. A scope in
    the fingerprint would mean four 167-session builds to fill one 2x2 table."""
    flat = IndexConfig(**LEAVES).fingerprint()
    for scope in SCOPES:
        cfg = agent_cfg(scope=scope, max_hops=7, critic='both')
        assert cfg.index.fingerprint() == flat


def test_the_four_scopes_are_the_two_by_two_and_nothing_else():
    """`retrieve` and `generate` own one stage each; `full` owns both. A fifth
    value would be a mechanism with no cell in the table."""
    assert SCOPES == ('', 'retrieve', 'generate', 'full')
    assert agent.owns_retrieval('retrieve') and agent.owns_retrieval('full')
    assert not agent.owns_retrieval('generate') and not agent.owns_retrieval('')
    assert agent.owns_generation('generate') and agent.owns_generation('full')
    assert not agent.owns_generation('retrieve') and not agent.owns_generation('')


# --- what is refused -------------------------------------------------------

def test_a_scope_this_installation_cannot_run_is_refused_never_substituted(
        monkeypatch):
    """The `leiden` rule. A row labelled `scope=full` that was actually served
    by the fixed pipeline is the worst artefact this lab can produce, because no
    other field on it disagrees."""
    monkeypatch.setattr(agent, 'agent_available', lambda: False)
    problems = agent_cfg(scope='retrieve').validate()
    assert len(problems) == 1
    assert 'uv sync --extra agent' in problems[0]
    assert 'refused' in problems[0]
    # ...and the control is unaffected: no agent, nothing to install.
    assert LabConfig().validate() == []


def test_the_writing_scopes_require_the_llm_answerer():
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
    problems = agent_cfg(scope='full', **{knob: value}).validate()
    assert any(knob in p for p in problems), problems


def test_an_unknown_scope_or_critic_is_refused_with_the_list():
    assert any('scope' in p for p in agent_cfg(scope='wander').validate())
    assert any('critic' in p
               for p in agent_cfg(scope='generate', critic='vibes').validate())
    assert CRITICS == ('grounded', 'both', 'none')


# --- what the panel shows --------------------------------------------------

def test_every_agent_knob_explains_itself():
    """`explain.missing() == []` is what stops a knob shipping unexplained, and
    the agent's group has to be inside that gate rather than beside it."""
    assert explain.missing() == []
    covered = set(explain.topics()) | explain.model_fields()
    for name in AgentConfig.__dataclass_fields__:
        assert f'agent.{name}' in covered, name


def test_the_agent_is_the_fourth_step_and_its_models_wear_its_ink():
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
    stub = Stub(plan='the diary entries about this', assess='SCORE: 0.9')
    monkeypatch.setattr(agent, '_ask', stub)
    outcome = agent.run(index, agent_cfg(scope='retrieve', max_hops=3),
                        question, query_date, llm=FakeChat())
    assert outcome.diagnostics['agent_stop'] == 'evidence-sufficient'
    assert outcome.diagnostics['agent_hops'] == 1
    assert outcome.diagnostics['agent_rewrites'] == 0
    assert outcome.contexts, 'a sufficient hop still has to return its evidence'
    assert stub.count('rewrite') == 0


def test_an_insufficient_verdict_rewrites_and_hops_again_up_to_the_cap(
        index, question, query_date, monkeypatch):
    stub = Stub(plan='...', assess='SCORE: 0.1', rewrite='خواب و بیخوابی')
    monkeypatch.setattr(agent, '_ask', stub)
    outcome = agent.run(index, agent_cfg(scope='retrieve', max_hops=3),
                        question, query_date, llm=FakeChat())
    assert outcome.diagnostics['agent_hops'] == 3
    assert outcome.diagnostics['agent_rewrites'] == 2, 'one per hop after the first'
    assert outcome.diagnostics['agent_stop'] == 'hop-cap'


def test_rewriting_off_still_hops_and_is_the_control_for_the_rewrite(
        index, question, query_date, monkeypatch):
    """`rewrite=False` is what says whether rewriting was the useful part or
    merely another look at the same pool."""
    stub = Stub(plan='...', assess='SCORE: 0.1')
    monkeypatch.setattr(agent, '_ask', stub)
    outcome = agent.run(index,
                        agent_cfg(scope='retrieve', max_hops=2, rewrite=False),
                        question, query_date, llm=FakeChat())
    assert outcome.diagnostics['agent_hops'] == 2
    assert outcome.diagnostics['agent_rewrites'] == 0
    assert stub.count('rewrite') == 0


def test_the_threshold_is_what_decides_sufficiency(index, question, query_date,
                                                   monkeypatch):
    monkeypatch.setattr(agent, '_ask', Stub(plan='...', assess='SCORE: 0.6'))
    lenient = agent.run(index, agent_cfg(scope='retrieve',
                                         evidence_threshold=0.5),
                        question, query_date, llm=FakeChat())
    strict = agent.run(index, agent_cfg(scope='retrieve', max_hops=2,
                                        evidence_threshold=0.8),
                       question, query_date, llm=FakeChat())
    assert lenient.diagnostics['agent_hops'] == 1
    assert strict.diagnostics['agent_hops'] == 2


def test_every_hop_retrieves_through_the_measured_pipeline(
        index, question, query_date, monkeypatch):
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
    monkeypatch.setattr(agent, '_ask', Stub(plan='...', assess='SCORE: 0.1'))
    outcome = agent.run(index, agent_cfg(scope='retrieve', max_hops=2,
                                         retrieval={'k': 4, 'candidates': 10}),
                        question, query_date, llm=FakeChat())
    assert len(seen) == 2, 'one retrieval per hop'
    assert all(cfg.k == 4 and cfg.candidates == 10 for cfg in seen)
    assert len(outcome.contexts) <= 4


# --- the loop: generation scope --------------------------------------------

def test_a_grounded_draft_ships_without_revision(index, question, query_date,
                                                 monkeypatch):
    stub = Stub(draft='جواب فارسی [s001]', critique='SCORE: 0.9')
    monkeypatch.setattr(agent, '_ask', stub)
    outcome = agent.run(index, agent_cfg(scope='generate'), question,
                        query_date, llm=FakeChat())
    assert outcome.answer == 'جواب فارسی [s001]'
    assert outcome.diagnostics['agent_revisions'] == 0
    assert outcome.diagnostics['agent_stop'] == 'grounded'
    assert outcome.diagnostics['agent_hops'] == 1, 'retrieval stays fixed'


def test_a_refused_draft_is_revised_up_to_the_cap(index, question, query_date,
                                                  monkeypatch):
    stub = Stub(draft=['first', 'second'], critique='SCORE: 0.1')
    monkeypatch.setattr(agent, '_ask', stub)
    outcome = agent.run(index, agent_cfg(scope='generate', max_revisions=1),
                        question, query_date, llm=FakeChat())
    assert outcome.diagnostics['agent_revisions'] == 1
    assert outcome.diagnostics['agent_stop'] == 'revision-cap'
    assert stub.count('draft') == 2


def test_the_critic_off_ships_the_draft_and_calls_no_critic(
        index, question, query_date, monkeypatch):
    """The control for the critic: without it the scope is one drafting call, so
    a `generate` row that beats `critic='none'` beat the critique, not the
    prompt."""
    stub = Stub(draft='جواب', critique='SCORE: 0.0')
    monkeypatch.setattr(agent, '_ask', stub)
    outcome = agent.run(index, agent_cfg(scope='generate', critic='none'),
                        question, query_date, llm=FakeChat())
    assert stub.count('critique') == 0
    assert outcome.answer == 'جواب'
    assert outcome.diagnostics['agent_stop'] == 'drafted'


def test_the_both_critic_also_asks_whether_the_answer_answers_the_question(
        index, question, query_date, monkeypatch):
    stub = Stub(draft='جواب', critique='SCORE: 0.9', completeness='SCORE: 0.9')
    monkeypatch.setattr(agent, '_ask', stub)
    agent.run(index, agent_cfg(scope='generate', critic='both'), question,
              query_date, llm=FakeChat())
    assert stub.count('critique') == 1
    assert stub.count('completeness') == 1
    # ...and 'grounded' asks only the first question.
    stub = Stub(draft='جواب', critique='SCORE: 0.9', completeness='SCORE: 0.9')
    monkeypatch.setattr(agent, '_ask', stub)
    agent.run(index, agent_cfg(scope='generate', critic='grounded'), question,
              query_date, llm=FakeChat())
    assert stub.count('completeness') == 0


# --- the loop: full, and the edge only it has ------------------------------

def test_only_the_full_scope_retrieves_again_after_a_bad_critique(
        index, question, query_date, monkeypatch):
    """The interaction term of the factorial. `generate` can only rewrite the
    answer it has; `full` can go back for different evidence, which is the one
    mechanism neither middle row holds."""
    def hops(scope):
        calls: list[str] = []
        real = pipeline.retrieve
        monkeypatch.setattr(pipeline, 'retrieve',
                            lambda *a, **k: (calls.append('r'), real(*a, **k))[1])
        monkeypatch.setattr(agent, '_ask',
                            Stub(plan='...', assess='SCORE: 0.9',
                                 draft='جواب', critique='SCORE: 0.1',
                                 rewrite='دوباره'))
        outcome = agent.run(index, agent_cfg(scope=scope, max_hops=2,
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
    """No stub: `FakeChat` echoes its prompt, so every verdict is unparsable.
    An unreadable sufficiency verdict must mean *insufficient*, never a
    number that clears the threshold, or an unreachable model turns the loop
    into a silent no-op."""
    outcome = agent.run(index, agent_cfg(scope='full', max_hops=2,
                                         max_revisions=1),
                        question, query_date, llm=FakeChat())
    assert outcome.diagnostics['agent_hops'] == 2, 'never stopped early'
    assert outcome.diagnostics['agent_stop'] in ('hop-cap', 'revision-cap')
    assert outcome.diagnostics['agent_unparsed'] > 0, 'and it is counted'


def test_the_verdict_parser_reads_a_score_and_refuses_to_invent_one():
    assert agent.verdict('SCORE: 0.8') == pytest.approx(0.8)
    assert agent.verdict('score 8/10') == pytest.approx(0.8)
    assert agent.verdict('YES') == 1.0
    assert agent.verdict('NO') == 0.0
    # Prose, an echo, and an empty reply are all *no opinion* — never a value.
    assert agent.verdict('FAKE: سؤال: چند بار ...') is None
    assert agent.verdict('') is None
    assert agent.verdict('it depends on what you mean by 0.9') is None


# --- cost, failure, and what the row says ----------------------------------

def test_the_cost_cap_ends_the_loop_and_names_itself(index, question,
                                                     query_date, monkeypatch):
    """A shape-bounded loop can still be expensive, and an unsweepable knob is
    not a knob."""
    stub = Stub(plan='...', assess='SCORE: 0.1', rewrite='...')
    monkeypatch.setattr(agent, '_ask', stub)
    outcome = agent.run(index, agent_cfg(scope='retrieve', max_hops=9,
                                         max_llm_calls=3),
                        question, query_date, llm=FakeChat())
    assert outcome.diagnostics['agent_stop'] == 'call-cap'
    assert outcome.diagnostics['agent_calls'] == 3
    assert len(stub.calls) == 3


def test_a_model_the_agent_cannot_reach_abstains_and_says_why(
        index, question, query_date, monkeypatch):
    """`_llm_answer`'s call one level up: one unreachable question must not end
    a run that has paid for twenty-nine others, and it must never quietly fall
    back to the fixed pipeline — that produces a row labelled with an agent that
    never ran."""
    monkeypatch.setattr(agent, '_ask',
                        Stub(plan=RuntimeError('daemon is down')))
    outcome = agent.run(index, agent_cfg(scope='full'), question, query_date,
                        llm=FakeChat())
    assert outcome.abstained
    assert outcome.answer == pipeline.REFUSAL
    assert 'daemon is down' in outcome.diagnostics['agent_error']
    assert outcome.diagnostics['agent_stop'] == 'error'


def test_the_agent_returns_the_outcome_the_rest_of_the_lab_already_scores(
        index, ground_truth, query_date, monkeypatch):
    """The agent fills the same `Outcome`, so scoring, RAGAS, the ledger and the
    Inspector need no second idea of what a result is."""
    monkeypatch.setattr(agent, '_ask',
                        Stub(plan='...', assess='SCORE: 0.9', draft='جواب [s001]',
                             critique='SCORE: 0.9'))
    asked = ground_truth['questions'][0]
    outcome = agent.run(index, agent_cfg(scope='full'), asked['question_fa'],
                        query_date, llm=FakeChat())
    assert isinstance(outcome, pipeline.Outcome)
    row = metrics.score_question(asked, outcome, k=3)
    assert row['n_hops'] == 1
    assert row['n_agent_calls'] >= 2
    assert row['agent_stop'] == 'grounded'
    assert 'n_contexts' in row and 'latency_ms' in row


def test_the_loop_counters_are_explained_measures_not_bare_numbers():
    """`explain.missing_metrics() == []` is the gate; these two join it rather
    than arriving on the dashboard as unlabelled integers."""
    assert explain.missing_metrics() == []
    defined = {m['key']: m for m in explain.measures()}
    for key in ('n_hops', 'n_agent_calls'):
        assert key in defined, key
        assert defined[key]['step'] == 'agent'
        assert defined[key]['formula']
    assert 'n_hops' in metrics.AGGREGATED
    assert 'n_agent_calls' in metrics.AGGREGATED


def test_a_traced_agent_records_every_node_it_visited(index, question,
                                                      query_date, monkeypatch):
    """The Inspector's ladder. "Refused because the diary is silent" and
    "refused after two hops found nothing" are different findings, and the trace
    is where the second one is legible."""
    monkeypatch.setattr(agent, '_ask',
                        Stub(plan='...', assess='SCORE: 0.1', rewrite='...',
                             draft='جواب', critique='SCORE: 0.9'))
    trace: dict = {}
    agent.run(index, agent_cfg(scope='full', max_hops=2), question, query_date,
              llm=FakeChat(), trace=trace)
    nodes = [visit['node'] for visit in trace['agent']]
    assert nodes[0] == 'plan'
    assert nodes.count('retrieve') == 2
    assert 'draft' in nodes and 'critique' in nodes
    assert all('hop' in visit for visit in trace['agent'])
    # The per-candidate ladder the Inspector already renders is still filled, so
    # the agent's traced run is not a second, poorer kind of trace.
    assert trace['candidates']


def test_tracing_an_agent_moves_no_number(index, question, query_date,
                                          monkeypatch):
    """`retrieve_traced`'s guarantee, one level up: the trace is a recording of
    the same run, so asking for it can never change a score."""
    def once():
        monkeypatch.setattr(agent, '_ask',
                            Stub(plan='...', assess='SCORE: 0.9', draft='جواب',
                                 critique='SCORE: 0.9'))
        return agent.run(index, agent_cfg(scope='full'), question, query_date,
                         llm=FakeChat())

    plain = once()
    trace: dict = {}
    monkeypatch.setattr(agent, '_ask',
                        Stub(plan='...', assess='SCORE: 0.9', draft='جواب',
                             critique='SCORE: 0.9'))
    traced = agent.run(index, agent_cfg(scope='full'), question, query_date,
                       llm=FakeChat(), trace=trace)
    assert [c.chunk_id for c in plain.contexts] == [c.chunk_id
                                                    for c in traced.contexts]
    assert plain.answer == traced.answer
    assert plain.diagnostics['agent_hops'] == traced.diagnostics['agent_hops']


# --- the graph itself ------------------------------------------------------

def test_the_loop_is_a_compiled_langgraph_with_the_edge_full_alone_has():
    """Not an incidental while-loop: the scopes differ by which nodes and edges
    the graph has, which is the thing a reader can check against the design."""
    pytest.importorskip('langgraph')
    for scope in ('retrieve', 'generate', 'full'):
        nodes = agent.graph_nodes(agent_cfg(scope=scope).agent)
        assert ('retrieve' in nodes) == agent.owns_retrieval(scope)
        assert ('critique' in nodes) == agent.owns_generation(scope)
    edges = agent.graph_edges(agent_cfg(scope='full').agent)
    assert ('critique', 'retrieve') in edges
    assert ('critique', 'retrieve') not in agent.graph_edges(
        agent_cfg(scope='generate').agent)


# --- the routes ------------------------------------------------------------

@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient

    from raglab.server import create_app
    return TestClient(create_app())


def _finished(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] not in ('running', 'cancelling'):
            return job
        time.sleep(0.01)
    raise AssertionError(f'job {job_id} still running after {timeout}s')


def test_the_panel_offers_every_scope_and_says_which_can_run(client):
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


def test_an_evaluation_with_a_scope_scores_records_and_names_the_loop(client,
                                                                     monkeypatch):
    """The whole way through: an agent run has to produce a row the leaderboard
    can read, with the loop's counters on every question and its shape in the
    notes — a row that says `scope=full` and nothing else would be two
    configurations wearing one label."""
    monkeypatch.setattr(agent, '_ask',
                        Stub(plan='...', assess='SCORE: 0.9', draft='جواب [s001]',
                             critique='SCORE: 0.9'))
    payload = agent_cfg(scope='full').to_dict() | {'limit': 2,
                                                  'ragas_mode': 'off'}
    res = client.post('/api/evaluations', json=payload)
    assert res.status_code == 202, res.text
    job = _finished(client, res.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    result = job['result']
    rows = result['rows']
    assert len(rows) == 2
    assert all(row['agent_scope'] == 'full' for row in rows)
    assert all(row['n_hops'] >= 1 and row['n_agent_calls'] >= 2 for row in rows)
    assert all(row['agent_stop'] for row in rows)
    assert any('agent scope=full' in note for note in result['notes'])
    # The means join the summary like every other per-question number.
    assert result['summary']['overall']['n_hops'] >= 1
    # And the Inspector's ladder came back with the run rather than being lost.
    assert result['traces'][0]['trace']['agent'][0]['node'] == 'plan'


def test_the_retrieval_route_shows_the_loop_and_never_answers(client,
                                                             monkeypatch):
    """`/api/retrievals` retrieves and stops. An agent that owns retrieval is
    part of what there is to show; the drafting half of `full` is an answering
    stage, so it must not run here however the scope is set."""
    stub = Stub(plan='...', assess='SCORE: 0.1', rewrite='دوباره',
                draft='این نباید اجرا شود')
    monkeypatch.setattr(agent, '_ask', stub)
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
    """Both run routes apply the same screen — the rule `/api/queries` and
    `/api/evaluations` already share about models, applied to the agent."""
    monkeypatch.setattr(agent, 'agent_available', lambda: False)
    payload = agent_cfg(scope='retrieve').to_dict() | {'limit': 1}
    for route in ('/api/evaluations', '/api/retrievals', '/api/queries'):
        body = payload | ({'question': 'چطور بودم؟'} if 'queries' in route else {})
        res = client.post(route, json=body)
        assert res.status_code == 400, (route, res.status_code)
        assert '--extra agent' in res.json()['detail'], route


# --- the two pages ---------------------------------------------------------

def _static(name: str) -> str:
    from raglab.server import STATIC
    return (STATIC / name).read_text(encoding='utf-8')


def test_both_pages_define_the_fourth_ink_and_neither_invents_it():
    """One ink per step, defined once per page with the same value. The lab and
    the Inspector are one instrument in two windows, so a step whose colour
    exists on one page only is a legend that lies on the other."""
    panel, sheet = _static('index.html'), _static('inspector.css')
    for page in (panel, sheet):
        assert '--step-agent:' in page
        assert '--step-agent-lit:' in page
    # The same value on both pages, not merely a token of the same name.
    ink = 'oklch(0.48 0.16 318)'
    assert ink in panel and ink in sheet
    assert 'data-step="agent"' in panel


def test_the_panel_has_a_control_for_every_agent_knob():
    """`explain.missing()` stops a knob shipping unexplained; this stops one
    shipping unreachable. A field with no control is a field the panel
    silently posts at its default, which is how a preset comes to lie."""
    panel = _static('index.html')
    models = {role.field for role in models_mod.ROLES}
    for name in AgentConfig.__dataclass_fields__:
        if f'agent.{name}' in models:
            # Model roles render from the served list; what must exist here
            # is the column the agent's group renders into.
            assert 'id="modelRoles-agent"' in panel
            continue
        assert f"$('{name}')" in panel, name
        assert f'id="{name}"' in panel, name


def test_the_inspector_renders_the_loop_beside_the_ranks():
    """The ladder is what makes an agent row readable: the candidate table says
    what came back, the ladder says after what."""
    js = _static('inspector.js')
    assert 'function agentLadder' in js
    assert 'trace.agent' in js
    assert 'agent-ladder' in _static('inspector.css')


def test_every_agent_node_reads_the_evidence_the_answerer_reads(index, question,
                                                               query_date,
                                                               monkeypatch):
    """The `generate` scope asks whether a critique loop writes better
    answers from the *same* evidence, so a draft node holding less of it
    makes the scope partly a measurement of truncation rather than of
    critique."""
    seen: dict[str, str] = {}

    def spy(llm, model, node, system, user):
        seen[node] = user
        return 'SCORE: 0.9'

    monkeypatch.setattr(agent, '_ask', spy)
    outcome = agent.run(index, agent_cfg(scope='full', critic='both'), question,
                        query_date, llm=FakeChat())
    handed = pipeline.context_blocks(outcome)
    assert len(handed) > 900, 'a shorter corpus than this cannot show the fault'
    for node in ('assess', 'draft', 'critique', 'completeness'):
        assert handed in seen[node], node


def test_the_panel_merges_a_remembered_config_over_the_served_groups():
    """A browser holding a config from before the agent group existed must
    not come up with blank agent controls — a blank number input reads as 0,
    and validation then refuses `max_hops` for a knob nobody touched. So the
    group list comes from the served defaults, never hard-coded in the page."""
    panel = _static('index.html')
    assert 'for (const group of Object.keys(defaults))' in panel
    assert "for (const group of ['index', 'retrieval', 'generation']) {\n    merged" \
        not in panel, 'the hard-coded group list is back in startingConfig'


def test_the_models_column_stays_the_right_hand_one_whatever_the_step_count():
    """A four-slot grid's auto-placement fills a row before wrapping, which
    would put the *models* card on a second row — breaking the rule that
    every model lives in the one right-hand column. Pinned rather than left
    to arithmetic."""
    panel = _static('index.html')
    assert '.bench > .rag-models { grid-column: -2 / -1; }' in panel
    assert 'repeat(4, minmax(0, 1fr)) minmax(0, 300px)' in panel
