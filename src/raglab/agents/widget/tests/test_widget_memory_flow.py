# this is a unit test
"""The policy-to-memory flow is selective and delivery ordered."""
from langchain_core.messages import AIMessage
import pytest
import threading
import time

from raglab.agents import widget
from raglab.agents.widget import long_term_memory as long_memory


@pytest.fixture(autouse=True)
def _clean_long_term_memory():
    """A clean store either side of the test — and no writer still running.

    The deferred decision outlives the call that started it, so the clear on
    the way out has to come after the last daemon this test started has
    finished. Clearing first and joining never would leave a write landing in
    widget.db after the *next* test's clear, failing an assertion in a test
    that started nothing."""
    widget.experiment_tools.set_experiment_reader(None)
    long_memory.clear_long_term_memory()
    yield
    _join_deferred_memory()
    long_memory.clear_long_term_memory()
    widget.experiment_tools.set_experiment_reader(None)


class _ExperimentReader:
    def __init__(self, dataset):
        self.dataset = dataset

    def experiment(self, experiment_id):
        return {'experiment_id': experiment_id, 'dataset': self.dataset}


class _Agent:
    def __init__(self, answer='authoritative answer'):
        self.answer = answer
        self.invocations = 0
        self.payloads = []

    def invoke(self, payload, config=None):
        self.invocations += 1
        self.payloads.append(payload)
        return {'messages': [
            *payload['messages'],
            AIMessage(content=self.answer),
        ]}


class _PolicyModel:
    def __init__(self, policy):
        self.policy = policy
        self.calls = []

    def with_structured_output(self, schema):
        self.calls.append(schema)
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.policy


def _setup(monkeypatch, policy, answer='authoritative answer'):
    agent = _Agent(answer)
    policy_model = _PolicyModel(policy)
    monkeypatch.setattr(widget.backends, '_memory_model',
                        lambda model: policy_model)
    monkeypatch.setattr(widget.backends, '_agent_for', lambda model: agent)
    return agent, policy_model


@pytest.fixture
def settled_memory(monkeypatch):
    """Run the post-answer memory decision on the calling thread.

    `ask` starts it on a thread of its own, which is the point of the ordering
    — but a test asking *what was filed* would then be racing it. The deferred
    work is the same call either way; this only removes the thread, so the
    decision is on `result['memory']` where the assertions can read it."""
    monkeypatch.setattr(
        widget.backends, '_defer_memory',
        lambda *args: widget.backends._finish_memory(*args) or {})


def _eventually(predicate):
    for _ in range(100):
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def _join_deferred_memory():
    """Wait for the deferred decision's own thread to finish.

    `_defer_memory` starts a daemon called `widget-memory`, and a test that
    returns while one is still inside `save_widget_memory` leaves it writing
    into the session's widget.db after the next test's autouse clear — a flake
    landing on whichever test runs next, which is the one place it cannot be
    explained from."""
    for thread in threading.enumerate():
        if thread.name == 'widget-memory':
            thread.join(5)


def test_deterministic_rejection_happens_before_agent_and_memory(monkeypatch):
    agent = _Agent()
    monkeypatch.setattr(widget.backends, '_agent_for', lambda model: agent)
    monkeypatch.setattr(widget.backends, '_memory_model',
                        lambda model: (_ for _ in ()).throw(
                            AssertionError('policy model was called')))
    monkeypatch.setattr(widget.tools, 'save_widget_memory',
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError('memory was saved')))

    result = widget.ask('Tell me a joke about penguins.',
                        model='openai/gpt-5-nano', thread='exp-reject')

    assert 'RAG lab' in result['reply']
    assert result['memory']['relevant'] is False
    assert agent.invocations == 0


def test_the_agent_answers_before_the_policy_model_is_asked_anything(monkeypatch):
    """The reader waits for one model, not two.

    The policy judges whether a finished turn is worth keeping, which is a
    question about an answer that does not exist yet, so it runs after the
    agent. Nothing may consult it on the way in."""
    order = []
    agent, policy_model = _setup(monkeypatch, {
        'relevant': True, 'should_save': False, 'dataset_id': '',
        'subtopic': '', 'reason': 'a fair question'})
    monkeypatch.setattr(widget.backends, '_memory_model',
                        lambda model: order.append('policy') or policy_model)
    answer = agent.invoke

    def watched(payload, config=None):
        order.append('agent')
        return answer(payload, config)

    agent.invoke = watched
    result = widget.ask('Which chunker should I compare?',
                        model='openai/gpt-5-nano', thread='order-exp')

    assert result['reply'] == 'authoritative answer'
    _eventually(lambda: 'policy' in order)
    assert order == ['agent', 'policy']


def test_two_turns_share_one_policy_client_and_reset_drops_it(monkeypatch):
    """The agent beside it was cached from the first day; this client was
    rebuilt on every turn, so each turn paid for a new connection to say one
    structured sentence. `reset()` drops both, because a retyped key must not
    leave a turn answered by one credential and filed by another."""
    built = []
    policy_model = _PolicyModel({
        'relevant': True, 'should_save': False, 'dataset_id': '',
        'subtopic': '', 'reason': 'a fair question'})
    monkeypatch.setattr(widget.backends, '_build_memory_model',
                        lambda model: built.append(model) or policy_model)
    monkeypatch.setattr(widget.backends, '_agent_for', lambda model: _Agent())
    widget.reset()
    try:
        for _ in range(2):
            widget.ask('Which chunker should I compare?',
                       model='openai/gpt-5-nano', thread='cache-exp')
        _eventually(lambda: len(policy_model.calls) == 2)
        assert built == ['openai/gpt-5-nano']

        widget.reset()
        widget.backends._memory_model('openai/gpt-5-nano')
        assert built == ['openai/gpt-5-nano', 'openai/gpt-5-nano']
    finally:
        widget.reset()


def test_an_irrelevant_policy_answers_the_question_and_files_nothing(
        monkeypatch, settled_memory):
    """Ruled on 2026-08-28: the model-judged relevance check is a save gate,
    not an answer gate. The widget writes no measured number, so its relevance
    check is a scope guard; a question the deterministic guard let through is
    answered, and the policy decides only whether anything is filed."""
    agent, _ = _setup(monkeypatch, {
        'relevant': False, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'misc', 'reason': 'not reusable',
    })
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError('summarizer was called')))

    result = widget.ask('Which index should I use?', model='openai/gpt-5-nano',
                        thread='exp-irrelevant')

    assert agent.invocations == 1
    assert result['reply'] == 'authoritative answer'
    assert result['memory']['should_save'] is False
    assert result['memory']['saved'] is False
    assert 'not reusable' in result['memory']['reason']
    assert long_memory.memory_context('diary-en') == ''


def test_relevant_policy_cannot_invent_dataset_without_active_experiment(
        monkeypatch, settled_memory):
    agent, _ = _setup(monkeypatch, {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'reusable',
    })

    result = widget.ask('Which retrieval setting should we retain?',
                        model='openai/gpt-5-nano', thread='general')

    assert agent.invocations == 1
    assert result['reply'] == 'authoritative answer'
    assert result['memory']['saved'] is False
    assert long_memory.memory_context('diary-en') == ''


def test_a_policy_naming_another_dataset_is_answered_and_not_filed(
        monkeypatch, settled_memory):
    """The mismatch stopped blocking the answer and still stops the write: the
    thread's dataset comes from the validated record, the policy's is model
    text, and a row filed under the wrong corpus is the lie CLAUDE.md forbids.
    The refusal names both so the reason says what happened."""
    class Reader:
        def experiment(self, experiment_id):
            return {'experiment_id': experiment_id, 'dataset': 'diary-en'}

    widget.experiment_tools.set_experiment_reader(Reader())
    agent, policy_model = _setup(monkeypatch, {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-fa',
        'subtopic': 'retrieval', 'reason': 'wrong dataset',
    })
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError('summarizer was called')))

    result = widget.ask('Which index should I use?',
                        model='openai/gpt-5-nano', thread='exp-context')

    assert agent.invocations == 1
    assert result['reply'] == 'authoritative answer'
    assert result['memory']['saved'] is False
    reason = result['memory']['reason']
    assert 'diary-fa' in reason and 'diary-en' in reason
    assert long_memory.memory_context('diary-en') == ''
    # The policy was still told which dataset the thread stands on.
    assert 'diary-en' in str(policy_model.messages)


def test_same_dataset_experiment_receives_dataset_and_global_memory(monkeypatch):
    class Reader:
        def experiment(self, experiment_id):
            return {'experiment_id': experiment_id, 'dataset': 'diary-en'}

    widget.experiment_tools.set_experiment_reader(Reader())
    try:
        long_memory.save_memory_update(
            'diary-en', 'old-exp', 'chunking', 'old question', 'old answer',
            'previous dataset finding')
        with long_memory._connect() as db:
            db.execute('INSERT INTO global_memory(id, summary, updated_at) '
                       "VALUES (1, 'existing cross-dataset context', 'now')")
            db.commit()
        long_memory.save_memory_update(
            'diary-fa', 'other-exp', 'retrieval', 'q', 'a', 'other finding')
        policy = {'relevant': True, 'should_save': False,
                  'dataset_id': 'diary-en', 'subtopic': 'retrieval',
                  'reason': 'useful'}
        agent, _ = _setup(monkeypatch, policy)
        widget.ask('Did reranking help?', model='openai/gpt-5-nano',
                   thread='new-exp')
    finally:
        widget.experiment_tools.set_experiment_reader(None)

    context = '\n'.join(str(m.content) for m in agent.payloads[0]['messages'])
    assert 'previous dataset finding' in context
    assert 'existing cross-dataset context' in context


def test_malformed_active_experiment_cannot_supply_dataset_identity(
        monkeypatch, settled_memory):
    """A record whose dataset is not a string identifies nothing, so the model
    text that names one cannot stand in for it. Ruled on 2026-08-28: that is a
    guard on the write and not on the answer — the question is answered, and
    nothing is filed under an experiment the records could not read."""
    class Reader:
        def experiment(self, experiment_id):
            return {'experiment_id': experiment_id, 'dataset': ['diary-en']}

    widget.experiment_tools.set_experiment_reader(Reader())
    agent, _ = _setup(monkeypatch, {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'reusable',
    })
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError('summarizer was called')))

    result = widget.ask('Which retrieval setting should we retain?',
                        model='openai/gpt-5-nano', thread='bad-context')

    assert agent.invocations == 1
    assert result['reply'] == 'authoritative answer'
    assert result['memory']['saved'] is False
    assert 'active experiment context' in result['memory']['reason']
    assert long_memory.memory_context('diary-en') == ''


def test_a_thread_about_a_run_the_records_do_not_know_yet_is_answered(
        monkeypatch, settled_memory):
    """The case that made the old pre-flight refusal wrong: an experiment that
    is still running has no ledger row yet, and that is exactly when a reader
    asks about it. The question is answered; the turn is not filed, because
    memory about an experiment this lab cannot name is memory about nothing."""
    class Reader:
        def experiment(self, experiment_id):
            return None            # not recorded yet

        def board_rows(self, limit=500):
            return []

    widget.experiment_tools.set_experiment_reader(Reader())
    agent, _ = _setup(monkeypatch, {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'reusable',
    })
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError('summarizer was called')))

    result = widget.ask('How is this run doing?', model='openai/gpt-5-nano',
                        thread='still-running')

    assert agent.invocations == 1
    assert result['reply'] == 'authoritative answer'
    assert result['memory']['saved'] is False
    assert result['memory']['should_save'] is False
    assert 'active experiment context' in result['memory']['reason']
    assert long_memory.memory_context('diary-en') == ''


def test_sync_returns_answer_before_deferred_memory_work(monkeypatch):
    widget.experiment_tools.set_experiment_reader(_ExperimentReader('diary-en'))
    policy = {'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
              'subtopic': 'retrieval', 'reason': 'reusable'}
    _setup(monkeypatch, policy, answer='answer first')
    events = []

    def finish(*args, **kwargs):
        events.append('memory')
        return {**policy, 'saved': True}

    def defer(*args, **kwargs):
        events.append('deferred')
        return {'status': 'pending', 'saved': False}

    monkeypatch.setattr(widget.backends, '_finish_memory', finish)
    monkeypatch.setattr(widget.backends, '_defer_memory', defer)

    result = widget.ask('What should we retain?', model='openai/gpt-5-nano',
                        thread='sync-order')

    assert result['reply'] == 'answer first'
    assert result['memory']['status'] == 'pending'
    assert events == ['deferred']


def test_accepted_turn_reads_context_and_aggregates_by_dataset(monkeypatch):
    widget.experiment_tools.set_experiment_reader(_ExperimentReader('diary-en'))
    long_memory.save_memory_update(
        'diary-en', 'old-exp', 'chunking', 'old question', 'old answer',
        'previous dataset finding')
    policy = {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'reusable',
    }
    agent, _ = _setup(monkeypatch, policy)
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: {
                            'dataset_summary': 'new dataset finding',
                            'global_summary': '',
                        })

    result = widget.ask('Did reranking help?', model='openai/gpt-5-nano',
                        thread='new-exp')

    assert result['memory']['status'] == 'pending'
    _eventually(lambda: 'new dataset finding' in
                long_memory.memory_context('diary-en'))
    assert 'previous dataset finding' in long_memory.memory_context('diary-en')
    assert any('previous dataset finding' in str(message.content)
               for message in agent.payloads[0]['messages'])


def test_accepted_turn_can_update_global_pattern(monkeypatch):
    class Reader:
        def experiment(self, experiment_id):
            return {'experiment_id': experiment_id, 'dataset': 'diary-fa'}

        def board_rows(self, limit=500):
            return [{'dataset': 'diary-en'}, {'dataset': 'diary-fa'}]

    widget.experiment_tools.set_experiment_reader(Reader())
    policy = {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-fa',
        'subtopic': 'chunking', 'reason': 'reusable',
    }
    try:
        _setup(monkeypatch, policy)
        with long_memory._connect() as db:
            db.execute("INSERT INTO global_memory(id, summary, updated_at) "
                       "VALUES (1, 'existing cross-dataset context', 'now')")
            db.commit()
        monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                            lambda **kwargs: {
                                'dataset_summary': 'Farsi session finding',
                                'global_summary': 'Session-aware chunking recurs',
                            })

        widget.ask('What pattern should we retain?',
                   model='openai/gpt-5-nano', thread='fa-exp')
        _eventually(lambda: 'Session-aware chunking recurs' in
                    long_memory.memory_context('diary-fa'))
    finally:
        widget.experiment_tools.set_experiment_reader(None)

    _eventually(lambda: 'Farsi session finding' in
                long_memory.memory_context('diary-fa'))
    context = long_memory.memory_context('diary-fa')
    assert 'Farsi session finding' in context
    assert 'Session-aware chunking recurs' in context


class _StreamAgent(_Agent):
    """The stub `agent.stream` both streaming tests drive: one piece, then the
    state the run ended in — the two modes `_stream_agent` reads."""

    def stream(self, payload, config=None, stream_mode=None):
        yield ('messages', (AIMessage(content=self.answer), {}))
        yield ('values', {'messages': [*payload['messages'],
                                       AIMessage(content=self.answer)]})


def test_stream_emits_authoritative_answer_then_defers_the_decision(monkeypatch):
    """The streamed turn's last event is a status, not a verdict.

    It was the verdict until 2026-08-28, which meant the event stream stayed
    open through the policy call, the summarizer call and two readings of the
    board — all after the reader had every word of the answer. The decision is
    the same decision and still happens; it happens on a thread of its own, the
    way `ask` has always handed it over, so the generator ends where the answer
    does and the event says `pending`."""
    widget.experiment_tools.set_experiment_reader(_ExperimentReader('smoke-mini'))
    policy = {
        'relevant': True, 'should_save': True, 'dataset_id': 'smoke-mini',
        'subtopic': 'retrieval', 'reason': 'reusable',
    }
    _setup(monkeypatch, policy, answer='streamed answer')
    events = []

    def summarize(**kwargs):
        events.append('summarize')
        return {'dataset_summary': 'stream finding', 'global_summary': ''}

    def save(**kwargs):
        events.append('save')
        return {'saved': True, 'dataset_id': 'smoke-mini'}

    monkeypatch.setattr(widget.backends, '_summarize_memory_update', summarize)
    monkeypatch.setattr(widget.tools, 'save_widget_memory', save)
    monkeypatch.setattr(widget.backends, '_agent_for',
                        lambda model: _StreamAgent('streamed answer'))
    output = widget.stream('What should we retain?',
                           model='openai/gpt-5-nano', thread='stream-exp')

    assert next(output) == {'delta': 'streamed answer'}
    reply = next(output)
    assert reply == {'reply': 'streamed answer',
                     'input_tokens': None, 'output_tokens': None}

    # The same status `ask` reports for the same question: one turn, one
    # contract, whichever way the answer was asked for.
    assert next(output) == {'memory': {'status': 'pending', 'saved': False}}
    with pytest.raises(StopIteration):
        next(output)

    # Deferred, not dropped: the write lands, just not on this connection.
    _eventually(lambda: events == ['summarize', 'save'])
    _join_deferred_memory()


def test_the_streamed_generator_ends_before_the_summarizer_does(monkeypatch):
    """The claim the deferral is for: no summarizer call is between the last
    event and the caller getting control back.

    The summarizer here blocks until this test releases it, and this test does
    not release it until the generator is exhausted: a generator that still
    waited on it would be the one thing that cannot finish. It also names the
    thread it ran on, which must not be the thread the events were read from."""
    widget.experiment_tools.set_experiment_reader(_ExperimentReader('smoke-mini'))
    _setup(monkeypatch, {
        'relevant': True, 'should_save': True, 'dataset_id': 'smoke-mini',
        'subtopic': 'retrieval', 'reason': 'reusable'}, answer='streamed answer')
    running = threading.Event()
    release = threading.Event()
    ran_on, summarized = [], []

    def summarize(**kwargs):
        ran_on.append(threading.current_thread())
        running.set()
        release.wait(30)
        summarized.append(True)
        return {'dataset_summary': 'stream finding', 'global_summary': ''}

    monkeypatch.setattr(widget.backends, '_summarize_memory_update', summarize)
    monkeypatch.setattr(widget.tools, 'save_widget_memory',
                        lambda **kwargs: {'saved': True,
                                          'dataset_id': 'smoke-mini'})
    monkeypatch.setattr(widget.backends, '_agent_for',
                        lambda model: _StreamAgent('streamed answer'))
    try:
        events = list(widget.stream('What should we retain?',
                                    model='openai/gpt-5-nano',
                                    thread='stream-defer'))
        # The generator is exhausted while the summarizer is still inside a
        # call this test has not released: it cannot have waited for one.
        assert running.wait(5)
        assert summarized == []
        assert events[-1] == {'memory': {'status': 'pending', 'saved': False}}
        assert ran_on[0] is not threading.current_thread()
    finally:
        # Released and then waited for: a daemon still inside the write when
        # this test returns would be writing into widget.db after the next
        # test's autouse clear.
        release.set()
        _join_deferred_memory()


def test_one_memory_pass_reads_the_board_once(monkeypatch):
    """`leaderboard.board_rows` reads up to `SCAN` run files off disk, and the
    pass needed the rows twice — once for the foreign-experiment refusal, once
    for the validated ids the write is checked against. Two full readings per
    saved turn, growing with `.runs/`; one snapshot now serves both."""
    class Counted:
        def __init__(self):
            self.reads = 0

        def experiment(self, experiment_id):
            return {'experiment_id': experiment_id, 'dataset': 'diary-en'}

        def board_rows(self, limit=500):
            self.reads += 1
            return [{'experiment_id': 'this-exp', 'dataset': 'diary-en'}]

    reader = Counted()
    widget.experiment_tools.set_experiment_reader(reader)
    monkeypatch.setattr(widget.backends, '_memory_model',
                        lambda model: _PolicyModel({
                            'relevant': True, 'should_save': True,
                            'dataset_id': 'diary-en', 'subtopic': 'retrieval',
                            'reason': 'reusable'}))
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: {'dataset_summary': 'a finding',
                                          'global_summary': ''})

    decision = widget.backends._finish_memory(
        'Did reranking help?', 'an answer naming nothing', 'openai/gpt-5-nano',
        'this-exp', 'diary-en', '', '')

    assert decision['saved'] is True
    assert reader.reads == 1


def test_a_turn_no_policy_client_can_judge_is_not_reported_as_pending(
        monkeypatch):
    """`pending` promises a decision is coming. Where no policy client can be
    built there is nothing to make one: the deferred work would return `None`
    on a daemon thread with nobody to tell, and the reader would be left
    watching for a verdict that can never arrive. That turn says `unavailable`,
    the same word the panel uses for a judge nobody could reach."""
    monkeypatch.setattr(widget.backends, '_agent_for', lambda model: _Agent())

    def unbuildable(model):
        raise widget.WidgetUnavailable('OPENROUTER_API_KEY is not set')

    monkeypatch.setattr(widget.backends, '_memory_model', unbuildable)
    monkeypatch.setattr(widget.backends, '_finish_memory',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError('a decision nobody can make was '
                                           'started anyway')))

    result = widget.ask('Which chunker should I compare?',
                        model='openai/gpt-5-nano', thread='keyless-exp')

    assert result['reply'] == 'authoritative answer'
    assert result['memory'] == {'status': 'unavailable', 'saved': False}


def test_save_failure_keeps_the_authoritative_answer(monkeypatch):
    widget.experiment_tools.set_experiment_reader(_ExperimentReader('diary-en'))
    policy = {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'reusable',
    }
    _setup(monkeypatch, policy, answer='answer survives storage failure')
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: {'dataset_summary': 'finding',
                                          'global_summary': ''})

    def fail(**kwargs):
        raise RuntimeError('disk is full')

    monkeypatch.setattr(widget.tools, 'save_widget_memory', fail)
    result = widget.ask('What should we retain?', model='openai/gpt-5-nano',
                        thread='save-failure')

    assert result['reply'] == 'answer survives storage failure'
    assert result['memory']['status'] == 'pending'


class _TwoDatasetReader:
    """Two recorded experiments on two corpora — the shape the real bug had."""
    ROWS = {'this-exp': 'meetings-de', 'other-exp': 'smoke-import-check'}

    def experiment(self, experiment_id):
        dataset = self.ROWS.get(experiment_id)
        return ({'experiment_id': experiment_id, 'dataset': dataset}
                if dataset else None)

    def board_rows(self, limit=500):
        return [{'experiment_id': i, 'dataset': d} for i, d in self.ROWS.items()]


def test_an_answer_about_another_dataset_is_not_filed_under_this_one(monkeypatch):
    """Seen 2026-08-27 in the developer's own widget.db: a `meetings-de`
    thread asked about 'the last experiment', the agent described the newest
    experiment overall — on `smoke-import-check` — and the summary of that
    answer was filed as memory *about meetings-de*. The thread's dataset is
    the row's provenance; an answer whose experiments belong to another corpus
    contradicts it, and a contradiction is a refusal, not a filing."""
    widget.experiment_tools.set_experiment_reader(_TwoDatasetReader())
    monkeypatch.setattr(widget.backends, '_memory_model',
                        lambda model: _PolicyModel({
                            'relevant': True, 'should_save': True,
                            'dataset_id': 'meetings-de',
                            'subtopic': 'overview', 'reason': 'reusable'}))
    answer = 'The latest experiment is other-exp on smoke-import-check.'
    out = widget.backends._finish_memory(
        'Tell me about the last experiment.', answer, 'openai/gpt-5-nano',
        'this-exp', 'meetings-de', '', 'you: an earlier question')
    assert out['saved'] is False
    assert 'other-exp' in out['reason'] and 'smoke-import-check' in out['reason']
    assert long_memory.memory_context('meetings-de') == ''
    with long_memory._connect() as db:
        assert db.execute('SELECT count(*) FROM memory_updates').fetchone()[0] == 0


def test_an_experiment_thread_tells_the_agent_which_experiment_it_is_about(monkeypatch):
    """The other half of the same incident: the agent was never told which
    experiment the thread belonged to, so 'the last experiment' could only
    mean the newest on the board. The turn now opens with the thread's own
    experiment and dataset, from the validated record, never from model text."""
    widget.experiment_tools.set_experiment_reader(_TwoDatasetReader())
    policy = {'relevant': True, 'should_save': False,
              'dataset_id': 'meetings-de', 'subtopic': '', 'reason': 'a question'}
    agent, _ = _setup(monkeypatch, policy)
    widget.ask('Tell me about the last experiment.', model='openai/gpt-5-nano',
               thread='this-exp')
    opening = str(agent.payloads[0]['messages'][0].content)
    assert 'this-exp' in opening and 'meetings-de' in opening


def test_a_threads_system_lines_are_written_once_not_once_per_turn(monkeypatch):
    """Seen in the developer trace: every turn appended the active-experiment
    line and the memory context again, so a five-turn thread carried ten
    system messages and the model reread them all each time. A system line
    already in the thread, word for word, is not added a second time; a memory
    context that changed is, because it says something new."""
    from raglab.agents.widget.tests.widget_examples import write_messages
    from langchain_core.messages import SystemMessage
    widget.experiment_tools.set_experiment_reader(_TwoDatasetReader())
    policy = {'relevant': True, 'should_save': False,
              'dataset_id': 'meetings-de', 'subtopic': '', 'reason': 'q'}
    agent, _ = _setup(monkeypatch, policy)
    widget.ask('first?', model='openai/gpt-5-nano', thread='this-exp')
    first = [str(m.content) for m in agent.payloads[0]['messages']
             if isinstance(m, SystemMessage)]
    assert len(first) == 2          # active experiment + memory context
    write_messages('this-exp', agent.payloads[0]['messages'] + [
        AIMessage(content='a')])   # what the graph would have kept
    widget.ask('second?', model='openai/gpt-5-nano', thread='this-exp')
    second = [str(m.content) for m in agent.payloads[1]['messages']
              if isinstance(m, SystemMessage)]
    assert second == []


def _reason_on_the_row(thread: str) -> str:
    from raglab.agents.widget import turn_logger
    rows = turn_logger.list_turns(thread)
    assert rows, f'no turn-log row was written for {thread}'
    return rows[-1]['memory_reason'] or ''


def test_a_deferred_decision_nobody_can_return_to_still_says_why(monkeypatch):
    """The decision runs on a daemon thread after the answer has gone, so its
    return value has no reader. Every non-save outcome used to end there: the
    summarizer raised, `save_error` was set on a dict the thread dropped, and
    the turn was unfiled forever with no record of why. It goes on the turn's
    own row now."""
    widget.experiment_tools.set_experiment_reader(_ExperimentReader('diary-en'))
    _setup(monkeypatch, {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'reusable'})
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: (_ for _ in ()).throw(
                            RuntimeError('the summarizer is down')))

    result = widget.ask('Which retrieval setting should we retain?',
                        model='openai/gpt-5-nano', thread='exp-lost-reason')

    assert result['memory'] == {'status': 'pending', 'saved': False}
    _join_deferred_memory()
    reason = _reason_on_the_row('exp-lost-reason')
    assert reason.startswith('not filed:')
    assert 'the summarizer is down' in reason
    assert long_memory.memory_context('diary-en') == ''


def test_a_refused_save_and_an_accepted_one_both_name_themselves_on_the_row(
        monkeypatch):
    """The column says what the memory pass decided, not only what went wrong.
    A column filled only by failures reads as a failure list, and a blank one
    cannot then be told from a pass that never happened."""
    widget.experiment_tools.set_experiment_reader(_ExperimentReader('diary-en'))
    _setup(monkeypatch, {
        'relevant': True, 'should_save': False, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'not worth keeping'})
    widget.ask('Was that useful?', model='openai/gpt-5-nano',
               thread='exp-declined')
    _join_deferred_memory()
    assert _reason_on_the_row('exp-declined') == (
        'not filed: not worth keeping')

    _setup(monkeypatch, {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'reusable'})
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: {'dataset_summary': 'a finding',
                                          'global_summary': ''})
    widget.ask('Which retrieval setting should we retain?',
               model='openai/gpt-5-nano', thread='exp-kept')
    _join_deferred_memory()
    assert _reason_on_the_row('exp-kept') == 'filed: reusable'


def test_a_turn_that_answered_survives_a_turn_log_write_that_failed(monkeypatch):
    """Three widget stores share one sqlite file behind three independent
    locks, and the deferred writer is on the stream path too — so one turn's
    write can find the file busy while the next turn is answering. An
    operational write is reported and never fatal: the reader keeps the answer
    the lab really produced."""
    import sqlite3

    widget.experiment_tools.set_experiment_reader(_ExperimentReader('diary-en'))
    _setup(monkeypatch, {
        'relevant': True, 'should_save': False, 'dataset_id': 'diary-en',
        'subtopic': '', 'reason': 'a question'})
    monkeypatch.setattr(widget.backends.turn_logger, 'log_turn',
                        lambda **kwargs: (_ for _ in ()).throw(
                            sqlite3.OperationalError('database is locked')))
    widget.HOOK_LOG.clear()

    result = widget.ask('Which chunker won?', model='openai/gpt-5-nano',
                        thread='exp-busy-file')

    assert result['reply'] == 'authoritative answer'
    assert result['memory'] == {'status': 'pending', 'saved': False}
    assert any('database is locked' in line for line in widget.HOOK_LOG), (
        'a failed operational write must be reported, not swallowed')
    _join_deferred_memory()


def test_the_streamed_turn_also_survives_a_turn_log_write_that_failed(
        monkeypatch):
    """The same guarantee on the path the browser actually uses."""
    import sqlite3

    widget.experiment_tools.set_experiment_reader(_ExperimentReader('diary-en'))
    agent, _ = _setup(monkeypatch, {
        'relevant': True, 'should_save': False, 'dataset_id': 'diary-en',
        'subtopic': '', 'reason': 'a question'})
    agent.stream = lambda payload, config=None, stream_mode=None: iter([
        ('values', {'messages': [*payload['messages'],
                                 AIMessage(content='streamed answer')]})])
    monkeypatch.setattr(widget.backends.turn_logger, 'log_turn',
                        lambda **kwargs: (_ for _ in ()).throw(
                            sqlite3.OperationalError('database is locked')))

    events = list(widget.stream('Which chunker won?',
                                model='openai/gpt-5-nano',
                                thread='exp-busy-stream'))

    assert events[-2]['reply'] == 'streamed answer'
    assert events[-1] == {'memory': {'status': 'pending', 'saved': False}}
    _join_deferred_memory()


def test_a_saved_turn_resolves_the_threads_dataset_exactly_once(monkeypatch):
    """`_preflight` settles the trusted dataset for the whole turn because each
    resolution is a ledger query plus a run-file read. The write gate then
    asked `validated_dataset_ids(thread)` for the same value a second time —
    and handed it the raw thread, bypassing `_experiment_of`, which this file
    calls its single reading of the thread's experiment."""
    class CountingReader(_ExperimentReader):
        def __init__(self, dataset):
            super().__init__(dataset)
            self.lookups = []

        def experiment(self, experiment_id):
            self.lookups.append(experiment_id)
            return super().experiment(experiment_id)

        def board_rows(self, limit=500):
            return [{'experiment_id': 'exp-other', 'dataset': 'meetings-de'}]

    reader = CountingReader('diary-en')
    widget.experiment_tools.set_experiment_reader(reader)
    _setup(monkeypatch, {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'reusable'})
    saved = {}
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: {'dataset_summary': 'a finding',
                                          'global_summary': 'a pattern'})
    monkeypatch.setattr(widget.tools, 'save_widget_memory',
                        lambda **kwargs: saved.update(kwargs) or {'saved': True})

    widget.ask('Which retrieval setting should we retain?',
               model='openai/gpt-5-nano', thread='exp-once')
    _join_deferred_memory()

    assert reader.lookups == ['exp-once']
    # The board's ids plus this thread's own, which is what the second lookup
    # was ever for.
    assert saved['validated_dataset_ids'] == {'meetings-de', 'diary-en'}
