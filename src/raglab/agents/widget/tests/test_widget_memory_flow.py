# this is a unit test
"""The policy-to-memory flow is selective and delivery ordered."""
from langchain_core.messages import AIMessage
import pytest
import time

from raglab.agents import widget
from raglab.agents.widget import long_term_memory as long_memory


@pytest.fixture(autouse=True)
def _clean_long_term_memory():
    widget.experiment_tools.set_experiment_reader(None)
    long_memory.clear_long_term_memory()
    yield
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


def _eventually(predicate):
    for _ in range(100):
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


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


def test_irrelevant_policy_result_does_not_write(monkeypatch):
    agent, _ = _setup(monkeypatch, {
        'relevant': False, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'misc', 'reason': 'not reusable',
    })
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError('summarizer was called')))

    result = widget.ask('Which index should I use?', model='openai/gpt-5-nano',
                        thread='exp-irrelevant')

    assert 'not reusable' in result['reply']
    assert result['memory']['should_save'] is False
    assert agent.invocations == 0
    assert long_memory.memory_context('diary-en') == ''


def test_relevant_policy_cannot_invent_dataset_without_active_experiment(monkeypatch):
    agent, _ = _setup(monkeypatch, {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
        'subtopic': 'retrieval', 'reason': 'reusable',
    })

    result = widget.ask('Which retrieval setting should we retain?',
                        model='openai/gpt-5-nano', thread='general')

    assert agent.invocations == 1
    assert result['reply'] == 'authoritative answer'
    assert result['memory']['saved'] is False
    assert result['memory']['status'] == 'not_saved'


def test_policy_dataset_mismatch_refuses_before_agent_and_save(monkeypatch):
    class Reader:
        def experiment(self, experiment_id):
            return {'experiment_id': experiment_id, 'dataset': 'diary-en'}

    widget.experiment_tools.set_experiment_reader(Reader())
    try:
        agent, policy_model = _setup(monkeypatch, {
            'relevant': True, 'should_save': True, 'dataset_id': 'diary-fa',
            'subtopic': 'retrieval', 'reason': 'wrong dataset',
        })
        result = widget.ask('Which index should I use?',
                            model='openai/gpt-5-nano', thread='exp-context')
    finally:
        widget.experiment_tools.set_experiment_reader(None)

    assert agent.invocations == 0
    assert result['memory']['saved'] is False
    assert 'dataset' in result['reply'].lower()
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


def test_malformed_active_experiment_cannot_supply_dataset_identity(monkeypatch):
    class Reader:
        def experiment(self, experiment_id):
            return {'experiment_id': experiment_id, 'dataset': ['diary-en']}

    widget.experiment_tools.set_experiment_reader(Reader())
    try:
        agent, _ = _setup(monkeypatch, {
            'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
            'subtopic': 'retrieval', 'reason': 'reusable',
        })
        result = widget.ask('Which retrieval setting should we retain?',
                            model='openai/gpt-5-nano', thread='bad-context')
    finally:
        widget.experiment_tools.set_experiment_reader(None)

    assert agent.invocations == 0
    assert result['memory']['blocked'] is True
    assert 'active experiment context' in result['reply']


def test_sync_returns_answer_before_deferred_memory_work(monkeypatch):
    widget.experiment_tools.set_experiment_reader(_ExperimentReader('diary-en'))
    policy = {'relevant': True, 'should_save': True, 'dataset_id': 'diary-en',
              'subtopic': 'retrieval', 'reason': 'reusable'}
    _setup(monkeypatch, policy, answer='answer first')
    events = []

    def finish(*args, **kwargs):
        events.append('memory')
        return {**policy, 'saved': True}

    monkeypatch.setattr(widget.backends, '_finish_memory', finish)
    monkeypatch.setattr(widget.backends, '_defer_memory',
                        lambda *args, **kwargs: events.append('deferred'))

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


def test_stream_emits_authoritative_answer_before_save(monkeypatch):
    widget.experiment_tools.set_experiment_reader(_ExperimentReader('smoke-mini'))
    policy = {
        'relevant': True, 'should_save': True, 'dataset_id': 'smoke-mini',
        'subtopic': 'retrieval', 'reason': 'reusable',
    }
    agent, _ = _setup(monkeypatch, policy, answer='streamed answer')
    events = []

    def summarize(**kwargs):
        events.append('summarize')
        return {'dataset_summary': 'stream finding', 'global_summary': ''}

    def save(**kwargs):
        events.append('save')
        return {'saved': True, 'dataset_id': 'smoke-mini'}

    monkeypatch.setattr(widget.backends, '_summarize_memory_update', summarize)
    monkeypatch.setattr(widget.tools, 'save_widget_memory', save)

    # The stream stub is supplied by the backend tests; this test pins the
    # save boundary by using the public stream path's final event.
    class StreamAgent(_Agent):
        def stream(self, payload, config=None, stream_mode=None):
            yield ('messages', (AIMessage(content='streamed answer'), {}))
            yield ('values', {'messages': [
                *payload['messages'], AIMessage(content='streamed answer')
            ]})

    stream_agent = StreamAgent('streamed answer')
    monkeypatch.setattr(widget.backends, '_agent_for', lambda model: stream_agent)
    output = widget.stream('What should we retain?',
                           model='openai/gpt-5-nano', thread='stream-exp')

    assert next(output) == {'delta': 'streamed answer'}
    reply = next(output)
    assert reply == {'reply': 'streamed answer',
                     'input_tokens': None, 'output_tokens': None}
    assert events == []
    assert long_memory.memory_context('smoke-mini') == ''

    status = next(output)
    assert status['memory']['saved'] is True
    assert status['memory']['dataset_id'] == 'smoke-mini'
    assert events == ['summarize', 'save']

    with pytest.raises(StopIteration):
        next(output)


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


def test_an_answer_about_another_dataset_is_not_filed_under_this_one():
    """Seen 2026-08-27 in the developer's own widget.db: a `meetings-de`
    thread asked about 'the last experiment', the agent described the newest
    experiment overall — on `smoke-import-check` — and the summary of that
    answer was filed as memory *about meetings-de*. The thread's dataset is
    the row's provenance; an answer whose experiments belong to another corpus
    contradicts it, and a contradiction is a refusal, not a filing."""
    widget.experiment_tools.set_experiment_reader(_TwoDatasetReader())
    decision = {'relevant': True, 'should_save': True,
                'dataset_id': 'meetings-de', 'subtopic': 'overview',
                'reason': 'reusable'}
    answer = 'The latest experiment is other-exp on smoke-import-check.'
    out = widget.backends._finish_memory(
        'Tell me about the last experiment.', answer, dict(decision),
        model=None, thread='this-exp')
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
