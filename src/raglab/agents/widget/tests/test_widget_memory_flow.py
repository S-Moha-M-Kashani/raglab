# this is a unit test
"""The policy-to-memory flow is selective and delivery ordered."""
from langchain_core.messages import AIMessage
import pytest

from raglab.agents import widget
from raglab.agents.widget import long_term_memory as long_memory


@pytest.fixture(autouse=True)
def _clean_long_term_memory():
    long_memory.clear_long_term_memory()
    yield
    long_memory.clear_long_term_memory()


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

    assert result['reply'] == 'authoritative answer'
    assert result['memory']['should_save'] is False
    assert agent.invocations == 1
    assert long_memory.memory_context('diary-en') == ''


def test_accepted_turn_reads_context_and_aggregates_by_dataset(monkeypatch):
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

    assert result['memory']['saved'] is True
    assert 'previous dataset finding' in long_memory.memory_context('diary-en')
    assert 'new dataset finding' in long_memory.memory_context('diary-en')
    assert any('previous dataset finding' in str(message.content)
               for message in agent.payloads[0]['messages'])


def test_accepted_turn_can_update_global_pattern(monkeypatch):
    policy = {
        'relevant': True, 'should_save': True, 'dataset_id': 'diary-fa',
        'subtopic': 'chunking', 'reason': 'reusable',
    }
    _setup(monkeypatch, policy)
    monkeypatch.setattr(widget.backends, '_summarize_memory_update',
                        lambda **kwargs: {
                            'dataset_summary': 'Farsi session finding',
                            'global_summary': 'Session-aware chunking recurs',
                        })

    widget.ask('What pattern should we retain?', model='openai/gpt-5-nano',
               thread='fa-exp')

    context = long_memory.memory_context('diary-fa')
    assert 'Farsi session finding' in context
    assert 'Session-aware chunking recurs' in context


def test_stream_emits_authoritative_answer_before_save(monkeypatch):
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
    assert result['memory']['saved'] is False
    assert 'disk is full' in result['memory']['save_error']
