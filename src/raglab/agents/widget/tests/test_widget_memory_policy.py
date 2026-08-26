"""Unit coverage for the widget's structured memory-policy contract."""

from langchain_core.messages import HumanMessage

from raglab.agents.widget import conversation_memory as memory
from raglab.agents.widget import hooks
from raglab.agents.widget import prompts


def test_widget_state_policy_fields_have_safe_defaults():
    # this is a unit test
    defaults = memory.widget_state_defaults('exp-42')
    assert defaults == {
        'relevant': False,
        'should_save': False,
        'dataset_id': '',
        'subtopic': '',
        'reason': '',
        'experiment_id': 'exp-42',
    }


def test_policy_state_never_allows_an_irrelevant_policy_to_save():
    # this is a unit test
    policy = memory.MemoryPolicy(
        relevant=False,
        should_save=True,
        dataset_id='diary-fa',
        subtopic='retrieval',
        reason='The model incorrectly requested a save.',
    )
    state = memory.policy_state(policy, ' exp-42 ')
    assert state == {
        'relevant': False,
        'should_save': False,
        'dataset_id': 'diary-fa',
        'subtopic': 'retrieval',
        'reason': 'The model incorrectly requested a save.',
        'experiment_id': 'exp-42',
    }


def test_relevance_guard_refuses_empty_long_and_unrelated_text():
    # this is a unit test
    assert 'question' in memory.relevance_guard('   ').lower()
    assert 'long' in memory.relevance_guard('x' * 501).lower()
    refusal = memory.relevance_guard("What's the weather in Berlin today?")
    assert refusal and 'RAG lab' in refusal


def test_question_length_limit_is_shared_by_guard_and_widget_hook():
    # this is a unit test
    assert hooks.MAX_QUESTION == memory.MAX_RELEVANCE_TEXT
    assert memory.relevance_guard('x' * hooks.MAX_QUESTION) is None
    assert memory.relevance_guard('x' * (hooks.MAX_QUESTION + 1))


def test_relevance_guard_accepts_a_lab_question():
    # this is a unit test
    assert memory.relevance_guard('Which chunker should I compare?') is None


def test_structured_policy_sets_dataset_and_subtopic_state():
    # this is a unit test
    class Model:
        def with_structured_output(self, schema):
            assert schema is memory.MemoryPolicy
            return self

        def invoke(self, messages):
            assert prompts.MEMORY_POLICY_PROMPT in messages[0][1]
            return {
                'relevant': True,
                'should_save': True,
                'dataset_id': 'diary-fa',
                'subtopic': 'retrieval',
                'reason': 'Useful guidance for a future run.',
            }

    policy = hooks.evaluate_memory_policy(
        'How did retrieval perform on this experiment?', Model(),
        experiment_id='exp-42')
    assert policy == memory.MemoryPolicy(
        relevant=True, should_save=True, dataset_id='diary-fa',
        subtopic='retrieval', reason='Useful guidance for a future run.')


def test_unavailable_or_malformed_policy_fails_closed():
    # this is a unit test
    class Unavailable:
        def with_structured_output(self, schema):
            raise RuntimeError('no policy model')

    class Malformed:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return {'relevant': 'yes'}

    for model in (Unavailable(), Malformed()):
        policy = hooks.evaluate_memory_policy('Which index should I use?', model)
        assert policy.relevant is False
        assert policy.should_save is False
        assert policy.dataset_id == ''
        assert policy.subtopic == ''
        assert policy.reason


def test_obvious_irrelevance_refusal_is_the_guard_copy():
    # this is a unit test
    message = HumanMessage(content='Tell me a joke about penguins.')
    refusal = hooks.refusal_for_message(message)
    assert refusal == memory.relevance_guard(message.content)


def test_six_existing_hooks_remain_unchanged():
    # this is a convention test
    assert len(hooks.MIDDLEWARE) == 6
    assert [hook.name for hook in hooks.MIDDLEWARE] == [
        'check_request', 'note_prompt', 'trim_and_call', 'log_tool_call',
        'check_reply', 'close_the_log']
