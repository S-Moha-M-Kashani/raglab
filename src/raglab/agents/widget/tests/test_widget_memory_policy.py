"""Unit coverage for the widget's structured memory-policy contract."""

from langchain_core.messages import HumanMessage

from raglab.agents.widget import conversation_memory as memory
from raglab.agents.widget import hooks
from raglab.agents.widget import prompts


def test_the_state_stamp_carries_only_what_the_turn_can_know():
    # this is a unit test
    """`dataset_stamp` writes the two things settled before the answer: which
    corpus the thread stands on, and which experiment it is about, stripped.

    It replaced `policy_state`, which flattened a memory verdict into four
    more channels. The verdict is taken after the answer now, so nothing
    evaluated one before the stamp and all four were written empty on every
    checkpoint — a record of a decision nobody had made. Neither the stamp nor
    `WidgetState` carries them any more."""
    assert memory.dataset_stamp('diary-fa', ' exp-42 ') == {
        'dataset_id': 'diary-fa',
        'experiment_id': 'exp-42',
    }
    assert memory.dataset_stamp() == {'dataset_id': '', 'experiment_id': ''}
    assert set(memory.WidgetState.__annotations__) & {
        'relevant', 'should_save', 'subtopic', 'reason'} == set()


def test_an_irrelevant_policy_can_never_carry_a_save_permission():
    # this is a unit test
    """The one guarantee `policy_state` used to hold, asserted where it now
    lives: the policy seam itself clears `should_save` on an irrelevant
    verdict, so no state channel has to normalize one afterwards."""
    class Model:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return {'relevant': False, 'should_save': True,
                    'dataset_id': 'diary-fa', 'subtopic': 'retrieval',
                    'reason': 'The model incorrectly requested a save.'}

    policy = hooks.evaluate_memory_policy('Which chunker won?', Model())
    assert policy.relevant is False
    assert policy.should_save is False


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


def test_four_existing_hooks_remain_unchanged():
    # this is a convention test
    assert len(hooks.MIDDLEWARE) == 4
    assert [hook.name for hook in hooks.MIDDLEWARE] == [
        'check_request', 'trim_and_call', 'log_tool_call', 'close_the_log']
