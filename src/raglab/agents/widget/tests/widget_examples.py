"""One way to seed a widget conversation, used by every test that needs
turns already sitting in the checkpointer before it makes its assertion.

The shape a checkpoint has to take is the saver's business, not the tests'.
`_write_messages` in `test_conversation_memory.py` first put messages into
`memory.saver()` this way — `empty_checkpoint()` filled in, with
`checkpoint_ns` present in the config because `SqliteSaver.put` reads it
without a default — and every other place that needs a seeded thread now
calls this rather than repeating that shape. A second, drifted copy is how a
langgraph upgrade that changes what the saver accepts gets fixed in one
place and left silently wrong in the other, surfacing later as a confusing
failure in whichever file still carries the old shape.
"""
from langgraph.checkpoint.base import empty_checkpoint

from raglab.agents.widget import conversation_memory as memory


def write_messages(thread: str, messages: list, experiment_id: str = '',
                   started_at: str = '2026-08-21T00:00:00+00:00') -> None:
    """Whatever messages a caller names, straight into the checkpointer without
    an LLM: the memory is what is under test, not the agent that fills it."""
    config = {'configurable': {'thread_id': thread, 'checkpoint_ns': ''}}
    saver = memory.saver()
    checkpoint = empty_checkpoint()
    checkpoint['id'] = f'{thread}-1'
    checkpoint['ts'] = started_at
    checkpoint['channel_values'] = {
        'messages': messages,
        'experiment_id': experiment_id, 'started_at': started_at}
    saver.put(config, checkpoint, {'source': 'update', 'step': 1}, {})
