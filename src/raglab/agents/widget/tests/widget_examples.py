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
import uuid

from langgraph.checkpoint.base import empty_checkpoint

from raglab.agents.widget import conversation_memory as memory


def write_messages(thread: str, messages: list) -> None:
    """Whatever messages a caller names, straight into the checkpointer without
    an LLM: the memory is what is under test, not the agent that fills it.

    The two state fields beside the messages are not invented here — they are
    whatever `memory.thread_stamp` would have written for this thread, which is
    what a real turn writes through `agent.invoke`. They used to be parameters
    with plausible-looking defaults, and that is exactly how a test came to
    assert values production had never once produced: the fields were declared
    on `WidgetState`, written by nothing, and read back as two empty strings by
    every reader that was not a test. A seeded thread has to look like a real
    one, or seeding it proves nothing about what the route reports."""
    stamp = memory.thread_stamp(thread)
    # `thread_stamp` leaves `started_at` out when the thread already has one —
    # that is what stops a later turn from moving it — so re-seeding a thread
    # keeps the moment the first seed wrote rather than inventing a second.
    started = stamp.get('started_at') or memory.history(thread)['started_at']
    config = {'configurable': {'thread_id': thread, 'checkpoint_ns': ''}}
    saver = memory.saver()
    checkpoint = empty_checkpoint()
    # A real checkpoint id, and one that says this is where the thread began.
    #
    # `f'{thread}-1'` read well and was unreadable to langgraph: `_algo`
    # unhexlifies a checkpoint id when a run resumes from one, so a seeded
    # thread could be read back but never handed to the compiled agent — which
    # is exactly what a test that drives production's own graph over a
    # pre-existing conversation has to do.
    #
    # The zero prefix is not decoration either. The saver returns a thread's
    # checkpoints newest-id-first, and a real one is time-ordered (uuid7), so
    # an id built from the thread's name alone could sort *above* the
    # checkpoint a later run writes — the seed would shadow the run, and a
    # test would read back the state it wrote rather than the state the graph
    # produced. Zeroed high bits put every seed before every run; the tail is
    # derived from the thread's name, so re-seeding one thread keeps the same
    # id and two threads never share one.
    checkpoint['id'] = ('00000000-0000-7000-8000-'
                        + uuid.uuid5(uuid.NAMESPACE_URL, thread).hex[-12:])
    checkpoint['ts'] = started
    checkpoint['channel_values'] = {'messages': messages, **stamp,
                                    'started_at': started}
    saver.put(config, checkpoint, {'source': 'update', 'step': 1}, {})
