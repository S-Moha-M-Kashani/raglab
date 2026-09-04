"""The helper in the corner of every surface.

Outside the measured seam: it writes no run, no ledger row and no number, which
is the only reason it may trace anywhere. A chat turn is a request rather than
a job, so these routes answer synchronously — except the streaming one, which
sends the same turn as it is written and can only report a late failure inside
the stream it already opened.
"""
import json

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from raglab.agents import widget

def _sent_events(events):
    """One iterator of dicts, encoded as server-sent events: `data: ` and the
    JSON, one object per line, blank line between. The lab's only streaming
    route, so this lives beside it rather than in the shared plumbing.

    A failure part-way through is encoded as an `error` event and the stream
    ends there. It cannot be a status code — those were spent on the first
    piece — and it must not be silence either: a page that saw the pieces stop
    with no word would have to guess whether the answer had finished, and
    guessing "finished" would show a fragment as a whole reply. Every
    exception is caught, not only the widget's own: an iterator that dies
    unexpectedly must still say so on the wire it was writing to.
    """
    try:
        for event in events:
            yield f'data: {json.dumps(_safe_widget_event(event))}\n\n'
    except Exception as error:
        yield f'data: {json.dumps({"error": str(error)})}\n\n'


def _safe_widget_event(event):
    """Keep widget responses to the reader-facing memory decision.

    The widget backend needs the full policy and save result to coordinate its
    own work, but those fields are internal implementation details and may
    contain dataset or model text. The panel exposes only the useful outcome,
    while leaving every other event field—including the authoritative reply—
    unchanged. A malformed or absent memory value is omitted rather than
    guessed at.

    Four statuses, and every one of them is a thing `widget.ask` and
    `widget.stream` can actually put on an event.

    `pending` is the ordinary case: both answer paths file after they answer,
    so the decision is still being taken when the turn lands. `unavailable` is
    a turn nobody judged — no policy client could be built. `not_filed` is a
    turn that ran to the end and must not be filed, which today means the
    tool-hop guard answered in the widget's own voice. `irrelevant` is the
    deterministic relevance guard's refusal (`backends._refused`), the one
    memory block that still arrives here as policy booleans.

    There were two more until 2026-08-28: `saved` and `not_saved`, read off a
    resolved verdict's `saved` flag. No caller can produce one any more — the
    verdict is taken on a thread of its own, after the response has gone, and
    it lands on the `widget_turn_log` row rather than on any event. Statuses
    this route cannot emit were removed rather than left standing, because a
    documented status is a promise about what a client may see, and a promise
    only a monkeypatched test could keep is not one.

    Only `pending`, `unavailable` and `not_filed` reach the panel — the
    deferred decision is the only memory event the page routes to a reader,
    and of those three the page writes a line for the first two (`widget.js`,
    `widgetMemoryStatus`). `irrelevant` is this API's alone, for a client
    reading the route rather than the page.
    """
    if not isinstance(event, dict) or 'memory' not in event:
        return event
    memory = event.get('memory')
    if not isinstance(memory, dict):
        return {key: value for key, value in event.items() if key != 'memory'}
    deferred = memory.get('status')
    if deferred in ('pending', 'unavailable', 'not_filed'):
        # A deferred or withheld decision carries no booleans, because there is
        # no verdict to put in them. Dropping the block would leave the reader
        # with no memory line at all, and no line reads as "nothing was worth
        # keeping" — a verdict, and one nobody gave.
        return event | {'memory': {'status': deferred}}
    if not all(isinstance(memory.get(key), bool)
               for key in ('relevant', 'should_save', 'saved')):
        return {key: value for key, value in event.items() if key != 'memory'}
    if memory.get('unavailable') is True:
        status = 'unavailable'
    elif not memory['relevant']:
        status = 'irrelevant'
    else:
        # A relevant, boolean-bearing memory block is a shape no answer path
        # builds. Rather than guess a verdict for it, the block is dropped the
        # way a malformed one is.
        return {key: value for key, value in event.items() if key != 'memory'}
    return event | {'memory': {'status': status}}


def register(app, context) -> None:
    """The widget owns its own memory and models, so nothing off the context is
    read here: these routes carry a message to it and its answer back."""

    @app.get('/api/widget')
    def widget_options():
        """The widget's own model list and the questions its empty log
        offers — served, because neither panel keeps a model list of its own,
        and because the starters are model-facing text, which in this project
        is a fixture rather than a string in a page. They ride the response
        that already exists: no new route, and no new import inside the
        widget package, which is a sealed leaf."""
        return {'models': [{'value': value, 'label': label}
                           for value, (_, label) in widget.WIDGET_MODELS.items()],
                'default': widget.DEFAULT_MODEL,
                'starters': widget.STARTERS}

    @app.post('/api/widget')
    def widget_chat(payload: dict):
        """The corner widget's endpoint: a question in, the widget's reply
        out. Synchronous, not a job — a chat turn is a request, not a run.
        An unknown model raises ValueError, answered as a 400 naming it."""
        message = (payload.get('message') or '').strip()
        if not message:
            raise HTTPException(400, 'message is empty')
        try:
            # The thread is the page's claim about which conversation this is —
            # the lab's active experiment, or `general`. The route only carries
            # it. The reply arrives with its token account and is served
            # unchanged.
            return _safe_widget_event(widget.ask(
                message, (payload.get('model') or '').strip(),
                thread=(payload.get('thread') or '').strip()))
        except widget.WidgetUnavailable as error:
            # The lab is up; its widget is not — the /api/queries split.
            raise HTTPException(502, str(error))

    @app.post('/api/widget/stream')
    def widget_stream(payload: dict):
        """The same turn as POST /api/widget, sent as it is written: one
        server-sent event per piece of the answer, then the reply event —
        carrying the reply the lab now holds and its token account, the very
        body the other route returns whole — and after it one memory event.
        The reply is the event with a `reply` key, not the last one: the
        memory event is last, and it says only what the decision about keeping
        this turn is at the moment it is written, because that decision now
        outlives the response (`backends._defer_memory`). A client that reads
        the final event as the reply reads the memory block as an answer. The
        page renders the pieces as they land and adopts the reply event, so
        what stays on screen is what the conversation log holds rather than
        whatever the pieces spelled.

        `widget.stream` raises before it yields anything, which is what keeps a
        refusal a status code: an unserved model is a 400 and an unreachable
        widget a 502, decided here, before the response opens. Once the first
        piece is out the status code is spent, so a failure after that can only
        be said inside the stream — an `error` event, and no `reply` event
        ever, because a half-written answer must never be handed over as a
        whole one."""
        message = (payload.get('message') or '').strip()
        if not message:
            raise HTTPException(400, 'message is empty')
        try:
            events = widget.stream(message,
                                   (payload.get('model') or '').strip(),
                                   thread=(payload.get('thread') or '').strip())
        except widget.WidgetUnavailable as error:
            raise HTTPException(502, str(error))
        return StreamingResponse(
            _sent_events(events), media_type='text/event-stream',
            # No cache anywhere in front of a conversation, and no proxy
            # buffering: a stream held back until it completes is the sudden
            # printing this route exists to end.
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/api/widget/history')
    def widget_history(thread: str = ''):
        """One conversation, as the widget holds it. This is what a page draws
        after a refresh: the lab is the only copy of the transcript, so a
        reader's log and the model's memory cannot drift apart. A thread nobody
        has used is empty, never a 404 — a conversation that has not happened
        yet is not an error."""
        return widget.history(thread)

    @app.delete('/api/widget/history')
    def widget_forget(thread: str = ''):
        """New Chat. Ends the conversation named and no other — the reader's
        other experiments keep theirs. Answers with the emptied thread, so the
        page redraws from the lab rather than assuming what it now holds."""
        widget.forget(thread)
        return widget.history(thread)
