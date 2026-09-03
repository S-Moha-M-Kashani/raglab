"""The widget in the corner of every surface.

Two shared files a page gains the helper by loading, and the routes that carry
one conversation: the log the lab holds is the only copy of the transcript, so
a reader's log and the model's memory cannot drift apart.
"""
import json
import re

from raglab.evaluation import deterministic_metrics as metrics


def test_the_widget_is_two_shared_files_every_surface_can_load(client):
    # this is an integration test
    """The widget is a helper any surface gains by loading it, not a feature of
    one page. Its rules and its script are served from the root like tokens.css
    and lab.js, so there is one definition rather than three copies."""
    css = client.get('/widget.css')
    js = client.get('/widget.js')
    assert css.status_code == 200
    assert css.headers['content-type'].startswith('text/css')
    assert js.status_code == 200
    assert js.headers['content-type'].startswith('application/javascript')
    assert '.widget-launch' in css.text
    assert 'widgetSay' in js.text
    assert '.widget-launch' not in client.get('/panel.css').text, (
        'the widget rules must live in one sheet, not two')


def test_every_surface_carries_the_widget(panel_texts):
    # this is a convention test
    """One helper, three surfaces. A reader who can ask a question on the
    Laboratory and not on the board is reading two different labs.

    Built over `served_lab.app`, not the shared `client` fixture — that
    fixture is `panel_server.create_app()` alone, and the Inspector is only
    mounted where the three surfaces actually come together, the way
    `test_inspector.py`'s own cross-surface checks already do."""
    from fastapi.testclient import TestClient
    from raglab.dashboard import served_lab

    inspector = TestClient(served_lab.app).get('/inspector/').text
    for name, page in (('index.html', panel_texts['index.html']),
                       ('leaderboard.html', panel_texts['leaderboard.html']),
                       ('inspector.html', inspector)):
        assert 'src="/widget.js"' in page, f'{name} does not load the widget'
        assert 'href="/widget.css"' in page, f'{name} does not style the widget'


def test_the_widget_sends_the_thread_it_is_in(panel_texts):
    # this is a convention test
    """The widget's memory is a thread in widget.db, not a page's lifetime. The
    POST must carry which thread, or every question lands in the same one."""
    script = panel_texts['widget.js']
    assert re.search(r"widgetStream\('/api/widget/stream',\s*\{[^}]*\bthread\b",
                     script), ('the widget POST must carry the thread id')
    assert 'crypto.randomUUID' not in script, (
        'a per-page id is exactly the reset this change removed')


def test_the_widget_types_the_answer_out_as_it_arrives(panel_texts):
    # this is a convention test
    """The reply used to land in one piece one round trip after Send. It comes
    from `/api/widget/stream` now, read as it arrives — and the pieces are only
    how it arrived: the final event carries the reply the lab's own log holds,
    and the bubble adopts that, so the screen and the transcript cannot differ.
    A stream that stops part-way leaves what came marked as stopped rather than
    dressed up as a whole answer."""
    script, style = panel_texts['widget.js'], panel_texts['widget.css']
    assert "'/api/widget/stream'" in script, (
        'the widget must ask the streaming route')
    assert 'getReader()' in script, (
        "the answer must be read as it arrives, not awaited as one body")
    assert 'widgetFinish(live, data.reply)' in script, (
        "the reply the lab holds must replace the pieces the page typed")
    assert 'widgetStopped(live)' in script, (
        'a stream that failed must not leave a fragment looking finished')
    assert '.widget-msg.bot.streaming::after' in style, (
        'an answer still being written must say so on screen')
    assert '.widget-msg.bot.stopped' in style, (
        'and one that stopped part-way must say that instead')


def test_the_widget_shows_the_token_account_under_a_reply(panel_texts):
    # this is a convention test
    """The account travels with the reply and the page shows it — a faint
    meta line, only when the backend reported one: an unreported account
    renders nothing rather than a made-up zero."""
    script = panel_texts['widget.js']
    assert 'input_tokens' in script, (
        'the widget must read the served token account')
    assert 'output_tokens' in script, (
        'both directions of the account, not just one')


def test_the_widget_serves_the_conversation_it_holds(client, monkeypatch):
    # this is an integration test
    """A refresh redraws the log from the lab, not from a copy in the browser:
    what a reader sees is exactly what the model remembers, so the two cannot
    drift apart. And what it serves about a thread — which experiment it is
    about, when it began — must be what a turn actually wrote, not two empty
    strings dressed as facts.

    A question is put through the real route, the real `ask`, the real graph
    and the real checkpointer; only the model is a fake, because the suite is
    offline and no test here may reach OpenRouter. That is what makes this an
    honest reading of what a reader would see: nothing between the POST and
    the GET is stubbed."""
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain.agents import create_agent

    from raglab.agents import widget
    from raglab.agents.widget import conversation_memory as memory
    from raglab.agents.widget.hooks import MIDDLEWARE

    memory.forget('exp-route')
    # A thread nobody has used says so with three empty answers rather than
    # with an error — the empty log and its starters are the honest rendering
    # of a conversation that has not happened yet.
    read = client.get('/api/widget/history', params={'thread': 'exp-route'})
    assert read.status_code == 200
    assert read.json() == {'thread': 'exp-route', 'experiment_id': '',
                           'started_at': '', 'turns': []}

    def fake_agent(model):
        # `_build_agent`'s own shape with the one part that needs a key and a
        # network swapped out: same state schema, same middleware, same
        # process-wide checkpointer, so what lands in widget.db is written by
        # the graph exactly as it would be in production. No tools, because a
        # fake chat model cannot bind them and a scripted reply calls none.
        return create_agent(GenericFakeChatModel(messages=iter([
                                AIMessage(content='four judged metrics decide')])),
                            system_prompt='x', middleware=MIDDLEWARE,
                            state_schema=memory.WidgetState,
                            checkpointer=memory.saver())

    widget.reset()
    monkeypatch.setattr(widget.backends, '_build_agent', fake_agent)
    try:
        said = client.post('/api/widget', json={'message': 'which metrics decide?',
                                                'model': 'openai/gpt-5-nano',
                                                'thread': 'exp-route'})
        assert said.status_code == 200, said.text
    finally:
        widget.reset()

    read = client.get('/api/widget/history', params={'thread': 'exp-route'})
    body = read.json()
    assert body['turns'] == [
        {'role': 'you', 'text': 'which metrics decide?'},
        {'role': 'bot', 'text': 'four judged metrics decide'}]
    # The two fields the route reports beside the turns. They were declared on
    # `WidgetState`, written by nothing, and pinned here as empty strings — a
    # route stating as fact about every thread the one thing it did not know.
    assert body['experiment_id'] == 'exp-route'
    assert body['started_at']


def test_the_route_serves_a_turns_token_account_when_one_was_reported(client):
    # this is an integration test
    """The route is a thin pass-through over `conversation_memory.history`, so
    what it proves here is that nothing between the checkpointer and the JSON
    response strips the account back off — the same seeding helper the unit
    tests use, read back through the actual FastAPI route rather than the
    Python function directly."""
    from langchain_core.messages import AIMessage, HumanMessage
    from raglab.agents.widget.tests.widget_examples import write_messages

    write_messages('exp-billed-route', [
        HumanMessage(content='what did that cost?'),
        AIMessage(content='1692 total', usage_metadata={
            'input_tokens': 1630, 'output_tokens': 62, 'total_tokens': 1692})])

    read = client.get('/api/widget/history', params={'thread': 'exp-billed-route'})
    assert read.status_code == 200
    assert read.json()['turns'] == [
        {'role': 'you', 'text': 'what did that cost?'},
        {'role': 'bot', 'text': '1692 total',
         'input_tokens': 1630, 'output_tokens': 62}]


def test_new_chat_empties_one_conversation_and_no_other(client):
    # this is an integration test
    """The only control that ends a conversation, and it ends exactly one."""
    from langchain_core.messages import AIMessage, HumanMessage
    from raglab.agents.widget.tests.widget_examples import write_messages

    # Seeded through the real saver, the same helper
    # `agents/widget/tests/test_conversation_memory.py` seeds its own threads
    # with — one definition of what a checkpoint has to look like, not two
    # that a langgraph upgrade could silently pull apart.
    for thread in ('exp-a', 'exp-b'):
        write_messages(thread, [HumanMessage(content='q'), AIMessage(content='a')])

    gone = client.delete('/api/widget/history', params={'thread': 'exp-a'})
    assert gone.status_code == 200
    assert gone.json()['turns'] == []
    kept = client.get('/api/widget/history', params={'thread': 'exp-b'})
    assert kept.json()['turns'] == [{'role': 'you', 'text': 'q'},
                                    {'role': 'bot', 'text': 'a'}]
