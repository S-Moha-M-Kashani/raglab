"""The developer's step-by-step checkout of one widget thread.

A page that exists only while `RAGLAB_DEV_KEY` is set, asks for the key on
itself, and must never print it back. What it shows is the thread as the widget
holds it — including the calls a turn made and the ones a budget dropped.
"""
from html import unescape as html_unescape

from raglab.configuration import lab_config as config


def test_the_dev_trace_page_opens_only_with_the_key_and_shows_the_tool_calls(client, monkeypatch):
    # this is an integration test
    """A developer page, not a reader's: every step the model took on a thread
    — system lines, tool calls with their arguments, tool results, replies.
    It answers to one key from the environment and to nothing else: no key
    configured and the page is a 404 rather than a 403, because a page that
    says "forbidden" has already said it exists. With a key configured, the
    page asks for it in a masked field — the key travels in a POST body, never
    in the address bar, the history or a link — and unlocks the browser with
    a session cookie the key is not written into."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from raglab.agents.widget.tests.widget_examples import write_messages
    write_messages('exp-dev', [
        HumanMessage(content='what ran?'),
        AIMessage(content='', tool_calls=[{'name': 'read_experiment',
                                           'args': {'experiment_id': 'exp-dev'},
                                           'id': 'c1'}]),
        ToolMessage(content='experiment exp-dev — baseline', tool_call_id='c1',
                    name='read_experiment'),
        AIMessage(content='the baseline.')])

    monkeypatch.delenv('RAGLAB_DEV_KEY', raising=False)
    assert client.get('/dev/trace').status_code == 404
    assert client.post('/dev/trace', data={'key': 'x'}).status_code == 404
    monkeypatch.setenv('RAGLAB_DEV_KEY', 'open-sesame')

    # Locked: the plate, a masked field, and not one thread name.
    plate = client.get('/dev/trace')
    assert plate.status_code == 200
    assert 'type="password"' in plate.text and 'name="key"' in plate.text
    assert 'exp-dev' not in plate.text
    # The key is never read from the address bar.
    assert 'exp-dev' not in client.get('/dev/trace', params={'key': 'open-sesame'}).text
    wrong = client.post('/dev/trace', data={'key': 'wrong'})
    assert wrong.status_code == 200
    assert 'did not match' in wrong.text and 'exp-dev' not in wrong.text
    assert 'set-cookie' not in wrong.headers

    unlocked = client.post('/dev/trace', data={'key': 'open-sesame'},
                           follow_redirects=False)
    assert unlocked.status_code == 303
    assert unlocked.headers['location'] == '/dev/trace'
    cookie = unlocked.headers['set-cookie']
    assert 'raglab_dev=' in cookie and 'HttpOnly' in cookie and 'SameSite=strict' in cookie.replace('Strict', 'strict')
    assert 'open-sesame' not in cookie

    index = client.get('/dev/trace')
    assert index.status_code == 200
    assert 'exp-dev' in index.text
    assert '1 question' in index.text and '4 step' in index.text
    assert 'what ran?' in index.text
    # Theme is stamped before paint, exactly as the three surfaces do it.
    head = index.text.split('</head>')[0]
    assert 'raglab-theme' in head and 'documentElement' in head
    page = client.get('/dev/trace', params={'thread': 'exp-dev'})
    assert page.status_code == 200
    # A checkout window must show the thread as it is now, not as the browser
    # last saw it: a reader who sent three more questions and pressed back saw
    # the two it had cached.
    assert page.headers['cache-control'] == 'no-store'
    for expected in ('what ran?', 'read_experiment', '&quot;experiment_id&quot;',
                     'experiment exp-dev — baseline', 'the baseline.'):
        assert expected in page.text
    # The page says what the model is standing on, not only what was said: the
    # standing system prompt (from the fixture, verbatim), and the window — how
    # many of the thread's messages the next call will actually see.
    from raglab.agents import widget
    assert widget.SYSTEM_PROMPT[:60] in html_unescape(page.text)
    assert f'last {widget.MAX_HISTORY}' in page.text
    # And it wears no step ink. The widget is a helper rather than a pipeline
    # stage, so the rule that keeps index orange, retrieval green and
    # generation blue off widget.css holds for its checkout window too — the
    # focus ring took generation blue until 2026-08-29 and now takes the ink
    # the focused element already carries, which contrasts on both themes
    # without naming a stage.
    assert '--step-' not in page.text, (
        'the dev trace is the widget\'s window, and the widget wears no '
        'step ink')
    assert 'outline:2px solid currentColor' in page.text

    # Lock: the cookie is forgotten server-side, so even a browser that kept
    # it is back at the plate.
    locked = client.post('/dev/trace/lock', follow_redirects=False)
    assert locked.status_code == 303
    assert 'exp-dev' not in client.get('/dev/trace').text


def test_the_dev_trace_dims_a_standing_line_a_newer_one_superseded(client, monkeypatch):
    # this is an integration test
    """The page's one claim is what the model was handed, so a memory context a
    newer one supersedes has to read like a trimmed step — the thread keeps it,
    but no call carries it. Dimming it is what makes filtering the prompt
    better than deleting from the log: the developer still sees everything the
    conversation accumulated, and sees which of it the model actually got.

    A system line the widget did not write is dimmed by nothing: it carries no
    standing marker, so nothing can supersede it."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from raglab.agents import widget
    from raglab.agents.widget.tests.widget_examples import write_messages

    def standing(text, mark):
        return SystemMessage(content=text,
                             additional_kwargs={widget.STANDING_LINE: mark})

    write_messages('exp-standing', [
        standing('about exp-standing', widget.IDENTITY_LINE),
        standing('memory v1', widget.MEMORY_LINE),
        standing('memory v2', widget.MEMORY_LINE),
        standing('memory v3', widget.MEMORY_LINE),
        SystemMessage(content='SAFETY: never quote a key'),
        HumanMessage(content='what ran?'), AIMessage(content='the baseline.')])
    monkeypatch.setenv('RAGLAB_DEV_KEY', 'open-sesame')
    client.post('/dev/trace', data={'key': 'open-sesame'})

    text = client.get('/dev/trace', params={'thread': 'exp-standing'}).text

    # The log shows every version — this page is the record, not the prompt.
    for line in ('memory v1', 'memory v2', 'memory v3'):
        assert line in text
    # The two older ones read as trimmed; the newest, the identity line and the
    # line the widget did not write do not.
    def dimmed(line):
        return f'<div class="system trimmed"><div class="card">' \
               f'<div class="kind">system</div><pre>{line}</pre>' in text
    assert dimmed('memory v1') and dimmed('memory v2')
    assert not dimmed('memory v3')
    assert not dimmed('about exp-standing')
    assert not dimmed('SAFETY: never quote a key')
    # And the standing panel says so in words, with the count.
    assert '2 superseded standing line(s)' in text


def test_the_dev_trace_says_a_closed_turns_tool_reply_travelled_as_a_stub(
        client, monkeypatch):
    # this is an integration test
    """The page's one claim is what the model was handed, so a tool reply the
    next call sends in reduced form must not read like one the model got whole.

    Dimming would be the wrong signal — a stub *was* sent — so the card says
    "reduced", shows the stub in the words the model actually received, and
    keeps the full body underneath, because the log is still the record. A
    developer asking "why did it say that?" would otherwise read twenty
    thousand characters the model never saw on that call and conclude the
    wrong thing."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from raglab.agents import widget
    from raglab.agents.widget.tests.widget_examples import write_messages

    body = 'a very long skill body. ' * 800
    write_messages('exp-stub', [
        HumanMessage(content='what about chunking?'),
        AIMessage(content='', tool_calls=[
            {'name': 'read_rag_skill', 'args': {'names': 'chunking-strategies'},
             'id': 'c1'}]),
        ToolMessage(content=body, tool_call_id='c1', name='read_rag_skill'),
        AIMessage(content='chunk by heading.')])
    monkeypatch.setenv('RAGLAB_DEV_KEY', 'open-sesame')
    client.post('/dev/trace', data={'key': 'open-sesame'})

    text = client.get('/dev/trace', params={'thread': 'exp-stub'}).text

    # The log holds the body, so the page shows it.
    assert body in html_unescape(text)
    # And says, in the model's own words, what went in its place.
    assert '· reduced' in text and 'sent as a stub' in text
    assert '[read_rag_skill(names=chunking-strategies)' in html_unescape(text)
    # Reduced is not trimmed: the reply was sent, so it is not dimmed.
    assert 'tool trimmed' not in text
    # The standing panel counts it and says the fence in words.
    assert '1 tool repl(y/ies)' in text
    assert 'the turn it is answering always carries its tool replies whole' in text
    assert str(widget.MAX_HISTORY_CHARS) in text


def test_the_dev_trace_dims_a_turn_the_character_budget_drops(client, monkeypatch):
    # this is an integration test
    """The page counted messages and never characters, so a thread holding one
    enormous answer rendered every step undimmed and drew no divider at all —
    while `hooks._within_budget` drops that whole turn from the next call. A
    developer asking "why did it say that?" read 25 KB as context the model had
    and it had never received it.

    This page's entire job is to say what the model was actually handed, so it
    applies the very rule the hook applies (`widget.history_budget_cut`) rather
    than a second imitation of it — measured on the stubs, dropped a whole turn
    at a time, and reported in the tense of the next call, which is why a
    finished last turn is history here rather than the turn being answered."""
    from langchain_core.messages import AIMessage, HumanMessage
    from raglab.agents import widget
    from raglab.agents.widget.tests.widget_examples import write_messages

    big = 'a very long answer. ' * 1_250          # 25,000 characters
    assert len(big) > widget.MAX_HISTORY_CHARS
    write_messages('exp-budget', [
        HumanMessage(content='the long one?'), AIMessage(content=big),
        HumanMessage(content='and the short one?'),
        AIMessage(content='short.')])
    monkeypatch.setenv('RAGLAB_DEV_KEY', 'open-sesame')
    client.post('/dev/trace', data={'key': 'open-sesame'})

    text = client.get('/dev/trace', params={'thread': 'exp-budget'}).text

    # The whole first turn is dimmed — the question with its answer, never the
    # answer alone, because the budget drops turns rather than messages.
    assert '<div class="human trimmed">' in text
    assert '<div class="ai trimmed">' in text
    # And the divider lands where the next call starts reading.
    assert 'from here on, sent to the model' in text
    kept = text.split('from here on, sent to the model')[1]
    assert 'and the short one?' in kept and 'short.' in kept
    assert big not in html_unescape(kept)
    # The standing panel counts both dropped steps, not zero.
    assert '2 oldest non-system step(s) are no longer sent' in text


def test_the_dev_trace_tells_a_turn_in_flight_from_one_whose_run_died(
        client, monkeypatch):
    # this is an integration test
    """The page reports in the tense of the next call, and which call that is
    is a fact about the world rather than a shape in the log.

    Three threads, and the middle one is why this test was rewritten. A thread
    whose last turn is **closed** has been answered, so its next call is a new
    question and that finished turn is history — budgeted, and dropped if it is
    too big. A thread that stops **mid-turn** looks the same either way and
    means two different things: a run still going, whose next model call
    continues it, or a run that died, whose next model call is a new question
    with that turn's whole body cut. This page used to read every unfinished
    turn as the first, which was true of one thread in this file and false of
    every dead one — and a dead turn is the state the branch exists for.

    What separates them is the row `_log_interrupted_turn` writes under the
    question's own id: a run that died owes one, a run still going does not.
    So the third thread here differs from the second by nothing but that row,
    and the page must say opposite things about the two."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from raglab.agents import widget
    from raglab.agents.widget import turn_logger
    from raglab.agents.widget.tests.widget_examples import write_messages
    monkeypatch.setenv('RAGLAB_DEV_KEY', 'open-sesame')
    client.post('/dev/trace', data={'key': 'open-sesame'})

    # One closed turn, too big to ride along. The next call is a new question,
    # so the whole turn is history and the whole turn goes.
    big = 'a very long answer. ' * 1_250          # 25,000 characters
    assert len(big) > widget.MAX_HISTORY_CHARS
    write_messages('exp-closed-last', [HumanMessage(content='the long one?'),
                                       AIMessage(content=big)])
    closed = client.get('/dev/trace', params={'thread': 'exp-closed-last'}).text
    assert '<div class="human trimmed">' in closed
    assert '<div class="ai trimmed">' in closed
    assert '2 oldest non-system step(s) are no longer sent' in closed

    # A turn still in flight — the state Task 5 is about, and the state every
    # tool hop passes through. The reply at the end is what the model is about
    # to read, so nothing here is dropped, dimmed or reduced.
    body = 'a very long skill body. ' * 900       # 21,600 characters
    assert len(body) > widget.MAX_HISTORY_CHARS
    write_messages('exp-inflight', [
        HumanMessage(content='q0'), AIMessage(content='a0'),
        HumanMessage(content='the long one?'),
        AIMessage(content='', tool_calls=[
            {'name': 'read_rag_skill', 'args': {'names': 'chunking'},
             'id': 'c1'}]),
        ToolMessage(content=body, tool_call_id='c1', name='read_rag_skill')])
    flight = client.get('/dev/trace', params={'thread': 'exp-inflight'}).text
    assert 'nothing has been trimmed yet' in flight
    assert 'trimmed"' not in flight        # the class, never the stylesheet
    assert 'from here on, sent to the model' not in flight
    assert 'sent as a stub' not in flight and '· reduced' not in flight
    # And the body really is there to be read, whole.
    assert body in html_unescape(flight)
    # The page says which of the two states this is, rather than leaving a
    # developer to infer it from what is not dimmed.
    assert 'no row says that run died, so it is still going' in flight
    assert widget.next_call_continues('exp-inflight')

    # The same shape, and a row saying the run behind it died. Nothing about
    # the messages changed; what changed is that the next call is now a new
    # question, so the turn is interrupted and its work is cut.
    write_messages('exp-died', [
        HumanMessage(content='q0', id='d0'), AIMessage(content='a0', id='d1'),
        HumanMessage(content='the long one?', id='d2'),
        AIMessage(content='', id='d3', tool_calls=[
            {'name': 'read_rag_skill', 'args': {'names': 'chunking'},
             'id': 'c1'}]),
        ToolMessage(content=body, tool_call_id='c1', id='d4',
                    name='read_rag_skill')])
    turn_logger.log_turn(thread_id='exp-died', experiment_id='exp-died',
                         dataset_id='diary-en', user_message_id='d2',
                         user_message='the long one?', status='interrupted',
                         status_reason='the model connection dropped mid-turn')
    died = client.get('/dev/trace', params={'thread': 'exp-died'}).text
    assert not widget.next_call_continues('exp-died')
    assert 'no row says that run died' not in died
    # The call and the 21 KB reply are cut; the question that opened the turn
    # keeps its place and travels with one line saying nothing answered it.
    assert '2 step(s) belong to a turn whose run died before it answered' in died
    assert died.count('not sent · this turn was interrupted') == 2
    assert widget.interrupted_note(2) in html_unescape(died)
    # The record still holds every character of it — only the prompt is shaped.
    assert body in html_unescape(died)


def test_the_dev_trace_marks_the_work_of_a_turn_that_never_answered(
        client, monkeypatch):
    # this is an integration test
    """A turn whose run died after a tool call, seen from the page.

    `hooks._close_interrupted` leaves that turn's unfinished work out of every
    later call and sends the question with one line saying nothing answered it.
    The page's single claim is what the model was handed, so it has to say the
    same: the call and the tool reply are dimmed and named as interrupted, the
    question is not, and the tool's body is still on the page — the record is
    what makes this a record, and nothing was deleted from it.

    The divider stays out of it on purpose. It answers "where does the next
    call start reading?", and an interrupted turn is dimmed in the middle of
    the tape rather than at its front; drawing the divider in front of one
    would promise that everything after it was sent."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from raglab.agents import widget
    from raglab.agents.widget.tests.widget_examples import write_messages
    monkeypatch.setenv('RAGLAB_DEV_KEY', 'open-sesame')
    client.post('/dev/trace', data={'key': 'open-sesame'})

    body = 'the recorded row, at length. ' * 20
    write_messages('exp-interrupted', [
        HumanMessage(content='what did that run score?'),
        AIMessage(content='', tool_calls=[
            {'name': 'read_experiment', 'args': {'experiment_id': 'e1'},
             'id': 'c1'}]),
        ToolMessage(content=body, tool_call_id='c1', name='read_experiment'),
        HumanMessage(content='what did that run score?'),
        AIMessage(content='It scored 0.71.')])

    text = client.get('/dev/trace', params={'thread': 'exp-interrupted'}).text

    # The unfinished half is dimmed and says which of the two reasons it is.
    assert '<div class="ai trimmed abandoned">' in text
    assert '<div class="tool tool trimmed abandoned">' in text
    assert text.count('not sent · this turn was interrupted') == 2
    # The question that opened it is not dimmed: it still travels, because a
    # follow-up needs to know what was asked — and the line that travels after
    # it is printed verbatim, the way a stub is. Dimming says what went; only
    # the words say what arrived.
    assert '<div class="human trimmed' not in text
    assert 'sent after this question' in text
    assert widget.interrupted_note(2) in html_unescape(text)
    # The standing panel counts them apart from a turn the window dropped.
    assert '2 step(s) belong to a turn whose run died before it answered' in text
    assert 'nothing has been trimmed yet' in text
    # And the body is still here to read — the log keeps everything.
    assert body in html_unescape(text)
    assert 'from here on, sent to the model' not in text


def test_the_dev_trace_agrees_with_the_real_next_call_on_random_threads():
    # this is an integration test
    """Four hundred random threads, and on each one the page's claim checked
    against the call the widget really makes next.

    Three times now this page has been found saying something the next call
    contradicted — a turn the character budget drops, a tool reply that travels
    as a stub, an interrupted turn read as one still in flight — and each was
    found by hand, on the one shape somebody thought to write down. The rules
    themselves are `conversation_memory`'s and `hooks`'s, shared on purpose, so
    what can still go wrong is not the rules but how this page composes them:
    which projection it takes the window over, and what it appends before it
    asks. That is a whole-shape property, and the way to test one is to
    generate shapes.

    Each thread is seeded with random turns — questions of 5, 200 or 3,000
    characters, some with a `read_rag_skill` hop whose reply is 40, 4,000 or
    21,000 characters, some left unanswered, standing memory lines dropped in
    between — and then the reader asks the next question through
    `backends._run` and the real compiled agent, so what the model was handed
    is recorded rather than imagined. A thread that ends mid-turn is given the
    `widget_turn_log` row that a run dying there really writes, which is the
    fact `widget.next_call_continues` reads.

    Three claims are compared per thread, and all three have been wrong at some
    point on this branch: which steps the call carries at all, which of them it
    carries whole rather than as a stub, and the lines an interrupted turn is
    replaced by. Under the premise this test replaced — an unclosed last turn
    read as the turn being answered — the disagreement is exactly the threads
    that end mid-turn and no others.
    """
    import random

    from langchain_core.messages import (AIMessage, HumanMessage,
                                         SystemMessage, ToolMessage)

    from raglab.agents import widget
    from raglab.agents.widget import backends, turn_logger
    from raglab.agents.widget import conversation_memory as memory
    from raglab.agents.widget.tests.prompt_payload_probe import (
        _RecordingModel, build_agent)
    from raglab.agents.widget.tests.widget_examples import write_messages
    from raglab.dashboard import dev_trace_page as page

    def standing(index):
        return SystemMessage(content=f'Dataset memory: note {index}',
                             id=f'm{index}',
                             additional_kwargs={
                                 memory.STANDING_LINE: memory.MEMORY_LINE})

    def a_thread(rng):
        """One random conversation, every message carrying the id its position
        gives it — the id is how a message handed to the model is matched back
        to the step the page made a claim about."""
        messages = []
        for _ in range(rng.randint(0, 2)):
            messages.append(standing(len(messages)))
        for turn in range(rng.randint(1, 7)):
            messages.append(HumanMessage(
                content=f'q{turn} ' + 'x' * rng.choice([5, 200, 3_000]),
                id=f'm{len(messages)}'))
            if rng.random() < 0.45:
                call = f'c{turn}'
                messages.append(AIMessage(
                    content='', id=f'm{len(messages)}',
                    tool_calls=[{'name': 'read_rag_skill',
                                 'args': {'names': 'chunking'}, 'id': call}]))
                messages.append(ToolMessage(
                    content=f'body{turn} ' + 'b' * rng.choice([40, 4_000,
                                                              21_000]),
                    tool_call_id=call, name='read_rag_skill',
                    id=f'm{len(messages)}'))
            if rng.random() < 0.75:
                messages.append(AIMessage(
                    content=f'a{turn} ' + 'y' * rng.choice([5, 300]),
                    id=f'm{len(messages)}'))
            if rng.random() < 0.15:
                messages.append(standing(len(messages)))
        return messages

    rng = random.Random(20260829)
    mid_turn = 0
    for shape in range(400):
        thread = f'trace-agreement-{shape}'
        messages = a_thread(rng)
        write_messages(thread, messages)
        # A thread that stops mid-turn stops for one of two reasons, and only
        # the dead one is followed by another question — so that is the one a
        # comparison against the next call can be made on, and the row is what
        # `_log_interrupted_turn` would have written for it.
        spoken = [m for m in messages
                  if memory.turn_shape(m) != memory.TURN_SYSTEM]
        last = memory.conversation_turns(
            [memory.turn_shape(m) for m in spoken])[-1]
        asked = spoken[last.start]
        if not last.closed and memory.turn_shape(asked) == memory.TURN_HUMAN:
            mid_turn += 1
            turn_logger.log_turn(
                thread_id=thread, experiment_id=thread, dataset_id='diary-en',
                user_message_id=asked.id, user_message=str(asked.content),
                status='interrupted',
                status_reason='the model connection dropped mid-turn')

        # What the page says, through the very calls `thread()` makes.
        steps = widget.trace(thread)['steps']
        continues = widget.next_call_continues(thread)
        dropped = {id(s) for s in page._dropped_from_the_window(steps,
                                                               continues)}
        stubbed = set(page._stubs(steps))
        notes = page._interrupted_notes(steps, continues)
        said_notes = [notes[id(s)] for s in steps if id(s) in notes]
        said_sent = {f'm{i}' for i, s in enumerate(steps)
                     if s['kind'] != 'system' and id(s) not in dropped}
        said_whole = {name for name in said_sent
                      if id(steps[int(name[1:])]) not in stubbed}

        # What the next call really carries.
        model = _RecordingModel(script=[AIMessage(content='ok')])
        agent = build_agent(model)
        payload, config = backends._run('and now?', thread, dataset='diary-en')
        agent.invoke(payload, config=config)
        handed = model.seen[0]
        held = {m.id: memory._text(m.content) for m in messages}
        sent = {m.id for m in handed if m.id in held
                and getattr(m, 'type', '') != 'system'}
        whole = {m.id for m in handed if m.id in sent
                 and memory._text(m.content) == held[m.id]}
        real_notes = [memory._text(m.content) for m in handed
                      if memory._text(m.content).startswith('[This question')]

        assert said_sent == sent, f'{thread}: the page named the wrong steps'
        assert said_whole == whole, f'{thread}: the page stubbed the wrong step'
        assert said_notes == real_notes, f'{thread}: wrong interrupted note(s)'

    # The generator really did produce both endings, or this proves one case
    # four hundred times.
    assert 50 < mid_turn < 350


def test_the_dev_key_never_appears_in_any_trace_response(client, monkeypatch):
    # this is an integration test
    """A key holding `#`, `%` and `&` — legal in .env, special in a URL — must
    unlock like any other, and must appear in no address, header or page the
    route sends back: the whole point of typing it into the page is that the
    browser's history and the server's access log never see it."""
    from langchain_core.messages import AIMessage, HumanMessage
    from raglab.agents.widget.tests.widget_examples import write_messages
    write_messages('exp-esc', [HumanMessage(content='hi'), AIMessage(content='hello')])
    key = 'ab#cd%ef&g'
    monkeypatch.setenv('RAGLAB_DEV_KEY', key)

    seen = [client.get('/dev/trace', params={'thread': 'exp-esc'}),
            client.post('/dev/trace', data={'key': key, 'next': 'exp-esc'},
                        follow_redirects=False)]
    assert seen[1].status_code == 303
    assert seen[1].headers['location'] == '/dev/trace?thread=exp-esc'
    seen.append(client.get(seen[1].headers['location']))
    seen.append(client.get('/dev/trace'))
    assert 'exp-esc' in seen[-1].text
    for response in seen:
        assert key not in response.text
        assert key not in str(response.headers)
