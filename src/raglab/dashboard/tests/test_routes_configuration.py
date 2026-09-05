"""What `/api/options` serves the panel to render itself from.

Everything here is a claim about the served response rather than the page: the
help text a knob carries, and the backends a lab with none installed names as
the way to fix that.
"""
from raglab.configuration import explainer_assembly as explain
from raglab.configuration import lab_config as config


# Topics whose brief is written rather than taken from the note's opening
# sentence, read from the one place that declares them — so this file holds no
# second copy of the list and a topic added there needs no edit here.
_DECLARED_BRIEFS = set(explain.BRIEF)


# --- the one dynamic, data-driven guard -------------------------------------

def test_the_panels_no_backend_hint_names_every_backend_that_would_fix_it(client):
    # this is a convention test
    """A hint that lists some of the ways out is worse than one that lists
    none, because a reader takes it for the whole set — so this fails the
    day a backend is added and the sentence is not. Built from the live
    provider list rather than a fixed row, since the row's own content is
    the thing under test."""
    page = client.get('/panel.js').text
    hint = [line for line in page.splitlines() if 'no LLM backend' in line]
    assert hint, 'the panel must say what to do when no backend is reachable'
    for provider in config.LLM_PROVIDERS:
        # 'fake' is not a way out: it answers without failing, which is the
        # problem rather than the fix.
        if provider and provider != 'fake':
            assert provider in hint[0], provider


def test_every_explainer_is_read_in_two_lengths(panel_texts, client):
    # this is a convention test
    """The `!` beside every knob is gone, and what replaced it is two lengths of
    one text: hover a knob's own name for a sentence, click it for the whole
    note. Four claims, because each one on its own is satisfiable while the
    feature is broken.

    The service serves both lengths under the same keys, from one text
    (`explain.briefs()` takes the opening sentence of `explain.topics()`), so a
    brief cannot drift from the note it opens.

    The trigger is the knob's own name, underlined — no mark. A checkbox is the
    exception and keeps one, because its words are already a click target that
    toggles it and they cannot also open a sentence.

    The hover lives in `lab.js`, once, for all three surfaces; and it obeys the
    two rules a hover reveal has to obey to be usable at all — a delay for a
    pointer crossing the page, none for a reader who tabbed here on purpose."""
    served = client.get('/api/options').json()
    assert set(served['brief']) == set(served['help']), (
        'both lengths, same keys, on the one payload the panel boots from — a '
        'topic with a note and no brief has nothing to show on hover')
    for topic, brief in served['brief'].items():
        assert served['help'][topic].startswith(brief) or topic in _DECLARED_BRIEFS, (
            f'{topic}: a brief that is not the note\'s own opening sentence '
            'has to be declared in explain.BRIEF, or it is a second copy '
            'nobody will keep in step')
    # The Inspector's half of this claim — its own route, its own marks — is in
    # test_inspector.py, where an Inspector client exists.

    js = panel_texts['panel.js']
    assert '>!</button>' not in js, (
        'the exclamation mark is retired on every trigger this page builds'
    )
    assert 'class="why-term"' in js and 'markTerm(label, found.topic)' in js, (
        "a knob's own name is the trigger now")
    assert "control.type === 'checkbox'" in js and '>?</button>' in js, (
        'a checkbox keeps a mark, because underlining words that already '
        'toggle a box would give one phrase two jobs')
    assert 'LabHelp.brief = (topic)' in js and 'LabHelp.full = (topic)' in js, (
        'the page resolves both lengths for the shared hover engine')

    lab = panel_texts['lab.js']
    assert 'HELP_HOVER_MS' in lab and 'showHelpBrief(trigger)' in lab
    assert "document.addEventListener('focusin'" in lab, (
        'a hover-only reveal is a reveal half the readers never get')
    focus = lab[lab.index("document.addEventListener('focusin'"):]
    focus = focus[:focus.index('});')]
    assert 'setTimeout' not in focus, (
        'no delay on focus: a reader who tabbed to a trigger has already asked')
    assert "trigger.setAttribute('aria-describedby', 'help-brief')" in lab, (
        'the sentence has to reach a screen reader, not only an eye')
    assert "box.dataset.more = String(helpHasMore(trigger, text))" in lab, (
        'the box offers "more" only when there is more — a brief that is '
        'already the whole note must not promise a second half')


def test_the_panels_split_plan_crosses_as_json_and_lands_normalized(client):
    # this is an integration test
    """The one knob the panel carries as a JSON list of objects.

    Both ends of that crossing, because each is satisfiable while the other is
    broken. Outward, the served defaults hand the panel the plan in its whole
    stored shape — a stage with a default left unsaid is a control that boots
    showing the wrong thing. Inward, the panel's lists become the tuple
    `fingerprint()` hashes, with every default filled, so the payload cannot
    depend on how the plan was carried here.
    """
    served = client.get('/api/options').json()['defaults']['index']
    assert served['split_plan'] == [
        {'kind': 'document'}, {'kind': 'drift', 'markers': [], 'when': 'always'}]

    cfg = config.LabConfig.from_dict({'index': {'split_plan': [
        {'kind': 'document'}, {'kind': 'separator', 'atoms': [{'text': '\n\n'}]}]}})
    assert cfg.index.split_plan == (
        {'kind': 'document'},
        {'kind': 'separator', 'when': 'over-budget', 'atoms': ({'text': '\n\n'},),
         'join': 'or'})
