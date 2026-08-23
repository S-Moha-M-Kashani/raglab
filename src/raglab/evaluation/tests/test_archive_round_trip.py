"""Export and import carry the whole experiment, at every rung of the ladder.

Two tests, one per direction, each run over every archive in
`archive_examples.ladder()`: the knobs alone, then plus what was indexed, then
plus what was retrieved, then plus what was generated — and beside that last
one, the same judged experiment with no trace kept. The ladder exists because
the format has several shapes and only the fullest one was ever exercised; a
codec that dropped the summaries, refused an archive with no questions in it,
or refused a run that was scored before traces were ever written down, would
have passed every test the repo had.

What makes these round trips mean anything is that the fixture's config has *no*
value left at its default (`SHIFTED_CONFIG`). A knob the codec silently drops
still reads back correct when the value it falls back to is the value it began
at, so a default-valued fixture cannot tell a carried knob from a reconstructed
one. The first assertion in each test is therefore the same: this fixture names
every field the dataclasses define, and moves every one of them.

The two directions are not symmetric in what they cover, and the tests say so
where it matters. `validate_archive` is the real server-side import
(`imported_archive_store`). The export a reader actually clicks, though, runs in
the browser — `ArchiveIO.completed` in `archive_io.js` — so `build_completed` is
the server-side twin of it, and the browser's own export is pinned over this
same ladder by `dashboard/tests/archive_ladder.test.js`.
"""
import copy
import json

import pytest

from raglab.configuration.lab_config import LabConfig
from raglab.evaluation import experiment_archive as archive
from raglab.evaluation.tests import archive_examples as examples


GROUPS = ('index', 'retrieval', 'generation')

LADDER = examples.ladder()
RUNG_IDS = [rung['name'] for rung in LADDER]


def _knobs(config: dict) -> dict:
    """A config flattened to `group.field` -> value, label included."""
    flat = {'label': config['label']}
    for group in GROUPS:
        for field, value in config[group].items():
            flat[f'{group}.{field}'] = value
    return flat


def _fixture_moves_every_knob():
    """The premise both tests rest on, checked before either one trusts it.

    Written against `LabConfig()` rather than a list kept here, so a knob added
    to a dataclass and forgotten in `SHIFTED_CONFIG` fails this rather than
    riding along untested.
    """
    defaults = _knobs(LabConfig().to_dict())
    shifted = _knobs(examples.SHIFTED_CONFIG)
    assert set(shifted) == set(defaults), (
        'the ladder fixture and the lab config name different knobs — '
        f'missing from the fixture: {sorted(set(defaults) - set(shifted))}; '
        f'unknown to the config: {sorted(set(shifted) - set(defaults))}')
    unmoved = sorted(knob for knob, value in shifted.items()
                     if value == defaults[knob])
    assert not unmoved, (
        'these knobs are still at their default, so a round trip that dropped '
        f'them would look correct: {unmoved}')


@pytest.mark.parametrize('rung', LADDER, ids=RUNG_IDS)
def test_exporting_writes_every_knob_and_every_stage_it_was_given(rung):
    # this is a unit test
    """Export invents nothing and omits nothing, at each rung.

    `build_completed` is handed only the result and the evidence the lab
    produced; everything in the archive it returns therefore has to have come
    from one of them. Rung 1 has neither, and is the one shape built by hand:
    an unrun experiment exports its knobs and must not grow an `evaluation`
    block describing readings that do not exist.
    """
    _fixture_moves_every_knob()
    settings = copy.deepcopy(rung['archive']['settings'])

    if rung['result'] is None:
        exported = archive.validate_archive({
            'format': archive.FORMAT, 'version': archive.VERSION,
            'settings': settings})
        assert 'evaluation' not in exported, (
            'a settings-only export must not fabricate an evaluation block')
    else:
        exported = archive.build_completed(
            settings, copy.deepcopy(rung['result']),
            copy.deepcopy(rung['evidence']))

    # Every knob, by name and by value, and the UI controls beside them.
    assert _knobs(exported['settings']['config']) == _knobs(examples.SHIFTED_CONFIG)
    assert exported['settings']['ui'] == examples.SHIFTED_UI

    # And every stage's evidence, in the amount this rung is supposed to carry.
    assert examples.contents(exported) == examples.CARRIED[rung['name']]

    # And then the whole archive, key for key: the assertions above name what
    # matters, this one catches whatever they forgot to look at.
    assert exported == rung['archive']


@pytest.mark.parametrize('rung', LADDER, ids=RUNG_IDS)
def test_importing_restores_every_knob_and_every_stage_it_was_sent(rung):
    # this is a unit test
    """Import is the same promise read backwards, at each rung.

    `validate_archive` is the server's trust boundary — the one
    `imported_archive_store` puts a ledger row behind — and it returns the
    archive unchanged, so "it imported" and "it imported everything" are the
    same claim only if something checks the second. The knobs get one check the
    export side cannot make: rebuilt through `LabConfig.from_dict`, which is how
    an imported experiment reaches the pipeline. A knob that survived as JSON
    but that the dataclasses drop on the way in would still be a lost knob.
    """
    _fixture_moves_every_knob()
    imported = archive.validate_archive(copy.deepcopy(rung['archive']))

    assert _knobs(imported['settings']['config']) == _knobs(examples.SHIFTED_CONFIG)
    assert imported['settings']['ui'] == examples.SHIFTED_UI

    # Compared as JSON, which is the only sense in which it can be: a knob
    # held as a tuple (`agentic_weights`) comes back from `normalized()` a
    # tuple, and it is the encoding, not Python, that the archive is written in.
    rebuilt = json.loads(json.dumps(
        LabConfig.from_dict(imported['settings']['config']).to_dict()))
    assert _knobs(rebuilt) == _knobs(examples.SHIFTED_CONFIG), (
        'the imported config does not survive the dataclasses that run it')

    assert examples.contents(imported) == examples.CARRIED[rung['name']]
    assert imported == rung['archive']

    if rung['result'] is not None:
        # The evidence, not merely counted but compared: the corpus a reader
        # would read, the chunks the Inspector would show, and the candidates
        # each trace ranked.
        inspector = imported['evaluation']['inspector']
        assert inspector['dataset']['corpus'] == examples.CORPUS
        assert inspector['dataset']['ground_truth'] == examples.GROUND_TRUTH
        assert inspector['chunks_by_session'] == examples.CHUNKS_BY_SESSION
        assert inspector['summaries'] == examples.SUMMARIES
        # Derived on the way in from the catalogue, never trusted as sent — so
        # this is the rung's metrics landing in the stages that own them.
        assert imported['evaluation']['stage_results'] == archive.stage_results(
            imported['evaluation']['result'],
            imported['evaluation']['metric_catalogue'])


def test_the_ladder_is_a_ladder():
    # this is a unit test
    """Each stacked rung carries everything the rung below it did, and more.

    Without this the fixtures could drift into unrelated archives, and "the
    retrieved rung covers the indexed one as well" would quietly stop being
    true while every other test here still passed.

    Only `SPINE` is walked. `scored-without-traces` is not a rung above
    `generated` but the same archive with its traces removed, so it belongs to
    the test below rather than to this one — folding it in here would either
    fail honestly or force the monotonic claim to be watered down for every
    rung to accommodate one that was never meant to make it.
    """
    counted = [examples.contents(rung['archive']) for rung in LADDER
               if rung['name'] in examples.SPINE]
    assert [rung['name'] for rung in LADDER
            if rung['name'] in examples.SPINE] == list(examples.SPINE)
    for below, above in zip(counted, counted[1:]):
        assert above != below, 'a rung adds nothing to the one below it'
        for key, carried in below.items():
            # Bools count as 0 and 1 here, which is the comparison wanted: a
            # stage present below may not go missing above.
            assert above[key] >= carried, f'{key} shrank up the ladder'


def test_the_scored_rung_is_the_generated_one_with_its_traces_gone():
    # this is a unit test
    """The branch rung differs from `generated` in the recording and nothing else.

    This is the shape 166 recorded evaluations are in: rows, judged metrics and
    a selection, no trace ever written. The archive format used to refuse it,
    because it required a trace per row, and refusing it would have thrown away
    real scores to protect evidence that was never collected. So the two rungs
    are compared field by field: everything the measurement consists of has to
    be identical, and only the traces and the candidates under them may differ.
    """
    rungs = {rung['name']: rung for rung in LADDER}
    generated = examples.contents(rungs['generated']['archive'])
    scored = examples.contents(rungs['scored-without-traces']['archive'])
    assert scored['traces'] == 0 and scored['candidates'] == 0
    assert generated['traces'] and generated['candidates']
    assert {key: value for key, value in scored.items()
            if key not in ('traces', 'candidates')} == {
        key: value for key, value in generated.items()
        if key not in ('traces', 'candidates')}, (
        'the two rungs must differ only in the recording, never in the reading')

    # And the measurement itself, not merely its counts: same rows, same judged
    # metrics, same selection. A rung that had quietly dropped a row would
    # otherwise still satisfy the counts above.
    for field in ('rows', 'ragas', 'summary', 'selection'):
        assert (rungs['scored-without-traces']['archive']['evaluation']['result'][field]
                == rungs['generated']['archive']['evaluation']['result'][field])
