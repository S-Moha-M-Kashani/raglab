"""The sweep's command line: what the documented options dispatch to, and what
it does instead of sweeping when the backend cannot judge."""
import pytest

from raglab.agents.extra_tools import sweep


def test_the_documented_options_reach_the_sweep_and_the_final_run(monkeypatch):
    # this is a unit test
    """The two things `main` can do, driven the way the README's commands drive
    them: the flags in the help text are parsed and handed on in order, and
    `--final` re-runs one candidate instead of sweeping all of them."""
    calls = []
    monkeypatch.setattr(sweep, 'sweep', lambda *args: calls.append(('sweep',) + args))
    monkeypatch.setattr(sweep, 'final', lambda *args: calls.append(('final',) + args))

    monkeypatch.setattr('sys.argv', ['raglab-sweep', '--limit', '10', '--workers',
                                     '3', '--only', 'A', 'F', '--balance', ''])
    sweep.main()
    assert calls == [('sweep', 10, 3, ['A', 'F'], '')]

    calls.clear()
    monkeypatch.setattr('sys.argv', ['raglab-sweep', '--final', 'F'])
    sweep.main()
    # limit=None: the point of the final run is the whole question set.
    assert calls == [('final', None, 6, 'F', sweep.SWEEP_BALANCE)]


def test_the_sweep_refuses_a_backend_that_cannot_judge_and_writes_no_run(monkeypatch):
    # this is a unit test
    """The suite's `fake` provider answers and judges without ever failing, so a
    sweep run on it would rank five candidates on numbers nothing measured. It
    exits non-zero before building anything, leaving `.runs/` as it found it."""
    before = sorted(sweep.RUNS_DIR.glob('*'))
    monkeypatch.setattr('sys.argv', ['raglab-sweep', '--limit', '1'])
    with pytest.raises(SystemExit, match='no LLM backend'):
        sweep.main()
    assert sorted(sweep.RUNS_DIR.glob('*')) == before


def test_a_candidate_letter_no_candidate_carries_is_named(monkeypatch):
    # this is a unit test
    """`--final Z` used to raise StopIteration — after loading the corpus and
    building a registry — which tells the reader neither what was wrong nor
    which letters exist."""
    monkeypatch.setattr('sys.argv', ['raglab-sweep', '--final', 'Z'])
    with pytest.raises(SystemExit, match="no candidate 'Z'"):
        sweep.main()
