"""The leaderboard → Laboratory handoff, which is JavaScript.

Opening a row of the board pins the Inspector to that experiment and makes the
same experiment's settings the Laboratory's. Both halves are browser files —
the slot the board writes, and the rule deciding which of a recorded config's
knobs this installation can actually serve — so the behaviour is pinned in
JavaScript against the served files and shelled out to here, skipping honestly
when node is absent the way the reveal's and the sorter's suites do.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

NEEDS_NODE = pytest.mark.skipif(
    shutil.which('node') is None,
    reason='node is absent; the handoff is a browser file and what decides '
           'which knobs survive it is written in JavaScript')


@NEEDS_NODE
def test_the_handoff_slot_and_its_reconciliation_hold():
    # this is an integration test
    result = subprocess.run(['node', '--test', 'experiment_handoff.test.js'],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@NEEDS_NODE
def test_the_board_hands_the_experiment_over_without_taking_the_link():
    # this is an integration test
    result = subprocess.run(['node', '--test', 'board_handoff.test.js'],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
