"""The column sorter's ordering logic, which is JavaScript."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# This is a unit test: it runs the JS suite for the browser file the panels share.
@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node is absent; the sorter is a browser file and '
                           'its ordering suite is written in JavaScript')
def test_the_column_sorter_orders_correctly():
    """`static/sorttable.js` decides what clicking a header does, and it is wrong
    in ways nobody notices — missing values sorting first, a third click that does
    not restore the order the service sent.

    That suite is written in JavaScript because the code is, and it stays that way
    after the move to a Python-only toolchain: a real test dropped to satisfy a
    slogan is a loss. So this shells out and skips honestly when node is not
    installed — the same shape as Lodestar's chat-memory tests skipping when their
    store is down."""
    result = subprocess.run(['node', '--test', 'tests/sorttable.test.js'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
