"""The column sorter's ordering logic, which is JavaScript."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# `static/sorttable.js` is a browser file, so its ordering suite is written
# in JavaScript and shelled out to here, skipping honestly when node is absent.
@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node is absent; the sorter is a browser file and '
                           'its ordering suite is written in JavaScript')
def test_the_column_sorter_orders_correctly():
    result = subprocess.run(['node', '--test', 'tests/sorttable.test.js'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
