"""The board's settings reveal — which one is open — which is JavaScript."""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


# `frontend/leaderboard.js` is a browser file, and what opens and closes the
# reveal stopped being CSS when the box moved into the top layer: a popover
# cannot be opened by a selector. So the behaviour is pinned in JavaScript,
# against the served file, and shelled out to here — skipping honestly when
# node is absent, the way the sorter's suite does.
@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node is absent; the reveal is a browser file and '
                           'what opens it is written in JavaScript')
def test_the_settings_reveal_opens_and_closes_correctly():
    # this is an integration test
    result = subprocess.run(['node', '--test', 'board_reveal.test.js'],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
