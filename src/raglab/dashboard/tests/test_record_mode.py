"""The Inspector's recorded-experiment mode — which record reaches which
explanation — which is JavaScript."""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


# `frontend/inspector.js` is a browser file, and the thing that was wrong about
# its read-only record mode is not a sentence but a *branch*: the ledger holds
# five record shapes and one test on `rows.length` could tell two stories, so a
# cancelled run, a one-off query, an errored retrieval and an imported archive
# were each handed an explanation belonging to another kind of experiment. Which
# branch a record reaches is behaviour, so it is pinned in JavaScript against
# the served file and shelled out to here — skipping honestly when node is
# absent, the way the sorter's and the board reveal's suites do.
@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node is absent; the recorded-experiment mode is a '
                           'browser file and which view it draws is written in '
                           'JavaScript')
def test_a_recorded_experiment_explains_itself_by_what_it_is():
    # this is an integration test
    result = subprocess.run(['node', '--test', 'record_mode.test.js'],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
