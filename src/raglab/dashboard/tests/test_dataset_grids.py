"""The dataset viewer's columns, its narrowing and its raw tree — which is
JavaScript."""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


# `frontend/dataset.js` is a browser file, and three of the page's promises are
# written there: that every column is one of the dataset's own declared labels,
# that a reading narrows its grid to exactly the rows it named, and that the
# raw view shows a value as recorded. An offline test of the served markup sees
# `<div id="dataset">` and nothing else, so the claims are pinned in JavaScript
# against the served file and shelled out to here — skipping honestly when node
# is absent, the way the board's own suites do.
@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node is absent; the viewer\'s columns are a '
                           'browser file and what they are derived from is '
                           'written in JavaScript')
def test_the_viewers_columns_come_from_the_corpuss_own_declarations():
    # this is an integration test
    result = subprocess.run(['node', '--test', 'dataset_grids.test.js'],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
