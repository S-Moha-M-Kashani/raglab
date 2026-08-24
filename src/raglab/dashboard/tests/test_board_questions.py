"""The board's `questions` column — what one row's own selection note reads
as — which is JavaScript."""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


# `frontend/leaderboard.js` is a browser file, and the column's own reading of
# a row (the note when it carries one, the bare count when it does not, sorted
# numerically either way) is written there — so it is pinned in JavaScript,
# against the served file, and shelled out to here — skipping honestly when
# node is absent, the way the reveal's suite does.
@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node is absent; the questions column is a '
                           'browser file and what it reads is written in '
                           'JavaScript')
def test_the_questions_column_reads_the_selection_note_or_falls_back():
    # this is an integration test
    result = subprocess.run(['node', '--test', 'board_questions.test.js'],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
