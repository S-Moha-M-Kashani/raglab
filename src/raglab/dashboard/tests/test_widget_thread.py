# this is a unit test
"""Runs the widget's thread contract under node, the way the board's handoff
contract already runs. The rules are a browser's, so they are checked in one.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

NEEDS_NODE = pytest.mark.skipif(
    shutil.which('node') is None,
    reason='node is absent; the thread rule is a browser file and which '
           'conversation is on screen is written in JavaScript')


@NEEDS_NODE
def test_which_conversation_the_widget_is_in():
    result = subprocess.run(['node', '--test', 'widget_thread.test.js'],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
