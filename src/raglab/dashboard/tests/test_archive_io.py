"""Run the browser archive codec's Node contract when Node is available."""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node is absent; the archive codec is a browser file and '
                    'its contract is written in JavaScript')
def test_the_archive_codec_contract():
    result = subprocess.run(['node', '--test', 'archive_io.test.js'],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
