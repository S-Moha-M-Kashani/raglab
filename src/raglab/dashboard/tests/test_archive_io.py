"""Run the browser archive codec's Node contracts when Node is available."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from raglab.configuration.lab_config import LabConfig
from raglab.evaluation.tests import archive_examples as examples

HERE = Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node is absent; the archive codec is a browser file and '
                    'its contract is written in JavaScript')
def test_the_archive_codec_contract():
    # this is an integration test
    result = subprocess.run(['node', '--test', 'archive_io.test.js'],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node is absent; the archive codec is a browser file and '
                    'its contract is written in JavaScript')
def test_the_archive_codec_carries_the_whole_ladder(tmp_path):
    # this is an integration test
    """The browser's export and import, over all four rungs of the ladder.

    The fixtures are generated here rather than written in JavaScript, from the
    same `archive_examples` the server-side contract uses and from `LabConfig`
    itself. That is the point: a knob added to the dataclasses reaches this test
    without anyone remembering to add it to a second list, and the two codecs
    are held to one archive rather than to two fixtures that may have drifted.
    """
    payload = {
        'config': examples.SHIFTED_CONFIG,
        'defaults': LabConfig().to_dict(),
        'ui': examples.SHIFTED_UI,
        'carried': examples.CARRIED,
        'rungs': examples.ladder(),
    }
    ladder = tmp_path / 'ladder.json'
    ladder.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False),
                      encoding='utf-8')
    result = subprocess.run(['node', '--test', 'archive_ladder.test.js'],
                            cwd=HERE, capture_output=True, text=True,
                            env={**os.environ, 'RAGLAB_LADDER': str(ladder)})
    assert result.returncode == 0, result.stdout + result.stderr
