"""Where this lab is allowed to listen, and what its launch line must carry."""
import re
from pathlib import Path

from raglab.cli import serve

ROOT = Path(__file__).resolve().parents[1]


def test_the_lab_and_the_inspector_take_no_port_lodestar_owns():
    """`RESERVED` is a copy of Lodestar's port list, not a live read — it can
    drift out of sync and must be updated by hand if Lodestar's changes."""
    assert serve.PANEL_PORT == 9002
    assert serve.INSPECTOR_PORT == 9003
    assert serve.PANEL_PORT != serve.INSPECTOR_PORT
    for port in (serve.PANEL_PORT, serve.INSPECTOR_PORT):
        assert port not in serve.RESERVED, (
            f':{port} belongs to {serve.RESERVED.get(port)}')


def test_the_documented_launch_installs_the_backend_the_default_embedder_needs():
    """Without the matching extra the service starts fine and only fails on
    the first index build, which reads as "the lab is broken" not "install
    this"."""
    config = (ROOT / 'src' / 'raglab' / 'config.py').read_text(encoding='utf-8')
    chosen = re.search(r"embedder: str = '([^']+)'", config)
    assert chosen, 'could not read the default embedder out of the lab config'
    needed = {'fastembed': 'semantic',
              'sentence-transformers': 'local-embeddings'}.get(chosen.group(1))
    if not needed:
        return                      # a hash embedder needs nothing
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert f'--extra {needed}' in readme, (
        f'the README must launch with --extra {needed}')


def test_no_lab_command_names_a_vector_database():
    """The lab's index is deliberately process memory, not a database."""
    pyproject = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    scripts = pyproject[pyproject.index('[project.scripts]'):]
    scripts = scripts[:scripts.index('\n[')]
    assert not re.search(r'chroma|CHROMA|DATABASE', scripts)


def test_the_one_command_runner_delegates_the_launch():
    """`raglab-lab` must call into the serving code rather than respell it —
    a second place that knows the port is the one that goes stale."""
    runner = (ROOT / 'src' / 'raglab' / 'cli' / 'lab.py').read_text(encoding='utf-8')
    assert 'serve.PANEL_PORT' in runner
    assert 'serve.panel()' in runner
    assert 'uvicorn.run' not in runner, (
        'the runner respells the launch — it must delegate to serve')


def test_every_entry_point_resolves_to_something_callable():
    """A name pointing at a function that does not exist fails only at the
    moment somebody runs the command, and nothing else would notice."""
    import importlib

    pyproject = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    block = pyproject[pyproject.index('[project.scripts]'):]
    block = block[:block.index('\n[')]
    entries = dict(re.findall(r'^(\S+) = "([^"]+)"$', block, re.M))
    assert set(entries) == {'raglab', 'raglab-inspector', 'raglab-lab',
                            'raglab-sweep', 'raglab-judgescreen',
                            'raglab-leaderboard'}
    for command, target in entries.items():
        module, _, function = target.partition(':')
        assert callable(getattr(importlib.import_module(module), function)), (
            f'{command} points at {target}, which is not callable')
