"""Where this lab is allowed to listen, and what its launch line must carry."""
import re
from pathlib import Path

from raglab.cli import serve

ROOT = Path(__file__).resolve().parents[1]


def test_the_lab_and_the_inspector_take_no_port_lodestar_owns():
    """These two services run on the same machine as Lodestar's stack, and a
    collision is a service that will not start on whichever one loses the race.

    The list is copied rather than read: the Lodestar repository is no longer a
    dependency, so nothing here can ask it. That is a real weakening — the two
    projects can now drift into a collision no test catches — and it is the
    accepted cost of the split. If Lodestar's allocation changes, this list has to
    be updated by hand."""
    assert serve.PANEL_PORT == 9002
    assert serve.INSPECTOR_PORT == 9003
    assert serve.PANEL_PORT != serve.INSPECTOR_PORT
    for port in (serve.PANEL_PORT, serve.INSPECTOR_PORT):
        assert port not in serve.RESERVED, (
            f':{port} belongs to {serve.RESERVED.get(port)}')


def test_the_documented_launch_installs_the_backend_the_default_embedder_needs():
    """The lab defaults to a Persian-tuned sentence-transformers model. Without
    the extra the service starts happily and then fails on the first index build
    — which reads as "the lab is broken", not "install this". The README is the
    launch line now that there is no npm script to hold it."""
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
    """The lab's index is process memory, deliberately: there is nothing to start
    first and nothing a later run can inherit from an earlier one by accident. A
    command naming a store would be the first step back toward one."""
    pyproject = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    scripts = pyproject[pyproject.index('[project.scripts]'):]
    scripts = scripts[:scripts.index('\n[')]
    assert not re.search(r'chroma|CHROMA|DATABASE', scripts)


def test_the_one_command_runner_delegates_the_launch():
    """`raglab-lab` runs the suite and then serves. It must reach the serving
    code rather than respell it: a second place that knows the port is the one
    that goes stale. In Lodestar the same rule kept `scripts/lab.mjs` from
    copying a uvicorn line carrying four version pins; those pins are in uv.lock
    now, and the port is what is left worth not repeating."""
    runner = (ROOT / 'src' / 'raglab' / 'cli' / 'lab.py').read_text(encoding='utf-8')
    assert 'serve.PANEL_PORT' in runner
    assert 'serve.panel()' in runner
    # `uvicorn.run`, not the word: the module docstring explains what it is
    # deliberately not respelling, and a test that cannot tell prose from code
    # would fail on the sentence saying why it passes.
    assert 'uvicorn.run' not in runner, (
        'the runner respells the launch — it must delegate to serve')


def test_every_entry_point_resolves_to_something_callable():
    """Six commands are the lab's whole surface now that there is no
    package.json. A name pointing at a function that does not exist is a command
    that fails at the moment somebody needs it, and nothing else would notice."""
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
