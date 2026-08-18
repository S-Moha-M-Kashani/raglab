"""Repo-wide guards that pin an absence or a source-text rule rather than a
behaviour: the dependency that was cut (lodestar/brain, chromadb), no
hardcoded machine path, the `.env.example` two-way contract, entry points
that resolve to real callables and name no database, ports that stay
distinct and unreserved, and the README launch line naming the extra the
default embedder needs. One tree walk over `src/` and `tests/`, computed
once, backs the source-scanning guards below.

Retires tests/test_no_lodestar.py, tests/test_config.py, tests/test_ports.py,
and folds in the chromadb-import scan from tests/test_store_index.py plus
the two "no vector database" checks from tests/test_raglab.py and the
`explain.missing()`/`explain.missing_metrics()` gate — all guards about an
absence or a registry's completeness, not about behaviour.

CLAUDE.md's "guard is an absence, not a check" trio: no raglab module
imports chromadb (here), a build opens no socket (tests/test_store_index.py
— stays there, it needs a real build), no entry point names a database
(here, via tomllib rather than string slicing).
"""
import importlib
import re
import tomllib
from pathlib import Path

import pytest

from raglab import config, explain
from raglab import server as lab_server
from raglab.cli import serve
from raglab.config import IndexConfig, LabSettings

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
TESTS = Path(__file__).resolve().parent
THIS_FILE = Path(__file__).resolve()

# The one tree walk every source-scanning guard below shares.
_PY_FILES = sorted([*SRC.rglob('*.py'), *TESTS.rglob('*.py')])
_SRC_FILES = [p for p in _PY_FILES if p.is_relative_to(SRC)]

with open(ROOT / 'pyproject.toml', 'rb') as _fh:
    _SCRIPTS = tomllib.load(_fh)['project']['scripts']


def test_the_tree_walk_finds_a_plausible_number_of_files():
    # this is a convention test
    """Every guard below is a `for path in _PY_FILES` loop, so a walk that
    silently matches nothing — wrong root, wrong glob, an exclusion that eats
    everything — makes every one of them pass vacuously. src/ holds 41
    modules and tests/ holds 26 at the time of writing (41 + 26 = 67, the
    count this assertion checks); anything under 50 means the walk found
    the wrong place."""
    assert len(_PY_FILES) > 50, (
        f'tree walk only found {len(_PY_FILES)} files — check the glob')


def test_nothing_imports_lodestar_brain():
    # this is a convention test
    """Prose mentioning lodestar_brain is fine; only a line the interpreter
    follows is not."""
    offenders = []
    for path in _PY_FILES:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(('import lodestar_brain', 'from lodestar_brain')):
                offenders.append(f'{path.name}:{number}')
    assert not offenders, f'lodestar_brain imported at {offenders}'


def test_no_raglab_module_imports_a_vector_database_client():
    # this is a convention test
    """chromadb is production's dependency, not the lab's. One import line
    would bring the persistence back, and it would look harmless. Broadened
    from the old scan (src/raglab's top level only) to every file the tree
    walk finds, so a chromadb import under raglab/cli or raglab/llm_tools —
    or in a test — would be caught too."""
    offenders = []
    for path in _PY_FILES:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith(('import ', 'from ')):
                continue
            if 'chromadb' in stripped or 'ChromaChatMemory' in stripped:
                offenders.append(f'{path.name}:{number}')
    assert not offenders, f'chromadb/ChromaChatMemory imported at {offenders}'


def test_the_lab_names_no_vector_database_at_all():
    # this is a convention test
    """A database the lab cannot name is one it cannot be pointed at by a
    typo, an old shell, or a copied command."""
    settings = LabSettings()
    assert [f for f in vars(settings) if 'chroma' in f or 'database' in f] == []
    with pytest.raises(TypeError):
        LabSettings(chroma_database='lodestar')


def test_the_lab_ignores_a_leftover_chroma_environment(monkeypatch):
    # this is a convention test
    """The board's Chroma stack runs whenever a board does, and a shell that
    ran the old lab commands still exports these; neither may reach the lab."""
    monkeypatch.setenv('RAGLAB_CHROMA_DATABASE', 'lodestar')
    monkeypatch.setenv('BRAIN_CHROMA_URL', 'http://localhost:8001')
    assert 'lodestar' not in repr(config.load_lab_settings())


def test_a_build_exposes_no_vector_store_gate():
    # this is a convention test
    """The hasattr half of what was one test in test_server.py, split because
    this half is a structural claim about the app (no gate to expose) and the
    other half (a real index build) is an integration test — it stays there.
    With the index in process memory there is nothing that can be down, so
    the gate that could refuse a build is gone rather than passing."""
    assert not hasattr(lab_server, 'require_chroma')


def test_no_path_is_hardcoded_to_one_machine():
    # this is a convention test
    """This file is excluded from its own scan, because a test that scans for
    a hardcoded-path string necessarily contains one."""
    offenders = [path.name for path in _PY_FILES
                 if path.resolve() != THIS_FILE
                 and any(quote + '/Users/' in path.read_text(encoding='utf-8')
                         or quote + '/home/' in path.read_text(encoding='utf-8')
                         for quote in ('"', "'"))]
    assert not offenders, f'a home directory is hardcoded in {offenders}'


# `env` as well as `os.environ`: `load_lab_settings` takes the mapping as an
# argument so a test can hand it one, and a name is what the reads go through.
_ENV_READS = re.compile(r"""
    (?:os\.environ|environ|env)\.get\(\s*['"]([A-Z][A-Z0-9_]{2,})['"]
  | (?:os\.environ|environ|env)\[\s*['"]([A-Z][A-Z0-9_]{2,})['"]\s*\]
  | getenv\(\s*['"]([A-Z][A-Z0-9_]{2,})['"]
  | setdefault\(\s*['"]([A-Z][A-Z0-9_]{2,})['"]
""", re.VERBOSE)

# `#VAR=` or `VAR=` at the start of a line — the template comments every
# variable out, so both spellings count as documented.
_ENV_DOCUMENTED = re.compile(r'^#?\s*([A-Z][A-Z0-9_]{2,})=')


def test_env_example_documents_every_variable_the_code_reads():
    # this is a convention test
    """Scans the src/ subset of the tree walk, not the tests — a variable a
    test invents for one assertion is not configuration anyone should be told
    about."""
    read = {name for path in _SRC_FILES
            for match in _ENV_READS.finditer(path.read_text(encoding='utf-8'))
            for name in match.groups() if name}
    documented = {match.group(1) for line in
                  (ROOT / '.env.example').read_text(encoding='utf-8').splitlines()
                  if (match := _ENV_DOCUMENTED.match(line))}
    read -= {'PATH', 'HOME'}
    assert read - documented == set(), 'read by the code, absent from .env.example'
    assert documented - read == set(), 'in .env.example, read by nothing'


def test_the_lab_and_the_inspector_take_no_port_lodestar_owns():
    # this is a convention test
    """`RESERVED` is a copy of Lodestar's port list, not a live read — it can
    drift out of sync and must be updated by hand if Lodestar's changes."""
    assert serve.PANEL_PORT == 9002
    assert serve.INSPECTOR_PORT == 9003
    assert serve.PANEL_PORT != serve.INSPECTOR_PORT
    for port in (serve.PANEL_PORT, serve.INSPECTOR_PORT):
        assert port not in serve.RESERVED, (
            f':{port} belongs to {serve.RESERVED.get(port)}')


def test_the_documented_launch_installs_the_backend_the_default_embedder_needs():
    # this is a convention test
    """Without the matching extra the service starts fine and only fails on
    the first index build, which reads as "the lab is broken" not "install
    this". Reads `IndexConfig().embedder` directly rather than a regex over
    `config.py`'s source, so a refactor that keeps the same default cannot
    break this test by moving the line it used to match."""
    chosen = IndexConfig().embedder
    needed = {'fastembed': 'semantic',
              'sentence-transformers': 'local-embeddings'}.get(chosen)
    if not needed:
        return                      # a hash embedder needs nothing
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert f'--extra {needed}' in readme, (
        f'the README must launch with --extra {needed}')


def test_the_one_command_runner_delegates_the_launch():
    # this is a convention test
    """`raglab-lab` must call into the serving code rather than respell it —
    a second place that knows the port is the one that goes stale."""
    runner = (ROOT / 'src' / 'raglab' / 'cli' / 'lab.py').read_text(encoding='utf-8')
    assert 'serve.PANEL_PORT' in runner
    assert 'serve.panel()' in runner
    assert 'uvicorn.run' not in runner, (
        'the runner respells the launch — it must delegate to serve')


def test_no_lab_command_names_a_vector_database():
    # this is a convention test
    """The lab's index is deliberately process memory, not a database. Parsed
    with tomllib rather than string-sliced, so a reformatted table (spacing,
    quoting, a comment on its own line) cannot silently stop the guard from
    reading the block it means to check."""
    for command, target in _SCRIPTS.items():
        assert not re.search(r'chroma|CHROMA|DATABASE', f'{command} {target}'), (
            command, target)


def test_every_entry_point_resolves_to_something_callable():
    # this is a convention test
    """A name pointing at a function that does not exist fails only at the
    moment somebody runs the command, and nothing else would notice. Parsed
    with tomllib and each target actually imported and resolved, not just
    pattern-matched against the text."""
    assert set(_SCRIPTS) == {'raglab', 'raglab-inspector', 'raglab-lab',
                             'raglab-sweep', 'raglab-judgescreen',
                             'raglab-leaderboard'}
    for command, target in _SCRIPTS.items():
        module, _, function = target.partition(':')
        assert callable(getattr(importlib.import_module(module), function)), (
            f'{command} points at {target}, which is not callable')


@pytest.mark.parametrize('gate', ['missing', 'missing_metrics'])
def test_nothing_ships_without_an_explainer(gate):
    # this is a convention test
    """The two gates that stop a knob or a metric shipping as a bare word or
    a bare number: a config field with no help text, and a key a run can
    report with nothing defining it."""
    assert getattr(explain, gate)() == []
