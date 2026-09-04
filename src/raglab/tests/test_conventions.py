"""Repo-wide guards that pin an absence or a source-text rule rather than a
behaviour: the dependency that was cut (the parent assistant's brain package, chromadb), no
hardcoded machine path, the `.env.example` two-way contract, entry points
that resolve to real callables and name no database, ports that stay
distinct and unreserved, and the README launch line naming the extra the
default embedder needs. One tree walk over `src/` and `tests/`, computed
once, backs the source-scanning guards below.

Retires the old parent-import scan, test_config.py, test_ports.py,
and folds in the chromadb-import scan from test_store_index.py plus
the two "no vector database" checks from test_raglab.py and the
`explain.missing()`/`explain.missing_metrics()` gate — all guards about an
absence or a registry's completeness, not about behaviour.

CLAUDE.md's "guard is an absence, not a check" trio: no raglab module
imports chromadb (here), a build opens no socket (test_store_index.py
— stays there, it needs a real build), no entry point names a database
(here, via tomllib rather than string slicing).
"""
import importlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from raglab.conftest import _spacing_literals, _track_literals
from raglab.configuration import lab_config as config
from raglab.configuration import explainer_assembly as explain
from raglab.dashboard import panel_server as lab_server
from raglab.dashboard.cli import serve
from raglab.configuration.lab_config import IndexConfig, LabSettings

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'src'
THIS_FILE = Path(__file__).resolve()

# The one tree walk every source-scanning guard below shares. Tests are
# colocated (each section's tests/ folder), so one walk over src/ sees both.
_PY_FILES = sorted(p for p in SRC.rglob('*.py') if '__pycache__' not in p.parts)
# The non-test subset — what actually ships: colocated tests/ folders and the
# conftest plumbing are test code, excluded from the guards that read only
# the lab itself (the env-example contract, the widget-leaf scan).
_SRC_FILES = [p for p in _PY_FILES
              if 'tests' not in p.parts and p.name != 'conftest.py']

with open(ROOT / 'pyproject.toml', 'rb') as _fh:
    _SCRIPTS = tomllib.load(_fh)['project']['scripts']

# The served stylesheets. Read from disk rather than over a route because this
# guard spans both surfaces at once — the routes belong to two different apps,
# and the claim is about the shared scale rather than about either page.
_SHEETS = SRC / 'raglab' / 'dashboard' / 'frontend'


def test_every_step_of_the_shared_scale_has_a_user():
    # this is a convention test
    """A step nobody reads is exactly what tokens.css exists to remove. It
    shipped with a --t-2xl (32px) reserved for a top bar that had not been built
    yet; the bar was built, it wanted --t-md, and the step sat there unread —
    which is how the last set of hand-set sizes started, one value kept "for
    later". If a step is worth having, something reads it now; if nothing does,
    the decision it records is a guess."""
    sheets = '\n'.join((_SHEETS / name).read_text(encoding='utf-8') for name in
                       ('tokens.css', 'chrome.css', 'panel.css',
                        'inspector.css', 'widget.css'))
    tokens = (_SHEETS / 'tokens.css').read_text(encoding='utf-8')
    declared = dict.fromkeys(re.findall(r'^\s*(--[a-z0-9-]+):', tokens, re.M))
    scale = [t for t in declared
             if re.match(r'--(s-|t-|radius-)|--(measure|gutter|bar-h|rail-h)$', t)]
    assert len(scale) > 15, (
        f'the scale regex matched only {len(scale)} steps — tokens.css was '
        'renamed or restructured and this guard is now passing vacuously')
    unread = [t for t in scale if f'var({t})' not in sheets]
    assert unread == [], (
        f'these steps of the shared scale are read by nothing: {unread}. Use '
        'them or delete them — a step kept for a future phase is a decision '
        'nothing has tested')


# (file, declaration, reason) — every margin, padding or gap on either surface
# that is deliberately off the ramp, and why. The guard below asserts this list
# is exactly what the sheets contain: a tenth entry fails it, and so does a
# retired ninth, so the list cannot rot into a blanket permission.
#
# Four kinds of reason appear here, and no fifth is expected:
#   - above the ramp: --s-7 (3rem) is its top step
#   - below the ramp: --s-px (2px) is its bottom step
#   - not a rem: a value meant to scale with one element's own type size
#   - a visual weight rather than a step, like a border width
SPACING_OFF_RAMP = [
    ('panel.css', 'padding: 0 0 5rem',
     'clearance under the whole page for the fixed widget launcher, which is '
     'taller than the ramp\'s top step — a bigger step invented for one use is '
     'a step with one user. It stays keyed to panel.css because it is a rule '
     'on the page: the launcher\'s own rules live in widget.css'),
    ('inspector.css', 'padding-bottom: 5rem',
     'clearance under the whole page for the fixed widget launcher, which is '
     'taller than the ramp\'s top step — the Laboratory reserves the same, for '
     'the same reason'),
    ('panel.css', 'padding-bottom: 2px',
     'the gap between a link and the rule underlining it, which is a border '
     'offset rather than spacing between things'),
    ('panel.css', 'padding: .05rem',
     "a sub-pixel nudge for a chip's vertical padding; --s-px (2px) is the "
     'ramp\'s bottom step and this sits under it'),
    ('inspector.css', 'padding: .08rem',
     'the same nudge for the archive pill, under the same bottom step'),
    ('inspector.css', 'padding: .38rem',
     "one button's vertical padding, tuned between --s-1 and --s-2 rather than "
     'snapped to either'),
    ('inspector.css', 'gap: 2px',
     'the hairline seam between the ladder bars — a visual weight, like a '
     'border width, not a distance between elements'),
    ('inspector.css', 'padding: 0 .3em',
     "em-relative to the layer badge's own mono size: the ramp is rem-based, so "
     "a value meant to scale with one element's type falls outside it"),
    ('inspector.css', 'padding: 0 .12em',
     'the same, for the evidence wash that has to hug the words it marks'),
    ('inspector.css', 'gap: var(--s-1) 1.3rem',
     'the column gap between metric badges, which lands between --s-4 and '
     '--s-5 — and note the row gap beside it is on the ramp'),
    ('dataset.css', 'margin-bottom: -1px',
     "the selected tab's own 2px rule pulling onto the tab strip's 1px one, so "
     'the chosen tab reads as attached to the panel it opened rather than '
     'sitting above a second line — the negative of a border width, which the '
     'ramp explicitly does not cover, and it must track that width rather than '
     'a spacing step'),
    ('chrome.css', 'margin-left: -1px',
     'the theme segments pulling onto each other so neighbours share one edge '
     'instead of drawing two — it is the negative of a border width, which the '
     'ramp explicitly does not cover, and it must track that width rather than '
     'a spacing step'),
]


def test_every_spacing_value_comes_off_the_ramp_or_is_named_here():
    # this is a convention test
    """The scale added ~220 token uses for spacing and no way to notice the
    220th-and-first hand-set value. The type ramp and the radius scale each got
    a literal-detector; spacing did not, on the argument that no regex could
    gate it without rejecting correct CSS. That argument was wrong: across four
    sheets there are nine literals, every one of them a deliberate exception
    with a reason, so what spacing needed was a named list — this one — and not
    an impossibility proof.

    Asserted as equality rather than containment, in both directions. A tenth
    literal fails because nothing explains it; a retired ninth fails too,
    because a list that keeps permitting what no longer exists stops describing
    the sheets and starts excusing them."""
    allowed = sorted((f, d) for f, d, _ in SPACING_OFF_RAMP)
    found = sorted((name, hit)
                   for name in ('tokens.css', 'chrome.css', 'panel.css',
                                'inspector.css', 'widget.css', 'dataset.css')
                   for hit in _spacing_literals(
                       (_SHEETS / name).read_text(encoding='utf-8')))
    assert found == allowed, (
        'the spacing ramp and this list disagree.\n'
        f'  not on the ramp and not named here: {sorted(set(found) - set(allowed))}\n'
        f'  named here but no longer in the sheets: {sorted(set(allowed) - set(found))}\n'
        'Take the value off the ramp only for one of the four reasons the list '
        'already carries, and write the reason beside the declaration too — a '
        'reader of the stylesheet cannot see this file.')


def test_every_letter_spacing_value_comes_from_the_label_recipe():
    # this is a convention test
    """One decision — how loose a small uppercase label is tracked — was made
    twenty times at eight values between .04 and .14em, and every one of the
    twenty sites is uppercase. `--label-track` is the answer; this is what stops
    the twenty-first site from asking again. No exception list, because unlike
    spacing there is no value here that has a reason to be off it: the three
    negative values are outside this guard by construction, since tightening
    three specific dense elements is not this recipe drifting."""
    for name in ('tokens.css', 'chrome.css', 'panel.css', 'inspector.css',
                 'widget.css', 'dataset.css'):
        css = (_SHEETS / name).read_text(encoding='utf-8')
        assert _track_literals(css) == [], (
            f'{name} spells its own tracking: {_track_literals(css)}. Read '
            '--label-track instead — and if this label genuinely wants a '
            'different looseness, that is a second recipe to name, not a '
            'number to write here')


def test_the_spacing_and_tracking_detectors_see_every_form():
    # this is a convention test
    """A regex that matches nothing passes a file-based assertion exactly like
    a correct one does, and both literal-detectors before these shipped with
    holes that had to be closed twice — a `font` shorthand with no line-height,
    then four radius corner longhands and three unit families. So each helper is
    fed strings it must catch and strings it must not, covering every form the
    property takes."""
    caught = [
        'a { margin: 4px }', 'a { padding: 1.5rem }',
        'a { margin-top: 4px }', 'a { padding-bottom: .5em }',
        'a { margin-block: 4px }', 'a { padding-inline-start: 2rem }',
        'a { margin-block-end: 1vh }', 'a { padding-inline: 3% }',
        'a { gap: 2px }', 'a { row-gap: 1ch }', 'a { column-gap: 2ex }',
        'a { grid-gap: 4px }', 'a { grid-row-gap: 4pt }',
        'a { margin-top: -2px }',
        'a { padding: var(--s-1) 1.3rem }',
        'a { margin: 0 auto 2rem }',
    ]
    missed = [
        'a { margin: 0 }', 'a { padding: 0 0 }', 'a { margin: 0 auto }',
        'a { padding: var(--s-2) var(--s-3) }',
        'a { gap: var(--s-1) }', 'a { margin-inline: var(--gutter) }',
        # not spacing: a border width, a position inset, a shadow offset
        'a { border-width: 2px }', 'a { top: 3rem }', 'a { left: -2px }',
        'a { box-shadow: 0 1px 2px black }',
        # a token whose own name carries digits and letters that look like units
        'a { padding: var(--s-px) var(--t-2xs) }',
        # its own documentation
        '/* padding: .38rem is deliberate here */ a { padding: var(--s-1) }',
    ]
    for css in caught:
        assert _spacing_literals(css) != [], css
    for css in missed:
        assert _spacing_literals(css) == [], css

    for css in ('a { letter-spacing: .13em }', 'a { letter-spacing: 0.07em }',
                'a { letter-spacing: 1px }', 'a { letter-spacing: .5ch }'):
        assert _track_literals(css) != [], css
    for css in ('a { letter-spacing: var(--label-track) }',
                'a { letter-spacing: 0 }', 'a { letter-spacing: normal }',
                'a { letter-spacing: -.01em }',
                '/* letter-spacing: .13em was the eighth value */ a { color: red }'):
        assert _track_literals(css) == [], css


def test_the_tree_walk_finds_a_plausible_number_of_files():
    # this is a convention test
    """Every guard below is a `for path in _PY_FILES` loop, so a walk that
    silently matches nothing — wrong root, wrong glob, an exclusion that eats
    everything — makes every one of them pass vacuously. src/ holds ~55 lab
    modules plus ~28 colocated test files at the time of writing; anything
    under 50 means the walk found the wrong place."""
    assert len(_PY_FILES) > 50, (
        f'tree walk only found {len(_PY_FILES)} files — check the glob')


def test_nothing_imports_the_parent_brain_package():
    # this is a convention test
    """`lodestar_brain` is the package of the production assistant this lab
    was extracted from. Prose mentioning it is fine; only a line the
    interpreter follows is not."""
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
        LabSettings(chroma_database='legacy')


def test_the_lab_ignores_a_leftover_chroma_environment(monkeypatch):
    # this is a convention test
    """The board's Chroma stack runs whenever a board does, and a shell that
    ran the old lab commands still exports these; neither may reach the lab."""
    monkeypatch.setenv('RAGLAB_CHROMA_DATABASE', 'legacy')
    monkeypatch.setenv('BRAIN_CHROMA_URL', 'http://localhost:8001')
    assert 'legacy' not in repr(config.load_lab_settings())


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


# D2's retired dataset dialect (`sessions`/`messages`/`mood`/`answerable`/
# `key_facts`/`type`/`difficulty`/`message_indices`/`question_fa` are gone
# from the code, not aliased). Only the seven of those that are not also
# ordinary English words with a legitimate life in prose — `sessions`,
# `messages`, `type` and `difficulty` read fine in a sentence about anything;
# these seven read, in code, only as the field they used to name.
_RETIRED_DIALECT = ('mood', 'valence', 'arousal', 'answerable', 'key_facts',
                     'message_indices', 'question_fa')
# The shape a dict/JSON key access takes — `row['key_facts']`,
# `document.get('mood')` — the word alone between its own quotes, nothing
# else inside them. A bare substring scan would also trip on legitimate
# prose that explains the retirement in the past tense
# (`importance_of`'s own docstring: "`mood.valence`/`arousal` are gone");
# names the diary's own domain content (the dataset-specific help text's
# "date, mood, topics, threads"); or names the *concept* the schema still
# has, just not as a stored field (`answerable`/`unanswerable` describing
# what `behavior` implies) — none of which wrap the word alone in quotes.
_RETIRED_DIALECT_AS_A_KEY = re.compile(
    r"""['"](?:%s)['"]""" % '|'.join(_RETIRED_DIALECT))


def test_no_raglab_module_reads_the_retired_dataset_dialect():
    # this is a convention test
    """Scoped to `_SRC_FILES` — what actually ships — not the full tree walk.
    An old-shape recorded row is a record, not a corpus (D2/D8), so a test
    fixture simulating one legitimately carries every word here forever:
    `archive_examples.py` documents its old-shape dict verbatim from the
    pre-migration code, and `experiment_handoff.test.js`/`panel_open.test.js`
    (not Python, so outside this walk regardless) deliberately recorded
    `key_facts_judge` — a different, still-current knob rename, not this
    dialect — to prove a retired field name is dropped and named rather than
    silently adopted. None of that is the dialect creeping back into the lab
    itself; this guard is about the code, not about a fixture's memory of
    what the code used to read."""
    offenders = []
    for path in _SRC_FILES:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if _RETIRED_DIALECT_AS_A_KEY.search(line):
                offenders.append(f'{path.name}:{number}: {line.strip()}')
    assert not offenders, f'retired dataset dialect read as a key at {offenders}'


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


def test_the_lab_takes_no_reserved_port():
    # this is a convention test
    """`RESERVED` is a hand-copied list of ports other local services own, not
    a live read — it can drift out of sync and must be updated by hand. There
    is one port now: the Inspector moved to /inspector on this one, so that a
    conversation and a theme choice can cross between the surfaces."""
    assert serve.PANEL_PORT == 9002
    assert serve.PANEL_PORT not in serve.RESERVED, (
        f':{serve.PANEL_PORT} belongs to {serve.RESERVED.get(serve.PANEL_PORT)}')
    assert not hasattr(serve, 'INSPECTOR_PORT'), (
        'the Inspector has no port of its own — it is a path on the lab')


def test_the_documented_launch_installs_the_backend_the_default_embedder_needs():
    # this is a convention test
    """Without the matching extra the service starts fine and only fails on
    the first index build, which reads as "the lab is broken" not "install
    this". Reads `IndexConfig().embedder` directly rather than a regex over
    `lab_config.py`'s source, so a refactor that keeps the same default cannot
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
    runner = (ROOT / 'src' / 'raglab' / 'dashboard' / 'cli'
              / 'lab.py').read_text(encoding='utf-8')
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
    assert set(_SCRIPTS) == {'raglab', 'raglab-lab', 'raglab-sweep',
                             'raglab-judgescreen', 'raglab-export',
                             'raglab-leaderboard'}
    for command, target in _SCRIPTS.items():
        module, _, function = target.partition(':')
        assert callable(getattr(importlib.import_module(module), function)), (
            f'{command} points at {target}, which is not callable')


def test_the_widget_package_is_a_deletable_leaf():
    # this is a convention test
    """The widget is a helper outside the measured seam, and removable:
    deleting src/raglab/agents/widget/ plus the handful of dashboard files
    listed below must strand nothing. Pinned in both directions with real
    import parsing rather than a regex, so a parenthesised import list cannot
    slip past.

    The direction is the claim. The widget package reaches the lab only
    through its two unmeasured edges — the CLI chat drive and the env
    settings — never the chat factory, the pipeline, evaluation or the stores
    (its skills corpus loader lives *inside* the package now, so it is no
    longer an edge at all). And the lab reaches the widget only from
    `dashboard/`, from the named files and nowhere else: the factory that
    wires it, the developer's trace page, its own route module (and the
    package line that names it), and the two sections that must tell it when
    what it cached changed underneath — a retyped OpenRouter key, an imported
    corpus. Nothing in `corpora/`, `evaluation/`, `rag_components/` or
    `llm_backends/` may reach in, which is what keeps the measured path free
    of it.

    Adding a name here is a decision, not a formality: it lengthens the list
    of files that a deletion of the widget has to edit."""
    import ast
    widget_pkg = 'raglab.agents.widget'
    widget_dir = SRC / 'raglab' / 'agents' / 'widget'
    allowed_into_lab = {'raglab.llm_backends.cli_subprocess_chat',
                        'raglab.configuration.env_settings'}
    # Paths rather than bare file names: `widget.py` alone would name two
    # different files here.
    allowed_to_reach_in = {
        'raglab/dashboard/panel_server.py',      # wires the key resolver and
                                                 # the experiment reader in
        'raglab/dashboard/dev_trace_page.py',    # renders one widget thread
        'raglab/dashboard/routes/__init__.py',   # names the route modules
        'raglab/dashboard/routes/widget.py',     # the widget's own routes
        'raglab/dashboard/routes/credentials.py',  # a retyped key: widget.reset()
        'raglab/dashboard/routes/datasets.py',   # an import: forget_board_dataset_ids()
    }

    def resolved(node, path) -> list[str]:
        """Absolute dotted targets of one import node, raglab ones only."""
        if isinstance(node, ast.Import):
            return [a.name for a in node.names if a.name.startswith('raglab')]
        parts = list(path.relative_to(SRC).with_suffix('').parts)
        if parts[-1] == '__init__':
            parts = parts[:-1]
        if node.level:
            base = parts[:len(parts) - node.level]
            module = '.'.join(base + (node.module.split('.') if node.module
                                      else []))
        else:
            module = node.module or ''
        if not module.startswith('raglab'):
            return []
        # `from <pkg> import x` — the package is imported, and x may itself be
        # a submodule; count both names, or a nested package would hide behind
        # its parent (`from raglab.agents import widget` reading as a mere
        # import of `raglab.agents`).
        if module:
            return [module] + [f'{module}.{a.name}' for a in node.names]
        return [a.name for a in node.names]

    reachers, escapes = [], []
    for path in _SRC_FILES:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        where = path.relative_to(SRC).as_posix()
        inside = path.is_relative_to(widget_dir)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in resolved(node, path):
                if inside:
                    if (not target.startswith(widget_pkg)
                            and not any(target == edge or target.startswith(edge + '.')
                                        for edge in allowed_into_lab)):
                        escapes.append(f'{path.name} imports {target}')
                elif 'widget' in target and where not in allowed_to_reach_in:
                    reachers.append(f'{where} imports {target}')
    assert not reachers, (
        'only these files may reach the widget — '
        f'{sorted(allowed_to_reach_in)} — but {reachers}')
    assert not escapes, f'the widget escaped its unmeasured edges: {escapes}'


def test_the_panel_context_holds_and_does_not_decide():
    # this is a convention test
    """The panel's routes are handed one container of the state they used to
    close over. It is allowed to hold data and callables the application
    factory built, and nothing else: the moment a method on it screens a
    config, picks a backend or shapes a response, it has become the service
    layer this project's complexity gate refuses, and fifty routes would then
    have somewhere to hide logic that belongs in the plumbing they share.

    Compared against a bare frozen dataclass rather than a written-out list of
    dunders, so the check keeps meaning the same thing when Python adds another
    generated attribute."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class _Reference:
        only: int

    context = lab_server.PanelContext
    assert dataclasses.is_dataclass(context)
    assert context.__dataclass_params__.frozen, 'the context must be frozen'
    # `vars()` below reads only the class's own dict, so a base class is where
    # a method could still hide. There is no base.
    assert context.__mro__[1:] == (object,), (
        'the context inherits from nothing — logic in a base is still logic')
    own = sorted(name for name in vars(context) if name not in vars(_Reference))
    assert not own, f'the context grew {own} — it holds, it does not decide'
    assert all(field.default is dataclasses.MISSING
               and field.default_factory is dataclasses.MISSING
               for field in dataclasses.fields(context)), (
        'every field is passed in at construction; none is decided here')


# The nine operations a mounted Inspector reads the lab through, and the panel
# route each one is the caller's side of. The seam and the route surface are
# two halves of one promise, and this table is where they are checked against
# each other.
_LAB_OPERATIONS = {
    'imported_archive': ('GET', '/api/imported-archives/{archive_id}'),
    'active_archive': ('GET', '/api/imported-archives/active'),
    'clear_active_archive': ('DELETE', '/api/imported-archives/active'),
    'experiment': ('GET', '/api/experiments/{experiment_id}'),
    'experiment_archive': ('GET', '/api/experiments/{experiment_id}/archive'),
    'experiment_questions': ('GET', '/api/experiments/{experiment_id}/questions'),
    'add_experiment_question': ('POST', '/api/experiments/{experiment_id}/questions'),
    'job': ('GET', '/api/jobs/{job_id}'),
    'jobs': ('GET', '/api/jobs'),
}


def test_the_inspector_owns_none_of_the_records_it_shows():
    # this is a convention test
    """The Inspector reads the lab; it never opens what the lab owns.

    That is the whole reason its records arrive through a seam rather than off
    disk: the ledger, the archive store and the corpus store have one writer,
    and a second reader that opened the files would be a second thing to keep
    in step with a record whose value is being written once. `Jobs()` with no
    recorder is the same rule for the scratch builds the Inspector does make —
    the lab passes `record=ledger.record` at its own construction site, and
    that contrast is what this guard is built on.

    The seam is the other half: nine operations, each the caller's side of a
    route the panel actually serves. A tenth that no route backs, or a route
    renamed underneath one, is drift the reader would only meet as an empty
    page."""
    from raglab.dashboard.service_route_plumbing import LabAccess

    inspector = next(path for path in _SRC_FILES
                     if path.name == 'inspector_server.py')
    source = inspector.read_text(encoding='utf-8')
    for owned in ('service_experiment_ledger', 'experiment_archive_store',
                  'corpus_store'):
        assert owned not in source, (
            f'the Inspector must not open {owned} — it asks the lab that owns it')
    assert 'jobs = Jobs(max_history=settings.max_job_history)' in source, (
        'the Inspector must construct its job table with no recorder')
    # The grep above reads the construction site; this reads what that site
    # actually builds. A `record` that defaulted to `ledger.record` would leave
    # the Inspector's literal call unchanged and make it a second writer of the
    # ledger anyway — the exact thing the greps are here to prevent.
    assert lab_server.Jobs().record is None, (
        'a job table built without a recorder must record nothing — the '
        'Inspector asks for no recorder and must not be given one by default')
    assert 'record=ledger.record' not in source, (
        "the Inspector must not adopt the lab's own recording call")
    panel = next(path for path in _SRC_FILES if path.name == 'panel_server.py')
    assert 'Jobs(record=ledger.record,' in panel.read_text(encoding='utf-8'), (
        'the lab, unlike the Inspector, does record — the contrast this guard '
        'depends on')

    named = {name for name in dir(LabAccess) if not name.startswith('_')}
    assert named == set(_LAB_OPERATIONS), (
        'the seam must name exactly the operations the Inspector needs')
    app = lab_server.create_app()
    access = app.state.lab_access
    served = {(method, route.path)
              for route in app.routes
              for method in getattr(route, 'methods', ())}
    for operation, (method, path) in _LAB_OPERATIONS.items():
        assert callable(getattr(access, operation, None)), (
            f'the panel answers no {operation}')
        assert (method, path) in served, (
            f'{operation} claims to be the caller of {method} {path}, which '
            'the panel does not serve')


def test_only_one_function_in_the_widget_writes_a_system_line():
    # this is a convention test
    """`backends._run` is the only author of the widget's system lines, and two
    other pieces of code now depend on that.

    `hooks.trim_and_call` leaves a superseded standing line out of a call, and
    `dev_trace_page` dims it, and both decide *which* line by reading the
    marker `_run` stamps on it. A system message built anywhere else in the
    package and written into `WidgetState` would carry no marker: safe, by
    design — an unmarked line is always sent — but it would also be a second
    author of the thread's standing text, and whoever adds one has to say what
    supersedes it. This guard is what makes them say so, by failing here first.

    Parsed rather than grepped, and scoped to the widget's own shipping files:
    a `SystemMessage` built for a one-off model call elsewhere in the lab never
    reaches this state and is none of this rule's business. Four spellings are
    caught — a bare call, one at module level, one through a module
    (`messages.SystemMessage(...)`), and one under an import alias — because a
    guard that pins one spelling teaches the next reader to use another. What
    it cannot see is a message handed in from outside the package; the failure
    mode there is the safe one, an unmarked line that is always sent.
    """
    import ast
    widget_dir = SRC / 'raglab' / 'agents' / 'widget'
    authors = set()
    for path in _SRC_FILES:
        if not path.is_relative_to(widget_dir):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = {'SystemMessage'}
        holders = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names |= {a.asname for a in node.names
                          if a.name == 'SystemMessage' and a.asname}
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    holders.setdefault(inner, node.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, 'id', '') or getattr(node.func, 'attr', '')
            if called in names:
                authors.add(f'{path.name}:{holders.get(node, "<module>")}')
    assert authors == {'backends.py:_run'}, (
        'only backends._run may write a system line into a widget thread; '
        f'found {sorted(authors)}')


def test_every_live_probe_gates_itself_and_the_secrets_guard_stands_down():
    # this is a convention test
    """The two halves of the live-run contract, pinned together because each
    is unsafe without the other. Every `*_live.py` test file must gate itself
    on being named on the pytest command line (`invocation_params.args`), or
    a directory sweep would silently pay for real model calls; and the
    conftest's secrets guard must carry its `_names_a_live_file` stand-down,
    or a named live run would find its keys blanked and skip forever —
    a probe that can never run is a guard that reports nothing."""
    live_files = [p for p in _PY_FILES
                  if p.name.endswith('_live.py') and 'tests' in p.parts]
    assert live_files, 'the walk found no live probes — check the glob'
    for path in live_files:
        assert 'invocation_params.args' in path.read_text(encoding='utf-8'), (
            f'{path.name} must skip unless named on the command line')
    conftest = (SRC / 'raglab' / 'conftest.py').read_text(encoding='utf-8')
    assert '_names_a_live_file(request)' in conftest, (
        'the secrets guard must stand down for a run that names a live file')


@pytest.mark.parametrize('gate', ['missing', 'missing_metrics', 'missing_briefs'])
def test_nothing_ships_without_an_explainer(gate):
    # this is a convention test
    """The three gates that stop a knob or a metric shipping as a bare word or
    a bare number: a config field with no help text, a key a run can report
    with nothing defining it, and — since the panel reads every explainer in
    two lengths — a topic whose one-sentence version is unusable.

    `missing_briefs` is the newest and the one worth explaining, because it can
    fail on text nobody thought they were breaking. A brief is the opening
    sentence of the help itself, taken in one place rather than written twice,
    so rewriting a help text rewrites its brief — and a rewrite that opens with
    a 400-character sentence, or with the shared dataset-specific caveat, or
    with nothing but the knob's own name, fails here. The fix is never to widen
    the limit: either the sentence stands on its own, or the topic declares a
    brief in `explain.BRIEF`."""
    assert getattr(explain, gate)() == []


def test_the_leaderboard_rule_describes_the_code_that_exists():
    # this is a convention test
    """A rule the code disobeys is worse than either rule alone. The board was
    deliberately flattened to one table per dataset — the owner's call about
    their own instrument — so the document moves with it. The stricter claim is
    not deleted: `group()`/`verdict()` still partition by question set and judge
    and still refuse to name a winner inside the combined error, because that is
    what a *sweep* ranks by and a sweep's candidates are comparable by
    construction."""
    guide = ROOT / 'CLAUDE.md'
    if not guide.exists():
        pytest.skip('the agent guide is kept on disk, not in the repository')
    text = guide.read_text(encoding='utf-8')
    assert 'one table per dataset' in text.lower(), (
        'the board is one table per dataset and the doc must say so')
    assert 'verdict' in text.lower() or 'comparability' in text.lower(), (
        'the sweep still groups before it ranks, and dropping that from the '
        'doc would invite someone to delete the machinery it protects')

    from raglab.evaluation import leaderboard
    assert hasattr(leaderboard, 'verdict') and hasattr(leaderboard, 'group'), (
        'the sweep imports both; flattening the board must not have cost them')
    assert hasattr(leaderboard, 'by_dataset')


# --- Two guards over prose. A tracked file that misstates the corpus size or
# names a path that moved is the same defect as a row that lies about what
# produced it: the reader believes it, and nothing else notices. Both have a
# mechanism — a folder added, a package moved — which is why they are pinned
# and the `npm run raglab` fossil next door was only deleted. ---

_SKILLS = ROOT / 'fixtures' / 'skills'

_NUMBER_WORDS = {'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
                 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16}


def test_the_skills_readme_counts_the_skills_that_are_on_disk():
    # this is a convention test
    """The corpus grows by adding a folder, so the count in its README goes
    stale without anything failing, and that count is load-bearing: the
    widget's own tool description quotes the size to justify listing every
    skill in one call. The loader is already held to the folders that exist
    (`test_skills.py`); this holds the prose to them."""
    on_disk = sum(1 for p in _SKILLS.iterdir() if (p / 'SKILL.md').exists())
    readme = (_SKILLS / 'README.md').read_text(encoding='utf-8')
    counted = re.findall(
        r'\b([A-Za-z]+)\s+(?:skills?|SKILL\.md|descriptions?)\b', readme)
    wrong = {word for word in counted
             if word.lower() in _NUMBER_WORDS
             and _NUMBER_WORDS[word.lower()] != on_disk}
    assert not wrong, f'{sorted(wrong)} counts a corpus that holds {on_disk}'


# A path is a promise the reader can check in one command, so a wrong one is
# caught by nobody until they check. URLs and `VAR=value` assignments are
# stripped first: a model slug (`jinaai/jina-reranker-v2-base-multilingual`)
# and a base URL both look like paths and are neither.
_PATHY = re.compile(r"""
    `([^`\n]*/[^`\n]*)`                      # anything backticked with a slash
  | (?<![\w./-])([\w.-]+(?:/[\w.-]+)*/)(?![\w.-])   # a bare directory: slash last
                                             #   (so a model slug, openai/gpt-5-nano,
                                             #   is not read as a directory)
  | (?<![\w./-])([\w-]+\.(?:py|json|md|db))  # or a bare file with a known suffix
""", re.VERBOSE)


def test_every_path_env_example_names_is_a_path_that_exists():
    # this is a convention test
    """The template already round-trips its *variables* against the code
    (above). Its prose names paths too — where the corpora live, which test
    enforces the contract, which package reads the widget's four — and every
    one of those went stale the moment the packages moved. A bare `name.py` is
    resolved against the whole tree rather than one directory, so a module that
    moved still counts as found; the claim is that the file exists, not that
    the sentence knows where it lives."""
    text = (ROOT / '.env.example').read_text(encoding='utf-8')
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'^#?\s*[A-Z][A-Z0-9_]*=.*$', '', text, flags=re.MULTILINE)
    names = {p.name for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}

    missing = []
    for match in _PATHY.finditer(text):
        ref = next(group for group in match.groups() if group)
        if ref.endswith('.py') and '/' not in ref:
            if ref not in names:
                missing.append(ref)
        elif not (ROOT / ref).exists():
            missing.append(ref)
    assert not missing, f'.env.example names {missing}, which do not exist'


def _run_pre_push(update: str) -> subprocess.CompletedProcess[str]:
    hook = ROOT / 'scripts' / 'git-hooks' / 'pre-push'
    assert hook.is_file(), 'the repository must ship its master-only push hook'
    return subprocess.run(
        [str(hook), 'origin', 'https://example.invalid/raglab.git'],
        input=update,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize('update', [
    f'refs/heads/master {"1" * 40} refs/heads/master {"0" * 40}\n',
    # Version tags are published with the branch they mark; the hook is an
    # allowlist and they are on it.
    f'refs/tags/v1.0.0 {"1" * 40} refs/tags/v1.0.0 {"0" * 40}\n',
    f'(delete) {"0" * 40} refs/heads/gone {"1" * 40}\n',
])
def test_the_push_hook_allows_master_and_its_version_tags(update):
    # this is a convention test
    result = _run_pre_push(update)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('update', [
    f'refs/heads/development {"1" * 40} refs/heads/development {"0" * 40}\n',
    f'refs/heads/feature/anything {"1" * 40} refs/heads/feature/anything {"0" * 40}\n',
    f'refs/tags/archive/pre-rewrite-master {"1" * 40} refs/tags/x {"0" * 40}\n',
    # The check is on the *local* ref, so development wearing another name on
    # the remote is refused too. That is the whole reason it is an allowlist.
    f'refs/heads/development {"1" * 40} refs/heads/master {"0" * 40}\n',
])
def test_the_push_hook_rejects_everything_except_master(update):
    # this is a convention test
    result = _run_pre_push(update)
    assert result.returncode != 0
    assert 'refused' in result.stderr


# --- The sanctioned duplication: two readable templates mirror the two JSON
# schemas that are the corpus/ground-truth contract's real source of truth.
# The duplication is deliberate (an author reads the template, a machine
# reads the schema) so it needs a guard that pins it the way the widget's
# prompt fixtures are pinned against the code that reads them — not
# byte-equal, since a schema and its template are two different documents,
# but key for key: every property a schema declares under "properties" must
# have a same-named counterpart in the template, at every depth an object,
# an array-of-one-example or an additionalProperties-map can be walked into,
# and nothing may appear in one and not the other. ---

_DATASETS = ROOT / 'fixtures' / 'corpus_groundtruth_datasets'


def _descend_schema(schema_node):
    """How to walk one level further into a schema node: ('object', the
    properties dict), ('array', the items schema), ('map', the
    additionalProperties schema — only when that schema itself declares
    properties, i.e. a label-declaration table), or (None, None) for a leaf
    or an opaque additionalProperties:true/string blob, which carries no
    fixed key set to mirror."""
    if not isinstance(schema_node, dict):
        return None, None
    if 'properties' in schema_node:
        return 'object', schema_node['properties']
    if schema_node.get('type') == 'array' and 'items' in schema_node:
        return 'array', schema_node['items']
    additional = schema_node.get('additionalProperties')
    if isinstance(additional, dict) and 'properties' in additional:
        return 'map', additional
    return None, None


def _template_keys(node):
    """A template dict's real declared keys: its own documentation
    (_read_me, _note) and the single <placeholder> key standing in for an
    additionalProperties map are markers, not properties."""
    return {k for k in node
            if not k.startswith('_') and not (k.startswith('<') and k.endswith('>'))}


def _template_placeholder_value(node):
    for key, value in node.items():
        if key.startswith('<') and key.endswith('>'):
            return value
    return None


def _mirror_mismatches(schema_node, template_node, path, exceptions):
    """Every place the schema and the template disagree about which keys
    exist at some depth, as human-readable strings; [] means they mirror
    each other fully from this node down. `exceptions` names paths (by the
    same dotted/bracketed spelling this function builds) where a template is
    allowed to abbreviate rather than restate a schema in full — the one
    case here is documented beside its own entry, not a blanket escape."""
    kind, child = _descend_schema(schema_node)
    if kind is None or path in exceptions:
        return []
    if kind == 'object':
        if not isinstance(template_node, dict):
            return [f'{path}: schema declares an object, template does not']
        schema_keys, template_keys = set(child), _template_keys(template_node)
        problems = []
        if schema_keys - template_keys:
            problems.append(f'{path}: in the schema, missing from the '
                             f'template: {sorted(schema_keys - template_keys)}')
        if template_keys - schema_keys:
            problems.append(f'{path}: in the template, missing from the '
                             f'schema: {sorted(template_keys - schema_keys)}')
        for key in schema_keys & template_keys:
            problems += _mirror_mismatches(child[key], template_node[key],
                                            f'{path}.{key}', exceptions)
        return problems
    if kind == 'array':
        if not isinstance(template_node, list) or not template_node:
            return [f'{path}: schema declares an array, template gives no '
                    'example item to mirror against']
        return _mirror_mismatches(child, template_node[0], f'{path}[]', exceptions)
    # kind == 'map'
    if not isinstance(template_node, dict):
        return [f'{path}: schema declares a keyed table, template is not an object']
    placeholder = _template_placeholder_value(template_node)
    if placeholder is None:
        return [f'{path}: template names no <placeholder> key for this table']
    return _mirror_mismatches(child, placeholder, path, exceptions)


def test_the_template_mirror_detector_sees_added_and_missing_keys():
    # this is a convention test
    """Proves the walker above actually notices drift in both directions
    before it is trusted on the real schemas: a key the schema declares and
    the template omits, and a key the template invents that the schema does
    not — one nested under an ordinary object, to exercise the recursive
    case a shallow key-set diff would miss."""
    schema = {'properties': {
        'a': {'type': 'string'},
        'b': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
    }}
    matching = {'a': '...', 'b': {'x': '...'}}
    assert _mirror_mismatches(schema, matching, 'root', set()) == []

    missing_in_template = {'a': '...', 'b': {}}
    assert _mirror_mismatches(schema, missing_in_template, 'root', set()) != []

    extra_in_template = {'a': '...', 'b': {'x': '...', 'y': 'surprise'}}
    assert _mirror_mismatches(schema, extra_in_template, 'root', set()) != []


def test_corpus_template_mirrors_corpus_schema_key_for_key():
    # this is a convention test
    """The sanctioned duplication, pinned: a key added to schema_corpus.json
    (such as the ranks label the agentic retriever's importance weight
    reads) with nothing added to corpus_template.json — or the reverse —
    fails here rather than reaching an author copying the template by hand."""
    schema = json.loads((_DATASETS / 'schema_corpus.json').read_text(encoding='utf-8'))
    template = json.loads((_DATASETS / 'corpus_template.json').read_text(encoding='utf-8'))
    problems = _mirror_mismatches(schema, template, 'root', set())
    assert problems == [], '\n'.join(problems)


# groundtruth_template.json abbreviates one node on purpose: a question
# label's own declaration (question_metadata_fields's entries) is the same
# twelve-key table label_fields uses in the corpus schema, and the template
# says so in its own "_note" ("Same shape as the corpus's label_fields")
# rather than restating the whole table a second time. That single,
# documented abbreviation is named here so it cannot silently grow a second
# one beside it.
_GROUNDTRUTH_MIRROR_EXCEPTIONS = {
    'root.groundtruth_dataset_metadata.question_metadata_fields',
}


def test_groundtruth_template_mirrors_groundtruth_schema_key_for_key():
    # this is a convention test
    """The same pin as the corpus pair, for the ground-truth schema and its
    template — including the D7 sentence that names a question label as a
    panel filter, which is prose in x-raglab-uses and therefore outside what
    this structural check walks (see the docstring on _descend_schema)."""
    schema = json.loads((_DATASETS / 'schema_groundtruth.json').read_text(encoding='utf-8'))
    template = json.loads((_DATASETS / 'groundtruth_template.json').read_text(encoding='utf-8'))
    problems = _mirror_mismatches(schema, template, 'root', _GROUNDTRUTH_MIRROR_EXCEPTIONS)
    assert problems == [], '\n'.join(problems)
