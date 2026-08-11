"""The dependency that was cut, pinned as an absence."""
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
TESTS = Path(__file__).resolve().parent


# This is a configuration invariant.
def test_nothing_imports_lodestar_brain():
    """The lab vendored the tokeniser and froze the preset precisely so this
    repository stands alone. An import is easy to add back by habit — from a
    docstring's example, or by copying a line out of the old repository — and it
    would work on this machine, which is the problem. The invariant is the
    absence, so it is asserted rather than remembered.

    Prose is allowed and deliberate: several modules explain what they were
    copied from and when. What must not exist is a line the interpreter follows."""
    offenders = []
    for path in [*SRC.rglob('*.py'), *TESTS.rglob('*.py')]:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(('import lodestar_brain',
                                    'from lodestar_brain')):
                offenders.append(f'{path.name}:{number}')
    assert not offenders, f'lodestar_brain imported at {offenders}'


# This is a configuration invariant.
def test_no_path_is_hardcoded_to_one_machine():
    """The suite used to read Lodestar's `app.js`, and the ledger wrote into that
    repository's `databases/test/`. Both were correct then; either one rewritten
    as an absolute path during the move would pass here and fail on any machine
    where the other checkout does not sit in the same place.

    A literal home directory is the shape that failure takes, and it is the only
    part of "reaches outside the repo" a text scan can judge — a marker like
    `brain/tests` appears in this project's own provenance notes, so scanning for
    it flags prose. The real proof that nothing reaches outside is the suite
    passing from a fresh clone, which no single assertion can stand in for.

    This file is skipped, because a test that scans for a string necessarily
    contains it. The cost is that this one file is unguarded; the alternative is
    assembling the needle from fragments, which hides what is being looked for
    from the person reading the test."""
    offenders = [path.name for path in [*SRC.rglob('*.py'), *TESTS.rglob('*.py')]
                 if path.resolve() != Path(__file__).resolve()
                 and any(quote + '/Users/' in path.read_text(encoding='utf-8')
                         or quote + '/home/' in path.read_text(encoding='utf-8')
                         for quote in ('"', "'"))]
    assert not offenders, f'a home directory is hardcoded in {offenders}'
