"""The dependency that was cut, pinned as an absence."""
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
TESTS = Path(__file__).resolve().parent


def test_nothing_imports_lodestar_brain():
    """Prose mentioning lodestar_brain is fine; only a line the interpreter
    follows is not."""
    offenders = []
    for path in [*SRC.rglob('*.py'), *TESTS.rglob('*.py')]:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(('import lodestar_brain',
                                    'from lodestar_brain')):
                offenders.append(f'{path.name}:{number}')
    assert not offenders, f'lodestar_brain imported at {offenders}'


def test_no_path_is_hardcoded_to_one_machine():
    """This file is excluded from its own scan, because a test that scans for
    a hardcoded-path string necessarily contains one."""
    offenders = [path.name for path in [*SRC.rglob('*.py'), *TESTS.rglob('*.py')]
                 if path.resolve() != Path(__file__).resolve()
                 and any(quote + '/Users/' in path.read_text(encoding='utf-8')
                         or quote + '/home/' in path.read_text(encoding='utf-8')
                         for quote in ('"', "'"))]
    assert not offenders, f'a home directory is hardcoded in {offenders}'
