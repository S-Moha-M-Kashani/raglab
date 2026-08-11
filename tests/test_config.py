"""`.env.example` against the code, in both directions."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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


# This is a configuration invariant.
def test_env_example_documents_every_variable_the_code_reads():
    """`.env.example` is the only list of what this lab can be configured with,
    so a variable missing from it is undiscoverable and one lingering in it after
    the code stopped reading it is a lie. Both directions are asserted for that
    reason — and the second direction is what caught `RAGLAB_URL` coming across
    in the move, a variable only Lodestar's Node proxy ever read.

    Scanned: the package. Not the tests — those set variables to exercise the
    readers above, and a value invented for one assertion is not configuration
    anyone should be told about."""
    read = {name for path in sorted((ROOT / 'src').rglob('*.py'))
            for match in _ENV_READS.finditer(path.read_text(encoding='utf-8'))
            for name in match.groups() if name}
    documented = {match.group(1) for line in
                  (ROOT / '.env.example').read_text(encoding='utf-8').splitlines()
                  if (match := _ENV_DOCUMENTED.match(line))}
    read -= {'PATH', 'HOME'}
    assert read - documented == set(), 'read by the code, absent from .env.example'
    assert documented - read == set(), 'in .env.example, read by nothing'
