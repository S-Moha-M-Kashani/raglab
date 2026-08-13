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


def test_env_example_documents_every_variable_the_code_reads():
    """Scans `src/` only, not the tests — a variable a test invents for one
    assertion is not configuration anyone should be told about."""
    read = {name for path in sorted((ROOT / 'src').rglob('*.py'))
            for match in _ENV_READS.finditer(path.read_text(encoding='utf-8'))
            for name in match.groups() if name}
    documented = {match.group(1) for line in
                  (ROOT / '.env.example').read_text(encoding='utf-8').splitlines()
                  if (match := _ENV_DOCUMENTED.match(line))}
    read -= {'PATH', 'HOME'}
    assert read - documented == set(), 'read by the code, absent from .env.example'
    assert documented - read == set(), 'in .env.example, read by nothing'
