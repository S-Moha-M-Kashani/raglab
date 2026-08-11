"""One command: prove the lab, then run it.

The lab's whole claim is that retrieval choices here were decided by
measurement. That claim is worth exactly as much as the suite behind it, so this
runner refuses to open the panel on code whose tests do not pass: a green
terminal above the URL is the point, not a convenience.

Was `scripts/lab.mjs` in the Lodestar repository, where it also had to avoid
respelling a uvicorn line carrying four version pins and an extra. Those are in
`uv.lock` and `serve.py` now; what remains worth not repeating is the port.
"""
import socket
import subprocess
import sys
from pathlib import Path

from . import serve

# The repo root, so the suite runs whatever directory the command was typed in:
# an entry point can be invoked from anywhere, and `tests/` cannot.
ROOT = Path(__file__).resolve().parents[3]

USAGE = f"""Usage: uv run raglab-lab [options]

Runs the lab's tests, then starts the lab on :{serve.PANEL_PORT}.

  --no-test    start the lab without running the suite first
  --all        run the whole suite, not just the lab's own file
  --test-only  run the suite and stop
"""


def _free(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(('127.0.0.1', port)) != 0


def main() -> None:
    argv = sys.argv[1:]
    if '--help' in argv or '-h' in argv:
        print(USAGE)
        return

    if '--no-test' not in argv:
        # The lab's own file by default. The full suite is a minute rather than
        # seconds, and the question this runner answers — "is the lab sound
        # enough to look at?" — is answered by the lab's tests.
        target = 'tests' if '--all' in argv else 'tests/test_raglab.py'
        print(f'\n\033[1m▸ {target}\033[0m')
        status = subprocess.run([sys.executable, '-m', 'pytest', target, '-q'],
                                cwd=ROOT).returncode
        if status != 0:
            print('\n\033[31m✗ tests failed — not starting the lab.\033[0m')
            print('  Start it anyway with:  uv run raglab-lab --no-test\n')
            sys.exit(status)
        print('\033[32m✓ tests pass\033[0m')
    if '--test-only' in argv:
        return

    if not _free(serve.PANEL_PORT):
        # Not an error worth failing on: the usual cause is a lab already
        # running, and the useful thing to print is where it is.
        print(f'\n\033[33m:{serve.PANEL_PORT} is already in use — a lab is '
              f'likely already up.\033[0m')
        print(f'  Panel:  http://localhost:{serve.PANEL_PORT}/\n')
        return

    print('\n\033[1m▸ starting the RAG lab\033[0m')
    print(f'  Panel:  http://localhost:{serve.PANEL_PORT}/')
    print('  First build downloads the embedding model (~2.2 GB) on first')
    print('  retrieval, not at boot.\n')
    serve.panel()

