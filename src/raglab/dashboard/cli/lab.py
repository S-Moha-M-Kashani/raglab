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

from raglab.dashboard.cli import serve

# The repo root, so the suite runs whatever directory the command was typed in:
# an entry point can be invoked from anywhere, and `tests/` cannot.
ROOT = Path(__file__).resolve().parents[4]

USAGE = f"""Usage: uv run raglab-lab [options]

Runs the lab's tests, then starts the lab on :{serve.PANEL_PORT}.

  --no-test    start the lab without running the suite first
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
        # The whole suite, not just test_raglab.py: the 2026-08
        # restructuring scattered that file's contents across
        # test_conventions.py, test_primitives.py, test_store_index.py and
        # others, so the narrow file no longer answers "is the lab sound
        # enough to look at?" on its own — it would miss a broken conftest
        # guard, a broken panel convention, or a broken Inspector route.
        # Measured: 542 cases, 11.2 s, so there is no longer a speed reason
        # to run less than everything.
        target = 'src/raglab'
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

