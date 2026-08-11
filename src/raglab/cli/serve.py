"""The two front doors, and the ports they bind.

The numbers live here rather than on a command line because there is no longer a
package.json to hold them and no Node test to read them back out of one. One
module owns them, `tests/test_ports.py` asserts them, and `lab.py` imports rather
than repeats them.
"""
import uvicorn

PANEL_PORT = 9002
INSPECTOR_PORT = 9003

# Lodestar's allocation on this machine, copied on 2026-08-11 when the lab moved
# out. Copied, not read: that repository is no longer a dependency, so nothing
# here can notice if these change. See test_ports.py for what that costs.
RESERVED = {
    3000: 'the Lodestar board',
    3001: 'the Lodestar test board',
    8001: 'the external vectordb-lab stack',
    8002: 'the external vectordb-lab stack',
    8003: "Lodestar's Chroma",
    8004: "Lodestar's test Chroma",
    9000: 'the Lodestar brain',
    9001: 'the paired test brain',
}


def panel() -> None:
    """The lab's panel. `uv run --extra local-embeddings raglab`."""
    uvicorn.run('raglab.server:app', port=PANEL_PORT)


def inspector() -> None:
    """The read-only Inspector. `uv run raglab-inspector`."""
    uvicorn.run('raglab.inspector:app', port=INSPECTOR_PORT)
