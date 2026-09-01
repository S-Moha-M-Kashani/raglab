"""The lab's front door, and the port it binds.

The number lives here rather than on a command line because there is no longer a
package.json to hold it and no Node test to read it back out of one. One
module owns it, `test_conventions.py` asserts it, and `lab.py` imports rather
than repeats it.
"""
import uvicorn

PANEL_PORT = 9002

# Ports already bound on the development machine by the production assistant
# this lab was extracted from and its neighbours, copied by hand on 2026-08-11
# when the lab moved out. Copied, not read: that system is no longer a
# dependency, so nothing here can notice if these change. See
# test_conventions.py for what that costs.
RESERVED = {
    3000: "the production assistant's board",
    3001: "the production assistant's test board",
    8001: 'the external vectordb-lab stack',
    8002: 'the external vectordb-lab stack',
    8003: "the production assistant's Chroma",
    8004: "the production assistant's test Chroma",
    9000: "the production assistant's brain service",
    9001: 'the paired test brain',
}


def panel() -> None:
    """The lab: the panel at the root, the Inspector at /inspector.
    `uv run --extra local-embeddings raglab`."""
    uvicorn.run('raglab.dashboard.served_lab:app', port=PANEL_PORT)
