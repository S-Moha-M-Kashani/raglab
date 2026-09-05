"""The lab as one served application: the panel at the root, the Inspector at /inspector.

One process and one origin, which is what the widget's conversation needs —
`localStorage` cannot cross origins, so the active experiment and the thread it
keys would be invisible to a second port.

One process is also why the Inspector is handed the panel's own reads here
rather than asking for them over a socket: the two halves are the same
process, and a loopback request between them would spend a worker thread on
each side of a call that is one attribute away — and could time out, so a lab
that is demonstrably running could report itself down.

This is its own module rather than three lines in `panel_server.py` because
`inspector_server.py` imports `Jobs` *from* `panel_server.py`: mounting from
inside the panel would close that into an import cycle. Nothing imports this
file — uvicorn names it, and `serve.panel()` is where the name is spelled.
"""
import os

from raglab.dashboard import inspector_server, panel_server
from raglab.dashboard.service_route_plumbing import LabAccess


def inspector_lab() -> LabAccess | None:
    """The panel's own reads, or nothing when the Inspector is pointed elsewhere.

    `RAGLAB_INSPECTOR_LAB_URL` is the only way the Inspector's lab is ever a
    lab this process does not contain, and it keeps that meaning: naming one
    leaves the Inspector to build its own HTTP access to the lab named, exactly
    as a standalone Inspector does.

    *When* the variable is read has changed: the mount below calls this once,
    as this module is imported, so the environment decides which mode the
    Inspector is in at start-up and a variable set afterwards changes nothing.
    The old forwarders read it on every request. That matters to nobody who
    serves the lab — the process is started with its environment — but it is a
    real difference, and a test that sets the variable must build the app after
    setting it."""
    if os.environ.get(inspector_server.LAB_URL_ENV):
        return None
    return panel_server.app.state.lab_access


app = panel_server.app
# Sub-apps keep their own middleware, so the Inspector's no-store header
# still rides its own responses.
app.mount('/inspector',
          inspector_server.create_inspector_app(lab=inspector_lab()))
