"""The lab as one served application: the panel at the root, the Inspector at /inspector.

One process and one origin, which is what the widget's conversation needs —
`localStorage` cannot cross origins, so the active experiment and the thread it
keys would be invisible to a second port.

This is its own module rather than three lines in `panel_server.py` because
`inspector_server.py` imports `Jobs` *from* `panel_server.py`: mounting from
inside the panel would close that into an import cycle. Nothing imports this
file — uvicorn names it, and `serve.panel()` is where the name is spelled.
"""
from raglab.dashboard import inspector_server, panel_server

app = panel_server.app
# Sub-apps keep their own middleware, so the Inspector's no-store header
# still rides its own responses.
app.mount('/inspector', inspector_server.app)
