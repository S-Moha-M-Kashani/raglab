"""The OpenRouter key the panel types.

Held in process memory and nowhere else — no file, no environment variable, no
log, no artifact — so it never reaches a run, a ledger row or an archive. The
widget is reset on either route because it caches a client built with whatever
key was in force when it last answered.
"""
from fastapi import HTTPException

from raglab.agents import widget
from raglab.llm_backends import openrouter_key_memory as credentials


def register(app, context) -> None:
    settings_now = context.settings_now

    @app.post('/api/credentials')
    def set_credentials(payload: dict):
        """Take the OpenRouter key from the panel, held for this process only, never recorded on a run."""
        try:
            credentials.set_key(payload.get('api_key') or '')
        except ValueError as error:
            raise HTTPException(400, str(error))
        widget.reset()
        return credentials.state(settings_now())

    @app.delete('/api/credentials')
    def clear_credentials():
        """Forget the key this panel supplied; never unsets the environment's own."""
        credentials.clear()
        widget.reset()
        return credentials.state(settings_now())
