"""The OpenRouter key, held for the life of this process and written nowhere.

The judged metrics are the four that choose the architecture, and reaching them
used to mean editing `.env` and restarting the lab — two steps at exactly the
moment you have discovered that the run you just watched could not be judged.
The panel takes the key instead, and this module is where it sits.

**In memory, and only in memory.** Nothing here opens a file, sets an
environment variable or logs a string, and `test_credentials.py` reads this
source to hold that. The three durable things this lab writes — a `.runs/` JSON
file, a `raglab.db` row, the terminal — are the account of the work and are meant
to be readable and shareable; a credential in any of them would be durable in
exactly the way a credential must not be. What survives a restart is therefore
nothing, which is the honest cost of the convenience: the environment variable
stays the way to start a lab that already has a key.

**The panel's key wins over the environment's, and clearing gives it back.**
"Clear" means "forget what I typed", never "unset the key" — a lab started with
`OPENROUTER_API_KEY` in its shell must end up exactly as it started.

**What is reported is set-ness, a masked tail, and where it came from** — never
the key. The tail answers the only question a panel has to be able to answer
about a credential ("is this the one I meant?"), and where it came from decides
who can remove it.
"""
from dataclasses import replace

from . import models
from .config import LabSettings

# The shortest thing that could be an OpenRouter key. Not a pattern: keys are
# issued by someone else and their shape is not ours to legislate — a lab
# pointed at a compatible proxy through OPENROUTER_BASE_URL may hold anything.
# What is checked is what a *mistake* looks like: an empty box, a pasted
# fragment, a value with whitespace in it from a copied shell line.
MIN_LENGTH = 20

_key: str = ''


def set_key(key: str) -> str:
    """Hold `key` for the rest of this process. Returns its masked hint.

    Raises `ValueError` with a sentence the panel can show. A key accepted
    silently is a run that fails much later at its first model call, with an
    error naming the model rather than the credential."""
    key = (key or '').strip()
    if not key:
        return _refuse('the key is empty')
    if any(character.isspace() for character in key):
        return _refuse('the key contains a space — a whole shell line may have '
                       'been pasted rather than the key itself')
    if len(key) < MIN_LENGTH:
        return _refuse(f'the key is {len(key)} characters, shorter than any '
                       f'issued one ({MIN_LENGTH} at least) — it looks like a '
                       'fragment')
    global _key
    _key = key
    # The catalogue is verified against OpenRouter's own model list and cached
    # per base url, and with no key that verification returns the empty set —
    # "nothing is available". Left in place, entering a key would leave every
    # remote model reading NA until the lab was restarted, which is the restart
    # this whole feature exists to remove.
    models.forget_live()
    return hint(_key)


def _refuse(reason: str) -> str:
    raise ValueError(f'that is not a usable API key: {reason}')


def clear() -> None:
    """Forget the key this process was given. The environment's own key, if
    there is one, is in force again immediately."""
    global _key
    _key = ''
    models.forget_live()


def held() -> bool:
    return bool(_key)


def apply(settings: LabSettings) -> LabSettings:
    """`settings` as they are, or carrying the key the panel supplied.

    Every route builds its run settings through here rather than reading the
    boot settings directly, so a key entered a second ago is in force without a
    restart — and so there is one place that decides which key wins."""
    if not _key:
        return settings
    return replace(settings, openrouter_api_key=_key)


def hint(key: str) -> str:
    """`sk-or…cdef` — enough to recognise a key, never enough to use one."""
    key = key or ''
    if len(key) <= 8:
        return '…'
    return f'{key[:5]}…{key[-4:]}'


def state(settings: LabSettings) -> dict:
    """What the panel is told: whether a key is in force, a masked tail, and
    which of the two ways in put it there."""
    if _key:
        return {'set': True, 'source': 'panel', 'hint': hint(_key)}
    if settings.openrouter_api_key:
        return {'set': True, 'source': 'environment',
                'hint': hint(settings.openrouter_api_key)}
    return {'set': False, 'source': '', 'hint': ''}
