"""The OpenRouter key the panel can set, held only in this process's memory.

Nothing here writes the key to a file, an environment variable, or a log —
`test_credentials.py` reads this module's source to hold that.
"""
from dataclasses import replace

from . import models
from .config import LabSettings

# A length floor, not a pattern: keys are issued by someone else and a proxy
# behind OPENROUTER_BASE_URL may accept a different shape, so this only catches
# an empty box, a pasted fragment, or a copied shell line with whitespace in it.
MIN_LENGTH = 20

_key: str = ''


def set_key(key: str) -> str:
    """Hold `key` for this process and return its masked hint, or raise ValueError with a reason to show."""
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
    # The verified-model cache is keyed per base url and holds the empty set with
    # no key; drop it so a newly-set key is reflected without restarting the lab.
    models.forget_live()
    return hint(_key)


def _refuse(reason: str) -> str:
    raise ValueError(f'that is not a usable API key: {reason}')


def clear() -> None:
    """Forget the key this process was given; the environment's own key, if any, is in force again."""
    global _key
    _key = ''
    models.forget_live()


def held() -> bool:
    return bool(_key)


def apply(settings: LabSettings) -> LabSettings:
    """`settings`, carrying the panel's key if one is held — the one place that decides which key wins."""
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
    """Whether a key is in force, a masked tail, and which of the two ways in put it there — never the key."""
    if _key:
        return {'set': True, 'source': 'panel', 'hint': hint(_key)}
    if settings.openrouter_api_key:
        return {'set': True, 'source': 'environment',
                'hint': hint(settings.openrouter_api_key)}
    return {'set': False, 'source': '', 'hint': ''}
