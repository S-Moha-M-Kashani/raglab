"""Reading a model's verdict, in the direction that costs work."""
import re

_YES = ('yes', 'بله', 'آری')
_NO = ('no', 'خیر', 'نه')
_NUMBER = re.compile(r'^\s*(?:score|verdict|rating)?\s*[:=]?\s*'
                     r'(\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?')


def verdict(text: str) -> float | None:
    """A model's verdict in [0,1], or None when it gave none. Unlike
    `retrieval.llm_scores`'s 0.5, never defaulted to a number: a single verdict
    deciding whether the loop stops cannot be split that way."""
    if not text:
        return None
    head = text.strip().lower()
    if any(head.startswith(word) for word in _YES):
        return 1.0
    if any(head.startswith(word) for word in _NO):
        return 0.0
    match = _NUMBER.match(head)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):                      # '8/10'
        scale = float(match.group(2)) or 1.0
        value = value / scale
    elif value > 1.0:                       # '8' on the 0-10 scale the prompts ask for
        value = value / 10.0
    return max(0.0, min(1.0, value))
