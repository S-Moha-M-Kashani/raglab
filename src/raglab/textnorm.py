"""Persian text normalisation and tokenisation for BM25, the lexical reranker,
and the time filter — the same tokeniser Lodestar's retrieval uses.
Two texts a reader would call identical must tokenise identically.
"""
# Vendored from lodestar_brain/textnorm.py at 057a755 — copied, not imported,
# so it can drift from Lodestar's tokeniser without anything noticing.

import re
import unicodedata

ZWNJ = '‌'  # نیم‌فاصله — a joiner, not a word boundary

# Arabic letterforms folded to their Persian equivalents (same sound, different codepoint).
_LETTER_FOLD = str.maketrans({
    'ي': 'ی', 'ك': 'ک', 'ﻯ': 'ی', 'ﻱ': 'ی', 'ٱ': 'ا', 'أ': 'ا', 'إ': 'ا',
    'آ': 'ا', 'ؤ': 'و', 'ئ': 'ی', 'ة': 'ه', 'ۀ': 'ه', 'ء': '',
})
# Harakat (fatha, damma, sukun, tanwin, ...) plus tatweel: decoration only.
_MARKS = re.compile('[ً-ْٓ-ٰٕـ]')
# Persian ۰-۹ and Arabic-Indic ٠-٩ → ASCII, so «۱۴۰۵» and "1405" are one token.
_DIGIT_FOLD = str.maketrans(
    {chr(0x06F0 + i): str(i) for i in range(10)} |
    {chr(0x0660 + i): str(i) for i in range(10)})
_WORD = re.compile(r'[^\W_]+(?:‌[^\W_]+)*', re.UNICODE)

# Deliberately short: an aggressive stop list would eat negations and pronouns
# that carry meaning in a diary («نمیدونم», «خودم»).
STOPWORDS = frozenset("""
که را از به با در و یا هم این اون آن یه یک بر بی تا هر چه چی می نمی است هست بود
بودم بوده شد شده می‌شه میشه کرد کردم کرده برای روی مثل ولی اما پس یعنی البته
دیگه دیگر خیلی خب حالا الان هنوز باید شاید انقدر اینقدر همه چون تو من ما شما اونا
""".split())


def normalize(text: str) -> str:
    """Idempotent: normalize(normalize(x)) == normalize(x)."""
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(_LETTER_FOLD).translate(_DIGIT_FOLD)
    text = _MARKS.sub('', text)
    return re.sub(r'[ \t ]+', ' ', text).strip()


def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    """Half-spaced compounds are emitted whole *and* split, since both forms
    appear in the corpus: «می‌خوام» must match «می خوام»."""
    out: list[str] = []
    for match in _WORD.findall(normalize(text).lower()):
        parts = match.split(ZWNJ)
        candidates = [match.replace(ZWNJ, '')] + (parts if len(parts) > 1 else [])
        for token in candidates:
            if len(token) < 2:
                continue
            if drop_stopwords and token in STOPWORDS:
                continue
            out.append(token)
    return out


def char_ngrams(text: str, n: int = 4) -> list[str]:
    """Recovers morphological overlap word tokens miss: Persian is heavily
    affixed (میخواستم / نمیخواستم / بخوام share a stem)."""
    flat = re.sub(r'\s+', ' ', normalize(text).lower().replace(ZWNJ, ''))
    if len(flat) <= n:
        return [flat] if flat else []
    return [flat[i:i + n] for i in range(len(flat) - n + 1)]


def sentences(text: str) -> list[str]:
    """Spoken-diary text is run-on, so «و بعدش» and «خلاصه» count as boundaries
    alongside . ! ? ؟ and newlines."""
    text = normalize(text)
    parts = re.split(r'(?<=[.!?؟…])\s+|\n+|(?:\s+(?:و بعدش|خلاصه|راستی)\s+)', text)
    return [p.strip() for p in parts if p and p.strip()]
