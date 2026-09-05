"""The lab's named text normalisers, and which one a corpus gets by default.

A normaliser is what the lexical stages tokenise with — BM25, the hash
embedders, the hierarchy's rare-word edges and summaries, a drift stage's
markers. Two are offered and the corpus's declared `language` picks between
them unless `IndexConfig.normalizer` names one: `persian`
(`farsi_text_normalizer`, which folds Arabic letterforms and digits and drops
a Persian stop list) and `neutral`, which normalises Unicode and folds
nothing. Whether Persian folding helps a German corpus is a measurable
question, so the table below is a default rather than a lock.
"""
import unicodedata

from raglab.rag_components.retrieval import farsi_text_normalizer as persian


class Neutral:
    """Unicode-normalising, language-blind: no letterform folds, no digit
    folds, no stop list. A token is a run of letters and digits."""

    @staticmethod
    def normalize(text: str) -> str:
        return ' '.join(unicodedata.normalize('NFKC', text).split())

    @staticmethod
    def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
        out, word = [], []
        for ch in Neutral.normalize(text).lower() + ' ':
            if unicodedata.category(ch)[0] in 'LN' or ch == persian.ZWNJ:
                word.append(ch)
            elif word:
                token = ''.join(word).replace(persian.ZWNJ, '')
                if len(token) >= 2:
                    out.append(token)
                word = []
        return out

    @staticmethod
    def char_ngrams(text: str, n: int = 4) -> list[str]:
        flat = Neutral.normalize(text).lower().replace(persian.ZWNJ, '')
        if len(flat) <= n:
            return [flat] if flat else []
        return [flat[i:i + n] for i in range(len(flat) - n + 1)]

    @staticmethod
    def sentences(text: str) -> list[str]:
        """Cut after `.`, `!`, `?`, `؟` or `…` followed by whitespace, and at newlines."""
        out, current = [], []
        text = unicodedata.normalize('NFKC', text)
        for i, ch in enumerate(text):
            if ch == '\n':
                out.append(''.join(current))
                current = []
                continue
            current.append(ch)
            after = text[i + 1] if i + 1 < len(text) else ' '
            if ch in '.!?؟…' and after.isspace():
                out.append(''.join(current))
                current = []
        out.append(''.join(current))
        return [p.strip() for p in out if p.strip()]


NEUTRAL = Neutral()
NORMALIZERS = {'persian': persian, 'neutral': NEUTRAL}
# What an unnamed normaliser means for a declared language; every language
# without an entry is neutral.
BY_LANGUAGE = {'fa': 'persian'}


def name_for(configured: str, language: str) -> str:
    return configured or BY_LANGUAGE.get(language, 'neutral')


def resolve(configured: str, language: str):
    """The normaliser a build runs — the named one, or the declared language's
    default. An unknown name is refused rather than replaced: a row must never
    lie about what produced it."""
    name = name_for(configured, language)
    if name not in NORMALIZERS:
        raise ValueError(f'unknown normalizer {name!r} — this installation '
                         f'provides {", ".join(NORMALIZERS)}')
    return NORMALIZERS[name]
