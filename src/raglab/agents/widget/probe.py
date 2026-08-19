"""The EN-Farsi alignment probe: a plain function over a real encoder.

`measure` is the measurement itself — pair cosine, mixed-pool retrieval, a
verdict — kept free of the tool wrapper (tools.py holds that) so a future
panel route could call it directly. It still lives inside the widget package
because the widget is its only caller today; promoting it beside embedding_backends.py
is a one-file move once a second caller exists.
"""
import json

from raglab.configuration.env_settings import ROOT

# The bilingual probe's default instrument: twelve diary-like sentence pairs,
# kept as a fixture rather than code — the skills-folder rule, so the pairs
# can change without touching Python — and fixed between calls so two
# measurements stay comparable.
PAIRS_FILE = ROOT / 'fixtures' / 'bilingual_probe_pairs.json'

# The shape a pairs payload must have, quoted by the tool's refusal and
# stated to the model by its YAML prompt — the config.HELP['run.dataset-file']
# pattern: the contract is announced where the data goes in.
PAIRS_SHAPE = ('a JSON list of at least two [english, farsi] pairs, e.g. '
               '[["I slept badly.", "بد خوابیدم."], '
               '["It rained.", "باران آمد."]]')


def _read_pairs(raw: str = ''):
    """The probe's pairs: the caller's JSON, or the bundled fixture. Returns
    (pairs, problem); a problem is a stated string the model can relay and
    correct itself against, never an exception."""
    source = raw.strip()
    try:
        data = json.loads(source) if source else json.loads(
            PAIRS_FILE.read_text(encoding='utf-8'))
    except Exception as error:
        return None, f'unreadable pairs ({error}); expected {PAIRS_SHAPE}'
    well_formed = (isinstance(data, list) and len(data) >= 2 and all(
        isinstance(pair, list) and len(pair) == 2
        and all(isinstance(text, str) and text.strip() for text in pair)
        for pair in data))
    if not well_formed:
        return None, f'malformed pairs; expected {PAIRS_SHAPE}'
    return [(en, fa) for en, fa in data], None


# name -> loaded encoder; the _AGENTS pattern — a 2 GB checkpoint must not be
# reloaded per question, and the cache dies with the process. Bounded,
# unlike _AGENTS, because the keys come from a model or a user naming any
# HuggingFace checkpoint: agents are a few kilobytes and four names, while
# a handful of encoder probes at gigabytes each would exhaust the lab
# process's memory.
_ENCODERS: dict = {}
MAX_ENCODERS = 2


def _load_encoder(name: str):
    """Lazy on both axes: the import needs the local-embeddings extra, and
    neither it nor the checkpoint may cost anything at module import."""
    from sentence_transformers import SentenceTransformer
    if name not in _ENCODERS:
        while len(_ENCODERS) >= MAX_ENCODERS:
            _ENCODERS.pop(next(iter(_ENCODERS)))
        _ENCODERS[name] = SentenceTransformer(name)
    return _ENCODERS[name]


def measure(model_name: str = '', pairs: str = '') -> str:
    """The probe over a real encoder — pair cosine, mixed-pool retrieval, a
    verdict, all in one stated sentence the model (or a person) can relay."""
    name = model_name.strip() or 'heydariAI/persian-embeddings'
    items, problem = _read_pairs(pairs)
    if problem:
        # Stated refusals the model can relay and correct against, not a
        # dead loop: the shape, the extra or the checkpoint — whichever is
        # missing is the whole answer.
        return f'cannot measure: {problem}'
    english = [en for en, _ in items]
    farsi = [fa for _, fa in items]
    try:
        import numpy as np
        encoder = _load_encoder(name)
        vectors = np.asarray(encoder.encode(english + farsi,
                                            normalize_embeddings=True))
    except Exception as error:
        return f'cannot measure {name}: {error}'
    n = len(items)
    sims = vectors[:n] @ vectors[n:].T
    pairs = np.diag(sims)
    mismatched = sims[~np.eye(n, dtype=bool)]
    pool = vectors @ vectors.T
    np.fill_diagonal(pool, -1.0)          # a query may not retrieve itself
    en_wins = int((pool[:n].argmax(axis=1) == np.arange(n) + n).sum())
    fa_wins = int((pool[n:].argmax(axis=1) == np.arange(n)).sum())
    separation = float(pairs.mean() - mismatched.mean())
    aligned = en_wins == n and fa_wins == n and separation >= 0.3
    verdict = 'aligned' if aligned else 'weak or no alignment'
    return (f'{name}, measured now on {n} English-Farsi sentence pairs: '
            f'translation pairs mean cosine {pairs.mean():.3f} '
            f'(min {pairs.min():.3f}), mismatched pairs mean '
            f'{mismatched.mean():.3f} (max {mismatched.max():.3f}); in a '
            f'mixed-language pool the English query finds its own Farsi '
            f'translation {en_wins}/{n} times and the Farsi query its '
            f'English one {fa_wins}/{n}. Verdict: {verdict}. This is a '
            f'sentence-scale probe on {n} short sentence pairs — '
            f'corpus-scale retrieval can still differ.')
