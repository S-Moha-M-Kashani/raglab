"""Embedders behind the production `Embedder` protocol: offline hash baselines
plus two backends (fastembed, sentence-transformers) that load real models.
Queries and passages embed separately (`query_vectors`) since E5-style models expect distinct `query:`/`passage:` prefixes.
"""
import hashlib
import re
from dataclasses import dataclass

import numpy as np

from . import textnorm

ASCII_DIM = 128


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity is only cosine if the vectors are unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class HashEmbedder:
    """Kept verbatim as the lab's baseline. Tokeniser `[a-z0-9]+` finds no
    tokens in Farsi, so every vector is the zero vector."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), ASCII_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r'[a-z0-9]+', text.lower()):
                digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
                out[i, digest % ASCII_DIM] += 1.0
        return _normalize(out)

TOKEN_DIM = 512
CHAR_DIM = 1024
DEFAULT_FASTEMBED = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_LOCAL = 'heydariAI/persian-embeddings'

# '' means "this backend's default", the same rule '' means RAGLAB_MODEL for chat roles.
BACKENDS = ('fastembed', 'sentence-transformers')
BACKEND_DEFAULTS = {
    'fastembed': DEFAULT_FASTEMBED,
    'sentence-transformers': DEFAULT_LOCAL,
}

# "any script" means not blind to Farsi, not that it understands it — both hash embedders are lexical.
LATIN_ONLY = 'Latin script only (a–z, 0–9)'
ANY_SCRIPT = 'any script, lexical only (no meaning)'
ENGLISH_ONLY = 'English only'
MULTI_50 = 'English + Farsi (50+ languages)'
MULTI_100 = 'English + Farsi (100+ languages)'
BY_MODEL = 'English + Farsi — depends on the model below'
FARSI_TUNED = 'Farsi + English (Persian-tuned)'


def _bucket(token: str, dim: int) -> int:
    return int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16) % dim


class TokenHashEmbedder:
    """Hashed bag of normalised words with sub-linear term frequency. Lexical, but at least it sees the language."""
    dim = TOKEN_DIM
    name = 'token-hash'

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in textnorm.tokens(text):
                slot = _bucket(token, self.dim)
                counts[slot] = counts.get(slot, 0.0) + 1.0
            for slot, count in counts.items():
                out[i, slot] = 1.0 + np.log(count)   # damp repetition
        return _normalize(out)


class CharHashEmbedder:
    """Hashed character n-grams; recovers shared stems across Persian affixes that whitespace tokens treat as unrelated words."""
    dim = CHAR_DIM
    name = 'char-hash'

    def __init__(self, n: int = 4):
        self.n = n

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for gram in textnorm.char_ngrams(text, self.n):
                out[i, _bucket(gram, self.dim)] += 1.0
        return _normalize(out)


class FastEmbedMultilingual:
    """Wraps fastembed directly. `factory` allows testing without a real download."""
    name = 'fastembed'

    def __init__(self, model_name: str, batch_size: int = 64,
                 query_prefix: str = '', passage_prefix: str = '', factory=None):
        self.model = (factory or _text_embedding)(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        # Model name is part of identity, so a cache keyed by embedder name distinguishes checkpoints.
        self.name = f'fastembed:{model_name}'
        self.dim = len(next(iter(self.model.embed(['probe']))))

    def _vectors(self, texts: list[str], prefix: str) -> np.ndarray:
        payload = [prefix + text for text in texts] if prefix else list(texts)
        vectors = np.array(list(self.model.embed(payload,
                                                 batch_size=self.batch_size)),
                           dtype=np.float32)
        return _normalize(vectors)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._vectors(list(texts), self.passage_prefix)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._vectors(list(texts), self.query_prefix)


def _text_embedding(model_name: str):
    from fastembed import TextEmbedding  # optional 'semantic' extra
    return TextEmbedding(model_name)


def _sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer  # 'local-embeddings'
    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    """Any HuggingFace checkpoint, through sentence-transformers. Pooling comes
    from the model's own config rather than being guessed. `factory` allows
    testing without a real download."""
    name = 'sentence-transformers'

    def __init__(self, model_name: str, batch_size: int = 32,
                 query_prefix: str = '', passage_prefix: str = '', factory=None):
        self.model = (factory or _sentence_transformer)(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        # Model name is part of identity, so a cache keyed by embedder name distinguishes checkpoints.
        self.name = f'st:{model_name}'
        # get_embedding_dimension replaced get_sentence_embedding_dimension in
        # sentence-transformers 5; fall back for the lab's floor version (3).
        dimension = (getattr(self.model, 'get_embedding_dimension', None)
                     or self.model.get_sentence_embedding_dimension)
        self.dim = int(dimension())

    def _vectors(self, texts: list[str], prefix: str) -> np.ndarray:
        payload = [prefix + text for text in texts] if prefix else list(texts)
        vectors = self.model.encode(payload, batch_size=self.batch_size,
                                    show_progress_bar=False,
                                    convert_to_numpy=True)
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._vectors(list(texts), self.passage_prefix)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._vectors(list(texts), self.query_prefix)


def query_vectors(embedder, texts: list[str]) -> np.ndarray:
    """Uses `embed_queries` when the embedder defines an asymmetric query side, else falls back to `embed`."""
    asymmetric = getattr(embedder, 'embed_queries', None)
    if callable(asymmetric):
        return asymmetric(list(texts))
    return embedder.embed(list(texts))


# --- what each option can actually read ------------------------------------

@dataclass(frozen=True)
class EmbedderHint:
    """One line of truth about an embedder kind, shown next to the dropdown."""
    kind: str
    label: str
    languages: str
    farsi: bool          # can it represent this corpus at all?
    note: str

    def as_dict(self, available: bool) -> dict:
        return {'kind': self.kind, 'label': self.label,
                'languages': self.languages, 'farsi': self.farsi,
                'note': self.note, 'available': available}


EMBEDDER_HINTS = (
    EmbedderHint('sentence-transformers',
                 'sentence-transformers — any HuggingFace model', BY_MODEL, True,
                 'The lab default, because it is the only backend that can reach '
                 'a Persian-tuned encoder: fastembed does not serve '
                 'heydariAI/persian-embeddings. Needs the local-embeddings extra '
                 'and downloads the weights once.'),
    EmbedderHint('fastembed', 'fastembed — a real transformer', BY_MODEL, True,
                 'Embeds meaning rather than letters (0.612 recall measured), '
                 'from its own short ONNX list. Which languages it covers is '
                 'decided by the model below — pick a multilingual one or this '
                 'becomes an English model reading a Farsi diary.'),
    EmbedderHint('ascii-hash', 'ascii-hash — the reference baseline',
                 LATIN_ONLY, False,
                 'tokenises [a-z0-9]+, so a Farsi sentence has no tokens and '
                 'every vector is zero: measured 0.014 session recall here, i.e. '
                 'chance. Kept as the baseline that makes the point.'),
    EmbedderHint('token-hash', 'token-hash — hashed Persian words', ANY_SCRIPT,
                 True,
                 'Unicode-aware bag of normalised words. Sees the language but '
                 'not the meaning, and misses «نمیخوام» against «میخواستم».'),
    EmbedderHint('char-hash', 'char-hash — hashed character 4-grams', ANY_SCRIPT,
                 True,
                 'Recovers Persian stems across affixes, which is why it is the '
                 'strongest option that downloads nothing: 0.386 recall against '
                 'ascii-hash\'s 0.014. Still lexical — a paraphrase with no '
                 'shared letters is invisible to it.'),
)

_HINTS = {hint.kind: hint for hint in EMBEDDER_HINTS}


@dataclass(frozen=True)
class EmbedModel:
    """A concrete embedding model and the backend that loads it. `backend`
    decides both the cost of picking it and how availability is checked."""
    id: str
    label: str
    languages: str
    farsi: bool
    source: str          # open | closed | unknown
    dim: int
    note: str
    backend: str = 'fastembed'
    # Shown directly in the dropdown rather than behind the explainer, since
    # that is the question being asked while it is open.
    tag: str = ''
    query_prefix: str = ''
    passage_prefix: str = ''

    def as_dict(self, available: bool) -> dict:
        return {'id': self.id, 'label': self.label, 'languages': self.languages,
                'farsi': self.farsi, 'source': self.source, 'dim': self.dim,
                'note': self.note, 'available': available,
                'backend': self.backend, 'tag': self.tag,
                'query_prefix': self.query_prefix,
                'passage_prefix': self.passage_prefix}


# Includes the two English-only baselines on purpose, so the brain's shipped
# choice can be measured against a real Farsi encoder.
EMBED_MODELS = (
    # ~512-token context, so a long chunk truncates.
    EmbedModel(DEFAULT_LOCAL, 'persian-embeddings (heydariAI)', FARSI_TUNED,
               True, 'open', 1024,
               'the lab default: fine-tuned on Persian specifically rather than '
               'multilingual-by-accident, ~2.2 GB, and the cheapest real encoder '
               'to try on this corpus. Loaded through sentence-transformers, '
               'which is what its model card recommends. ~512-token context.',
               backend='sentence-transformers', tag='lab default'),
    EmbedModel('intfloat/multilingual-e5-small', 'multilingual-e5-small',
               MULTI_100, True, 'open', 384,
               'the e5 retrieval recipe at a fifth of e5-large\'s size, and its '
               'weights are already on disk here. Served through '
               'sentence-transformers — this fastembed does not carry it — and '
               'it needs its query/passage prefixes to perform, which the lab '
               'applies for you.',
               backend='sentence-transformers',
               query_prefix='query: ', passage_prefix='passage: '),
    EmbedModel(DEFAULT_FASTEMBED, 'paraphrase-multilingual-MiniLM-L12-v2',
               MULTI_50, True, 'open', 384,
               'the fastembed default: smallest multilingual option, ~120 MB, '
               '384 dims. Fine as a floor, and the weakest of the Farsi-capable '
               'models.'),
    EmbedModel('BAAI/bge-small-en-v1.5', 'bge-small-en-v1.5', ENGLISH_ONLY,
               False, 'open', 384,
               'what the brain hardwires today. Here as the baseline: it will '
               'return confident numbers that mean nothing on Farsi text.'),
    EmbedModel('sentence-transformers/all-MiniLM-L6-v2', 'all-MiniLM-L6-v2',
               ENGLISH_ONLY, False, 'open', 384,
               'the most-copied embedder on the internet, and the wrong one for '
               'this corpus.'),
)

MODEL_IDS = tuple(model.id for model in EMBED_MODELS)
_MODELS = {model.id: model for model in EMBED_MODELS}


def fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except Exception:
        return False
    return True


def fastembed_models() -> frozenset:
    """Model ids this installation of fastembed can serve; the empty set
    (never a guess) if fastembed is missing or its API changed."""
    try:
        from fastembed import TextEmbedding
        return frozenset(entry['model']
                         for entry in TextEmbedding.list_supported_models()
                         if entry.get('model'))
    except Exception:
        return frozenset()


def sentence_transformers_available() -> bool:
    """Import-checked: without the extra, every HuggingFace checkpoint is NA."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def backend_availability() -> dict:
    """Which of the two model backends can be used right now."""
    return {'fastembed': fastembed_available(),
            'sentence-transformers': sentence_transformers_available()}


def embedder_hints(settings=None) -> list[dict]:
    """One hint per embedder kind. Hash kinds are always available; the two
    model backends answer for themselves."""
    live = backend_availability()
    return [hint.as_dict(live.get(hint.kind, True)) for hint in EMBEDDER_HINTS]


def embed_model_catalogue(settings=None) -> list[dict]:
    """The embedding-model dropdown, each entry's availability checked against its own backend."""
    default_id = getattr(settings, 'fastembed_model', None) or DEFAULT_FASTEMBED
    served = fastembed_models()
    live = backend_availability()
    known = list(EMBED_MODELS)
    if default_id not in _MODELS:
        # An id set by RAGLAB_FASTEMBED_MODEL is by definition one the user wants.
        known.insert(0, EmbedModel(default_id, default_id,
                                   'coverage not recorded — check the model card',
                                   False, 'unknown', 0,
                                   'named by RAGLAB_FASTEMBED_MODEL'))

    def usable(model: EmbedModel) -> bool:
        if not live.get(model.backend):
            return False
        # fastembed also publishes which models it serves; honour that on top of the import check.
        return model.id in served if model.backend == 'fastembed' else True

    entries = [model.as_dict(usable(model)) for model in known]
    entries.sort(key=lambda entry: not entry['available'])
    # '' pins nothing: switching backend switches model without a second edit.
    return [{'id': '', 'label': 'the backend\'s own default',
             'languages': BY_MODEL, 'farsi': True, 'source': 'default',
             'dim': 0, 'available': True, 'backend': '', 'tag': '',
             'note': 'sentence-transformers → '
                     f'{_short(DEFAULT_LOCAL)}; fastembed → '
                     f'{_short(default_id)} (RAGLAB_FASTEMBED_MODEL)',
             'query_prefix': '', 'passage_prefix': ''}] + entries


def _short(model_id: str) -> str:
    return model_id.rsplit('/', 1)[-1]


def resolve_model(kind: str, settings=None, model: str = '') -> str:
    """The model a kind will load: what was pinned, else that backend's own default."""
    if kind not in BACKENDS:
        return ''
    if model:
        return model
    if kind == 'fastembed':
        # RAGLAB_FASTEMBED_MODEL keeps working exactly as before.
        return getattr(settings, 'fastembed_model', None) or DEFAULT_FASTEMBED
    return BACKEND_DEFAULTS[kind]


def language_note(kind: str, model: str = '') -> str:
    """One line for a run's notes, naming what the embedder can actually read."""
    hint = _HINTS.get(kind)
    if kind in BACKENDS and model:
        entry = _MODELS.get(model)
        coverage = entry.languages if entry else 'coverage not recorded'
        return f'embedder {kind} on {model}: {coverage}'
    coverage = hint.languages if hint else 'coverage not recorded'
    return f'embedder {kind}: {coverage}'


def make_embedder(kind: str, settings=None, model: str = ''):
    if kind == 'ascii-hash':
        embedder = HashEmbedder()
        embedder.name = 'ascii-hash'          # type: ignore[attr-defined]
        embedder.dim = 128                    # type: ignore[attr-defined]
        return embedder
    if kind == 'token-hash':
        return TokenHashEmbedder()
    if kind == 'char-hash':
        return CharHashEmbedder()
    if kind in BACKENDS:
        name = resolve_model(kind, settings, model)
        entry = _MODELS.get(name)
        prefixes = {'query_prefix': entry.query_prefix if entry else '',
                    'passage_prefix': entry.passage_prefix if entry else ''}
        if kind == 'fastembed':
            return FastEmbedMultilingual(name, **prefixes)
        return SentenceTransformerEmbedder(name, **prefixes)
    raise ValueError(f'unknown lab embedder: {kind!r}')
