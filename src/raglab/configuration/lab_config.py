"""Lab settings and the IndexConfig/RetrievalConfig/GenerationConfig objects driving the pipeline.

Only IndexConfig enters the index fingerprint; the other three sweep for free
against a build already made. There is deliberately no vector-storage setting
here — the index lives in process memory (`store.MemoryVectors`), so nothing
here can be pointed at real chat memory by a typo or a stale shell.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# Re-exported so every existing `from .config import ROOT` (etc.) keeps
# working: these moved to env_settings.py, which is where `load_env_file` reads
# `ROOT / '.env'` from.
from .env_settings import (ROOT, RUNS_DIR, LLM_PROVIDERS, PROVIDER_MODELS,  # noqa: F401
                       load_env_file, LabSettings, load_lab_settings,
                       settings_for_provider)

# Re-exported: the option tuples live in option_vocabularies.py — the lab's closed
# vocabularies — because knob_dependencies.py needs a dozen of them and must not
# import .config to get them (that was the circular import this replaced).
# Every one of the nineteen existing importers still reaches them as
# `config.CHUNKERS` etc.
from .option_vocabularies import (CHUNKERS, CHAR_SIZED_CHUNKERS, OVERLAP_CHUNKERS,  # noqa: F401
                      EMBEDDERS, MODEL_EMBEDDERS, HIERARCHIES,
                      GRAPH_HIERARCHIES, CLUSTER_HIERARCHIES,
                      LEVELLED_HIERARCHIES, TUNED_HIERARCHIES, GRAPH_SOURCES,
                      KNN_SOURCES, SUMMARIZERS, RETRIEVERS, SUMMARY_SCOPES,
                      RERANKERS, GRADERS, ANSWERERS)

# Re-exported: DEPENDENCIES lives in knob_dependencies.py, which now reads the
# option tuples from option_vocabularies.py rather than from here, so this import carries
# no ordering requirement of its own — kept in the same spot as before.
from .knob_dependencies import DEPENDENCIES, dependency_state  # noqa: F401,E402


@dataclass(frozen=True)
class Step:
    """One of the stages a knob or a model can belong to; served rather than reinvented per frontend.

    Only the *meaning* lives here — the ink for each step is a CSS token.
    `label` titles a panel of knobs, `short` tags a model group where a whole clause would not fit."""
    key: str        # matches the config group, and the CSS ink token
    short: str      # 'Index'
    label: str      # 'Index — what gets stored'
    note: str       # when it runs, and therefore what a change there costs


STEPS = (
    Step('index', 'Index', 'Index — what gets stored',
         'Runs once per corpus. It decides what can ever be found, so a change '
         'here rebuilds the collection and invalidates the numbers below it.'),
    Step('retrieval', 'Retrieval', 'Retrieval & ranking',
         'Runs on every question against an index that already exists — which '
         'is why these are the knobs worth sweeping first.'),
    Step('generation', 'Generation', 'Generation & scoring',
         'Turns the retrieved contexts into a Farsi answer, refuses when the '
         'diary is silent, and grades what it wrote.'),
)


@dataclass(frozen=True)
class IndexConfig:
    """What gets indexed. Its fingerprint names the in-memory index."""
    # Which corpus. '' is the built-in Farsi diary; anything else names a file
    # under fixtures/corpus_groundtruth_datasets/ or .datasets/ (see raglab/dataset_import_contract.py).
    # It belongs here rather than beside the run controls because it decides what
    # is stored: an index built over one corpus must never be handed a question
    # from another, and the fingerprint is what makes that impossible.
    dataset: str = ''
    chunker: str = 'semantic-drift'
    chunk_chars: int = 500
    overlap: int = 100          # fixed-overlap only
    contextual: bool = True     # prepend a situating header to every chunk
    # Persian-tuned by default, since the corpus is a Farsi diary. '' below
    # means the backend's recommended model (embedding.BACKEND_DEFAULTS).
    embedder: str = 'sentence-transformers'
    embed_model: str = ''       # model-backed kinds only; '' = backend default
    # --- the summary hierarchy, all of it built with no model call ----------
    # '' is flat, and flat is fingerprinted exactly as it was before these seven
    # fields existed (see fingerprint()).
    hierarchy: str = ''
    graph_source: str = 'hybrid'
    graph_knn: int = 8
    # One dial, two meanings, because a reader should not have to hold two
    # knobs that grey each other out: for the graph partitions it is the
    # modularity resolution γ, and for the clusterings it is the group count
    # `k = max(2, round(granularity * sqrt(n / 2)))`. 1.0 is therefore "this
    # family's own default" — the same idiom `''` already carries for a model.
    granularity: float = 1.0
    hierarchy_levels: int = 1
    min_group: int = 3
    summarizer: str = 'centroid'

    # Fields that describe the hierarchy and mean nothing without one. Listed
    # once, because both `normalized()` and `fingerprint()` need the same answer.
    HIERARCHY_FIELDS = ('graph_source', 'graph_knn', 'granularity',
                        'hierarchy_levels', 'min_group', 'summarizer')

    def normalized(self) -> 'IndexConfig':
        # A model that is not consulted is blanked, so it cannot invalidate the
        # fingerprint and cost a rebuild nobody asked for.
        return replace(self, embed_model=(self.embed_model
                                          if self.embedder in MODEL_EMBEDDERS
                                          else ''))

    def fingerprint(self) -> str:
        fields = asdict(self.normalized())
        # A new field here would otherwise change every hash and break every
        # collection name already recorded in `.runs/`.
        if not fields.get('dataset'):
            fields.pop('dataset', None)
        # Same rule, seven fields at once: `hierarchy=''` means none of them
        # ran, so a stale graph_knn left in a browser must not name a second
        # index holding byte-identical rows.
        if not fields.get('hierarchy'):
            for name in ('hierarchy', *self.HIERARCHY_FIELDS):
                fields.pop(name, None)
        else:
            # Within a hierarchy, the same rule one level down: a graph knob no
            # partition reads must not cost a rebuild.
            if fields['hierarchy'] not in GRAPH_HIERARCHIES:
                fields.pop('graph_source', None)
                fields.pop('graph_knn', None)
            elif fields.get('graph_source') not in KNN_SOURCES:
                fields.pop('graph_knn', None)
            if fields['hierarchy'] not in TUNED_HIERARCHIES:
                fields.pop('granularity', None)
            if fields['hierarchy'] not in LEVELLED_HIERARCHIES:
                fields.pop('hierarchy_levels', None)
        payload = json.dumps(fields, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def collection(self) -> str:
        return f'raglab-{self.fingerprint()}'


@dataclass(frozen=True)
class RetrievalConfig:
    """Everything between a question and the assembled context."""
    retriever: str = 'hybrid-rrf'
    k: int = 8                       # contexts handed to the answerer
    candidates: int = 40             # depth taken from each retriever
    rrf_k: int = 60
    time_filter: bool = True         # resolve Farsi time words into a date range
    # On by default: free, no LLM call, and measured positive on this fixture.
    multi_query: bool = True
    hyde: bool = False               # LLM hypothetical answer as the query
    expansion_model: str = ''        # HyDE only; '' = LabSettings.llm_model
    mmr_lambda: float = 1.0          # 1.0 = pure relevance, lower = diversify
    reranker: str = 'lexical'
    rerank_depth: int = 20
    reranker_model: str = ''         # reranker='llm' only
    recency_half_life_days: float = 180.0
    agentic_weights: tuple[float, float, float] = (1.0, 0.3, 0.2)
    grader: str = 'none'             # gate that makes abstention possible
    grade_threshold: float = 0.0
    grader_model: str = ''           # grader='llm' only
    max_context_chars: int = 6000
    # --- what retrieval does with an index that has summaries in it ---------
    # Retrieval settings, not index ones, so they stay outside the fingerprint
    # and all four scopes sweep free against a single build.
    summary_scope: str = 'mixed'
    # Applied before the candidate cut, never after: a summary that had not
    # already survived the cut could never be promoted into it, and a boost
    # applied afterwards is a no-op that looks like a knob.
    summary_boost: float = 1.0
    summary_levels: str = ''         # '' = every level

    def normalized(self) -> 'RetrievalConfig':
        return replace(self, agentic_weights=tuple(self.agentic_weights))


@dataclass(frozen=True)
class GenerationConfig:
    answerer: str = 'extractive'
    model: str = ''                  # '' = LabSettings.llm_model
    fact_judge: bool = False         # LLM check of the ground truth's derived_facts
    judge_model: str = ''            # the fact judge
    ragas_model: str = ''            # RAGAS's own judge, kept separate on purpose


@dataclass(frozen=True)
class LabConfig:
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    label: str = ''

    @classmethod
    def from_dict(cls, data: dict) -> 'LabConfig':
        """Build from the panel's JSON, ignoring unknown keys so a stale browser tab cannot crash a run."""
        def pick(kind, payload):
            fields = {f for f in kind.__dataclass_fields__}
            return kind(**{k: v for k, v in (payload or {}).items() if k in fields})
        index = pick(IndexConfig, data.get('index'))
        retrieval = pick(RetrievalConfig, data.get('retrieval'))
        generation = pick(GenerationConfig, data.get('generation'))
        return cls(index=index.normalized(), retrieval=retrieval.normalized(),
                   generation=generation, label=data.get('label', ''))

    def to_dict(self) -> dict:
        return {'index': asdict(self.index), 'retrieval': asdict(self.retrieval),
                'generation': asdict(self.generation), 'label': self.label}

    def validate(self) -> list[str]:
        """Returns every problem found; raises nothing."""
        bad = []
        checks = ((self.index.chunker, CHUNKERS, 'chunker'),
                  (self.index.embedder, EMBEDDERS, 'embedder'),
                  (self.index.hierarchy, HIERARCHIES, 'hierarchy'),
                  (self.index.graph_source, GRAPH_SOURCES, 'graph_source'),
                  (self.index.summarizer, SUMMARIZERS, 'summarizer'),
                  (self.retrieval.summary_scope, SUMMARY_SCOPES, 'summary_scope'),
                  (self.retrieval.retriever, RETRIEVERS, 'retriever'),
                  (self.retrieval.reranker, RERANKERS, 'reranker'),
                  (self.retrieval.grader, GRADERS, 'grader'),
                  (self.generation.answerer, ANSWERERS, 'answerer'))
        for value, allowed, name in checks:
            if value not in allowed:
                bad.append(f'unknown {name}: {value!r} (expected one of '
                           f'{", ".join(allowed)})')
        if self.retrieval.k < 1:
            bad.append('k must be >= 1')
        # A grouping whose library is not installed is refused, never silently
        # substituted for another one.
        if self.index.hierarchy:
            from ..rag_components.indexing.summary_hierarchy_builder import HIERARCHY_EXTRAS, hierarchy_available
            if not hierarchy_available(self.index.hierarchy):
                bad.append(
                    f'{self.index.hierarchy} needs a package this installation '
                    f'does not have: install it with '
                    f'{HIERARCHY_EXTRAS[self.index.hierarchy]}. It is refused rather '
                    f'than replaced by another grouping.')
            if self.index.hierarchy_levels < 1:
                bad.append('hierarchy_levels must be >= 1')
            if self.index.min_group < 2:
                bad.append('min_group must be >= 2 — a group of one is a chunk')
            if self.index.graph_knn < 1 and self.index.graph_source in KNN_SOURCES:
                bad.append('graph_knn must be >= 1')
        # A model belongs to exactly one backend; a mismatch is a validation
        # error rather than a silent fall back to the backend's own default.
        wanted = self.index.embed_model
        if wanted and self.index.embedder in MODEL_EMBEDDERS:
            from ..rag_components.indexing.embedding_backends import EMBED_MODELS
            served_by = {model.id: model.backend for model in EMBED_MODELS}
            backend = served_by.get(wanted)
            if backend and backend != self.index.embedder:
                bad.append(f'{wanted} is served by {backend}, not '
                           f'{self.index.embedder}: set the embedder to '
                           f'{backend} or pick one of its models')
        return bad


def _production_config() -> dict:
    """The settings the *shipped* Assistant retrieves with, in lab terms — the values live in `production_baseline_snapshot.py`."""
    from ..evaluation import production_baseline_snapshot as baseline
    return baseline.production_config(LabConfig().to_dict())


def __getattr__(name: str):
    """`PRODUCTION_CONFIG`, built on first access (PEP 562) since deriving it imports LangChain."""
    if name == 'PRODUCTION_CONFIG':
        value = _production_config()
        globals()[name] = value        # built once per process
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


# Re-exported: HELP lives in knob_help_text.py, pure UI copy that depends on nothing here.
from .knob_help_text import HELP  # noqa: F401,E402
