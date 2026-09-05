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
# Every one of the existing importers still reaches them as
# `config.EMBEDDERS` etc.
from .option_vocabularies import (STAGE_KINDS, COMBINATORS, STAGE_WHEN,  # noqa: F401
                      NORMALIZERS, CHUNK_UNITS,
                      EMBEDDERS, MODEL_EMBEDDERS, HIERARCHIES,
                      GRAPH_HIERARCHIES, CLUSTER_HIERARCHIES,
                      LEVELLED_HIERARCHIES, TUNED_HIERARCHIES, GRAPH_SOURCES,
                      KNN_SOURCES, SUMMARIZERS, RETRIEVERS, SUMMARY_SCOPES,
                      RERANKERS, GRADERS, ANSWERERS)

# Re-exported: DEPENDENCIES lives in knob_dependencies.py, which now reads the
# option tuples from option_vocabularies.py rather than from here, so this import carries
# no ordering requirement of its own — kept in the same spot as before.
from .knob_dependencies import DEPENDENCIES, dependency_state, inert_knobs  # noqa: F401,E402
from . import split_plan as plan  # noqa: E402


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
         'Turns the retrieved contexts into an answer in the corpus\'s own '
         'language, refuses when the corpus is silent, and grades what it '
         'wrote.'),
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
    # Where a document is cut: an ordered list of stages, the document always
    # first, each subdividing what the one before produced — see
    # split_plan.py for the two forms it is written in. Every chunker the
    # lab used to name is one such list.
    split_plan: tuple[dict, ...] = plan.DEFAULT
    # The budget that closes the plan: a piece still larger than this after
    # the last stage is divided to fit, and `overlap` is repeated between the
    # pieces that division makes. `chunk_unit` says what the number counts.
    chunk_chars: int = 500
    chunk_unit: str = 'characters'
    overlap: int = 0
    # How a document's parts read as one text: joined by `part_join`, each
    # part's text as the corpus recorded it unless `part_prefix` names a
    # declared part-level label to write in front of it (`role: …`). Both
    # default to the corpus template's own rule — a bare newline, no prefix.
    part_join: str = '\n'
    part_prefix: str = ''
    # Which text normaliser the lexical stages tokenise with. '' follows the
    # corpus's declared language (text_normalizers.BY_LANGUAGE).
    normalizer: str = ''
    contextual: bool = True     # prepend a situating header to every chunk
    # The default backend's own recommended model is Persian-tuned, picked for
    # the bundled Farsi diary; '' below means whatever the chosen backend
    # recommends (embedding.BACKEND_DEFAULTS), so it follows the corpus.
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

    def __post_init__(self):
        # The plan is given its one canonical shape on construction — lists
        # retupled, a stage's unsaid defaults filled — so a config built in
        # code and one read off the panel's JSON validate and hash alike.
        object.__setattr__(self, 'split_plan', plan.normalize(self.split_plan))

    def normalized(self) -> 'IndexConfig':
        # A model that is not consulted is blanked, so it cannot invalidate the
        # fingerprint and cost a rebuild nobody asked for. The plan needs no
        # step here: `__post_init__` gave it its one canonical shape.
        return replace(self, embed_model=(self.embed_model
                                          if self.embedder in MODEL_EMBEDDERS
                                          else ''))

    def fingerprint(self) -> str:
        fields = asdict(self.normalized())
        # A new field here would otherwise change every hash and break every
        # collection name already recorded in `.runs/`.
        if not fields.get('dataset'):
            fields.pop('dataset', None)
        # Same rule for the four corpus-neutral knobs: each is inert at its
        # default, and an inert knob must not name a second index.
        for name, default in (('part_join', '\n'), ('part_prefix', ''),
                              ('normalizer', ''), ('chunk_unit', 'characters')):
            if fields.get(name) == default:
                fields.pop(name, None)
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
    # On by default: free, no LLM call, and measured positive on the bundled
    # Farsi diary.
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
        checks = ((self.index.embedder, EMBEDDERS, 'embedder'),
                  (self.index.normalizer, NORMALIZERS, 'normalizer'),
                  (self.index.chunk_unit, CHUNK_UNITS, 'chunk_unit'),
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
        bad.extend(self._plan_problems())
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

    def _plan_problems(self) -> list[str]:
        """The plan's own rules, and the two knobs that read the corpus's
        declared labels — checked against the selected corpus, so a boundary
        on a label it never declares is refused before a build begins rather
        than cutting nothing and saying so nowhere."""
        bad = []
        if self.index.chunk_chars < 1:
            bad.append('chunk_chars must be >= 1')
        if self.index.overlap < 0:
            bad.append('overlap must be >= 0')
        # Only a real model has units to count; the hash embedders see
        # characters, so a budget in tokens over one would be a budget in
        # characters wearing the wrong name.
        if (self.index.chunk_unit == 'tokens'
                and self.index.embedder not in MODEL_EMBEDDERS):
            bad.append(f'chunk_unit=tokens needs an embedder that loads a '
                       f'model ({", ".join(MODEL_EMBEDDERS)}); '
                       f'{self.index.embedder} reports no model units, so the '
                       'budget is refused rather than counted in characters')
        label_fields = self._declared_labels()
        bad.extend(plan.problems(self.index.split_plan, label_fields))
        prefix = self.index.part_prefix
        if prefix and label_fields is not None:
            definition = label_fields.get(prefix)
            if not definition or 'part' not in (definition.get('applies_to') or []):
                bad.append(f'part_prefix names no part-level label of the '
                           f'selected corpus: {prefix!r}')
        return bad

    def _declared_labels(self) -> dict | None:
        """The selected corpus's `label_fields`, or None when the corpus
        cannot be read here — then the plan is checked for shape alone, and
        the build is where an unknown dataset is refused."""
        from ..corpora import dataset_import_contract as datasets
        try:
            corpus = datasets.load_corpus(self.index.dataset)
        except (ValueError, OSError):
            return None
        return (corpus.get('corpus_dataset_metadata') or {}).get('label_fields') or {}


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
