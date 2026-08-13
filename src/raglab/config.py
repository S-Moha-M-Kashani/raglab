"""Lab settings and the IndexConfig/RetrievalConfig/GenerationConfig/AgentConfig objects driving the pipeline.

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
# working: these moved to settings.py, which is where `load_env_file` reads
# `ROOT / '.env'` from.
from .settings import (ROOT, RUNS_DIR, LLM_PROVIDERS, PROVIDER_MODELS,  # noqa: F401
                       load_env_file, LabSettings, load_lab_settings,
                       settings_for_provider)

# Every option tuple leads with the value the lab actually defaults to, since
# both panels render these directly as offered choices;
# `test_every_option_list_leads_with_the_default` keeps a moved default from
# quietly staying buried in the list.
CHUNKERS = ('semantic-drift', 'fixed', 'fixed-overlap', 'message', 'turn-pair',
            'session')
# Chunkers that read chunk_chars/overlap, per chunking.py's own branches — the
# rest emit one piece per message, pair or day and ignore both numbers.
CHAR_SIZED_CHUNKERS = ('semantic-drift', 'fixed', 'fixed-overlap')
OVERLAP_CHUNKERS = ('fixed-overlap',)
# fastembed (its own ONNX list) and sentence-transformers (any HuggingFace
# checkpoint) load a named model; the hash embedders load none.
EMBEDDERS = ('sentence-transformers', 'fastembed',
             'ascii-hash', 'token-hash', 'char-hash')
MODEL_EMBEDDERS = ('fastembed', 'sentence-transformers')
# How chunks are grouped before being summarised beside the leaves; '' is flat
# and the default. Three families, in the order worth reading them: graph
# partitions, embedding clusterings, the declared control. Grouping is always
# over *chunks*, never entities — this is not GraphRAG, and `bipartite-terms`
# below is the closest honest analogue.
HIERARCHIES = ('', 'louvain', 'leiden', 'label-prop',
               'raptor', 'agglomerative', 'kmeans', 'metadata')
GRAPH_HIERARCHIES = ('louvain', 'leiden', 'label-prop')
CLUSTER_HIERARCHIES = ('raptor', 'agglomerative', 'kmeans')
# Groupings that can be asked for more than one level; label-prop and kmeans
# produce one partition and stop.
LEVELLED_HIERARCHIES = ('raptor', 'agglomerative', 'louvain', 'leiden')
# Groupings that read `granularity` (see IndexConfig for its two meanings).
# label-prop is the control precisely because it has no such parameter.
TUNED_HIERARCHIES = GRAPH_HIERARCHIES[:2] + CLUSTER_HIERARCHIES
# Declared metadata is deliberately absent as an edge source: that grouping is
# measured on its own as hierarchy='metadata', a control rather than an input here.
GRAPH_SOURCES = ('hybrid', 'knn', 'lexical', 'bipartite-terms')
KNN_SOURCES = ('hybrid', 'knn')
# All extractive: a build that called a model would be unsweepable and would
# let the `fake` provider fill the index with invention no field contradicts.
SUMMARIZERS = ('centroid', 'lead-idf', 'mmr', 'card')
RETRIEVERS = ('hybrid-rrf', 'dense', 'bm25')
# 'mixed' is the default so building a hierarchy changes nothing about
# retrieval until this knob moves.
SUMMARY_SCOPES = ('mixed', 'leaves', 'summaries', 'drill-down')
RERANKERS = ('lexical', 'none', 'recency', 'agentic', 'cross-encoder', 'llm')
GRADERS = ('none', 'lexical', 'llm')
ANSWERERS = ('extractive', 'none', 'llm')
# The 2x2 the agent is built on: retrieval-agent {off,on} x generation-agent
# {off,on}, so a row can attribute its win to a stage rather than to "the
# agent". '' is the off control. 'full' deliberately changes both against the
# control and is only interpretable beside the two middle rows — never alone.
SCOPES = ('', 'retrieve', 'generate', 'full')
# What the generation agent checks before shipping a draft. 'none' is the
# control for whether the critique bought anything at all.
CRITICS = ('grounded', 'both', 'none')
# Ascending, and the order a sample's uneven remainder is handed out in.
DIFFICULTIES = ('easy', 'medium', 'hard')
# How a limited run picks its questions. See evaluate.select_questions.
BALANCES = ('stride', 'difficulty')


# Re-exported: DEPENDENCIES lives in dependencies.py, and imports the option
# tuples above back from here — this line must stay below their definitions,
# since dependencies.py reads them off this (at that point still loading)
# module rather than duplicating them.
from .dependencies import DEPENDENCIES, dependency_state  # noqa: F401,E402


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
    Step('agent', 'Agent', 'Agent — the loop around the stages',
         'Off by default. Hands one stage above to a bounded LangGraph loop that '
         'can look again: retrieval, generation, or both. It changes no knob '
         'above it — it runs them repeatedly and decides when to stop.'),
)


@dataclass(frozen=True)
class IndexConfig:
    """What gets indexed. Its fingerprint names the in-memory index."""
    # Which corpus. '' is the built-in Farsi diary; anything else names a file
    # under fixtures/groundtruth_datasets/ or .datasets/ (see raglab/datasets.py).
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
    key_facts_judge: bool = False    # LLM check of ground-truth key_facts
    judge_model: str = ''            # the key-facts judge
    ragas_model: str = ''            # RAGAS's own judge, kept separate on purpose


@dataclass(frozen=True)
class AgentConfig:
    """Which stage a bounded loop owns, and how far it may go.

    Nothing here is an index field, so all four scopes sweep free against a
    single build. Every loop is bounded twice: `max_hops`/`max_revisions` bound
    its shape, `max_llm_calls` bounds its cost; whichever cap ends a run is
    named in `agent_stop` on the row.
    """
    scope: str = ''                  # '' | retrieve | generate | full
    max_hops: int = 3                # retrieval loops; retrieve/full
    rewrite: bool = True             # rewrite the query between hops
    evidence_threshold: float = 0.5  # the sufficiency verdict a hop must clear
    max_revisions: int = 1           # generation retries; generate/full
    critic: str = 'grounded'         # grounded | both | none
    max_llm_calls: int = 12          # the cost ceiling, per question
    plan_model: str = ''             # '' = LabSettings.llm_model
    critic_model: str = ''


@dataclass(frozen=True)
class LabConfig:
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
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
        agent = pick(AgentConfig, data.get('agent'))
        return cls(index=index.normalized(), retrieval=retrieval.normalized(),
                   generation=generation, agent=agent,
                   label=data.get('label', ''))

    def to_dict(self) -> dict:
        return {'index': asdict(self.index), 'retrieval': asdict(self.retrieval),
                'generation': asdict(self.generation),
                'agent': asdict(self.agent), 'label': self.label}

    def validate(self) -> list[str]:
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
                  (self.generation.answerer, ANSWERERS, 'answerer'),
                  (self.agent.scope, SCOPES, 'agent scope'),
                  (self.agent.critic, CRITICS, 'agent critic'))
        for value, allowed, name in checks:
            if value not in allowed:
                bad.append(f'unknown {name}: {value!r} (expected one of '
                           f'{", ".join(allowed)})')
        if self.retrieval.k < 1:
            bad.append('k must be >= 1')
        # A grouping whose library is not installed is refused, never silently
        # substituted for another one.
        if self.index.hierarchy:
            from .hierarchy import EXTRAS, hierarchy_available
            if not hierarchy_available(self.index.hierarchy):
                bad.append(
                    f'{self.index.hierarchy} needs a package this installation '
                    f'does not have: install it with '
                    f'{EXTRAS[self.index.hierarchy]}. It is refused rather '
                    f'than replaced by another grouping.')
            if self.index.hierarchy_levels < 1:
                bad.append('hierarchy_levels must be >= 1')
            if self.index.min_group < 2:
                bad.append('min_group must be >= 2 — a group of one is a chunk')
            if self.index.graph_knn < 1 and self.index.graph_source in KNN_SOURCES:
                bad.append('graph_knn must be >= 1')
        # A scope this installation cannot run is refused, never silently
        # served by the fixed pipeline.
        if self.agent.scope in SCOPES and self.agent.scope:
            from .agent import EXTRA, agent_available
            if not agent_available():
                bad.append(
                    f'agent scope {self.agent.scope!r} needs a package this '
                    f'installation does not have: install it with {EXTRA}. It '
                    f'is refused rather than run without the agent.')
            # Under `extractive` there is no LLM draft to critique or revise —
            # refused rather than silently promoting the answerer.
            elif (self.agent.scope in ('generate', 'full')
                    and self.generation.answerer != 'llm'):
                bad.append(
                    f'agent scope {self.agent.scope!r} owns generation, so the '
                    f'answerer must be llm (it is '
                    f'{self.generation.answerer!r}): the agent drafts, '
                    f'critiques and revises with a model')
            if self.agent.max_hops < 1:
                bad.append('agent max_hops must be >= 1 — a scope that owns '
                           'retrieval still retrieves once')
            if self.agent.max_revisions < 0:
                bad.append('agent max_revisions must be >= 0')
            if self.agent.max_llm_calls < 1:
                bad.append('agent max_llm_calls must be >= 1')
        # A model belongs to exactly one backend; a mismatch is a validation
        # error rather than a silent fall back to the backend's own default.
        wanted = self.index.embed_model
        if wanted and self.index.embedder in MODEL_EMBEDDERS:
            from .embedding import EMBED_MODELS
            served_by = {model.id: model.backend for model in EMBED_MODELS}
            backend = served_by.get(wanted)
            if backend and backend != self.index.embedder:
                bad.append(f'{wanted} is served by {backend}, not '
                           f'{self.index.embedder}: set the embedder to '
                           f'{backend} or pick one of its models')
        return bad


def _production_config() -> dict:
    """The settings the *shipped* Assistant retrieves with, in lab terms — the values live in `baseline.py`."""
    from . import baseline
    return baseline.production_config(LabConfig().to_dict())


def __getattr__(name: str):
    """`PRODUCTION_CONFIG`, built on first access (PEP 562) since deriving it imports LangChain."""
    if name == 'PRODUCTION_CONFIG':
        value = _production_config()
        globals()[name] = value        # built once per process
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


# Re-exported: HELP lives in help.py, pure UI copy that depends on nothing here.
from .help import HELP  # noqa: F401,E402
