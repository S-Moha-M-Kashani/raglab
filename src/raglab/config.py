"""Lab settings and the IndexConfig/RetrievalConfig/GenerationConfig/AgentConfig objects driving the pipeline.

Only IndexConfig enters the index fingerprint; the other three sweep for free
against a build already made. There is deliberately no vector-storage setting
here — the index lives in process memory (`store.MemoryVectors`), so nothing
here can be pointed at real chat memory by a typo or a stale shell.
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]        # the raglab repo root
# At the repo root, not beside the code: runs are the account of the work, and
# burying them inside src/ would read as build output.
RUNS_DIR = ROOT / '.runs'

# Local Ollama is the default so a judged run's hundreds of calls cannot
# silently spend API credit; naming another provider is an explicit opt-in.
# `claude`/`codex` are a CLI on this machine (clichat.py), not an endpoint, and need no key.
LLM_PROVIDERS = ('', 'openrouter', 'ollama', 'claude', 'codex', 'fake')

# One default model per backend, since a slug only means something to the
# backend that serves it. The local default is the model the judge screen has a
# row for (`.screens/`); 'fake' keeps the remote slug because it ignores the
# model entirely.
PROVIDER_MODELS = {'openrouter': 'openai/gpt-5-nano',
                   'ollama': '4skl/gemma4-e2b-mtp',
                   'claude': 'sonnet',
                   'codex': 'gpt-5.6-luna',
                   'fake': 'openai/gpt-5-nano'}


def load_env_file(path: Path | None = None) -> None:
    """Load repo-root .env into the environment without overriding what is already set."""
    path = path or ROOT / '.env'
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class LabSettings:
    openrouter_api_key: str = ''
    openrouter_base_url: str = 'https://openrouter.ai/api/v1'
    # See LLM_PROVIDERS. RAGLAB_LLM=ollama runs every LLM stage on a model on
    # this machine, which is what makes the expensive candidates measurable
    # without buying credit.
    llm_provider: str = 'ollama'
    ollama_base_url: str = 'http://localhost:11434/v1'
    # A setting, not an argv constant, because it moves the numbers; the two
    # CLIs accept different values and `clichat.checked_effort` refuses one the
    # chosen CLI does not — an unaccepted value can exit 0 with no text at all.
    cli_effort: str = 'low'
    # '' = the provider's own default (PROVIDER_MODELS), resolved in __post_init__.
    # It must follow the provider: a remote slug left standing under
    # RAGLAB_LLM=ollama made every run refuse for a model nobody picked.
    llm_model: str = ''
    fastembed_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    # Multilingual on purpose: fastembed's default rerankers are English-only
    # and score Farsi pairs as noise. Override with RAGLAB_CROSS_ENCODER.
    cross_encoder_model: str = 'jinaai/jina-reranker-v2-base-multilingual'

    def __post_init__(self):
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(
                f'unknown RAGLAB_LLM {self.llm_provider!r}; expected one of '
                + ', '.join(repr(name) for name in LLM_PROVIDERS))
        # Only when unset: overwriting a stated model would label a run with
        # one model while another scored it.
        if not self.llm_model:
            object.__setattr__(self, 'llm_model', PROVIDER_MODELS[self.provider])

    @property
    def provider(self) -> str:
        """The backend a chat model will actually be built with — never ''."""
        if self.llm_provider:
            return self.llm_provider
        return 'openrouter' if self.openrouter_api_key else 'fake'

    @property
    def llm_ready(self) -> bool:
        """Whether an LLM stage would reach a real model — not `bool(key)`, since 'fake' never fails either."""
        return self.provider in ('openrouter', 'ollama', 'claude', 'codex')


def load_lab_settings(env: dict | None = None) -> LabSettings:
    load_env_file()
    env = os.environ if env is None else env
    # BRAIN_CHROMA_URL and RAGLAB_CHROMA_DATABASE are deliberately not read: an
    # experiment must not find the board's Chroma stack just because it is there.
    return LabSettings(
        openrouter_api_key=env.get('OPENROUTER_API_KEY', ''),
        openrouter_base_url=env.get('OPENROUTER_BASE_URL',
                                    'https://openrouter.ai/api/v1'),
        llm_provider=env.get('RAGLAB_LLM', 'ollama'),
        ollama_base_url=env.get('RAGLAB_OLLAMA_BASE_URL',
                                'http://localhost:11434/v1'),
        cli_effort=env.get('RAGLAB_CLI_EFFORT', 'low'),
        llm_model=env.get('RAGLAB_MODEL', ''),
        fastembed_model=env.get(
            'RAGLAB_FASTEMBED_MODEL',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
        cross_encoder_model=env.get('RAGLAB_CROSS_ENCODER',
                                    'jinaai/jina-reranker-v2-base-multilingual'),
    )


def settings_for_provider(settings: LabSettings, provider: str) -> LabSettings:
    """One run's backend override, e.g. the panel's mode dropdown; '' passes settings through untouched.

    The old backend's *default* model does not survive the switch (PROVIDER_MODELS),
    but a model the user explicitly named (RAGLAB_MODEL) is never replaced."""
    if not provider:
        return settings
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f'unknown provider {provider!r}; expected one of '
            + ', '.join(repr(name) for name in LLM_PROVIDERS if name))
    model = ('' if settings.llm_model == PROVIDER_MODELS.get(settings.provider)
             else settings.llm_model)
    return replace(settings, llm_provider=provider, llm_model=model)


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


# Which controls are live, served here rather than duplicated per panel so the
# two cannot grey out different knobs. `on` lists the enabling values,
# `on_true` means a boolean must be set, and `reason` completes "disabled because …".
DEPENDENCIES = {
    'index.chunk_chars': {
        'field': 'index.chunker', 'on': list(CHAR_SIZED_CHUNKERS),
        'reason': 'the message, turn-pair and session chunkers cut on structure, '
                  'not on length'},
    'index.overlap': {
        'field': 'index.chunker', 'on': list(OVERLAP_CHUNKERS),
        'reason': 'only the fixed-overlap chunker slides a window'},
    'index.embed_model': {
        'field': 'index.embedder', 'on': list(MODEL_EMBEDDERS),
        'reason': 'the hash embedders load no model'},
    'index.graph_source': {
        'field': 'index.hierarchy', 'on': list(GRAPH_HIERARCHIES),
        'reason': 'the embedding clusterings and the metadata grouping build no '
                  'graph'},
    'index.graph_knn': {
        'field': 'index.graph_source', 'on': list(KNN_SOURCES),
        'reason': 'this edge source builds no nearest-neighbour edges'},
    'index.granularity': {
        'field': 'index.hierarchy', 'on': list(TUNED_HIERARCHIES),
        'reason': 'label propagation has no granularity parameter — that is what '
                  'makes it the control — and the metadata groups are given '
                  'rather than chosen'},
    'index.hierarchy_levels': {
        'field': 'index.hierarchy', 'on': list(LEVELLED_HIERARCHIES),
        'reason': 'this grouping produces one level and stops'},
    'index.min_group': {
        'field': 'index.hierarchy', 'on': [h for h in HIERARCHIES if h],
        'reason': 'nothing is grouped'},
    'index.summarizer': {
        'field': 'index.hierarchy', 'on': [h for h in HIERARCHIES if h],
        'reason': 'nothing is grouped, so nothing is summarised'},
    # The three below gate on an *index* field from the retrieval group, which
    # `dependency_state` resolves because it reads the whole config dict: what
    # retrieval may do with summaries is decided by whether the build wrote any.
    'retrieval.summary_scope': {
        'field': 'index.hierarchy', 'on': [h for h in HIERARCHIES if h],
        'reason': 'this index is flat — it holds no summaries to scope'},
    'retrieval.summary_boost': {
        'field': 'index.hierarchy', 'on': [h for h in HIERARCHIES if h],
        'reason': 'this index is flat — there is nothing to boost'},
    'retrieval.summary_levels': {
        'field': 'index.hierarchy', 'on': list(LEVELLED_HIERARCHIES),
        'reason': 'only a grouping with more than one level has levels to '
                  'choose between'},
    'retrieval.rrf_k': {
        'field': 'retrieval.retriever', 'on': ['hybrid-rrf'],
        'reason': 'only hybrid-rrf fuses two rankings'},
    'retrieval.rerank_depth': {
        'field': 'retrieval.reranker', 'on': [r for r in RERANKERS if r != 'none'],
        'reason': 'nothing is reranked'},
    'retrieval.reranker_model': {
        'field': 'retrieval.reranker', 'on': ['llm'],
        'reason': 'only the llm reranker calls a model'},
    'retrieval.recency_half_life_days': {
        'field': 'retrieval.reranker', 'on': ['recency', 'agentic'],
        'reason': 'only the recency and agentic rerankers weigh age'},
    'retrieval.agentic_weights': {
        'field': 'retrieval.reranker', 'on': ['agentic'],
        'reason': 'only the agentic reranker has weights to balance'},
    'retrieval.grade_threshold': {
        'field': 'retrieval.grader', 'on': [g for g in GRADERS if g != 'none'],
        'reason': 'the gate is off, so nothing is scored to threshold'},
    'retrieval.grader_model': {
        'field': 'retrieval.grader', 'on': ['llm'],
        'reason': 'only the llm gate calls a model'},
    'retrieval.expansion_model': {
        'field': 'retrieval.hyde', 'on_true': True,
        'reason': 'HyDE is off — multi-query expansion is rule-based and uses no '
                  'model'},
    'generation.model': {
        'field': 'generation.answerer', 'on': ['llm'],
        'reason': 'only the llm answerer calls a model'},
    'generation.judge_model': {
        'field': 'generation.key_facts_judge', 'on_true': True,
        'reason': 'the key-facts judge is off'},
    # Greyed out per *scope*, except `critic_model`, which gates on the critic —
    # resolved transitively by `dependency_state`, so turning the critic off
    # also greys the model it would have used.
    'agent.max_hops': {
        'field': 'agent.scope', 'on': ['retrieve', 'full'],
        'reason': 'this scope does not own retrieval, so it takes exactly one '
                  'hop'},
    'agent.rewrite': {
        'field': 'agent.scope', 'on': ['retrieve', 'full'],
        'reason': 'only a scope that hops has a query to rewrite between hops'},
    'agent.evidence_threshold': {
        'field': 'agent.scope', 'on': ['retrieve', 'full'],
        'reason': 'nothing asks whether the evidence is sufficient — retrieval '
                  'runs once'},
    'agent.max_revisions': {
        'field': 'agent.scope', 'on': ['generate', 'full'],
        'reason': 'this scope does not own generation, so the answerer writes '
                  'once'},
    'agent.critic': {
        'field': 'agent.scope', 'on': ['generate', 'full'],
        'reason': 'this scope does not own generation, so there is no draft to '
                  'critique'},
    'agent.max_llm_calls': {
        'field': 'agent.scope', 'on': [s for s in SCOPES if s],
        'reason': 'no agent is running, so there is no loop to put a ceiling on'},
    'agent.plan_model': {
        'field': 'agent.scope', 'on': ['retrieve', 'full'],
        'reason': 'planning, rewriting and the sufficiency verdict all belong '
                  'to the retrieval loop'},
    'agent.critic_model': {
        'field': 'agent.critic', 'on': [c for c in CRITICS if c != 'none'],
        'reason': 'the critic is off, so nothing reads a critic model'},
}


def dependency_state(cfg_dict: dict) -> dict:
    """For one config, `{'<group>.<field>': {'enabled': bool, 'reason': str}}` for every dependent field.

    A control whose owner is itself dead is dead, and reports the owner's
    reason rather than its own — resolved transitively, not as a special case,
    since a two-deep chain has the same defect."""
    state: dict = {}

    def resolve(key: str, seen: frozenset) -> dict:
        if key in state:
            return state[key]
        rule = DEPENDENCIES[key]
        owner = rule['field']
        group, _, name = owner.partition('.')
        current = (cfg_dict.get(group) or {}).get(name)
        enabled = (bool(current) if rule.get('on_true')
                   else current in rule.get('on', ()))
        reason = '' if enabled else rule['reason']
        # A cycle would be a bug in the table above, not a runtime condition;
        # the guard stops it taking the whole panel down with it.
        if enabled and owner in DEPENDENCIES and owner not in seen:
            above = resolve(owner, seen | {key})
            if not above['enabled']:
                enabled, reason = False, above['reason']
        state[key] = {'enabled': enabled, 'reason': reason}
        return state[key]

    for key in DEPENDENCIES:
        resolve(key, frozenset())
    return state


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


# Every knob explains itself in the panel, next to the control. A field added
# above without a line below fails test_every_configuration_factor_has_an_explainer.
# Keys are '<group>.<field>'; model fields are explained by models.ROLES
# instead, and 'run.*' describes controls that belong to one run, not a config.
HELP = {
    'index.dataset': (
        'Which corpus this experiment measures against. The built-in one is a '
        'year of Farsi diary chat with 112 ground-truth questions, and every '
        'finding in docs/report is about it; the bundled samples are there to '
        'tell a general result from one that is true of Farsi diaries. Changing '
        'it rebuilds the index — a corpus is what gets stored — and the '
        'leaderboard groups by it before anything else, because two corpora are '
        'not two configurations of one measurement. Import your own with the '
        'button beside it: docs/groundtruth-dataset-contract.md is the shape, '
        'and the lab refuses a dataset whose evidence quotes are not verbatim '
        'in the messages they cite.'),
    # `run.` not `index.` — this is not a field, it's the control beside one —
    # and the key's last segment is the file input's own id, which is how the
    # panel finds an explainer at all. The one entry written as a shape rather
    # than prose (`p.explain` keeps its newlines); every other text is one line.
    'run.dataset-file': (
        'One JSON file holding a corpus and the questions it can be asked. '
        'Three keys, all required, checked in full and refused — never '
        'repaired — with every problem reported at once:\n'
        '\n'
        '"dataset": {\n'
        '  "id": "support-en",\n'
        '  "name": "Product support tickets",\n'
        '  "language": "en"}\n'
        'id is 2–40 characters of a–z, 0–9 and hyphens, and becomes the '
        'value recorded on every run and every leaderboard row; language picks '
        'the contextual header\'s language and is what lets the panel say that '
        'an English-only embedder cannot read this corpus. Optional: '
        'description, query_date.\n'
        '\n'
        '"sessions": [{\n'
        '  "session_id": "t-1",\n'
        '  "date": "2025-11-04",\n'
        '  "messages": [{"role": "user", "content": "…"}]}]\n'
        'One conversation per entry, at least one entry, session_id unique '
        'because evidence points at it; date is YYYY-MM-DD, since the time '
        'filter and every recency score read it as a number; role is "user" or '
        '"assistant". Optional: time, source, topics, threads, and mood '
        '{"label", "valence", "arousal"} — an absent mood is neutral, not an '
        'error, and a corpus that is not a diary should not have to invent '
        'one.\n'
        '\n'
        '"questions": [{\n'
        '  "id": "q-1",\n'
        '  "type": "single-hop",\n'
        '  "difficulty": "easy",\n'
        '  "answerable": true,\n'
        '  "question": "…",\n'
        '  "answer": "…",\n'
        '  "evidence": [{"session_id": "t-1",\n'
        '                "message_indices": [0],\n'
        '                "quote": "…"}]}]\n'
        'type is one of single-hop, temporal, multi-hop, aggregation, '
        'knowledge-update, commitment, entity, pattern, habit, abstention, '
        'adversarial — the list is closed because a run filters and reports by '
        'it. difficulty is easy, medium or hard, and is load bearing: a '
        'balanced sample takes an equal share of each. answer and evidence are '
        'required when answerable is true, and the unanswerable questions are '
        'what measures whether a pipeline knows to refuse. Optional: '
        'question_en, key_facts, time_scope, query_date.\n'
        '\n'
        'The rule that earns its cost: every evidence quote must appear '
        'verbatim in a message it cites. Quote recall, the Inspector\'s '
        'highlighted spans and the offline context metrics are all computed '
        'against those strings, so a corpus that misquotes itself does not '
        'score worse — it scores confidently about text it never contained. '
        'The full contract, with a commented skeleton and the four bundled '
        'samples that meet it: docs/groundtruth-dataset-contract.md.'),
    'index.chunker': (
        'How a day of chat is cut into the pieces that get embedded. '
        '"fixed" packs 500 characters regardless of meaning; "message" keeps one '
        'message per piece; "turn-pair" keeps a question with its answer; '
        '"session" stores the whole day; "semantic-drift" cuts where the topic '
        'actually changes, which is what the default measures best on.'),
    'index.chunk_chars': (
        'Target size of one piece. Small pieces retrieve precisely but often lose '
        'the sentence that answers the question; large ones keep the answer but '
        'drag unrelated text into the context and dilute precision.'),
    'index.overlap': (
        'Characters repeated between neighbouring pieces, so a sentence sitting '
        'on a boundary is not cut in half. Only the "fixed-overlap" chunker uses '
        'it.'),
    'index.contextual': (
        'Prepend a one-line header — date, mood, storyline — to every chunk '
        'before embedding it (Anthropic call this contextual retrieval). A diary '
        'chunk that says "بهتر شد" is unsearchable without knowing what "it" was. '
        'Built from metadata alone, so it costs no model call and no summary.'),
    'index.embedder': (
        'Turns text into the vector the index is searched by, and the one choice that '
        'decides whether anything else matters. Each option says which languages '
        'it can represent: "ascii-hash" is what the brain ships today and reads '
        'Latin script only, so Farsi embeds to the zero vector — measured at 0.01 '
        'recall, i.e. chance. The other two hash embedders see any script but '
        'only as letters, never as meaning. Two options load a real model, and '
        'they differ in what that costs: "fastembed" runs its own short ONNX list '
        'locally; "sentence-transformers" runs any HuggingFace checkpoint, which '
        'is the only way to reach the Persian-tuned encoders — it needs '
        'the local-embeddings extra and downloads weights. Whichever you pick, '
        'the model below decides the languages.'),
    'index.embed_model': (
        'Which real embedding model the chosen backend loads. This is where Farsi '
        'is won '
        'or lost: the famous ones (bge-small-en, all-MiniLM-L6) are English-only '
        'and will return confident numbers that measure nothing on a Persian '
        'diary, so every entry states its coverage and the multilingual ones are '
        'listed by name. Bigger is not automatically better — e5 is trained for '
        'retrieval while the paraphrase models are trained for similarity — and '
        'the E5 family needs its "query:"/"passage:" prefixes, which the lab '
        'applies for you. Changing this rebuilds the index, because it changes '
        'what is stored.'),
    'index.hierarchy': (
        'Groups the chunks and indexes one summary per group *beside* them — '
        'the leaves always stay, because a summary that drops "the sixth '
        'rejection" makes the counting question unanswerable forever. Three '
        'families: "louvain", "leiden" and "label-prop" partition a graph built '
        'over the chunks; "raptor", "agglomerative" and "kmeans" cluster the '
        'chunk vectors; "metadata" groups by the corpus\'s own storylines and is '
        'the control — that grouping was measured in July 2026 and deleted for '
        'scoring within 0.006 of no hierarchy at all. **These are not GraphRAG.** '
        'GraphRAG extracts entities with a model; this lab builds offline, so '
        'the nodes are chunks and "bipartite-terms" below is the closest honest '
        'analogue. Expect Leiden and Louvain to tie at this corpus size: '
        'Leiden\'s advantage is over badly-connected communities, which needs '
        'scale to show.'),
    'index.graph_source': (
        'What an edge between two chunks means. "knn" is cosine similarity — '
        'each chunk joined to its nearest neighbours; "lexical" is shared rare '
        'words, which is what catches names and numbers a vector blurs; '
        '"hybrid" is both. "bipartite-terms" makes the rare words nodes too and '
        'partitions chunks and terms together, so a community has a nameable '
        'subject — the nearest thing to an entity graph available without a '
        'model. The corpus\'s declared topics and storylines are deliberately '
        'not an edge source: that grouping is measurable on its own as '
        'hierarchy="metadata", and mixing it in here would re-derive an already '
        'answered question inside a new one.'),
    'index.graph_knn': (
        'How many nearest neighbours each chunk is joined to. Low values leave '
        'the graph in disconnected pieces and every piece becomes its own '
        'community; high values connect everything to everything and modularity '
        'collapses toward one giant group. The build reports both, so this is a '
        'knob you can tune by reading the index statistics rather than by '
        'running an evaluation.'),
    'index.granularity': (
        'How coarse the grouping is, and it means two different things because '
        'the two families take two different parameters. For the graph '
        'partitions it is the modularity resolution: above 1.0 gives more, '
        'smaller communities. For the clusterings it is the group count, taken '
        'as granularity × √(n/2) over the leaf chunks — the usual rule of '
        'thumb, about 27 groups on the built-in diary. Either way 1.0 means '
        '"this family\'s own default".'),
    'index.hierarchy_levels': (
        'How many times to group the groups. Level 1 summarises chunks; level 2 '
        'summarises those summaries. More levels answer broader questions and '
        'cost precision, because a summary of summaries is two extractions away '
        'from anything the diarist actually said.'),
    'index.min_group': (
        'The smallest group worth summarising. Below it the members are left as '
        'leaves and no summary row is written — a "summary" of two chunks is '
        'the two chunks with a header on top, and it competes against its own '
        'members in the search.'),
    'index.summarizer': (
        'How a group becomes one piece of text, without a model — a build that '
        'called an LLM would take hours instead of seconds and would let the '
        'offline fake backend fill the index with confident invention. '
        '"centroid" concatenates the members nearest the group\'s centre; '
        '"lead-idf" takes the sentences covering the most rare words; "mmr" '
        'picks members for coverage without repetition; "card" writes no prose '
        'at all — top terms, date span, member count, session ids — which is '
        'the cheapest and the most likely to help a counting question, because '
        'it states a number instead of asking the model to count chunks.'),
    'retrieval.summary_scope': (
        'What the search is allowed to see. "mixed" puts summaries and leaves '
        'in one pool, so a hierarchy changes nothing until you move this; '
        '"leaves" ignores the summaries entirely and is the control that says '
        'whether building them bought anything; "summaries" searches only them. '
        '"drill-down" retrieves among summaries and then expands each to its '
        'members — the shape of GraphRAG\'s local search, and the one mechanism '
        'that answers the failure this lab actually measured: in July 2026 the '
        'habit ledger was correct and reachable and was retrieved for 1 question '
        'in 24, because twenty times more leaves outvoted it.'),
    'retrieval.summary_boost': (
        'Multiplies every summary\'s score before the candidates are cut. 1.0 is '
        'off. Applied before the cut and never after, because a summary that '
        'had not already survived the cut cannot be promoted into it — that '
        'version was measured and was a no-op that looked like a knob. Be '
        'careful with it even so: a boost lifts every summary equally, so it '
        'buys visibility for whichever kind of group is most numerous, which is '
        'how the same idea failed in July. "drill-down" is the targeted '
        'alternative.'),
    'retrieval.summary_levels': (
        'Which levels of the hierarchy may be retrieved, as a space-separated '
        'list ("1", "1 2"). Empty means all of them. Worth setting when a deep '
        'hierarchy is answering broad questions well and specific ones badly: '
        'the top level is the one most likely to be retrieved for everything.'),
    'retrieval.retriever': (
        '"dense" searches vectors (meaning), "bm25" searches words (exact names, '
        'numbers, rare terms), "hybrid-rrf" runs both and fuses the two rankings '
        'with Reciprocal Rank Fusion. Hybrid wins here because a diary is full of '
        'proper nouns a vector blurs.'),
    'retrieval.k': (
        'How many chunks the answerer finally sees. Raising it finds more evidence '
        'and lowers precision; it is the single knob that moves recall and '
        'precision in opposite directions.'),
    'retrieval.candidates': (
        'How deep each retriever looks before fusion and reranking. Cheap to '
        'raise — nothing reads these yet — and it is what gives the reranker '
        'something to find.'),
    'retrieval.rrf_k': (
        'The constant in Reciprocal Rank Fusion (1/(k+rank)). Higher flattens the '
        'ranking, so agreement between the two retrievers matters more than either '
        'one being confident.'),
    'retrieval.time_filter': (
        'Reads Farsi time words — «آذر», «تابستون», «سه ماه پیش» — as a Jalali '
        'date range and restricts the search to it. Without this, "what happened '
        'in آذر" retrieves the whole year.'),
    'retrieval.multi_query': (
        'Searches several rewrites of the question (stripped of question words, '
        'keywords only) and merges the hits. Rule-based, so it costs nothing: '
        'measured quote recall 0.489 → 0.512 with no model call.'),
    'retrieval.hyde': (
        'Writes a hypothetical diary answer with a model and searches with that '
        'instead of the question, because an answer looks more like the text you '
        'are hunting for than a question does. Costs one LLM call per query.'),
    'retrieval.mmr_lambda': (
        'Maximal Marginal Relevance. At 1.0 the top k are simply the best-scoring '
        'chunks, which on a diary often means five chunks from one afternoon. '
        'Lower it to trade some relevance for spread across days.'),
    'retrieval.reranker': (
        'Re-scores the candidates before the cut to k. "lexical" is free IDF '
        'coverage; "recency" prefers recent entries; "agentic" is the Generative '
        'Agents mix of relevance + recency + importance; "cross-encoder" reads '
        'question and chunk together with a real model; "llm" asks a model to '
        'score each one.'),
    'retrieval.rerank_depth': (
        'How many candidates the reranker actually reads. The reranker is the '
        'expensive stage, so this is the cost dial: depth 20 with k 8 means '
        'twenty chunks scored to choose eight.'),
    'retrieval.recency_half_life_days': (
        'How fast the recency reranker forgets. At 180 days an entry from six '
        'months ago counts half as much as today — right for "how am I doing '
        'lately", wrong for "what happened last summer".'),
    'retrieval.agentic_weights': (
        'The three weights of the agentic reranker: relevance, recency, '
        'importance. Importance comes from the chunk itself (decisions, promises '
        'and dated commitments score higher than small talk).'),
    'retrieval.grader': (
        'The gate that makes abstention possible: chunks scoring below the '
        'threshold are dropped, and if nothing survives the pipeline refuses '
        'instead of answering from noise. "none" means every question gets an '
        'answer, including the ones the diary never mentions.'),
    'retrieval.grade_threshold': (
        'The score a chunk must clear to survive the gate. Measured: the lexical '
        'gate has no usable setting (0.6 caught 6 of 8 unanswerable questions but '
        'wrongly refused 52% of the answerable ones), while an LLM gate at 0.4 '
        'refused 5 of 5 with 3% false refusals.'),
    'retrieval.max_context_chars': (
        'Budget for the assembled context. When it is exceeded whole chunks are '
        'dropped, never truncated — half a diary entry reads as a complete one '
        'and invites an answer from a sentence whose second half changed the '
        'meaning.'),
    'generation.answerer': (
        '"none" measures retrieval alone. "extractive" quotes the longest '
        'sentence from each top chunk — deterministic, free, and honest about '
        'being quoting rather than answering. "llm" actually writes the answer.'),
    'generation.key_facts_judge': (
        'Scores each answer against the ground truth\'s atomic key facts with a '
        'model. The facts are English and the answers Farsi, so no lexical metric '
        'can do this — and it is the metric that exposed generation as the '
        'bottleneck (coverage 0.261 against faithfulness 0.743).'),
    'agent.scope': (
        'Which stage a bounded loop is allowed to own, and the only setting '
        'here that changes what runs. "off" is the fixed pipeline every number '
        'in docs/report was measured on. "retrieval" lets the agent look again: '
        'plan → retrieve → is this enough? → rewrite → retrieve, with '
        'generation left exactly as it is. "generation" holds retrieval fixed '
        'and lets the agent draft, critique its own draft against the retrieved '
        'text, and revise. "both" is the two together plus the one edge neither '
        'has alone — a bad critique can send it back for different evidence. '
        'The four are a 2×2 on purpose: "both" changes two things at once, so '
        'it means something only beside the two middle rows and nothing on its '
        'own. Every retrieval and generation knob above still applies on every '
        'hop; the agent runs them repeatedly, it does not replace them.'),
    'agent.max_hops': (
        'How many times the retrieval loop may go round before it settles for '
        'what it has. Each hop is a fresh retrieval plus a sufficiency verdict, '
        'so this is the retrieval scope\'s cost dial. 1 makes the agent a single '
        'extra model call over the fixed pipeline, which is the honest floor to '
        'compare the rest against.'),
    'agent.rewrite': (
        'Rewrite the query between hops, rather than searching the same words '
        'again. Off, the hops differ only in what the sufficiency verdict asked '
        'for — which is the control that says whether rewriting was the useful '
        'part of the loop or whether looking twice was.'),
    'agent.evidence_threshold': (
        'The sufficiency verdict a hop has to clear for the agent to stop '
        'looking. High values hop more and cost more; low values make the loop '
        'a formality. An unreadable verdict counts as *insufficient* — never as '
        'a number that clears this bar — because a model that cannot be reached '
        'must not look like a loop that succeeded.'),
    'agent.max_revisions': (
        'How many times a refused draft may be rewritten. 0 means the critic '
        'reports and nothing acts on it, which is worth running: it separates '
        '"the critic was right" from "the revision helped".'),
    'agent.critic': (
        'What the agent checks before it ships a draft. "grounded" asks only '
        'whether every claim is supported by the retrieved text — the '
        'hallucination check. "both" also asks whether the draft actually '
        'answers the question, which is the failure the key-facts judge '
        'measured (coverage 0.261 against faithfulness 0.743). "none" ships the '
        'first draft and is the control: a generation row that cannot beat it '
        'paid for a critique that changed nothing.'),
    'agent.max_llm_calls': (
        'A hard ceiling on model calls per question, whatever the loop still '
        'wants to do. The shape caps above bound how many times it goes round; '
        'this bounds what that costs, because a judged sweep multiplies it by '
        'thirty questions and four candidates. When this is what ends a run the '
        'row says so ("call-cap"), so a cheap-looking score can never be a '
        'truncated loop nobody noticed.'),
    'run.mode': (
        'Where the LLM stages run. "Local (Ollama)" is the lab default — free '
        'and private — and resets every stage to the lab\'s own defaults. '
        '"OpenRouter" switches the backend and presets the full pipeline onto '
        'gpt-5-nano (HyDE, LLM reranker, relevance gate, answerer and both '
        'judges); the relevance gate prefers a purpose-built reranker when '
        'OpenRouter\'s model list verifies one. The embedder stays the local '
        'Persian-tuned encoder either way. Picking a mode overwrites those '
        'stage choices; every knob can still be changed afterwards.'),
    'run.openrouter_key': (
        'The key the OpenRouter backend calls with, entered here instead of in '
        '.env so a lab already running can reach a remote model. It is held in '
        'the lab process and written nowhere — not to a run file, not to the '
        'experiment ledger, not to your browser — so it is forgotten when the '
        'lab stops; OPENROUTER_API_KEY in the environment is still how a lab '
        'starts with one. Setting it does not change which backend runs: that '
        'is the dropdown above, and a model on this machine needs no key at '
        'all.'),
    'run.ragas_mode': (
        '"offline" scores the retrieved context against the ground-truth quotes '
        'with string similarity — no model, no key, no variance. "judged" adds '
        'faithfulness, answer relevancy and factual correctness, which need a '
        'model. "off" skips RAGAS.'),
    'run.ragas_limit': (
        'How many questions RAGAS scores, when judged metrics make the full set '
        'too slow or too expensive.'),
    'run.limit': (
        'How many ground-truth questions to score. The subset is taken by '
        'striding, not truncating, so a limit of 10 still covers every question '
        'type instead of ten single-hop ones.'),
    'run.types': (
        'Restrict the run to certain question types — single-hop, multi-hop, '
        'temporal, counting, latest-state, unanswerable, adversarial. The type '
        'breakdown is usually more informative than the headline.'),
    'run.difficulty': 'Restrict the run to easy, medium or hard questions.',
    'run.balance': (
        'How a limited run chooses its questions. "difficulty" takes an equal '
        'share of easy, medium and hard; "stride" spreads across the set as it '
        'is, which means medium — 57 of the 112 — gets about half the sample. '
        'That matters because the four deciding metrics are means over '
        'questions, so a skewed sample measures one band and reports it as the '
        'pipeline. The sweep uses 49 balanced (17/16/16); runs before '
        '2026-07-31 used stride, which is why the setting is recorded on every '
        'row rather than assumed.'),
    'run.workers': (
        'How many questions are scored in parallel. Only worth raising when a '
        'stage calls a model, where wall-clock is dominated by waiting.'),
    'run.label': (
        'What this run is called in the leaderboard. Worth writing: a row named '
        '"semantic-drift" tells you nothing three days later.'),
}
