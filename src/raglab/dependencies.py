"""Which controls the panel greys out, and why."""

from .options import (CHAR_SIZED_CHUNKERS, OVERLAP_CHUNKERS, MODEL_EMBEDDERS,
                      GRAPH_HIERARCHIES, KNN_SOURCES, TUNED_HIERARCHIES,
                      LEVELLED_HIERARCHIES, HIERARCHIES, RERANKERS, GRADERS,
                      SCOPES, CRITICS)


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
