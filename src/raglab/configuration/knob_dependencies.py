"""Which controls the panel greys out, and why."""

from raglab.configuration.option_vocabularies import (
    CHAR_SIZED_CHUNKERS,
    OVERLAP_CHUNKERS,
    MODEL_EMBEDDERS,
    GRAPH_HIERARCHIES,
    KNN_SOURCES,
    TUNED_HIERARCHIES,
    LEVELLED_HIERARCHIES,
    HIERARCHIES,
    RERANKERS,
    GRADERS)


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
    # `time_filter` reads a *dataset* fact, not another knob — the new source
    # this table gains for D5: a corpus with no declared date label has no
    # time behaviour at all, whatever else is set. `dependency_state` is
    # handed this fact under a synthetic `dataset` group beside `index`/
    # `retrieval`/`generation`, resolved by the same `on_true` rule every
    # boolean knob above uses — the mechanism does not change, only what it
    # is told.
    'retrieval.time_filter': {
        'field': 'dataset.date_label', 'on_true': True,
        'reason': 'this corpus declares no date label'},
    # The two below keep their original reranker-based rule — the pipeline
    # (question_to_answer_pipeline.py) reads `recency_half_life_days` and
    # `agentic_weights` only inside the branches for the rerankers named
    # there, so a knob nothing reads must grey out regardless of the corpus —
    # and gain a second, independent condition on top of it: `also`, a
    # second `{field, on_true/on, reason}` check of exactly the same shape,
    # which must *also* hold. Composed rather than replaced, since either
    # fact alone makes the knob meaningless: a dated corpus with the lexical
    # reranker greys for the reranker reason, and the recency reranker over a
    # dateless corpus greys for the dataset reason.
    'retrieval.recency_half_life_days': {
        'field': 'retrieval.reranker', 'on': ['recency', 'agentic'],
        'reason': 'only the recency and agentic rerankers weigh age',
        'also': {'field': 'dataset.date_label', 'on_true': True,
                 'reason': 'this corpus declares no date label'}},
    'retrieval.agentic_weights': {
        'field': 'retrieval.reranker', 'on': ['agentic'],
        'reason': 'only the agentic reranker has weights to balance',
        'also': {'field': 'dataset.ranks_label', 'on_true': True,
                 'reason': 'this corpus declares no ranks label'}},
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
        'field': 'generation.fact_judge', 'on_true': True,
        'reason': 'the fact judge is off'},
}


def _check(cfg_dict: dict, condition: dict) -> tuple[bool, str]:
    """One `{field, on_true|on, reason}` condition against a config dict:
    whether the named field's current value satisfies it, and the reason if
    not. The one piece every entry in `DEPENDENCIES` shares, whether it is a
    rule's primary condition or its `also`."""
    group, _, name = condition['field'].partition('.')
    current = (cfg_dict.get(group) or {}).get(name)
    enabled = (bool(current) if condition.get('on_true')
               else current in condition.get('on', ()))
    return enabled, ('' if enabled else condition['reason'])


def dependency_state(cfg_dict: dict) -> dict:
    """For one config, `{'<group>.<field>': {'enabled': bool, 'reason': str}}` for every dependent field.

    A control whose owner is itself dead is dead, and reports the owner's
    reason rather than its own — resolved transitively, not as a special
    case, since a two-deep chain has the same defect. A rule's `also` is a
    second, independent condition of the same shape: both must hold, and the
    primary condition (and its transitive chain) is checked first, so a
    control killed by its owner keeps reporting the owner's reason rather
    than the second condition's."""
    state: dict = {}

    def resolve(key: str, seen: frozenset) -> dict:
        if key in state:
            return state[key]
        rule = DEPENDENCIES[key]
        owner = rule['field']
        enabled, reason = _check(cfg_dict, rule)
        # A cycle would be a bug in the table above, not a runtime condition;
        # the guard stops it taking the whole panel down with it.
        if enabled and owner in DEPENDENCIES and owner not in seen:
            above = resolve(owner, seen | {key})
            if not above['enabled']:
                enabled, reason = False, above['reason']
        if enabled and 'also' in rule:
            also_enabled, also_reason = _check(cfg_dict, rule['also'])
            if not also_enabled:
                enabled, reason = False, also_reason
        state[key] = {'enabled': enabled, 'reason': reason}
        return state[key]

    for key in DEPENDENCIES:
        resolve(key, frozenset())
    return state


def _field_present(cfg_dict: dict, field: str) -> bool:
    """Whether `field`'s dotted group and name were both actually recorded
    in `cfg_dict` — the distinction `_check` cannot draw, since it folds a
    field nobody set and a field set to something falsy into the same
    `None`."""
    group, _, name = field.partition('.')
    group_dict = cfg_dict.get(group)
    return isinstance(group_dict, dict) and name in group_dict


def _disabled_by_known_fields(cfg_dict: dict, state: dict, key: str,
                               seen: frozenset) -> bool:
    """Retraces the one branch of `dependency_state`'s chain that actually
    disabled `key` — its own condition, then its owner's, then its `also` —
    in the same order that resolve() checks them, to ask whether the
    condition that fired named a field this config recorded. `_check` is
    reused for every condition; only the walk is repeated, since
    `dependency_state` does not hand back which branch decided."""
    rule = DEPENDENCIES[key]
    owner = rule['field']
    enabled, _ = _check(cfg_dict, rule)
    if not enabled:
        return _field_present(cfg_dict, owner)
    if (owner in DEPENDENCIES and owner not in seen
            and not state[owner]['enabled']):
        return _disabled_by_known_fields(cfg_dict, state, owner, seen | {key})
    also = rule.get('also')
    if also is not None:
        also_enabled, _ = _check(cfg_dict, also)
        if not also_enabled:
            return _field_present(cfg_dict, also['field'])
    raise AssertionError(
        f'{key}: dependency_state reported this disabled but no branch of '
        'the same walk disabled it — the table and this function disagree')


def inert_knobs(cfg_dict: dict) -> dict[str, str]:
    """Which recorded knobs this config never read, and why — {} when none.

    Built on `dependency_state` rather than beside it: a knob is inert when
    that table reports it disabled *and* every condition on the chain that
    disabled it — the rule's own, an owner's, an `also` — named a field this
    config actually recorded. A field the config never wrote is a question
    nobody asked, not an answer of no, so it marks nothing inert (unknown is
    not inert). The knob's own group and field must also have been
    recorded, or there is nothing here to call inert — a config that never
    wrote `retrieval` at all has nothing to mark there, only unlabelled."""
    state = dependency_state(cfg_dict)
    inert: dict[str, str] = {}
    for key, result in state.items():
        if result['enabled'] or not _field_present(cfg_dict, key):
            continue
        if _disabled_by_known_fields(cfg_dict, state, key, frozenset()):
            inert[key] = result['reason']
    return inert
