"""The shipped-assistant preset, after it stopped being derived."""
from raglab import baseline


# This is a unit test.
def test_the_preset_is_a_dated_snapshot_and_says_so():
    """The button claims to be the real system. It used to import
    lodestar_brain's own constants, so it could not drift; frozen, it can — so
    the one thing it must never do is present itself as current. The date and the
    commit are part of the value, not a comment."""
    assert baseline.SNAPSHOT_DATE == '2026-08-11'
    assert len(baseline.SNAPSHOT_COMMIT) >= 7
    assert baseline.SNAPSHOT_DATE in baseline.LABEL
    assert 'shipped assistant' in baseline.LABEL


# This is a unit test.
def test_the_preset_mirrors_what_lodestar_shipped():
    """Two honest differences are asserted rather than smoothed over, because
    the label says "the real system" and not "the best one we found". Lodestar
    splits with a recursive 500/100 splitter, so the mirror is fixed-overlap at
    those sizes — not semantic-drift, which the sweep preferred but the brain
    does not ship. And it prepends no situating header, so contextual is off."""
    preset = baseline.production_config({
        'index': {'chunker': 'semantic-drift', 'chunk_chars': 900,
                  'overlap': 0, 'contextual': True, 'embedder': 'hash',
                  'embed_model': 'x'},
        'retrieval': {'retriever': 'dense', 'k': 4, 'candidates': 10,
                      'rrf_k': 1, 'time_filter': False, 'multi_query': False,
                      'hyde': True, 'mmr_lambda': 0.5, 'reranker': 'none',
                      'rerank_depth': 5, 'grader': 'none',
                      'grade_threshold': 0.9, 'agentic_weights': (1, 2)},
        'generation': {'answerer': 'extractive'},
        'run': {'limit': 5},
    })
    assert preset['index'] == {
        'chunker': 'fixed-overlap', 'chunk_chars': 500, 'overlap': 100,
        'contextual': False, 'embedder': 'sentence-transformers',
        'embed_model': ''}
    assert preset['retrieval'] == {
        'retriever': 'hybrid-rrf', 'k': 8, 'candidates': 40, 'rrf_k': 60,
        'time_filter': True, 'multi_query': True, 'hyde': False,
        'mmr_lambda': 1.0, 'reranker': 'lexical', 'rerank_depth': 20,
        'grader': 'llm', 'grade_threshold': 0.4, 'agentic_weights': [1, 2]}
    assert preset['generation'] == {'answerer': 'llm'}
    assert preset['label'] == baseline.LABEL


# This is a unit test.
def test_every_field_of_the_default_config_survives():
    """Built over the defaults so every field is present: the panel fills its
    whole form from this, and a knob Lodestar has no opinion on (the recency
    half-life, which its lexical reranker never reads) should read as the lab
    default rather than blank."""
    preset = baseline.production_config({
        'index': {}, 'retrieval': {'agentic_weights': ()},
        'generation': {}, 'run': {'limit': 5, 'half_life_days': 90},
    })
    assert preset['run'] == {'limit': 5, 'half_life_days': 90}


# This is a unit test.
def test_agentic_weights_are_served_as_a_list():
    """A list, not the dataclass's tuple: this dict is served as JSON, where a
    tuple arrives as a list anyway, and "what the panel receives" should equal
    what this module holds rather than merely resemble it."""
    preset = baseline.production_config(
        {'index': {}, 'retrieval': {'agentic_weights': (0.5, 0.5)},
         'generation': {}, 'run': {}})
    assert preset['retrieval']['agentic_weights'] == [0.5, 0.5]
    assert isinstance(preset['retrieval']['agentic_weights'], list)


# This is a unit test.
def test_the_snapshot_does_not_mutate_the_defaults_it_was_given():
    """`LabConfig().to_dict()` is a fresh dict per call today, but the preset is
    built once per process and served on every /api/options — writing through to
    a caller's dict would make the lab's own defaults drift into the shipped
    preset's values the first time anybody looked at the panel."""
    defaults = {'index': {'chunker': 'semantic-drift'},
                'retrieval': {'agentic_weights': (1,), 'k': 4},
                'generation': {'answerer': 'extractive'}, 'run': {}}
    baseline.production_config(defaults)
    assert defaults['index']['chunker'] == 'semantic-drift'
    assert defaults['retrieval']['k'] == 4
    assert defaults['generation']['answerer'] == 'extractive'
