"""The shipped-assistant preset, after it stopped being derived."""
import json

from raglab.configuration import lab_config as config
from raglab.evaluation import production_baseline_snapshot as baseline


def test_the_preset_is_a_dated_snapshot_and_says_so():
    # this is a unit test
    """The date and commit are part of the served value, not a comment — the
    snapshot must say it is frozen, since it can no longer prove it by
    importing the source it mirrors."""
    assert baseline.SNAPSHOT_DATE == '2026-08-11'
    assert len(baseline.SNAPSHOT_COMMIT) >= 7
    assert baseline.SNAPSHOT_DATE in baseline.LABEL
    assert 'shipped assistant' in baseline.LABEL


def test_the_preset_mirrors_what_the_production_assistant_shipped():
    # this is a unit test
    """Two honest differences from the sweep's own winner: fixed-overlap
    500/100 rather than semantic-drift, and no contextual header — the label
    says "the real system", not "the best one we found". `agentic_weights`
    is served as a **list**, not the dataclass's tuple, since this dict is
    served as JSON — the input here is a tuple, so the equality check below
    only passes if the conversion really happened. `delimiters` is pinned
    empty rather than carried over: the input sets one and the shipped
    assistant, which split on whitespace alone, must not inherit it."""
    preset = baseline.production_config({
        'index': {'chunker': 'semantic-drift', 'chunk_chars': 900,
                  'overlap': 0, 'delimiters': ('\n\n',), 'contextual': True,
                  'embedder': 'hash', 'embed_model': 'x'},
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
        'delimiters': [], 'contextual': False,
        'embedder': 'sentence-transformers', 'embed_model': ''}
    assert preset['retrieval'] == {
        'retriever': 'hybrid-rrf', 'k': 8, 'candidates': 40, 'rrf_k': 60,
        'time_filter': True, 'multi_query': True, 'hyde': False,
        'mmr_lambda': 1.0, 'reranker': 'lexical', 'rerank_depth': 20,
        'grader': 'llm', 'grade_threshold': 0.4, 'agentic_weights': [1, 2]}
    assert isinstance(preset['retrieval']['agentic_weights'], list)
    assert preset['generation'] == {'answerer': 'llm'}
    assert preset['label'] == baseline.LABEL


def test_every_field_of_the_default_config_survives():
    # this is a unit test
    """Built over the defaults, so a knob the snapshot has no opinion on (the
    recency half-life) reads as the lab default rather than blank."""
    preset = baseline.production_config({
        'index': {}, 'retrieval': {'agentic_weights': ()},
        'generation': {}, 'run': {'limit': 5, 'half_life_days': 90},
    })
    assert preset['run'] == {'limit': 5, 'half_life_days': 90}


def test_the_snapshot_does_not_mutate_the_defaults_it_was_given():
    # this is a unit test
    """The preset is built once per process and served on every /api/options —
    writing through to the caller's dict would drift the lab's own defaults."""
    defaults = {'index': {'chunker': 'semantic-drift'},
                'retrieval': {'agentic_weights': (1,), 'k': 4},
                'generation': {'answerer': 'extractive'}, 'run': {}}
    baseline.production_config(defaults)
    assert defaults['index']['chunker'] == 'semantic-drift'
    assert defaults['retrieval']['k'] == 4
    assert defaults['generation']['answerer'] == 'extractive'


def test_the_shipped_settings_archive_is_a_credential_free_snapshot():
    # this is a unit test
    archived = json.loads((config.ROOT / 'fixtures' /
                          'production_baseline_settings.json').read_text(encoding='utf-8'))
    assert archived['format'] == 'raglab-experiment'
    assert archived['version'] == 1
    assert 'evaluation' not in archived
    assert archived['settings']['config'] == config.PRODUCTION_CONFIG
    assert archived['settings']['ui'] == {
        'mode': '', 'ragas_mode': 'offline', 'limit': 0,
        'ragas_limit': 0, 'types': [],
    }

    def credential_shaped(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = key.lower().replace('_', '')
                if normalized in {'apikey', 'openrouterkey', 'password',
                                  'clientsecret', 'accesstoken', 'authorization'}:
                    return key
                found = credential_shaped(nested)
                if found:
                    return found
        if isinstance(value, list):
            for nested in value:
                found = credential_shaped(nested)
                if found:
                    return found
        return None

    assert credential_shaped(archived) is None
