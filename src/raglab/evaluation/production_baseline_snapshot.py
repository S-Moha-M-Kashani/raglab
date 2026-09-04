"""A dated snapshot of the retrieval config the production assistant shipped, not a live import.

Re-snapshot deliberately from the constants named in the comments below —
`retrieve_fuse_rerank_grade.py` for the numbers, `config.Settings` for the two backend defaults.
"""

SNAPSHOT_DATE = '2026-08-11'
# The parent assistant's retrieve_fuse_rerank_grade.py at this commit; the two Settings
# defaults below came from its lab_config.py in the same tree.
SNAPSHOT_COMMIT = '04cafac'
LABEL = f'the shipped assistant (snapshot {SNAPSHOT_DATE})'


def production_config(defaults: dict) -> dict:
    """The shipped pipeline as a full run config, built over `defaults` so every field is present."""
    # Copy per group: this preset is served on every /api/options, and writing
    # through to the caller's dict would leak these values into the lab's defaults.
    preset = {group: dict(fields) for group, fields in defaults.items()}
    preset['label'] = LABEL
    preset['index'] |= {
        # The shipped assistant packed words up to the budget and slid a
        # window: the whole document, cut by the budget alone — no stage
        # between the document and the budget, and no paragraph boundary
        # preferred over a word cut. A list rather than the dataclass's tuple,
        # like `agentic_weights` below, because this dict is served as JSON.
        'split_plan': [{'kind': 'document'}],
        'chunk_chars': 500,          # retrieval.CHUNK_SIZE
        'overlap': 100,              # retrieval.CHUNK_OVERLAP
        'contextual': False,
        'embedder': 'sentence-transformers',   # config.Settings.embedder
        'embed_model': ''}
    preset['retrieval'] |= {
        'retriever': 'hybrid-rrf',
        'k': 8,                      # retrieval.TOP_K
        'candidates': 40,            # retrieval.CANDIDATES
        'rrf_k': 60,                 # retrieval.RRF_K
        'time_filter': True,
        'multi_query': True,
        'hyde': False,
        'mmr_lambda': 1.0,
        'reranker': 'lexical',
        'rerank_depth': 20,          # retrieval.RERANK_DEPTH
        'grader': 'llm',             # config.Settings.grader
        'grade_threshold': 0.4,      # retrieval.GRADE_THRESHOLD
        # A list, matching what this dict actually serves as JSON.
        'agentic_weights': list(preset['retrieval'].get('agentic_weights', ()))}
    preset['generation'] |= {'answerer': 'llm'}
    return preset
