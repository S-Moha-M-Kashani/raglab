"""What Lodestar shipped, frozen on the day the lab moved out.

Until 2026-08-11 this was derived: `config.PRODUCTION_CONFIG` imported
`lodestar_brain.config` and `lodestar_brain.retrieval` and read their constants,
so a preset claiming to be the real system could not drift from it. The lab no
longer shares a repository with that code, so the values are literals and the
guarantee is gone. What replaces it is honesty: the date and the commit are part
of the label the panel shows, so the preset presents itself as a snapshot rather
than as the truth.

If Lodestar's retrieval changes, this file is wrong and nothing here will notice.
Re-snapshot it deliberately, from the constants named in the comments below —
`retrieval.py` for the numbers, `config.Settings` for the two backend defaults.
"""

SNAPSHOT_DATE = '2026-08-11'
# lodestar_brain/retrieval.py at this commit; the two Settings defaults below
# came from lodestar_brain/config.py in the same tree (development, fa959e2).
SNAPSHOT_COMMIT = '04cafac'
LABEL = f'the shipped assistant (snapshot {SNAPSHOT_DATE})'


def production_config(defaults: dict) -> dict:
    """The shipped pipeline, over `LabConfig().to_dict()`.

    Two honest differences from the lab's measured winner are worth naming,
    because the label says "the real system" and not "the best one we found".
    Lodestar splits with a recursive 500/100 splitter, so its mirror here is
    `fixed-overlap` at those sizes — not `semantic-drift`, which the sweep
    preferred but the brain does not ship. And it prepends no situating header,
    so `contextual` is off. Everything else is the pipeline in
    `retrieval.CardIndex.search`: hybrid dense+BM25 fused with RRF, expanded
    queries, the Farsi time filter, a lexical rerank, then the LLM gate.

    Built over the defaults so every field is present: the panel fills its whole
    form from this, and a knob the shipped brain has no opinion on (the recency
    half-life, which its lexical reranker never reads) should read as the lab
    default rather than blank.

    The groups are copied, not aliased: this preset is built once per process and
    served on every `/api/options`, so writing through to the caller's dict would
    push the shipped values into the lab's own defaults the first time anybody
    opened the panel.
    """
    preset = {group: dict(fields) for group, fields in defaults.items()}
    preset['label'] = LABEL
    preset['index'] |= {
        'chunker': 'fixed-overlap',
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
        # A list, not the dataclass's tuple: this dict is served as JSON, where a
        # tuple arrives as a list anyway, and "what the panel receives" should
        # equal what this module holds rather than merely resemble it.
        'agentic_weights': list(preset['retrieval'].get('agentic_weights', ()))}
    preset['generation'] |= {'answerer': 'llm'}
    return preset
