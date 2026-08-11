"""The summary hierarchy: grouping, summarising, and what retrieval may do with
the rows it wrote.

The corpus is Farsi, so these use `token-hash` rather than the `ascii-hash` the
rest of the offline suite reaches for: ascii-hash embeds Farsi to the zero
vector — the measurement that moved the brain's default — and a graph built over
zero vectors has no edges, so it would test the empty case forever while looking
like it tested the feature.
"""
import numpy as np
import pytest

from raglab import config, hierarchy, pipeline
from raglab.config import (GRAPH_HIERARCHIES, HIERARCHIES, IndexConfig,
                           LabConfig, LabSettings, RetrievalConfig)
from raglab.corpus import load_diary, load_ground_truth
from raglab.index import IndexRegistry

LAB_SETTINGS = LabSettings(openrouter_api_key='', llm_provider='fake')

# One session per chunk keeps these fast: the grouping is what is under test,
# not the chunker, and 167 leaves is enough for every partition to be non-trivial.
LEAVES = dict(chunker='session', embedder='token-hash', contextual=False)


@pytest.fixture(scope='module')
def diary():
    return load_diary()


@pytest.fixture(scope='module')
def registry(diary):
    return IndexRegistry(LAB_SETTINGS, diary)


# --- the fingerprint -------------------------------------------------------

# This is a unit test.
def test_a_flat_index_is_fingerprinted_exactly_as_it_was_before_hierarchies():
    """Seven new fields, and none of them may touch a flat index's name.

    The literals are the same ones `test_the_built_in_corpus_keeps_the_
    fingerprints_already_recorded` pins for the dataset field, and for the same
    reason: every `index.collection` already written into `.runs/` names an
    index a rebuild has to be able to reproduce. A new field that entered the
    hash would leave a whole leaderboard describing collections that can no
    longer be built by name.
    """
    assert IndexConfig().fingerprint() == '804444ae65db'
    assert IndexConfig().collection() == 'raglab-804444ae65db'
    assert IndexConfig(embedder='ascii-hash').fingerprint() == '9d62a8c374b6'


# This is a unit test.
def test_hierarchy_knobs_left_over_from_another_config_do_not_name_a_new_index():
    """A stale `graph_knn` in a browser tab must not cost a 167-session rebuild
    of an index that is byte-identical to the one already built. `hierarchy=''`
    means nothing below it ran."""
    assert (IndexConfig(graph_knn=99, granularity=3.0, summarizer='card',
                        hierarchy_levels=4, min_group=9).fingerprint()
            == IndexConfig().fingerprint())


# This is a unit test.
def test_a_knob_the_chosen_grouping_never_reads_does_not_cost_a_rebuild():
    """The same argument one level down: k-means builds no graph, so the number
    of graph neighbours cannot change what it stored."""
    assert (IndexConfig(hierarchy='kmeans', graph_knn=99).fingerprint()
            == IndexConfig(hierarchy='kmeans').fingerprint())
    # ...but a knob it *does* read must.
    assert (IndexConfig(hierarchy='kmeans', granularity=2.0).fingerprint()
            != IndexConfig(hierarchy='kmeans').fingerprint())


# This is a unit test.
def test_two_groupings_are_two_indexes():
    """Leiden and Louvain partition the same graph differently, so they store
    different rows and must never share a collection."""
    seen = {IndexConfig(hierarchy=name).fingerprint() for name in HIERARCHIES}
    assert len(seen) == len(HIERARCHIES)


# --- availability, refused rather than substituted -------------------------

# This is a unit test.
def test_a_grouping_whose_library_is_missing_is_refused_and_never_substituted(
        monkeypatch):
    """The embedder rule applied to partitions.

    A row labelled `leiden` that Louvain actually produced is the one artefact
    this lab must not make, because nothing else on the row contradicts it. The
    availability *check* is stubbed rather than the option list, so this asserts
    on the rule and not on which extras happen to be installed here.
    """
    monkeypatch.setattr(hierarchy, 'hierarchy_available',
                        lambda name: name != 'leiden')
    problems = LabConfig(index=IndexConfig(hierarchy='leiden')).validate()
    assert problems, 'an unavailable grouping must not validate'
    assert 'leiden' in problems[0]
    assert 'graph-index' in problems[0], 'the error has to say what to install'
    assert 'louvain' not in problems[0].lower(), 'never offer a substitute'
    # Everything else still validates: one missing wheel is not a broken lab.
    assert LabConfig(index=IndexConfig(hierarchy='louvain')).validate() == []


# This is a unit test.
def test_availability_is_verified_rather_than_asserted():
    """`available()` answers per grouping, and says what to install. networkx
    and scikit-learn are core dependencies, so those five are always true."""
    answers = hierarchy.available()
    for name in ('louvain', 'label-prop', 'raptor', 'agglomerative', 'kmeans'):
        assert answers[name]['available'], f'{name} runs on a core dependency'
    assert 'graph-index' in answers['leiden']['install']


# --- the build -------------------------------------------------------------

# This is an integration test (real in-memory index, offline hash embedder).
@pytest.mark.parametrize('name', [h for h in HIERARCHIES if h])
def test_every_grouping_writes_summaries_beside_the_leaves_it_grouped(
        registry, name):
    """Additive, always: the leaves a flat build produced are all still there,
    by id. Replacing a session with its summary is what loses information
    permanently — a summary that drops "the sixth rejection" makes the counting
    question unanswerable forever."""
    flat = registry.get(IndexConfig(**LEAVES))
    grouped = registry.get(IndexConfig(**LEAVES, hierarchy=name))

    leaf_ids = {c.id for c in flat.chunks}
    assert leaf_ids <= {c.id for c in grouped.chunks}
    summaries = [c for c in grouped.chunks if c.layer == 'summary']
    assert summaries, f'{name} produced no summary at all'
    assert len(grouped.chunks) == len(leaf_ids) + len(summaries)
    for summary in summaries:
        assert summary.text.strip()
        assert summary.level >= 1 and summary.group_id
        assert set(summary.member_ids) <= leaf_ids


# This is an integration test.
def test_the_metadata_control_reproduces_the_corpus_own_storylines(registry,
                                                                   diary):
    """The grouping that was measured and deleted on 2026-07-31, kept as a
    control so the old finding and the new one can be read off one table. It has
    to be the thing it claims to be: one group per declared thread."""
    grouped = registry.get(IndexConfig(**LEAVES, hierarchy='metadata'))
    summaries = [c for c in grouped.chunks if c.layer == 'summary']
    assert len(summaries) == len(diary['threads']) == 18


# This is an integration test.
def test_a_build_reports_what_the_grouping_did(registry):
    """"The index built" is not a result. A partition with no community
    structure makes every score under it uninformative, and that is worth
    knowing before reading them."""
    stats = registry.get(IndexConfig(**LEAVES, hierarchy='louvain')).stats
    report = stats.hierarchy
    assert report['hierarchy'] == 'louvain'
    assert report['groups'] >= 2 and report['summaries'] == report['groups']
    assert report['modularity'] is not None, 'a graph method reports modularity'
    assert report['nodes'] and report['edges'] and report['components']
    assert 0.0 < report['coverage'] <= 1.0
    assert report['per_level'][0]['level'] == 1
    assert stats.leaves < stats.chunks, 'summaries are counted as rows'

    clustered = registry.get(IndexConfig(**LEAVES, hierarchy='kmeans')).stats
    assert clustered.hierarchy['silhouette'] is not None
    assert clustered.hierarchy['modularity'] is None, 'k-means builds no graph'


# This is an integration test.
def test_a_flat_build_reports_no_hierarchy_at_all(registry):
    """None rather than an empty block: "no hierarchy" and "a hierarchy that
    found nothing" are different facts about a build."""
    assert registry.get(IndexConfig(**LEAVES)).stats.hierarchy is None


# This is an integration test.
def test_building_a_hierarchy_opens_no_socket_and_calls_no_model(monkeypatch,
                                                                 diary):
    """The decision that makes a hierarchy sweepable at all. It is asserted
    rather than assumed, because a summariser that reached for a model would
    also let the offline `fake` backend fill an index with confident invention
    that no field on the resulting row contradicts."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError('a build must not open a socket')

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(pipeline, 'lab_chat', refuse)
    fresh = IndexRegistry(LAB_SETTINGS, diary)
    index = fresh.get(IndexConfig(**LEAVES, hierarchy='louvain'))
    assert any(c.layer == 'summary' for c in index.chunks)


# This is an integration test.
def test_a_deeper_hierarchy_groups_its_own_summaries(registry):
    """Level 2 is the same operation applied to level 1's output, which is what
    makes "recursive" mean something rather than being a second parameter."""
    index = registry.get(IndexConfig(**LEAVES, hierarchy='agglomerative',
                                     hierarchy_levels=2, granularity=2.0))
    levels = {c.level for c in index.chunks if c.layer == 'summary'}
    assert levels == {1, 2}
    top = [c for c in index.chunks if c.level == 2][0]
    assert all(mid.startswith('summary:') for mid in top.member_ids)


# --- the summarisers -------------------------------------------------------

# This is a unit test.
@pytest.mark.parametrize('name', config.SUMMARIZERS)
def test_every_summariser_writes_text_from_the_group_and_calls_no_model(name):
    from raglab.chunking import Chunk
    chunks = [Chunk(id=f'c{i}', text=f'session {i} tax office installment {i}',
                    session_id=f's{i}', date=f'2026-01-{i + 1:02d}')
              for i in range(6)]
    vectors = np.eye(6, dtype=np.float32)
    idf = hierarchy._idf_of([c.text for c in chunks])
    text = hierarchy.summarize(chunks, vectors, [0, 1, 2], name, idf)
    assert text.strip()


# This is a unit test.
def test_the_card_summariser_states_the_count_rather_than_implying_it():
    """The cheapest summariser, and the one most likely to help a counting
    question: it states a number instead of asking the model to count retrieved
    chunks — the task the 2026-07-31 record identifies as the one a language
    model is worst at."""
    from raglab.chunking import Chunk
    chunks = [Chunk(id=f'c{i}', text=f'روز {i} باشگاه رفتم',
                    session_id=f's{i}', date=f'2026-04-{i + 1:02d}')
              for i in range(5)]
    idf = hierarchy._idf_of([c.text for c in chunks])
    card = hierarchy.summarize(chunks, np.eye(5, dtype=np.float32),
                               [0, 1, 2, 3], 'card', idf)
    assert 'group of 4' in card
    assert '2026-04-01' in card and '2026-04-04' in card


# --- retrieval over an index that has summaries ----------------------------

def _question(ground_truth):
    return ground_truth['questions'][0]


# This is an integration test.
def test_the_leaves_scope_retrieves_exactly_what_a_flat_index_would(registry):
    """The control has to actually be a control: if `leaves` moved a single
    context, no row using it could say whether building the summaries cost
    anything."""
    ground_truth = load_ground_truth()
    question = _question(ground_truth)
    flat = registry.get(IndexConfig(**LEAVES))
    grouped = registry.get(IndexConfig(**LEAVES, hierarchy='louvain'))

    plain = pipeline.retrieve(flat, RetrievalConfig(reranker='none'),
                              question['question_fa'], question['query_date'])
    control = pipeline.retrieve(
        grouped, RetrievalConfig(reranker='none', summary_scope='leaves'),
        question['question_fa'], question['query_date'])
    assert [c.chunk_id for c in control.contexts] == \
           [c.chunk_id for c in plain.contexts]


# This is an integration test.
def test_the_summaries_scope_retrieves_only_summaries(registry):
    ground_truth = load_ground_truth()
    question = _question(ground_truth)
    grouped = registry.get(IndexConfig(**LEAVES, hierarchy='louvain'))
    outcome = pipeline.retrieve(
        grouped, RetrievalConfig(reranker='none', summary_scope='summaries'),
        question['question_fa'], question['query_date'])
    assert outcome.contexts
    assert all(grouped.by_id[c.chunk_id].layer == 'summary'
               for c in outcome.contexts)
    assert outcome.diagnostics['contexts_by_layer']['leaf'] == 0


# This is an integration test.
def test_drill_down_expands_each_summary_to_the_members_it_stands_for(registry):
    """The mechanism the 2026-07-31 post-mortem asked for and `rollup_boost` was
    not: summaries compete only against summaries, so being outnumbered twenty
    to one by leaves cannot happen, and the members arrive as evidence the
    answerer can quote."""
    ground_truth = load_ground_truth()
    question = _question(ground_truth)
    grouped = registry.get(IndexConfig(**LEAVES, hierarchy='louvain'))
    outcome = pipeline.retrieve(
        grouped, RetrievalConfig(reranker='none', summary_scope='drill-down',
                                 max_context_chars=10 ** 7),
        question['question_fa'], question['query_date'])
    expanded = [c for c in outcome.contexts if c.expanded_from]
    assert expanded, 'drill-down retrieved a summary and expanded nothing'
    for context in expanded:
        parent = grouped.by_id[context.expanded_from]
        assert parent.layer == 'summary'
        assert context.chunk_id in parent.member_ids
    counts = outcome.diagnostics['contexts_by_layer']
    assert counts['summary'] and counts['expanded']


# This is an integration test.
def test_a_boost_promotes_a_summary_into_the_candidate_cut(registry):
    """Applied before the cut, never after. There are far more leaves than
    summaries, so a summary that had not already survived the cut could not be
    promoted into it — that version was measured in the 2026-07-30 sweep and was
    a no-op that looked like a knob."""
    ground_truth = load_ground_truth()
    question = _question(ground_truth)
    grouped = registry.get(IndexConfig(**LEAVES, hierarchy='metadata'))
    base = RetrievalConfig(reranker='none', k=5, rerank_depth=5)
    plain = pipeline.retrieve(grouped, base, question['question_fa'],
                              question['query_date'])
    boosted = pipeline.retrieve(grouped, replace_boost(base, 50.0),
                                question['question_fa'], question['query_date'])

    def summaries(outcome):
        return sum(1 for c in outcome.contexts
                   if grouped.by_id[c.chunk_id].layer == 'summary')

    assert summaries(boosted) > summaries(plain)
    assert boosted.diagnostics['summaries_boosted'] > 0


def replace_boost(cfg: RetrievalConfig, value: float) -> RetrievalConfig:
    from dataclasses import replace
    return replace(cfg, summary_boost=value)


# This is an integration test.
def test_mixed_is_the_default_so_a_hierarchy_changes_no_retrieval_by_itself():
    """Building a hierarchy must not move a number on its own — the first row is
    then a clean answer to one question, and drill-down is a second candidate
    rather than a confound in the first."""
    assert RetrievalConfig().summary_scope == 'mixed'
    assert RetrievalConfig().summary_boost == 1.0


# This is a unit test.
def test_the_store_filter_and_the_bm25_mask_agree_about_layers(registry):
    """Hybrid fusion compares two candidate pools. If the `where` clause and the
    mask disagree about which rows exist, the two halves of the search are
    silently looking at different corpora."""
    from raglab import query as query_mod
    from raglab.store import matches
    grouped = registry.get(IndexConfig(**LEAVES, hierarchy='louvain'))
    for scope in ('mixed', 'leaves', 'summaries'):
        cfg = RetrievalConfig(summary_scope=scope)
        layers, levels = pipeline.summary_filter(cfg)
        where = query_mod.layer_clause(None, layers, levels)
        mask = pipeline._mask(grouped, None, layers, levels)
        for i, chunk in enumerate(grouped.chunks):
            assert bool(mask[i]) == matches(chunk.metadata(), where), (
                f'{scope}: {chunk.id} disagrees between the two halves')


# This is a unit test.
def test_every_hierarchy_control_is_dead_until_it_means_something():
    """What the panels grey out by, including the three retrieval knobs that
    gate on an *index* field — what retrieval may do with summaries is decided
    by whether the build wrote any."""
    flat = config.dependency_state(LabConfig().to_dict())
    for key in ('index.graph_source', 'index.granularity', 'index.min_group',
                'index.summarizer', 'retrieval.summary_scope',
                'retrieval.summary_boost'):
        assert not flat[key]['enabled'], f'{key} means nothing on a flat index'
        assert flat[key]['reason'], f'{key} is greyed out with no reason given'

    louvain = config.dependency_state(
        LabConfig(index=IndexConfig(hierarchy='louvain')).to_dict())
    assert louvain['index.graph_source']['enabled']
    assert louvain['index.granularity']['enabled']
    assert louvain['retrieval.summary_scope']['enabled']

    # Label propagation is the control precisely because it has no granularity.
    label_prop = config.dependency_state(
        LabConfig(index=IndexConfig(hierarchy='label-prop')).to_dict())
    assert not label_prop['index.granularity']['enabled']
    assert label_prop['index.graph_source']['enabled']

    # k-means builds no graph, so neither edge knob applies.
    kmeans = config.dependency_state(
        LabConfig(index=IndexConfig(hierarchy='kmeans')).to_dict())
    assert not kmeans['index.graph_source']['enabled']
    assert not kmeans['index.graph_knn']['enabled']
    assert kmeans['index.granularity']['enabled']


# This is a unit test.
def test_the_graph_methods_are_named_as_chunk_graphs_and_not_as_graphrag():
    """A reader who sees `leiden` on a leaderboard row will think GraphRAG
    unless told otherwise, and GraphRAG's graph is over LLM-extracted entities.
    The help text is where that is corrected, so it is pinned."""
    text = config.HELP['index.hierarchy']
    assert 'GraphRAG' in text
    assert 'bipartite-terms' in config.HELP['index.graph_source']
    assert all(name in text for name in GRAPH_HIERARCHIES)
