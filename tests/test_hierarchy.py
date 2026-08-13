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

def test_a_flat_index_is_fingerprinted_exactly_as_it_was_before_hierarchies():
    """Seven new fields, none of which may touch a flat index's name — every
    `index.collection` already written into `.runs/` names an index a
    rebuild has to be able to reproduce."""
    assert IndexConfig().fingerprint() == '804444ae65db'
    assert IndexConfig().collection() == 'raglab-804444ae65db'
    assert IndexConfig(embedder='ascii-hash').fingerprint() == '9d62a8c374b6'


def test_hierarchy_knobs_left_over_from_another_config_do_not_name_a_new_index():
    """A stale `graph_knn` in a browser tab must not cost a 167-session rebuild
    of an index that is byte-identical to the one already built. `hierarchy=''`
    means nothing below it ran."""
    assert (IndexConfig(graph_knn=99, granularity=3.0, summarizer='card',
                        hierarchy_levels=4, min_group=9).fingerprint()
            == IndexConfig().fingerprint())


def test_a_knob_the_chosen_grouping_never_reads_does_not_cost_a_rebuild():
    """The same argument one level down: k-means builds no graph, so the number
    of graph neighbours cannot change what it stored."""
    assert (IndexConfig(hierarchy='kmeans', graph_knn=99).fingerprint()
            == IndexConfig(hierarchy='kmeans').fingerprint())
    # ...but a knob it *does* read must.
    assert (IndexConfig(hierarchy='kmeans', granularity=2.0).fingerprint()
            != IndexConfig(hierarchy='kmeans').fingerprint())


def test_two_groupings_are_two_indexes():
    """Leiden and Louvain partition the same graph differently, so they store
    different rows and must never share a collection."""
    seen = {IndexConfig(hierarchy=name).fingerprint() for name in HIERARCHIES}
    assert len(seen) == len(HIERARCHIES)


# --- availability, refused rather than substituted -------------------------

def test_a_grouping_whose_library_is_missing_is_refused_and_never_substituted(
        monkeypatch):
    """The availability *check* is stubbed rather than the option list, so
    this asserts on the rule and not on which extras happen to be installed
    here."""
    monkeypatch.setattr(hierarchy, 'hierarchy_available',
                        lambda name: name != 'leiden')
    problems = LabConfig(index=IndexConfig(hierarchy='leiden')).validate()
    assert problems, 'an unavailable grouping must not validate'
    assert 'leiden' in problems[0]
    assert 'graph-index' in problems[0], 'the error has to say what to install'
    assert 'louvain' not in problems[0].lower(), 'never offer a substitute'
    assert LabConfig(index=IndexConfig(hierarchy='louvain')).validate() == []


def test_availability_is_verified_rather_than_asserted():
    """`available()` answers per grouping, and says what to install. networkx
    and scikit-learn are core dependencies, so those five are always true."""
    answers = hierarchy.available()
    for name in ('louvain', 'label-prop', 'raptor', 'agglomerative', 'kmeans'):
        assert answers[name]['available'], f'{name} runs on a core dependency'
    assert 'graph-index' in answers['leiden']['install']


# --- the build -------------------------------------------------------------

# Real in-memory index, offline hash embedder.
@pytest.mark.parametrize('name', [h for h in HIERARCHIES if h])
def test_every_grouping_writes_summaries_beside_the_leaves_it_grouped(
        registry, name):
    """Additive, always: replacing a leaf with its summary loses information
    permanently, unlike keeping both."""
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


def test_the_metadata_control_reproduces_the_corpus_own_storylines(registry,
                                                                   diary):
    """Kept as a control against the old, deleted rollups: it has to be the
    thing it claims to be, one group per declared thread."""
    grouped = registry.get(IndexConfig(**LEAVES, hierarchy='metadata'))
    summaries = [c for c in grouped.chunks if c.layer == 'summary']
    assert len(summaries) == len(diary['threads']) == 18


def test_a_build_reports_what_the_grouping_did(registry):
    """A partition with no community structure makes every score under it
    uninformative, and that is worth knowing before reading them."""
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


def test_a_flat_build_reports_no_hierarchy_at_all(registry):
    """None rather than an empty block: "no hierarchy" and "a hierarchy that
    found nothing" are different facts about a build."""
    assert registry.get(IndexConfig(**LEAVES)).stats.hierarchy is None


def test_building_a_hierarchy_opens_no_socket_and_calls_no_model(monkeypatch,
                                                                 diary):
    """The decision that makes a hierarchy sweepable at all — a summariser
    that reached for a model would let the offline `fake` backend fill an
    index with confident invention no field on the row contradicts."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError('a build must not open a socket')

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(pipeline, 'lab_chat', refuse)
    fresh = IndexRegistry(LAB_SETTINGS, diary)
    index = fresh.get(IndexConfig(**LEAVES, hierarchy='louvain'))
    assert any(c.layer == 'summary' for c in index.chunks)


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


def test_the_card_summariser_states_the_count_rather_than_implying_it():
    """States a number instead of asking a model to count retrieved chunks —
    counting is a task a language model is bad at."""
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


def test_drill_down_expands_each_summary_to_the_members_it_stands_for(registry):
    """Summaries compete only against summaries, so being outnumbered by
    leaves cannot happen, and the members arrive as evidence the answerer
    can quote."""
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


def test_a_boost_promotes_a_summary_into_the_candidate_cut(registry):
    """Applied before the cut, never after: there are far more leaves than
    summaries, so a summary that had not already survived the cut could not
    be promoted into it — that would be a no-op that looked like a knob."""
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


def test_mixed_is_the_default_so_a_hierarchy_changes_no_retrieval_by_itself():
    """Building a hierarchy must not move a number on its own — the first row is
    then a clean answer to one question, and drill-down is a second candidate
    rather than a confound in the first."""
    assert RetrievalConfig().summary_scope == 'mixed'
    assert RetrievalConfig().summary_boost == 1.0


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
        mask = pipeline._allowed(grouped, None, layers, levels)
        for i, chunk in enumerate(grouped.chunks):
            assert bool(mask[i]) == matches(chunk.metadata(), where), (
                f'{scope}: {chunk.id} disagrees between the two halves')


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


# Real evaluation, offline embedder, fake LLM.
def test_a_run_records_whether_the_hierarchy_was_actually_retrieved(diary):
    """"The hierarchy was configured" and "the hierarchy was retrieved" are
    different facts, and a row that scores flat is uninterpretable without
    the second."""
    from raglab import evaluate
    ground_truth = load_ground_truth()
    registry = IndexRegistry(LAB_SETTINGS, diary)
    result = evaluate.run_eval(
        registry, ground_truth,
        LabConfig(index=IndexConfig(**LEAVES, hierarchy='metadata'),
                  retrieval=RetrievalConfig(reranker='none', summary_boost=20.0)),
        LAB_SETTINGS, limit=4, ragas_mode='off')
    assert all('n_summaries' in row for row in result.rows)
    assert sum(row['n_summaries'] for row in result.rows) > 0, (
        'a boosted metadata hierarchy that reached no context would make this '
        'metric untestable, not merely unused')
    assert 'n_summaries' in result.summary['overall']


# FastAPI TestClient.
def test_the_build_route_refuses_an_unavailable_grouping_by_name(monkeypatch):
    """A 400 naming what to install, not a 500 from an import three frames
    down. Both run routes already apply one screen; a build applies the half of
    it that describes what gets stored."""
    from fastapi.testclient import TestClient

    from raglab import server
    monkeypatch.setattr(hierarchy, 'hierarchy_available',
                        lambda name: name != 'leiden')
    client = TestClient(server.create_app())
    response = client.post('/api/indexes',
                           json={'index': {'hierarchy': 'leiden'}})
    assert response.status_code == 400
    assert 'graph-index' in response.json()['detail']


# FastAPI TestClient.
def test_both_panels_are_served_the_hierarchy_lists_rather_than_keeping_them():
    from fastapi.testclient import TestClient

    from raglab import server
    options = TestClient(server.create_app()).get('/api/options').json()
    assert options['hierarchies'] == list(HIERARCHIES)
    assert options['summary_scopes'] == list(config.SUMMARY_SCOPES)
    assert options['summarizers'] == list(config.SUMMARIZERS)
    assert options['hierarchy_support']['louvain']['available'] is True
    assert 'retrieval.summary_scope' in options['dependencies']


# Reads the panel's own source.
def test_the_panel_resolves_a_dependency_chain_the_way_the_service_does():
    """Resolving the dependency rules happens per keystroke in the browser
    without a round trip, so the resolution exists twice and the two copies
    must agree — a single-level resolver once left `graph_knn` live under a
    grouping that builds no graph at all, because it only asked whether the
    edge *source* builds kNN edges rather than resolving transitively."""
    from pathlib import Path
    panel = (Path(__file__).resolve().parents[1] / 'src' / 'raglab' / 'static'
             / 'index.html').read_text(encoding='utf-8')
    assert 'function dependencyState(' in panel, (
        'the panel must resolve chains, not just single rules')
    assert 'resolve(rule.field' in panel, 'the resolution has to be transitive'

    # And the rule it exists for, from the service side.
    kmeans = config.dependency_state(
        LabConfig(index=IndexConfig(hierarchy='kmeans')).to_dict())
    assert not kmeans['index.graph_knn']['enabled']
    assert kmeans['index.graph_knn']['reason'] == \
        kmeans['index.graph_source']['reason'], (
            'a control killed by its owner reports the owner\'s reason — its '
            'own would describe a condition that is not why it is dead')


def test_the_graph_methods_are_named_as_chunk_graphs_and_not_as_graphrag():
    """A reader who sees `leiden` on a leaderboard row will think GraphRAG
    unless told otherwise, and GraphRAG's graph is over LLM-extracted entities.
    The help text is where that is corrected, so it is pinned."""
    text = config.HELP['index.hierarchy']
    assert 'GraphRAG' in text
    assert 'bipartite-terms' in config.HELP['index.graph_source']
    assert all(name in text for name in GRAPH_HIERARCHIES)


# --- determinism: one fingerprint must name one index ------------------------

# Every token here is shared by exactly two documents, so every token has the
# same IDF — a pure tie on the per-chunk "top terms by IDF" cut, which is the
# state the ordering bug lives in.
TIED_DOCS = 20


def _tied_corpus() -> list[str]:
    return [' '.join(f'p{min(i, j):02d}x{max(i, j):02d}'
                     for j in range(TIED_DOCS) if j != i)
            for i in range(TIED_DOCS)]


def test_terms_that_tie_on_idf_are_chosen_by_a_stated_rule():
    """A tie must be broken by something written down, not by `set`
    iteration order, which Python randomises per process — everything
    downstream (edges, partition, summaries, fingerprint) follows it. The
    rule asserted here is lexicographic: among terms of equal IDF, the ones
    that sort first win."""
    postings, idf = hierarchy._term_postings(_tied_corpus(), top_terms=12)

    # the premise: every candidate really does tie, so the cut is arbitrary
    # unless something breaks it
    assert len({round(v, 12) for v in idf.values()}) == 1, \
        'the corpus no longer produces a tie, so it tests nothing'

    for doc in range(TIED_DOCS):
        chosen = sorted(t for t, docs in postings.items() if doc in docs)
        candidates = sorted(t for t in idf
                            if t.startswith(f'p{doc:02d}x') or t.endswith(f'x{doc:02d}'))
        assert len(chosen) == 12, f'doc {doc} kept {len(chosen)} terms, expected 12'
        assert chosen == candidates[:12], (
            f'doc {doc} kept an arbitrary twelve of its {len(candidates)} tied '
            'terms rather than the twelve that sort first')


# Two subprocesses, so it tests the thing that actually varies:
# PYTHONHASHSEED, which is fixed within any one process.
def test_the_same_corpus_builds_the_same_graph_in_a_different_process():
    """One fingerprint must name one index, across processes and not merely
    within one. `SEED` fixes Louvain's own RNG, not the order of the input
    it is handed — and string hashing is fixed for the life of a process, so
    only two subprocesses can see this vary at all."""
    import json
    import os
    import subprocess
    import sys

    program = (
        'import json, numpy as np;'
        'from raglab.hierarchy import build_graph;'
        f'n = {TIED_DOCS};'
        "texts = [' '.join('p%02dx%02d' % (min(i, j), max(i, j))"
        '           for j in range(n) if j != i) for i in range(n)];'
        "graph = build_graph(texts, np.zeros((n, 4), dtype=np.float32), 'lexical', 0);"
        "print(json.dumps(sorted([min(a, b), max(a, b)] for a, b in graph.edges())))"
    )

    def edges_under(seed: str) -> list:
        env = dict(os.environ, PYTHONHASHSEED=seed)
        done = subprocess.run([sys.executable, '-c', program], env=env,
                              capture_output=True, encoding='utf-8', timeout=180)
        assert done.returncode == 0, done.stderr[-2000:]
        return json.loads(done.stdout.strip().splitlines()[-1])

    first, second = edges_under('1'), edges_under('2')
    assert first, 'the corpus produced no lexical edges, so it tests nothing'
    assert first == second, (
        f'the same corpus produced {len(first)} edges in one process and '
        f'{len(second)} in another, sharing '
        f'{len(set(map(tuple, first)) & set(map(tuple, second)))} — a build is '
        'not reproducible, so its fingerprint names an index that cannot be '
        'rebuilt')
