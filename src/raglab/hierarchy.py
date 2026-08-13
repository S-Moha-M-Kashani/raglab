"""Grouping chunks, and writing one summary per group — all of it offline.

The lab shipped five summary layers and deleted every one of them on
2026-07-31, because the candidate that removed them scored within 0.006 of the
baseline. This module is not that revert. Two things are different, and both
come out of the post-mortem in `docs/rag-architecture.md`:

* **The groups are discovered, not declared.** The deleted rollups grouped by
  what the corpus already said about itself — session, month, storyline. Here a
  group is a community in a graph built over the chunks, or a cluster of their
  vectors. `hierarchy='metadata'` keeps the old grouping as a *control*, so the
  old finding and the new one can be read off the same table instead of seven
  months apart.
* **Nothing here calls a model.** All four summarisers are extractive. A build
  that made LLM calls would take hours rather than seconds — so nobody would
  sweep it, which is the only reason to put a thing in a lab — and it would let
  the offline `fake` backend fill a collection with confident invention that no
  field on the resulting row would contradict.

**These are not GraphRAG, and the help text says so out loud.** GraphRAG's graph
is over entities and relations that a model extracted. There is no offline
entity extractor for Farsi, so the nodes here are chunks; `bipartite-terms`
promotes rare words to nodes as well, which is the closest honest analogue and
the only source under which a community has a nameable subject.

Everything is deterministic. Louvain and k-means both take a seed, fixed here
rather than exposed, because a grouping that changes between two builds of the
same fingerprint would make an index name a lie.

**A seed was not enough, and this file claimed otherwise until 2026-08-13.** A
seed fixes a method's own RNG; it says nothing about the order of the input the
method is handed. `_term_postings` cut its per-chunk top terms with a stable sort
over a `set`, so terms tying on IDF — which on a real corpus is most of them —
were kept in hash order, and Python randomises string hashing per process. Three
fresh processes built the diary under one config and one fingerprint into 8, 8
and 6 groups. The tie-break is now the token itself, and
`test_the_same_corpus_builds_the_same_graph_in_a_different_process` holds the
line from a second process, because a single-process test cannot see this class
of bug at all.
"""
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import textnorm
from .chunking import Chunk

# One seed for everything that would otherwise wander. Not a knob: two builds of
# one fingerprint must produce one index, and a seed in the config would be a
# knob whose only effect is to make that untrue.
SEED = 20260812

# What to install when a grouping's library is missing. Only Leiden has one —
# everything else runs on networkx and scikit-learn, which are core.
EXTRAS = {
    'leiden': "uv sync --extra graph-index",
    'louvain': 'nothing — networkx is a core dependency',
    'label-prop': 'nothing — networkx is a core dependency',
    'raptor': 'nothing — scikit-learn is a core dependency',
    'agglomerative': 'nothing — scikit-learn is a core dependency',
    'kmeans': 'nothing — scikit-learn is a core dependency',
    'metadata': 'nothing — it reads the corpus',
}


def hierarchy_available(name: str) -> bool:
    """Whether this grouping can actually run *here*.

    Verified by import rather than guessed from a list, for the reason the
    embedder catalogue is verified: NA has to mean one thing — this
    installation cannot load it — or it rots into meaning nothing.
    """
    if not name:
        return True
    if name == 'leiden':
        try:
            import igraph            # noqa: F401
            import leidenalg         # noqa: F401
            return True
        except Exception:
            return False
    if name in ('louvain', 'label-prop'):
        try:
            import networkx          # noqa: F401
            return True
        except Exception:
            return False
    if name in ('raptor', 'agglomerative', 'kmeans'):
        try:
            import sklearn           # noqa: F401
            return True
        except Exception:
            return False
    return name == 'metadata'


def available() -> dict:
    """Every grouping → whether this installation can run it, and what to
    install when it cannot. Served to both panels, so neither guesses."""
    return {name: {'available': hierarchy_available(name),
                   'install': EXTRAS.get(name, '')}
            for name in EXTRAS}


# --- the graph -------------------------------------------------------------

def _knn_edges(vectors: np.ndarray, k: int) -> dict[tuple[int, int], float]:
    """Each chunk joined to its `k` nearest neighbours by cosine.

    Symmetric by construction: an edge is kept once, under the ordered pair, and
    carries the larger of the two similarities — a mutual-kNN rule would drop
    the long-tail chunk that only *one* other chunk considers a neighbour, which
    on a diary is exactly the unusual day worth finding.
    """
    edges: dict[tuple[int, int], float] = {}
    if vectors.shape[0] < 2:
        return edges
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms < 1e-12, 1.0, norms)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -2.0)
    k = min(k, similarity.shape[0] - 1)
    for i, row in enumerate(similarity):
        for j in np.argpartition(-row, k - 1)[:k] if k > 0 else []:
            weight = float(row[j])
            if weight <= 0:
                continue        # a zero-vector embedder joins nothing
            key = (i, int(j)) if i < j else (int(j), i)
            edges[key] = max(edges.get(key, 0.0), weight)
    return edges


def _term_postings(texts: list[str], top_terms: int = 12
                   ) -> tuple[dict[str, list[int]], dict[str, float]]:
    """The rare words each chunk is *about*, and how rare each one is.

    Only the top `top_terms` by IDF per chunk: every chunk shares its common
    words with every other, so keeping them would join the whole corpus into
    one community and measure nothing.
    """
    tokenised = [textnorm.tokens(text) for text in texts]
    document_count: dict[str, int] = defaultdict(int)
    for tokens in tokenised:
        for token in set(tokens):
            document_count[token] += 1
    n = max(1, len(tokenised))
    idf = {token: math.log(1 + (n - count + 0.5) / (count + 0.5))
           for token, count in document_count.items()}
    postings: dict[str, list[int]] = defaultdict(list)
    for i, tokens in enumerate(tokenised):
        # The token itself is the tie-break, and it is load bearing. IDF is a
        # function of document frequency, so on any real corpus a great many
        # terms tie exactly — and `sorted` is stable, which means ties used to
        # keep the order `set` iteration happened to offer. Python randomises
        # string hashing per process, so *which* of the tied terms survived this
        # cut was different every run, and the lexical edges, the graph, the
        # partition and the summaries all followed it.
        #
        # Measured 2026-08-13 before the fix: three fresh processes built the
        # diary under one identical config and one identical fingerprint
        # (`raglab-6561f330c7c8`) into 8, 8 and 6 groups, at modularity 0.2657,
        # 0.2689 and 0.2732. `SEED` never covered this — it fixes Louvain's own
        # RNG, not the order of the input it is handed — so an index name was a
        # claim no rebuild could honour, which is the one thing a fingerprint
        # exists to prevent.
        ranked = sorted(set(tokens),
                        key=lambda t: (-idf.get(t, 0.0), t))[:top_terms]
        for token in ranked:
            postings[token].append(i)
    return postings, idf


def _lexical_edges(texts: list[str]) -> dict[tuple[int, int], float]:
    """Chunks sharing a rare word, weighted by how rare it is.

    A term appearing in more than a fifth of the corpus is skipped: it would
    add a clique of that size and swamp the partition.
    """
    postings, idf = _term_postings(texts)
    edges: dict[tuple[int, int], float] = defaultdict(float)
    ceiling = max(2, len(texts) // 5)
    for token, docs in postings.items():
        if len(docs) < 2 or len(docs) > ceiling:
            continue
        weight = idf.get(token, 0.0)
        for a_index, a in enumerate(docs):
            for b in docs[a_index + 1:]:
                edges[(a, b)] += weight
    return dict(edges)


def _normalised(edges: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Edge weights to [0,1], so a cosine and an IDF sum can be added together.
    They share no scale, exactly as a BM25 score and a cosine do not — the same
    reason RRF fuses ranks rather than scores."""
    if not edges:
        return edges
    high = max(edges.values())
    if high <= 0:
        return {key: 0.0 for key in edges}
    return {key: value / high for key, value in edges.items()}


def build_graph(texts: list[str], vectors: np.ndarray, source: str, knn: int):
    """The chunk graph, as a networkx Graph. Node ids are chunk indices.

    `bipartite-terms` returns a graph over chunks *and* terms; the caller
    projects the communities back onto the chunk nodes, which is what makes a
    community have a nameable subject.
    """
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(len(texts)))
    if source == 'bipartite-terms':
        postings, idf = _term_postings(texts)
        ceiling = max(2, len(texts) // 5)
        for token, docs in postings.items():
            if len(docs) < 2 or len(docs) > ceiling:
                continue
            node = f'term:{token}'
            graph.add_node(node, term=True)
            for doc in docs:
                graph.add_edge(doc, node, weight=idf.get(token, 1.0))
        return graph
    edges: dict[tuple[int, int], float] = {}
    if source in ('knn', 'hybrid'):
        edges = dict(_normalised(_knn_edges(vectors, knn)))
    if source in ('lexical', 'hybrid'):
        for key, weight in _normalised(_lexical_edges(texts)).items():
            edges[key] = edges.get(key, 0.0) + weight
    for (a, b), weight in edges.items():
        graph.add_edge(a, b, weight=weight)
    return graph


# --- partitioning ----------------------------------------------------------

def _partition_graph(graph, method: str, granularity: float) -> list[list[int]]:
    """Communities as lists of chunk indices. Term nodes are dropped on the way
    out, so `bipartite-terms` and the other sources return the same shape."""
    import networkx as nx
    if graph.number_of_nodes() == 0:
        return []
    if method == 'leiden':
        communities = _leiden(graph, granularity)
    elif method == 'label-prop':
        communities = [set(c) for c in
                       nx.algorithms.community.asyn_lpa_communities(
                           graph, weight='weight', seed=SEED)]
    else:   # 'louvain'
        communities = nx.algorithms.community.louvain_communities(
            graph, weight='weight', resolution=granularity, seed=SEED)
    out = []
    for community in communities:
        members = sorted(node for node in community if isinstance(node, int))
        if members:
            out.append(members)
    return out


def _leiden(graph, granularity: float) -> list[set]:
    """Leiden over an igraph copy of the same graph.

    A copy rather than a rewrite of the construction above, so every source
    feeds every method and the only thing that differs between a `leiden` row
    and a `louvain` row is the partition — which is the comparison.
    """
    import igraph
    import leidenalg
    nodes = list(graph.nodes())
    position = {node: i for i, node in enumerate(nodes)}
    g = igraph.Graph(n=len(nodes))
    weights = []
    for a, b, data in graph.edges(data=True):
        g.add_edge(position[a], position[b])
        weights.append(float(data.get('weight', 1.0)))
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights=weights or None, resolution_parameter=granularity, seed=SEED)
    return [{nodes[i] for i in community} for community in partition]


def _cluster_count(n: int, granularity: float) -> int:
    """`granularity × √(n/2)`, the usual rule of thumb — so 1.0 is the group
    count anyone would have picked by hand, and the dial reads the same way it
    does for the graph methods (higher = more, smaller groups)."""
    base = math.sqrt(max(1, n) / 2.0)
    return max(2, min(n, int(round(max(0.05, granularity) * base))))


def _cluster_vectors(vectors: np.ndarray, method: str,
                     granularity: float) -> list[list[int]]:
    """Groups from the chunk vectors. RAPTOR's GMM is soft — a chunk can sit in
    two groups, which no partition here can express — so it is the one method
    that can put a chunk under two summaries."""
    n = vectors.shape[0]
    if n < 2:
        return [list(range(n))] if n else []
    k = _cluster_count(n, granularity)
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=SEED,
                        n_init=10).fit_predict(vectors)
        return _by_label(labels, k)
    if method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=k,
                                         linkage='ward').fit_predict(vectors)
        return _by_label(labels, k)
    # 'raptor' — soft assignment, so membership is a probability threshold
    # rather than an argmax. 1/k is "more likely than uniform chance", which is
    # the weakest claim that still means something.
    from sklearn.mixture import GaussianMixture
    reduced = _reduced(vectors)
    model = GaussianMixture(n_components=k, random_state=SEED,
                            covariance_type='diag', reg_covar=1e-4)
    probabilities = model.fit(reduced).predict_proba(reduced)
    floor = 1.0 / k
    groups = []
    for column in range(k):
        members = [i for i in range(n) if probabilities[i, column] >= floor]
        if members:
            groups.append(members)
    return groups


def _reduced(vectors: np.ndarray) -> np.ndarray:
    """PCA down to something a mixture model can fit. A GMM over 768 dimensions
    on a thousand points is fitting more parameters than it has data for; UMAP
    is what RAPTOR uses and is a heavy dependency for a step whose job is only
    to stop the covariance from being singular."""
    from sklearn.decomposition import PCA
    width = min(vectors.shape[1], max(2, min(50, vectors.shape[0] - 1)))
    if width >= vectors.shape[1]:
        return vectors
    return PCA(n_components=width, random_state=SEED).fit_transform(vectors)


def _by_label(labels, k: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        groups[int(label)].append(i)
    return [g for g in groups if g]


def _metadata_groups(chunks: list[Chunk]) -> list[list[int]]:
    """The deleted rollup, as a control: one group per storyline the corpus
    declares. Threads and not topics — the thread rollup is the one that was
    measured and deleted, and topics are per-session and noisier, so grouping on
    them would be a new thing wearing the control's name."""
    by_thread: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        for thread in chunk.threads:
            by_thread[thread].append(i)
    return [members for _, members in sorted(by_thread.items())]


# --- the summaries ---------------------------------------------------------

@dataclass
class Group:
    """One group of chunk indices at one level, and what it became."""
    level: int
    group_id: str
    members: list[int]
    label: str = ''


def _centroid_order(vectors: np.ndarray, members: list[int]) -> list[int]:
    """Members ordered by closeness to their own centre."""
    block = vectors[members]
    centre = block.mean(axis=0)
    norm = np.linalg.norm(centre)
    if norm < 1e-12:
        return list(members)
    scores = block @ (centre / norm)
    return [members[i] for i in np.argsort(-scores)]


def _sentences(text: str) -> list[str]:
    found = textnorm.sentences(text)
    return found or ([text] if text.strip() else [])


def _idf_of(texts: list[str]) -> dict[str, float]:
    _, idf = _term_postings(texts, top_terms=10**6)
    return idf


def summarize(chunks: list[Chunk], vectors: np.ndarray, members: list[int],
              summarizer: str, idf: dict[str, float], budget: int = 900) -> str:
    """One group → one piece of text, extractively.

    `budget` is characters. It is not a knob: a summary long enough to dominate
    its own members in the search is not a summary, and the number that matters
    — how a summary competes against a leaf — is `summary_boost` on the
    retrieval side, where it can be swept without a rebuild.
    """
    texts = [chunks[i].body for i in members]
    if summarizer == 'card':
        return _card(chunks, members, idf)
    if summarizer == 'centroid':
        picked = _centroid_order(vectors, members)
        return _fit([chunks[i].body for i in picked], budget)
    if summarizer == 'mmr':
        from .retrieval import mmr
        block = vectors[members]
        centre = block.mean(axis=0)
        norm = np.linalg.norm(centre) or 1.0
        relevance = (block @ (centre / norm)).astype(np.float32)
        order = mmr(block.astype(np.float32), relevance, len(members), 0.5)
        return _fit([chunks[members[i]].body for i in order], budget)
    # 'lead-idf' — the sentences that cover the most rare words, greedily, and
    # never the same word twice: coverage is the point, so a second sentence
    # about the term already covered adds nothing.
    scored: list[tuple[float, str]] = []
    for text in texts:
        for sentence in _sentences(text):
            tokens = set(textnorm.tokens(sentence))
            if tokens:
                scored.append((sum(idf.get(t, 0.0) for t in tokens), sentence))
    scored.sort(key=lambda pair: -pair[0])
    taken, seen, used = [], set(), 0
    for _, sentence in scored:
        tokens = set(textnorm.tokens(sentence))
        if tokens <= seen:
            continue
        if used + len(sentence) > budget and taken:
            break
        taken.append(sentence)
        seen |= tokens
        used += len(sentence)
    return ' '.join(taken) or _fit(texts, budget)


def _card(chunks: list[Chunk], members: list[int], idf: dict[str, float]) -> str:
    """No prose: the terms, the span, the count, the sessions.

    The cheapest summariser and plausibly the most useful one for a counting
    question, because it *states* a number instead of asking the model to count
    retrieved chunks — the task the 2026-07-31 record identifies as the one a
    language model is worst at.
    """
    weights: dict[str, float] = defaultdict(float)
    for i in members:
        for token in set(textnorm.tokens(chunks[i].body)):
            weights[token] += idf.get(token, 0.0)
    terms = [t for t, _ in sorted(weights.items(), key=lambda kv: -kv[1])[:12]]
    dates = sorted({chunks[i].date for i in members if chunks[i].date})
    sessions = sorted({chunks[i].session_id for i in members
                       if chunks[i].session_id})
    span = f'{dates[0]} … {dates[-1]}' if dates else '—'
    return (f'[group of {len(members)} · {span} · {len(sessions)} sessions]\n'
            f'{" ".join(terms)}\n'
            f'{" ".join(sessions[:40])}')


def _fit(texts: list[str], budget: int) -> str:
    out, used = [], 0
    for text in texts:
        if out and used + len(text) > budget:
            break
        out.append(text)
        used += len(text)
    return '\n'.join(out)


# --- the whole thing -------------------------------------------------------

@dataclass
class HierarchyStats:
    """What the grouping did. Reported because "the index built" is not a
    result: a partition with modularity 0.05 found no community structure, and
    the retrieval scores under it are uninformative — which is worth knowing
    before reading them rather than after."""
    hierarchy: str = ''
    graph_source: str = ''
    summarizer: str = ''
    levels: int = 0
    groups: int = 0
    summaries: int = 0
    per_level: list = field(default_factory=list)
    modularity: float | None = None      # graph methods
    silhouette: float | None = None      # clustering methods
    coverage: float = 0.0                # leaves inside a summarised group
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    components: int = 0
    avg_summary_chars: float = 0.0
    seconds: float = 0.0
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _group_once(chunks: list[Chunk], vectors: np.ndarray, cfg,
                stats: HierarchyStats) -> list[list[int]]:
    """One level of grouping, by whichever family `cfg.hierarchy` names."""
    if cfg.hierarchy == 'metadata':
        return _metadata_groups(chunks)
    if cfg.hierarchy in ('raptor', 'agglomerative', 'kmeans'):
        groups = _cluster_vectors(vectors, cfg.hierarchy, cfg.granularity)
        if stats.silhouette is None:
            stats.silhouette = _silhouette(vectors, groups)
        return groups
    graph = build_graph([c.text for c in chunks], vectors,
                        cfg.graph_source, cfg.graph_knn)
    groups = _partition_graph(graph, cfg.hierarchy, cfg.granularity)
    if not stats.nodes:
        import networkx as nx
        stats.nodes = graph.number_of_nodes()
        stats.edges = graph.number_of_edges()
        stats.density = round(nx.density(graph), 5) if stats.nodes > 1 else 0.0
        stats.components = nx.number_connected_components(graph)
        stats.modularity = _modularity(graph, groups)
    return groups


def _modularity(graph, groups: list[list[int]]) -> float | None:
    import networkx as nx
    if not groups or graph.number_of_edges() == 0:
        return None
    covered = [set(g) for g in groups]
    loose = set(graph.nodes()) - {node for g in covered for node in g}
    partition = covered + [{node} for node in loose]
    try:
        return round(float(nx.algorithms.community.modularity(
            graph, partition, weight='weight')), 4)
    except Exception:
        # An overlapping partition (RAPTOR's soft groups reaching a graph
        # method through a later level) has no defined modularity. Reporting
        # nothing is right; failing a build over a statistic is not.
        return None


def _silhouette(vectors: np.ndarray, groups: list[list[int]]) -> float | None:
    from sklearn.metrics import silhouette_score
    labels = np.full(vectors.shape[0], -1)
    for label, members in enumerate(groups):
        for i in members:
            labels[i] = label       # a soft member counts under its last group
    keep = labels >= 0
    if keep.sum() < 3 or len(set(labels[keep].tolist())) < 2:
        return None
    try:
        return round(float(silhouette_score(vectors[keep], labels[keep])), 4)
    except Exception:
        return None


def build(chunks: list[Chunk], vectors: np.ndarray, cfg, embedder,
          stats: HierarchyStats) -> list[Chunk]:
    """Every summary row this config asks for, over the leaves it is given.

    Returns the new chunks only — the caller keeps the leaves, always. Level 1
    groups the leaves; level 2 groups level 1's summaries, and so on, so a
    deeper hierarchy is the same operation applied to its own output.
    """
    import time
    started = time.time()
    stats.hierarchy = cfg.hierarchy
    stats.graph_source = (cfg.graph_source if cfg.hierarchy in
                          ('louvain', 'leiden', 'label-prop') else '')
    stats.summarizer = cfg.summarizer
    idf = _idf_of([c.text for c in chunks])

    summaries: list[Chunk] = []
    current, current_vectors = chunks, vectors
    covered: set[str] = set()
    levels = max(1, cfg.hierarchy_levels)
    for level in range(1, levels + 1):
        if len(current) < max(2, cfg.min_group):
            stats.notes.append(
                f'level {level}: only {len(current)} rows to group — stopped')
            break
        groups = [g for g in _group_once(current, current_vectors, cfg, stats)
                  if len(g) >= cfg.min_group]
        if not groups:
            stats.notes.append(
                f'level {level}: no group reached min_group={cfg.min_group}')
            break
        written: list[Chunk] = []
        for number, members in enumerate(groups):
            group_id = f'h{level}-{number:03d}'
            text = summarize(current, current_vectors, members,
                             cfg.summarizer, idf)
            if not text.strip():
                continue
            written.append(_summary_chunk(current, members, text, level,
                                          group_id))
            if level == 1:
                covered.update(current[i].id for i in members)
        if not written:
            break
        stats.per_level.append({
            'level': level, 'groups': len(written),
            'min': min(len(g) for g in groups),
            'median': int(np.median([len(g) for g in groups])),
            'max': max(len(g) for g in groups),
            'singletons': sum(1 for g in groups if len(g) == 1)})
        summaries.extend(written)
        if level < levels:
            current = written
            current_vectors = np.array(
                embedder.embed([c.text for c in written]), dtype=np.float32)

    stats.levels = len(stats.per_level)
    stats.groups = sum(entry['groups'] for entry in stats.per_level)
    stats.summaries = len(summaries)
    stats.coverage = round(len(covered) / max(1, len(chunks)), 4)
    stats.avg_summary_chars = (round(float(np.mean([len(c.text)
                                                    for c in summaries])), 1)
                               if summaries else 0.0)
    stats.seconds = round(time.time() - started, 2)
    return summaries


def _summary_chunk(members_of: list[Chunk], members: list[int], text: str,
                   level: int, group_id: str) -> Chunk:
    """A summary row, carrying the union of its members' date span so the time
    filter keeps working over it, and their ids so it can be expanded to them
    without a second lookup."""
    block = [members_of[i] for i in members]
    dates = sorted(c.date for c in block if c.date)
    spans_from = [c.span_from for c in block if c.span_from]
    spans_to = [c.span_to for c in block if c.span_to]
    topics: list[str] = []
    threads: list[str] = []
    for chunk in block:
        for topic in chunk.topics:
            if topic not in topics:
                topics.append(topic)
        for thread in chunk.threads:
            if thread not in threads:
                threads.append(thread)
    sessions = {c.session_id for c in block if c.session_id}
    return Chunk(
        id=f'summary:{group_id}', text=text,
        # One session's summary keeps that session's id, so the ground truth's
        # own unit still lines up; a summary spanning several claims none,
        # because naming one of them would put a citation on a row that is
        # mostly about the others.
        session_id=next(iter(sessions)) if len(sessions) == 1 else '',
        date=dates[0] if dates else '',
        span_from=min(spans_from) if spans_from else 0,
        span_to=max(spans_to) if spans_to else 0,
        importance=round(float(np.mean([c.importance for c in block])), 3),
        topics=tuple(topics[:12]), threads=tuple(threads[:12]),
        layer='summary', level=level, group_id=group_id,
        member_ids=tuple(c.id for c in block))
