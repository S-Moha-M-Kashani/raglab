"""The graph's shape, as data: which stages a scope owns, the state channels,
and each scope's nodes and edges for anything that renders or checks them."""
from typing import Any, TypedDict

from raglab.configuration.lab_config import AgentConfig


def owns_retrieval(scope: str) -> bool:
    return scope in ('retrieve', 'full')


def owns_generation(scope: str) -> bool:
    return scope in ('generate', 'full')


class State(TypedDict, total=False):
    question: str
    query: str
    plan: str
    # LangGraph reads this class to know its channels; an update naming an undeclared key is refused.
    last: Any
    hops: int
    rewrites: int
    revisions: int
    calls: int
    unparsed: int
    sufficient: bool
    draft: str
    stop: str
    error: str


def _shape(cfg: AgentConfig) -> tuple[tuple[str, ...],
                                      tuple[tuple[str, str], ...]]:
    """This scope's nodes/edges as data for `graph_nodes`/`graph_edges`; `_Loop.build_graph` hand-wires its own edges rather than reading this table, and the two genuinely disagree on one declared-but-unreachable path-map entry (`rewrite=True`'s unconditional `'retrieve'` from `assess`) — so trust the hand-wiring, not this table, if they ever seem to disagree."""
    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    if owns_retrieval(cfg.scope):
        nodes += ['plan', 'retrieve', 'assess']
        edges += [('plan', 'retrieve'), ('retrieve', 'assess')]
        if cfg.rewrite:
            nodes.append('rewrite')
            edges += [('assess', 'rewrite'), ('rewrite', 'retrieve')]
        else:
            edges.append(('assess', 'retrieve'))
    if owns_generation(cfg.scope):
        nodes.append('draft')
        if owns_retrieval(cfg.scope):
            edges.append(('assess', 'draft'))
        if cfg.critic != 'none':
            nodes.append('critique')
            edges += [('draft', 'critique'), ('critique', 'draft')]
            if owns_retrieval(cfg.scope):
                # The interaction term: only `full` can answer a bad critique with different evidence rather than rewording.
                edges.append(('critique', 'retrieve'))
    return tuple(nodes), tuple(edges)


def graph_nodes(cfg: AgentConfig) -> tuple[str, ...]:
    return _shape(cfg)[0]


def graph_edges(cfg: AgentConfig) -> tuple[tuple[str, str], ...]:
    return _shape(cfg)[1]
