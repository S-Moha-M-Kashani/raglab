# index.graph_knn — how many nearest neighbours each chunk is joined to

- **Step:** Index. **Fingerprinted:** yes, when a kNN-bearing graph source is
  in play (`hybrid`, `knn`). **Default:** 8.

## What the knob does
Sets `k` in the k-nearest-neighbour graph the partition runs on. The build
reports the resulting component structure and modularity, so this is a knob you
can tune by reading index statistics rather than by running an evaluation.

## What it means scientifically
`k` controls graph **connectivity**, and community detection is acutely
sensitive to it:

- Too low and the graph fragments: many small connected components, each of
  which is trivially its own community. Modularity looks fine and means nothing
  — you have recovered the sampling noise of the neighbour lists.
- Too high and the graph approaches a dense blob: intra- and inter-group edge
  densities converge, modularity collapses toward one giant community, and the
  partition stops carrying information.

There is a percolation-style transition between the two regimes, which is why
the useful range is narrow and worth checking rather than assuming. High-
dimensional embeddings add a second effect: **hubness** — a few chunks appear in
disproportionately many neighbour lists — which at large `k` pulls unrelated
regions together through those hubs.

## Why RAG architectures have this knob
Because a graph over chunks is not given by the data; it is constructed, and `k`
is the main free parameter of that construction. Any claim about "louvain vs
leiden" is conditional on it: the comparison is only meaningful in the regime
where both are partitioning a connected, non-degenerate graph.

## When it is useful
- **Diagnose first, sweep second:** read the reported component count and
  modularity. Many components → raise `k`; one community covering most chunks →
  lower it.
- **Scale with the corpus:** a larger corpus tolerates (and needs) a larger `k`
  before it fragments; the same value is not equally sparse at 200 chunks and
  20,000.
- **Sparser** when `index.graph_source = lexical` already gives dense
  co-occurrence edges; the two sources add up under `hybrid`.

## Interactions
Only read when `index.graph_source` is `hybrid` or `knn`; ignored (and dropped
from the fingerprint) for `lexical` and `bipartite-terms`. Interacts with
`index.granularity`, which is the *other* dial on the same partition: `k`
changes the graph, γ changes what counts as a community in it.
