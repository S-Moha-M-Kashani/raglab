# index.granularity — how coarse the grouping is (one dial, two meanings)

- **Step:** Index. **Fingerprinted:** yes, for the tuned families only.
- **Default:** 1.0, which always means "this family's own default".
- **Read by:** `louvain`, `leiden` (as modularity resolution) and `raptor`,
  `agglomerative`, `kmeans` (as group count). `label-prop` reads nothing —
  which is precisely why it is the control.

## What the knob does
For the graph partitions it is the **modularity resolution γ**: above 1.0 gives
more, smaller communities. For the clusterings it is the **group count**, taken
as `k = max(2, round(granularity × √(n/2)))` over the leaf chunks. One control
rather than two, because a reader should not have to hold two knobs that grey
each other out.

## What it means scientifically
Both readings answer the same unanswerable question — *how many groups are
there?* — which no clustering method determines from data alone.

- **Resolution γ** enters the modularity objective as the weight on the
  null-model term (the Reichardt–Bornholdt generalisation). It exists because
  plain modularity has a **resolution limit** (Fortunato and Barthélemy): below
  a scale set by the total number of edges, genuinely separate communities get
  merged no matter how well-separated they are. γ moves that scale; it does not
  remove it.
- **`k = √(n/2)`** is the standard rule-of-thumb for the number of clusters —
  it grows sublinearly with corpus size, which is the right shape (a corpus ten
  times larger does not have ten times more themes) but is a heuristic, not an
  estimate.

Either way, the number of groups is a *choice about the question being asked*,
not a property discovered in the corpus.

## Why RAG architectures have this knob
Group size decides what a summary is about. Coarse groups produce summaries that
answer broad questions and say nothing specific; fine groups produce summaries
that are barely more than their members. Since the right abstraction depends on
the questions, the lab exposes the dial and measures instead of guessing.

## When it is useful
- **Raise it** (more, smaller groups) when summaries read as vague and
  cross-topic — when a group's members clearly belong to several subjects.
- **Lower it** when summaries are near-duplicates of single chunks and
  `index.min_group` is already suppressing many groups.
- **Compare against `label-prop` and `metadata`**: if a tuned family only wins at
  a carefully chosen γ, that is a fragile result worth reporting as such.

## Interactions
`index.graph_knn` shapes the graph γ is applied to; `index.hierarchy_levels`
compounds the effect (a coarse level 1 leaves little to group at level 2);
`index.min_group` silently discards groups this knob made too small.
