# index.hierarchy_levels — how many times to group the groups

- **Step:** Index. **Fingerprinted:** yes, for the multi-level families.
- **Default:** 1. **Read by:** `raptor`, `agglomerative`, `louvain`, `leiden`.
  `label-prop` and `kmeans` produce one partition and stop.

## What the knob does
Level 1 summarises chunks; level 2 summarises those summaries, and so on. Every
level is indexed beside the leaves, not instead of them.

## What it means scientifically
This builds an **abstraction ladder** — RAPTOR's recursive tree — and the
scientific point is that each rung is *lossy compression* of the rung below.
Extractive summarisation at level 2 selects from text that was already selected
at level 1, so information loss compounds multiplicatively with depth, and so
does any selection bias in the summariser. A level-3 summary is three
extractions away from anything the corpus actually says.

The payoff is coverage of query granularity: a deep tree contains a unit at
roughly the right scale for almost any question, which is what makes
"collapsed-tree" retrieval (search all levels at once) attractive. The cost is
that the highest levels are generic, and generic text has high average
similarity to many queries — the hubness problem again — so it gets retrieved
for everything.

## Why RAG architectures have this knob
Because corpora differ in how many meaningful scales they have. A one-year diary
has entries, weeks and seasons; a support-ticket dump may have tickets and
products and nothing above. Depth should be measured, not assumed, and each
extra level costs build time and index size for a shrinking return.

## When it is useful
- **Level 1** is the right default and usually where most of the benefit is.
- **Level 2+** when the corpus is large and genuinely multi-scale, and when
  broad questions still fail with one level of summaries.
- **Stop increasing it** when the top level starts being retrieved for
  everything; the targeted fix is `retrieval.summary_levels` (restrict which
  levels may be retrieved) rather than rebuilding shallower.

## Interactions
`index.granularity` decides how much is left to group at the next level;
`index.min_group` truncates the ladder in practice by refusing to summarise
small groups; `retrieval.summary_levels` and `retrieval.summary_scope` decide
which of the levels a query may see at all.
