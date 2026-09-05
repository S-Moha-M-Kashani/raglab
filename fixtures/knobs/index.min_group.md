# index.min_group — the smallest group worth summarising

- **Step:** Index. **Fingerprinted:** yes, whenever a hierarchy is built.
- **Default:** 3.

## What the knob does
Groups below this size are left as leaves and no summary row is written for
them.

## What it means scientifically
Two reasons, one statistical and one about retrieval:

1. **A tiny group's summary carries no new information.** A "summary" of two
   chunks is the two chunks with a header on top; extractive summarisation of a
   set that small is close to the identity function. In estimation terms, a
   centroid over two points has high variance and no shrinkage benefit — the
   group statistic is not more stable than its members.
2. **It competes with its own members.** The summary and the chunks it was made
   from are near-duplicates in one index, so they occupy several of the same
   top-k slots and crowd out genuinely different evidence. This is the
   redundancy problem MMR exists to fix, created here at index time instead of
   query time.

There is a third, practical effect: community detection and clustering both
produce a long tail of singletons and pairs, so without a floor most summary
rows are noise, and the summary population's statistics (and any boost applied
to it) are dominated by that noise.

## Why RAG architectures have this knob
Hierarchical indexes are only worth their extra rows if each row adds a level of
abstraction. The floor is what keeps the summary layer meaningful rather than a
partial copy of the leaf layer.

## When it is useful
- **Raise it** when the index reports many summaries but few large groups, or
  when retrieved sets are full of a summary plus the chunks it summarises.
- **Lower it** only on small corpora where every group is small and you are
  deliberately testing whether any summary layer helps.
- **Read it together with the reported group-size distribution** rather than
  tuning it blind.

## Interactions
`index.granularity` and `index.graph_knn` determine the group-size distribution
this knob cuts; `retrieval.summary_boost` amplifies whatever survives it, so a
low floor plus a boost is the configuration most likely to fill the top-k with
near-duplicate summaries.
