# retrieval.summary_levels — which levels of the hierarchy may be retrieved

- **Step:** Retrieval. **Default:** `''` = every level.
- **Format:** a space-separated list, e.g. `1` or `1 2`.

## What the knob does
Restricts retrieval to the named levels of the summary hierarchy.

## What it means scientifically
Levels are abstraction strata, and they are not equally retrievable. A top-level
summary is generic by construction: it mentions many topics briefly, which gives
it moderate similarity to *almost any* query. In high-dimensional embedding
spaces this is the **hubness** phenomenon — a few points sit near many others
and appear in a disproportionate share of nearest-neighbour lists — and generic
summaries are exactly the kind of point that becomes a hub. The symptom is
recognisable: broad questions are answered well while specific ones degrade,
because the same one or two high-level summaries are retrieved for everything and
consume the top-k slots.

Restricting the eligible levels is the direct fix. It is a **structural** control
rather than a score adjustment (compare `retrieval.summary_boost`, which
re-weights but cannot exclude), and because it is a retrieval knob, all the level
combinations of one deep build can be compared without rebuilding anything.

There is a measurement point too: reporting which levels were eligible is part of
describing what was measured. "A three-level RAPTOR index" and "a three-level
index searched at level 1 only" are different systems.

## Why RAG architectures have this knob
Because building depth and using depth are separate decisions. A deep hierarchy
is cheap insurance at build time; which strata a query should see is a per-query
question, and the honest way to answer it is to measure each restriction.

## When it is useful
- **Set it** when a deep hierarchy answers broad questions well and specific ones
  badly — the classic signature of the top level being retrieved for everything.
- **Level 1 only** to test whether the deeper levels bought anything at all.
- **Top levels only** for deliberately global, sensemaking questions.
- **Leave empty** on a one-level hierarchy, where it has nothing to restrict.

## Interactions
Requires `index.hierarchy` with `index.hierarchy_levels > 1` to be meaningful,
and a `retrieval.summary_scope` that includes summaries. A cleaner instrument
than `retrieval.summary_boost` for the same class of problem.
