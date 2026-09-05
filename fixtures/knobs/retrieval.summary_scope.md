# retrieval.summary_scope — what the search is allowed to see

- **Step:** Retrieval (so all four scopes sweep free against one build).
- **Default:** `mixed`. **Values:** `mixed`, `leaves`, `summaries`,
  `drill-down`.

## What the knob does
`mixed` puts summaries and leaves in one pool, so building a hierarchy changes
nothing about retrieval until this knob moves. `leaves` ignores the summaries
entirely and is the control that says whether building them bought anything.
`summaries` searches only them. `drill-down` retrieves among summaries and then
expands each to its members.

## What it means scientifically
This is where the hierarchical index is actually cashed in, and the four options
are four retrieval theories:

- **One pool (`mixed`)** assumes units of different abstraction levels compete
  fairly on similarity. They do not. Leaves outnumber summaries by a large
  factor, so in a top-k over one pool the leaves simply **outvote** the
  summaries: a summary can be correct, reachable and almost never retrieved.
  That is a base-rate effect, not a scoring bug, and it is the failure this lab
  measured.
- **`leaves`** is the ablation that makes the summary layer's value measurable at
  all. Without it, "the hierarchy helped" is unfalsifiable.
- **`summaries`** is the global/thematic extreme — the shape of GraphRAG's
  *global* search, where the answer is synthesised from community-level
  descriptions rather than from source text.
- **`drill-down`** is the two-stage shape: retrieve coarse units, then expand
  each to its members. This is GraphRAG's *local* search, and the same pattern
  as the parent-document / small-to-big retrievers: **search at the abstraction
  level that matches the query, read at the level that contains the facts.** It
  is the targeted answer to the outvoting problem, because summaries only
  compete against summaries in the first stage.

## Why RAG architectures have this knob
Because question granularity varies and index granularity is fixed at build
time. Scope is the query-time dial that reconciles them — and, unlike the index
knobs, it costs no rebuild, so all four readings of one hierarchy can be
compared against a single build.

## When each option is useful
- **`drill-down`** for broad questions on a hierarchy you have already built —
  usually the option that makes the summary layer pay.
- **`leaves`** for factoid questions, and always as the control in any claim
  about hierarchies.
- **`summaries`** for "what were the themes" questions and for cheap coverage of
  a very large corpus.
- **`mixed`** as the neutral default; do not read it as "the best of both".

## Interactions
Requires `index.hierarchy`; `retrieval.summary_boost` is the blunt alternative
to `drill-down`, `retrieval.summary_levels` restricts which levels are eligible,
and `retrieval.k`/`max_context_chars` decide how much an expansion can bring
back.
