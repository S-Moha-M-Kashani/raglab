# retrieval.mmr_lambda — trade relevance for diversity in the final set

- **Step:** Retrieval. **Default:** 1.0 (pure relevance; lower diversifies).

## What the knob does
At 1.0 the top `k` are simply the best-scoring chunks, which often means several
chunks from one document. Lowering it trades some relevance for spread across
documents.

## What it means scientifically
This is **Maximal Marginal Relevance** (Carbonell and Goldstein, 1998), the
canonical formulation of relevance-with-redundancy-control. MMR selects
greedily, at each step maximising

`λ · relevance(chunk, query) − (1 − λ) · max similarity(chunk, already-selected)`

so λ = 1 is plain top-k and λ → 0 is pure novelty. The underlying observation is
that **relevance is not additive**: the second copy of the same fact adds
nothing, yet a ranker that scores documents independently cannot know that. Top-k
by score is therefore systematically redundant, and the redundancy is worst
exactly where retrieval works best — a strong match usually has near-duplicate
neighbours (adjacent chunks, overlapping windows, a summary and its members).

In decision terms it is a coverage objective: with a fixed budget of `k` slots
and a question that may need evidence from several documents, maximising
expected coverage beats maximising per-item score.

## Why RAG architectures have this knob
Because the context window is a budget and duplicates spend it for nothing.
Diversity also changes what the *answer* can be: a set of chunks all from one
document supports one narrative, and a spread set lets the answerer notice
disagreement or aggregate across sources.

## When it is useful
- **Lower it** for multi-document synthesis ("summarise what happened", "compare
  X across sources"), for ambiguous questions with several readings, and when
  `index.overlap` or a summary layer is filling the top-k with near-duplicates.
- **Keep it at 1.0** for single-fact lookups: the answer lives in one place, and
  forcing spread can push out the second-best chunk from the right document in
  favour of an irrelevant one from elsewhere.
- **Diagnose first:** if the retrieved set is visibly repetitive in the
  Inspector, this is the knob; if it is diverse but wrong, it is not.

## Interactions
Applied when selecting the final `retrieval.k` from the candidate pool, so it
interacts with `retrieval.candidates` (needs alternatives to choose from),
`index.overlap` and the summary knobs (both manufacture near-duplicates).
