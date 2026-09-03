# retrieval.rrf_k — the constant in Reciprocal Rank Fusion

- **Step:** Retrieval. **Default:** 60 — the value from the original RRF paper.
- **Only read when** `retrieval.retriever = hybrid-rrf`.

## What the knob does
RRF scores a document as `Σ_retrievers 1/(rrf_k + rank)`. This knob is that
constant.

## What it means scientifically
`rrf_k` is a **smoothing / discounting parameter** that decides how sharply the
top of each ranking dominates the fused result:

- **Small `rrf_k`** makes the reciprocal curve steep: rank 1 is worth much more
  than rank 5, so one retriever's confident top hit can win alone. The fusion
  behaves like "trust whoever is most certain".
- **Large `rrf_k`** flattens the curve toward equality: differences between
  ranks matter less, and what dominates is *how many* retrievers ranked a
  document highly at all. The fusion behaves like a vote, rewarding agreement
  over confidence.

The reason RRF works without calibration is that it discards scores entirely and
uses only ranks, which are comparable across systems by construction. The price
is that it also discards genuine confidence information — `rrf_k` is the single
dial that decides how much of the *rank* signal is kept.

## Why RAG architectures have this knob
Fusing a lexical and a dense retriever means combining two incomparable score
scales. Rank fusion is the standard robust answer, and its one parameter
encodes a prior about which retriever's certainty to believe. On corpora where
one retriever is much stronger, a steep curve helps; where they are
complementary, a flat curve helps.

## When it is useful
- **Leave it at 60** unless you have a reason: it is the published default and a
  reasonable prior.
- **Lower it** when one retriever is clearly better and its top hits are being
  diluted by the weaker one.
- **Raise it** when both retrievers are decent but noisy, and you want the
  documents both of them like — a good setting when precision matters more than
  finding the single best passage.

## Interactions
Meaningless unless two rankings are being fused (`hybrid-rrf`). Interacts with
`retrieval.candidates` (deeper pools give more overlap to reward) and with
`retrieval.reranker`, which re-scores whatever fusion produced.
