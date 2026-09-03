# retrieval.k — how many chunks the answerer finally sees

- **Step:** Retrieval. **Default:** 8.

## What the knob does
The size of the final context set handed to the answerer, after fusion,
reranking and the relevance gate.

## What it means scientifically
This is the single knob that moves **recall and precision in opposite
directions**, and both are measured here: two of the four deciding metrics are
context precision and context recall.

- Raising `k` monotonically raises the chance the supporting evidence is present
  (recall@k never decreases with k).
- Raising `k` lowers the *density* of relevant text in the context, and readers
  are demonstrably sensitive to that: irrelevant retrieved passages degrade
  generated answers (the "distraction" effect), and evidence placed in the
  middle of a long context is used less reliably than evidence at the beginning
  or end (**lost in the middle**).

So the answer quality curve in `k` is typically an inverted U, not a staircase,
and its peak depends on the reader's robustness, the chunk size, and how
redundant the top of the ranking is.

## Why RAG architectures have this knob
Because "retrieve more" is the intuitive fix for a missed answer and it is only
half right: it fixes recall failures and creates precision failures. Exposing
`k` — and reporting precision and recall separately rather than one blended
score — is what makes that trade visible instead of a matter of taste.

## When it is useful
- **Raise it** when context recall is the failing metric, or when the correct
  chunk is being found at rank 9–15 (visible in the Inspector).
- **Lower it** when faithfulness or judged context precision is poor while
  recall is comfortable, or when the reader is small and easily distracted.
- **Keep it fixed** while sweeping any other retrieval knob: `k` changes the
  denominator of the precision metrics, so moving it alongside another knob
  confounds the comparison.

## Interactions
`retrieval.candidates` and `retrieval.rerank_depth` must both exceed `k` for
reranking to mean anything (depth 20 with k 8 = twenty chunks scored to choose
eight). `retrieval.max_context_chars` can silently drop whole chunks below `k`.
`retrieval.mmr_lambda` decides whether the k slots are spent on near-duplicates.
