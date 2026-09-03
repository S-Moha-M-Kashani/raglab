# retrieval.retriever — dense, sparse, or both fused

- **Step:** Retrieval (per question; outside the index fingerprint, so it
  sweeps free against one build). **Default:** `hybrid-rrf`.
- **Values:** `hybrid-rrf`, `dense`, `bm25`.

## What the knob does
`dense` searches vectors (meaning). `bm25` searches words (exact names, numbers,
rare terms). `hybrid-rrf` runs both and fuses the two rankings with Reciprocal
Rank Fusion.

## What it means scientifically
The two retrievers fail in opposite directions, which is the whole reason for
hybrids:

- **Sparse (BM25)** scores term overlap with saturation and length
  normalisation. It is exact, interpretable and unbeatable on rare tokens — a
  name, an id, a number — but it suffers the classic **vocabulary mismatch**
  problem: a question that uses different words than the document scores zero.
- **Dense (bi-encoder)** scores similarity in a learned space, so paraphrase
  works, but rare surface forms are blurred: a proper noun the encoder never saw
  is pushed toward the neighbourhood of things that merely look like it.
- **RRF** (Cormack, Clarke and Büttcher) fuses by *rank*, not score:
  `Σ 1/(rrf_k + rank)`. Because it never compares raw scores, it needs no
  calibration between systems whose scores are on incomparable scales — the
  practical reason it beats score-weighted fusion in the general case. Fusion
  rewards **agreement**: a document both retrievers rank highly wins over one
  either retriever loves alone.

## Why RAG architectures have this knob
Because most real corpora contain both regimes at once — paraphrasable prose
*and* identifiers that must match exactly — and a single retriever is a bet on
one of them. Hybrid retrieval is the standard production answer, and the knob is
here so the bet is measured rather than assumed.

## When each option is useful
- **`hybrid-rrf`** as the default, and especially on corpora full of proper
  nouns, dates, ids and numbers.
- **`bm25`** as a baseline that is often shockingly strong, and as a diagnostic:
  if BM25 alone matches the hybrid, the encoder is contributing nothing (check
  `index.embed_model`'s language coverage).
- **`dense`** when questions are paraphrases of the content and share almost no
  vocabulary with it, and to isolate the encoder's contribution.

## Interactions
`retrieval.rrf_k` only matters here when fusing; `retrieval.candidates` sets how
deep each retriever goes *before* fusion, which is what gives fusion and the
reranker something to work with. A hash `index.embedder` makes `dense` a null
run and turns `hybrid-rrf` into BM25 with extra steps.
