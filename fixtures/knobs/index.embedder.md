# index.embedder — which function turns text into the vector the index is searched by

- **Step:** Index. **Fingerprinted:** yes. **Default:** `sentence-transformers`.
- **Values:** `sentence-transformers`, `fastembed` (both load a real model),
  `ascii-hash`, `token-hash`, `char-hash` (no model at all).

## What the knob does
Chooses the embedding backend — and therefore whether anything downstream can
work. `fastembed` runs its own short local ONNX list; `sentence-transformers`
runs any HuggingFace checkpoint, which is the only route to an encoder tuned
for one language (a Farsi-tuned encoder for a Farsi corpus). The three hash
embedders load nothing: `ascii-hash` tokenises Latin letters and digits only,
so a non-Latin corpus embeds to the zero vector; `token-hash` and `char-hash`
see any script, but only as letters, never as meaning.

## What it means scientifically
Dense retrieval is a **bi-encoder** architecture: query and document are
embedded independently and compared by cosine similarity, which is what makes
search cheap and also what caps its quality — all the semantics must survive
compression into one vector, with no interaction between query and document.
The hash options are **feature hashing** (the hashing trick): random projection
of surface tokens into a fixed space. They preserve exact-match structure and
nothing else, which is exactly why they belong here — they are the *null model*.
`ascii-hash` on a non-Latin corpus is the strongest null there is: every vector
is zero, similarity is undefined, and any metric that still looks respectable
is measuring something other than retrieval. That is a floor to compare against,
not a working option.

## Why RAG architectures have this knob
Because the encoder is the one component whose failure is silent and total. A
tokenizer that cannot see a script, or a model trained on the wrong domain,
returns confident numbers over noise. Making the embedder a knob — with
deliberate degenerate baselines alongside real models — turns "is the retrieval
good?" into a measurement rather than an assumption.

## When it is useful
- **Real work:** `sentence-transformers` for language-specific or unusual
  domains; `fastembed` when a small local ONNX model is enough and startup cost
  matters.
- **Ablations:** a hash embedder answers "how much of my score comes from
  meaning and how much from word overlap?" — especially informative when
  `retrieval.retriever` is `hybrid-rrf`, where BM25 can carry a run whose dense
  half is broken.

## Interactions
`index.embed_model` names the checkpoint and decides language coverage; the
hash kinds ignore it and blank it in the fingerprint. Requires the
`local-embeddings` extra for real models and downloads weights on first use.
