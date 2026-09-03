# index.graph_source — what an edge between two chunks means

- **Step:** Index. **Fingerprinted:** yes, but only when the hierarchy is a
  graph partition; otherwise dropped so it cannot cost a rebuild.
- **Values:** `hybrid` (default), `knn`, `lexical`, `bipartite-terms`.
- **Read by:** `louvain`, `leiden`, `label-prop`.

## What the knob does
Builds the graph the partition runs on. `knn` joins each chunk to its nearest
neighbours by cosine similarity. `lexical` joins chunks that share rare words.
`hybrid` is both. `bipartite-terms` promotes the rare words to nodes and
partitions chunks and terms together, so a community comes out with a nameable
subject.

## What it means scientifically
A community-detection result is a function of the graph, and the graph is a
modelling choice — the algorithm cannot rescue a bad one. The three sources
encode three different similarity theories:

- **kNN similarity graph:** the standard construction in spectral and
  graph-based clustering, resting on the manifold assumption that semantically
  related texts are close in embedding space. It inherits the encoder's biases,
  including the tendency of high-dimensional embeddings to make some points
  *hubs* that appear in many neighbour lists.
- **Lexical (shared rare terms):** IDF-weighted co-occurrence. It catches
  exactly what a dense vector blurs — names, numbers, identifiers, technical
  terms — because rarity, not meaning, is the signal.
- **Bipartite chunk–term graph:** co-clustering, in the sense of Dhillon's
  bipartite spectral formulation: documents and terms partitioned jointly, so
  each community is a set of chunks *plus* the terms that characterise it. That
  is the nearest thing to an entity graph reachable without a model, and it is
  what makes a community labelable rather than just a number.

A corpus's declared topics are deliberately *not* an edge source: that grouping
is measurable on its own as `index.hierarchy = metadata`, and mixing it in here
would re-derive an already-answered question inside a new one.

## Why RAG architectures have this knob
Graph-based grouping only pays off if the edges mean something for the queries
you care about. Making the edge definition explicit is what separates "graph
RAG helped" from "cosine kNN helped".

## When it is useful
- **`lexical`** on corpora dense with proper nouns, ids and numbers — the same
  regime where BM25 beats a dense retriever.
- **`knn`** on paraphrase-heavy prose where wording varies and meaning repeats.
- **`hybrid`** as the default compromise, and the sane starting point.
- **`bipartite-terms`** when you want communities you can *name*, or when a
  summary needs a subject rather than a centroid.

## Interactions
`index.graph_knn` applies only to the two kNN-bearing sources (`hybrid`, `knn`)
and is dropped from the fingerprint otherwise. Downstream, edge quality shows up
in the partition's modularity and community-count statistics the build reports.
