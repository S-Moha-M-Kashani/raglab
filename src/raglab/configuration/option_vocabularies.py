"""The lab's closed vocabularies — every dropdown's allowed values. Imports nothing from this package."""

# Every option tuple leads with the value the lab actually defaults to, since
# both panels render these directly as offered choices;
# `test_every_option_list_leads_with_the_default` keeps a moved default from
# quietly staying buried in the list.
# The split plan's vocabulary (configuration/split_plan.py holds the plan
# itself). A stage is one of five kinds: `document` is always first, `part`
# and `label` cut at part boundaries, `separator` cuts text at a literal
# string, `drift` cuts where consecutive parts stop resembling each other.
STAGE_KINDS = ('document', 'part', 'label', 'separator', 'drift')
# Within one stage, atoms combine one way or the other — never both, so no
# precedence rule is ever needed.
COMBINATORS = ('or', 'and')
# Whether a stage cuts every piece or only one still over the budget.
STAGE_WHEN = ('always', 'over-budget')
# Which text normaliser tokenises the corpus for the lexical stages. '' is
# the default and means "whatever the corpus's declared language calls for"
# (text_normalizers.BY_LANGUAGE); a name overrides that for any corpus.
NORMALIZERS = ('', 'persian', 'neutral')
# What the size budget counts: characters, or the units the embedding model reads.
CHUNK_UNITS = ('characters', 'tokens')
# fastembed (its own ONNX list) and sentence-transformers (any HuggingFace
# checkpoint) load a named model; the hash embedders load none.
EMBEDDERS = ('sentence-transformers', 'fastembed',
             'ascii-hash', 'token-hash', 'char-hash')
MODEL_EMBEDDERS = ('fastembed', 'sentence-transformers')
# How chunks are grouped before being summarised beside the leaves; '' is flat
# and the default. Three families, in the order worth reading them: graph
# partitions, embedding clusterings, the declared control. Grouping is always
# over *chunks*, never entities — this is not GraphRAG, and `bipartite-terms`
# below is the closest honest analogue.
HIERARCHIES = ('', 'louvain', 'leiden', 'label-prop',
               'raptor', 'agglomerative', 'kmeans', 'metadata')
GRAPH_HIERARCHIES = ('louvain', 'leiden', 'label-prop')
CLUSTER_HIERARCHIES = ('raptor', 'agglomerative', 'kmeans')
# Groupings that can be asked for more than one level; label-prop and kmeans
# produce one partition and stop.
LEVELLED_HIERARCHIES = ('raptor', 'agglomerative', 'louvain', 'leiden')
# Groupings that read `granularity` (see IndexConfig for its two meanings).
# label-prop is the control precisely because it has no such parameter.
TUNED_HIERARCHIES = GRAPH_HIERARCHIES[:2] + CLUSTER_HIERARCHIES
# Declared metadata is deliberately absent as an edge source: that grouping is
# measured on its own as hierarchy='metadata', a control rather than an input here.
GRAPH_SOURCES = ('hybrid', 'knn', 'lexical', 'bipartite-terms')
KNN_SOURCES = ('hybrid', 'knn')
# All extractive: a build that called a model would be unsweepable and would
# let the `fake` provider fill the index with invention no field contradicts.
SUMMARIZERS = ('centroid', 'lead-idf', 'mmr', 'card')
RETRIEVERS = ('hybrid-rrf', 'dense', 'bm25')
# 'mixed' is the default so building a hierarchy changes nothing about
# retrieval until this knob moves.
SUMMARY_SCOPES = ('mixed', 'leaves', 'summaries', 'drill-down')
RERANKERS = ('lexical', 'none', 'recency', 'agentic', 'cross-encoder', 'llm')
GRADERS = ('none', 'lexical', 'llm')
ANSWERERS = ('extractive', 'none', 'llm')
