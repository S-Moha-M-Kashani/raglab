"""The lab's closed vocabularies — every dropdown's allowed values. Imports nothing from this package."""

# Every option tuple leads with the value the lab actually defaults to, since
# both panels render these directly as offered choices;
# `test_every_option_list_leads_with_the_default` keeps a moved default from
# quietly staying buried in the list.
CHUNKERS = ('semantic-drift', 'fixed', 'fixed-overlap', 'message', 'turn-pair',
            'session')
# Chunkers that read chunk_chars/overlap, per chunking.py's own branches — the
# rest emit one piece per message, pair or day and ignore both numbers.
CHAR_SIZED_CHUNKERS = ('semantic-drift', 'fixed', 'fixed-overlap')
OVERLAP_CHUNKERS = ('fixed-overlap',)
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
# The 2x2 the agent is built on: retrieval-agent {off,on} x generation-agent
# {off,on}, so a row can attribute its win to a stage rather than to "the
# agent". '' is the off control. 'full' deliberately changes both against the
# control and is only interpretable beside the two middle rows — never alone.
SCOPES = ('', 'retrieve', 'generate', 'full')
# What the generation agent checks before shipping a draft. 'none' is the
# control for whether the critique bought anything at all.
CRITICS = ('grounded', 'both', 'none')
# Ascending, and the order a sample's uneven remainder is handed out in.
DIFFICULTIES = ('easy', 'medium', 'hard')
# How a limited run picks its questions. See evaluate.select_questions.
BALANCES = ('stride', 'difficulty')
