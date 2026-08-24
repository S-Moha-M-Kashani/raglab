"""Panel help text, keyed `<group>.<field>`, served over `/api/explain`."""

# Every knob explains itself in the panel, next to the control. A field added
# above without a line below fails test_every_configuration_factor_has_an_explainer.
# Keys are '<group>.<field>'; model fields are explained by models.ROLES
# instead, and 'run.*' describes controls that belong to one run, not a config.
#
# The lab is corpus-agnostic, so this text is too: a knob is described by what
# it does to any corpus, and a measured number says which corpus produced it.
# A few knobs really are wired to the bundled Farsi diary's own language and
# metadata; each of those opens with DATASET_SPECIFIC and then says exactly
# what happens elsewhere, so a reader is never left guessing which is which.

# One sentence, in one place, so the flag reads identically on every surface
# that carries it (embedder hints, model roles, metric help, disabled reasons).
DATASET_SPECIFIC = (
    'Dataset-specific: written for the bundled Farsi diary, not portable to a '
    'corpus in another language. It stays in the lab as the worked example of '
    'a general rule — a filter that knows its corpus\'s language and metadata '
    'beats a language-neutral pipeline.')

HELP = {
    'index.dataset': (
        'Which corpus this experiment measures against. It is the first thing '
        'the leaderboard groups by, because two corpora are two measurements '
        'rather than two configurations of one, and changing it rebuilds the '
        'index. The bundled default is a synthetic Farsi diary with 112 '
        'ground-truth questions; four smaller corpora ship beside it so a '
        'general finding can be told from a fact about Farsi diaries. Import '
        'your own with the button beside it — the ! there states the shape a '
        'file must have — and a dataset whose evidence quotes are not verbatim '
        'in the messages they cite is refused.'),
    # `run.` not `index.` — this is not a field, it's the control beside one —
    # and the key's last segment is the file input's own id, which is how the
    # panel finds an explainer at all. The one entry written as a shape rather
    # than prose (`p.explain` keeps its newlines); every other text is one line.
    'run.dataset-file': (
        'A dataset is two JSON files, paired by id: a corpus and the ground '
        'truth measured against it. Both are checked in full against '
        'fixtures/corpus_groundtruth_datasets/schema_corpus.json and '
        'schema_groundtruth.json and refused — never repaired — with every '
        'problem reported at once.\n'
        '\n'
        'The corpus: "corpus_dataset_metadata": {"dataset": "support-tickets", '
        '"name": "…", "language": "en"} — dataset is the id every run and '
        'leaderboard row records, 2+ characters, never a filename. '
        '"corpus_documents": [{"corpus_document_id": 1, "document_content": '
        '[{"text": "…"}]}] — one entry per document, one part per turn; a '
        'label used anywhere (on a document, a part, a question) must be '
        'declared in label_fields, which states its type, which levels it may '
        'apply to, and — for a closed set — its allowed values. One label '
        'typed "date-time" drives time filtering and recency; one numeric '
        'label declaring ranks:true drives importance; neither is required.\n'
        '\n'
        'The ground truth: "groundtruth_dataset_metadata": {"name": "…", '
        '"corpus_ref": {"dataset": "support-tickets"}} — corpus_ref.dataset '
        'must name the corpus above. "groundtruth_dataset": [{'
        '"groundtruth_question_id": 1, "question": "…", "expected_answer": '
        '{"behavior": "answer", "text": "…", "derived_facts": [{'
        '"derived_fact_id": 1, "fact": "…"}]}, "relevant_corpus_documents": '
        '[{"corpus_document_id": 1, "evidence": [{"text": "…", '
        '"fidelity": "verbatim"}]}]}] — question is the retrieval query as '
        'asked, nothing else here may be handed to it. behavior is "answer", '
        '"abstain" (nothing in the corpus answers it — empty '
        'relevant_corpus_documents) or "correct_premise" (a false premise, '
        'corrected from the corpus); only abstain scores as unanswerable. '
        'text is the reference two of the four judged metrics compare '
        'against; derived_facts are the atomic claims a correct answer must '
        'contain, reported but non-voting. evidence.fidelity is "verbatim" '
        '(copied character for character — checked against the document, and '
        'the only kind lexical quote recall may score), "paraphrase" (the '
        'document\'s own claim, restated) or "computed" (worked out from a '
        'label, stated nowhere in the text).\n'
        '\n'
        'The rule that earns its cost: every verbatim evidence entry must '
        'appear character for character in the document it cites. Quote '
        'recall, the Inspector\'s highlighted spans and the offline context '
        'metrics are all computed against those strings, so a ground truth '
        'that misquotes its corpus does not score worse — it scores '
        'confidently about text it never contained. Every bundled pair in '
        'fixtures/corpus_groundtruth_datasets/ meets it and is the working '
        'reference; corpus_template.json and groundtruth_template.json in '
        'that folder are the templates to copy, and schema_corpus.json / '
        'schema_groundtruth.json are the contract checked above, not files '
        'to import.'),
    'index.chunker': (
        'How one session is cut into the pieces that get embedded. "fixed" '
        'packs words up to a character budget; "fixed-overlap" slides a window '
        'so a sentence on a boundary appears in both neighbours; "message" '
        'keeps one message per piece; "turn-pair" keeps a question with its '
        'answer; "session" stores the whole session as one piece. '
        '"semantic-drift" cuts where consecutive messages stop resembling each '
        'other — the bottom third of that session\'s own similarity '
        'distribution, so no absolute threshold is assumed — and also at a '
        'size ceiling and at a short list of topic-change phrases, which is '
        'the only part of it written for one language.'),
    'index.chunk_chars': (
        'How large a piece may be, in characters. Only the three length-based '
        'splitters read it; the rest cut on structure and grey it out. "fixed" '
        'and "fixed-overlap" treat it as the size of one piece, so it behaves '
        'as a target; "semantic-drift" treats it as a maximum, cutting a '
        'segment once adding the next message would take it past twice this '
        'value. Small pieces retrieve precisely and often lose the sentence '
        'that answers the question; large ones keep it and dilute precision.'),
    'index.overlap': (
        'Characters repeated between neighbouring pieces, so a sentence '
        'sitting on a boundary is not cut in half. Only "fixed-overlap" slides '
        'a window, so only it reads this; an overlap at or above the piece '
        'size is halved rather than looping forever.'),
    'index.contextual': (
        'Prepend a one-line header to every chunk before embedding it '
        '(Anthropic call this contextual retrieval). A chunk that says "it '
        'got better" is unsearchable without knowing what "it" was. Built '
        'from the corpus\'s own metadata, so it costs no model call and no '
        'summary: every label the corpus declares at the document level '
        'that the document actually carries, in declaration order, list '
        'values joined by the corpus\'s own language comma. A label the '
        'document does not carry is left out rather than shown as a '
        'placeholder, and a label that rates another one\'s confidence is '
        'never written into chunk text — a caveat on a label is not '
        'something to embed. The header is written in the corpus\'s '
        'declared language.'),
    'index.embedder': (
        'Turns text into the vector the index is searched by, and the one '
        'choice that decides whether anything else matters. "ascii-hash" '
        'tokenises Latin letters and digits only, so a non-Latin corpus embeds '
        'to the zero vector — it is the baseline that makes the point, not a '
        'working option. "token-hash" and "char-hash" see any script, but only '
        'as letters, never as meaning. Two options load a real model and '
        'differ in what that costs: "fastembed" runs its own short ONNX list '
        'locally, "sentence-transformers" runs any HuggingFace checkpoint, '
        'which is the only way to reach an encoder tuned for one language — a '
        'Farsi-tuned one, for the bundled diary. It needs the '
        'local-embeddings extra and downloads weights. Whichever you pick, the '
        'model below decides which languages are covered.'),
    'index.embed_model': (
        'Which real embedding model the chosen backend loads, and where a '
        'non-English corpus is won or lost: the best-known encoders '
        '(bge-small-en, all-MiniLM-L6) are English-only and will return '
        'confident numbers that measure nothing on text they cannot read, and '
        'the bundled corpus is Farsi — so every entry states its coverage and '
        'the multilingual ones are listed by name. Bigger is not '
        'automatically better — e5 is trained for retrieval while the '
        'paraphrase models are trained for similarity — '
        'and the E5 family needs its "query:"/"passage:" prefixes, which the '
        'lab applies for you. Changing this rebuilds the index, because it '
        'changes what is stored.'),
    'index.hierarchy': (
        'Groups the chunks and indexes one summary per group *beside* them — '
        'the leaves always stay, because a summary that drops a detail makes '
        'the question about that detail unanswerable forever. Three families: '
        '"louvain", "leiden" and "label-prop" partition a graph built over the '
        'chunks; "raptor", "agglomerative" and "kmeans" cluster the chunk '
        'vectors; "metadata" groups by whatever storylines the corpus itself '
        'declares and is the control. **These are not GraphRAG.** GraphRAG '
        'extracts entities with a model; this lab builds offline, so the nodes '
        'are chunks and "bipartite-terms" below is the closest honest '
        'analogue. Expect leiden and louvain to tie on a small corpus: '
        'leiden\'s advantage is over badly-connected communities, which needs '
        'scale to show.'),
    'index.graph_source': (
        'What an edge between two chunks means. "knn" is cosine similarity — '
        'each chunk joined to its nearest neighbours; "lexical" is shared rare '
        'words, which is what catches names and numbers a vector blurs; '
        '"hybrid" is both. "bipartite-terms" makes the rare words nodes too '
        'and partitions chunks and terms together, so a community has a '
        'nameable subject — the nearest thing to an entity graph without a '
        'model. A corpus\'s declared topics are deliberately not an edge '
        'source: that grouping is measurable on its own as '
        'hierarchy="metadata", and mixing it in here would re-derive an '
        'already answered question inside a new one.'),
    'index.graph_knn': (
        'How many nearest neighbours each chunk is joined to. Low values leave '
        'the graph in disconnected pieces and every piece becomes its own '
        'community; high values connect everything to everything and '
        'modularity collapses toward one giant group. The build reports both, '
        'so this is a knob you can tune by reading the index statistics rather '
        'than by running an evaluation.'),
    'index.granularity': (
        'How coarse the grouping is, and it means two different things because '
        'the two families take two different parameters. For the graph '
        'partitions it is the modularity resolution: above 1.0 gives more, '
        'smaller communities. For the clusterings it is the group count, taken '
        'as granularity × √(n/2) over the leaf chunks — the usual rule of '
        'thumb. Either way 1.0 means "this family\'s own default".'),
    'index.hierarchy_levels': (
        'How many times to group the groups. Level 1 summarises chunks; level '
        '2 summarises those summaries. More levels answer broader questions '
        'and cost precision, because a summary of summaries is two extractions '
        'away from anything the corpus actually says.'),
    'index.min_group': (
        'The smallest group worth summarising. Below it the members are left '
        'as leaves and no summary row is written — a "summary" of two chunks '
        'is the two chunks with a header on top, and it competes against its '
        'own members in the search.'),
    'index.summarizer': (
        'How a group becomes one piece of text, without a model — a build that '
        'called an LLM would take hours instead of seconds and would let the '
        'offline fake backend fill the index with confident invention. '
        '"centroid" concatenates the members nearest the group\'s centre; '
        '"lead-idf" takes the sentences covering the most rare words; "mmr" '
        'picks members for coverage without repetition; "card" writes no prose '
        'at all — top terms, date span, member count, session ids — which is '
        'the cheapest and the most likely to help a counting question, because '
        'it states a number instead of asking the model to count chunks.'),
    'retrieval.summary_scope': (
        'What the search is allowed to see. "mixed" puts summaries and leaves '
        'in one pool, so a hierarchy changes nothing until you move this; '
        '"leaves" ignores the summaries entirely and is the control that says '
        'whether building them bought anything; "summaries" searches only '
        'them. "drill-down" retrieves among summaries and then expands each to '
        'its members — the shape of GraphRAG\'s local search, and the answer '
        'to a failure measured here: a summary can be correct and reachable '
        'and still almost never retrieved, because twenty times more leaves '
        'outvote it.'),
    'retrieval.summary_boost': (
        'Multiplies every summary\'s score before the candidates are cut. 1.0 '
        'is off. Applied before the cut and never after, because a summary '
        'that had not already survived the cut cannot be promoted into it — '
        'that version was measured and was a no-op that looked like a knob. Be '
        'careful with it even so: a boost lifts every summary equally, so it '
        'buys visibility for whichever kind of group is most numerous. '
        '"drill-down" is the targeted alternative.'),
    'retrieval.summary_levels': (
        'Which levels of the hierarchy may be retrieved, as a space-separated '
        'list ("1", "1 2"). Empty means all of them. Worth setting when a deep '
        'hierarchy is answering broad questions well and specific ones badly: '
        'the top level is the one most likely to be retrieved for everything.'),
    'retrieval.retriever': (
        '"dense" searches vectors (meaning), "bm25" searches words (exact '
        'names, numbers, rare terms), "hybrid-rrf" runs both and fuses the two '
        'rankings with Reciprocal Rank Fusion. Hybrid usually wins on a corpus '
        'full of proper nouns, which a vector blurs and a word match does not.'),
    'retrieval.k': (
        'How many chunks the answerer finally sees. Raising it finds more '
        'evidence and lowers precision; it is the single knob that moves '
        'recall and precision in opposite directions.'),
    'retrieval.candidates': (
        'How deep each retriever looks before fusion and reranking. Cheap to '
        'raise — nothing reads these yet — and it is what gives the reranker '
        'something to find.'),
    'retrieval.rrf_k': (
        'The constant in Reciprocal Rank Fusion (1/(k+rank)). Higher flattens '
        'the ranking, so agreement between the two retrievers matters more '
        'than either one being confident.'),
    'retrieval.time_filter': (
        DATASET_SPECIFIC + ' It reads Persian time language — «آذر», '
        '«تابستون», «سه ماه پیش» — as a Jalali date range and restricts the '
        'search to it, which is what stops a question about one month '
        'retrieving the whole year. A question in any other language matches '
        'nothing here, so the filter simply never fires and the search is '
        'unrestricted. Greyed out on a corpus with no label typed date-time '
        '(D5): with no dates on a chunk there is no span to restrict to, and '
        'the same absence leaves the recency reranker inert and a summary\'s '
        'date span blank.'),
    'retrieval.multi_query': (
        DATASET_SPECIFIC + ' It searches several rule-based rewrites of the '
        'question — the question itself, a keyword-only form with the '
        'interrogatives stripped, and a synonym variant — and merges the hits. '
        'It costs nothing, and on the bundled diary moved quote recall from '
        '0.489 to 0.512 with no model call. Both word lists are Persian and '
        'hand-written, so on another corpus every rewrite collapses back to '
        'the original question and this is a no-op.'),
    'retrieval.hyde': (
        DATASET_SPECIFIC + ' It writes a hypothetical answer with a model and '
        'searches with that instead of the question, on the theory that an '
        'answer looks more like the text you are hunting for than a question '
        'does. Costs one LLM call per query. The prompt asks in Persian for a '
        'diary-style paragraph, so on another corpus it does not go quiet — it '
        'searches with text in the wrong language and register, which is worse '
        'than leaving it off.'),
    'retrieval.mmr_lambda': (
        'Maximal Marginal Relevance. At 1.0 the top k are simply the '
        'best-scoring chunks, which often means several chunks from one '
        'session. Lower it to trade some relevance for spread across '
        'sessions.'),
    'retrieval.reranker': (
        'Re-scores the candidates before the cut to k. "lexical" is free IDF '
        'coverage; "recency" prefers recent entries; "agentic" is the '
        'Generative Agents mix of relevance + recency + importance; '
        '"cross-encoder" reads question and chunk together with a real model; '
        '"llm" asks a model to score each one.'),
    'retrieval.rerank_depth': (
        'How many candidates the reranker actually reads. The reranker is the '
        'expensive stage, so this is the cost dial: depth 20 with k 8 means '
        'twenty chunks scored to choose eight.'),
    'retrieval.recency_half_life_days': (
        'How fast the recency reranker forgets. At 180 days an entry from six '
        'months ago counts half as much as today — right for "how am I doing '
        'lately", wrong for "what happened last summer". Greyed out on a '
        'corpus that declares no date-time label (D5): with no date on a '
        'chunk there is nothing to weigh by age.'),
    'retrieval.agentic_weights': (
        'The three weights of the agentic reranker: relevance, recency, '
        'importance. Importance is the corpus\'s own numeric label declaring '
        'ranks: true (D6), rescaled to 0–1 by its declared minimum/maximum — '
        'a rating, a severity, whatever that corpus chose to rank chunks by. A '
        'corpus that declares no ranks label gives every chunk 0.0 importance, '
        'so that third weight shifts all scores equally and changes no '
        'ranking; the knob greys out for the same reason.'),
    'retrieval.grader': (
        'The gate that makes abstention possible: chunks scoring below the '
        'threshold are dropped, and if nothing survives the pipeline refuses '
        'instead of answering from noise. "none" means every question gets an '
        'answer, including the ones the corpus never mentions.'),
    'retrieval.grade_threshold': (
        'The score a chunk must clear to survive the gate. Measured on the '
        'bundled diary: the lexical gate had no usable setting (0.6 caught 6 '
        'of 8 unanswerable questions but wrongly refused 52% of the answerable '
        'ones), while an LLM gate at 0.4 refused 5 of 5 with 3% false '
        'refusals.'),
    'retrieval.max_context_chars': (
        'Budget for the assembled context. When it is exceeded whole chunks '
        'are dropped, never truncated — half an entry reads as a complete one '
        'and invites an answer from a sentence whose second half changed the '
        'meaning.'),
    'generation.answerer': (
        '"none" measures retrieval alone. "extractive" quotes the longest '
        'sentence from each of the top three chunks, tagged with its session — '
        'deterministic, free, and honest about quoting rather than answering. '
        '"llm" actually writes the answer.'),
    'generation.fact_judge': (
        'Scores each answer against the ground truth\'s atomic derived_facts '
        'with a model. It is the only way to score facts when the answer and '
        'the reference are not in the same language — the bundled diary '
        'answers in Farsi against English facts, which no lexical metric can '
        'compare — and it is the metric that exposed generation as that '
        'corpus\'s bottleneck (coverage 0.261 against faithfulness 0.743).'),
    'run.mode': (
        'Where the LLM stages run. "Local (Ollama)" is the lab default — free '
        'and private — and resets every stage to the lab\'s own defaults. '
        '"OpenRouter" switches the backend and presets the full pipeline onto '
        'gpt-5-nano (HyDE, LLM reranker, relevance gate, answerer and both '
        'judges); the relevance gate prefers a purpose-built reranker when '
        'OpenRouter\'s model list verifies one. No mode touches the index, so '
        'the embedder is left exactly where you set it. Picking a mode '
        'overwrites those stage choices; every knob can still be changed '
        'afterwards.'),
    'run.openrouter_key': (
        'The key the OpenRouter backend calls with, entered here instead of in '
        '.env so a lab already running can reach a remote model. It is held in '
        'the lab process and written nowhere — not to a run file, not to the '
        'experiment ledger, not to your browser — so it is forgotten when the '
        'lab stops; OPENROUTER_API_KEY in the environment is still how a lab '
        'starts with one. Setting it does not change which backend runs: that '
        'is the dropdown above, and a model on this machine needs no key at '
        'all.'),
    'run.ragas_mode': (
        '"offline" scores the retrieved context against the ground-truth '
        'quotes with string similarity — no model, no key, no variance. '
        '"judged" adds the five model-graded metrics: faithfulness, answer '
        'relevancy, factual correctness, and judged context precision and '
        'recall. Four of those five are what a configuration is actually '
        'chosen by. "off" skips RAGAS.'),
    'run.ragas_limit': (
        'How many questions RAGAS scores, when judged metrics make the full '
        'set too slow or too expensive.'),
    'run.limit': (
        'How many ground-truth questions to score. The subset is never the '
        'first n — it is drawn by the sampling below, striding across the set '
        'or taking an equal share of a labelled band — so a limit of 10 still '
        'covers the whole set instead of ten of one kind.'),
    'run.labels': (
        'Restrict the run to questions whose declared labels match. One '
        'switch-group per label the loaded ground truth declares with a '
        'closed set of values or a glossary — read from that dataset, so the '
        'choices differ from one corpus to the next rather than naming a '
        'fixed list every corpus must share.'),
    'run.balance': (
        'How a limited run chooses its questions. Naming a question label '
        'takes an equal share of that label\'s own values — "difficulty" '
        'takes an equal share of easy, medium and hard on a corpus that '
        'declares it; "" (stride) spreads across the set as it is, which '
        'means a corpus\'s most common band dominates the sample — on the '
        'bundled diary that is medium, 57 of 112, about half. It matters '
        'because the four deciding metrics are means over questions, so a '
        'skewed sample measures one band and reports it as the pipeline. The '
        'setting is recorded on every row rather than assumed, because it has '
        'not always been the same.'),
    'run.workers': (
        'How many questions are scored in parallel. Only worth raising when a '
        'stage calls a model, where wall-clock is dominated by waiting.'),
    'run.label': (
        'What this run is called in the leaderboard. Worth writing: a row '
        'named "semantic-drift" tells you nothing three days later.'),
}
