# Bundled datasets

Five corpora the lab can be pointed at, one of them the bundled default (the
Farsi diary). They are here to answer one question the lab could not answer
with a single fixture: **is this finding about retrieval, or about Farsi
diaries?**

Every dataset is a pair of files, joined by the id each declares:
`<name>_corpus.json` and `<name>_groundtruth.json`. `diary-fa` is an ordinary
pair like the other four — no native schema, no special-cased loader; it
satisfies the same `dataset_import_contract.validate()` contract every import
is held to, and `corpus_template.json`/`groundtruth_template.json` in this
folder are the templates for a new one.

Each also stands in for a use case from the map in
`skills/rag-use-case-architectures/SKILL.md`: the diary is the personal-memory
row, `support-en` the customer-support row, `meetings-de` the meeting-notes
row, and `research-multihop` the multi-hop research row — so a row's suggested
starting architecture can be tried against a corpus of its own shape without
leaving the repository.

They are read-only reference points. Anything imported through the panel lands
in `.datasets/` instead, which is git-ignored.

| pair | language | sessions | questions | what it is for |
| --- | --- | --- | --- | --- |
| `diary_year_fa_corpus.json` / `diary_year_fa_groundtruth.json` | Farsi | 167 | many | The bundled default: a year of synthetic colloquial Farsi diary chat — the flagship case study, not the project's scope. |
| `support_en_corpus.json` / `support_en_groundtruth.json` | English | 20 | 15 | A year of support conversations for a fictional analytics product. Latin script, short factual turns, dates and identifiers — the corpus where an ASCII embedder is genuinely competitive rather than scoring at chance, which is the control the Farsi finding never had. |
| `meetings_de_corpus.json` / `meetings_de_groundtruth.json` | German | 15 | 12 | Weekly meeting notes for a fictional software team. A third script family and compound nouns, where an English-only embedder separates sharply from a multilingual one. Each question carries an English translation. |
| `research_multihop_corpus.json` / `research_multihop_groundtruth.json` | English | 18 | 14 | Reading notes following one replication dispute. Deliberately weighted to multi-hop and aggregation: most answers need two or three sessions, so a pipeline that retrieves one perfect chunk still scores badly. |
| `smoke_mini_corpus.json` / `smoke_mini_groundtruth.json` | English | 5 | 6 | Five sessions, six questions, all three difficulty bands. Too small to measure anything and not meant to — it is what you run end to end in seconds to check a pipeline works before spending an hour on one that does not. Declares no date label and no `ranks` label on purpose, so the suite's fastest corpus also exercises those absent-declaration paths on every run. |

All five are synthetic. Every person, company, product and finding in them is
fictional; none of it is anybody's data.

## What makes them trustworthy

Every one satisfies the contract `datasets.validate()` enforces, and
`tests/test_datasets.py` re-checks that on every run of the suite. The rule that
matters is the last one: **each evidence quote appears verbatim in the message it
cites.** Quote recall, the Inspector's green evidence spans and the offline RAGAS
context metrics are all computed against those strings, so a corpus that
misquotes itself does not produce a worse score — it produces a confident score
about text that was never there.

They were authored so that the rule holds by construction rather than by
proofreading: each quote is a variable in the authoring script and the message it
belongs to is written by interpolating that same variable, so the two cannot
drift apart.

## What they are not

They are small. Fifteen questions cannot decide an architecture — the deciding
metrics are means over questions and a mean over fifteen has an error bar wide
enough to swallow most of the candidate field. Use them the way a control is
used: to find out whether something measured on the diary survives contact with
a different language, a different domain, or a different question shape.
