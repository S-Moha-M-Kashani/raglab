# Bundled datasets

Four corpora the lab can be pointed at, beside the built-in Farsi diary. They
are here to answer one question the lab could not answer with a single fixture:
**is this finding about retrieval, or about Farsi diaries?**

They are read-only reference points. Anything imported through the panel lands
in `.datasets/` instead, which is git-ignored.

| file | language | sessions | questions | what it is for |
| --- | --- | --- | --- | --- |
| `support-en.json` | English | 20 | 15 | A year of support conversations for a fictional analytics product. Latin script, short factual turns, dates and identifiers — the corpus where an ASCII embedder is genuinely competitive rather than scoring at chance, which is the control the Farsi finding never had. |
| `meetings-de.json` | German | 15 | 12 | Weekly meeting notes for a fictional software team. A third script family and compound nouns, where an English-only embedder separates sharply from a multilingual one. Each question carries an English translation. |
| `research-multihop.json` | English | 18 | 14 | Reading notes following one replication dispute. Deliberately weighted to multi-hop and aggregation: most answers need two or three sessions, so a pipeline that retrieves one perfect chunk still scores badly. |
| `smoke-mini.json` | English | 5 | 6 | Five sessions, six questions, all three difficulty bands. Too small to measure anything and not meant to — it is what you run end to end in seconds to check a pipeline works before spending an hour on one that does not. |

All four are synthetic. Every person, company, product and finding in them is
fictional; none of it is anybody's data.

## What makes them trustworthy

Every one satisfies `docs/groundtruth-dataset-contract.md`, and
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
