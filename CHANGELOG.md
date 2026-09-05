# Changelog

Every entry below is a release point on `master` and carries the tag of the same
name. The ladder is the history: one landing, one release point, one tag, and
the version in `pyproject.toml` names the newest of them.

Dates are the day the release point was built. Versions follow
[semantic versioning](https://semver.org/) — a major number for a break, a
minor for a capability, a patch for a fix.

## [v1.0.0] — 2026-09-05

**The workbench reaches 1.0.** Eighteen landings since the previous release
point, and the reason 1.0 sits here rather than one rung earlier: the knob
surface underneath an experiment settled.

- **One split plan replaces the six chunkers and the delimiter list.** Where a
  document is cut is now a single ordered list of stages, stored in the form the
  fingerprint hashes and built on the panel stage by stage. A recorded
  experiment carries the plan where it used to carry a chunker's name.
- **The Inspector reads the lab it is part of.** Mounted, it asks the panel
  through nine named operations and gets a function call — no socket, no port,
  no `503`. Only `RAGLAB_INSPECTOR_LAB_URL` puts it in another process.
- **The served routes have a shape a reader can open**: one module per section
  under `dashboard/routes/`, each with its own tests, and a frozen context they
  read state off rather than decide with.
- **The lab is bounded in memory.** `RAGLAB_MAX_INDEXES` and
  `RAGLAB_MAX_JOB_HISTORY` cap what the process keeps, a pinned index is never
  evicted, and a build forced by an eviction says so on its row.
- **The helper gets a page per knob** and a ranked search over them, plus
  starters only this lab can answer.
- **The dataset page** was rebuilt, an import stops landing in the developer's
  own folder, and both boards took the new table look.
- **The container serves every backend it offers.**
- **A browser suite that owns its lab**: a live end-to-end journey on a real
  model, a guard that no surface squeezes its own text, and a killed run that no
  longer leaves its lab behind.
- **The files a stranger looks for**: this changelog, `SECURITY.md`, forms for
  the two kinds of issue the licence can accept, a pull request template saying
  why there is no third, weekly Dependabot on the CI actions and the Python
  dependencies, and an `.editorconfig`.

The major version says the workbench is complete enough to be used and cited as
it stands, not that it stopped changing. Rough edges in the panel are expected
and are what the patch number is for.

## [v0.34.0] — 2026-09-03

**The workbench reaches its first complete shape.** Since 0.33 the lab gained
six things: dataset copy that names no corpus, plainer widget wording, a left
setup panel that explains its own knobs, an opt-in browser suite driving all
three reader surfaces, a command line for the terminal tools, and a widget that
says which step it is on.

*This point carried the tag `v1.0.0` for two days. It was renumbered when 1.0
was moved forward to the release above, which is where the split plan settled
the knob surface.*

## [v0.33.0] — 2026-09-01

**Publish-ready lab.** The repository becomes presentable to a public GitHub
audience: a proprietary portfolio licence, a README that leads with Quick start
and drops the course-reviewer framing, a widget knowledge base rewritten to
fourteen verified topics, package metadata, a CI workflow, an offline
presentation deck, and no private codenames in any public-facing text.

## [v0.32.1] — 2026-08-31

**The helper stops describing a lab it no longer lives in.** Its fact sheet had
kept the shape of the repository this lab was extracted from. It now names
`diary-en` as the bundled default among seven corpora, adds the corpus store and
the dev trace page, and describes this lab's own architecture.

## [v0.32.0] — 2026-08-29

**The helper's prompt gets smaller and more honest.**

## [v0.31.2] — 2026-08-28

**Answer first, then judge.** The helper hands over its answer before deciding
whether the turn is worth remembering, so the reader waits one model round trip
instead of two. The tool-hop budget was off by one and is now derived from the
guard. The README gains three chapters: the agent shape and the six it is not,
the case against the usual front-end stacks, and an honest count of what is
missing.

## [v0.31.1] — 2026-08-28

**The push guard is held to the contract it now has.** It had moved into
`scripts/git-hooks` and become a plain allowlist, so its test was checking a
file that had gone. This release also carries the four hooks themselves and the
release script that built the commit.

## [v0.31.0] — 2026-08-28

**A seventh bundled corpus: `nosrat-fa`.** 167 tutoring sessions in Persian with
126 questions over them — the control the set was missing, because every other
second corpus also changes language, and this one keeps the diary's Farsi while
changing everything else about the material.

## [v0.30.0] — 2026-08-28

**A record's chunks tab reads its own archive**, so an old experiment shows the
text it actually indexed instead of nothing. The developer trace page also stops
taking its key from the address bar, where it was landing in browser history.

## [v0.29.1] — 2026-08-27
**Trace index rows render correctly.**

## [v0.29.0] — 2026-08-27
**Per-thread trace counts.**

## [v0.28.0] — 2026-08-27
**Trace context in the developer page.**

## [v0.27.2] — 2026-08-27
**Trace caching and system prompts hardened.**

## [v0.27.1] — 2026-08-27
**A busy widget database is tolerated rather than fatal.**

## [v0.27.0] — 2026-08-27
**Keyed widget trace pages.**

## [v0.26.4] — 2026-08-27
**Memory provenance survives a change of experiment.**

## [v0.26.3] — 2026-08-27
**The widget database record is persisted complete.**

## [v0.26.2] — 2026-08-27
**Remote pushes are restricted to `master`.**

## [v0.26.1] — 2026-08-26
**Experiment state and answer language are preserved.**

## [v0.26.0] — 2026-08-24
**The English diary becomes the default corpus.** An empty
`IndexConfig.dataset` still means `diary-fa`, so recorded fingerprints keep
their meaning.

## [v0.25.0] — 2026-08-24
**Corpus and ground-truth schemas are generalised**, so every bundled dataset is
an ordinary validated pair rather than a special case.

## [v0.24.0] — 2026-08-23
**Explanations rewritten, and experiments archive themselves.**

## [v0.23.0] — 2026-08-22
**The widget streams its replies.**

## [v0.22.0] — 2026-08-22
**One widget thread, shared across all surfaces.**

## [v0.21.0] — 2026-08-21
**Experiment analysis and lab handoff are connected**, so the board's open
button makes a recorded experiment's settings the panel's.

## [v0.20.0] — 2026-08-21
**Containerised deployment.**

## [v0.19.0] — 2026-08-21
**The measured agent loop is removed.** Simpler, and one fewer thing claiming to
be a stage.

## [v0.18.0] — 2026-08-21
**Dataset leaderboards and filtering.**

## [v0.17.0] — 2026-08-20
**Accessible day and night themes** — two, and no third.

## [v0.16.0] — 2026-08-20
**A shared frontend design system**, with colour meaning pipeline step.

## [v0.15.0] — 2026-08-20
**Portable experiment archives.**

## [v0.14.0] — 2026-08-19
**OpenRouter credentials are secured** — a typed key lives in process memory and
nowhere else.

## [v0.13.0] — 2026-08-19
**Widget conversations and token accounts are persisted.**

## [v0.12.0] — 2026-08-19
**Packages align with application surfaces.**

## [v0.11.0] — 2026-08-19
**The widget and its optional tracing are isolated** into a deletable leaf.

## [v0.10.0] — 2026-08-18
**RAG skills and bilingual probes.**

## [v0.9.0] — 2026-08-18
**LangChain 1.x orchestration adopted.**

## [v0.8.0] — 2026-08-15
**The assistant widget is integrated.**

## [v0.7.0] — 2026-08-14
**Configuration, frontend and tests are modularised.**

## [v0.6.0] — 2026-08-13
**Imports, fixtures and services consolidated.**

## [v0.5.0] — 2026-08-13
**Scoped retrieval and live inspection.**

## [v0.4.0] — 2026-08-12
**Hierarchical indexing and CLI backends.**

## [v0.3.0] — 2026-08-12
**Dataset evaluation and leaderboards.**

## [v0.2.0] — 2026-08-11
**The Laboratory, the Inspector, and credentials.**

## [v0.1.0] — 2026-08-11
**The measured RAG workbench is established.** Chunking and retrieval choices
are decided by measurement against a ground-truth corpus, never by taste.
