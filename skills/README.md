# `skills/` — the advanced-RAG corpus

Fourteen skills, one folder each, written in the
[Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
open format: ten techniques from the current RAG literature, plus four working
skills — the use-case → architecture map with its per-use-case experiment
ladders, the experiment methodology, and the two research-tracking skills.
They exist to be read by a person and, later, retrieved by the panel's widget
— one copy, two consumers, no drift.

They are **not** part of the measured seam. Nothing here is imported by
`raglab/`, nothing here changes a number, and nothing here is a claim about
any bundled corpus. Each file separates what the literature *measured* from
what this lab has measured — findings are labelled by the corpus they were
taken on — and the final section of every skill says how the technique maps
onto the knobs that already exist here.

## The fourteen

| Skill | The question it answers |
| --- | --- |
| [`chunking-strategies`](chunking-strategies/SKILL.md) | How should documents be split, and do the clever splitters actually beat recursive? |
| [`contextual-retrieval`](contextual-retrieval/SKILL.md) | What if every chunk carried a blurb saying where it came from? |
| [`hybrid-retrieval-fusion`](hybrid-retrieval-fusion/SKILL.md) | Dense or sparse — and how do you merge two ranked lists honestly? |
| [`reranking-late-interaction`](reranking-late-interaction/SKILL.md) | Recall is fine but the top 5 are wrong. What re-scores them? |
| [`hierarchical-graph-rag`](hierarchical-graph-rag/SKILL.md) | What do you retrieve for a question no single chunk answers? |
| [`query-transformation`](query-transformation/SKILL.md) | The query and the corpus are written in different registers. Rewrite which one? |
| [`adaptive-corrective-rag`](adaptive-corrective-rag/SKILL.md) | Should this query even be retrieved for, and what happens when the evidence is bad? |
| [`agentic-rag`](agentic-rag/SKILL.md) | When is a loop around retrieval worth *n* model calls per question? |
| [`multilingual-rag`](multilingual-rag/SKILL.md) | Which components fail silently outside English? |
| [`rag-evaluation`](rag-evaluation/SKILL.md) | How do you know any of the above helped? |
| [`rag-use-case-architectures`](rag-use-case-architectures/SKILL.md) | Which architecture is the educated first guess for this use case? |
| [`rag-experiment-methodology`](rag-experiment-methodology/SKILL.md) | How do you iterate — dev set, held-out test, error analysis — without fooling yourself? |
| [`rag-research-radar`](rag-research-radar/SKILL.md) | Where does new research land, and how does a technique earn a file here? |
| [`rag-source-watchlist`](rag-source-watchlist/SKILL.md) | Which sources carry news about *this project's* concerns, and what lands where? |

The reading order for a new project: `rag-use-case-architectures` picks the
starting candidate, `rag-experiment-methodology` runs the loop, `rag-evaluation`
says what a number is allowed to mean, and the technique skills are the levers
the loop reaches for. `rag-research-radar` keeps all of it current — it carries
a dated frontier snapshot and the procedure for adding the next skill — and
`rag-source-watchlist` is that procedure's address book for this repository's
own concerns.

## Why this format

A `SKILL.md` is YAML frontmatter plus a Markdown body. The spec requires exactly
two fields:

```yaml
---
name: contextual-retrieval        # lowercase, digits, hyphens; must equal the folder name
description: What it does, and when to use it.
---
```

`name` caps at 64 characters and must match the parent folder exactly or the
skill will not load. `description` caps at 1024 characters and has to state both
halves — what the skill covers *and* when to reach for it — because that string
is the only thing an agent reads when deciding whether to open the body. Avoid
angle brackets anywhere in the frontmatter: they can inject unintended
instructions into a system prompt.

Everything else is optional and unrecognised keys are ignored, so the format
costs nothing to adopt and is portable across runtimes.

Three reasons it is the right choice here over a single `ADVANCED_RAG.md`:

- **The description is a free retrieval index.** "What it does and when to use
  it" is exactly the routing signal a search tool needs. A keyword match over
  fourteen descriptions is cheap and honest; a keyword match over one
  2000-line file returns the file.
- **Progressive disclosure.** The body is only paid for when the skill is
  actually opened. That is the same two-level retrieve-then-expand pattern as
  `drill-down` in `raglab/hierarchy.py` — search the summaries, expand the one
  that matched.
- **One corpus, two readers.** Claude Code loads these from `.claude/skills/`;
  the widget can load the same files from disk. Neither needs its own copy.

## Using them from Claude Code

`.claude/skills/` is where Claude Code looks. Symlink rather than copy, so there
is one file to edit:

```bash
mkdir -p .claude/skills
for d in skills/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "../../$d" ".claude/skills/$(basename "$d")"
done
```

## Feeding them to the widget — designed, gated, not yet implemented

`raglab/widget.py` currently answers from `KNOWLEDGE_BASE`, a dict of seven
strings about this project, through a `search_knowledge_base` tool that keyword-
matches keys and bodies. Making it a RAG expert means adding this corpus beside
that dict. The design is pinned by `tests/test_skills.py` on the
`feature/rag-skills` branch (a test-first gate: the test exists, the
implementation follows its approval):

- **A loader, `raglab/skills.py`** — reads `skills/*/SKILL.md` from disk at
  call time, splits frontmatter from body, caches in process memory by file
  mtime. Nothing from the Markdown is copied into Python; the folder is the
  source of truth, and a convention test holds the loader to serving exactly
  the folders that exist. The one hand-written string is `DISTINCTIONS`, the
  guide to how the near-neighbour skills differ, which the same test forces to
  name every skill.
- **Two tools, not one.** `search_rag_skills(query)` returns names and
  descriptions — fourteen short lines, always affordable. `read_rag_skill(names)`
  returns bodies, several per call but capped at three, reporting an unknown
  name while still serving the known ones. The model routes on the first and
  pays for the second only when it commits. Collapsing them into one tool that
  returns matching bodies puts several thousand tokens into the loop for a
  question that wanted one line.
- **A keyword match, on purpose.** Fourteen descriptions are small enough that
  lexical matching works, and reaching for the lab's own embedder would put
  the widget inside the measured seam, which its module header refuses.
- **A system prompt that keeps the two corpora distinct**: `KNOWLEDGE_BASE` is
  *this project*, `skills/` is *the field*. The widget must not answer "what
  does this lab do" out of a technique paper, and must not present a
  literature claim as a measurement taken here.
- **The CLI backends get the index only.** `CliChat` has no `bind_tools`, so
  the two CLI options answer in one call with the knowledge base inlined.
  Fourteen skill bodies cannot be inlined; the CLI prompt carries the names
  and descriptions and says the bodies are out of reach there — an option
  states what it can do.

## Keeping them honest

- Every measured claim carries its source. Where a number came from this
  repository, the skill says so and names the date or the run.
- Literature numbers are reported as that corpus's result, not as a prediction
  about this one. Nothing in `skills/` licenses a change to a default here; only
  a row in `.runs/` does.
- Techniques that are *not* implemented here say so plainly, and say what would
  have to change — a store, a build rule, a dependency — rather than implying a
  knob exists.
- These files are prose, so they can rot. When a technique lands as a real knob,
  its skill's final section is the thing to update.
