# RAG lab

## The project

This project presents and refines a flexible RAG laboratory for users who want
to evaluate RAG approaches against their own corpus and ground-truth dataset,
with metrics and tables for each step, while leaving the decision to launch a
new experiment with the user. The laboratory accepts these datasets, runs the
selected approach, retrieves answers for comparison with the ground truth,
reports poor results thoroughly, and provides a helper assistant with
conversation memory, cost tracking, safety guards, and LangSmith tracing. The
helper responds only when asked, selects tools for the question, and,
depending on each result, chains another lookup, answers with insights, or
asks for clarification so the user can decide what to try next. It calls
OpenRouter and tools for experiment data, per-experiment and long-term memory,
and the stored RAG skills in `fixtures/skills/`, while reading the corpus,
ground-truth data, experiment results, metrics, and tables.

**Why it is useful.** Chunk size, retriever, reranker, and embedder choices are
usually made by taste; here they are made by measurement, and the helper turns
a table of numbers into a next step without ever making that step for you.

**Target users.** Engineers and students building a RAG system over their own
documents who need evidence for a design decision, and reviewers who want to
audit how an answer was produced.

## Overview

RAG lab is an offline-first retrieval workbench. It measures chunking,
retrieval, reranking, embedding, and generation choices against corpora with
known answers, so architecture decisions are based on evidence rather than
preference.

The repository includes seven datasets: `diary-en`, `diary-fa`, `support-en`,
`meetings-de`, `research-multihop`, `smoke-mini`, and `nosrat-fa`. Each is a
corpus together with the ground-truth questions a run is scored against. The
fresh panel default is `diary-en`; `diary-fa` remains the legacy identity for
older fingerprints.

## Quick start

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), and Node.js for
the browser-contract tests. Docker Compose is optional.

```sh
uv sync --extra local-embeddings
cp .env.example .env
uv run raglab
```

Open <http://localhost:9002> — the panel at `/`, the Inspector at
`/inspector`, the board at `/leaderboard`. With `RAGLAB_DEV_KEY` set in `.env`,
`/dev/trace` shows every widget conversation step by step — it asks for the
key on the page.

The default backend is local Ollama:

```sh
ollama pull 4skl/gemma4-e2b-mtp
```

Set `RAGLAB_LLM` in `.env` to choose another backend:

- `ollama` — local model server; set `RAGLAB_MODEL` if needed.
- `openrouter` — remote model; set `OPENROUTER_API_KEY`.
- `claude` or `codex` — a locally installed and authenticated CLI.
- `fake` — offline test backend only.

The first local index build downloads the embedding model, approximately
2.2 GB. The panel can start before a model is available; answering and judging
require a working backend.

## Run an experiment

1. Start the panel and choose or import a dataset.
2. Configure the index, retrieval, and generation settings.
3. Press **Build**, then **Run evaluation**.
4. Change one knob at a time and run again to make comparisons meaningful.
5. Open an experiment to inspect its retrieved evidence, answers, and metrics.
6. Export a completed experiment when you need to share its settings and
   evidence.

The four deciding metrics are faithfulness, answer relevancy, context
precision, and context recall. A ranking is valid only within the same dataset,
question set, and judge. The leaderboard groups experiments by dataset but
does not claim a universal winner.

## Datasets

Built-in datasets are ordinary corpus/ground-truth JSON pairs under
`fixtures/corpus_groundtruth_datasets/`. Import your own pair from the panel
or through `POST /api/datasets`. The pair must satisfy the declared dataset
contract; invalid data is rejected rather than repaired silently.

## The Ask widget

The **✳ Ask** helper is available on the Laboratory, Inspector, and Leaderboard.
It answers questions about the project, RAG techniques, and recorded
experiments. OpenRouter models can use the experiment and skills tools. Claude
and Codex CLI models are keyless but cannot use tools or retain graph
conversation state.

The widget keeps three kinds of memory in its local SQLite store:

- Short-term memory: the current `thread_id`, normally one experiment ID or
  `general`.
- Dataset memory: accepted summaries from all experiments on one dataset,
  including extracted subtopics such as reranking or semantic drift.
- Global memory: compact patterns that generalize across validated datasets.

Only relevant RAG-lab questions may enter long-term memory. A policy step
decides relevance, dataset, subtopic, and whether saving is allowed. Irrelevant
questions are refused and are not persisted. The summary writer runs after the
answer is delivered.

With `RAGLAB_DEV_KEY` set, `/dev/trace` asks for that key in a masked field
(it never appears in the address bar) and then lists every widget thread; open
one to see the system lines, the question, each tool call and reply, the
answer, and the token account. The browser stays unlocked until you press
**Lock** or the server restarts. Read-only; a 404 when no key is configured.

Widget turns can also be traced to LangSmith — the widget alone, since it
writes no run and no ledger row. Set the four `LANGSMITH_*` variables from
`.env.example` and restart the server. Two details that silently disable it:
`LANGSMITH_TRACING` must be the lowercase word `true`, and
`LANGSMITH_ENDPOINT` must match your key's region (US keys use
`https://api.smith.langchain.com`, EU keys `https://eu.api.smith.langchain.com`;
a mismatch is a 403).

## Stored data

All durable application data is local and git-ignored:

- `.runs/` — one JSON result per evaluation.
- `databases/raglab.db` — experiment/job ledger.
- `databases/corpora.db` — content-addressed imported corpora.
- `databases/widget.db` — widget checkpoints, readable turn logs, and long-term
  memory. Set `RAGLAB_WIDGET_DB` to override it.
- `.datasets/` — imported dataset files.

Inside `widget.db`:

- `checkpoints` and `writes` are internal LangGraph state tables.
- `widget_turn_log` is the readable one-row-per-question audit log, including
  messages, tool steps, token totals, latency, status, and memory link.
- `dataset_memory` stores one accumulated summary per dataset.
- `global_memory` stores cross-dataset patterns.
- `memory_updates` stores summary provenance: experiment, dataset, subtopic,
  question, and answer.

The widget never writes experiment runs, ledger rows, scores, or ranking data.
Its OpenRouter key, when entered in the panel, stays in process memory only.

## Commands

```sh
uv run raglab                         # serve the application on :9002
uv run raglab-lab --test-only         # run the offline preflight suite
uv run raglab-sweep                   # run one-knob-at-a-time candidates
uv run raglab-judgescreen --models MODEL
uv run raglab-leaderboard             # print recorded experiments
```

`raglab-sweep`, `raglab-judgescreen` and `raglab-leaderboard` live in
`agents/extra_tools/`; they import the lab and no frontend route reaches them.

## Examples

Questions the Ask widget is built for:

- On an experiment: *"Why is context recall low here?"* — it reads the
  experiment's retrieved evidence and metrics, then answers with what the
  retriever missed and which knob touches that.
- On the Leaderboard: *"Which of these runs is comparable to exp 12?"* — it
  refuses to name a winner across judges and explains which columns must match.
- On the Laboratory: *"What does reranking buy me on a Farsi corpus?"* — it
  looks up the stored skill in `fixtures/skills/` and any dataset memory from
  earlier runs on that corpus.
- Ambiguous: *"Is this good?"* — it asks which experiment and which metric you
  mean instead of guessing.
- Off-topic: *"Write me a poem"* — the relevance guard declines, and nothing is
  written to long-term memory.

## Design decisions

- **No vector database.** The index lives in process memory and dies with the
  process, so a build opens no socket and every experiment is reproducible
  from its recorded fingerprint alone.
- **A row never lies about what produced it.** A model the backend does not
  serve is refused, never substituted; a judge that cannot be reached refuses
  to score rather than returning a passing default.
- **Exactly four judged metrics decide** (faithfulness, answer relevancy,
  context precision, context recall). Everything else is reported and does not
  vote, so a change in one knob cannot be argued into a win by a metric nobody
  agreed on in advance.
- **One knob per sweep candidate,** models held fixed, and the answerer and the
  judge must be different models — otherwise a difference cannot be attributed.
- **The helper is outside the measured seam.** It writes no run, no ledger row
  and no score, which is the only reason it may keep memory, bill tokens and
  trace to LangSmith without contaminating an experiment.
- **Plain HTML/JS front end on a FastAPI backend** rather than a framework: the
  three pages share one token file and one widget bundle, and the browser
  contract is tested with `node --test` without a build step. The case against
  the two usual alternatives is in *Why not Next.js or Streamlit* below.
- **Prompts, tool descriptions and skills are fixtures, not code**
  (`fixtures/prompts/*.yaml`, `fixtures/skills/*/SKILL.md`), pinned byte-equal
  by a test, so a prompt change is a reviewed diff.

## Agent architecture

The Ask helper is a **single tool-calling loop** — the shape usually called
ReAct: the model is handed every tool at once, chooses one, reads what came
back, and decides from that result whether to look something up again or
answer. It is `langchain.agents.create_agent` with four middleware
(`agents/widget/hooks.py`), one cached agent per OpenRouter model, and one
LangGraph SQLite checkpoint thread per experiment.

That shape was chosen against the alternatives, not by default.

| Agent type | What it is | Why it was or was not used |
|---|---|---|
| **No agent — a prompt chain** | A fixed sequence of calls; the model never chooses a step. | Rejected. The question decides which record must be read: *"why is recall low here?"* needs the run, *"what does reranking buy me?"* needs a skill file. A fixed order would read both every time and still be wrong for the third question. |
| **Router / classifier** | One model call picks one tool, that tool answers, done. | Rejected. Predictable and cheap, but it cannot chain. *"Why is context recall low here?"* is three lookups where the second depends on the first: read the experiment, then read the questions whose evidence was missed, then read the skill that names the knob. |
| **Tool-calling loop (ReAct)** | Model picks a tool, reads the result, loops or answers. | **Chosen.** It is the smallest shape that can chain a lookup onto the result of the previous one. Its real cost is that the path is not fixed and can loop, so the loop is bounded twice: `MAX_TOOL_HOPS = 8`, and a `RECURSION_LIMIT` derived from it so the guard, not the ceiling, is what fires. |
| **Plan-and-execute** | The model writes a plan first, then executes each step. | Rejected. It buys the most on long multi-stage tasks; here a turn is one to three hops, so the planning call is roughly a third more cost per question — and the plan is stale the moment the first tool returns a number that changes what is worth reading next. |
| **Multi-agent / supervisor** | Specialist agents behind a coordinator. | Rejected. Every tool reads the same lab's records; there is no second domain for a specialist to own. The coordination overhead is real, and a second agent would make the one rule this package must keep — that it writes no run, no ledger row and no score — harder to see in one file. |
| **Reflection / critic loop** | A second model pass grades the first answer and it is rewritten. | Rejected for the answer, used elsewhere. Self-grading doubles the cost and asks the model to be its own judge, which is exactly what this project refuses to do for experiments. The honesty rules are enforced deterministically instead — the tool-hop guard, the provenance check that refuses to file a memory under the wrong dataset, and the length cap. Two *separate* structured calls do exist around the turn: the memory policy that decides whether anything may be stored, and the summary writer that runs after the answer is delivered. |
| **Autonomous / goal-driven agent** | Given an objective, it acts until it is met — here, launching experiments on its own. | Rejected on purpose. The product rule is that the human decides what to run. The helper reads and advises; the Build and Run buttons stay with the reader. |

Two footnotes a reviewer should have:

- **The CLI path is not an agent at all.** `claude` and `codex` run as one
  subprocess per call with the knowledge base inlined into the prompt, because
  `CliChat` has no `bind_tools`. They cannot call a tool, cannot stream, and
  write no conversation state. The model menu says so in each label — an
  option states what it cannot do.
- **Middleware is where the loop is made safe.** `check_request` caps the
  question before it can cost anything, `trim_and_call` bounds the history
  window while keeping the standing system lines, `log_tool_call` lets a tool
  failure through after recording it rather than swallowing it, and
  `close_the_log` closes the account of the run.

## Why not Next.js or Streamlit

The brief's usual stacks were both considered. The stack was chosen the same
way a chunker is chosen in this lab: by what the change costs in **evidence**,
not by preference.

**Against Node.** The web server was never the hard part. Underneath it sit
three layers with no JavaScript equivalent: the scoring library (`ragas`), the
maths and graph libraries (`numpy`, `scikit-learn`, `networkx`, `leidenalg`),
and the local embedding models. Rewriting them means proving the new versions
score identically to the old — and until that is proven, every recorded result
becomes incomparable. That is discarding the evidence base, not migrating it.
The usual reward, one language everywhere, is worth nothing here, because this
front end has no build tooling to unify with.

**Against Streamlit.** It is Python, so the science is safe; the objection is
architectural. Streamlit dissolves the boundary the rest of the project speaks
through — the HTTP API that the command-line entry points, the tests and the
helper's terminal mode all use. It cannot express a separately-permissioned
read-only Inspector, or a developer page that genuinely 404s when no key is
configured. It fights long-running cancellable jobs, because it re-runs the
whole script on every interaction. It cannot express the two-theme cascade
(`[data-theme]` outranking `prefers-color-scheme`). And roughly 3,400 lines of
browser-contract tests run by `node --test`, plus the served-surface assertions
in `dashboard/tests/`, would need a real browser instead.

**The honest concessions.** Streamlit would have given three things free that
were paid for by hand here: sortable tables, file upload and download for
dataset import and experiment export, and neat chat streaming. That is a real
cost, and it was weighed rather than dismissed.

## Limitations and reflection

### Optional tasks implemented

**The count: 7 optional tasks fully implemented and 4 more in part — 2 easy,
3 medium and 2 hard in full, plus 1 easy, 2 medium and 1 hard partially.**
The bonus criterion asks for two medium and one hard; three medium and two
hard are here in full. A task is only called *implemented* below if it works
end to end; anything short of that is marked *partial* and says what is
missing.

| Difficulty | Fully implemented | Partial | Not built | Offered in total |
|---|---|---|---|---|
| Easy | 2 | 1 | 2 | 5 |
| Medium | 3 | 2 | 3 | 8 |
| Hard | 2 | 1 | 2 | 5 |
| **All** | **7** | **4** | **7** | **18** |

**Easy — 2 of 5 in full, 1 partial**

- *Choose from a list of LLMs* — the helper's ⚙ menu offers four models, and
  the lab's own backend is chosen with `RAGLAB_LLM`.
- *Interactive help / chatbot guide* — the Ask helper rides all three
  surfaces, and every knob carries its own help text
  (`knob_help_text.py`, gated by `explain.missing() == []`).
- *Personality* — **partial.** The helper has one deliberate voice: brief,
  evidence-bound, refusing to state a number a tool did not return. It is not
  reader-selectable.
- *Temperature and max-token sliders* — **not built, on purpose.** A
  generation setting that changes an answer belongs on the row that records
  the answer, and the helper writes no row.

**Medium — 3 of 8 in full, 2 partial**

- *Token usage and cost* — **partial.** Every turn reports its token account
  in the widget and stores it in `widget_turn_log`; it is counted in tokens,
  never converted to money.
- *Long-term and short-term memory* — both. Short term is the LangGraph
  SQLite checkpoint per thread; long term is `dataset_memory`, `global_memory`
  and `memory_updates`, written only after a policy step accepts the turn.
- *Five or more function tools* — ten. **The UI toggle for enabling and
  disabling them is not built.**
- *Multi-model support* — OpenRouter (two models), Ollama, the Claude CLI, the
  Codex CLI, and a `fake` backend for the offline suite.
- *A security guard, with developer settings kept apart from the reader's
  experience* — the request cap and topic guard, the AST-whitelist calculator,
  the in-memory-only API key, and `/dev/trace` behind `RAGLAB_DEV_KEY`.
- **Not built:** a tool that calls an external API, user authentication, and a
  response-rating feedback loop.

**Hard — 2 of 5 in full, 1 partial**

- *Agentic RAG* — the helper retrieves over the skills corpus in two layers:
  a cheap catalogue search, then full bodies capped at three per call.
- *LLM observability* — LangSmith tracing for widget turns only, which is
  allowed precisely because the helper is outside the measured seam.
- *An evaluation report using RAGAS* — **partial.** RAGAS grades the lab's own
  pipeline on four judged metrics; it has never been pointed at the helper's
  answers.
- **Not built:** an agent that learns from user feedback, and integration with
  an external data source.

### Known limitations

- **No rating, so no feedback loop and no learning agent.** This is the
  largest gap against the optional list. The storage for it already exists:
  `widget_turn_log` has one row per question and `memory_updates` records why
  each summary was kept, so a verdict column and a route are the missing
  pieces, not a redesign.
- **No login and no rate limit on `/api/widget`.** This is a localhost tool;
  anyone who can reach port 9002 can spend the configured OpenRouter budget.
  Exposing the port without putting something in front of it is not safe.
- **Two of the four model options are degraded agents.** The CLI models cannot
  call tools, keep no thread state and cannot stream. The labels admit it, but
  a reader who picks one gets a different product.
- **The run log is process-wide.** `HOOK_LOG` is one bounded deque shared by
  every concurrent turn, so two readers asking at once produce an interleaved
  account. It is capped, so it leaks nothing and cannot grow — but it can
  mislead.
- **The deterministic topic guard is a small word list** and is easy to walk
  around. The model policy step behind it is the real relevance check; the
  guard only saves a model call on the obvious cases.
- **Long-term memory is model-written.** A wrong summary persists until
  `clear_long_term_memory` is called. Provenance is recorded for every update,
  but there is no review surface for a reader to correct one.
- **The helper is not itself measured.** The project's whole argument is that
  choices should be made by measurement, and the one component never subjected
  to that is the assistant.
- **Single process, in-memory index.** The index dies with the process by
  design; that is the reproducibility rule, but it also means a rebuild is the
  price of every restart.

### What would come next, in order

1. A rating on each answer, stored on the turn row, then dataset memory
   weighted by it — that closes the feedback loop and the learning agent
   together, on storage that already exists.
2. Point the lab's own RAGAS harness at the helper's answers, with the skills
   corpus as ground truth, and publish the four metrics for the assistant the
   way they are published for an experiment.
3. A rate limit and a single shared secret on the API, so the lab can be run
   somewhere other than a laptop.
4. A memory review surface: the accepted summaries, their provenance, and a
   delete button.

### When prompt engineering, RAG, or an agent is the right answer

This project ended up using all three, in layers, which makes the boundary
easy to state:

- **Prompt engineering** is enough when the model already knows the answer and
  only the shape is wrong. That is why the prompts and tool descriptions here
  are fixtures under review (`fixtures/prompts/*.yaml`) rather than strings
  buried in code — the cheapest fix is the one you can diff.
- **RAG** is needed when the answer is in text the model has never seen: this
  project's own facts, the twelve RAG skill files, a reader's private corpus.
  Retrieval, not a longer prompt, is what puts that text in front of the model.
- **An agent** is needed when the *question* decides which lookups happen and
  in what order, and a later lookup depends on what an earlier one returned.
  *"Why is recall low here, and what should I change?"* cannot be served by any
  fixed sequence, which is the whole justification for the tool loop.

The measured pipeline is RAG. The helper is an agent over that RAG plus the
recorded results. The prompts are engineering. Each layer is used where the
cheaper one below it stops working.

## Development and testing

The canonical offline check is:

```sh
uv run pytest src/raglab -q
```

It uses the `fake` backend, temporary databases, and blank credentials. The
intentional live probes are excluded unless run explicitly:

```sh
uv run pytest src/raglab/agents/widget/tests/test_skills_live.py -v -s
```

Browser contracts can be run directly from their directory:

```sh
cd src/raglab/dashboard/tests
node --test panel_open.test.js board_reveal.test.js
```

Branch from `development`; `master` is the squash-merged release branch.

## Docker

```sh
mkdir -p .runs databases .datasets
docker compose up -d --build
```

Open <http://localhost:9002>. Stop it with:

```sh
docker compose down
```

The container uses host Ollama through `host.docker.internal:11434` by
default. Embedding data is kept in the named `hf-cache` volume.
