---
name: agentic-rag
description: Put a bounded control loop around retrieval and generation so the system can plan, retrieve again, inspect its own evidence, rewrite the query, and decide when it has enough — instead of running one fixed pass. Use for multi-hop questions, questions whose answer requires evidence the first query cannot name, and anywhere a single retrieve-then-generate pass demonstrably returns partial evidence. Covers the loop shapes, the caps every loop needs, why the agent must reuse the measured pipeline rather than bring its own retrieval, and how to attribute a win to the loop rather than to everything that changed with it.
---

# Agentic RAG

**What it is.** A fixed pipeline decides everything up front: one query, one
retrieval, one generation. An agentic pipeline makes retrieval a *decision
process* — the model plans, retrieves, looks at what came back, judges whether it
is enough, reformulates, retrieves again, and stops when the evidence supports an
answer or when a budget runs out.

The patterns come from general agent work — ReAct's interleaving of reasoning and
action, reflection, planning, tool use — applied to the specific case where the
tool is a retriever. The 2026 literature calls the large end of this "deep
research agents".

## The two loops, which are separate

Almost every agentic RAG system is one or both of these, and they are worth
keeping apart because they fix different things:

- **A retrieval loop.** plan → retrieve → judge the evidence → rewrite → retrieve
  again. Fixes *insufficient evidence*. Nothing here touches generation.
- **A generation loop.** draft → critique → revise. Fixes *ungrounded or poor
  answers* over evidence that was already adequate.

Run both and you have a system with a `critique → retrieve` edge: the drafting
stage can send the system back for more evidence. That is the interaction, and it
is genuinely more capable than either loop alone — but a run with both switched
on at once cannot tell you which one earned the improvement.

**Treat it as a 2×2**, not as four ideas: retrieval loop {off, on} × generation
loop {off, on}. The both-on cell is the interaction term and is readable only
beside the two single-loop cells. "We added an agent and the score went up"
attributes nothing.

## Rules that keep an agentic result meaningful

**The loop goes around the measured pipeline, never past it.** Every retrieval
hop must call the same retrieval function with the same configuration, so all the
retrieval knobs still apply per hop. An agent that brings its own retrieval is a
second, unswept pipeline, and its results are incomparable with every row already
recorded.

**Bound it twice, and record which bound fired.** Cap the *shape* (max hops, max
revisions) and cap the *cost* (max model calls) separately, because they fail
differently — a loop can spend its call budget inside one hop. Then put the
reason the loop stopped on the row: evidence-sufficient, hop-cap, call-cap,
revision-cap, grounded, abstained, error. "Refused because the corpus is silent",
"refused because it ran out of hops" and "refused because the model was
unreachable" are three findings, and without a stop reason they are one
indistinguishable row.

If the framework has its own recursion limit, derive it from your caps rather
than letting it fire — a framework ceiling reported as a hop cap names a limit
nobody configured.

**Return the best hop, not the last.** A rewrite can make things worse, and
evidence already found must not be spent discovering that.

**Read an unreadable verdict in the direction that costs work.** Unparsable
sufficiency means insufficient; unparsable groundedness means not grounded. A
verdict that decides whether the loop stops cannot be scored neutrally the way a
per-document relevance score can — see `adaptive-corrective-rag` for what a
neutral score does to a threshold.

**A self-critic is a weak check.** The critic usually runs on the answerer's own
model, which is fine for what it is — a cheap revision signal — and is not
evidence. What ranks the run must remain an independent judge. Local models have
been measured approving essentially everything.

**Report the shape as data.** Emit the node and edge counts from the compiled
graph rather than from a hand-written description, so the documentation and the
thing that ran cannot disagree about what a mode does.

## Is the loop even beating a strong baseline?

The control most agentic-search papers skip: a well-tuned single-pass retrieval.
2026 work asking "is lexical retrieval sufficient for agentic search?" found a
strong lexical baseline under an agent loop competitive with far more elaborate
machinery. Before attributing a win to the agent, check it against a widened
`top_k` and a reranker — both far cheaper than *n* model calls per question.

Open problems named across the 2026 surveys: evaluation, coordination, memory
management, efficiency, and governance. Evaluation is first for a reason — an
agent's output is a trajectory, and scoring only the final answer discards the
part that varies.

## In this lab

Implemented as the `raglab/agents/agentic_rag/` package, `AgentConfig.scope`, behind `uv sync --extra
agent`, and built as the 2×2 above:

- `''` fixed pipeline · `retrieve` retrieval loop · `generate` generation loop ·
  `full` both. **`full` breaks the one-knob-per-candidate rule on purpose** and is
  therefore not a candidate on its own — it carries the `critique → retrieve` edge
  and is readable only beside candidates `I` and `J` in `sweep.candidates`.
- Every hop calls `pipeline.retrieve` with the run's own `RetrievalConfig`, so all
  twenty-odd retrieval knobs apply per hop.
- **No scope is an index field**, so all four sweep free against one build — the
  same property the four `summary_scope` values have.
- `max_hops` / `max_revisions` bound the shape, `max_llm_calls` bounds the cost,
  and `agent_stop` names which fired, with `n_hops` and `n_agent_calls` beside it.
  LangGraph's `recursion_limit` is derived from the caps.
- `agentic_rag.verdict()` returns `None` rather than a number for an unreadable verdict
  — deliberately *not* reusing `retrieval.llm_scores`'s 0.5, which means something
  different in its own context.
- A scope this installation cannot run is refused, never substituted;
  `generate` and `full` additionally require `answerer='llm'` rather than
  silently promoting it. A row labelled `scope=full` that ran no agent is the
  worst artefact the lab can produce.
- `agentic_rag.shape._shape` reports the graph's nodes and edges as data.
- `/api/retrievals` narrows a retrieval-owning scope to its search half and forces
  the answerer off, since that route retrieves and stops.
- **No LangSmith and no checkpointer** on this path. A lab that phones a tracing
  service to explain its own numbers has put the account of a run somewhere its
  own records are not. (The panel's helper widget traces, and is outside the
  measured seam precisely so that it may.)
- The plum step ink `--step-agent` exists because the loop is not a pipeline stage
  and must not wear a stage's colour.

## Sources

- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136)
- [Deep Research Agents: A Roadmap](https://www.emergentmind.com/papers/2506.18096)
- [Deep Research: A Survey of Autonomous Research Agents](https://arxiv.org/html/2508.12752)
- [Rethinking Agentic Search with Pi-Serini: Is Lexical Retrieval Sufficient?](https://arxiv.org/pdf/2605.10848)
- [AgentIR: Reasoning-Aware Retrieval for Deep Research Agents](https://arxiv.org/pdf/2603.04384)
