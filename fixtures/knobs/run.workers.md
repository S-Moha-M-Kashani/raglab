# run.workers — how many questions are scored in parallel

- **Step:** Run control. **Only worth raising when a stage calls a model.**

## What the knob does
Sets the concurrency of the evaluation loop: how many questions are in flight at
once.

## What it means scientifically
Concurrency is the one knob that should change **wall clock and nothing else**,
and it is worth being precise about why, and about the ways that can fail:

- **The gain is bounded by what you are waiting for.** When a stage calls a
  model, per-question time is dominated by network or inference latency, so
  requests overlap almost perfectly and throughput scales nearly linearly with
  workers — Little's law, with the queue mostly idle. A fully local, CPU-bound
  pipeline (hash embedder, extractive answerer, offline metrics) is already
  saturating the machine, and extra workers only add contention.
- **Rate limits invert the gain.** Against a hosted backend, too much concurrency
  produces throttling, retries and — worst of all — errored stages, which show up
  as degraded rows rather than as slow ones. A row that carries a reason for
  degradation is honest, but it is still a row you did not want.
- **It must not change the numbers.** Each question is scored independently, so
  results are order-independent by construction. If a metric moves with worker
  count, that is a bug (shared state, or a rate limit being absorbed as a
  passing default), not a tuning opportunity — which is why a stage that cannot
  reach its model refuses to score rather than returning a default.

Also worth remembering for local backends: a CLI-driven call is unbilled but not
free, because it costs a process spawn and wall clock; concurrency there is
bounded by machine memory as much as by the model.

## Why RAG architectures have this knob
Because judged evaluation is latency-bound and serial evaluation of a hundred
questions against a remote judge takes long enough to discourage measuring at
all. Parallelism is what makes the honest, expensive measurement practical.

## When it is useful
- **Raise it** for judged runs, LLM rerankers, LLM gates and HyDE — every
  configuration whose per-question time is spent waiting.
- **Keep it low** for local model backends near their memory limit, and for
  offline-only runs where it buys nothing.
- **Lower it** at the first sign of throttling or errored stages.

## Interactions
Multiplies the call volume created by `retrieval.rerank_depth`,
`retrieval.grader`, `generation.answerer` and the judges; interacts with
`run.mode` (a local backend's ceiling is the machine, a remote one's is the
account).
