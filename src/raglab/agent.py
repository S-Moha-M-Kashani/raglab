"""The scoped RAG agent: a bounded LangGraph loop around the measured stages.

The point of this module is not that the lab can run an agent — it is that a row
can say *which stage* the loop paid off in. `AgentConfig.scope` hands exactly one
stage to the graph (`retrieve`, `generate`, or `full` for both), so the four
values are a 2x2 rather than four unrelated candidates. See
docs/plans/2026-08-13-rag-agent-design.md for the table and why `full` is only
interpretable beside the two middle rows.

Three properties are load bearing, and each one is a test:

**The loop goes around the pipeline, never past it.** Every hop calls
`pipeline.retrieve` with the run's own `RetrievalConfig`, so all twenty-odd
retrieval knobs still apply on every hop. An agent with its own retrieval would
be a second pipeline nobody has swept, and its row would be incomparable with
every row already in `.runs/`.

**An unreadable verdict is read in the direction that costs work.** An
unparsable sufficiency verdict means *insufficient* and an unparsable
groundedness verdict means *not grounded* — never a value that clears a
threshold. This is the 2026-08-02 gate fault stated as a rule: there, an
unreachable model scored 0.5 against a 0.4 bar, so `grader='llm'` became a no-op
that no field on the row contradicted. A loop that "succeeds" because its judge
is broken is the same artefact.

**The loop returns its best hop, not its last.** A rewrite can make things
worse, and evidence already found must not be spent on finding out. `best`
tracks the highest sufficiency verdict seen, earlier hops winning ties.

Nothing here persists: no checkpointer, no thread id, no LangSmith. The lab's
account of a run is `.runs/`, the ledger and the Inspector's trace, and an agent
must not become the first thing that reports somewhere else.
"""
import re
import time
from typing import Any, TypedDict

from . import pipeline
from .config import AgentConfig, LabConfig
from .llm import lab_chat
from .models import Roles

EXTRA = 'uv sync --extra agent'

# What a critic verdict has to clear. A constant rather than a seventh knob: the
# critic answers a yes/no question ("is every claim supported?"), so a dial on it
# would be a second control doing `critic`'s job — and the knob that decides how
# hard the *retrieval* loop tries already exists as `evidence_threshold`.
CRITIC_BAR = 0.5


def agent_available() -> bool:
    """Whether a scope can actually run *here*.

    Verified by import rather than read off a list, for the reason
    `hierarchy_available` is: NA has to keep meaning one thing — this
    installation cannot load it.
    """
    try:
        import langgraph                                    # noqa: F401
        from langgraph.graph import StateGraph              # noqa: F401
    except Exception:
        return False
    return True


def available() -> dict:
    """Every scope → whether this installation can run it, and what to install
    when it cannot. Served to the panel, so it never offers what the lab
    refuses."""
    from .config import SCOPES
    ready = agent_available()
    return {scope: {'available': True if not scope else ready,
                    'install': '' if not scope else EXTRA}
            for scope in SCOPES}


def owns_retrieval(scope: str) -> bool:
    return scope in ('retrieve', 'full')


def owns_generation(scope: str) -> bool:
    return scope in ('generate', 'full')


# --- reading a model's verdict ---------------------------------------------

_YES = ('yes', 'بله', 'آری')
_NO = ('no', 'خیر', 'نه')
_NUMBER = re.compile(r'^\s*(?:score|verdict|rating)?\s*[:=]?\s*'
                     r'(\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?')


def verdict(text: str) -> float | None:
    """A model's verdict in [0,1], or None when it did not give one.

    `None` is not a failure to be papered over — it is the honest reading of
    prose, an echo, or an empty reply, and every caller here turns it into the
    conservative outcome rather than a number. `retrieval.llm_scores` maps an
    unparsed line to 0.5 because there it means "no opinion about this one
    document among ten"; a single verdict that decides whether the loop stops
    cannot be split that way.
    """
    if not text:
        return None
    head = text.strip().lower()
    if any(head.startswith(word) for word in _YES):
        return 1.0
    if any(head.startswith(word) for word in _NO):
        return 0.0
    match = _NUMBER.match(head)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):                      # '8/10'
        scale = float(match.group(2)) or 1.0
        value = value / scale
    elif value > 1.0:                       # '8' on the 0-10 scale the prompts ask for
        value = value / 10.0
    return max(0.0, min(1.0, value))


# --- the one model seam ----------------------------------------------------

class _Exhausted(Exception):
    """The per-question call ceiling was reached. Not an error: a bound doing
    its job, reported as `agent_stop='call-cap'`."""


def _ask(llm, model: str, node: str, system: str, user: str) -> str:
    """Every model call the agent makes goes through here.

    One seam, so the call ceiling cannot be bypassed by a node added later, and
    so a test can answer per node without knowing a prompt's wording. `node` is
    unused in the call itself and present for exactly that reason.
    """
    turn = lab_chat(llm, [{'role': 'system', 'content': system},
                          {'role': 'user', 'content': user}], model)
    return (turn.content or '').strip()


PLAN_PROMPT = (
    'You plan retrieval over a personal Farsi diary. In one short sentence, say '
    'what evidence would answer the question — dates, names, events to look for. '
    'Do not answer the question itself.')
ASSESS_PROMPT = (
    'You judge whether retrieved diary excerpts are enough to answer a question. '
    'Reply with exactly one line: "SCORE: n" where n is 0-10 — 10 means the '
    'excerpts fully answer it, 0 means they are irrelevant. No other text.')
REWRITE_PROMPT = (
    'Rewrite a search query over a Farsi personal diary so it retrieves the '
    'missing evidence. Reply with the query only — keywords in Farsi, no '
    'explanation, no question words.')
CRITIQUE_PROMPT = (
    'You check a Farsi answer against the diary excerpts it was written from. '
    'Reply with exactly one line: "SCORE: n" where n is 0-10 — 10 means every '
    'claim in the answer is supported by the excerpts, 0 means it is invented. '
    'No other text.')
COMPLETENESS_PROMPT = (
    'You check whether a Farsi answer actually answers the question that was '
    'asked. Reply with exactly one line: "SCORE: n" where n is 0-10 — 10 means '
    'it answers it directly and completely, 0 means it does not answer it. No '
    'other text.')


def _contexts_block(outcome, limit: int = 700) -> str:
    return '\n\n'.join(
        f'[{c.session_id or c.chunk_id} | {c.date}]\n{c.text[:limit]}'
        for c in outcome.contexts) if outcome else '(nothing retrieved)'


# --- the graph's shape, as data -------------------------------------------

class State(TypedDict, total=False):
    question: str
    query: str
    plan: str
    # The last hop's Outcome. Declared, because LangGraph reads this class to
    # know its channels and refuses an update naming a key it has never heard of.
    last: Any
    hops: int
    rewrites: int
    revisions: int
    calls: int
    unparsed: int
    sufficient: bool
    draft: str
    stop: str
    error: str


def _shape(cfg: AgentConfig) -> tuple[tuple[str, ...],
                                      tuple[tuple[str, str], ...]]:
    """This scope's nodes and edges — including the conditional targets.

    Returned as data so `graph_nodes` / `graph_edges`, the panel's help text and
    the compiled graph cannot disagree about what a scope does. The edge
    `('critique', 'retrieve')` exists under exactly one scope, which is the
    entire reason `full` is worth running.
    """
    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    if owns_retrieval(cfg.scope):
        nodes += ['plan', 'retrieve', 'assess']
        edges += [('plan', 'retrieve'), ('retrieve', 'assess')]
        if cfg.rewrite:
            nodes.append('rewrite')
            edges += [('assess', 'rewrite'), ('rewrite', 'retrieve')]
        else:
            edges.append(('assess', 'retrieve'))
    if owns_generation(cfg.scope):
        nodes.append('draft')
        if owns_retrieval(cfg.scope):
            edges.append(('assess', 'draft'))
        if cfg.critic != 'none':
            nodes.append('critique')
            edges += [('draft', 'critique'), ('critique', 'draft')]
            if owns_retrieval(cfg.scope):
                # The interaction term: only `full` can answer a bad critique by
                # going back for different evidence rather than rewording the
                # same claim.
                edges.append(('critique', 'retrieve'))
    return tuple(nodes), tuple(edges)


def graph_nodes(cfg: AgentConfig) -> tuple[str, ...]:
    return _shape(cfg)[0]


def graph_edges(cfg: AgentConfig) -> tuple[tuple[str, str], ...]:
    return _shape(cfg)[1]


def _guard(fn):
    """Every node, wrapped: a stop already set is honoured, the call ceiling is a
    stop rather than a crash, and an unreachable model ends the loop with its
    reason on the row instead of taking the run down.

    The alternative — letting the exception out of `graph.invoke` — loses the
    counters, and "refused after two hops" and "refused because the model was
    unreachable" are exactly the two readings this feature exists to separate.
    """
    def wrapped(state: State) -> dict:
        if state.get('stop'):
            return {}
        try:
            return fn(state)
        except _Exhausted:
            return {'stop': 'call-cap'}
        except Exception as error:
            return {'stop': 'error',
                    'error': f'{type(error).__name__}: {error}'[:200]}
    wrapped.__name__ = getattr(fn, '__name__', 'node')
    return wrapped


def run(index, cfg: LabConfig, question: str, query_date: str, llm=None,
        models: Roles | None = None, trace: dict | None = None):
    """One question through the scoped loop, returning the same `Outcome` the
    fixed pipeline returns.

    The same shape deliberately: scoring, RAGAS, the ledger and the Inspector
    then need no second idea of what a result is, and an agent row is comparable
    with a pipeline row on every metric the lab already computes.
    """
    agent_cfg = cfg.agent
    if not agent_cfg.scope:
        raise ValueError(
            'agent.run needs a scope — the fixed pipeline is pipeline.retrieve. '
            'A caller that reaches here with no scope would produce a row '
            'labelled with an agent that never ran.')
    roles = models or Roles()
    started = time.perf_counter()
    visits: list[dict] = []
    hop_traces: dict[int, dict] = {}
    # The best hop seen, and its verdict. A later hop only wins on a strictly
    # higher verdict, so a rewrite that made things worse cannot spend the
    # evidence an earlier hop already found.
    best: dict[str, Any] = {'outcome': None, 'verdict': -1.0, 'hop': 0}

    def note(node: str, hop: int, detail: str = '') -> None:
        visits.append({'node': node, 'hop': hop, 'detail': detail[:200]})

    def ask(state: State, node: str, model: str, system: str, user: str
            ) -> tuple[str, dict]:
        """A model call, with the ceiling checked *before* it is spent."""
        if state.get('calls', 0) >= agent_cfg.max_llm_calls:
            raise _Exhausted()
        text = _ask(llm, model, node, system, user)
        return text, {'calls': state.get('calls', 0) + 1}

    def do_retrieve(state: State) -> dict:
        hop = state.get('hops', 0) + 1
        hop_trace: dict = {} if trace is not None else None
        outcome = pipeline.retrieve(index, cfg.retrieval, state['query'],
                                    query_date, llm=llm, models=roles,
                                    trace=hop_trace)
        if hop_trace is not None:
            hop_traces[hop] = hop_trace
        if best['outcome'] is None:
            best.update(outcome=outcome, verdict=-1.0, hop=hop)
        note('retrieve', hop, f'{len(outcome.contexts)} contexts for '
                              f'{state["query"][:60]}')
        return {'hops': hop, 'last': outcome}

    def do_plan(state: State) -> dict:
        text, spent = ask(state, 'plan', roles.plan, PLAN_PROMPT,
                          f'Question: {state["question"]}')
        note('plan', 0, text)
        return spent | {'plan': text}

    def do_assess(state: State) -> dict:
        outcome = state.get('last')
        text, spent = ask(state, 'assess', roles.plan, ASSESS_PROMPT,
                          f'Question: {state["question"]}\n'
                          f'What we are looking for: {state.get("plan", "")}\n\n'
                          f'Excerpts:\n{_contexts_block(outcome)}')
        score = verdict(text)
        note('assess', state.get('hops', 0),
             f'{"unparsed" if score is None else round(score, 2)}')
        # Unparsed means insufficient: keep looking. Never a number that clears
        # the threshold — see the module docstring.
        if score is not None and score > best['verdict']:
            best.update(outcome=outcome, verdict=score, hop=state.get('hops', 0))
        return spent | {
            'sufficient': bool(score is not None
                               and score >= agent_cfg.evidence_threshold),
            'unparsed': state.get('unparsed', 0) + (1 if score is None else 0)}

    def do_rewrite(state: State) -> dict:
        text, spent = ask(state, 'rewrite', roles.plan, REWRITE_PROMPT,
                          f'Question: {state["question"]}\n'
                          f'Already tried: {state["query"]}\n'
                          f'Still missing: {state.get("plan", "")}')
        note('rewrite', state.get('hops', 0), text)
        # A model that returns nothing usable leaves the query alone rather than
        # searching for an empty string.
        return spent | {'query': text or state['query'],
                        'rewrites': state.get('rewrites', 0) + 1}

    def do_draft(state: State) -> dict:
        outcome = best['outcome']
        previous = state.get('draft')
        user = (f'سؤال: {state["question"]}\n\n'
                f'تکه‌های دفترچه:\n{_contexts_block(outcome, 900)}')
        if previous:
            user += (f'\n\nپیش‌نویس قبلی که رد شد:\n{previous}\n'
                     'دوباره بنویس و فقط به تکه‌های بالا تکیه کن.')
        text, spent = ask(state, 'draft', roles.answer, pipeline.ANSWER_PROMPT,
                          user)
        note('draft', state.get('hops', 0), text)
        return spent | {
            'draft': text or pipeline.REFUSAL,
            'revisions': state.get('revisions', 0) + (1 if previous else 0)}

    def do_critique(state: State) -> dict:
        outcome = best['outcome']
        body = (f'Question: {state["question"]}\n\nAnswer:\n{state["draft"]}\n\n'
                f'Excerpts:\n{_contexts_block(outcome)}')
        text, spent = ask(state, 'critique', roles.critic, CRITIQUE_PROMPT, body)
        grounded = verdict(text)
        unparsed = 1 if grounded is None else 0
        note('critique', state.get('hops', 0),
             f'grounded={"unparsed" if grounded is None else round(grounded, 2)}')
        passed = grounded is not None and grounded >= CRITIC_BAR
        state = dict(state) | spent
        if passed and agent_cfg.critic == 'both':
            text, more = ask(state, 'completeness', roles.critic,
                             COMPLETENESS_PROMPT, body)
            spent = {'calls': more['calls']}
            complete = verdict(text)
            unparsed += 1 if complete is None else 0
            note('completeness', state.get('hops', 0),
                 f'complete={"unparsed" if complete is None else round(complete, 2)}')
            passed = complete is not None and complete >= CRITIC_BAR
        return spent | {'sufficient': passed,
                        'unparsed': state.get('unparsed', 0) + unparsed}

    # --- routing -----------------------------------------------------------

    def after_assess(state: State) -> str:
        if state.get('stop'):
            return END
        if state.get('sufficient'):
            return 'draft' if owns_generation(agent_cfg.scope) else END
        if state.get('hops', 0) >= agent_cfg.max_hops:
            # Out of hops. Under a generation scope the drafting still happens —
            # with the best evidence found — and the row says the cap ended the
            # search rather than a verdict.
            return 'draft' if owns_generation(agent_cfg.scope) else END
        return 'rewrite' if agent_cfg.rewrite else 'retrieve'

    def after_draft(state: State) -> str:
        if state.get('stop'):
            return END
        return 'critique' if agent_cfg.critic != 'none' else END

    def after_critique(state: State) -> str:
        if state.get('stop'):
            return END
        if state.get('sufficient'):
            return END
        if state.get('revisions', 0) >= agent_cfg.max_revisions:
            return END
        # `full` alone may answer a bad critique with different evidence; every
        # other scope can only redraft from what it has.
        if (owns_retrieval(agent_cfg.scope)
                and state.get('hops', 0) < agent_cfg.max_hops):
            return 'retrieve'
        return 'draft'

    from langgraph.graph import END, StateGraph

    graph = StateGraph(State)
    builders = {'plan': do_plan, 'retrieve': do_retrieve, 'assess': do_assess,
                'rewrite': do_rewrite, 'draft': do_draft,
                'critique': do_critique}
    nodes = graph_nodes(agent_cfg)
    for name in nodes:
        graph.add_node(name, _guard(builders[name]))
    if owns_retrieval(agent_cfg.scope):
        graph.set_entry_point('plan')
        graph.add_edge('plan', 'retrieve')
        graph.add_edge('retrieve', 'assess')
        targets = ['retrieve', END] + (['draft']
                                       if owns_generation(agent_cfg.scope) else [])
        if agent_cfg.rewrite:
            targets.append('rewrite')
            graph.add_edge('rewrite', 'retrieve')
        graph.add_conditional_edges('assess', after_assess, targets)
    else:
        graph.set_entry_point('draft')
    if owns_generation(agent_cfg.scope):
        if agent_cfg.critic == 'none':
            graph.add_edge('draft', END)
        else:
            graph.add_conditional_edges('draft', after_draft, ['critique', END])
            after = ['draft', END]
            if owns_retrieval(agent_cfg.scope):
                after.append('retrieve')
            graph.add_conditional_edges('critique', after_critique, after)

    # A scope that does not own retrieval retrieves exactly once, before the
    # graph runs: that is what "retrieval held fixed" means, and it keeps the
    # generation row's evidence identical to candidate F's.
    initial: State = {'question': question, 'query': question, 'hops': 0,
                      'rewrites': 0, 'revisions': 0, 'calls': 0, 'unparsed': 0}
    if not owns_retrieval(agent_cfg.scope):
        fixed_trace: dict = {} if trace is not None else None
        outcome = pipeline.retrieve(index, cfg.retrieval, question, query_date,
                                    llm=llm, models=roles, trace=fixed_trace)
        if fixed_trace is not None:
            hop_traces[1] = fixed_trace
        best.update(outcome=outcome, verdict=-1.0, hop=1)
        initial |= {'hops': 1, 'last': outcome}
        note('retrieve', 1, f'{len(outcome.contexts)} contexts (fixed retrieval)')

    if best['outcome'] is not None and not best['outcome'].contexts:
        # Nothing to draft from and nothing to critique. Refusing here is the
        # pipeline's own rule (`pipeline.answer` on an abstained outcome), and it
        # keeps the row honest: `abstained`, not a critique of an empty context.
        final: State = dict(initial) | {'stop': 'abstained'}
    else:
        # The recursion limit is derived from the caps rather than left at
        # LangGraph's default 25: a lab knob that silently hit a framework
        # ceiling would report `hop-cap` for a limit nobody configured.
        rounds = 4 * (agent_cfg.max_hops + agent_cfg.max_revisions) + 10
        final = dict(graph.compile().invoke(
            initial, config={'recursion_limit': rounds}))

    outcome = best['outcome']
    if outcome is None:
        # The loop died before it retrieved anything (an unreachable model on the
        # planning call). An empty Outcome, so the caller still gets one shape.
        outcome = pipeline.Outcome(question=question, contexts=[],
                                   abstained=True)
    stop = final.get('stop') or _terminal(agent_cfg, final)
    if owns_generation(agent_cfg.scope):
        if stop in ('error', 'abstained'):
            outcome.answer = pipeline.REFUSAL
            outcome.abstained = True
        else:
            outcome.answer = final.get('draft') or pipeline.REFUSAL
            if pipeline.reads_as_refusal(outcome.answer, 'llm'):
                outcome.abstained = True
    else:
        # The fixed answerer, exactly as an unagented run would call it: this
        # scope owns retrieval only, so generation must stay comparable.
        if stop == 'error':
            outcome.answer = pipeline.REFUSAL
            outcome.abstained = True
        else:
            outcome = pipeline.answer(outcome, cfg.generation, llm=llm,
                                      models=roles)
    outcome.diagnostics = dict(outcome.diagnostics) | {
        'agent_scope': agent_cfg.scope,
        'agent_hops': final.get('hops', 0),
        'agent_rewrites': final.get('rewrites', 0),
        'agent_revisions': final.get('revisions', 0),
        'agent_calls': final.get('calls', 0),
        'agent_unparsed': final.get('unparsed', 0),
        'agent_stop': stop}
    if final.get('error'):
        # Why the agent gave up, when it gave up because it could not reach its
        # model. Same argument as `answer_error` one stage down: without it, a
        # CliError and "the diary is silent" are the same row, and RAGAS judges
        # both with confident, low faithfulness.
        outcome.diagnostics['agent_error'] = final['error']
    outcome.timings = dict(outcome.timings) | {
        'agent_ms': round((time.perf_counter() - started) * 1000, 1)}
    if trace is not None:
        winning = hop_traces.get(best['hop']) or {}
        trace.update(winning)
        trace['agent'] = visits
        trace['agent_hop'] = best['hop']
    return outcome


def note_for(cfg: AgentConfig) -> str:
    """One line describing the loop, for a run's notes.

    The caps are named, not only the scope, for the reason `models.note_for`
    names the CLI effort: they move the numbers while leaving the label
    identical, so two rows differing only in `max_hops` would be ranked as a
    comparable pair. The config dict on the row carries the values; this is what
    a reader sees without opening it.
    """
    parts = [f'agent scope={cfg.scope}']
    if owns_retrieval(cfg.scope):
        parts.append(f'max_hops={cfg.max_hops}')
        parts.append('rewrite' if cfg.rewrite else 'no rewrite')
        parts.append(f'evidence>={cfg.evidence_threshold}')
    if owns_generation(cfg.scope):
        parts.append(f'critic={cfg.critic}')
        parts.append(f'max_revisions={cfg.max_revisions}')
    parts.append(f'<={cfg.max_llm_calls} calls/question')
    return ', '.join(parts)


def _terminal(cfg: AgentConfig, state: dict) -> str:
    """Why the loop stopped, when no cap or failure claimed it first.

    Named rather than inferred by a reader: "found what it needed", "ran out of
    hops" and "ran out of revisions" are three different findings about a
    configuration, and the July 2026 post-mortem is the cost of having to
    reconstruct which one happened by hand.
    """
    if owns_generation(cfg.scope):
        if cfg.critic == 'none':
            return 'drafted'
        if state.get('sufficient'):
            return 'grounded'
        if state.get('revisions', 0) >= cfg.max_revisions:
            return 'revision-cap'
        return 'drafted'
    if state.get('sufficient'):
        return 'evidence-sufficient'
    if state.get('hops', 0) >= cfg.max_hops:
        return 'hop-cap'
    return 'stopped'
