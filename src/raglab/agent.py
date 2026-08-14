"""The scoped RAG agent: a bounded LangGraph loop around the pipeline's own
stages. `AgentConfig.scope` picks retrieve/generate/full; the loop always calls
`pipeline.retrieve` with the run's own `RetrievalConfig`, never its own retrieval.
"""
import re
import time
from typing import Any, TypedDict

from . import pipeline
from .config import AgentConfig, LabConfig
from .llm import lab_chat
from .models import Roles

AGENT_EXTRA = 'uv sync --extra agent'

# A constant, not a knob: the critic answers a yes/no question, and the "how hard should it try" knob already exists as `evidence_threshold`.
CRITIC_BAR = 0.5


def agent_available() -> bool:
    """Verified by import, not read off a list, so NA means one thing: this installation cannot load it."""
    try:
        import langgraph                                    # noqa: F401
        from langgraph.graph import StateGraph              # noqa: F401
    except Exception:
        return False
    return True


def _stategraph():
    """The one place `_Loop.build_graph` reaches into langgraph. Deferred like
    `agent_available`'s own import above rather than a module-level import, so
    `import raglab.agent` still succeeds when the `agent` extra is not
    installed — only a call to `run` needs this, and `LabConfig.validate()`
    already refused a scope this installation cannot run before `run` is
    ever reached."""
    from langgraph.graph import END, StateGraph
    return END, StateGraph


def available() -> dict:
    """Every scope → whether this installation can run it, and what to install if not."""
    from .config import SCOPES
    ready = agent_available()
    return {scope: {'available': True if not scope else ready,
                    'install': '' if not scope else AGENT_EXTRA}
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
    """A model's verdict in [0,1], or None when it gave none. Unlike
    `retrieval.llm_scores`'s 0.5, never defaulted to a number: a single verdict
    deciding whether the loop stops cannot be split that way."""
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
    """The per-question call ceiling was reached; reported as `agent_stop='call-cap'`."""


def _ask(llm, model: str, node: str, system: str, user: str) -> str:
    """Every model call routes through here, so the ceiling cannot be bypassed
    by a node added later. `node` is otherwise unused, kept so a test can
    identify the call site without parsing prompt text."""
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


def _contexts_block(outcome) -> str:
    """Exactly what the answerer is handed (`pipeline.context_blocks`), so every
    node judges the same text rather than a truncated view."""
    if outcome is None or not outcome.contexts:
        return '(nothing retrieved)'
    return pipeline.context_blocks(outcome)


# --- the graph's shape, as data -------------------------------------------

class State(TypedDict, total=False):
    question: str
    query: str
    plan: str
    # LangGraph reads this class to know its channels; an update naming an undeclared key is refused.
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
    """This scope's nodes/edges as data for `graph_nodes`/`graph_edges`; `_Loop.build_graph` hand-wires its own edges rather than reading this table, and the two genuinely disagree on one declared-but-unreachable path-map entry (`rewrite=True`'s unconditional `'retrieve'` from `assess`) — so trust the hand-wiring, not this table, if they ever seem to disagree."""
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
                # The interaction term: only `full` can answer a bad critique with different evidence rather than rewording.
                edges.append(('critique', 'retrieve'))
    return tuple(nodes), tuple(edges)


def graph_nodes(cfg: AgentConfig) -> tuple[str, ...]:
    return _shape(cfg)[0]


def graph_edges(cfg: AgentConfig) -> tuple[tuple[str, str], ...]:
    return _shape(cfg)[1]


def _guard(fn):
    """Wraps every node: an existing stop is honoured, the call ceiling ends
    the loop rather than crashing it, and an unreachable model's reason lands
    on `state['stop']` instead of taking the run down."""
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


class _Loop:
    """One question's mutable loop state plus the resources every node and router needs, so they read `self.x` rather than each closing over the same names; one instance per call to `run`, never reused across questions."""

    def __init__(self, index, cfg: LabConfig, question: str, query_date: str,
                llm, roles: Roles, trace: dict | None):
        self.index = index
        self.cfg = cfg
        self.agent_cfg = cfg.agent
        self.question = question
        self.query_date = query_date
        self.llm = llm
        self.roles = roles
        self.trace = trace
        self.visits: list[dict] = []
        self.hop_traces: dict[int, dict] = {}
        # A later hop wins only on a strictly higher verdict, so a bad rewrite cannot spend evidence already found.
        self.best: dict[str, Any] = {'outcome': None, 'verdict': -1.0, 'hop': 0}
        # Set only once `build_graph` has imported langgraph; the routers
        # that read it only ever run inside a compiled graph, after that.
        self.end: str = ''
        # A scope without retrieval retrieves exactly once, before the graph runs.
        self.initial: State = {'question': question, 'query': question,
                               'hops': 0, 'rewrites': 0, 'revisions': 0,
                               'calls': 0, 'unparsed': 0}

    def note(self, node: str, hop: int, detail: str = '') -> None:
        self.visits.append({'node': node, 'hop': hop, 'detail': detail[:200]})

    def ask(self, state: State, node: str, model: str, system: str, user: str
           ) -> tuple[str, dict]:
        """A model call, with the ceiling checked *before* it is spent."""
        if state.get('calls', 0) >= self.agent_cfg.max_llm_calls:
            raise _Exhausted()
        text = _ask(self.llm, model, node, system, user)
        return text, {'calls': state.get('calls', 0) + 1}

    # --- nodes ---------------------------------------------------------

    def do_retrieve(self, state: State) -> dict:
        hop = state.get('hops', 0) + 1
        hop_trace: dict = {} if self.trace is not None else None
        outcome = pipeline.retrieve(self.index, self.cfg.retrieval,
                                    state['query'], self.query_date,
                                    llm=self.llm, models=self.roles,
                                    trace=hop_trace)
        if hop_trace is not None:
            self.hop_traces[hop] = hop_trace
        if self.best['outcome'] is None:
            self.best.update(outcome=outcome, verdict=-1.0, hop=hop)
        self.note('retrieve', hop, f'{len(outcome.contexts)} contexts for '
                                   f'{state["query"][:60]}')
        return {'hops': hop, 'last': outcome}

    def do_plan(self, state: State) -> dict:
        text, spent = self.ask(state, 'plan', self.roles.plan, PLAN_PROMPT,
                               f'Question: {state["question"]}')
        self.note('plan', 0, text)
        return spent | {'plan': text}

    def do_assess(self, state: State) -> dict:
        outcome = state.get('last')
        text, spent = self.ask(state, 'assess', self.roles.plan, ASSESS_PROMPT,
                               f'Question: {state["question"]}\n'
                               f'What we are looking for: {state.get("plan", "")}\n\n'
                               f'Excerpts:\n{_contexts_block(outcome)}')
        score = verdict(text)
        self.note('assess', state.get('hops', 0),
                  f'{"unparsed" if score is None else round(score, 2)}')
        # Unparsed means insufficient — never a value that clears the threshold.
        if score is not None and score > self.best['verdict']:
            self.best.update(outcome=outcome, verdict=score,
                            hop=state.get('hops', 0))
        return spent | {
            'sufficient': bool(score is not None
                               and score >= self.agent_cfg.evidence_threshold),
            'unparsed': state.get('unparsed', 0) + (1 if score is None else 0)}

    def do_rewrite(self, state: State) -> dict:
        text, spent = self.ask(state, 'rewrite', self.roles.plan, REWRITE_PROMPT,
                               f'Question: {state["question"]}\n'
                               f'Already tried: {state["query"]}\n'
                               f'Still missing: {state.get("plan", "")}')
        self.note('rewrite', state.get('hops', 0), text)
        # An empty reply leaves the query alone rather than searching for nothing.
        return spent | {'query': text or state['query'],
                        'rewrites': state.get('rewrites', 0) + 1}

    def do_draft(self, state: State) -> dict:
        outcome = self.best['outcome']
        previous = state.get('draft')
        user = (f'سؤال: {state["question"]}\n\n'
                f'تکه‌های دفترچه:\n{_contexts_block(outcome)}')
        if previous:
            user += (f'\n\nپیش‌نویس قبلی که رد شد:\n{previous}\n'
                     'دوباره بنویس و فقط به تکه‌های بالا تکیه کن.')
        text, spent = self.ask(state, 'draft', self.roles.answer,
                               pipeline.ANSWER_PROMPT, user)
        self.note('draft', state.get('hops', 0), text)
        return spent | {
            'draft': text or pipeline.REFUSAL,
            'revisions': state.get('revisions', 0) + (1 if previous else 0)}

    def do_critique(self, state: State) -> dict:
        outcome = self.best['outcome']
        body = (f'Question: {state["question"]}\n\nAnswer:\n{state["draft"]}\n\n'
                f'Excerpts:\n{_contexts_block(outcome)}')
        text, spent = self.ask(state, 'critique', self.roles.critic,
                               CRITIQUE_PROMPT, body)
        grounded = verdict(text)
        unparsed = 1 if grounded is None else 0
        self.note('critique', state.get('hops', 0),
                  f'grounded={"unparsed" if grounded is None else round(grounded, 2)}')
        passed = grounded is not None and grounded >= CRITIC_BAR
        state = dict(state) | spent
        if passed and self.agent_cfg.critic == 'both':
            text, more = self.ask(state, 'completeness', self.roles.critic,
                                  COMPLETENESS_PROMPT, body)
            spent = {'calls': more['calls']}
            complete = verdict(text)
            unparsed += 1 if complete is None else 0
            self.note('completeness', state.get('hops', 0),
                      f'complete={"unparsed" if complete is None else round(complete, 2)}')
            passed = complete is not None and complete >= CRITIC_BAR
        return spent | {'sufficient': passed,
                        'unparsed': state.get('unparsed', 0) + unparsed}

    # --- routing ---------------------------------------------------------

    def after_assess(self, state: State) -> str:
        if state.get('stop'):
            return self.end
        if state.get('sufficient'):
            return 'draft' if owns_generation(self.agent_cfg.scope) else self.end
        if state.get('hops', 0) >= self.agent_cfg.max_hops:
            # Out of hops: still draft (with the best evidence found) under a generation scope.
            return 'draft' if owns_generation(self.agent_cfg.scope) else self.end
        return 'rewrite' if self.agent_cfg.rewrite else 'retrieve'

    def after_draft(self, state: State) -> str:
        if state.get('stop'):
            return self.end
        return 'critique' if self.agent_cfg.critic != 'none' else self.end

    def after_critique(self, state: State) -> str:
        if state.get('stop'):
            return self.end
        if state.get('sufficient'):
            return self.end
        if state.get('revisions', 0) >= self.agent_cfg.max_revisions:
            return self.end
        # Only `full` may answer a bad critique with different evidence; every other scope can only redraft.
        if (owns_retrieval(self.agent_cfg.scope)
                and state.get('hops', 0) < self.agent_cfg.max_hops):
            return 'retrieve'
        return 'draft'

    # --- graph construction and execution ---------------------------------

    def build_graph(self):
        """The hand-wiring `run` used to do inline, one edge per line, kept
        exactly as measured — see `_shape`'s docstring for why this is not
        driven from `_shape`'s own edge table."""
        end, StateGraph = _stategraph()
        self.end = end
        agent_cfg = self.agent_cfg
        graph = StateGraph(State)
        builders = {'plan': self.do_plan, 'retrieve': self.do_retrieve,
                   'assess': self.do_assess, 'rewrite': self.do_rewrite,
                   'draft': self.do_draft, 'critique': self.do_critique}
        for name in graph_nodes(agent_cfg):
            graph.add_node(name, _guard(builders[name]))
        if owns_retrieval(agent_cfg.scope):
            graph.set_entry_point('plan')
            graph.add_edge('plan', 'retrieve')
            graph.add_edge('retrieve', 'assess')
            targets = ['retrieve', end] + (['draft']
                                           if owns_generation(agent_cfg.scope) else [])
            if agent_cfg.rewrite:
                targets.append('rewrite')
                graph.add_edge('rewrite', 'retrieve')
            graph.add_conditional_edges('assess', self.after_assess, targets)
        else:
            graph.set_entry_point('draft')
        if owns_generation(agent_cfg.scope):
            if agent_cfg.critic == 'none':
                graph.add_edge('draft', end)
            else:
                graph.add_conditional_edges('draft', self.after_draft,
                                            ['critique', end])
                after = ['draft', end]
                if owns_retrieval(agent_cfg.scope):
                    after.append('retrieve')
                graph.add_conditional_edges('critique', self.after_critique, after)
        return graph

    def fixed_retrieve(self) -> None:
        """The retrieval a scope with no retrieval hop still needs, run once
        before the graph rather than inside a node — `owns_retrieval` is
        false for this scope, so no node here ever calls `pipeline.retrieve`
        itself."""
        fixed_trace: dict = {} if self.trace is not None else None
        outcome = pipeline.retrieve(self.index, self.cfg.retrieval,
                                    self.question, self.query_date,
                                    llm=self.llm, models=self.roles,
                                    trace=fixed_trace)
        if fixed_trace is not None:
            self.hop_traces[1] = fixed_trace
        self.best.update(outcome=outcome, verdict=-1.0, hop=1)
        self.initial |= {'hops': 1, 'last': outcome}
        self.note('retrieve', 1,
                 f'{len(outcome.contexts)} contexts (fixed retrieval)')

    def invoke(self, graph) -> State:
        # Derived from the caps, not LangGraph's default 25, so a hit ceiling reports a cap the config actually set.
        rounds = 4 * (self.agent_cfg.max_hops + self.agent_cfg.max_revisions) + 10
        return dict(graph.compile().invoke(
            self.initial, config={'recursion_limit': rounds}))

    def finalize(self, final: State, started: float) -> pipeline.Outcome:
        """The loop's final state, turned into the `Outcome` shape the fixed
        pipeline returns: the best hop's evidence, the winning answer (or the
        canonical refusal), diagnostics naming why the loop stopped, timings,
        and — when the caller asked for one — the Inspector's trace."""
        agent_cfg = self.agent_cfg
        outcome = self.best['outcome']
        if outcome is None:
            # The loop died before retrieving anything; an empty Outcome so the caller still gets one shape.
            outcome = pipeline.Outcome(question=self.question, contexts=[],
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
            # The fixed answerer, exactly as an unagented run would call it — this scope owns retrieval only.
            if stop == 'error':
                outcome.answer = pipeline.REFUSAL
                outcome.abstained = True
            else:
                outcome = pipeline.answer(outcome, self.cfg.generation,
                                          llm=self.llm, models=self.roles)
        outcome.diagnostics = dict(outcome.diagnostics) | {
            'agent_scope': agent_cfg.scope,
            'agent_hops': final.get('hops', 0),
            'agent_rewrites': final.get('rewrites', 0),
            'agent_revisions': final.get('revisions', 0),
            'agent_calls': final.get('calls', 0),
            'agent_unparsed': final.get('unparsed', 0),
            'agent_stop': stop}
        if final.get('error'):
            # Names why the agent gave up when it could not reach its model, so that and a genuine refusal aren't the same row.
            outcome.diagnostics['agent_error'] = final['error']
        outcome.timings = dict(outcome.timings) | {
            'agent_ms': round((time.perf_counter() - started) * 1000, 1)}
        if self.trace is not None:
            winning = self.hop_traces.get(self.best['hop']) or {}
            self.trace.update(winning)
            self.trace['agent'] = self.visits
            self.trace['agent_hop'] = self.best['hop']
        return outcome


def run(index, cfg: LabConfig, question: str, query_date: str, llm=None,
        models: Roles | None = None, trace: dict | None = None):
    """One question through the scoped loop, returning the same `Outcome`
    shape the fixed pipeline returns, so an agent row is comparable with a
    pipeline row on every metric the lab already computes."""
    if not cfg.agent.scope:
        raise ValueError(
            'agent.run needs a scope — the fixed pipeline is pipeline.retrieve. '
            'A caller that reaches here with no scope would produce a row '
            'labelled with an agent that never ran.')
    started = time.perf_counter()
    loop = _Loop(index, cfg, question, query_date, llm, models or Roles(), trace)

    if not owns_retrieval(loop.agent_cfg.scope):
        loop.fixed_retrieve()

    if loop.best['outcome'] is not None and not loop.best['outcome'].contexts:
        # Nothing to draft or critique from — the same rule pipeline.answer applies to an abstained outcome.
        final: State = dict(loop.initial) | {'stop': 'abstained'}
    else:
        final = loop.invoke(loop.build_graph())

    return loop.finalize(final, started)


def note_for(cfg: AgentConfig) -> str:
    """One line describing the loop for a run's notes. Caps are named, not just
    the scope, since they move the numbers while leaving the label identical."""
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
    """Why the loop stopped, when no cap or failure claimed it first."""
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
