"""The entry seam callers use: one question through the scoped loop, and the
one line describing the loop for a run's notes."""
import time

from raglab.configuration.lab_config import AgentConfig, LabConfig
from raglab.llm_backends.model_role_catalogue import Roles
from raglab.agents.agentic_rag.loop import _Loop
from raglab.agents.agentic_rag.shape import owns_generation, owns_retrieval


def run(index, cfg: LabConfig, question: str, query_date: str, llm=None,
        models: Roles | None = None, trace: dict | None = None):
    """One question through the scoped loop, returning the same `Outcome`
    shape the fixed pipeline returns, so an agent row is comparable with a
    pipeline row on every metric the lab already computes."""
    if not cfg.agent.scope:
        raise ValueError(
            'agentic_rag.run needs a scope — the fixed pipeline is '
            'pipeline.retrieve. A caller that reaches here with no scope would '
            'produce a row labelled with an agent that never ran.')
    started = time.perf_counter()
    loop = _Loop(index, cfg, question, query_date, llm, models or Roles(), trace)

    if not owns_retrieval(loop.agent_cfg.scope):
        loop.fixed_retrieve()

    if loop.best['outcome'] is not None and not loop.best['outcome'].contexts:
        # Nothing to draft or critique from — the same rule pipeline.answer applies to an abstained outcome.
        final = dict(loop.initial) | {'stop': 'abstained'}
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
