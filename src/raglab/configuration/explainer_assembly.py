"""Assembles the explainer text for every knob and model role, served over the API.

Text lives beside the definition it describes (`config.HELP`, `models.ROLES`);
this module only assembles it and reports what is missing — `missing()`
returning anything is a test failure, so a new field cannot ship unexplained.
"""
from raglab.configuration.lab_config import (
    HELP,
    AgentConfig,
    GenerationConfig,
    IndexConfig,
    RetrievalConfig)
from raglab.evaluation.deterministic_metrics import (
    AGGREGATED,
    MEASURE_HELP,
    MEASURES)
from raglab.llm_backends.model_role_catalogue import ROLE_HELP, ROLES
from raglab.evaluation.ragas_judged_metrics import (
    LLM_METRICS,
    OFFLINE_METRICS,
    RAGAS_MEASURE_HELP,
    RAGAS_MEASURES)

GROUPS = (('index', IndexConfig), ('retrieval', RetrievalConfig),
          ('generation', GenerationConfig), ('agent', AgentConfig))


def topics() -> dict[str, str]:
    """Every explainable key → its text: '<group>.<field>', 'model.<role>', 'run.<field>', 'metric.<key>'."""
    return (dict(HELP) | dict(ROLE_HELP) | dict(MEASURE_HELP)
            | dict(RAGAS_MEASURE_HELP))


def measures() -> list[dict]:
    """Every metric the panel can print, ours then RAGAS's, in display order."""
    return [measure.as_dict() for measure in MEASURES + RAGAS_MEASURES]


def missing_metrics() -> list[str]:
    """Metrics a run can report with nothing explaining them."""
    defined = {measure.key for measure in MEASURES + RAGAS_MEASURES}
    reported = (set(AGGREGATED) | {'headline'} | set(OFFLINE_METRICS)
                | set(LLM_METRICS))
    return sorted(reported - defined)


def model_fields() -> set[str]:
    """Config fields that are explained by a model role rather than by HELP."""
    return {role.field for role in ROLES}


def missing() -> list[str]:
    """Configuration fields with no explainer, in '<group>.<field>' form."""
    covered = set(topics()) | model_fields()
    return [f'{group}.{name}'
            for group, kind in GROUPS
            for name in kind.__dataclass_fields__
            if f'{group}.{name}' not in covered]
