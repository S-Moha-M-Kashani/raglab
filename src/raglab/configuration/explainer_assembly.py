"""Assembles the explainer text for every knob and model role, served over the API.

Text lives beside the definition it describes (`config.HELP`, `models.ROLES`);
this module only assembles it and reports what is missing — `missing()`
returning anything is a test failure, so a new field cannot ship unexplained.
"""
import re

from raglab.configuration.knob_help_text import DATASET_SPECIFIC as HELP_FLAG
from raglab.configuration.lab_config import (
    HELP,
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
          ('generation', GenerationConfig))

# --- the brief ------------------------------------------------------------
# Every explainer is now read in two lengths: a brief on hover, the whole text
# on a click. A brief is not new information and it is not a second source of
# truth — it is the *opening sentence* of the text that already lives beside
# the definition, taken here, in the one module whose job is assembling that
# text. So there is nothing to keep in step: rewrite the help and the brief
# follows.
#
# The exceptions are below, and they are exceptions to a rule rather than a
# free-text field: a topic lands here only when its opening sentence cannot
# serve as a brief, and `missing_briefs()` — a gate, like `missing()` — says
# which those are. Three ways a sentence fails:
#
#   too long   — the sentence runs past BRIEF_LIMIT (`index.embed_model` opens
#                with 423 characters), and a hover box the size of a paragraph
#                is the thing the brief exists to avoid;
#   too short  — it names the topic without saying anything ("Maximal Marginal
#                Relevance.", "The SQuAD-style measure.");
#   the flag   — the text opens with the shared DATASET_SPECIFIC caveat, so its
#                first sentence describes every dataset-specific knob equally
#                and this one not at all.
BRIEF_LIMIT = 150
BRIEF_FLOOR = 25

BRIEF = {
    # too long
    'index.embed_model':
        'Which embedding model the backend loads — the one choice a '
        'non-English corpus is won or lost on. Changing it rebuilds the index.',
    'index.hierarchy':
        'Groups the chunks and indexes one summary per group beside them; the '
        'leaves always stay. Not GraphRAG — the nodes are chunks.',
    'index.summarizer':
        'How a group becomes one piece of text, without a model: nearest '
        'members, rare-word sentences, coverage picks, or a card of counts.',
    'retrieval.retriever':
        'Whether the search matches meaning (dense), words (bm25), or both '
        'rankings fused (hybrid-rrf).',
    'retrieval.grader':
        'The gate that makes abstention possible: chunks below the threshold '
        'are dropped, and if none survive the pipeline refuses.',
    'metric.factual_correctness(mode=f1)':
        'Whether the answer matches the reference answer, both directions '
        'counted — being right, not merely grounded. A judge model scores it.',
    'metric.false_abstention':
        'How often the evidence was there and the answerer still said it had '
        'nothing. Read it beside abstention.',
    'metric.headline':
        'One comparable number for the leaderboard: retrieval, quote '
        'survival, ranking and correct refusal, weighted in that order.',
    'metric.non_llm_context_recall':
        "RAGAS's context recall with the judge removed: is each ground-truth "
        'quote inside some retrieved chunk? It does not vote.',
    'model.expand':
        "The model that writes HyDE's hypothetical answer. Multi-query "
        'expansion is rule-based and uses none.',
    # too short
    'retrieval.mmr_lambda':
        'Maximal Marginal Relevance: at 1.0 the top k are simply the '
        'best-scoring chunks; lower it to spread them across documents.',
    'generation.answerer':
        'Who writes the answer: nobody ("none" measures retrieval alone), a '
        'quote from the top chunks ("extractive"), or a model ("llm").',
    'metric.answer_token_f1':
        'Word overlap between the answer and the reference, both directions — '
        'the F1 that does not punish a short correct answer.',
    # the flag
    'retrieval.time_filter':
        'Reads time language in the question as a date range and restricts '
        'the search to it. Dataset-specific: written for the Farsi diary.',
    'retrieval.multi_query':
        'Searches several rule-based rewrites of the question and merges the '
        'hits, with no model call. Dataset-specific: the rewrites are Persian.',
    'retrieval.hyde':
        'Writes a hypothetical answer with a model and searches with that '
        'instead of the question. One LLM call each; the prompt is Persian.',
}

# The flag's own opening words, read rather than repeated: a text that starts
# with them cannot have its first sentence taken as a brief.
_FLAG = HELP_FLAG.split(':')[0] + ':'


def opening_sentence(text: str) -> str:
    """The text up to its first sentence end. Whitespace-anchored on purpose:
    `0.5` and `metrics.score_question` are not sentence ends."""
    return re.split(r'(?<=[.!?])\s', text.strip(), maxsplit=1)[0]


def brief(topic: str, text: str) -> str:
    """One sentence for a hover box: the declared brief, or the text's own
    opening sentence where that serves."""
    return BRIEF.get(topic) or opening_sentence(text)


def briefs() -> dict[str, str]:
    """Every explainable key -> its one-sentence version, same keys as `topics()`."""
    return {topic: brief(topic, text) for topic, text in topics().items()}


def missing_briefs() -> list[str]:
    """Topics whose brief is unusable, with the reason — a gate, like `missing()`.

    A failure here is never fixed by loosening the rule: either the help text's
    opening sentence is rewritten to stand on its own, or the topic declares a
    brief in `BRIEF`.
    """
    problems = []
    for topic, text in sorted(topics().items()):
        one = brief(topic, text)
        if not one:
            problems.append(f'{topic}: no text at all')
        elif len(one) > BRIEF_LIMIT:
            problems.append(f'{topic}: {len(one)} characters, over {BRIEF_LIMIT}')
        elif len(one) < BRIEF_FLOOR:
            problems.append(f'{topic}: {len(one)} characters, under {BRIEF_FLOOR}')
        elif one.startswith(_FLAG):
            problems.append(f'{topic}: opens with the dataset-specific flag, '
                            'which describes every such knob and none of them')
        elif one[-1] not in '.!?':
            problems.append(f'{topic}: does not end a sentence')
    return problems


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
