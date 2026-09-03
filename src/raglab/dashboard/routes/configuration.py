"""Everything the panel needs to render itself, and whether the lab is up.

`/api/options` is one response because the panel is one page: the closed
vocabularies, the knob defaults and their dependency table, the pipeline steps
and their ink, the model catalogues this installation can actually load, every
explainer in two lengths, the loaded corpus, and what is installed. Served
rather than duplicated into each surface, so two panels cannot grey out
different knobs for the same stated reason.
"""
from pathlib import Path

from raglab.configuration.lab_config import (
    ANSWERERS,
    CHUNKERS,
    DEPENDENCIES,
    EMBEDDERS,
    GRADERS,
    GRAPH_SOURCES,
    HIERARCHIES,
    RERANKERS,
    RETRIEVERS,
    ROOT,
    RUNS_DIR,
    STEPS,
    SUMMARIZERS,
    SUMMARY_SCOPES,
    LabConfig)
from raglab.configuration import explainer_assembly as explain
from raglab.corpora import dataset_import_contract as datasets
from raglab.dashboard.routes.datasets import _dataset_options, _question_vocab
from raglab.evaluation import ragas_judged_metrics as ragas_eval
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.llm_backends import model_role_catalogue as models
from raglab.llm_backends import openrouter_key_memory as credentials
from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.rag_components.indexing import (
    summary_hierarchy_builder as hierarchy)
from raglab.rag_components.retrieval import (
    retrieve_fuse_rerank_grade as retrieval)

def _relative(path: Path) -> str:
    """Repo-relative path for the panel, or absolute when it's outside the repo (`relative_to` raises)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _catalogue_vocab() -> dict:
    """The closed vocabularies for the three pipeline stages: chunker/embedder, retriever/reranker, grader/answerer."""
    return {
        'chunkers': list(CHUNKERS), 'embedders': list(EMBEDDERS),
        'retrievers': list(RETRIEVERS), 'rerankers': list(RERANKERS),
        'graders': list(GRADERS), 'answerers': list(ANSWERERS),
    }


def _hierarchy_options() -> dict:
    """The summary hierarchy: grouping, graph edges, summariser, and what retrieval may do with the rows written."""
    return {
        'hierarchies': list(HIERARCHIES),
        'graph_sources': list(GRAPH_SOURCES),
        'summarizers': list(SUMMARIZERS),
        'summary_scopes': list(SUMMARY_SCOPES),
        # Verified by import, never guessed, so NA keeps meaning one thing:
        # this installation cannot load it.
        'hierarchy_support': hierarchy.available(),
    }


def _config_defaults() -> dict:
    served = LabConfig().to_dict()
    # The panel starts on the served DEFAULT corpus, named explicitly. The
    # dataclass default stays '' — the legacy spelling of BUILTIN that keeps
    # every recorded fingerprint meaning what it meant.
    served['index']['dataset'] = datasets.DEFAULT
    return {
        # Served rather than duplicated per panel, so both grey out the same
        # knobs for the same stated reason.
        'dependencies': DEPENDENCIES,
        'defaults': served,
    }


def _step_list() -> dict:
    return {'steps': [{'key': step.key, 'short': step.short, 'label': step.label,
                       'note': step.note} for step in STEPS]}


def _model_catalogues(live) -> dict:
    return {
        'embedder_hints': embedding.embedder_hints(live),
        'embed_models': embedding.embed_model_catalogue(live),
        'models': models.catalogue(live),
        'model_roles': [role.as_dict() for role in models.ROLES],
        'modes': models.mode_catalogue(live),
    }


def _metric_help() -> dict:
    # Label, step, formula and library per metric, so a name cannot
    # drift from its definition. Two lengths of every explainer, from one
    # source: `brief` is the opening sentence of `help`, taken by
    # `explain.briefs()` rather than written a second time, so a hover box and
    # the text it opens out into cannot come to say different things.
    return {'metrics': explain.measures(), 'help': explain.topics(),
            'brief': explain.briefs()}


def _corpus_summary(diary: dict, ground_truth: dict) -> dict:
    documents = diary.get('corpus_documents') or []
    return {'corpus': {
        'documents': len(documents),
        'parts': sum(len(document.get('document_content') or [])
                     for document in documents),
        'questions': len(ground_truth.get('groundtruth_dataset') or []),
        'query_date': (ground_truth.get('groundtruth_dataset_metadata') or {}
                       ).get('default_question_asked_at', '')[:10],
    }}


def _capabilities(live) -> dict:
    return {'capabilities': {
        'fastembed': embedding.fastembed_available(),
        'sentence_transformers': embedding.sentence_transformers_available(),
        'cross_encoder': retrieval.cross_encoder_available(
            live.cross_encoder_model),
        'cross_encoder_model': live.cross_encoder_model,
        'fastembed_model': live.fastembed_model,
        # 'a real model is reachable', not 'a key exists' — under
        # RAGLAB_LLM=ollama every stage runs locally with no key.
        'llm': live.llm_ready,
        'llm_provider': live.provider,
        'llm_model': live.llm_model,
        'ollama_base_url': live.ollama_base_url,
        'ragas': ragas_eval.availability(live).as_dict(),
        'openrouter_key': credentials.state(live),
        # Stated positively, since the index is thrown away with the
        # process rather than merely "no service named".
        'storage': {'index': 'memory',
                    'runs': str(RUNS_DIR.relative_to(ROOT)),
                    'experiments': _relative(ledger.db_path())},
    }}


def register(app, context) -> None:
    settings_now, registry = context.settings_now, context.registry
    diary, ground_truth = context.corpus, context.ground_truth

    @app.get('/api/options')
    def options():
        """Everything the panel needs to render itself, including what is actually installed."""
        live = settings_now()
        return (_catalogue_vocab() | _hierarchy_options()
                | _question_vocab(ground_truth) | _config_defaults() | _step_list()
                | _model_catalogues(live) | _metric_help()
                | _corpus_summary(diary, ground_truth) | _capabilities(live)
                | _dataset_options() | {'indexes': registry.known()})

    @app.get('/api/health')
    def health():
        # No dependency to report: the lab is up or it is not running.
        return {'ok': True, 'storage': 'memory'}
