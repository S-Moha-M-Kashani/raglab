"""The measured stages, and the job table all three of them run on.

Building an index, retrieving over the selected questions, and asking one
question end to end are three routes rather than three flags on one, so the
step that is cheap to repeat while a knob moves stays cheap. None of them is a
request: each answers 202 with a job id and a Location the panel polls, because
the work outwaits any HTTP timeout. Preconditions still refuse synchronously.
"""
from fastapi import HTTPException

from raglab.configuration.lab_config import (
    LabConfig,
    settings_for_provider)
from raglab.corpora import dataset_import_contract as datasets
from raglab.dashboard.service_presentation import (
    chunks_by_session,
    mark_gold,
    summary_rows)
from raglab.dashboard.service_route_plumbing import (
    _accepted,
    _with_backend,
    cancel_checker,
    scaled_progress,
    screen)
from raglab.evaluation import run_evaluation as evaluate
from raglab.llm_backends import model_role_catalogue as models
from raglab.llm_backends.chat_model_factory import lab_llm
from raglab.rag_components import question_to_answer_pipeline as pipeline


def register(app, context) -> dict:
    """Returns the job-table operations the Inspector reads through."""
    settings_now, questions_for = context.settings_now, context.questions_for
    registry, dataset_lock, jobs = (
        context.registry, context.dataset_lock, context.jobs)
    ground_truth = context.ground_truth

    @app.post('/api/indexes')
    def build_index(payload: dict):
        cfg = LabConfig.from_dict(payload)
        force = bool(payload.get('force'))
        # The same screen the run routes apply: a missing library fails as a
        # 400 naming what to install, not a 500 from an import three frames
        # down. Only the index half is checked — a build reads no model.
        problems = [p for p in cfg.validate()
                    if not p.startswith(('unknown retriever', 'unknown reranker',
                                         'unknown grader', 'unknown answerer',
                                         'unknown summary_scope', 'k must be'))]
        if problems:
            raise HTTPException(400, '; '.join(problems))

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled)
            check_cancelled()
            with dataset_lock(cfg.index.dataset), registry.hold(cfg.index):
                check_cancelled()
                index = registry.get(cfg.index, progress=report, force=force)
                return {'collection': index.stats.collection,
                        'chunks': index.stats.chunks,
                        'leaves': index.stats.leaves,
                        'avg_chars': index.stats.avg_chars,
                        'p95_chars': index.stats.p95_chars,
                        'embed_dim': index.stats.embed_dim,
                        'build_seconds': index.stats.build_seconds,
                        # None on a flat build, distinguishing "no hierarchy"
                        # from "a hierarchy that found nothing".
                        'hierarchy': index.stats.hierarchy,
                        'reused': index.stats.reused, 'notes': index.stats.notes,
                        # So a follower (the Inspector) can render what was
                        # built without holding its own index; both halves,
                        # since `chunks_by_session` alone omits every summary a
                        # grouping wrote.
                        'chunks_by_session': chunks_by_session(index),
                        'summaries': summary_rows(index)}

        return _accepted(jobs.start('index', work, config=cfg.to_dict()))

    @app.post('/api/retrievals')
    def start_retrieval(payload: dict):
        """Retrieval only, over the questions the eval card has selected — no answering, no judging, no run file.

        Its own route rather than a flag on `/api/evaluations`, so it stays the
        step affordable to repeat while moving one knob. Takes the same
        selection arguments as an evaluation on purpose."""
        cfg = LabConfig.from_dict(payload)
        run_settings = settings_for_provider(settings_now(),
                                             payload.get('provider') or '')
        screen(cfg, run_settings)

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled)
            check_cancelled()
            with dataset_lock(cfg.index.dataset), registry.hold(cfg.index):
                check_cancelled()
                return evaluate.run_retrieval(
                    registry, questions_for(cfg), cfg, run_settings,
                    labels=payload.get('labels') or None,
                    limit=payload.get('limit') or None,
                    balance=payload.get('balance') or '',
                    progress=report, cancelled=check_cancelled)

        return _accepted(jobs.start('retrieve', work,
                                    config=_with_backend(cfg, run_settings)))

    @app.get('/api/jobs')
    def list_jobs():
        """Every job this process has run, newest first — id/kind/state/config only, not the full result."""
        return {'jobs': jobs.list()}

    @app.get('/api/jobs/{job_id}')
    def job_status(job_id: str):
        return jobs.get(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel_job(job_id: str):
        return jobs.cancel(job_id)

    @app.post('/api/queries')
    def ad_hoc_query(payload: dict):
        """Run one question through the current settings and return every stage — still a job, since the
        index it builds implicitly can outwait any HTTP timeout; preconditions still refuse synchronously."""
        cfg = LabConfig.from_dict(payload)
        question = (payload.get('question') or '').strip()
        if not question:
            raise HTTPException(400, 'question is required')
        # The same screen /api/evaluations applies, so the two routes cannot
        # disagree about which configs are legal.
        run_settings = settings_for_provider(settings_now(),
                                             payload.get('provider') or '')
        screen(cfg, run_settings)
        requested_query_date = payload.get('query_date')

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled)
            check_cancelled()
            with dataset_lock(cfg.index.dataset), registry.hold(cfg.index):
                check_cancelled()
                asked = questions_for(cfg)
                query_date = requested_query_date or (
                    asked.get('groundtruth_dataset_metadata') or {}
                    ).get('default_question_asked_at', '2026-07-28T00:00:00Z')[:10]
                # The implicit build is the long silent part — hand it the
                # front of the bar, or it all happens on 'starting 0%'.
                index = registry.get(
                    cfg.index, progress=scaled_progress(report, 0.7))
                llm = lab_llm(run_settings)
                roles = models.resolve(cfg, run_settings)
                report('retrieving', 0.75, question[:80])
                # Traced rather than plain `retrieve`, for the per-step ranks
                # the Inspector's table needs.
                outcome, trace = pipeline.retrieve_traced(
                    index, cfg.retrieval, question, query_date,
                    llm=llm, models=roles)
                report('answering', 0.9)
                outcome = pipeline.answer(
                    outcome, cfg.generation, llm=llm, models=roles)
                # Exact match only, never fuzzy: everything else stays plainly
                # ungraded rather than guessed at.
                gt_question = next(
                    (q for q in asked['groundtruth_dataset']
                     if q['question'] == question), None)
                if gt_question is not None:
                    quotes = [
                        ev['text']
                        for relevant in gt_question.get(
                            'relevant_corpus_documents') or []
                        for ev in relevant.get('evidence') or []
                        if ev.get('fidelity') == 'verbatim']
                    gold_flags = mark_gold(
                        [c['text'] for c in trace['candidates']], quotes)
                    question_id = gt_question['groundtruth_question_id']
                else:
                    gold_flags = [False] * len(trace['candidates'])
                    question_id = None
                for candidate, gold in zip(trace['candidates'], gold_flags):
                    candidate['gold'] = gold
                    candidate['question_id'] = question_id
                return (outcome.as_dict()
                        | {'models': roles.as_dict(), 'trace': trace,
                           'question_id': question_id})

        return _accepted(jobs.start('query', work,
                                    config=_with_backend(cfg, run_settings)))

    @app.get('/api/questions')
    def questions(limit: int = 200, dataset: str = ''):
        """The ground truth without its answers, for picking a question in the query panel."""
        asked = (datasets.load(dataset)[1] if dataset else ground_truth)
        return {'questions': [
            {'id': q['groundtruth_question_id'],
             'labels': q.get('question_metadata') or {},
             'question': q['question'],
             'behavior': q['expected_answer']['behavior'],
             'evidence_sessions': [
                 str(relevant.get('corpus_document_id', ''))
                 for relevant in q.get('relevant_corpus_documents') or []]}
            for q in asked['groundtruth_dataset'][:limit]]}

    # The job table, as the Inspector reads it: the summary list it follows,
    # and one job's full body.
    return {'jobs': list_jobs, 'job': job_status}
