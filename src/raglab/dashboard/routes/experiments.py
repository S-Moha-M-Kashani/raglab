"""Evaluations, the experiments they become, and the board built from both.

An evaluation is the one job that scores: it runs the pipeline over a selection
of ground-truth questions, writes a run file and a ledger row, and archives
itself. Reading those two records back is the rest of this module — one
experiment by id, its portable archive, the questions added to it since, and
the board, which puts every experiment that touched one corpus in one table and
names no winner.
"""
from fastapi import HTTPException

from raglab.configuration import explainer_assembly as explain
from raglab.configuration.lab_config import (
    LLM_PROVIDERS,
    GenerationConfig,
    IndexConfig,
    LabConfig,
    RetrievalConfig,
    settings_for_provider)
from raglab.corpora import dataset_import_contract as datasets
from raglab.dashboard.service_presentation import gold_available
from raglab.dashboard.service_route_plumbing import (
    _accepted,
    _find_question,
    _with_backend,
    cancel_checker,
    scaled_progress,
    screen)
from raglab.evaluation import deterministic_metrics as metrics
from raglab.evaluation import experiment_archive as archive
from raglab.evaluation import experiment_archive_store as archive_store
from raglab.evaluation import leaderboard
from raglab.evaluation import run_evaluation as evaluate
from raglab.evaluation import service_experiment_ledger as ledger
from raglab.llm_backends import model_role_catalogue as models
from raglab.llm_backends.chat_model_factory import lab_llm
from raglab.rag_components import question_to_answer_pipeline as pipeline

def _recorded_config_problem(recorded: object) -> str | None:
    """The strict counterpart to ``LabConfig.from_dict`` for old records.

    The normal parser is intentionally forgiving of a stale *browser* payload.
    A recorded experiment is different evidence: dropping a retired field or
    coercing a malformed value here would run a question under settings the
    record never named.  ``provider`` is execution metadata Jobs adds beside a
    LabConfig, not a knob, and is the one accepted top-level companion field.
    """
    if not isinstance(recorded, dict) or not recorded:
        return 'recorded experiment has no config'
    lab_fields = set(LabConfig.__dataclass_fields__)
    unknown = sorted(set(recorded) - lab_fields - {'provider'})
    if unknown:
        return f'{unknown[0]} is not a knob this lab reads any more'
    if 'label' in recorded and not isinstance(recorded['label'], str):
        return 'recorded label has malformed type'

    for name, kind in (('index', IndexConfig),
                       ('retrieval', RetrievalConfig),
                       ('generation', GenerationConfig)):
        if name not in recorded:
            return f'recorded {name} is missing'
        knobs = recorded[name]
        if not isinstance(knobs, dict):
            return f'recorded {name} has malformed shape'
        retired = sorted(set(knobs) - set(kind.__dataclass_fields__))
        if retired:
            return f'{retired[0]} is not a knob this lab reads any more'
        missing = sorted(set(kind.__dataclass_fields__) - set(knobs))
        if missing:
            return f'recorded {name}.{missing[0]} is missing'
        for knob, value in knobs.items():
            expected = kind.__dataclass_fields__[knob].type
            if expected is str and not isinstance(value, str):
                return f'recorded {name}.{knob} has malformed type'
            if expected is bool and type(value) is not bool:
                return f'recorded {name}.{knob} has malformed type'
            if expected is int and type(value) is not int:
                return f'recorded {name}.{knob} has malformed type'
            if expected is float and (type(value) not in (int, float)):
                return f'recorded {name}.{knob} has malformed type'
            if knob == 'agentic_weights' and (
                    not isinstance(value, (list, tuple))
                    or len(value) != 3
                    or any(type(weight) not in (int, float)
                           for weight in value)):
                return f'recorded {name}.{knob} has malformed type'
    if 'provider' not in recorded:
        return 'recorded provider is missing'
    if not isinstance(recorded['provider'], str):
        return 'recorded provider has malformed type'
    provider = recorded['provider']
    if provider and provider not in LLM_PROVIDERS:
        return f'unknown recorded provider: {provider!r}'
    return None


def _recorded_config(detail: object, row: dict | None = None) -> object:
    """The config a job recorded, with only its separately-recorded backend.

    Run results predate the backend field and keep their `config` in the
    portable archive spelling, where provider is deliberately execution
    metadata rather than a knob.  The ledger records that resolved provider in
    its own column from the same job.  Joining those two stored facts restores
    the complete job config; it never consults the lab's current provider and
    never mutates the detail object being read.
    """
    config = detail.get('config') if isinstance(detail, dict) else None
    provider = (row or {}).get('provider')
    if (isinstance(config, dict) and 'provider' not in config
            and isinstance(provider, str) and provider):
        return config | {'provider': provider}
    return config


def _archive_ui(payload: dict) -> dict:
    """The panel controls of one run request, in the shape an archive records
    them (`settings.ui`).

    The knobs are a `LabConfig` and travel as one; these five are not — they
    are how much of the corpus was scored and by which backend, which the
    config cannot say. Read back off the request that carried them rather than
    off the result, because the result never held them.

    `mode` is the panel's dropdown, and the request carries the *provider* that
    dropdown resolved to, so it is resolved back through the same table
    (`models.MODES`) rather than guessed at. A provider no mode offers leaves
    it blank, which is what a run started outside the panel honestly is.
    """
    provider = payload.get('provider') or ''
    return {
        'mode': next((mode.key for mode in models.MODES
                      if mode.provider == provider), ''),
        'ragas_mode': payload.get('ragas_mode', 'offline'),
        'limit': int(payload.get('limit') or 0),
        'ragas_limit': int(payload.get('ragas_limit') or 0),
        # D7: a question filter is now one switch-group per question label
        # the dataset itself declares, so there is no fixed vocabulary left
        # to validate `labels`/`balance` against here — the panel checks them
        # against the dataset the archived config names.
        'labels': dict(payload.get('labels') or {}),
        'balance': payload.get('balance') or '',
    }


def register(app, context) -> None:
    settings_now, questions_for = context.settings_now, context.questions_for
    registry, dataset_lock, jobs = (
        context.registry, context.dataset_lock, context.jobs)

    @app.post('/api/evaluations')
    def start_evaluation(payload: dict):
        cfg = LabConfig.from_dict(payload)
        # The mode dropdown's backend override, applied before the screen so
        # the settings that refuse a model are the settings that would run it.
        run_settings = settings_for_provider(settings_now(),
                                             payload.get('provider') or '')
        screen(cfg, run_settings)
        run_execution = {
            'provider': run_settings.provider,
            'models': models.resolve(cfg, run_settings).as_dict(),
        }
        metric_catalogue = explain.measures()
        archive_ui = _archive_ui(payload)

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled)
            check_cancelled()
            with dataset_lock(cfg.index.dataset), registry.hold(cfg.index):
                # Snapshot and index use share this boundary. A replacement of
                # the same id cannot put new corpus evidence beside old chunks.
                check_cancelled()
                run_corpus, run_truth = datasets.load(cfg.index.dataset)
                result = evaluate.run_eval(
                    registry, run_truth, cfg, run_settings,
                    labels=payload.get('labels') or None,
                    limit=payload.get('limit') or None,
                    balance=payload.get('balance') or '',
                    ragas_mode=payload.get('ragas_mode', 'offline'),
                    ragas_limit=payload.get('ragas_limit') or None,
                    workers=int(payload.get('workers', 1)), progress=report,
                    # Always traced, so the Inspector is never blank after a run —
                    # a recording of the same retrieval, so no score can move.
                    trace=True, cancelled=check_cancelled)
                # Added here, not inside `as_dict`: the run file `save_run`
                # writes stays the summary, with no trace or chunk text in it.
                return result.as_dict() | {
                    'traces': result.traces,
                    'chunks_by_session': result.chunks_by_session,
                    'summaries': result.summaries,
                    'archive_evidence': {
                        'execution': run_execution,
                        'metric_catalogue': metric_catalogue,
                        'inspector': {
                            'dataset': {
                                'id': cfg.index.dataset or datasets.BUILTIN,
                                'corpus': run_corpus,
                                'ground_truth': run_truth,
                            },
                            'chunks_by_session': result.chunks_by_session,
                            'summaries': result.summaries,
                            'traces': result.traces,
                        },
                    }}

        def keep_archive(job: dict) -> None:
            """One finished evaluation, written down as the archive a reader
            would have downloaded.

            The evidence is already assembled for the browser; all this adds is
            the knob surface it was produced under — the config off the result
            itself, so the two can never disagree, and the panel controls this
            request carried. Handed to the store as a whole object rather than
            column by column: `build_completed` is the export codec's own
            server-side twin, so an experiment on record and an experiment
            exported are the same thing said twice.

            Reached only from `Jobs.run`, and only for a job that finished:
            unfinished work is not saved, and a failure here is reported on the
            job rather than failing it.
            """
            result = job.get('result')
            evidence = result.get('archive_evidence') if isinstance(result, dict) else None
            if not isinstance(evidence, dict):
                return
            canonical = {key: value for key, value in result.items()
                         if key != 'archive_evidence'}
            archive_store.store_completed(
                {'config': canonical['config'], 'ui': archive_ui}, canonical,
                evidence)

        return _accepted(jobs.start('run', work,
                                    config=_with_backend(cfg, run_settings),
                                    archive=keep_archive))

    @app.get('/api/evaluations')
    def evaluations(limit: int = 50):
        # `total` beside the rows, since this listing is bounded.
        return {'runs': evaluate.list_runs(limit),
                'total': evaluate.count_runs()}

    @app.get('/api/leaderboard')
    def leaderboard_board(dataset: str = '', limit: int = 500):
        """One board per dataset: every experiment that touched one corpus.

        `dataset=''` is the built-in default, `dataset='*'` is every experiment
        — the table that used to sit on the lab page, which is the same
        population with no filter and so an option in the same picker rather
        than a second surface.

        The grouping and the row shape come from `evaluation.leaderboard`, the
        same module `raglab-leaderboard` prints from, so the page and the
        command line cannot describe the same records differently. This route is
        why that module lives in `evaluation/` rather than among the terminal
        tools no route reaches."""
        wanted = dataset or '*'
        running = jobs.get(jobs.current) if jobs.current else None
        is_running = bool(running and running['state'] in ('running', 'cancelling'))
        if is_running:
            # A live job has no durable row yet. It still belongs on the board,
            # newest first, so leaving the Laboratory does not make the reader
            # lose the experiment currently in progress.
            rows = leaderboard.board_rows(limit)
            rows.append(leaderboard.live_job_record(running))
            rows.sort(key=lambda row: row.get('started_at', ''), reverse=True)
            if wanted != '*':
                rows = [row for row in rows if row.get('dataset') == wanted]
            ordering = 'newest'
        else:
            boards = leaderboard.build_board(limit)
            # `every_row`, not the boards concatenated: the page's own prose says
            # the order it was served in is the ranking, and a concatenation is
            # ordered by dataset block instead.
            rows = (leaderboard.every_row(boards) if wanted == '*' else
                    next((b.rows for b in boards if b.dataset == wanted), []))
            ordering = 'score'
        return {'dataset': wanted,
                'datasets': [found.as_dict() for found in datasets.catalogue()],
                'rows': rows, 'ordering': ordering, 'running': is_running}

    @app.get('/api/experiments')
    def experiments(limit: int = 200):
        """Everything this lab has ever finished, newest first — beside the leaderboard, never in it.

        An index build has no questions and no score, so a numbered table it
        appeared in would make a rank claim about work that measured nothing."""
        return {'experiments': ledger.experiments(limit)}

    @app.get('/api/experiments/{experiment_id}')
    def experiment_detail(experiment_id: str):
        """One experiment by id, from whichever of the two records holds it.

        The ledger first, then the run file, because the board is built from
        both and a row it lists must be a row this answers. An evaluation older
        than the ledger has only its run file, and those are most of the scored
        rows there are."""
        found = leaderboard.experiment(experiment_id)
        if found is None:
            raise HTTPException(404, 'unknown experiment')
        row = ledger.experiment(experiment_id)
        run = evaluate.load_run(experiment_id)
        # `experiment_record` is the projection the board's own rows are, so the
        # row a reader clicked and the page that opens it cannot give two
        # accounts of one experiment — and any later reader that wants the same
        # experiment without the evidence asks the same function rather than
        # writing the ledger-versus-run precedence a third time. What a page
        # needs on top *is* the evidence: the ledger's own `detail`, which
        # carries the whole job result for a row it recorded, and the run file
        # itself for an evaluation older than the ledger.
        return found | {'detail': (row or {}).get('detail') or run or {}}

    @app.get('/api/experiments/{experiment_id}/archive')
    def experiment_archive_route(experiment_id: str):
        """One experiment as the portable archive the export button writes.

        Its own route rather than a shape change to `/api/experiments/{id}`,
        because that one has a second reader: the Inspector's recorded mode
        reads `detail`, `state`, `kind` and `error` off it, and would go blank
        if it started receiving an archive instead.

        This is what the board's open button hands to the panel, and it is the
        *same object* a downloaded file carries — so opening a row and importing
        its export are one path with one strictness, not two that can drift.

        The corpus is spliced back in on the way out (`archive_store.serve`)
        from the content-addressed corpus store, and a version that store does
        not hold is a refusal rather than a substitution: 409, because the
        archive is intact and it is this installation that has moved.
        """
        # A question is a complete, durable record but deliberately not a
        # completed evaluation: it has one answer and trace, no run file or
        # corpus snapshot to turn into an export archive.  The open handoff
        # therefore carries only the settings it actually recorded.  It does
        # not touch the parent, manufacture evaluation evidence, or splice in
        # today's corpus; a later Inspector visit reads the question row's own
        # ledger evidence through its ordinary recorded-mode route.
        question = ledger.experiment(experiment_id)
        if (question or {}).get('kind') == 'question' and question.get('state') == 'done':
            detail = question.get('detail')
            recorded = _recorded_config(detail, question)
            problem = _recorded_config_problem(recorded)
            if problem:
                raise HTTPException(409, problem)
            handoff = {
                'format': archive.FORMAT,
                'version': archive.VERSION,
                'settings': {
                    'config': {name: recorded[name]
                               for name in LabConfig.__dataclass_fields__},
                    'ui': _archive_ui(detail | {
                        'provider': recorded['provider']}),
                },
            }
            try:
                return archive.validate_archive(handoff)
            except archive.ArchiveError as error:
                raise HTTPException(409, str(error)) from error

        db = archive_store.connect(ledger.db_path())
        try:
            found = archive_store.serve(db, experiment_id)
        except archive_store.ArchiveStoreError as error:
            raise HTTPException(409, str(error))
        finally:
            db.close()
        if found is None:
            raise HTTPException(
                404, f'{experiment_id} has no complete archive: only '
                'experiments whose evidence survives in full are archived')
        return found

    @app.post('/api/experiments/{experiment_id}/questions')
    def add_recorded_question(experiment_id: str, payload: dict):
        """Run one ground-truth question under an experiment's recorded config.

        The parent record is evidence of work already done, so this route only
        ever starts a distinct `question` job whose result names what it
        annotates.  It never accepts a config from the browser: accepting one
        would make the Inspector claim an experiment supplied settings it did
        not actually record.
        """
        if leaderboard.experiment(experiment_id) is None:
            raise HTTPException(404, f'unknown experiment: {experiment_id}')
        row = ledger.experiment(experiment_id)
        run = evaluate.load_run(experiment_id)
        detail = (row or {}).get('detail') or run or {}
        if not isinstance(detail, dict):
            raise HTTPException(409, 'recorded experiment detail is malformed')
        if detail.get('format') == 'raglab-experiment':
            raise HTTPException(409, 'recorded archive has no runnable job config')
        recorded = _recorded_config(detail, row)
        problem = _recorded_config_problem(recorded)
        if problem:
            raise HTTPException(409, problem)

        cfg = LabConfig.from_dict(recorded)
        asked = questions_for(cfg)
        question_id = payload.get('question_id') if isinstance(payload, dict) else None
        question = _find_question(asked, question_id)
        if question is None:
            raise HTTPException(404, f'unknown question id: {question_id!r}')
        run_settings = settings_for_provider(settings_now(),
                                             recorded.get('provider') or '')
        screen(cfg, run_settings)
        job_config = _with_backend(cfg, run_settings)
        selected_id = str(question_id)

        def work(report, cancelled):
            check_cancelled = cancel_checker(cancelled)
            check_cancelled()
            with dataset_lock(cfg.index.dataset), registry.hold(cfg.index):
                check_cancelled()
                index = registry.get(cfg.index,
                                     progress=scaled_progress(report, 0.6))
                llm = lab_llm(run_settings)
                roles = models.resolve(cfg, run_settings)
                query_date = (asked.get('groundtruth_dataset_metadata') or {}
                              ).get('default_question_asked_at',
                                    '2026-07-28T00:00:00Z')[:10]
                report('retrieving', 0.65, question['question'][:80])
                outcome, trace = pipeline.retrieve_traced(
                    index, cfg.retrieval, question['question'], query_date,
                    llm=llm, models=roles)
                trace_row = evaluate.trace_row(
                    question, trace,
                    gold_present=gold_available(
                        index, metrics.verbatim_quotes(question)))
                check_cancelled()
                report('answering', 0.85)
                outcome = pipeline.answer(outcome, cfg.generation, llm=llm,
                                          models=roles)
                answer_row = evaluate.json_safe(
                    metrics.score_question(question, outcome, cfg.retrieval.k))
                return {
                    'label': f'adds {selected_id} to {experiment_id}',
                    'annotates': experiment_id,
                    'question_id': selected_id,
                    'config': job_config,
                    'models': roles.as_dict(),
                    'selection': {'n': 1, 'question_ids': [selected_id]},
                    'traces': [trace_row],
                    'rows': [answer_row],
                }

        return _accepted(jobs.start('question', work, config=job_config))

    @app.get('/api/experiments/{experiment_id}/questions')
    def recorded_questions(experiment_id: str):
        if leaderboard.experiment(experiment_id) is None:
            raise HTTPException(404, f'unknown experiment: {experiment_id}')
        return {'questions': ledger.annotations(experiment_id)}

    @app.get('/api/evaluations/{run_id}')
    def evaluation_detail(run_id: str):
        data = evaluate.load_run(run_id)
        if data is None:
            raise HTTPException(404, 'unknown run')
        return data
