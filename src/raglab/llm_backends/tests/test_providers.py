"""The local backend (a model on this machine) and the provider modes that
switch the whole lab between it, OpenRouter and the CLI backends."""
from dataclasses import replace

import pytest

from raglab.llm_backends import cli_subprocess_chat as clichat
from raglab.configuration import lab_config as config
from raglab.evaluation import run_evaluation as evaluate
from raglab.llm_backends import model_role_catalogue as models
from raglab.evaluation import ragas_judged_metrics as ragas_eval
from raglab.agents.extra_tools import sweep
from raglab.configuration.lab_config import (
    GenerationConfig,
    LabConfig,
    LabSettings)

from raglab.conftest import LAB_SETTINGS, OLLAMA_SETTINGS

# Every id the local catalogue can offer — used to stand in for a reachable
# Ollama daemon serving everything measured here, so a test can assert on the
# catalogue's *shape* without ever asking a real daemon whether it agrees.
_ALL_OLLAMA_IDS = frozenset(option.id for option in models.OLLAMA_MODELS)


# --- the local backend: a model on this machine ----------------------------
# The four deciding metrics are judged, so an unkeyed lab could measure
# nothing at all. Ollama closes that gap *honestly*: a run judged locally
# must be labelled locally, and an unloadable slug must stop the run rather
# than quietly fall back to another backend.

def test_the_lab_provider_resolves_to_a_real_backend_or_the_fake():
    # this is a unit test
    """Local is the default; another backend is always an explicit choice."""
    assert LabSettings(openrouter_api_key='').provider == 'ollama'
    assert LabSettings(openrouter_api_key='sk-x').provider == 'ollama'
    assert LabSettings(openrouter_api_key='sk-x', llm_provider='openrouter').provider == 'openrouter'
    # A named provider is a commitment, and outranks whether a key happens to
    # exist: with 'ollama' set, a key in the environment must not divert the run
    # to a paid API.
    assert LabSettings(openrouter_api_key='sk-x',
                       llm_provider='ollama').provider == 'ollama'


def test_an_unknown_lab_provider_raises_rather_than_falling_back():
    # this is a unit test
    with pytest.raises(ValueError, match='RAGLAB_LLM'):
        LabSettings(llm_provider='ollamma')


def test_llm_ready_asks_whether_a_real_model_is_reachable_not_whether_a_key_is():
    # this is a unit test
    """The distinction the whole change rests on. The fake provider answers and
    grades every question without ever failing, so 'has a backend' and 'has a
    key' are different questions — and a local judge needs no key at all."""
    assert LabSettings(openrouter_api_key='').llm_ready
    assert LabSettings(openrouter_api_key='sk-x').llm_ready
    assert LabSettings(llm_provider='ollama').llm_ready
    assert not LabSettings(openrouter_api_key='sk-x', llm_provider='fake').llm_ready


def test_the_lab_builds_its_local_model_through_its_own_seam():
    # this is a unit test
    """`LabSettings` itself has to satisfy the factory, not a stand-in.
    `test_llm.py` covers the endpoint and timeout against a stub; this
    covers the real settings object a caller actually holds."""
    from raglab.llm_backends.chat_model_factory import (
    LOCAL_TIMEOUT,
    make_chat_model)
    built = make_chat_model(replace(OLLAMA_SETTINGS,
                                    ollama_base_url='http://localhost:11434/v1'))
    assert str(built.openai_api_base) == 'http://localhost:11434/v1'
    assert built.model_name == 'gemma4:e2b'
    assert built.request_timeout == LOCAL_TIMEOUT


def test_the_ragas_judge_follows_the_provider_instead_of_hardcoding_openrouter():
    # this is a unit test
    """The judge must not be the one stage RAGLAB_LLM cannot move — a local
    answerer with a remote judge is a paid run that looks free."""
    from raglab.llm_backends.chat_model_factory import judge_llm
    judge = judge_llm(replace(OLLAMA_SETTINGS, llm_model='gemma4:e2b'),
                      'qwen3.5:2b')
    assert str(judge.openai_api_base) == OLLAMA_SETTINGS.ollama_base_url
    # The judge slug reaches the wire because RAGAS binds the model at
    # construction and never forwards one per request.
    assert judge.model_name == 'qwen3.5:2b'


def test_ragas_availability_accepts_a_local_judge_with_no_api_key():
    # this is a unit test
    from raglab.evaluation import ragas_judged_metrics as ragas_eval
    status = ragas_eval.availability(OLLAMA_SETTINGS)
    if status.installed:
        assert status.llm_ready, status.notes
    fake = ragas_eval.availability(LAB_SETTINGS)
    if fake.installed:
        assert not fake.llm_ready
        assert any('ollama' in note for note in fake.notes), (
            'the note has to name the way out, not just the missing key')


def test_the_judge_load_table_throttles_by_process_cost_not_a_rate_limit():
    # this is a unit test
    """RAGAS's default concurrency queued requests past the client timeout
    against a laptop model, and a CLI call is a whole process rather than a
    socket — so both throttle harder than a plain HTTP backend. Concurrency
    and timeout cannot change *what* a judge scores, only whether the score
    arrives, so tuning them per backend is not a thumb on the scale. Every
    backend a run can use has a row here — the fallback exists for an
    unknown one, not as the row for a known one — and the answering phase
    reads the same table for the same reason: the cap is a fact about the
    machine, not about which phase is running."""
    local = ragas_eval.judge_load(OLLAMA_SETTINGS)
    remote = ragas_eval.judge_load(replace(LAB_SETTINGS,
                                           openrouter_api_key='sk-x',
                                           llm_provider='openrouter'))
    assert local['max_workers'] < remote['max_workers']
    assert local['timeout'] > remote['timeout']
    assert local['timeout'] >= 600, 'calls under load were measured at 80–92s'
    for provider in ('claude', 'codex'):
        load = ragas_eval.judge_load(config.LabSettings(llm_provider=provider))
        assert load['max_workers'] == 3
        assert load['timeout'] >= 600
    for provider in config.LLM_PROVIDERS:
        if provider:
            assert provider in ragas_eval.JUDGE_LOAD, provider
    cli = replace(LAB_SETTINGS, llm_provider='claude')
    cap = ragas_eval.JUDGE_LOAD['claude']['max_workers']
    assert sweep.capped_workers(6, cli) == cap
    # A lower figure is a deliberate choice about someone's machine; the cap
    # exists to stop an unmeasured default, not to overrule a measured one.
    assert sweep.capped_workers(1, cli) == 1
    # And it changes nothing where a call is a socket rather than a process.
    for provider in ('openrouter', 'ollama', 'fake'):
        settings = replace(LAB_SETTINGS, llm_provider=provider)
        assert sweep.capped_workers(6, settings) == 6, provider


def test_a_run_records_which_backend_judged_it():
    # this is a unit test
    """A decision score is comparable only within one judge, and the model slug
    alone does not say whether it ran locally or was paid for."""
    note = models.note_for(LabConfig(), OLLAMA_SETTINGS)
    assert 'ollama' in note
    assert 'fake' in models.note_for(LabConfig(), LAB_SETTINGS)


def test_a_cli_run_records_the_effort_because_the_effort_moves_the_numbers():
    # this is a unit test
    """`RAGLAB_CLI_EFFORT` changes the scores, so two rows on claude/sonnet
    at `low` and at `max` must not be labelled identically. Only the CLI
    backends: for the other three, effort is a field nothing reads."""
    low = replace(LAB_SETTINGS, llm_provider='claude', llm_model='sonnet',
                  cli_effort='low')
    hard = replace(low, cli_effort='max')
    assert 'effort=low' in models.note_for(LabConfig(), low)
    assert 'effort=max' in models.note_for(LabConfig(), hard)
    assert models.note_for(LabConfig(), low) != models.note_for(LabConfig(), hard)
    for other in (LAB_SETTINGS, OLLAMA_SETTINGS,
                  replace(LAB_SETTINGS, llm_provider='openrouter')):
        assert 'effort' not in models.note_for(LabConfig(), other)


def test_the_dropdown_offers_the_local_models_when_the_backend_is_local(
        monkeypatch):
    # this is a unit test
    """Two lists, not one: an OpenRouter slug is not something Ollama can load,
    and a local tag is not something OpenRouter serves. One merged dropdown would
    offer every user half a menu of choices that cannot work."""
    monkeypatch.setattr(models, 'ollama_ids', lambda settings: _ALL_OLLAMA_IDS)
    local = {e['id'] for e in models.catalogue(OLLAMA_SETTINGS)}
    remote = {e['id'] for e in models.catalogue(LAB_SETTINGS)}
    assert 'qwen3.5:2b' in local and 'gemma4:e2b' in local
    assert 'openai/gpt-5-nano' in remote
    assert 'qwen3.5:2b' not in remote
    # The lists are disjoint, which is the property that matters. Note it cannot
    # be checked by the shape of a slug: `4skl/gemma4-e2b-mtp` is a namespaced
    # Ollama tag and contains a '/' exactly like an OpenRouter one.
    assert not ({o.id for o in models.CHAT_MODELS}
                & {o.id for o in models.OLLAMA_MODELS})
    assert not (local - {''}) & {o.id for o in models.CHAT_MODELS}


def test_every_local_model_says_what_it_is_for():
    # this is a unit test
    """The catalogue rule applies to the local list too: the licence is part
    of the label and every option says why you would pick it."""
    for option in models.OLLAMA_MODELS:
        assert option.source == 'open', option.id
        assert option.note, option.id
        assert option.label


def test_a_model_the_local_backend_does_not_serve_stops_the_run(monkeypatch):
    # this is a unit test
    """The embedder rule applied to chat models: a mismatch is a validation
    error, never a silent fallback to whatever the backend actually served."""
    monkeypatch.setattr(models, 'served_ids',
                        lambda settings: frozenset({'gemma4:e2b'}))
    cfg = LabConfig(generation=GenerationConfig(ragas_model='qwen3.5:2b'))
    problems = models.provider_problems(cfg, OLLAMA_SETTINGS)
    assert problems and 'qwen3.5:2b' in problems[0]
    assert models.provider_problems(
        LabConfig(generation=GenerationConfig(ragas_model='gemma4:e2b')),
        OLLAMA_SETTINGS) == []


def test_an_unreachable_daemon_claims_nothing_rather_than_refusing_everything(
        monkeypatch):
    # this is a unit test
    """"Cannot check" and "not there" are different facts. With the daemon down
    the served list is empty, and a guard that read that as "serves nothing"
    would refuse every run on a machine that was merely idle."""
    monkeypatch.setattr(models, 'served_ids', lambda settings: frozenset())
    cfg = LabConfig(generation=GenerationConfig(ragas_model='anything:1b'))
    assert models.provider_problems(cfg, OLLAMA_SETTINGS) == []


def test_a_remote_slug_is_never_refused_on_the_strength_of_a_listing(monkeypatch):
    # this is a unit test
    """OpenRouter's list is authoritative in one direction only: everything on it
    works, but a slug missing from it may still be valid — the routing suffixes
    (`:free`, `:floor`) do not appear as ids. Blocking runs that used to work is a
    worse failure than the mislabelled row this guard exists to prevent, so the
    refusal is scoped to the local backend, whose tag list *is* authoritative both
    ways."""
    keyed = replace(LAB_SETTINGS, openrouter_api_key='sk-x')
    monkeypatch.setattr(models, 'openrouter_ids',
                        lambda settings: frozenset({'openai/gpt-5-nano'}))
    cfg = LabConfig(generation=GenerationConfig(ragas_model='openai/gpt-5-mini:floor'))
    assert models.provider_problems(cfg, keyed) == []


def test_the_local_tag_list_is_read_from_the_daemon_not_guessed(monkeypatch):
    # this is a unit test
    """Availability is verified, never inferred from the shape of a slug."""
    calls = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'models': [{'name': 'qwen3.5:2b'},
                               {'name': 'gemma4:e2b:latest'}]}

    def fake_get(url, timeout=None):
        calls['url'] = url
        return Response()

    import httpx
    monkeypatch.setattr(httpx, 'get', fake_get)
    monkeypatch.setattr(models, '_LIVE', {})
    ids = models.ollama_ids(OLLAMA_SETTINGS)
    # /api/tags, not the OpenAI-compatible /v1/models: a tag is what `ollama
    # pull` names and what a run should be labelled with.
    assert calls['url'] == 'http://localhost:11434/api/tags'
    assert 'qwen3.5:2b' in ids
    # Both spellings, because `ollama run gemma4:e2b` works even though the
    # daemon prints the ':latest' form.
    assert 'gemma4:e2b' in ids and 'gemma4:e2b:latest' in ids


def test_the_sweep_candidates_use_the_pinned_answer_and_judge_models():
    # this is a unit test
    """The sweep's two model pins are env-settable so a local run needs no edit
    to the file — every candidate must actually use them, or the pin is
    decorative."""
    for cfg in sweep.candidates():
        assert cfg.generation.model == sweep.ANSWER_MODEL, cfg.label
        assert cfg.generation.ragas_model == sweep.JUDGE_MODEL, cfg.label


@pytest.mark.parametrize('provider,pair', list(sweep.PAIRINGS.items()))
def test_every_pairing_keeps_the_judge_apart_from_the_answerer(provider, pair):
    # this is a unit test
    """A model grading its own output is not evidence, and these four metrics
    are the whole basis of the ranking — so no provider's default pairing may
    let the answerer judge itself, and both models must be ones that
    provider's own catalogue actually serves. A slug only means something to
    the backend that serves it, so crossing them is the failure
    `provider_problems` exists to stop, and a default must not be the thing
    that trips it."""
    assert pair['answerer'] != pair['judge'], provider
    served = models.known_models(config.LabSettings(llm_provider=provider))
    slugs = {m.id for m in served}
    assert pair['answerer'] in slugs, (provider, pair)
    assert pair['judge'] in slugs, (provider, pair)


def test_an_explicit_model_is_never_replaced_by_the_provider_default():
    # this is a unit test
    """The resolution is for the *unset* case only. Overwriting a stated model
    would mean a run labelled with one model was scored by another."""
    local = LabSettings(llm_provider='ollama', llm_model='gemma4:e2b')
    assert local.llm_model == 'gemma4:e2b'


def test_every_provider_has_a_default_model_its_own_catalogue_offers():
    # this is a unit test
    """A slug only means something to the backend that serves it, so each
    backend's default has to appear in that backend's own list."""
    for provider, model in config.PROVIDER_MODELS.items():
        served = models.known_models(
            config.LabSettings(llm_provider=provider, llm_model=model))
        assert model in {m.id for m in served}, (provider, model)


def test_a_cli_backend_counts_as_a_real_model_and_names_its_own_default():
    # this is a unit test
    """A CLI reaches a real model, so it may produce leaderboard rows, and
    the default model follows the backend the same way it does everywhere
    else: a remote slug left standing under a CLI is a default that cannot
    run."""
    for provider, expected in (('claude', 'sonnet'), ('codex', 'gpt-5.6-luna')):
        settings = config.LabSettings(llm_provider=provider)
        assert settings.llm_ready is True
        assert settings.llm_model == expected
    # And switching backends does not carry the old backend's default across.
    remote = config.LabSettings(llm_provider='openrouter')
    assert (config.settings_for_provider(remote, 'claude').llm_model
            == 'sonnet')


def test_the_reasoning_effort_is_a_setting_rather_than_an_argv_constant():
    # this is a unit test
    """Effort moves the numbers, so it must be readable off the config
    rather than buried in an argv constant. That it reaches the actual
    argv is `test_the_configured_effort_is_the_one_that_reaches_the_argv`."""
    assert config.LabSettings().cli_effort == 'low'
    assert config.load_lab_settings({'RAGLAB_CLI_EFFORT': 'high'}).cli_effort \
        == 'high'


def test_the_process_memory_ceilings_are_settings_a_sweep_still_fits_under():
    # this is a unit test
    """Both tables the process holds for its own lifetime — built indexes and
    the job table — are bounded by a stated number rather than by the
    machine. The index default has to be wide enough that a sweep over the
    widest single index knob reuses every index it builds, or the cache stops
    being one. Zero means unbounded, and a value nobody can read falls back to
    the default rather than making the lab unstartable over a cache size."""
    settings = config.LabSettings()
    widest = max(len(vocabulary) for vocabulary in
                 (config.EMBEDDERS, config.HIERARCHIES,
                  config.SUMMARIZERS, config.GRAPH_SOURCES))
    assert settings.max_indexes >= widest

    read = config.load_lab_settings({'RAGLAB_MAX_INDEXES': '0',
                                     'RAGLAB_MAX_JOB_HISTORY': '5'})
    assert read.max_indexes == 0 and read.max_job_history == 5
    assert config.load_lab_settings({'RAGLAB_MAX_INDEXES': 'lots'}).max_indexes \
        == settings.max_indexes


def test_the_local_pairing_is_the_one_that_was_screened():
    # this is a convention test
    """The judge is part of the apparatus, so the default judge has to be a
    model `.screens/` has a row for — a default nobody screened is
    judge-shopping with extra steps. Pinned for both CLI backends this lab
    ships a screened pairing for."""
    assert sweep.PAIRINGS['ollama'] == {'answerer': '4skl/gemma4-e2b-mtp',
                                        'judge': 'gemma4:e2b'}
    assert sweep.PAIRINGS['claude'] == {'answerer': 'sonnet', 'judge': 'opus'}


def test_the_sweep_refuses_a_backend_it_has_no_screened_pair_for(monkeypatch):
    # this is a unit test
    """Codex has one verified alias here, so there is no honest
    answerer/judge pair. The order of the two refusals matters: `'' == ''`
    is true, so a self-grading check placed first would report the wrong
    fault when what is missing is any pair at all."""
    monkeypatch.setattr(sweep, '_PROVIDER', 'codex')
    monkeypatch.setattr(sweep, 'ANSWER_MODEL', '')
    monkeypatch.setattr(sweep, 'JUDGE_MODEL', '')
    monkeypatch.setenv('RAGLAB_LLM', 'codex')
    with pytest.raises(SystemExit) as refused:
        sweep.judged_settings()
    said = str(refused.value)
    assert 'no answerer/judge pair' in said
    assert 'pinned pair' in said, 'the refusal has to say what is missing'
    # The equality branch's own wording must not be what came back: with both
    # pins empty it would fire first under the wrong order and name the wrong
    # fault.
    assert 'are both' not in said


def test_the_sweep_refuses_a_judge_that_grades_its_own_answers(monkeypatch,
                                                              tmp_path):
    # this is a unit test
    monkeypatch.setattr(sweep, 'ANSWER_MODEL', 'gemma4:e2b')
    monkeypatch.setattr(sweep, 'JUDGE_MODEL', 'gemma4:e2b')
    monkeypatch.setattr(sweep, 'load_lab_settings', lambda: OLLAMA_SETTINGS)
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    with pytest.raises(SystemExit, match='own output'):
        sweep.judged_settings()


def test_the_sweep_starts_with_a_local_judge_and_no_api_key(monkeypatch):
    # this is a unit test
    """The guard used to test for a credential, so anyone judging locally was
    sent away from a run they could have made."""
    monkeypatch.setattr(sweep, 'load_lab_settings', lambda: OLLAMA_SETTINGS)
    assert sweep.judged_settings().provider == 'ollama'


# --- provider modes: local vs OpenRouter -------------------------------------
# The models column grows a mode dropdown. A mode is a served preset: which
# backend runs the LLM stages and which model each stage defaults to. Served
# rather than kept in a frontend, so the two panels cannot disagree about what
# picking "openrouter" configures.


def test_the_lab_offers_a_backend_for_every_place_a_model_can_run():
    # this is a unit test
    # Local first: it is the lab default, and an option list leads with its
    # default here (see test_every_option_list_leads_with_the_default).
    assert [mode.key for mode in models.MODES] == ['local', 'openrouter',
                                                   'claude', 'codex']
    by_key = {mode.key: mode for mode in models.MODES}
    assert by_key['local'].provider == 'ollama'
    assert by_key['openrouter'].provider == 'openrouter'
    # A CLI mode's key *is* its provider: there is one way to run each of them.
    assert by_key['claude'].provider == 'claude'
    assert by_key['codex'].provider == 'codex'
    # A mode explains itself like every other control on the page.
    assert all(mode.label and mode.note for mode in models.MODES)


@pytest.mark.parametrize('key,expected_model', [
    ('claude', 'sonnet'),
    ('codex', 'gpt-5.6-luna'),
    ('openrouter', 'openai/gpt-5-nano'),
    ('local', None),
])
def test_a_mode_presets_the_full_pipeline_on_its_own_model(
        monkeypatch, key, expected_model):
    # this is a unit test
    """The same preset shape every mode applies: HyDE, the LLM reranker, the
    gate, the answerer and both judges — all pointed at one model. `local` is
    the exception: it resets to the lab's own defaults rather than presetting
    a model, since it *is* the lab default. The index is deliberately
    untouched in every case: the embedder stays local wherever the chat
    models run."""
    if key == 'openrouter':
        monkeypatch.setattr(models, 'openrouter_ids', lambda settings: frozenset())
    patch = models.mode_config(key, LAB_SETTINGS)
    assert 'index' not in patch
    # Every mode presets exactly the same two groups — a field one mode sets
    # and another forgets would leak a remote model into a local run's label.
    # `mode_config` asserts this internally too (model_role_catalogue.py:466-467); this is
    # the belt, so the buckle failing does not silently turn the `local` case
    # below into a no-op loop over an empty dict.
    assert set(patch) == {'retrieval', 'generation'}
    if expected_model is None:
        defaults = LabConfig().to_dict()
        for group, names in patch.items():
            for name, value in names.items():
                assert value == defaults[group][name], f'{group}.{name}'
        return
    ret, gen = patch['retrieval'], patch['generation']
    assert ret['hyde'] is True and ret['expansion_model'] == expected_model
    assert ret['reranker'] == 'llm' and ret['reranker_model'] == expected_model
    assert ret['grader'] == 'llm'
    # For 'openrouter' this is the fallback (nothing on OpenRouter's own list
    # verified a purpose-built reranker); for the CLI modes it is the only
    # option, since a cohere slug means nothing to a CLI.
    assert ret['grader_model'] == expected_model
    assert ret['grade_threshold'] == 0.4     # the measured gate setting
    assert gen['answerer'] == 'llm' and gen['model'] == expected_model
    assert gen['fact_judge'] is True and gen['judge_model'] == expected_model
    assert gen['ragas_model'] == expected_model


def test_an_unknown_mode_raises_rather_than_being_guessed():
    # this is a unit test
    """No auto modes anywhere in this repo."""
    with pytest.raises(ValueError):
        models.mode_config('cloud', LAB_SETTINGS)


def test_a_cli_catalogue_reports_availability_from_the_binary_never_the_alias(
        monkeypatch):
    # this is a unit test
    """There is no /api/tags to ask a CLI, and an alias cannot be checked
    without paying for a call, so availability is the binary: present makes
    every alias offerable (a known one or the user's own unpublished pick —
    nothing here claims to know that alias exists, only that the backend
    does); absent marks every alias NA without dropping any of them, so the
    user's own choice is never hidden."""
    monkeypatch.setattr(clichat.shutil, 'which',
                        lambda name: '/usr/bin/claude' if name == 'claude' else None)
    settings = config.LabSettings(llm_provider='claude')
    entries = {e['id']: e for e in models.catalogue(settings)}
    assert entries['sonnet']['available'] is True
    assert entries['opus']['available'] is True
    gone = config.LabSettings(llm_provider='codex')
    assert all(not e['available'] for e in models.catalogue(gone)
               if e['source'] != 'default')

    unpublished = config.LabSettings(llm_provider='claude',
                                     llm_model='some-unpublished-alias')
    monkeypatch.setattr(clichat.shutil, 'which', lambda name: None)
    offered = [e for e in models.catalogue(unpublished) if e['source'] != 'default']
    assert offered and not any(e['available'] for e in offered)
    monkeypatch.setattr(clichat.shutil, 'which', lambda name: '/usr/bin/claude')
    back = {e['id']: e for e in models.catalogue(unpublished)}
    assert back['some-unpublished-alias']['available'] is True
    # And the HTTP backends keep the old reading: an unanswered daemon checked
    # nothing, so the user's choice stands rather than being shown as NA.
    monkeypatch.setattr(models, 'ollama_ids', lambda settings: frozenset())
    local = config.LabSettings(llm_provider='ollama', llm_model='some-tag')
    assert {e['id']: e for e in models.catalogue(local)}['some-tag']['available']


def test_a_backend_whose_command_is_absent_stops_the_run_naming_it(monkeypatch):
    # this is a unit test
    """Only the *binary* is refused — "cannot check" and "not there" are
    different facts, so an unknown alias is left for the CLI's own error at
    call time."""
    monkeypatch.setattr(clichat.shutil, 'which', lambda name: None)
    settings = config.LabSettings(llm_provider='claude')
    problems = models.provider_problems(LabConfig(), settings)
    assert len(problems) == 1 and 'claude' in problems[0]
    assert 'not installed' in problems[0]

    monkeypatch.setattr(clichat.shutil, 'which', lambda name: '/usr/bin/claude')
    cfg = LabConfig(generation=GenerationConfig(model='some-unpublished-alias'))
    assert models.provider_problems(cfg, settings) == []


def test_the_gate_prefers_a_cohere_reranker_the_account_can_reach(monkeypatch):
    # this is a unit test
    """A slug OpenRouter's own model list does not verify falls back to
    gpt-5-nano rather than gambling a run on it — "cannot verify" must not
    become a refused run."""
    def gate(served):
        monkeypatch.setattr(models, 'openrouter_ids',
                            lambda settings: frozenset(served))
        return models.mode_config('openrouter',
                                  LAB_SETTINGS)['retrieval']['grader_model']

    assert gate({'cohere/rerank-4-fast', 'cohere/rerank-4-pro',
                 'openai/gpt-5-nano'}) == 'cohere/rerank-4-fast'
    assert gate({'cohere/rerank-4-pro',
                 'openai/gpt-5-nano'}) == 'cohere/rerank-4-pro'
    assert gate({'openai/gpt-5-nano'}) == 'openai/gpt-5-nano'
    assert gate(set()) == 'openai/gpt-5-nano'


def test_a_provider_override_rebuilds_the_settings_it_names():
    # this is a unit test
    """The dropdown must move the backend, not just the model labels: a run
    whose models say openrouter while the settings still say ollama would be
    refused over models the user never picked."""
    swapped = config.settings_for_provider(LAB_SETTINGS, 'openrouter')
    assert swapped.provider == 'openrouter'
    # The old backend's default model must not survive the switch — a slug
    # only means something to the backend that serves it...
    assert swapped.llm_model == config.PROVIDER_MODELS['openrouter']
    back = config.settings_for_provider(swapped, 'ollama')
    assert back.llm_model == config.PROVIDER_MODELS['ollama']
    # ...but an explicitly named model (RAGLAB_MODEL) is never replaced.
    named = replace(LAB_SETTINGS, llm_model='someone/custom-7b')
    assert (config.settings_for_provider(named, 'openrouter').llm_model
            == 'someone/custom-7b')
    # '' means "no override": the settings pass through untouched.
    assert config.settings_for_provider(LAB_SETTINGS, '') is LAB_SETTINGS
    with pytest.raises(ValueError):
        config.settings_for_provider(LAB_SETTINGS, 'huggingface')


def test_options_serves_the_provider_modes(client, monkeypatch):
    # this is an integration test
    """The one HTTP-level check of the mode dropdown: both panels read this
    same served list, so `formatConfig` and the leaderboard cannot disagree
    with each other about what a mode configures. Stubbed against a fake but
    fully-served Ollama daemon, since `/api/options` builds every mode's own
    catalogue — including 'local' — regardless of which backend is active."""
    monkeypatch.setattr(models, 'ollama_ids', lambda settings: _ALL_OLLAMA_IDS)
    # And against a fake but reachable OpenRouter, regardless of whether this
    # machine's own environment happens to carry a real API key — a test must
    # not pass or fail depending on that.
    monkeypatch.setattr(models, 'openrouter_ids', lambda settings: frozenset())
    body = client.get('/api/options').json()
    modes = {mode['key']: mode for mode in body['modes']}
    assert set(modes) == {'local', 'openrouter', 'claude', 'codex'}
    assert modes['openrouter']['provider'] == 'openrouter'
    served = modes['openrouter']['config']
    assert served['generation']['model'] == 'openai/gpt-5-nano'
    # The gate default falls back to the answerer model when nothing on
    # OpenRouter's own list verifies a purpose-built reranker — deterministic
    # here since `openrouter_ids` is stubbed empty above.
    assert served['retrieval']['grader_model'] == 'openai/gpt-5-nano'
    # The dropdown explains itself behind the same '!' as every other control.
    assert 'run.mode' in body['help']
    # A slug only means something to the backend that serves it, so the mode
    # that moves the backend must bring that backend's model list with it.
    remote = {option['id'] for option in modes['openrouter']['models']}
    local = {option['id'] for option in modes['local']['models']}
    assert 'openai/gpt-5-nano' in remote
    assert '4skl/gemma4-e2b-mtp' in local
    # Disjoint apart from the '' lab-default entry and a model the user named
    # by RAGLAB_MODEL — an explicitly named model is offered everywhere by the
    # catalogue's own rule. The two known lists must never blur into one
    # dropdown of half-unusable choices.
    named = body['capabilities']['llm_model']
    assert (remote & local) <= {'', named}


def test_both_run_routes_refuse_an_unknown_provider(client):
    # this is an integration test
    """Both run routes apply the same screen — the two disagreeing about which
    configs are legal was a bug once already."""
    for route in ('/api/queries', '/api/evaluations'):
        res = client.post(route, json={'question': 'x',
                                       'provider': 'huggingface'})
        assert res.status_code == 400, route
        assert 'huggingface' in res.json()['detail']


def test_a_mode_only_presets_models_its_own_catalogue_offers(monkeypatch):
    # this is a unit test
    """Every model a mode presets must be offered by the catalogue that
    same mode carries — a dropdown filled from the boot provider's
    catalogue would silently wipe an unofferable preset back to ''. Stubbed
    against a fake but fully-served Ollama daemon, since `mode_catalogue`
    builds the 'local' mode's own catalogue regardless of which backend is
    active."""
    monkeypatch.setattr(models, 'openrouter_ids', lambda settings: frozenset())
    monkeypatch.setattr(models, 'ollama_ids', lambda settings: _ALL_OLLAMA_IDS)
    for entry in models.mode_catalogue(LAB_SETTINGS):
        offered = {option['id'] for option in entry['models']}
        for group, names in entry['config'].items():
            for name, value in names.items():
                if name.endswith('model') and value:
                    assert value in offered, (
                        f"{entry['key']} presets {group}.{name}={value!r} "
                        'but its own catalogue does not offer it')
