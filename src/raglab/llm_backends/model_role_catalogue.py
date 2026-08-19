"""Which language model runs which stage; each stage carries its own choice rather than sharing one.

Every option says where its weights stand (`open`/`closed`). The remote
catalogue offers only models verified reachable on this account; the local
(Ollama) list keeps NA meaning "not pulled yet", since a daemon can be asked directly.
"""
from dataclasses import dataclass, fields

from raglab.llm_backends import cli_subprocess_chat as clichat
from raglab.configuration.lab_config import (
    LabConfig,
    LabSettings,
    settings_for_provider)

SOURCES = ('default', 'open', 'closed', 'unknown')


@dataclass(frozen=True)
class ModelOption:
    id: str                  # the OpenRouter slug that goes on the wire
    label: str               # what the dropdown shows
    source: str              # open | closed | unknown
    note: str = ''           # why you would pick it
    verified: bool = False   # this lab has actually produced numbers with it

    def as_dict(self, live: frozenset) -> dict:
        return {'id': self.id, 'label': self.label, 'source': self.source,
                'note': self.note, 'available': self.verified or self.id in live}


# Deliberately short. A dropdown of sixty slugs is not a choice either, and every
# entry here is one somebody might reasonably want on Farsi diary text.
CHAT_MODELS = (
    ModelOption('openai/gpt-5-nano', 'GPT-5 nano', 'closed', verified=True,
                note='every grade in .runs/ so far was measured on this'),
    ModelOption('openai/gpt-5-mini', 'GPT-5 mini', 'closed',
                note='the same family with more capacity — the obvious A/B'),
    ModelOption('anthropic/claude-haiku-4.5', 'Claude Haiku 4.5', 'closed',
                note='fast, and strong at "answer only from this context"'),
    ModelOption('google/gemini-2.5-flash', 'Gemini 2.5 Flash', 'closed',
                note='cheap long context, useful for the summary pass'),
)

# A separate list because an OpenRouter slug is not a thing Ollama can load and
# a local tag is not a thing OpenRouter serves; the catalogue serves whichever
# list matches the active provider.
OLLAMA_MODELS = (
    ModelOption('gemma4:e2b', 'Gemma 4 E2B', 'open',
                note='read the Farsi correctly and scored 4/5 on the judge '
                     'screen; ~17s per judged call, because it reasons before '
                     'every verdict'),
    ModelOption('qwen3.5:2b', 'Qwen3.5 2B', 'open',
                note='the speed candidate for the judge — Qwen is the stronger '
                     'multilingual family at this size, but the two nearest-size '
                     'models screened so far were both constant predictors, so '
                     'screen it before trusting a row it produced'),
    ModelOption('4skl/gemma4-e2b-mtp', 'Gemma 4 E2B (MTP)', 'open',
                note='multi-token-prediction build of the above, and 3.7 GB at '
                     'Q4_0 against 7.2 GB at Q4_K_M — so it is a throughput '
                     'test, and the quantisation differs too, which means a '
                     'quality difference from the base build is not ruled out'),
    ModelOption('deepseek-r1:8b', 'DeepSeek-R1 8B', 'open',
                note='the strongest reasoner installed at a size that still '
                     'answers, but reasoning is the cost here: the judge is the '
                     'high-volume stage, and a model that spends hundreds of '
                     'thinking tokens per verdict pays that on every call'),
    ModelOption('dolphin-mixtral:8x7b', 'Dolphin Mixtral 8x7B', 'open',
                note='the largest thing installed; a 47B mixture is slow enough '
                     'that it is an answerer, not a 276-call judge'),
    ModelOption('llama3.1:8b', 'Llama 3.1 8B', 'open',
                note='read the Farsi but was a constant predictor as a judge — '
                     'usable to answer, never to grade'),
)

# Reached by running the CLI already installed and logged in on this machine
# (cli_subprocess_chat.py) — a separate list, since an OpenRouter slug is not a thing
# `claude --model` accepts. None of these is `verified`: that flag would make an
# option available unconditionally, but here availability can be checked (the
# binary exists or it does not), so an unverified option showing as usable would
# disagree with `provider_problems` refusing the run.
CLAUDE_MODELS = (
    ModelOption('sonnet', 'Claude Sonnet (CLI)', 'closed',
                note='what this backend was measured on, all at effort=low: '
                     '3.9s per call on a short grade probe, 5.6s on the lab\'s '
                     'real grade prompt, and ~7.4s per call-slot across a '
                     'judged run — a longer prompt costs more, so the probe '
                     'figure is the floor rather than the price'),
    ModelOption('opus', 'Claude Opus (CLI)', 'closed',
                note='the sweep\'s judge under this backend — a model grading '
                     'its own output is not evidence, so the answerer stays '
                     'sonnet'),
    ModelOption('haiku', 'Claude Haiku (CLI)', 'closed',
                note='the speed candidate: the judge is ~420 calls per run, '
                     'and each one here is a process spawn as well as a turn'),
)

# The codex CLI publishes no model list to verify against. Luna is the
# cost-sensitive default for high-volume judging; Terra remains the
# previously-used quality alternative. RAGLAB_MODEL names another, and
# `catalogue` offers whatever it names.
CODEX_MODELS = (
    ModelOption('gpt-5.6-luna', 'GPT-5.6 Luna (Codex CLI)', 'closed',
                note='the cost-sensitive default for high-volume judging, run '
                     'at low reasoning effort'),
    ModelOption('gpt-5.6-terra', 'GPT-5.6 Terra (Codex CLI)', 'closed',
                note='8.2s per call at effort=low, and every call carries '
                     'codex\'s own ~18.5k-token agent preamble, which no flag '
                     'removes'),
)


@dataclass(frozen=True)
class ModelRole:
    key: str            # 'answer'
    label: str          # 'Answer'
    field: str          # 'generation.model'
    help: str
    only_when: str      # when this role is actually consulted

    @property
    def step(self) -> str:
        """Which pipeline step this model serves, derived from `field` so the two cannot disagree."""
        return self.field.partition('.')[0]

    def as_dict(self) -> dict:
        return {'key': self.key, 'label': self.label, 'field': self.field,
                'help': self.help, 'only_when': self.only_when,
                'step': self.step}


ROLES = (
    ModelRole('expand', 'Query rewriting (HyDE)', 'retrieval.expansion_model',
              'Invents a plausible diary answer and searches with that instead '
              'of the question, on the theory that an answer looks more like the '
              'text you are hunting for than a question does. Only HyDE uses a '
              'model; multi-query expansion is rule-based and free.',
              'HyDE is on'),
    ModelRole('rerank', 'Reranker', 'retrieval.reranker_model',
              'Reads the top candidates and scores each one against the '
              'question. It sits in the latency path of every query, so a slow '
              'model here is felt on every single question.',
              'Reranker = llm'),
    ModelRole('grade', 'Relevance gate', 'retrieval.grader_model',
              'Decides whether a chunk is relevant at all. This is what lets '
              'the lab abstain rather than answer from noise: measured here, an '
              'LLM gate at 0.4 refused all five unanswerable questions while '
              'wrongly refusing 3% of the answerable ones — the lexical gate had '
              'no threshold that could do both.',
              'Gate = llm'),
    ModelRole('answer', 'Answer', 'generation.model',
              'Writes the Farsi answer from the retrieved context, cites session '
              'ids, and is the stage that must refuse when the diary is silent. '
              'Generation is the current bottleneck — faithfulness 0.743 against '
              'key-fact coverage 0.261 — so this is the interesting dropdown.',
              'Answerer = llm'),
    ModelRole('judge', 'Key-facts judge', 'generation.judge_model',
              'Checks a Farsi answer against the ground truth\'s English key '
              'facts. It is translating as well as judging, which is why no '
              'deterministic metric can replace it — and why a weak model here '
              'produces confidently wrong scores.',
              'the key-facts judge is on'),
    ModelRole('ragas', 'RAGAS judge', 'generation.ragas_model',
              'The model RAGAS uses for faithfulness, answer relevancy and '
              'factual correctness. Separate from the answerer on purpose: a '
              'model grading its own output is not evidence.',
              'RAGAS = judged'),
    ModelRole('plan', 'Agent planner', 'agent.plan_model',
              'Runs the retrieval loop\'s three thinking steps: what evidence '
              'would answer this, is what we found sufficient, and what should '
              'the next query be. It is asked once per hop, so a slow model here '
              'multiplies by max_hops on every question — the same arithmetic '
              'that makes the reranker\'s choice matter.',
              'the agent scope owns retrieval'),
    ModelRole('critic', 'Agent critic', 'agent.critic_model',
              'Reads a draft answer against the contexts it was written from and '
              'says whether every claim is supported — and, under "both", '
              'whether the draft answers the question at all. This is the stage '
              'that can catch what faithfulness would later punish, so a weak '
              'model here approves exactly the answers a strong judge will mark '
              'down.',
              'the agent scope owns generation and the critic is on'),
)

ROLE_HELP = {f'model.{role.key}': role.help for role in ROLES}


@dataclass(frozen=True)
class Roles:
    """The model each stage will actually ask for. '' means "whatever the
    provider defaults to", which is how RAGLAB_MODEL keeps working."""
    expand: str = ''
    rerank: str = ''
    grade: str = ''
    answer: str = ''
    judge: str = ''
    ragas: str = ''
    plan: str = ''
    critic: str = ''

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def chosen(cfg: LabConfig, role: ModelRole) -> str:
    group, _, name = role.field.partition('.')
    return getattr(getattr(cfg, group), name) or ''


def resolve(cfg: LabConfig, settings: LabSettings) -> Roles:
    """Config → concrete model per stage, filling blanks with the lab default."""
    picked = {role.key: chosen(cfg, role) or settings.llm_model for role in ROLES}
    return Roles(**picked)


_LIVE: dict[str, frozenset] = {}


def forget_live() -> None:
    """Drop the cached per-url availability, so `credentials.set_key` re-asks instead of leaving stale NAs."""
    _LIVE.clear()


def openrouter_ids(settings: LabSettings) -> frozenset:
    """Model ids OpenRouter is currently serving, or an empty set.

    Best-effort: a failure here means "cannot verify" (shows as NA), never an
    exception — the lab must still run with no network at all."""
    if not settings.openrouter_api_key:
        return frozenset()
    if settings.openrouter_base_url in _LIVE:
        return _LIVE[settings.openrouter_base_url]
    ids: frozenset = frozenset()
    try:
        import httpx
        res = httpx.get(f'{settings.openrouter_base_url.rstrip("/")}/models',
                        timeout=6.0)
        res.raise_for_status()
        ids = frozenset(m['id'] for m in res.json().get('data', []) if m.get('id'))
    except Exception:
        ids = frozenset()
    _LIVE[settings.openrouter_base_url] = ids
    return ids


def ollama_ids(settings: LabSettings) -> frozenset:
    """Model tags Ollama is serving on this machine, or an empty set.

    Authoritative in both directions, unlike OpenRouter's list: a tag absent
    here genuinely cannot be loaded, which lets `provider_problems` refuse a run
    rather than merely mark it NA. `/api/tags`, not `/v1/models`, since a tag is
    what `ollama pull` names."""
    if settings.provider != 'ollama':
        return frozenset()
    root = settings.ollama_base_url.rstrip('/').removesuffix('/v1')
    if root in _LIVE:
        return _LIVE[root]
    ids: frozenset = frozenset()
    try:
        import httpx
        res = httpx.get(f'{root}/api/tags', timeout=6.0)
        res.raise_for_status()
        ids = frozenset(m['name'] for m in res.json().get('models', [])
                        if m.get('name'))
        # Ollama reports 'qwen3.5:2b'; `ollama run qwen3.5` works too, so both
        # spellings are offered rather than only the one the daemon happens to
        # print. A run's notes still record whatever the user picked.
        ids |= frozenset(name.removesuffix(':latest') for name in ids)
    except Exception:
        ids = frozenset()
    _LIVE[root] = ids
    return ids


def cli_ids(settings: LabSettings) -> frozenset:
    """Aliases the configured CLI backend can be asked for, or an empty set.

    The checkable fact for a CLI is the *binary*, since there is no `/api/tags`
    and an alias cannot be verified without paying for a call. Present, its
    catalogue plus whatever RAGLAB_MODEL named is offerable; absent, nothing is."""
    if settings.provider not in clichat.CLIS:
        return frozenset()
    if not clichat.cli_available(settings.provider):
        return frozenset()
    return frozenset({option.id for option in known_models(settings)}
                     | {settings.llm_model})


def served_ids(settings: LabSettings) -> frozenset:
    """Whatever the active backend is serving right now."""
    if settings.provider == 'ollama':
        return ollama_ids(settings)
    if settings.provider in clichat.CLIS:
        return cli_ids(settings)
    return openrouter_ids(settings)


# The candidate list per backend, because a slug only means something to the
# backend that serves it. Four lists now, and a dropdown that mixed them would
# offer most users a menu that mostly cannot work.
_CATALOGUES = {'ollama': OLLAMA_MODELS, 'claude': CLAUDE_MODELS,
               'codex': CODEX_MODELS}


def known_models(settings: LabSettings) -> tuple[ModelOption, ...]:
    return _CATALOGUES.get(settings.provider, CHAT_MODELS)


def provider_problems(cfg: LabConfig, settings: LabSettings) -> list[str]:
    """Model choices the active backend cannot serve — a validation error, never a silent fallback.

    This guard only ever refuses what it has verified absent. For `ollama` that
    is the model, via `/api/tags` (an unreachable daemon claims nothing, since
    "cannot check" and "not there" are different facts). For a CLI there is no
    such list, so the verifiable fact is the binary; an unknown alias is left to
    fail at call time. `openrouter`'s list is authoritative in one direction
    only (routing suffixes like `:free` don't appear as ids), so it is refused
    nothing here."""
    if settings.provider in clichat.CLIS:
        # The one check that can be made: an alias is not refused here, since
        # nothing verified it absent.
        if clichat.cli_available(settings.provider):
            return []
        return [f'{settings.provider} is the chosen backend but its command is '
                f'not installed on this machine — install it, or pick a backend '
                f'that is here']
    if settings.provider != 'ollama':
        return []
    served = served_ids(settings)
    if not served:
        return []
    picked = {role.key: chosen(cfg, role) for role in ROLES}
    picked['(lab default)'] = settings.llm_model
    return [f'{settings.provider} does not serve {model!r}, asked for by '
            f'{role} — pull it, or pick one it does serve'
            for role, model in sorted(picked.items())
            if model and model not in served]


def catalogue(settings: LabSettings) -> list[dict]:
    """The dropdown contents: the lab default first, then every candidate with its licence and availability."""
    live = served_ids(settings)
    known = list(known_models(settings))
    if settings.llm_model not in {option.id for option in known}:
        # An id set by RAGLAB_MODEL is offered even though nothing is known
        # about its weights. With a served list, membership decides; with none
        # on an HTTP backend the daemon simply didn't answer, so the user's
        # choice is taken at face value; with none on a CLI backend the
        # emptiness is a checked fact (the command is not there), so this alias
        # is NA too — never offering a model the lab actually refuses to run.
        verified = not live and settings.provider not in clichat.CLIS
        known.insert(0, ModelOption(settings.llm_model, settings.llm_model,
                                    'unknown', verified=verified,
                                    note='named by RAGLAB_MODEL'))
    entries = [option.as_dict(live) for option in known]
    # Usable models first; NA sinks to the bottom without leaving the list.
    entries.sort(key=lambda entry: not entry['available'])
    return [{'id': '', 'label': f'lab default ({settings.llm_model})',
             'source': 'default', 'available': True,
             'note': 'follows RAGLAB_MODEL, so one change moves every stage'}
            ] + entries


@dataclass(frozen=True)
class ProviderMode:
    """One entry in the panel's mode dropdown: which backend runs the LLM
    stages, and therefore which per-stage preset picking it applies."""
    key: str        # 'local'
    label: str      # what the dropdown shows
    provider: str   # the LabSettings.llm_provider this mode runs on
    note: str       # why you would pick it


# Local first: it is the lab default — a default must never silently spend API
# credit — so this list leads with it like every other option list here.
MODES = (
    ProviderMode('local', 'Local (Ollama)', 'ollama',
                 note='every LLM stage on a model on this machine — free and '
                      'private — with the lab\'s own stage defaults: '
                      'extractive answerer, lexical reranker, no gate'),
    ProviderMode('openrouter', 'OpenRouter', 'openrouter',
                 note='the full LLM pipeline on gpt-5-nano — HyDE, LLM '
                      'reranker, relevance gate, answerer and both judges — '
                      'while the embedder stays the local Persian-tuned '
                      'encoder, the measured winner'),
    ProviderMode('claude', 'Claude (CLI)', 'claude',
                 note='the full LLM pipeline on the Claude Code CLI already '
                      'logged in on this machine, so no API key is needed at '
                      'all — HyDE, LLM reranker, relevance gate, answerer and '
                      'both judges. At effort=low a call cost 3.9s on a short '
                      'probe and 5.6–7.4s on the prompts the lab actually '
                      'sends, and the calls bill your Claude account rather '
                      'than nothing'),
    ProviderMode('codex', 'Codex (CLI)', 'codex',
                 note='the same full pipeline on the Codex CLI. ~8.2s per call '
                      'at effort=low, and every call carries codex\'s own '
                      '~18.5k-token agent preamble, which no flag removes'),
)

# What the openrouter mode runs every stage on: the model every grade in
# .runs/ so far was measured on (see CHAT_MODELS).
MODE_MODEL = 'openai/gpt-5-nano'

# And what each of the other LLM-pipeline modes runs on. Keyed by mode, not by
# provider, so the table that says "this mode presets that model" is one table.
# `local` is absent on purpose: it presets the lab's own defaults, which is a
# reset rather than a model choice.
MODE_MODELS = {'openrouter': MODE_MODEL, 'claude': 'sonnet',
               'codex': 'gpt-5.6-luna'}

# Preferred relevance-gate models, in order: a purpose-built reranker beats a
# prompted chat model at this job, but one is picked only when OpenRouter's own
# model list verifies it — resolved when the preset is served, never at run
# time, since a runtime fallback would label a row with a model that never scored it.
GATE_MODELS = ('cohere/rerank-4-fast', 'cohere/rerank-4-pro')

# The fields a mode presets, and only these — index is absent since the
# embedder is the measured winner regardless of where the chat models run.
# `mode_config`'s closing assertion holds every mode to exactly this set.
_MODE_FIELDS = {
    'retrieval': ('hyde', 'expansion_model', 'reranker', 'reranker_model',
                  'grader', 'grader_model', 'grade_threshold'),
    'generation': ('answerer', 'model', 'key_facts_judge', 'judge_model',
                   'ragas_model'),
}


def gate_model(settings: LabSettings) -> str:
    """The openrouter mode's relevance-gate default: the first of GATE_MODELS
    that OpenRouter's model list verifies, else MODE_MODEL. An empty list means
    "cannot verify", and an unverifiable model must not be a default."""
    served = openrouter_ids(settings)
    for candidate in GATE_MODELS:
        if candidate in served:
            return candidate
    return MODE_MODEL


def mode_config(key: str, settings: LabSettings) -> dict:
    """The config patch a mode applies, shaped like LabConfig.to_dict().
    Unknown mode raises — no auto modes anywhere in this repo."""
    if key not in {mode.key for mode in MODES}:
        raise ValueError(f'unknown mode {key!r}; expected one of '
                         + ', '.join(repr(mode.key) for mode in MODES))
    if key in MODE_MODELS:
        model = MODE_MODELS[key]
        patch = {
            'retrieval': {
                'hyde': True, 'expansion_model': model,
                'reranker': 'llm', 'reranker_model': model,
                'grader': 'llm',
                # A purpose-built reranker beats a prompted chat model at the
                # gate, but that chain resolves against OpenRouter's own
                # catalogue — a slug from it means nothing to a CLI, so only
                # that mode consults it.
                'grader_model': (gate_model(settings) if key == 'openrouter'
                                 else model),
                # Measured threshold: balances refusing unanswerable questions
                # against wrongly refusing answerable ones.
                'grade_threshold': 0.4},
            'generation': {
                'answerer': 'llm', 'model': model,
                'key_facts_judge': True, 'judge_model': model,
                'ragas_model': model},
        }
    else:
        defaults = LabConfig().to_dict()
        patch = {group: {name: defaults[group][name] for name in names}
                 for group, names in _MODE_FIELDS.items()}
    assert {g: set(f) for g, f in patch.items()} == {
        g: set(f) for g, f in _MODE_FIELDS.items()}
    return patch


def mode_catalogue(settings: LabSettings) -> list[dict]:
    """Each mode with the backend it runs on, its config patch, and that backend's own model catalogue."""
    return [{'key': mode.key, 'label': mode.label, 'provider': mode.provider,
             'note': mode.note, 'config': mode_config(mode.key, settings),
             'models': catalogue(settings_for_provider(settings, mode.provider))}
            for mode in MODES]


def note_for(cfg: LabConfig, settings: LabSettings) -> str:
    """A one-line description of the stage/model split, for a run's notes.

    Names the backend too, since the same slug is a different measurement
    depending on where it ran. On a CLI backend it also names the reasoning
    effort, which moves the numbers but is deliberately not a `LabConfig` field
    or in the index fingerprint — so the row is self-describing even though
    `leaderboard.group` cannot tell two efforts apart on its own."""
    roles = resolve(cfg, settings)
    used = {role.key: getattr(roles, role.key) for role in ROLES}
    distinct = sorted(set(used.values()))
    where = settings.provider
    if settings.provider in clichat.CLIS:
        where = f'{where} (effort={settings.cli_effort})'
    if len(distinct) == 1:
        return f'every LLM stage ran on {distinct[0]} via {where}'
    return (f'models per stage (via {where}): '
            + ', '.join(f'{key}={value}' for key, value in used.items()))
