"""Which language model runs which stage.

Seven stages of the lab can call a model, and they want different things from
one. The summariser runs once per session (157 calls per build) and wants cheap;
the key-facts judge runs once per question and wants the strongest thing
available; the reranker sits in the latency path of every query. Sharing one
model across all of them makes those trade-offs unmeasurable, so each stage
carries its own choice and nothing here is hard-coded.

Two rules the catalogue keeps:

**Every option says where its weights stand.** On a Farsi corpus the open-weight
models are the interesting ones — they can be run locally later, which is the
whole direction of the brain's LLMProvider seam — so `open` / `closed` is part of
the label, not a footnote.

**The remote catalogue offers only models this account can reach.** The old rule
— an unrun model stays listed as NA, "worth trying, nobody measured it yet" —
was checked against the wire on 2026-08-02 and had rotted: all six open-weight
options answered 404 ("no endpoints matching your guardrail restrictions and
data policy"), so NA had stopped meaning "unmeasured" and started meaning
"broken", the one thing a dropdown must not hide. The local list keeps the old
rule, because there NA is honest: a tag that is merely not pulled is one
`ollama pull` away, and the daemon is asked directly rather than guessed at.
Availability is still verified against OpenRouter's own model list when a key is
present, and never guessed when it is not.
"""
from dataclasses import dataclass, fields

from . import clichat
from .config import LabConfig, LabSettings, settings_for_provider

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

# Served by Ollama on this machine, and therefore a different list: an OpenRouter
# slug is not a thing Ollama can load, and a local tag is not a thing OpenRouter
# serves. Mixing them in one dropdown would offer every user roughly half a menu
# of choices that cannot work, so the catalogue serves whichever list matches the
# active provider.
#
# Free, private, and slow — which reorders the trade-offs completely. On
# OpenRouter the interesting question is capability per cent; here it is capability
# per second, because the judge is ~276 calls per run and a reasoning model that
# spends 500 thinking tokens per verdict turns one candidate into an hour.
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
# (clichat.py), so a third and fourth list — an OpenRouter slug is not a thing
# `claude --model` accepts, and an alias is not a thing OpenRouter serves.
# Deliberately three entries: the answerer, the judge that must not be the
# answerer, and the speed candidate. `fable` is reachable through RAGLAB_MODEL
# and is not offered, because nothing here has measured it.
#
# **None of these is `verified`**, and that is not a claim that nothing was
# measured — `sonnet` and `gpt-5.6-terra` both answered here. `verified` makes an
# option available *unconditionally* (`ModelOption.as_dict` returns
# `verified or id in live`), which is the right escape hatch when availability
# cannot be checked. Here it can: the binary either exists or it does not. A
# verified option would show as usable on a machine with no `claude` installed,
# while `provider_problems` refused the run — the panel offering what the lab
# refuses. The measurement lives in the note instead.
CLAUDE_MODELS = (
    ModelOption('sonnet', 'Claude Sonnet (CLI)', 'closed',
                note='what this backend was measured on: 3.9s per call at '
                     'effort=low on the lab\'s own grade prompt'),
    ModelOption('opus', 'Claude Opus (CLI)', 'closed',
                note='the sweep\'s judge under this backend — a model grading '
                     'its own output is not evidence, so the answerer stays '
                     'sonnet'),
    ModelOption('haiku', 'Claude Haiku (CLI)', 'closed',
                note='the speed candidate: the judge is ~420 calls per run, '
                     'and each one here is a process spawn as well as a turn'),
)

# One entry, and honestly so: the codex CLI publishes no model list to verify
# against, and this is the alias that has actually answered here. RAGLAB_MODEL
# names another, and `catalogue` offers whatever it names.
CODEX_MODELS = (
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
        """Which pipeline step this model serves, and therefore which colour it
        wears in the panel. Derived from the field rather than declared twice:
        an ink that disagreed with where the value is stored would be worse than
        no ink at all."""
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
    """Drop what has been verified about every backend's model list.

    Availability is asked of the backend and cached per url, which is right for
    a panel that reloads its options on every visit and wrong the moment the
    credential changes underneath it: with no key, "what does OpenRouter serve"
    answers the empty set, and a cached empty set reads as *nothing is
    available*. `credentials.set_key` calls this, so entering a key in the panel
    re-asks rather than leaving every remote model marked NA until a restart."""
    _LIVE.clear()


def openrouter_ids(settings: LabSettings) -> frozenset:
    """Model ids OpenRouter is currently serving, or an empty set.

    Best-effort by design: the lab must run with no network at all, so a failure
    here means "cannot verify" (everything unverified shows as NA) and never an
    exception. Cached per base URL — the panel asks for options on every visit."""
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

    Verified, never guessed — the same rule the embedder catalogue keeps. And
    unlike OpenRouter's list this one is authoritative in both directions: a tag
    absent here genuinely cannot be loaded, which is what lets
    `provider_problems` refuse a run instead of merely marking it NA.

    `/api/tags` rather than the OpenAI-compatible `/v1/models`, because a tag is
    what `ollama pull` names and what a run should be labelled with."""
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

    For a CLI the checkable fact is the *binary*: there is no `/api/tags` to
    ask, and an alias cannot be verified without paying for a call. So with the
    command present its catalogue is offerable — plus whatever RAGLAB_MODEL
    named, which is by definition a model the user wants — and with the command
    absent nothing is, which is what makes the catalogue say NA instead of
    claiming an alias on a machine that cannot run any of them.
    """
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
    """Model choices the active backend cannot serve.

    A validation error, never a silent fallback — the embedder rule applied to
    chat models, for the same reason: a leaderboard row labelled qwen3.5:2b that
    was actually scored by gpt-5-mini is the single worst artefact this lab can
    produce, and no field on the row would contradict it.

    Two refusals, and one rule joining them: **this guard only ever refuses what
    it has verified absent**, so what it checks is whatever the backend can be
    asked. For `ollama` that is the model — `/api/tags` answers with the tags it
    serves, and an unreachable daemon claims *nothing*, because "cannot check"
    and "not there" are different facts. For a CLI there is no `/api/tags` and an
    alias cannot be checked without paying for a call, so the verifiable fact is
    the **binary**: a missing command is refused and an unknown alias is not, the
    CLI naming an alias it cannot serve better at call time than anything guessed
    here.

    `openrouter` is therefore refused nothing. Its list is authoritative in one
    direction only: everything on it works, but a slug missing from it may still
    be perfectly valid — the routing suffixes (`:free`, `:floor`) do not appear as
    ids. Refusing on that basis would block runs that used to work, which is a
    worse failure than the one this guard prevents."""
    if settings.provider in clichat.CLIS:
        # One check, and it is the one that can be made. An alias is *not*
        # refused here: nothing verified it absent, and this guard only ever
        # refuses what it has verified — the CLI names an alias it cannot serve
        # itself, at call time, better than anything guessed here could.
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
    """The dropdown contents: the lab default first, then every candidate with
    its licence and whether it can be used right now."""
    live = served_ids(settings)
    known = list(known_models(settings))
    if settings.llm_model not in {option.id for option in known}:
        # An id set by RAGLAB_MODEL is by definition one the user wants, so it is
        # offered even though nothing is known about its weights.
        #
        # `verified=not live` is the honest reading of the two cases: with a
        # served list in hand, membership of that list decides — including for
        # the user's own pick, which is exactly the case provider_problems will
        # refuse a run over. With no list, nothing can be checked, so the user's
        # choice is taken at face value rather than shown as NA.
        known.insert(0, ModelOption(settings.llm_model, settings.llm_model,
                                    'unknown', verified=not live,
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
                      'both judges. ~3.9s per call at effort=low, and the calls '
                      'bill your Claude account rather than nothing'),
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
               'codex': 'gpt-5.6-terra'}

# Preferred relevance-gate models, in order. A purpose-built reranker (query +
# text → relevance score) beats a prompted chat model at exactly this job — but
# one is picked only when OpenRouter's own model list verifies it, and the
# resolution happens when the preset is served, never at run time: a runtime
# fallback would produce a row labelled with a model that did not score it.
# Measured 2026-07-31: OpenRouter's catalogue had no rerank entries at all
# (test_no_knob_offers_an_openrouter_embedding_or_rerank_model), so today this
# chain resolves to MODE_MODEL; the preference stands so the day the list
# gains one, the preset starts offering it without a code change.
GATE_MODELS = ('cohere/rerank-4-fast', 'cohere/rerank-4-pro')

# The fields a mode presets — and only these. Index is deliberately absent:
# heydariAI/persian-embeddings is the measured winner regardless of where the
# chat models run. **Every** mode patches exactly these fields, read off this one
# table — the three in MODE_MODELS by naming their own model at each stage, and
# `local` by writing the lab's own defaults back into the same set — so switching
# back is a full reset rather than a remote model leaking into a local run's
# label. `mode_config`'s closing assertion is what holds the two paths to it.
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
                # The measured setting: an LLM gate at 0.4 refused all five
                # unanswerable questions with 3% false refusals.
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
    """The dropdown contents: each mode with the backend it runs on, the exact
    patch picking it applies, and — because a slug only means something to the
    backend that serves it — that backend's own model catalogue. Without the
    list riding along, the panels filled their dropdowns from the boot
    provider's catalogue, could not display the preset under the other mode,
    and their config-follows-the-panel rule silently wiped it back to ''."""
    return [{'key': mode.key, 'label': mode.label, 'provider': mode.provider,
             'note': mode.note, 'config': mode_config(mode.key, settings),
             'models': catalogue(settings_for_provider(settings, mode.provider))}
            for mode in MODES]


def note_for(cfg: LabConfig, settings: LabSettings) -> str:
    """A one-line description of the stage/model split, for a run's notes.

    The backend is named as well as the models, because the same slug is a
    different measurement depending on where it ran: 'fake' means every LLM
    number on the row is meaningless, and local versus remote is the difference
    between a free row and a paid one.

    On a CLI backend the reasoning effort is named too, for exactly that reason:
    it moves the numbers (the grade probe scored 8 under `low` where the default
    scored 9), and without it two rows on claude/sonnet at `effort=low` and
    `effort=max` are labelled identically. `leaderboard.group` would then rank
    them against each other as a comparable pair, which is this repository's one
    unbreakable rule broken by a knob. **The row is now self-describing but the
    grouping is not**: effort is deliberately not a `LabConfig` field and not in
    the index fingerprint — that is a larger change, and doing it here would put
    an LLM setting into the config that decides whether an index needs
    rebuilding. So two efforts still group together; the note is what tells the
    reader they were two."""
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
