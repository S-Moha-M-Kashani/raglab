"""LLM backend settings: the repo paths, the provider/model tables, and `LabSettings`."""
import os
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]        # the raglab repo root
# At the repo root, not beside the code: runs are the account of the work, and
# burying them inside src/ would read as build output.
RUNS_DIR = ROOT / '.runs'

# Local Ollama is the default so a judged run's hundreds of calls cannot
# silently spend API credit; naming another provider is an explicit opt-in.
# `claude`/`codex` are a CLI on this machine (cli_subprocess_chat.py), not an endpoint, and need no key.
LLM_PROVIDERS = ('', 'openrouter', 'ollama', 'claude', 'codex', 'fake')

# One default model per backend, since a slug only means something to the
# backend that serves it. The local default is the model the judge screen has a
# row for (`.screens/`); 'fake' keeps the remote slug because it ignores the
# model entirely.
PROVIDER_MODELS = {'openrouter': 'openai/gpt-5-nano',
                   'ollama': '4skl/gemma4-e2b-mtp',
                   'claude': 'sonnet',
                   'codex': 'gpt-5.6-luna',
                   'fake': 'openai/gpt-5-nano'}


def load_env_file(path: Path | None = None) -> None:
    """Load repo-root .env into the environment without overriding what is already set."""
    path = path or ROOT / '.env'
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class LabSettings:
    openrouter_api_key: str = ''
    openrouter_base_url: str = 'https://openrouter.ai/api/v1'
    # See LLM_PROVIDERS. RAGLAB_LLM=ollama runs every LLM stage on a model on
    # this machine, which is what makes the expensive candidates measurable
    # without buying credit.
    llm_provider: str = 'ollama'
    ollama_base_url: str = 'http://localhost:11434/v1'
    # A setting, not an argv constant, because it moves the numbers; the two
    # CLIs accept different values and `clichat.checked_effort` refuses one the
    # chosen CLI does not — an unaccepted value can exit 0 with no text at all.
    cli_effort: str = 'low'
    # '' = the provider's own default (PROVIDER_MODELS), resolved in __post_init__.
    # It must follow the provider: a remote slug left standing under
    # RAGLAB_LLM=ollama made every run refuse for a model nobody picked.
    llm_model: str = ''
    fastembed_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    # Multilingual on purpose: fastembed's default rerankers are English-only
    # and score Farsi pairs as noise. Override with RAGLAB_CROSS_ENCODER.
    cross_encoder_model: str = 'jinaai/jina-reranker-v2-base-multilingual'

    def __post_init__(self):
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(
                f'unknown RAGLAB_LLM {self.llm_provider!r}; expected one of '
                + ', '.join(repr(name) for name in LLM_PROVIDERS))
        # Only when unset: overwriting a stated model would label a run with
        # one model while another scored it.
        if not self.llm_model:
            object.__setattr__(self, 'llm_model', PROVIDER_MODELS[self.provider])

    @property
    def provider(self) -> str:
        """The backend a chat model will actually be built with — never ''."""
        if self.llm_provider:
            return self.llm_provider
        return 'openrouter' if self.openrouter_api_key else 'fake'

    @property
    def llm_ready(self) -> bool:
        """Whether an LLM stage would reach a real model — not `bool(key)`, since 'fake' never fails either."""
        return self.provider in ('openrouter', 'ollama', 'claude', 'codex')


def load_lab_settings(env: dict | None = None) -> LabSettings:
    load_env_file()
    env = os.environ if env is None else env
    # BRAIN_CHROMA_URL and RAGLAB_CHROMA_DATABASE are deliberately not read: an
    # experiment must not find the board's Chroma stack just because it is there.
    return LabSettings(
        openrouter_api_key=env.get('OPENROUTER_API_KEY', ''),
        openrouter_base_url=env.get('OPENROUTER_BASE_URL',
                                    'https://openrouter.ai/api/v1'),
        llm_provider=env.get('RAGLAB_LLM', 'ollama'),
        ollama_base_url=env.get('RAGLAB_OLLAMA_BASE_URL',
                                'http://localhost:11434/v1'),
        cli_effort=env.get('RAGLAB_CLI_EFFORT', 'low'),
        llm_model=env.get('RAGLAB_MODEL', ''),
        fastembed_model=env.get(
            'RAGLAB_FASTEMBED_MODEL',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
        cross_encoder_model=env.get('RAGLAB_CROSS_ENCODER',
                                    'jinaai/jina-reranker-v2-base-multilingual'),
    )


def settings_for_provider(settings: LabSettings, provider: str) -> LabSettings:
    """One run's backend override, e.g. the panel's mode dropdown; '' passes settings through untouched.

    The old backend's *default* model does not survive the switch (PROVIDER_MODELS),
    but a model the user explicitly named (RAGLAB_MODEL) is never replaced."""
    if not provider:
        return settings
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f'unknown provider {provider!r}; expected one of '
            + ', '.join(repr(name) for name in LLM_PROVIDERS if name))
    model = ('' if settings.llm_model == PROVIDER_MODELS.get(settings.provider)
             else settings.llm_model)
    return replace(settings, llm_provider=provider, llm_model=model)
