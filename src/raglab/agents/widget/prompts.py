"""The widget's model-facing text, loaded from its fixtures.

Model-facing text is data, not code (the skills-folder and bilingual-pairs
rule): every prompt lives under fixtures/prompts/, so editing what the model
reads is editing a fixture. Three pages: `widget.yaml` (the two system
prompts and the empty log's four starters), `widget_tools.yaml` (one
description per tool, bound in tools.py), and `widget_knowledge.yaml` (the
facts search_knowledge_base serves and the CLI system prompt inlines).
"""
from raglab.configuration.env_settings import ROOT

PROMPTS_DIR = ROOT / 'fixtures' / 'prompts'


def _prompts(name: str) -> dict:
    """One YAML page from the prompts folder, parsed at import — the pages
    are small, and a missing key must fail loudly here rather than serve a
    tool with no description."""
    import yaml
    return yaml.safe_load(
        (PROMPTS_DIR / f'{name}.yaml').read_text(encoding='utf-8'))


_PROMPTS = _prompts('widget')
_TOOL_PROMPTS = _prompts('widget_tools')

SYSTEM_PROMPT = _PROMPTS['system'].strip()
MEMORY_POLICY_PROMPT = _PROMPTS['memory_policy'].strip()
ACTIVE_EXPERIMENT_PROMPT = _PROMPTS['active_experiment'].strip()

# The four questions the empty log offers. Model-facing text like the prompts
# either side of it: clicking one sends exactly this string.
STARTERS: list = [line.strip() for line in _PROMPTS['starters']]

# The project facts the widget answers from. A dict rather than the raw page
# so a caller iterates key -> text exactly as the old in-code table read.
KNOWLEDGE_BASE: dict = {key: text.strip()
                        for key, text in _prompts('widget_knowledge').items()}
