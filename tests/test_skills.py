"""The skills corpus and the widget tools that serve it.

`skills/` holds thirteen SKILL.md files — frontmatter (name, description) plus
a Markdown body. `raglab/skills.py` is the loader, and the widget gains two
tools over it: `search_rag_skills` routes on the descriptions (the cheap
layer), `read_rag_skill` pays for bodies only on commit, several at a time but
capped. The same drill-down shape as `summary_scope='drill-down'`, one level
up. Nothing here reaches a network; the corpus is files in this repository.
"""
import os

import pytest

from raglab import skills, widget
from raglab.settings import ROOT


# --- the loader ----------------------------------------------------------

def test_the_index_serves_every_skill_with_its_description():
    # this is a unit test
    index = skills.index()
    assert 'rag-evaluation' in index
    assert 'chunking-strategies' in index
    for name, description in index.items():
        assert isinstance(description, str) and description
        # the frontmatter contract: name matches its folder
        assert (ROOT / 'skills' / name / 'SKILL.md').is_file()


def test_the_index_covers_every_skill_folder_in_the_repository():
    # this is a convention test
    """A skill someone adds must be served without touching code — and a
    folder the loader silently drops is rot no screen reports."""
    folders = {p.parent.name for p in (ROOT / 'skills').glob('*/SKILL.md')}
    assert folders == set(skills.index())


def test_a_body_is_the_markdown_without_its_frontmatter():
    # this is a unit test
    text = skills.body('rag-evaluation')
    assert '# RAG Evaluation' in text
    assert not text.startswith('---')
    assert 'description:' not in text.split('\n', 1)[0]


def test_an_unknown_name_raises_with_the_valid_names_in_the_message():
    # this is a unit test
    with pytest.raises(KeyError) as caught:
        skills.body('zeppelin-rag')
    assert 'rag-evaluation' in str(caught.value)


def test_search_matches_on_descriptions_and_bodies_case_insensitively():
    # this is a unit test
    names = [name for name, _ in skills.search('JUDGE screening')]
    assert 'rag-evaluation' in names
    assert skills.search('zeppelin xylophone') == []


def test_the_loader_reads_a_custom_root_and_skips_a_malformed_file(tmp_path):
    # this is a unit test
    """One broken file must not take the whole corpus down — the widget is a
    helper, and a helper that dies on one bad page serves nothing."""
    good = tmp_path / 'good-skill'
    good.mkdir()
    (good / 'SKILL.md').write_text(
        '---\nname: good-skill\ndescription: A well-formed skill.\n---\n\nBody.\n',
        encoding='utf-8')
    bad = tmp_path / 'bad-skill'
    bad.mkdir()
    (bad / 'SKILL.md').write_text('no frontmatter at all\n', encoding='utf-8')
    (tmp_path / 'not-a-skill').mkdir()   # folder without SKILL.md: ignored
    index = skills.index(root=tmp_path)
    assert index == {'good-skill': 'A well-formed skill.'}
    assert 'Body.' in skills.body('good-skill', root=tmp_path)


def test_an_edited_skill_is_served_fresh(tmp_path):
    # this is a unit test
    folder = tmp_path / 'living-skill'
    folder.mkdir()
    page = folder / 'SKILL.md'
    page.write_text('---\nname: living-skill\ndescription: First.\n---\nOne.\n',
                    encoding='utf-8')
    assert skills.index(root=tmp_path)['living-skill'] == 'First.'
    page.write_text('---\nname: living-skill\ndescription: Second.\n---\nTwo.\n',
                    encoding='utf-8')
    stamp = page.stat()
    os.utime(page, (stamp.st_atime + 2, stamp.st_mtime + 2))
    assert skills.index(root=tmp_path)['living-skill'] == 'Second.'


# --- the disambiguation guide --------------------------------------------

def test_the_distinctions_guide_names_every_skill():
    # this is a convention test
    """The guide is what keeps thirteen near-neighbours apart for the model —
    a skill it does not mention is a skill it cannot disambiguate."""
    for name in skills.index():
        assert name in skills.DISTINCTIONS, f'{name} missing from DISTINCTIONS'


def test_the_index_text_carries_names_descriptions_and_distinctions():
    # this is a unit test
    """`index_text()` is the one formatted block both consumers inline — the
    search tool's no-match reply and the CLI system prompt."""
    text = skills.index_text()
    index = skills.index()
    for name, description in index.items():
        assert name in text
        # the characterisation travels with the name, not summarised away
        assert description[:60] in text
    assert skills.DISTINCTIONS.strip() in text


# --- the two widget tools ------------------------------------------------

def test_the_widget_offers_both_skill_tools():
    # this is a unit test
    names = {tool.name for tool in widget.TOOLS}
    assert 'search_rag_skills' in names
    assert 'read_rag_skill' in names


def test_search_rag_skills_returns_the_characterisation_not_just_names():
    # this is a unit test
    reply = widget.search_rag_skills.invoke({'query': 'error analysis dev test split'})
    assert 'rag-experiment-methodology' in reply
    index = skills.index()
    assert index['rag-experiment-methodology'][:60] in reply


def test_search_rag_skills_lists_the_whole_catalogue_when_nothing_matches():
    # this is a unit test
    reply = widget.search_rag_skills.invoke({'query': 'zeppelin xylophone'})
    for name in skills.index():
        assert name in reply


def test_read_rag_skill_returns_several_bodies_each_under_its_name():
    # this is a unit test
    reply = widget.read_rag_skill.invoke(
        {'names': 'rag-evaluation, rag-experiment-methodology'})
    assert '# RAG Evaluation' in reply
    assert '# RAG Experiment Methodology' in reply
    # each body is labelled, so two skills cannot blur into one answer
    assert reply.index('rag-evaluation') < reply.index('rag-experiment-methodology')


def test_read_rag_skill_caps_how_many_bodies_one_call_returns():
    # this is a unit test
    """Bodies are the expensive layer; a call that asked for the whole corpus
    would put ~100 KB into the loop. The cap is stated in the reply."""
    asked = ', '.join(list(skills.index())[:5])
    reply = widget.read_rag_skill.invoke({'names': asked})
    assert reply.count('\n# ') + reply.count('\n\n# ') <= widget.MAX_SKILL_READS + 1
    assert str(widget.MAX_SKILL_READS) in reply
    assert widget.MAX_SKILL_READS == 3


def test_read_rag_skill_reports_an_unknown_name_and_serves_the_known_one():
    # this is a unit test
    """A model that mistypes one name must not lose the whole call — the
    reply serves what it can and says what it could not."""
    reply = widget.read_rag_skill.invoke({'names': 'rag-evaluation, zeppelin-rag'})
    assert '# RAG Evaluation' in reply
    assert 'zeppelin-rag' in reply
    assert 'rag-research-radar' in reply    # the valid names are offered back


# --- the prompts keep the two corpora apart ------------------------------

def test_the_system_prompt_separates_project_facts_from_field_knowledge():
    # this is a unit test
    """`KNOWLEDGE_BASE` is this project; `skills/` is the field. The prompt
    names both tools and the boundary, so the widget neither answers 'what
    does this lab do' from a literature file nor presents a literature claim
    as a measurement taken here."""
    prompt = widget.SYSTEM_PROMPT
    assert 'search_rag_skills' in prompt
    assert 'read_rag_skill' in prompt
    assert 'search_knowledge_base' in prompt


def test_the_cli_prompt_inlines_the_index_and_says_bodies_are_out_of_reach():
    # this is a unit test
    """A CLI backend runs one call with no tools: it gets the descriptions —
    an option states what it can do — and must say the bodies are not
    readable there rather than inventing their content."""
    system = widget._cli_system()
    for name in skills.index():
        assert name in system
    for key in widget.KNOWLEDGE_BASE:
        assert key in system
