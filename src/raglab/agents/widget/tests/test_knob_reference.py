"""The knob reference and the two widget tools that serve it.

`fixtures/knobs/` holds one Markdown page per knob of the lab's own knob
surface: a title line `# <group>.<field> — <one-line summary>`, then what the
knob does, what it means scientifically, why RAG architectures have such a
knob, when it is useful, and which knobs it interacts with.
`raglab/agents/widget/knob_reference.py` is the loader, and the widget gains
two tools over it — the same cheap-then-expensive shape as the skills corpus:
`search_knobs` routes on the title summaries (the cheap layer), `read_knob`
pays for whole pages only on commit, several at a time but capped.

The corpus is files in this repository; nothing here reaches a network. The
claim that earns the convention test below is coverage: a knob the panel
explains but this folder does not is a knob the widget cannot answer for.
"""
import pytest

from raglab.agents.widget import knob_reference as knobs
from raglab.agents import widget
from raglab.configuration.env_settings import ROOT
from raglab.configuration.knob_help_text import HELP
from raglab.llm_backends.model_role_catalogue import ROLES


# --- the loader ----------------------------------------------------------

def test_the_index_covers_every_knob_the_lab_explains_and_no_others():
    # this is a convention test
    """The knob surface is `HELP` plus the model roles, which carry their own
    help instead of a HELP entry. A knob missing here is one the widget cannot
    answer for; a page for a knob the lab no longer has is rot that would
    answer confidently about a control nobody can set. The summary each page
    is served under is checked here too, since it is the layer a search routes
    on: a page whose title said nothing would be unfindable."""
    served = set(HELP) | {role.field for role in ROLES}
    index = knobs.index()
    assert set(index) == served
    listing = (ROOT / 'fixtures' / 'knobs' / 'README.md').read_text(encoding='utf-8')
    for key, summary in index.items():
        # the title-line contract: the key names its own file, and the summary
        # is the title's second half — neither the hash nor the key again
        assert (ROOT / 'fixtures' / 'knobs' / f'{key}.md').is_file()
        assert summary and not summary.startswith('#') and key not in summary
        # and the folder's own README lists it. `index.delimiters` arrived on
        # 2026-09-04 from another branch and had to be added there by hand;
        # the list is the reader's map of the corpus, so it rots silently
        # unless something reads it.
        assert key in listing, f'{key} is missing from fixtures/knobs/README.md'


def test_a_page_is_the_whole_markdown_of_that_knobs_file():
    # this is a unit test
    page = knobs.page('retrieval.summary_boost')
    assert page.startswith('# retrieval.summary_boost — ')
    assert 'candidate cut' in page
    assert '## Interactions' in page
    # the page states which step it belongs to, because that is what a change
    # there costs
    assert '**Step:** Retrieval' in page


def test_an_unknown_key_raises_with_valid_keys_in_the_message():
    # this is a unit test
    with pytest.raises(KeyError) as caught:
        knobs.page('index.zeppelin')
    assert 'index.split_plan' in str(caught.value)


def test_related_knobs_are_the_ones_that_page_names_in_its_interactions():
    # this is a convention test
    """The interaction lines are the graph inside the corpus: they are what
    lets the widget answer "if I raise this, what else moves" without a graph
    engine. Every neighbour must be a knob that exists, and no page is its own
    neighbour."""
    index = knobs.index()
    related = knobs.related('retrieval.summary_boost')
    assert 'index.hierarchy' in related
    assert 'retrieval.summary_scope' in related
    for key in index:
        neighbours = knobs.related(key)
        assert key not in neighbours, f'{key} lists itself as related'
        assert set(neighbours) <= set(index), (
            f'{key} names a knob that has no page: '
            f'{sorted(set(neighbours) - set(index))}')


def test_the_loader_reads_a_custom_root_and_skips_a_page_with_no_title_line(tmp_path):
    # this is a unit test
    """One malformed page must not take the corpus down — the widget is a
    helper, and a helper that dies on one bad file serves nothing."""
    (tmp_path / 'index.good.md').write_text(
        '# index.good — a well-formed page\n\nBody.\n\n'
        '## Interactions\n`index.split_plan` moves with it.\n', encoding='utf-8')
    (tmp_path / 'index.bad.md').write_text('no title line at all\n',
                                           encoding='utf-8')
    (tmp_path / 'README.md').write_text('# Not a knob — prose\n', encoding='utf-8')
    index = knobs.index(root=tmp_path)
    assert index == {'index.good': 'a well-formed page'}
    assert 'Body.' in knobs.page('index.good', root=tmp_path)
    assert knobs.related('index.good', root=tmp_path) == ('index.split_plan',)


def test_search_matches_on_keys_summaries_and_bodies_case_insensitively():
    # this is a unit test
    keys = [key for key, _ in knobs.search('RECIPROCAL rank fusion')]
    assert 'retrieval.rrf_k' in keys
    assert 'retrieval.summary_boost' in [k for k, _ in knobs.search('summary_boost')]
    assert knobs.search('zeppelin xylophone') == []


def test_search_ranks_the_closest_knobs_first_and_bounds_what_it_returns():
    # this is a unit test
    """The regression a real turn found on 2026-09-03: asked "which questions
    did retrieval miss in my last run, and which knob fixes that?", the model
    called search_knobs eight times in a row and the hop guard stopped the
    turn. The cause was here, not in the guard — matching any query word
    anywhere in a 2.6 KB page returns most of the corpus, unranked and
    alphabetical, so no reply moved the model forward and it kept rewording.

    A search must therefore rank (a key or summary match beats a mention
    buried in prose) and it must be bounded, so one reply is a shortlist to
    commit to rather than a wall to narrow."""
    everything = knobs.search('which knob fixes missed retrieval')
    assert len(everything) <= knobs.MAX_SEARCH_HITS < len(knobs.index())

    # ranked: the knob whose own key is asked for leads, and a page that only
    # mentions the word in passing does not outrank it
    ranked = [key for key, _ in knobs.search('reranker')]
    assert ranked[0] == 'retrieval.reranker'
    if 'index.overlap' in ranked:
        # index.overlap mentions rerankers once, in passing; rerank_depth is
        # about them
        assert ranked.index('retrieval.rerank_depth') < ranked.index('index.overlap')

    # a query naming a knob exactly gets that knob, not a page that cites it
    assert knobs.search('retrieval.summary_levels')[0][0] == 'retrieval.summary_levels'

    # question words score nothing. The first ranking measured put the three
    # *_model knobs on top of the failing question, because their summaries
    # open with "which model …" and `which` counted as a summary match.
    assert knobs.search('which what how does') == []
    top = [key for key, _ in knobs.search('which knob fixes missed retrieval')][:3]
    assert not all(key.endswith('_model') for key in top), top


def test_search_knobs_says_how_many_more_matched_when_it_caps():
    # this is a unit test
    """A shortlist that hid the rest silently would invite the same rewording
    loop: the model has to know the cap is why it is seeing eight."""
    reply = widget.search_knobs.invoke(
        {'query': 'which knob fixes missed retrieval'})
    assert reply.count('\n') < len(knobs.index())
    assert 'more' in reply.lower()


# --- the two widget tools ------------------------------------------------

def test_the_widget_offers_both_knob_tools_and_names_them_in_its_prompt():
    # this is a convention test
    """A tool the system prompt never mentions is a tool the model does not
    know it has — the skills pair is named there for the same reason."""
    names = {tool.name for tool in widget.TOOLS}
    for tool in ('search_knobs', 'read_knob'):
        assert tool in names
        assert tool in widget.SYSTEM_PROMPT


def test_search_knobs_returns_the_summary_beside_the_key():
    # this is a unit test
    reply = widget.search_knobs.invoke({'query': 'diversity redundancy'})
    assert 'retrieval.mmr_lambda' in reply
    assert knobs.index()['retrieval.mmr_lambda'][:40] in reply


def test_search_knobs_lists_the_whole_knob_surface_when_nothing_matches():
    # this is a unit test
    reply = widget.search_knobs.invoke({'query': 'zeppelin xylophone'})
    for key in ('index.split_plan', 'retrieval.k', 'generation.answerer', 'run.limit'):
        assert key in reply


def test_read_knob_returns_the_page_with_the_knobs_it_interacts_with():
    # this is a unit test
    reply = widget.read_knob.invoke({'keys': 'retrieval.summary_boost'})
    assert '## Interactions' in reply
    # the neighbours travel with the page, so the model can follow them
    # without a second search
    assert 'index.hierarchy' in reply


def test_read_knob_returns_several_pages_each_under_its_key():
    # this is a unit test
    reply = widget.read_knob.invoke({'keys': 'index.split_plan, retrieval.k'})
    assert reply.index('index.split_plan') < reply.index('retrieval.k')
    assert '=== index.split_plan ===' in reply
    assert '=== retrieval.k ===' in reply


def test_read_knob_caps_how_many_pages_one_call_returns():
    # this is a unit test
    """Pages are the expensive layer; a call that asked for the whole surface
    would put ~130 KB into the loop. The cap is stated in the reply, so the
    model knows to ask again rather than assuming it saw everything."""
    asked = ', '.join(list(knobs.index())[:5])
    reply = widget.read_knob.invoke({'keys': asked})
    assert reply.count('\n=== ') <= widget.MAX_KNOB_READS
    assert str(widget.MAX_KNOB_READS) in reply
    assert widget.MAX_KNOB_READS == 3


def test_read_knob_reports_an_unknown_key_and_serves_the_known_one():
    # this is a unit test
    """A model that mistypes one key must not lose the whole call — the reply
    serves what it can and says what it could not."""
    reply = widget.read_knob.invoke({'keys': 'index.split_plan, index.zeppelin'})
    assert '=== index.split_plan ===' in reply
    assert 'index.zeppelin' in reply
    assert 'index.chunk_chars' in reply     # the valid keys are offered back
