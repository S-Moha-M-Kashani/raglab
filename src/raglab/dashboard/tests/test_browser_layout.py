# this is an end-to-end test
"""Nothing on any surface is squeezed out of its box.

Every other browser journey asks whether a control *works*: press it, and the
right thing happens. None of them asks whether a reader can *read* the result,
and that turned out to be a real gap. The split plan's typed line shared a
flex row with a dropdown whose options are whole sentences; the dropdown took
the column, the line was shrunk to about one character wide, and
`overflow-wrap: anywhere` turned `document / drift …` into a vertical column
of single letters half a screen tall. Every test passed — the element existed,
was visible, and held exactly the right text.

So this file asserts geometry, which is the one thing the rest of the suite
never looks at, and it asserts only what cannot be argued with:

* no surface scrolls sideways, which is the repo's own stated rule for wide
  content — a table or a diagram scrolls inside its own region, the page body
  never does; and
* no visible element is narrower than a character is wide while being taller
  than a paragraph, which is the exact signature of the failure above and has
  no legitimate cause.

Deliberately not a screenshot comparison: a pixel baseline fails on every font
and renderer, needs regenerating after every honest change, and teaches a
maintainer to regenerate it without looking. These two rules cost nothing to
keep true and cannot be satisfied by a broken page.
"""
import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser

#: The four surfaces a reader can open, each at a width where a real reader
#: would use it. The narrow pass matters: the panel's controls sit in a column
#: that gets tighter as the setup drawer opens, which is where a row that only
#: just fits stops fitting.
SURFACES = ('/', '/inspector', '/leaderboard', '/dataset?dataset=smoke-mini')
WIDTHS = (1440, 1024)

#: A horizontal scrollbar of a pixel or two is a rounding artefact of a
#: fractional layout, not a reader-visible overflow.
SLACK = 2

#: Narrower than one character of the smallest type on the page, and taller
#: than a paragraph. Nothing legitimate is that shape; a squeezed flex item
#: holding a sentence is exactly that shape.
TOO_NARROW = 40
TALL = 100

#: What the rule below is measured against: elements the reader can see, that
#: hold text of their own rather than inheriting a child's.
SQUEEZED = """() => {
  const bad = [];
  for (const el of document.querySelectorAll('body *')) {
    const own = [...el.childNodes]
      .filter((n) => n.nodeType === 3).map((n) => n.textContent.trim()).join('');
    if (own.length < 8) continue;
    const box = el.getBoundingClientRect();
    if (box.width === 0 && box.height === 0) continue;
    if (box.width < %(narrow)d && box.height > %(tall)d) {
      bad.push(`${el.tagName.toLowerCase()}#${el.id || ''}.${el.className || ''} `
        + `${Math.round(box.width)}x${Math.round(box.height)} — ${own.slice(0, 40)}`);
    }
  }
  return bad;
}""" % {'narrow': TOO_NARROW, 'tall': TALL}


@pytest.mark.parametrize('path', SURFACES)
@pytest.mark.parametrize('width', WIDTHS)
def test_no_surface_scrolls_sideways_and_no_text_is_squeezed_into_a_column(
        lab_server, page, path, width):
    page.set_viewport_size({'width': width, 'height': 1200})
    page.goto(f'{lab_server}{path}')
    # The one call every page makes on load, waited for by its visible effect
    # rather than by a clock: until it lands, the page is mostly empty and
    # nothing about its layout is worth measuring.
    page.wait_for_load_state('networkidle')

    overflow = page.evaluate(
        '() => document.documentElement.scrollWidth'
        ' - document.documentElement.clientWidth')
    assert overflow <= SLACK, (
        f'{path} at {width}px scrolls {overflow}px sideways — wide content '
        'belongs in its own scroll region, never the page body')

    squeezed = page.evaluate(SQUEEZED)
    assert squeezed == [], (
        f'{path} at {width}px has text squeezed into a narrow column: '
        f'{squeezed}')
