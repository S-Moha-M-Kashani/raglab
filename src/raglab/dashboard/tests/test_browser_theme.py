# this is an end-to-end test
"""Two themes and no third, proved where the rule can actually break.

The theme is one attribute on `<html>`, one key in `localStorage` and a
guarded `prefers-color-scheme` block in `tokens.css` — and the whole point of
the guard is that a reader's explicit choice outranks the machine's. None of
that is visible to an offline assertion on markup: only a real browser
resolves a media query, cascades a custom property and reports the colour the
page actually painted. So this journey drives the three radios on a real
Chromium, emulates a machine set to dark and to light, and reads back the
computed ground rather than a class name. It also follows one choice across a
reload and across all three surfaces, because the three pages share one origin
and one key, which is the only reason the control can be the same control
everywhere.
"""
import pytest

pytest.importorskip('playwright.sync_api',
                    reason='browser suite: uv sync --extra browser-tests')

pytestmark = pytest.mark.browser

from playwright.sync_api import expect  # noqa: E402  (after the skip guard)


def _open_settings(page):
    """The three radios live behind the settings disc; open it once."""
    if not page.locator('#app-settings-panel').is_visible():
        page.click('#app-settings')
    expect(page.locator('#app-settings-panel')).to_be_visible()


def _choose(page, name: str):
    """Pick a theme the way a reader does — by clicking its label.

    The radio itself is a clipped one-pixel input (`chrome.css`), so the label
    is the real target, and clicking it is what fires the `change` listener
    `lab.js` hangs on the group.
    """
    _open_settings(page)
    page.click(f'#theme-control label[for="theme-{name}"]')
    expect(page.locator(f'#theme-{name}')).to_be_checked()


def _stored_theme(page):
    return page.evaluate("() => localStorage.getItem('raglab-theme')")


def _ground(page) -> str:
    """The colour the page actually painted behind everything else.

    `body { background: var(--plate) }` on all three surfaces, so this is the
    end of the whole cascade — the token, the theme block that assigned it and
    the media query that may have overridden both.
    """
    return page.evaluate('() => getComputedStyle(document.body).backgroundColor')


def _colour_scheme(page) -> str:
    return page.evaluate(
        '() => getComputedStyle(document.documentElement).colorScheme')


def _is_dark(colour: str) -> bool:
    """Dark enough that the browser's own widgets would be drawn dark too."""
    channels = [int(part) for part in colour[colour.index('(') + 1:
                                             colour.index(')')].split(',')[:3]]
    return sum(channels) / 3 < 128


def test_day_stamps_the_root_stores_the_choice_and_paints_light(panel):
    _choose(panel, 'day')

    expect(panel.locator('html')).to_have_attribute('data-theme', 'day')
    assert _stored_theme(panel) == 'day'
    assert not _is_dark(_ground(panel)), _ground(panel)
    assert _colour_scheme(panel) == 'light'


def test_night_stamps_the_root_and_the_ground_actually_changes(panel):
    _choose(panel, 'day')
    daylight = _ground(panel)

    _choose(panel, 'night')

    expect(panel.locator('html')).to_have_attribute('data-theme', 'night')
    assert _stored_theme(panel) == 'night'
    assert _ground(panel) != daylight
    assert _is_dark(_ground(panel)), _ground(panel)
    assert _colour_scheme(panel) == 'dark'


def test_auto_stores_nothing_and_lets_the_machine_decide(panel):
    """Auto is the absence of a choice: no attribute, no key, no third theme."""
    _choose(panel, 'night')
    chosen_night = _ground(panel)

    _choose(panel, 'auto')

    assert panel.evaluate(
        "() => document.documentElement.hasAttribute('data-theme')") is False
    assert _stored_theme(panel) is None

    panel.emulate_media(color_scheme='dark')
    assert _is_dark(_ground(panel)), _ground(panel)
    # The same values either way in: Night chosen and Night inherited are one
    # palette, written once and assigned by two blocks.
    assert _ground(panel) == chosen_night

    panel.emulate_media(color_scheme='light')
    assert not _is_dark(_ground(panel)), _ground(panel)


def test_the_reader_outranks_the_machine(panel):
    """A dark machine plus an explicit Day is a light page — that is the guard."""
    panel.emulate_media(color_scheme='dark')
    _choose(panel, 'auto')
    assert _is_dark(_ground(panel)), 'the machine was not honoured on Auto'

    _choose(panel, 'day')

    expect(panel.locator('html')).to_have_attribute('data-theme', 'day')
    assert not _is_dark(_ground(panel)), _ground(panel)
    assert _colour_scheme(panel) == 'light'


def test_the_choice_survives_a_reload(panel):
    _choose(panel, 'night')

    panel.reload()

    expect(panel.locator('html')).to_have_attribute('data-theme', 'night')
    expect(panel.locator('#theme-night')).to_be_checked()
    assert _is_dark(_ground(panel)), _ground(panel)


def test_the_choice_crosses_the_three_surfaces(panel, lab_server):
    """One origin, one key: the Laboratory's choice is in force on the others."""
    _choose(panel, 'night')

    for path in ('/inspector', '/leaderboard'):
        panel.goto(f'{lab_server}{path}')
        expect(panel.locator('html')).to_have_attribute('data-theme', 'night')
        expect(panel.locator('#theme-night')).to_be_checked()
        assert _is_dark(_ground(panel)), f'{path}: {_ground(panel)}'


def test_the_three_radios_are_one_control_with_one_choice_at_a_time(board):
    """The board wears the same control, and it is a radio group, not three switches."""
    for name, stored in (('night', 'night'), ('day', 'day'), ('auto', None)):
        _choose(board, name)
        assert _stored_theme(board) == stored
        checked = board.locator('#theme-control input:checked')
        expect(checked).to_have_count(1)
        expect(checked).to_have_id(f'theme-{name}')


def test_row_height_is_the_readers_to_set_and_it_survives_a_reload(dataset_page):
    """The second setting, stored the way the first one is.

    Compact is the absence of a stored value, exactly as Auto is for the theme:
    the sheet's own `:root` is the compact one, and the attribute only ever
    names the departure from it. Asserted through the padding a cell actually
    computes rather than through the attribute alone — the attribute is a name,
    and the row height is the thing the reader asked for.
    """
    def padding():
        # The page draws its tables from one fetch after load, so the cell has
        # to be waited for on the first read and again after the reload.
        dataset_page.wait_for_selector('.data-table tbody td')
        return dataset_page.evaluate(
            "() => getComputedStyle(document.querySelector"
            "('.data-table tbody td')).paddingTop")

    def choose(name):
        _open_settings(dataset_page)
        dataset_page.click(f'#density-control label[for="density-{name}"]')
        expect(dataset_page.locator(f'#density-{name}')).to_be_checked()

    def stored():
        return dataset_page.evaluate(
            "() => localStorage.getItem('raglab-density')")

    tight = padding()
    choose('comfortable')
    expect(dataset_page.locator('html')).to_have_attribute(
        'data-density', 'comfortable')
    assert stored() == 'comfortable'
    roomy = padding()
    assert float(roomy[:-2]) > float(tight[:-2]), (roomy, tight)

    dataset_page.reload()
    expect(dataset_page.locator('#density-comfortable')).to_be_checked()
    assert padding() == roomy

    choose('compact')
    expect(dataset_page.locator('html')).not_to_have_attribute(
        'data-density', 'comfortable')
    assert stored() is None, 'compact is the absence of a stored value'
    assert padding() == tight
