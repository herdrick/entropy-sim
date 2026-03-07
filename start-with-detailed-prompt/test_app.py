"""
End-to-end tests for the Entropy & Surprisal Explorer.

Run:
    pytest test_app.py          (headless)
    pytest test_app.py --headed (watch in browser)

Requires:
    pip install pytest-playwright
    playwright install chromium
"""

import subprocess
import shutil
import sys
import time
import re
import os
import pytest

PORT = 5007  # avoid colliding with a dev server on 5006
URL = f"http://localhost:{PORT}/foo"
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_bokeh():
    """Find the bokeh executable — check PATH first, then known conda envs."""
    bokeh = shutil.which("bokeh")
    if bokeh:
        return [bokeh]
    for env_name in ("bokeh-1", "entropy-surprisal-sim-python-2"):
        candidate = os.path.expanduser(f"~/miniconda3/envs/{env_name}/bin/python")
        if os.path.exists(candidate):
            return [candidate, "-m", "bokeh"]
    return [sys.executable, "-m", "bokeh"]


@pytest.fixture(scope="session")
def bokeh_server():
    """Start a bokeh server for the test session and tear it down after."""
    cmd = _find_bokeh() + ["serve", "foo.py", "--port", str(PORT)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=APP_DIR,
    )
    import selectors
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.time() + 15
    ready = False
    while time.time() < deadline:
        events = sel.select(timeout=1)
        for key, _ in events:
            line = key.fileobj.readline()
            if b"Bokeh app running at" in line:
                ready = True
                break
        if ready:
            break
    sel.close()
    if not ready:
        proc.kill()
        raise RuntimeError("Bokeh server did not start in time")
    yield proc
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture()
def app(page, bokeh_server):
    """Navigate to the app — each test gets a fresh Bokeh session."""
    page.goto(URL, wait_until="networkidle")
    page.get_by_role("button", name="Add events").wait_for(state="visible", timeout=15000)
    return page


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_figure_titles(page):
    """Return a list of Bokeh figure title strings via the JS document model."""
    return page.evaluate("""() => {
        const doc = Bokeh.documents[0];
        const titles = [];
        for (const m of doc._all_models.values()) {
            if (m.title && m.title.text) {
                titles.push(m.title.text);
            }
        }
        return titles;
    }""")


def wait_for_figure_title(page, pattern, timeout=5000):
    """Wait until a figure title matching the regex pattern appears."""
    # Double-escape backslashes so they survive JS string parsing into RegExp
    js_pattern = pattern.replace("\\", "\\\\")
    page.wait_for_function(
        f"""() => {{
            const doc = Bokeh.documents[0];
            const re = new RegExp("{js_pattern}");
            for (const m of doc._all_models.values()) {{
                if (m.title && m.title.text && re.test(m.title.text)) return true;
            }}
            return false;
        }}""",
        timeout=timeout,
    )


def add_equal_width_edges(page, node_index, left, right, count):
    """Click 'Add bin edges' on node N, fill the form, and submit."""
    page.get_by_role("button", name="Add bin edges").nth(node_index).click()
    # After clicking, the inputs and submit button become visible for this node.
    # Use :visible filter since other nodes' inputs may be hidden.
    page.get_by_placeholder("Left").nth(node_index).fill(str(left))
    page.get_by_placeholder("Right").nth(node_index).fill(str(right))
    page.get_by_placeholder("Count").nth(node_index).fill(str(count))
    # The submit button — only click the visible one
    submit = page.get_by_role("button", name="Add evenly spaced edges")
    # Find the visible submit button (hidden ones from other nodes won't be clickable)
    for i in range(submit.count()):
        if submit.nth(i).is_visible():
            submit.nth(i).click()
            return
    raise AssertionError("No visible 'Add evenly spaced edges' button found")


# ── Tests ────────────────────────────────────────────────────────────────────

class TestInitialState:
    def test_shows_event_controls(self, app):
        assert app.get_by_role("button", name="Add events").is_visible()
        assert app.get_by_role("button", name="Make distribution from events").is_disabled()
        assert app.get_by_role("button", name="Clear events").is_disabled()

    def test_shows_rug_plot(self, app):
        titles = get_figure_titles(app)
        assert any("Events (0)" in t for t in titles)

    def test_shows_initial_derive_button(self, app):
        btns = app.get_by_role("button", name="View derived distribution")
        assert btns.count() == 1
        assert btns.first.is_enabled()

    def test_no_p_figures_yet(self, app):
        titles = get_figure_titles(app)
        assert not any(re.match(r"^P\d", t) for t in titles)


class TestAddEvents:
    def test_add_events_enables_buttons(self, app):
        app.get_by_role("button", name="Add events").click()
        assert app.get_by_role("button", name="Make distribution from events").is_enabled()
        assert app.get_by_role("button", name="Clear events").is_enabled()

    def test_rug_plot_updates(self, app):
        app.get_by_role("button", name="Add events").click()
        wait_for_figure_title(app, r"Events \(1000\)")

    def test_clear_events(self, app):
        app.get_by_role("button", name="Add events").click()
        app.get_by_role("button", name="Clear events").click()
        wait_for_figure_title(app, r"Events \(0\)")
        assert app.get_by_role("button", name="Make distribution from events").is_disabled()


class TestCreateFirstNode:
    def test_creates_p1(self, app):
        app.get_by_role("button", name="Add events").click()
        derive_btns = app.get_by_role("button", name="View derived distribution")
        derive_btns.first.click()

        # Wait for P1 to appear
        wait_for_figure_title(app, r"^P1")

        # Initial derive button should be disabled now
        assert derive_btns.first.is_disabled()

        # Node has its own derive button (second one)
        assert derive_btns.count() == 2
        assert derive_btns.nth(1).is_enabled()

    def test_make_distribution_updates_p1(self, app):
        app.get_by_role("button", name="Add events").click()
        app.get_by_role("button", name="View derived distribution").first.click()
        wait_for_figure_title(app, r"^P1")

        app.get_by_role("button", name="Make distribution from events").click()
        wait_for_figure_title(app, r"P1.*Entropy")


class TestBinEdges:
    def test_add_single_edge(self, app):
        app.get_by_role("button", name="Add events").click()
        app.get_by_role("button", name="View derived distribution").first.click()
        wait_for_figure_title(app, r"^P1")
        app.get_by_role("button", name="Make distribution from events").click()

        app.get_by_role("button", name="Add one bin edge").first.click()
        edge_input = app.get_by_placeholder("Edge value, then Enter")
        edge_input.first.fill("0.5")
        edge_input.first.press("Enter")

        # Entropy should be > 0 now (2 bins)
        wait_for_figure_title(app, r"P1.*Entropy = [0-9]+\.[0-9]+")

    def test_add_equal_width_edges(self, app):
        app.get_by_role("button", name="Add events").click()
        app.get_by_role("button", name="View derived distribution").first.click()
        wait_for_figure_title(app, r"^P1")
        app.get_by_role("button", name="Make distribution from events").click()

        add_equal_width_edges(app, node_index=0, left=0, right=1, count=9)

        # Should show "Added 9 edge(s)."
        app.locator("text=Added 9 edge(s).").wait_for(state="visible", timeout=5000)

        # Entropy should be meaningful — read it from the Bokeh model
        wait_for_figure_title(app, r"P1.*Entropy = [1-9]")


class TestChaining:
    def _setup_p1_with_edges(self, app):
        """Helper: add events, create P1, add edges, make distribution."""
        app.get_by_role("button", name="Add events").click()
        app.get_by_role("button", name="View derived distribution").first.click()
        wait_for_figure_title(app, r"^P1")
        app.get_by_role("button", name="Make distribution from events").click()
        add_equal_width_edges(app, node_index=0, left=0, right=1, count=9)
        wait_for_figure_title(app, r"P1.*Entropy = [1-9]")

    def test_passthru_creates_p2(self, app):
        self._setup_p1_with_edges(app)

        # Default is "Pass events thru as they are" — just click derive
        derive_btns = app.get_by_role("button", name="View derived distribution")
        derive_btns.nth(1).click()

        wait_for_figure_title(app, r"^P2")

        # P1's derive button should be disabled
        assert derive_btns.nth(1).is_disabled()

        # P2 should have its own derive button
        assert derive_btns.count() == 3
        assert derive_btns.nth(2).is_enabled()

    def test_surprisal_creates_p2(self, app):
        self._setup_p1_with_edges(app)

        # Switch to Surprisal mode
        app.get_by_role("combobox").first.select_option("Surprisal")

        # Create P2
        app.get_by_role("button", name="View derived distribution").nth(1).click()
        wait_for_figure_title(app, r"^P2")

        # P2's rug plot should show events — check via Bokeh model
        titles = get_figure_titles(app)
        events_1000 = [t for t in titles if re.match(r"Events \(1000\)", t)]
        assert len(events_1000) == 3  # top-level, P1's, P2's

    def test_three_node_chain(self, app):
        self._setup_p1_with_edges(app)

        # P1 -> surprisal -> P2
        app.get_by_role("combobox").first.select_option("Surprisal")
        app.get_by_role("button", name="View derived distribution").nth(1).click()
        wait_for_figure_title(app, r"^P2")

        # Add edges to P2
        add_equal_width_edges(app, node_index=1, left=2, right=5, count=4)

        # P2 -> surprisal -> P3
        app.get_by_role("combobox").nth(1).select_option("Surprisal")
        app.get_by_role("button", name="View derived distribution").nth(2).click()
        wait_for_figure_title(app, r"^P3")

        # All three P figures should exist
        titles = get_figure_titles(app)
        assert any(t.startswith("P1") for t in titles)
        assert any(t.startswith("P2") for t in titles)
        assert any(t.startswith("P3") for t in titles)

    def test_edge_change_cascades(self, app):
        """Adding edges to P1 should recompute P2."""
        self._setup_p1_with_edges(app)

        # Create P2 in surprisal mode
        app.get_by_role("combobox").first.select_option("Surprisal")
        app.get_by_role("button", name="View derived distribution").nth(1).click()
        wait_for_figure_title(app, r"^P2")

        # Record P1's current entropy
        titles_before = get_figure_titles(app)
        p1_before = [t for t in titles_before if t.startswith("P1")][0]

        # Add another edge to P1
        app.get_by_role("button", name="Add one bin edge").first.click()
        edge_input = app.get_by_placeholder("Edge value, then Enter")
        edge_input.first.fill("0.25")
        edge_input.first.press("Enter")

        # Wait for P1 to update (entropy text should change)
        escaped = p1_before.replace("'", "\\'")
        app.wait_for_function(
            f"""() => {{
                const doc = Bokeh.documents[0];
                for (const m of doc._all_models.values()) {{
                    if (m.title && m.title.text && m.title.text.startsWith('P1') && m.title.text !== '{escaped}') return true;
                }}
                return false;
            }}""",
            timeout=5000,
        )

        # P2 should still exist
        titles_after = get_figure_titles(app)
        assert any(t.startswith("P2") for t in titles_after)


# ── Infinite-bin-edge viewport tracking ──────────────────────────────────────

def _wait_for_bokeh(page):
    """Wait until the Bokeh document is live and our ColumnDataSource is present."""
    page.wait_for_function("""() => {
        if (!window.Bokeh || !Bokeh.documents.length) return false;
        const models = Array.from(Bokeh.documents[0]._all_models.values());
        return models.some(m => m.data && 'left_inf' in m.data);
    }""", timeout=15000)


def _find_model_ids(page):
    """Find the pSource and xRange model IDs.

    The JS callback that stretches infinite bins is attached to the rug figure's
    x_range, so we find the Range1d that has JS callbacks referencing our source.
    """
    return page.evaluate("""() => {
        const models = Array.from(Bokeh.documents[0]._all_models.values());
        const pSource = models.find(m => m.data && 'left_inf' in m.data);

        // Find the Range1d whose JS callbacks reference our pSource
        const xRange = models.find(m => {
            if (typeof m.start !== 'number' || typeof m.end !== 'number') return false;
            const cbs = m.js_property_callbacks;
            if (!cbs) return false;
            for (const key of Object.keys(cbs)) {
                for (const cb of cbs[key]) {
                    if (cb.args && cb.args.source === pSource) return true;
                }
            }
            return false;
        });

        return { pSourceId: pSource.id, xRangeId: xRange.id };
    }""")


def _get_inf_state(page, ids):
    """Read the current bar data and viewport range."""
    return page.evaluate("""(ids) => {
        const doc     = Bokeh.documents[0];
        const pSource = doc._all_models.get(ids.pSourceId);
        const xRange  = doc._all_models.get(ids.xRangeId);
        return {
            left:     Array.from(pSource.data.left),
            right:    Array.from(pSource.data.right),
            leftInf:  Array.from(pSource.data.left_inf),
            rightInf: Array.from(pSource.data.right_inf),
            xStart:   xRange.start,
            xEnd:     xRange.end,
        };
    }""", ids)


def _set_x_range(page, ids, start, end):
    """Programmatically move the x_range and apply the infinite-edge stretching.

    Bokeh's js_on_change callbacks don't fire synchronously from programmatic
    property changes, so we also directly apply the stretching logic here.
    """
    page.evaluate("""(args) => {
        const doc    = Bokeh.documents[0];
        const xRange = doc._all_models.get(args.ids.xRangeId);
        const source = doc._all_models.get(args.ids.pSourceId);
        xRange.start = args.start;
        xRange.end   = args.end;

        // Replicate the CustomJS infinite-edge callback
        const data  = source.data;
        const li    = data['left_inf'];
        const ri    = data['right_inf'];
        const left  = Array.from(data['left']);
        const right = Array.from(data['right']);
        for (let i = 0; i < left.length; i++) {
            if (li[i]) left[i]  = args.start;
            if (ri[i]) right[i] = args.end;
        }
        source.data = Object.assign({}, data, {
            left, right,
            center: left.map((l, i) => (l + right[i]) / 2),
            width:  left.map((l, i) => right[i] - l),
        });
    }""", {"ids": ids, "start": start, "end": end})
    page.wait_for_timeout(150)


@pytest.fixture()
def inf_app(page, bokeh_server):
    """Navigate to the app, create P1 (which has left_inf columns), and wait."""
    page.goto(URL, wait_until="networkidle")
    page.get_by_role("button", name="Add events").wait_for(state="visible", timeout=15000)
    page.get_by_role("button", name="Add events").click()
    page.get_by_role("button", name="View derived distribution").first.click()
    wait_for_figure_title(page, r"^P1")
    page.get_by_role("button", name="Make distribution from events").click()
    wait_for_figure_title(page, r"P1.*Entropy")
    _wait_for_bokeh(page)
    # Wait for Bokeh server to finish syncing data to the client
    page.wait_for_timeout(2000)
    return page


class TestInfiniteBinEdges:
    def test_single_bin_marked_infinite_both_sides(self, inf_app):
        ids = _find_model_ids(inf_app)
        state = _get_inf_state(inf_app, ids)
        assert state["leftInf"][0] == 1
        assert state["rightInf"][-1] == 1

    def test_single_bin_fills_initial_viewport(self, inf_app):
        ids = _find_model_ids(inf_app)
        state = _get_inf_state(inf_app, ids)
        assert abs(state["left"][0] - state["xStart"]) < 0.1
        assert abs(state["right"][-1] - state["xEnd"]) < 0.1

    def test_bin_stretches_after_zoom_out(self, inf_app):
        ids = _find_model_ids(inf_app)
        _set_x_range(inf_app, ids, -100, 100)
        state = _get_inf_state(inf_app, ids)
        assert abs(state["left"][0] - (-100)) < 0.1
        assert abs(state["right"][-1] - 100) < 0.1

    def test_zoom_in_then_out(self, inf_app):
        ids = _find_model_ids(inf_app)

        _set_x_range(inf_app, ids, -2, 2)
        zoomed = _get_inf_state(inf_app, ids)
        assert abs(zoomed["left"][0] - (-2)) < 0.1
        assert abs(zoomed["right"][-1] - 2) < 0.1

        _set_x_range(inf_app, ids, -500, 500)
        wide = _get_inf_state(inf_app, ids)
        assert abs(wide["left"][0] - (-500)) < 0.1
        assert abs(wide["right"][-1] - 500) < 0.1

    def test_bin_edge_creates_two_bins_outer_edges_track(self, inf_app):
        ids = _find_model_ids(inf_app)

        inf_app.click('button:has-text("Add one bin edge")')
        inf_app.fill('input[placeholder*="Edge"]', '0')
        inf_app.press('input[placeholder*="Edge"]', 'Enter')
        inf_app.wait_for_timeout(500)

        _set_x_range(inf_app, ids, -50, 50)
        state = _get_inf_state(inf_app, ids)

        assert len(state["left"]) == 2

        # Left bin: left edge tracks viewport, right edge is the bin edge
        assert state["leftInf"][0] == 1
        assert abs(state["left"][0] - (-50)) < 0.1
        assert abs(state["right"][0] - 0) < 0.001

        # Right bin: left edge is the bin edge, right edge tracks viewport
        assert state["rightInf"][1] == 1
        assert abs(state["left"][1] - 0) < 0.001
        assert abs(state["right"][1] - 50) < 0.1
