import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, CustomJS, Div, TextInput, Button, Row, Column, Spacer
)
from bokeh.layouts import layout
import events as ev

# ── Constants ────────────────────────────────────────────────────────────────
X_MIN, X_MAX = -10, 10
LAPLACE_ALPHA = 1  # pseudocount per bin

# ── State ────────────────────────────────────────────────────────────────────
all_events = np.array([], dtype=float)   # accumulated raw events
fenceposts = []                          # sorted list of interior bin edges

# ── Helpers ──────────────────────────────────────────────────────────────────

def bin_edges():
    """Full edge array: -inf, interior fenceposts (sorted), +inf."""
    return np.array([-np.inf] + sorted(fenceposts) + [np.inf])


def compute_probs(edges, event_arr):
    """
    Return (lefts, rights, probs) for a histogram with LaPlace smoothing.
    Outer edges may be -inf / +inf; counting uses searchsorted on interior edges.
    """
    n_bins = len(edges) - 1
    interior = edges[1:-1]  # finite fenceposts only
    if len(event_arr) > 0:
        indices = np.searchsorted(interior, event_arr)
        counts = np.bincount(indices, minlength=n_bins).astype(float)
    else:
        counts = np.zeros(n_bins)
    smoothed = np.where(counts == 0, LAPLACE_ALPHA, counts)  # Laplace smoothing: only empty bins
    probs = smoothed / smoothed.sum()
    lefts = edges[:-1]
    rights = edges[1:]
    return lefts, rights, probs


def make_source_data(lefts, rights, probs, x_start=X_MIN, x_end=X_MAX):
    """
    Build the dict for p_source.  Infinite outer edges are clipped to
    x_start/x_end for the initial render; left_inf/right_inf masks let the
    JS range callback extend them to the live viewport on every pan/zoom.
    """
    left_inf  = np.isneginf(lefts).astype(int)
    right_inf = np.isposinf(rights).astype(int)
    dl = np.where(left_inf,  x_start, lefts)
    dr = np.where(right_inf, x_end,   rights)
    return dict(
        left=dl, right=dr, top=probs,
        center=(dl + dr) / 2, width=dr - dl,
        color=bar_colors(len(probs)),
        left_inf=left_inf, right_inf=right_inf,
    )


def entropy_bits(probs):
    p = np.array(probs)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def bar_colors(n):
    return ["#4878CF"] * n


# ── Data sources ─────────────────────────────────────────────────────────────

# Rug plot (top figure) — raw events
rug_source = ColumnDataSource(dict(x=[], y=[]))

# P bar chart — seeded with the single uniform bin
edges0 = bin_edges()
lefts0, rights0, probs0 = compute_probs(edges0, all_events)
p_source = ColumnDataSource(make_source_data(lefts0, rights0, probs0))

# ── Figures ──────────────────────────────────────────────────────────────────

TOOLS = "pan,xwheel_zoom,xbox_zoom,reset,save"
PLOT_WIDTH = 900

# Top rug figure
rug_fig = figure(
    width=PLOT_WIDTH, height=80,
    x_range=(X_MIN, X_MAX), y_range=(-0.5, 0.5),
    tools=TOOLS, toolbar_location="right",
    title="Events (0)",
)
rug_fig.yaxis.visible = False
rug_fig.ygrid.visible = False
rug_fig.segment(
    x0="x", y0=-0.4, x1="x", y1=0.4,
    source=rug_source,
    line_color="#888888", line_width=1, alpha=0.02,
)

# P distribution figure — shares x_range with rug so zoom is linked
p_fig = figure(
    width=PLOT_WIDTH, height=380,
    x_range=rug_fig.x_range,
    tools=TOOLS, toolbar_location="right",
    title=f"P  |  Entropy = {entropy_bits(probs0):.4f} bits",
)
p_fig.quad(
    left="left", right="right", top="top", bottom=0,
    source=p_source,
    fill_color="color", line_color="white", alpha=0.8,
)
p_fig.xaxis.axis_label = "Value"
p_fig.yaxis.axis_label = "Probability"

# JS callback: whenever the shared x_range changes, stretch infinite-edge bars
# to fill the current viewport so they always look like they extend to ±∞.
_range_cb = CustomJS(args=dict(source=p_source, x_range=rug_fig.x_range), code="""
    const data  = source.data;
    const li    = data['left_inf'];
    const ri    = data['right_inf'];
    const left  = data['left'].slice();
    const right = data['right'].slice();
    const xstart = x_range.start;
    const xend   = x_range.end;
    for (let i = 0; i < left.length; i++) {
        if (li[i]) left[i]  = xstart;
        if (ri[i]) right[i] = xend;
    }
    const center = left.map((l, i) => (l + right[i]) / 2);
    const width  = left.map((l, i) => right[i] - l);
    data['left']   = left;
    data['right']  = right;
    data['center'] = center;
    data['width']  = width;
    source.change.emit();
""")
rug_fig.x_range.js_on_change('start', _range_cb)
rug_fig.x_range.js_on_change('end',   _range_cb)

# ── Update helpers ────────────────────────────────────────────────────────────

def refresh_p(event_arr):
    """Recompute and redraw the P distribution."""
    edges = bin_edges()
    lefts, rights, probs = compute_probs(edges, event_arr)
    p_source.data = make_source_data(
        lefts, rights, probs,
        x_start=rug_fig.x_range.start,
        x_end=rug_fig.x_range.end,
    )
    p_fig.title.text = f"P  |  Entropy = {entropy_bits(probs):.4f} bits"


def refresh_rug():
    """Update the top rug figure from all_events."""
    rug_source.data = dict(x=all_events, y=np.zeros(len(all_events)))
    rug_fig.title.text = f"Events ({len(all_events)})"
    has_events = len(all_events) > 0
    make_dist_btn.disabled = not has_events
    clear_events_btn.disabled = not has_events


# ── Controls ─────────────────────────────────────────────────────────────────

# Number of events to generate
n_events_input = TextInput(value="1000", title="", width=80)

add_events_btn = Button(label="Add events", button_type="success", width=120)
make_dist_btn = Button(label="Make distribution from events", button_type="primary", width=240, disabled=True)
clear_events_btn = Button(label="Clear events", button_type="warning", width=120, disabled=True)

divide_bin_btn = Button(label="Add one bin edge", button_type="default", width=120)
fencepost_input = TextInput(
    placeholder="Edge value, then Enter",
    width=220,
    visible=False,
)
fencepost_status = Div(text="", width=300, styles={"color": "red", "font-size": "13px"})

equal_width_btn = Button(label="Add bin edges", button_type="default", width=120)
ew_left_input = TextInput(placeholder="Left", width=80, visible=False)
ew_right_input = TextInput(placeholder="Right", width=80, visible=False)
ew_count_input = TextInput(placeholder="Count", width=80, visible=False)
ew_submit_btn = Button(label="Add evenly spaced edges", button_type="success", width=200, visible=False)
ew_preview = Div(text="", width=200, styles={"font-size": "13px", "line-height": "2.2"})
ew_status = Div(text="", width=300, styles={"color": "red", "font-size": "13px"})

# ── Callbacks ─────────────────────────────────────────────────────────────────

def cb_ew_count_change(attr, old, new):
    try:
        count = int(new)
        if count < 1:
            raise ValueError
    except ValueError:
        ew_preview.text = ""
        return
    # new edges that don't already exist
    try:
        left = float(ew_left_input.value)
        right = float(ew_right_input.value)
        step = (right - left) / (count + 1)
        new_edges = [left + step * (i + 1) for i in range(count)]
        new_unique = [e for e in new_edges if e not in fenceposts]
    except (ValueError, ZeroDivisionError):
        new_unique = list(range(count))  # assume all are new if left/right not set yet
    total_bins = len(fenceposts) + len(new_unique) + 1  # +1: n edges = n+1 bins
    ew_preview.text = f"→ {total_bins} bins total"

def cb_add_events():
    global all_events
    try:
        n = int(n_events_input.value)
        if n <= 0:
            raise ValueError
    except ValueError:
        n = 1000
        n_events_input.value = "1000"
    new_ev = ev.get_events(n)
    all_events = np.concatenate([all_events, new_ev])
    refresh_rug()


def cb_make_dist():
    # Build new P from current events; old P is replaced
    refresh_p(event_arr=all_events)


def cb_clear_events():
    global all_events
    all_events = np.array([], dtype=float)
    rug_source.data = dict(x=[], y=[])
    rug_fig.title.text = "Events (0)"
    make_dist_btn.disabled = True
    clear_events_btn.disabled = True
    # Note: do NOT change P or its rug overlay


def cb_divide_bin():
    fencepost_input.visible = not fencepost_input.visible
    fencepost_status.text = ""


def cb_fencepost(attr, old, new):
    global fenceposts
    val_str = new.strip()
    if not val_str:
        return
    try:
        val = float(val_str)
    except ValueError:
        fencepost_status.text = f"'{val_str}' is not a valid number."
        fencepost_input.value = ""
        return
    if val in fenceposts:
        fencepost_status.text = f"{val} is already a fencepost."
        fencepost_input.value = ""
        return
    fenceposts.append(val)
    fencepost_status.text = f"Added fencepost at {val}."
    fencepost_input.value = ""
    fencepost_input.visible = False
    # Recompute P — keep using same events if any
    refresh_p(event_arr=all_events)


def cb_equal_width_toggle():
    vis = not ew_left_input.visible
    ew_left_input.visible = vis
    ew_right_input.visible = vis
    ew_count_input.visible = vis
    ew_submit_btn.visible = vis
    ew_status.text = ""
    ew_preview.text = ""


def cb_equal_width_submit():
    global fenceposts
    try:
        left = float(ew_left_input.value)
        right = float(ew_right_input.value)
        count = int(ew_count_input.value)
    except ValueError:
        ew_status.text = "Enter valid numbers for left, right, and count."
        return
    if right <= left:
        ew_status.text = "Right must be greater than left."
        return
    if count < 1:
        ew_status.text = "Count must be at least 1."
        return
    step = (right - left) / (count + 1)
    new_edges = [left + step * (i + 1) for i in range(count)]
    added = [e for e in new_edges if e not in fenceposts]
    fenceposts.extend(added)
    ew_status.text = f"Added {len(added)} edge(s)."
    ew_preview.text = ""
    ew_left_input.visible = False
    ew_right_input.visible = False
    ew_count_input.visible = False
    ew_submit_btn.visible = False
    refresh_p(event_arr=all_events)


add_events_btn.on_click(cb_add_events)
make_dist_btn.on_click(cb_make_dist)
clear_events_btn.on_click(cb_clear_events)
divide_bin_btn.on_click(cb_divide_bin)
divide_bin_btn.js_on_click(CustomJS(args=dict(inp=fencepost_input), code="""
    setTimeout(() => {
        const el = inp.el?.querySelector?.('input');
        if (el) el.focus();
    }, 100);
"""))
fencepost_input.on_change("value", cb_fencepost)
equal_width_btn.on_click(cb_equal_width_toggle)
ew_count_input.on_change("value_input", cb_ew_count_change)
ew_submit_btn.on_click(cb_equal_width_submit)

# ── Layout ────────────────────────────────────────────────────────────────────

top_controls = Row(
    add_events_btn,
    Div(text="<b>n =</b>", styles={"line-height": "2.2", "margin-left": "6px"}),
    n_events_input,
    Spacer(width=20),
    make_dist_btn,
    Spacer(width=20),
    clear_events_btn,
)

divide_row = Row(divide_bin_btn, fencepost_input, fencepost_status)
equal_width_row = Row(equal_width_btn, ew_left_input, ew_right_input, ew_count_input, ew_submit_btn, ew_preview, ew_status)

root = Column(
    top_controls,
    rug_fig,
    p_fig,
    divide_row,
    equal_width_row,
)

curdoc().add_root(root)
curdoc().title = "Entropy & Surprisal Explorer"
