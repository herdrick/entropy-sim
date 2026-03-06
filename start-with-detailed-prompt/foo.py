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
root_events = np.array([], dtype=float)   # accumulated raw events
interior_edges = []                          # interior bin edges (sorted)

# ── Helpers ──────────────────────────────────────────────────────────────────

def bin_edges():
    """Full edge array: -inf, interior edges (sorted), +inf."""
    return np.array([-np.inf] + sorted(interior_edges) + [np.inf])


def compute_probabilities(edges, event_arr):
    """
    Return probabilities. Uses LaPlace smoothing.
    Outer edges may be -inf / +inf; counting uses searchsorted on interior edges.
    """
    n_bins = len(edges) - 1
    interior = edges[1:-1]  # finite interior edges only
    if len(event_arr) > 0:
        indices = np.searchsorted(interior, event_arr)
        counts = np.bincount(indices, minlength=n_bins).astype(float)
    else:
        counts = np.zeros(n_bins)
    smoothed = np.where(counts == 0, LAPLACE_ALPHA, counts)  # Laplace smoothing: only empty bins
    return smoothed / smoothed.sum()


def make_column_data_source_data(edges, probs, x_start=X_MIN, x_end=X_MAX):
    """
    Build the dict for p_column_data_source.  Infinite outer edges are clipped to
    x_start/x_end for the initial render; left_inf/right_inf masks let the
    JS range callback extend them to the live viewport on every pan/zoom.
    """
    lefts = edges[:-1]
    rights = edges[1:]
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
probs0 = compute_probabilities(edges0, root_events)
p_column_data_source = ColumnDataSource(make_column_data_source_data(edges0, probs0))

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
    source=p_column_data_source,
    fill_color="color", line_color="white", alpha=0.8,
)
p_fig.xaxis.axis_label = "Value"
p_fig.yaxis.axis_label = "Probability"

# JS callback: whenever the shared x_range changes, stretch infinite-edge bars
# to fill the current viewport so they always look like they extend to ±∞.
_range_callback = CustomJS(args=dict(source=p_column_data_source, x_range=rug_fig.x_range), code="""
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
rug_fig.x_range.js_on_change('start', _range_callback)
rug_fig.x_range.js_on_change('end',   _range_callback)

# ── Update helpers ────────────────────────────────────────────────────────────

def refresh_p(event_arr):
    """Recompute and redraw the P distribution."""
    edges = bin_edges()
    probs = compute_probabilities(edges, event_arr)
    p_column_data_source.data = make_column_data_source_data(
        edges, probs,
        x_start=rug_fig.x_range.start,
        x_end=rug_fig.x_range.end,
    )
    p_fig.title.text = f"P  |  Entropy = {entropy_bits(probs):.4f} bits"


def refresh_rug():
    """Update the top rug figure from root_events."""
    rug_source.data = dict(x=root_events, y=np.zeros(len(root_events)))
    rug_fig.title.text = f"Events ({len(root_events)})"
    has_events = len(root_events) > 0
    make_dist_btn.disabled = not has_events
    clear_events_btn.disabled = not has_events


# ── Controls ─────────────────────────────────────────────────────────────────

# Number of events to generate
n_events_input = TextInput(value="1000", title="", width=80)

add_events_btn = Button(label="Add events", button_type="success", width=120)
make_dist_btn = Button(label="Make distribution from events", button_type="primary", width=240, disabled=True)
clear_events_btn = Button(label="Clear events", button_type="warning", width=120, disabled=True)

divide_bin_btn = Button(label="Add one bin edge", button_type="default", width=120)
edge_input = TextInput(
    placeholder="Edge value, then Enter",
    width=220,
    visible=False,
)
edge_status = Div(text="", width=300, styles={"color": "red", "font-size": "13px"})

equal_width_btn = Button(label="Add bin edges", button_type="default", width=120)
equal_width_left_input = TextInput(placeholder="Left", width=80, visible=False)
equal_width_right_input = TextInput(placeholder="Right", width=80, visible=False)
equal_width_count_input = TextInput(placeholder="Count", width=80, visible=False)
equal_width_submit_btn = Button(label="Add evenly spaced edges", button_type="success", width=200, visible=False)
equal_width_preview = Div(text="", width=200, styles={"font-size": "13px", "line-height": "2.2"})
equal_width_status = Div(text="", width=300, styles={"color": "red", "font-size": "13px"})

# ── Callbacks ─────────────────────────────────────────────────────────────────

def on_equal_width_count_change(attr, old, new):
    try:
        count = int(new)
        if count < 1:
            raise ValueError
    except ValueError:
        equal_width_preview.text = ""
        return
    # new edges that don't already exist
    try:
        left = float(equal_width_left_input.value)
        right = float(equal_width_right_input.value)
        step = (right - left) / (count + 1)
        new_edges = [left + step * (i + 1) for i in range(count)]
        new_unique = [e for e in new_edges if e not in interior_edges]
    except (ValueError, ZeroDivisionError):
        new_unique = list(range(count))  # assume all are new if left/right not set yet
    total_bins = len(interior_edges) + len(new_unique) + 1  # +1: n edges = n+1 bins
    equal_width_preview.text = f"→ {total_bins} bins total"

def on_add_events():
    global root_events
    try:
        n = int(n_events_input.value)
        if n <= 0:
            raise ValueError
    except ValueError:
        n = 1000
        n_events_input.value = "1000"
    new_ev = ev.get_events(n)
    root_events = np.concatenate([root_events, new_ev])
    refresh_rug()


def on_make_dist():
    # Build new P from current events; old P is replaced
    refresh_p(event_arr=root_events)


def on_clear_events():
    global root_events
    root_events = np.array([], dtype=float)
    rug_source.data = dict(x=[], y=[])
    rug_fig.title.text = "Events (0)"
    make_dist_btn.disabled = True
    clear_events_btn.disabled = True
    # Note: do NOT change P or its rug overlay


def on_divide_bin():
    edge_input.visible = not edge_input.visible
    edge_status.text = ""


def on_edge_input(attr, old, new):
    global interior_edges
    val_str = new.strip()
    if not val_str:
        return
    try:
        val = float(val_str)
    except ValueError:
        edge_status.text = f"'{val_str}' is not a valid number."
        edge_input.value = ""
        return
    if val in interior_edges:
        edge_status.text = f"{val} is already a bin edge."
        edge_input.value = ""
        return
    interior_edges.append(val)
    edge_status.text = f"Added bin edge at {val}."
    edge_input.value = ""
    edge_input.visible = False
    # Recompute P — keep using same events if any
    refresh_p(event_arr=root_events)


def on_equal_width_toggle():
    vis = not equal_width_left_input.visible
    equal_width_left_input.visible = vis
    equal_width_right_input.visible = vis
    equal_width_count_input.visible = vis
    equal_width_submit_btn.visible = vis
    equal_width_status.text = ""
    equal_width_preview.text = ""


def on_equal_width_submit():
    global interior_edges
    try:
        left = float(equal_width_left_input.value)
        right = float(equal_width_right_input.value)
        count = int(equal_width_count_input.value)
    except ValueError:
        equal_width_status.text = "Enter valid numbers for left, right, and count."
        return
    if right <= left:
        equal_width_status.text = "Right must be greater than left."
        return
    if count < 1:
        equal_width_status.text = "Count must be at least 1."
        return
    step = (right - left) / (count + 1)
    new_edges = [left + step * (i + 1) for i in range(count)]
    added = [e for e in new_edges if e not in interior_edges]
    interior_edges.extend(added)
    equal_width_status.text = f"Added {len(added)} edge(s)."
    equal_width_preview.text = ""
    equal_width_left_input.visible = False
    equal_width_right_input.visible = False
    equal_width_count_input.visible = False
    equal_width_submit_btn.visible = False
    refresh_p(event_arr=root_events)


add_events_btn.on_click(on_add_events)
make_dist_btn.on_click(on_make_dist)
clear_events_btn.on_click(on_clear_events)
divide_bin_btn.on_click(on_divide_bin)
divide_bin_btn.js_on_click(CustomJS(args=dict(inp=edge_input), code="""
    setTimeout(() => {
        const el = inp.el?.querySelector?.('input');
        if (el) el.focus();
    }, 100);
"""))
edge_input.on_change("value", on_edge_input)
equal_width_btn.on_click(on_equal_width_toggle)
equal_width_count_input.on_change("value_input", on_equal_width_count_change)
equal_width_submit_btn.on_click(on_equal_width_submit)

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

divide_row = Row(divide_bin_btn, edge_input, edge_status)
equal_width_row = Row(equal_width_btn, equal_width_left_input, equal_width_right_input, equal_width_count_input, equal_width_submit_btn, equal_width_preview, equal_width_status)

root = Column(
    top_controls,
    rug_fig,
    p_fig,
    divide_row,
    equal_width_row,
)

curdoc().add_root(root)
curdoc().title = "Entropy & Surprisal Explorer"
