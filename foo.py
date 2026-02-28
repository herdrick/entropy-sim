import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, Div, TextInput, Button, Row, Column, Spacer
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
    """Full edge array including domain boundaries."""
    return np.array([X_MIN] + sorted(fenceposts) + [X_MAX])


def compute_probs(edges, event_arr):
    """
    Return (lefts, rights, probs) for a histogram with LaPlace smoothing.
    If event_arr is None, all bins get equal weight (pure LaPlace).
    """
    n_bins = len(edges) - 1
    #counts = np.zeros(n_bins)
    raw, _ = np.histogram(event_arr, bins=edges)
    counts = raw.astype(float)
    # LaPlace smoothing
    smoothed = counts + LAPLACE_ALPHA
    probs = smoothed / smoothed.sum()
    lefts = edges[:-1]
    rights = edges[1:]
    return lefts, rights, probs


def entropy_bits(probs):
    p = np.array(probs)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def bar_colors(n):
    return ["#4878CF"] * n


# ── Initial distribution ──────────────────────────────────────────────────────
edges0 = bin_edges()
lefts0, rights0, probs0 = compute_probs(edges0, all_events)
widths0 = rights0 - lefts0
centers0 = (lefts0 + rights0) / 2

# ── Data sources ─────────────────────────────────────────────────────────────

# Rug plot (top figure) — raw events
rug_source = ColumnDataSource(dict(x=[], y=[]))

# P bar chart
p_source = ColumnDataSource(dict(
    left=lefts0, right=rights0, top=probs0,
    center=centers0, width=widths0,
    color=bar_colors(len(probs0)),
))

# Rug overlay on P figure (events shown after "make distribution from events")
p_rug_source = ColumnDataSource(dict(x=[], y=[]))

# ── Figures ──────────────────────────────────────────────────────────────────

TOOLS = "pan,xwheel_zoom,xbox_zoom,reset,save"
PLOT_WIDTH = 900

# Top rug figure
rug_fig = figure(
    width=PLOT_WIDTH, height=80,
    x_range=(X_MIN, X_MAX), y_range=(-0.5, 0.5),
    tools=TOOLS, toolbar_location="right",
    title="Events (rug plot)",
)
rug_fig.yaxis.visible = False
rug_fig.ygrid.visible = False
rug_fig.segment(
    x0="x", y0=-0.4, x1="x", y1=0.4,
    source=rug_source,
    line_color="#e07b39", line_width=1,
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
# Rug overlay inside P fig (shows events after "make distribution")
p_fig.segment(
    x0="x", y0=0, x1="x", y1="y",
    source=p_rug_source,
    line_color="#e07b39", line_width=1, alpha=0.6,
)
p_fig.xaxis.axis_label = "Value"
p_fig.yaxis.axis_label = "Probability"

# ── Update helpers ────────────────────────────────────────────────────────────

def refresh_p(event_arr=None):
    """Recompute and redraw the P distribution."""
    edges = bin_edges()
    lefts, rights, probs = compute_probs(edges, event_arr)
    widths = rights - lefts
    centers = (lefts + rights) / 2
    p_source.data = dict(
        left=lefts, right=rights, top=probs,
        center=centers, width=widths,
        color=bar_colors(len(probs)),
    )
    p_fig.title.text = f"P  |  Entropy = {entropy_bits(probs):.4f} bits"


def refresh_rug():
    """Update the top rug figure from all_events."""
    rug_source.data = dict(x=all_events, y=np.zeros(len(all_events)))


# ── Controls ─────────────────────────────────────────────────────────────────

# Number of events to generate
n_events_input = TextInput(value="1000", title="", width=80)

add_events_btn = Button(label="Add events", button_type="success", width=120)
make_dist_btn = Button(label="Make distribution from events", button_type="primary", width=240)
clear_events_btn = Button(label="Clear events", button_type="warning", width=120)

divide_bin_btn = Button(label="Divide a bin", button_type="default", width=130)
fencepost_input = TextInput(
    placeholder="Fencepost value, then Enter",
    width=220,
    visible=False,
)
fencepost_status = Div(text="", width=300, styles={"color": "red", "font-size": "13px"})

# ── Callbacks ─────────────────────────────────────────────────────────────────

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
    # Show events as rug overlay on P figure
    if len(all_events) > 0:
        rug_h = p_fig.y_range.end * 0.05  # 5 % of chart height
        p_rug_source.data = dict(
            x=all_events,
            y=np.full(len(all_events), rug_h if rug_h > 0 else 0.01),
        )
    else:
        p_rug_source.data = dict(x=[], y=[])


def cb_clear_events():
    global all_events
    all_events = np.array([], dtype=float)
    rug_source.data = dict(x=[], y=[])
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
    if val <= X_MIN or val >= X_MAX:
        fencepost_status.text = f"Fencepost must be in ({X_MIN}, {X_MAX})."
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
    refresh_p(event_arr=all_events if len(all_events) > 0 else None)
    # Refresh P's rug overlay if it was showing
    if len(p_rug_source.data["x"]) > 0:
        cb_make_dist()


add_events_btn.on_click(cb_add_events)
make_dist_btn.on_click(cb_make_dist)
clear_events_btn.on_click(cb_clear_events)
divide_bin_btn.on_click(cb_divide_bin)
fencepost_input.on_change("value", cb_fencepost)

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

root = Column(
    top_controls,
    rug_fig,
    p_fig,
    divide_row,
)

curdoc().add_root(root)
curdoc().title = "Entropy & Surprisal Explorer"
