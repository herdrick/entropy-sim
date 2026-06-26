import numpy as np
np.set_printoptions(formatter={'float': lambda x: f"{x},"})

from scipy.stats import norm as scipy_norm
from dataclasses import dataclass, field
from typing import Optional
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, CustomJS, Div, TextInput, Button, Row, Column, Spacer, Select,
    RadioGroup, RadioButtonGroup, Slider, HoverTool, Range1d,
)
import events as ev

# ── Constants ────────────────────────────────────────────────────────────────
X_MIN, X_MAX = -10, 10
PRIOR_ALPHA_DEFAULT = 0
PRIOR_MU_DEFAULT = 0
PRIOR_SIGMA_DEFAULT = 5
TOOLS = "xpan,xwheel_zoom,xbox_zoom,reset,save"
PLOT_WIDTH = 900
MAX_ITER = 1000
CONVERGENCE_TOL = 1e-8

# ── State ────────────────────────────────────────────────────────────────────
root_events = np.array([], dtype=float)
all_events: np.ndarray = np.array([], dtype=float)
history_index: int = 0
_transport_cb_guard: bool = False


# ── Helpers (same as main.py) ─────────────────────────────────────────────────

def bin_counts(edges, event_arr):
    n_bins = len(edges) - 1
    interior = edges[1:-1]
    if len(event_arr) > 0:
        indices = np.searchsorted(interior, event_arr)
        return np.bincount(indices, minlength=n_bins).astype(float)
    return np.zeros(n_bins)


def gaussian_prior_mass(edges, mu=PRIOR_MU_DEFAULT, sigma=PRIOR_SIGMA_DEFAULT):
    cdf_vals = scipy_norm.cdf(edges, loc=mu, scale=sigma)
    return np.diff(cdf_vals)


def compute_probabilities(edges, event_arr, alpha=PRIOR_ALPHA_DEFAULT,
                          mu=PRIOR_MU_DEFAULT, sigma=PRIOR_SIGMA_DEFAULT):
    counts = bin_counts(edges, event_arr)
    prior = gaussian_prior_mass(edges, mu, sigma)
    smoothed = counts + alpha * prior
    total = smoothed.sum()
    if total > 0:
        return smoothed / total
    return np.ones(len(counts)) / len(counts)


def make_column_data_source_data(edges, probs, counts=None, x_start=X_MIN, x_end=X_MAX, use_density=True):
    lefts = edges[:-1]
    rights = edges[1:]
    left_inf  = np.isneginf(lefts).astype(int)
    right_inf = np.isposinf(rights).astype(int)
    dl = np.where(left_inf,  x_start, lefts)
    dr = np.where(right_inf, x_end,   rights)
    widths = dr - dl
    density = np.where(widths > 0, probs / widths, 0.0)
    edge_left_str  = ["-∞" if np.isneginf(e) else f"{e:.4g}" for e in lefts]
    edge_right_str = ["+∞" if np.isposinf(e) else f"{e:.4g}" for e in rights]
    if counts is None:
        counts = np.zeros(len(probs))
    total = counts.sum()
    raw_prob = counts / total if total > 0 else np.zeros(len(probs))
    left_actual  = np.where(left_inf,  -1e308, lefts)
    right_actual = np.where(right_inf,  1e308, rights)
    return dict(
        left=dl, right=dr, top=density if use_density else probs, prob=probs, density=density,
        center=(dl + dr) / 2, width=widths,
        color=bar_colors(len(probs)),
        left_inf=left_inf, right_inf=right_inf,
        count=counts, raw_prob=raw_prob,
        edge_left_str=edge_left_str, edge_right_str=edge_right_str,
        left_actual=left_actual, right_actual=right_actual,
    )


def entropy_bits(probs):
    p = np.array(probs)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def bar_colors(n):
    return ["#4878CF"] * n


# ── Fixed-point iteration ─────────────────────────────────────────────────────

def compute_fixed_point_iterations(events, edges, alpha, mu, sigma):
    """Count event-based surprisal-map iterations until the distribution converges."""
    if len(events) == 0:
        return None
    interior = edges[1:-1]
    current_events = events.copy()
    probs = compute_probabilities(edges, current_events, alpha, mu, sigma)
    for i in range(MAX_ITER):
        bin_indices = np.clip(np.searchsorted(interior, current_events), 0, len(probs) - 1)
        new_events = -np.log2(probs[bin_indices])
        new_probs = compute_probabilities(edges, new_events, alpha, mu, sigma)
        if np.max(np.abs(new_probs - probs)) < CONVERGENCE_TOL:
            return i + 1
        current_events = new_events
        probs = new_probs
    return None  # did not converge


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class PNode:
    interior_edges: list = field(default_factory=list)
    events: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    figure: object = None
    source: ColumnDataSource = None
    edge_line_source: ColumnDataSource = None
    split_point_slider: Slider = None
    equal_width_left_slider: Slider = None
    equal_width_right_slider: Slider = None
    equal_width_count_slider: Slider = None
    y_mode_radio: RadioGroup = None
    y_scale_toggle: Select = None
    prior_alpha_slider: Slider = None
    prior_mu_slider: Slider = None
    prior_sigma_slider: Slider = None
    current_edges: np.ndarray = None
    current_probs: np.ndarray = None
    layout: Column = None
    edge_panel: object = None
    single_edges: list = field(default_factory=list)
    add_single_edge_input: object = None
    add_single_edge_btn: object = None
    freeze_edge_btn: object = None
    y_range_adaptive: bool = True


node: Optional[PNode] = None
convergence_div = Div(
    text="<i>Add events to compute fixed-point iterations.</i>",
    styles={"font-size": "15px", "margin-top": "10px"},
)


def recompute():
    if node is None:
        return
    edges = np.array([-np.inf] + sorted(node.interior_edges) + [np.inf])
    alpha = node.prior_alpha_slider.value
    mu    = node.prior_mu_slider.value
    sigma = node.prior_sigma_slider.value
    counts = bin_counts(edges, node.events)
    probs  = compute_probabilities(edges, node.events, alpha=alpha, mu=mu, sigma=sigma)
    node.current_edges = edges
    node.current_probs = probs
    use_density = node.y_mode_radio.active == 1
    cds_data = make_column_data_source_data(
        edges, probs, counts=counts,
        x_start=node.figure.x_range.start,
        x_end=node.figure.x_range.end,
        use_density=use_density,
    )
    node.source.data = cds_data
    if node.y_range_adaptive:
        max_top = float(np.max(cds_data['top'])) if len(cds_data['top']) > 0 else 1.0
        node.figure.y_range.end = max_top * 1.05
    else:
        node.figure.y_range.end = 1.0
    node.figure.title.text = f"P1  |  entropy = {entropy_bits(probs):.4f} bits"
    node.edge_line_source.data = dict(x=sorted(node.interior_edges))

    n_iter = compute_fixed_point_iterations(node.events, edges, alpha, mu, sigma)
    if n_iter is None and len(node.events) == 0:
        convergence_div.text = "<i>Add events to compute fixed-point iterations.</i>"
    elif n_iter is None:
        convergence_div.text = f"<b>Did not converge within {MAX_ITER} iterations.</b>"
    else:
        s = "iteration" if n_iter == 1 else "iterations"
        convergence_div.text = f"<b>Fixed point reached in {n_iter} {s}.</b>"


def make_node(initial_events):
    global node
    n = PNode()
    n.events = initial_events

    edges0 = np.array([-np.inf, np.inf])
    probs0 = compute_probabilities(edges0, initial_events)
    n.source = ColumnDataSource(make_column_data_source_data(edges0, probs0, use_density=False))

    n.figure = figure(
        width=PLOT_WIDTH, height=380,
        x_range=(X_MIN, X_MAX),
        y_range=Range1d(0, 1),
        tools=TOOLS, toolbar_location="right",
        title="P1  |  entropy = 0.0000 bits",
    )
    quad_renderer = n.figure.quad(
        left="left", right="right", top="top", bottom=0,
        source=n.source,
        fill_color="color", line_color="black", alpha=0.8,
    )
    hover = HoverTool(renderers=[quad_renderer], tooltips=[
        ("Bin",                       "@edge_left_str to @edge_right_str"),
        ("Count",                     "@count{0}"),
        ("Probability before prior",  "@raw_prob{0.0000}"),
        ("Probability",               "@prob{0.0000}"),
        ("Density",                   "@density{0.0000}"),
    ])
    n.figure.add_tools(hover)

    n.edge_line_source = ColumnDataSource(dict(x=[]))
    n.figure.ray(x="x", y=0, length=0, angle=np.pi/2,
                 source=n.edge_line_source,
                 line_color="black", line_alpha=0.08, line_width=1)

    n.figure.xgrid.grid_line_color = None
    n.figure.ygrid.grid_line_color = None
    n.figure.xaxis.axis_label = "Value"
    n.figure.yaxis.axis_label = "Probability"

    n.y_mode_radio   = RadioGroup(labels=["Probability", "Probability density"], active=0, inline=True)
    n.y_scale_toggle = Select(value="adaptive",
                              options=[("fixed", "Y: fixed 0–1"), ("adaptive", "Y: adaptive")],
                              width=140)

    n.prior_alpha_slider = Slider(start=0,    end=5,  value=PRIOR_ALPHA_DEFAULT, step=0.1, title="Prior strength α", width=250)
    n.prior_mu_slider    = Slider(start=-10,  end=10, value=PRIOR_MU_DEFAULT,    step=0.1, title="Prior mean μ",     width=250)
    n.prior_sigma_slider = Slider(start=0.1,  end=20, value=PRIOR_SIGMA_DEFAULT, step=0.1, title="Prior std dev σ",  width=250)

    _range_cb = CustomJS(
        args=dict(source=n.source, x_range=n.figure.x_range, y_mode=n.y_mode_radio),
        code="""
        const data  = source.data;
        const li    = data['left_inf'];
        const ri    = data['right_inf'];
        const prob  = data['prob'];
        const left  = data['left'].slice();
        const right = data['right'].slice();
        const xstart = x_range.start, xend = x_range.end;
        for (let i = 0; i < left.length; i++) {
            if (li[i]) left[i]  = xstart;
            if (ri[i]) right[i] = xend;
        }
        const center  = left.map((l, i) => (l + right[i]) / 2);
        const width   = left.map((l, i) => right[i] - l);
        const density = prob.map((p, i) => width[i] > 0 ? p / width[i] : 0);
        const top = y_mode.active === 1 ? density : prob.slice();
        source.data = {...data, left, right, center, width, density, top};
    """)
    n.figure.x_range.js_on_change('start', _range_cb)
    n.figure.x_range.js_on_change('end',   _range_cb)

    n.split_point_slider        = Slider(start=X_MIN, end=X_MAX, value=0.0,  step=0.1, title="Split point",             width=250)
    n.equal_width_left_slider   = Slider(start=X_MIN, end=X_MAX, value=-3.0, step=0.1, title="Evenly spaced: left",     width=250)
    n.equal_width_right_slider  = Slider(start=X_MIN, end=X_MAX, value=3.0,  step=0.1, title="Evenly spaced: right",    width=250)
    n.equal_width_count_slider  = Slider(start=0,     end=5000,   value=0,    step=1,   title="Evenly spaced: edge count", width=250)

    n.add_single_edge_input = TextInput(placeholder="Value…", width=120)
    n.add_single_edge_btn   = Button(label="Add",        width=55)
    n.freeze_edge_btn       = Button(label="Freeze edge", width=100)

    def _sync_and_recompute(nd=n):
        edges = {nd.split_point_slider.value} | set(nd.single_edges)
        count = int(nd.equal_width_count_slider.value)
        left  = nd.equal_width_left_slider.value
        right = nd.equal_width_right_slider.value
        if count > 0 and right > left:
            step = (right - left) / (count + 1)
            for i in range(count):
                edges.add(left + step * (i + 1))
        nd.interior_edges = sorted(edges)
        recompute()

    def on_bin_edge_slider_change(attr, old, new): _sync_and_recompute()

    def on_y_mode_change(attr, old, new, nd=n):
        nd.figure.yaxis.axis_label = "Probability density" if new == 1 else "Probability"
        data = nd.source.data
        nd.source.data = {**data, 'top': data['density'] if new == 1 else data['prob']}

    def on_y_scale_toggle(attr, old, new, nd=n):
        nd.y_range_adaptive = (new == "adaptive")
        recompute()

    def on_prior_change(attr, old, new): recompute()

    def on_freeze_edge(nd=n):
        val = nd.split_point_slider.value
        if val not in nd.single_edges:
            nd.single_edges.append(val)
        _sync_and_recompute()

    def on_add_single_edge(nd=n):
        val_str = nd.add_single_edge_input.value.strip()
        if not val_str:
            return
        try:
            val = float(val_str)
        except ValueError:
            return
        if val not in nd.single_edges:
            nd.single_edges.append(val)
        nd.add_single_edge_input.value = ""
        _sync_and_recompute()

    for _s in (n.split_point_slider, n.equal_width_left_slider,
               n.equal_width_right_slider, n.equal_width_count_slider):
        _s.on_change("value", on_bin_edge_slider_change)
    n.y_mode_radio.on_change("active", on_y_mode_change)
    n.prior_alpha_slider.on_change("value", on_prior_change)
    n.prior_mu_slider.on_change("value",    on_prior_change)
    n.prior_sigma_slider.on_change("value", on_prior_change)
    n.freeze_edge_btn.on_click(on_freeze_edge)
    n.add_single_edge_btn.on_click(on_add_single_edge)
    n.y_scale_toggle.on_change("value", on_y_scale_toggle)

    n.interior_edges = [n.split_point_slider.value]

    edge_panel = Column(
        Row(n.add_single_edge_input, Spacer(width=5), n.add_single_edge_btn),
        Spacer(height=4),
        Row(n.split_point_slider, Spacer(width=8), n.freeze_edge_btn),
        Spacer(height=10),
        n.equal_width_left_slider,
        n.equal_width_right_slider,
        n.equal_width_count_slider,
    )
    n.edge_panel = edge_panel
    n.layout = Column(
        Row(n.prior_alpha_slider, Spacer(width=20), n.prior_mu_slider, Spacer(width=20), n.prior_sigma_slider),
        Row(n.figure, Spacer(width=20), edge_panel),
        Row(n.y_mode_radio, Spacer(width=20), n.y_scale_toggle),
    )
    node = n
    return n


# ── Top-level event controls (same as main.py) ────────────────────────────────

n_events_input       = TextInput(value="1000", title="", width=80)
family_select        = Select(value=ev.FAMILY_NAMES[0], options=ev.FAMILY_NAMES, width=150)
append_replace_radio = RadioButtonGroup(labels=["Append", "Replace"], active=0)
_current_param_sliders: list = []
dist_params_row      = Row()
add_events_btn       = Button(label="Add events",   button_type="success", width=120)
clear_events_btn     = Button(label="Clear events", button_type="warning", width=120, disabled=True)
single_event_input        = TextInput(placeholder="Add event at value…", width=200)
single_event_count_input  = TextInput(value="1", width=60, title="")
single_event_status       = Div(text="", width=200, styles={"color": "red", "font-size": "13px", "line-height": "2.2"})

history_back_btn = Button(label="◀", width=50, disabled=True)
history_fwd_btn  = Button(label="▶", width=50, disabled=True)
history_slider   = Slider(start=0, end=1, value=0, step=1, title="", sizing_mode="stretch_width", disabled=True)
history_label    = Div(text="Step 0 of 0", styles={"line-height": "2.2", "font-size": "13px"})


def update_transport_state():
    global _transport_cb_guard
    n = len(all_events)
    _transport_cb_guard = True
    history_slider.end      = max(n, 1)
    history_slider.value    = history_index
    history_slider.disabled = n == 0
    _transport_cb_guard = False
    history_label.text           = f"Step {history_index} of {n}"
    history_back_btn.disabled    = history_index == 0
    history_fwd_btn.disabled     = history_index == n
    clear_events_btn.disabled    = n == 0


def on_make_dist():
    if node is None:
        return
    node.events = root_events.copy()
    recompute()


def on_add_events():
    global root_events, all_events, history_index
    if append_replace_radio.active == 1:
        do_replace()
        return
    try:
        n = int(n_events_input.value)
        if n <= 0:
            raise ValueError
    except ValueError:
        n = 1000
        n_events_input.value = "1000"
    was_at_end = history_index == len(all_events)
    new_ev = ev.get_events(n, family_select.value, get_current_params())
    all_events = np.concatenate([all_events, new_ev])
    if was_at_end:
        history_index = len(all_events)
    root_events = all_events[:history_index].copy()
    update_transport_state()
    on_make_dist()


def on_clear_events():
    global root_events, all_events, history_index
    root_events   = np.array([], dtype=float)
    all_events    = np.array([], dtype=float)
    history_index = 0
    on_make_dist()
    update_transport_state()


def on_single_event_input(attr, old, new):
    global root_events, all_events, history_index
    val_str = new.strip()
    if not val_str:
        return
    try:
        val = float(val_str)
    except ValueError:
        single_event_status.text  = f"'{val_str}' is not a valid number."
        single_event_input.value  = ""
        return
    try:
        count = int(single_event_count_input.value)
    except ValueError:
        count = 0
    n = max(count, 1)
    if append_replace_radio.active == 1:
        all_events    = np.full(n, val)
        history_index = n
    else:
        was_at_end = history_index == len(all_events)
        all_events = np.concatenate([all_events, np.full(n, val)])
        if was_at_end:
            history_index = len(all_events)
    root_events = all_events[:history_index].copy()
    single_event_status.text = f"Added {n} event{'s' if n > 1 else ''} at {val}."
    single_event_input.value = ""
    update_transport_state()
    on_make_dist()


def make_param_sliders(family_name):
    return [
        Slider(start=spec["start"], end=spec["end"], value=spec["value"],
               step=spec["step"], title=spec["name"], width=200)
        for spec in ev.FAMILIES[family_name]["params"]
    ]


def get_current_params():
    return {s.title: s.value for s in _current_param_sliders}


def do_replace():
    global root_events, all_events, history_index
    try:
        n = int(n_events_input.value)
        if n <= 0:
            raise ValueError
    except ValueError:
        n = 1000
    new_ev        = ev.get_events(n, family_select.value, get_current_params())
    all_events    = new_ev.copy()
    history_index = len(all_events)
    root_events   = all_events.copy()
    update_transport_state()
    on_make_dist()


def on_param_slider_change(attr, old, new):
    if append_replace_radio.active == 1:
        do_replace()


def on_family_change(attr, old, new):
    global _current_param_sliders
    _current_param_sliders      = make_param_sliders(new)
    dist_params_row.children    = list(_current_param_sliders)
    for s in _current_param_sliders:
        s.on_change("value", on_param_slider_change)
    if append_replace_radio.active == 1:
        do_replace()


add_events_btn.on_click(on_add_events)
clear_events_btn.on_click(on_clear_events)
single_event_input.on_change("value", on_single_event_input)
family_select.on_change("value", on_family_change)

_current_param_sliders   = make_param_sliders(ev.FAMILY_NAMES[0])
dist_params_row.children = list(_current_param_sliders)
for _s in _current_param_sliders:
    _s.on_change("value", on_param_slider_change)


def apply_history_index(idx):
    global root_events, history_index
    history_index = max(0, min(idx, len(all_events)))
    root_events   = all_events[:history_index].copy()
    update_transport_state()
    on_make_dist()


def on_history_slider_change(attr, old, new):
    if _transport_cb_guard:
        return
    apply_history_index(int(new))


history_slider.on_change("value", on_history_slider_change)
history_back_btn.on_click(lambda: apply_history_index(history_index - 1))
history_fwd_btn.on_click( lambda: apply_history_index(history_index + 1))

# ── Initialize ────────────────────────────────────────────────────────────────

make_node(root_events.copy())
recompute()

# ── Layout ────────────────────────────────────────────────────────────────────

top_controls = Column(
    Row(
        family_select, Spacer(width=10), dist_params_row, Spacer(width=20),
        append_replace_radio, Spacer(width=20),
        add_events_btn,
        Div(text="<b>n =</b>", styles={"line-height": "2.2", "margin-left": "6px"}),
        n_events_input, Spacer(width=20),
        clear_events_btn,
    ),
    Row(
        single_event_input,
        Div(text="<b>count:</b>", styles={"line-height": "2.2", "margin-left": "6px"}),
        single_event_count_input,
        single_event_status,
    ),
)

transport_row = Row(
    history_back_btn, Spacer(width=5),
    history_slider,   Spacer(width=5),
    history_fwd_btn,  Spacer(width=10),
    history_label,
    sizing_mode="stretch_width",
)

curdoc().add_root(Column(top_controls, transport_row, node.layout, convergence_div))
curdoc().title = "Surprisal Fixed Point"
