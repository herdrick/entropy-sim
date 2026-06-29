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

from viz_simplex3d import make_simplex3d_panel, update_simplex3d_panel
from bin_selection import BinTracker, make_bin_lock_ui
from viz_radial import make_radial_panel, update_radial_panel
from viz_scatter_matrix import make_scatter_matrix_panel, update_scatter_matrix_panel
from viz_parallel_coords import make_parallel_coords_panel, update_parallel_coords_panel

# ── Constants ────────────────────────────────────────────────────────────────
X_MIN, X_MAX = -10, 300
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
_step_cb_handle: list = [None]  # holds add_next_tick_callback handle during step-through animation


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

def compute_fixed_point_iterations(events, init_edges, surp_edges, alpha, mu, sigma):
    """Return (n_iter, converged_probs) or (None, None) if no convergence.

    init_edges: bins for the initial distribution (P1).
    surp_edges: bins used for every surprisal distribution (S(P1), S(S(P1)), ...).
    """
    if len(events) == 0:
        return None, None
    init_interior = init_edges[1:-1]
    surp_interior = surp_edges[1:-1]
    # Map original events through P1 → first surprisal events
    init_probs = compute_probabilities(init_edges, events, alpha, mu, sigma)
    bin_idx = np.clip(np.searchsorted(init_interior, events), 0, len(init_probs) - 1)
    current_events = -np.log2(init_probs[bin_idx])
    # First surprisal distribution (S(P1))
    probs = compute_probabilities(surp_edges, current_events, alpha, mu, sigma)
    for i in range(MAX_ITER):
        bin_idx = np.clip(np.searchsorted(surp_interior, current_events), 0, len(probs) - 1)
        new_events = -np.log2(probs[bin_idx])
        new_probs = compute_probabilities(surp_edges, new_events, alpha, mu, sigma)
        if np.max(np.abs(new_probs - probs)) < CONVERGENCE_TOL:
            return i + 1, new_probs
        current_events = new_events
        probs = new_probs
    return None, None  # did not converge


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
surp_node: Optional[PNode] = None
session_record: int = 0       # highest n_iter seen this session
session_record_rows: list = [] # list of HTML strings, one per record
all_simplex_fixed_points: list = []  # each entry: full converged prob vector (any length)
_fp_barchart_figures: dict = {}  # tuple(edges) -> {'source', 'fig', 'n', 'y_max'}
_FP_BAR_ALPHA = 0.15
_FP_BAR_COLOR = "#2266AA"

clear_simplex_btn = Button(label="Clear points", width=110, button_type="warning")
simplex_stats_div = Div(width=160, height=460, styles={"font-size": "13px", "padding-left": "12px"})

# Viz panel state — populated after make_node() call below
_simplex3d_state = None
_radial_state = None
_scatter_state = None
_parallel_state = None
_bin_lock_layout = None
_locked_bins_state = None
tracker = BinTracker()


def update_simplex_stats():
    from collections import Counter
    counts = Counter()
    for fp in all_simplex_fixed_points:
        for i in [i for i, v in enumerate(fp) if v > 1e-12][:3]:
            counts[i] += 1
    if not counts:
        simplex_stats_div.text = ""
        return
    rows = "".join(
        f"<tr><td>p{i+1}</td><td style='padding-left:8px;text-align:right'>{c}×</td></tr>"
        for i, c in counts.most_common(10)
    )
    simplex_stats_div.text = (
        f"<b>Top bins</b> (n={len(all_simplex_fixed_points)})"
        f"<table style='margin-top:6px'>{rows}</table>"
    )


convergence_div = Div(
    text="<i>Add events to compute fixed-point iterations.</i>",
    styles={"font-size": "15px", "margin-top": "10px"},
)


def _update_surp_node():
    """Compute and display the first surprisal distribution S(P1) in surp_node."""
    if node is None or surp_node is None:
        return
    edges = node.current_edges
    probs = node.current_probs
    if edges is None or probs is None or len(node.events) == 0:
        empty_edges = np.array([-np.inf, np.inf])
        surp_node.source.data = make_column_data_source_data(empty_edges, np.array([1.0]), use_density=False)
        surp_node.current_edges = empty_edges
        surp_node.current_probs = np.array([1.0])
        surp_node.figure.title.text = "S(P1) — First Surprisal Distribution"
        return
    interior = edges[1:-1]
    bin_idx = np.clip(np.searchsorted(interior, node.events), 0, len(probs) - 1)
    surp_events = -np.log2(probs[bin_idx])
    surp_node.events = surp_events
    s_edges = np.array([-np.inf] + sorted(surp_node.interior_edges) + [np.inf])
    alpha = node.prior_alpha_slider.value
    mu    = node.prior_mu_slider.value
    sigma = node.prior_sigma_slider.value
    s_counts = bin_counts(s_edges, surp_events)
    s_probs  = compute_probabilities(s_edges, surp_events, alpha, mu, sigma)
    surp_node.current_edges = s_edges
    surp_node.current_probs = s_probs
    use_density = surp_node.y_mode_radio.active == 1
    cds = make_column_data_source_data(
        s_edges, s_probs, counts=s_counts,
        x_start=surp_node.figure.x_range.start,
        x_end=surp_node.figure.x_range.end,
        use_density=use_density,
    )
    surp_node.source.data = cds
    if surp_node.y_range_adaptive:
        max_top = float(np.max(cds['top'])) if len(cds['top']) > 0 else 1.0
        surp_node.figure.y_range.end = max_top * 1.05
    else:
        surp_node.figure.y_range.end = 1.0
    surp_node.figure.title.text = f"S(P1)  |  entropy = {entropy_bits(s_probs):.4f} bits"
    surp_node.edge_line_source.data = dict(x=sorted(surp_node.interior_edges))


def _get_active_bins():
    """Return (bin_indices, bin_labels) based on lock state."""
    if _locked_bins_state is not None and _locked_bins_state['locked']:
        indices = _locked_bins_state['bins']
        labels = _locked_bins_state['labels']
    else:
        indices = tracker.get_active_bins(min_freq=0.1)
        labels = tracker.get_bin_labels(indices)
    return indices, labels


def _refresh_viz_panels():
    """Update all 4 viz panels with current data."""
    if _simplex3d_state is None:
        return
    if _locked_bins_state and '_refresh_status' in _locked_bins_state:
        _locked_bins_state['_refresh_status']()
    indices, labels = _get_active_bins()
    update_simplex3d_panel(_simplex3d_state, all_simplex_fixed_points, indices, labels)
    new_radial = update_radial_panel(_radial_state, all_simplex_fixed_points, indices, labels)
    _radial_wrap.children = [new_radial]
    new_scatter = update_scatter_matrix_panel(_scatter_state, all_simplex_fixed_points, indices, labels)
    _scatter_wrap.children = [new_scatter]
    new_parallel = update_parallel_coords_panel(_parallel_state, all_simplex_fixed_points, indices, labels)
    _parallel_wrap.children = [new_parallel]


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

    _update_surp_node()
    surp_edges = (np.array([-np.inf] + sorted(surp_node.interior_edges) + [np.inf])
                  if surp_node is not None else edges)
    n_iter, fixed_probs = compute_fixed_point_iterations(node.events, edges, surp_edges, alpha, mu, sigma)
    if n_iter is None and len(node.events) == 0:
        convergence_div.text = "<i>Add events to compute fixed-point iterations.</i>"
        return
    elif n_iter is None:
        convergence_div.text = f"<b>Did not converge within {MAX_ITER} iterations.</b>"
        return

    if fixed_probs is not None:
        all_simplex_fixed_points.append(fixed_probs.copy())
        tracker.record(fixed_probs)
        update_simplex_stats()
        _refresh_viz_panels()
        _add_to_fp_barchart(surp_edges, fixed_probs)

    s = "iteration" if n_iter == 1 else "iterations"
    current_line = f"<b>Fixed point reached in {n_iter} {s}.</b>"

    global session_record, session_record_rows
    if n_iter > session_record:
        session_record = n_iter
        dist_desc = family_select.value
        if _current_param_sliders:
            param_str = ", ".join(f"{sl.title}={sl.value:.3g}" for sl in _current_param_sliders)
            dist_desc += f"({param_str})"
        n_events = len(node.events)
        n_interior = len(node.interior_edges)
        if n_interior > 0:
            e_min = min(node.interior_edges)
            e_max = max(node.interior_edges)
            edge_desc = f"{n_interior} interior edges [{e_min:.3g}–{e_max:.3g}]"
        else:
            edge_desc = "no interior edges"
        row = (f"<b>{n_iter}</b> {s} — {dist_desc} | "
               f"events count: {n_events} | {edge_desc}")
        session_record_rows.insert(0, row)

    if session_record_rows:
        records_html = "<br>".join(session_record_rows)
        convergence_div.text = (
            current_line
            + "<br><br><b>Records this session:</b><br>"
            + records_html
        )
    else:
        convergence_div.text = current_line


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


def make_surp_node():
    global surp_node
    n = PNode()
    n.events = np.array([], dtype=float)

    edges0 = np.array([-np.inf, np.inf])
    probs0 = np.array([1.0])
    n.source = ColumnDataSource(make_column_data_source_data(edges0, probs0, use_density=False))

    n.figure = figure(
        width=PLOT_WIDTH, height=380,
        x_range=(0, 50),
        y_range=Range1d(0, 1),
        tools=TOOLS, toolbar_location="right",
        title="S(P1) — First Surprisal Distribution",
    )
    quad_renderer = n.figure.quad(
        left="left", right="right", top="top", bottom=0,
        source=n.source,
        fill_color="color", line_color="black", alpha=0.8,
    )
    hover = HoverTool(renderers=[quad_renderer], tooltips=[
        ("Bin",    "@edge_left_str to @edge_right_str"),
        ("Count",  "@count{0}"),
        ("Probability before prior", "@raw_prob{0.0000}"),
        ("Probability", "@prob{0.0000}"),
        ("Density",     "@density{0.0000}"),
    ])
    n.figure.add_tools(hover)
    n.edge_line_source = ColumnDataSource(dict(x=[]))
    n.figure.ray(x="x", y=0, length=0, angle=np.pi/2,
                 source=n.edge_line_source,
                 line_color="black", line_alpha=0.08, line_width=1)
    n.figure.xgrid.grid_line_color = None
    n.figure.ygrid.grid_line_color = None
    n.figure.xaxis.axis_label = "Surprisal (bits)"
    n.figure.yaxis.axis_label = "Probability"

    n.y_mode_radio   = RadioGroup(labels=["Probability", "Probability density"], active=0, inline=True)
    n.y_scale_toggle = Select(value="adaptive",
                              options=[("fixed", "Y: fixed 0–1"), ("adaptive", "Y: adaptive")],
                              width=140)

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

    n.split_point_slider       = Slider(start=0, end=20, value=5.0,  step=0.1, title="Split point",              width=250)
    n.equal_width_left_slider  = Slider(start=0, end=20, value=0.0,  step=0.1, title="Evenly spaced: left",      width=250)
    n.equal_width_right_slider = Slider(start=0, end=20, value=10.0, step=0.1, title="Evenly spaced: right",     width=250)
    n.equal_width_count_slider = Slider(start=0, end=1000, value=0,  step=1,   title="Evenly spaced: edge count", width=250)

    n.add_single_edge_input = TextInput(placeholder="Value…", width=120)
    n.add_single_edge_btn   = Button(label="Add",         width=55)
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
        Row(n.figure, Spacer(width=20), edge_panel),
        Row(n.y_mode_radio, Spacer(width=20), n.y_scale_toggle),
    )
    surp_node = n
    return n


# ── Top-level event controls (same as main.py) ────────────────────────────────

n_events_input       = TextInput(value="1000", title="", width=80)
family_select        = Select(value=ev.FAMILY_NAMES[0], options=ev.FAMILY_NAMES, width=150)
append_replace_radio = RadioButtonGroup(labels=["Append", "Replace"], active=0)
_current_param_sliders: list = []
dist_params_row      = Row()
add_events_btn            = Button(label="Add events",             button_type="success", width=120)
add_events_one_by_one_btn = Button(label="Add events (one by one)", button_type="success", width=190)
clear_events_btn          = Button(label="Clear events",           button_type="warning", width=120, disabled=True)
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


def on_clear_simplex():
    global all_simplex_fixed_points
    all_simplex_fixed_points = []
    _fp_barchart_figures.clear()
    _fp_barchart_wrap.children = []
    tracker.reset()
    update_simplex_stats()
    _refresh_viz_panels()

clear_simplex_btn.on_click(on_clear_simplex)

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


def on_add_events_one_by_one():
    global all_events

    if _step_cb_handle[0] is not None:
        try:
            curdoc().remove_next_tick_callback(_step_cb_handle[0])
        except Exception:
            pass
        _step_cb_handle[0] = None
        add_events_one_by_one_btn.label = "Add events (one by one)"
        return

    try:
        n = int(n_events_input.value)
        if n <= 0:
            raise ValueError
    except ValueError:
        n = 1000
        n_events_input.value = "1000"

    new_ev = ev.get_events(n, family_select.value, get_current_params())
    all_events = np.concatenate([all_events, new_ev])
    target_index = len(all_events)

    add_events_one_by_one_btn.label = "Stop"

    def step():
        if _step_cb_handle[0] is None:
            add_events_one_by_one_btn.label = "Add events (one by one)"
            return
        apply_history_index(history_index + 1)
        if history_index >= target_index:
            _step_cb_handle[0] = None
            add_events_one_by_one_btn.label = "Add events (one by one)"
        else:
            _step_cb_handle[0] = curdoc().add_next_tick_callback(step)

    _step_cb_handle[0] = curdoc().add_next_tick_callback(step)


add_events_one_by_one_btn.on_click(on_add_events_one_by_one)

# ── Initialize ────────────────────────────────────────────────────────────────

make_node(root_events.copy())
make_surp_node()

# Initialize viz panels
_simplex3d_layout, _simplex3d_state = make_simplex3d_panel([], [], [])
_radial_layout, _radial_state = make_radial_panel([], [], [])
_scatter_layout, _scatter_state = make_scatter_matrix_panel([], [], [])
_parallel_layout, _parallel_state = make_parallel_coords_panel([], [], [])
_bin_lock_layout, _locked_bins_state = make_bin_lock_ui(tracker)

# Wrapper columns so _refresh_viz_panels can replace panel contents when bins change
_radial_wrap = Column(_radial_layout)
_scatter_wrap = Column(_scatter_layout)
_parallel_wrap = Column(_parallel_layout)

_fp_barchart_wrap = Column()  # populated as fixed points accumulate


def _add_to_fp_barchart(edges, probs):
    """Add one fixed-point distribution to the overlay bar chart for its binning group."""
    key = tuple(float(e) for e in edges)
    x_start, x_end = 0.0, 50.0

    lefts = np.where(np.isneginf(edges[:-1]), x_start, edges[:-1])
    rights = np.where(np.isposinf(edges[1:]), x_end, edges[1:])
    bottom = np.zeros(len(probs))

    new_data = dict(left=lefts, right=rights, top=probs, bottom=bottom)
    y_max = float(np.max(probs)) if len(probs) > 0 else 1.0

    if key not in _fp_barchart_figures:
        source = ColumnDataSource(new_data)
        interior = edges[1:-1]
        if len(interior) == 0:
            edge_desc = "no interior edges"
        else:
            edge_desc = f"{len(interior)} interior edge{'s' if len(interior) != 1 else ''}: " + \
                        ", ".join(f"{e:.4g}" for e in interior)
        n_bins = len(probs)
        fig = figure(
            width=PLOT_WIDTH, height=280,
            x_range=(x_start, x_end),
            y_range=Range1d(0, y_max * 1.1 or 1.0),
            tools=TOOLS, toolbar_location="right",
            title=f"Fixed-point overlays — 1 point, {n_bins} bins ({edge_desc})",
        )
        fig.quad(left="left", right="right", top="top", bottom="bottom",
                 source=source,
                 fill_color=_FP_BAR_COLOR, line_color=None,
                 fill_alpha=_FP_BAR_ALPHA)
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        fig.xaxis.axis_label = "Surprisal (bits)"
        fig.yaxis.axis_label = "Probability"
        _fp_barchart_figures[key] = {'source': source, 'fig': fig, 'n': 1, 'y_max': y_max}
        _fp_barchart_wrap.children = [state['fig'] for state in _fp_barchart_figures.values()]
    else:
        state = _fp_barchart_figures[key]
        state['n'] += 1
        state['y_max'] = max(state['y_max'], y_max)
        old = state['source'].data
        state['source'].data = {k: np.concatenate([old[k], new_data[k]]) for k in new_data}
        n_bins = len(probs)
        state['fig'].title.text = (
            f"Fixed-point overlays — {state['n']} points, {n_bins} bins"
        )
        state['fig'].y_range.end = state['y_max'] * 1.1

recompute()

# ── Layout ────────────────────────────────────────────────────────────────────

top_controls = Column(
    Row(
        family_select, Spacer(width=10), dist_params_row, Spacer(width=20),
        append_replace_radio, Spacer(width=20),
        add_events_btn, Spacer(width=10),
        add_events_one_by_one_btn,
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

simplex_section = Column(
    Row(clear_simplex_btn, Spacer(width=20), _bin_lock_layout),
    Row(_simplex3d_layout, simplex_stats_div),
    Row(_radial_wrap, _parallel_wrap),
    _scatter_wrap,
)

curdoc().add_root(Column(top_controls, transport_row, node.layout, surp_node.layout, convergence_div, _fp_barchart_wrap, simplex_section))
curdoc().title = "Surprisal Fixed Point"
