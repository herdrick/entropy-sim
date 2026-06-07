import numpy as np
np.set_printoptions(formatter={'float': lambda x: f"{x},"})  # Note: This leaves a trailing comma at the very end of the array, but it will restore commas between the elements in your server logs.

from scipy.stats import norm as scipy_norm
from dataclasses import dataclass, field
from typing import Optional
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, CustomJS, Div, TextInput, Button, Row, Column, Spacer, Select,
    CheckboxGroup, RadioGroup, RadioButtonGroup, Slider, HoverTool,
)
import events as ev

# ── Constants ────────────────────────────────────────────────────────────────
X_MIN, X_MAX = -10, 10
PRIOR_ALPHA_DEFAULT = 1    # pseudocount scale
PRIOR_MU_DEFAULT = 0       # prior mean
PRIOR_SIGMA_DEFAULT = 5    # prior std dev
TOOLS = "xpan,xwheel_zoom,xbox_zoom,reset,save"
PLOT_WIDTH = 900

# ── State ────────────────────────────────────────────────────────────────────
root_events = np.array([], dtype=float)   # accumulated raw events
root_node: Optional["PNode"] = None       # head of the singly-linked list

all_events: np.ndarray = np.array([], dtype=float)  # all events ever (reset on Clear)
history_index: int = 0  # == number of events visible at current step
_transport_cb_guard: bool = False
_column_count: int = 1
_all_nodes: list = []


@dataclass
class PNode:
    output_mode: str = "surprisal"       # "passthru" | "surprisal"
    interior_edges: list = field(default_factory=list)
    events: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    figure: object = None
    source: ColumnDataSource = None
    edge_line_source: ColumnDataSource = None
    child: Optional["PNode"] = None
    parent: Optional["PNode"] = None
    # UI widgets
    derive_dropdown: Select = None
    derive_btn: Button = None
    split_point_slider: Slider = None
    equal_width_left_slider: Slider = None
    equal_width_right_slider: Slider = None
    equal_width_count_slider: Slider = None
    y_mode_radio: RadioGroup = None
    prior_alpha_slider: Slider = None
    prior_mu_slider: Slider = None
    prior_sigma_slider: Slider = None
    kl_div_display: Div = None
    current_edges: np.ndarray = None
    current_probs: np.ndarray = None
    layout: Column = None
    edge_panel: object = None
    propagates: bool = False
    gang_checkbox: CheckboxGroup = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def bin_counts(edges, event_arr):
    n_bins = len(edges) - 1
    interior = edges[1:-1]
    if len(event_arr) > 0:
        indices = np.searchsorted(interior, event_arr)
        return np.bincount(indices, minlength=n_bins).astype(float)
    return np.zeros(n_bins)


def gaussian_prior_mass(edges, mu=PRIOR_MU_DEFAULT, sigma=PRIOR_SIGMA_DEFAULT):
    """Prior mass in each bin from a Gaussian(mu, sigma) distribution."""
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
    # Edge labels for hover: show "-inf" / "+inf" for infinite edges
    edge_left_str = ["-\u221e" if np.isneginf(e) else f"{e:.4g}" for e in lefts]
    edge_right_str = ["+\u221e" if np.isposinf(e) else f"{e:.4g}" for e in rights]
    if counts is None:
        counts = np.zeros(len(probs))
    total = counts.sum()
    raw_prob = counts / total if total > 0 else np.zeros(len(probs))
    return dict(
        left=dl, right=dr, top=density if use_density else probs, prob=probs, density=density,
        center=(dl + dr) / 2, width=widths,
        color=bar_colors(len(probs)),
        left_inf=left_inf, right_inf=right_inf,
        count=counts, raw_prob=raw_prob,
        edge_left_str=edge_left_str, edge_right_str=edge_right_str,
    )


def entropy_bits(probs):
    p = np.array(probs)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def kl_divergence_bits(p_edges, p_probs, q_edges, q_probs):
    """D_KL(P||Q) in bits, or None if undefined.

    Defined when every non-zero-prob bin of P has a matching bin in Q
    (same two edges) whose probability is also non-zero. Prior smoothing
    (if any) is already baked into p_probs/q_probs before this is called.
    """
    q_map = {(float(q_edges[i]), float(q_edges[i + 1])): float(q_probs[i])
             for i in range(len(q_probs))}
    total = 0.0
    for i in range(len(p_probs)):
        p = float(p_probs[i])
        if p <= 0:
            continue
        key = (float(p_edges[i]), float(p_edges[i + 1]))
        q = q_map.get(key)
        if q is None or q <= 0:
            return None
        total += p * np.log2(p / q)
    return float(total)


def bar_colors(n):
    return ["#4878CF"] * n


def node_index(node):
    idx, cur = 0, root_node
    while cur is not node:
        cur = cur.child
        idx += 1
    return idx


def rebuild_grid():
    n = _column_count
    for node in _all_nodes:
        node.figure.width = PLOT_WIDTH
        for s in (node.prior_alpha_slider, node.prior_mu_slider, node.prior_sigma_slider):
            s.width = 250
        node.layout.children[0] = Row(
            node.prior_alpha_slider, Spacer(width=20),
            node.prior_mu_slider, Spacer(width=20),
            node.prior_sigma_slider,
        )
        node.layout.children[1] = Row(node.figure, Spacer(width=20), node.edge_panel)
        node.kl_div_display.width = None
        node.layout.children[3] = Row(
            node.derive_dropdown, node.derive_btn,
            node.gang_checkbox, node.kl_div_display,
        )
    base = root_col.children[:3]
    node_rows = []
    for i in range(0, len(_all_nodes), n):
        chunk = _all_nodes[i:i+n]
        node_rows.append(Row(*[nd.layout for nd in chunk], spacing=0, sizing_mode="fixed") if len(chunk) > 1 else chunk[0].layout)
    root_col.children = base + node_rows


# ── Core recomputation ───────────────────────────────────────────────────────

def propagate_params_down(node):
    """Push node's params to its child. Recurses if child also propagates."""
    child = node.child
    if child is None:
        return
    child.split_point_slider.value = node.split_point_slider.value
    child.equal_width_left_slider.value = node.equal_width_left_slider.value
    child.equal_width_right_slider.value = node.equal_width_right_slider.value
    child.equal_width_count_slider.value = node.equal_width_count_slider.value
    child.prior_alpha_slider.value = node.prior_alpha_slider.value
    child.prior_mu_slider.value = node.prior_mu_slider.value
    child.prior_sigma_slider.value = node.prior_sigma_slider.value
    if child.propagates:
        propagate_params_down(child)


def recompute_from(node):
    if node is None:
        return
    edges = np.array([-np.inf] + sorted(node.interior_edges) + [np.inf])
    alpha = node.prior_alpha_slider.value if node.prior_alpha_slider is not None else PRIOR_ALPHA_DEFAULT
    mu = node.prior_mu_slider.value if node.prior_mu_slider is not None else PRIOR_MU_DEFAULT
    sigma = node.prior_sigma_slider.value if node.prior_sigma_slider is not None else PRIOR_SIGMA_DEFAULT
    counts = bin_counts(edges, node.events)
    probs = compute_probabilities(edges, node.events, alpha=alpha, mu=mu, sigma=sigma)
    node.current_edges = edges
    node.current_probs = probs
    use_density = node.y_mode_radio is not None and node.y_mode_radio.active == 1
    node.source.data = make_column_data_source_data(
        edges, probs, counts=counts,
        x_start=node.figure.x_range.start,
        x_end=node.figure.x_range.end,
        use_density=use_density,
    )
    idx = node_index(node)
    node.figure.title.text = (
        f"P{idx+1}  |  entropy = {entropy_bits(probs):.4f} bits"
    )

    # Update bin edge vertical lines (interior edges only, not ±inf)
    interior = sorted(node.interior_edges)
    node.edge_line_source.data = dict(x=interior)

    # Push output to child
    if node.child is not None:
        if node.output_mode == "passthru":
            node.child.events = node.events.copy()
        else:  # surprisal
            interior = edges[1:-1]
            bin_indices = np.searchsorted(interior, node.events)
            node.child.events = -np.log2(probs[bin_indices])
        recompute_from(node.child)

    refresh_kl_display(node)
    if node.parent is not None:
        refresh_kl_display(node.parent)


def refresh_kl_display(node):
    """Show KL(node ‖ parent) with ↑ and KL(node ‖ child) with ↓, when each exists."""
    if node.kl_div_display is None or node.current_edges is None:
        return
    edges, probs = node.current_edges, node.current_probs
    parts = []
    parent = node.parent
    if parent is not None and parent.current_edges is not None:
        kl_up = kl_divergence_bits(edges, probs, parent.current_edges, parent.current_probs)
        if kl_up is not None:
            parts.append(f"KL divergence ↑ {kl_up:.4f} bits")
    child = node.child
    if child is not None and child.current_edges is not None:
        kl_down = kl_divergence_bits(edges, probs, child.current_edges, child.current_probs)
        if kl_down is not None:
            parts.append(f"KL divergence ↓ {kl_down:.4f} bits")
    node.kl_div_display.text = " &nbsp;&nbsp; ".join(parts)


# ── PNode factory ────────────────────────────────────────────────────────────

def make_p_node(initial_events):
    node = PNode()
    node.events = initial_events

    edges0 = np.array([-np.inf, np.inf])
    probs0 = compute_probabilities(edges0, initial_events)
    node.source = ColumnDataSource(make_column_data_source_data(edges0, probs0, use_density=False))

    node.figure = figure(
        width=PLOT_WIDTH, height=380,
        x_range=(X_MIN, X_MAX),
        tools=TOOLS, toolbar_location="right",
        title="P  |  Entropy = 0.0000 bits",
    )
    quad_renderer = node.figure.quad(
        left="left", right="right", top="top", bottom=0,
        source=node.source,
        fill_color="color", line_color="black", alpha=0.8,
    )
    hover = HoverTool(renderers=[quad_renderer], tooltips=[
        ("Bin", "@edge_left_str to @edge_right_str"),
        ("Count", "@count{0}"),
        ("Probability before prior", "@raw_prob{0.0000}"),
        ("Probability", "@prob{0.0000}"),
        ("Density", "@density{0.0000}"),
    ])
    node.figure.add_tools(hover)
    # Vertical lines at bin edges (full plot height)
    node.edge_line_source = ColumnDataSource(dict(x=[]))
    node.figure.ray(x="x", y=0, length=0, angle=np.pi/2,
                    source=node.edge_line_source,
                    line_color="black", line_width=1)

    node.figure.xaxis.axis_label = "Value"
    node.figure.yaxis.axis_label = "Probability"

    # Y-mode radio: probability vs probability density
    node.y_mode_radio = RadioGroup(labels=["Probability", "Probability density"], active=0, inline=True)

    # Gang checkbox (hidden until node is linked to a parent)
    node.gang_checkbox = CheckboxGroup(labels=["Copy params to child node"], active=[])

    # Gaussian prior sliders
    node.prior_alpha_slider = Slider(
        start=0, end=5, value=PRIOR_ALPHA_DEFAULT, step=0.1,
        title="Prior strength α", width=250,
    )
    node.prior_mu_slider = Slider(
        start=-10, end=10, value=PRIOR_MU_DEFAULT, step=0.1,
        title="Prior mean μ", width=250,
    )
    node.prior_sigma_slider = Slider(
        start=0.1, end=20, value=PRIOR_SIGMA_DEFAULT, step=0.1,
        title="Prior std dev σ", width=250,
    )

    # JS callback for infinite-edge stretching
    _range_cb = CustomJS(args=dict(source=node.source, x_range=node.figure.x_range, y_mode=node.y_mode_radio), code="""
        const data  = source.data;
        const li    = data['left_inf'];
        const ri    = data['right_inf'];
        const prob  = data['prob'];
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
        const density = prob.map((p, i) => width[i] > 0 ? p / width[i] : 0);
        const top = y_mode.active === 1 ? density : prob.slice();
        source.data = {...data, left, right, center, width, density, top};
    """)
    node.figure.x_range.js_on_change('start', _range_cb)
    node.figure.x_range.js_on_change('end',   _range_cb)

    # ── Bin edge controls ────────────────────────────────────────────────
    node.split_point_slider = Slider(
        start=X_MIN, end=X_MAX, value=0.0, step=0.1,
        title="Split point", width=250,
    )
    node.equal_width_left_slider = Slider(
        start=X_MIN, end=X_MAX, value=-3.0, step=0.1,
        title="Evenly spaced: left", width=250,
    )
    node.equal_width_right_slider = Slider(
        start=X_MIN, end=X_MAX, value=3.0, step=0.1,
        title="Evenly spaced: right", width=250,
    )
    node.equal_width_count_slider = Slider(
        start=0, end=20, value=0, step=1,
        title="Evenly spaced: edge count", width=250,
    )

    # ── Derive controls ──────────────────────────────────────────────────
    node.derive_dropdown = Select(
        value="Surprisal",
        options=["Pass events thru as they are", "Surprisal"],
        width=250,
    )
    node.derive_btn = Button(label="View derived distribution", button_type="primary", width=220)
    node.kl_div_display = Div(text="", styles={"line-height": "2.2", "margin-left": "10px", "font-size": "13px"})

    # ── Per-node callbacks ───────────────────────────────────────────────

    def _sync_edges_and_recompute(n=node):
        edges = {n.split_point_slider.value}
        count = int(n.equal_width_count_slider.value)
        left = n.equal_width_left_slider.value
        right = n.equal_width_right_slider.value
        if count > 0 and right > left:
            step = (right - left) / (count + 1)
            for i in range(count):
                edges.add(left + step * (i + 1))
        n.interior_edges = sorted(edges)
        recompute_from(n)

    def on_bin_edge_slider_change(attr, old, new, n=node):
        _sync_edges_and_recompute(n)

    def on_output_mode_change(attr, old, new, n=node):
        n.output_mode = "passthru" if new == "Pass events thru as they are" else "surprisal"
        if n.child is not None:
            recompute_from(n)

    def on_y_mode_change(attr, old, new, n=node):
        n.figure.yaxis.axis_label = "Probability density" if new == 1 else "Probability"
        # Switch top between density and prob
        data = n.source.data
        if new == 1:
            n.source.data = {**data, 'top': data['density']}
        else:
            n.source.data = {**data, 'top': data['prob']}

    def on_prior_change(attr, old, new, n=node):
        recompute_from(n)

    def on_propagate_change(attr, old, new, n=node):
        n.propagates = 0 in new

    def on_derive(n=node):
        create_child_node(n)

    # Wire up callbacks
    for _s in (node.split_point_slider, node.equal_width_left_slider,
               node.equal_width_right_slider, node.equal_width_count_slider):
        _s.on_change("value", on_bin_edge_slider_change)
    node.y_mode_radio.on_change("active", on_y_mode_change)
    node.prior_alpha_slider.on_change("value", on_prior_change)
    node.prior_mu_slider.on_change("value", on_prior_change)
    node.prior_sigma_slider.on_change("value", on_prior_change)
    node.derive_dropdown.on_change("value", on_output_mode_change)
    node.derive_btn.on_click(on_derive)
    node.gang_checkbox.on_change("active", on_propagate_change)

    # ── Layout for this node ─────────────────────────────────────────────
    edge_panel = Column(
        node.split_point_slider,
        Spacer(height=10),
        node.equal_width_left_slider,
        node.equal_width_right_slider,
        node.equal_width_count_slider,
    )
    node.edge_panel = edge_panel

    # Initialize interior_edges from slider defaults (callbacks only fire on change)
    node.interior_edges = [node.split_point_slider.value]

    derive_row = Row(node.derive_dropdown, node.derive_btn, node.gang_checkbox, node.kl_div_display)

    prior_row = Row(node.prior_alpha_slider, Spacer(width=20), node.prior_mu_slider, Spacer(width=20), node.prior_sigma_slider)
    plot_and_edges = Row(node.figure, Spacer(width=20), edge_panel)
    node.layout = Column(prior_row, plot_and_edges, node.y_mode_radio, derive_row)

    return node


def create_child_node(parent_node):
    """Create a new PNode as a child of parent_node and append to layout."""
    global root_node

    # Compute the events this child will receive
    if parent_node is not None:
        if parent_node.output_mode == "passthru":
            child_events = parent_node.events.copy()
        else:  # surprisal
            edges = np.array([-np.inf] + sorted(parent_node.interior_edges) + [np.inf])
            alpha = parent_node.prior_alpha_slider.value if parent_node.prior_alpha_slider else PRIOR_ALPHA_DEFAULT
            mu = parent_node.prior_mu_slider.value if parent_node.prior_mu_slider else PRIOR_MU_DEFAULT
            sigma = parent_node.prior_sigma_slider.value if parent_node.prior_sigma_slider else PRIOR_SIGMA_DEFAULT
            probs = compute_probabilities(edges, parent_node.events, alpha=alpha, mu=mu, sigma=sigma)
            interior = edges[1:-1]
            bin_indices = np.searchsorted(interior, parent_node.events)
            child_events = -np.log2(probs[bin_indices])
    else:
        child_events = root_events.copy()

    new_node = make_p_node(child_events)

    if parent_node is not None:
        parent_node.child = new_node
        new_node.parent = parent_node
        parent_node.derive_btn.disabled = True
        if parent_node.propagates:
            new_node.propagates = True
            new_node.gang_checkbox.active = [0]
            propagate_params_down(parent_node)
    else:
        root_node = new_node
        initial_derive_btn.disabled = True

    # Append to node list and rebuild grid
    _all_nodes.append(new_node)
    rebuild_grid()

    # Recompute so it shows a distribution (also refreshes KL displays for new_node and its parent)
    recompute_from(new_node)


# ── Top-level event controls ─────────────────────────────────────────────────

n_events_input = TextInput(value="1000", title="", width=80)
family_select = Select(value=ev.FAMILY_NAMES[0], options=ev.FAMILY_NAMES, width=150)
append_replace_radio = RadioButtonGroup(labels=["Append", "Replace"], active=0)
_current_param_sliders: list = []
dist_params_row = Row()
add_events_btn = Button(label="Add events", button_type="success", width=120)
clear_events_btn = Button(label="Clear events", button_type="warning", width=120, disabled=True)
single_event_input = TextInput(placeholder="Add event at value…", width=200)
single_event_count_input = TextInput(value="1", width=60, title="")
single_event_status = Div(text="", width=200, styles={"color": "red", "font-size": "13px", "line-height": "2.2"})

# Initial "View derived distribution" button (before any nodes exist)
initial_derive_btn = Button(label="View derived distribution", button_type="primary", width=220)

col_count_radio = RadioButtonGroup(labels=["1 col", "2 col", "3 col"], active=0)

def on_col_count_change(attr, old, new):
    global _column_count
    _column_count = new + 1
    rebuild_grid()

col_count_radio.on_change("active", on_col_count_change)

history_back_btn = Button(label="◀", width=50, disabled=True)
history_fwd_btn = Button(label="▶", width=50, disabled=True)
history_slider = Slider(start=0, end=1, value=0, step=1, title="", sizing_mode="stretch_width", disabled=True)
history_label = Div(text="Step 0 of 0", styles={"line-height": "2.2", "font-size": "13px"})


def update_transport_state():
    global _transport_cb_guard
    n = len(all_events)
    at_end = history_index == n
    at_start = history_index == 0
    _transport_cb_guard = True
    history_slider.end = max(n, 1)
    history_slider.value = history_index
    history_slider.disabled = n == 0
    _transport_cb_guard = False
    history_label.text = f"Step {history_index} of {n}"
    history_back_btn.disabled = at_start
    history_fwd_btn.disabled = at_end
    clear_events_btn.disabled = len(all_events) == 0


def refresh_rug():
    update_transport_state()


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
    refresh_rug()
    on_make_dist()


def on_make_dist():
    if root_node is None:
        return
    root_node.events = root_events.copy()
    recompute_from(root_node)


def on_clear_events():
    global root_events, all_events, history_index
    root_events = np.array([], dtype=float)
    all_events = np.array([], dtype=float)
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
        single_event_status.text = f"'{val_str}' is not a valid number."
        single_event_input.value = ""
        return
    try:
        count = int(single_event_count_input.value)
    except ValueError:
        count = 0
    n = max(count, 1)
    if append_replace_radio.active == 1:
        all_events = np.full(n, val)
        history_index = n
    else:
        was_at_end = history_index == len(all_events)
        all_events = np.concatenate([all_events, np.full(n, val)])
        if was_at_end:
            history_index = len(all_events)
    root_events = all_events[:history_index].copy()
    single_event_status.text = f"Added {n} event{'s' if n > 1 else ''} at {val}."
    single_event_input.value = ""
    refresh_rug()
    on_make_dist()


def make_param_sliders(family_name):
    sliders = []
    for spec in ev.FAMILIES[family_name]["params"]:
        s = Slider(
            start=spec["start"], end=spec["end"],
            value=spec["value"], step=spec["step"],
            title=spec["name"], width=200,
        )
        sliders.append(s)
    return sliders


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
    new_ev = ev.get_events(n, family_select.value, get_current_params())
    all_events = new_ev.copy()
    history_index = len(all_events)
    root_events = all_events.copy()
    refresh_rug()
    on_make_dist()


def on_param_slider_change(attr, old, new):
    if append_replace_radio.active == 1:
        do_replace()


def on_family_change(attr, old, new):
    global _current_param_sliders
    _current_param_sliders = make_param_sliders(new)
    dist_params_row.children = list(_current_param_sliders)
    for s in _current_param_sliders:
        s.on_change("value", on_param_slider_change)
    if append_replace_radio.active == 1:
        do_replace()


def on_initial_derive():
    create_child_node(None)


add_events_btn.on_click(on_add_events)
clear_events_btn.on_click(on_clear_events)
single_event_input.on_change("value", on_single_event_input)
initial_derive_btn.on_click(on_initial_derive)
family_select.on_change("value", on_family_change)

# Initialize param sliders for the default family
_current_param_sliders = make_param_sliders(ev.FAMILY_NAMES[0])
dist_params_row.children = list(_current_param_sliders)
for _s in _current_param_sliders:
    _s.on_change("value", on_param_slider_change)


def apply_history_index(idx):
    global root_events, history_index
    history_index = max(0, min(idx, len(all_events)))
    root_events = all_events[:history_index].copy()
    refresh_rug()
    on_make_dist()


def on_history_slider_change(attr, old, new):
    if _transport_cb_guard:
        return
    apply_history_index(int(new))


def on_history_back():
    apply_history_index(history_index - 1)


def on_history_fwd():
    apply_history_index(history_index + 1)


history_slider.on_change("value", on_history_slider_change)
history_back_btn.on_click(on_history_back)
history_fwd_btn.on_click(on_history_fwd)

# ── Layout ────────────────────────────────────────────────────────────────────

top_controls = Column(
    Row(
        family_select,
        Spacer(width=10),
        dist_params_row,
        Spacer(width=20),
        append_replace_radio,
        Spacer(width=20),
        add_events_btn,
        Div(text="<b>n =</b>", styles={"line-height": "2.2", "margin-left": "6px"}),
        n_events_input,
        Spacer(width=20),
        clear_events_btn,
    ),
    Row(
        single_event_input,
        Div(text="<b>count:</b>", styles={"line-height": "2.2", "margin-left": "6px"}),
        single_event_count_input,
        single_event_status,
        Spacer(width=30),
        Div(text="<b>Layout:</b>", styles={"line-height": "2.2"}),
        col_count_radio,
    ),
)

transport_row = Row(
    history_back_btn,
    Spacer(width=5),
    history_slider,
    Spacer(width=5),
    history_fwd_btn,
    Spacer(width=10),
    history_label,
    sizing_mode="stretch_width",
)

root_col = Column(
    top_controls,
    transport_row,
    initial_derive_btn,
)

curdoc().add_root(root_col)
curdoc().title = "Entropy & Surprisal Explorer"
