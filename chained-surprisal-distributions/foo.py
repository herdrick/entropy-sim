# run with "bokeh serve start-with-detailed-prompt/foo.py --dev"
import numpy as np
np.set_printoptions(formatter={'float': lambda x: f"{x},"})  # Note: This leaves a trailing comma at the very end of the array, but it will restore commas between the elements in your server logs.

from scipy.stats import norm as scipy_norm
from dataclasses import dataclass, field
from typing import Optional
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, CustomJS, Div, TextInput, Button, Row, Column, Spacer, Select,
    CheckboxGroup, RadioGroup, Slider, HoverTool,
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

event_history: list = [np.array([], dtype=float)]  # snapshots; index 0 = empty
history_index: int = 0
_transport_cb_guard: bool = False


@dataclass
class PNode:
    output_mode: str = "surprisal"       # "passthru" | "surprisal"
    interior_edges: list = field(default_factory=list)
    events: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    figure: object = None
    source: ColumnDataSource = None
    rug_fig: object = None
    rug_source: ColumnDataSource = None
    edge_line_source: ColumnDataSource = None
    child: Optional["PNode"] = None
    parent: Optional["PNode"] = None
    # UI widgets
    derive_dropdown: Select = None
    derive_btn: Button = None
    edge_input: TextInput = None
    edge_status: Div = None
    divide_bin_btn: Button = None
    equal_width_btn: Button = None
    equal_width_left_input: TextInput = None
    equal_width_right_input: TextInput = None
    equal_width_count_input: TextInput = None
    equal_width_submit_btn: Button = None
    equal_width_edge_at_ends: CheckboxGroup = None
    equal_width_preview: Div = None
    equal_width_status: Div = None
    y_mode_radio: RadioGroup = None
    prior_alpha_slider: Slider = None
    prior_mu_slider: Slider = None
    prior_sigma_slider: Slider = None
    kl_div_display: Div = None
    current_edges: np.ndarray = None
    current_probs: np.ndarray = None
    layout: Column = None
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


# ── Core recomputation ───────────────────────────────────────────────────────

def propagate_params_down(node):
    """Push node's params to its child. Recurses if child also propagates."""
    child = node.child
    if child is None:
        return
    child.interior_edges = list(node.interior_edges)
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

    # Update this node's rug plot
    node.rug_source.data = dict(x=node.events, y=np.zeros(len(node.events)))
    node.rug_fig.title.text = f"Events ({len(node.events)})"

    # Push output to child
    if node.child is not None:
        if node.output_mode == "passthru":
            node.child.events = node.events.copy()
        else:  # surprisal
            interior = edges[1:-1]
            bin_indices = np.searchsorted(interior, node.events)
            node.child.events = -np.log2(probs[bin_indices])
        recompute_from(node.child)
        update_kl_display(node)
    else:
        if node.kl_div_display is not None:
            node.kl_div_display.text = ""
    if node.parent is not None:
        update_kl_display(node.parent)


def update_kl_display(parent):
    child = parent.child
    if child is None or parent.kl_div_display is None:
        return
    pe, pp = parent.current_edges, parent.current_probs
    ce, cp = child.current_edges, child.current_probs
    if pe is None or ce is None:
        parent.kl_div_display.text = ""
        return
    pi = node_index(parent) + 1
    ci = node_index(child) + 1
    kl_pc = kl_divergence_bits(pe, pp, ce, cp)
    kl_cp = kl_divergence_bits(ce, cp, pe, pp)
    parts = []
    if kl_pc is not None:
        parts.append(f"KL divergence ↓ {kl_pc:.4f} bits")
    if kl_cp is not None:
        parts.append(f"KL divergence ↑ {kl_cp:.4f} bits")
    parent.kl_div_display.text = " &nbsp;&nbsp; ".join(parts)


# ── PNode factory ────────────────────────────────────────────────────────────

def make_p_node(initial_events):
    node = PNode()
    node.events = initial_events

    # Rug plot for this node
    node.rug_source = ColumnDataSource(dict(x=initial_events, y=np.zeros(len(initial_events))))
    node.rug_fig = figure(
        width=PLOT_WIDTH, height=80,
        x_range=(X_MIN, X_MAX), y_range=(-0.5, 0.5),
        tools=TOOLS, toolbar_location="right",
        title=f"Events ({len(initial_events)})",
    )
    node.rug_fig.yaxis.visible = False
    node.rug_fig.ygrid.visible = False
    node.rug_fig.segment(
        x0="x", y0=-0.4, x1="x", y1=0.4,
        source=node.rug_source,
        line_color="#888888", line_width=1, alpha=0.02,
    )

    # P distribution figure — independent x_range, shared with its own rug
    edges0 = np.array([-np.inf, np.inf])
    probs0 = compute_probabilities(edges0, initial_events)
    node.source = ColumnDataSource(make_column_data_source_data(edges0, probs0, use_density=False))

    node.figure = figure(
        width=PLOT_WIDTH, height=380,
        x_range=node.rug_fig.x_range,
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
    _range_cb = CustomJS(args=dict(source=node.source, x_range=node.rug_fig.x_range, y_mode=node.y_mode_radio), code="""
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
    node.rug_fig.x_range.js_on_change('start', _range_cb)
    node.rug_fig.x_range.js_on_change('end',   _range_cb)

    # ── Bin edge controls ────────────────────────────────────────────────
    node.divide_bin_btn = Button(label="Add one bin edge", button_type="default", width=120)
    node.edge_input = TextInput(placeholder="Edge value, then Enter", width=220, visible=False)
    node.edge_status = Div(text="", width=300, styles={"color": "red", "font-size": "13px"})

    node.equal_width_btn = Button(label="Add bin edges", button_type="default", width=120)
    node.equal_width_left_input = TextInput(placeholder="Left", width=80, visible=False)
    node.equal_width_right_input = TextInput(placeholder="Right", width=80, visible=False)
    node.equal_width_count_input = TextInput(placeholder="Count", width=80, visible=False)
    node.equal_width_submit_btn = Button(label="Add evenly spaced edges", button_type="success", width=200, visible=False)
    node.equal_width_edge_at_ends = CheckboxGroup(labels=["also add edges at the start and end of the interval"], active=[], visible=False)
    node.equal_width_preview = Div(text="", width=200, styles={"font-size": "13px", "line-height": "2.2"})
    node.equal_width_status = Div(text="", width=300, styles={"color": "red", "font-size": "13px"})

    # ── Derive controls ──────────────────────────────────────────────────
    node.derive_dropdown = Select(
        value="Surprisal",
        options=["Pass events thru as they are", "Surprisal"],
        width=250,
    )
    node.derive_btn = Button(label="View derived distribution", button_type="primary", width=220)
    node.kl_div_display = Div(text="", width=600, styles={"line-height": "2.2", "margin-left": "10px", "font-size": "13px"})

    # ── Per-node callbacks ───────────────────────────────────────────────

    def on_divide_bin(n=node):
        n.edge_input.visible = not n.edge_input.visible
        n.edge_status.text = ""

    def on_edge_input(attr, old, new, n=node):
        val_str = new.strip()
        if not val_str:
            return
        try:
            val = float(val_str)
        except ValueError:
            n.edge_status.text = f"'{val_str}' is not a valid number."
            n.edge_input.value = ""
            return
        if val in n.interior_edges:
            n.edge_status.text = f"{val} is already a bin edge."
            n.edge_input.value = ""
            return
        n.interior_edges.append(val)
        n.edge_status.text = f"Added bin edge at {val}."
        n.edge_input.value = ""
        n.edge_input.visible = False
        recompute_from(n)

    def on_equal_width_toggle(n=node):
        vis = not n.equal_width_left_input.visible
        n.equal_width_left_input.visible = vis
        n.equal_width_right_input.visible = vis
        n.equal_width_count_input.visible = vis
        n.equal_width_submit_btn.visible = vis
        n.equal_width_edge_at_ends.visible = vis
        n.equal_width_status.text = ""
        n.equal_width_preview.text = ""

    def update_equal_width_preview(n=node):
        try:
            count = int(n.equal_width_count_input.value_input or n.equal_width_count_input.value)
            if count < 1:
                raise ValueError
        except ValueError:
            n.equal_width_preview.text = ""
            return
        include_ends = 0 in n.equal_width_edge_at_ends.active
        try:
            left = float(n.equal_width_left_input.value)
            right = float(n.equal_width_right_input.value)
            step = (right - left) / (count + 1)
            new_edges = [left + step * (i + 1) for i in range(count)]
            if include_ends:
                new_edges.insert(0, left)
                new_edges.append(right)
            new_unique = [e for e in new_edges if e not in n.interior_edges]
        except (ValueError, ZeroDivisionError):
            new_unique = list(range(count))
        total_bins = len(n.interior_edges) + len(new_unique) + 1
        n.equal_width_preview.text = f"→ {total_bins} bins total"

    def on_equal_width_count_change(attr, old, new, n=node):
        update_equal_width_preview(n)

    def on_equal_width_checkbox_change(attr, old, new, n=node):
        update_equal_width_preview(n)

    def on_equal_width_submit(n=node):
        try:
            left = float(n.equal_width_left_input.value)
            right = float(n.equal_width_right_input.value)
            count = int(n.equal_width_count_input.value)
        except ValueError:
            n.equal_width_status.text = "Enter valid numbers for left, right, and count."
            return
        if right <= left:
            n.equal_width_status.text = "Right must be greater than left."
            return
        if count < 1:
            n.equal_width_status.text = "Count must be at least 1."
            return
        include_ends = 0 in n.equal_width_edge_at_ends.active
        step = (right - left) / (count + 1)
        new_edges = [left + step * (i + 1) for i in range(count)]
        if include_ends:
            new_edges.insert(0, left)
            new_edges.append(right)
        added = [e for e in new_edges if e not in n.interior_edges]
        n.interior_edges.extend(added)
        n.equal_width_status.text = f"Added {len(added)} edge(s)."
        n.equal_width_preview.text = ""
        n.equal_width_left_input.visible = False
        n.equal_width_right_input.visible = False
        n.equal_width_count_input.visible = False
        n.equal_width_submit_btn.visible = False
        n.equal_width_edge_at_ends.visible = False
        recompute_from(n)

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
    node.divide_bin_btn.on_click(on_divide_bin)
    node.divide_bin_btn.js_on_click(CustomJS(args=dict(inp=node.edge_input), code="""
        setTimeout(() => {
            const el = inp.el?.querySelector?.('input');
            if (el) el.focus();
        }, 100);
    """))
    node.edge_input.on_change("value", on_edge_input)
    node.equal_width_btn.on_click(on_equal_width_toggle)
    node.equal_width_count_input.on_change("value_input", on_equal_width_count_change)
    node.equal_width_count_input.on_change("value", on_equal_width_count_change)
    node.equal_width_edge_at_ends.on_change("active", on_equal_width_checkbox_change)
    node.equal_width_submit_btn.on_click(on_equal_width_submit)
    node.y_mode_radio.on_change("active", on_y_mode_change)
    node.prior_alpha_slider.on_change("value", on_prior_change)
    node.prior_mu_slider.on_change("value", on_prior_change)
    node.prior_sigma_slider.on_change("value", on_prior_change)
    node.derive_dropdown.on_change("value", on_output_mode_change)
    node.derive_btn.on_click(on_derive)
    node.gang_checkbox.on_change("active", on_propagate_change)

    # ── Layout for this node ─────────────────────────────────────────────
    divide_section = Row(node.divide_bin_btn, node.edge_input, node.edge_status)
    equal_width_inputs_row = Row(
        node.equal_width_left_input,
        node.equal_width_right_input,
        node.equal_width_count_input,
    )
    equal_width_section = Column(
        Row(node.equal_width_btn, node.equal_width_preview, node.equal_width_status),
        equal_width_inputs_row,
        node.equal_width_submit_btn,
        node.equal_width_edge_at_ends,
    )
    edge_panel = Column(divide_section, Spacer(height=10), equal_width_section)

    derive_row = Row(node.derive_dropdown, node.derive_btn, node.gang_checkbox, node.kl_div_display)

    prior_row = Row(node.prior_alpha_slider, Spacer(width=20), node.prior_mu_slider, Spacer(width=20), node.prior_sigma_slider)
    plot_and_edges = Row(node.figure, Spacer(width=20), edge_panel)
    node.layout = Column(prior_row, node.rug_fig, plot_and_edges, node.y_mode_radio, derive_row)

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

    # Append to root layout
    root_col.children.append(new_node.layout)

    # Recompute so it shows a distribution
    recompute_from(new_node)
    if parent_node is not None:
        update_kl_display(parent_node)


# ── Top-level event controls ─────────────────────────────────────────────────

n_events_input = TextInput(value="1000", title="", width=80)
source_select = Select(value=ev.SOURCE_NAMES[0], options=ev.SOURCE_NAMES, width=200)
add_events_btn = Button(label="Add events", button_type="success", width=120)
clear_events_btn = Button(label="Clear events", button_type="warning", width=120, disabled=True)
single_event_input = TextInput(placeholder="Add event at value…", width=200)
single_event_count_input = TextInput(value="1", width=60, title="")
single_event_status = Div(text="", width=200, styles={"color": "red", "font-size": "13px", "line-height": "2.2"})

# Top-level rug plot (raw events, before any PNode)
rug_source = ColumnDataSource(dict(x=[], y=[]))
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

# Initial "View derived distribution" button (before any nodes exist)
initial_derive_btn = Button(label="View derived distribution", button_type="primary", width=220)

history_back_btn = Button(label="◀", width=50, disabled=True)
history_fwd_btn = Button(label="▶", width=50, disabled=True)
history_slider = Slider(start=0, end=1, value=0, step=1, title="", width=350, disabled=True)
history_label = Div(text="Step 0 of 0", styles={"line-height": "2.2", "font-size": "13px"})


def update_transport_state():
    global _transport_cb_guard
    n = len(event_history) - 1
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
    locked = not at_end
    add_events_btn.disabled = locked
    clear_events_btn.disabled = locked or len(root_events) == 0
    single_event_input.disabled = locked


def refresh_rug():
    rug_source.data = dict(x=root_events, y=np.zeros(len(root_events)))
    rug_fig.title.text = f"Events ({len(root_events)})"
    update_transport_state()


def on_add_events():
    global root_events
    try:
        n = int(n_events_input.value)
        if n <= 0:
            raise ValueError
    except ValueError:
        n = 1000
        n_events_input.value = "1000"
    new_ev = ev.get_events(n, source_select.value)
    root_events = np.concatenate([root_events, new_ev])
    push_history(root_events)
    refresh_rug()
    on_make_dist()


def on_make_dist():
    if root_node is None:
        return
    root_node.events = root_events.copy()
    recompute_from(root_node)


def on_clear_events():
    global root_events, event_history, history_index
    root_events = np.array([], dtype=float)
    event_history = [np.array([], dtype=float)]
    history_index = 0
    rug_source.data = dict(x=[], y=[])
    rug_fig.title.text = "Events (0)"
    on_make_dist()
    update_transport_state()


def on_single_event_input(attr, old, new):
    global root_events
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
    root_events = np.concatenate([root_events, np.full(n, val)])
    push_history(root_events)
    single_event_status.text = f"Added {n} event{'s' if n > 1 else ''} at {val}."
    single_event_input.value = ""
    refresh_rug()
    on_make_dist()


def on_initial_derive():
    create_child_node(None)


add_events_btn.on_click(on_add_events)
clear_events_btn.on_click(on_clear_events)
single_event_input.on_change("value", on_single_event_input)
initial_derive_btn.on_click(on_initial_derive)


def push_history(events_arr):
    global event_history, history_index
    event_history = event_history[:history_index + 1]
    event_history.append(events_arr.copy())
    history_index = len(event_history) - 1


def apply_history_index(idx):
    global root_events, history_index
    history_index = max(0, min(idx, len(event_history) - 1))
    root_events = event_history[history_index].copy()
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

top_controls = Row(
    source_select,
    Spacer(width=10),
    add_events_btn,
    Div(text="<b>n =</b>", styles={"line-height": "2.2", "margin-left": "6px"}),
    n_events_input,
    Spacer(width=20),
    clear_events_btn,
    Spacer(width=30),
    single_event_input,
    Div(text="<b>count:</b>", styles={"line-height": "2.2", "margin-left": "6px"}),
    single_event_count_input,
    single_event_status,
)

transport_row = Row(
    history_back_btn,
    Spacer(width=5),
    history_slider,
    Spacer(width=5),
    history_fwd_btn,
    Spacer(width=10),
    history_label,
)

root_col = Column(
    top_controls,
    transport_row,
    rug_fig,
    initial_derive_btn,
)

curdoc().add_root(root_col)
curdoc().title = "Entropy & Surprisal Explorer"
