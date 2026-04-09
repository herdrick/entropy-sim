# run with "bokeh serve start-with-detailed-prompt/foo.py --dev"
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, CustomJS, Div, TextInput, Button, Row, Column, Spacer, Select,
    CheckboxGroup, RadioGroup, Slider,
)
import events as ev

# ── Constants ────────────────────────────────────────────────────────────────
X_MIN, X_MAX = -10, 10
LAPLACE_ALPHA_DEFAULT = 1  # pseudocount per bin
TOOLS = "xpan,xwheel_zoom,xbox_zoom,reset,save"
PLOT_WIDTH = 900

# ── State ────────────────────────────────────────────────────────────────────
root_events = np.array([], dtype=float)   # accumulated raw events
root_node: Optional["PNode"] = None       # head of the singly-linked list


@dataclass
class PNode:
    output_mode: str = "passthru"       # "passthru" | "surprisal"
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
    laplace_slider: Slider = None
    kl_div_display: Div = None
    current_edges: np.ndarray = None
    current_probs: np.ndarray = None
    layout: Column = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def bin_counts(edges, event_arr):
    n_bins = len(edges) - 1
    interior = edges[1:-1]
    if len(event_arr) > 0:
        indices = np.searchsorted(interior, event_arr)
        return np.bincount(indices, minlength=n_bins).astype(float)
    return np.zeros(n_bins)


def compute_probabilities(edges, event_arr, alpha=LAPLACE_ALPHA_DEFAULT):
    counts = bin_counts(edges, event_arr)
    smoothed = counts + alpha
    return smoothed / smoothed.sum()


def make_column_data_source_data(edges, probs, x_start=X_MIN, x_end=X_MAX, use_density=True):
    lefts = edges[:-1]
    rights = edges[1:]
    left_inf  = np.isneginf(lefts).astype(int)
    right_inf = np.isposinf(rights).astype(int)
    dl = np.where(left_inf,  x_start, lefts)
    dr = np.where(right_inf, x_end,   rights)
    widths = dr - dl
    density = np.where(widths > 0, probs / widths, 0.0)
    return dict(
        left=dl, right=dr, top=density if use_density else probs, prob=probs, density=density,
        center=(dl + dr) / 2, width=widths,
        color=bar_colors(len(probs)),
        left_inf=left_inf, right_inf=right_inf,
    )


def entropy_bits(probs):
    p = np.array(probs)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def kl_divergence_bits(p_edges, p_probs, q_edges, q_probs):
    """D_KL(P||Q) in bits, or None if undefined.

    Defined when every non-zero-prob bin of P has a matching bin in Q
    (same two edges) whose probability is also non-zero. Laplace smoothing
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

def recompute_from(node):
    if node is None:
        return
    edges = np.array([-np.inf] + sorted(node.interior_edges) + [np.inf])
    alpha = node.laplace_slider.value if node.laplace_slider is not None else LAPLACE_ALPHA_DEFAULT
    probs = compute_probabilities(edges, node.events, alpha=alpha)
    node.current_edges = edges
    node.current_probs = probs
    use_density = node.y_mode_radio is not None and node.y_mode_radio.active == 1
    node.source.data = make_column_data_source_data(
        edges, probs,
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
    node.figure.quad(
        left="left", right="right", top="top", bottom=0,
        source=node.source,
        fill_color="color", line_color="black", alpha=0.8,
    )
    # Vertical lines at bin edges (full plot height)
    node.edge_line_source = ColumnDataSource(dict(x=[]))
    node.figure.ray(x="x", y=0, length=0, angle=np.pi/2,
                    source=node.edge_line_source,
                    line_color="black", line_width=1)

    node.figure.xaxis.axis_label = "Value"
    node.figure.yaxis.axis_label = "Probability"

    # Y-mode radio: probability vs probability density
    node.y_mode_radio = RadioGroup(labels=["Probability", "Probability density"], active=0, inline=True)

    # Laplace smoothing alpha slider
    node.laplace_slider = Slider(
        start=0, end=5, value=LAPLACE_ALPHA_DEFAULT, step=0.1,
        title="Laplace smoothing α", width=250,
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
        data['left']   = left;
        data['right']  = right;
        data['center'] = center;
        data['width']  = width;
        data['density'] = density;
        data['top']    = top;
        source.change.emit();
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
        value="Pass events thru as they are",
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

    def on_laplace_alpha_change(attr, old, new, n=node):
        recompute_from(n)

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
    node.laplace_slider.on_change("value", on_laplace_alpha_change)
    node.derive_dropdown.on_change("value", on_output_mode_change)
    node.derive_btn.on_click(on_derive)

    # ── Layout for this node ─────────────────────────────────────────────
    divide_row = Row(node.divide_bin_btn, node.edge_input, node.edge_status)
    equal_width_row = Row(
        node.equal_width_btn, node.equal_width_left_input,
        node.equal_width_right_input,
        node.equal_width_count_input, node.equal_width_submit_btn,
        node.equal_width_edge_at_ends,
        node.equal_width_preview, node.equal_width_status,
    )
    derive_row = Row(node.derive_dropdown, node.derive_btn, node.kl_div_display)

    y_mode_row = Row(node.y_mode_radio, Spacer(width=30), node.laplace_slider)
    node.layout = Column(node.rug_fig, node.figure, y_mode_row, divide_row, equal_width_row, derive_row)

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
            alpha = parent_node.laplace_slider.value if parent_node.laplace_slider else LAPLACE_ALPHA_DEFAULT
            probs = compute_probabilities(edges, parent_node.events, alpha=alpha)
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
add_events_btn = Button(label="Add events", button_type="success", width=120)
make_dist_btn = Button(label="Make distribution from events", button_type="primary", width=240, disabled=True)
clear_events_btn = Button(label="Clear events", button_type="warning", width=120, disabled=True)

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


def refresh_rug():
    rug_source.data = dict(x=root_events, y=np.zeros(len(root_events)))
    rug_fig.title.text = f"Events ({len(root_events)})"
    has_events = len(root_events) > 0
    make_dist_btn.disabled = not has_events
    clear_events_btn.disabled = not has_events


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
    if root_node is None:
        return
    root_node.events = root_events.copy()
    recompute_from(root_node)


def on_clear_events():
    global root_events
    root_events = np.array([], dtype=float)
    rug_source.data = dict(x=[], y=[])
    rug_fig.title.text = "Events (0)"
    make_dist_btn.disabled = True
    clear_events_btn.disabled = True


def on_initial_derive():
    create_child_node(None)


add_events_btn.on_click(on_add_events)
make_dist_btn.on_click(on_make_dist)
clear_events_btn.on_click(on_clear_events)
initial_derive_btn.on_click(on_initial_derive)

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

root_col = Column(
    top_controls,
    rug_fig,
    initial_derive_btn,
)

curdoc().add_root(root_col)
curdoc().title = "Entropy & Surprisal Explorer"
