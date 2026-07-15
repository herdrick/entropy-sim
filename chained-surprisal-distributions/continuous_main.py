"""Continuous (KDE-based) analogue of main.py.

Each node fits a continuous density to its events via a Gaussian-KDE blended
with a Gaussian(mu, sigma) prior (blend weight n/(n+alpha)), instead of a
histogram over user-chosen bin edges. Deriving a child node computes
S(x) = -log2(density(x)) directly at each event -- no bin lookup needed.

Domain note: the root node (P1) lives in "value" units; every derived node
from then on lives in "bits" units (S(P1), S(S(P1)), ... are all self-information
quantities in the same units). KL/W1 comparisons are only mathematically
meaningful between nodes in the same domain, so they're shown between a node
and its parent only when the parent is itself a bits-domain node (depth >= 2),
and between a node and its child always (children of a bits-domain node are
automatically bits-domain too). The root's comparison to its own child is
skipped since value units and bits units aren't the same measure.

The four bin-simplex viz panels from main.py's sibling fixed_point.py have no
continuous analogue and aren't present here either.
"""
import numpy as np
np.set_printoptions(formatter={'float': lambda x: f"{x},"})

from scipy.stats import norm as scipy_norm, gaussian_kde
from scipy.interpolate import CubicSpline
from dataclasses import dataclass, field
from typing import Optional, Callable
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, Div, TextInput, Button, Row, Column, Spacer, Select,
    CheckboxGroup, RadioButtonGroup, Slider, HoverTool, Range1d, Span,
)
import events as ev

# ── Constants ────────────────────────────────────────────────────────────────
X_MIN, X_MAX = -10, 30
S_MIN, S_MAX = 0, 25
GRID_N = 400
PRIOR_ALPHA_DEFAULT = 0
PRIOR_MU_DEFAULT = 0
PRIOR_SIGMA_DEFAULT = 5
BANDWIDTH_DEFAULT = 1.0
GMM_COMPONENTS_DEFAULT = 2
GMM_COMPONENTS_MAX = 8
ADAPTIVE_KDE_SENSITIVITY = 0.5  # exponent on local/global density ratio -> bandwidth scaling
DENSITY_METHODS = [("kde", "KDE"), ("adaptive_kde", "Adaptive KDE"), ("gmm", "GMM"), ("bspline", "B-spline")]
TOOLS = "xpan,xwheel_zoom,xbox_zoom,reset,save"
PLOT_WIDTH = 900

VALUE_GRID = np.linspace(X_MIN, X_MAX, GRID_N)
SURP_GRID = np.linspace(S_MIN, S_MAX, GRID_N)

# ── State ────────────────────────────────────────────────────────────────────
root_events = np.array([], dtype=float)
root_node: Optional["PNode"] = None
all_events: np.ndarray = np.array([], dtype=float)
history_index: int = 0
_transport_cb_guard: bool = False
_column_count: int = 1
_all_nodes: list = []
_step_cb_handle: list = [None]
_trace_indices: list = []

TRACE_PALETTE = ["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"]

# ── Busy indicator ───────────────────────────────────────────────────────────
# Bokeh only pushes document changes to the browser after a callback returns,
# so to actually *see* "Working…" while a slow recomputation runs, the work
# has to be deferred to the next tick: show the indicator now, run the real
# callback body on the next tick, then clear it.
busy_div = Div(text="⏳ Working…", visible=False, styles={
    "position": "fixed", "top": "12px", "left": "50%", "transform": "translateX(-50%)",
    "z-index": "9999", "background": "#fff7ed", "border": "1px solid #b45309",
    "border-radius": "6px", "padding": "6px 16px", "box-shadow": "0 2px 8px rgba(0,0,0,0.15)",
    "color": "#b45309", "font-weight": "bold", "font-size": "14px",
})


def busy(fn):
    def wrapped():
        busy_div.visible = True

        def run():
            try:
                fn()
            finally:
                busy_div.visible = False
        curdoc().add_next_tick_callback(run)
    return wrapped


def busy_change(fn):
    def wrapped(attr, old, new):
        busy_div.visible = True

        def run():
            try:
                fn(attr, old, new)
            finally:
                busy_div.visible = False
        curdoc().add_next_tick_callback(run)
    return wrapped


@dataclass
class PNode:
    events: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    figure: object = None
    source: ColumnDataSource = None
    child: Optional["PNode"] = None
    parent: Optional["PNode"] = None
    depth: int = 0
    derive_btn: Button = None
    prior_alpha_slider: Slider = None
    prior_mu_slider: Slider = None
    prior_sigma_slider: Slider = None
    bandwidth_slider: Slider = None
    method_select: Select = None
    n_components_slider: Slider = None
    y_scale_toggle: Select = None
    kl_div_display: Div = None
    current_density: object = None
    grid: np.ndarray = None
    layout: Column = None
    propagates: bool = False
    gang_checkbox: CheckboxGroup = None
    y_range_adaptive: bool = True
    trace_source: ColumnDataSource = None
    trace_spans: list = field(default_factory=list)
    rug_source: ColumnDataSource = None
    rug_glyph: object = None


# ── Density helpers ────────────────────────────────────────────────────────

def _em_gmm_pdf(events, n_components):
    """Fit a 1-D Gaussian mixture via EM. Refit from scratch on every call, so
    this stays cheap and deterministic: a fixed small iteration count, and
    components seeded from quantile splits of the sorted events rather than
    random init, so repeated refits on the same data land in the same place."""
    n = len(events)
    k = max(1, min(n_components, n))
    splits = np.array_split(np.sort(events), k)
    means = np.array([np.mean(s) for s in splits])
    stds = np.array([max(np.std(s), 1e-2) for s in splits])
    weights = np.full(k, 1.0 / k)
    x = events[:, None]
    for _ in range(25):
        comp = weights[None, :] * scipy_norm.pdf(x, means[None, :], stds[None, :])
        denom = np.clip(comp.sum(axis=1, keepdims=True), 1e-300, None)
        resp = comp / denom
        Nk = np.clip(resp.sum(axis=0), 1e-8, None)
        means = (resp * x).sum(axis=0) / Nk
        var = (resp * (x - means[None, :]) ** 2).sum(axis=0) / Nk
        stds = np.sqrt(np.maximum(var, 1e-4))
        weights = Nk / n

    def pdf(xq):
        xq = np.asarray(xq, dtype=float)
        flat = xq.reshape(-1, 1)
        vals = (weights[None, :] * scipy_norm.pdf(flat, means[None, :], stds[None, :])).sum(axis=1)
        return vals.reshape(xq.shape)
    return pdf


def _adaptive_kde_pdf(events, bw_factor):
    """Balloon-style KDE: a pilot fixed-bandwidth KDE gives a rough density at
    each event, then each event's own kernel bandwidth is shrunk in dense
    regions and widened in sparse ones (by (local/geometric-mean-density) **
    -ADAPTIVE_KDE_SENSITIVITY), instead of one bandwidth for the whole
    dataset."""
    pilot = gaussian_kde(events, bw_method=lambda k: k.scotts_factor() * bw_factor)
    pilot_at_events = np.clip(pilot(events), 1e-300, None)
    g = np.exp(np.mean(np.log(pilot_at_events)))
    h0 = float(np.sqrt(pilot.covariance[0, 0]))
    local_h = np.clip(h0 * (pilot_at_events / g) ** (-ADAPTIVE_KDE_SENSITIVITY), 1e-3, None)

    def pdf(xq):
        xq = np.asarray(xq, dtype=float)
        flat = xq.reshape(-1, 1)
        z = (flat - events[None, :]) / local_h[None, :]
        vals = (np.exp(-0.5 * z ** 2) / (local_h[None, :] * np.sqrt(2 * np.pi))).mean(axis=1)
        return vals.reshape(xq.shape)
    return pdf


def _bspline_pdf(events, bw_factor):
    """Bin events into a histogram, Gaussian-smooth the bin heights (kernel
    width in bins set by bw_factor), then fit a cubic spline through the
    sqrt of the smoothed heights -- squaring on evaluation guarantees the
    result stays non-negative without constrained optimization. Cheap to
    refit (histogram + spline through O(sqrt(n)) points, not O(n)).

    Unlike KDE/GMM/Adaptive-KDE, this has compact support: outside the
    padded histogram range the density is exactly 0, not a shrinking tail,
    so with no prior (alpha=0) an event far outside the observed range gets
    surprisal clipped to the -log2(1e-300) ceiling rather than a large but
    finite value."""
    n = len(events)
    n_bins = max(6, min(60, int(np.sqrt(n)) + 2))
    lo, hi = np.min(events), np.max(events)
    pad = (hi - lo) * 0.15 + 1e-6
    edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
    counts, edges = np.histogram(events, bins=edges)
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    hist_density = counts / (n * widths)

    sigma_bins = max(0.6, bw_factor * 1.5)
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    k = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (k / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(hist_density, radius, mode="constant")
    smoothed = np.convolve(padded, kernel, mode="valid")

    spline = CubicSpline(centers, np.sqrt(np.clip(smoothed, 0, None)), extrapolate=False)

    def raw(xq):
        xq = np.asarray(xq, dtype=float)
        vals = spline(xq)
        vals = np.where(np.isnan(vals), 0.0, vals)
        return np.clip(vals, 0, None) ** 2

    grid = np.linspace(edges[0], edges[-1], 400)
    mass = np.trapezoid(raw(grid), grid)
    mass = mass if mass > 1e-12 else 1.0
    return lambda xq: raw(xq) / mass


def make_density_fn(events, alpha, mu, sigma, method, bw_factor, n_components) -> Callable[[np.ndarray], np.ndarray]:
    prior_pdf = lambda x: scipy_norm.pdf(np.asarray(x, dtype=float), loc=mu, scale=sigma)
    n = len(events)
    if n == 0:
        return lambda x: np.zeros_like(np.asarray(x, dtype=float))
    if n == 1 or np.ptp(events) < 1e-9:
        center = float(np.mean(events))
        narrow = max(sigma, 1e-2) * 0.05
        raw_pdf = lambda x: scipy_norm.pdf(np.asarray(x, dtype=float), loc=center, scale=narrow)
    elif method == "gmm":
        raw_pdf = _em_gmm_pdf(events, n_components)
    elif method == "adaptive_kde":
        raw_pdf = _adaptive_kde_pdf(events, bw_factor)
    elif method == "bspline":
        raw_pdf = _bspline_pdf(events, bw_factor)
    else:
        kde = gaussian_kde(events, bw_method=lambda k: k.scotts_factor() * bw_factor)
        raw_pdf = lambda x: kde(np.asarray(x, dtype=float))
    if alpha <= 0:
        return raw_pdf
    w = n / (n + alpha)
    return lambda x: w * raw_pdf(x) + (1 - w) * prior_pdf(x)


def surprisal_bits(x, density_fn):
    # This treats density(x) directly as if it were a probability -- there's no
    # bin lookup / bin width anywhere. That's the differential-entropy
    # convention (matches differential_entropy_bits below, -int p*log2(p) dx),
    # not the discrete-Shannon one, and it comes with real caveats:
    #
    # - A true discretized surprisal would be S = -log2(P) where
    #   P ~= density(x) * dx for some bin width dx. Using density(x) alone is
    #   equivalent to silently fixing dx = 1 in whatever units x is in.
    # - Not scale-invariant: rescale x (e.g. events *= 10) and density scales
    #   by 1/10, shifting every surprisal value by a constant log2(10) bits.
    #   Absolute numbers only mean something relative to x's units; comparisons
    #   *within* one density_fn are still meaningful.
    # - density(x) can exceed 1 (e.g. a narrow Gaussian), so surprisal can go
    #   negative -- unlike discrete Shannon surprisal, which is always >= 0.
    # - Strictly speaking, a continuous RV has P(X=x) = 0 at any single point,
    #   so there's no well-defined "information content of this exact event";
    #   this is a differential-entropy analogue, not that quantity.
    p = np.clip(density_fn(x), 1e-300, None)
    return -np.log2(p)


def differential_entropy_bits(density_fn, grid):
    p = np.clip(density_fn(grid), 1e-300, None)
    return float(np.trapezoid(-p * np.log2(p), grid))


def kl_divergence_bits(p_fn, q_fn, grid):
    p = np.clip(p_fn(grid), 1e-300, None)
    q = np.clip(q_fn(grid), 1e-300, None)
    integrand = np.where(p > 1e-12, p * np.log2(p / q), 0.0)
    return float(np.trapezoid(integrand, grid))


def wasserstein_distance(p_fn, q_fn, grid):
    p = np.clip(p_fn(grid), 0, None)
    q = np.clip(q_fn(grid), 0, None)
    p_mass = np.trapezoid(p, grid)
    q_mass = np.trapezoid(q, grid)
    if p_mass <= 0 or q_mass <= 0:
        return float("nan")
    p = p / p_mass
    q = q / q_mass
    dx = np.diff(grid)
    F_p = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * dx)])
    F_q = np.concatenate([[0.0], np.cumsum((q[1:] + q[:-1]) / 2 * dx)])
    return float(np.trapezoid(np.abs(F_p - F_q), grid))


def _param_row(nd):
    """Only show the slider(s) relevant to the currently-selected fit method:
    bandwidth for the two KDE variants, component count for GMM."""
    children = [nd.prior_alpha_slider, Spacer(width=20), nd.prior_mu_slider, Spacer(width=20),
                nd.prior_sigma_slider]
    if nd.method_select.value in ("kde", "adaptive_kde", "bspline"):
        children += [Spacer(width=20), nd.bandwidth_slider]
    if nd.method_select.value == "gmm":
        children += [Spacer(width=20), nd.n_components_slider]
    children += [Spacer(width=20), nd.method_select]
    return Row(*children)


def rebuild_grid():
    n = _column_count
    for nd in _all_nodes:
        nd.figure.width = PLOT_WIDTH
        for s in (nd.prior_alpha_slider, nd.prior_mu_slider, nd.prior_sigma_slider,
                  nd.bandwidth_slider, nd.n_components_slider):
            s.width = 250
        nd.layout.children[0] = _param_row(nd)
        nd.layout.children[1] = nd.figure
        nd.kl_div_display.width = None
        nd.layout.children[3] = Row(
            nd.derive_btn, nd.gang_checkbox, nd.kl_div_display,
        )
    base = root_col.children[:5]
    node_rows = []
    for i in range(0, len(_all_nodes), n):
        chunk = _all_nodes[i:i + n]
        node_rows.append(Row(*[nd.layout for nd in chunk], spacing=0, sizing_mode="fixed") if len(chunk) > 1 else chunk[0].layout)
    root_col.children = base + node_rows


# ── Core recomputation ───────────────────────────────────────────────────────

def propagate_params_down(nd):
    child = nd.child
    if child is None:
        return
    child.prior_alpha_slider.value = nd.prior_alpha_slider.value
    child.prior_mu_slider.value = nd.prior_mu_slider.value
    child.prior_sigma_slider.value = nd.prior_sigma_slider.value
    child.bandwidth_slider.value = nd.bandwidth_slider.value
    child.n_components_slider.value = nd.n_components_slider.value
    child.method_select.value = nd.method_select.value
    recompute_from(child)
    propagate_params_down(child)


def recompute_from(nd):
    if nd is None:
        return
    alpha = nd.prior_alpha_slider.value
    mu = nd.prior_mu_slider.value
    sigma = nd.prior_sigma_slider.value
    bw = nd.bandwidth_slider.value
    density_fn = make_density_fn(nd.events, alpha, mu, sigma,
                                  nd.method_select.value, bw, nd.n_components_slider.value)
    nd.current_density = density_fn
    y = density_fn(nd.grid)
    nd.source.data = dict(x=nd.grid, y=y)
    if nd.y_range_adaptive:
        nd.figure.y_range.end = float(np.max(y)) * 1.05 if len(y) else 1.0
    else:
        nd.figure.y_range.end = 1.0
    label = "P1" if nd.depth == 0 else f"P{nd.depth + 1}"
    nd.figure.title.text = f"{label}  |  entropy = {differential_entropy_bits(density_fn, nd.grid):.4f} bits"

    rug_h = nd.figure.y_range.end * 0.03
    nd.rug_source.data = dict(x=nd.events, y0=np.zeros(len(nd.events)), y1=np.full(len(nd.events), rug_h))

    if nd.child is not None:
        nd.child.events = surprisal_bits(nd.events, density_fn)
        recompute_from(nd.child)

    refresh_kl_display(nd)
    if nd.parent is not None:
        refresh_kl_display(nd.parent)


_KL_LINK = '<a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence" target="_blank">KL</a>'
_W1_LINK = '<a href="https://en.wikipedia.org/wiki/Wasserstein_metric" target="_blank">W1</a>'


def refresh_kl_display(nd):
    """Show KL/W1 to parent (up-arrow) and child (down-arrow) when both sides
    share the same units (see module docstring): comparisons to a parent only
    apply when the parent is itself bits-domain (depth >= 2); comparisons to
    a child always apply once this node is itself bits-domain (depth >= 1),
    since a bits-domain node's child is automatically bits-domain too."""
    if nd.kl_div_display is None or nd.current_density is None:
        return
    lines = []
    if nd.depth >= 2 and nd.parent is not None and nd.parent.current_density is not None:
        kl_up = kl_divergence_bits(nd.current_density, nd.parent.current_density, SURP_GRID)
        w1_up = wasserstein_distance(nd.current_density, nd.parent.current_density, SURP_GRID)
        lines.append(f"↑ {_KL_LINK} {kl_up:.4f} bits &nbsp; {_W1_LINK} {w1_up:.4f}")
    if nd.depth >= 1 and nd.child is not None and nd.child.current_density is not None:
        kl_down = kl_divergence_bits(nd.current_density, nd.child.current_density, SURP_GRID)
        w1_down = wasserstein_distance(nd.current_density, nd.child.current_density, SURP_GRID)
        lines.append(f"↓ {_KL_LINK} {kl_down:.4f} bits &nbsp; {_W1_LINK} {w1_down:.4f}")
    nd.kl_div_display.text = "<br>".join(lines)


def refresh_trace_display():
    """Mark the currently-traced event indices on every existing node's curve
    (a vertical dashed line + point at that node's value for the event) and
    summarize the per-node chain of values in trace_summary_div."""
    for nd in _all_nodes:
        for sp in nd.trace_spans:
            try:
                nd.figure.center.remove(sp)
            except ValueError:
                pass
        nd.trace_spans = []
        xs, ys, colors = [], [], []
        for i, idx in enumerate(_trace_indices):
            if idx >= len(nd.events) or nd.current_density is None:
                continue
            color = TRACE_PALETTE[i % len(TRACE_PALETTE)]
            x_val = float(nd.events[idx])
            y_val = float(nd.current_density(np.array([x_val]))[0])
            xs.append(x_val)
            ys.append(y_val)
            colors.append(color)
            span = Span(location=x_val, dimension="height", line_color=color,
                        line_dash="dashed", line_width=2)
            nd.figure.add_layout(span)
            nd.trace_spans.append(span)
        nd.trace_source.data = dict(x=xs, y=ys, color=colors)

    lines = []
    for i, idx in enumerate(_trace_indices):
        color = TRACE_PALETTE[i % len(TRACE_PALETTE)]
        chain = []
        for nd in _all_nodes:
            if idx >= len(nd.events):
                break
            label = "P1" if nd.depth == 0 else f"P{nd.depth + 1}"
            unit = "" if nd.depth == 0 else " bits"
            chain.append(f"{label}={nd.events[idx]:.3f}{unit}")
        if chain:
            lines.append(f'<span style="color:{color}">●</span> ' + " &rarr; ".join(chain))
    trace_summary_div.text = "<br>".join(lines)


# ── PNode factory ────────────────────────────────────────────────────────────

def make_p_node(initial_events, depth):
    nd = PNode()
    nd.events = initial_events
    nd.depth = depth
    is_surprisal = depth > 0
    nd.grid = SURP_GRID if is_surprisal else VALUE_GRID
    x_range = (S_MIN, S_MAX) if is_surprisal else (X_MIN, X_MAX)
    x_label = "Surprisal (bits)" if is_surprisal else "Value"

    nd.source = ColumnDataSource(data=dict(x=[], y=[]))
    nd.figure = figure(
        width=PLOT_WIDTH, height=380,
        x_range=x_range,
        y_range=Range1d(0, 1),
        tools=TOOLS, toolbar_location="right",
        title="P  |  entropy = 0.0000 bits",
    )
    nd.figure.varea(x="x", y1=0, y2="y", source=nd.source, fill_color="#4878CF", fill_alpha=0.4)
    line_renderer = nd.figure.line(x="x", y="y", source=nd.source, line_color="#2255AA", line_width=2)
    hover = HoverTool(renderers=[line_renderer], mode="vline", tooltips=[
        ("x", "@x{0.000}"),
        ("density", "@y{0.0000}"),
    ])
    nd.figure.add_tools(hover)
    nd.trace_source = ColumnDataSource(data=dict(x=[], y=[], color=[]))
    nd.figure.scatter(x="x", y="y", source=nd.trace_source, color="color", size=9,
                       line_color="black", line_width=1, level="overlay")
    nd.rug_source = ColumnDataSource(data=dict(x=[], y0=[], y1=[]))
    nd.rug_glyph = nd.figure.segment(
        x0="x", y0="y0", x1="x", y1="y1", source=nd.rug_source,
        line_color="#444444", line_width=1,
        line_alpha=rug_alpha_slider.value, visible=(0 in rug_checkbox.active),
    )
    nd.figure.xgrid.grid_line_color = None
    nd.figure.ygrid.grid_line_color = None
    nd.figure.xaxis.axis_label = x_label
    nd.figure.yaxis.axis_label = "Density"

    nd.y_scale_toggle = Select(value="adaptive",
                                options=[("fixed", "Y: fixed 0–1"), ("adaptive", "Y: adaptive")],
                                width=140)
    nd.gang_checkbox = CheckboxGroup(labels=["Control all descendants' parameters"], active=[])

    alpha_end = 1000 if not is_surprisal else 50
    mu_range = (-10, 10) if not is_surprisal else (0, 20)
    sigma_range = (0.1, 20) if not is_surprisal else (0.1, 10)
    nd.prior_alpha_slider = Slider(start=0, end=alpha_end, value=PRIOR_ALPHA_DEFAULT, step=1, title="Prior strength α", width=250)
    nd.prior_mu_slider = Slider(start=mu_range[0], end=mu_range[1], value=PRIOR_MU_DEFAULT, step=0.1, title="Prior mean μ", width=250)
    nd.prior_sigma_slider = Slider(start=sigma_range[0], end=sigma_range[1], value=PRIOR_SIGMA_DEFAULT, step=0.1, title="Prior std dev σ", width=250)
    nd.bandwidth_slider = Slider(start=0.1, end=3.0, value=BANDWIDTH_DEFAULT, step=0.05, title="Bandwidth / smoothing factor", width=250)
    nd.n_components_slider = Slider(start=1, end=GMM_COMPONENTS_MAX, value=GMM_COMPONENTS_DEFAULT, step=1, title="GMM components", width=250)
    nd.method_select = Select(value="kde", options=DENSITY_METHODS, title="Fit method", width=140)

    nd.derive_btn = Button(label="View derived distribution", button_type="primary", width=220)
    nd.kl_div_display = Div(text="", styles={"line-height": "2.2", "margin-left": "10px", "font-size": "13px"})

    def on_param_change(attr, old, new, n=nd):
        recompute_from(n)
        if n.propagates:
            propagate_params_down(n)
        refresh_trace_display()

    def on_method_change(attr, old, new, n=nd):
        n.layout.children[0] = _param_row(n)
        recompute_from(n)
        if n.propagates:
            propagate_params_down(n)
        refresh_trace_display()

    def on_y_scale_toggle(attr, old, new, n=nd):
        n.y_range_adaptive = (new == "adaptive")
        recompute_from(n)

    def on_propagate_change(attr, old, new, n=nd):
        n.propagates = 0 in new
        if n.propagates:
            propagate_params_down(n)

    def on_derive(n=nd):
        create_child_node(n)

    for s in (nd.prior_alpha_slider, nd.prior_mu_slider, nd.prior_sigma_slider,
              nd.bandwidth_slider, nd.n_components_slider):
        s.on_change("value", on_param_change)
    nd.method_select.on_change("value", on_method_change)
    nd.y_scale_toggle.on_change("value", on_y_scale_toggle)
    nd.gang_checkbox.on_change("active", on_propagate_change)
    nd.derive_btn.on_click(busy(on_derive))

    derive_row = Row(nd.derive_btn, nd.gang_checkbox, nd.kl_div_display)
    nd.layout = Column(_param_row(nd), nd.figure, Row(nd.y_scale_toggle), derive_row)

    return nd


def create_child_node(parent_node):
    global root_node

    if parent_node is not None:
        alpha = parent_node.prior_alpha_slider.value
        mu = parent_node.prior_mu_slider.value
        sigma = parent_node.prior_sigma_slider.value
        bw = parent_node.bandwidth_slider.value
        density_fn = make_density_fn(parent_node.events, alpha, mu, sigma,
                                      parent_node.method_select.value, bw, parent_node.n_components_slider.value)
        child_events = surprisal_bits(parent_node.events, density_fn)
        depth = parent_node.depth + 1
    else:
        child_events = root_events.copy()
        depth = 0

    new_node = make_p_node(child_events, depth)

    if parent_node is not None:
        parent_node.child = new_node
        new_node.parent = parent_node
        parent_node.derive_btn.disabled = True
        new_node.propagates = parent_node.propagates
        new_node.gang_checkbox.active = list(parent_node.gang_checkbox.active)
        if parent_node.propagates:
            propagate_params_down(parent_node)
    else:
        root_node = new_node
        initial_derive_btn.disabled = True

    _all_nodes.append(new_node)
    rebuild_grid()
    recompute_from(new_node)
    refresh_trace_display()


# ── Top-level event controls (same as main.py) ────────────────────────────────

n_events_input = TextInput(value="100", title="", width=80)
family_select = Select(value=ev.FAMILY_NAMES[0], options=ev.FAMILY_NAMES, width=150)
append_replace_radio = RadioButtonGroup(labels=["Append", "Replace"], active=0)
_current_param_sliders: list = []
dist_params_row = Row()
add_events_btn = Button(label="Add events", button_type="success", width=120)
add_events_one_by_one_btn = Button(label="Add events (one by one)", button_type="success", width=190)
clear_events_btn = Button(label="Clear events", button_type="warning", width=120, disabled=True)
single_event_input = TextInput(placeholder="Add event at value…", width=200)
single_event_count_input = TextInput(value="1", width=60, title="")
single_event_status = Div(text="", width=200, styles={"color": "red", "font-size": "13px", "line-height": "2.2"})
trace_checkbox = CheckboxGroup(labels=["Trace new events"], active=[])
clear_traces_btn = Button(label="Clear traces", button_type="default", width=110)
trace_summary_div = Div(text="", styles={"font-size": "13px", "line-height": "1.8"})
rug_checkbox = CheckboxGroup(labels=["Show event rug"], active=[])
rug_alpha_slider = Slider(start=0.0, end=1.0, value=0.15, step=0.01, title="Rug opacity", width=200)


def on_rug_checkbox_change(attr, old, new):
    visible = 0 in new
    for nd in _all_nodes:
        nd.rug_glyph.visible = visible


def on_rug_alpha_change(attr, old, new):
    for nd in _all_nodes:
        nd.rug_glyph.glyph.line_alpha = new


rug_checkbox.on_change("active", on_rug_checkbox_change)
rug_alpha_slider.on_change("value", on_rug_alpha_change)


def set_trace_new_indices(n):
    """Add the last n events in root_events to the traced set, if tracing is on.
    Traces accumulate across add-actions and are only ever cleared via the
    explicit Clear traces button."""
    global _trace_indices
    if 0 in trace_checkbox.active and n > 0:
        start = max(0, len(root_events) - n)
        _trace_indices = _trace_indices + list(range(start, len(root_events)))


def on_clear_traces():
    global _trace_indices
    _trace_indices = []
    refresh_trace_display()


clear_traces_btn.on_click(on_clear_traces)

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
    _transport_cb_guard = True
    history_slider.end = max(n, 1)
    history_slider.value = history_index
    history_slider.disabled = n == 0
    _transport_cb_guard = False
    history_label.text = f"Step {history_index} of {n}"
    history_back_btn.disabled = history_index == 0
    history_fwd_btn.disabled = history_index == n
    clear_events_btn.disabled = len(all_events) == 0


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
        n = 100
        n_events_input.value = "100"
    was_at_end = history_index == len(all_events)
    new_ev = ev.get_events(n, family_select.value, get_current_params())
    all_events = np.concatenate([all_events, new_ev])
    if was_at_end:
        history_index = len(all_events)
    root_events = all_events[:history_index].copy()
    set_trace_new_indices(n)
    update_transport_state()
    on_make_dist()


def on_make_dist():
    if root_node is None:
        return
    root_node.events = root_events.copy()
    recompute_from(root_node)
    refresh_trace_display()


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
    set_trace_new_indices(n)
    update_transport_state()
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
        n = 100
    new_ev = ev.get_events(n, family_select.value, get_current_params())
    all_events = new_ev.copy()
    history_index = len(all_events)
    root_events = all_events.copy()
    set_trace_new_indices(n)
    update_transport_state()
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


add_events_btn.on_click(busy(on_add_events))
clear_events_btn.on_click(busy(on_clear_events))
single_event_input.on_change("value", busy_change(on_single_event_input))
initial_derive_btn.on_click(busy(on_initial_derive))
family_select.on_change("value", on_family_change)

_current_param_sliders = make_param_sliders(ev.FAMILY_NAMES[0])
dist_params_row.children = list(_current_param_sliders)
for _s in _current_param_sliders:
    _s.on_change("value", on_param_slider_change)


def apply_history_index(idx):
    global root_events, history_index
    history_index = max(0, min(idx, len(all_events)))
    root_events = all_events[:history_index].copy()
    update_transport_state()
    on_make_dist()


def on_history_slider_change(attr, old, new):
    if _transport_cb_guard:
        return
    apply_history_index(int(new))


history_slider.on_change("value", busy_change(on_history_slider_change))
history_back_btn.on_click(busy(lambda: apply_history_index(history_index - 1)))
history_fwd_btn.on_click(busy(lambda: apply_history_index(history_index + 1)))


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
        n = 100
        n_events_input.value = "100"

    new_ev = ev.get_events(n, family_select.value, get_current_params())
    all_events = np.concatenate([all_events, new_ev])
    target_index = len(all_events)

    add_events_one_by_one_btn.label = "Stop"

    def step():
        if _step_cb_handle[0] is None:
            add_events_one_by_one_btn.label = "Add events (one by one)"
            return
        apply_history_index(history_index + 1)
        set_trace_new_indices(1)
        refresh_trace_display()
        if history_index >= target_index:
            _step_cb_handle[0] = None
            add_events_one_by_one_btn.label = "Add events (one by one)"
        else:
            _step_cb_handle[0] = curdoc().add_next_tick_callback(step)

    _step_cb_handle[0] = curdoc().add_next_tick_callback(step)


add_events_one_by_one_btn.on_click(busy(on_add_events_one_by_one))

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
        Spacer(width=10),
        add_events_one_by_one_btn,
        Div(text="<b>n =</b>", styles={"line-height": "2.2", "margin-left": "6px"}),
        n_events_input,
        Spacer(width=20),
        clear_events_btn,
        Spacer(width=20),
        trace_checkbox,
        clear_traces_btn,
    ),
    Row(
        single_event_input,
        Div(text="<b>count:</b>", styles={"line-height": "2.2", "margin-left": "6px"}),
        single_event_count_input,
        single_event_status,
        Spacer(width=30),
        Div(text="<b>Layout:</b>", styles={"line-height": "2.2"}),
        col_count_radio,
        Spacer(width=30),
        rug_checkbox,
        rug_alpha_slider,
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
    busy_div,
    top_controls,
    transport_row,
    trace_summary_div,
    initial_derive_btn,
)

curdoc().add_root(root_col)
curdoc().title = "Entropy & Surprisal Explorer (Continuous)"
