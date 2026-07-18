"""Continuous (KDE-based) analogue of fixed_point.py.

Instead of binning events into a histogram, each node fits a continuous
density via a Gaussian-KDE blended with a Gaussian(mu, sigma) prior
(blend weight n/(n+alpha), the continuous analogue of additive/pseudocount
smoothing). Surprisal S(x) = -log2(density(x) * dx) is evaluated directly at
each event, where dx is a user-adjustable bin width (defaults to 1) standing
in for an actual bin lookup, and the fixed-point iteration repeats that
transform + refit until the density stops changing on a fixed grid.

The four bin-simplex viz panels (simplex3d, radial, scatter matrix, parallel
coords) from fixed_point.py have no continuous analogue (they plot vectors
of per-bin probabilities) and are intentionally dropped here.
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
    RadioButtonGroup, Slider, HoverTool, Range1d, CheckboxGroup,
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
BIN_WIDTH_DEFAULT = 1.0
GMM_COMPONENTS_DEFAULT = 2
GMM_COMPONENTS_MAX = 8
ADAPTIVE_KDE_SENSITIVITY = 0.5  # exponent on local/global density ratio -> bandwidth scaling
DENSITY_METHODS = [("kde", "KDE"), ("adaptive_kde", "Adaptive KDE"), ("gmm", "GMM"), ("bspline", "B-spline")]
TOOLS = "xpan,xwheel_zoom,xbox_zoom,reset,save"
PLOT_WIDTH = 900
MAX_ITER = 1000
WASSERSTEIN_TOL_LOG10_DEFAULT = -10

VALUE_GRID = np.linspace(X_MIN, X_MAX, GRID_N)
SURP_GRID = np.linspace(S_MIN, S_MAX, GRID_N)

# ── State ────────────────────────────────────────────────────────────────────
root_events = np.array([], dtype=float)
all_events: np.ndarray = np.array([], dtype=float)
history_index: int = 0
_transport_cb_guard: bool = False
_step_cb_handle: list = [None]


# ── Density helpers ────────────────────────────────────────────────────────

def _em_gmm_pdf(events, n_components):
    """Fit a 1-D Gaussian mixture via EM. Refit from scratch on every call
    (including up to MAX_ITER times inside the fixed-point loop), so this
    stays cheap and deterministic: a fixed small iteration count, and
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
    """Fit a continuous density to `events`, blended with a Gaussian(mu,sigma)
    prior. Blend weight is n/(n+alpha) -- the continuous analogue of adding
    alpha pseudocounts worth of prior mass to a histogram."""
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


def surprisal_bits(x, density_fn, dx=1.0):
    # A true discretized surprisal is S = -log2(P) where P ~= density(x) * dx
    # for some bin width dx -- density alone is a density, not a probability.
    # dx defaults to 1 (in whatever units x is in), so the bin-width slider at
    # dx=1 reproduces the old no-bin-width behavior exactly; moving it just
    # adds a constant -log2(dx) shift to every event's surprisal (bigger bins
    # -> more probability mass per bin -> lower surprisal).
    #
    # That constant shift is also why this quantity isn't scale-invariant on
    # its own: rescaling x rescales density inversely, which the dx slider can
    # compensate for but doesn't have to, since comparisons *within* one
    # density_fn at a fixed dx are meaningful regardless.
    #
    # Remaining caveats even with dx set "correctly":
    # - density(x) * dx can still exceed 1 (e.g. a narrow Gaussian with a wide
    #   bin), so surprisal can go negative -- unlike discrete Shannon
    #   surprisal, which is always >= 0.
    # - Strictly speaking, a continuous RV has P(X=x) = 0 at any single point;
    #   density(x) * dx is only an approximation of the probability mass in a
    #   bin of width dx centered at x, not an exact one.
    p = np.clip(density_fn(x) * dx, 1e-300, None)
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


def compute_fixed_point_iterations(events, alpha, mu, sigma, method, bw_factor, n_components, tol, dx=1.0):
    """Return (n_iter, final_density_fn, final_events, history) or all-None if no
    convergence.

    Mirrors fixed_point.py's iteration: map events -> S(P1) samples, then
    repeatedly re-transform through the current density and refit, until
    the distribution stops moving: the Wasserstein (W1) distance between
    P_i and P_i+1 drops below tol. W1 is used (rather than a
    raw pointwise density comparison) because it accounts for how much
    probability mass actually shifted, not just the worst-case height
    difference at any single point.

    `history` is a list of (kl_forward, kl_backward, w1) tuples, one per
    iteration, measuring the divergence between that iteration's density
    (P_i) and the next (P_i+1): forward is KL(P_i‖P_i+1), backward is
    KL(P_i+1‖P_i) -- the two differ because KL is asymmetric, while W1 is a
    true metric and has only one value per step.
    """
    if len(events) == 0:
        return None, None, None, None
    p1_density = make_density_fn(events, alpha, mu, sigma, method, bw_factor, n_components)
    current_events = surprisal_bits(events, p1_density, dx)
    density = make_density_fn(current_events, alpha, mu, sigma, method, bw_factor, n_components)
    history = []
    for i in range(MAX_ITER):
        new_events = surprisal_bits(current_events, density, dx)
        new_density = make_density_fn(new_events, alpha, mu, sigma, method, bw_factor, n_components)
        w1 = wasserstein_distance(density, new_density, SURP_GRID)
        history.append((
            kl_divergence_bits(density, new_density, SURP_GRID),
            kl_divergence_bits(new_density, density, SURP_GRID),
            w1,
        ))
        if w1 < tol:
            return i + 1, new_density, new_events, history
        current_events = new_events
        density = new_density
    return None, None, None, None


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class PNode:
    events: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    figure: object = None
    source: ColumnDataSource = None
    prior_alpha_slider: Slider = None
    prior_mu_slider: Slider = None
    prior_sigma_slider: Slider = None
    bandwidth_slider: Slider = None
    bin_width_slider: Slider = None
    bin_width_panel: object = None
    method_select: Select = None
    n_components_slider: Slider = None
    y_scale_toggle: Select = None
    current_density: object = None
    grid: np.ndarray = None
    layout: Column = None
    y_range_adaptive: bool = True
    rug_source: ColumnDataSource = None
    rug_glyph: object = None


node: Optional[PNode] = None
surp_node: Optional[PNode] = None
session_record: int = 0
session_record_rows: list = []

rug_checkbox = CheckboxGroup(labels=["Show event rug"], active=[])
rug_alpha_slider = Slider(start=0.0, end=1.0, value=0.15, step=0.01, title="Rug opacity", width=200)
tol_slider = Slider(start=-12, end=-3, step=1, value=WASSERSTEIN_TOL_LOG10_DEFAULT,
                     title="Convergence tolerance (log₁₀)", width=200)
tol_slider.on_change("value", lambda attr, old, new: recompute())

clear_overlays_btn = Button(label="Clear overlays", width=130, button_type="warning")
_overlay_source = ColumnDataSource(data=dict(xs=[], ys=[]))
_overlay_count = 0
_overlay_all_events: np.ndarray = np.array([], dtype=float)
_overlay_figure = figure(
    width=PLOT_WIDTH, height=280,
    x_range=(S_MIN, S_MAX),
    y_range=Range1d(0, 1),
    tools=TOOLS, toolbar_location="right",
    title="Fixed-point overlays — 0 points",
)
_overlay_figure.multi_line(xs="xs", ys="ys", source=_overlay_source,
                           line_color="#2266AA", line_alpha=0.25, line_width=2)
_overlay_figure.xgrid.grid_line_color = None
_overlay_figure.ygrid.grid_line_color = None
_overlay_figure.xaxis.axis_label = "Surprisal (bits)"
_overlay_figure.yaxis.axis_label = "Density"
_overlay_rug_source = ColumnDataSource(data=dict(x=[], y0=[], y1=[]))
_overlay_rug_glyph = _overlay_figure.segment(
    x0="x", y0="y0", x1="x", y1="y1", source=_overlay_rug_source,
    line_color="#444444", line_width=1,
    line_alpha=rug_alpha_slider.value, visible=(0 in rug_checkbox.active),
)


def _rug_glyphs():
    return [node.rug_glyph, surp_node.rug_glyph, _overlay_rug_glyph]


def on_rug_checkbox_change(attr, old, new):
    visible = 0 in new
    for g in _rug_glyphs():
        g.visible = visible


def on_rug_alpha_change(attr, old, new):
    for g in _rug_glyphs():
        g.glyph.line_alpha = new


rug_checkbox.on_change("active", on_rug_checkbox_change)
rug_alpha_slider.on_change("value", on_rug_alpha_change)


def _add_to_overlay(density_fn, events):
    global _overlay_count, _overlay_all_events
    y = density_fn(SURP_GRID)
    old = _overlay_source.data
    _overlay_source.data = dict(
        xs=list(old['xs']) + [SURP_GRID],
        ys=list(old['ys']) + [y],
    )
    _overlay_count += 1
    y_max = max([np.max(yy) for yy in _overlay_source.data['ys']], default=1.0)
    _overlay_figure.y_range.end = y_max * 1.1 or 1.0
    _overlay_figure.title.text = f"Fixed-point overlays — {_overlay_count} points"

    _overlay_all_events = np.concatenate([_overlay_all_events, events])
    rug_h = _overlay_figure.y_range.end * 0.03
    _overlay_rug_source.data = dict(
        x=_overlay_all_events,
        y0=np.zeros(len(_overlay_all_events)),
        y1=np.full(len(_overlay_all_events), rug_h),
    )


def on_clear_overlays():
    global _overlay_count, _overlay_all_events
    _overlay_source.data = dict(xs=[], ys=[])
    _overlay_count = 0
    _overlay_all_events = np.array([], dtype=float)
    _overlay_rug_source.data = dict(x=[], y0=[], y1=[])
    _overlay_figure.title.text = "Fixed-point overlays — 0 points"


clear_overlays_btn.on_click(on_clear_overlays)

_progression_source_log = ColumnDataSource(data=dict(iter=[], kl_fwd=[], kl_bwd=[], w1=[]))
_progression_source_linear = ColumnDataSource(data=dict(iter=[], kl_fwd=[], kl_bwd=[], w1=[]))


def _make_progression_figure(source, log_scale):
    kwargs = dict(
        width=PLOT_WIDTH, height=280,
        tools=TOOLS, toolbar_location="right",
        title="Convergence progression (last run)",
    )
    if log_scale:
        kwargs["y_axis_type"] = "log"
    else:
        kwargs["y_range"] = Range1d(0, 1)
    fig = figure(**kwargs)
    fig.line(x="iter", y="kl_fwd", source=source,
             line_color="#2266AA", line_width=2, legend_label="KL forward  (Pᵢ‖Pᵢ₊₁)")
    fig.circle(x="iter", y="kl_fwd", source=source, color="#2266AA", size=4)
    fig.line(x="iter", y="kl_bwd", source=source,
             line_color="#CC6633", line_width=2, legend_label="KL backward (Pᵢ₊₁‖Pᵢ)")
    fig.circle(x="iter", y="kl_bwd", source=source, color="#CC6633", size=4)
    fig.line(x="iter", y="w1", source=source,
             line_color="#339933", line_width=2, legend_label="W1")
    fig.circle(x="iter", y="w1", source=source, color="#339933", size=4)
    fig.xgrid.grid_line_color = None
    fig.xaxis.axis_label = "Iteration"
    fig.yaxis.axis_label = "Divergence (log scale)" if log_scale else "Divergence"
    fig.legend.click_policy = "hide"
    fig.add_layout(fig.legend[0], "right")
    return fig


_progression_figure_log = _make_progression_figure(_progression_source_log, log_scale=True)
_progression_figure_linear = _make_progression_figure(_progression_source_linear, log_scale=False)
_progression_container = Column(_progression_figure_log)

progression_y_scale_select = Select(
    value="log", options=[("log", "Y: log scale"), ("linear", "Y: linear scale")], width=150,
)


def on_progression_y_scale_change(attr, old, new):
    _progression_container.children = [
        _progression_figure_linear if new == "linear" else _progression_figure_log
    ]


progression_y_scale_select.on_change("value", on_progression_y_scale_change)


def _log_safe(vals):
    """NaN-out non-positive values instead of clamping to a floor -- a log
    axis can't plot zero anyway, and clamping makes near-zero and
    barely-below-threshold values look identical."""
    arr = np.asarray(vals, dtype=float)
    return np.where(arr > 0, arr, np.nan)


def _update_progression_plot(history):
    if not history:
        empty = dict(iter=[], kl_fwd=[], kl_bwd=[], w1=[])
        _progression_source_log.data = empty
        _progression_source_linear.data = empty
        return
    iters = list(range(1, len(history) + 1))
    kl_fwd_raw = [h[0] for h in history]
    kl_bwd_raw = [h[1] for h in history]
    w1_raw = [h[2] for h in history]
    _progression_source_log.data = dict(
        iter=iters,
        kl_fwd=_log_safe(kl_fwd_raw),
        kl_bwd=_log_safe(kl_bwd_raw),
        w1=_log_safe(w1_raw),
    )
    _progression_source_linear.data = dict(
        iter=iters,
        kl_fwd=np.clip(kl_fwd_raw, 0, None),
        kl_bwd=np.clip(kl_bwd_raw, 0, None),
        w1=np.clip(w1_raw, 0, None),
    )
    y_max = max(np.max(kl_fwd_raw), np.max(kl_bwd_raw), np.max(w1_raw))
    _progression_figure_linear.y_range.end = y_max * 1.1 or 1.0

convergence_div = Div(
    text="<i>Add events to compute fixed-point iterations.</i>",
    styles={"font-size": "15px", "margin-top": "10px"},
)


def _update_curve(nd: PNode, density_fn, grid):
    y = density_fn(grid)
    nd.source.data = dict(x=grid, y=y)
    nd.current_density = density_fn
    if nd.y_range_adaptive:
        nd.figure.y_range.end = float(np.max(y)) * 1.05 if len(y) else 1.0
    else:
        nd.figure.y_range.end = 1.0
    rug_h = nd.figure.y_range.end * 0.03
    nd.rug_source.data = dict(x=nd.events, y0=np.zeros(len(nd.events)), y1=np.full(len(nd.events), rug_h))


def _update_surp_node():
    """First surprisal distribution S(P1), one transform only (no iteration)."""
    if node is None or surp_node is None:
        return
    if node.current_density is None or len(node.events) == 0:
        empty_density = lambda x: np.zeros_like(np.asarray(x, dtype=float))
        surp_node.events = np.array([], dtype=float)
        _update_curve(surp_node, empty_density, SURP_GRID)
        surp_node.figure.title.text = "S(P1) — First Surprisal Distribution"
        return
    surp_events = surprisal_bits(node.events, node.current_density, node.bin_width_slider.value)
    surp_node.events = surp_events
    alpha = surp_node.prior_alpha_slider.value
    mu = surp_node.prior_mu_slider.value
    sigma = surp_node.prior_sigma_slider.value
    bw = surp_node.bandwidth_slider.value
    density_fn = make_density_fn(surp_events, alpha, mu, sigma,
                                  surp_node.method_select.value, bw, surp_node.n_components_slider.value)
    _update_curve(surp_node, density_fn, SURP_GRID)
    surp_node.figure.title.text = (
        f"S(P1)  |  entropy = {differential_entropy_bits(density_fn, SURP_GRID):.4f} bits"
    )


def recompute():
    if node is None:
        return
    alpha = node.prior_alpha_slider.value
    mu = node.prior_mu_slider.value
    sigma = node.prior_sigma_slider.value
    bw = node.bandwidth_slider.value
    method = node.method_select.value
    n_components = node.n_components_slider.value
    density_fn = make_density_fn(node.events, alpha, mu, sigma, method, bw, n_components)
    _update_curve(node, density_fn, VALUE_GRID)
    node.figure.title.text = (
        f"P1  |  entropy = {differential_entropy_bits(density_fn, VALUE_GRID):.4f} bits"
    )

    _update_surp_node()

    n_iter, fixed_density, fixed_events, history = compute_fixed_point_iterations(
        node.events, alpha, mu, sigma, method, bw, n_components, tol=10 ** tol_slider.value,
        dx=node.bin_width_slider.value)
    if n_iter is None and len(node.events) == 0:
        convergence_div.text = "<i>Add events to compute fixed-point iterations.</i>"
        _update_progression_plot(None)
        return
    elif n_iter is None:
        convergence_div.text = f"<b>Did not converge within {MAX_ITER} iterations.</b>"
        _update_progression_plot(None)
        return

    _update_progression_plot(history)
    _add_to_overlay(fixed_density, fixed_events)
    fixed_entropy = differential_entropy_bits(fixed_density, SURP_GRID)

    s = "iteration" if n_iter == 1 else "iterations"
    current_line = f"<b>Fixed point reached in {n_iter} {s}.</b> Entropy = {fixed_entropy:.4f} bits."

    global session_record, session_record_rows
    if n_iter > session_record:
        session_record = n_iter
        dist_desc = family_select.value
        if _current_param_sliders:
            param_str = ", ".join(f"{sl.title}={sl.value:.3g}" for sl in _current_param_sliders)
            dist_desc += f"({param_str})"
        n_events = len(node.events)
        row = (f"<b>{n_iter}</b> {s} — {dist_desc} | "
               f"events count: {n_events} | α={alpha:.3g}, μ={mu:.3g}, σ={sigma:.3g}, bw={bw:.2g}")
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


def _make_curve_figure(title, x_range, x_label):
    source = ColumnDataSource(data=dict(x=[], y=[]))
    fig = figure(
        width=PLOT_WIDTH, height=380,
        x_range=x_range,
        y_range=Range1d(0, 1),
        tools=TOOLS, toolbar_location="right",
        title=title,
    )
    fig.varea(x="x", y1=0, y2="y", source=source, fill_color="#4878CF", fill_alpha=0.4)
    line_renderer = fig.line(x="x", y="y", source=source, line_color="#2255AA", line_width=2)
    hover = HoverTool(renderers=[line_renderer], mode="vline", tooltips=[
        ("x", "@x{0.000}"),
        ("density", "@y{0.0000}"),
    ])
    fig.add_tools(hover)
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.xaxis.axis_label = x_label
    fig.yaxis.axis_label = "Density"
    rug_source = ColumnDataSource(data=dict(x=[], y0=[], y1=[]))
    rug_glyph = fig.segment(
        x0="x", y0="y0", x1="x", y1="y1", source=rug_source,
        line_color="#444444", line_width=1,
        line_alpha=rug_alpha_slider.value, visible=(0 in rug_checkbox.active),
    )
    return fig, source, rug_source, rug_glyph


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


def make_node(initial_events, alpha_end=5, x_range=(X_MIN, X_MAX), x_label="Value",
              mu_range=(-10, 10), sigma_range=(0.1, 20), title="P1  |  entropy = 0.0000 bits"):
    n = PNode()
    n.events = initial_events
    n.grid = VALUE_GRID

    n.figure, n.source, n.rug_source, n.rug_glyph = _make_curve_figure(title, x_range, x_label)

    n.prior_alpha_slider = Slider(start=0, end=alpha_end, value=PRIOR_ALPHA_DEFAULT, step=0.1, title="Prior strength α", width=250)
    n.prior_mu_slider = Slider(start=mu_range[0], end=mu_range[1], value=PRIOR_MU_DEFAULT, step=0.1, title="Prior mean μ", width=250)
    n.prior_sigma_slider = Slider(start=sigma_range[0], end=sigma_range[1], value=PRIOR_SIGMA_DEFAULT, step=0.1, title="Prior std dev σ", width=250)
    n.bandwidth_slider = Slider(start=0.1, end=3.0, value=BANDWIDTH_DEFAULT, step=0.05, title="Bandwidth / smoothing factor", width=250)
    n.bin_width_slider = Slider(start=0.01, end=5.0, value=BIN_WIDTH_DEFAULT, step=0.01, title="Bin width Δx", width=250)
    n.n_components_slider = Slider(start=1, end=GMM_COMPONENTS_MAX, value=GMM_COMPONENTS_DEFAULT, step=1, title="GMM components", width=250)
    n.method_select = Select(value="kde", options=DENSITY_METHODS, title="Fit method", width=140)
    n.y_scale_toggle = Select(value="adaptive",
                               options=[("fixed", "Y: fixed 0–1"), ("adaptive", "Y: adaptive")],
                               width=140)

    def on_param_change(attr, old, new):
        recompute()

    def on_method_change(attr, old, new, nd=n):
        nd.layout.children[0] = _param_row(nd)
        recompute()

    def on_y_scale_toggle(attr, old, new, nd=n):
        nd.y_range_adaptive = (new == "adaptive")
        recompute()

    for s in (n.prior_alpha_slider, n.prior_mu_slider, n.prior_sigma_slider,
              n.bandwidth_slider, n.bin_width_slider, n.n_components_slider):
        s.on_change("value", on_param_change)
    n.method_select.on_change("value", on_method_change)
    n.y_scale_toggle.on_change("value", on_y_scale_toggle)

    n.bin_width_panel = Column(n.bin_width_slider)
    n.layout = Column(
        _param_row(n),
        Row(n.figure, Spacer(width=20), n.bin_width_panel),
        Row(n.y_scale_toggle),
    )
    return n


# ── Top-level event controls (same as fixed_point.py) ─────────────────────────

n_events_input = TextInput(value="1000", title="", width=80)
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
    clear_events_btn.disabled = n == 0


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
    new_ev = ev.get_events(n, family_select.value, get_current_params())
    all_events = new_ev.copy()
    history_index = len(all_events)
    root_events = all_events.copy()
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


add_events_btn.on_click(on_add_events)
clear_events_btn.on_click(on_clear_events)
single_event_input.on_change("value", on_single_event_input)
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


history_slider.on_change("value", on_history_slider_change)
history_back_btn.on_click(lambda: apply_history_index(history_index - 1))
history_fwd_btn.on_click(lambda: apply_history_index(history_index + 1))


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

node = make_node(root_events.copy(), alpha_end=5, x_range=(X_MIN, X_MAX), x_label="Value",
                 title="P1  |  entropy = 0.0000 bits")
surp_node = make_node(np.array([], dtype=float), alpha_end=5, x_range=(S_MIN, S_MAX),
                      x_label="Surprisal (bits)", mu_range=(0, 20), sigma_range=(0.1, 10),
                      title="S(P1) — First Surprisal Distribution")

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
        Spacer(width=30),
        rug_checkbox,
        rug_alpha_slider,
        tol_slider,
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
    history_slider, Spacer(width=5),
    history_fwd_btn, Spacer(width=10),
    history_label,
    sizing_mode="stretch_width",
)

curdoc().add_root(Column(
    top_controls, transport_row,
    node.layout, surp_node.layout,
    convergence_div,
    Row(progression_y_scale_select),
    _progression_container,
    Row(clear_overlays_btn),
    _overlay_figure,
))
curdoc().title = "Surprisal Fixed Point (Continuous)"
