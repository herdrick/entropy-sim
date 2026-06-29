import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from io import BytesIO
import base64

from bokeh.models import ColumnDataSource, Div, Slider, Select, Row, Column, Spacer

_NO_BIN = "__none__"


def _render_simplex(fixed_points, bin_a, bin_b, bin_c, elev=26, azim=47, point_alpha=0.1):
    """Render 3D simplex PNG; returns HTML img tag string."""
    fig = plt.figure(figsize=(5.2, 4.6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tri = Poly3DCollection([verts], alpha=0.12, facecolor='steelblue', edgecolor='none')
    ax.add_collection3d(tri)
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        ax.plot3D(*zip(verts[i], verts[j]), color='#444', lw=1.4, alpha=0.6)

    labels = [
        f"p{bin_a+1}" if bin_a >= 0 else "—",
        f"p{bin_b+1}" if bin_b >= 0 else "—",
        f"p{bin_c+1}" if bin_c >= 0 else "—",
    ]
    offsets = [(0.07, -0.07, -0.07), (-0.07, 0.07, -0.07), (-0.07, -0.07, 0.07)]
    for v, label, off in zip(verts, labels, offsets):
        if label != "—":
            ax.text(v[0]+off[0], v[1]+off[1], v[2]+off[2],
                    label, fontsize=12, ha='center', va='center')

    pts = []
    for fp in fixed_points:
        p = np.zeros(3)
        for k, idx in enumerate([bin_a, bin_b, bin_c]):
            if idx >= 0 and idx < len(fp):
                p[k] = fp[idx]
        s = p.sum()
        if s > 0:
            p /= s
            pts.append(p)

    if pts:
        arr = np.array(pts)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2],
                   c='blue', alpha=point_alpha, s=28, zorder=5, depthshade=True)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xlabel(labels[0], labelpad=4)
    ax.set_ylabel(labels[1], labelpad=4)
    ax.set_zlabel(labels[2], labelpad=4)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f'Fixed points on simplex  (n={len(pts)})', fontsize=11, pad=6)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_b64}" style="display:block;"/>'


def _build_options(fixed_points):
    """Return list of (value_str, label) for every bin that has ever had mass."""
    if not fixed_points:
        return []
    max_len = max(len(fp) for fp in fixed_points)
    opts = []
    for i in range(max_len):
        vals = [fp[i] for fp in fixed_points if i < len(fp)]
        if any(v > 1e-12 for v in vals):
            avg = float(np.mean(vals))
            opts.append((str(i), f"p{i+1} | {avg:.4f}"))
    return opts


def _parse_idx(val):
    """Select widget value → int bin index, or -1 for (none)."""
    if val == _NO_BIN:
        return -1
    try:
        return int(val)
    except (ValueError, TypeError):
        return -1


def make_simplex3d_panel(fixed_points, bin_indices, bin_labels):
    """Returns (layout, state)."""
    opts = _build_options(fixed_points)
    none_opt = [(_NO_BIN, "(none)")]
    full_opts = none_opt + opts

    def _default(k):
        return opts[k][0] if k < len(opts) else _NO_BIN

    elev_slider  = Slider(start=-90, end=90,  value=26,  step=1,    title="Elevation",     width=200)
    azim_slider  = Slider(start=0,   end=360, value=47,  step=1,    title="Azimuth",       width=200)
    alpha_slider = Slider(start=0.0, end=1.0, value=0.1, step=0.01, title="Opacity",  width=200)

    bin_a_select = Select(title="Vertex A", value=_default(0), options=full_opts, width=170)
    bin_b_select = Select(title="Vertex B", value=_default(1), options=full_opts, width=170)
    bin_c_select = Select(title="Vertex C", value=_default(2), options=full_opts, width=170)

    simplex_div = Div(width=520, height=460)

    state = {
        'source': ColumnDataSource(data=dict()),
        'figure': simplex_div,
        'elev_slider': elev_slider,
        'azim_slider': azim_slider,
        'alpha_slider': alpha_slider,
        'bin_a_select': bin_a_select,
        'bin_b_select': bin_b_select,
        'bin_c_select': bin_c_select,
        'layout': None,
        '_fixed_points': list(fixed_points),
    }

    def _rerender():
        simplex_div.text = _render_simplex(
            state['_fixed_points'],
            _parse_idx(bin_a_select.value),
            _parse_idx(bin_b_select.value),
            _parse_idx(bin_c_select.value),
            elev=int(elev_slider.value),
            azim=int(azim_slider.value),
            point_alpha=float(alpha_slider.value),
        )

    state['_rerender'] = _rerender

    elev_slider.on_change('value',    lambda a, o, n: _rerender())
    azim_slider.on_change('value',    lambda a, o, n: _rerender())
    alpha_slider.on_change('value',   lambda a, o, n: _rerender())
    bin_a_select.on_change('value',   lambda a, o, n: _rerender())
    bin_b_select.on_change('value',   lambda a, o, n: _rerender())
    bin_c_select.on_change('value',   lambda a, o, n: _rerender())

    _rerender()

    layout = Column(
        Row(bin_a_select, Spacer(width=8), bin_b_select, Spacer(width=8), bin_c_select),
        Row(elev_slider, Spacer(width=20), azim_slider, Spacer(width=20), alpha_slider),
        simplex_div,
    )
    state['layout'] = layout
    return layout, state


def update_simplex3d_panel(state, fixed_points, bin_indices, bin_labels):
    """Update fixed points, refresh dropdown options, re-render. Returns layout."""
    state['_fixed_points'] = list(fixed_points)

    opts = _build_options(fixed_points)
    none_opt = [(_NO_BIN, "(none)")]
    opt_vals = {v for v, _ in opts}

    selects = [state['bin_a_select'], state['bin_b_select'], state['bin_c_select']]

    # Auto-assign first 3 seen bins when all selects are still at (none)
    if opts and all(sel.value == _NO_BIN for sel in selects):
        for k, sel in enumerate(selects):
            if k < len(opts):
                sel.value = opts[k][0]

    # Preserve currently-selected bins even if they have no data right now
    # (e.g. after Clear points). Add them back with a "(no data)" label so
    # Bokeh doesn't auto-reset the select value.
    extra = []
    for sel in selects:
        v = sel.value
        if v != _NO_BIN and v not in opt_vals:
            try:
                extra.append((v, f"p{int(v)+1} | (no data)"))
            except (ValueError, TypeError):
                pass

    full_opts = none_opt + opts + extra
    for sel in selects:
        sel.options = full_opts

    state['_rerender']()
    return state['layout']
