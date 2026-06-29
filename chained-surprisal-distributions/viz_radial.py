import numpy as np
from bokeh.models import ColumnDataSource, Slider, Div, Row, Column, LabelSet, HoverTool
from bokeh.plotting import figure
from bokeh.layouts import column, row


LINE_COLOR = "#4878CF"


def _spoke_angles(n):
    return [np.pi / 2 + 2 * np.pi * k / n for k in range(n)]


def _polygon_coords(fixed_points, bin_indices):
    """Return (xs, ys) lists-of-lists, one closed polygon per fixed point."""
    n = len(bin_indices)
    angles = _spoke_angles(n)
    xs, ys = [], []
    for fp in fixed_points:
        px, py = [], []
        for k, bi in enumerate(bin_indices):
            r = float(fp[bi]) if bi < len(fp) else 0.0
            px.append(r * np.cos(angles[k]))
            py.append(r * np.sin(angles[k]))
        # close the polygon
        px.append(px[0])
        py.append(py[0])
        xs.append(px)
        ys.append(py)
    return xs, ys


def _build_figure(fixed_points, bin_indices, bin_labels, alpha):
    n = len(bin_indices)
    angles = _spoke_angles(n)

    fig = figure(
        width=460,
        height=460,
        title=f"Radar chart (n={len(fixed_points)} fixed points, {n} bins)",
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        x_range=(-1.4, 1.4),
        y_range=(-1.4, 1.4),
    )
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    fig.xgrid.visible = False
    fig.ygrid.visible = False

    # reference circles
    theta = np.linspace(0, 2 * np.pi, 100)
    for r in [0.25, 0.5, 0.75, 1.0]:
        fig.line(
            r * np.cos(theta),
            r * np.sin(theta),
            line_color='#cccccc',
            line_dash='dashed',
            line_width=0.8,
        )

    # spokes from center to radius 1.0
    spoke_xs = [[0.0, np.cos(a)] for a in angles]
    spoke_ys = [[0.0, np.sin(a)] for a in angles]
    fig.multi_line(spoke_xs, spoke_ys, line_color='#cccccc', line_width=0.8)

    # axis labels at outer end of each spoke
    label_r = 1.12
    label_source = ColumnDataSource(data=dict(
        x=[label_r * np.cos(a) for a in angles],
        y=[label_r * np.sin(a) for a in angles],
        label=list(bin_labels),
    ))
    labels = LabelSet(
        x='x', y='y', text='label',
        source=label_source,
        text_align='center',
        text_baseline='middle',
        text_font_size='10pt',
    )
    fig.add_layout(labels)

    # polygons
    xs, ys = _polygon_coords(fixed_points, bin_indices)
    source = ColumnDataSource(data=dict(xs=xs, ys=ys))
    glyph = fig.multi_line(
        xs='xs', ys='ys',
        source=source,
        line_color=LINE_COLOR,
        line_alpha=alpha,
        line_width=1.5,
    )

    return fig, source, glyph


def make_radial_panel(fixed_points, bin_indices, bin_labels):
    """
    fixed_points: list of numpy arrays
    bin_indices: list of int (0-based)
    bin_labels: list of str (e.g. ["p1", "p3", "p5"])
    Returns (layout, state)
    """
    if not bin_indices:
        div = Div(text="No active bins selected.")
        layout = column(div)
        state = {
            'source': None,
            'figure': None,
            'alpha_slider': None,
            'multi_line_glyph': None,
            'layout': layout,
            'bin_indices': list(bin_indices),
            'bin_labels': list(bin_labels),
        }
        return layout, state

    alpha_default = 0.3
    fig, source, glyph = _build_figure(
        fixed_points, bin_indices, bin_labels, alpha_default
    )

    alpha_slider = Slider(
        start=0.0, end=1.0, value=alpha_default, step=0.05, title="Opacity"
    )

    def _on_alpha(attr, old, new):
        glyph.glyph.line_alpha = new

    alpha_slider.on_change('value', _on_alpha)

    layout = column(alpha_slider, fig)

    state = {
        'source': source,
        'figure': fig,
        'alpha_slider': alpha_slider,
        'multi_line_glyph': glyph,
        'layout': layout,
        'bin_indices': list(bin_indices),
        'bin_labels': list(bin_labels),
    }
    return layout, state


def update_radial_panel(state, fixed_points, bin_indices, bin_labels):
    """Updates in-place when possible. Returns layout."""
    same = (
        list(bin_indices) == list(state.get('bin_indices') or [])
        and bool(bin_indices)
    )

    if same and state.get('source') is not None:
        xs, ys = _polygon_coords(fixed_points, bin_indices)
        state['source'].data = dict(xs=xs, ys=ys)
        fig = state['figure']
        if fig is not None:
            fig.title.text = (
                f"Radar chart (n={len(fixed_points)} fixed points, "
                f"{len(bin_indices)} bins)"
            )
        return state['layout']

    # rebuild from scratch
    prev_alpha = state['alpha_slider'].value if state.get('alpha_slider') else None
    layout, new_state = make_radial_panel(fixed_points, bin_indices, bin_labels)
    if prev_alpha is not None and new_state.get('alpha_slider'):
        new_state['alpha_slider'].value = prev_alpha
        new_state['multi_line_glyph'].glyph.line_alpha = prev_alpha
    state.clear()
    state.update(new_state)
    return layout
