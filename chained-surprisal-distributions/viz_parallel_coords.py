import numpy as np
from bokeh.models import ColumnDataSource, Slider, Div, Row, Column, FixedTicker
from bokeh.plotting import figure
from bokeh.layouts import column, row


def _build_xs_ys(fixed_points, bin_indices):
    n = len(bin_indices)
    xs = []
    ys = []
    for fp in fixed_points:
        xs.append(list(range(n)))
        ys.append([fp[bin_idx] if bin_idx < len(fp) else 0.0 for bin_idx in bin_indices])
    return xs, ys


def make_parallel_coords_panel(fixed_points, bin_indices, bin_labels):
    """
    fixed_points: list of numpy arrays
    bin_indices: list of int (0-based)
    bin_labels: list of str (e.g. ["p1", "p3", "p5"])
    Returns (layout, state)
    """
    if not bin_indices:
        layout = column(Div(text="No active bins selected."))
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

    n = len(bin_indices)
    xs, ys = _build_xs_ys(fixed_points, bin_indices)
    source = ColumnDataSource(data=dict(xs=xs, ys=ys))

    fig = figure(
        width=600,
        height=380,
        title=f"Parallel coordinates (n={len(fixed_points)} fixed points)",
        x_range=(-0.5, n - 0.5),
        y_range=(0, 1),
    )

    renderer = fig.multi_line(
        xs='xs',
        ys='ys',
        source=source,
        line_color="#4878CF",
        line_alpha=0.4,
        line_width=1.2,
    )

    for i in range(n):
        fig.line([i, i], [0, 1], line_color='#888888', line_width=1.0)

    fig.xaxis.ticker = FixedTicker(ticks=list(range(n)))
    fig.xaxis.major_label_overrides = {i: bin_labels[i] for i in range(n)}
    fig.yaxis.axis_label = "Probability (0-1)"

    slider = Slider(start=0.0, end=1.0, value=0.4, step=0.01, title="Line alpha")

    def _on_alpha_change(attr, old, new):
        renderer.glyph.line_alpha = new

    slider.on_change('value', _on_alpha_change)

    layout = column(slider, fig)

    state = {
        'source': source,
        'figure': fig,
        'alpha_slider': slider,
        'multi_line_glyph': renderer,
        'layout': layout,
        'bin_indices': list(bin_indices),
        'bin_labels': list(bin_labels),
    }
    return layout, state


def update_parallel_coords_panel(state, fixed_points, bin_indices, bin_labels):
    """Updates in-place when possible. Returns layout."""
    if list(bin_indices) != list(state.get('bin_indices', [])):
        layout, new_state = make_parallel_coords_panel(fixed_points, bin_indices, bin_labels)
        state.clear()
        state.update(new_state)
        return layout

    if not bin_indices:
        return state['layout']

    xs, ys = _build_xs_ys(fixed_points, bin_indices)
    state['source'].data = dict(xs=xs, ys=ys)
    state['figure'].title.text = f"Parallel coordinates (n={len(fixed_points)} fixed points)"
    return state['layout']
