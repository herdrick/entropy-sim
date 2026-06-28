import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, Slider, LabelSet
from bokeh.plotting import figure
from bokeh.layouts import column, row

PRIMARY_COLOR = "#4878CF"

# 2D triangle vertices for barycentric-to-Cartesian projection
_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])


def _project_first3_nonzero(fp):
    """Return (barycentric_coords, labels) using first 3 non-zero bins of fp."""
    nz = [i for i, v in enumerate(fp) if v > 1e-12][:3]
    p = np.zeros(3)
    for k, i in enumerate(nz):
        p[k] = fp[i]
    s = p.sum()
    if s > 0:
        p /= s
    labels = [f'p{nz[k]+1}' if k < len(nz) else '' for k in range(3)]
    return p, labels


def _compute_columns(fixed_points):
    """Project all fixed points; return data dict and the last point's vertex labels."""
    xs, ys, bas, bbs, bcs = [], [], [], [], []
    last_labels = ['p?', 'p?', 'p?']
    for fp in fixed_points:
        bary, labels = _project_first3_nonzero(fp)
        xy = bary[0] * _V[0] + bary[1] * _V[1] + bary[2] * _V[2]
        xs.append(xy[0])
        ys.append(xy[1])
        bas.append(bary[0])
        bbs.append(bary[1])
        bcs.append(bary[2])
        last_labels = [labels[k] if labels[k] else 'p?' for k in range(3)]
    data = dict(x=xs, y=ys, ba=bas, bb=bbs, bc=bcs)
    return data, last_labels


def _vertex_label_data(labels):
    return dict(
        x=[_V[0, 0], _V[1, 0], _V[2, 0]],
        y=[_V[0, 1], _V[1, 1], _V[2, 1]],
        text=[labels[0], labels[1], labels[2]],
    )


def make_simplex3d_panel(fixed_points, bin_indices, bin_labels):
    """fixed_points: list of numpy arrays. bin_indices: list of int. bin_labels: list of str.
    Returns (layout, state) where layout is a Column and state is a dict."""
    data, last_labels = _compute_columns(fixed_points)
    source = ColumnDataSource(data=data)

    fig = figure(
        width=520,
        height=460,
        title=f"Fixed points on simplex (n={len(fixed_points)})",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        match_aspect=True,
    )
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    fig.xgrid.visible = False
    fig.ygrid.visible = False

    # Triangle edges
    tri_x = [_V[0, 0], _V[1, 0], _V[2, 0], _V[0, 0]]
    tri_y = [_V[0, 1], _V[1, 1], _V[2, 1], _V[0, 1]]
    fig.line(tri_x, tri_y, line_color="#888888", line_width=1.5)

    # Vertex labels
    vertex_label_source = ColumnDataSource(data=_vertex_label_data(last_labels))
    vertex_labels = LabelSet(
        x='x', y='y', text='text', source=vertex_label_source,
        text_font_size='11pt', text_color='#333333',
        x_offset=4, y_offset=4,
    )
    fig.add_layout(vertex_labels)

    alpha_slider = Slider(start=0.0, end=1.0, value=0.5, step=0.01, title="Point alpha")

    scatter_glyph = fig.scatter(
        x='x', y='y', source=source, size=9,
        fill_color=PRIMARY_COLOR, line_color=PRIMARY_COLOR,
        fill_alpha=alpha_slider.value, line_alpha=alpha_slider.value,
    )

    hover = HoverTool(
        renderers=[scatter_glyph],
        tooltips=[
            ("p_a", "@ba{0.000}"),
            ("p_b", "@bb{0.000}"),
            ("p_c", "@bc{0.000}"),
        ],
    )
    fig.add_tools(hover)

    def _on_alpha(attr, old, new):
        scatter_glyph.glyph.fill_alpha = new
        scatter_glyph.glyph.line_alpha = new

    alpha_slider.on_change('value', _on_alpha)

    slider_row = row(alpha_slider)
    layout = column(slider_row, fig)

    state = {
        'source': source,
        'figure': fig,
        'alpha_slider': alpha_slider,
        'scatter_glyph': scatter_glyph,
        'vertex_label_source': vertex_label_source,
        'layout': layout,
    }
    return layout, state


def update_simplex3d_panel(state, fixed_points, bin_indices, bin_labels):
    """Updates in-place. Returns layout (same Column)."""
    data, last_labels = _compute_columns(fixed_points)
    state['source'].data = data
    state['vertex_label_source'].data = _vertex_label_data(last_labels)
    state['figure'].title.text = f"Fixed points on simplex (n={len(fixed_points)})"
    return state['layout']
