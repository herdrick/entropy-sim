import numpy as np
from bokeh.models import ColumnDataSource, Div, Column
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column

MAX_N = 8
CELL_W = 130
CELL_H = 130
SCATTER_COLOR = "#4878CF"


def _extract_values(fixed_points, bin_idx):
    return [float(fp[bin_idx]) if bin_idx < len(fp) else 0.0 for fp in fixed_points]


def _build_shared_data(fixed_points, bin_indices):
    data = {}
    for col, bin_idx in enumerate(bin_indices):
        data['b{}'.format(col)] = _extract_values(fixed_points, bin_idx)
    return data


def _histogram(values):
    hist, edges = np.histogram(values, bins=20, range=(0, 1))
    return dict(top=hist.tolist(), left=edges[:-1].tolist(), right=edges[1:].tolist())


def _empty_panel():
    div = Div(text="No active bins selected.")
    layout = Column(div)
    state = {
        'source': None,
        'figure': None,
        'layout': layout,
        'bin_indices': [],
        'bin_labels': [],
        'diag_sources': [],
    }
    return layout, state


def _build(fixed_points, bin_indices, bin_labels):
    if not bin_indices:
        return _empty_panel()

    n = min(len(bin_indices), MAX_N)
    bin_indices = list(bin_indices[:n])
    bin_labels = list(bin_labels[:n])

    shared_source = ColumnDataSource(_build_shared_data(fixed_points, bin_indices))

    col_x_ranges = [None] * n
    row_y_ranges = [None] * n

    diag_sources = []
    grid = []

    for i in range(n):
        row_figs = []
        for j in range(n):
            kwargs = dict(width=CELL_W, height=CELL_H, tools="")
            if col_x_ranges[j] is not None:
                kwargs['x_range'] = col_x_ranges[j]
            if i != j and row_y_ranges[i] is not None:
                kwargs['y_range'] = row_y_ranges[i]

            fig = figure(**kwargs)

            if col_x_ranges[j] is None:
                col_x_ranges[j] = fig.x_range
            if i != j and row_y_ranges[i] is None:
                row_y_ranges[i] = fig.y_range

            if i == j:
                values = shared_source.data['b{}'.format(i)]
                diag_source = ColumnDataSource(_histogram(values))
                diag_sources.append(diag_source)
                fig.quad(top='top', bottom=0, left='left', right='right',
                         source=diag_source, fill_color=SCATTER_COLOR,
                         line_color="black", alpha=0.6)
            else:
                fig.scatter(x='b{}'.format(j), y='b{}'.format(i),
                            source=shared_source, alpha=0.3, size=5,
                            color=SCATTER_COLOR)

            if i == n - 1:
                fig.xaxis.axis_label = bin_labels[j]
            else:
                fig.xaxis.major_label_text_font_size = '0pt'

            if j == 0:
                fig.yaxis.axis_label = bin_labels[i]
            else:
                fig.yaxis.major_label_text_font_size = '0pt'

            row_figs.append(fig)
        grid.append(row_figs)

    gp = gridplot(grid, merge_tools=False)
    layout = column(gp)

    state = {
        'source': shared_source,
        'figure': gp,
        'layout': layout,
        'bin_indices': bin_indices,
        'bin_labels': bin_labels,
        'diag_sources': diag_sources,
    }
    return layout, state


def make_scatter_matrix_panel(fixed_points, bin_indices, bin_labels):
    """
    fixed_points: list of numpy arrays
    bin_indices: list of int (0-based), capped at 8
    bin_labels: list of str (e.g. ["p1", "p3", "p5"])
    Returns (layout, state)
    """
    return _build(fixed_points, bin_indices, bin_labels)


def update_scatter_matrix_panel(state, fixed_points, bin_indices, bin_labels):
    """Updates in-place when possible. Returns layout."""
    n = min(len(bin_indices), MAX_N)
    capped_indices = list(bin_indices[:n])

    if capped_indices != list(state.get('bin_indices', [])):
        layout, new_state = _build(fixed_points, bin_indices, bin_labels)
        state.clear()
        state.update(new_state)
        return layout

    if state['source'] is not None:
        state['source'].data = _build_shared_data(fixed_points, capped_indices)
        for i, diag_source in enumerate(state['diag_sources']):
            values = _extract_values(fixed_points, capped_indices[i])
            diag_source.data = _histogram(values)

    return state['layout']
