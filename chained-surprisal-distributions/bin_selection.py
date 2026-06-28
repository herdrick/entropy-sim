import numpy as np
from bokeh.models import ColumnDataSource, Div, Slider, Row, Column, Spacer, Toggle


class BinTracker:
    def __init__(self):
        self._counts = {}
        self._n_runs = 0

    def record(self, probs):
        """Record one fixed-point probability vector (numpy array)."""
        probs = np.asarray(probs)
        self._n_runs += 1
        for idx in np.where(probs > 1e-12)[0]:
            idx = int(idx)
            self._counts[idx] = self._counts.get(idx, 0) + 1

    def reset(self):
        """Clear all recorded data."""
        self._counts = {}
        self._n_runs = 0

    def get_active_bins(self, min_freq=0.1):
        """Return list of bin indices that appeared with probability > 1e-12
        in at least min_freq fraction of runs. Returns sorted list of ints."""
        if self._n_runs == 0:
            return []
        return sorted(
            idx
            for idx, count in self._counts.items()
            if count / self._n_runs >= min_freq
        )

    def get_bin_labels(self, indices):
        """Return list of label strings for given bin indices.
        Label format: bin index 0 -> 'p1', index 4 -> 'p5'."""
        return ["p{}".format(int(idx) + 1) for idx in indices]

    @property
    def n_runs(self):
        """Number of fixed points recorded."""
        return self._n_runs

    @property
    def n_bins_seen(self):
        """Total number of distinct bin indices seen across all runs."""
        return len(self._counts)


def make_bin_lock_ui(tracker):
    """Returns (layout, locked_bins_state)

    layout: a Bokeh Column containing the UI
    locked_bins_state: a plain mutable dict: {'locked': bool, 'bins': list, 'labels': list}
    """
    locked_bins_state = {'locked': False, 'bins': [], 'labels': []}

    toggle_btn = Toggle(label="Lock bins", active=False)
    min_freq_slider = Slider(
        start=0, end=100, value=10, step=1, title="Min frequency %", width=200
    )
    status_div = Div(text="")

    def _current_min_freq():
        return min_freq_slider.value / 100.0

    def _update_status_unlocked():
        bins = tracker.get_active_bins(_current_min_freq())
        labels = tracker.get_bin_labels(bins)
        if labels:
            label_str = ", ".join(labels)
            status_div.text = "Unlocked &mdash; {} active bins ({})".format(
                len(bins), label_str
            )
        else:
            status_div.text = "Unlocked &mdash; 0 active bins"

    def _update_status_locked():
        labels = locked_bins_state['labels']
        if labels:
            status_div.text = "Locked to: {}".format(", ".join(labels))
        else:
            status_div.text = "Locked to: (none)"

    def _on_toggle(attr, old, new):
        if new:
            bins = tracker.get_active_bins(_current_min_freq())
            labels = tracker.get_bin_labels(bins)
            locked_bins_state['locked'] = True
            locked_bins_state['bins'] = bins
            locked_bins_state['labels'] = labels
            toggle_btn.label = "Unlock bins"
            _update_status_locked()
        else:
            locked_bins_state['locked'] = False
            locked_bins_state['bins'] = []
            locked_bins_state['labels'] = []
            toggle_btn.label = "Lock bins"
            _update_status_unlocked()

    def _on_min_freq(attr, old, new):
        if not locked_bins_state['locked']:
            _update_status_unlocked()

    toggle_btn.on_change('active', _on_toggle)
    min_freq_slider.on_change('value', _on_min_freq)

    _update_status_unlocked()

    def refresh_status():
        """Call after new fixed points are recorded to keep status current."""
        if not locked_bins_state['locked']:
            _update_status_unlocked()

    layout = Column(
        Row(toggle_btn, Spacer(width=10), min_freq_slider),
        status_div,
    )

    locked_bins_state['_refresh_status'] = refresh_status
    return layout, locked_bins_state
