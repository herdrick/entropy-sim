"""
Interactive Entropy Simulator
Visualizes entropy, surprisal, and distribution convergence with mystery sources.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons, TextBox
from matplotlib import cm


# --- Mystery Sources ---
SOURCES = {
    'Source A': {'dist': stats.uniform(0, 1), 'name': 'Uniform(0,1)'},
    'Source B': {'dist': stats.beta(2, 5), 'name': 'Beta(2,5)'},
    'Source C': {'dist': stats.beta(0.5, 0.5), 'name': 'Beta(0.5,0.5)'},
    'Source D': {'dist': stats.beta(5, 5), 'name': 'Beta(5,5)'},
    'Source E': {'dist': stats.beta(0.3, 0.3), 'name': 'Beta(0.3,0.3)'},
    'Source F': {'dist': None, 'name': 'Mixture: 0.5*Beta(2,2) + 0.5*Beta(20,20)'},
}

def sample_source(key):
    if key == 'Source F':
        if np.random.random() < 0.5:
            return stats.beta.rvs(2, 2)
        else:
            return stats.beta.rvs(20, 20)
    return SOURCES[key]['dist'].rvs()

def source_pdf(key, x):
    if key == 'Source F':
        return 0.5 * stats.beta.pdf(x, 2, 2) + 0.5 * stats.beta.pdf(x, 20, 20)
    return SOURCES[key]['dist'].pdf(x)

def source_cdf(key, x):
    if key == 'Source F':
        return 0.5 * stats.beta.cdf(x, 2, 2) + 0.5 * stats.beta.cdf(x, 20, 20)
    return SOURCES[key]['dist'].cdf(x)

def entropy_of_source(key):
    """Compute theoretical binned entropy in bits (matches histogram computation)."""
    cdf_at_edges = source_cdf(key, BIN_EDGES)
    bin_probs = np.zeros(N_FINITE_BINS + 2)
    bin_probs[0] = cdf_at_edges[0]                        # (-inf, 0)
    bin_probs[1:N_FINITE_BINS + 1] = np.diff(cdf_at_edges)  # finite bins
    bin_probs[-1] = 1.0 - cdf_at_edges[-1]                # [1, +inf)
    safe = np.where(bin_probs > 0, bin_probs, 1.0)
    return -np.sum(np.where(bin_probs > 0, bin_probs * np.log2(safe), 0.0))


# --- Binned entropy computation ---
N_BIN_EDGES = 21
BIN_EDGES = np.linspace(0, 1, N_BIN_EDGES)
N_FINITE_BINS = N_BIN_EDGES - 1
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH = 1.0 / N_FINITE_BINS



def get_bin_idx(value):
    """Map value to bin index: 0=left overflow, 1..N_FINITE_BINS=finite, N_FINITE_BINS+1=right overflow."""
    if value < BIN_EDGES[0]:
        return 0
    if value >= BIN_EDGES[-1]:
        return N_FINITE_BINS + 1
    return np.searchsorted(BIN_EDGES, value, side='right')

def compute_binned_entropy(counts):
    """Entropy in bits from histogram counts."""
    counts = np.where(counts == 0, 1, counts)  # Laplace smoothing: only empty bins
    total = counts.sum()
    probs = counts / total
    safe = np.where(probs > 0, probs, 1.0)
    return -np.sum(np.where(probs > 0, probs * np.log2(safe), 0.0))

def surprisal_of_event(value, counts):
    """Surprisal in bits for a single event given current histogram."""
    bin_idx = get_bin_idx(value)
    counts = np.where(counts == 0, 1, counts)  # Laplace smoothing: only empty bins
    total = counts.sum()
    prob = counts[bin_idx] / total
    return -np.log2(prob)


class EntropySimulator:
    def __init__(self):
        self.current_source = 'Source A'
        self.playing = False
        self.revealed = False
        self.speed = 10  # events per second
        self.reset_data()
        self.setup_figure()
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / self.speed))
        self.timer.add_callback(self.step)
        self.timer.start()

    def reset_data(self):
        self.events = []
        self.counts = np.zeros(N_FINITE_BINS + 2, dtype=float)
        self.entropy_history = []
        self.surprisal_history = []
        self.running_avg_surprisal = []

    def setup_figure(self):
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.patch.set_facecolor('#1a1a2e')
        self.fig.suptitle('Entropy Simulator', color='white', fontsize=16, fontweight='bold')

        # Main grid: 2x2 panels with space for controls on the right
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.4],
                                   left=0.07, right=0.97, top=0.92, bottom=0.06,
                                   wspace=0.35, hspace=0.35)

        # --- Panel 1: Histogram (top-left) ---
        self.ax_hist = self.fig.add_subplot(gs[0, 0])
        self.ax_hist.set_facecolor('#16213e')
        self.ax_hist.set_title('Live Histogram', color='white', fontsize=11)
        self.ax_hist.set_xlabel('Value', color='#aaa', fontsize=9)
        self.ax_hist.set_ylabel('Probability', color='#aaa', fontsize=9)
        self.ax_hist.set_xlim(0, 1)
        self.ax_hist.tick_params(colors='#aaa')
        self.hist_bars = self.ax_hist.bar(BIN_CENTERS, np.zeros(N_FINITE_BINS),
                                          width=BIN_WIDTH * 0.9, color='#0f3460',
                                          edgecolor='#e94560', linewidth=0.5)
        self.hist_pdf_line = None  # for reveal
        # Overflow bins (initially zero-width, stretched to view edges on zoom)
        self.left_overflow_bar = self.ax_hist.bar([0], [0], width=0, color='#0f3460',
                                                   edgecolor='#e94560', linewidth=0.5)[0]
        self.right_overflow_bar = self.ax_hist.bar([1], [0], width=0, color='#0f3460',
                                                    edgecolor='#e94560', linewidth=0.5)[0]
        # Bin edge markers at 0 and 1
        self.ax_hist.axvline(0, color='#aaa', linewidth=0.8, linestyle='--', alpha=0.5)
        self.ax_hist.axvline(1, color='#aaa', linewidth=0.8, linestyle='--', alpha=0.5)

        # Scroll-to-zoom on histogram x-axis
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

        # --- Panel 2: Entropy over time (top-right) ---
        self.ax_entropy = self.fig.add_subplot(gs[0, 1])
        self.ax_entropy.set_facecolor('#16213e')
        self.ax_entropy.set_title('Entropy Over Time', color='white', fontsize=11)
        self.ax_entropy.set_xlabel('Events', color='#aaa', fontsize=9)
        self.ax_entropy.set_ylabel('Entropy (bits)', color='#aaa', fontsize=9)
        self.ax_entropy.tick_params(colors='#aaa')
        self.entropy_line, = self.ax_entropy.plot([], [], color='#e94560', linewidth=1.5)
        self.entropy_theory_line = None

        # --- Panel 3: Surprisal stream (bottom-left) ---
        self.ax_surprisal = self.fig.add_subplot(gs[1, 0])
        self.ax_surprisal.set_facecolor('#16213e')
        self.ax_surprisal.set_title('Surprisal Stream', color='white', fontsize=11)
        self.ax_surprisal.set_xlabel('Events', color='#aaa', fontsize=9)
        self.ax_surprisal.set_ylabel('Surprisal (bits)', color='#aaa', fontsize=9)
        self.ax_surprisal.tick_params(colors='#aaa')
        self.surprisal_scatter = self.ax_surprisal.scatter([], [], s=8, c=[], cmap='coolwarm_r',
                                                           vmin=0, vmax=8)
        self.surprisal_avg_line, = self.ax_surprisal.plot([], [], color='#e2d810',
                                                          linewidth=1.5, label='Running avg')

        # --- Panel 4: Event visualizer (bottom-right) ---
        self.ax_event = self.fig.add_subplot(gs[1, 1])
        self.ax_event.set_facecolor('#16213e')
        self.ax_event.set_title('Latest Event', color='white', fontsize=11)
        self.ax_event.set_xlim(-0.05, 1.05)
        self.ax_event.set_ylim(-0.5, 1.5)
        self.ax_event.set_yticks([])
        self.ax_event.tick_params(colors='#aaa')
        # Number line
        self.ax_event.plot([0, 1], [0, 0], color='#aaa', linewidth=2)
        for tick in np.arange(0, 1.1, 0.1):
            self.ax_event.plot([tick, tick], [-0.08, 0.08], color='#aaa', linewidth=1)
        self.event_marker, = self.ax_event.plot([], [], 'o', color='#e94560',
                                                 markersize=14, zorder=5)
        self.event_text = self.ax_event.text(0.5, 0.9, '', transform=self.ax_event.transAxes,
                                              color='white', fontsize=10, ha='center',
                                              va='center', family='monospace')

        # --- Controls panel (right column) ---
        ax_controls = self.fig.add_subplot(gs[:, 2])
        ax_controls.set_visible(False)

        # Play/Pause button
        ax_play = self.fig.add_axes([0.72, 0.82, 0.12, 0.05])
        self.btn_play = Button(ax_play, 'Play', color='#0f3460', hovercolor='#e94560')
        self.btn_play.label.set_color('white')
        self.btn_play.on_clicked(self.toggle_play)

        # Reset button
        ax_reset = self.fig.add_axes([0.86, 0.82, 0.09, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset', color='#0f3460', hovercolor='#e94560')
        self.btn_reset.label.set_color('white')
        self.btn_reset.on_clicked(self.on_reset)

        # Reveal button
        ax_reveal = self.fig.add_axes([0.72, 0.75, 0.23, 0.05])
        self.btn_reveal = Button(ax_reveal, 'Reveal source', color='#533483',
                                  hovercolor='#e94560')
        self.btn_reveal.label.set_color('white')
        self.btn_reveal.on_clicked(self.on_reveal)

        # Speed slider
        ax_speed = self.fig.add_axes([0.72, 0.68, 0.23, 0.03])
        self.slider_speed = Slider(ax_speed, 'Speed', 1, 100, valinit=10, valstep=1,
                                    color='#e94560')
        self.slider_speed.label.set_color('white')
        self.slider_speed.valtext.set_color('white')
        self.slider_speed.on_changed(self.on_speed_change)

        # Source radio buttons
        ax_radio = self.fig.add_axes([0.72, 0.25, 0.23, 0.38])
        ax_radio.set_facecolor('#16213e')
        source_labels = list(SOURCES.keys())
        self.radio = RadioButtons(ax_radio, source_labels, active=0,
                                   activecolor='#e94560')
        for label in self.radio.labels:
            label.set_color('white')
            label.set_fontsize(10)
        self.radio.on_clicked(self.on_source_change)

        # Source label
        ax_radio.set_title('Mystery Source', color='white', fontsize=11, pad=10)

        # Add Event button
        ax_add_event = self.fig.add_axes([0.72, 0.17, 0.23, 0.05])
        self.btn_add_event = Button(ax_add_event, 'Manually add event', color='#0f3460',
                                     hovercolor='#e94560')
        self.btn_add_event.label.set_color('white')
        self.btn_add_event.on_clicked(self.on_add_event)

        # Text box for manual event entry (initially hidden)
        self.ax_textbox = self.fig.add_axes([0.72, 0.10, 0.23, 0.05])
        self.textbox = TextBox(self.ax_textbox, '', initial='',
                               color='#16213e', hovercolor='#16213e')
        self.textbox.text_disp.set_color('white')
        self.ax_textbox.set_visible(False)
        self.textbox.on_submit(self.on_submit_event)

        self._update_theory_line()

    def _update_theory_line(self):
        """Add/update theoretical entropy dashed line (hidden until revealed)."""
        h = entropy_of_source(self.current_source)
        if self.entropy_theory_line is not None:
            self.entropy_theory_line.remove()
        self.entropy_theory_line = self.ax_entropy.axhline(
            y=h, color='#e2d810', linestyle='--', linewidth=1, alpha=0.7,
            label=f'True entropy: {h:.3f} bits')
        self.entropy_theory_line.set_visible(self.revealed)
        legend = self.ax_entropy.get_legend()
        if legend:
            legend.remove()
        if self.revealed:
            self.ax_entropy.legend(loc='upper right', fontsize=8,
                                    facecolor='#16213e', edgecolor='#aaa',
                                    labelcolor='white')

    def toggle_play(self, event=None):
        self.playing = not self.playing
        self.btn_play.label.set_text('Pause' if self.playing else 'Play')
        self.fig.canvas.draw_idle()

    def on_reset(self, event=None):
        self.reset_data()
        self.revealed = False
        self.btn_reveal.label.set_text('Reveal Distribution')
        self._clear_reveal()
        self._update_theory_line()
        self._redraw_all()
        self.fig.canvas.draw_idle()

    def on_reveal(self, event=None):
        self.revealed = not self.revealed
        if self.revealed:
            name = SOURCES[self.current_source]['name']
            self.btn_reveal.label.set_text(f'{name}')
            # Overlay true PDF on histogram, extending to visible range
            xlim = self.ax_hist.get_xlim()
            x_lo = max(xlim[0], 0.001)
            x_hi = min(xlim[1], 0.999)
            x = np.linspace(x_lo, x_hi, 500)
            pdf = source_pdf(self.current_source, x)
            # Scale PDF to match histogram (probability per bin = pdf * bin_width)
            if self.hist_pdf_line is not None:
                self.hist_pdf_line.remove()
            self.hist_pdf_line, = self.ax_hist.plot(x, pdf * BIN_WIDTH, color='#e2d810',
                                                     linewidth=2, linestyle='-', alpha=0.9,
                                                     label='True PDF (scaled)')
            self.ax_hist.legend(loc='upper right', fontsize=7,
                                 facecolor='#16213e', edgecolor='#aaa',
                                 labelcolor='white')
        else:
            self.btn_reveal.label.set_text('Reveal Distribution')
            self._clear_reveal()
        # Show/hide the theory entropy line
        self._update_theory_line()
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        """Scroll-to-zoom on histogram x-axis."""
        if event.inaxes != self.ax_hist:
            return
        zoom_factor = 1.2
        if event.button == 'up':
            scale = 1.0 / zoom_factor
        elif event.button == 'down':
            scale = zoom_factor
        else:
            return
        xlim = self.ax_hist.get_xlim()
        xdata = event.xdata
        new_left = xdata - (xdata - xlim[0]) * scale
        new_right = xdata + (xlim[1] - xdata) * scale
        self.ax_hist.set_xlim(new_left, new_right)
        self._stretch_outermost_bars()
        self._update_pdf_overlay()
        self.fig.canvas.draw_idle()

    def _stretch_outermost_bars(self):
        """Stretch overflow bars to fill visible x range beyond [0, 1]."""
        xlim = self.ax_hist.get_xlim()
        # Left overflow: from visible left edge to 0
        left_edge = min(xlim[0], BIN_EDGES[0])
        self.left_overflow_bar.set_x(left_edge)
        self.left_overflow_bar.set_width(BIN_EDGES[0] - left_edge)
        # Right overflow: from 1 to visible right edge
        right_edge = max(xlim[1], BIN_EDGES[-1])
        self.right_overflow_bar.set_x(BIN_EDGES[-1])
        self.right_overflow_bar.set_width(right_edge - BIN_EDGES[-1])

    def _update_pdf_overlay(self):
        """Redraw PDF overlay to cover visible x range."""
        if not self.revealed or self.hist_pdf_line is None:
            return
        xlim = self.ax_hist.get_xlim()
        # Clamp to valid domain for beta distributions (avoid 0 and 1 edges)
        x_lo = max(xlim[0], 0.001)
        x_hi = min(xlim[1], 0.999)
        if x_lo >= x_hi:
            return
        x = np.linspace(x_lo, x_hi, 500)
        pdf = source_pdf(self.current_source, x)
        self.hist_pdf_line.set_data(x, pdf * BIN_WIDTH)

    def _clear_reveal(self):
        if self.hist_pdf_line is not None:
            self.hist_pdf_line.remove()
            self.hist_pdf_line = None
        legend = self.ax_hist.get_legend()
        if legend:
            legend.remove()

    def on_source_change(self, label):
        self.current_source = label
        self.on_reset()

    def on_speed_change(self, val):
        self.speed = int(val)
        self.timer.interval = max(10, int(1000 / self.speed))

    def on_add_event(self, event=None):
        self.ax_textbox.set_visible(True)
        self.fig.canvas.draw_idle()
        # Focus the text box so the user can type immediately
        self.textbox.begin_typing()

    def on_submit_event(self, text):
        try:
            value = float(text)
        except ValueError:
            self.textbox.set_val('')
            return
        if value < 0 or value > 1:
            self.textbox.set_val('')
            return
        self._add_event(value)
        self.textbox.set_val('')

    def _add_event(self, value):
        """Add a single event value to the collection and update all displays."""
        self.events.append(value)
        s = surprisal_of_event(value, self.counts)
        # Update histogram counts
        bin_idx = get_bin_idx(value)
        self.counts[bin_idx] += 1
        h = compute_binned_entropy(self.counts)
        self.entropy_history.append(h)
        self.surprisal_history.append(s)
        avg = np.mean(self.surprisal_history)
        self.running_avg_surprisal.append(avg)
        self._redraw_all()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def step(self):
        if not self.playing:
            return
        value = sample_source(self.current_source)
        self._add_event(value)

    def _redraw_all(self):
        n = len(self.events)

        # --- Histogram ---
        total = self.counts.sum()
        if total > 0:
            probs = self.counts / total
        else:
            probs = np.zeros(N_FINITE_BINS + 2)
        # Finite bins (indices 1..N_FINITE_BINS in counts)
        for bar, p in zip(self.hist_bars, probs[1:N_FINITE_BINS + 1]):
            bar.set_height(p)
        # Overflow bins
        self.left_overflow_bar.set_height(probs[0])
        self.right_overflow_bar.set_height(probs[-1])
        ymax = max(probs.max() * 1.3, 0.01) if total > 0 else 0.2
        self.ax_hist.set_ylim(0, ymax)

        # --- Entropy ---
        if n > 0:
            self.entropy_line.set_data(range(1, n + 1), self.entropy_history)
            self.ax_entropy.set_xlim(1, max(n, 10))
            all_h = self.entropy_history + [entropy_of_source(self.current_source)]
            self.ax_entropy.set_ylim(min(all_h) - 0.3, max(all_h) + 0.3)
        else:
            self.entropy_line.set_data([], [])
            h_theory = entropy_of_source(self.current_source)
            self.ax_entropy.set_xlim(1, 10)
            self.ax_entropy.set_ylim(h_theory - 1, h_theory + 1)

        # --- Surprisal scatter ---
        if n > 0:
            indices = np.arange(1, n + 1)
            colors = np.array(self.surprisal_history)
            offsets = np.column_stack([indices, colors])
            self.surprisal_scatter.set_offsets(offsets)
            self.surprisal_scatter.set_array(colors)
            self.surprisal_avg_line.set_data(indices, self.running_avg_surprisal)
            self.ax_surprisal.set_xlim(1, max(n, 10))
            smax = max(max(self.surprisal_history), 1)
            self.ax_surprisal.set_ylim(0, smax * 1.2)
        else:
            self.surprisal_scatter.set_offsets(np.empty((0, 2)))
            self.surprisal_avg_line.set_data([], [])
            self.ax_surprisal.set_xlim(1, 10)
            self.ax_surprisal.set_ylim(0, 8)

        # --- Event visualizer ---
        if n > 0:
            last = self.events[-1]
            self.event_marker.set_data([last], [0])
            s = self.surprisal_history[-1]
            h_est = self.entropy_history[-1]
            self.event_text.set_text(
                f'Value: {last:.4f}\n'
                f'Surprisal: {s:.2f} bits\n'
                f'Entropy est: {h_est:.3f} bits\n'
                f'Events: {n}'
            )
        else:
            self.event_marker.set_data([], [])
            self.event_text.set_text('No events yet')

        # Stretch outermost bars to fill visible x range
        self._stretch_outermost_bars()
        self._update_pdf_overlay()


def main():
    sim = EntropySimulator()
    plt.show()


if __name__ == '__main__':
    main()
