"""
Interactive Entropy Simulator
Visualizes entropy, surprisal, and distribution convergence with mystery sources.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
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

def entropy_of_source(key):
    """Compute theoretical binned entropy in bits (matches histogram computation)."""
    from scipy.integrate import quad
    bin_probs = np.zeros(N_BINS)
    for i in range(N_BINS):
        if key == 'Source F':
            p, _ = quad(lambda x: 0.5 * stats.beta.pdf(x, 2, 2) + 0.5 * stats.beta.pdf(x, 20, 20),
                        BIN_EDGES[i], BIN_EDGES[i + 1])
        else:
            p = SOURCES[key]['dist'].cdf(BIN_EDGES[i + 1]) - SOURCES[key]['dist'].cdf(BIN_EDGES[i])
        bin_probs[i] = p
    bin_probs = bin_probs[bin_probs > 0]
    return -np.sum(bin_probs * np.log2(bin_probs))


# --- Binned entropy computation ---
N_BINS = 20
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH = 1.0 / N_BINS

def compute_binned_entropy(counts):
    """Entropy in bits from histogram counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def surprisal_of_event(value, counts):
    """Surprisal in bits for a single event given current histogram."""
    bin_idx = np.clip(np.searchsorted(BIN_EDGES, value, side='right') - 1, 0, N_BINS - 1)
    counts = counts + 1
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
        self.counts = np.zeros(N_BINS, dtype=float)
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
        self.hist_bars = self.ax_hist.bar(BIN_CENTERS, np.zeros(N_BINS),
                                          width=BIN_WIDTH * 0.9, color='#0f3460',
                                          edgecolor='#e94560', linewidth=0.5)
        self.hist_pdf_line = None  # for reveal

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
            # Overlay true PDF on histogram
            x = np.linspace(0.001, 0.999, 500)
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

    def step(self):
        if not self.playing:
            return
        # Generate event
        value = sample_source(self.current_source)
        self.events.append(value)

        # Update histogram counts
        bin_idx = np.clip(np.searchsorted(BIN_EDGES, value, side='right') - 1, 0, N_BINS - 1)
        self.counts[bin_idx] += 1

        # Compute entropy
        h = compute_binned_entropy(self.counts)
        self.entropy_history.append(h)

        # Compute surprisal (use counts BEFORE this event for surprisal, but after is fine too)
        s = surprisal_of_event(value, self.counts)
        self.surprisal_history.append(s)
        # Running average
        avg = np.mean(self.surprisal_history)
        self.running_avg_surprisal.append(avg)

        self._redraw_all()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _redraw_all(self):
        n = len(self.events)

        # --- Histogram ---
        total = self.counts.sum()
        if total > 0:
            probs = self.counts / total
        else:
            probs = np.zeros(N_BINS)
        for bar, p in zip(self.hist_bars, probs):
            bar.set_height(p)
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


def main():
    sim = EntropySimulator()
    plt.show()


if __name__ == '__main__':
    main()
