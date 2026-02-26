#!/usr/bin/env python3
"""
ENTROPY EXPLORER
An interactive simulator for building intuition about entropy,
surprisal, and information theory.

Controls:
  [1-5]   Switch between hidden oracles
  [SPACE] Pause / resume event stream
  [r]     Reveal / hide the oracle's true distribution
  [c]     Clear data (same oracle)
  [+/-]   Increase / decrease event speed
  [q]     Quit
"""

import sys
import math
import time
import random
import threading
import tty
import termios
import select
from collections import deque
from typing import Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.align import Align
except ImportError:
    print("Install rich: pip install rich")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# ORACLE DEFINITIONS  (the "black boxes")
# ─────────────────────────────────────────────────────────────────

def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


ORACLES = [
    {
        "name": "ORACLE  Ⅰ",
        "color": "bright_cyan",
        "hint": "It speaks with perfect impartiality.",
        "sample": lambda: random.random(),
        "reveal": (
            "Uniform(0, 1)",
            "Every value in [0,1] is equally likely. "
            "This achieves MAXIMUM entropy for a continuous source — "
            "no outcome is more probable than any other.",
        ),
    },
    {
        "name": "ORACLE  Ⅱ",
        "color": "bright_magenta",
        "hint": "It is of two minds — always.",
        "sample": lambda: _clip(
            random.gauss(0.2, 0.07) if random.random() < 0.5
            else random.gauss(0.8, 0.07)
        ),
        "reveal": (
            "Bimodal Gaussian  (μ₁=0.2, μ₂=0.8, σ=0.07)",
            "Two equally likely clusters. High entropy overall because "
            "neither cluster is 'preferred' — but values in the middle "
            "are genuinely rare and very surprising!",
        ),
    },
    {
        "name": "ORACLE  Ⅲ",
        "color": "bright_green",
        "hint": "It always seeks the center.",
        "sample": lambda: _clip(random.gauss(0.5, 0.10)),
        "reveal": (
            "Gaussian(μ=0.5, σ=0.10)  clipped to [0,1]",
            "Bell curve centered at 0.5. Lower entropy than Uniform — "
            "events cluster predictably. Values near 0 or 1 carry "
            "enormous surprisal because they're so improbable.",
        ),
    },
    {
        "name": "ORACLE  Ⅳ",
        "color": "yellow",
        "hint": "It is repelled by the middle.",
        "sample": lambda: random.betavariate(0.35, 0.35),
        "reveal": (
            "Beta(α=0.35, β=0.35)  — U-shaped",
            "Events cluster near 0 AND near 1, rarely in the middle. "
            "The center is the surprising zone. High entropy because "
            "two very different regions are each common.",
        ),
    },
    {
        "name": "ORACLE  Ⅴ",
        "color": "bright_red",
        "hint": "It always drifts the same direction.",
        "sample": lambda: random.betavariate(0.5, 5.0),
        "reveal": (
            "Beta(α=0.5, β=5.0)  — strongly left-skewed",
            "Almost all events fall near 0. Lowest entropy of the five — "
            "very predictable. But when a high value appears, "
            "it carries enormous surprisal. One rare event = many bits!",
        ),
    },
]

# ─────────────────────────────────────────────────────────────────
# ENTROPY ESTIMATOR
# ─────────────────────────────────────────────────────────────────

N_BINS = 20
MAX_BITS = math.log2(N_BINS)  # ~4.32 bits


class EntropyEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = [0] * N_BINS
        self.total = 0
        self._surprisals: deque = deque(maxlen=300)
        self._total_bits = 0.0

    def observe(self, x: float) -> float:
        """Record event x ∈ [0,1]; return estimated surprisal in bits."""
        b = min(int(x * N_BINS), N_BINS - 1)
        # Laplace-smoothed probability (add-1 smoothing across all bins)
        p = (self.counts[b] + 1) / (self.total + N_BINS)
        surprisal = -math.log2(p)
        self.counts[b] += 1
        self.total += 1
        self._surprisals.append(surprisal)
        self._total_bits += surprisal
        return surprisal

    def entropy(self) -> float:
        """Shannon entropy estimate (bits)."""
        denom = self.total + N_BINS
        h = 0.0
        for c in self.counts:
            p = (c + 1) / denom
            h -= p * math.log2(p)
        return h

    def hist_fractions(self):
        """Heights relative to the tallest bin (for display)."""
        m = max(self.counts) if self.counts else 1
        if m == 0:
            return [0.0] * N_BINS
        return [c / m for c in self.counts]

    def hist_probs(self):
        """Smoothed probability per bin."""
        denom = self.total + N_BINS
        return [(c + 1) / denom for c in self.counts]

    def avg_surprisal(self) -> Optional[float]:
        if not self._surprisals:
            return None
        return sum(self._surprisals) / len(self._surprisals)

    def confidence(self) -> float:
        """0→1: how much to trust the estimate."""
        return min(1.0, self.total / 250)

    def total_bits(self) -> float:
        return self._total_bits


# ─────────────────────────────────────────────────────────────────
# SURPRISAL COLOR / LABEL HELPERS
# ─────────────────────────────────────────────────────────────────

_S_THRESHOLDS = [
    (1.0, "bright_blue",   "expected  "),
    (2.0, "cyan",          "likely    "),
    (3.0, "green",         "notable   "),
    (4.0, "yellow",        "unusual   "),
    (5.0, "orange3",       "rare!     "),
    (99., "bright_red",    "SHOCKING  "),
]


def s_color(s: float) -> str:
    for thr, col, _ in _S_THRESHOLDS:
        if s < thr:
            return col
    return "bright_red"


def s_label(s: float) -> str:
    for thr, _, lbl in _S_THRESHOLDS:
        if s < thr:
            return lbl
    return "SHOCKING  "


# ─────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────

def render_histogram(est: EntropyEstimator, color: str, rows: int = 8) -> Text:
    """P(x) per bin (Laplace-smoothed, Σ=1).
    1 char per bin = 20 chars wide. 6-char y-axis. Total = 26 chars/row."""
    probs = est.hist_probs()         # smoothed, sums to 1
    max_p = max(probs)
    uniform_p = 1.0 / N_BINS        # = 0.05 for N_BINS=20

    # Which display row (1=bottom, rows=top) straddles the uniform level?
    uniform_row = max(1, min(rows, math.ceil(uniform_p / max_p * rows)))

    text = Text()
    for row in range(rows, 0, -1):
        frac_hi = row / rows
        frac_lo = (row - 1) / rows
        is_ref = (row == uniform_row)

        # 6-char y-axis column (always exactly 6 chars)
        if row == rows:
            text.append(f"{max_p:.3f}", style="dim")   # 5 chars
            text.append("▕", style="dim")              # 1 char
        elif is_ref:
            text.append(" 1/N ", style="yellow dim")   # 5 chars
            text.append("╞", style="yellow dim")       # 1 char
        else:
            text.append("     ", style="")             # 5 chars
            text.append("│", style="dim")              # 1 char

        # 1 char per bin = N_BINS chars
        for p in probs:
            frac = p / max_p
            if frac >= frac_hi:
                text.append("█", style=color)
            elif frac > frac_lo:
                partial = (frac - frac_lo) / (frac_hi - frac_lo)
                ch = "▆" if partial > 0.75 else "▄" if partial > 0.50 else "▂"
                text.append(ch, style=color + " dim")
            elif is_ref:
                text.append("╌", style="yellow dim")
            else:
                text.append(" ")
        text.append("\n")

    # Bottom axis (6-char y-pad + N_BINS dashes)
    text.append("     └", style="dim")
    text.append("─" * N_BINS + "\n", style="dim")
    # x-axis labels at 0, 0.5, 1.0
    mid = N_BINS // 2
    text.append("      0", style="dim")
    text.append(" " * (mid - 2), style="dim")
    text.append("0.5", style="dim")
    text.append(" " * (N_BINS - mid - 4), style="dim")
    text.append("1.0\n", style="dim")

    # Compact p-value summary
    min_p = min(probs)
    text.append("      ", style="")
    text.append(f"min={min_p:.3f} ", style="dim")
    text.append(f"1/N={uniform_p:.3f} ", style="yellow dim")
    text.append(f"max={max_p:.3f}", style="dim")
    return text


def render_dotline(positions: deque, width: int = 56) -> Text:
    """Number line showing recent event positions as colored dots."""
    slots = [(" ", "")] * width
    items = list(positions)
    n = len(items)
    for i, (x, s) in enumerate(items):
        pos = min(int(x * width), width - 1)
        age = i / n if n > 0 else 1.0
        dot = "●" if age > 0.65 else ("•" if age > 0.30 else "·")
        slots[pos] = (dot, s_color(s))
    text = Text()
    text.append("▏", style="dim")
    for ch, col in slots:
        if ch == " ":
            text.append("─", style="dim")
        else:
            text.append(ch, style=col + " bold")
    text.append("▏", style="dim")
    return text


def render_entropy_bar(est: EntropyEstimator, width: int = 34) -> Text:
    """Colored entropy progress bar."""
    h = est.entropy()
    frac = h / MAX_BITS if MAX_BITS > 0 else 0
    filled = int(frac * width)
    if frac < 0.35:
        c = "bright_blue"
    elif frac < 0.60:
        c = "cyan"
    elif frac < 0.80:
        c = "green"
    elif frac < 0.93:
        c = "yellow"
    else:
        c = "bright_red"
    text = Text()
    text.append("  ▐", style="dim")
    text.append("█" * filled, style=c + " bold")
    text.append("░" * (width - filled), style="dim")
    text.append("▌  ", style="dim")
    text.append(f"{h:.2f}", style="bold")
    text.append(f" / {MAX_BITS:.2f} bits", style="dim")
    return text


def render_stream(events: deque, bar_w: int = 11) -> Text:
    """Scrolling event list: value, p̂(x), surprisal bar, bits, label."""
    text = Text()
    items = list(events)[-14:]
    items.reverse()
    for i, (val, surp) in enumerate(items):
        col = s_color(surp)
        lbl = s_label(surp)
        dim = "dim" if i > 4 else ""
        bold = "bold" if i == 0 else ""
        # p̂ is recoverable: surprisal = -log2(p)  →  p = 2^(-surprisal)
        p_hat = 2.0 ** (-surp)
        filled = min(int((surp / (MAX_BITS + 1)) * bar_w), bar_w)
        bar = "▪" * filled + "·" * (bar_w - filled)
        style = f"{col} {bold} {dim}".strip()
        text.append(f"  {val:.4f}", style=style)
        text.append(f"  p={p_hat:.3f}", style=f"dim {dim}")
        text.append("  │", style="dim")
        text.append(bar, style=style)
        text.append("│", style="dim")
        text.append(f" {surp:4.1f}b", style=style)
        text.append(f"  {lbl}\n", style=f"italic {dim}")
    return text


# ─────────────────────────────────────────────────────────────────
# APPLICATION STATE
# ─────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.oracle_idx = 0
        self.estimator = EntropyEstimator()
        self.events: deque = deque(maxlen=20)       # (value, surprisal)
        self.positions: deque = deque(maxlen=80)    # (value, surprisal)
        self.revealed = False
        self.paused = False
        self.running = True
        self.count = 0
        self._flash = ""
        self._flash_exp = 0.0

    @property
    def oracle(self):
        return ORACLES[self.oracle_idx]

    def switch(self, idx: int):
        with self.lock:
            self.oracle_idx = idx % len(ORACLES)
            self.estimator.reset()
            self.events.clear()
            self.positions.clear()
            self.revealed = False
            self.count = 0

    def add_event(self, x: float):
        with self.lock:
            s = self.estimator.observe(x)
            self.events.append((x, s))
            self.positions.append((x, s))
            self.count += 1

    def flash(self, msg: str, secs: float = 2.5):
        self._flash = msg
        self._flash_exp = time.time() + secs

    def flash_str(self) -> str:
        if self._flash and time.time() < self._flash_exp:
            return self._flash
        return ""


# ─────────────────────────────────────────────────────────────────
# DISPLAY BUILDER
# ─────────────────────────────────────────────────────────────────

def make_display(state: AppState) -> Layout:
    oracle = state.oracle
    est = state.estimator
    col = oracle["color"]

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )
    layout["left"].split_column(
        Layout(name="hist_panel", ratio=3),
        Layout(name="meter_panel", ratio=2),
    )
    layout["right"].split_column(
        Layout(name="dotline_panel", size=6),
        Layout(name="stream_panel"),
    )

    # ── HEADER ────────────────────────────────────────────────────
    status = "⏸ PAUSED" if state.paused else "▶ LIVE"
    fmsg = state.flash_str()
    hdr = Text(justify="center")
    hdr.append(" ◉ ENTROPY EXPLORER ◉ ", style="bold white on blue")
    hdr.append(f"   {status}   ", style="bold dim")
    hdr.append(f"  {oracle['name']}  ", style=f"bold {col}")
    hdr.append(f"  n = {state.count:,}  ", style="dim")
    if fmsg:
        hdr.append(f"   ★ {fmsg} ★", style="bold yellow")
    layout["header"].update(Panel(Align(hdr, "center"), box=box.HEAVY))

    # ── HISTOGRAM PANEL ───────────────────────────────────────────
    # Keep content height stable whether hidden or revealed.
    # Revealed description lives in the meter panel (below), not here.
    hp = Text()
    hp.append(f"\n  {oracle['name']}\n", style=f"bold {col}")
    if state.revealed:
        rname, _ = oracle["reveal"]
        hp.append(f"  ✦ {rname}\n\n", style=f"{col} italic")
    else:
        hp.append(f"  \"{oracle['hint']}\"\n\n", style="dim italic")
    hp.append_text(render_histogram(est, col))
    layout["hist_panel"].update(
        Panel(hp, title="[dim]P(x) — probability per bin  (Σ = 1)[/]",
              box=box.ROUNDED))

    # ── ENTROPY METER ─────────────────────────────────────────────
    mp = Text()
    h = est.entropy()
    frac = h / MAX_BITS
    conf = est.confidence()
    avg_s = est.avg_surprisal()
    total_b = est.total_bits()

    mp.append("\n  H(X) — Shannon Entropy\n\n", style="bold")
    mp.append_text(render_entropy_bar(est, width=32))
    mp.append("\n\n")

    if frac < 0.35:
        desc = "Very predictable — few bits per event"
    elif frac < 0.60:
        desc = "Structured — some patterns visible"
    elif frac < 0.80:
        desc = "Fairly random — hard to predict"
    elif frac < 0.93:
        desc = "High entropy — quite chaotic"
    else:
        desc = "Near-maximum entropy — pure chaos!"
    mp.append(f"  {desc}\n", style="italic dim")

    if state.revealed:
        _, rnote = oracle["reveal"]
        mp.append("\n", style="")
        # word-wrap at ~38 chars
        words = rnote.split()
        line, out = "  ", []
        for w in words:
            if len(line) + len(w) + 1 > 40:
                out.append(line)
                line = "  " + w + " "
            else:
                line += w + " "
        out.append(line)
        mp.append("\n".join(out) + "\n", style="dim italic")
    else:
        if avg_s is not None:
            mp.append(f"\n  Avg surprisal : {avg_s:.2f} bits/event\n", style="dim")
        mp.append(f"  Total info    : {total_b:.0f} bits received\n", style="dim")
        conf_filled = int(conf * 12)
        conf_bar = "█" * conf_filled + "░" * (12 - conf_filled)
        mp.append(f"  Estimate conf : [{conf_bar}] {int(conf*100)}%", style="dim")

    layout["meter_panel"].update(
        Panel(mp, title="[dim]Entropy Meter[/]", box=box.ROUNDED))

    # ── DOT LINE ──────────────────────────────────────────────────
    dp = Text()
    dp.append("\n  Where on [0 … 1] are events landing?\n\n  ", style="dim")
    dp.append_text(render_dotline(state.positions, width=55))
    dp.append("\n\n  ")
    for s_val, lbl_col, label in [
        (0.5, "bright_blue", "expected"),
        (2.5, "green",       "notable"),
        (4.5, "orange3",     "rare"),
        (6.0, "bright_red",  "shocking"),
    ]:
        dp.append("● ", style=s_color(s_val) + " bold")
        dp.append(label + "  ", style="dim")
    layout["dotline_panel"].update(
        Panel(dp, title="[dim]Number Line — dot color = surprisal[/]",
              box=box.ROUNDED))

    # ── EVENT STREAM ──────────────────────────────────────────────
    sp = Text()
    sp.append("\n  value    p(x)      │ surprisal   │ bits  character\n",
              style="dim")
    sp.append("  " + "─" * 52 + "\n", style="dim")
    sp.append_text(render_stream(state.events))
    sp.append("\n")
    sp.append("  ℹ  Surprisal(x) = −log₂ p(x).  "
              "Entropy H = 𝔼[surprisal].\n", style="dim italic")
    sp.append("     Low entropy ↔ predictable ↔ easy to compress.\n",
              style="dim italic")
    layout["stream_panel"].update(
        Panel(sp, title="[dim]Event Stream[/]", box=box.ROUNDED))

    # ── FOOTER ────────────────────────────────────────────────────
    ft = Text(justify="center")
    controls = [
        ("[1-5]", "Oracle"),
        ("[SPC]",  "Pause"),
        ("[r]",    "Reveal"),
        ("[c]",    "Clear"),
        ("[+/-]",  "Speed"),
        ("[q]",    "Quit"),
    ]
    for key, desc in controls:
        ft.append(key, style="bold yellow")
        ft.append(f" {desc}    ", style="dim")
    layout["footer"].update(Panel(Align(ft, "center"), box=box.HEAVY))

    return layout


# ─────────────────────────────────────────────────────────────────
# BACKGROUND EVENT GENERATOR
# ─────────────────────────────────────────────────────────────────

def event_loop(state: AppState, speed: list):
    while state.running:
        if not state.paused:
            x = state.oracle["sample"]()
            state.add_event(x)
        time.sleep(speed[0])


# ─────────────────────────────────────────────────────────────────
# KEYBOARD (non-blocking raw input)
# ─────────────────────────────────────────────────────────────────

def getch() -> Optional[str]:
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    console = Console()
    state = AppState()
    speed = [0.30]  # seconds between events

    # seed with a few events so the display isn't empty
    for _ in range(8):
        x = state.oracle["sample"]()
        state.add_event(x)

    gen = threading.Thread(target=event_loop, args=(state, speed), daemon=True)
    gen.start()

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setraw(fd)

    try:
        with Live(make_display(state), refresh_per_second=12,
                  screen=True, console=console) as live:
            while state.running:
                live.update(make_display(state))

                ch = getch()
                if ch:
                    if ch in ("q", "Q", "\x03", "\x04"):
                        state.running = False

                    elif ch in "12345":
                        idx = int(ch) - 1
                        state.switch(idx)
                        state.flash(f"Switched to {ORACLES[idx]['name']}")

                    elif ch == " ":
                        state.paused = not state.paused
                        state.flash("Paused — press SPACE to resume"
                                    if state.paused else "Resumed")

                    elif ch in ("r", "R"):
                        state.revealed = not state.revealed
                        if state.revealed:
                            state.flash(f"Revealed: {state.oracle['reveal'][0]}")
                        else:
                            state.flash("Distribution hidden")

                    elif ch in ("c", "C"):
                        idx = state.oracle_idx
                        state.switch(idx)
                        state.flash("Cleared — fresh start")

                    elif ch in ("+", "="):
                        speed[0] = max(0.04, speed[0] * 0.65)
                        state.flash(f"Speed ↑  {1/speed[0]:.1f} events/sec")

                    elif ch == "-":
                        speed[0] = min(3.0, speed[0] / 0.65)
                        state.flash(f"Speed ↓  {1/speed[0]:.1f} events/sec")

                time.sleep(0.04)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        console.clear()
        console.print()
        console.print("[bold cyan]  ◉ Thanks for exploring entropy! ◉[/]")
        console.print()
        console.print("  [dim]Key insight:[/] Entropy = average surprisal of a source.")
        console.print("  [dim]High H → unpredictable → each event carries more information.[/]")
        console.print("  [dim]Low  H → predictable  → events are mostly redundant.[/]")
        console.print()


if __name__ == "__main__":
    main()
