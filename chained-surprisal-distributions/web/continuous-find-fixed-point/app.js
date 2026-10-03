// Continuous surprisal fixed-point finder — port of continuous_find_fixed_point.py
// Each node fits a continuous density (KDE / Adaptive KDE / GMM / B-spline) blended
// with a Gaussian(mu,sigma) prior, then iterates events -> surprisal -> refit until
// the surprisal distribution stops moving (Wasserstein distance < tol).

// ───────────────────────── Constants ─────────────────────────
const X_MIN = -10, X_MAX = 30;
const S_MIN = 0, S_MAX = 25;
const GRID_N = 400;
const PRIOR_ALPHA_DEFAULT = 0;
const PRIOR_MU_DEFAULT = 0;
const PRIOR_SIGMA_DEFAULT = 5;
const BANDWIDTH_DEFAULT = 1.0;
const BIN_WIDTH_DEFAULT = 1.0;
const GMM_COMPONENTS_DEFAULT = 2;
const GMM_COMPONENTS_MAX = 8;
const ADAPTIVE_KDE_SENSITIVITY = 0.5;
const MAX_ITER = 1000;
const WASSERSTEIN_TOL_LOG10_DEFAULT = -10;

function linspace(a, b, n) {
  const out = new Array(n);
  if (n === 1) { out[0] = a; return out; }
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = a + step * i;
  return out;
}

const VALUE_GRID = linspace(X_MIN, X_MAX, GRID_N);
const SURP_GRID = linspace(S_MIN, S_MAX, GRID_N);

// ───────────────────────── Numeric helpers ─────────────────────────
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr, ddof = 0) {
  const n = arr.length;
  if (n - ddof <= 0) return 0;
  const m = mean(arr);
  const v = arr.reduce((a, x) => a + (x - m) * (x - m), 0) / (n - ddof);
  return Math.sqrt(v);
}
function ptp(arr) { return Math.max(...arr) - Math.min(...arr); }
function clipArr(arr, lo, hi) {
  return arr.map(v => {
    let x = v;
    if (lo !== null && x < lo) x = lo;
    if (hi !== null && x > hi) x = hi;
    return x;
  });
}
function clipScalar(x, lo, hi) {
  if (lo !== null && x < lo) x = lo;
  if (hi !== null && x > hi) x = hi;
  return x;
}
function trapz(y, x) {
  let s = 0;
  for (let i = 0; i < y.length - 1; i++) s += (y[i] + y[i + 1]) / 2 * (x[i + 1] - x[i]);
  return s;
}
function normalPdf(x, mu, sigma) {
  const s = Math.max(sigma, 1e-300);
  const z = (x - mu) / s;
  return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
}
function geometricMeanOfLogs(arr) {
  const m = mean(arr.map(v => Math.log(Math.max(v, 1e-300))));
  return Math.exp(m);
}

// ───────────────────────── Not-a-knot cubic spline (extrapolate=False) ─────────────────────────
// Solves for second derivatives M_i via a dense linear system (not-a-knot
// boundary rows + standard interior rows), matching scipy.interpolate.CubicSpline's
// default bc_type. Small systems only (<=61 points here), so Gaussian elimination
// is simpler and plenty fast rather than a banded/tridiagonal solver.
function cubicSplineNotAKnot(xs, ys) {
  const n = xs.length - 1; // n+1 points, n intervals
  const h = new Array(n);
  for (let i = 0; i < n; i++) h[i] = xs[i + 1] - xs[i];

  const A = Array.from({ length: n + 1 }, () => new Array(n + 1).fill(0));
  const b = new Array(n + 1).fill(0);

  if (n >= 3) {
    // Not-a-knot: third derivative continuous at x_1 and x_{n-1}
    A[0][0] = h[1];
    A[0][1] = -(h[0] + h[1]);
    A[0][2] = h[0];
    b[0] = 0;
    A[n][n - 2] = h[n - 1];
    A[n][n - 1] = -(h[n - 2] + h[n - 1]);
    A[n][n] = h[n - 2];
    b[n] = 0;
  } else {
    // Fall back to natural spline for degenerate tiny inputs.
    A[0][0] = 1; b[0] = 0;
    A[n][n] = 1; b[n] = 0;
  }
  for (let i = 1; i < n; i++) {
    A[i][i - 1] = h[i - 1];
    A[i][i] = 2 * (h[i - 1] + h[i]);
    A[i][i + 1] = h[i];
    b[i] = 6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
  }

  const M = gaussianSolve(A, b);

  return function spline(xq) {
    return xq.map(x => {
      if (x < xs[0] || x > xs[n]) return NaN;
      let lo = 0, hi = n - 1;
      if (x >= xs[n]) lo = n - 1;
      else {
        // binary search for interval containing x
        let l = 0, r = n;
        while (l < r) {
          const mid = (l + r) >> 1;
          if (xs[mid + 1] <= x) l = mid + 1; else r = mid;
        }
        lo = Math.min(l, n - 1);
      }
      const hi_ = h[lo];
      const a = xs[lo + 1] - x;
      const bb = x - xs[lo];
      const term1 = M[lo] * (a * a * a) / (6 * hi_);
      const term2 = M[lo + 1] * (bb * bb * bb) / (6 * hi_);
      const term3 = (ys[lo] / hi_ - M[lo] * hi_ / 6) * a;
      const term4 = (ys[lo + 1] / hi_ - M[lo + 1] * hi_ / 6) * bb;
      return term1 + term2 + term3 + term4;
    });
  };
}

function gaussianSolve(A, b) {
  const n = A.length;
  const M = A.map(row => row.slice());
  const rhs = b.slice();
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (piv !== col) { [M[col], M[piv]] = [M[piv], M[col]]; [rhs[col], rhs[piv]] = [rhs[piv], rhs[col]]; }
    const pv = M[col][col];
    if (Math.abs(pv) < 1e-14) continue;
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / pv;
      if (f === 0) continue;
      for (let c = col; c < n; c++) M[r][c] -= f * M[col][c];
      rhs[r] -= f * rhs[col];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = rhs[r];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = Math.abs(M[r][r]) < 1e-14 ? 0 : s / M[r][r];
  }
  return x;
}

// ───────────────────────── Density fitting ─────────────────────────

// EM Gaussian mixture, seeded from quantile splits of sorted events (not random
// init), so repeated refits on the same data land in the same place.
function emGmmPdf(events, nComponents) {
  const n = events.length;
  const k = Math.max(1, Math.min(nComponents, n));
  const sorted = events.slice().sort((a, b) => a - b);
  const splits = [];
  for (let i = 0; i < k; i++) {
    const start = Math.floor((i * n) / k);
    const end = Math.floor(((i + 1) * n) / k);
    splits.push(sorted.slice(start, Math.max(end, start + 1)));
  }
  let means = splits.map(s => mean(s));
  let stds = splits.map(s => Math.max(std(s), 1e-2));
  let weights = new Array(k).fill(1 / k);

  for (let iter = 0; iter < 25; iter++) {
    const resp = events.map(x => {
      const comp = new Array(k);
      let denom = 0;
      for (let j = 0; j < k; j++) {
        comp[j] = weights[j] * normalPdf(x, means[j], stds[j]);
        denom += comp[j];
      }
      denom = Math.max(denom, 1e-300);
      return comp.map(c => c / denom);
    });
    const Nk = new Array(k).fill(0);
    for (let j = 0; j < k; j++) {
      for (let i = 0; i < n; i++) Nk[j] += resp[i][j];
      Nk[j] = Math.max(Nk[j], 1e-8);
    }
    const newMeans = new Array(k).fill(0);
    for (let j = 0; j < k; j++) {
      for (let i = 0; i < n; i++) newMeans[j] += resp[i][j] * events[i];
      newMeans[j] /= Nk[j];
    }
    const newVar = new Array(k).fill(0);
    for (let j = 0; j < k; j++) {
      for (let i = 0; i < n; i++) {
        const d = events[i] - newMeans[j];
        newVar[j] += resp[i][j] * d * d;
      }
      newVar[j] /= Nk[j];
    }
    means = newMeans;
    stds = newVar.map(v => Math.sqrt(Math.max(v, 1e-4)));
    weights = Nk.map(v => v / n);
  }

  return function pdf(xs) {
    return xs.map(x => {
      let s = 0;
      for (let j = 0; j < k; j++) s += weights[j] * normalPdf(x, means[j], stds[j]);
      return s;
    });
  };
}

// Scott's rule bandwidth factor for 1-D data, as used by scipy.stats.gaussian_kde.
function scottsFactor(n) { return Math.pow(n, -1 / 5); }

function kdePdf(events, bwFactor) {
  const n = events.length;
  const h = scottsFactor(n) * bwFactor * std(events, 1);
  const hSafe = Math.max(h, 1e-300);
  return function pdf(xs) {
    return xs.map(x => {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const z = (x - events[i]) / hSafe;
        s += Math.exp(-0.5 * z * z) / (hSafe * Math.sqrt(2 * Math.PI));
      }
      return s / n;
    });
  };
}

// Balloon-style KDE: a pilot fixed-bandwidth KDE estimates local density at each
// event, then each event's own kernel bandwidth is shrunk in dense regions and
// widened in sparse ones.
function adaptiveKdePdf(events, bwFactor) {
  const n = events.length;
  const h0 = scottsFactor(n) * bwFactor * std(events, 1);
  const h0Safe = Math.max(h0, 1e-300);
  const pilotAt = events.map(x => {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const z = (x - events[i]) / h0Safe;
      s += Math.exp(-0.5 * z * z) / (h0Safe * Math.sqrt(2 * Math.PI));
    }
    return Math.max(s / n, 1e-300);
  });
  const g = geometricMeanOfLogs(pilotAt);
  const localH = pilotAt.map(p => Math.max(h0Safe * Math.pow(p / g, -ADAPTIVE_KDE_SENSITIVITY), 1e-3));

  return function pdf(xs) {
    return xs.map(x => {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const hi = localH[i];
        const z = (x - events[i]) / hi;
        s += Math.exp(-0.5 * z * z) / (hi * Math.sqrt(2 * Math.PI));
      }
      return s / n;
    });
  };
}

// Histogram -> Gaussian-smoothed bin heights -> cubic spline through sqrt(height)
// (squaring on evaluation keeps the result non-negative). Compact support: density
// is exactly 0 outside the padded histogram range.
function bsplinePdf(events, bwFactor) {
  const n = events.length;
  const nBins = Math.max(6, Math.min(60, Math.floor(Math.sqrt(n)) + 2));
  const lo = Math.min(...events), hi = Math.max(...events);
  const pad = (hi - lo) * 0.15 + 1e-6;
  const edges = linspace(lo - pad, hi + pad, nBins + 1);
  const counts = new Array(nBins).fill(0);
  for (const x of events) {
    let idx = Math.floor((x - edges[0]) / ((edges[nBins] - edges[0]) / nBins));
    idx = clipScalar(idx, 0, nBins - 1);
    counts[idx] += 1;
  }
  const widths = new Array(nBins);
  const centers = new Array(nBins);
  for (let i = 0; i < nBins; i++) {
    widths[i] = edges[i + 1] - edges[i];
    centers[i] = (edges[i] + edges[i + 1]) / 2;
  }
  const histDensity = counts.map((c, i) => c / (n * widths[i]));

  const sigmaBins = Math.max(0.6, bwFactor * 1.5);
  const radius = Math.max(1, Math.ceil(3 * sigmaBins));
  const kernel = [];
  for (let k = -radius; k <= radius; k++) kernel.push(Math.exp(-0.5 * (k / sigmaBins) * (k / sigmaBins)));
  const kSum = kernel.reduce((a, b) => a + b, 0);
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;

  const padded = new Array(histDensity.length + 2 * radius).fill(0);
  for (let i = 0; i < histDensity.length; i++) padded[i + radius] = histDensity[i];
  const smoothed = new Array(nBins).fill(0);
  for (let i = 0; i < nBins; i++) {
    let s = 0;
    for (let k = 0; k < kernel.length; k++) s += padded[i + k] * kernel[k];
    smoothed[i] = s;
  }

  const sqrtHeights = smoothed.map(v => Math.sqrt(Math.max(v, 0)));
  const spline = cubicSplineNotAKnot(centers, sqrtHeights);

  function raw(xs) {
    const vals = spline(xs);
    return vals.map(v => {
      if (isNaN(v)) return 0;
      return Math.max(v, 0) ** 2;
    });
  }

  const grid = linspace(edges[0], edges[nBins], 400);
  let mass = trapz(raw(grid), grid);
  if (!(mass > 1e-12)) mass = 1.0;
  return function pdf(xs) { return raw(xs).map(v => v / mass); };
}

function makeDensityFn(events, alpha, mu, sigma, method, bwFactor, nComponents) {
  const priorPdf = (xs) => xs.map(x => normalPdf(x, mu, sigma));
  const n = events.length;
  if (n === 0) return (xs) => xs.map(() => 0);

  let rawPdf;
  if (n === 1 || ptp(events) < 1e-9) {
    const center = mean(events);
    const narrow = Math.max(sigma, 1e-2) * 0.05;
    rawPdf = (xs) => xs.map(x => normalPdf(x, center, narrow));
  } else if (method === 'gmm') {
    rawPdf = emGmmPdf(events, nComponents);
  } else if (method === 'adaptive_kde') {
    rawPdf = adaptiveKdePdf(events, bwFactor);
  } else if (method === 'bspline') {
    rawPdf = bsplinePdf(events, bwFactor);
  } else {
    rawPdf = kdePdf(events, bwFactor);
  }

  if (alpha <= 0) return rawPdf;
  const w = n / (n + alpha);
  return function pdf(xs) {
    const r = rawPdf(xs);
    const p = priorPdf(xs);
    return xs.map((_, i) => w * r[i] + (1 - w) * p[i]);
  };
}

// ───────────────────────── Surprisal / entropy / divergence ─────────────────────────

function surprisalBits(xs, densityFn, dx = 1.0) {
  const d = densityFn(xs);
  return d.map(v => -Math.log2(Math.max(v * dx, 1e-300)));
}

function differentialEntropyBits(densityFn, grid) {
  const p = densityFn(grid).map(v => Math.max(v, 1e-300));
  const integrand = p.map(v => -v * Math.log2(v));
  return trapz(integrand, grid);
}

function klDivergenceBits(pFn, qFn, grid) {
  const p = pFn(grid).map(v => Math.max(v, 1e-300));
  const q = qFn(grid).map(v => Math.max(v, 1e-300));
  const integrand = p.map((pv, i) => (pv > 1e-12 ? pv * Math.log2(pv / q[i]) : 0));
  return trapz(integrand, grid);
}

function wassersteinDistance(pFn, qFn, grid) {
  let p = pFn(grid).map(v => Math.max(v, 0));
  let q = qFn(grid).map(v => Math.max(v, 0));
  const pMass = trapz(p, grid);
  const qMass = trapz(q, grid);
  if (!(pMass > 0) || !(qMass > 0)) return NaN;
  p = p.map(v => v / pMass);
  q = q.map(v => v / qMass);
  const dx = [];
  for (let i = 0; i < grid.length - 1; i++) dx.push(grid[i + 1] - grid[i]);
  const Fp = [0], Fq = [0];
  for (let i = 0; i < dx.length; i++) {
    Fp.push(Fp[i] + (p[i] + p[i + 1]) / 2 * dx[i]);
    Fq.push(Fq[i] + (q[i] + q[i + 1]) / 2 * dx[i]);
  }
  const diff = Fp.map((v, i) => Math.abs(v - Fq[i]));
  return trapz(diff, grid);
}

function computeFixedPointIterations(events, alpha, mu, sigma, method, bwFactor, nComponents, tol, dx) {
  if (events.length === 0) return { nIter: null, density: null, events: null, history: null };
  const p1Density = makeDensityFn(events, alpha, mu, sigma, method, bwFactor, nComponents);
  let currentEvents = surprisalBits(events, p1Density, dx);
  let density = makeDensityFn(currentEvents, alpha, mu, sigma, method, bwFactor, nComponents);
  const history = [];
  for (let i = 0; i < MAX_ITER; i++) {
    const newEvents = surprisalBits(currentEvents, density, dx);
    const newDensity = makeDensityFn(newEvents, alpha, mu, sigma, method, bwFactor, nComponents);
    const w1 = wassersteinDistance(density, newDensity, SURP_GRID);
    history.push([
      klDivergenceBits(density, newDensity, SURP_GRID),
      klDivergenceBits(newDensity, density, SURP_GRID),
      w1,
    ]);
    if (w1 < tol) return { nIter: i + 1, density: newDensity, events: newEvents, history };
    currentEvents = newEvents;
    density = newDensity;
  }
  return { nIter: null, density: null, events: null, history: null };
}

// ───────────────────────── Event families (port of events.py) ─────────────────────────

const FAMILIES = {
  Uniform: {
    params: [
      { name: 'low', start: -10, end: 10, value: 0.0, step: 0.1 },
      { name: 'high', start: -10, end: 10, value: 1.0, step: 0.1 },
    ],
    sample: (p) => p.low + Math.random() * Math.max(p.high - p.low, 1e-6),
  },
  Normal: {
    params: [
      { name: 'μ', start: -10, end: 10, value: 0.0, step: 0.1 },
      { name: 'σ', start: 0.1, end: 10, value: 1.0, step: 0.1 },
    ],
    sample: (p) => p['μ'] + p['σ'] * jStat.normal.sample(0, 1),
  },
  Beta: {
    params: [
      { name: 'α', start: 0.1, end: 10, value: 2.0, step: 0.1 },
      { name: 'β', start: 0.1, end: 10, value: 5.0, step: 0.1 },
    ],
    sample: (p) => jStat.beta.sample(p['α'], p['β']),
  },
  Exponential: {
    params: [
      { name: 'λ', start: 0.1, end: 10, value: 1.0, step: 0.1 },
    ],
    sample: (p) => -Math.log(Math.random()) / p['λ'],
  },
};
const FAMILY_NAMES = Object.keys(FAMILIES);

function getEvents(n, family, params) {
  const fn = FAMILIES[family].sample;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = fn(params);
  return out;
}

// ───────────────────────── Chart.js rug plugin ─────────────────────────
// Draws vertical tick marks at the given x-values near the bottom of a chart,
// approximating the Bokeh "segment" rug glyphs. Registered per-chart.
function makeRugPlugin(getXs, getColor, getAlphaVisible) {
  return {
    id: 'rugPlugin-' + Math.random().toString(36).slice(2),
    afterDraw(chart) {
      const { alpha, visible } = getAlphaVisible();
      if (!visible) return;
      const xs = getXs();
      if (!xs || xs.length === 0) return;
      const xScale = chart.scales.x, yScale = chart.scales.y;
      const ctx = chart.ctx;
      const yBottom = yScale.getPixelForValue(yScale.min);
      const rugFrac = (yScale.max - yScale.min) * 0.03;
      const yTop = yScale.getPixelForValue(yScale.min + rugFrac);
      ctx.save();
      ctx.strokeStyle = getColor();
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const x of xs) {
        if (x < xScale.min || x > xScale.max) continue;
        const px = xScale.getPixelForValue(x);
        ctx.moveTo(px, yBottom);
        ctx.lineTo(px, yTop);
      }
      ctx.stroke();
      ctx.restore();
    },
  };
}

// ───────────────────────── App state ─────────────────────────

Chart.defaults.color = '#aaaaaa';
Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';

let rootEvents = [];
let allEvents = [];
let historyIndex = 0;
let transportGuard = false;
let oneByOneHandle = null; // non-null while "Add events (one by one)" animation is running

let sessionRecord = 0;
const sessionRecordRows = [];

const rugState = { visible: false, alpha: 0.15 };

function makeNode(opts) {
  // opts: canvasId, titleEl, xMin, xMax, alphaEnd, muRange, sigmaRange, methodRowEl,
  //       binWidthEl (optional), yScaleSelectEl
  const node = {
    events: [],
    grid: opts.grid,
    method: 'kde',
    alpha: PRIOR_ALPHA_DEFAULT,
    mu: PRIOR_MU_DEFAULT,
    sigma: opts.sigmaDefault !== undefined ? opts.sigmaDefault : PRIOR_SIGMA_DEFAULT,
    bw: BANDWIDTH_DEFAULT,
    nComponents: GMM_COMPONENTS_DEFAULT,
    binWidth: BIN_WIDTH_DEFAULT,
    yRangeAdaptive: true,
    currentDensity: null,
    chart: null,
  };

  const ctx = document.getElementById(opts.canvasId).getContext('2d');
  node.chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'density',
        data: [],
        borderColor: '#2255AA',
        backgroundColor: 'rgba(72,120,207,0.4)',
        fill: true,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      scales: {
        x: { type: 'linear', min: opts.xMin, max: opts.xMax, title: { display: true, text: opts.xLabel } },
        y: { min: 0, max: 1, title: { display: true, text: 'Density' } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: (c) => `density: ${c.parsed.y.toFixed(4)}`,
            title: (items) => `x: ${items[0].parsed.x.toFixed(3)}`,
          },
        },
      },
    },
    plugins: [makeRugPlugin(
      () => node.events,
      () => '#444444',
      () => ({ alpha: rugState.alpha, visible: rugState.visible }),
    )],
  });

  node.titleEl = document.getElementById(opts.titleId);
  node.yScaleSelectEl = document.getElementById(opts.yScaleSelectId);
  node.yScaleSelectEl.addEventListener('change', () => {
    node.yRangeAdaptive = node.yScaleSelectEl.value === 'adaptive';
    recompute();
  });

  node.paramRowEl = document.getElementById(opts.paramRowId);
  buildParamRow(node, opts);

  if (opts.binWidthSliderId) {
    node.binWidthSliderEl = document.getElementById(opts.binWidthSliderId);
    node.binWidthValEl = document.getElementById(opts.binWidthValId);
    node.binWidthSliderEl.addEventListener('input', () => {
      node.binWidth = parseFloat(node.binWidthSliderEl.value);
      node.binWidthValEl.textContent = node.binWidth.toFixed(2);
      recompute();
    });
  } else {
    node.binWidthSliderEl = null;
  }

  return node;
}

function buildParamRow(node, opts) {
  node.paramRowEl.innerHTML = '';

  const alphaSlider = makeLabeledSlider('Prior strength α', 0, opts.alphaEnd !== undefined ? opts.alphaEnd : 5, 0.1, node.alpha,
    (v) => { node.alpha = v; recompute(); });
  const muSlider = makeLabeledSlider('Prior mean μ', opts.muRange[0], opts.muRange[1], 0.1, node.mu,
    (v) => { node.mu = v; recompute(); });
  const sigmaSlider = makeLabeledSlider('Prior std dev σ', opts.sigmaRange[0], opts.sigmaRange[1], 0.1, node.sigma,
    (v) => { node.sigma = v; recompute(); });
  const bwSlider = makeLabeledSlider('Bandwidth / smoothing factor', 0.1, 3.0, 0.05, node.bw,
    (v) => { node.bw = v; recompute(); });
  const nCompSlider = makeLabeledSlider('GMM components', 1, GMM_COMPONENTS_MAX, 1, node.nComponents,
    (v) => { node.nComponents = Math.round(v); recompute(); });

  node.paramRowEl.appendChild(alphaSlider.el);
  node.paramRowEl.appendChild(muSlider.el);
  node.paramRowEl.appendChild(sigmaSlider.el);

  const methodShowsBw = ['kde', 'adaptive_kde', 'bspline'].includes(node.method);
  bwSlider.el.style.display = methodShowsBw ? '' : 'none';
  nCompSlider.el.style.display = node.method === 'gmm' ? '' : 'none';
  node.paramRowEl.appendChild(bwSlider.el);
  node.paramRowEl.appendChild(nCompSlider.el);

  const methodSelect = document.createElement('select');
  for (const [val, label] of [['kde', 'KDE'], ['adaptive_kde', 'Adaptive KDE'], ['gmm', 'GMM'], ['bspline', 'B-spline']]) {
    const o = document.createElement('option');
    o.value = val; o.textContent = label;
    if (val === node.method) o.selected = true;
    methodSelect.appendChild(o);
  }
  methodSelect.addEventListener('change', () => {
    node.method = methodSelect.value;
    bwSlider.el.style.display = ['kde', 'adaptive_kde', 'bspline'].includes(node.method) ? '' : 'none';
    nCompSlider.el.style.display = node.method === 'gmm' ? '' : 'none';
    recompute();
  });
  node.paramRowEl.appendChild(methodSelect);

  node._sliders = { alphaSlider, muSlider, sigmaSlider, bwSlider, nCompSlider, methodSelect };
}

function makeLabeledSlider(title, start, end, step, value, onChange) {
  const el = document.createElement('label');
  el.className = 'field';
  const span = document.createElement('span');
  span.className = 'val';
  span.textContent = `${title}: ${value}`;
  const input = document.createElement('input');
  input.type = 'range';
  input.min = start; input.max = end; input.step = step; input.value = value;
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    span.textContent = `${title}: ${v}`;
    onChange(v);
  });
  el.appendChild(span);
  el.appendChild(input);
  return { el, input };
}

function updateCurve(node, densityFn, grid) {
  const y = densityFn(grid);
  node.chart.data.datasets[0].data = grid.map((x, i) => ({ x, y: y[i] }));
  node.currentDensity = densityFn;
  const yMax = y.length ? Math.max(...y) : 1.0;
  node.chart.options.scales.y.max = node.yRangeAdaptive ? (yMax * 1.05 || 1.0) : 1.0;
  node.chart.update('none');
}

// ───────────────────────── Nodes ─────────────────────────

let node, surpNode;

function initNodes() {
  node = makeNode({
    canvasId: 'p1-canvas', titleId: 'p1-title', paramRowId: 'p1-param-row',
    yScaleSelectId: 'p1-yscale-select',
    binWidthSliderId: 'p1-binwidth-slider', binWidthValId: 'p1-binwidth-val',
    xMin: X_MIN, xMax: X_MAX, xLabel: 'Value', grid: VALUE_GRID,
    alphaEnd: 5, muRange: [-10, 10], sigmaRange: [0.1, 20], sigmaDefault: PRIOR_SIGMA_DEFAULT,
  });
  surpNode = makeNode({
    canvasId: 'sp1-canvas', titleId: 'sp1-title', paramRowId: 'sp1-param-row',
    yScaleSelectId: 'sp1-yscale-select',
    xMin: S_MIN, xMax: S_MAX, xLabel: 'Surprisal (bits)', grid: SURP_GRID,
    alphaEnd: 5, muRange: [0, 20], sigmaRange: [0.1, 10], sigmaDefault: 5,
  });
}

function updateSurpNode() {
  if (node.currentDensity === null || node.events.length === 0) {
    surpNode.events = [];
    updateCurve(surpNode, (xs) => xs.map(() => 0), SURP_GRID);
    surpNode.titleEl.textContent = 'S(P1) — First Surprisal Distribution';
    return;
  }
  const dx = node.binWidthSliderEl ? node.binWidth : 1.0;
  const surpEvents = surprisalBits(node.events, node.currentDensity, dx);
  surpNode.events = surpEvents;
  const densityFn = makeDensityFn(surpEvents, surpNode.alpha, surpNode.mu, surpNode.sigma,
    surpNode.method, surpNode.bw, surpNode.nComponents);
  updateCurve(surpNode, densityFn, SURP_GRID);
  surpNode.titleEl.textContent = `S(P1)  |  entropy = ${differentialEntropyBits(densityFn, SURP_GRID).toFixed(4)} bits`;
}

// ───────────────────────── Overlay chart ─────────────────────────

let overlayChart, overlayCount = 0;
let overlayAllEvents = [];

function initOverlayChart() {
  const ctx = document.getElementById('overlay-canvas').getContext('2d');
  overlayChart = new Chart(ctx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      scales: {
        x: { type: 'linear', min: S_MIN, max: S_MAX, title: { display: true, text: 'Surprisal (bits)' } },
        y: { min: 0, max: 1, title: { display: true, text: 'Density' } },
      },
      plugins: { legend: { display: false } },
    },
    plugins: [makeRugPlugin(
      () => overlayAllEvents,
      () => '#444444',
      () => ({ alpha: rugState.alpha, visible: rugState.visible }),
    )],
  });
}

function addToOverlay(densityFn, events) {
  const y = densityFn(SURP_GRID);
  overlayChart.data.datasets.push({
    data: SURP_GRID.map((x, i) => ({ x, y: y[i] })),
    borderColor: 'rgba(34,102,170,0.25)',
    borderWidth: 2,
    pointRadius: 0,
    fill: false,
    tension: 0,
  });
  overlayCount += 1;
  let yMax = 1.0;
  for (const ds of overlayChart.data.datasets) {
    for (const pt of ds.data) if (pt.y > yMax) yMax = pt.y;
  }
  overlayChart.options.scales.y.max = yMax * 1.1 || 1.0;
  document.getElementById('overlay-title').textContent = `Fixed-point overlays — ${overlayCount} points`;

  overlayAllEvents = overlayAllEvents.concat(events);
  overlayChart.update('none');
}

function onClearOverlays() {
  overlayChart.data.datasets = [];
  overlayCount = 0;
  overlayAllEvents = [];
  overlayChart.options.scales.y.max = 1.0;
  document.getElementById('overlay-title').textContent = 'Fixed-point overlays — 0 points';
  overlayChart.update('none');
}

// ───────────────────────── Progression chart ─────────────────────────

let progressionChart;
let progressionYScale = 'log';

function logSafe(vals) { return vals.map(v => (v > 0 ? v : null)); }

function initProgressionChart() {
  const ctx = document.getElementById('progression-canvas').getContext('2d');
  progressionChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: 'KL forward  (Pᵢ‖Pᵢ₊₁)', data: [], borderColor: '#2266AA', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#2266AA', tension: 0, spanGaps: false },
        { label: 'KL backward (Pᵢ₊₁‖Pᵢ)', data: [], borderColor: '#CC6633', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#CC6633', tension: 0, spanGaps: false },
        { label: 'W1', data: [], borderColor: '#339933', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#339933', tension: 0, spanGaps: false },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      scales: {
        x: { type: 'linear', title: { display: true, text: 'Iteration' } },
        y: { type: 'logarithmic', title: { display: true, text: 'Divergence (log scale)' } },
      },
      plugins: { legend: { display: true, position: 'right' } },
    },
  });
}

function updateProgressionPlot(history) {
  if (!history || history.length === 0) {
    for (const ds of progressionChart.data.datasets) ds.data = [];
    progressionChart.update('none');
    return;
  }
  const iters = history.map((_, i) => i + 1);
  const klFwdRaw = history.map(h => h[0]);
  const klBwdRaw = history.map(h => h[1]);
  const w1Raw = history.map(h => h[2]);

  let seriesFwd, seriesBwd, seriesW1, yMax;
  if (progressionYScale === 'log') {
    seriesFwd = logSafe(klFwdRaw); seriesBwd = logSafe(klBwdRaw); seriesW1 = logSafe(w1Raw);
  } else {
    seriesFwd = klFwdRaw.map(v => Math.max(v, 0));
    seriesBwd = klBwdRaw.map(v => Math.max(v, 0));
    seriesW1 = w1Raw.map(v => Math.max(v, 0));
    yMax = Math.max(...klFwdRaw, ...klBwdRaw, ...w1Raw);
  }
  progressionChart.data.datasets[0].data = iters.map((x, i) => ({ x, y: seriesFwd[i] }));
  progressionChart.data.datasets[1].data = iters.map((x, i) => ({ x, y: seriesBwd[i] }));
  progressionChart.data.datasets[2].data = iters.map((x, i) => ({ x, y: seriesW1[i] }));

  if (progressionYScale === 'linear') {
    progressionChart.options.scales.y.max = yMax * 1.1 || 1.0;
    progressionChart.options.scales.y.min = 0;
  } else {
    delete progressionChart.options.scales.y.max;
    delete progressionChart.options.scales.y.min;
  }
  progressionChart.update('none');
}

function setProgressionYScale(scale) {
  progressionYScale = scale;
  progressionChart.options.scales.y.type = scale === 'log' ? 'logarithmic' : 'linear';
  progressionChart.options.scales.y.title.text = scale === 'log' ? 'Divergence (log scale)' : 'Divergence';
}

// ───────────────────────── Top-level event controls ─────────────────────────

let appendReplaceActive = 0; // 0 = append, 1 = replace
let currentParamSliders = []; // [{name, el, input}]

const familySelect = document.getElementById('family-select');
const distParamsRow = document.getElementById('dist-params-row');
const nEventsInput = document.getElementById('n-events-input');
const addEventsBtn = document.getElementById('add-events-btn');
const addEventsOneByOneBtn = document.getElementById('add-events-one-by-one-btn');
const clearEventsBtn = document.getElementById('clear-events-btn');
const singleEventInput = document.getElementById('single-event-input');
const singleEventCountInput = document.getElementById('single-event-count-input');
const singleEventStatus = document.getElementById('single-event-status');

const historyBackBtn = document.getElementById('history-back-btn');
const historyFwdBtn = document.getElementById('history-fwd-btn');
const historySlider = document.getElementById('history-slider');
const historyLabel = document.getElementById('history-label');

const tolSlider = document.getElementById('tol-slider');
const tolVal = document.getElementById('tol-val');
const rugCheckbox = document.getElementById('rug-checkbox');
const rugAlphaSlider = document.getElementById('rug-alpha-slider');
const rugAlphaVal = document.getElementById('rug-alpha-val');

const convergenceDiv = document.getElementById('convergence-div');

function initFamilySelect() {
  for (const name of FAMILY_NAMES) {
    const o = document.createElement('option');
    o.value = name; o.textContent = name;
    familySelect.appendChild(o);
  }
  familySelect.value = FAMILY_NAMES[0];
  rebuildParamSliders(FAMILY_NAMES[0]);
  familySelect.addEventListener('change', () => {
    rebuildParamSliders(familySelect.value);
    if (appendReplaceActive === 1) doReplace();
  });
}

function rebuildParamSliders(familyName) {
  distParamsRow.innerHTML = '';
  currentParamSliders = FAMILIES[familyName].params.map(spec => {
    const s = makeLabeledSlider(spec.name, spec.start, spec.end, spec.step, spec.value, () => {
      if (appendReplaceActive === 1) doReplace();
    });
    s.name = spec.name;
    distParamsRow.appendChild(s.el);
    return s;
  });
}

function getCurrentParams() {
  const out = {};
  for (const s of currentParamSliders) out[s.name] = parseFloat(s.input.value);
  return out;
}

document.getElementById('append-replace-toggle').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  appendReplaceActive = parseInt(btn.dataset.val);
  for (const b of e.currentTarget.children) b.classList.toggle('active', b === btn);
});

function updateTransportState() {
  const n = allEvents.length;
  transportGuard = true;
  historySlider.max = Math.max(n, 1);
  historySlider.value = historyIndex;
  historySlider.disabled = n === 0;
  transportGuard = false;
  historyLabel.textContent = `Step ${historyIndex} of ${n}`;
  historyBackBtn.disabled = historyIndex === 0;
  historyFwdBtn.disabled = historyIndex === n;
  clearEventsBtn.disabled = n === 0;
}

function onMakeDist() {
  node.events = rootEvents.slice();
  recompute();
}

function parseN() {
  const n = parseInt(nEventsInput.value);
  if (!Number.isFinite(n) || n <= 0) { nEventsInput.value = '1000'; return 1000; }
  return n;
}

function onAddEvents() {
  if (appendReplaceActive === 1) { doReplace(); return; }
  const n = parseN();
  const wasAtEnd = historyIndex === allEvents.length;
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = allEvents.concat(newEv);
  if (wasAtEnd) historyIndex = allEvents.length;
  rootEvents = allEvents.slice(0, historyIndex);
  updateTransportState();
  onMakeDist();
}

function doReplace() {
  const n = parseN();
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = newEv;
  historyIndex = allEvents.length;
  rootEvents = allEvents.slice();
  updateTransportState();
  onMakeDist();
}

function onClearEvents() {
  rootEvents = []; allEvents = []; historyIndex = 0;
  onMakeDist();
  updateTransportState();
}

function onSingleEventInput() {
  const valStr = singleEventInput.value.trim();
  if (!valStr) return;
  const val = parseFloat(valStr);
  if (!Number.isFinite(val)) {
    singleEventStatus.textContent = `'${valStr}' is not a valid number.`;
    singleEventInput.value = '';
    return;
  }
  let count = parseInt(singleEventCountInput.value);
  if (!Number.isFinite(count)) count = 0;
  const n = Math.max(count, 1);
  if (appendReplaceActive === 1) {
    allEvents = new Array(n).fill(val);
    historyIndex = n;
  } else {
    const wasAtEnd = historyIndex === allEvents.length;
    allEvents = allEvents.concat(new Array(n).fill(val));
    if (wasAtEnd) historyIndex = allEvents.length;
  }
  rootEvents = allEvents.slice(0, historyIndex);
  singleEventStatus.textContent = `Added ${n} event${n > 1 ? 's' : ''} at ${val}.`;
  singleEventInput.value = '';
  updateTransportState();
  onMakeDist();
}

function applyHistoryIndex(idx) {
  historyIndex = Math.max(0, Math.min(idx, allEvents.length));
  rootEvents = allEvents.slice(0, historyIndex);
  updateTransportState();
  onMakeDist();
}

function onAddEventsOneByOne() {
  if (oneByOneHandle !== null) {
    clearTimeout(oneByOneHandle);
    oneByOneHandle = null;
    addEventsOneByOneBtn.textContent = 'Add events (one by one)';
    return;
  }
  const n = parseN();
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = allEvents.concat(newEv);
  const targetIndex = allEvents.length;

  addEventsOneByOneBtn.textContent = 'Stop';

  function step() {
    if (oneByOneHandle === null) { addEventsOneByOneBtn.textContent = 'Add events (one by one)'; return; }
    applyHistoryIndex(historyIndex + 1);
    if (historyIndex >= targetIndex) {
      oneByOneHandle = null;
      addEventsOneByOneBtn.textContent = 'Add events (one by one)';
    } else {
      oneByOneHandle = setTimeout(step, 0);
    }
  }
  oneByOneHandle = setTimeout(step, 0);
}

// ───────────────────────── Recompute ─────────────────────────

function recompute() {
  const alpha = node.alpha, mu = node.mu, sigma = node.sigma;
  const bw = node.bw, method = node.method, nComponents = node.nComponents;
  const densityFn = makeDensityFn(node.events, alpha, mu, sigma, method, bw, nComponents);
  updateCurve(node, densityFn, VALUE_GRID);
  node.titleEl.textContent = `P1  |  entropy = ${differentialEntropyBits(densityFn, VALUE_GRID).toFixed(4)} bits`;

  updateSurpNode();

  const dx = node.binWidthSliderEl ? node.binWidth : 1.0;
  const tol = Math.pow(10, parseFloat(tolSlider.value));
  const result = computeFixedPointIterations(node.events, alpha, mu, sigma, method, bw, nComponents, tol, dx);

  if (result.nIter === null && node.events.length === 0) {
    convergenceDiv.innerHTML = '<i>Add events to compute fixed-point iterations.</i>';
    updateProgressionPlot(null);
    return;
  } else if (result.nIter === null) {
    convergenceDiv.innerHTML = `<b>Did not converge within ${MAX_ITER} iterations.</b>`;
    updateProgressionPlot(null);
    return;
  }

  updateProgressionPlot(result.history);
  addToOverlay(result.density, result.events);
  const fixedEntropy = differentialEntropyBits(result.density, SURP_GRID);

  const s = result.nIter === 1 ? 'iteration' : 'iterations';
  let currentLine = `<b>Fixed point reached in ${result.nIter} ${s}.</b> Entropy = ${fixedEntropy.toFixed(4)} bits.`;

  if (result.nIter > sessionRecord) {
    sessionRecord = result.nIter;
    let distDesc = familySelect.value;
    if (currentParamSliders.length) {
      const paramStr = currentParamSliders.map(sl => `${sl.name}=${parseFloat(sl.input.value).toPrecision(3)}`).join(', ');
      distDesc += `(${paramStr})`;
    }
    const nEvents = node.events.length;
    const row = `<b>${result.nIter}</b> ${s} — ${distDesc} | events count: ${nEvents} | ` +
      `α=${alpha.toPrecision(3)}, μ=${mu.toPrecision(3)}, σ=${sigma.toPrecision(3)}, bw=${bw.toPrecision(2)}`;
    sessionRecordRows.unshift(row);
  }

  if (sessionRecordRows.length) {
    convergenceDiv.innerHTML = currentLine + '<br><br><b>Records this session:</b><br>' + sessionRecordRows.join('<br>');
  } else {
    convergenceDiv.innerHTML = currentLine;
  }
}

// ───────────────────────── Wiring ─────────────────────────

function initControls() {
  addEventsBtn.addEventListener('click', onAddEvents);
  addEventsOneByOneBtn.addEventListener('click', onAddEventsOneByOne);
  clearEventsBtn.addEventListener('click', onClearEvents);
  singleEventInput.addEventListener('change', onSingleEventInput);
  singleEventInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') onSingleEventInput(); });

  historySlider.addEventListener('input', () => {
    if (transportGuard) return;
    applyHistoryIndex(parseInt(historySlider.value));
  });
  historyBackBtn.addEventListener('click', () => applyHistoryIndex(historyIndex - 1));
  historyFwdBtn.addEventListener('click', () => applyHistoryIndex(historyIndex + 1));

  tolSlider.addEventListener('input', () => {
    tolVal.textContent = tolSlider.value;
    recompute();
  });

  rugCheckbox.addEventListener('change', () => {
    rugState.visible = rugCheckbox.checked;
    node.chart.update('none'); surpNode.chart.update('none'); overlayChart.update('none');
  });
  rugAlphaSlider.addEventListener('input', () => {
    rugState.alpha = parseFloat(rugAlphaSlider.value);
    rugAlphaVal.textContent = rugState.alpha.toFixed(2);
    node.chart.update('none'); surpNode.chart.update('none'); overlayChart.update('none');
  });

  document.getElementById('clear-overlays-btn').addEventListener('click', onClearOverlays);
  document.getElementById('progression-yscale-select').addEventListener('change', (e) => {
    setProgressionYScale(e.target.value);
    recompute();
  });
}

function init() {
  initFamilySelect();
  initNodes();
  initOverlayChart();
  initProgressionChart();
  initControls();
  updateTransportState();
  recompute();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
