"use strict";
/* Continuous (density-based) analogue of the discrete iterated_surprisal_distributions
 * web port. Each node fits a continuous density to its events via one of three
 * methods (KDE / adaptive-KDE / EM-GMM / B-spline), blended with a Gaussian(mu,sigma)
 * prior weighted by pseudocount alpha. Deriving a child node maps each parent event x
 * to surprisal S(x) = -log2(density(x) * dx). Ported 1:1 from
 * continuous_iterated_surprisal_distributions.py — see that file for the reference math.
 */

// ───────────────────────────── Constants ──────────────────────────────────
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
const DENSITY_METHODS = [["kde", "KDE"], ["adaptive_kde", "Adaptive KDE"], ["gmm", "GMM"], ["bspline", "B-spline"]];
const PLOT_WIDTH = 860, PLOT_HEIGHT = 360;
const TRACE_PALETTE = ["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"];

const KL_LINK = '<a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence" target="_blank">KL</a>';
const W1_LINK = '<a href="https://en.wikipedia.org/wiki/Wasserstein_metric" target="_blank">W1</a>';

function linspace(a, b, n) {
  const out = new Float64Array(n);
  if (n === 1) { out[0] = a; return out; }
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = a + step * i;
  return out;
}
const VALUE_GRID = linspace(X_MIN, X_MAX, GRID_N);
const SURP_GRID = linspace(S_MIN, S_MAX, GRID_N);

// ───────────────────────────── Math utilities ──────────────────────────────
function trapz(y, x) {
  let s = 0;
  for (let i = 0; i < y.length - 1; i++) s += (y[i] + y[i + 1]) * 0.5 * (x[i + 1] - x[i]);
  return s;
}
function cumtrapz(y, x) {
  const out = new Float64Array(y.length);
  for (let i = 1; i < y.length; i++) out[i] = out[i - 1] + (y[i - 1] + y[i]) * 0.5 * (x[i] - x[i - 1]);
  return out;
}
function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
function stdPop(a) { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2; return Math.sqrt(s / a.length); }
function stdUnbiased(a) {
  if (a.length < 2) return 0;
  const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
  return Math.sqrt(s / (a.length - 1));
}
function ptp(a) { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return hi - lo; }
function clip(x, lo, hi) { return Math.max(lo, hi === undefined ? x : Math.min(x, hi)); }
function clipArr(a, lo, hi) { const out = new Float64Array(a.length); for (let i = 0; i < a.length; i++) out[i] = clip(a[i], lo, hi); return out; }
function normPdf(x, loc, scale) {
  const z = (x - loc) / scale;
  return Math.exp(-0.5 * z * z) / (scale * Math.sqrt(2 * Math.PI));
}
function toArray(x) { return (x instanceof Float64Array || Array.isArray(x)) ? x : [x]; }

// ───────────────────────────── RNG / event families ────────────────────────
function randUniform(lo, hi) { return lo + (hi - lo) * Math.random(); }
function randNormal(mu, sigma) {
  let u1 = Math.random(), u2 = Math.random();
  if (u1 < 1e-12) u1 = 1e-12;
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}
function randExponential(rate) {
  const u = Math.random();
  return -Math.log(1 - u) / rate;
}
function randGamma(shape) {
  // Marsaglia-Tsang (shape >= 1); boost trick (Ahrens-Dieter) for shape < 1.
  if (shape < 1) {
    const g = randGamma(shape + 1);
    const u = Math.random();
    return g * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      x = randNormal(0, 1);
      v = (1 + c * x) ** 3;
    } while (v <= 0);
    const u = Math.random();
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
}
function randBeta(a, b) {
  const x = randGamma(a), y = randGamma(b);
  return x / (x + y);
}

const FAMILIES = {
  Uniform: {
    params: [
      { name: "low", start: -10, end: 10, value: 0.0, step: 0.1 },
      { name: "high", start: -10, end: 10, value: 1.0, step: 0.1 },
    ],
    fn: (p, n) => { const out = new Float64Array(n); const scale = Math.max(p["high"] - p["low"], 1e-6); for (let i = 0; i < n; i++) out[i] = randUniform(p["low"], p["low"] + scale); return out; },
  },
  Normal: {
    params: [
      { name: "μ", start: -10, end: 10, value: 0.0, step: 0.1 },
      { name: "σ", start: 0.1, end: 10, value: 1.0, step: 0.1 },
    ],
    fn: (p, n) => { const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = randNormal(p["μ"], p["σ"]); return out; },
  },
  Beta: {
    params: [
      { name: "α", start: 0.1, end: 10, value: 2.0, step: 0.1 },
      { name: "β", start: 0.1, end: 10, value: 5.0, step: 0.1 },
    ],
    fn: (p, n) => { const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = randBeta(p["α"], p["β"]); return out; },
  },
  Exponential: {
    params: [
      { name: "λ", start: 0.1, end: 10, value: 1.0, step: 0.1 },
    ],
    fn: (p, n) => { const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = randExponential(p["λ"]); return out; },
  },
};
const FAMILY_NAMES = Object.keys(FAMILIES);
function getEvents(n, family, params) { return FAMILIES[family].fn(params, n); }

// ───────────────────────────── Density fits ─────────────────────────────────

function _emGmmPdf(events, nComponents) {
  const n = events.length;
  const k = Math.max(1, Math.min(nComponents, n));
  const sorted = Float64Array.from(events).sort();
  // numpy.array_split: first (n % k) chunks get one extra element.
  const base = Math.floor(n / k), rem = n % k;
  const splits = [];
  let off = 0;
  for (let i = 0; i < k; i++) {
    const size = base + (i < rem ? 1 : 0);
    splits.push(sorted.slice(off, off + size));
    off += size;
  }
  let means = splits.map(mean);
  let stds = splits.map(s => Math.max(stdPop(s), 1e-2));
  let weights = new Array(k).fill(1 / k);

  for (let iter = 0; iter < 25; iter++) {
    // Built with an explicit loop (not events.map/Float64Array.map): each row is itself
    // an array of k responsibilities, which Float64Array.map would silently coerce to NaN.
    const resp = new Array(n);
    for (let i = 0; i < n; i++) {
      const x = events[i];
      const comp = new Array(k);
      let denom = 0;
      for (let j = 0; j < k; j++) { comp[j] = weights[j] * normPdf(x, means[j], stds[j]); denom += comp[j]; }
      denom = Math.max(denom, 1e-300);
      resp[i] = comp.map(c => c / denom);
    }
    const Nk = new Array(k).fill(0);
    for (let j = 0; j < k; j++) { for (let i = 0; i < n; i++) Nk[j] += resp[i][j]; Nk[j] = Math.max(Nk[j], 1e-8); }
    const newMeans = new Array(k).fill(0);
    for (let j = 0; j < k; j++) { let s = 0; for (let i = 0; i < n; i++) s += resp[i][j] * events[i]; newMeans[j] = s / Nk[j]; }
    const newVar = new Array(k).fill(0);
    for (let j = 0; j < k; j++) { let s = 0; for (let i = 0; i < n; i++) s += resp[i][j] * (events[i] - newMeans[j]) ** 2; newVar[j] = Math.max(s / Nk[j], 1e-4); }
    means = newMeans;
    stds = newVar.map(Math.sqrt);
    weights = Nk.map(v => v / n);
  }

  return (xq) => {
    const arr = toArray(xq);
    const out = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      let s = 0;
      for (let j = 0; j < k; j++) s += weights[j] * normPdf(arr[i], means[j], stds[j]);
      out[i] = s;
    }
    return out;
  };
}

function _adaptiveKdePdf(events, bwFactor) {
  const n = events.length;
  const scottsFactor = Math.pow(n, -1 / 5);
  const h0 = scottsFactor * bwFactor * stdUnbiased(events);
  const sqrt2pi = Math.sqrt(2 * Math.PI);
  const pilotAt = (xq) => {
    const out = new Float64Array(xq.length);
    for (let i = 0; i < xq.length; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) { const z = (xq[i] - events[j]) / h0; s += Math.exp(-0.5 * z * z); }
      out[i] = s / (n * h0 * sqrt2pi);
    }
    return out;
  };
  const pilotAtEvents = clipArr(pilotAt(events), 1e-300, Infinity);
  let logSum = 0;
  for (let i = 0; i < n; i++) logSum += Math.log(pilotAtEvents[i]);
  const g = Math.exp(logSum / n);
  const localH = new Float64Array(n);
  for (let i = 0; i < n; i++) localH[i] = clip(h0 * Math.pow(pilotAtEvents[i] / g, -ADAPTIVE_KDE_SENSITIVITY), 1e-3, Infinity);

  return (xq) => {
    const arr = toArray(xq);
    const out = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) {
        const h = localH[j];
        const z = (arr[i] - events[j]) / h;
        s += Math.exp(-0.5 * z * z) / (h * sqrt2pi);
      }
      out[i] = s / n;
    }
    return out;
  };
}

function _kdePdf(events, bwFactor) {
  const n = events.length;
  const scottsFactor = Math.pow(n, -1 / 5);
  const h0 = scottsFactor * bwFactor * stdUnbiased(events);
  const sqrt2pi = Math.sqrt(2 * Math.PI);
  return (xq) => {
    const arr = toArray(xq);
    const out = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) { const z = (arr[i] - events[j]) / h0; s += Math.exp(-0.5 * z * z); }
      out[i] = s / (n * h0 * sqrt2pi);
    }
    return out;
  };
}

// Natural-not-a-knot cubic spline through (xs, ys); matches scipy.interpolate.CubicSpline
// with its default bc_type='not-a-knot'. Evaluates to NaN outside [xs[0], xs[last]].
function cubicSplineNotAKnot(xs, ys) {
  const n = xs.length;
  const h = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = xs[i + 1] - xs[i];

  const A = Array.from({ length: n }, () => new Float64Array(n));
  const b = new Float64Array(n);

  if (n >= 4) {
    A[0][0] = h[1]; A[0][1] = -(h[0] + h[1]); A[0][2] = h[0]; b[0] = 0;
    A[n - 1][n - 3] = h[n - 2]; A[n - 1][n - 2] = -(h[n - 3] + h[n - 2]); A[n - 1][n - 1] = h[n - 3]; b[n - 1] = 0;
  } else {
    // Degenerate fallback (shouldn't occur given n_bins >= 6): natural spline.
    A[0][0] = 1; b[0] = 0;
    A[n - 1][n - 1] = 1; b[n - 1] = 0;
  }
  for (let i = 1; i < n - 1; i++) {
    A[i][i - 1] = h[i - 1];
    A[i][i] = 2 * (h[i - 1] + h[i]);
    A[i][i + 1] = h[i];
    b[i] = 6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
  }

  const M = solveLinear(A, b);

  return {
    xs, ys, h, M,
    evaluate(x) {
      if (x < xs[0] || x > xs[n - 1]) return NaN;
      let i = 0;
      while (i < n - 2 && x > xs[i + 1]) i++;
      const hi = h[i];
      const a = (xs[i + 1] - x), c = (x - xs[i]);
      return (M[i] * a ** 3) / (6 * hi) + (M[i + 1] * c ** 3) / (6 * hi)
        + (ys[i] / hi - M[i] * hi / 6) * a + (ys[i + 1] / hi - M[i + 1] * hi / 6) * c;
    },
  };
}

function solveLinear(A, b) {
  const n = b.length;
  const M = A.map(row => Float64Array.from(row));
  const x = Float64Array.from(b);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (piv !== col) { [M[col], M[piv]] = [M[piv], M[col]]; [x[col], x[piv]] = [x[piv], x[col]]; }
    const pivVal = M[col][col] || 1e-300;
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / pivVal;
      if (f === 0) continue;
      for (let c = col; c < n; c++) M[r][c] -= f * M[col][c];
      x[r] -= f * x[col];
    }
  }
  const sol = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * sol[c];
    sol[r] = s / (M[r][r] || 1e-300);
  }
  return sol;
}

function _bsplinePdf(events, bwFactor) {
  const n = events.length;
  const nBins = clip(Math.floor(Math.sqrt(n)) + 2, 6, 60);
  let lo = Infinity, hi = -Infinity;
  for (const v of events) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const pad = (hi - lo) * 0.15 + 1e-6;
  const edges = linspace(lo - pad, hi + pad, nBins + 1);
  const counts = new Float64Array(nBins);
  for (const v of events) {
    let idx = Math.floor((v - edges[0]) / (edges[edges.length - 1] - edges[0]) * nBins);
    idx = clip(idx, 0, nBins - 1);
    // Refine with exact edge search (handles non-uniform float error near boundaries).
    while (idx > 0 && v < edges[idx]) idx--;
    while (idx < nBins - 1 && v >= edges[idx + 1]) idx++;
    counts[idx]++;
  }
  const widths = new Float64Array(nBins), centers = new Float64Array(nBins), histDensity = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) {
    widths[i] = edges[i + 1] - edges[i];
    centers[i] = (edges[i] + edges[i + 1]) / 2;
    histDensity[i] = counts[i] / (n * widths[i]);
  }

  const sigmaBins = Math.max(0.6, bwFactor * 1.5);
  const radius = Math.max(1, Math.ceil(3 * sigmaBins));
  const kernel = [];
  let kSum = 0;
  for (let k = -radius; k <= radius; k++) { const v = Math.exp(-0.5 * (k / sigmaBins) ** 2); kernel.push(v); kSum += v; }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;
  const padded = new Float64Array(nBins + 2 * radius);
  padded.set(histDensity, radius);
  const smoothed = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) {
    let s = 0;
    for (let j = 0; j < kernel.length; j++) s += padded[i + j] * kernel[j];
    smoothed[i] = s;
  }
  const sqrtSmoothed = Float64Array.from(smoothed, v => Math.sqrt(Math.max(v, 0)));
  const spline = cubicSplineNotAKnot(centers, sqrtSmoothed);

  const raw = (xq) => {
    const arr = toArray(xq);
    const out = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      let v = spline.evaluate(arr[i]);
      if (Number.isNaN(v)) v = 0;
      v = Math.max(v, 0);
      out[i] = v * v;
    }
    return out;
  };

  const grid = linspace(edges[0], edges[edges.length - 1], 400);
  let mass = trapz(raw(grid), grid);
  if (!(mass > 1e-12)) mass = 1.0;
  return (xq) => { const r = raw(xq); const out = new Float64Array(r.length); for (let i = 0; i < r.length; i++) out[i] = r[i] / mass; return out; };
}

function makeDensityFn(events, alpha, mu, sigma, method, bwFactor, nComponents) {
  const priorPdf = (xq) => { const arr = toArray(xq); const out = new Float64Array(arr.length); for (let i = 0; i < arr.length; i++) out[i] = normPdf(arr[i], mu, sigma); return out; };
  const n = events.length;
  if (n === 0) return (xq) => new Float64Array(toArray(xq).length);

  let rawPdf;
  if (n === 1 || ptp(events) < 1e-9) {
    const center = mean(events);
    const narrow = Math.max(sigma, 1e-2) * 0.05;
    rawPdf = (xq) => { const arr = toArray(xq); const out = new Float64Array(arr.length); for (let i = 0; i < arr.length; i++) out[i] = normPdf(arr[i], center, narrow); return out; };
  } else if (method === "gmm") {
    rawPdf = _emGmmPdf(events, nComponents);
  } else if (method === "adaptive_kde") {
    rawPdf = _adaptiveKdePdf(events, bwFactor);
  } else if (method === "bspline") {
    rawPdf = _bsplinePdf(events, bwFactor);
  } else {
    rawPdf = _kdePdf(events, bwFactor);
  }

  if (alpha <= 0) return rawPdf;
  const w = n / (n + alpha);
  return (xq) => {
    const r = rawPdf(xq), p = priorPdf(xq);
    const out = new Float64Array(r.length);
    for (let i = 0; i < r.length; i++) out[i] = w * r[i] + (1 - w) * p[i];
    return out;
  };
}

function surprisalBits(xArr, densityFn, dx) {
  const p = densityFn(xArr);
  const out = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) out[i] = -Math.log2(clip(p[i] * dx, 1e-300, Infinity));
  return out;
}
function differentialEntropyBits(densityFn, grid) {
  const p = clipArr(densityFn(grid), 1e-300, Infinity);
  const integrand = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) integrand[i] = -p[i] * Math.log2(p[i]);
  return trapz(integrand, grid);
}
function klDivergenceBits(pFn, qFn, grid) {
  const p = clipArr(pFn(grid), 1e-300, Infinity), q = clipArr(qFn(grid), 1e-300, Infinity);
  const integrand = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) integrand[i] = p[i] > 1e-12 ? p[i] * Math.log2(p[i] / q[i]) : 0;
  return trapz(integrand, grid);
}
function wassersteinDistance(pFn, qFn, grid) {
  let p = clipArr(pFn(grid), 0, Infinity), q = clipArr(qFn(grid), 0, Infinity);
  const pMass = trapz(p, grid), qMass = trapz(q, grid);
  if (!(pMass > 0) || !(qMass > 0)) return NaN;
  const pn = new Float64Array(p.length), qn = new Float64Array(q.length);
  for (let i = 0; i < p.length; i++) { pn[i] = p[i] / pMass; qn[i] = q[i] / qMass; }
  const Fp = cumtrapz(pn, grid), Fq = cumtrapz(qn, grid);
  const diff = new Float64Array(Fp.length);
  for (let i = 0; i < Fp.length; i++) diff[i] = Math.abs(Fp[i] - Fq[i]);
  return trapz(diff, grid);
}

// ───────────────────────────── DOM helpers ──────────────────────────────────
function el(tag, props = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "style") Object.assign(e.style, v);
    else if (k === "className") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) if (c) e.appendChild(c);
  return e;
}
function fmt(x, d = 4) { return Number.isFinite(x) ? x.toFixed(d) : String(x); }

function createSlider({ min, max, value, step, title, width = 220 }) {
  const valueSpan = el("span", { className: "val", text: String(value) });
  const input = el("input", { type: "range", min, max, step, value });
  const wrapper = el("label", { className: "field", style: { width: width + "px" } }, [
    el("span", {}, [document.createTextNode(title + ": "), valueSpan]),
    input,
  ]);
  input.addEventListener("input", () => { valueSpan.textContent = Number(input.value).toLocaleString(undefined, { maximumFractionDigits: 3 }); });
  return { wrapper, input, title, get value() { return Number(input.value); }, set value(v) { input.value = v; valueSpan.textContent = Number(v).toLocaleString(undefined, { maximumFractionDigits: 3 }); } };
}
function createSelect({ options, value, width = 140, title }) {
  const select = el("select", { style: { width: width + "px" } },
    options.map(([v, label]) => el("option", { value: v, text: label })));
  select.value = value;
  const wrapper = title ? el("label", { className: "field" }, [el("span", { text: title }), select]) : select;
  return { wrapper, select, get value() { return select.value; }, set value(v) { select.value = v; } };
}
function createToggleGroup(labels, activeIdx, onChange) {
  const buttons = labels.map((l, i) => el("button", {
    className: "toggle-group" + (i === activeIdx ? " active" : ""), type: "button", text: l,
  }));
  let active = activeIdx;
  buttons.forEach((btn, i) => btn.addEventListener("click", () => {
    active = i;
    buttons.forEach((b, j) => b.classList.toggle("active", j === i));
    onChange(i);
  }));
  const wrapper = el("div", { className: "row", style: { gap: "0", marginBottom: "0" } }, buttons);
  return { wrapper, get active() { return active; }, set active(i) { active = i; buttons.forEach((b, j) => b.classList.toggle("active", j === i)); } };
}
function createCheckbox(label) {
  const input = el("input", { type: "checkbox" });
  const wrapper = el("label", { className: "checkbox-row" }, [input, document.createTextNode(label)]);
  return { wrapper, input, get checked() { return input.checked; }, set checked(v) { input.checked = v; } };
}

// ───────────────────────────── Busy indicator ───────────────────────────────
const busyDiv = document.getElementById("busy-indicator");
function showBusy() { busyDiv.style.display = "block"; }
function hideBusy() { busyDiv.style.display = "none"; }
function busy(fn) {
  return (...args) => {
    showBusy();
    setTimeout(() => { try { fn(...args); } finally { hideBusy(); } }, 0);
  };
}
function busyChange(fn) {
  return (...args) => {
    if (suspendRecompute) return;
    showBusy();
    setTimeout(() => { try { fn(...args); } finally { hideBusy(); } }, 0);
  };
}

// ───────────────────────────── Global state ─────────────────────────────────
let rootEvents = new Float64Array(0);
let rootNode = null;
let allEvents = new Float64Array(0);
let historyIndex = 0;
let transportCbGuard = false;
let columnCount = 1;
const allNodes = [];
let stepHandle = null;
let traceIndices = [];
let suspendRecompute = false;
let rugVisible = false;
let rugAlpha = 0.15;

function concatF64(a, b) { const out = new Float64Array(a.length + b.length); out.set(a, 0); out.set(b, a.length); return out; }

// ───────────────────────────── PNode factory ────────────────────────────────

function makePNode(initialEvents, depth) {
  const isSurprisal = depth > 0;
  const grid = isSurprisal ? SURP_GRID : VALUE_GRID;
  const domain = isSurprisal ? { start: S_MIN, end: S_MAX } : { start: X_MIN, end: X_MAX };
  const xLabel = isSurprisal ? "Surprisal (bits)" : "Value";

  const nd = {
    events: initialEvents, depth, isSurprisal, grid, domain,
    viewRange: { start: domain.start, end: domain.end },
    yRangeEnd: 1.0, yAdaptive: true,
    densityY: new Float64Array(grid.length),
    currentDensity: null,
    child: null, parent: null,
    propagates: false,
    traceMarkers: [],
    titleText: "P  |  entropy = 0.0000 bits",
    hover: { x: null },
  };

  // Canvas + reset button + hover tip
  const canvas = el("canvas", { className: "density-canvas", width: PLOT_WIDTH, height: PLOT_HEIGHT });
  const resetBtn = el("button", { className: "canvas-reset-btn", type: "button", title: "Reset view", text: "⟲" });
  const hoverTip = el("div", { className: "hover-tip" });
  const canvasWrap = el("div", { className: "node-canvas-wrap" }, [canvas, resetBtn, hoverTip]);
  nd.canvas = canvas; nd.ctx = canvas.getContext("2d");

  resetBtn.addEventListener("click", () => { nd.viewRange = { start: domain.start, end: domain.end }; drawNode(nd); });

  let dragging = false, dragStartX = 0, dragStartRange = null;
  canvas.addEventListener("mousedown", (e) => { dragging = true; dragStartX = e.offsetX; dragStartRange = { ...nd.viewRange }; });
  window.addEventListener("mouseup", () => { dragging = false; });
  canvas.addEventListener("mousemove", (e) => {
    const { left, plotW } = canvasMetrics();
    if (dragging) {
      const dxPx = e.offsetX - dragStartX;
      const span = dragStartRange.end - dragStartRange.start;
      const dxData = -(dxPx / plotW) * span;
      nd.viewRange = { start: dragStartRange.start + dxData, end: dragStartRange.end + dxData };
      drawNode(nd);
      hoverTip.style.display = "none";
      return;
    }
    const xData = nd.viewRange.start + ((e.offsetX - left) / plotW) * (nd.viewRange.end - nd.viewRange.start);
    if (e.offsetX < left || e.offsetX > left + plotW) { hoverTip.style.display = "none"; return; }
    let bestI = 0, bestD = Infinity;
    for (let i = 0; i < nd.grid.length; i++) { const d = Math.abs(nd.grid[i] - xData); if (d < bestD) { bestD = d; bestI = i; } }
    hoverTip.style.display = "block";
    hoverTip.style.left = (e.offsetX + 10) + "px";
    hoverTip.style.top = (e.offsetY - 10) + "px";
    hoverTip.textContent = `x = ${nd.grid[bestI].toFixed(3)}, density = ${nd.densityY[bestI].toFixed(4)}`;
  });
  canvas.addEventListener("mouseleave", () => { hoverTip.style.display = "none"; });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const { left, plotW } = canvasMetrics();
    const anchor = nd.viewRange.start + ((e.offsetX - left) / plotW) * (nd.viewRange.end - nd.viewRange.start);
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    const fullSpan = domain.end - domain.start;
    let newSpan = clip((nd.viewRange.end - nd.viewRange.start) * factor, fullSpan * 0.02, fullSpan * 5);
    const frac = (anchor - nd.viewRange.start) / (nd.viewRange.end - nd.viewRange.start);
    nd.viewRange = { start: anchor - frac * newSpan, end: anchor + (1 - frac) * newSpan };
    drawNode(nd);
  }, { passive: false });
  function canvasMetrics() {
    const left = 46, right = 10, top = 26, bottom = 34;
    return { left, top, plotW: canvas.width - left - right, plotH: canvas.height - top - bottom };
  }
  nd._canvasMetrics = canvasMetrics;

  // ── Sliders / selects ─────────────────────────────────────────────────
  const alphaEnd = isSurprisal ? 50 : 1000;
  const muRange = isSurprisal ? [0, 20] : [-10, 10];
  const sigmaRange = isSurprisal ? [0.1, 10] : [0.1, 20];

  const alphaSlider = createSlider({ min: 0, max: alphaEnd, value: PRIOR_ALPHA_DEFAULT, step: 1, title: "Prior strength α" });
  const muSlider = createSlider({ min: muRange[0], max: muRange[1], value: PRIOR_MU_DEFAULT, step: 0.1, title: "Prior mean μ" });
  const sigmaSlider = createSlider({ min: sigmaRange[0], max: sigmaRange[1], value: PRIOR_SIGMA_DEFAULT, step: 0.1, title: "Prior std dev σ" });
  const bwSlider = createSlider({ min: 0.1, max: 3.0, value: BANDWIDTH_DEFAULT, step: 0.05, title: "Bandwidth / smoothing factor" });
  const binWidthSlider = createSlider({ min: 0.01, max: 5.0, value: BIN_WIDTH_DEFAULT, step: 0.01, title: "Bin width Δx" });
  const nCompSlider = createSlider({ min: 1, max: GMM_COMPONENTS_MAX, value: GMM_COMPONENTS_DEFAULT, step: 1, title: "GMM components" });
  const methodSelect = createSelect({ options: DENSITY_METHODS, value: "kde", width: 140, title: "Fit method" });
  const yScaleSelect = createSelect({ options: [["fixed", "Y: fixed 0–1"], ["adaptive", "Y: adaptive"]], value: "adaptive", width: 150 });
  const gangCheckbox = createCheckbox("Control all descendants' parameters");
  const deriveBtn = el("button", { className: "primary", type: "button", text: "View derived distribution" });
  const klDiv = el("div", { className: "kl-display" });

  Object.assign(nd, { alphaSlider, muSlider, sigmaSlider, bwSlider, binWidthSlider, nCompSlider, methodSelect, yScaleSelect, gangCheckbox, deriveBtn, klDiv });

  const paramRow = el("div", { className: "row" });
  function renderParamRow() {
    const children = [alphaSlider.wrapper, muSlider.wrapper, sigmaSlider.wrapper];
    if (["kde", "adaptive_kde", "bspline"].includes(methodSelect.value)) children.push(bwSlider.wrapper);
    if (methodSelect.value === "gmm") children.push(nCompSlider.wrapper);
    children.push(methodSelect.wrapper);
    paramRow.replaceChildren(...children);
  }
  renderParamRow();
  nd.renderParamRow = renderParamRow;

  const binWidthPanel = el("div", { className: "bin-width-panel" }, [binWidthSlider.wrapper]);
  const figureRow = el("div", { className: "row node-figure-row" }, [canvasWrap, binWidthPanel]);
  const deriveRow = el("div", { className: "row" }, [deriveBtn, gangCheckbox.wrapper, klDiv]);
  const panel = el("div", { className: "node-panel" }, [paramRow, figureRow, el("div", { className: "row" }, [yScaleSelect.wrapper]), deriveRow]);
  nd.layoutEl = panel;

  // ── Callbacks ───────────────────────────────────────────────────────────
  function onParamChange() {
    if (suspendRecompute) return;
    if (nd.propagates) propagateParamsDown(nd);
    recomputeFrom(nd);
    refreshTraceDisplay();
  }
  function onMethodChange() {
    if (suspendRecompute) return;
    renderParamRow();
    if (nd.propagates) propagateParamsDown(nd);
    recomputeFrom(nd);
    refreshTraceDisplay();
  }
  [alphaSlider, muSlider, sigmaSlider, bwSlider, binWidthSlider, nCompSlider].forEach(s => {
    s.input.addEventListener("change", busyChange(onParamChange));
  });
  methodSelect.select.addEventListener("change", busyChange(onMethodChange));
  yScaleSelect.select.addEventListener("change", () => { nd.yAdaptive = (yScaleSelect.value === "adaptive"); recomputeFrom(nd); });
  gangCheckbox.input.addEventListener("change", () => {
    nd.propagates = gangCheckbox.checked;
    if (nd.propagates) { propagateParamsDown(nd); recomputeFrom(nd); refreshTraceDisplay(); }
  });
  deriveBtn.addEventListener("click", busy(() => createChildNode(nd)));

  return nd;
}

function drawNode(nd) {
  const ctx = nd.ctx, canvas = nd.canvas;
  const { left, top, plotW, plotH } = nd._canvasMetrics();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0a0f1e"; ctx.fillRect(0, 0, canvas.width, canvas.height);

  const { start: vStart, end: vEnd } = nd.viewRange;
  const xToPx = (x) => left + ((x - vStart) / (vEnd - vStart)) * plotW;
  const yToPx = (y) => top + plotH - (y / nd.yRangeEnd) * plotH;

  ctx.save();
  ctx.beginPath(); ctx.rect(left, top, plotW, plotH); ctx.clip();

  // Density curve + area fill
  ctx.beginPath();
  for (let i = 0; i < nd.grid.length; i++) {
    const px = xToPx(nd.grid[i]), py = yToPx(nd.densityY[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.lineTo(xToPx(nd.grid[nd.grid.length - 1]), yToPx(0));
  ctx.lineTo(xToPx(nd.grid[0]), yToPx(0));
  ctx.closePath();
  ctx.fillStyle = "rgba(72,120,207,0.4)"; ctx.fill();

  ctx.beginPath();
  for (let i = 0; i < nd.grid.length; i++) {
    const px = xToPx(nd.grid[i]), py = yToPx(nd.densityY[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = "#2255AA"; ctx.lineWidth = 2; ctx.stroke();

  // Rug
  if (rugVisible) {
    const rugH = nd.yRangeEnd * 0.03;
    ctx.strokeStyle = `rgba(68,68,68,${rugAlpha})`; ctx.lineWidth = 1;
    for (let i = 0; i < nd.events.length; i++) {
      const px = xToPx(nd.events[i]);
      ctx.beginPath(); ctx.moveTo(px, yToPx(0)); ctx.lineTo(px, yToPx(rugH)); ctx.stroke();
    }
  }

  // Trace markers
  for (const m of nd.traceMarkers) {
    const px = xToPx(m.x);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = m.color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, top + plotH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(px, yToPx(m.y), 5, 0, 2 * Math.PI);
    ctx.fillStyle = m.color; ctx.fill();
    ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.stroke();
  }
  ctx.restore();

  // Axes
  ctx.strokeStyle = "#444"; ctx.lineWidth = 1;
  ctx.strokeRect(left, top, plotW, plotH);
  ctx.fillStyle = "#9aa0c0"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  const nTicks = 6;
  for (let i = 0; i <= nTicks; i++) {
    const x = vStart + (i / nTicks) * (vEnd - vStart);
    const px = xToPx(x);
    ctx.fillText(x.toFixed(1), px, top + plotH + 14);
  }
  ctx.textAlign = "right";
  const nYTicks = 4;
  for (let i = 0; i <= nYTicks; i++) {
    const y = (i / nYTicks) * nd.yRangeEnd;
    ctx.fillText(y.toFixed(2), left - 6, yToPx(y) + 3);
  }
  ctx.textAlign = "center";
  ctx.fillText(nd.isSurprisal ? "Surprisal (bits)" : "Value", left + plotW / 2, canvas.height - 4);
  ctx.save();
  ctx.translate(12, top + plotH / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("Density", 0, 0);
  ctx.restore();

  // Title
  ctx.textAlign = "left";
  ctx.fillStyle = "#fff"; ctx.font = "bold 12px sans-serif";
  ctx.fillText(nd.titleText, left, 16);
}

// ───────────────────────────── Core recomputation ───────────────────────────

function propagateParamsDown(nd) {
  const child = nd.child;
  if (child === null) return;
  suspendRecompute = true;
  try {
    child.alphaSlider.value = nd.alphaSlider.value;
    child.muSlider.value = nd.muSlider.value;
    child.sigmaSlider.value = nd.sigmaSlider.value;
    child.bwSlider.value = nd.bwSlider.value;
    child.binWidthSlider.value = nd.binWidthSlider.value;
    child.nCompSlider.value = nd.nCompSlider.value;
    child.methodSelect.value = nd.methodSelect.value;
  } finally {
    suspendRecompute = false;
  }
  child.renderParamRow();
  propagateParamsDown(child);
}

function recomputeFrom(nd) {
  if (nd === null) return;
  const alpha = nd.alphaSlider.value, mu = nd.muSlider.value, sigma = nd.sigmaSlider.value;
  const bw = nd.bwSlider.value;
  const densityFn = makeDensityFn(nd.events, alpha, mu, sigma, nd.methodSelect.value, bw, nd.nCompSlider.value);
  nd.currentDensity = densityFn;
  const y = densityFn(nd.grid);
  nd.densityY = y;
  if (nd.yAdaptive) {
    let mx = 0; for (let i = 0; i < y.length; i++) if (y[i] > mx) mx = y[i];
    // A floor avoids a 0/0 -> NaN canvas coordinate when there are no events yet
    // (y is all exact zeros), which the Python original's Bokeh range tolerates
    // but a raw canvas divide does not.
    nd.yRangeEnd = y.length ? Math.max(mx * 1.05, 1e-6) : 1.0;
  } else {
    nd.yRangeEnd = 1.0;
  }
  const label = nd.depth === 0 ? "P1" : `P${nd.depth + 1}`;
  nd.titleText = `${label}  |  entropy = ${fmt(differentialEntropyBits(densityFn, nd.grid))} bits`;
  drawNode(nd);

  if (nd.child !== null) {
    nd.child.events = surprisalBits(nd.events, densityFn, nd.binWidthSlider.value);
    recomputeFrom(nd.child);
  }

  refreshKlDisplay(nd);
  if (nd.parent !== null) refreshKlDisplay(nd.parent);
}

function refreshKlDisplay(nd) {
  const lines = [];
  if (nd.depth >= 2 && nd.parent !== null && nd.parent.currentDensity !== null) {
    const klUp = klDivergenceBits(nd.currentDensity, nd.parent.currentDensity, SURP_GRID);
    const w1Up = wassersteinDistance(nd.currentDensity, nd.parent.currentDensity, SURP_GRID);
    lines.push(`↑ ${KL_LINK} ${fmt(klUp)} bits &nbsp; ${W1_LINK} ${fmt(w1Up)}`);
  }
  if (nd.depth >= 1 && nd.child !== null && nd.child.currentDensity !== null) {
    const klDown = klDivergenceBits(nd.currentDensity, nd.child.currentDensity, SURP_GRID);
    const w1Down = wassersteinDistance(nd.currentDensity, nd.child.currentDensity, SURP_GRID);
    lines.push(`↓ ${KL_LINK} ${fmt(klDown)} bits &nbsp; ${W1_LINK} ${fmt(w1Down)}`);
  }
  nd.klDiv.innerHTML = lines.join("<br>");
}

function refreshTraceDisplay() {
  for (const nd of allNodes) {
    const markers = [];
    for (let i = 0; i < traceIndices.length; i++) {
      const idx = traceIndices[i];
      if (idx >= nd.events.length || nd.currentDensity === null) continue;
      const color = TRACE_PALETTE[i % TRACE_PALETTE.length];
      const xVal = nd.events[idx];
      const yVal = nd.currentDensity([xVal])[0];
      markers.push({ x: xVal, y: yVal, color });
    }
    nd.traceMarkers = markers;
    drawNode(nd);
  }

  const lines = [];
  for (let i = 0; i < traceIndices.length; i++) {
    const idx = traceIndices[i];
    const color = TRACE_PALETTE[i % TRACE_PALETTE.length];
    const chain = [];
    for (const nd of allNodes) {
      if (idx >= nd.events.length) break;
      const label = nd.depth === 0 ? "P1" : `P${nd.depth + 1}`;
      const unit = nd.depth === 0 ? "" : " bits";
      chain.push(`${label}=${nd.events[idx].toFixed(3)}${unit}`);
    }
    if (chain.length) lines.push(`<span style="color:${color}">●</span> ` + chain.join(" &rarr; "));
  }
  document.getElementById("trace-summary").innerHTML = lines.join("<br>");
}

function rebuildGrid() {
  const grid = document.getElementById("node-grid");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = `repeat(${columnCount}, max-content)`;
  grid.replaceChildren(...allNodes.map(nd => nd.layoutEl));
}

function createChildNode(parentNode) {
  let childEvents, depth;
  if (parentNode !== null) {
    const alpha = parentNode.alphaSlider.value, mu = parentNode.muSlider.value, sigma = parentNode.sigmaSlider.value;
    const bw = parentNode.bwSlider.value;
    const densityFn = makeDensityFn(parentNode.events, alpha, mu, sigma, parentNode.methodSelect.value, bw, parentNode.nCompSlider.value);
    childEvents = surprisalBits(parentNode.events, densityFn, parentNode.binWidthSlider.value);
    depth = parentNode.depth + 1;
  } else {
    childEvents = Float64Array.from(rootEvents);
    depth = 0;
  }

  const newNode = makePNode(childEvents, depth);

  if (parentNode !== null) {
    parentNode.child = newNode;
    newNode.parent = parentNode;
    parentNode.deriveBtn.disabled = true;
    newNode.propagates = parentNode.propagates;
    newNode.gangCheckbox.checked = parentNode.gangCheckbox.checked;
    if (parentNode.propagates) propagateParamsDown(parentNode);
  } else {
    rootNode = newNode;
    initialDeriveBtn.disabled = true;
  }

  allNodes.push(newNode);
  rebuildGrid();
  recomputeFrom(newNode);
  refreshTraceDisplay();
}

// ───────────────────────────── Top-level controls ───────────────────────────

const nEventsInput = el("input", { type: "text", value: "100", style: { width: "70px" } });
nEventsInput.value = "100";
const familySelect = createSelect({ options: FAMILY_NAMES.map(n => [n, n]), value: FAMILY_NAMES[0], width: 150 });
const appendReplaceToggle = createToggleGroup(["Append", "Replace"], 0, () => {});
const distParamsRow = el("div", { className: "row", style: { marginBottom: "0" } });
const addEventsBtn = el("button", { className: "success", type: "button", text: "Add events" });
const addOneByOneBtn = el("button", { className: "success", type: "button", text: "Add events (one by one)" });
const clearEventsBtn = el("button", { className: "warning", type: "button", text: "Clear events", disabled: true });
clearEventsBtn.disabled = true;
const singleEventInput = el("input", { type: "text", placeholder: "Add event at value…", style: { width: "200px" } });
const singleEventCountInput = el("input", { type: "text", value: "1", style: { width: "50px" } });
const singleEventStatus = el("span", { className: "status-text" });
const traceCheckbox = createCheckbox("Trace new events");
const clearTracesBtn = el("button", { type: "button", className: "small", text: "Clear traces" });
const rugCheckbox = createCheckbox("Show event rug");
const rugAlphaSlider = createSlider({ min: 0, max: 1, value: 0.15, step: 0.01, title: "Rug opacity", width: 180 });
const colCountToggle = createToggleGroup(["1 col", "2 col", "3 col"], 0, (i) => { columnCount = i + 1; rebuildGrid(); });

const historyBackBtn = el("button", { type: "button", text: "◀", disabled: true, style: { width: "44px" } });
const historyFwdBtn = el("button", { type: "button", text: "▶", disabled: true, style: { width: "44px" } });
historyBackBtn.disabled = true; historyFwdBtn.disabled = true;
const historySlider = el("input", { type: "range", id: "history-slider", min: 0, max: 1, value: 0, step: 1, disabled: true });
historySlider.disabled = true;
const historyLabel = el("span", { id: "transport-label", text: "Step 0 of 0" });

const initialDeriveBtn = el("button", { className: "primary", type: "button", text: "View derived distribution" });

let currentParamSliders = [];

function makeParamSliders(familyName) {
  return FAMILIES[familyName].params.map(spec => createSlider({ min: spec.start, max: spec.end, value: spec.value, step: spec.step, title: spec.name, width: 180 }));
}
function getCurrentParams() {
  const out = {};
  for (const s of currentParamSliders) out[s.title] = s.value;
  return out;
}
function onParamSliderChange() {
  if (appendReplaceToggle.active === 1) doReplace();
}
function renderParamSliders(name) {
  currentParamSliders = makeParamSliders(name);
  distParamsRow.replaceChildren(...currentParamSliders.map(s => s.wrapper));
  currentParamSliders.forEach(s => s.input.addEventListener("input", onParamSliderChange));
}

function updateTransportState() {
  const n = allEvents.length;
  transportCbGuard = true;
  historySlider.max = Math.max(n, 1);
  historySlider.value = historyIndex;
  historySlider.disabled = n === 0;
  transportCbGuard = false;
  historyLabel.textContent = `Step ${historyIndex} of ${n}`;
  historyBackBtn.disabled = historyIndex === 0;
  historyFwdBtn.disabled = historyIndex === n;
  clearEventsBtn.disabled = allEvents.length === 0;
}

function onMakeDist() {
  if (rootNode === null) return;
  rootNode.events = Float64Array.from(rootEvents);
  recomputeFrom(rootNode);
  refreshTraceDisplay();
}

function setTraceNewIndices(n) {
  if (traceCheckbox.checked && n > 0) {
    const start = Math.max(0, rootEvents.length - n);
    const added = [];
    for (let i = start; i < rootEvents.length; i++) added.push(i);
    traceIndices = traceIndices.concat(added);
  }
}

function parseN() {
  const v = parseInt(nEventsInput.value, 10);
  if (!Number.isFinite(v) || v <= 0) { nEventsInput.value = "100"; return 100; }
  return v;
}

function onAddEvents() {
  if (appendReplaceToggle.active === 1) { doReplace(); return; }
  const n = parseN();
  const wasAtEnd = historyIndex === allEvents.length;
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = concatF64(allEvents, newEv);
  if (wasAtEnd) historyIndex = allEvents.length;
  rootEvents = allEvents.slice(0, historyIndex);
  setTraceNewIndices(n);
  updateTransportState();
  onMakeDist();
}

function doReplace() {
  const n = parseN();
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = Float64Array.from(newEv);
  historyIndex = allEvents.length;
  rootEvents = Float64Array.from(allEvents);
  setTraceNewIndices(n);
  updateTransportState();
  onMakeDist();
}

function onClearEvents() {
  rootEvents = new Float64Array(0);
  allEvents = new Float64Array(0);
  historyIndex = 0;
  onMakeDist();
  updateTransportState();
}

function onSingleEventInput() {
  const valStr = singleEventInput.value.trim();
  if (!valStr) return;
  const val = parseFloat(valStr);
  if (!Number.isFinite(val)) {
    singleEventStatus.textContent = `'${valStr}' is not a valid number.`;
    singleEventInput.value = "";
    return;
  }
  let count = parseInt(singleEventCountInput.value, 10);
  if (!Number.isFinite(count)) count = 0;
  const n = Math.max(count, 1);
  if (appendReplaceToggle.active === 1) {
    allEvents = new Float64Array(n).fill(val);
    historyIndex = n;
  } else {
    const wasAtEnd = historyIndex === allEvents.length;
    allEvents = concatF64(allEvents, new Float64Array(n).fill(val));
    if (wasAtEnd) historyIndex = allEvents.length;
  }
  rootEvents = allEvents.slice(0, historyIndex);
  singleEventStatus.textContent = `Added ${n} event${n > 1 ? "s" : ""} at ${val}.`;
  singleEventInput.value = "";
  setTraceNewIndices(n);
  updateTransportState();
  onMakeDist();
}

function applyHistoryIndex(idx) {
  historyIndex = clip(idx, 0, allEvents.length);
  rootEvents = allEvents.slice(0, historyIndex);
  updateTransportState();
  onMakeDist();
}

function onClearTraces() {
  traceIndices = [];
  refreshTraceDisplay();
}

function onAddEventsOneByOne() {
  if (stepHandle !== null) {
    clearTimeout(stepHandle);
    stepHandle = null;
    addOneByOneBtn.textContent = "Add events (one by one)";
    return;
  }
  const n = parseN();
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = concatF64(allEvents, newEv);
  const targetIndex = allEvents.length;
  addOneByOneBtn.textContent = "Stop";

  function step() {
    if (stepHandle === null) { addOneByOneBtn.textContent = "Add events (one by one)"; return; }
    applyHistoryIndex(historyIndex + 1);
    setTraceNewIndices(1);
    refreshTraceDisplay();
    if (historyIndex >= targetIndex) {
      stepHandle = null;
      addOneByOneBtn.textContent = "Add events (one by one)";
    } else {
      stepHandle = setTimeout(step, 0);
    }
  }
  stepHandle = setTimeout(step, 0);
}

function onRugCheckboxChange() {
  rugVisible = rugCheckbox.checked;
  for (const nd of allNodes) drawNode(nd);
}
function onRugAlphaChange() {
  rugAlpha = rugAlphaSlider.value;
  for (const nd of allNodes) drawNode(nd);
}

addEventsBtn.addEventListener("click", busy(onAddEvents));
clearEventsBtn.addEventListener("click", busy(onClearEvents));
singleEventInput.addEventListener("change", busyChange(onSingleEventInput));
singleEventInput.addEventListener("keydown", (e) => { if (e.key === "Enter") singleEventInput.blur(); });
initialDeriveBtn.addEventListener("click", busy(() => createChildNode(null)));
familySelect.select.addEventListener("change", () => { renderParamSliders(familySelect.value); if (appendReplaceToggle.active === 1) doReplace(); });
addOneByOneBtn.addEventListener("click", busy(onAddEventsOneByOne));
clearTracesBtn.addEventListener("click", onClearTraces);
rugCheckbox.input.addEventListener("change", onRugCheckboxChange);
rugAlphaSlider.input.addEventListener("input", onRugAlphaChange);
historySlider.addEventListener("input", busyChange(() => { if (transportCbGuard) return; applyHistoryIndex(parseInt(historySlider.value, 10)); }));
historyBackBtn.addEventListener("click", busy(() => applyHistoryIndex(historyIndex - 1)));
historyFwdBtn.addEventListener("click", busy(() => applyHistoryIndex(historyIndex + 1)));

renderParamSliders(FAMILY_NAMES[0]);

// ───────────────────────────── Layout ───────────────────────────────────────

document.getElementById("top-controls").append(
  el("div", { className: "row" }, [
    familySelect.wrapper, distParamsRow, appendReplaceToggle.wrapper,
    addEventsBtn, addOneByOneBtn,
    el("span", { html: "<b>n =</b>" }), nEventsInput,
    clearEventsBtn, traceCheckbox.wrapper, clearTracesBtn,
  ]),
  el("div", { className: "row" }, [
    singleEventInput,
    el("span", { html: "<b>count:</b>" }), singleEventCountInput, singleEventStatus,
    el("span", { html: "<b>Layout:</b>" }), colCountToggle.wrapper,
    rugCheckbox.wrapper, rugAlphaSlider.wrapper,
  ]),
);

document.getElementById("transport-panel").append(
  el("div", { className: "row", style: { marginBottom: "0" } }, [
    historyBackBtn, historySlider, historyFwdBtn, historyLabel,
  ]),
);

document.getElementById("initial-derive-row").append(initialDeriveBtn);

updateTransportState();
