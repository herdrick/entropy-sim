// Surprisal Fixed Point — web port of find_fixed_point.py
import { createSimplex3DPanel, updateSimplex3DPanel } from './viz-simplex3d.js';
import { createRadialPanel, updateRadialPanel } from './viz-radial.js';
import { createScatterMatrixPanel, updateScatterMatrixPanel } from './viz-scatter-matrix.js';
import { createParallelCoordsPanel, updateParallelCoordsPanel } from './viz-parallel-coords.js';

// ───────────────────────── Constants ─────────────────────────
const X_MIN = -10, X_MAX = 300;
const PRIOR_ALPHA_DEFAULT = 0, PRIOR_MU_DEFAULT = 0, PRIOR_SIGMA_DEFAULT = 5;
const MAX_ITER = 1000;
const BAR_COLOR = '#4878CF';

// ───────────────────────── Math helpers (iterated_surprisal_distributions.py) ─────────────────────────
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function normalCdf(x, mu, sigma) {
  if (x === -Infinity) return 0;
  if (x === Infinity) return 1;
  return 0.5 * (1 + erf((x - mu) / (sigma * Math.SQRT2)));
}

function searchsortedLeft(arr, v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function binCounts(edges, events) {
  const nBins = edges.length - 1;
  const interior = edges.slice(1, -1);
  const counts = new Array(nBins).fill(0);
  for (let i = 0; i < events.length; i++) {
    const idx = searchsortedLeft(interior, events[i]);
    counts[Math.min(idx, nBins - 1)]++;
  }
  return counts;
}

function gaussianPriorMass(edges, mu, sigma) {
  const cdf = edges.map(e => normalCdf(e, mu, sigma));
  const out = new Array(edges.length - 1);
  for (let i = 0; i < out.length; i++) out[i] = cdf[i + 1] - cdf[i];
  return out;
}

function computeProbabilities(edges, events, alpha, mu, sigma) {
  const counts = binCounts(edges, events);
  const prior = gaussianPriorMass(edges, mu, sigma);
  const smoothed = counts.map((c, i) => c + alpha * prior[i]);
  const total = smoothed.reduce((a, b) => a + b, 0);
  if (total > 0) return smoothed.map(s => s / total);
  return new Array(counts.length).fill(1 / counts.length);
}

function entropyBits(probs) {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log2(p);
  return h;
}

function clip(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function fmtG(x, sig) {
  if (x === Infinity) return '+∞';
  if (x === -Infinity) return '-∞';
  if (x === 0) return '0';
  return Number(x.toPrecision(sig)).toString();
}

// ───────────────────────── Fixed-point iteration ─────────────────────────
function computeFixedPointIterations(events, initEdges, surpEdges, alpha, mu, sigma, tol) {
  if (events.length === 0) return [null, null];
  const initInterior = initEdges.slice(1, -1);
  const surpInterior = surpEdges.slice(1, -1);
  const initProbs = computeProbabilities(initEdges, events, alpha, mu, sigma);
  let currentEvents = events.map(e => {
    const idx = clip(searchsortedLeft(initInterior, e), 0, initProbs.length - 1);
    return -Math.log2(initProbs[idx]);
  });
  let probs = computeProbabilities(surpEdges, currentEvents, alpha, mu, sigma);
  for (let i = 0; i < MAX_ITER; i++) {
    const newEvents = currentEvents.map(e => {
      const idx = clip(searchsortedLeft(surpInterior, e), 0, probs.length - 1);
      return -Math.log2(probs[idx]);
    });
    const newProbs = computeProbabilities(surpEdges, newEvents, alpha, mu, sigma);
    let maxDiff = 0;
    for (let j = 0; j < newProbs.length; j++) maxDiff = Math.max(maxDiff, Math.abs(newProbs[j] - probs[j]));
    if (maxDiff < tol) return [i + 1, newProbs];
    currentEvents = newEvents;
    probs = newProbs;
  }
  return [null, null];
}

// ───────────────────────── Random samplers (events.py port) ─────────────────────────
function sampleNormal01() {
  let u1 = Math.random();
  while (u1 === 0) u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
function sampleGamma(shape) {
  if (shape < 1) {
    const u = Math.random();
    return sampleGamma(shape + 1) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = sampleNormal01(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
}
function sampleBeta(a, b) {
  const x = sampleGamma(a), y = sampleGamma(b);
  return x / (x + y);
}

const FAMILIES = {
  'Uniform': {
    params: [
      { name: 'low', start: -10, end: 10, value: 0.0, step: 0.1 },
      { name: 'high', start: -10, end: 10, value: 1.0, step: 0.1 },
    ],
    sample: (p) => {
      const scale = Math.max(p['high'] - p['low'], 1e-6);
      return p['low'] + Math.random() * scale;
    },
  },
  'Normal': {
    params: [
      { name: 'μ', start: -10, end: 10, value: 0.0, step: 0.1 },
      { name: 'σ', start: 0.1, end: 10, value: 1.0, step: 0.1 },
    ],
    sample: (p) => p['μ'] + p['σ'] * sampleNormal01(),
  },
  'Beta': {
    params: [
      { name: 'α', start: 0.1, end: 10, value: 2.0, step: 0.1 },
      { name: 'β', start: 0.1, end: 10, value: 5.0, step: 0.1 },
    ],
    sample: (p) => sampleBeta(p['α'], p['β']),
  },
  'Exponential': {
    params: [
      { name: 'λ', start: 0.1, end: 10, value: 1.0, step: 0.1 },
    ],
    sample: (p) => -Math.log(1 - Math.random()) / p['λ'],
  },
};
const FAMILY_NAMES = Object.keys(FAMILIES);

function getEvents(n, family, params) {
  const fn = FAMILIES[family].sample;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = fn(params);
  return out;
}

// ───────────────────────── BinTracker (bin_selection.py port) ─────────────────────────
class BinTracker {
  constructor() { this._counts = new Map(); this._nRuns = 0; }
  record(probs) {
    this._nRuns++;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > 1e-12) this._counts.set(i, (this._counts.get(i) || 0) + 1);
    }
  }
  reset() { this._counts = new Map(); this._nRuns = 0; }
  getActiveBins(minFreq = 0.1) {
    if (this._nRuns === 0) return [];
    const out = [];
    for (const [idx, count] of this._counts) if (count / this._nRuns >= minFreq) out.push(idx);
    return out.sort((a, b) => a - b);
  }
  getBinLabels(indices) { return indices.map(i => `p${i + 1}`); }
}
const tracker = new BinTracker();

// ───────────────────────── Canvas histogram node panel ─────────────────────────
const CANVAS_W = 640, CANVAS_H = 320;
const PAD_L = 48, PAD_R = 12, PAD_T = 10, PAD_B = 34;

function domainToPixelX(node, x) {
  return PAD_L + (x - node.xViewStart) / (node.xViewEnd - node.xViewStart) * (CANVAS_W - PAD_L - PAD_R);
}
function pixelToDomainX(node, px) {
  return node.xViewStart + (px - PAD_L) / (CANVAS_W - PAD_L - PAD_R) * (node.xViewEnd - node.xViewStart);
}
function domainToPixelY(node, y, yEnd) {
  return CANVAS_H - PAD_B - (y / yEnd) * (CANVAS_H - PAD_T - PAD_B);
}

function computeNodeData(node, edges, probs, counts) {
  node.currentEdges = edges;
  node.currentProbs = probs;
  node.currentCounts = counts || new Array(probs.length).fill(0);
  node.edgeLeftStrs = edges.slice(0, -1).map(l => l === -Infinity ? '-∞' : fmtG(l, 4));
  node.edgeRightStrs = edges.slice(1).map(r => r === Infinity ? '+∞' : fmtG(r, 4));
  const total = node.currentCounts.reduce((a, b) => a + b, 0);
  node.rawProb = total > 0 ? node.currentCounts.map(c => c / total) : new Array(probs.length).fill(0);
}

function drawNode(node) {
  const ctx = node.ctx;
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  const edges = node.currentEdges, probs = node.currentProbs;
  if (!edges || !probs) return;
  const n = probs.length;
  const xStart = node.xViewStart, xEnd = node.xViewEnd;
  const lefts = edges.slice(0, -1), rights = edges.slice(1);
  const dl = new Array(n), dr = new Array(n), top = new Array(n), density = new Array(n);
  for (let i = 0; i < n; i++) {
    dl[i] = lefts[i] === -Infinity ? xStart : lefts[i];
    dr[i] = rights[i] === Infinity ? xEnd : rights[i];
    const width = dr[i] - dl[i];
    density[i] = width > 0 ? probs[i] / width : 0;
    top[i] = node.yModeActive === 1 ? density[i] : probs[i];
  }
  let yEnd;
  if (node.yRangeAdaptive) {
    const maxTop = n > 0 ? Math.max(...top) : 1.0;
    yEnd = maxTop > 0 ? maxTop * 1.05 : 1.0;
  } else {
    yEnd = 1.0;
  }
  node.lastBars = [];
  for (let i = 0; i < n; i++) {
    if (dr[i] <= xStart || dl[i] >= xEnd) continue;
    const clL = Math.max(dl[i], xStart), clR = Math.min(dr[i], xEnd);
    const pxL = domainToPixelX(node, clL), pxR = domainToPixelX(node, clR);
    const pxTop = domainToPixelY(node, top[i], yEnd);
    const pxBase = domainToPixelY(node, 0, yEnd);
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = BAR_COLOR;
    ctx.fillRect(pxL, pxTop, Math.max(pxR - pxL, 0.5), pxBase - pxTop);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.strokeRect(pxL, pxTop, pxR - pxL, pxBase - pxTop);
    node.lastBars.push({ dl: dl[i], dr: dr[i], index: i });
  }
  // interior edge lines
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  for (const e of node.interiorEdges) {
    if (e >= xStart && e <= xEnd) {
      const px = domainToPixelX(node, e);
      ctx.beginPath();
      ctx.moveTo(px, PAD_T);
      ctx.lineTo(px, CANVAS_H - PAD_B);
      ctx.stroke();
    }
  }
  // axes
  ctx.strokeStyle = '#555';
  ctx.beginPath();
  ctx.moveTo(PAD_L, CANVAS_H - PAD_B);
  ctx.lineTo(CANVAS_W - PAD_R, CANVAS_H - PAD_B);
  ctx.moveTo(PAD_L, PAD_T);
  ctx.lineTo(PAD_L, CANVAS_H - PAD_B);
  ctx.stroke();
  ctx.fillStyle = '#9aa5c0';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  const nTicks = 5;
  for (let t = 0; t <= nTicks; t++) {
    const xv = xStart + (xEnd - xStart) * t / nTicks;
    const px = domainToPixelX(node, xv);
    ctx.fillText(fmtG(xv, 3), px, CANVAS_H - PAD_B + 14);
  }
  ctx.textAlign = 'right';
  ctx.fillText(fmtG(yEnd, 3), PAD_L - 4, PAD_T + 8);
  ctx.fillText('0', PAD_L - 4, CANVAS_H - PAD_B);
  ctx.textAlign = 'center';
  ctx.fillText(node.xAxisLabel, (PAD_L + CANVAS_W - PAD_R) / 2, CANVAS_H - 4);
  node._yEnd = yEnd;
}

function showTooltip(node, evt) {
  const rect = node.canvas.getBoundingClientRect();
  const mx = evt.clientX - rect.left, my = evt.clientY - rect.top;
  const domainX = pixelToDomainX(node, mx);
  let hit = null;
  for (const b of node.lastBars) {
    if (domainX >= b.dl && domainX <= b.dr) { hit = b; break; }
  }
  if (!hit) { node.tooltip.style.display = 'none'; return; }
  const i = hit.index;
  node.tooltip.innerHTML =
    `Bin: ${node.edgeLeftStrs[i]} to ${node.edgeRightStrs[i]}<br>` +
    `Count: ${node.currentCounts[i]}<br>` +
    `Probability before prior: ${node.rawProb[i].toFixed(4)}<br>` +
    `Probability: ${node.currentProbs[i].toFixed(4)}<br>` +
    `Density: ${(node.currentProbs[i] / Math.max(hit.dr - hit.dl, 1e-12)).toFixed(4)}`;
  node.tooltip.style.left = (mx + 12) + 'px';
  node.tooltip.style.top = (my + 12) + 'px';
  node.tooltip.style.display = 'block';
}

function syncEdges(node) {
  const edgeSet = new Set([node.dom.splitSlider.valueAsNumber, ...node.singleEdges]);
  const count = parseInt(node.dom.countSlider.value, 10);
  const left = node.dom.leftSlider.valueAsNumber;
  const right = node.dom.rightSlider.valueAsNumber;
  if (count > 0 && right > left) {
    const step = (right - left) / (count + 1);
    for (let i = 0; i < count; i++) edgeSet.add(left + step * (i + 1));
  }
  node.interiorEdges = [...edgeSet].sort((a, b) => a - b);
  if (node.onEdgesChanged) node.onEdgesChanged();
}

function createHistogramNode(container, opts) {
  const { xAxisLabel, viewStart, viewEnd, splitDefault, leftDefault, rightDefault, hasPrior } = opts;
  const ymodeName = 'ymode-' + Math.random().toString(36).slice(2);
  container.innerHTML = `
    <div class="node-title"></div>
    ${hasPrior ? `
    <div class="row">
      <div class="slider-field"><label class="field-label">Prior strength α <span class="slider-value" data-alpha-val></span></label><input type="range" data-alpha min="0" max="5" step="0.1" value="${PRIOR_ALPHA_DEFAULT}"></div>
      <div class="slider-field"><label class="field-label">Prior mean μ <span class="slider-value" data-mu-val></span></label><input type="range" data-mu min="-10" max="10" step="0.1" value="${PRIOR_MU_DEFAULT}"></div>
      <div class="slider-field"><label class="field-label">Prior std dev σ <span class="slider-value" data-sigma-val></span></label><input type="range" data-sigma min="0.1" max="20" step="0.1" value="${PRIOR_SIGMA_DEFAULT}"></div>
    </div>` : ''}
    <div class="node-panel">
      <div class="node-canvas-wrap">
        <canvas class="hist-canvas" width="${CANVAS_W}" height="${CANVAS_H}"></canvas>
        <div class="hist-tooltip" data-tooltip></div>
      </div>
      <div class="edge-panel">
        <div class="row">
          <input type="text" data-single-edge-input placeholder="Value…" style="width:100px">
          <button data-add-single-edge-btn>Add</button>
        </div>
        <div class="slider-field">
          <label class="field-label">Split point <span class="slider-value" data-split-val></span></label>
          <input type="range" data-split min="${X_MIN}" max="${X_MAX}" step="0.1" value="${splitDefault}">
        </div>
        <button data-freeze-edge-btn style="align-self:flex-start">Freeze edge</button>
        <div class="slider-field">
          <label class="field-label">Evenly spaced: left <span class="slider-value" data-left-val></span></label>
          <input type="range" data-left min="${X_MIN}" max="${X_MAX}" step="0.1" value="${leftDefault}">
        </div>
        <div class="slider-field">
          <label class="field-label">Evenly spaced: right <span class="slider-value" data-right-val></span></label>
          <input type="range" data-right min="${X_MIN}" max="${X_MAX}" step="0.1" value="${rightDefault}">
        </div>
        <div class="slider-field">
          <label class="field-label">Evenly spaced: edge count <span class="slider-value" data-count-val></span></label>
          <input type="range" data-count min="0" max="5000" step="1" value="0">
        </div>
      </div>
    </div>
    <div class="row">
      <div class="radio-group">
        <label><input type="radio" name="${ymodeName}" value="0" checked> Probability</label>
        <label><input type="radio" name="${ymodeName}" value="1"> Probability density</label>
      </div>
      <select data-y-scale>
        <option value="adaptive" selected>Y: adaptive</option>
        <option value="fixed">Y: fixed 0-1</option>
      </select>
    </div>
  `;

  const node = {
    interiorEdges: [splitDefault],
    singleEdges: [],
    events: [],
    currentEdges: null, currentProbs: null, currentCounts: null,
    edgeLeftStrs: [], edgeRightStrs: [], rawProb: [],
    yModeActive: 0,
    yRangeAdaptive: true,
    xViewStart: viewStart, xViewEnd: viewEnd,
    initialView: [viewStart, viewEnd],
    xAxisLabel,
    lastBars: [],
    onEdgesChanged: null,
    titleEl: container.querySelector('.node-title'),
    canvas: container.querySelector('canvas.hist-canvas'),
    tooltip: container.querySelector('[data-tooltip]'),
    dom: {
      splitSlider: container.querySelector('[data-split]'),
      freezeBtn: container.querySelector('[data-freeze-edge-btn]'),
      singleEdgeInput: container.querySelector('[data-single-edge-input]'),
      addSingleEdgeBtn: container.querySelector('[data-add-single-edge-btn]'),
      leftSlider: container.querySelector('[data-left]'),
      rightSlider: container.querySelector('[data-right]'),
      countSlider: container.querySelector('[data-count]'),
      yScaleSelect: container.querySelector('[data-y-scale]'),
      priorAlphaSlider: container.querySelector('[data-alpha]'),
      priorMuSlider: container.querySelector('[data-mu]'),
      priorSigmaSlider: container.querySelector('[data-sigma]'),
    },
  };
  node.ctx = node.canvas.getContext('2d');

  const splitVal = container.querySelector('[data-split-val]');
  const leftVal = container.querySelector('[data-left-val]');
  const rightVal = container.querySelector('[data-right-val]');
  const countVal = container.querySelector('[data-count-val]');
  splitVal.textContent = splitDefault;
  leftVal.textContent = leftDefault;
  rightVal.textContent = rightDefault;
  countVal.textContent = '0';

  node.dom.splitSlider.addEventListener('input', () => { splitVal.textContent = node.dom.splitSlider.value; syncEdges(node); });
  node.dom.leftSlider.addEventListener('input', () => { leftVal.textContent = node.dom.leftSlider.value; syncEdges(node); });
  node.dom.rightSlider.addEventListener('input', () => { rightVal.textContent = node.dom.rightSlider.value; syncEdges(node); });
  node.dom.countSlider.addEventListener('input', () => { countVal.textContent = node.dom.countSlider.value; syncEdges(node); });

  node.dom.freezeBtn.addEventListener('click', () => {
    const val = node.dom.splitSlider.valueAsNumber;
    if (!node.singleEdges.includes(val)) node.singleEdges.push(val);
    syncEdges(node);
  });
  node.dom.addSingleEdgeBtn.addEventListener('click', () => {
    const raw = node.dom.singleEdgeInput.value.trim();
    if (!raw) return;
    const val = parseFloat(raw);
    if (isNaN(val)) return;
    if (!node.singleEdges.includes(val)) node.singleEdges.push(val);
    node.dom.singleEdgeInput.value = '';
    syncEdges(node);
  });

  container.querySelectorAll(`input[name="${ymodeName}"]`).forEach(radio => {
    radio.addEventListener('change', () => {
      if (radio.checked) { node.yModeActive = parseInt(radio.value, 10); drawNode(node); }
    });
  });
  node.dom.yScaleSelect.addEventListener('change', () => {
    node.yRangeAdaptive = node.dom.yScaleSelect.value === 'adaptive';
    drawNode(node);
  });

  if (hasPrior) {
    const alphaVal = container.querySelector('[data-alpha-val]');
    const muVal = container.querySelector('[data-mu-val]');
    const sigmaVal = container.querySelector('[data-sigma-val]');
    alphaVal.textContent = PRIOR_ALPHA_DEFAULT;
    muVal.textContent = PRIOR_MU_DEFAULT;
    sigmaVal.textContent = PRIOR_SIGMA_DEFAULT;
    node.dom.priorAlphaSlider.addEventListener('input', () => { alphaVal.textContent = node.dom.priorAlphaSlider.value; if (node.onEdgesChanged) node.onEdgesChanged(); });
    node.dom.priorMuSlider.addEventListener('input', () => { muVal.textContent = node.dom.priorMuSlider.value; if (node.onEdgesChanged) node.onEdgesChanged(); });
    node.dom.priorSigmaSlider.addEventListener('input', () => { sigmaVal.textContent = node.dom.priorSigmaSlider.value; if (node.onEdgesChanged) node.onEdgesChanged(); });
  }

  // pan / zoom
  let dragging = false, dragStartX = 0, dragStartView = null;
  node.canvas.addEventListener('mousedown', (e) => {
    dragging = true; dragStartX = e.clientX; dragStartView = [node.xViewStart, node.xViewEnd];
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - dragStartX;
    const domainDx = dx / (CANVAS_W - PAD_L - PAD_R) * (dragStartView[1] - dragStartView[0]);
    node.xViewStart = dragStartView[0] - domainDx;
    node.xViewEnd = dragStartView[1] - domainDx;
    drawNode(node);
  });
  window.addEventListener('mouseup', () => { dragging = false; });
  node.canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = node.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const domainX = pixelToDomainX(node, mouseX);
    const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
    const newWidth = (node.xViewEnd - node.xViewStart) * factor;
    const ratio = (domainX - node.xViewStart) / (node.xViewEnd - node.xViewStart);
    node.xViewStart = domainX - ratio * newWidth;
    node.xViewEnd = node.xViewStart + newWidth;
    drawNode(node);
  }, { passive: false });
  node.canvas.addEventListener('dblclick', () => {
    node.xViewStart = node.initialView[0];
    node.xViewEnd = node.initialView[1];
    drawNode(node);
  });
  node.canvas.addEventListener('mousemove', (e) => showTooltip(node, e));
  node.canvas.addEventListener('mouseleave', () => { node.tooltip.style.display = 'none'; });

  return node;
}

// ───────────────────────── App state ─────────────────────────
let allEvents = [];
let historyIndex = 0;
let rootEvents = [];
let sessionRecord = 0;
let sessionRecordRows = [];
let allSimplexFixedPoints = [];
const fpBarCharts = new Map();
let fpBarAlpha = 0.15;
let currentParamSliders = [];
const lockedBinsState = { locked: false, bins: [], labels: [] };
let transportGuard = false;
let animHandle = null;

const nodeP1 = createHistogramNode(document.getElementById('p1-panel'), {
  xAxisLabel: 'Value', viewStart: X_MIN, viewEnd: X_MAX,
  splitDefault: 0.0, leftDefault: -3.0, rightDefault: 3.0, hasPrior: true,
});
const nodeS = createHistogramNode(document.getElementById('surp-panel'), {
  xAxisLabel: 'Surprisal (bits)', viewStart: 0, viewEnd: 50,
  splitDefault: 5.0, leftDefault: 0.0, rightDefault: 20.0, hasPrior: false,
});
nodeP1.onEdgesChanged = () => appRecompute();
nodeS.onEdgesChanged = () => appRecompute();

const convergenceDiv = document.getElementById('convergence-div');
const familySelect = document.getElementById('family-select');
const distParamsRow = document.getElementById('dist-params-row');
const nEventsInput = document.getElementById('n-events-input');
const addEventsBtn = document.getElementById('add-events-btn');
const addEventsOneByOneBtn = document.getElementById('add-events-one-by-one-btn');
const clearEventsBtn = document.getElementById('clear-events-btn');
const singleEventInput = document.getElementById('single-event-input');
const singleEventCountInput = document.getElementById('single-event-count-input');
const singleEventStatus = document.getElementById('single-event-status');
const tolSlider = document.getElementById('tol-slider');
const tolValueSpan = document.getElementById('tol-value');
const historyBackBtn = document.getElementById('history-back-btn');
const historyFwdBtn = document.getElementById('history-fwd-btn');
const historySlider = document.getElementById('history-slider');
const historyLabel = document.getElementById('history-label');
const fpBarAlphaSlider = document.getElementById('fp-bar-alpha-slider');
const fpBarAlphaValueSpan = document.getElementById('fp-bar-alpha-value');
const fpBarchartWrapEl = document.getElementById('fp-barchart-wrap');
const clearSimplexBtn = document.getElementById('clear-simplex-btn');
const simplexStatsDiv = document.getElementById('simplex-stats-div');
const lockToggle = document.getElementById('lock-bins-toggle');
const minFreqSlider = document.getElementById('min-freq-slider');
const minFreqValueSpan = document.getElementById('min-freq-value');
const binLockStatusEl = document.getElementById('bin-lock-status');

function isReplaceActive() {
  return document.querySelector('input[name="append-replace"]:checked').value === 'replace';
}
function getCurrentParams() {
  const out = {};
  for (const s of currentParamSliders) out[s.title] = s.el.valueAsNumber;
  return out;
}
function makeParamSliders(familyName) {
  distParamsRow.innerHTML = '';
  currentParamSliders = FAMILIES[familyName].params.map(spec => {
    const field = document.createElement('div');
    field.className = 'slider-field';
    field.innerHTML = `<label class="field-label">${spec.name} <span class="slider-value"></span></label>
      <input type="range" min="${spec.start}" max="${spec.end}" step="${spec.step}" value="${spec.value}">`;
    const input = field.querySelector('input');
    const valSpan = field.querySelector('.slider-value');
    valSpan.textContent = spec.value;
    input.addEventListener('input', () => {
      valSpan.textContent = input.value;
      if (isReplaceActive()) doReplace();
    });
    distParamsRow.appendChild(field);
    return { el: input, title: spec.name };
  });
}

FAMILY_NAMES.forEach(name => {
  const opt = document.createElement('option');
  opt.value = name; opt.textContent = name;
  familySelect.appendChild(opt);
});
familySelect.value = FAMILY_NAMES[0];
makeParamSliders(FAMILY_NAMES[0]);
familySelect.addEventListener('change', () => {
  makeParamSliders(familySelect.value);
  if (isReplaceActive()) doReplace();
});

function updateTransportState() {
  const n = allEvents.length;
  transportGuard = true;
  historySlider.max = String(Math.max(n, 1));
  historySlider.value = String(historyIndex);
  historySlider.disabled = n === 0;
  transportGuard = false;
  historyLabel.textContent = `Step ${historyIndex} of ${n}`;
  historyBackBtn.disabled = historyIndex === 0;
  historyFwdBtn.disabled = historyIndex === n;
  clearEventsBtn.disabled = n === 0;
}

function onMakeDist() {
  nodeP1.events = rootEvents.slice();
  appRecompute();
}

function onAddEvents() {
  if (isReplaceActive()) { doReplace(); return; }
  let n = parseInt(nEventsInput.value, 10);
  if (!n || n <= 0) { n = 1000; nEventsInput.value = '1000'; }
  const wasAtEnd = historyIndex === allEvents.length;
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = allEvents.concat(newEv);
  if (wasAtEnd) historyIndex = allEvents.length;
  rootEvents = allEvents.slice(0, historyIndex);
  updateTransportState();
  onMakeDist();
}

function onClearEvents() {
  allEvents = []; rootEvents = []; historyIndex = 0;
  onMakeDist();
  updateTransportState();
}

function onSingleEventCommit() {
  const raw = singleEventInput.value.trim();
  if (!raw) return;
  const val = parseFloat(raw);
  if (isNaN(val)) {
    singleEventStatus.textContent = `'${raw}' is not a valid number.`;
    singleEventInput.value = '';
    return;
  }
  let count = parseInt(singleEventCountInput.value, 10);
  if (isNaN(count)) count = 0;
  const n = Math.max(count, 1);
  if (isReplaceActive()) {
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

function doReplace() {
  let n = parseInt(nEventsInput.value, 10);
  if (!n || n <= 0) n = 1000;
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = newEv.slice();
  historyIndex = allEvents.length;
  rootEvents = allEvents.slice();
  updateTransportState();
  onMakeDist();
}

function applyHistoryIndex(idx) {
  historyIndex = clip(idx, 0, allEvents.length);
  rootEvents = allEvents.slice(0, historyIndex);
  updateTransportState();
  onMakeDist();
}

function onAddEventsOneByOne() {
  if (animHandle !== null) {
    clearTimeout(animHandle);
    animHandle = null;
    addEventsOneByOneBtn.textContent = 'Add events (one by one)';
    return;
  }
  let n = parseInt(nEventsInput.value, 10);
  if (!n || n <= 0) { n = 1000; nEventsInput.value = '1000'; }
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = allEvents.concat(newEv);
  const targetIndex = allEvents.length;
  addEventsOneByOneBtn.textContent = 'Stop';

  function step() {
    if (animHandle === null) { addEventsOneByOneBtn.textContent = 'Add events (one by one)'; return; }
    applyHistoryIndex(historyIndex + 1);
    if (historyIndex >= targetIndex) {
      animHandle = null;
      addEventsOneByOneBtn.textContent = 'Add events (one by one)';
    } else {
      animHandle = setTimeout(step, 0);
    }
  }
  animHandle = setTimeout(step, 0);
}

addEventsBtn.addEventListener('click', onAddEvents);
addEventsOneByOneBtn.addEventListener('click', onAddEventsOneByOne);
clearEventsBtn.addEventListener('click', onClearEvents);
singleEventInput.addEventListener('change', onSingleEventCommit);
singleEventInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') onSingleEventCommit(); });
tolSlider.addEventListener('input', () => { tolValueSpan.textContent = tolSlider.value; appRecompute(); });

historySlider.addEventListener('input', () => {
  if (transportGuard) return;
  applyHistoryIndex(parseInt(historySlider.value, 10));
});
historyBackBtn.addEventListener('click', () => applyHistoryIndex(historyIndex - 1));
historyFwdBtn.addEventListener('click', () => applyHistoryIndex(historyIndex + 1));

// ───────────────────────── Fixed-point overlay bar charts ─────────────────────────
function edgeKey(edges) {
  return edges.map(e => e === Infinity ? 'inf' : e === -Infinity ? '-inf' : e.toFixed(6)).join(',');
}

function renderFpBarchart(entry) {
  entry.titleEl.textContent = entry.n === 1
    ? `Fixed-point overlays — 1 point, ${entry.nBins} bins (${entry.edgeDesc})`
    : `Fixed-point overlays — ${entry.n} points, ${entry.nBins} bins`;
  const ctx = entry.ctx, w = entry.canvas.width, h = entry.canvas.height;
  ctx.clearRect(0, 0, w, h);
  const padL = 40, padR = 10, padT = 10, padB = 24;
  const xStart = 0, xEnd = 20, yEnd = (entry.yMax * 1.1) || 1.0;
  const xPix = x => padL + (x - xStart) / (xEnd - xStart) * (w - padL - padR);
  const yPix = y => h - padB - (y / yEnd) * (h - padT - padB);
  ctx.fillStyle = `rgba(34,102,170,${fpBarAlpha})`;
  for (let i = 0; i < entry.xPoints.length; i++) {
    const px = xPix(entry.xPoints[i]), py = yPix(entry.yPoints[i]);
    ctx.beginPath(); ctx.arc(px, py, 3, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.strokeStyle = '#555';
  ctx.beginPath();
  ctx.moveTo(padL, h - padB); ctx.lineTo(w - padR, h - padB);
  ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB);
  ctx.stroke();
  ctx.fillStyle = '#9aa5c0';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Surprisal (bits)', w / 2, h - 4);
}

function addToFpBarchart(edges, probs) {
  const key = edgeKey(edges);
  const xStart = 0, xEnd = 20;
  const lefts = edges.slice(0, -1).map(e => e === -Infinity ? xStart : e);
  const rights = edges.slice(1).map(e => e === Infinity ? xEnd : e);
  const centers = lefts.map((l, i) => (l + rights[i]) / 2);
  const yMaxNew = probs.length ? Math.max(...probs) : 1.0;
  let entry = fpBarCharts.get(key);
  if (!entry) {
    const interior = edges.slice(1, -1);
    const edgeDesc = interior.length === 0
      ? 'no interior edges'
      : `${interior.length} interior edge${interior.length !== 1 ? 's' : ''}: ` + interior.map(e => fmtG(e, 4)).join(', ');
    const wrap = document.createElement('div');
    const titleEl = document.createElement('div');
    titleEl.className = 'overlay-title';
    const canvas = document.createElement('canvas');
    canvas.className = 'overlay-canvas';
    canvas.width = 640; canvas.height = 180;
    wrap.appendChild(titleEl); wrap.appendChild(canvas);
    fpBarchartWrapEl.appendChild(wrap);
    entry = { canvas, ctx: canvas.getContext('2d'), titleEl, xPoints: centers.slice(), yPoints: probs.slice(), n: 1, yMax: yMaxNew, edgeDesc, nBins: probs.length };
    fpBarCharts.set(key, entry);
  } else {
    entry.xPoints.push(...centers);
    entry.yPoints.push(...probs);
    entry.n += 1;
    entry.yMax = Math.max(entry.yMax, yMaxNew);
    entry.nBins = probs.length;
  }
  renderFpBarchart(entry);
}

fpBarAlphaSlider.addEventListener('input', () => {
  fpBarAlpha = parseFloat(fpBarAlphaSlider.value);
  fpBarAlphaValueSpan.textContent = fpBarAlpha.toFixed(2);
  for (const e of fpBarCharts.values()) renderFpBarchart(e);
});

// ───────────────────────── Viz panels (simplex3d / radial / scatter matrix / parallel coords) ─────────────────────────
const simplex3dState = createSimplex3DPanel(document.getElementById('simplex3d-panel'));
const radialState = createRadialPanel(document.getElementById('radial-panel'));
const scatterState = createScatterMatrixPanel(document.getElementById('scatter-matrix-panel'));
const parallelState = createParallelCoordsPanel(document.getElementById('parallel-panel'));

function updateSimplexStats() {
  const counts = new Map();
  for (const fp of allSimplexFixedPoints) {
    let found = 0;
    for (let i = 0; i < fp.length && found < 3; i++) {
      if (fp[i] > 1e-12) { counts.set(i, (counts.get(i) || 0) + 1); found++; }
    }
  }
  if (counts.size === 0) { simplexStatsDiv.innerHTML = ''; return; }
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
  const rows = sorted.map(([i, c]) => `<tr><td>p${i + 1}</td><td style="padding-left:8px;text-align:right">${c}×</td></tr>`).join('');
  simplexStatsDiv.innerHTML = `<b>Top bins</b> (n=${allSimplexFixedPoints.length})<table style="margin-top:6px">${rows}</table>`;
}

function currentMinFreq() { return parseFloat(minFreqSlider.value) / 100; }
function updateStatusUnlocked() {
  const bins = tracker.getActiveBins(currentMinFreq());
  const labels = tracker.getBinLabels(bins);
  binLockStatusEl.textContent = labels.length
    ? `Unlocked — ${bins.length} active bins (${labels.join(', ')})`
    : 'Unlocked — 0 active bins';
}
function updateStatusLocked() {
  const labels = lockedBinsState.labels;
  binLockStatusEl.textContent = labels.length ? `Locked to: ${labels.join(', ')}` : 'Locked to: (none)';
}
lockToggle.addEventListener('change', () => {
  if (lockToggle.checked) {
    const bins = tracker.getActiveBins(currentMinFreq());
    const labels = tracker.getBinLabels(bins);
    lockedBinsState.locked = true; lockedBinsState.bins = bins; lockedBinsState.labels = labels;
    updateStatusLocked();
  } else {
    lockedBinsState.locked = false; lockedBinsState.bins = []; lockedBinsState.labels = [];
    updateStatusUnlocked();
  }
  refreshVizPanels();
});
minFreqSlider.addEventListener('input', () => {
  minFreqValueSpan.textContent = minFreqSlider.value;
  if (!lockedBinsState.locked) updateStatusUnlocked();
});
updateStatusUnlocked();

// Note (faithful to bin_selection.py / find_fixed_point.py): the active bins actually fed
// to the 4 visualization panels while unlocked always use a fixed 10% frequency threshold —
// the min-frequency slider only controls the lock-time snapshot and the status line preview.
function getActiveBinsForViz() {
  if (lockedBinsState.locked) return [lockedBinsState.bins, lockedBinsState.labels];
  const bins = tracker.getActiveBins(0.1);
  return [bins, tracker.getBinLabels(bins)];
}

function refreshVizPanels() {
  if (!lockedBinsState.locked) updateStatusUnlocked();
  const [indices, labels] = getActiveBinsForViz();
  updateSimplex3DPanel(simplex3dState, allSimplexFixedPoints, indices, labels);
  updateRadialPanel(radialState, allSimplexFixedPoints, indices, labels);
  updateScatterMatrixPanel(scatterState, allSimplexFixedPoints, indices, labels);
  updateParallelCoordsPanel(parallelState, allSimplexFixedPoints, indices, labels);
}

clearSimplexBtn.addEventListener('click', () => {
  allSimplexFixedPoints = [];
  fpBarCharts.clear();
  fpBarchartWrapEl.innerHTML = '';
  tracker.reset();
  updateSimplexStats();
  refreshVizPanels();
});

// ───────────────────────── Core recompute pipeline (find_fixed_point.py: recompute / _update_surp_node) ─────────────────────────
function updateSurpNode(alpha, mu, sigma) {
  const edges = nodeP1.currentEdges, probs = nodeP1.currentProbs;
  if (!edges || !probs || nodeP1.events.length === 0) {
    const emptyEdges = [-Infinity, Infinity];
    computeNodeData(nodeS, emptyEdges, [1.0], null);
    drawNode(nodeS);
    nodeS.titleEl.textContent = 'S(P1) — First Surprisal Distribution';
    return;
  }
  const interior = edges.slice(1, -1);
  const surpEvents = nodeP1.events.map(e => {
    const idx = clip(searchsortedLeft(interior, e), 0, probs.length - 1);
    return -Math.log2(probs[idx]);
  });
  nodeS.events = surpEvents;
  const sEdges = [-Infinity, ...[...nodeS.interiorEdges].sort((a, b) => a - b), Infinity];
  const sCounts = binCounts(sEdges, surpEvents);
  const sProbs = computeProbabilities(sEdges, surpEvents, alpha, mu, sigma);
  computeNodeData(nodeS, sEdges, sProbs, sCounts);
  drawNode(nodeS);
  nodeS.titleEl.textContent = `S(P1)  |  entropy = ${entropyBits(sProbs).toFixed(4)} bits`;
}

function appRecompute() {
  const edges = [-Infinity, ...[...nodeP1.interiorEdges].sort((a, b) => a - b), Infinity];
  const alpha = nodeP1.dom.priorAlphaSlider.valueAsNumber;
  const mu = nodeP1.dom.priorMuSlider.valueAsNumber;
  const sigma = nodeP1.dom.priorSigmaSlider.valueAsNumber;
  const counts = binCounts(edges, nodeP1.events);
  const probs = computeProbabilities(edges, nodeP1.events, alpha, mu, sigma);
  computeNodeData(nodeP1, edges, probs, counts);
  drawNode(nodeP1);
  nodeP1.titleEl.textContent = `P1  |  entropy = ${entropyBits(probs).toFixed(4)} bits`;

  updateSurpNode(alpha, mu, sigma);

  const surpEdges = [-Infinity, ...[...nodeS.interiorEdges].sort((a, b) => a - b), Infinity];
  const tol = Math.pow(10, parseFloat(tolSlider.value));
  const [nIter, fixedProbs] = computeFixedPointIterations(nodeP1.events, edges, surpEdges, alpha, mu, sigma, tol);

  if (nIter === null && nodeP1.events.length === 0) {
    convergenceDiv.innerHTML = '<i>Add events to compute fixed-point iterations.</i>';
    return;
  } else if (nIter === null) {
    convergenceDiv.innerHTML = `<b>Did not converge within ${MAX_ITER} iterations.</b>`;
    return;
  }

  if (fixedProbs) {
    allSimplexFixedPoints.push(fixedProbs.slice());
    tracker.record(fixedProbs);
    updateSimplexStats();
    refreshVizPanels();
    addToFpBarchart(surpEdges, fixedProbs);
  }

  const s = nIter === 1 ? 'iteration' : 'iterations';
  const currentLine = `<b>Fixed point reached in ${nIter} ${s}.</b>`;

  if (nIter > sessionRecord) {
    sessionRecord = nIter;
    let distDesc = familySelect.value;
    if (currentParamSliders.length) {
      const paramStr = currentParamSliders.map(sl => `${sl.title}=${fmtG(sl.el.valueAsNumber, 3)}`).join(', ');
      distDesc += `(${paramStr})`;
    }
    const nEvents = nodeP1.events.length;
    const nInterior = nodeP1.interiorEdges.length;
    let edgeDesc;
    if (nInterior > 0) {
      const eMin = Math.min(...nodeP1.interiorEdges);
      const eMax = Math.max(...nodeP1.interiorEdges);
      edgeDesc = `${nInterior} interior edges [${fmtG(eMin, 3)}–${fmtG(eMax, 3)}]`;
    } else {
      edgeDesc = 'no interior edges';
    }
    const row = `<b>${nIter}</b> ${s} — ${distDesc} | events count: ${nEvents} | ${edgeDesc}`;
    sessionRecordRows.unshift(row);
  }

  convergenceDiv.innerHTML = sessionRecordRows.length
    ? currentLine + '<br><br><b>Records this session:</b><br>' + sessionRecordRows.join('<br>')
    : currentLine;
}

// ───────────────────────── Init ─────────────────────────
nodeP1.titleEl.textContent = 'P1  |  entropy = 0.0000 bits';
nodeS.titleEl.textContent = 'S(P1) — First Surprisal Distribution';
updateTransportState();
appRecompute();
