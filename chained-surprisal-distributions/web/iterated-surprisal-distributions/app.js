// --- Entropy & Surprisal Explorer: Chained Distributions (Web Port) ---
//
// Port of iterated_surprisal_distributions.py / events.py. See that file for
// the authoritative math; this mirrors it function-for-function.

// =============================================================================
// Constants
// =============================================================================

const X_MIN = -10, X_MAX = 30;
const PRIOR_ALPHA_DEFAULT = 0;
const PRIOR_MU_DEFAULT = 0;
const PRIOR_SIGMA_DEFAULT = 5;

// =============================================================================
// Math helpers
// =============================================================================

function erf(x) {
  // Abramowitz & Stegun 7.1.26, max abs error 1.5e-7.
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

function lowerBound(sortedArr, v) {
  // First index i such that sortedArr[i] >= v (numpy searchsorted, side='left').
  let lo = 0, hi = sortedArr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedArr[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function binCounts(edges, events) {
  const nBins = edges.length - 1;
  const interior = edges.slice(1, -1);
  const counts = new Array(nBins).fill(0);
  for (let i = 0; i < events.length; i++) {
    counts[lowerBound(interior, events[i])]++;
  }
  return counts;
}

function gaussianPriorMass(edges, mu, sigma) {
  const cdfs = edges.map((e) => normalCdf(e, mu, sigma));
  const out = new Array(edges.length - 1);
  for (let i = 0; i < out.length; i++) out[i] = cdfs[i + 1] - cdfs[i];
  return out;
}

function computeProbabilities(edges, events, alpha, mu, sigma) {
  const counts = binCounts(edges, events);
  const prior = gaussianPriorMass(edges, mu, sigma);
  const smoothed = counts.map((c, i) => c + alpha * prior[i]);
  const total = smoothed.reduce((a, b) => a + b, 0);
  if (total > 0) return smoothed.map((s) => s / total);
  return new Array(counts.length).fill(1 / counts.length);
}

function entropyBits(probs) {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log2(p);
  return h;
}

function klDivergenceBits(pEdges, pProbs, qEdges, qProbs) {
  const qMap = new Map();
  for (let i = 0; i < qProbs.length; i++) {
    qMap.set(`${qEdges[i]}|${qEdges[i + 1]}`, qProbs[i]);
  }
  let total = 0;
  for (let i = 0; i < pProbs.length; i++) {
    const p = pProbs[i];
    if (p <= 0) continue;
    const q = qMap.get(`${pEdges[i]}|${pEdges[i + 1]}`);
    if (q === undefined || q <= 0) return null;
    total += p * Math.log2(p / q);
  }
  return total;
}

function clipEdgesForWasserstein(edges) {
  return edges.map((e) => (e === -Infinity ? X_MIN : e === Infinity ? X_MAX : e));
}

function buildPiecewiseCdf(edges, probs) {
  const xs = [edges[0]];
  const fs = [0];
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    xs.push(edges[i + 1]);
    fs.push(cum);
  }
  return { xs, fs };
}

function interpLinear(xs, fs, x) {
  if (x <= xs[0]) return fs[0];
  if (x >= xs[xs.length - 1]) return fs[fs.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return fs[lo] + t * (fs[hi] - fs[lo]);
}

function wassersteinDistance(pEdges, pProbs, qEdges, qProbs) {
  const pe = clipEdgesForWasserstein(pEdges);
  const qe = clipEdgesForWasserstein(qEdges);
  const { xs: px, fs: pf } = buildPiecewiseCdf(pe, pProbs);
  const { xs: qx, fs: qf } = buildPiecewiseCdf(qe, qProbs);
  const allX = Array.from(new Set([...px, ...qx])).sort((a, b) => a - b);
  let total = 0;
  let prevX = allX[0];
  let prevDiff = Math.abs(interpLinear(px, pf, prevX) - interpLinear(qx, qf, prevX));
  for (let i = 1; i < allX.length; i++) {
    const x = allX[i];
    const diff = Math.abs(interpLinear(px, pf, x) - interpLinear(qx, qf, x));
    total += (x - prevX) * (diff + prevDiff) / 2;
    prevX = x;
    prevDiff = diff;
  }
  return total;
}

// =============================================================================
// Event source families & samplers
// =============================================================================

function sampleStdNormal() {
  let u1 = Math.random();
  if (u1 === 0) u1 = 1e-12;
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
    do {
      x = sampleStdNormal();
      v = 1 + c * x;
    } while (v <= 0);
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
  Uniform: {
    params: [
      { name: "low", start: -10, end: 10, value: 0.0, step: 0.1 },
      { name: "high", start: -10, end: 10, value: 1.0, step: 0.1 },
    ],
    sample: (p) => {
      const scale = Math.max(p["high"] - p["low"], 1e-6);
      return p["low"] + Math.random() * scale;
    },
  },
  Normal: {
    params: [
      { name: "μ", start: -10, end: 10, value: 0.0, step: 0.1 },
      { name: "σ", start: 0.1, end: 10, value: 1.0, step: 0.1 },
    ],
    sample: (p) => p["μ"] + p["σ"] * sampleStdNormal(),
  },
  Beta: {
    params: [
      { name: "α", start: 0.1, end: 10, value: 2.0, step: 0.1 },
      { name: "β", start: 0.1, end: 10, value: 5.0, step: 0.1 },
    ],
    sample: (p) => sampleBeta(p["α"], p["β"]),
  },
  Exponential: {
    params: [{ name: "λ", start: 0.1, end: 10, value: 1.0, step: 0.1 }],
    sample: (p) => -Math.log(Math.random()) / p["λ"],
  },
};
const FAMILY_NAMES = Object.keys(FAMILIES);

function getEvents(n, familyName, params) {
  const fn = FAMILIES[familyName].sample;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = fn(params);
  return out;
}

// =============================================================================
// Global state
// =============================================================================

let rootEvents = [];
let allEvents = [];
let historyIndex = 0;
let transportGuard = false;
const allNodes = [];
let rootNode = null;
let oneByOneHandle = null; // non-null while animating
let traceEnabled = true;

// =============================================================================
// Small DOM helpers
// =============================================================================

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) e.appendChild(c);
  return e;
}

function makeSlider({ label, min, max, step, value, width, formatter }) {
  const wrap = el("div", { class: "slider-field" });
  wrap.style.width = width ? width + "px" : "";
  const labelRow = el("div", { class: "slider-label" });
  const nameSpan = el("span", { text: label });
  const valueSpan = el("span", { text: formatter ? formatter(value) : String(value) });
  labelRow.appendChild(nameSpan);
  labelRow.appendChild(valueSpan);
  const input = el("input", { type: "range", min, max, step, value });
  wrap.appendChild(labelRow);
  wrap.appendChild(input);
  const api = {
    wrap, input, valueSpan,
    get value() { return parseFloat(input.value); },
    set value(v) {
      input.value = v;
      valueSpan.textContent = formatter ? formatter(v) : String(v);
    },
    onChange(fn) {
      input.addEventListener("input", () => {
        valueSpan.textContent = formatter ? formatter(api.value) : String(api.value);
        fn(api.value);
      });
    },
  };
  return api;
}

function makeRadioGroup(container, onChange) {
  const buttons = Array.from(container.querySelectorAll("button"));
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      onChange(btn.dataset.value, buttons.indexOf(btn));
    });
  });
  return {
    get active() {
      return buttons.findIndex((b) => b.classList.contains("active"));
    },
  };
}

// =============================================================================
// PNode: one histogram panel in the chain
// =============================================================================

let nodeCounter = 0;

function makeNode(initialEvents, isSurprisal) {
  const node = {
    id: nodeCounter++,
    events: initialEvents.slice(),
    isSurprisal,
    interiorEdges: [],
    singleEdges: [],
    parent: null,
    child: null,
    propagates: false,
    yMode: "prob", // 'prob' | 'density'
    yRangeAdaptive: true,
    currentEdges: [-Infinity, Infinity],
    currentProbs: null,
    currentCounts: null,
    bars: [], // last-drawn bar geometry, for hit-testing
    highlight: null, // {left,right,top,bottom} in data coords, or null
  };
  node.currentProbs = computeProbabilities(node.currentEdges, node.events, PRIOR_ALPHA_DEFAULT, PRIOR_MU_DEFAULT, PRIOR_SIGMA_DEFAULT);
  node.currentCounts = binCounts(node.currentEdges, node.events);

  const sMin = isSurprisal ? 0 : X_MIN;
  const sMax = isSurprisal ? 20 : X_MAX;
  const cMax = isSurprisal ? 1000 : 5000;
  node.xRange = { start: sMin, end: sMax };
  node.defaultXRange = { start: sMin, end: sMax };

  // ---- DOM scaffold ----
  const panel = el("div", { class: "node-panel" });
  const title = el("div", { class: "node-title", text: "P | entropy = 0.0000 bits" });

  const priorAlpha = makeSlider({ label: "Prior strength α", min: 0, max: 1000, step: 1, value: PRIOR_ALPHA_DEFAULT, width: 250 });
  const priorMu = makeSlider({ label: "Prior mean μ", min: -10, max: 10, step: 0.1, value: PRIOR_MU_DEFAULT, width: 250, formatter: (v) => v.toFixed(1) });
  const priorSigma = makeSlider({ label: "Prior std dev σ", min: 0.1, max: 20, step: 0.1, value: PRIOR_SIGMA_DEFAULT, width: 250, formatter: (v) => v.toFixed(1) });
  const priorRow = el("div", { class: "node-bottom-row" });
  priorRow.appendChild(priorAlpha.wrap);
  priorRow.appendChild(priorMu.wrap);
  priorRow.appendChild(priorSigma.wrap);

  const canvasWrap = el("div", { class: "node-canvas-wrap" });
  const canvas = el("canvas", { width: 660, height: 360 });
  const tooltip = el("div", { class: "node-tooltip" });
  canvasWrap.appendChild(canvas);
  canvasWrap.appendChild(tooltip);

  // Edge controls
  const addEdgeInput = el("input", { type: "text", placeholder: "Value…" });
  const addEdgeBtn = el("button", { text: "Add" });
  const edgeAddRow = el("div", { class: "edge-row" });
  edgeAddRow.appendChild(addEdgeInput);
  edgeAddRow.appendChild(addEdgeBtn);

  const splitSlider = makeSlider({ label: "Split point", min: sMin, max: sMax, step: 0.1, value: isSurprisal ? 5.0 : 0.0, formatter: (v) => v.toFixed(1) });
  const freezeBtn = el("button", { text: "Freeze edge" });
  const splitRow = el("div", { class: "edge-row" });
  splitRow.appendChild(splitSlider.wrap);
  splitRow.appendChild(freezeBtn);

  const ewLeft = makeSlider({ label: "Evenly spaced: left", min: sMin, max: sMax, step: 0.1, value: isSurprisal ? 0.0 : -3.0, formatter: (v) => v.toFixed(1) });
  const ewRight = makeSlider({ label: "Evenly spaced: right", min: sMin, max: sMax, step: 0.1, value: isSurprisal ? 10.0 : 3.0, formatter: (v) => v.toFixed(1) });
  const ewCount = makeSlider({ label: "Evenly spaced: edge count", min: 0, max: cMax, step: 1, value: 0 });

  const edgePanel = el("div", { class: "edge-panel" });
  edgePanel.appendChild(edgeAddRow);
  edgePanel.appendChild(splitRow);
  edgePanel.appendChild(ewLeft.wrap);
  edgePanel.appendChild(ewRight.wrap);
  edgePanel.appendChild(ewCount.wrap);

  const plotRow = el("div", { class: "node-plot-row" });
  plotRow.appendChild(canvasWrap);
  plotRow.appendChild(edgePanel);

  // y mode / scale row
  const yModeRadioEl = el("div", { class: "radio-group" });
  const yProbBtn = el("button", { text: "Probability", class: "active" });
  const yDensBtn = el("button", { text: "Probability density" });
  yModeRadioEl.appendChild(yProbBtn);
  yModeRadioEl.appendChild(yDensBtn);

  const yScaleSelect = el("select");
  yScaleSelect.appendChild(el("option", { value: "adaptive", text: "Y: adaptive" }));
  yScaleSelect.appendChild(el("option", { value: "fixed", text: "Y: fixed 0–1" }));
  yScaleSelect.value = "adaptive";

  const yRow = el("div", { class: "node-bottom-row" });
  yRow.appendChild(yModeRadioEl);
  yRow.appendChild(yScaleSelect);

  // derive row
  const deriveBtn = el("button", { class: "primary", text: "View derived distribution" });
  const gangLabel = el("label", { class: "checkbox-field" });
  const gangCheckbox = el("input", { type: "checkbox" });
  gangLabel.appendChild(gangCheckbox);
  gangLabel.appendChild(document.createTextNode("Control all descendants' parameters"));
  const klDisplay = el("div", { class: "kl-display" });

  const deriveRow = el("div", { class: "node-bottom-row" });
  deriveRow.appendChild(deriveBtn);
  deriveRow.appendChild(gangLabel);
  deriveRow.appendChild(klDisplay);

  panel.appendChild(title);
  panel.appendChild(priorRow);
  panel.appendChild(plotRow);
  panel.appendChild(yRow);
  panel.appendChild(deriveRow);

  node.dom = {
    panel, title, canvas, ctx: canvas.getContext("2d"), tooltip,
    priorAlpha, priorMu, priorSigma,
    addEdgeInput, addEdgeBtn, splitSlider, freezeBtn, ewLeft, ewRight, ewCount,
    yProbBtn, yDensBtn, yScaleSelect, deriveBtn, gangCheckbox, klDisplay,
  };

  // ---- behavior ----

  function syncEdgesAndRecompute() {
    const edgeSet = new Set([splitSlider.value, ...node.singleEdges]);
    const count = Math.round(ewCount.value);
    const left = ewLeft.value, right = ewRight.value;
    if (count > 0 && right > left) {
      const step = (right - left) / (count + 1);
      for (let i = 1; i <= count; i++) edgeSet.add(left + step * i);
    }
    node.interiorEdges = Array.from(edgeSet).sort((a, b) => a - b);
    recomputeFrom(node);
  }
  node.syncEdgesAndRecompute = syncEdgesAndRecompute;
  node.interiorEdges = [splitSlider.value];

  function onBinEdgeSliderChange() {
    syncEdgesAndRecompute();
    if (node.propagates) propagateParamsDown(node);
  }
  [splitSlider, ewLeft, ewRight, ewCount].forEach((s) => s.onChange(onBinEdgeSliderChange));

  function setYMode(mode) {
    node.yMode = mode;
    yProbBtn.classList.toggle("active", mode === "prob");
    yDensBtn.classList.toggle("active", mode === "density");
    drawNode(node);
  }
  yProbBtn.addEventListener("click", () => setYMode("prob"));
  yDensBtn.addEventListener("click", () => setYMode("density"));

  yScaleSelect.addEventListener("change", () => {
    node.yRangeAdaptive = yScaleSelect.value === "adaptive";
    recomputeFrom(node);
  });

  function onPriorChange() {
    recomputeFrom(node);
    if (node.propagates) propagateParamsDown(node);
  }
  priorAlpha.onChange(onPriorChange);
  priorMu.onChange(onPriorChange);
  priorSigma.onChange(onPriorChange);

  gangCheckbox.addEventListener("change", () => {
    node.propagates = gangCheckbox.checked;
    if (node.propagates) propagateParamsDown(node);
  });

  freezeBtn.addEventListener("click", () => {
    const val = splitSlider.value;
    if (!node.singleEdges.includes(val)) node.singleEdges.push(val);
    syncEdgesAndRecompute();
  });

  addEdgeBtn.addEventListener("click", () => {
    const valStr = addEdgeInput.value.trim();
    if (!valStr) return;
    const val = parseFloat(valStr);
    if (Number.isNaN(val)) return;
    if (!node.singleEdges.includes(val)) node.singleEdges.push(val);
    addEdgeInput.value = "";
    syncEdgesAndRecompute();
    if (node.propagates) propagateParamsDown(node);
  });
  addEdgeInput.addEventListener("keydown", (e) => { if (e.key === "Enter") addEdgeBtn.click(); });

  deriveBtn.addEventListener("click", () => createChildNode(node));

  attachCanvasInteraction(node);

  return node;
}

// =============================================================================
// Canvas drawing & hit-testing
// =============================================================================

const MARGIN = { left: 56, right: 14, top: 14, bottom: 32 };

function dataToPixel(node, cw, ch, x, y, yEnd) {
  const pw = cw - MARGIN.left - MARGIN.right;
  const ph = ch - MARGIN.top - MARGIN.bottom;
  const px = MARGIN.left + ((x - node.xRange.start) / (node.xRange.end - node.xRange.start)) * pw;
  const py = MARGIN.top + ph - (y / yEnd) * ph;
  return [px, py];
}

function pixelToDataX(node, cw, x) {
  const pw = cw - MARGIN.left - MARGIN.right;
  return node.xRange.start + ((x - MARGIN.left) / pw) * (node.xRange.end - node.xRange.start);
}

function computeBars(node) {
  const edges = node.currentEdges, probs = node.currentProbs, counts = node.currentCounts;
  const totalCount = counts.reduce((a, b) => a + b, 0);
  const bars = [];
  for (let i = 0; i < probs.length; i++) {
    const leftEdge = edges[i], rightEdge = edges[i + 1];
    const leftInf = leftEdge === -Infinity, rightInf = rightEdge === Infinity;
    const dl = leftInf ? node.xRange.start : leftEdge;
    const dr = rightInf ? node.xRange.end : rightEdge;
    const width = dr - dl;
    const density = width > 0 ? probs[i] / width : 0;
    const top = node.yMode === "density" ? density : probs[i];
    bars.push({
      dl, dr, top, count: counts[i], prob: probs[i], density,
      rawProb: totalCount > 0 ? counts[i] / totalCount : 0,
      leftActual: leftEdge, rightActual: rightEdge,
      edgeLeftStr: leftInf ? "-∞" : formatNum(leftEdge),
      edgeRightStr: rightInf ? "+∞" : formatNum(rightEdge),
    });
  }
  return bars;
}

function formatNum(x) {
  return Number(x.toPrecision(4)).toString();
}

function drawNode(node) {
  const canvas = node.dom.canvas;
  const ctx = node.dom.ctx;
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth || 660, ch = canvas.clientHeight || 360;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  node.bars = computeBars(node);

  let yEnd;
  if (node.yRangeAdaptive) {
    const maxTop = node.bars.length ? Math.max(...node.bars.map((b) => b.top)) : 1;
    yEnd = maxTop > 0 ? maxTop * 1.05 : 1;
  } else {
    yEnd = 1;
  }
  node.yEnd = yEnd;

  // background
  ctx.fillStyle = "#0e1530";
  ctx.fillRect(MARGIN.left, MARGIN.top, cw - MARGIN.left - MARGIN.right, ch - MARGIN.top - MARGIN.bottom);

  // bars
  for (const b of node.bars) {
    const [x0, y0] = dataToPixel(node, cw, ch, b.dl, b.top, yEnd);
    const [x1, y1] = dataToPixel(node, cw, ch, b.dr, 0, yEnd);
    ctx.fillStyle = "rgba(72, 120, 207, 0.8)";
    ctx.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.strokeStyle = "black";
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
  }

  // interior edge marker lines
  ctx.strokeStyle = "rgba(0,0,0,0.25)";
  ctx.lineWidth = 1;
  for (const e of node.interiorEdges) {
    if (e <= node.xRange.start || e >= node.xRange.end) continue;
    const [x] = dataToPixel(node, cw, ch, e, 0, yEnd);
    ctx.beginPath();
    ctx.moveTo(x, MARGIN.top);
    ctx.lineTo(x, ch - MARGIN.bottom);
    ctx.stroke();
  }

  // highlight overlay
  if (node.highlight && node.highlight.left.length) {
    ctx.fillStyle = "rgba(255,0,255,0.5)";
    for (let i = 0; i < node.highlight.left.length; i++) {
      const [x0, y0] = dataToPixel(node, cw, ch, node.highlight.left[i], node.highlight.top[i], yEnd);
      const [x1, y1] = dataToPixel(node, cw, ch, node.highlight.right[i], node.highlight.bottom[i], yEnd);
      ctx.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    }
  }

  // axes
  ctx.strokeStyle = "#555";
  ctx.beginPath();
  ctx.moveTo(MARGIN.left, MARGIN.top);
  ctx.lineTo(MARGIN.left, ch - MARGIN.bottom);
  ctx.lineTo(cw - MARGIN.right, ch - MARGIN.bottom);
  ctx.stroke();

  ctx.fillStyle = "#aaa";
  ctx.font = "11px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const xTicks = 6;
  for (let i = 0; i <= xTicks; i++) {
    const xv = node.xRange.start + (node.xRange.end - node.xRange.start) * (i / xTicks);
    const [x] = dataToPixel(node, cw, ch, xv, 0, yEnd);
    ctx.beginPath();
    ctx.moveTo(x, ch - MARGIN.bottom);
    ctx.lineTo(x, ch - MARGIN.bottom + 4);
    ctx.stroke();
    ctx.fillText(formatNum(xv), x, ch - MARGIN.bottom + 6);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const yv = (yEnd * i) / yTicks;
    const [, y] = dataToPixel(node, cw, ch, node.xRange.start, yv, yEnd);
    ctx.beginPath();
    ctx.moveTo(MARGIN.left - 4, y);
    ctx.lineTo(MARGIN.left, y);
    ctx.stroke();
    ctx.fillText(yv.toFixed(2), MARGIN.left - 6, y);
  }
  ctx.save();
  ctx.translate(14, MARGIN.top + (ch - MARGIN.top - MARGIN.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(node.yMode === "density" ? "Probability density" : "Probability", 0, 0);
  ctx.restore();
  ctx.textAlign = "center";
  ctx.fillText("Value", MARGIN.left + (cw - MARGIN.left - MARGIN.right) / 2, ch - 10);
}

function attachCanvasInteraction(node) {
  const canvas = node.dom.canvas;
  const tooltip = node.dom.tooltip;
  let dragging = false;
  let dragStartX = 0;
  let dragStartRange = null;

  function hitTest(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const cw = rect.width;
    const dataX = pixelToDataX(node, cw, clientX - rect.left);
    for (let i = 0; i < node.bars.length; i++) {
      const b = node.bars[i];
      if (dataX >= b.dl && dataX < b.dr) return i;
    }
    return -1;
  }

  canvas.addEventListener("mousemove", (e) => {
    if (dragging) {
      const rect = canvas.getBoundingClientRect();
      const pw = rect.width - MARGIN.left - MARGIN.right;
      const dxData = ((dragStartX - e.clientX) / pw) * (dragStartRange.end - dragStartRange.start);
      node.xRange.start = dragStartRange.start + dxData;
      node.xRange.end = dragStartRange.end + dxData;
      drawNode(node);
      return;
    }
    const idx = hitTest(e.clientX, e.clientY);
    if (idx === -1) {
      tooltip.style.display = "none";
      if (traceEnabled) clearAllHighlights();
      return;
    }
    const b = node.bars[idx];
    const rect = canvas.getBoundingClientRect();
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX - rect.left + 12) + "px";
    tooltip.style.top = (e.clientY - rect.top - 10) + "px";
    tooltip.innerHTML =
      `Bin: ${b.edgeLeftStr} to ${b.edgeRightStr}<br>` +
      `Count: ${b.count}<br>` +
      `Probability before prior: ${b.rawProb.toFixed(4)}<br>` +
      `Probability: ${b.prob.toFixed(4)}<br>` +
      `Density: ${b.density.toFixed(4)}`;
    if (traceEnabled) traceHover(node, idx); else clearAllHighlights();
  });

  canvas.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
    dragging = false;
    if (traceEnabled) clearAllHighlights();
  });

  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    dragStartX = e.clientX;
    dragStartRange = { start: node.xRange.start, end: node.xRange.end };
    tooltip.style.display = "none";
  });
  window.addEventListener("mouseup", () => { dragging = false; });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const cw = rect.width;
    const dataX = pixelToDataX(node, cw, e.clientX - rect.left);
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    const start = dataX - (dataX - node.xRange.start) * factor;
    const end = dataX + (node.xRange.end - dataX) * factor;
    node.xRange.start = start;
    node.xRange.end = end;
    drawNode(node);
  }, { passive: false });

  canvas.addEventListener("dblclick", () => {
    node.xRange.start = node.defaultXRange.start;
    node.xRange.end = node.defaultXRange.end;
    drawNode(node);
  });
}

// =============================================================================
// Trace hover feature
// =============================================================================

function clearAllHighlights() {
  let changed = false;
  for (const n of allNodes) {
    if (n.highlight) { n.highlight = null; changed = true; drawNode(n); }
  }
  return changed;
}

function emptyHighlight() { return { left: [], right: [], top: [], bottom: [] }; }

function traceHover(sourceNode, barIndex) {
  // Clear everyone first.
  for (const n of allNodes) n.highlight = null;

  const b = sourceNode.bars[barIndex];

  // Walk up: ancestors.
  let ranges = [[b.leftActual, b.rightActual]];
  let cur = sourceNode.parent;
  while (cur) {
    const hl = emptyHighlight();
    const nextRanges = [];
    for (const bar of cur.bars) {
      const surp = -Math.log2(bar.prob);
      for (const rng of ranges) {
        if (surp >= rng[0] && surp <= rng[1]) {
          hl.left.push(bar.dl); hl.right.push(bar.dr); hl.top.push(bar.top); hl.bottom.push(0);
          nextRanges.push([bar.leftActual, bar.rightActual]);
          break;
        }
      }
    }
    cur.highlight = hl;
    ranges = nextRanges;
    if (ranges.length === 0) break;
    cur = cur.parent;
  }

  // Walk down: descendants.
  let surp = -Math.log2(b.prob);
  cur = sourceNode.child;
  while (cur) {
    const hl = emptyHighlight();
    let nextSurp = null;
    for (const bar of cur.bars) {
      if (surp >= bar.leftActual && surp < bar.rightActual) {
        hl.left.push(bar.dl); hl.right.push(bar.dr); hl.top.push(bar.top); hl.bottom.push(0);
        nextSurp = -Math.log2(bar.prob);
        break;
      }
    }
    cur.highlight = hl;
    if (nextSurp === null) break;
    surp = nextSurp;
    cur = cur.child;
  }

  for (const n of allNodes) drawNode(n);
}

// =============================================================================
// Core recomputation
// =============================================================================

function nodeIndex(node) { return allNodes.indexOf(node); }

function recomputeFrom(node) {
  if (!node) return;
  const edges = [-Infinity, ...node.interiorEdges.slice().sort((a, b) => a - b), Infinity];
  const alpha = node.dom.priorAlpha.value;
  const mu = node.dom.priorMu.value;
  const sigma = node.dom.priorSigma.value;
  const counts = binCounts(edges, node.events);
  const probs = computeProbabilities(edges, node.events, alpha, mu, sigma);
  node.currentEdges = edges;
  node.currentProbs = probs;
  node.currentCounts = counts;

  drawNode(node);

  const idx = nodeIndex(node);
  node.dom.title.textContent = `P${idx + 1}  |  entropy = ${entropyBits(probs).toFixed(4)} bits`;

  if (node.child) {
    const interior = edges.slice(1, -1);
    const childEvents = node.events.map((v) => -Math.log2(probs[lowerBound(interior, v)]));
    node.child.events = childEvents;
    recomputeFrom(node.child);
  }

  refreshKlDisplay(node);
  if (node.parent) refreshKlDisplay(node.parent);
}

const KL_LINK = '<a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence" target="_blank">KL</a>';
const W1_LINK = '<a href="https://en.wikipedia.org/wiki/Wasserstein_metric" target="_blank">W1</a>';

function refreshKlDisplay(node) {
  if (!node.currentEdges) return;
  const lines = [];
  const { currentEdges: edges, currentProbs: probs } = node;
  if (node.parent && node.parent.currentEdges) {
    const klUp = klDivergenceBits(edges, probs, node.parent.currentEdges, node.parent.currentProbs);
    const w1Up = wassersteinDistance(edges, probs, node.parent.currentEdges, node.parent.currentProbs);
    const klStr = klUp !== null ? `${klUp.toFixed(4)} bits` : "undefined";
    lines.push(`&uarr; ${KL_LINK} ${klStr} &nbsp; ${W1_LINK} ${w1Up.toFixed(4)}`);
  }
  if (node.child && node.child.currentEdges) {
    const klDown = klDivergenceBits(edges, probs, node.child.currentEdges, node.child.currentProbs);
    const w1Down = wassersteinDistance(edges, probs, node.child.currentEdges, node.child.currentProbs);
    const klStr = klDown !== null ? `${klDown.toFixed(4)} bits` : "undefined";
    lines.push(`&darr; ${KL_LINK} ${klStr} &nbsp; ${W1_LINK} ${w1Down.toFixed(4)}`);
  }
  node.dom.klDisplay.innerHTML = lines.join("<br>");
}

function propagateParamsDown(node) {
  const child = node.child;
  if (!child) return;
  child.singleEdges = node.singleEdges.slice();
  child.dom.splitSlider.value = node.dom.splitSlider.value;
  child.dom.ewLeft.value = node.dom.ewLeft.value;
  child.dom.ewRight.value = node.dom.ewRight.value;
  child.dom.ewCount.value = node.dom.ewCount.value;
  child.dom.priorAlpha.value = node.dom.priorAlpha.value;
  child.dom.priorMu.value = node.dom.priorMu.value;
  child.dom.priorSigma.value = node.dom.priorSigma.value;
  child.syncEdgesAndRecompute();
  if (!child.propagates) {
    child.propagates = true;
    child.dom.gangCheckbox.checked = true;
  }
  propagateParamsDown(child);
}

// =============================================================================
// Node creation & grid layout
// =============================================================================

function createChildNode(parentNode) {
  let childEvents;
  let isSurprisal;
  if (parentNode) {
    const edges = [-Infinity, ...parentNode.interiorEdges.slice().sort((a, b) => a - b), Infinity];
    const alpha = parentNode.dom.priorAlpha.value;
    const mu = parentNode.dom.priorMu.value;
    const sigma = parentNode.dom.priorSigma.value;
    const probs = computeProbabilities(edges, parentNode.events, alpha, mu, sigma);
    const interior = edges.slice(1, -1);
    childEvents = parentNode.events.map((v) => -Math.log2(probs[lowerBound(interior, v)]));
    isSurprisal = true;
  } else {
    childEvents = rootEvents.slice();
    isSurprisal = false;
  }

  const newNode = makeNode(childEvents, isSurprisal);

  if (parentNode) {
    parentNode.child = newNode;
    newNode.parent = parentNode;
    parentNode.dom.deriveBtn.disabled = true;
    if (parentNode.propagates) propagateParamsDown(parentNode);
  } else {
    rootNode = newNode;
    document.getElementById("initial-derive-wrap").style.display = "none";
  }

  allNodes.push(newNode);
  rebuildGrid();
  recomputeFrom(newNode);
}

function rebuildGrid() {
  const grid = document.getElementById("node-grid");
  grid.innerHTML = "";
  for (const n of allNodes) grid.appendChild(n.dom.panel);
  for (const n of allNodes) drawNode(n);
}

// =============================================================================
// Top-level event controls
// =============================================================================

let currentParamSliderApis = [];

function getCurrentParams() {
  const out = {};
  for (const api of currentParamSliderApis) out[api.label] = api.value;
  return out;
}

function makeParamSliders(familyName) {
  const row = document.getElementById("dist-params-row");
  row.innerHTML = "";
  currentParamSliderApis = [];
  for (const spec of FAMILIES[familyName].params) {
    const s = makeSlider({
      label: spec.name, min: spec.start, max: spec.end, step: spec.step, value: spec.value,
      width: 170, formatter: (v) => v.toFixed(2),
    });
    s.label = spec.name;
    s.onChange(() => { if (appendReplaceRadio.active === 1) doReplace(); });
    row.appendChild(s.wrap);
    currentParamSliderApis.push(s);
  }
}

function onMakeDist() {
  if (!rootNode) return;
  rootNode.events = rootEvents.slice();
  recomputeFrom(rootNode);
}

function updateTransportState() {
  const n = allEvents.length;
  const atEnd = historyIndex === n;
  const atStart = historyIndex === 0;
  transportGuard = true;
  historySlider.max = String(Math.max(n, 1));
  historySlider.value = String(historyIndex);
  historySlider.disabled = n === 0;
  transportGuard = false;
  historyLabel.textContent = `Step ${historyIndex} of ${n}`;
  historyBackBtn.disabled = atStart;
  historyFwdBtn.disabled = atEnd;
  clearEventsBtn.disabled = allEvents.length === 0;
}

function applyHistoryIndex(idx) {
  historyIndex = Math.max(0, Math.min(idx, allEvents.length));
  rootEvents = allEvents.slice(0, historyIndex);
  updateTransportState();
  onMakeDist();
}

function onAddEvents() {
  if (appendReplaceRadio.active === 1) { doReplace(); return; }
  let n = parseInt(nEventsInput.value, 10);
  if (!Number.isFinite(n) || n <= 0) { n = 1000; nEventsInput.value = "1000"; }
  const wasAtEnd = historyIndex === allEvents.length;
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = allEvents.concat(newEv);
  if (wasAtEnd) historyIndex = allEvents.length;
  rootEvents = allEvents.slice(0, historyIndex);
  updateTransportState();
  onMakeDist();
}

function doReplace() {
  let n = parseInt(nEventsInput.value, 10);
  if (!Number.isFinite(n) || n <= 0) n = 1000;
  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = newEv.slice();
  historyIndex = allEvents.length;
  rootEvents = allEvents.slice();
  updateTransportState();
  onMakeDist();
}

function onClearEvents() {
  rootEvents = [];
  allEvents = [];
  historyIndex = 0;
  onMakeDist();
  updateTransportState();
}

function onSingleEventInput() {
  const valStr = singleEventInput.value.trim();
  if (!valStr) return;
  const val = parseFloat(valStr);
  const statusEl = document.getElementById("single-event-status");
  if (Number.isNaN(val)) {
    statusEl.textContent = `'${valStr}' is not a valid number.`;
    singleEventInput.value = "";
    return;
  }
  let count = parseInt(singleEventCountInput.value, 10);
  if (!Number.isFinite(count)) count = 0;
  const n = Math.max(count, 1);
  if (appendReplaceRadio.active === 1) {
    allEvents = new Array(n).fill(val);
    historyIndex = n;
  } else {
    const wasAtEnd = historyIndex === allEvents.length;
    allEvents = allEvents.concat(new Array(n).fill(val));
    if (wasAtEnd) historyIndex = allEvents.length;
  }
  rootEvents = allEvents.slice(0, historyIndex);
  statusEl.textContent = `Added ${n} event${n > 1 ? "s" : ""} at ${val}.`;
  singleEventInput.value = "";
  updateTransportState();
  onMakeDist();
}

function onAddEventsOneByOne() {
  if (oneByOneHandle !== null) {
    oneByOneHandle = null;
    addOneByOneBtn.textContent = "Add events (one by one)";
    return;
  }
  let n = parseInt(nEventsInput.value, 10);
  if (!Number.isFinite(n) || n <= 0) { n = 1000; nEventsInput.value = "1000"; }

  const newEv = getEvents(n, familySelect.value, getCurrentParams());
  allEvents = allEvents.concat(newEv);
  const targetIndex = allEvents.length;

  addOneByOneBtn.textContent = "Stop";
  oneByOneHandle = {};

  function step() {
    if (oneByOneHandle === null) {
      addOneByOneBtn.textContent = "Add events (one by one)";
      return;
    }
    applyHistoryIndex(historyIndex + 1);
    if (historyIndex >= targetIndex) {
      oneByOneHandle = null;
      addOneByOneBtn.textContent = "Add events (one by one)";
    } else {
      requestAnimationFrame(step);
    }
  }
  requestAnimationFrame(step);
}

// =============================================================================
// Wiring top-level DOM
// =============================================================================

const familySelect = document.getElementById("family-select");
FAMILY_NAMES.forEach((name) => familySelect.appendChild(el("option", { value: name, text: name })));

const appendReplaceRadio = makeRadioGroup(document.getElementById("append-replace-radio"), () => {
  if (appendReplaceRadio.active === 1) doReplace();
});

const addEventsBtn = document.getElementById("add-events-btn");
const addOneByOneBtn = document.getElementById("add-events-one-by-one-btn");
const clearEventsBtn = document.getElementById("clear-events-btn");
const nEventsInput = document.getElementById("n-events-input");
const singleEventInput = document.getElementById("single-event-input");
const singleEventCountInput = document.getElementById("single-event-count-input");
const traceCheckbox = document.getElementById("trace-checkbox");
const historyBackBtn = document.getElementById("history-back-btn");
const historyFwdBtn = document.getElementById("history-fwd-btn");
const historySlider = document.getElementById("history-slider");
const historyLabel = document.getElementById("history-label");
const initialDeriveBtn = document.getElementById("initial-derive-btn");

addEventsBtn.addEventListener("click", onAddEvents);
addOneByOneBtn.addEventListener("click", onAddEventsOneByOne);
clearEventsBtn.addEventListener("click", onClearEvents);
singleEventInput.addEventListener("keydown", (e) => { if (e.key === "Enter") onSingleEventInput(); });
singleEventInput.addEventListener("blur", onSingleEventInput);
initialDeriveBtn.addEventListener("click", () => createChildNode(null));
familySelect.addEventListener("change", () => {
  makeParamSliders(familySelect.value);
  if (appendReplaceRadio.active === 1) doReplace();
});
traceCheckbox.addEventListener("change", () => {
  traceEnabled = traceCheckbox.checked;
  if (!traceEnabled) clearAllHighlights();
});

historySlider.addEventListener("input", () => {
  if (transportGuard) return;
  applyHistoryIndex(parseInt(historySlider.value, 10));
});
historyBackBtn.addEventListener("click", () => applyHistoryIndex(historyIndex - 1));
historyFwdBtn.addEventListener("click", () => applyHistoryIndex(historyIndex + 1));

makeParamSliders(FAMILY_NAMES[0]);
updateTransportState();

window.addEventListener("resize", () => { for (const n of allNodes) drawNode(n); });
