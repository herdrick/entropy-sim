// --- Entropy Simulator (Web Version) ---
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

// =============================================================================
// Constants & Binning
// =============================================================================

// 21 edges produce 20 bins.
// Bin 0 = (-inf, edge[1])        — leftmost, open on left
// Bin i (1..18) = [edge[i], edge[i+1])  — interior
// Bin 19 = [edge[19], +inf)      — rightmost, open on right
const N_BIN_EDGES = 21;
const BIN_EDGES = Array.from({ length: N_BIN_EDGES }, (_, i) => i / (N_BIN_EDGES - 1));
const TOTAL_BINS = N_BIN_EDGES - 1; // 20
const BIN_WIDTH = 1.0 / (N_BIN_EDGES - 1); // 0.05

const COLORS = {
  accent: '#e94560',
  highlight: '#e2d810',
  bg: '#1a1a2e',
  panel: '#16213e',
  barFill: 0x0f3460,
  barEdge: 0xe94560,
};

function getBin(value) {
  if (value < BIN_EDGES[1]) return 0;
  if (value >= BIN_EDGES[N_BIN_EDGES - 1]) return TOTAL_BINS - 1;
  let lo = 1, hi = N_BIN_EDGES - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (BIN_EDGES[mid] <= value) lo = mid; else hi = mid - 1;
  }
  return lo;
}

// =============================================================================
// Mystery Sources
// =============================================================================

const SOURCES = {
  'Source A': { name: 'Uniform(0,1)', type: 'uniform' },
  'Source B': { name: 'Beta(2,5)', alpha: 2, beta: 5, type: 'beta' },
  'Source C': { name: 'Beta(0.5,0.5)', alpha: 0.5, beta: 0.5, type: 'beta' },
  'Source D': { name: 'Beta(5,5)', alpha: 5, beta: 5, type: 'beta' },
  'Source E': { name: 'Beta(0.3,0.3)', alpha: 0.3, beta: 0.3, type: 'beta' },
  'Source F': {
    name: 'Mixture: 0.5*Beta(2,2)+0.5*Beta(20,20)', type: 'mixture',
    components: [{ alpha: 2, beta: 2, weight: 0.5 }, { alpha: 20, beta: 20, weight: 0.5 }],
  },
};

function sampleSource(key) {
  const src = SOURCES[key];
  if (src.type === 'uniform') return Math.random();
  if (src.type === 'beta') return jStat.beta.sample(src.alpha, src.beta);
  const r = Math.random();
  let cum = 0;
  for (const c of src.components) {
    cum += c.weight;
    if (r < cum) return jStat.beta.sample(c.alpha, c.beta);
  }
  const last = src.components[src.components.length - 1];
  return jStat.beta.sample(last.alpha, last.beta);
}

function sourcePdf(key, x) {
  const src = SOURCES[key];
  if (src.type === 'uniform') return (x >= 0 && x <= 1) ? 1.0 : 0.0;
  if (src.type === 'beta') return jStat.beta.pdf(x, src.alpha, src.beta);
  return src.components.reduce((s, c) => s + c.weight * jStat.beta.pdf(x, c.alpha, c.beta), 0);
}

function sourceCdf(key, x) {
  const src = SOURCES[key];
  if (src.type === 'uniform') return x <= 0 ? 0 : x >= 1 ? 1 : x;
  if (src.type === 'beta') return jStat.beta.cdf(x, src.alpha, src.beta);
  return src.components.reduce((s, c) => s + c.weight * jStat.beta.cdf(x, c.alpha, c.beta), 0);
}

// =============================================================================
// Entropy & Surprisal Math (all in bits)
// =============================================================================

function modelEntropy(counts) {
  const smoothed = counts.map(c => c + 1);
  const total = smoothed.reduce((a, b) => a + b, 0);
  let h = 0;
  for (const s of smoothed) {
    const p = s / total;
    h -= p * Math.log2(p);
  }
  return h;
}

function eventSurprisal(value, counts) {
  const idx = getBin(value);
  const smoothed = counts.map(c => c + 1);
  const total = smoothed.reduce((a, b) => a + b, 0);
  return -Math.log2(smoothed[idx] / total);
}

function sourceEntropy(key) {
  const cdfAtEdges = BIN_EDGES.map(e => sourceCdf(key, e));
  const binProbs = new Array(TOTAL_BINS).fill(0);
  // Bin 0: (-inf, edge[1]) = CDF(edge[1])
  binProbs[0] = cdfAtEdges[1];
  // Interior bins 1..18: [edge[i], edge[i+1])
  for (let i = 1; i < TOTAL_BINS - 1; i++) {
    binProbs[i] = cdfAtEdges[i + 1] - cdfAtEdges[i];
  }
  // Bin 19: [edge[19], +inf) = 1 - CDF(edge[19])
  binProbs[TOTAL_BINS - 1] = 1.0 - cdfAtEdges[N_BIN_EDGES - 1];

  let h = 0;
  for (const p of binProbs) {
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

// =============================================================================
// Application State
// =============================================================================

const state = {
  currentSource: 'Source A',
  playing: false,
  revealed: false,
  speed: 10,
  events: [],
  counts: new Array(TOTAL_BINS).fill(0),
  entropyHistory: [],
  surprisalHistory: [],
  runningAvgSurprisal: [],
  timer: null,
};

function resetData() {
  state.events = [];
  state.counts = new Array(TOTAL_BINS).fill(0);
  state.entropyHistory = [];
  state.surprisalHistory = [];
  state.runningAvgSurprisal = [];
}

// =============================================================================
// Three.js Histogram (Panel 1)
// =============================================================================

let scene, camera, renderer, orbitControls;
let barMeshes = [[]]; // barMeshes[row][col] — single row for 1D, grid for future 2D
let pdfLine = null;
let initialCameraPosition, initialCameraTarget;

function initHistogram() {
  const container = document.getElementById('histogram-container');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.bg);

  const aspect = container.offsetWidth / container.offsetHeight;
  const xSpan = 1.4;
  const ySpan = xSpan / aspect;
  camera = new THREE.OrthographicCamera(-0.2, 1.2, ySpan * 0.8, -ySpan * 0.1, -10, 10);
  camera.position.set(0.5, 0.3, 5);
  camera.lookAt(0.5, 0.3, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.offsetWidth, container.offsetHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.insertBefore(renderer.domElement, container.firstChild);

  orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.rotateSpeed = 0.7;
  orbitControls.zoomSpeed = 0.8;
  orbitControls.panSpeed = 1.0;

  initialCameraPosition = camera.position.clone();
  initialCameraTarget = orbitControls.target.clone();

  // Create bar meshes
  barMeshes = [[]];
  const barColor = new THREE.Color(COLORS.barFill);
  const edgeColor = new THREE.Color(COLORS.barEdge);

  for (let i = 0; i < TOTAL_BINS; i++) {
    const barW = BIN_WIDTH * 0.9;
    const geo = new THREE.BoxGeometry(barW, 1, 0.001);
    const mat = new THREE.MeshBasicMaterial({ color: barColor.clone() });
    const mesh = new THREE.Mesh(geo, mat);

    // Bin center positions
    let xCenter;
    if (i === 0) {
      xCenter = BIN_EDGES[0] + BIN_WIDTH / 2; // leftmost displayed same width
    } else if (i === TOTAL_BINS - 1) {
      xCenter = BIN_EDGES[N_BIN_EDGES - 1] + BIN_WIDTH / 2; // rightmost
    } else {
      xCenter = (BIN_EDGES[i] + BIN_EDGES[i + 1]) / 2;
    }

    mesh.position.set(xCenter, 0, 0);
    mesh.scale.y = 0.0001;
    mesh.userData = { row: 0, col: i };
    scene.add(mesh);

    // Wireframe edges
    const edgeGeo = new THREE.EdgesGeometry(geo);
    const edgeMat = new THREE.LineBasicMaterial({ color: edgeColor.clone() });
    mesh.add(new THREE.LineSegments(edgeGeo, edgeMat));

    barMeshes[0].push(mesh);
  }

  // X-axis line
  const axisMat = new THREE.LineBasicMaterial({ color: 0xaaaaaa });
  const xAxisGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.1, 0, 0), new THREE.Vector3(1.15, 0, 0),
  ]);
  scene.add(new THREE.Line(xAxisGeo, axisMat));

  // Y-axis line
  const yAxisGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.05, 0, 0), new THREE.Vector3(-0.05, 0.5, 0),
  ]);
  scene.add(new THREE.Line(yAxisGeo, axisMat.clone()));

  // Dashed markers at 0 and 1
  const markerMat = new THREE.LineDashedMaterial({
    color: 0xaaaaaa, transparent: true, opacity: 0.4, dashSize: 0.02, gapSize: 0.01,
  });
  for (const xVal of [0, 1]) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(xVal, 0, 0.01), new THREE.Vector3(xVal, 0.5, 0.01),
    ]);
    const line = new THREE.Line(geo, markerMat.clone());
    line.computeLineDistances();
    scene.add(line);
  }

  // Reset camera button
  document.getElementById('resetCameraBtn').addEventListener('click', () => {
    camera.position.copy(initialCameraPosition);
    orbitControls.target.copy(initialCameraTarget);
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    orbitControls.update();
  });

  // Resize handling
  const ro = new ResizeObserver(() => {
    const w = container.offsetWidth;
    const h = container.offsetHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h);
    const asp = w / h;
    const yR = 1.4 / asp;
    camera.left = -0.2;
    camera.right = 1.2;
    camera.top = yR * 0.8;
    camera.bottom = -yR * 0.1;
    camera.updateProjectionMatrix();
  });
  ro.observe(container);

  // Render loop
  (function animate() {
    requestAnimationFrame(animate);
    orbitControls.update();
    renderer.render(scene, camera);
  })();
}

// =============================================================================
// Chart.js Charts (Panels 2 & 3)
// =============================================================================

let entropyChart, surprisalChart;

function initCharts() {
  Chart.defaults.color = '#aaaaaa';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';

  // Panel 2: Entropy over time
  entropyChart = new Chart(document.getElementById('entropy-canvas'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Model entropy',
          data: [],
          borderColor: COLORS.accent,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: 'Source entropy',
          data: [],
          borderColor: COLORS.highlight,
          borderWidth: 1.5,
          borderDash: [6, 3],
          pointRadius: 0,
          hidden: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { title: { display: true, text: 'Events' }, min: 1, max: 10 },
        y: { title: { display: true, text: 'Entropy (bits)' } },
      },
      plugins: { legend: { display: false } },
    },
  });

  // Panel 3: Surprisal scatter
  surprisalChart = new Chart(document.getElementById('surprisal-canvas'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Surprisal',
          data: [],
          backgroundColor: COLORS.accent,
          pointRadius: 2.5,
        },
        {
          label: 'Running average',
          data: [],
          type: 'line',
          borderColor: COLORS.highlight,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { title: { display: true, text: 'Events' }, min: 1, max: 10 },
        y: { title: { display: true, text: 'Surprisal (bits)' }, min: 0, max: 8 },
      },
      plugins: { legend: { display: false } },
    },
  });
}

// =============================================================================
// Update Functions
// =============================================================================

function updateHistogram() {
  const total = state.counts.reduce((a, b) => a + b, 0);
  const probs = total > 0 ? state.counts.map(c => c / total) : new Array(TOTAL_BINS).fill(0);

  for (let i = 0; i < TOTAL_BINS; i++) {
    const mesh = barMeshes[0][i];
    const h = probs[i];
    mesh.scale.y = Math.max(h, 0.0001);
    mesh.position.y = h / 2;
  }
}

function updatePdfOverlay() {
  if (pdfLine) {
    scene.remove(pdfLine);
    pdfLine = null;
  }
  if (!state.revealed) return;

  const pts = [];
  const n = 200;
  for (let i = 0; i <= n; i++) {
    const x = 0.001 + (0.998 * i) / n;
    const y = sourcePdf(state.currentSource, x) * BIN_WIDTH;
    pts.push(new THREE.Vector3(x, y, 0.02));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({
    color: parseInt(COLORS.highlight.slice(1), 16), linewidth: 2,
  });
  pdfLine = new THREE.Line(geo, mat);
  scene.add(pdfLine);
}

function updateEntropyChart() {
  const n = state.entropyHistory.length;
  const labels = Array.from({ length: n }, (_, i) => i + 1);
  entropyChart.data.labels = labels;
  entropyChart.data.datasets[0].data = state.entropyHistory;

  if (n > 0) {
    entropyChart.options.scales.x.min = 1;
    entropyChart.options.scales.x.max = Math.max(n, 10);
    const srcH = sourceEntropy(state.currentSource);
    const allH = [...state.entropyHistory, srcH];
    entropyChart.options.scales.y.min = Math.min(...allH) - 0.3;
    entropyChart.options.scales.y.max = Math.max(...allH) + 0.3;
  } else {
    entropyChart.options.scales.x.min = 1;
    entropyChart.options.scales.x.max = 10;
    delete entropyChart.options.scales.y.min;
    delete entropyChart.options.scales.y.max;
  }

  // Source entropy reference line
  const ds = entropyChart.data.datasets[1];
  if (state.revealed && n > 0) {
    const srcH = sourceEntropy(state.currentSource);
    ds.data = labels.map(() => srcH);
    ds.hidden = false;
    ds.label = `Source entropy: ${srcH.toFixed(3)} bits`;
    entropyChart.options.plugins.legend.display = true;
  } else {
    ds.data = [];
    ds.hidden = true;
    entropyChart.options.plugins.legend.display = state.revealed;
  }

  entropyChart.update('none');
}

function updateSurprisalChart() {
  const n = state.surprisalHistory.length;
  surprisalChart.data.datasets[0].data = state.surprisalHistory.map((s, i) => ({ x: i + 1, y: s }));
  surprisalChart.data.datasets[1].data = state.runningAvgSurprisal.map((a, i) => ({ x: i + 1, y: a }));

  if (n > 0) {
    surprisalChart.options.scales.x.min = 1;
    surprisalChart.options.scales.x.max = Math.max(n, 10);
    surprisalChart.options.scales.y.max = Math.max(...state.surprisalHistory, 1) * 1.2;
  } else {
    surprisalChart.options.scales.x.min = 1;
    surprisalChart.options.scales.x.max = 10;
    surprisalChart.options.scales.y.max = 8;
  }
  surprisalChart.options.scales.y.min = 0;

  surprisalChart.update('none');
}

function updateLatestEvent() {
  const el = document.getElementById('latest-event-content');
  const n = state.events.length;
  if (n === 0) {
    el.innerHTML = '<div class="event-stat">No events yet</div>';
    return;
  }
  const v = state.events[n - 1];
  const s = state.surprisalHistory[n - 1];
  const h = state.entropyHistory[n - 1];
  el.innerHTML =
    `<div class="event-stat"><span class="event-label">Value:</span> ${v.toFixed(4)}</div>` +
    `<div class="event-stat"><span class="event-label">Surprisal:</span> ${s.toFixed(2)} bits</div>` +
    `<div class="event-stat"><span class="event-label">Entropy of model:</span> ${h.toFixed(7)} bits</div>` +
    `<div class="event-stat"><span class="event-label">Events:</span> ${n}</div>`;
}

function updateAll() {
  updateHistogram();
  updateEntropyChart();
  updateSurprisalChart();
  updateLatestEvent();
  if (state.revealed) updatePdfOverlay();
}

// =============================================================================
// Core Event Loop
// =============================================================================

function addEvent(value) {
  // Compute surprisal BEFORE updating counts
  const s = eventSurprisal(value, state.counts);
  state.counts[getBin(value)] += 1;
  state.events.push(value);
  state.entropyHistory.push(modelEntropy(state.counts));
  state.surprisalHistory.push(s);
  const cumSum = state.surprisalHistory.reduce((a, b) => a + b, 0);
  state.runningAvgSurprisal.push(cumSum / state.surprisalHistory.length);
  updateAll();
}

function step() {
  if (!state.playing) return;
  addEvent(sampleSource(state.currentSource));
}

function startTimer() {
  stopTimer();
  state.timer = setInterval(step, Math.max(10, Math.round(1000 / state.speed)));
}

function stopTimer() {
  if (state.timer !== null) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

// =============================================================================
// UI Controls
// =============================================================================

function initControls() {
  const playBtn = document.getElementById('btn-play');
  playBtn.addEventListener('click', () => {
    state.playing = !state.playing;
    playBtn.textContent = state.playing ? 'Pause' : 'Play';
    if (state.playing) startTimer(); else stopTimer();
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    state.playing = false;
    playBtn.textContent = 'Play';
    stopTimer();
    state.revealed = false;
    document.getElementById('btn-reveal').textContent = 'Reveal Distribution';
    resetData();
    updatePdfOverlay();
    updateAll();
  });

  const revealBtn = document.getElementById('btn-reveal');
  revealBtn.addEventListener('click', () => {
    state.revealed = !state.revealed;
    revealBtn.textContent = state.revealed
      ? SOURCES[state.currentSource].name
      : 'Reveal Distribution';
    updatePdfOverlay();
    updateEntropyChart();
  });

  const speedSlider = document.getElementById('speed-slider');
  const speedVal = document.getElementById('speed-value');
  speedSlider.addEventListener('input', () => {
    state.speed = parseInt(speedSlider.value);
    speedVal.textContent = state.speed;
    if (state.playing) startTimer();
  });

  document.querySelectorAll('input[name="source"]').forEach(radio => {
    radio.addEventListener('change', () => {
      state.currentSource = radio.value;
      document.getElementById('btn-reset').click();
    });
  });

  const addEventBtn = document.getElementById('btn-add-event');
  const eventInput = document.getElementById('event-input');
  addEventBtn.addEventListener('click', () => {
    const val = parseFloat(eventInput.value);
    if (!isNaN(val)) {
      addEvent(val);
      eventInput.value = '';
    }
  });
  eventInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') addEventBtn.click();
  });
}

// =============================================================================
// Init
// =============================================================================

function init() {
  initHistogram();
  initCharts();
  initControls();
  updateAll();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
