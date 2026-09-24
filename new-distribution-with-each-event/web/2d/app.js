// --- Entropy Simulator (Web Version, 2D: continuous x categorical) ---
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// =============================================================================
// Constants & Binning (continuous variable X)
// =============================================================================

// N_BIN_EDGES edges produce N_BIN_EDGES-1 equal-width bins.
// Bin 0 = (-inf, edge[1])              — leftmost, open on left
// Bin i (1..N-2) = [edge[i], edge[i+1]) — interior
// Bin N-1 = [edge[N-1], +inf)          — rightmost, open on right
let N_BIN_EDGES, BIN_EDGES, TOTAL_BINS, BIN_WIDTH;

function setBinEdgeCount(n) {
  N_BIN_EDGES = n;
  BIN_EDGES = Array.from({ length: N_BIN_EDGES }, (_, i) => i / (N_BIN_EDGES - 1));
  TOTAL_BINS = N_BIN_EDGES - 1;
  BIN_WIDTH = 1.0 / (N_BIN_EDGES - 1);
}

setBinEdgeCount(21);

const COLORS = {
  accent: '#e94560',
  highlight: '#e2d810',
  marginalX: '#3fa7d6',
  marginalY: '#4caf50',
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

function categoryLabel(i) {
  return String.fromCharCode(97 + i); // 0 -> 'a', 1 -> 'b', ...
}

// =============================================================================
// Mystery Source (continuous variable X, on [0,1])
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

// =============================================================================
// Mystery Alphabet (categorical variable Y, over {a, b, c, ...})
// =============================================================================

const ALPHABETS = {
  'Alphabet A': { name: 'Uniform', type: 'uniform' },
  'Alphabet B': { name: 'Zipf (s=1)', type: 'zipf', s: 1 },
  'Alphabet C': { name: 'Zipf (s=2)', type: 'zipf', s: 2 },
  'Alphabet D': { name: 'Geometric (r=0.5)', type: 'geometric', r: 0.5 },
  'Alphabet E': { name: 'Dominant symbol (90%)', type: 'dominant', p: 0.9 },
};

function alphabetPmf(key, n) {
  const src = ALPHABETS[key];
  if (src.type === 'uniform') return new Array(n).fill(1 / n);
  if (src.type === 'zipf') {
    const raw = Array.from({ length: n }, (_, i) => 1 / Math.pow(i + 1, src.s));
    const total = raw.reduce((a, b) => a + b, 0);
    return raw.map(v => v / total);
  }
  if (src.type === 'geometric') {
    const raw = Array.from({ length: n }, (_, i) => Math.pow(src.r, i));
    const total = raw.reduce((a, b) => a + b, 0);
    return raw.map(v => v / total);
  }
  // dominant
  if (n === 1) return [1];
  const rest = (1 - src.p) / (n - 1);
  return Array.from({ length: n }, (_, i) => i === 0 ? src.p : rest);
}

function sampleAlphabet(key, n) {
  const pmf = alphabetPmf(key, n);
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < pmf.length; i++) {
    cum += pmf[i];
    if (r < cum) return i;
  }
  return pmf.length - 1;
}

// =============================================================================
// Entropy & Surprisal Math (all in bits, Laplace/add-1 smoothed)
// =============================================================================

function entropyOfCounts(counts) {
  const smoothed = counts.map(c => c + 1);
  const total = smoothed.reduce((a, b) => a + b, 0);
  let h = 0;
  for (const s of smoothed) {
    const p = s / total;
    h -= p * Math.log2(p);
  }
  return h;
}

function marginalCountsX(counts2d) {
  const nCols = counts2d[0].length;
  const marg = new Array(nCols).fill(0);
  for (const row of counts2d) for (let c = 0; c < nCols; c++) marg[c] += row[c];
  return marg;
}

function marginalCountsY(counts2d) {
  return counts2d.map(row => row.reduce((a, b) => a + b, 0));
}

function jointEventSurprisal(value, catIdx, counts2d) {
  const col = getBin(value);
  const nCols = counts2d[0].length;
  const flat = counts2d.flat();
  const smoothed = flat.map(c => c + 1);
  const total = smoothed.reduce((a, b) => a + b, 0);
  const idx = catIdx * nCols + col;
  return -Math.log2(smoothed[idx] / total);
}

// =============================================================================
// Application State
// =============================================================================

function makeCounts2d(alphabetSize) {
  return Array.from({ length: alphabetSize }, () => new Array(TOTAL_BINS).fill(0));
}

const state = {
  currentSource: 'Source A',
  currentAlphabet: 'Alphabet A',
  alphabetSize: 3,
  playing: false,
  revealed: false,
  speed: 10,
  events: [], // { value, catIdx }
  counts: makeCounts2d(3),
  jointEntropyHistory: [],
  marginalXEntropyHistory: [],
  marginalYEntropyHistory: [],
  surprisalHistory: [],
  runningAvgSurprisal: [],
  timer: null,
};

function resetData() {
  state.events = [];
  state.counts = makeCounts2d(state.alphabetSize);
  state.jointEntropyHistory = [];
  state.marginalXEntropyHistory = [];
  state.marginalYEntropyHistory = [];
  state.surprisalHistory = [];
  state.runningAvgSurprisal = [];
}

// Changing the bin count re-bins all events seen so far into the new bins.
// It does not touch the entropy/surprisal histories — those are per-event
// snapshots computed under whatever bin count was active at the time, and
// stay frozen; only events added after the change use the new bins.
function setBinCount(n) {
  setBinEdgeCount(n + 1);
  state.counts = makeCounts2d(state.alphabetSize);
  for (const { value, catIdx } of state.events) {
    state.counts[catIdx][getBin(value)] += 1;
  }
  buildHistogramGrid();
}

// =============================================================================
// Three.js Histogram (Panel 1): grid of bars over (bin, category)
// =============================================================================

let scene, camera, renderer, orbitControls;
let barMeshes = []; // barMeshes[row=category][col=bin]
let pdfLine = null;
let gridHelperObjects = [];
let initialCameraPosition, initialCameraTarget;
let cameraMode = 'perspective';
let scaffoldingVisible = true;
const ORTHO_VIEW_SIZE = 1.6;

function zStep() {
  return 1.0 / state.alphabetSize;
}

function xCenterForBin(i) {
  if (i === 0) return BIN_EDGES[0] + BIN_WIDTH / 2;
  if (i === TOTAL_BINS - 1) return BIN_EDGES[N_BIN_EDGES - 1] + BIN_WIDTH / 2;
  return (BIN_EDGES[i] + BIN_EDGES[i + 1]) / 2;
}

function zCenterForCategory(row) {
  return (row + 0.5) * zStep();
}

function makeTextSprite(text, { fontSize = 48, color = '#ccc', scale = 0.09 } = {}) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.font = `${fontSize}px sans-serif`;
  const textWidth = ctx.measureText(text).width;
  canvas.width = textWidth + 16;
  canvas.height = fontSize * 1.4;
  ctx.font = `${fontSize}px sans-serif`;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 8, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  const aspect = canvas.width / canvas.height;
  sprite.scale.set(scale * aspect, scale, 1);
  return sprite;
}

function clearHistogramGrid() {
  for (const row of barMeshes) {
    for (const mesh of row) {
      scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
  }
  barMeshes = [];
  for (const obj of gridHelperObjects) scene.remove(obj);
  gridHelperObjects = [];
}

function buildHistogramGrid() {
  clearHistogramGrid();

  const barColor = new THREE.Color(COLORS.barFill);
  const edgeColor = new THREE.Color(COLORS.barEdge);
  const barW = BIN_WIDTH * 0.9;
  const barD = zStep() * 0.9;

  for (let row = 0; row < state.alphabetSize; row++) {
    const rowMeshes = [];
    for (let col = 0; col < TOTAL_BINS; col++) {
      const geo = new THREE.BoxGeometry(barW, 1, barD);
      const mat = new THREE.MeshBasicMaterial({ color: barColor.clone() });
      const mesh = new THREE.Mesh(geo, mat);

      mesh.position.set(xCenterForBin(col), 0, zCenterForCategory(row));
      mesh.scale.y = 0.0001;
      mesh.userData = { row, col };
      scene.add(mesh);

      const edgeGeo = new THREE.EdgesGeometry(geo);
      const edgeMat = new THREE.LineBasicMaterial({ color: edgeColor.clone() });
      mesh.add(new THREE.LineSegments(edgeGeo, edgeMat));

      rowMeshes.push(mesh);
    }
    barMeshes.push(rowMeshes);
  }

  // X-axis line (continuous var)
  const axisMat = new THREE.LineBasicMaterial({ color: 0xaaaaaa });
  const xAxisGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.1, 0, 0), new THREE.Vector3(1.15, 0, 0),
  ]);
  const xAxis = new THREE.Line(xAxisGeo, axisMat);
  scene.add(xAxis);
  gridHelperObjects.push(xAxis);

  // Z-axis line (categorical var)
  const zAxisGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, -0.05), new THREE.Vector3(0, 0, 1.05),
  ]);
  const zAxis = new THREE.Line(zAxisGeo, axisMat.clone());
  scene.add(zAxis);
  gridHelperObjects.push(zAxis);

  // Y-axis line
  const yAxisGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.05, 0, -0.05), new THREE.Vector3(-0.05, 0.5, -0.05),
  ]);
  const yAxis = new THREE.Line(yAxisGeo, axisMat.clone());
  scene.add(yAxis);
  gridHelperObjects.push(yAxis);

  // Category boundary markers along z
  const markerMat = new THREE.LineDashedMaterial({
    color: 0xaaaaaa, transparent: true, opacity: 0.4, dashSize: 0.02, gapSize: 0.01,
  });
  for (let row = 0; row <= state.alphabetSize; row++) {
    const z = row * zStep();
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.05, 0, z), new THREE.Vector3(1.1, 0, z),
    ]);
    const line = new THREE.Line(geo, markerMat.clone());
    line.computeLineDistances();
    scene.add(line);
    gridHelperObjects.push(line);
  }

  // Axis name labels
  const xLabel = makeTextSprite('X (continuous)');
  xLabel.position.set(1.22, 0, 0);
  scene.add(xLabel);
  gridHelperObjects.push(xLabel);

  const yLabel = makeTextSprite('P');
  yLabel.position.set(-0.05, 0.55, -0.05);
  scene.add(yLabel);
  gridHelperObjects.push(yLabel);

  const zLabel = makeTextSprite('Y (category)');
  zLabel.position.set(0, 0, 1.15);
  scene.add(zLabel);
  gridHelperObjects.push(zLabel);

  // X-axis tick labels at 0 and 1
  for (const xVal of [0, 1]) {
    const tick = makeTextSprite(String(xVal), { fontSize: 40, scale: 0.065 });
    tick.position.set(xVal, -0.04, 0);
    scene.add(tick);
    gridHelperObjects.push(tick);
  }

  // Category letter labels along the z-axis
  for (let row = 0; row < state.alphabetSize; row++) {
    const label = makeTextSprite(categoryLabel(row), { fontSize: 40, scale: 0.065 });
    label.position.set(-0.1, 0, zCenterForCategory(row));
    scene.add(label);
    gridHelperObjects.push(label);
  }

  for (const obj of gridHelperObjects) obj.visible = scaffoldingVisible;
}

function setScaffoldingVisible(visible) {
  scaffoldingVisible = visible;
  for (const obj of gridHelperObjects) obj.visible = visible;
}

function makeCamera(mode, aspect) {
  if (mode === 'orthographic') {
    return new THREE.OrthographicCamera(
      -ORTHO_VIEW_SIZE * aspect / 2, ORTHO_VIEW_SIZE * aspect / 2,
      ORTHO_VIEW_SIZE / 2, -ORTHO_VIEW_SIZE / 2, -10, 20,
    );
  }
  return new THREE.PerspectiveCamera(45, aspect, 0.01, 20);
}

function setCameraMode(mode) {
  cameraMode = mode;
  const container = document.getElementById('histogram-container');
  const aspect = container.offsetWidth / container.offsetHeight;
  const prevPosition = camera ? camera.position.clone() : new THREE.Vector3(1.6, 1.1, 1.9);
  const prevTarget = orbitControls ? orbitControls.target.clone() : new THREE.Vector3(0.5, 0.15, 0.5);

  if (orbitControls) orbitControls.dispose();

  camera = makeCamera(mode, aspect);
  camera.position.copy(prevPosition);
  camera.lookAt(prevTarget);
  camera.updateProjectionMatrix();

  orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.rotateSpeed = 0.7;
  orbitControls.zoomSpeed = 0.8;
  orbitControls.panSpeed = 1.0;
  orbitControls.target.copy(prevTarget);
  orbitControls.update();

  const toggleBtn = document.getElementById('toggleCameraBtn');
  if (toggleBtn) toggleBtn.textContent = mode === 'perspective' ? 'Perspective' : 'Orthographic';
}

function initHistogram() {
  const container = document.getElementById('histogram-container');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.bg);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.offsetWidth, container.offsetHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.insertBefore(renderer.domElement, container.firstChild);

  setCameraMode('perspective');

  initialCameraPosition = camera.position.clone();
  initialCameraTarget = orbitControls.target.clone();

  buildHistogramGrid();

  // Reset camera button
  document.getElementById('resetCameraBtn').addEventListener('click', () => {
    camera.position.copy(initialCameraPosition);
    orbitControls.target.copy(initialCameraTarget);
    camera.updateProjectionMatrix();
    orbitControls.update();
  });

  // Perspective / orthographic toggle
  document.getElementById('toggleCameraBtn').addEventListener('click', () => {
    setCameraMode(cameraMode === 'perspective' ? 'orthographic' : 'perspective');
  });

  // Scaffolding toggle
  document.getElementById('scaffolding-toggle').addEventListener('change', (e) => {
    setScaffoldingVisible(e.target.checked);
  });

  // Resize handling
  const ro = new ResizeObserver(() => {
    const w = container.offsetWidth;
    const h = container.offsetHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h);
    const aspect = w / h;
    if (camera.isPerspectiveCamera) {
      camera.aspect = aspect;
    } else {
      camera.left = -ORTHO_VIEW_SIZE * aspect / 2;
      camera.right = ORTHO_VIEW_SIZE * aspect / 2;
      camera.top = ORTHO_VIEW_SIZE / 2;
      camera.bottom = -ORTHO_VIEW_SIZE / 2;
    }
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

  // Panel 2: Entropy over time — joint + both marginals
  entropyChart = new Chart(document.getElementById('entropy-canvas'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Joint entropy H(X,Y)',
          data: [],
          borderColor: COLORS.accent,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: 'Marginal H(X)',
          data: [],
          borderColor: COLORS.marginalX,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: 'Marginal H(Y)',
          data: [],
          borderColor: COLORS.marginalY,
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
        y: { title: { display: true, text: 'Entropy (bits)' } },
      },
      plugins: { legend: { display: true, labels: { boxWidth: 12, font: { size: 9 } } } },
    },
  });

  // Panel 3: Surprisal scatter (joint)
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
        y: { title: { display: true, text: 'Joint surprisal (bits)' }, min: 0, max: 8 },
      },
      plugins: { legend: { display: true, labels: { boxWidth: 12, font: { size: 9 } } } },
    },
  });
}

// =============================================================================
// Update Functions
// =============================================================================

function updateHistogram() {
  const flat = state.counts.flat();
  const total = flat.reduce((a, b) => a + b, 0);

  for (let row = 0; row < state.alphabetSize; row++) {
    for (let col = 0; col < TOTAL_BINS; col++) {
      const h = total > 0 ? state.counts[row][col] / total : 0;
      const mesh = barMeshes[row][col];
      mesh.scale.y = Math.max(h, 0.0001);
      mesh.position.y = h / 2;
    }
  }
}

function updatePdfOverlay() {
  if (pdfLine) {
    scene.remove(pdfLine);
    pdfLine = null;
  }
  if (!state.revealed) return;

  // Draw the continuous PDF as a curtain at each category's z-slot, scaled by that category's probability.
  const pmf = alphabetPmf(state.currentAlphabet, state.alphabetSize);
  const group = new THREE.Group();
  const n = 200;
  for (let row = 0; row < state.alphabetSize; row++) {
    const z = zCenterForCategory(row);
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const x = 0.001 + (0.998 * i) / n;
      const y = sourcePdf(state.currentSource, x) * BIN_WIDTH * pmf[row];
      pts.push(new THREE.Vector3(x, y, z));
    }
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({
      color: parseInt(COLORS.highlight.slice(1), 16), linewidth: 2,
    });
    group.add(new THREE.Line(geo, mat));
  }
  pdfLine = group;
  scene.add(pdfLine);
}

function updateEntropyChart() {
  const n = state.jointEntropyHistory.length;
  const labels = Array.from({ length: n }, (_, i) => i + 1);
  entropyChart.data.labels = labels;
  entropyChart.data.datasets[0].data = state.jointEntropyHistory;
  entropyChart.data.datasets[1].data = state.marginalXEntropyHistory;
  entropyChart.data.datasets[2].data = state.marginalYEntropyHistory;

  if (n > 0) {
    entropyChart.options.scales.x.min = 1;
    entropyChart.options.scales.x.max = Math.max(n, 10);
    const allH = [
      ...state.jointEntropyHistory,
      ...state.marginalXEntropyHistory,
      ...state.marginalYEntropyHistory,
    ];
    entropyChart.options.scales.y.min = Math.min(...allH) - 0.3;
    entropyChart.options.scales.y.max = Math.max(...allH) + 0.3;
  } else {
    entropyChart.options.scales.x.min = 1;
    entropyChart.options.scales.x.max = 10;
    delete entropyChart.options.scales.y.min;
    delete entropyChart.options.scales.y.max;
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
  const { value, catIdx } = state.events[n - 1];
  const s = state.surprisalHistory[n - 1];
  const hJoint = state.jointEntropyHistory[n - 1];
  const hX = state.marginalXEntropyHistory[n - 1];
  const hY = state.marginalYEntropyHistory[n - 1];
  el.innerHTML =
    `<div class="event-stat"><span class="event-label">X:</span> ${value.toFixed(4)}</div>` +
    `<div class="event-stat"><span class="event-label">Y:</span> ${categoryLabel(catIdx)}</div>` +
    `<div class="event-stat"><span class="event-label">Joint surprisal:</span> ${s.toFixed(2)} bits</div>` +
    `<div class="event-stat"><span class="event-label">H(X,Y):</span> ${hJoint.toFixed(4)} bits &nbsp;` +
    `H(X): ${hX.toFixed(4)} &nbsp; H(Y): ${hY.toFixed(4)}</div>` +
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

function addEvent(value, catIdx) {
  const s = jointEventSurprisal(value, catIdx, state.counts);
  state.counts[catIdx][getBin(value)] += 1;
  state.events.push({ value, catIdx });

  state.jointEntropyHistory.push(entropyOfCounts(state.counts.flat()));
  state.marginalXEntropyHistory.push(entropyOfCounts(marginalCountsX(state.counts)));
  state.marginalYEntropyHistory.push(entropyOfCounts(marginalCountsY(state.counts)));

  state.surprisalHistory.push(s);
  const cumSum = state.surprisalHistory.reduce((a, b) => a + b, 0);
  state.runningAvgSurprisal.push(cumSum / state.surprisalHistory.length);
  updateAll();
}

function step() {
  if (!state.playing) return;
  addEvent(sampleSource(state.currentSource), sampleAlphabet(state.currentAlphabet, state.alphabetSize));
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

function rebuildCategorySelect() {
  const select = document.getElementById('event-category-select');
  select.innerHTML = '';
  for (let i = 0; i < state.alphabetSize; i++) {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = categoryLabel(i);
    select.appendChild(opt);
  }
}

function initControls() {
  const playBtn = document.getElementById('btn-play');
  playBtn.addEventListener('click', () => {
    state.playing = !state.playing;
    playBtn.textContent = state.playing ? 'Pause' : 'Play';
    if (state.playing) startTimer(); else stopTimer();
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    state.revealed = false;
    document.getElementById('btn-reveal').textContent = 'Reveal Distributions';
    resetData();
    buildHistogramGrid();
    updatePdfOverlay();
    updateAll();
  });

  const revealBtn = document.getElementById('btn-reveal');
  revealBtn.addEventListener('click', () => {
    state.revealed = !state.revealed;
    revealBtn.textContent = state.revealed
      ? `${SOURCES[state.currentSource].name} × ${ALPHABETS[state.currentAlphabet].name}`
      : 'Reveal Distributions';
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

  const binCountSlider = document.getElementById('bin-count-slider');
  const binCountVal = document.getElementById('bin-count-value');
  binCountSlider.addEventListener('input', () => {
    const n = parseInt(binCountSlider.value);
    binCountVal.textContent = n;
    setBinCount(n);
    updateAll();
  });

  document.querySelectorAll('input[name="source"]').forEach(radio => {
    radio.addEventListener('change', () => {
      state.currentSource = radio.value;
      if (state.revealed) {
        revealBtn.textContent = `${SOURCES[state.currentSource].name} × ${ALPHABETS[state.currentAlphabet].name}`;
      }
      updatePdfOverlay();
      updateEntropyChart();
    });
  });

  document.querySelectorAll('input[name="alphabet"]').forEach(radio => {
    radio.addEventListener('change', () => {
      state.currentAlphabet = radio.value;
      if (state.revealed) {
        revealBtn.textContent = `${SOURCES[state.currentSource].name} × ${ALPHABETS[state.currentAlphabet].name}`;
      }
      updatePdfOverlay();
      updateEntropyChart();
    });
  });

  const alphabetSizeSlider = document.getElementById('alphabet-size-slider');
  const alphabetSizeVal = document.getElementById('alphabet-size-value');
  alphabetSizeSlider.addEventListener('input', () => {
    state.alphabetSize = parseInt(alphabetSizeSlider.value);
    alphabetSizeVal.textContent = state.alphabetSize;
    resetData();
    buildHistogramGrid();
    rebuildCategorySelect();
    updatePdfOverlay();
    updateAll();
  });

  const addEventBtn = document.getElementById('btn-add-event');
  const eventInput = document.getElementById('event-input');
  const categorySelect = document.getElementById('event-category-select');
  addEventBtn.addEventListener('click', () => {
    const val = parseFloat(eventInput.value);
    const catIdx = parseInt(categorySelect.value);
    if (!isNaN(val) && !isNaN(catIdx)) {
      addEvent(val, catIdx);
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
  rebuildCategorySelect();
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
