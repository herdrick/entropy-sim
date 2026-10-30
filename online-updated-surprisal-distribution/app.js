// --- Entropy Simulator (Web Version, N-D: any number of continuous/categorical vars) ---
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// =============================================================================
// Mystery Sources (continuous variables) & Alphabets (categorical variables)
// =============================================================================

const SOURCES = {
  'Source A': { name: 'Uniform(0,1)', glyph: '●', type: 'uniform' },
  'Source B': { name: 'Beta(2,5)', glyph: '▲', alpha: 2, beta: 5, type: 'beta' },
  'Source C': { name: 'Beta(0.5,0.5)', glyph: '■', alpha: 0.5, beta: 0.5, type: 'beta' },
  'Source D': { name: 'Beta(5,5)', glyph: '◆', alpha: 5, beta: 5, type: 'beta' },
  'Source E': { name: 'Beta(0.3,0.3)', glyph: '★', alpha: 0.3, beta: 0.3, type: 'beta' },
  'Source F': {
    name: 'Mixture: 0.5*Beta(2,2)+0.5*Beta(20,20)', glyph: '✚', type: 'mixture',
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

const ALPHABETS = {
  'Alphabet A': { name: 'Uniform', glyph: '●', type: 'uniform' },
  'Alphabet B': { name: 'Zipf (s=1)', glyph: '▲', type: 'zipf', s: 1 },
  'Alphabet C': { name: 'Zipf (s=2)', glyph: '■', type: 'zipf', s: 2 },
  'Alphabet D': { name: 'Geometric (r=0.5)', glyph: '◆', type: 'geometric', r: 0.5 },
  'Alphabet E': { name: 'Dominant symbol (90%)', glyph: '★', type: 'dominant', p: 0.9 },
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

function categoryLabel(i) {
  return String.fromCharCode(97 + i); // 0 -> 'a', 1 -> 'b', ...
}

// =============================================================================
// Continuous binning — each continuous variable owns its own bin config,
// since different variables can have independently-chosen bin counts.
// =============================================================================

// N_EDGES edges produce N_EDGES-1 equal-width bins.
// Bin 0 = (-inf, edge[1])              — leftmost, open on left
// Bin i (1..N-2) = [edge[i], edge[i+1]) — interior
// Bin N-1 = [edge[N-1], +inf)          — rightmost, open on right
function makeBinConfig(totalBins) {
  const nEdges = totalBins + 1;
  const edges = Array.from({ length: nEdges }, (_, i) => i / (nEdges - 1));
  return { nEdges, edges, totalBins, width: 1 / totalBins };
}

function getBin(binCfg, value) {
  const { edges, totalBins, nEdges } = binCfg;
  if (value < edges[1]) return 0;
  if (value >= edges[nEdges - 1]) return totalBins - 1;
  let lo = 1, hi = nEdges - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (edges[mid] <= value) lo = mid; else hi = mid - 1;
  }
  return lo;
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

// Surprisal of a single value against a variable's own (smoothed) counts,
// evaluated BEFORE that count is incremented for the new event.
function marginalSurprisal(counts, idx) {
  const smoothed = counts.map(c => c + 1);
  const total = smoothed.reduce((a, b) => a + b, 0);
  return -Math.log2(smoothed[idx] / total);
}

// =============================================================================
// Moments Math (continuous variables only — a categorical alphabet is
// unordered, so statistical moments aren't a meaningful summary of it)
// =============================================================================

const MOMENT_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];

function snapFloatNoise(x) {
  return x === 0 ? 0 : Number(x.toPrecision(9));
}

// Padding the span by 10% before rounding guarantees visible headroom above/below the
// data even when a value lands exactly on what would otherwise become the axis edge
// (e.g. a line pinned at a trivial constant like 1) — without this, such a line would
// hug the plot border instead of sitting visibly inside it.
const AXIS_PADDING_FRACTION = 0.1;

function niceAxisBounds(rawMin, rawMax) {
  let min = snapFloatNoise(rawMin);
  let max = snapFloatNoise(rawMax);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * AXIS_PADDING_FRACTION;
  min -= pad;
  max += pad;
  const roughStep = (max - min) / 5;
  const exponent = Math.floor(Math.log10(roughStep));
  const fraction = roughStep / Math.pow(10, exponent);
  const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  const step = niceFraction * Math.pow(10, exponent);
  return { min: Math.floor(min / step) * step, max: Math.ceil(max / step) * step };
}

function ordinal(n) {
  if (n % 10 === 1 && n % 100 !== 11) return `${n}st`;
  if (n % 10 === 2 && n % 100 !== 12) return `${n}nd`;
  if (n % 10 === 3 && n % 100 !== 13) return `${n}rd`;
  return `${n}th`;
}

function momentLabel(type, order) {
  if (order === 0) return '0th moment';
  if (type === 'raw') return order === 1 ? 'Mean' : `${ordinal(order)} moment`;
  if (type === 'central') {
    if (order === 1) return `${ordinal(order)} moment (≡ 0)`;
    if (order === 2) return 'Variance';
    return `${ordinal(order)} moment`;
  }
  // standardized
  if (order === 1) return `${ordinal(order)} moment (≡ 0)`;
  if (order === 2) return `${ordinal(order)} moment (≡ 1)`;
  if (order === 3) return 'Skewness';
  if (order === 4) return 'Kurtosis';
  return `${ordinal(order)} moment`;
}

function binomial(n, k) {
  let r = 1;
  for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1);
  return r;
}

function centralMoment(powerSums, n, mean, j) {
  let m = 0;
  for (let k = 0; k <= j; k++) {
    const sk = k === 0 ? n : powerSums[k];
    m += binomial(j, k) * Math.pow(-mean, j - k) * sk;
  }
  return m / n;
}

function computeMoment(powerSums, n, j) {
  if (j === 0) return { raw: 1, central: 1, standardized: 1 };
  const mean = powerSums[1] / n;
  const raw = powerSums[j] / n;
  const central = centralMoment(powerSums, n, mean, j);
  const variance = j === 2 ? central : centralMoment(powerSums, n, mean, 2);
  const sigma = Math.sqrt(Math.max(variance, 0));
  const standardized = sigma > 0 ? central / Math.pow(sigma, j) : 0;
  return { raw, central, standardized };
}

// Recomputes the full history (all three types) for moment order `order` by replaying
// `values` from scratch — used only when the slider requests an order for the first time.
// Returns the running power sums too, so the caller can resume incremental updates from
// the right total instead of restarting at 0 — they live in the same object as the
// history they were computed from, so there's nothing else to keep in sync.
function backfillMomentOrder(order, values) {
  const history = { raw: [], central: [], standardized: [], powerSums: new Array(order + 1).fill(0) };
  for (let i = 0; i < values.length; i++) {
    for (let k = 1; k <= order; k++) history.powerSums[k] += Math.pow(values[i], k);
    const { raw, central, standardized } = computeMoment(history.powerSums, i + 1, order);
    history.raw.push(raw);
    history.central.push(central);
    history.standardized.push(standardized);
  }
  return history;
}

// =============================================================================
// Application State
// =============================================================================

const COLORS = {
  accent: '#e94560',
  highlight: '#e2d810',
  bg: '#1a1a2e',
  barFill: 0x0f3460,
  barEdge: 0xe94560,
};

// Bars (and their edges) render at this opacity, user-adjustable via the opacity slider.
const DEFAULT_BAR_OPACITY = 0.25;

let nextVarId = 1;

function createVariable(kind) {
  const v = {
    id: nextVarId++,
    kind, // 'continuous' | 'categorical'
    entropyHistory: new Array(state.events.length).fill(null),
  };
  if (kind === 'continuous') {
    v.source = 'Source A';
    v.binCfg = makeBinConfig(7);
    v.counts = new Array(7).fill(0);
    v.momentsCache = {}; // { order: { raw, central, standardized, powerSums } }
  } else {
    v.alphabet = 'Alphabet A';
    v.alphabetSize = 3;
    v.counts = new Array(3).fill(0);
  }
  v.label = `Var ${v.id}`;
  return v;
}

const state = {
  variables: [],
  events: [], // array of { [varId]: value } — value is a float for continuous, a category index for categorical
  playing: false,
  sourcesHidden: true,
  barOpacity: DEFAULT_BAR_OPACITY,
  speed: 10,
  surprisalHistory: [],
  runningAvgSurprisal: [],
  histoVarA: null,
  histoVarB: null,
  histoCounts2d: [[0]], // [row][col], row = axis B, col = axis A
  momentVarId: null,
  maxMomentOrder: 4,
  momentType: 'standardized',
  logScale: false,
  timer: null,
};

function findVariable(id) {
  return state.variables.find(v => v.id === id) || null;
}

function continuousVariables() {
  return state.variables.filter(v => v.kind === 'continuous');
}

// Values this variable has seen, in order — only defined for events from the
// point this variable was added onward, since earlier events predate it.
function ownValues(v) {
  return state.events.filter(e => v.id in e).map(e => e[v.id]);
}

function resetData() {
  state.events = [];
  state.surprisalHistory = [];
  state.runningAvgSurprisal = [];
  for (const v of state.variables) {
    v.entropyHistory = [];
    v.counts = v.kind === 'continuous' ? new Array(v.binCfg.totalBins).fill(0) : new Array(v.alphabetSize).fill(0);
    if (v.kind === 'continuous') v.momentsCache = {};
  }
  rebuildHistoGrid();
}

// =============================================================================
// Moments cache (per continuous variable)
// =============================================================================

function ensureMomentsTrackedUpTo(v, order) {
  let missing = false;
  for (let k = 0; k <= order; k++) {
    if (!(k in v.momentsCache)) { missing = true; break; }
  }
  if (!missing) return;

  const values = ownValues(v);
  for (let k = 0; k <= order; k++) {
    if (k in v.momentsCache) continue;
    v.momentsCache[k] = backfillMomentOrder(k, values);
  }
}

// =============================================================================
// Histogram pair axis descriptors
// =============================================================================

// Describes how to lay a variable's values out along one axis of the 3D grid,
// in [0,1] either way (continuous bin edges, or evenly-spaced category slots) —
// so a continuous-continuous, continuous-categorical, or categorical-categorical
// pair all share the exact same grid-building code.
function axisDescriptor(v) {
  if (!v) {
    return { size: 1, posFor: () => 0.5, barSize: 0.9, ticks: [], name: '(none)' };
  }
  if (v.kind === 'continuous') {
    const { edges, totalBins, width } = v.binCfg;
    return {
      size: totalBins,
      posFor: (i) => (edges[i] + edges[i + 1]) / 2,
      barSize: width * 0.9,
      ticks: [{ pos: 0, label: '0' }, { pos: 1, label: '1' }],
      name: `${v.label} (continuous)`,
      indexFor: (value) => getBin(v.binCfg, value),
    };
  }
  const size = v.alphabetSize;
  const step = 1 / size;
  return {
    size,
    posFor: (i) => (i + 0.5) * step,
    barSize: step * 0.9,
    ticks: Array.from({ length: size }, (_, i) => ({ pos: (i + 0.5) * step, label: categoryLabel(i) })),
    name: `${v.label} (categorical)`,
    indexFor: (value) => value,
  };
}

function rebuildHistoGrid() {
  const varA = findVariable(state.histoVarA);
  const varB = findVariable(state.histoVarB);
  const axisA = axisDescriptor(varA);
  const axisB = axisDescriptor(varB);

  state.histoCounts2d = Array.from({ length: axisB.size }, () => new Array(axisA.size).fill(0));
  if (varA) {
    for (const e of state.events) {
      if (!(varA.id in e)) continue;
      if (varB && !(varB.id in e)) continue;
      const col = axisA.indexFor(e[varA.id]);
      const row = varB ? axisB.indexFor(e[varB.id]) : 0;
      state.histoCounts2d[row][col] += 1;
    }
  }
  buildHistogramGrid();
}

// =============================================================================
// Three.js Histogram (Panel 1): grid of bars over (axis A, axis B)
// =============================================================================

let scene, camera, renderer, orbitControls;
let barMeshes = []; // barMeshes[row][col]
let pdfLine = null;
let gridHelperObjects = [];
let initialCameraPosition, initialCameraTarget;
let cameraMode = 'perspective';
let scaffoldingVisible = true;
const ORTHO_VIEW_SIZE = 1.6;

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

  const varA = findVariable(state.histoVarA);
  const varB = findVariable(state.histoVarB);
  const axisA = axisDescriptor(varA);
  const axisB = axisDescriptor(varB);

  const barColor = new THREE.Color(COLORS.barFill);
  const edgeColor = new THREE.Color(COLORS.barEdge);

  for (let row = 0; row < axisB.size; row++) {
    const rowMeshes = [];
    for (let col = 0; col < axisA.size; col++) {
      const geo = new THREE.BoxGeometry(axisA.barSize, 1, axisB.barSize);
      const mat = new THREE.MeshBasicMaterial({
        color: barColor.clone(),
        transparent: true,
        opacity: state.barOpacity,
      });
      const mesh = new THREE.Mesh(geo, mat);

      mesh.position.set(axisA.posFor(col), 0, axisB.posFor(row));
      mesh.scale.y = 0.0001;
      mesh.userData = { row, col };
      scene.add(mesh);

      const edgeGeo = new THREE.EdgesGeometry(geo);
      const edgeMat = new THREE.LineBasicMaterial({
        color: edgeColor.clone(),
        transparent: true,
        opacity: state.barOpacity,
      });
      const edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
      mesh.add(edgeLines);
      mesh.userData.edgeLines = edgeLines;

      rowMeshes.push(mesh);
    }
    barMeshes.push(rowMeshes);
  }

  const axisMat = new THREE.LineBasicMaterial({ color: 0xaaaaaa });
  const xAxisGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.1, 0, 0), new THREE.Vector3(1.15, 0, 0),
  ]);
  const xAxis = new THREE.Line(xAxisGeo, axisMat);
  scene.add(xAxis);
  gridHelperObjects.push(xAxis);

  if (varB) {
    const zAxisGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, -0.05), new THREE.Vector3(0, 0, 1.05),
    ]);
    const zAxis = new THREE.Line(zAxisGeo, axisMat.clone());
    scene.add(zAxis);
    gridHelperObjects.push(zAxis);
  }

  const yAxisGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.05, 0, -0.05), new THREE.Vector3(-0.05, 0.5, -0.05),
  ]);
  const yAxis = new THREE.Line(yAxisGeo, axisMat.clone());
  scene.add(yAxis);
  gridHelperObjects.push(yAxis);

  // Row-boundary markers along the depth axis (only meaningful with a second variable)
  if (varB) {
    const markerMat = new THREE.LineDashedMaterial({
      color: 0xaaaaaa, transparent: true, opacity: 0.4, dashSize: 0.02, gapSize: 0.01,
    });
    for (let row = 0; row <= axisB.size; row++) {
      const z = row / axisB.size;
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-0.05, 0, z), new THREE.Vector3(1.1, 0, z),
      ]);
      const line = new THREE.Line(geo, markerMat.clone());
      line.computeLineDistances();
      scene.add(line);
      gridHelperObjects.push(line);
    }
  }

  const xLabel = makeTextSprite(axisA.name);
  xLabel.position.set(1.22, 0, 0);
  scene.add(xLabel);
  gridHelperObjects.push(xLabel);

  const yLabel = makeTextSprite('ℙ');
  yLabel.position.set(-0.05, 0.55, -0.05);
  scene.add(yLabel);
  gridHelperObjects.push(yLabel);

  if (varB) {
    const zLabel = makeTextSprite(axisB.name);
    zLabel.position.set(0, 0, 1.15);
    scene.add(zLabel);
    gridHelperObjects.push(zLabel);
  }

  for (const tick of axisA.ticks) {
    const label = makeTextSprite(tick.label, { fontSize: 40, scale: 0.065 });
    label.position.set(tick.pos, -0.04, 0);
    scene.add(label);
    gridHelperObjects.push(label);
  }

  if (varB) {
    for (const tick of axisB.ticks) {
      const label = makeTextSprite(tick.label, { fontSize: 40, scale: 0.065 });
      label.position.set(-0.1, 0, tick.pos);
      scene.add(label);
      gridHelperObjects.push(label);
    }
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

  const perspBtn = document.getElementById('perspectiveCameraBtn');
  const orthoBtn = document.getElementById('orthographicCameraBtn');
  if (perspBtn) perspBtn.classList.toggle('active', mode === 'perspective');
  if (orthoBtn) orthoBtn.classList.toggle('active', mode === 'orthographic');
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

  rebuildHistoGrid();

  document.getElementById('resetCameraBtn').addEventListener('click', () => {
    camera.position.copy(initialCameraPosition);
    orbitControls.target.copy(initialCameraTarget);
    camera.updateProjectionMatrix();
    orbitControls.update();
  });

  document.getElementById('perspectiveCameraBtn').addEventListener('click', () => setCameraMode('perspective'));
  document.getElementById('orthographicCameraBtn').addEventListener('click', () => setCameraMode('orthographic'));

  document.getElementById('scaffolding-toggle').addEventListener('change', (e) => {
    setScaffoldingVisible(e.target.checked);
  });

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

  (function animate() {
    requestAnimationFrame(animate);
    orbitControls.update();
    renderer.render(scene, camera);
  })();
}

// =============================================================================
// Chart.js Charts (Panels 2, 3 & 4)
// =============================================================================

let entropyChart, surprisalChart, momentsChart;

function initCharts() {
  Chart.defaults.color = '#aaaaaa';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';

  entropyChart = new Chart(document.getElementById('entropy-canvas'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { title: { display: true, text: 'Events' }, min: 1, max: 10 },
        y: { title: { display: true, text: 'Entropy (bits)' } },
      },
      plugins: {
        legend: { display: true, labels: { usePointStyle: true, pointStyle: 'line', boxWidth: 20, font: { size: 9 } } },
      },
    },
  });

  surprisalChart = new Chart(document.getElementById('surprisal-canvas'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Surprisal',
          data: [],
          backgroundColor: COLORS.accent,
          borderColor: COLORS.accent,
          pointStyle: 'circle',
          pointRadius: 2.5,
        },
        {
          label: 'Running average',
          data: [],
          type: 'line',
          borderColor: COLORS.highlight,
          borderWidth: 1.5,
          pointStyle: 'line',
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
      plugins: {
        legend: { display: true, labels: { usePointStyle: true, boxWidth: 12, font: { size: 9 } } },
      },
    },
  });

  momentsChart = new Chart(document.getElementById('moments-canvas'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { title: { display: true, text: 'Events' }, min: 1, max: 10 },
        y: { title: { display: true, text: 'Value' }, ticks: {} },
      },
      plugins: {
        legend: { display: true, labels: { usePointStyle: true, boxWidth: 12, font: { size: 9 } } },
      },
    },
  });
}

function buildMomentDataset(v, order) {
  return {
    label: momentLabel(state.momentType, order),
    data: v.momentsCache[order][state.momentType],
    borderColor: MOMENT_COLORS[order % MOMENT_COLORS.length],
    borderWidth: 1.5,
    pointRadius: 0,
    pointStyle: 'line',
    tension: 0,
  };
}

// =============================================================================
// Update Functions
// =============================================================================

function updateHistogram() {
  const flat = state.histoCounts2d.flat();
  const total = flat.reduce((a, b) => a + b, 0);

  for (let row = 0; row < barMeshes.length; row++) {
    for (let col = 0; col < barMeshes[row].length; col++) {
      const h = total > 0 ? state.histoCounts2d[row][col] / total : 0;
      const mesh = barMeshes[row][col];
      mesh.scale.y = Math.max(h, 0.0001);
      mesh.position.y = h / 2;
    }
  }

  document.getElementById('event-count-badge').textContent = `Events: ${state.events.length}`;
}

function updateBarOpacity() {
  for (const row of barMeshes) {
    for (const mesh of row) {
      mesh.material.opacity = state.barOpacity;
      mesh.userData.edgeLines.material.opacity = state.barOpacity;
    }
  }
}

function updatePdfOverlay() {
  if (pdfLine) {
    scene.remove(pdfLine);
    pdfLine = null;
  }
  if (state.sourcesHidden) return;

  const varA = findVariable(state.histoVarA);
  const varB = findVariable(state.histoVarB);
  if (!varA || varA.kind !== 'continuous') return;

  const group = new THREE.Group();
  const n = 200;
  const axisB = axisDescriptor(varB);
  const rowWeights = (varB && varB.kind === 'categorical')
    ? alphabetPmf(varB.alphabet, varB.alphabetSize)
    : [1];
  const rows = varB ? (varB.kind === 'categorical' ? varB.alphabetSize : 1) : 1;

  for (let row = 0; row < rows; row++) {
    // A continuous varB has no closed-form PMF per bin to weight by, so the
    // overlay only renders a meaningful curtain per row when varB is categorical
    // (or absent) — for continuous x continuous pairs it draws a single curve
    // at row 0, under the assumption the two are independent.
    const z = varB ? axisB.posFor(row) : 0.5;
    const weight = varB && varB.kind === 'categorical' ? rowWeights[row] : 1;
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const x = 0.001 + (0.998 * i) / n;
      const y = sourcePdf(varA.source, x) * varA.binCfg.width * weight;
      pts.push(new THREE.Vector3(x, y, z));
    }
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({
      color: parseInt(COLORS.highlight.slice(1), 16), linewidth: 2,
      transparent: true, opacity: 0.8,
    });
    group.add(new THREE.Line(geo, mat));
    if (rows === 1) break;
  }
  pdfLine = group;
  scene.add(pdfLine);
}

function updateEntropyChart() {
  const n = state.events.length;
  const labels = Array.from({ length: n }, (_, i) => i + 1);
  entropyChart.data.labels = labels;
  entropyChart.data.datasets = state.variables.map((v, i) => ({
    label: `H(${v.label})`,
    data: v.entropyHistory,
    borderColor: MOMENT_COLORS[i % MOMENT_COLORS.length],
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0,
  }));

  if (n > 0) {
    entropyChart.options.scales.x.min = 1;
    entropyChart.options.scales.x.max = Math.max(n, 10);
    const allH = state.variables.flatMap(v => v.entropyHistory.filter(x => x !== null));
    if (allH.length > 0) {
      entropyChart.options.scales.y.min = Math.min(...allH) - 0.3;
      entropyChart.options.scales.y.max = Math.max(...allH) + 0.3;
    }
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

function updateMomentsChart() {
  const v = findVariable(state.momentVarId);
  if (!v) {
    momentsChart.data.labels = [];
    momentsChart.data.datasets = [];
    momentsChart.update('none');
    return;
  }

  ensureMomentsTrackedUpTo(v, state.maxMomentOrder);

  const n = ownValues(v).length;
  const labels = Array.from({ length: n }, (_, i) => i + 1);
  momentsChart.data.labels = labels;
  momentsChart.data.datasets = Array.from({ length: state.maxMomentOrder + 1 }, (_, i) => buildMomentDataset(v, i));
  const visibleValues = momentsChart.data.datasets.flatMap(ds => ds.data);

  momentsChart.options.scales.y.type = state.logScale ? 'logarithmic' : 'linear';
  momentsChart.options.scales.y.ticks.callback = state.logScale
    ? (value) => Number(value).toExponential(0)
    : undefined;

  if (n > 0) {
    momentsChart.options.scales.x.min = 1;
    momentsChart.options.scales.x.max = Math.max(n, 10);
    if (state.logScale) {
      delete momentsChart.options.scales.y.min;
      delete momentsChart.options.scales.y.max;
    } else {
      const bounds = niceAxisBounds(Math.min(...visibleValues), Math.max(...visibleValues));
      momentsChart.options.scales.y.min = bounds.min;
      momentsChart.options.scales.y.max = bounds.max;
    }
  } else {
    momentsChart.options.scales.x.min = 1;
    momentsChart.options.scales.x.max = 10;
    delete momentsChart.options.scales.y.min;
    delete momentsChart.options.scales.y.max;
  }

  momentsChart.update('none');
}

function updateAll() {
  updateHistogram();
  updateEntropyChart();
  updateSurprisalChart();
  updateMomentsChart();
  if (!state.sourcesHidden) updatePdfOverlay();
}

// =============================================================================
// Core Event Loop
// =============================================================================

// valuesByVarId: { [varId]: value } — a float for continuous vars, a category
// index for categorical vars. Must include a value for every current variable.
function addEvent(valuesByVarId) {
  state.events.push(valuesByVarId);

  let jointSurprisal = 0;
  for (const v of state.variables) {
    const val = valuesByVarId[v.id];
    const idx = v.kind === 'continuous' ? getBin(v.binCfg, val) : val;
    jointSurprisal += marginalSurprisal(v.counts, idx);
    v.counts[idx] += 1;
    v.entropyHistory.push(entropyOfCounts(v.counts));

    if (v.kind === 'continuous') {
      const n = ownValues(v).length;
      for (const key of Object.keys(v.momentsCache)) {
        const k = Number(key);
        const cache = v.momentsCache[k];
        for (let m = 1; m <= k; m++) cache.powerSums[m] += Math.pow(val, m);
        const { raw, central, standardized } = computeMoment(cache.powerSums, n, k);
        cache.raw.push(raw);
        cache.central.push(central);
        cache.standardized.push(standardized);
      }
    }
  }

  state.surprisalHistory.push(jointSurprisal);
  const cumSum = state.surprisalHistory.reduce((a, b) => a + b, 0);
  state.runningAvgSurprisal.push(cumSum / state.surprisalHistory.length);

  const varA = findVariable(state.histoVarA);
  const varB = findVariable(state.histoVarB);
  if (varA && varA.id in valuesByVarId && (!varB || varB.id in valuesByVarId)) {
    const axisA = axisDescriptor(varA);
    const axisB = axisDescriptor(varB);
    const col = axisA.indexFor(valuesByVarId[varA.id]);
    const row = varB ? axisB.indexFor(valuesByVarId[varB.id]) : 0;
    state.histoCounts2d[row][col] += 1;
  }

  updateAll();
}

function sampleAllVariables() {
  const values = {};
  for (const v of state.variables) {
    values[v.id] = v.kind === 'continuous' ? sampleSource(v.source) : sampleAlphabet(v.alphabet, v.alphabetSize);
  }
  return values;
}

function step() {
  if (!state.playing) return;
  addEvent(sampleAllVariables());
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
// UI: Variable cards, histogram-pair selectors, moments selector, manual event
// =============================================================================

function buildDistributionRadios(container, name, options, checkedKey, onChange) {
  container.innerHTML = '';
  for (const key of Object.keys(options)) {
    const label = document.createElement('label');
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = name;
    radio.value = key;
    if (key === checkedKey) radio.checked = true;
    radio.addEventListener('change', () => onChange(key));
    const text = document.createElement('span');
    text.className = 'dist-label';
    text.dataset.key = key;
    label.appendChild(radio);
    label.appendChild(text);
    container.appendChild(label);
  }
  renderDistributionLabels(container, options);
}

function renderDistributionLabels(container, options) {
  container.querySelectorAll('.dist-label').forEach(span => {
    const opt = options[span.dataset.key];
    span.textContent = state.sourcesHidden ? opt.glyph : opt.name;
    span.classList.toggle('dist-glyph', state.sourcesHidden);
  });
}

function renderAllDistributionLabels() {
  document.querySelectorAll('.source-group').forEach(container => {
    const options = container.dataset.kind === 'continuous' ? SOURCES : ALPHABETS;
    renderDistributionLabels(container, options);
  });
}

// Rebuilds everything whose *options* depend on the current variable list
// (histogram-pair dropdowns, moments-variable dropdown, manual-event inputs).
// The variable cards themselves are rebuilt separately by renderVariableList.
function refreshVariableDependentUI() {
  renderHistoSelectors();
  renderMomentVarSelector();
  renderManualEventInputs();
}

function renderHistoSelectors() {
  const selA = document.getElementById('histo-var-a');
  const selB = document.getElementById('histo-var-b');

  if (state.histoVarA !== null && !findVariable(state.histoVarA)) state.histoVarA = null;
  if (state.histoVarB !== null && !findVariable(state.histoVarB)) state.histoVarB = null;

  for (const [sel, selected] of [[selA, state.histoVarA], [selB, state.histoVarB]]) {
    sel.innerHTML = '';
    const noneOpt = document.createElement('option');
    noneOpt.value = '';
    noneOpt.textContent = '(none)';
    sel.appendChild(noneOpt);
    for (const v of state.variables) {
      const opt = document.createElement('option');
      opt.value = String(v.id);
      opt.textContent = v.label;
      if (v.id === selected) opt.selected = true;
      sel.appendChild(opt);
    }
    if (selected === null) sel.value = '';
  }
}

function renderMomentVarSelector() {
  const sel = document.getElementById('moment-var-select');
  const options = continuousVariables();

  if (state.momentVarId === null || !findVariable(state.momentVarId) || findVariable(state.momentVarId).kind !== 'continuous') {
    state.momentVarId = options.length > 0 ? options[0].id : null;
  }

  sel.innerHTML = '';
  for (const v of options) {
    const opt = document.createElement('option');
    opt.value = String(v.id);
    opt.textContent = v.label;
    if (v.id === state.momentVarId) opt.selected = true;
    sel.appendChild(opt);
  }
}

function renderManualEventInputs() {
  const container = document.getElementById('manual-event-list');
  container.innerHTML = '';
  for (const v of state.variables) {
    const row = document.createElement('div');
    row.className = 'manual-event-row';
    const span = document.createElement('span');
    span.textContent = v.label;
    row.appendChild(span);

    if (v.kind === 'continuous') {
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'e.g. 0.42';
      input.dataset.varId = String(v.id);
      row.appendChild(input);
    } else {
      const select = document.createElement('select');
      select.dataset.varId = String(v.id);
      for (let i = 0; i < v.alphabetSize; i++) {
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = categoryLabel(i);
        select.appendChild(opt);
      }
      row.appendChild(select);
    }
    container.appendChild(row);
  }
}

function renderVariableList() {
  const container = document.getElementById('variables-list');
  container.innerHTML = '';
  for (const v of state.variables) {
    container.appendChild(buildVariableCard(v));
  }
  renderAllDistributionLabels();
}

function buildVariableCard(v) {
  const card = document.createElement('div');
  card.className = 'variable-panel';

  const h2 = document.createElement('h2');
  const titleSpan = document.createElement('span');
  titleSpan.textContent = v.label;
  const kindSpan = document.createElement('span');
  kindSpan.className = 'variable-kind-tag';
  kindSpan.textContent = v.kind;
  h2.appendChild(titleSpan);
  h2.appendChild(kindSpan);
  card.appendChild(h2);

  const distLabel = document.createElement('div');
  distLabel.className = 'dist-sublabel';
  distLabel.textContent = 'Distribution';
  card.appendChild(distLabel);

  const radios = document.createElement('div');
  radios.className = 'source-group';
  radios.dataset.kind = v.kind;
  card.appendChild(radios);

  if (v.kind === 'continuous') {
    buildDistributionRadios(radios, `source-${v.id}`, SOURCES, v.source, (key) => {
      v.source = key;
      updatePdfOverlay();
      updateEntropyChart();
    });

    const sliderGroup = document.createElement('div');
    sliderGroup.className = 'slider-group';
    const label = document.createElement('label');
    const valSpan = document.createElement('span');
    valSpan.textContent = v.binCfg.totalBins;
    label.appendChild(valSpan);
    label.appendChild(document.createTextNode(' equal width bins'));
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '1';
    slider.max = '200';
    slider.value = String(v.binCfg.totalBins);
    slider.addEventListener('input', () => {
      const n = parseInt(slider.value);
      valSpan.textContent = n;
      v.binCfg = makeBinConfig(n);
      const values = ownValues(v);
      v.counts = new Array(n).fill(0);
      for (const val of values) v.counts[getBin(v.binCfg, val)] += 1;
      rebuildHistoGrid();
      updateAll();
    });
    sliderGroup.appendChild(label);
    sliderGroup.appendChild(slider);
    card.appendChild(sliderGroup);
  } else {
    buildDistributionRadios(radios, `alphabet-${v.id}`, ALPHABETS, v.alphabet, (key) => {
      v.alphabet = key;
      updatePdfOverlay();
      updateEntropyChart();
    });

    const sliderGroup = document.createElement('div');
    sliderGroup.className = 'slider-group';
    const label = document.createElement('label');
    label.appendChild(document.createTextNode('Alphabet size'));
    const valSpan = document.createElement('span');
    valSpan.textContent = v.alphabetSize;
    label.appendChild(valSpan);
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '1';
    slider.max = '10';
    slider.value = String(v.alphabetSize);
    slider.addEventListener('input', () => {
      v.alphabetSize = parseInt(slider.value);
      valSpan.textContent = v.alphabetSize;
      v.counts = new Array(v.alphabetSize).fill(0);
      v.entropyHistory = new Array(state.events.length).fill(null);
      rebuildHistoGrid();
      renderManualEventInputs();
      updateAll();
    });
    sliderGroup.appendChild(label);
    sliderGroup.appendChild(slider);
    card.appendChild(sliderGroup);
  }

  return card;
}

function addVariable(kind) {
  const v = createVariable(kind);
  state.variables.push(v);
  if (state.variables.length === 2 && state.histoVarB === null) {
    state.histoVarB = v.id;
  }
  renderVariableList();
  refreshVariableDependentUI();
  rebuildHistoGrid();
  updateAll();
}

function initControls() {
  const playBtn = document.getElementById('btn-play');
  playBtn.addEventListener('click', () => {
    state.playing = !state.playing;
    playBtn.textContent = state.playing ? 'Pause' : 'Play';
    if (state.playing) startTimer(); else stopTimer();
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    resetData();
    updateAll();
  });

  const opacitySlider = document.getElementById('bar-opacity-slider');
  opacitySlider.value = String(state.barOpacity);
  opacitySlider.addEventListener('input', () => {
    state.barOpacity = parseFloat(opacitySlider.value);
    updateBarOpacity();
  });

  const hideCheckbox = document.getElementById('hide-distributions');
  hideCheckbox.addEventListener('change', () => {
    state.sourcesHidden = hideCheckbox.checked;
    renderAllDistributionLabels();
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

  const momentOrderSlider = document.getElementById('moment-order-slider');
  const momentOrderVal = document.getElementById('moment-order-value');
  momentOrderSlider.addEventListener('input', () => {
    state.maxMomentOrder = parseInt(momentOrderSlider.value);
    momentOrderVal.textContent = state.maxMomentOrder;
    updateMomentsChart();
  });

  document.querySelectorAll('input[name="moment-type"]').forEach(radio => {
    radio.addEventListener('change', () => {
      state.momentType = radio.value;
      updateMomentsChart();
    });
  });

  document.getElementById('moment-log-scale').addEventListener('change', (e) => {
    state.logScale = e.target.checked;
    updateMomentsChart();
  });

  document.getElementById('moment-var-select').addEventListener('change', (e) => {
    state.momentVarId = parseInt(e.target.value);
    updateMomentsChart();
  });

  document.getElementById('histo-var-a').addEventListener('change', (e) => {
    state.histoVarA = e.target.value === '' ? null : parseInt(e.target.value);
    renderHistoSelectors();
    rebuildHistoGrid();
    updateAll();
  });

  document.getElementById('histo-var-b').addEventListener('change', (e) => {
    state.histoVarB = e.target.value === '' ? null : parseInt(e.target.value);
    rebuildHistoGrid();
    updateAll();
  });

  document.getElementById('btn-add-continuous').addEventListener('click', () => addVariable('continuous'));
  document.getElementById('btn-add-categorical').addEventListener('click', () => addVariable('categorical'));

  const addEventBtn = document.getElementById('btn-add-event');
  addEventBtn.addEventListener('click', () => {
    const values = {};
    let valid = true;
    for (const v of state.variables) {
      const el = document.querySelector(`#manual-event-list [data-var-id="${v.id}"]`);
      if (!el) { valid = false; continue; }
      if (v.kind === 'continuous') {
        const val = parseFloat(el.value);
        if (isNaN(val)) { valid = false; continue; }
        values[v.id] = val;
      } else {
        values[v.id] = parseInt(el.value);
      }
    }
    if (valid && state.variables.length > 0) {
      addEvent(values);
      document.querySelectorAll('#manual-event-list input[type="text"]').forEach(el => { el.value = ''; });
    }
  });
}

// =============================================================================
// Init
// =============================================================================

function init() {
  const v = createVariable('continuous');
  state.variables.push(v);
  state.histoVarA = v.id;

  renderVariableList();
  refreshVariableDependentUI();
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
