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

// Bars go translucent while the source distributions are shown, so the curves aren't
// hidden inside opaque bar volumes.
const BAR_OPACITY_WITH_SOURCES = 0.25;
const LINE_OPACITY_WITH_SOURCES = 0.85;

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

// =============================================================================
// Mystery Alphabet (categorical variable Y, over {a, b, c, ...})
// =============================================================================

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
// Moments Math (continuous variable X only — Y is an unordered categorical
// alphabet, for which statistical moments aren't a meaningful summary)
// =============================================================================

// Fixed categorical colors, one per moment order (identity, never cycled/reassigned).
// This palette's 8 validated slots are the practical ceiling on distinguishable lines;
// nothing else in this file hardcodes an order limit, so raising the slider's max in
// the HTML (index.html) is the only change needed to track more orders — colors just
// wrap past 8 (see buildMomentDataset) rather than this file needing an update too.
const MOMENT_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];

// Standard "nice numbers" rounding (Heckbert): picks a 1/2/5-times-a-power-of-ten
// step sized to the data's span, then rounds the bounds outward to multiples of it —
// so axis endpoints land on readable numbers instead of raw min-0.3 / max+0.3 values.
// Collapses float noise (e.g. a value mathematically exactly 1, computed as
// 1.0000000000000002 because it went through a sqrt-then-squared division) so it
// doesn't push the bound below past a step boundary that Math.ceil then jumps a
// full extra step beyond — 9 significant figures is far finer than any axis needs.
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

// The selected type (raw/central/standardized) is already named by the radio buttons,
// so labels don't repeat it — just the order, plus the conventional name where one exists.
// Trivial constants are named explicitly (1st central moment is always exactly 0;
// 2nd standardized moment is always exactly 1 — both are identities, not data).
function momentLabel(type, order) {
  // The 0th moment is trivially 1 for every type (it's just E[X^0] = 1, the
  // normalization condition) — same identity regardless of raw/central/standardized.
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

// Central moment of order j from power sums about zero (powerSums[m] = sum of x^m, for m >= 1; the
// "0th power sum" is just n, substituted directly since powerSums[0] is unused/unmaintained).
function centralMoment(powerSums, n, mean, j) {
  let m = 0;
  for (let k = 0; k <= j; k++) {
    const sk = k === 0 ? n : powerSums[k];
    m += binomial(j, k) * Math.pow(-mean, j - k) * sk;
  }
  return m / n;
}

// Returns { raw, central, standardized } for moment order j, from power sums about zero.
function computeMoment(powerSums, n, j) {
  // Shortcut order 0: it's trivially 1 regardless of the data (E[X^0] = 1), and doesn't
  // need powerSums[1]/[2] to exist — those might not be tracked if order 0 alone was requested.
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
// X values from scratch — used only when the slider requests an order for the first time.
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

function makeCounts2d(alphabetSize) {
  return Array.from({ length: alphabetSize }, () => new Array(TOTAL_BINS).fill(0));
}

const state = {
  currentSource: 'Source A',
  currentAlphabet: 'Alphabet A',
  alphabetSize: 3,
  playing: false,
  sourcesHidden: true,
  speed: 10,
  events: [], // { value, catIdx }
  counts: makeCounts2d(3),
  jointEntropyHistory: [],
  marginalXEntropyHistory: [],
  marginalYEntropyHistory: [],
  surprisalHistory: [],
  runningAvgSurprisal: [],
  momentsCacheX: {}, // { order: { raw: [...], central: [...], standardized: [...], powerSums: [...] } } — its own keys ARE the tracked-order set
  maxMomentOrder: 4, // highest order currently displayed
  momentType: 'standardized',
  logScale: false,
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
  state.momentsCacheX = {};
}

// Ensures every order from 0 up to `order` has a full, up-to-date history in momentsCacheX.
// momentsCacheX's own keys are the source of truth for which orders are tracked — this
// just fills in whichever ones in the range are missing; already-tracked orders are
// simply skipped (they're kept up to date incrementally in addEvent).
function ensureMomentsTrackedUpTo(order) {
  // This runs on every event (via updateMomentsChart), so avoid the O(n) value
  // extraction below unless some order in range actually still needs backfilling.
  let missing = false;
  for (let k = 0; k <= order; k++) {
    if (!(k in state.momentsCacheX)) { missing = true; break; }
  }
  if (!missing) return;

  const values = state.events.map(e => e.value);
  for (let k = 0; k <= order; k++) {
    if (k in state.momentsCacheX) continue;
    state.momentsCacheX[k] = backfillMomentOrder(k, values);
  }
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
      const mat = new THREE.MeshBasicMaterial({
        color: barColor.clone(),
        transparent: true,
        opacity: state.sourcesHidden ? 1 : BAR_OPACITY_WITH_SOURCES,
      });
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

  const yLabel = makeTextSprite('ℙ');
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

  buildHistogramGrid();

  // Reset camera button
  document.getElementById('resetCameraBtn').addEventListener('click', () => {
    camera.position.copy(initialCameraPosition);
    orbitControls.target.copy(initialCameraTarget);
    camera.updateProjectionMatrix();
    orbitControls.update();
  });

  // Perspective / orthographic camera mode
  document.getElementById('perspectiveCameraBtn').addEventListener('click', () => {
    setCameraMode('perspective');
  });
  document.getElementById('orthographicCameraBtn').addEventListener('click', () => {
    setCameraMode('orthographic');
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

let entropyChart, surprisalChart, momentsChart;

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
          label: 'Marginal entropy H(X)',
          data: [],
          borderColor: COLORS.marginalX,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: 'Marginal entropy H(Y)',
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
      plugins: {
        legend: {
          display: true,
          labels: { usePointStyle: true, pointStyle: 'line', boxWidth: 20, font: { size: 9 } },
        },
      },
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
        y: { title: { display: true, text: 'Joint surprisal (bits)' }, min: 0, max: 8 },
      },
      plugins: {
        legend: {
          display: true,
          labels: { usePointStyle: true, boxWidth: 12, font: { size: 9 } },
        },
      },
    },
  });

  // Panel 4: Moments of X over time — datasets are built on demand in updateMomentsChart,
  // not preallocated here, so the legend only ever lists orders currently shown.
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
        legend: {
          display: true,
          labels: { usePointStyle: true, boxWidth: 12, font: { size: 9 } },
        },
      },
    },
  });
}

// Color wraps past the palette's 8 validated slots rather than crashing — see the
// MOMENT_COLORS comment for why this file has no hardcoded order ceiling to keep in sync.
function buildMomentDataset(order) {
  return {
    label: momentLabel(state.momentType, order),
    data: state.momentsCacheX[order][state.momentType],
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

  document.getElementById('event-count-badge').textContent = `Events: ${state.events.length}`;
}

function updateBarOpacity() {
  const opacity = state.sourcesHidden ? 1 : BAR_OPACITY_WITH_SOURCES;
  for (const row of barMeshes) {
    for (const mesh of row) {
      mesh.material.opacity = opacity;
    }
  }
}

function updatePdfOverlay() {
  if (pdfLine) {
    scene.remove(pdfLine);
    pdfLine = null;
  }
  if (state.sourcesHidden) return;

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
      transparent: true, opacity: LINE_OPACITY_WITH_SOURCES,
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

function updateMomentsChart() {
  ensureMomentsTrackedUpTo(state.maxMomentOrder);

  const n = state.events.length;
  const labels = Array.from({ length: n }, (_, i) => i + 1);
  momentsChart.data.labels = labels;

  // Built fresh to exactly state.maxMomentOrder entries — nothing hidden-but-listed,
  // so the legend never shows orders that aren't currently selected.
  momentsChart.data.datasets = Array.from({ length: state.maxMomentOrder + 1 }, (_, i) => buildMomentDataset(i));
  const visibleValues = momentsChart.data.datasets.flatMap(ds => ds.data);

  momentsChart.options.scales.y.type = state.logScale ? 'logarithmic' : 'linear';
  // Chart.js's default log-scale ticks print with ~13 decimal places; compact exponential
  // notation is readable at the magnitudes moments actually span (1e-6 to 1e+2 or so).
  momentsChart.options.scales.y.ticks.callback = state.logScale
    ? (value) => Number(value).toExponential(0)
    : undefined;

  if (n > 0) {
    momentsChart.options.scales.x.min = 1;
    momentsChart.options.scales.x.max = Math.max(n, 10);
    if (state.logScale) {
      // Logarithmic scales can't represent zero/negative values (several moment types
      // legitimately produce them — e.g. skewness, or the trivial ≡0 line); let
      // Chart.js auto-range over whatever positive values are present instead of
      // forcing a padded min/max the way the linear case does.
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

  const n = state.events.length;
  for (const key of Object.keys(state.momentsCacheX)) {
    const k = Number(key);
    const cache = state.momentsCacheX[k];
    for (let m = 1; m <= k; m++) cache.powerSums[m] += Math.pow(value, m);
    const { raw, central, standardized } = computeMoment(cache.powerSums, n, k);
    cache.raw.push(raw);
    cache.central.push(central);
    cache.standardized.push(standardized);
  }

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

function buildDistributionRadios(containerId, name, options, checkedKey, onChange) {
  const container = document.getElementById(containerId);
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
  renderDistributionLabels(containerId, options);
}

function renderDistributionLabels(containerId, options) {
  const container = document.getElementById(containerId);
  container.querySelectorAll('.dist-label').forEach(span => {
    const opt = options[span.dataset.key];
    span.textContent = state.sourcesHidden ? opt.glyph : opt.name;
    span.classList.toggle('dist-glyph', state.sourcesHidden);
  });
}

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
    resetData();
    buildHistogramGrid();
    updatePdfOverlay();
    updateAll();
  });

  const hideCheckbox = document.getElementById('hide-distributions');
  hideCheckbox.addEventListener('change', () => {
    state.sourcesHidden = hideCheckbox.checked;
    renderDistributionLabels('source-radios', SOURCES);
    renderDistributionLabels('alphabet-radios', ALPHABETS);
    updateBarOpacity();
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

  buildDistributionRadios('source-radios', 'source', SOURCES, state.currentSource, (key) => {
    state.currentSource = key;
    updatePdfOverlay();
    updateEntropyChart();
  });

  buildDistributionRadios('alphabet-radios', 'alphabet', ALPHABETS, state.currentAlphabet, (key) => {
    state.currentAlphabet = key;
    updatePdfOverlay();
    updateEntropyChart();
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
