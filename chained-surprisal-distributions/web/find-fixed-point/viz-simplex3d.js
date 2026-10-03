// 3D simplex scatter of fixed points (viz_simplex3d.py port) — Three.js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const NO_BIN = '__none__';

function buildOptions(fixedPoints) {
  if (!fixedPoints.length) return [];
  const maxLen = Math.max(...fixedPoints.map(fp => fp.length));
  const opts = [];
  for (let i = 0; i < maxLen; i++) {
    const vals = fixedPoints.filter(fp => i < fp.length).map(fp => fp[i]);
    if (vals.some(v => v > 1e-12)) {
      const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
      opts.push({ value: String(i), label: `p${i + 1} | ${avg.toFixed(4)}` });
    }
  }
  return opts;
}

function parseIdx(val) {
  if (val === NO_BIN) return -1;
  const n = parseInt(val, 10);
  return isNaN(n) ? -1 : n;
}

function fillSelect(select, opts, keepValue) {
  const prev = keepValue !== undefined ? keepValue : select.value;
  select.innerHTML = '';
  const noneOpt = document.createElement('option');
  noneOpt.value = NO_BIN; noneOpt.textContent = '(none)';
  select.appendChild(noneOpt);
  for (const o of opts) {
    const el = document.createElement('option');
    el.value = o.value; el.textContent = o.label;
    select.appendChild(el);
  }
  if ([...select.options].some(o => o.value === prev)) select.value = prev;
}

export function createSimplex3DPanel(container) {
  container.innerHTML = `
    <div class="viz-panel-title">3D simplex — fixed-point locations</div>
    <div class="row" style="margin-bottom:6px">
      <select data-a style="width:140px"></select>
      <select data-b style="width:140px"></select>
      <select data-c style="width:140px"></select>
    </div>
    <div class="row" style="margin-bottom:6px">
      <div class="slider-field" style="min-width:120px"><label class="field-label">Elevation <span data-elev-val></span></label><input type="range" data-elev min="-90" max="90" step="1" value="26"></div>
      <div class="slider-field" style="min-width:120px"><label class="field-label">Azimuth <span data-azim-val></span></label><input type="range" data-azim min="0" max="360" step="1" value="47"></div>
      <div class="slider-field" style="min-width:120px"><label class="field-label">Opacity <span data-alpha-val></span></label><input type="range" data-alpha min="0" max="1" step="0.01" value="0.1"></div>
    </div>
    <div data-canvas-wrap style="width:460px;height:400px;position:relative;"></div>
  `;
  const selA = container.querySelector('[data-a]');
  const selB = container.querySelector('[data-b]');
  const selC = container.querySelector('[data-c]');
  const elevSlider = container.querySelector('[data-elev]');
  const azimSlider = container.querySelector('[data-azim]');
  const alphaSlider = container.querySelector('[data-alpha]');
  const elevVal = container.querySelector('[data-elev-val]');
  const azimVal = container.querySelector('[data-azim-val]');
  const alphaVal = container.querySelector('[data-alpha-val]');
  elevVal.textContent = elevSlider.value;
  azimVal.textContent = azimSlider.value;
  alphaVal.textContent = alphaSlider.value;
  const canvasWrap = container.querySelector('[data-canvas-wrap]');

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0f1e);
  const camera = new THREE.PerspectiveCamera(45, 460 / 400, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(460, 400);
  canvasWrap.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0.33, 0.33, 0.33);

  // triangle (simplex face) + edges
  const verts = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const triGeo = new THREE.BufferGeometry().setFromPoints(verts);
  triGeo.setIndex([0, 1, 2]);
  const triMat = new THREE.MeshBasicMaterial({ color: 0x4878cf, opacity: 0.12, transparent: true, side: THREE.DoubleSide });
  scene.add(new THREE.Mesh(triGeo, triMat));
  const edgeGeo = new THREE.BufferGeometry().setFromPoints([verts[0], verts[1], verts[2], verts[0]]);
  scene.add(new THREE.Line(edgeGeo, new THREE.LineBasicMaterial({ color: 0x888888 })));
  const axesHelper = new THREE.AxesHelper(0.05);
  scene.add(axesHelper);

  const pointsGeo = new THREE.BufferGeometry();
  pointsGeo.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
  const pointsMat = new THREE.PointsMaterial({ color: 0x3355ff, size: 0.02, transparent: true, opacity: 0.1, sizeAttenuation: true });
  const pointsObj = new THREE.Points(pointsGeo, pointsMat);
  scene.add(pointsObj);

  const labelDivs = [document.createElement('div'), document.createElement('div'), document.createElement('div')];
  labelDivs.forEach(d => {
    d.style.position = 'absolute';
    d.style.color = '#ccc';
    d.style.fontSize = '11px';
    d.style.pointerEvents = 'none';
    canvasWrap.appendChild(d);
  });

  function setCameraFromElevAzim(elevDeg, azimDeg) {
    const r = 2.2;
    const elev = elevDeg * Math.PI / 180, azim = azimDeg * Math.PI / 180;
    const x = r * Math.cos(elev) * Math.cos(azim) + 0.33;
    const y = r * Math.sin(elev) + 0.33;
    const z = r * Math.cos(elev) * Math.sin(azim) + 0.33;
    camera.position.set(x, y, z);
    camera.lookAt(0.33, 0.33, 0.33);
  }
  setCameraFromElevAzim(26, 47);

  function updateLabelPositions(labels) {
    const positions = verts;
    for (let i = 0; i < 3; i++) {
      labelDivs[i].textContent = labels[i] || '';
      const v = positions[i].clone().project(camera);
      labelDivs[i].style.left = ((v.x * 0.5 + 0.5) * 460) + 'px';
      labelDivs[i].style.top = ((-v.y * 0.5 + 0.5) * 400) + 'px';
    }
  }

  let lastLabels = ['', '', ''];
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    updateLabelPositions(lastLabels);
    renderer.render(scene, camera);
  }
  animate();

  const state = {
    container, selA, selB, selC, elevSlider, azimSlider, alphaSlider,
    scene, camera, renderer, controls, pointsObj, pointsMat,
    _fixedPoints: [], _rerender: null,
  };

  function rerender() {
    const a = parseIdx(selA.value), b = parseIdx(selB.value), c = parseIdx(selC.value);
    const pts = [];
    for (const fp of state._fixedPoints) {
      const p = [0, 0, 0];
      [a, b, c].forEach((idx, k) => { if (idx >= 0 && idx < fp.length) p[k] = fp[idx]; });
      const s = p[0] + p[1] + p[2];
      if (s > 0) pts.push([p[0] / s, p[1] / s, p[2] / s]);
    }
    const flat = new Float32Array(pts.length * 3);
    pts.forEach((p, i) => { flat[i * 3] = p[0]; flat[i * 3 + 1] = p[1]; flat[i * 3 + 2] = p[2]; });
    pointsGeo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
    pointsGeo.computeBoundingSphere();
    pointsMat.opacity = parseFloat(alphaSlider.value);
    lastLabels = [
      a >= 0 ? `p${a + 1}` : '—',
      b >= 0 ? `p${b + 1}` : '—',
      c >= 0 ? `p${c + 1}` : '—',
    ];
  }
  state._rerender = rerender;

  elevSlider.addEventListener('input', () => { elevVal.textContent = elevSlider.value; setCameraFromElevAzim(parseFloat(elevSlider.value), parseFloat(azimSlider.value)); });
  azimSlider.addEventListener('input', () => { azimVal.textContent = azimSlider.value; setCameraFromElevAzim(parseFloat(elevSlider.value), parseFloat(azimSlider.value)); });
  alphaSlider.addEventListener('input', () => { alphaVal.textContent = alphaSlider.value; rerender(); });
  [selA, selB, selC].forEach(sel => sel.addEventListener('change', rerender));

  rerender();
  return state;
}

export function updateSimplex3DPanel(state, fixedPoints, _bin_indices, _bin_labels) {
  state._fixedPoints = fixedPoints;
  const opts = buildOptions(fixedPoints);
  const optVals = new Set(opts.map(o => o.value));
  const sels = [state.selA, state.selB, state.selC];

  if (opts.length && sels.every(s => s.value === NO_BIN || s.value === '')) {
    sels.forEach((s, k) => { if (k < opts.length) s.value = opts[k].value; });
  }
  const extra = [];
  for (const s of sels) {
    const v = s.value;
    if (v && v !== NO_BIN && !optVals.has(v)) {
      const idx = parseInt(v, 10);
      if (!isNaN(idx)) extra.push({ value: v, label: `p${idx + 1} | (no data)` });
    }
  }
  const fullOpts = [...opts, ...extra];
  for (const s of sels) fillSelect(s, fullOpts);

  state._rerender();
}
