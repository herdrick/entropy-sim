// Radar / spider chart of fixed points across active bins (viz_radial.py port)
const LINE_COLOR = '#4878CF';
const W = 420, H = 420;

function spokeAngles(n) {
  return Array.from({ length: n }, (_, k) => Math.PI / 2 + 2 * Math.PI * k / n);
}

function polygonCoords(fixedPoints, binIndices) {
  const angles = spokeAngles(binIndices.length);
  return fixedPoints.map(fp => {
    const pts = binIndices.map((bi, k) => {
      const r = bi < fp.length ? fp[bi] : 0.0;
      return [r * Math.cos(angles[k]), r * Math.sin(angles[k])];
    });
    pts.push(pts[0]);
    return pts;
  });
}

function render(state) {
  const { ctx, binIndices, binLabels, fixedPoints, alphaSlider, titleEl } = state;
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H / 2, scale = Math.min(W, H) / 2 / 1.4;
  const toPx = (x, y) => [cx + x * scale, cy - y * scale];

  titleEl.textContent = `Radar chart (n=${fixedPoints.length} fixed points, ${binIndices.length} bins)`;

  // reference circles
  ctx.strokeStyle = '#555';
  ctx.setLineDash([4, 3]);
  for (const r of [0.25, 0.5, 0.75, 1.0]) {
    ctx.beginPath();
    for (let t = 0; t <= 64; t++) {
      const ang = 2 * Math.PI * t / 64;
      const [px, py] = toPx(r * Math.cos(ang), r * Math.sin(ang));
      t === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // spokes + labels
  const angles = spokeAngles(binIndices.length);
  ctx.strokeStyle = '#555';
  ctx.fillStyle = '#9aa5c0';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  angles.forEach((a, k) => {
    const [px, py] = toPx(Math.cos(a), Math.sin(a));
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();
    const [lx, ly] = toPx(1.12 * Math.cos(a), 1.12 * Math.sin(a));
    ctx.fillText(binLabels[k], lx, ly);
  });

  // polygons
  const alpha = parseFloat(alphaSlider.value);
  ctx.strokeStyle = LINE_COLOR;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1.5;
  for (const poly of polygonCoords(fixedPoints, binIndices)) {
    ctx.beginPath();
    poly.forEach(([x, y], i) => {
      const [px, py] = toPx(x, y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

export function createRadialPanel(container) {
  const state = { container, binIndices: [], binLabels: [], fixedPoints: [] };
  rebuild(state);
  return state;
}

function rebuild(state) {
  const { container } = state;
  if (!state.binIndices.length) {
    container.innerHTML = `<div class="viz-panel-title">Radar chart</div><div class="no-data-msg">No active bins selected.</div>`;
    state.ctx = null;
    return;
  }
  container.innerHTML = `
    <div class="viz-panel-title"></div>
    <div class="slider-field" style="max-width:200px;margin-bottom:6px"><label class="field-label">Opacity</label><input type="range" min="0" max="1" step="0.05" value="0.3"></div>
    <canvas width="${W}" height="${H}"></canvas>
  `;
  state.titleEl = container.querySelector('.viz-panel-title');
  state.alphaSlider = container.querySelector('input[type=range]');
  state.canvas = container.querySelector('canvas');
  state.ctx = state.canvas.getContext('2d');
  state.alphaSlider.addEventListener('input', () => render(state));
}

export function updateRadialPanel(state, fixedPoints, binIndices, binLabels) {
  const prevAlpha = state.alphaSlider ? state.alphaSlider.value : null;
  const changed = JSON.stringify(binIndices) !== JSON.stringify(state.binIndices);
  state.fixedPoints = fixedPoints;
  state.binIndices = binIndices;
  state.binLabels = binLabels;
  if (changed || !state.ctx) {
    rebuild(state);
    if (prevAlpha !== null && state.alphaSlider) state.alphaSlider.value = prevAlpha;
  }
  if (state.ctx) render(state);
}
