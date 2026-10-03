// Parallel coordinates plot of fixed points across active bins (viz_parallel_coords.py port)
const W = 560, H = 340;
const LINE_COLOR = '#4878CF';

function render(state) {
  const { ctx, binIndices, binLabels, fixedPoints, alphaSlider, titleEl } = state;
  ctx.clearRect(0, 0, W, H);
  const n = binIndices.length;
  const padL = 30, padR = 30, padT = 14, padB = 28;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const xPix = i => padL + (n === 1 ? plotW / 2 : (plotW * i) / (n - 1));
  const yPix = v => padT + plotH - v * plotH;

  titleEl.textContent = `Parallel coordinates (n=${fixedPoints.length} fixed points)`;

  ctx.strokeStyle = '#888';
  ctx.fillStyle = '#9aa5c0';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) {
    const px = xPix(i);
    ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, padT + plotH); ctx.stroke();
    ctx.fillText(binLabels[i], px, H - 8);
  }

  const alpha = parseFloat(alphaSlider.value);
  ctx.strokeStyle = LINE_COLOR;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1.2;
  for (const fp of fixedPoints) {
    ctx.beginPath();
    binIndices.forEach((bi, i) => {
      const v = bi < fp.length ? fp[bi] : 0.0;
      const px = xPix(i), py = yPix(v);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function rebuild(state) {
  const { container, binIndices } = state;
  if (!binIndices.length) {
    container.innerHTML = `<div class="viz-panel-title">Parallel coordinates</div><div class="no-data-msg">No active bins selected.</div>`;
    state.ctx = null;
    return;
  }
  container.innerHTML = `
    <div class="viz-panel-title"></div>
    <div class="slider-field" style="max-width:200px;margin-bottom:6px"><label class="field-label">Opacity</label><input type="range" min="0" max="1" step="0.01" value="0.4"></div>
    <canvas width="${W}" height="${H}"></canvas>
  `;
  state.titleEl = container.querySelector('.viz-panel-title');
  state.alphaSlider = container.querySelector('input[type=range]');
  state.canvas = container.querySelector('canvas');
  state.ctx = state.canvas.getContext('2d');
  state.alphaSlider.addEventListener('input', () => render(state));
}

export function createParallelCoordsPanel(container) {
  const state = { container, binIndices: [], binLabels: [], fixedPoints: [] };
  rebuild(state);
  return state;
}

export function updateParallelCoordsPanel(state, fixedPoints, binIndices, binLabels) {
  const changed = JSON.stringify(binIndices) !== JSON.stringify(state.binIndices);
  const prevAlpha = state.alphaSlider ? state.alphaSlider.value : null;
  state.fixedPoints = fixedPoints;
  state.binIndices = binIndices;
  state.binLabels = binLabels;
  if (changed || !state.ctx) {
    rebuild(state);
    if (prevAlpha !== null && state.alphaSlider) state.alphaSlider.value = prevAlpha;
  }
  if (state.ctx) render(state);
}
