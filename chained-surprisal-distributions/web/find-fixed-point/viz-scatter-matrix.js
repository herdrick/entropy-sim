// Pairwise scatter matrix of fixed points across active bins (viz_scatter_matrix.py port)
const MAX_N = 8;
const CELL = 110;
const SCATTER_COLOR = '#4878CF';

function extractValues(fixedPoints, binIdx) {
  return fixedPoints.map(fp => binIdx < fp.length ? fp[binIdx] : 0.0);
}

function histogram(values, nBins = 20) {
  const counts = new Array(nBins).fill(0);
  for (const v of values) {
    let idx = Math.floor(v * nBins);
    idx = Math.max(0, Math.min(nBins - 1, idx));
    counts[idx]++;
  }
  return counts;
}

function render(state) {
  const { fixedPoints, binIndices, binLabels, alphaSlider } = state;
  const n = Math.min(binIndices.length, MAX_N);
  const alpha = parseFloat(alphaSlider.value);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const canvas = state.cells[i][j];
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, CELL, CELL);
      ctx.strokeStyle = '#333';
      ctx.strokeRect(0.5, 0.5, CELL - 1, CELL - 1);
      if (i === j) {
        const values = extractValues(fixedPoints, binIndices[i]);
        const counts = histogram(values);
        const maxC = Math.max(...counts, 1);
        const barW = CELL / counts.length;
        ctx.fillStyle = SCATTER_COLOR;
        ctx.globalAlpha = 0.7;
        counts.forEach((c, k) => {
          const h = (c / maxC) * (CELL - 4);
          ctx.fillRect(k * barW, CELL - h, barW - 1, h);
        });
        ctx.globalAlpha = 1;
      } else {
        const xs = extractValues(fixedPoints, binIndices[j]);
        const ys = extractValues(fixedPoints, binIndices[i]);
        ctx.fillStyle = SCATTER_COLOR;
        ctx.globalAlpha = alpha;
        for (let k = 0; k < xs.length; k++) {
          const px = xs[k] * (CELL - 8) + 4;
          const py = CELL - (ys[k] * (CELL - 8) + 4);
          ctx.beginPath(); ctx.arc(px, py, 2.2, 0, 2 * Math.PI); ctx.fill();
        }
        ctx.globalAlpha = 1;
      }
    }
  }
}

function rebuild(state) {
  const { container, binIndices, binLabels } = state;
  const n = Math.min(binIndices.length, MAX_N);
  if (n === 0) {
    container.innerHTML = `<div class="viz-panel-title">Scatter matrix</div><div class="no-data-msg">No active bins selected.</div>`;
    state.cells = [];
    return;
  }
  container.innerHTML = `
    <div class="viz-panel-title">Scatter matrix</div>
    <div class="slider-field" style="max-width:200px;margin-bottom:6px"><label class="field-label">Opacity</label><input type="range" min="0" max="1" step="0.01" value="0.3"></div>
    <div class="scatter-matrix-grid" data-grid></div>
  `;
  state.alphaSlider = container.querySelector('input[type=range]');
  const grid = container.querySelector('[data-grid]');
  grid.style.gridTemplateColumns = `repeat(${n}, ${CELL}px)`;
  grid.style.width = `${n * CELL + (n - 1) * 2}px`;
  state.cells = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      const canvas = document.createElement('canvas');
      canvas.width = CELL; canvas.height = CELL;
      grid.appendChild(canvas);
      row.push(canvas);
    }
    state.cells.push(row);
  }
  // axis labels row/col
  const labelsWrap = document.createElement('div');
  labelsWrap.style.display = 'flex';
  labelsWrap.style.justifyContent = 'space-around';
  labelsWrap.style.width = `${n * CELL}px`;
  labelsWrap.style.fontSize = '10px';
  labelsWrap.style.color = '#9aa5c0';
  labelsWrap.style.marginTop = '2px';
  binIndices.slice(0, n).forEach((_, k) => {
    const l = document.createElement('span');
    l.textContent = binLabels[k];
    labelsWrap.appendChild(l);
  });
  container.appendChild(labelsWrap);
  state.alphaSlider.addEventListener('input', () => render(state));
}

export function createScatterMatrixPanel(container) {
  const state = { container, binIndices: [], binLabels: [], fixedPoints: [], cells: [] };
  rebuild(state);
  return state;
}

export function updateScatterMatrixPanel(state, fixedPoints, binIndices, binLabels) {
  const n = Math.min(binIndices.length, MAX_N);
  const capped = binIndices.slice(0, n);
  const changed = JSON.stringify(capped) !== JSON.stringify(state.binIndices);
  const prevAlpha = state.alphaSlider ? state.alphaSlider.value : null;
  state.fixedPoints = fixedPoints;
  state.binIndices = capped;
  state.binLabels = binLabels.slice(0, n);
  if (changed || !state.cells.length) {
    rebuild(state);
    if (prevAlpha !== null && state.alphaSlider) state.alphaSlider.value = prevAlpha;
  }
  if (state.cells.length) render(state);
}
