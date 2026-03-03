// @ts-check
const { test, expect } = require('playwright/test');

// ── Bokeh JS helpers ──────────────────────────────────────────────────────────

/** Wait until the Bokeh document is live and our ColumnDataSource is present. */
async function waitForBokeh(page) {
  await page.waitForFunction(() => {
    if (!window.Bokeh || !Bokeh.documents.length) return false;
    const models = Array.from(Bokeh.documents[0]._all_models.values());
    return models.some(m => m.data && 'left_inf' in m.data);
  }, { timeout: 15_000 });
}

/**
 * Find the model IDs we care about once and reuse them across helpers.
 * Returns { pSourceId, xRangeId }.
 */
async function findModelIds(page) {
  return page.evaluate(() => {
    const models = Array.from(Bokeh.documents[0]._all_models.values());

    // Our ColumnDataSource has a left_inf column
    const pSource = models.find(m => m.data && 'left_inf' in m.data);

    // The shared x Range1d starts at -10 / ends at 10; y ranges have different values
    const xRange = models.find(m =>
      typeof m.start === 'number' && Math.abs(m.start - (-10)) < 1 &&
      typeof m.end   === 'number' && Math.abs(m.end   -   10) < 1
    );

    return { pSourceId: pSource.id, xRangeId: xRange.id };
  });
}

/** Read the current bar data and viewport range. */
async function getState(page, ids) {
  return page.evaluate((ids) => {
    const doc     = Bokeh.documents[0];
    const pSource = doc._all_models.get(ids.pSourceId);
    const xRange  = doc._all_models.get(ids.xRangeId);
    return {
      left:     Array.from(pSource.data.left),
      right:    Array.from(pSource.data.right),
      leftInf:  Array.from(pSource.data.left_inf),
      rightInf: Array.from(pSource.data.right_inf),
      xStart:   xRange.start,
      xEnd:     xRange.end,
    };
  }, ids);
}

/** Programmatically move the x_range and let the JS callback settle. */
async function setXRange(page, ids, start, end) {
  await page.evaluate((args) => {
    const xRange = Bokeh.documents[0]._all_models.get(args.ids.xRangeId);
    xRange.start = args.start;
    xRange.end   = args.end;
  }, { ids, start, end });
  await page.waitForTimeout(150);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

test.describe('infinite bin edges track viewport', () => {
  let ids;

  test.beforeEach(async ({ page }) => {
    await page.goto('/foo');
    await waitForBokeh(page);
    ids = await findModelIds(page);
  });

  test('single bin is marked infinite on both sides', async ({ page }) => {
    const state = await getState(page, ids);
    expect(state.leftInf[0]).toBe(1);
    expect(state.rightInf[state.rightInf.length - 1]).toBe(1);
  });

  test('single bin fills the initial viewport', async ({ page }) => {
    const state = await getState(page, ids);
    expect(state.left[0]).toBeCloseTo(state.xStart, 1);
    expect(state.right[state.right.length - 1]).toBeCloseTo(state.xEnd, 1);
  });

  test('bin stretches to fill a wider viewport after zoom out', async ({ page }) => {
    await setXRange(page, ids, -100, 100);
    const state = await getState(page, ids);
    expect(state.left[0]).toBeCloseTo(-100, 1);
    expect(state.right[state.right.length - 1]).toBeCloseTo(100, 1);
  });

  test('zoom in then out: bin always matches viewport', async ({ page }) => {
    await setXRange(page, ids, -2, 2);
    const zoomed = await getState(page, ids);
    expect(zoomed.left[0]).toBeCloseTo(-2, 1);
    expect(zoomed.right[zoomed.right.length - 1]).toBeCloseTo(2, 1);

    await setXRange(page, ids, -500, 500);
    const wide = await getState(page, ids);
    expect(wide.left[0]).toBeCloseTo(-500, 1);
    expect(wide.right[wide.right.length - 1]).toBeCloseTo(500, 1);
  });

  test('fencepost creates two bins; outer edges still track viewport', async ({ page }) => {
    // Reveal the fencepost input and add a fencepost at 0
    await page.click('button:has-text("Divide a bin")');
    await page.fill('input[placeholder*="Fencepost"]', '0');
    await page.press('input[placeholder*="Fencepost"]', 'Enter');
    await page.waitForTimeout(500); // round-trip to Python server

    await setXRange(page, ids, -50, 50);
    const state = await getState(page, ids);

    expect(state.left.length).toBe(2);

    // Left bin: left edge tracks viewport, right edge is the fencepost
    expect(state.leftInf[0]).toBe(1);
    expect(state.left[0]).toBeCloseTo(-50, 1);
    expect(state.right[0]).toBeCloseTo(0, 3);

    // Right bin: left edge is the fencepost, right edge tracks viewport
    expect(state.rightInf[1]).toBe(1);
    expect(state.left[1]).toBeCloseTo(0, 3);
    expect(state.right[1]).toBeCloseTo(50, 1);
  });
});
