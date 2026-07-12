#!/usr/bin/env node
// Minimal browser driver for chained-surprisal-distributions' Bokeh apps.
// Launches its OWN isolated Chromium instance (separate profile from any
// shared playwright-mcp browser), so it never hits "Browser is already
// in use" when another session holds that lock.
//
// Reads newline-delimited commands from stdin:
//   nav <url>                        navigate
//   click <exact button text>        click a button by accessible name
//   fill <label prefix> <text>       fill a text input whose label starts with <label prefix>
//   press <key>                      e.g. press Enter
//   wait <ms>                        fixed pause (only for animations, not loading)
//   screenshot <path>                full-page screenshot
//   console                          print captured console errors so far
//
// Example:
//   node driver.mjs <<'EOF'
//   nav http://localhost:5006/continuous_main
//   click Add events
//   screenshot /tmp/after_add.png
//   console
//   EOF

import { chromium } from 'playwright';
import readline from 'node:readline';

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
page.on('pageerror', err => errors.push(String(err)));

const rl = readline.createInterface({ input: process.stdin });

for await (const raw of rl) {
  const line = raw.trim();
  if (!line || line.startsWith('#')) continue;
  const [cmd, ...rest] = line.split(' ');
  const arg = rest.join(' ');
  try {
    if (cmd === 'nav') {
      await page.goto(arg, { waitUntil: 'networkidle' });
      await page.waitForSelector('.bk-Column', { timeout: 15000 });
      console.log(`[nav] loaded ${arg}`);
    } else if (cmd === 'click') {
      await page.getByRole('button', { name: arg, exact: true }).click();
      console.log(`[click] ${arg}`);
    } else if (cmd === 'fill') {
      const [label, ...text] = rest;
      await page.getByPlaceholder(label, { exact: false }).fill(text.join(' '));
      console.log(`[fill] ${label} = ${text.join(' ')}`);
    } else if (cmd === 'press') {
      await page.keyboard.press(arg);
      console.log(`[press] ${arg}`);
    } else if (cmd === 'wait') {
      await page.waitForTimeout(Number(arg));
    } else if (cmd === 'screenshot') {
      await page.screenshot({ path: arg, fullPage: true });
      console.log(`[screenshot] ${arg}`);
    } else if (cmd === 'console') {
      console.log('[console errors]', errors.length ? errors : 'none');
    } else {
      console.log(`[error] unknown command: ${cmd}`);
    }
  } catch (e) {
    console.log(`[error] ${cmd} ${arg}: ${e.message}`);
  }
}

await browser.close();
