// @ts-check
const { defineConfig } = require('playwright/test');

module.exports = defineConfig({
  testDir: 'tests',
  use: {
    baseURL: 'http://localhost:5006',
  },
  webServer: {
    command: 'bokeh serve foo.py --port 5006',
    url: 'http://localhost:5006/foo',
    timeout: 20_000,
    reuseExistingServer: !process.env.CI,
    stderr: 'pipe',
  },
});
