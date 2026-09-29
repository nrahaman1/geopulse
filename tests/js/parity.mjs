// Runs the browser engine on inputs prepared by tests/test_js_parity.py and writes its outputs back as JSON.
// Usage: node parity.mjs <engine.js> <in.json> <out.json>
import fs from "node:fs";
import { pathToFileURL } from "node:url";

const [, , enginePath, inPath, outPath] = process.argv;
const E = await import(pathToFileURL(enginePath).href);
const inp = JSON.parse(fs.readFileSync(inPath, "utf8"));
const f32 = (a) => Float32Array.from(a, (v) => (v === null ? NaN : v));
const bands = (arrays) => Object.fromEntries(Object.entries(arrays).map(([k, b]) => [k, b.map(f32)]));
const safe = (fn) => { try { return fn(); } catch (err) { return { error: err.message }; } };

const out = {
  grids: inp.aois.map((g) => E.makeGrid(g)),
  areas: inp.aois.map((g) => E.areaKm2(g)),
  utm: inp.points.map(([epsg, lon, lat]) => E.projection(epsg).forward(lon, lat)),
  utm_inverse: inp.points.map(([epsg, lon, lat]) => E.projection(epsg).inverse(...E.projection(epsg).forward(lon, lat))),
  requests: inp.requests.map((r) => safe(() => E.makeRequest(r))),
  selection: inp.selection.map(([n, period, how]) => E.selectScenes([...Array(n).keys()], period, how)),
  starts: inp.starts.map(([n, size, stride]) => E.starts(n, size, stride)),
  masks: inp.masks.map(({ grid, geom }) => Array.from(E.aoiMask(grid, geom))),
  baseline: inp.cases.map((c) => {
    const r = E.baselinePredict(bands(c.arrays), c.task, c.npx);
    return { target: Array.from(r.target), change: Array.from(r.change), uncertainty: Array.from(r.uncertainty), severity: r.severity && Array.from(r.severity) };
  }),
  tensors: inp.tensors.map((c) => Object.fromEntries(Object.entries(E.toTensors(bands(c.arrays), c.sensors, c.npx)).map(([k, v]) => [k, Array.from(v)]))),
  tasks: E.TASKS,
};
fs.writeFileSync(outPath, JSON.stringify(out)); // NaN -> null
