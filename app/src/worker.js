// GeoPulse browser worker: runs a full GeoPulse analysis on the visitor's machine.
// STAC search + SAS signing (Planetary Computer), COG window reads (geotiff.js, HTTP range requests), compositing
// and physics (engine.js), the trained model (ONNX Runtime Web: WebGPU, else WebAssembly), then maps and downloads.
import { contours } from "d3-contour";
import { BaseClient, BaseResponse, fromCustomClient, writeArrayBuffer } from "geotiff";
import * as E from "./engine.js";

const PC = "https://planetarycomputer.microsoft.com/api";
const TILE = 256, STRIDE = 192;
let ort; // ONNX Runtime (and its ~26 MB of WebAssembly) loads only when a model runs

const log = (msg) => self.postMessage({ type: "log", msg });

self.onmessage = async ({ data }) => {
  if (data.type !== "run") return;
  try {
    self.postMessage({ type: "result", result: await run(data.request, data.model, data.version) });
  } catch (err) {
    self.postMessage({ type: "error", message: err?.message || String(err) });
  }
};

// Transient failures (a dropped connection, "Failed to fetch", 429/5xx throttling) are retried with backoff; a big
// AOI makes hundreds of range reads and one of them failing used to end the whole job.
const TRIES = 6;
const retryable = (status) => status === 429 || status >= 500;
const backoff = (i) => new Promise((res) => setTimeout(res, 500 * 2 ** i));
async function fetchRetry(url, opts = {}) {
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(url, opts);
      if (!retryable(r.status) || i === TRIES - 1) return r;
    } catch (err) {
      if (i === TRIES - 1) throw new Error(`network error after ${TRIES} tries: ${err.message} (${new URL(url).host})`);
    }
    await backoff(i);
  }
}

// At most MAX_READS range requests in flight: browsers queue the rest anyway, and long queues time out.
const MAX_READS = 12;
let reading = 0;
const readQueue = [];
async function throttled(fn) {
  while (reading >= MAX_READS) await new Promise((res) => readQueue.push(res));
  reading++;
  try { return await fn(); } finally { reading--; readQueue.shift()?.(); }
}

class Bytes extends BaseResponse {
  constructor(status, headers, data) { super(); this._status = status; this.headers = headers; this.data = data; }
  get status() { return this._status; }
  getHeader(name) { return this.headers.get(name) || undefined; }
  async getData() { return this.data; }
}

// geotiff.js client whose reads (headers and body) are retried and throttled.
class RetryClient extends BaseClient {
  request({ headers, signal } = {}) {
    return throttled(async () => {
      for (let i = 0; ; i++) {
        try {
          const r = await fetch(this.url, { headers, signal });
          if (!retryable(r.status) || i === TRIES - 1) return new Bytes(r.status, r.headers, await r.arrayBuffer());
        } catch (err) {
          if (signal?.aborted || i === TRIES - 1) throw new Error(`imagery read failed after ${TRIES} tries: ${err.message}`);
        }
        await backoff(i);
      }
    });
  }
}

async function pool(tasks, n) {
  const out = new Array(tasks.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(n, tasks.length) }, async () => {
    while (next < tasks.length) { const i = next++; out[i] = await tasks[i](); }
  }));
  return out;
}

// ------------------------------------------------------------------ Planetary Computer

async function search(sensor, aoi, datetime) {
  const body = { collections: [E.COLLECTIONS[sensor]], intersects: aoi, limit: 100 };
  if (sensor !== "dem") body.datetime = datetime;
  if (sensor === "s2") Object.assign(body, { filter: { op: "<", args: [{ property: "eo:cloud_cover" }, E.MAX_S2_CLOUD] }, "filter-lang": "cql2-json" });
  const r = await fetchRetry(`${PC}/stac/v1/search`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`STAC search failed (${r.status}) for ${sensor}`);
  const fc = await r.json();
  return fc.features.sort((a, b) => (a.properties.datetime || "").localeCompare(b.properties.datetime || ""));
}

// SAS tokens are per storage account + container (as the planetary-computer Python client signs): the per-collection
// token can belong to a different account than an asset's (the DEM's does) and then reads fail with 403.
const tokens = {};
async function sign(href) {
  const u = new URL(href), key = `${u.hostname.split(".")[0]}/${u.pathname.split("/")[1]}`;
  const t = tokens[key];
  if (!t || Date.parse(t["msft:expiry"]) - Date.now() < 5 * 60e3) {
    const r = await fetchRetry(`${PC}/sas/v1/token/${key}`);
    if (!r.ok) throw new Error(`could not get a read token for ${key} (${r.status})`);
    tokens[key] = await r.json();
  }
  return `${href}${href.includes("?") ? "&" : "?"}${tokens[key].token}`;
}

const describe = (item, frac) => ({
  id: item.id, collection: item.collection, datetime: item.properties.datetime, cloud_cover: item.properties["eo:cloud_cover"] ?? null,
  orbit: item.properties["sat:orbit_state"] ?? null, relative_orbit: item.properties["sat:relative_orbit"] ?? null,
  platform: item.properties.platform ?? null, ...(frac === undefined ? {} : { valid_fraction: Math.round(frac * 1e4) / 1e4 }),
});

// ------------------------------------------------------------------ reading and compositing

function georef(img) {
  const [x0, y0] = img.getOrigin(), [rx, ry] = img.getResolution(), keys = img.getGeoKeys();
  const epsg = keys.ProjectedCSTypeGeoKey || keys.GeographicTypeGeoKey;
  const point = keys.GTRasterTypeGeoKey === 2; // PixelIsPoint: GDAL moves the origin half a pixel, and so do we
  const nd = img.getGDALNoData();
  return { epsg, x0: point ? x0 - rx / 2 : x0, y0: point ? y0 - ry / 2 : y0, rx, ry, nodata: nd === null ? null : Number(nd) };
}

function readerFor(grid) {
  const coordCache = new Map(), npx = grid.width * grid.height;
  return async function readBand(href, method) {
    const tiff = await fromCustomClient(new RetryClient(await sign(href)), { allowFullFile: false });
    const img = await tiff.getImage();
    const geo = georef(img), W = img.getWidth(), H = img.getHeight();
    const key = `${geo.epsg}|${geo.x0}|${geo.y0}|${geo.rx}|${geo.ry}`;
    if (!coordCache.has(key)) coordCache.set(key, E.sourceCoords(grid, geo.epsg, geo));
    const coords = coordCache.get(key);
    const win = E.windowFor(coords, W, H);
    if (!win) return new Float32Array(npx).fill(NaN);
    const [raster] = await img.readRasters({ window: win, samples: [0] });
    const data = new Float32Array(raster.length);
    for (let i = 0; i < raster.length; i++) data[i] = geo.nodata !== null && raster[i] === geo.nodata ? NaN : raster[i];
    return E.resample({ data, x0: win[0], y0: win[1], width: win[2] - win[0], height: win[3] - win[1] }, coords, W, H, method);
  };
}

async function s2Scene(item, readBand, npx) {
  const scl = await readBand(item.assets.SCL.href, "nearest");
  const ok = new Uint8Array(npx);
  let n = 0;
  for (let k = 0; k < npx; k++) if (!Number.isNaN(scl[k]) && !E.S2_INVALID_SCL.has(scl[k])) { ok[k] = 1; n++; }
  if (n / npx < 0.02) return null; // essentially all cloud or outside the tile
  // Processing baseline >= 04.00 (Jan 2022+) stores DN with a +1000 offset.
  const offset = parseFloat(item.properties["s2:processing_baseline"] || "0") >= 4 ? -1000 : 0;
  const bands = await Promise.all(E.S2_BANDS.map((b) => readBand(item.assets[b].href, "bilinear")));
  for (const band of bands) for (let k = 0; k < npx; k++) band[k] = ok[k] ? (band[k] + offset) / 10000 : NaN;
  return bands;
}

async function s1Scene(item, readBand, npx) {
  const bands = await Promise.all(E.S1_BANDS.map((b) => readBand(item.assets[b].href, "bilinear")));
  for (const band of bands) for (let k = 0; k < npx; k++) if (!(band[k] > 0)) band[k] = NaN;
  return bands;
}

// Composites of recent requests stay in memory, so trying another model, task or sensor on the same area is instant.
const cache = new Map();
async function cached(key, compute) {
  if (!cache.has(key)) {
    if (cache.size >= 12) cache.delete(cache.keys().next().value);
    cache.set(key, await compute());
  }
  return cache.get(key);
}

async function prepare(req, grid, spec) {
  const npx = grid.width * grid.height, readBand = readerFor(grid);
  const arrays = {}, scenes = {}, warnings = [];
  const method = { pre: spec.composite[0], post: spec.composite[1] };
  for (const sensor of req.sensors) {
    const S = sensor.toUpperCase();
    let found = { pre: await search(sensor, req.aoi, req.before), post: await search(sensor, req.aoi, req.after) };
    if (sensor === "s1") {
      found = E.matchOrbits(found, spec.scenes);
      if (found.restricted) log(`  S1: restricted pre-event scenes to relative orbit(s) ${found.restricted.filter(Boolean).join(", ")}`);
    }
    for (const period of E.TIMES) {
      const items = E.selectScenes(found[period], period, spec.scenes);
      if (!items.length) {
        warnings.push(`no ${S} scenes in ${period}-event window`);
        log(`! ${S} ${period}: no scenes`);
        continue;
      }
      const key = JSON.stringify([sensor, method[period], items.map((it) => it.id), grid]);
      if (!cache.has(key)) log(`  ${S} ${period}: reading ${items.length} scene(s)…`);
      const bands = await cached(key, async () => {
        const load = sensor === "s2" ? s2Scene : s1Scene;
        const loaded = (await pool(items.map((it) => () => load(it, readBand, npx)), 3)).filter(Boolean);
        const nb = sensor === "s2" ? E.S2_BANDS.length : E.S1_BANDS.length;
        return loaded.length ? E.composite(loaded, method[period], sensor, npx) : Array.from({ length: nb }, () => new Float32Array(npx).fill(NaN));
      });
      const frac = E.validFraction(bands);
      if (frac < 0.01) {
        warnings.push(`${S} ${period}-event composite has no valid pixels (clouds?)`);
        continue;
      }
      arrays[`${sensor}_${period}`] = bands;
      scenes[`${sensor}_${period}`] = items.map((it) => describe(it, frac));
      log(`✓ ${S} ${period}: ${items.length} scene(s), ${Math.round(frac * 100)}% valid`);
    }
  }
  const demItems = await search("dem", req.aoi);
  if (demItems.length) {
    arrays.dem = await cached(JSON.stringify(["dem", demItems.map((it) => it.id), grid]), async () => {
      const tiles = await pool(demItems.map((it) => () => readBand(it.assets.data.href, "bilinear")), 4);
      const elev = new Float32Array(npx).fill(NaN);
      for (const t of tiles) for (let k = 0; k < npx; k++) if (Number.isNaN(elev[k])) elev[k] = t[k];
      return [elev, E.slopeDeg(elev, grid)];
    });
    scenes.dem = demItems.map((it) => describe(it));
    log("✓ DEM + slope");
  }
  return { arrays, scenes, warnings };
}

// ------------------------------------------------------------------ model (ONNX Runtime Web)

const sessions = new Map();
async function session(card) {
  if (sessions.has(card.url)) return sessions.get(card.url);
  // The JSEP WebGPU backend (plus WebAssembly). The newer `onnxruntime-web/webgpu` EP gave wrong GPFT outputs
  // (1.30.0: 1.76 vs 3.28 km² flooded on the Emilia test box), so it is not used.
  ort ??= await import("onnxruntime-web");
  log(`  loading ${card.model_id} (${card.onnx.file})…`);
  const bytes = new Uint8Array(await (await fetchRetry(card.url)).arrayBuffer());
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (b) => b.toString(16).padStart(2, "0")).join("");
  if (card.onnx.sha256 && digest !== card.onnx.sha256) throw new Error(`checksum mismatch for ${card.onnx.file}`);
  let s = null;
  for (const ep of self.navigator.gpu ? ["webgpu", "wasm"] : ["wasm"]) {
    try {
      const sess = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
      if (await reproduces(sess, card.onnx.probe)) { s = { sess, ep }; break; }
      log(`! ${ep} computed ${card.model_id} wrongly on this device (self-check failed); falling back`);
    } catch (err) {
      log(`! ${ep} unavailable (${String(err.message || err).slice(0, 80)}); falling back`);
    }
  }
  if (!s) throw new Error("no ONNX Runtime backend in this browser computes the model correctly");
  sessions.set(card.url, s);
  return s;
}

// The model card carries the logits PyTorch produced for a fixed input. A backend is trusted only if it reproduces
// them: GPU backends and drivers can compute a model wrongly without raising any error.
async function reproduces(sess, probe) {
  if (!probe) return true; // cards exported before the self-check existed
  const feeds = Object.fromEntries(Object.entries(probe.shapes).map(([name, dims]) => [name, new ort.Tensor("float32", E.probeValues(name, dims), dims)]));
  const L = (await sess.run(feeds)).logits.data;
  let sum = 0, abs = 0;
  for (const v of L) { sum += v; abs += Math.abs(v); }
  const tol = 1e-3 * probe.absmean + 1e-4;
  return Math.abs(sum / L.length - probe.mean) <= tol && Math.abs(abs / L.length - probe.absmean) <= tol;
}

async function predictOnnx(card, A, req, grid) {
  const { sess, ep } = await session(card);
  const { width: W, height: H } = grid, npx = W * H;
  const T = E.toTensors(A, req.sensors, npx);
  const tasks = card.onnx.tasks, ti = tasks.indexOf(req.task), nOut = tasks.length + 1, p = card.onnx.dropout;
  if (ti < 0) throw new Error(`${card.model_id} cannot map ${req.task}`);
  const mc = ep === "webgpu" ? 8 : 3; // MC-dropout passes; fewer on CPU to keep waits reasonable
  log(`✓ Model ${card.model_id} v${card.version} on ${ep === "webgpu" ? "WebGPU" : "WebAssembly"} (${mc} MC-dropout passes)`);
  const channels = Object.fromEntries(card.onnx.inputs.filter((k) => !k.startsWith("mask")).map((k) => [k, T[k].length / npx]));
  const offsets = [];
  for (const r of E.starts(Math.max(H, TILE), TILE, STRIDE)) for (const c of E.starts(Math.max(W, TILE), TILE, STRIDE)) offsets.push([r, c]);
  const accT = new Float32Array(npx), accC = new Float32Array(npx), cnt = new Float32Array(npx);
  const wsum = [new Float32Array(npx), new Float32Array(npx)];
  const T2 = TILE * TILE, sigm = (x) => 1 / (1 + Math.exp(-x));
  for (let o = 0; o < offsets.length; o++) {
    const [r, c] = offsets[o];
    const feeds = {};
    for (const [name, C] of Object.entries(channels)) {
      const buf = new Float32Array(C * T2), src = T[name];
      for (let ch = 0; ch < C; ch++) for (let y = 0; y < TILE && r + y < H; y++) {
        const so = ch * npx + (r + y) * W + c, dO = ch * T2 + y * TILE;
        buf.set(src.subarray(so, so + Math.min(TILE, W - c)), dO);
      }
      feeds[name] = new ort.Tensor("float32", buf, [1, C, TILE, TILE]);
    }
    const pt = new Float32Array(T2), pc = new Float32Array(T2);
    let w0 = null;
    for (let m = 0; m < mc; m++) {
      for (const [name, C] of Object.entries(card.onnx.masks)) {
        const mask = new Float32Array(C);
        for (let i = 0; i < C; i++) mask[i] = Math.random() >= p ? 1 / (1 - p) : 0;
        feeds[name] = new ort.Tensor("float32", mask, [1, C, 1, 1]);
      }
      const out = await sess.run(feeds);
      const L = out.logits.data;
      for (let k = 0; k < T2; k++) { pt[k] += sigm(L[ti * T2 + k]) / mc; pc[k] += sigm(L[(nOut - 1) * T2 + k]) / mc; }
      if (m === 0) w0 = out.weights.data.slice();
    }
    for (let y = 0; y < TILE && r + y < H; y++) for (let x = 0; x < TILE && c + x < W; x++) {
      const g = (r + y) * W + c + x, t = y * TILE + x;
      accT[g] += pt[t]; accC[g] += pc[t]; cnt[g]++;
      wsum[0][g] += w0[t]; wsum[1][g] += w0[T2 + t];
    }
    if (o % 4 === 3 || o === offsets.length - 1) log(`  inference ${o + 1}/${offsets.length} tiles`);
  }
  const target = new Float32Array(npx), change = new Float32Array(npx), unc = new Float32Array(npx);
  const wmean = [0, 0];
  let nObs = 0;
  for (let k = 0; k < npx; k++) {
    const observed = T.s1_post_valid[k] + T.s2_post_valid[k] > 0;
    if (!observed) { target[k] = change[k] = unc[k] = NaN; continue; }
    target[k] = accT[k] / cnt[k]; change[k] = accC[k] / cnt[k]; unc[k] = E.binaryEntropy(target[k]);
    wmean[0] += wsum[0][k] / cnt[k]; wmean[1] += wsum[1][k] / cnt[k]; nObs++;
  }
  const weights = nObs ? { s1: wmean[0] / nObs, s2: wmean[1] / nObs } : { s1: null, s2: null };
  return { target, change, uncertainty: unc, weights, ep, mc };
}

// ------------------------------------------------------------------ outputs

async function png(rgba, width, height) {
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
  return canvas.convertToBlob({ type: "image/png" });
}

async function geotiff(values, grid, nodata) {
  const buf = await writeArrayBuffer(values, {
    width: grid.width, height: grid.height, ModelPixelScale: [grid.res, grid.res, 0], ModelTiepoint: [0, 0, 0, grid.x0, grid.y0, 0],
    GTModelTypeGeoKey: 1, GTRasterTypeGeoKey: 1, ProjectedCSTypeGeoKey: grid.epsg, GDAL_NODATA: String(nodata),
  });
  return new Blob([buf], { type: "image/tiff" });
}

function extent(prob, grid, crs) {
  const values = Array.from(prob, (v) => (Number.isNaN(v) ? 0 : v));
  const [mp] = contours().size([grid.width, grid.height]).thresholds([E.THRESHOLD])(values);
  const toLonLat = ([x, y]) => crs.inverse(grid.x0 + x * grid.res, grid.y0 - y * grid.res).map((v) => Math.round(v * 1e6) / 1e6);
  const ringArea = (ring) => { let a = 0; for (let i = 0; i < ring.length - 1; i++) a += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]; return Math.abs(a) / 2; };
  const feats = [];
  for (const poly of mp.coordinates) {
    const km2 = (ringArea(poly[0]) - poly.slice(1).reduce((s, r) => s + ringArea(r), 0)) * grid.res ** 2 / 1e6;
    if (km2 < 0.002) continue; // < 0.2 ha specks, like the server's sieve
    feats.push({ type: "Feature", geometry: { type: "Polygon", coordinates: poly.map((ring) => ring.map(toLonLat)) }, properties: { area_km2: Math.round(km2 * 1e5) / 1e5 } });
  }
  feats.sort((a, b) => b.properties.area_km2 - a.properties.area_km2);
  return { type: "FeatureCollection", features: feats.slice(0, 5000) };
}

const json = (obj) => new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });

async function run(req, card, version) {
  const t0 = performance.now();
  const spec = E.TASKS[req.task], grid = E.makeGrid(req.aoi), npx = grid.width * grid.height, crs = E.projection(grid.epsg);
  log(`✓ Grid EPSG:${grid.epsg}, ${grid.width}×${grid.height} px @ ${grid.res} m — computing in your browser`);
  const inp = await prepare(req, grid, spec);
  const A = inp.arrays, warnings = [...req.warnings, ...inp.warnings];
  if (!("s1_post" in A) && !("s2_post" in A)) throw new Error(`no usable post-event observations for this AOI/window: ${warnings.join("; ")}`);
  if (!("s1_pre" in A) && !("s2_pre" in A)) {
    if (req.task !== "flood") throw new Error(`${req.task} mapping needs pre-event observations; none were usable in the before window`);
    warnings.push("no pre-event observations: permanent water cannot be separated from flood water");
  }
  let maps, engine;
  if (card.model_id === "threshold-baseline") {
    const sub = Object.fromEntries(Object.entries(A).filter(([k]) => k === "dem" || req.sensors.includes(k.slice(0, 2))));
    maps = E.baselinePredict(sub, req.task, npx);
    engine = "browser · physics baseline";
    log(`✓ Model loaded: ${card.model_id} v${card.version}`);
  } else {
    maps = await predictOnnx(card, A, req, grid);
    if (req.task === "wildfire" && "s2_pre" in A && "s2_post" in A && req.sensors.includes("s2")) {
      maps.severity = new Uint8Array(npx);
      for (let k = 0; k < npx; k++) maps.severity[k] = E.severityClass(A, k);
    }
    engine = `browser · ONNX Runtime Web (${maps.ep}, ${maps.mc} MC passes)`;
  }
  const inside = E.aoiMask(grid, req.aoi);
  for (let k = 0; k < npx; k++) {
    if (!inside[k]) { maps.target[k] = maps.change[k] = maps.uncertainty[k] = NaN; continue; }
    // Physical constraint, not learned: nothing on pre-event open water burns or loses canopy.
    if (req.task !== "flood" && E.preWater(A, k)) { maps.target[k] = 0; maps.uncertainty[k] = 0; }
  }
  if (maps.severity) for (let k = 0; k < npx; k++) {
    const valid = inside[k] && !Number.isNaN(maps.target[k]) && maps.severity[k] !== E.IGNORE;
    maps.severity[k] = valid ? (maps.target[k] >= E.THRESHOLD ? maps.severity[k] : 0) : E.IGNORE;
  }
  log("✓ Inference complete — rendering maps");

  const target = spec.target, [rgb, deep] = E.TARGET_COLORS[target];
  const layerImgs = { target: E.ramp(maps.target, E.probPalette(rgb, deep)), change: E.ramp(maps.change, E.PALETTES.change), uncertainty: E.ramp(maps.uncertainty, E.PALETTES.uncertainty) };
  if (maps.severity) {
    const img = new Uint8ClampedArray(npx * 4);
    for (let k = 0; k < npx; k++) img.set(E.SEVERITY_RGBA[maps.severity[k] === E.IGNORE ? 0 : maps.severity[k]], k * 4);
    layerImgs.severity = img;
  }
  for (const p of E.TIMES) {
    if (`s2_${p}` in A) layerImgs[`s2_${p}`] = E.rgbImage(A[`s2_${p}`], npx, inside);
    if (`s1_${p}` in A) layerImgs[`s1_${p}`] = E.grayImage(A[`s1_${p}`], npx, inside);
  }
  const files = {};
  let bounds = null;
  for (const [k, rgba] of Object.entries(layerImgs)) {
    const m = E.toWebMercator(rgba, grid);
    bounds = m.bounds;
    files[`layers/${k}.png`] = await png(m.rgba, m.width, m.height);
  }
  const names = { probability: `${target}_probability.tif`, extent: `${target}_extent.geojson` };
  files[names.probability] = await geotiff(maps.target, grid, "nan");
  files["uncertainty.tif"] = await geotiff(maps.uncertainty, grid, "nan");
  files["change_probability.tif"] = await geotiff(maps.change, grid, "nan");
  if (maps.severity) files[`${target}_severity.tif`] = await geotiff(maps.severity, grid, 255);
  files[names.extent] = json(extent(maps.target, grid, crs));

  const runtime = (performance.now() - t0) / 1000;
  const summary = E.summarize({ request: req, grid, maps, scenes: inp.scenes, warnings, card, runtime, engine });
  summary.files = names;
  files["summary.json"] = json(summary);
  files["provenance.json"] = json({
    model: `${card.model_id}-v${card.version}`, onnx_sha256: card.onnx?.sha256 ?? null, geopulse: version, engine,
    request: Object.fromEntries(Object.entries(req).filter(([k]) => k !== "aoi")),
    compositing: { pre: spec.composite[0], post: spec.composite[1], scenes: spec.scenes },
    sensors: Object.fromEntries(Object.entries(inp.scenes).map(([k, v]) => [k, v.map((s) => s.id)])), scenes: inp.scenes,
    target_crs: `EPSG:${grid.epsg}`, resolution_m: grid.res, grid, created: new Date().toISOString(),
  });
  log(`✓ Done in ${runtime.toFixed(0)} s`);
  return { summary, layers: { bounds, layers: Object.keys(layerImgs), target, extent: names.extent }, files };
}
