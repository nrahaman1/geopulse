// GeoPulse browser engine: the GeoPulse pipeline in plain JavaScript, so the visitor's own computer does the work.
// Pure functions only — no DOM, no network (worker.js does I/O), so Node can test it. It mirrors
// geopulse/{grid,data,baseline,model,pipeline}.py, and tests/test_js_parity.py checks that both agree.

export const TASKS = {
  flood: { title: "Flood inundation", target: "flood", verb: "flooded", composite: ["median", "first"], scenes: "closest", seasonal: false },
  wildfire: { title: "Wildfire / burn", target: "burn", verb: "burned", composite: ["median", "median"], scenes: "closest", seasonal: false },
  vegetation: { title: "Vegetation disturbance", target: "disturbance", verb: "disturbed", composite: ["median", "median"], scenes: "spread", seasonal: true },
};
export const SENSORS = { s1: 2, s2: 6 };
export const TIMES = ["pre", "post"];
export const S1_BANDS = ["vv", "vh"];
export const S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"];
export const S2_INVALID_SCL = new Set([0, 1, 3, 8, 9, 10]);
export const COLLECTIONS = { s1: "sentinel-1-rtc", s2: "sentinel-2-l2a", dem: "cop-dem-glo-30" };
export const MAX_SCENES = 4;
export const MAX_S2_CLOUD = 60;
export const MAX_WINDOW_DAYS = 120;
export const MAX_SEASON_GAP_DAYS = 45;
export const REVIEW_UNCERTAINTY = 0.8;
export const THRESHOLD = 0.5;
export const IGNORE = 255;
export const NORM = {
  s1: { mean: [-12.0, -19.0], std: [5.0, 5.0] },
  s2: { mean: [0.08, 0.1, 0.1, 0.25, 0.2, 0.14], std: [0.06, 0.06, 0.08, 0.1, 0.1, 0.09] },
};
export const PHYSICS = {
  VV_WATER_DB: -18.0, VV_DROP_DB: 3.0, VV_DROP_CEILING_DB: -11.0, MNDWI_WATER: 0.0, DNBR_BURN: 0.1,
  DNBR_SEVERITY: [0.1, 0.27, 0.44, 0.66], NDVI_FUEL: 0.1, NDVI_DROP: 0.2, NDVI_CANOPY: 0.5, VH_DROP_DB: 2.0,
  VH_CANOPY_DB: -17.0, SAR_ONLY_BURN_UNCERTAINTY: 0.5, CHANGE_DB: 3.0, CHANGE_INDEX: 0.2,
};
const SENSOR_ALIASES = { s1: "s1", "sentinel-1": "s1", sentinel1: "s1", s2: "s2", "sentinel-2": "s2", sentinel2: "s2" };

// ------------------------------------------------------------------ projections (WGS84 UTM, Krüger n^6 series)

const A_WGS84 = 6378137, F_WGS84 = 1 / 298.257223563, K0 = 0.9996;
const N3 = F_WGS84 / (2 - F_WGS84);
const E_WGS84 = Math.sqrt(F_WGS84 * (2 - F_WGS84));
const [n, n2, n3, n4, n5, n6] = [N3, N3 ** 2, N3 ** 3, N3 ** 4, N3 ** 5, N3 ** 6];
const A_RECT = (A_WGS84 / (1 + n)) * (1 + n2 / 4 + n4 / 64 + n6 / 256);
const ALPHA = [
  n / 2 - (2 * n2) / 3 + (5 * n3) / 16 + (41 * n4) / 180 - (127 * n5) / 288 + (7891 * n6) / 37800,
  (13 * n2) / 48 - (3 * n3) / 5 + (557 * n4) / 1440 + (281 * n5) / 630 - (1983433 * n6) / 1935360,
  (61 * n3) / 240 - (103 * n4) / 140 + (15061 * n5) / 26880 + (167603 * n6) / 181440,
  (49561 * n4) / 161280 - (179 * n5) / 168 + (6601661 * n6) / 7257600,
  (34729 * n5) / 80640 - (3418889 * n6) / 1995840,
  (212378941 * n6) / 319334400,
];
const BETA = [
  n / 2 - (2 * n2) / 3 + (37 * n3) / 96 - n4 / 360 - (81 * n5) / 512 + (96199 * n6) / 604800,
  n2 / 48 + n3 / 15 - (437 * n4) / 1440 + (46 * n5) / 105 - (1118711 * n6) / 3870720,
  (17 * n3) / 480 - (37 * n4) / 840 - (209 * n5) / 4480 + (5569 * n6) / 90720,
  (4397 * n4) / 161280 - (11 * n5) / 504 - (830251 * n6) / 7257600,
  (4583 * n5) / 161280 - (108847 * n6) / 3991680,
  (20648693 * n6) / 638668800,
];
const RAD = Math.PI / 180;

function utm(zone, south) {
  const lon0 = ((zone - 1) * 6 - 180 + 3) * RAD, fn = south ? 10000000 : 0;
  const forward = (lon, lat) => {
    const phi = lat * RAD, lam = lon * RAD - lon0;
    const tau = Math.tan(phi);
    const sigma = Math.sinh(E_WGS84 * Math.atanh((E_WGS84 * tau) / Math.sqrt(1 + tau * tau)));
    const tauP = tau * Math.sqrt(1 + sigma * sigma) - sigma * Math.sqrt(1 + tau * tau);
    const xiP = Math.atan2(tauP, Math.cos(lam));
    const etaP = Math.asinh(Math.sin(lam) / Math.sqrt(tauP * tauP + Math.cos(lam) ** 2));
    let xi = xiP, eta = etaP;
    for (let j = 1; j <= 6; j++) {
      xi += ALPHA[j - 1] * Math.sin(2 * j * xiP) * Math.cosh(2 * j * etaP);
      eta += ALPHA[j - 1] * Math.cos(2 * j * xiP) * Math.sinh(2 * j * etaP);
    }
    return [500000 + K0 * A_RECT * eta, fn + K0 * A_RECT * xi];
  };
  const inverse = (x, y) => {
    const xi = (y - fn) / (K0 * A_RECT), eta = (x - 500000) / (K0 * A_RECT);
    let xiP = xi, etaP = eta;
    for (let j = 1; j <= 6; j++) {
      xiP -= BETA[j - 1] * Math.sin(2 * j * xi) * Math.cosh(2 * j * eta);
      etaP -= BETA[j - 1] * Math.cos(2 * j * xi) * Math.sinh(2 * j * eta);
    }
    const sinhEta = Math.sinh(etaP), sinXi = Math.sin(xiP), cosXi = Math.cos(xiP);
    const tauP = sinXi / Math.sqrt(sinhEta * sinhEta + cosXi * cosXi);
    let tau = tauP;
    for (let i = 0; i < 20; i++) {
      const sigma = Math.sinh(E_WGS84 * Math.atanh((E_WGS84 * tau) / Math.sqrt(1 + tau * tau)));
      const tauI = tau * Math.sqrt(1 + sigma * sigma) - sigma * Math.sqrt(1 + tau * tau);
      const d = ((tauP - tauI) / Math.sqrt(1 + tauI * tauI)) *
        ((1 + (1 - E_WGS84 ** 2) * tau * tau) / ((1 - E_WGS84 ** 2) * Math.sqrt(1 + tau * tau)));
      tau += d;
      if (Math.abs(d) < 1e-12) break;
    }
    return [(Math.atan2(sinhEta, cosXi) + lon0) / RAD, Math.atan(tau) / RAD];
  };
  return { forward, inverse };
}

/** Projection for the EPSG codes Sentinel data on the Planetary Computer uses: WGS84 UTM zones and lon/lat. */
export function projection(epsg) {
  if (epsg === 4326) return { forward: (lon, lat) => [lon, lat], inverse: (x, y) => [x, y] };
  if ((epsg > 32600 && epsg <= 32660) || (epsg > 32700 && epsg <= 32760)) return utm(epsg % 100, epsg > 32700);
  throw new Error(`unsupported CRS EPSG:${epsg}`);
}

const MERC_R = 6378137;
export const mercator = {
  forward: (lon, lat) => [MERC_R * lon * RAD, MERC_R * Math.log(Math.tan(Math.PI / 4 + (lat * RAD) / 2))],
  inverse: (x, y) => [x / MERC_R / RAD, (2 * Math.atan(Math.exp(y / MERC_R)) - Math.PI / 2) / RAD],
};

// ------------------------------------------------------------------ AOI, grid, request (geopulse/grid.py + pipeline.py)

export class RequestError extends Error {}

function rings(geom) {
  return geom.type === "MultiPolygon" ? geom.coordinates : [geom.coordinates];
}

export function aoiGeometry(obj) {
  if (obj?.type === "FeatureCollection") {
    if ((obj.features || []).length !== 1) throw new RequestError("FeatureCollection AOI must contain exactly one feature");
    obj = obj.features[0];
  }
  if (obj?.type === "Feature") obj = obj.geometry || {};
  if (!obj || !["Polygon", "MultiPolygon"].includes(obj.type)) throw new RequestError("AOI must be a Polygon or MultiPolygon");
  for (const poly of rings(obj)) {
    for (const ring of poly) {
      const [a, b] = [ring[0], ring[ring.length - 1]];
      if (ring.length < 4 || a[0] !== b[0] || a[1] !== b[1]) throw new RequestError("AOI rings must be closed with at least 4 positions");
      for (const [x, y] of ring) if (!(x >= -180 && x <= 180 && y >= -90 && y <= 90)) throw new RequestError("AOI coordinates must be lon/lat (EPSG:4326)");
    }
  }
  return obj;
}

export function geomBounds(geom) {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const poly of rings(geom)) for (const ring of poly) for (const [x, y] of ring) {
    w = Math.min(w, x); s = Math.min(s, y); e = Math.max(e, x); n = Math.max(n, y);
  }
  return [w, s, e, n];
}

export function utmEpsg(lon, lat) {
  const zone = Math.min(Math.floor((lon + 180) / 6) + 1, 60);
  return (lat >= 0 ? 32600 : 32700) + zone;
}

export function areaKm2(geom) {
  const [w, s, e, n] = geomBounds(geom);
  const p = projection(utmEpsg((w + e) / 2, (s + n) / 2));
  let total = 0;
  for (const poly of rings(geom)) poly.forEach((ring, k) => {
    const pts = ring.map(([x, y]) => p.forward(x, y));
    let a = 0;
    for (let i = 0; i < pts.length - 1; i++) a += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1];
    total += k === 0 ? Math.abs(a) / 2 : -Math.abs(a) / 2;
  });
  return total / 1e6;
}

/** Same deterministic grid as geopulse.grid.make_grid: UTM zone of the AOI centre, edges snapped to `res`. */
export function makeGrid(geom, res = 10) {
  const [w, s, e, n] = geomBounds(geom);
  const epsg = utmEpsg((w + e) / 2, (s + n) / 2);
  const p = projection(epsg);
  const xs = [], ys = [];
  const k = 21; // like rasterio.warp.transform_bounds(densify_pts=21)
  for (let i = 0; i <= k + 1; i++) {
    const t = i / (k + 1);
    for (const [lon, lat] of [[w + t * (e - w), s], [w + t * (e - w), n], [w, s + t * (n - s)], [e, s + t * (n - s)]]) {
      const [x, y] = p.forward(lon, lat);
      xs.push(x); ys.push(y);
    }
  }
  const minx = Math.floor(Math.min(...xs) / res) * res, miny = Math.floor(Math.min(...ys) / res) * res;
  const maxx = Math.ceil(Math.max(...xs) / res) * res, maxy = Math.ceil(Math.max(...ys) / res) * res;
  return { epsg, res, x0: minx, y0: maxy, width: Math.round((maxx - minx) / res), height: Math.round((maxy - miny) / res) };
}

function parseDay(v) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(v))) throw new RequestError(`bad date ${v}: use YYYY-MM-DD`);
  const d = new Date(`${v}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) throw new RequestError(`bad date ${v}`);
  return d;
}

export function parseWindow(w) {
  const [a, b] = (typeof w === "string" ? w.split("/") : w).map(parseDay);
  const days = Math.round((b - a) / 864e5);
  if (days < 0 || days > MAX_WINDOW_DAYS) throw new RequestError(`window must be ordered and at most ${MAX_WINDOW_DAYS} days`);
  return [a, b];
}

const iso = (d) => d.toISOString().slice(0, 10);

export function seasonGap(b, a) {
  const doy = ([x, y]) => {
    const mid = new Date(x.getTime() + Math.floor(Math.round((y - x) / 864e5) / 2) * 864e5);
    return Math.round((mid - Date.UTC(mid.getUTCFullYear(), 0, 1)) / 864e5) + 1;
  };
  const d = Math.abs(doy(b) - doy(a));
  return Math.min(d, 365 - d);
}

/** Validate and normalise a request exactly like geopulse.pipeline.make_request. */
export function makeRequest({ aoi, before, after, task = "flood", sensors = ["s1", "s2"], model = "auto", maxKm2 = 1000 }) {
  if (!(task in TASKS)) throw new RequestError(`unknown task ${task}; supported: ${Object.keys(TASKS)}`);
  if (typeof sensors === "string") sensors = sensors.split(",");
  sensors = [...new Set(sensors.map((s) => {
    const k = SENSOR_ALIASES[String(s).trim().toLowerCase()];
    if (!k) throw new RequestError(`unknown sensor ${s}; use s1, s2`);
    return k;
  }))];
  if (!sensors.length) throw new RequestError("at least one sensor is required");
  const geometry = aoiGeometry(aoi);
  const km2 = areaKm2(geometry);
  if (!(km2 >= 0.01 && km2 <= maxKm2)) throw new RequestError(`AOI area ${km2.toFixed(1)} km² outside allowed range 0.01–${maxKm2} km²`);
  const b = parseWindow(before), a = parseWindow(after);
  if (b[1] >= a[0]) throw new RequestError("the before window must end before the after window starts");
  const warnings = [];
  const gap = seasonGap(b, a);
  if (TASKS[task].seasonal && gap > MAX_SEASON_GAP_DAYS) {
    warnings.push(`before/after windows are ${gap} days apart in the seasonal cycle: phenology may be mapped as disturbance; use the same season in different years`);
  }
  return {
    task, aoi: geometry, before: `${iso(b[0])}/${iso(b[1])}`, after: `${iso(a[0])}/${iso(a[1])}`, sensors, model,
    resolution: 10, aoi_km2: Math.round(km2 * 1000) / 1000, warnings,
  };
}

/** Pixel-centre-in-AOI mask on the grid (rasterio geometry_mask(invert=True) semantics). */
export function aoiMask(grid, geom) {
  const p = projection(grid.epsg);
  const polys = rings(geom).map((poly) => poly.map((ring) => ring.map(([x, y]) => p.forward(x, y))));
  const mask = new Uint8Array(grid.width * grid.height);
  for (let r = 0; r < grid.height; r++) {
    const y = grid.y0 - (r + 0.5) * grid.res;
    const xs = [];
    for (const poly of polys) for (const ring of poly) for (let i = 0; i < ring.length - 1; i++) {
      const [x1, y1] = ring[i], [x2, y2] = ring[i + 1];
      if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) xs.push(x1 + ((y - y1) / (y2 - y1)) * (x2 - x1));
    }
    xs.sort((a, b) => a - b);
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const c0 = Math.max(0, Math.ceil((xs[i] - grid.x0) / grid.res - 0.5));
      const c1 = Math.min(grid.width - 1, Math.floor((xs[i + 1] - grid.x0) / grid.res - 0.5));
      for (let c = c0; c <= c1; c++) mask[r * grid.width + c] = 1;
    }
  }
  return mask;
}

// ------------------------------------------------------------------ scene selection (geopulse/data.py)

export function selectScenes(items, period, how = "closest") {
  if (how === "spread" && items.length > MAX_SCENES) {
    const idx = [];
    for (let i = 0; i < MAX_SCENES; i++) idx.push(Math.round((i * (items.length - 1)) / (MAX_SCENES - 1)));
    return [...new Set(idx)].map((i) => items[i]);
  }
  return period === "pre" ? items.slice(-MAX_SCENES) : items.slice(0, MAX_SCENES);
}

/** SAR change is only meaningful between like geometries: keep pre scenes from the post scenes' orbits. */
export function matchOrbits(found, how = "closest") {
  const orbit = (i) => i.properties?.["sat:relative_orbit"];
  const postOrbits = new Set(selectScenes(found.post, "post", how).map(orbit));
  const same = found.pre.filter((i) => postOrbits.has(orbit(i)));
  return same.length && same.length < found.pre.length ? { ...found, pre: same, restricted: [...postOrbits] } : found;
}

// ------------------------------------------------------------------ warping onto the grid

/** Fractional source-pixel coordinates (pixel-edge convention) of every grid pixel centre. Exact transforms on a
 *  lattice, bilinear in between: sub-centimetre at 10 m for AOIs of tens of km. */
export function sourceCoords(grid, srcEpsg, geo, step = 16) {
  const { width: W, height: H } = grid;
  const col = new Float32Array(W * H), row = new Float32Array(W * H);
  const toSrcPixel = (x, y) => [(x - geo.x0) / geo.rx, (y - geo.y0) / geo.ry];
  if (srcEpsg === grid.epsg) {
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const [u, v] = toSrcPixel(grid.x0 + (c + 0.5) * grid.res, grid.y0 - (r + 0.5) * grid.res);
      col[r * W + c] = u; row[r * W + c] = v;
    }
    return { col, row };
  }
  const g = projection(grid.epsg), s = projection(srcEpsg);
  const lc = [], lr = [];
  for (let c = 0; c < W; c += step) lc.push(c);
  if (lc[lc.length - 1] !== W - 1) lc.push(W - 1);
  for (let r = 0; r < H; r += step) lr.push(r);
  if (lr[lr.length - 1] !== H - 1) lr.push(H - 1);
  const LU = new Float64Array(lc.length * lr.length), LV = new Float64Array(lc.length * lr.length);
  lr.forEach((r, j) => lc.forEach((c, i) => {
    const [lon, lat] = g.inverse(grid.x0 + (c + 0.5) * grid.res, grid.y0 - (r + 0.5) * grid.res);
    const [u, v] = toSrcPixel(...s.forward(lon, lat));
    LU[j * lc.length + i] = u; LV[j * lc.length + i] = v;
  }));
  let j = 0;
  for (let r = 0; r < H; r++) {
    while (j < lr.length - 2 && r > lr[j + 1]) j++;
    const ty = lr[j + 1] === lr[j] ? 0 : (r - lr[j]) / (lr[j + 1] - lr[j]);
    let i = 0;
    for (let c = 0; c < W; c++) {
      while (i < lc.length - 2 && c > lc[i + 1]) i++;
      const tx = lc[i + 1] === lc[i] ? 0 : (c - lc[i]) / (lc[i + 1] - lc[i]);
      const a = j * lc.length + i, b = a + 1, d = a + lc.length, e = d + 1;
      col[r * W + c] = (1 - ty) * ((1 - tx) * LU[a] + tx * LU[b]) + ty * ((1 - tx) * LU[d] + tx * LU[e]);
      row[r * W + c] = (1 - ty) * ((1 - tx) * LV[a] + tx * LV[b]) + ty * ((1 - tx) * LV[d] + tx * LV[e]);
    }
  }
  return { col, row };
}

/** Source window [x0, y0, x1, y1] (clamped) that covers the coordinates, or null if they miss the image. */
export function windowFor(coords, imgW, imgH, margin = 2) {
  let u0 = Infinity, v0 = Infinity, u1 = -Infinity, v1 = -Infinity;
  for (let k = 0; k < coords.col.length; k++) {
    const u = coords.col[k], v = coords.row[k];
    if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v;
  }
  const x0 = Math.max(0, Math.floor(u0) - margin), y0 = Math.max(0, Math.floor(v0) - margin);
  const x1 = Math.min(imgW, Math.ceil(u1) + margin), y1 = Math.min(imgH, Math.ceil(v1) + margin);
  return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : null;
}

/** Resample a source window onto the grid. Invalid samples (NaN) are skipped and bilinear weights renormalised,
 *  like GDAL's warper with nodata. Pixels mapping outside the source image are NaN. */
export function resample(win, coords, imgW, imgH, method = "bilinear") {
  const { data, x0, y0, width: ww, height: wh } = win;
  const out = new Float32Array(coords.col.length).fill(NaN);
  const at = (i, j) => (i < x0 || j < y0 || i >= x0 + ww || j >= y0 + wh ? NaN : data[(j - y0) * ww + (i - x0)]);
  for (let k = 0; k < out.length; k++) {
    const u = coords.col[k], v = coords.row[k];
    if (!(u >= 0 && v >= 0 && u < imgW && v < imgH)) continue;
    if (method === "nearest") { out[k] = at(Math.floor(u), Math.floor(v)); continue; }
    const fu = u - 0.5, fv = v - 0.5, i0 = Math.floor(fu), j0 = Math.floor(fv), tx = fu - i0, ty = fv - j0;
    let acc = 0, wsum = 0;
    for (const [i, j, w] of [[i0, j0, (1 - tx) * (1 - ty)], [i0 + 1, j0, tx * (1 - ty)], [i0, j0 + 1, (1 - tx) * ty], [i0 + 1, j0 + 1, tx * ty]]) {
      if (w <= 0) continue;
      const val = at(Math.min(Math.max(i, 0), imgW - 1), Math.min(Math.max(j, 0), imgH - 1));
      if (!Number.isNaN(val)) { acc += w * val; wsum += w; }
    }
    if (wsum > 0) out[k] = acc / wsum;
  }
  return out;
}

// ------------------------------------------------------------------ compositing (geopulse/data.py)

/** scenes: array of per-scene band arrays (each Float32Array, time-ordered). Returns band arrays. */
export function composite(scenes, method, sensor, npx) {
  const nb = scenes[0].length, out = [];
  for (let b = 0; b < nb; b++) out.push(new Float32Array(npx).fill(NaN));
  const vals = new Float64Array(scenes.length);
  for (let k = 0; k < npx; k++) {
    if (method === "first") {
      let pick = 0;
      for (let s = 0; s < scenes.length; s++) {
        if (scenes[s].every((band) => !Number.isNaN(band[k]))) { pick = s; break; }
      }
      for (let b = 0; b < nb; b++) out[b][k] = scenes[pick][b][k];
    } else {
      for (let b = 0; b < nb; b++) {
        let m = 0;
        for (let s = 0; s < scenes.length; s++) { const v = scenes[s][b][k]; if (!Number.isNaN(v)) vals[m++] = v; }
        if (!m) continue;
        const sorted = Array.from(vals.subarray(0, m)).sort((x, y) => x - y);
        out[b][k] = m % 2 ? sorted[(m - 1) / 2] : (sorted[m / 2 - 1] + sorted[m / 2]) / 2;
      }
    }
  }
  if (sensor === "s1") for (const band of out) for (let k = 0; k < npx; k++) band[k] = 10 * Math.log10(band[k]);
  return out;
}

export function validFraction(bands) {
  let ok = 0;
  const npx = bands[0].length;
  for (let k = 0; k < npx; k++) if (bands.every((b) => !Number.isNaN(b[k]))) ok++;
  return ok / npx;
}

/** Elevation mosaic -> [elevation, slope in degrees] (numpy.gradient semantics). */
export function slopeDeg(elev, grid) {
  const { width: W, height: H, res } = grid;
  const out = new Float32Array(W * H);
  const e = (r, c) => elev[r * W + c];
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const gx = W < 2 ? 0 : c === 0 ? (e(r, 1) - e(r, 0)) / res : c === W - 1 ? (e(r, W - 1) - e(r, W - 2)) / res : (e(r, c + 1) - e(r, c - 1)) / (2 * res);
    const gy = H < 2 ? 0 : r === 0 ? (e(1, c) - e(0, c)) / res : r === H - 1 ? (e(H - 1, c) - e(H - 2, c)) / res : (e(r + 1, c) - e(r - 1, c)) / (2 * res);
    out[r * W + c] = Math.atan(Math.hypot(gx, gy)) / RAD;
  }
  return out;
}

// ------------------------------------------------------------------ physics baselines (geopulse/baseline.py)

const sig = (x) => 1 / (1 + Math.exp(-Math.min(Math.max(x, -30), 30)));
const fmax = (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.max(a, b)); // numpy.fmax: NaN loses
const nd = (a, b) => (a - b) / (a + b + 1e-6);
const both = (A, s) => `${s}_pre` in A && `${s}_post` in A;
const mndwi = (s2, k) => nd(s2[1][k], s2[4][k]);
const ndvi = (s2, k) => nd(s2[3][k], s2[2][k]);
const nbr = (s2, k) => nd(s2[3][k], s2[5][k]);
export const binaryEntropy = (p) => {
  const q = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return -(q * Math.log2(q) + (1 - q) * Math.log2(1 - q));
};

function waterProb(A, period, k) {
  const P = PHYSICS, out = {};
  if (`s1_${period}` in A) {
    const vv = A[`s1_${period}`][0][k];
    let p = sig((P.VV_WATER_DB - vv) / 1.5);
    if (period === "post" && "s1_pre" in A) {
      const q = sig((A.s1_pre[0][k] - vv - P.VV_DROP_DB) / 1.0) * sig((P.VV_DROP_CEILING_DB - vv) / 1.5);
      p = fmax(p, q);
    }
    out.s1 = p;
  }
  if (`s2_${period}` in A) out.s2 = sig((mndwi(A[`s2_${period}`], k) - P.MNDWI_WATER) / 0.05);
  return out;
}

const vhDrop = (A, k) => {
  const pre = A.s1_pre[1][k], post = A.s1_post[1][k];
  return sig((pre - post - PHYSICS.VH_DROP_DB) / 0.6) * sig((pre - PHYSICS.VH_CANOPY_DB) / 1.0);
};

function nanmean(vals) {
  let s = 0, m = 0;
  for (const v of vals) if (!Number.isNaN(v)) { s += v; m++; }
  return m ? s / m : NaN;
}

export function severityClass(A, k) {
  const d = nbr(A.s2_pre, k) - nbr(A.s2_post, k);
  if (Number.isNaN(d)) return IGNORE;
  return PHYSICS.DNBR_SEVERITY.filter((t) => d >= t).length;
}

export function preWater(A, k, threshold = 0.8) {
  const p = waterProb(A, "pre", k);
  let w = "s1" in p ? p.s1 : NaN;
  if ("s2" in p && !Number.isNaN(p.s2)) w = p.s2;
  return (Number.isNaN(w) ? 0 : w) > threshold;
}

/** target / change / uncertainty (+ severity for wildfire), NaN where nothing was observed. */
export function baselinePredict(A, task, npx) {
  const P = PHYSICS;
  const target = new Float32Array(npx), change = new Float32Array(npx), unc = new Float32Array(npx);
  const sev = task === "wildfire" && both(A, "s2") ? new Uint8Array(npx) : null;
  for (let k = 0; k < npx; k++) {
    let t, u;
    if (task === "flood") {
      const pre = waterProb(A, "pre", k), post = waterProb(A, "post", k);
      const wPre = nanmean(Object.values(pre));
      t = nanmean(Object.values(post)) * (1 - (Number.isNaN(wPre) ? 0 : wPre));
      u = binaryEntropy(t);
      if ("s1" in post && "s2" in post) u = fmax(u, Math.abs(post.s1 - post.s2));
    } else if (task === "wildfire") {
      const s2 = both(A, "s2") ? sig((nbr(A.s2_pre, k) - nbr(A.s2_post, k) - P.DNBR_BURN) / 0.025) * sig((ndvi(A.s2_pre, k) - P.NDVI_FUEL) / 0.03) : NaN;
      const s1 = both(A, "s1") ? vhDrop(A, k) : NaN;
      const sarOnly = Number.isNaN(s2) && !Number.isNaN(s1);
      t = sarOnly ? s1 : s2;
      u = sarOnly ? fmax(binaryEntropy(t), P.SAR_ONLY_BURN_UNCERTAINTY) : binaryEntropy(t);
    } else {
      const p = {};
      if (both(A, "s2")) {
        const pre = ndvi(A.s2_pre, k);
        p.s2 = sig((pre - ndvi(A.s2_post, k) - P.NDVI_DROP) / 0.04) * sig((pre - P.NDVI_CANOPY) / 0.05);
      }
      if (both(A, "s1")) p.s1 = vhDrop(A, k);
      t = nanmean(Object.values(p));
      u = binaryEntropy(t);
      if ("s1" in p && "s2" in p) u = fmax(u, Math.abs(p.s1 - p.s2));
    }
    const ch = [];
    if (both(A, "s1")) {
      const d = Math.max(Math.abs(A.s1_post[0][k] - A.s1_pre[0][k]), Math.abs(A.s1_post[1][k] - A.s1_pre[1][k]));
      ch.push(sig((d - P.CHANGE_DB) / 0.5));
    }
    if (both(A, "s2")) {
      const d = Math.max(...[mndwi, ndvi, nbr].map((f) => Math.abs(f(A.s2_post, k) - f(A.s2_pre, k))));
      ch.push(sig((d - P.CHANGE_INDEX) / 0.05));
    }
    target[k] = t;
    change[k] = ch.length ? nanmean(ch) : t;
    unc[k] = Number.isNaN(t) ? NaN : u;
    if (sev) sev[k] = severityClass(A, k);
  }
  return { target, change, uncertainty: unc, severity: sev };
}

// ------------------------------------------------------------------ learned model inputs (geopulse/model.py)

function nanmedianArray(a) {
  const v = Array.from(a).filter((x) => !Number.isNaN(x)).sort((x, y) => x - y);
  if (!v.length) return NaN;
  return v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
}

/** Normalised model inputs, channel-major Float32Array per input (to_tensors). */
export function toTensors(A, sensors, npx) {
  const out = {};
  for (const [s, nb] of Object.entries(SENSORS)) for (const t of TIMES) {
    const key = `${s}_${t}`, x = new Float32Array(nb * npx), v = new Float32Array(npx);
    if (key in A && sensors.includes(s)) {
      const bands = A[key], { mean, std } = NORM[s];
      for (let k = 0; k < npx; k++) {
        if (!bands.every((b) => !Number.isNaN(b[k]))) continue;
        v[k] = 1;
        for (let b = 0; b < nb; b++) x[b * npx + k] = (bands[b][k] - mean[b]) / std[b];
      }
    }
    out[key] = x; out[`${key}_valid`] = v;
  }
  const dem = new Float32Array(2 * npx);
  if ("dem" in A) {
    const [elev, slope] = A.dem, med = nanmedianArray(elev);
    for (let k = 0; k < npx; k++) {
      const r = (elev[k] - med) / 20, s = slope[k] / 10;
      dem[k] = Number.isNaN(r) ? 0 : r;
      dem[npx + k] = Number.isNaN(s) ? 0 : s;
    }
  }
  out.dem = dem;
  return out;
}

/** Window origins covering n (last one shifted inward), as geopulse.grid._starts. */
export function starts(n, size, stride) {
  if (n <= size) return [0];
  const out = [];
  for (let s = 0; s < n - size; s += stride) out.push(s);
  out.push(n - size);
  return out;
}

// ------------------------------------------------------------------ outputs (geopulse/pipeline.py)

export const TARGET_COLORS = {
  flood: [[40, 120, 255], [20, 90, 255]],
  burn: [[235, 70, 30], [200, 20, 10]],
  disturbance: [[190, 70, 235], [150, 30, 220]],
};
export const PALETTES = {
  change: [[0.2, [255, 170, 0, 0]], [0.5, [255, 170, 0, 140]], [1.0, [255, 120, 0, 230]]],
  uncertainty: [[0.0, [60, 20, 90, 0]], [0.3, [120, 40, 140, 110]], [0.7, [230, 80, 90, 190]], [1.0, [255, 210, 60, 230]]],
};
export const SEVERITY_RGBA = [[0, 0, 0, 0], [127, 255, 212, 220], [255, 255, 0, 220], [255, 140, 0, 230], [255, 0, 0, 235]];
export const SEVERITY_CLASSES = ["low", "moderate-low", "moderate-high", "high"];

export const probPalette = ([r, g, b], [dr, dg, db]) => [[0.2, [r, g, b, 0]], [0.5, [r, g, b, 150]], [1.0, [dr, dg, db, 235]]];

/** Piecewise-linear RGBA colormap (numpy.interp with left=0); NaN -> transparent. */
export function ramp(values, stops) {
  const out = new Uint8ClampedArray(values.length * 4);
  for (let k = 0; k < values.length; k++) {
    const v = values[k];
    if (Number.isNaN(v) || v < stops[0][0]) continue;
    let i = 0;
    while (i < stops.length - 1 && v > stops[i + 1][0]) i++;
    const [x0, c0] = stops[i], [x1, c1] = stops[Math.min(i + 1, stops.length - 1)];
    const t = v >= x1 ? 1 : (v - x0) / (x1 - x0);
    for (let ch = 0; ch < 4; ch++) out[k * 4 + ch] = c0[ch] + t * (c1[ch] - c0[ch]);
  }
  return out;
}

export function rgbImage(s2, npx, mask) {
  const out = new Uint8ClampedArray(npx * 4);
  for (let k = 0; k < npx; k++) {
    if (mask && !mask[k]) continue;
    let any = false;
    [2, 1, 0].forEach((b, ch) => {
      const v = s2[b][k];
      const x = Number.isNaN(v) ? 0 : Math.pow(Math.min(Math.max(v / 0.3, 0), 1), 1 / 1.4) * 255;
      out[k * 4 + ch] = x;
      any ||= x > 0;
    });
    out[k * 4 + 3] = any ? 255 : 0;
  }
  return out;
}

export function grayImage(s1, npx, mask) {
  const out = new Uint8ClampedArray(npx * 4);
  for (let k = 0; k < npx; k++) {
    if (mask && !mask[k]) continue;
    const v = s1[0][k];
    const x = Number.isNaN(v) ? 0 : Math.min(Math.max((v + 25) / 25, 0), 1) * 255;
    out[k * 4] = out[k * 4 + 1] = out[k * 4 + 2] = x;
    out[k * 4 + 3] = x > 0 ? 255 : 0;
  }
  return out;
}

/** Reproject an RGBA grid image to Web Mercator (nearest) so MapLibre can drape it; returns lon/lat bounds. */
export function toWebMercator(rgba, grid, step = 16) {
  const g = projection(grid.epsg);
  const { width: W, height: H, res, x0, y0 } = grid;
  let mx0 = Infinity, my0 = Infinity, mx1 = -Infinity, my1 = -Infinity;
  for (let i = 0; i <= 20; i++) {
    const t = i / 20;
    for (const [x, y] of [[x0 + t * W * res, y0], [x0 + t * W * res, y0 - H * res], [x0, y0 - t * H * res], [x0 + W * res, y0 - t * H * res]]) {
      const [mx, my] = mercator.forward(...g.inverse(x, y));
      mx0 = Math.min(mx0, mx); my0 = Math.min(my0, my); mx1 = Math.max(mx1, mx); my1 = Math.max(my1, my);
    }
  }
  const dw = W, dh = Math.max(1, Math.round((W * (my1 - my0)) / (mx1 - mx0)));
  const out = new Uint8ClampedArray(dw * dh * 4);
  // lattice of grid-pixel coordinates for mercator pixel centres
  const lc = [], lr = [];
  for (let c = 0; c < dw; c += step) lc.push(c);
  if (lc[lc.length - 1] !== dw - 1) lc.push(dw - 1);
  for (let r = 0; r < dh; r += step) lr.push(r);
  if (lr[lr.length - 1] !== dh - 1) lr.push(dh - 1);
  const LU = [], LV = [];
  for (const r of lr) for (const c of lc) {
    const mx = mx0 + ((c + 0.5) / dw) * (mx1 - mx0), my = my1 - ((r + 0.5) / dh) * (my1 - my0);
    const [x, y] = g.forward(...mercator.inverse(mx, my));
    LU.push((x - x0) / res); LV.push((y0 - y) / res);
  }
  let j = 0;
  for (let r = 0; r < dh; r++) {
    while (j < lr.length - 2 && r > lr[j + 1]) j++;
    const ty = lr[j + 1] === lr[j] ? 0 : (r - lr[j]) / (lr[j + 1] - lr[j]);
    let i = 0;
    for (let c = 0; c < dw; c++) {
      while (i < lc.length - 2 && c > lc[i + 1]) i++;
      const tx = lc[i + 1] === lc[i] ? 0 : (c - lc[i]) / (lc[i + 1] - lc[i]);
      const a = j * lc.length + i, b = a + 1, d = a + lc.length, e = d + 1;
      const u = (1 - ty) * ((1 - tx) * LU[a] + tx * LU[b]) + ty * ((1 - tx) * LU[d] + tx * LU[e]);
      const v = (1 - ty) * ((1 - tx) * LV[a] + tx * LV[b]) + ty * ((1 - tx) * LV[d] + tx * LV[e]);
      const sc = Math.floor(u), sr = Math.floor(v);
      if (sc < 0 || sr < 0 || sc >= W || sr >= H) continue;
      out.set(rgba.subarray((sr * W + sc) * 4, (sr * W + sc) * 4 + 4), (r * dw + c) * 4);
    }
  }
  const [w, s] = mercator.inverse(mx0, my0), [e, n] = mercator.inverse(mx1, my1);
  return { rgba: out, width: dw, height: dh, bounds: [w, s, e, n] };
}

/** summary.json content, as geopulse.pipeline.write_outputs builds it. */
export function summarize({ request, grid, maps, scenes, warnings, card, runtime, engine }) {
  const spec = TASKS[request.task], pxKm2 = grid.res ** 2 / 1e6;
  const { target: p, uncertainty: u, severity } = maps;
  let observed = 0, hit = 0, review = 0, conf = 0, usum = 0;
  for (let k = 0; k < p.length; k++) {
    if (Number.isNaN(p[k])) continue;
    observed++; usum += u[k];
    if (p[k] >= THRESHOLD) { hit++; conf += Math.max(p[k], 1 - p[k]); }
    if (u[k] >= REVIEW_UNCERTAINTY) review++;
  }
  const r3 = (x) => Math.round(x * 1000) / 1000;
  const s = {
    task: request.task, target: spec.target, affected_label: spec.verb,
    model: { id: card.model_id, version: card.version }, aoi_km2: request.aoi_km2,
    observed_km2: r3(observed * pxKm2), affected_km2: r3(hit * pxKm2), review_km2: r3(review * pxKm2),
    mean_confidence_affected: hit ? r3(conf / hit) : null, mean_uncertainty: observed ? r3(usum / observed) : null,
    modality_weights: maps.weights || null, scenes: Object.fromEntries(Object.entries(scenes).map(([k, v]) => [k, v.length])),
    warnings, runtime_s: Math.round(runtime * 10) / 10, engine,
    decision_policy: { threshold: THRESHOLD, review_if_uncertainty_ge: REVIEW_UNCERTAINTY },
  };
  if (severity) {
    s.severity_km2 = Object.fromEntries(SEVERITY_CLASSES.map((name, i) => {
      let c = 0;
      for (let k = 0; k < severity.length; k++) if (severity[k] === i + 1) c++;
      return [name, r3(c * pxKm2)];
    }));
  }
  return s;
}

// Fixed input for the ONNX self-check (identical to model.py::probe_values): validity and dropout masks are 1,
// everything else a deterministic pattern in [-0.5, 0.5).
export function probeValues(name, dims) {
  const n = dims.reduce((a, b) => a * b, 1), out = new Float32Array(n);
  const ones = name.endsWith("_valid") || name.startsWith("mask_");
  for (let i = 0; i < n; i++) out[i] = ones ? 1 : ((i * 7919) % 1000) / 1000 - 0.5;
  return out;
}
