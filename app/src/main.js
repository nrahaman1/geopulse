import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import mapWorker from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import * as E from "./engine.js";
import "./style.css";

const $ = (id) => document.getElementById(id);
// Two compute engines. "browser": worker.js runs the whole pipeline on this machine (STAC + COG reads, compositing,
// physics, ONNX model on WebGPU/WebAssembly). "server": GeoPulse's Python engine (PyTorch on CUDA/Metal/CPU), either
// the one the desktop app runs on this PC or a `geopulse serve` that serves this page. GitHub Pages has only "browser".
const TAURI = window.__TAURI__; // set inside the desktop app
const RELEASES = "https://github.com/nrahaman1/geopulse/releases/latest";
const BROWSER_MAX_KM2 = 300;
const BASELINE = { model_id: "threshold-baseline", version: "1.1.0", tasks: ["flood", "wildfire", "vegetation"] };
let SERVER = false, SERVER_URL = "", TOKEN = "", SERVER_INFO = {}, ENGINE = "browser", worker = null, VERSION = "?";
let browserModels = [BASELINE], serverModels = [];
const BROWSER_JOBS = new Map(); // id -> { job, res: { summary, layers, files: {name: Blob} } }, oldest first
const KEEP_JOBS = 10;
const withToken = (url) => (TOKEN ? `${url}${url.includes("?") ? "&" : "?"}token=${TOKEN}` : url);
const api = async (path, opts = {}) => {
  const auth = TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {};
  const r = await fetch(SERVER_URL + path, { ...opts, headers: { ...opts.headers, ...auth } });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
};

const basemap = () => ({
  version: 8,
  sources: {
    dark: { type: "raster", tileSize: 256, maxzoom: 16, attribution: "Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors",
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"] },
    imagery: { type: "raster", tileSize: 256, maxzoom: 19, attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"] },
  },
  layers: [
    { id: "dark", type: "raster", source: "dark" },
    { id: "imagery", type: "raster", source: "imagery", layout: { visibility: "none" } },
  ],
});

maplibregl.setWorkerUrl(mapWorker); // bundled: MapLibre would look for it next to its own (bundled) module
const view = { center: [-40, 30], zoom: 2 };
const map = new maplibregl.Map({ container: "map", style: basemap(), ...view, attributionControl: { compact: true } });
const before = new maplibregl.Map({ container: "map-before", style: basemap(), ...view, attributionControl: false });
map.addControl(new maplibregl.NavigationControl(), "bottom-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

// Keep the two maps in lockstep (the "before" map is only visible in swipe mode).
let syncing = false;
const sync = (from, to) => from.on("move", () => {
  if (syncing) return;
  syncing = true;
  to.jumpTo({ center: from.getCenter(), zoom: from.getZoom(), bearing: from.getBearing(), pitch: from.getPitch() });
  syncing = false;
});
sync(map, before); sync(before, map);

// ------------------------------------------------------------------ AOI
let aoi = null;
const emptyFC = { type: "FeatureCollection", features: [] };
// Resolves once, after the first "load". map.loaded() can read false again during redraws, when "load" never refires.
const mapReady = new Promise((resolve) => map.once("load", resolve));
map.on("load", () => {
  map.addSource("aoi", { type: "geojson", data: aoi ?? emptyFC }); // an AOI chosen before the map loaded shows now
  map.addLayer({ id: "aoi-line", type: "line", source: "aoi", paint: { "line-color": "#3fb6c8", "line-width": 2, "line-dasharray": [2, 1] } });
});

function areaKm2(geom) {
  const ring = (geom.type === "Polygon" ? geom.coordinates : geom.coordinates[0])[0];
  const R = 6371.0088, rad = Math.PI / 180;
  let a = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1, y1] = ring[i], [x2, y2] = ring[i + 1];
    a += (x2 - x1) * rad * (2 + Math.sin(y1 * rad) + Math.sin(y2 * rad));
  }
  return Math.abs(a * R * R / 2);
}

function setAoi(geom, fit = true) {
  if (geom.type === "FeatureCollection") geom = geom.features[0].geometry;
  if (geom.type === "Feature") geom = geom.geometry;
  aoi = geom;
  map.getSource("aoi")?.setData(geom);
  $("aoi-info").textContent = `${geom.type} · ${areaKm2(geom).toFixed(1)} km²`;
  if (fit) {
    const pts = (geom.type === "Polygon" ? [geom.coordinates] : geom.coordinates).flat(2);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 80, duration: 800 });
  }
}

// ------------------------------------------------------------------ place search
// Nominatim (OpenStreetMap) on submit only: its usage policy forbids search-as-you-type. Coordinates need no lookup.
const COORDS = /^\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*$/;
const AOI_KM = 10;
function boxAround(lon, lat, km = AOI_KM) {
  const dLat = km / 2 / 110.574, dLon = km / 2 / (111.32 * Math.cos((lat * Math.PI) / 180));
  return { type: "Polygon", coordinates: [[[lon - dLon, lat - dLat], [lon + dLon, lat - dLat], [lon + dLon, lat + dLat], [lon - dLon, lat + dLat], [lon - dLon, lat - dLat]]] };
}
function goTo(name, lon, lat) {
  $("search-results").replaceChildren();
  $("example").value = "";
  setAoi(boxAround(lon, lat));
  $("aoi-info").textContent = `${AOI_KM} × ${AOI_KM} km box around ${name} — draw a box to change it`;
}
function searchMessage(text) {
  $("search-results").replaceChildren(Object.assign(document.createElement("div"), { className: "msg", textContent: text }));
}
$("search").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("search-q").value.trim();
  if (!q) return;
  const m = q.match(COORDS);
  if (m && Math.abs(+m[1]) <= 90 && Math.abs(+m[2]) <= 180) return goTo(`${+m[1]}, ${+m[2]}`, +m[2], +m[1]);
  searchMessage("Searching…");
  try {
    const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&q=${encodeURIComponent(q)}`;
    const r = await fetch(url, { headers: { "Accept-Language": navigator.language || "en" } });
    if (!r.ok) throw new Error(r.statusText || `HTTP ${r.status}`);
    const hits = await r.json();
    if (!hits.length) return searchMessage(`No place found for “${q}”. Try adding a region or country.`);
    if (hits.length === 1) return goTo(hits[0].name || q, +hits[0].lon, +hits[0].lat);
    $("search-results").replaceChildren(
      ...hits.map((h) => Object.assign(document.createElement("button"), {
        type: "button", textContent: h.display_name, onclick: () => goTo(h.name || h.display_name.split(",")[0], +h.lon, +h.lat),
      })),
      Object.assign(document.createElement("div"), { className: "attr", textContent: "Places © OpenStreetMap contributors (Nominatim)" }),
    );
  } catch (err) {
    searchMessage(`Place search unavailable (${err.message}). Coordinates like 39.76, -121.62 always work.`);
  }
};
$("search-q").onkeydown = (e) => { if (e.key === "Escape") $("search-results").replaceChildren(); };

let drawing = null;
$("draw").onclick = () => {
  drawing = { start: null };
  map.getCanvas().style.cursor = "crosshair";
  map.dragPan.disable();
  $("aoi-info").textContent = "Drag on the map to draw a box.";
};
const box = (a, b) => ({ type: "Polygon", coordinates: [[[a.lng, a.lat], [b.lng, a.lat], [b.lng, b.lat], [a.lng, b.lat], [a.lng, a.lat]]] });
map.on("mousedown", (e) => { if (drawing) drawing.start = e.lngLat; });
map.on("mousemove", (e) => { if (drawing?.start) map.getSource("aoi").setData(box(drawing.start, e.lngLat)); });
map.on("mouseup", (e) => {
  if (!drawing?.start) return;
  const g = box(drawing.start, e.lngLat);
  drawing = null;
  map.dragPan.enable();
  map.getCanvas().style.cursor = "";
  $("example").value = "";
  setAoi(g, false);
});

$("upload-btn").onclick = () => $("upload").click();
$("upload").onchange = async (e) => {
  try { setAoi(JSON.parse(await e.target.files[0].text())); $("example").value = ""; }
  catch (err) { $("aoi-info").textContent = `Could not read GeoJSON: ${err.message}`; }
};

// ------------------------------------------------------------------ examples, models, jobs
let examples = [];
const TASK_ICON = { flood: "💧", wildfire: "🔥", vegetation: "🌲" };
const TASK_HINT = {
  flood: "Before: a dry period. After: the days right after the event (earliest clear view is used).",
  wildfire: "Before: weeks before ignition. After: after containment (post-fire median). Severity from dNBR.",
  vegetation: "Use the SAME season in different years (e.g. Jun–Aug 2019 vs Jun–Aug 2021) so phenology cancels.",
};
const TARGET = {
  flood: ["Flood", "#2878ff", "#145aff"],
  wildfire: ["Burn", "#eb461e", "#c8140a"],
  vegetation: ["Disturbance", "#be46eb", "#961edc"],
};
function setTask(task) {
  $("task").value = task;
  $("task-hint").textContent = TASK_HINT[task];
  const keep = $("model").value;
  $("model").replaceChildren(new Option("auto", "auto"));
  const models = ENGINE === "browser" ? browserModels : serverModels;
  for (const m of models.filter((m) => m.tasks.includes(task))) $("model").add(new Option(`${m.model_id} v${m.version}`, m.model_id));
  $("model").value = [...$("model").options].some((o) => o.value === keep) ? keep : "auto";
}
$("task").onchange = () => setTask($("task").value);
const setDates = (w, a, b) => { const [x, y] = Array.isArray(w) ? w : w.split("/"); $(a).value = x; $(b).value = y; };
const WEBGPU = !!self.navigator.gpu;
function setEngine(engine) {
  ENGINE = engine;
  document.querySelector(`input[name=engine][value=${engine}]`).checked = true;
  const where = WEBGPU ? "GPU (WebGPU)" : "CPU (WebAssembly)";
  $("engine-hint").innerHTML = engine === "browser"
    ? `${TAURI ? "Built-in engine" : "Runs on your computer"}: imagery streams straight from the Microsoft Planetary Computer and the model runs on your ${where}. Nothing is uploaded. Areas up to ${BROWSER_MAX_KM2} km².`
      + (TAURI ? "" : ` For your full GPU and larger areas, <a href="${RELEASES}" target="_blank" rel="noopener">get the desktop app</a>.`)
    : `${TAURI ? "GeoPulse's Python engine on this PC" : "The GeoPulse server that serves this page"}: ${SERVER_INFO.device}. Full pipeline, areas up to ${SERVER_INFO.max_job_km2} km².`;
  $("version").textContent = `v${VERSION} · ${engine === "browser" ? (WEBGPU ? "WebGPU" : "WebAssembly") : SERVER_INFO.device.split(" · ")[0]}`;
  setTask($("task").value || "flood");
}
for (const el of document.querySelectorAll("input[name=engine]")) el.onchange = () => setEngine(el.value);

async function connectServer() {
  SERVER_INFO = await api("/health");
  serverModels = await api("/models");
  SERVER = true;
  loadCases();
  $("engine-server").hidden = false;
  $("engine-server").querySelector("input").disabled = false;
  $("server-label").textContent = `${TAURI ? "This PC" : "GeoPulse server"} · ${SERVER_INFO.device}`;
  setEngine("server");
  refreshJobs();
}

// Desktop app: GeoPulse's Python engine on this PC. The first start installs Python and PyTorch (CUDA when an NVIDIA
// GPU is present) with uv into the app's data folder; later starts take seconds. The built-in engine works meanwhile.
async function desktopEngine(install = false) {
  const { invoke } = TAURI.core;
  const status = await invoke("engine_status");
  $("engine-server").hidden = false;
  $("engine-server").querySelector("input").disabled = true;
  $("server-label").textContent = `This PC · Python engine${status.gpu ? ` · ${status.gpu}` : ""}`;
  if (!status.installed && !install) {
    $("engine-setup").hidden = false;
    $("engine-start").textContent = `Install the PC engine (one-time, ~${status.download})`;
    $("engine-start").title = `Python and PyTorch (${status.gpu ? "CUDA, for your NVIDIA GPU" : "CPU"}) into the app's data folder`;
    return;
  }
  $("engine-setup").hidden = true;
  const lines = [status.installed ? "Starting the GeoPulse engine on this PC…" : "Installing the GeoPulse engine on this PC (one time)…"];
  const quiet = status.installed; // routine starts stay out of the log unless they fail
  if (!quiet) logLines(lines);
  engineBusy(lines[0]);
  const unlisten = await TAURI.event.listen("engine-log", ({ payload }) => {
    lines.push(payload);
    if (!quiet) logLines(lines.slice(-300));
  });
  const unstage = await TAURI.event.listen("engine-stage", ({ payload }) => engineBusy(payload));
  try {
    ({ url: SERVER_URL, token: TOKEN } = await invoke("engine_start", { variant: status.variant }));
    await connectServer();
    if (!quiet) logLines([...lines, `✓ GeoPulse engine ready: ${SERVER_INFO.device}`]);
  } catch (err) {
    logLines([...lines.slice(-40), `✗ The PC engine did not start: ${err}`]);
    $("engine-setup").hidden = false;
    $("engine-start").textContent = "Retry the PC engine";
  } finally {
    unlisten();
    unstage();
    engineBusy(null);
  }
}

// Under "This PC" while its engine starts: a spinner, the current step and the seconds so far, so a start that takes
// a while (PyTorch loading, or installing what an app update changed) never looks frozen.
let busyTimer = null;
function engineBusy(step) {
  $("engine-status").hidden = !step;
  $("engine-status-text").textContent = step ?? "";
  if (step && !busyTimer) {
    const t0 = Date.now();
    $("engine-status-time").textContent = "";
    busyTimer = setInterval(() => { $("engine-status-time").textContent = `${Math.round((Date.now() - t0) / 1000)} s`; }, 1000);
  } else if (!step) {
    clearInterval(busyTimer);
    busyTimer = null;
  }
}
$("engine-start").onclick = () => desktopEngine(true);

// Desktop app: a new release (signed; the app checks the signature) downloads in the background and installs once no
// analysis or engine setup is running; the installer then restarts GeoPulse.
async function selfUpdate(engineReady) {
  const update = await TAURI.updater.check().catch(() => null);
  if (!update) return;
  await update.download();
  await engineReady.catch(() => {});
  while ($("run").disabled) await new Promise((r) => setTimeout(r, 15000));
  logLines([`Installing GeoPulse ${update.version}: the app restarts in a moment…`]);
  await TAURI.core.invoke("engine_stop");
  await update.install(); // Windows: the installer takes over and reopens the app
  await TAURI.process.relaunch(); // macOS and Linux
}

async function init() {
  VERSION = (await fetch("site.json").then((r) => r.json()).catch(() => ({}))).version ?? "?";
  const index = await fetch("weights/index.json").then((r) => (r.ok ? r.json() : [])).catch(() => []);
  // The checksum in the URL makes a changed model a new URL, so browser and service-worker caches never go stale.
  browserModels = [...index.map((c) => ({ ...c, tasks: c.onnx.tasks, url: new URL(`weights/${c.onnx.file}?v=${c.onnx.sha256.slice(0, 12)}`, document.baseURI).href })), BASELINE];
  setEngine("browser");
  examples = await fetch("examples.json").then((r) => r.json()).catch(() => []);
  fillExamples(examples[0]?.id ?? "");
  $("example").onchange();
  loadCases();
  await loadBrowserJobs();
  refreshJobs();
  if (TAURI) selfUpdate(desktopEngine()).catch((err) => console.warn("update:", err));
  else if (await fetch("health").then((r) => r.ok && r.headers.get("content-type")?.includes("json")).catch(() => false)) await connectServer();
}
$("example").onchange = () => {
  const id = $("example").value;
  const ex = examples.find((x) => x.id === id) ?? added.find((x) => x.id === id);
  showCase(added.find((x) => x.id === id));
  if (!ex) return;
  setAoi(ex.aoi); // the AOI is set at once; the map draws it when it has loaded (slow or blocked basemaps included)
  setTask(ex.task || "flood");
  if (ex.before) { setDates(ex.before, "b0", "b1"); setDates(ex.after, "a0", "a1"); }
  if (ex.sensors) for (const k of ["s1", "s2"]) $(k).checked = ex.sensors.includes(k);
};

// ------------------------------------------------------------------ Scout: recent events
// Events the Scout found in official alerts (GDACS, Copernicus EMS, NASA EONET) and the news. "Recent events" lists
// them with their sources; "Add to examples" copies one into the example list (kept in this browser or app), where it
// fills the form like any example. Nothing runs until "Run analysis".
let cases = [], scoutStatus = null;
const LIVE = "https://nrahaman1.github.io/geopulse/"; // the live Scout publishes here every 6 hours
const SCOUT_DOC = "https://github.com/nrahaman1/geopulse/blob/main/docs/SCOUT.md";
const getJson = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))));
let added = (() => { try { return JSON.parse(localStorage.getItem("geopulse.added") || "[]"); } catch { return []; } })();
const saveAdded = () => { try { localStorage.setItem("geopulse.added", JSON.stringify(added)); } catch {} };
const ago = (iso) => {
  const h = Math.round((Date.now() - Date.parse(iso)) / 3.6e6);
  return h < 1 ? "less than an hour ago" : h < 48 ? `${h} h ago` : `${Math.round(h / 24)} days ago`;
};

function fillExamples(selected = $("example").value) {
  const option = (x) => new Option(`${TASK_ICON[x.task] || ""} ${x.title || x.id}`, x.id);
  $("example").replaceChildren(new Option("— choose —", ""), ...examples.map(option));
  if (added.length) $("example").append(el("optgroup", { label: "Added from the Scout" }, ...added.map(option)));
  $("example").value = selected;
}

async function loadCases() {
  // The live Scout's cases are part of the GitHub Pages build; the desktop app and `geopulse serve` fetch them from
  // there. A Scout run on this machine (`geopulse scout discover`) adds its own through the engine.
  const [live, status] = await Promise.all(["cases.json", "scout-status.json"].map((f) =>
    getJson(f).catch(() => getJson(LIVE + f)).catch(() => null)));
  const local = SERVER ? await api("/scout/cases").catch(() => []) : [];
  const byId = new Map([...(live ?? []), ...local].map((c) => [c.id, c]));
  cases = [...byId.values()];
  scoutStatus = status;
  added = added.map((a) => byId.get(a.id) ?? a); // the Scout's latest view of an added event (new imagery, sources)
  saveAdded();
  fillExamples();
  $("scout-open").textContent = `📡 Recent events${cases.length ? ` (${cases.length})` : ""}`;
  if (status?.finished) $("scout-hint").textContent = `Floods, wildfires and forest loss the GeoPulse Scout found in alerts and the news; updated ${ago(status.finished)}.`;
  if ($("scout").open) renderScout();
}

// News titles and quotes are untrusted text: built with textContent, never as HTML.
function el(tag, props = {}, ...kids) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...kids);
  return node;
}
const SOURCE = { gdacs: "GDACS alert", cems: "Copernicus EMS activation", eonet: "NASA EONET" };
function caseDetails(c) {
  const sc = c.scenes;
  const when = c.event_start && (c.event_end && c.event_end !== c.event_start ? `${c.event_start} to ${c.event_end}` : c.event_start);
  const at = c.lat != null && `${Math.abs(c.lat).toFixed(2)}°${c.lat < 0 ? "S" : "N"} ${Math.abs(c.lon).toFixed(2)}°${c.lon < 0 ? "W" : "E"}`;
  const facts = [when && `event ${when}`, at, c.aoi_km2 && `${Math.round(c.aoi_km2)} km²`,
    sc && `S1 ${sc.s1_before}→${sc.s1_after}, S2 ${sc.s2_before}→${sc.s2_after} scenes (before→after)`];
  const sources = c.sources.map((s) => {
    const link = /^https?:\/\//.test(s.url ?? "") ? el("a", { href: s.url, target: "_blank", rel: "noopener", textContent: s.title || s.url }) : el("span", { textContent: s.title ?? "" });
    const who = s.kind === "news" ? s.outlet : SOURCE[s.kind] ?? s.kind;
    return el("li", {}, `${who} · ${s.date ?? ""} · `, link, ...(s.quote ? [el("q", { textContent: s.quote })] : []));
  });
  return [
    el("div", { className: "case-head" }, el("span", { className: `conf-${c.confidence}`, textContent: `${c.confidence} confidence` }), ` · ${c.status}`),
    el("div", { textContent: facts.filter(Boolean).join(" · ") }),
    el("div", { className: "src-label", textContent: c.sources.length > 1 ? `${c.sources.length} sources` : "Source" }),
    el("ul", {}, ...sources),
  ];
}
function showCase(c) {
  $("case-info").hidden = !c;
  if (!c) return;
  const remove = el("button", { type: "button", textContent: "Remove from examples" });
  remove.onclick = () => { added = added.filter((a) => a.id !== c.id); saveAdded(); fillExamples(""); showCase(null); };
  $("case-info").replaceChildren(...caseDetails(c), el("div", { className: "btns" }, remove));
}

function scoutAbout() {
  const intro = "Every 6 hours the Scout reads official disaster alerts and the news. An open-weights language model "
    + "reads each article; code then checks every quote word for word against the article and locates the place "
    + "with OpenStreetMap. Each event below lists where it comes from. ";
  const how = el("a", { href: SCOUT_DOC, target: "_blank", rel: "noopener", textContent: "How the Scout works ↗" });
  const s = scoutStatus;
  if (!s?.sources) return [intro, how];
  const name = { gdacs: "GDACS", cems: "Copernicus EMS", eonet: "NASA EONET", gdelt: "GDELT" };
  const runs = Object.entries(s.sources).map(([k, v]) => `${name[k] ?? k.replace(/^rss:/, "")} `
    + (v.ok ? `✓ ${v.events ?? v.articles ?? 0} ${"events" in v ? "alerts" : "articles"}` : "✗ unavailable"));
  const news = s.news ? ` · ${s.news.read} articles read by ${s.news.model}` : "";
  return [intro, how, el("div", { className: "run", textContent: `Last run ${ago(s.finished)}: ${runs.join(" · ")}${news}` })];
}
function renderScout() {
  const hazard = document.querySelector("input[name=scout-hazard]:checked").value;
  const shown = cases.filter((c) => !hazard || c.task === hazard);
  $("scout-about").replaceChildren(...scoutAbout());
  $("scout-list").replaceChildren(...(shown.length ? shown.map(eventItem) : [el("li", { className: "hint",
    textContent: cases.length ? "No events of this kind right now." : "The recent events could not be loaded (offline?)." })]));
}
function eventItem(c) {
  const mine = added.some((a) => a.id === c.id);
  const ready = "before" in c; // "too recent": the after-event window has not started yet
  const btn = el("button", { type: "button", className: mine ? "" : "primary", disabled: !mine && !ready,
    textContent: mine ? "In your examples · show it" : ready ? "Add to examples" : "Too recent to add: no after-event imagery yet" });
  btn.onclick = () => {
    if (!mine) { added.push(c); saveAdded(); }
    fillExamples(c.id);
    $("example").onchange();
    $("scout").close();
  };
  return el("li", { className: "case" }, el("div", { className: "ev-title", textContent: `${TASK_ICON[c.task] || ""} ${c.title}` }),
    ...caseDetails(c), el("div", { className: "btns" }, btn));
}
$("scout-open").onclick = () => { renderScout(); $("scout").showModal(); };
$("scout-close").onclick = () => $("scout").close();
$("scout").onclick = (e) => { if (e.target === $("scout")) $("scout").close(); }; // a click on the backdrop
for (const r of document.querySelectorAll("input[name=scout-hazard]")) r.onchange = renderScout;

// Public deployments hide /jobs; each browser then remembers the jobs it started.
const myJobs = () => { try { return JSON.parse(localStorage.getItem("geopulse.jobs") || "[]"); } catch { return []; } };
const remember = (id) => { try { localStorage.setItem("geopulse.jobs", JSON.stringify([id, ...myJobs()].slice(0, 20))); } catch {} };

// Browser jobs and their result files live in IndexedDB (as GeoLibre keeps projects), so they survive reloads.
// Without IndexedDB (some private windows) they last until the tab closes.
const DB = new Promise((resolve, reject) => {
  const r = indexedDB.open("geopulse", 1);
  r.onupgradeneeded = () => r.result.createObjectStore("jobs", { keyPath: "job.id" });
  r.onsuccess = () => resolve(r.result);
  r.onerror = () => reject(r.error);
});
async function store(mode, fn) {
  const t = (await DB).transaction("jobs", mode);
  const req = fn(t.objectStore("jobs"));
  return new Promise((resolve, reject) => {
    t.oncomplete = () => resolve(req?.result);
    t.onerror = t.onabort = () => reject(t.error);
  });
}
async function loadBrowserJobs() {
  const recs = await store("readonly", (s) => s.getAll()).catch(() => []);
  for (const rec of recs.sort((a, b) => a.job.created.localeCompare(b.job.created))) BROWSER_JOBS.set(rec.job.id, rec);
}
async function saveBrowserJob(rec) {
  BROWSER_JOBS.set(rec.job.id, rec);
  for (const old of [...BROWSER_JOBS.keys()].slice(0, -KEEP_JOBS)) await deleteBrowserJob(old);
  const { urls, ...keep } = rec;
  await store("readwrite", (s) => s.put(keep));
}
async function deleteBrowserJob(id) {
  for (const u of Object.values(BROWSER_JOBS.get(id)?.urls ?? {})) URL.revokeObjectURL(u);
  BROWSER_JOBS.delete(id);
  await store("readwrite", (s) => s.delete(id)).catch(() => {});
}

async function refreshJobs() {
  const local = [...BROWSER_JOBS.values()].reverse().map(({ job }) => ({ ...job, task: job.request.task, aoi_km2: job.request.aoi_km2, local: true }));
  const server = !SERVER ? [] : await api("/jobs").catch(async () => (await Promise.all(myJobs().map((id) => api(`/jobs/${id}`).catch(() => null))))
    .filter(Boolean).map((j) => ({ ...j, task: j.request.task, aoi_km2: j.request.aoi_km2 })));
  const jobs = [...local, ...server];
  if (!jobs.length) return $("jobs").replaceChildren(Object.assign(document.createElement("span"), { className: "hint", textContent: "none yet" }));
  $("jobs").replaceChildren(...jobs.slice(0, 12).map((j) => {
    const d = document.createElement("div");
    const when = j.local ? new Date(j.created).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : j.id;
    d.innerHTML = `<span>${TASK_ICON[j.task] || ""} ${when} · ${j.aoi_km2?.toFixed?.(0) ?? "?"} km²</span><span class="st-${j.status}">${j.local ? "in browser" : j.status}</span>`;
    d.title = j.local ? `Computed in this browser (${j.id}); kept on this device` : `Server job ${j.id}`;
    d.onclick = () => j.status === "succeeded" ? showResults(j.id) : watch(j.id);
    if (j.local) {
      const x = Object.assign(document.createElement("button"), { type: "button", className: "del", textContent: "×", title: "Delete this result from this device" });
      x.onclick = async (e) => { e.stopPropagation(); await deleteBrowserJob(j.id); refreshJobs(); };
      d.append(x);
    }
    return d;
  }));
}

// ------------------------------------------------------------------ run
function logLines(lines) {
  const el = $("log");
  el.style.display = "block";
  el.innerHTML = lines.map((l) => {
    const cls = l.startsWith("✓") ? "ok" : l.startsWith("!") ? "warn" : l.startsWith("✗") ? "err" : "";
    return `<div class="${cls}">${l.replace(/</g, "&lt;")}</div>`;
  }).join("");
  el.scrollTop = el.scrollHeight;
}

$("run").onclick = async () => {
  if (!aoi) return logLines(["✗ Choose an area of interest first."]);
  const sensors = ["s1", "s2"].filter((s) => $(s).checked);
  if (!sensors.length) return logLines(["✗ Select at least one sensor."]);
  const body = {
    task: $("task").value, aoi, sensors, model: $("model").value,
    before: [$("b0").value, $("b1").value], after: [$("a0").value, $("a1").value],
  };
  if (ENGINE === "browser") return runInBrowser(body);
  try {
    const job = await api("/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    remember(job.id);
    watch(job.id);
  } catch (err) { logLines([`✗ ${err.message}`]); }
};

function runInBrowser(body) {
  let request;
  try { request = E.makeRequest({ ...body, maxKm2: BROWSER_MAX_KM2 }); }
  catch (err) { return logLines([`✗ ${err.message}`]); }
  const fits = browserModels.filter((m) => m.tasks.includes(request.task));
  const card = body.model === "auto" ? fits[0] : fits.find((m) => m.model_id === body.model) ?? BASELINE;
  const lines = [`browser job: ${request.task} with ${card.model_id}`];
  logLines(lines);
  $("run").disabled = true;
  startProgress();
  worker ??= new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onerror = (e) => { lines.push(`✗ worker failed: ${e.message || "could not start"}`); logLines(lines); $("run").disabled = false; endProgress(); };
  worker.onmessage = async ({ data }) => {
    if (data.type === "log") { lines.push(data.msg); logLines(lines); return; }
    if (data.type === "progress") { setProgress(data.value, data.label); return; }
    $("run").disabled = false;
    endProgress();
    if (data.type === "error") { lines.push(`✗ ${data.message}`); logLines(lines); return; }
    const id = [...crypto.getRandomValues(new Uint8Array(6))].map((b) => b.toString(16).padStart(2, "0")).join("");
    const job = { id, status: "succeeded", request, log: lines, created: new Date().toISOString(), summary: data.result.summary };
    const { summary, layers, files } = data.result;
    try { await saveBrowserJob({ job, res: { summary, layers, files } }); }
    catch (err) { lines.push(`! not kept after this tab closes (browser storage: ${err.message || err.name})`); logLines(lines); }
    refreshJobs();
    showResults(id);
  };
  worker.postMessage({ type: "run", request, model: card, version: VERSION });
}

async function watch(id) {
  $("run").disabled = true;
  startProgress();
  try {
    for (;;) {
      const job = await api(`/jobs/${id}`);
      logLines([`job ${id}: ${job.status}`, ...job.log, ...(job.error ? [`✗ ${job.error}`] : [])]);
      if (job.started) progressT0 = Date.parse(job.started);
      setProgress(job.progress, job.stage ?? (job.status === "queued" ? "Waiting for the engine" : null));
      if (job.status === "succeeded") { await showResults(id); break; }
      if (job.status === "failed") break;
      await new Promise((r) => setTimeout(r, 1500));
    }
  } finally { $("run").disabled = false; endProgress(); refreshJobs(); }
}

// ------------------------------------------------------------------ progress
// A native <progress>: indeterminate until the engine reports a fraction, and it never moves backwards.
let progressT0 = 0, progressValue = 0, progressTimer = null;
function startProgress() {
  clearInterval(progressTimer);
  progressT0 = Date.now();
  progressValue = 0;
  $("progress-bar").removeAttribute("value");
  $("progress-label").textContent = "Starting…";
  $("progress").hidden = false;
  progressTimer = setInterval(showProgressMeta, 1000);
  showProgressMeta();
}
function setProgress(value, label) {
  if (typeof value === "number") $("progress-bar").value = progressValue = Math.max(progressValue, Math.min(value, 1));
  if (label) $("progress-label").textContent = label;
  showProgressMeta();
}
function showProgressMeta() {
  const s = Math.max(0, Math.round((Date.now() - progressT0) / 1000));
  const pct = $("progress-bar").hasAttribute("value") ? `${Math.floor(progressValue * 100)}% · ` : "";
  $("progress-meta").textContent = `${pct}${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function endProgress() {
  clearInterval(progressTimer);
  $("progress").hidden = true;
}

// ------------------------------------------------------------------ results
const PRED = ["change", "uncertainty", "severity", "target"];
const IMG = ["s2_pre", "s2_post", "s1_pre", "s1_post"];
let current = null;

function clearResult(m) {
  for (const id of [...IMG, ...PRED, "outline"]) {
    if (m.getLayer(id)) m.removeLayer(id);
    if (m.getSource(id)) m.removeSource(id);
  }
}

function addImage(m, id, url, b, opacity) {
  const [w, s, e, n] = b;
  m.addSource(id, { type: "image", url, coordinates: [[w, n], [e, n], [e, s], [w, s]] });
  m.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": opacity, "raster-resampling": "nearest", "raster-fade-duration": 0 } }, m.getLayer("aoi-line") ? "aoi-line" : undefined);
}

async function serverResults(id) {
  const res = await api(`/jobs/${id}/results`);
  res.files = Object.fromEntries(Object.entries(res.files).map(([k, url]) => [k, withToken(SERVER_URL + url)]));
  return { res, job: await api(`/jobs/${id}`) };
}

async function showResults(id) {
  const rec = BROWSER_JOBS.get(id);
  if (rec) rec.urls ??= Object.fromEntries(Object.entries(rec.res.files).map(([k, blob]) => [k, URL.createObjectURL(blob)]));
  const { res, job } = rec ? { res: { ...rec.res, files: rec.urls }, job: rec.job } : await serverResults(id);
  if (rec) logLines(job.log);
  current = { id, res };
  setAoi(job.request.aoi, true);
  // The form now describes this job, ready to tweak and rerun.
  $("example").value = "";
  setTask(job.request.task);
  setDates(job.request.before, "b0", "b1");
  setDates(job.request.after, "a0", "a1");
  for (const k of ["s1", "s2"]) $(k).checked = job.request.sensors.includes(k);

  const s = res.summary;
  $("results").hidden = false;
  $("m-hit").textContent = s.affected_km2.toFixed(2);
  $("k-hit").textContent = `${s.affected_label} km²`;
  $("k-conf").textContent = `mean confidence (${s.affected_label})`;
  $("m-review").textContent = s.review_km2.toFixed(2);
  $("m-conf").textContent = s.mean_confidence_affected == null ? "–" : `${(s.mean_confidence_affected * 100).toFixed(0)}%`;
  $("severity").innerHTML = s.severity_km2 ? "<label>Burn severity (dNBR, Key &amp; Benson 2006)</label>" +
    Object.entries(s.severity_km2).map(([k, v]) => `<div class="mono" style="font-size:12px">${k.padEnd(14)} ${v.toFixed(2)} km²</div>`).join("") : "";
  $("m-obs").textContent = s.observed_km2.toFixed(1);
  $("sensors-used").innerHTML = ["s1_pre", "s1_post", "s2_pre", "s2_post"].map((k) =>
    `${k.replace("_", " ").toUpperCase().padEnd(8)} ${s.scenes[k] ? `${s.scenes[k]} scene(s)` : '<span style="color:var(--bad)">missing</span>'}`).join("<br>")
    + `<br>MODEL    ${s.model.id} v${s.model.version}`;
  $("weights").innerHTML = s.modality_weights ? "<label>Modality reliability weight (post-event, learned)</label>" +
    Object.entries(s.modality_weights).filter(([, v]) => v != null).map(([k, v]) =>
      `<div class="mono" style="font-size:12px">${k.toUpperCase()} ${(v * 100).toFixed(0)}%</div><div class="bar"><div style="width:${v * 100}%"></div></div>`).join("") : "";
  $("warnings").replaceChildren(...s.warnings.map((w) => Object.assign(document.createElement("li"), { textContent: w })));
  $("files").innerHTML = Object.keys(res.files).filter((k) => !k.startsWith("layers")).map((k) => `<a href="${res.files[k]}" download="${k}">${k}</a>`).join("");
  // A cross-origin link (the desktop app's engine) ignores `download` and would open the file in place of the app.
  for (const a of $("files").querySelectorAll("a")) if (new URL(a.href).origin !== location.origin) a.onclick = saveFile;

  // Map layers need the map; the numbers and downloads above do not wait for it.
  await mapReady;
  if (current.id !== id) return; // another result was opened meanwhile
  const { bounds, layers } = res.layers;
  const f = (name) => res.files[`layers/${name}.png`];
  clearResult(map); clearResult(before);
  for (const k of IMG) if (layers.includes(k)) {
    addImage(map, k, f(k), bounds, 1);
    if (k.endsWith("_pre")) addImage(before, k, f(k), bounds, 1);
  }
  for (const k of PRED) if (layers.includes(k)) addImage(map, k, f(k), bounds, $("opacity").value / 100);
  const [tname, c0, c1] = TARGET[job.request.task] || TARGET.flood;
  $("target-name").textContent = tname;
  $("target-legend").style.background = `linear-gradient(90deg,${c0}00,${c0}96 40%,${c1}eb)`;
  map.addSource("outline", { type: "geojson", data: res.files[res.layers.extent] });
  map.addLayer({ id: "outline", type: "line", source: "outline", paint: { "line-color": "#ffffff", "line-opacity": 0.7, "line-width": 0.8 } });
  // Default visibility: optical if we have it, otherwise SAR.
  const afterImg = layers.includes("s2_post") ? "s2_post" : "s1_post";
  for (const el of document.querySelectorAll("#layers .check[data-layer]")) {
    const k = el.dataset.layer, box = el.querySelector("input");
    const exists = k === "outline" || layers.includes(k);
    el.classList.toggle("disabled", !exists);
    if (IMG.includes(k)) box.checked = k === afterImg;
    applyLayer(k, box.checked && exists);
  }
  const beforeImg = layers.includes("s2_pre") ? "s2_pre" : "s1_pre";
  for (const k of IMG) if (before.getLayer(k)) before.setLayoutProperty(k, "visibility", k === beforeImg ? "visible" : "none");
}

async function saveFile(e) {
  e.preventDefault();
  const a = e.currentTarget;
  const url = URL.createObjectURL(await (await fetch(a.href)).blob());
  Object.assign(document.createElement("a"), { href: url, download: a.download }).click();
  setTimeout(() => URL.revokeObjectURL(url), 60e3);
}

// Desktop app: web links open in the system browser, not inside the app window.
if (TAURI) document.addEventListener("click", (e) => {
  const a = e.target.closest?.("a[href^='https://']");
  if (a) { e.preventDefault(); TAURI.opener.openUrl(a.href); }
});

function applyLayer(k, on) {
  const vis = on ? "visible" : "none";
  if (map.getLayer(k)) map.setLayoutProperty(k, "visibility", vis);
}
for (const el of document.querySelectorAll("#layers .check[data-layer]")) {
  el.querySelector("input").onchange = (e) => applyLayer(el.dataset.layer, e.target.checked);
}
$("opacity").oninput = () => { for (const k of PRED) if (map.getLayer(k)) map.setPaintProperty(k, "raster-opacity", $("opacity").value / 100); };
for (const b of document.querySelectorAll("[data-basemap]")) b.onclick = () => {
  for (const m of [map, before]) for (const id of ["dark", "imagery"]) m.setLayoutProperty(id, "visibility", id === b.dataset.basemap ? "visible" : "none");
};

// ------------------------------------------------------------------ swipe
let swipeX = 0.5;
function placeSwipe() {
  const w = $("maps").clientWidth, x = swipeX * w;
  $("swipe").style.left = `${x}px`;
  $("map-before").style.clipPath = `inset(0 ${w - x}px 0 0)`;
}
$("swipe-toggle").onchange = (e) => {
  const on = e.target.checked;
  for (const id of ["map-before", "swipe", "lbl-before", "lbl-after"]) $(id).style.display = on ? "block" : "none";
  if (on) { before.resize(); before.jumpTo({ center: map.getCenter(), zoom: map.getZoom() }); placeSwipe(); }
};
$("swipe").querySelector(".grip").onpointerdown = (e) => {
  e.target.setPointerCapture(e.pointerId);
  e.target.onpointermove = (ev) => {
    const r = $("maps").getBoundingClientRect();
    swipeX = Math.min(0.98, Math.max(0.02, (ev.clientX - r.left) / r.width));
    placeSwipe();
  };
  e.target.onpointerup = () => { e.target.onpointermove = null; };
};
new ResizeObserver(() => { map.resize(); before.resize(); placeSwipe(); }).observe($("maps"));

if (innerWidth < 800) $("layers").open = false;  // small screens: keep the map visible

init().catch((err) => logLines([`✗ could not start: ${err.message}`]));
