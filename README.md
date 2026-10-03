<p align="center"><img src="docs/assets/geopulse-logo.png" alt="GeoPulse: Earth Intelligence" width="420"></p>

# GeoPulse

Open-source multimodal geospatial AI for Earth-change intelligence.

[![CI](https://github.com/nrahaman1/geopulse/actions/workflows/ci.yml/badge.svg)](https://github.com/nrahaman1/geopulse/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Download the desktop app](https://img.shields.io/github/v/release/nrahaman1/geopulse?label=desktop%20app&logo=github)](https://github.com/nrahaman1/geopulse/releases/latest)
[![Open the web app](https://img.shields.io/badge/%F0%9F%9B%B0%EF%B8%8F-open%20the%20web%20app-3fb6c8.svg)](https://nrahaman1.github.io/geopulse/)

## ▶ Get GeoPulse

| | |
|---|---|
| **Desktop app** — Windows, macOS, Linux | **[Download from Releases](https://github.com/nrahaman1/geopulse/releases/latest)**: `GeoPulse_…_x64-setup.exe` (Windows), `…_aarch64.dmg` / `…_x64.dmg` (macOS), `….AppImage` / `….deb` (Linux). Runs GeoPulse's full Python engine on **your PC and its GPU** (NVIDIA CUDA, Apple Metal) with areas up to 1500 km², plus the built-in WebGPU engine. The first start of the PC engine downloads Python and PyTorch once (~3 GB with CUDA, ~400 MB CPU-only). [How it works](#how-it-runs). |
| **Web app** — nothing to install | **<https://nrahaman1.github.io/geopulse/>**: the same app, computing in your browser (WebGPU, else WebAssembly). Nothing is uploaded and no server is involved. Areas up to 300 km². Installable as a PWA. |
| **Python** (CLI, SDK, API) | see [Quickstart](#quickstart), `docker compose up`, or [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/nrahaman1/geopulse?quickstart=1) |

Installing a new version over an old one upgrades it in place, and from v0.5.0 the desktop app updates itself: it
checks for a new release at startup and installs it when no analysis is running.

The installers are not code-signed yet: Windows SmartScreen asks you to confirm ("More info" → "Run anyway"), and on
macOS open the app once with right-click → Open.

```text
      Sentinel-1 SAR  +  Sentinel-2 optical  +  Copernicus DEM
                              ↓
          STAC discovery → deterministic grid → task-aware pre/post composites
                              ↓
          GeoPulse Fusion (quality-gated; one head per task + shared change head)
                              ↓
            Flood  |  Wildfire / burn (+ dNBR severity)  |  Vegetation disturbance
                              ↓
     Probability + uncertainty + polygons + provenance (COG / GeoJSON / STAC)
```

**Status: v0.5 (alpha), three tasks.** A desktop app, a web app and a Python package, all on public data. What is
implemented and why: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); results and caveats:
[MODEL_CARD.md](MODEL_CARD.md); what comes next: [ROADMAP.md](ROADMAP.md).

## Quickstart

```bash
pip install https://github.com/nrahaman1/geopulse/releases/download/v0.5.3/geopulse_eo-0.5.3-py3-none-any.whl
# Python 3.11+. The wheel on each release includes the web app; the command and import are `geopulse`.
geopulse models pull         # trained checkpoints from GitHub Releases, SHA-256 verified
geopulse doctor              # GDAL, PROJ, PyTorch, CUDA, STAC connectivity
geopulse serve               # API + web map at http://127.0.0.1:8000  (API docs at /docs)
```

From a clone, with the example events:

```bash
git clone https://github.com/nrahaman1/geopulse && cd geopulse
uv sync --extra gpu          # CUDA PyTorch on Windows/Linux (Metal on macOS); or --extra cpu
uv run geopulse models pull
uv run geopulse infer --config examples/flood_emilia_conselice/request.yaml
uv run geopulse infer --config examples/wildfire_palisades_la/request.yaml
uv run geopulse infer --config examples/vegetation_para_deforestation/request.yaml
```

Or with Docker (CPU): `docker compose up`, then open http://127.0.0.1:8000.

`infer` writes to `outputs/<example>/`, named by what the task maps (`flood`, `burn`, `disturbance`):

```text
<target>_probability.tif  change_probability.tif  uncertainty.tif   (Cloud-Optimized GeoTIFFs)
burn_severity.tif         (wildfire: Key & Benson dNBR classes 1 low … 4 high)
<target>_extent.geojson   summary.json   provenance.json   item.json (STAC Item)
quicklook.png             layers/*.png + layers.json                (web-map overlays)
```

No account or API key is needed: imagery comes from the Microsoft Planetary Computer STAC with anonymous
signed URLs. First runs download only the COG windows covering the AOI (~1–2 min per 100–200 km²); repeat
runs hit the local cache in `data/cache/`.

## Tasks

| Task | Map | How the windows are used | Pick windows like |
|---|---|---|---|
| `flood` | open-water inundation | pre = median; post = earliest clear view (floods recede) | dry weeks before → days after |
| `wildfire` | burned area + severity | pre = median before ignition; post = median after containment | weeks before → after containment |
| `vegetation` | stand-replacing canopy loss | both = median of scenes spread evenly across the window | **same season**, different years |

Vegetation requests whose windows differ by more than 45 days in the annual cycle get a warning: otherwise
senescence and green-up look like disturbance. Burn and disturbance maps are set to 0 on pre-event open water.

## Interfaces

**CLI**

```bash
geopulse search   --aoi aoi.geojson --start 2024-09-01 --end 2024-10-10 --sensor s1 --sensor s2
geopulse infer    --aoi aoi.geojson --task wildfire --before 2024-12-01/2025-01-06 --after 2025-02-01/2025-03-31
geopulse dataset build configs/data/wildfire_bench.yaml
geopulse train    configs/experiments/multitask_gpft.yaml
geopulse evaluate --model gpft-multitask-mini --dataset data/bench/geopulse-bench-wildfire --split test
geopulse evaluate --model threshold-baseline  --dataset data/bench/geopulse-bench-wildfire --split test
geopulse models [list|pull|push]  |  geopulse dataset [build|pull|push]  |  serve | benchmark | doctor
```

**Python SDK**

```python
from geopulse import GeoPulse

model = GeoPulse.from_pretrained("auto")          # newest trained checkpoint, else the physics baseline
result = model.predict(aoi="examples/wildfire_camp_paradise/aoi.geojson", task="wildfire",
                       before=("2018-09-15", "2018-11-07"), after=("2018-11-26", "2019-01-31"))
result.save("outputs/camp/")
```

**REST API** (FastAPI, OpenAPI at `/docs`)

```text
GET  /health             GET  /models   /models/{id}     GET /examples
POST /search             POST /jobs     GET /jobs   /jobs/{id}   /jobs/{id}/results   /jobs/{id}/files/{path}
POST /jobs/{id}/cancel
POST /predict/sync       (≤ 25 km²)      GET /metrics     (Prometheus text)
```

Jobs are validated up front (task, geometry, ≤ 500 km², ordered windows ≤ 120 days, known sensors), run on a
background worker, and persisted under `outputs/jobs/<id>/`.

**Web map** (`/`, and the hosted platform): search a place or `lat, lon` (sets a 10 × 10 km box), pick one of 14 examples (💧 flood,
🔥 wildfire, 🌲 vegetation), draw a box or upload GeoJSON;
set task, windows, sensors and model (filtered by task); run. Layers for the task probability, burn severity,
uncertainty, change, extent polygons and pre/post S1/S2 imagery; opacity; before/after swipe; metrics (affected km²,
review-recommended km², confidence, severity breakdown, sensors used, learned modality weights); downloads.
Everything needed to run fits one screen: the side panel has *Set up*, *Result* and *Jobs* tabs, and its run bar (Run,
or Stop while a job runs, with the current stage and progress) never scrolls away. **Recent events** sits beside the
place search.
**Compute** chooses where a job runs: the *PC engine* (desktop app), the *GeoPulse server* (when `geopulse serve`
serves the page) or *built-in* (this browser, always available). Built-in jobs and their files are kept on the
device (IndexedDB), 10 at most.

**Scout** ([docs/SCOUT.md](docs/SCOUT.md)): every 6 hours a GitHub Actions run reads official alerts (GDACS,
Copernicus EMS, NASA EONET) and the news with an open-weights LLM, and proposes each new flood, wildfire or forest-loss
event as a ready-to-run case. The **Recent events** button lists them with their sources (alerts, news links and
the quoted sentences), so you can see where each comes from; **Add to examples** puts one in your example list,
and nothing runs until you press Run. Locally: `geopulse scout discover` (needs `--extra scout` and an
OpenAI-compatible LLM such as Ollama).

## How it runs

GeoPulse follows [GeoLibre](https://github.com/opengeos/GeoLibre): one web app ([`app/`](app/), Vite) ships three ways,
and the heavy lifting happens on the user's own machine.

- **Desktop app** ([`app/src-tauri/`](app/src-tauri/), [Tauri 2](https://tauri.app)): a small native shell around
  the web app. Like GeoLibre's processing sidecar, GeoPulse's Python engine is a locked project bundled with the app
  (`pyproject.toml` + `uv.lock` + the `geopulse` package) that the bundled [uv](https://docs.astral.sh/uv/)
  installs on first use into the app's data folder, with PyTorch built for CUDA when an NVIDIA GPU is present
  (Metal on Apple silicon, CPU otherwise). The engine listens on 127.0.0.1 only, on a random port, accepts only
  requests carrying a per-launch token, and exits with the app. It runs the full Python pipeline (rasterio/GDAL
  reads, PyTorch on the GPU) with areas up to 1500 km².
- **Web app** on GitHub Pages: static files and a service worker (PWA) that caches the app, ONNX Runtime and the
  models after first use. Every job runs in the visitor's browser (below).
- **`geopulse serve`**: the same web app from the Python package, with its API as the engine.

The built-in (browser) engine, [`app/src/worker.js`](app/src/worker.js) and [`app/src/engine.js`](app/src/engine.js),
redoes the Python pipeline in JavaScript:

1. STAC search on the Planetary Computer, scene selection, per-container SAS signing (all CORS-enabled).
2. HTTP range reads of only the COG windows over the AOI ([geotiff.js](https://geotiffjs.github.io/)), retried on
   transient network errors and throttled, warped to the same 10 m UTM grid as Python (own UTM projection,
   bilinear/nearest with nodata renormalisation).
3. Task-aware composites, SCL cloud masks, dB conversion, DEM slope, physics baseline and dNBR severity.
4. The GeoPulse model as ONNX ([ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)) on **WebGPU** (8 MC-dropout
   passes) or WebAssembly (3 passes), SHA-256 verified. Before a backend is trusted it must reproduce the model
   card's self-check (logits PyTorch produced for a fixed input); a GPU backend or driver that computes the model
   wrongly is caught and WebAssembly is used instead.
5. Polygons ([d3-contour](https://github.com/d3/d3-contour)), map overlays, GeoTIFF/GeoJSON/summary/provenance downloads.

Same events, browser vs Python server (both `gpft-multitask-mini`, 8 MC passes):

| Event | Browser | Python | Browser time (cold, WebGPU) |
|---|---:|---:|---:|
| Emilia 2023 flood, flooded km² | 17.68 | 17.70 | 178 s |
| Palisades 2025 fire, burned km² | 73.79 | 73.76 | 182 s |
| Pará 2020 forest loss, disturbed km² | 16.31 | 16.04 | 287 s |

Most of the time is spent downloading imagery; reruns with other sensors or models reuse it while the tab is open.
Without WebGPU a model pass is ~12× slower, so WebAssembly uses 3 passes. `tests/test_js_parity.py` checks the engine against the Python reference
under Node.js (grids, projections, requests, masks, baseline, model inputs); `tests/test_onnx.py` checks ONNX against PyTorch.

## Models

| Model | What it is |
|---|---|
| `threshold-baseline` | Training-free physics for all tasks: SAR VV + log-ratio drop and MNDWI (flood); dNBR on vegetated pixels (burn); NDVI + VH drop on canopy (vegetation). |
| `gpft-<task>-mini` | GPFT-mini (0.54 M params) for one task: per-sensor adapters, per-pixel quality-gated fusion, change encoder, U-Net, MC-dropout uncertainty, temperature-scaled. Trained with sensor dropout. |
| `gpft-<task>-mini-nodrop` | Same, without sensor dropout (ablation). |
| `gpft-multitask-mini` | One GPFT-mini with flood, burn and disturbance heads, trained on all three benchmarks. |

## Benchmarks: GeoPulse-Bench v0.1

Three benchmarks, split by event, with an out-of-distribution test event each
([configs/data/](configs/data/)); `bench.audit` rejects overlapping AOIs across splits.

| Task | Labels | Test event (never seen in training) |
|---|---|---|
| flood | weak S1+S2 consensus | Pakistan 2022 (continent) |
| wildfire | MTBS burn severity (USGS/USFS) | Martin Fire 2018, Nevada sagebrush (ecoregion) |
| vegetation | Hansen Global Forest Change loss year | Novo Progresso, Pará 2020 (biome) |

Test IoU with both sensors, one seed ([MODEL_CARD.md](MODEL_CARD.md) has the full robustness matrices):

| Model | Flood | Wildfire | Vegetation |
|---|---:|---:|---:|
| threshold-baseline | n/a (labels built from it) | **0.901**\* | 0.212 |
| gpft-<task>-mini (sensor dropout) | 0.940 | 0.017 | 0.496 |
| gpft-<task>-mini-nodrop | **0.975** | 0.232 | 0.521 |
| gpft-multitask-mini | 0.959 | 0.707 | **0.560** |

\* 0.076 before a fuel-gate fix that was diagnosed on this test event; see the model card.

What these say: multi-task training improved every out-of-distribution test over the matching single-task model;
dNBR physics transferred to an unseen fire ecoregion far better than models trained on five California fires;
learned models beat fixed thresholds ~2.5× on vegetation disturbance; sensor dropout did not help in any task.

Reproduce (download the prebuilt tiles with `geopulse dataset pull`, or rebuild them from STAC):

```bash
for b in flood wildfire vegetation; do geopulse dataset build configs/data/${b}_bench.yaml; done
geopulse train configs/experiments/multitask_gpft.yaml        # and <task>_gpft[_nodrop].yaml
geopulse evaluate --model gpft-multitask-mini --dataset data/bench/geopulse-bench-vegetation --split test
```

Systems (RTX 4060 Laptop, 256 px tiles, batch 16): 10.1 ms/tile FP32, 1.9 GB peak; FP16 autocast is not faster
for this 0.54 M-parameter model.

## Downloads (everything is on GitHub)

| What | Where | Command |
|---|---|---|
| Desktop installers + Python wheel | [Releases](https://github.com/nrahaman1/geopulse/releases/latest) (`v*` tags) | built by `.github/workflows/release.yml` |
| Web app | [GitHub Pages](https://nrahaman1.github.io/geopulse/) | built by `.github/workflows/pages.yml` on every push |
| Trained checkpoints, ONNX exports, cards, eval reports | release [`models-v1`](https://github.com/nrahaman1/geopulse/releases/tag/models-v1) | `geopulse models pull` / `models push` |
| GeoPulse-Bench tiles (flood, wildfire, vegetation) | release [`bench-v1`](https://github.com/nrahaman1/geopulse/releases/tag/bench-v1) | `geopulse dataset pull` / `dataset push` |

## Development

```bash
uv run pytest                     # offline: grid alignment, physics per task, modality combinations, multi-task,
                                  # API, releases, browser engine vs Python (needs Node.js), ONNX vs PyTorch
uv run ruff check geopulse tests
cd app && npm ci
npm run dev                       # the web app at http://localhost:5173 (hot reload)
npm run build                     # dist/: what GitHub Pages serves; `geopulse serve` serves it too
npm run uv && npm run tauri dev   # the desktop app (needs Rust; `npm run uv` fetches uv for this machine)
npm run tauri build               # installers under src-tauri/target/release/bundle/
```

Layout: `geopulse/{tasks,stac,grid,data,baseline,model,bench,train,pipeline,releases,api,cli}.py`;
`app/` (web app: `index.html`, `src/{main,engine,worker}.js`; desktop shell: `src-tauri/`); `configs/`, `examples/`,
`tests/`, `docs/`.
Contributions welcome: [CONTRIBUTING.md](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md) ·
[Security](SECURITY.md) · [Changelog](CHANGELOG.md). Citing GeoPulse: [CITATION.cff](CITATION.cff).

## Licenses

Code: Apache-2.0 ([LICENSE](LICENSE)). Data: see [DATA_LICENSES.md](DATA_LICENSES.md) — contains modified
Copernicus Sentinel data; Sentinel-1 RTC © Catalyst/Microsoft, CC BY 4.0; MTBS (public domain); Hansen GFC (CC BY 4.0).
