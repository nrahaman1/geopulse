# GeoPulse

Open-source multimodal geospatial AI for Earth-change intelligence.

[![CI](https://github.com/nrahaman1/geopulse/actions/workflows/ci.yml/badge.svg)](https://github.com/nrahaman1/geopulse/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow.svg)](https://huggingface.co/nafizrahaman/geopulse-gpft-mini)
[![Benchmark on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-GeoPulse--Bench-yellow.svg)](https://huggingface.co/datasets/nafizrahaman/geopulse-bench)
[![Open the platform](https://img.shields.io/badge/%F0%9F%9B%B0%EF%B8%8F-open%20the%20platform-3fb6c8.svg)](https://nrahaman1.github.io/geopulse/)

## ▶ Open the platform

| | |
|---|---|
| **In your browser** — nothing to install | **<https://nrahaman1.github.io/geopulse/>** (mirror: <https://nafizrahaman-geopulse.static.hf.space>) — the full platform on any place: search, pick dates and a task, run. **Your own computer does the work** ([how](#in-browser-engine)); nothing is uploaded and no server is involved. Areas up to 300 km². |
| **Python server in the cloud** — free | [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/nrahaman1/geopulse?quickstart=1) — builds in a few minutes, then the map opens by itself (areas up to 500 km²). |
| **On your machine** (GPU optional) | see [Quickstart](#quickstart) or `docker compose up`. |

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

**Status: v0.3 (alpha), three tasks.** Runs in a web browser, or on a laptop with or without a GPU, against public data. What is
implemented and why: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); results and caveats:
[MODEL_CARD.md](MODEL_CARD.md); what comes next: [ROADMAP.md](ROADMAP.md).

## Quickstart

```bash
pip install https://github.com/nrahaman1/geopulse/archive/refs/heads/main.zip   # Python 3.11+
# (PyPI package `geopulse-eo` coming; the command and import are `geopulse`)
geopulse models pull         # trained checkpoints from Hugging Face, SHA-256 verified
geopulse doctor              # GDAL, PROJ, PyTorch, CUDA, STAC connectivity
geopulse serve               # API + web map at http://127.0.0.1:8000  (API docs at /docs)
```

From a clone, with the example events:

```bash
git clone https://github.com/nrahaman1/geopulse && cd geopulse
uv sync                      # CUDA PyTorch on Windows/Linux; CPU-only: see CONTRIBUTING.md
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
POST /predict/sync       (≤ 25 km²)      GET /metrics     (Prometheus text)
```

Jobs are validated up front (task, geometry, ≤ 500 km², ordered windows ≤ 120 days, known sensors), run on a
background worker, and persisted under `outputs/jobs/<id>/`.

**Web map** (`/`, and the hosted platform): search a place or `lat, lon` (sets a 10 × 10 km box), pick one of 13 examples (💧 flood,
🔥 wildfire, 🌲 vegetation), draw a box or upload GeoJSON;
set task, windows, sensors and model (filtered by task); run. Layers for the task probability, burn severity,
uncertainty, change, extent polygons and pre/post S1/S2 imagery; opacity; before/after swipe; metrics (affected km²,
review-recommended km², confidence, severity breakdown, sensors used, learned modality weights); downloads.
**Compute** chooses where a job runs: *this browser* (always available) or *GeoPulse server* (when the page is served
by `geopulse serve`). Browser jobs and their files are kept on the device (IndexedDB), 10 at most.

## In-browser engine

Following [GeoLibre](https://github.com/opengeos/GeoLibre), the hosted platform is static files and every job runs on
the visitor's machine: [`geopulse/web/worker.js`](geopulse/web/worker.js) (a Web Worker) and
[`geopulse/web/engine.js`](geopulse/web/engine.js) redo the Python pipeline in JavaScript.

1. STAC search on the Planetary Computer, scene selection, per-container SAS signing (all CORS-enabled).
2. HTTP range reads of only the COG windows over the AOI ([geotiff.js](https://geotiffjs.github.io/)), warped to the
   same 10 m UTM grid as Python (own UTM projection, bilinear/nearest with nodata renormalisation).
3. Task-aware composites, SCL cloud masks, dB conversion, DEM slope, physics baseline and dNBR severity.
4. The GeoPulse model as ONNX ([ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)) on **WebGPU** (8 MC-dropout
   passes) or WebAssembly (3 passes), downloaded from Hugging Face and SHA-256 verified.
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

## Hugging Face

| What | Where | Command |
|---|---|---|
| Trained checkpoints + eval reports | [nafizrahaman/geopulse-gpft-mini](https://huggingface.co/nafizrahaman/geopulse-gpft-mini) | `geopulse models pull` |
| GeoPulse-Bench tiles (flood, wildfire, vegetation) | [nafizrahaman/geopulse-bench](https://huggingface.co/datasets/nafizrahaman/geopulse-bench) | `geopulse dataset pull` |
| ONNX models for the browser | same repo (`*.onnx`, `index.json`) | `geopulse models export-onnx --model all`, then `models push` |
| In-browser platform (static Space, mirror of GitHub Pages) | [nafizrahaman/geopulse](https://huggingface.co/spaces/nafizrahaman/geopulse) | `python scripts/build_web.py site && hf upload nafizrahaman/geopulse site --repo-type space` |
| Full live Space (Docker, needs HF PRO hardware) | ready in [deploy/huggingface-space/](deploy/huggingface-space/) | `hf upload <user>/<space> deploy/huggingface-space --repo-type space` |

## Development

```bash
uv run pytest          # offline: grid alignment, physics per task, modality combinations, multi-task, API, Hub,
                       # browser engine vs Python (needs Node.js), ONNX vs PyTorch
uv run ruff check geopulse tests scripts
python scripts/build_web.py site && python -m http.server -d site   # the static platform at http://localhost:8000
```

Layout: `geopulse/{tasks,stac,grid,data,baseline,model,bench,train,pipeline,hub,api,cli}.py`,
`geopulse/web/{index.html,engine.js,worker.js}`, `configs/`, `examples/`, `tests/`, `docs/`, `deploy/`.
Contributions welcome: [CONTRIBUTING.md](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md) ·
[Security](SECURITY.md) · [Changelog](CHANGELOG.md). Citing GeoPulse: [CITATION.cff](CITATION.cff).

## Licenses

Code: Apache-2.0 ([LICENSE](LICENSE)). Data: see [DATA_LICENSES.md](DATA_LICENSES.md) — contains modified
Copernicus Sentinel data; Sentinel-1 RTC © Catalyst/Microsoft, CC BY 4.0; MTBS (public domain); Hansen GFC (CC BY 4.0).
