# Changelog

All notable changes are recorded here. The project follows [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change APIs and output schemas.

## [0.5.3] — 2026-10-03

### Added
- **Stop** a running analysis: the Run button turns into Stop while a job runs. The built-in engine stops at once; a
  PC-engine or server job stops at its next progress report (`POST /jobs/{id}/cancel`; new job status `cancelled`).
- A search box in **Recent events** (place, country, source, quoted text).

### Changed
- Layout: a wider side panel with *Set up*, *Result* and *Jobs* tabs, so setting up and running fit one screen without
  scrolling; the run bar (Run/Stop, progress, log) stays at the bottom of the panel; a result opens in the *Result* tab.
  **Recent events** moved beside the map's place search, with a count.
- Extent polygons are drawn in bright yellow on a dark casing (were thin white lines), so they stand out on the dark
  basemap, on imagery and over every task colour.

## [0.5.2] — 2026-10-02

### Changed
- Desktop app: while the PC engine starts, a spinner under "This PC" shows the current step (checking or updating the
  Python environment, checking the trained models, loading PyTorch on the GPU) and the seconds so far, so the start no
  longer looks frozen.

## [0.5.1] — 2026-10-01

### Added
- Example event: the Trishuli debris flood of 26 August 2026 in Nuwakot, Nepal (Bidur/Trishuli Bazar, Betrawati,
  Devighat), Sentinel-1 and Sentinel-2.

### Changed
- The Scout's events are no longer mixed into the example list. A **Recent events** button opens them in a dialog with
  each event's sources (alerts, news links, quoted sentences), its location and imagery, and the health of the last
  Scout run; **Add to examples** puts an event in the example list under *Added from the Scout* (kept in this browser
  or app, refreshed while the Scout still lists it), where it can be run or removed.

### Fixed
- Scout: the first live run accepted two flood forecasts and placed three stories at the wrong spot (a stadium, an
  industrial site, a housing estate named "State"). Evidence quotes that only warn or forecast no longer count, and a
  geocoded place must be a place (not a building) whose name is the one the story gives; names come in English.

## [0.5.0] — 2026-09-30

### Added
- **GeoPulse Scout** (`geopulse/scout/`, `geopulse scout discover|list|run`, [docs/SCOUT.md](docs/SCOUT.md)): finds
  floods, wildfires and forest loss in official alerts (GDACS, Copernicus EMS, NASA EONET) and in the news (GDELT and
  topic RSS feeds, read by an open-weights LLM through any OpenAI-compatible endpoint, Qwen3 8B on Ollama by default),
  verifies quotes and places, geocodes them and plans each event as a ready-to-run case with before/after windows and
  an imagery check. Optional extra `scout` (Scrapling). `GET /scout/cases`.
- The live Scout: `.github/workflows/scout.yml` runs every 6 hours on GitHub Actions, keeps its state on the
  `scout-data` branch and republishes GitHub Pages; the web and desktop apps list the cases as *Recent events*.
- Self-update for the desktop app (tauri-plugin-updater): at startup the app checks the latest GitHub Release,
  downloads a signed update in the background and installs it once no analysis is running.
- The official GeoPulse logo: app icons, web app favicon and PWA icons, the header and the README.

### Changed
- Windows releases ship only the NSIS installer (`…_x64-setup.exe`), which upgrades an installed GeoPulse in place;
  the MSI is no longer built, so updates never install a second copy.

## [0.4.1] — 2026-09-29

### Added
- A progress bar while an analysis runs, for every engine (built-in, PC engine, `geopulse serve`): the current stage
  ("Reading Sentinel-2 after (2/4 scenes)", "Running the model (12/81 tiles)"), percent and elapsed time. Both
  pipelines report the same stages (`pipeline.PROGRESS` = `engine.js::PROGRESS`); jobs expose `progress` and `stage`.
- Branch protection for `main`: no force pushes or deletion; CI must pass (repository admins may push directly).

## [0.4.0] — 2026-09-29

### Added
- Desktop app for Windows, macOS and Linux (Tauri 2, `app/src-tauri`), after GeoLibre: the web app in a native
  window plus GeoPulse's Python engine on the user's PC. On first use the bundled uv installs Python and PyTorch
  (CUDA with an NVIDIA GPU, Metal on Apple silicon, else CPU) into the app's data folder; the engine serves on
  127.0.0.1 with a per-launch token, takes areas up to 1500 km² and exits with the app. Installers are built by
  `.github/workflows/release.yml` and attached to each `v*` release.
- The web app is a Vite project (`app/`): npm dependencies bundled instead of CDNs, code-split (MapLibre, geotiff
  decoders and ONNX Runtime load only when needed), a service worker that caches the app, ONNX Runtime and the
  models (installable PWA). The same build is served by GitHub Pages, the desktop app and `geopulse serve`.
- ONNX self-check: model cards carry the logits PyTorch produced for a fixed input; the browser trusts a GPU
  backend only if it reproduces them, else it falls back to WebAssembly.
- `geopulse/releases.py`: checkpoints, ONNX exports and benchmark tiles are GitHub Release assets (`models-v1`,
  `bench-v1`), downloaded with the standard library and SHA-256 verified.
- `gpu` / `cpu` extras choose the PyTorch build (`uv sync --extra gpu`); Apple-silicon GPUs are used (MPS).
- `/health` reports the compute device and the area limit; `geopulse serve --exit-with-parent`; `GEOPULSE_TOKEN`.

### Changed
- ONNX Runtime Web uses its JSEP WebGPU build: the newer `webgpu` build computed GPFT-mini wrongly (1.76 vs 3.28 km²
  flooded on a test box).

### Removed
- Hugging Face: the Hub integration (`huggingface-hub` dependency, `geopulse/hub.py`), the Spaces and their deploy files.

### Fixed
- Browser engine: a single dropped or throttled imagery read ("Failed to fetch") no longer fails the job; reads are
  retried with backoff and at most 12 are in flight.
- Checkpoints load by name from a relative models directory again (`models/models/...` error).
- Hidden controls (the server option, the setup button) no longer show: `display` rules overrode `hidden`.

## [0.3.0] — 2026-09-29

### Added
- In-browser engine (after [GeoLibre](https://github.com/opengeos/GeoLibre)): the full platform runs on the visitor's
  machine as static files. `geopulse/web/worker.js` + `engine.js` do STAC search, per-container SAS signing, COG range
  reads (geotiff.js), warping onto the Python grid, compositing, physics, the ONNX model on WebGPU or WebAssembly
  (ONNX Runtime Web, MC dropout), polygons and GeoTIFF/GeoJSON downloads. Browser and Python results agree within
  1.7 % on the flood, wildfire and vegetation examples.
- Hosted platform on GitHub Pages (<https://nrahaman1.github.io/geopulse/>, `.github/workflows/pages.yml`), mirrored
  on the static Hugging Face Space; `scripts/build_web.py` builds it.
- Browser jobs and their result files persist on the device (IndexedDB, newest 10) and can be deleted from the list.
- `geopulse models export-onnx`: ONNX export with explicit dropout masks; the Hub model repo carries `*.onnx` and `index.json`.
- A "Compute" choice in the web map (this browser / GeoPulse server when one serves the page).
- Tests: the JavaScript engine against the Python reference under Node.js, and ONNX against PyTorch.

### Removed
- The precomputed online showcase (`scripts/export_static.py`); the hosted platform computes any place instead.

### Fixed
- Model cards list newest-first by their `created` date, not file modification time.

## [0.2.0] — 2026-09-29

### Added
- Wildfire / burn mapping with Key & Benson (2006) dNBR severity classes (`burn_severity.tif`).
- Vegetation disturbance (stand-replacing canopy loss) with a same-season comparison and a seasonal-mismatch warning.
- `geopulse/tasks.py`: one registry for per-task compositing, scene selection and output naming.
- Multi-task GPFT-mini (one head per task + shared change head); v0.1 flood checkpoints still load.
- GeoPulse-Bench wildfire (MTBS labels) and vegetation (Hansen GFC labels) benchmarks; `evaluate` can score the
  physics baseline on the same tiles.
- Hugging Face Hub integration: `geopulse models pull|push`, `geopulse dataset pull|push`, checksum-verified
  downloads, generated model and dataset cards.
- Place search in the web map (OpenStreetMap Nominatim, or `lat, lon`), which sets a 10 × 10 km AOI.
- Online showcase on a static Hugging Face Space (`scripts/export_static.py`: web map + precomputed example results),
  a GitHub Codespaces dev container that opens the full platform, and `GEOPULSE_MC_PASSES` for small CPU deployments.
- Dockerfile, compose file and a Hugging Face Space definition; public mode (`GEOPULSE_PUBLIC=1`) that hides other
  visitors' jobs; `GEOPULSE_MAX_JOB_KM2`.
- CI (lint, tests on Python 3.11–3.14, wheel contents, lockfile), Docker smoke test, tag-driven release to PyPI,
  Dependabot, pre-commit, issue and PR templates, governance files.

### Changed
- Distribution renamed to `geopulse-eo` on PyPI (the import name and CLI stay `geopulse`).
- Outputs are named by target (`flood_*`, `burn_*`, `disturbance_*`); summaries use `affected_km2`.
- SAR change evidence uses a softer slope (speckle-aware "stable" band).

### Fixed
- Docker image: install `libexpat1`, which rasterio's Linux wheels need and slim images no longer ship.
- Web map: example and search AOIs no longer wait forever when the map redraws after its first load.
- Remote reads now time out instead of hanging a job forever.
- Burn and disturbance maps are forced to zero on pre-event open water.
- Vegetation scene selection samples both windows evenly (it compared late-summer with early-summer scenes).
- Burn fuel gate lowered from NDVI 0.2 to 0.10 so sparse shrubland can burn.
- The web page is revalidated on upgrade instead of served stale from the browser cache.

## [0.1.0] — 2026-09-28

### Added
- Local-first flood MVP: STAC search, deterministic UTM grids, S1/S2/DEM compositing with caching, physics
  baseline, GPFT-mini with sensor dropout and MC-dropout uncertainty, weak-label flood benchmark, CLI, SDK,
  FastAPI jobs, MapLibre web map, COG/GeoJSON/STAC outputs with provenance.
