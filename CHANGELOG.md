# Changelog

All notable changes are recorded here. The project follows [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change APIs and output schemas.

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
