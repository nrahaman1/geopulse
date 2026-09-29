# Changelog

All notable changes are recorded here. The project follows [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change APIs and output schemas.

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
- Live demo on Hugging Face Spaces; `GEOPULSE_MC_PASSES` for small CPU deployments.
- Dockerfile, compose file and a Hugging Face Space definition; public mode (`GEOPULSE_PUBLIC=1`) that hides other
  visitors' jobs; `GEOPULSE_MAX_JOB_KM2`.
- CI (lint, tests on Python 3.11–3.14, wheel contents, lockfile), Docker smoke test, tag-driven release to PyPI,
  Dependabot, pre-commit, issue and PR templates, governance files.

### Changed
- Distribution renamed to `geopulse-eo` on PyPI (the import name and CLI stay `geopulse`).
- Outputs are named by target (`flood_*`, `burn_*`, `disturbance_*`); summaries use `affected_km2`.
- SAR change evidence uses a softer slope (speckle-aware "stable" band).

### Fixed
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
