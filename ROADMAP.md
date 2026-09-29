# Roadmap

GeoPulse is heading for a v1.0 that is technically deep, reproducible and polished rather than broad. Status
against the v1.0 definition of done:

## Done (v0.4)

- Data: reproducible STAC search, Sentinel-1 and Sentinel-2 pipelines, Copernicus DEM, deterministic tiling,
  split audit, provenance, read cache.
- Tasks: flood, wildfire/burn (+ dNBR severity), vegetation disturbance.
- ML: physics baselines, GPFT-mini (quality-gated fusion, sensor dropout, MC-dropout uncertainty, temperature
  scaling), multi-task training, out-of-distribution test events, sensor-ablation robustness matrices.
- Engineering: Python package, CLI, SDK, FastAPI with async jobs, COG/GeoJSON/STAC outputs, web map, Docker,
  CI, model registry with checksums, weights and benchmark tiles as GitHub Release assets.
- Platform: the full pipeline in the visitor's browser (WebGPU/WebAssembly, ONNX, self-checked), a PWA on
  GitHub Pages; a desktop app (Windows, macOS, Linux) that runs the Python engine on the user's GPU.

## Next (v0.5 — make the numbers trustworthy)

- [ ] Hand-labelled validation tiles for flood (cloudy and ambiguous pixels), replacing weak-label-only scoring.
- [ ] Multiple seeds and event-level bootstrap confidence intervals.
- [ ] Rangeland and grassland fires in wildfire training; hold out a different ecoregion for testing.
- [ ] Experiment: physics indices (dNBR, NDVI, MNDWI) as extra input channels.
- [ ] Georegistration-error sensitivity test (shift one modality by 0.25–2 px).
- [ ] Error-analysis panels per task (false positives and negatives by cause).

## Later (towards v1.0)

- [ ] Landsat 8/9 adapter and ERA5-Land weather context.
- [ ] Pretrained EO backbone (Clay, Prithvi or AnySat) versus from-scratch, with frozen and fine-tuned probes.
- [ ] Low-shot adaptation curves (1–100 labelled tiles) on unseen regions.
- [ ] Calibration study: does uncertainty rise when a sensor is missing or geography shifts?
- [ ] FP16 export benchmarks; performance regression test in CI.
- [ ] Browser engine: keep downloaded composites in IndexedDB so reruns after a reload skip the downloads.
- [ ] Documentation site, tutorials, benchmark report, demo video.
- [ ] Preprint, if the results justify one.

Ideas and help are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).
