---
title: GeoPulse
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
license: apache-2.0
short_description: Flood, wildfire & forest-loss maps computed in your browser
models:
  - nafizrahaman/geopulse-gpft-mini
datasets:
  - nafizrahaman/geopulse-bench
---

# GeoPulse

The full GeoPulse platform, running on **your** computer. Search any place (or pick an example), choose before/after
dates and a task (flood, wildfire/burn, vegetation disturbance), and your browser:

1. searches the Microsoft Planetary Computer STAC catalog and streams only the needed Sentinel-1, Sentinel-2 and
   Copernicus DEM windows (HTTP range reads of Cloud-Optimized GeoTIFFs),
2. composites and harmonises them on a 10 m UTM grid,
3. runs the GeoPulse multi-task model (ONNX Runtime Web on WebGPU, or WebAssembly) with MC-dropout uncertainty,
4. maps the result and lets you download GeoTIFFs, GeoJSON polygons, a summary and provenance.

No server does any computing and nothing is uploaded. Source, docs and benchmarks:
<https://github.com/nrahaman1/geopulse> (built with `scripts/build_web.py`).
