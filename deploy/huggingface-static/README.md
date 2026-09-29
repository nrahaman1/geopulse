---
title: GeoPulse
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
license: apache-2.0
short_description: Flood, wildfire and forest-loss maps from Sentinel-1/2
models:
  - nafizrahaman/geopulse-gpft-mini
datasets:
  - nafizrahaman/geopulse-bench
---

# GeoPulse — online showcase

The GeoPulse web map with precomputed results for 13 example events (floods, wildfires, forest loss), made by the
multi-task GPFT-mini model from Sentinel-1, Sentinel-2 and the Copernicus DEM. Search a place, explore the layers,
compare before/after imagery and download the results.

To analyse **any** place and dates, run the full platform for free in
[GitHub Codespaces](https://codespaces.new/nrahaman1/geopulse?quickstart=1) or locally — see
<https://github.com/nrahaman1/geopulse>. Built with `scripts/export_static.py`.
