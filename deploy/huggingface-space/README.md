---
title: GeoPulse
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Flood, wildfire and forest-loss maps from Sentinel-1/2
models:
  - nafizrahaman/geopulse-gpft-mini
---

# GeoPulse demo

Pick an example event (flood, wildfire, vegetation disturbance) or draw an area, choose before/after windows, and
GeoPulse fetches Sentinel-1, Sentinel-2 and Copernicus DEM data from the Microsoft Planetary Computer, runs the
multi-task GPFT-mini model and maps the change with uncertainty.

Free CPU hardware: jobs are capped at 150 km² and take a few minutes. Source, docs and benchmarks:
<https://github.com/nrahaman1/geopulse>. Model card: <https://huggingface.co/nafizrahaman/geopulse-gpft-mini>.
