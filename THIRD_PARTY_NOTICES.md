# Third-party notices

GeoPulse's own code is Apache-2.0 (`LICENSE`). It depends on, but does not vendor, the software below; each is
installed from its own distribution under its own license. Data licenses are in `DATA_LICENSES.md`.

## Python dependencies (direct)

| Package | License |
|---|---|
| numpy | BSD-3-Clause (plus 0BSD, MIT, Zlib for bundled parts) |
| rasterio | BSD-3-Clause; its wheels bundle GDAL (MIT/X11-style) and PROJ (MIT) |
| pystac, pystac-client | Apache-2.0 |
| planetary-computer | MIT (© Microsoft) |
| PyYAML | MIT |
| PyTorch (torch) | BSD-3-Clause |
| FastAPI | MIT (on Starlette, BSD-3-Clause; Pydantic, MIT) |
| uvicorn | BSD-3-Clause |

Development only: pytest (MIT), httpx (BSD-3-Clause), ruff (MIT), pre-commit (MIT), onnx and onnxruntime (MIT).

## Bundled into the web and desktop app (app/, from npm)

| Component | License / terms |
|---|---|
| MapLibre GL JS 6 | BSD-3-Clause |
| geotiff.js 3 (and its decoders: pako MIT, zstd, lerc Apache-2.0) | MIT |
| ONNX Runtime Web 1.30 | MIT (© Microsoft) |
| d3-contour 4 and d3-array | ISC |
| Workbox (service worker, via vite-plugin-pwa) | MIT |
| Tauri 2 (desktop shell) and tauri-plugin-opener | Apache-2.0 / MIT |
| uv (bundled with the desktop app, installs its Python engine) | Apache-2.0 / MIT (© Astral) |

Build tools: Vite (MIT), vite-plugin-pwa (MIT), yaml (ISC), @tauri-apps/cli (Apache-2.0 / MIT).

## Services used at run time

| Service | Terms |
|---|---|
| Esri World Imagery and World Dark Gray Canvas tiles | Esri Terms of Use; attribution shown on the map |
| Nominatim place search (OpenStreetMap Foundation) | ODbL data, © OpenStreetMap contributors; Nominatim usage policy (one request per submitted search, no autocomplete) |

Also: Microsoft Planetary Computer STAC API and blob storage (anonymous signed URLs); Google Cloud Storage for
Hansen Global Forest Change tiles; GitHub (Pages for the web app; release assets for installers, checkpoints and
benchmark tiles); the desktop engine's first start downloads Python (python-build-standalone) and packages from PyPI
and download.pytorch.org through uv.
