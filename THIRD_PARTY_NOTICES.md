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
| huggingface-hub | Apache-2.0 |

Development only: pytest (MIT), httpx (BSD-3-Clause), ruff (MIT), pre-commit (MIT), onnx and onnxruntime (MIT).

## Loaded by the web map at run time (not redistributed)

| Component | License / terms |
|---|---|
| MapLibre GL JS 4.7.1 (from unpkg CDN) | BSD-3-Clause |
| geotiff.js 3.0.5 (from jsDelivr, in-browser engine) | MIT |
| ONNX Runtime Web 1.30.0 (from jsDelivr, in-browser engine) | MIT (© Microsoft) |
| d3-contour 4.0.2 and d3-array (from jsDelivr, in-browser engine) | ISC |
| Esri World Imagery and World Dark Gray Canvas tiles | Esri Terms of Use; attribution shown on the map |
| Nominatim place search (OpenStreetMap Foundation) | ODbL data, © OpenStreetMap contributors; Nominatim usage policy (one request per submitted search, no autocomplete) |

## Services used at run time

Microsoft Planetary Computer STAC API and blob storage (anonymous signed URLs); Google Cloud Storage for Hansen
Global Forest Change tiles; Hugging Face Hub for published checkpoints, ONNX models and benchmark tiles; GitHub Pages hosts the static platform.
