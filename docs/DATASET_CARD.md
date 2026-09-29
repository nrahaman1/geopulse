# GeoPulse-Bench v0.1

Event-split benchmarks for multimodal Earth-change mapping, built by
[GeoPulse](https://github.com/nrahaman1/geopulse) from Sentinel-1 RTC, Sentinel-2 L2A and Copernicus DEM GLO-30
(Microsoft Planetary Computer). Published as the assets of the
[`bench-v1` release](https://github.com/nrahaman1/geopulse/releases/tag/bench-v1): one zip per benchmark and
`SHA256SUMS`. Download and verify with `geopulse dataset pull` (or `geopulse dataset pull geopulse-bench-wildfire`).
Each folder is one benchmark:

| Folder | Task | Events (train / val / test) | Tiles | Labels |
|---|---|---|---:|---|
| `geopulse-bench-flood` | flood | 5 / 1 / 1 (test: Pakistan 2022) | 236 | weak: S1+S2 consensus |
| `geopulse-bench-wildfire` | burned area | 5 / 1 / 1 (test: Martin Fire 2018, Nevada sagebrush) | 212 | MTBS burn severity |
| `geopulse-bench-vegetation` | stand-replacing forest loss | 3 / 1 / 1 (test: Pará, Brazil 2020) | 176 | Hansen GFC v1.12 loss year |

## Format

Each tile is `<event>/<row>_<col>.npz`, 256 × 256 px at 10 m on a UTM grid, float16, normalised as GeoPulse's
`to_tensors` does (see `geopulse/model.py::NORM`):

- `s1_pre`, `s1_post` (2 × H × W: VV, VH dB), `s2_pre`, `s2_post` (6 × H × W: B02 B03 B04 B08 B11 B12), each with a
  `*_valid` mask (1 × H × W); `dem` (2 × H × W: height above the AOI median / 20, slope / 10).
- `labels` (2 × H × W, uint8): `[task target, generic change]`, 0 / 1, 255 = ignore.

`index.json` lists every tile with its event and split, the manifest SHA-256, label source per event, scene IDs
used per event and period, and event bounds.

## How it was built

`geopulse dataset build configs/data/<task>_bench.yaml` (manifests are in the GeoPulse repository): STAC search per
event window, per-task compositing (flood post = earliest valid, otherwise median; vegetation samples both windows
evenly across the same season), reprojection straight onto a snapped 10 m grid, then tiling. Tiles with < 2 %
labelled pixels are dropped.

## Labels and known issues

- **Flood labels are weak**: pixels where SAR and optical agree confidently. Scores against them measure agreement
  with that consensus, not ground truth; the ambiguous pixels (cloud, flooded vegetation, urban) are ignored.
- **Wildfire**: MTBS low/moderate/high = burned; outside mapped fires and "increased greenness" = unburned;
  "unburned to low" and non-mapping areas = ignored. Fires below MTBS's size threshold are not mapped.
- **Vegetation**: Hansen loss in the label year = 1; forest (≥ 30 % cover in 2000) with no loss 2001–2024 = 0; loss in
  other years = ignored. Partial disturbance (thinning, partial windthrow) counts as "no loss" in Hansen.
- Positive fractions vary widely by event (vegetation: 1–49 %). Change labels are conservative consensus.

## Licenses and required notices

Mixed open data; details in `DATA_LICENSES.md`:

- Contains modified Copernicus Sentinel data (2017–2023).
- Sentinel-1 RTC © Catalyst / Microsoft, CC BY 4.0.
- Copernicus DEM GLO-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under
  COPERNICUS by the European Union and ESA.
- MTBS: U.S. Government work (USGS / USDA Forest Service), public domain.
- Hansen et al. (2013) Global Forest Change v1.12, CC BY 4.0.

## Citation

Cite GeoPulse (see `CITATION.cff` in the repository) and the label sources: MTBS (mtbs.gov) and
Hansen, M. C. et al. (2013), *Science* 342, 850–853.
