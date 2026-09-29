# GeoPulse architecture (v0.2, local-first: flood, wildfire, vegetation disturbance)

```text
request (AOI, windows, task, sensors)      geopulse/pipeline.py::make_request   — one validator for CLI, API, SDK
        │
        ▼
deterministic UTM grid                     geopulse/grid.py                     — snapped 10 m pixel edges
        │
        ▼
STAC search (Planetary Computer)           geopulse/stac.py                     — sensors, not collections
        │
        ▼
warp COG windows onto the grid             geopulse/data.py::read_to_grid       — no intermediate resampling
per-task compositing (tasks.py)            geopulse/data.py::composite          — cached by (items, grid, method)
        │
        ▼
model registry → baseline | GPFT-mini      geopulse/model.py                    — "auto" = newest checkpoint for the task
        │
        ▼
COGs, GeoJSON, summary, provenance,        geopulse/pipeline.py::write_outputs  — {flood|burn|disturbance}_*.tif
STAC item, Web-Mercator PNG layers                                                  + burn_severity.tif
        │
        ▼
FastAPI jobs + MapLibre web map            geopulse/api.py, geopulse/web/index.html
```

Training side: `geopulse/bench.py` (manifest → tiles + label source + leakage audit) → `geopulse/train.py`
(multi-task heads, sensor dropout, masked BCE + Dice, temperature scaling, robustness matrix, baseline scoring).

## Tasks (`geopulse/tasks.py`)

| Task | Output | Pre composite | Post composite | Physics baseline | Benchmark labels |
|---|---|---|---|---|---|
| flood | `flood_*` | median | earliest valid (floods recede) | SAR VV threshold + log-ratio drop; MNDWI > 0; new water only | weak S1+S2 consensus |
| wildfire | `burn_*`, `burn_severity.tif` | median | median (suppresses smoke/cloud) | dNBR > 0.10 on vegetated pixels; SAR VH drop only fills optical gaps | MTBS (USGS/USFS) |
| vegetation | `disturbance_*` | median, same season a year before | median, same season a year after | NDVI drop on canopy pixels + SAR VH drop | Hansen GFC loss year |

## GPFT-mini

```text
S1 pre/post ─ SAR adapter ─┐                        ┌─ per-pixel softmax gate over sensors,
S2 pre/post ─ optical adapter ─┤ quality-gated fusion ├─ unobserved (cloud / missing) → weight 0
                               └──────────┬───────────┘
                   [pre, post, post−pre, |post−pre|] + terrain(DEM rel. height, slope)
                                          │
                              U-Net context (3 levels, GroupNorm, Dropout2d)
                                          │
              heads: one logit per task (flood, wildfire, vegetation) + shared change logit, ÷ temperature
              uncertainty: entropy of MC-dropout mean (8 passes)
```

~0.54 M parameters. Adapters are shared across time (Siamese). GroupNorm instead of BatchNorm because
sensor dropout zeroes whole modalities per sample, which makes batch statistics meaningless. A model trained on
one benchmark has one task head; the multi-task model has three. v0.1 flood checkpoints (no `tasks` key) load as
flood-only.

## Decisions (ADR-style)

**ADR-001 STAC as the discovery layer.** Provider details live in `stac.py` (collection names, signing).
Everything else asks for `s1`, `s2`, `dem`, `mtbs`. Swapping to Earth Search or a private catalog touches one file.

**ADR-002 Warp straight onto a snapped grid.** Each band is read through a `WarpedVRT` whose transform is the
AOI grid, so remote COG windows are fetched once and resampled once. The grid origin is snapped to multiples
of the resolution, which makes grids deterministic and cache keys stable. `tests/test_grid.py` checks that
a whole-pixel offset comes back as an exact shift (no half-pixel error). Remote reads have connect/read timeouts:
without them one stalled HTTP connection hung a job indefinitely.

**ADR-003 Physics before learning.** Compositing and baselines encode sensor physics explicitly:
SAR is composited in linear power and converted to dB afterwards; pre-event SAR is restricted to the post-event
relative orbit; Sentinel-2 processing baseline ≥ 04.00 has its +1000 DN offset removed; flood post-event composites
take the earliest valid observation because floods recede; SAR flood detection uses absolute backscatter *and*
log-ratio change, because wind-roughened floodwater can sit well above −18 dB (≈ −14 dB at Hamburg 2019).
Burn severity classes use the Key & Benson (2006) dNBR breaks. C-band SAR sees fire only indirectly, so the burn
baseline uses it only where optical is missing and floors its uncertainty at 0.5.

**ADR-004 COG + GeoJSON + STAC item outputs.** Rasters are written through GDAL's COG driver; every run writes
`provenance.json` (scene IDs, compositing, grid, model hash, git SHA) and a STAC Item so results can be catalogued.

**ADR-005 Independent labels where they exist, weak labels where they don't.** Wildfire uses MTBS and vegetation
uses Hansen GFC: both come from Landsat and analysts, independent of GeoPulse's Sentinel inputs, so the physics
baseline and the learned model can be compared fairly. Flood still uses S1+S2 consensus labels, and scores
against those measure agreement with the consensus, not ground truth. Any event can point `label:` at a
hand-labelled raster.

**ADR-006 Event-level splits with an audit.** Tiles never split randomly. `bench.audit` rejects a manifest where
an event appears twice or AOIs of different splits overlap. Each test event is out of distribution: a continent
(flood: Pakistan), an ecoregion (wildfire: Great Basin sagebrush) or a biome (vegetation: Amazon) absent from training.

**ADR-007 Seasonal design for vegetation.** Disturbance is measured between the same season in different years
(label year Y: pre in Y−1, post in Y+1), so senescence and green-up cancel. Requests whose windows are more than
45 days apart in the annual cycle get an explicit warning.

**ADR-008 One masked loss for single- and multi-task training.** Each benchmark's labels are expanded into the
model's channels; other tasks' channels are IGNORE. A flood tile never pushes the burn head, and the same code trains
one head or three.

**ADR-009 Local-first services.** One in-process worker thread and JSON job records on disk. Enough for a single
machine; the `ponytail:` comment in `api.py` marks where Redis + RQ/Celery goes when jobs must survive restarts.

**ADR-010 Apache-2.0 for code; data licenses tracked separately** in `DATA_LICENSES.md`.
