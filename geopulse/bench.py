"""GeoPulse-Bench: build tiled training data from an event manifest, split by event, with a leakage audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml
from rasterio.enums import Resampling
from rasterio.features import bounds as geom_bounds
from rasterio.warp import transform_bounds

from . import stac
from .baseline import IGNORE, weak_change, weak_labels
from .data import PIPELINE_VERSION, Log, prepare, read_to_grid
from .grid import Grid, aoi_geometry, bbox_geometry, make_grid
from .model import to_tensors
from .tasks import TASKS

MIN_LABELED = 0.02  # keep tiles with at least 2% labeled pixels
GFC = (
    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/Hansen_GFC-2024-v1.12_{layer}_{tile}.tif"
)
GFC_FOREST = 30  # % tree cover in 2000 that counts as forest


def event_geometry(ev: dict) -> dict:
    return bbox_geometry(*ev["bbox"]) if "bbox" in ev else aoi_geometry(ev["geometry"])


def mtbs_labels(geometry: dict, grid: Grid, year: int) -> np.ndarray:
    """MTBS burn severity -> 1 burned (low/moderate/high), 0 unburned (outside mapped fires, increased greenness).

    'Unburned to low' (1) and the non-mapping mask (6) stay IGNORE. Fires below MTBS's size threshold are not
    mapped, so they would be labelled unburned: pick AOIs around mapped fires.
    """
    items = stac.search("mtbs", geometry, str(year))
    if not items:
        raise ValueError(f"no MTBS mosaic for {year} (the collection covers CONUS, 1984-2018)")
    a = read_to_grid(items[0].assets["burn-severity"].href, grid, Resampling.nearest)  # NaN = background
    out = np.full(a.shape, IGNORE, "uint8")
    out[np.isnan(a) | (a == 5)] = 0
    out[np.isin(a, (2, 3, 4))] = 1
    return out


def _gfc_tiles(geometry: dict) -> list[str]:
    w, s, e, n = geom_bounds(geometry)
    tiles = set()
    for lon in (w, e):
        for lat in (s, n):
            top, left = math.ceil(lat / 10) * 10, math.floor(lon / 10) * 10
            tiles.add(f"{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'E' if left >= 0 else 'W'}")
    return sorted(tiles)


def _gfc(layer: str, geometry: dict, grid: Grid) -> np.ndarray:
    out = np.full((grid.height, grid.width), np.nan, "float32")
    for tile in _gfc_tiles(geometry):
        a = read_to_grid(GFC.format(layer=layer, tile=tile), grid, Resampling.nearest)
        out = np.where(np.isnan(out), a, out)
    return out


def hansen_labels(geometry: dict, grid: Grid, year: int) -> np.ndarray:
    """Hansen et al. Global Forest Change: 1 = stand-replacing loss in `year`; 0 = forest (≥30% cover in 2000)
    with no loss 2001-2024; loss in any other year is IGNORE (it may fall inside or outside the windows)."""
    loss = _gfc("lossyear", geometry, grid)
    cover = _gfc("treecover2000", geometry, grid)
    out = np.full(loss.shape, IGNORE, "uint8")
    out[(loss == 0) & (cover >= GFC_FOREST)] = 0
    out[loss == year - 2000] = 1
    return out


def task_labels(ev: dict, task: str, geometry: dict, grid: Grid, arrays: dict) -> tuple[np.ndarray, str]:
    """(2,H,W) uint8 [task target, change] and a label-source string for the index."""
    spec = ev.get("label")
    if spec is None:
        if task != "flood":
            raise ValueError(f"event {ev['id']}: {task} events need a `label:` source (mtbs, hansen or a raster)")
        return weak_labels(arrays), "weak:s1+s2-consensus"
    if isinstance(spec, str):
        spec = {"source": "raster", "path": spec}
    if spec["source"] == "mtbs":
        target = mtbs_labels(geometry, grid, spec["year"])
    elif spec["source"] == "hansen":
        target = hansen_labels(geometry, grid, spec["year"])
    elif spec["source"] == "raster":
        a = read_to_grid(str(spec["path"]), grid, Resampling.nearest)
        target = np.where(np.isfinite(a), a, IGNORE).astype("uint8")
    else:
        raise ValueError(f"unknown label source {spec['source']!r}")
    source = f"{spec['source']}:{spec.get('year', spec.get('path'))}"
    return np.stack([target, weak_change(arrays)]), source


def build(manifest_path: str | Path, out_root: str | Path = "data/bench", log: Log = print) -> Path:
    manifest_path = Path(manifest_path)
    m = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    task = m.get("task", "flood")
    tile = int(m.get("tile", 256))
    out = Path(out_root) / m["dataset"]
    index = {
        "dataset": m["dataset"],
        "version": m["version"],
        "task": task,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "pipeline": PIPELINE_VERSION,
        "tile": tile,
        "label_source": {},
        "events": {},
        "tiles": [],
    }
    audit(m["events"])
    for ev in m["events"]:
        log(f"— event {ev['id']} ({ev['split']})")
        geometry = event_geometry(ev)
        grid = make_grid(geometry, m.get("resolution", 10))
        if min(grid.width, grid.height) < tile:
            raise ValueError(f"event {ev['id']} AOI is smaller than one {tile}px tile")
        spec = TASKS[task]
        inp = prepare(geometry, grid, ev["pre"], ev["post"], ("s1", "s2"), log, spec["composite"], spec["scenes"])
        labels, index["label_source"][ev["id"]] = task_labels(ev, task, geometry, grid, inp.arrays)
        t = to_tensors(inp.arrays)
        (out / ev["id"]).mkdir(parents=True, exist_ok=True)
        kept = 0
        for r, c in grid.tiles(tile):
            lab = labels[:, r : r + tile, c : c + tile]
            if (lab[0] != IGNORE).mean() < MIN_LABELED:
                continue
            rel = f"{ev['id']}/{r}_{c}.npz"
            np.savez_compressed(
                out / rel, labels=lab, **{k: v[:, r : r + tile, c : c + tile].astype("float16") for k, v in t.items()}
            )
            lab0 = lab[0][lab[0] != IGNORE]
            index["tiles"].append(
                {
                    "event": ev["id"],
                    "split": ev["split"],
                    "path": rel,
                    "positive_fraction": round(float(lab0.mean()), 4),
                }
            )
            kept += 1
        labeled = labels[0] != IGNORE
        index["events"][ev["id"]] = {
            "split": ev["split"],
            "bounds": list(transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)),
            "tiles": kept,
            "labeled_fraction": round(float(labeled.mean()), 4),
            "positive_fraction": round(float(labels[0][labeled].mean()), 4) if labeled.any() else None,
            "scenes": {k: [s["id"] for s in v] for k, v in inp.scenes.items()},
            "warnings": inp.warnings,
        }
        log(f"✓ {ev['id']}: {kept} tiles, {labeled.mean():.0%} labeled")
    (out / "index.json").write_text(json.dumps(index, indent=2))
    log(f"✓ Dataset written: {out} ({len(index['tiles'])} tiles)")
    return out


def audit(events: list[dict]) -> None:
    """Fail on geographic leakage: an event id in two splits, or overlapping AOIs assigned to different splits."""
    seen: set[str] = set()
    boxes = []
    for ev in events:
        if ev["id"] in seen:
            raise ValueError(f"event {ev['id']} appears more than once")
        seen.add(ev["id"])
        w, s, e, n = geom_bounds(event_geometry(ev))
        for other, (ow, os_, oe, on) in boxes:
            if other["split"] != ev["split"] and w < oe and ow < e and s < on and os_ < n:
                raise ValueError(f"leakage: {ev['id']} ({ev['split']}) overlaps {other['id']} ({other['split']})")
        boxes.append((ev, (w, s, e, n)))
