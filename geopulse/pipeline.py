"""End-to-end inference: request -> grid -> inputs -> model -> COGs, GeoJSON, summary, provenance, map layers."""

from __future__ import annotations

import datetime as dt
import json
import time
import warnings
from itertools import pairwise
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import shapes, sieve
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, transform_geom

from . import baseline
from . import model as models
from .baseline import IGNORE
from .data import PIPELINE_VERSION, Inputs, Log, prepare
from .grid import Grid, aoi_geometry, area_km2, make_grid
from .tasks import TASKS

SENSOR_ALIASES = {"s1": "s1", "sentinel-1": "s1", "sentinel1": "s1", "s2": "s2", "sentinel-2": "s2", "sentinel2": "s2"}
MAX_AOI_KM2 = 1000.0
MAX_WINDOW_DAYS = 120
MAX_SEASON_GAP_DAYS = 45  # vegetation: windows further apart in the annual cycle risk mapping phenology
REVIEW_UNCERTAINTY = 0.8  # abstain ("review recommended") above this normalized entropy
THRESHOLD = 0.5
SEVERITY_CLASSES = ("low", "moderate-low", "moderate-high", "high")  # Key & Benson (2006) dNBR classes 1-4


class RequestError(ValueError):
    pass


def _window(w) -> tuple[dt.date, dt.date]:
    a, b = w.split("/") if isinstance(w, str) else w
    try:
        a, b = dt.date.fromisoformat(str(a)), dt.date.fromisoformat(str(b))
    except ValueError as e:
        raise RequestError(f"bad date window {w!r}: use YYYY-MM-DD/YYYY-MM-DD") from e
    if a > b or (b - a).days > MAX_WINDOW_DAYS:
        raise RequestError(f"window {w!r} must be ordered and at most {MAX_WINDOW_DAYS} days")
    return a, b


def season_gap(b: tuple[dt.date, dt.date], a: tuple[dt.date, dt.date]) -> int:
    """Days between the windows' midpoints on the annual cycle (0 = same season, 182 = opposite)."""
    doy = [(w[0] + (w[1] - w[0]) / 2).timetuple().tm_yday for w in (b, a)]
    d = abs(doy[0] - doy[1])
    return min(d, 365 - d)


def make_request(
    aoi: dict,
    before,
    after,
    task: str = "flood",
    sensors=("s1", "s2"),
    model: str = "auto",
    resolution: float = 10.0,
    max_km2: float = MAX_AOI_KM2,
) -> dict:
    """Validate and normalize a request. Shared by the CLI, API and SDK so every entry point has the same rules."""
    if task not in TASKS:
        raise RequestError(f"unknown task {task!r}; supported: {list(TASKS)}")
    if isinstance(sensors, str):
        sensors = sensors.split(",")
    try:
        sensors = tuple(dict.fromkeys(SENSOR_ALIASES[s.strip().lower()] for s in sensors))
    except KeyError as e:
        raise RequestError(f"unknown sensor {e.args[0]!r}; use s1, s2") from None
    if not sensors:
        raise RequestError("at least one sensor is required")
    try:
        geometry = aoi_geometry(aoi)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise RequestError(f"invalid AOI: {e}") from None
    km2 = area_km2(geometry)
    if not 0.01 <= km2 <= max_km2:
        raise RequestError(f"AOI area {km2:.1f} km² outside allowed range 0.01–{max_km2:g} km²")
    b, a = _window(before), _window(after)
    if b[1] >= a[0]:
        raise RequestError("the before window must end before the after window starts")
    if not 10 <= resolution <= 100:
        raise RequestError("resolution must be 10–100 m")
    notes = []
    if TASKS[task]["seasonal"] and (gap := season_gap(b, a)) > MAX_SEASON_GAP_DAYS:
        notes.append(
            f"before/after windows are {gap} days apart in the seasonal cycle: phenology may be mapped as "
            "disturbance; use the same season in different years"
        )
    return {
        "task": task,
        "aoi": geometry,
        "before": f"{b[0]}/{b[1]}",
        "after": f"{a[0]}/{a[1]}",
        "sensors": list(sensors),
        "model": model,
        "resolution": float(resolution),
        "aoi_km2": round(km2, 3),
        "warnings": notes,
    }


def run(request: dict, out_dir: str | Path, log: Log = print, inputs: Inputs | None = None) -> dict:
    """Run a validated request (see make_request); returns summary.json content. `inputs` skips acquisition."""
    t0 = time.time()
    task = request["task"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    geometry = request["aoi"]
    grid = inputs.grid if inputs else make_grid(geometry, request["resolution"])
    log(f"✓ Grid {grid.crs}, {grid.width}×{grid.height} px @ {grid.res:g} m")
    if inputs is None:
        spec = TASKS[task]
        inputs = prepare(
            geometry,
            grid,
            request["before"],
            request["after"],
            tuple(request["sensors"]),
            log,
            spec["composite"],
            spec["scenes"],
        )
    inputs.warnings[:0] = request.get("warnings", [])
    if not any(k in inputs.arrays for k in ("s1_post", "s2_post")):
        raise RequestError("no usable post-event observations for this AOI/window: " + "; ".join(inputs.warnings))
    if not any(k in inputs.arrays for k in ("s1_pre", "s2_pre")):
        if task != "flood":
            raise RequestError(f"{task} mapping needs pre-event observations; none were usable in the before window")
        inputs.warnings.append("no pre-event observations: permanent water cannot be separated from flood water")
    card = models.resolve(request.get("model", "auto"), task)
    log(f"✓ Model loaded: {card['model_id']} v{card['version']}")
    maps = models.predict(card, inputs.arrays, tuple(request["sensors"]), task)
    inside = grid.mask(geometry)
    for k in ("target", "change", "uncertainty"):
        maps[k] = np.where(inside, maps[k], np.nan).astype("float32")
    if task != "flood":
        # Physical constraint, not learned: a model never shown the sea once painted ocean as burned (Palisades 2025).
        water = inside & baseline.pre_water(inputs.arrays)
        maps["target"][water], maps["uncertainty"][water] = 0.0, 0.0
    if "severity" in maps:  # classes only where the task map calls the pixel burned
        burned = np.nan_to_num(maps["target"]) >= THRESHOLD
        valid = inside & np.isfinite(maps["target"]) & (maps["severity"] != IGNORE)
        maps["severity"] = np.where(valid, np.where(burned, maps["severity"], 0), IGNORE).astype("uint8")
    log("✓ Inference complete")
    summary = write_outputs(out, request, grid, inputs, maps, card, time.time() - t0)
    log(f"✓ Outputs: {out}")
    return summary


# --------------------------------------------------------------------------- outputs


def write_cog(path: Path, arr: np.ndarray, grid: Grid, nodata=np.nan) -> None:
    arr = arr if arr.ndim == 3 else arr[None]
    profile = dict(
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=arr.shape[0],
        dtype=arr.dtype.name,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
    )
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(arr)
        rio_copy(mem.name, str(path), driver="COG", COMPRESS="DEFLATE", PREDICTOR="2" if arr.dtype.kind == "f" else "1")


def _rgb(s2: np.ndarray) -> np.ndarray:
    rgb = np.clip(s2[[2, 1, 0]] / 0.3, 0, 1) ** (1 / 1.4)
    return np.nan_to_num(rgb * 255).astype("uint8")


def _gray(s1: np.ndarray) -> np.ndarray:
    g = np.clip((s1[0] + 25) / 25, 0, 1)
    return np.repeat(np.nan_to_num(g * 255)[None], 3, 0).astype("uint8")


def _ramp(v: np.ndarray, stops: list[tuple[float, tuple[int, int, int, int]]]) -> np.ndarray:
    """Piecewise-linear RGBA colormap; NaN -> transparent."""
    xs = [s for s, _ in stops]
    out = np.stack([np.interp(np.nan_to_num(v, nan=-1), xs, [c[i] for _, c in stops], left=0) for i in range(4)])
    out[:, np.isnan(v)] = 0
    return out.astype("uint8")


def _prob_palette(rgb: tuple[int, int, int], deep: tuple[int, int, int]) -> list:
    return [(0.2, (*rgb, 0)), (0.5, (*rgb, 150)), (1.0, (*deep, 235))]


TARGET_COLORS = {  # (probability ramp, deep end, quicklook overlay)
    "flood": ((40, 120, 255), (20, 90, 255)),
    "burn": ((235, 70, 30), (200, 20, 10)),
    "disturbance": ((190, 70, 235), (150, 30, 220)),
}
PALETTES = {
    "change": [(0.2, (255, 170, 0, 0)), (0.5, (255, 170, 0, 140)), (1.0, (255, 120, 0, 230))],
    "uncertainty": [
        (0.0, (60, 20, 90, 0)),
        (0.3, (120, 40, 140, 110)),
        (0.7, (230, 80, 90, 190)),
        (1.0, (255, 210, 60, 230)),
    ],
}
SEVERITY_RGBA = np.array(
    [(0, 0, 0, 0), (127, 255, 212, 220), (255, 255, 0, 220), (255, 140, 0, 230), (255, 0, 0, 235)], "uint8"
)


def _web_layer(path: Path, rgba: np.ndarray, grid: Grid) -> list[float]:
    """Reproject an RGB(A) uint8 image to Web Mercator PNG so MapLibre can drape it exactly. Returns lon/lat bounds."""
    if rgba.shape[0] == 3:
        rgba = np.concatenate([rgba, np.where(rgba.any(0), 255, 0)[None].astype("uint8")])
    dst_crs = "EPSG:3857"
    transform, w, h = calculate_default_transform(grid.crs, dst_crs, grid.width, grid.height, *grid.bounds)
    dst = np.zeros((4, h, w), "uint8")
    reproject(
        rgba,
        dst,
        src_transform=grid.transform,
        src_crs=grid.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    _png(path, dst)
    west, north = transform.c, transform.f
    east, south = west + w * transform.a, north + h * transform.e
    return list(transform_bounds(dst_crs, "EPSG:4326", west, south, east, north))


def _png(path: Path, img: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)  # PNGs are pictures; bounds live in layers.json
        with rasterio.open(
            path, "w", driver="PNG", width=img.shape[2], height=img.shape[1], count=img.shape[0], dtype="uint8"
        ) as f:
            f.write(img)


def _polygons(prob: np.ndarray, grid: Grid, limit: int = 5000) -> dict:
    mask = np.nan_to_num(prob) >= THRESHOLD
    mask = sieve(mask.astype("uint8"), size=20).astype(bool)  # drop < 0.2 ha specks at 10 m
    feats = []
    for geom, _ in shapes(mask.astype("uint8"), mask=mask, transform=grid.transform):
        feats.append(
            {
                "type": "Feature",
                "geometry": transform_geom(grid.crs, "EPSG:4326", geom, precision=6),
                "properties": {"area_km2": round(_ring_area(geom), 5)},
            }
        )
    feats.sort(key=lambda f: -f["properties"]["area_km2"])
    return {"type": "FeatureCollection", "features": feats[:limit]}


def _ring_area(geom: dict) -> float:
    total = 0.0
    for k, ring in enumerate(geom["coordinates"]):
        a = 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in pairwise(ring)))
        total += a if k == 0 else -a
    return total / 1e6


def write_outputs(out: Path, request: dict, grid: Grid, inputs: Inputs, maps: dict, card: dict, runtime: float) -> dict:
    spec = TASKS[request["task"]]
    target = spec["target"]
    prob, change, unc = maps["target"], maps["change"], maps["uncertainty"]
    files = {"probability": f"{target}_probability.tif", "extent": f"{target}_extent.geojson"}
    for name, arr in ((files["probability"], prob), ("change_probability.tif", change), ("uncertainty.tif", unc)):
        write_cog(out / name, arr, grid)
    if "severity" in maps:
        files["severity"] = f"{target}_severity.tif"
        write_cog(out / files["severity"], maps["severity"], grid, nodata=IGNORE)
    (out / files["extent"]).write_text(json.dumps(_polygons(prob, grid)))

    observed = np.isfinite(prob)
    px_km2 = grid.res**2 / 1e6
    hit = observed & (prob >= THRESHOLD)
    review = observed & (unc >= REVIEW_UNCERTAINTY)
    conf = np.maximum(prob, 1 - prob)
    a = inputs.arrays
    rgb, deep = TARGET_COLORS[target]

    # Quicklook: the task map over post-event optical (or SAR when optical is missing).
    base = (
        _rgb(a["s2_post"])
        if "s2_post" in a
        else _gray(a["s1_post"])
        if "s1_post" in a
        else np.zeros((3, grid.height, grid.width), "uint8")
    )
    ql = base.astype("float32")
    ql[:, hit] = 0.45 * ql[:, hit] + 0.55 * np.array(deep)[:, None]
    _png(out / "quicklook.png", ql.astype("uint8"))

    # Web map layers (Web Mercator PNGs + lon/lat bounds).
    (out / "layers").mkdir(exist_ok=True)
    layers = {"target": _ramp(prob, _prob_palette(rgb, deep))}
    layers |= {k: _ramp(maps[k], PALETTES[k]) for k in ("change", "uncertainty")}
    if "severity" in maps:
        layers["severity"] = np.moveaxis(
            SEVERITY_RGBA[np.where(maps["severity"] == IGNORE, 0, maps["severity"])], -1, 0
        )
    inside = grid.mask(request["aoi"])
    for p in ("pre", "post"):
        if f"s2_{p}" in a:
            layers[f"s2_{p}"] = _rgb(a[f"s2_{p}"]) * inside
        if f"s1_{p}" in a:
            layers[f"s1_{p}"] = _gray(a[f"s1_{p}"]) * inside
    bounds = None
    for k, img in layers.items():
        bounds = _web_layer(out / "layers" / f"{k}.png", img, grid)
    (out / "layers.json").write_text(
        json.dumps({"bounds": bounds, "layers": list(layers), "target": target, "extent": files["extent"]})
    )

    summary = {
        "task": request["task"],
        "target": target,
        "affected_label": spec["verb"],
        "model": {"id": card["model_id"], "version": card["version"]},
        "aoi_km2": request.get("aoi_km2"),
        "observed_km2": round(float(observed.sum() * px_km2), 3),
        "affected_km2": round(float(hit.sum() * px_km2), 3),
        "review_km2": round(float(review.sum() * px_km2), 3),
        "mean_confidence_affected": round(float(conf[hit].mean()), 3) if hit.any() else None,
        "mean_uncertainty": round(float(np.nanmean(unc)), 3) if observed.any() else None,
        "modality_weights": maps.get("weights"),
        "scenes": {k: len(v) for k, v in inputs.scenes.items()},
        "warnings": inputs.warnings,
        "files": files,
        "runtime_s": round(runtime, 1),
        "decision_policy": {"threshold": THRESHOLD, "review_if_uncertainty_ge": REVIEW_UNCERTAINTY},
    }
    if "severity" in maps:
        summary["severity_km2"] = {
            name: round(float((maps["severity"] == k + 1).sum() * px_km2), 3) for k, name in enumerate(SEVERITY_CLASSES)
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    created = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    provenance = {
        "model": f"{card['model_id']}-v{card['version']}",
        "checkpoint_sha256": card.get("checkpoint_sha256"),
        "git_sha": models.git_sha(),
        "dataset_pipeline": PIPELINE_VERSION,
        "request": {k: v for k, v in request.items() if k != "aoi"},
        "compositing": dict(zip(("pre", "post"), spec["composite"], strict=True)) | {"scenes": spec["scenes"]},
        "sensors": {k: [s["id"] for s in v] for k, v in inputs.scenes.items()},
        "scenes": inputs.scenes,
        "target_crs": grid.crs,
        "resolution_m": grid.res,
        "grid": grid.to_dict(),
        "inference_precision": "fp32",
        "created": created,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    w, s, e, n = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)
    cog = "image/tiff; application=geotiff; profile=cloud-optimized"
    assets = {
        "classification": {"href": f"./{files['probability']}", "type": cog, "roles": ["data"]},
        "change_probability": {"href": "./change_probability.tif", "type": cog, "roles": ["data"]},
        "uncertainty": {"href": "./uncertainty.tif", "type": cog, "roles": ["data"]},
        "vectorized_change": {"href": f"./{files['extent']}", "type": "application/geo+json", "roles": ["data"]},
        "quicklook": {"href": "./quicklook.png", "type": "image/png", "roles": ["overview"]},
        "provenance": {"href": "./provenance.json", "type": "application/json", "roles": ["metadata"]},
    }
    if "severity" in files:
        assets["severity"] = {"href": f"./{files['severity']}", "type": cog, "roles": ["data"]}
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": ["https://stac-extensions.github.io/projection/v1.1.0/schema.json"],
        "id": f"geopulse-{request['task']}-{created.replace(':', '')}",
        "geometry": request["aoi"],
        "bbox": [w, s, e, n],
        "properties": {
            "datetime": None,
            "start_datetime": request["before"].split("/")[0] + "T00:00:00Z",
            "end_datetime": request["after"].split("/")[1] + "T23:59:59Z",
            "created": created,
            "proj:code": grid.crs,
            "geopulse:task": request["task"],
            "geopulse:model": provenance["model"],
            "geopulse:affected_km2": summary["affected_km2"],
        },
        "assets": assets,
        "links": [],
    }
    (out / "item.json").write_text(json.dumps(item, indent=2))
    return summary
