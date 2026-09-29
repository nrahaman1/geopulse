"""Acquisition and harmonization: STAC items -> arrays on one grid -> pre/post composites with quality masks."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from . import stac
from .grid import Grid

PIPELINE_VERSION = "0.1.0"

S1_BANDS = ("vv", "vh")  # gamma0 RTC, linear power on the provider; converted to dB after compositing
S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")  # blue, green, red, NIR, SWIR1, SWIR2
S2_INVALID_SCL = (0, 1, 3, 8, 9, 10)  # nodata, saturated, cloud shadow, cloud medium/high, cirrus
MAX_SCENES = 4  # per sensor per period, the scenes closest in time to the event
MAX_S2_CLOUD = 60.0

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    # Without these a stalled connection blocks a read (and the job worker) forever.
    "GDAL_HTTP_CONNECTTIMEOUT": "20",
    "GDAL_HTTP_TIMEOUT": "90",
    "VSI_CACHE": "TRUE",
}

Log = Callable[[str], None]
Progress = Callable[[float, str], None]  # (fraction done 0..1, what is happening)
SENSOR_NAMES = {"s1": "Sentinel-1", "s2": "Sentinel-2"}
PERIOD_NAMES = {"pre": "before", "post": "after"}


@dataclass
class Inputs:
    """Everything the models need for one AOI. Arrays are float32 on `grid`, NaN where invalid.

    s1_pre/s1_post: (2,H,W) VV,VH in dB.  s2_pre/s2_post: (6,H,W) surface reflectance.
    dem: (2,H,W) elevation m, slope deg.
    """

    grid: Grid
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    scenes: dict[str, list[dict]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def valid(self, key: str) -> np.ndarray:
        return np.isfinite(self.arrays[key]).all(0)


def cache_dir() -> Path:
    return Path(os.environ.get("GEOPULSE_CACHE", "data/cache"))


def read_to_grid(href: str, grid: Grid, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Warp one band straight onto the target grid (remote COG windows are fetched lazily). NaN = no data."""
    with (
        rasterio.Env(**GDAL_ENV),
        rasterio.open(href) as src,
        WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
            add_alpha=src.nodata is None,
        ) as vrt,
    ):
        a = vrt.read(1).astype("float32")
        # No declared nodata: the added alpha band marks pixels outside the source footprint.
        a[(vrt.read(2) if src.nodata is None else vrt.read_masks(1)) == 0] = np.nan
    return a


def _cached(parts: list, compute: Callable[[], np.ndarray]) -> np.ndarray:
    """Deterministic cache keyed by inputs (item ids, grid, pipeline version), not by call site."""
    key = hashlib.sha1(json.dumps([PIPELINE_VERSION, *parts], sort_keys=True).encode()).hexdigest()[:20]
    path = cache_dir() / f"{key}.npy"
    if path.exists():
        return np.load(path)
    arr = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def s2_scene(item, grid: Grid) -> np.ndarray | None:
    """(6,H,W) reflectance with clouds/shadow/nodata as NaN, or None if the scene is essentially unusable."""
    scl = read_to_grid(item.assets["SCL"].href, grid, Resampling.nearest)
    ok = np.isfinite(scl) & ~np.isin(scl, S2_INVALID_SCL)
    if ok.mean() < 0.02:
        return None
    # Processing baseline >= 04.00 (Jan 2022+) stores DN with a +1000 offset.
    offset = -1000.0 if float(item.properties.get("s2:processing_baseline") or 0) >= 4 else 0.0
    with ThreadPoolExecutor(len(S2_BANDS)) as pool:
        bands = list(pool.map(lambda b: read_to_grid(item.assets[b].href, grid), S2_BANDS))
    x = (np.stack(bands) + offset) / 10000.0
    x[:, ~ok] = np.nan
    return x


def s1_scene(item, grid: Grid) -> np.ndarray:
    """(2,H,W) VV/VH gamma0 in linear power, NaN where missing."""
    x = np.stack([read_to_grid(item.assets[b].href, grid) for b in S1_BANDS])
    x[~(x > 0)] = np.nan
    return x


def composite(
    sensor: str, items: list, grid: Grid, method: str = "median", done: Callable[[int], None] | None = None
) -> np.ndarray:
    """Per-pixel composite of time-ordered scenes.

    median: robust stable baseline (pre-event).  first: earliest valid observation (post-event), because floods
    recede and a multi-date median would dilute the very signal we are looking for.
    """

    def compute():
        load = s2_scene if sensor == "s2" else s1_scene
        count = itertools.count(1)

        def one(item):  # reports each loaded scene to `done` (progress)
            scene = load(item, grid)
            if done:
                done(next(count))
            return scene

        with ThreadPoolExecutor(4) as pool:
            scenes = [s for s in pool.map(one, items) if s is not None]
        if not scenes:
            return np.full((len(S2_BANDS if sensor == "s2" else S1_BANDS), grid.height, grid.width), np.nan, "float32")
        stack = np.stack(scenes)
        if method == "first":
            idx = np.isfinite(stack).all(1).argmax(0)
            out = np.take_along_axis(stack, idx[None, None], 0)[0]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels stay NaN
                out = np.nanmedian(stack, axis=0)
        # Composite in linear power, then dB: never mix representations.
        return (10 * np.log10(out) if sensor == "s1" else out).astype("float32")

    return _cached([sensor, method, sorted(i.id for i in items), grid.to_dict()], compute)


def dem(items: list, grid: Grid) -> np.ndarray:
    def compute():
        elev = np.full((grid.height, grid.width), np.nan, "float32")
        for item in items:
            a = read_to_grid(item.assets["data"].href, grid)
            elev = np.where(np.isnan(elev), a, elev)
        gy, gx = np.gradient(elev, grid.res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
        return np.stack([elev, slope])

    return _cached(["dem", sorted(i.id for i in items), grid.to_dict()], compute)


def select(items: list, period: str, how: str = "closest") -> list:
    """Pick at most MAX_SCENES of the (oldest-first) items.

    closest: nearest the event boundary (end of the pre window, start of the post window) — floods, fires.
    spread: evenly across the window, so two same-season windows sample the same phenology. "closest" would pair
    late-summer pre scenes with early-summer post scenes and turn green-up into apparent change.
    """
    if how == "spread" and len(items) > MAX_SCENES:
        idx = np.linspace(0, len(items) - 1, MAX_SCENES).round().astype(int)
        return [items[i] for i in dict.fromkeys(idx.tolist())]
    return items[-MAX_SCENES:] if period == "pre" else items[:MAX_SCENES]


def prepare(
    geometry: dict,
    grid: Grid,
    before: str,
    after: str,
    sensors: tuple[str, ...] = ("s1", "s2"),
    log: Log = print,
    methods: tuple[str, str] = ("median", "first"),
    selection: str = "closest",
    progress: Progress | None = None,
) -> Inputs:
    """Search, fetch and composite every sensor for both periods (pre, post methods; scene selection), plus terrain.

    `progress` gets the fraction of this stage done: one equal share per composite and the DEM, advanced per scene."""
    inp = Inputs(grid)
    windows = {"pre": before, "post": after}
    method = dict(zip(windows, methods, strict=True))
    units, unit = len(sensors) * 2 + 1, 0

    def step(frac: float, label: str) -> None:
        if progress:
            progress((unit + frac) / units, label)

    for sensor in sensors:
        step(0, f"Searching {SENSOR_NAMES[sensor]} scenes")
        found = {
            p: stac.search(sensor, geometry, w, max_cloud=MAX_S2_CLOUD if sensor == "s2" else None)
            for p, w in windows.items()
        }
        if sensor == "s1":
            found = _match_orbits(found, log, selection)
        for period, items in found.items():
            key = f"{sensor}_{period}"
            items = select(items, period, selection)
            what, n = f"{SENSOR_NAMES[sensor]} {PERIOD_NAMES[period]}", len(items)
            if not items:
                inp.warnings.append(f"no {sensor.upper()} scenes in {period}-event window")
                log(f"! {sensor.upper()} {period}: no scenes")
                unit += 1
                continue
            log(f"  {sensor.upper()} {period}: compositing {len(items)} scene(s)")
            step(0, f"Reading {what} (0/{n} scenes)")
            arr = composite(
                sensor,
                items,
                grid,
                method[period],
                done=lambda k, w=what, n=n: step(k / n, f"Reading {w} ({k}/{n} scenes)"),
            )
            unit += 1
            frac = float(np.isfinite(arr).all(0).mean())
            if frac < 0.01:
                inp.warnings.append(f"{sensor.upper()} {period}-event composite has no valid pixels (clouds?)")
                continue
            inp.arrays[key] = arr
            inp.scenes[key] = [stac.describe(i) | {"valid_fraction": round(frac, 4)} for i in items]
            log(f"✓ {sensor.upper()} {period}: {len(items)} scene(s), {frac:.0%} valid")
    step(0, "Reading terrain (Copernicus DEM)")
    dem_items = stac.search("dem", geometry)
    if dem_items:
        inp.arrays["dem"] = dem(dem_items, grid)
        inp.scenes["dem"] = [stac.describe(i) for i in dem_items]
        log("✓ DEM + slope")
    return inp


def _match_orbits(found: dict[str, list], log: Log, selection: str = "closest") -> dict[str, list]:
    """SAR change is only meaningful between like geometries: keep pre scenes from the post scenes' orbit."""
    post_orbits = {i.properties.get("sat:relative_orbit") for i in select(found["post"], "post", selection)}
    same = [i for i in found["pre"] if i.properties.get("sat:relative_orbit") in post_orbits]
    if same and len(same) < len(found["pre"]):
        log(f"  S1: restricted pre-event scenes to relative orbit(s) {sorted(o for o in post_orbits if o)}")
        found = {**found, "pre": same}
    return found
