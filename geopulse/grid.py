"""Deterministic AOI grids: UTM CRS, snapped origin, fixed resolution, tile layout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from rasterio.features import bounds as geom_bounds
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import transform_bounds, transform_geom


def aoi_geometry(obj: dict) -> dict:
    """Accept a GeoJSON geometry, Feature or FeatureCollection; return one Polygon/MultiPolygon geometry."""
    if obj.get("type") == "FeatureCollection":
        feats = obj.get("features") or []
        if len(feats) != 1:
            raise ValueError("FeatureCollection AOI must contain exactly one feature")
        obj = feats[0]
    if obj.get("type") == "Feature":
        obj = obj.get("geometry") or {}
    if obj.get("type") not in ("Polygon", "MultiPolygon"):
        raise ValueError("AOI must be a Polygon or MultiPolygon")
    polys = obj["coordinates"] if obj["type"] == "MultiPolygon" else [obj["coordinates"]]
    for poly in polys:
        for ring in poly:
            if len(ring) < 4 or ring[0] != ring[-1]:
                raise ValueError("AOI rings must be closed with at least 4 positions")
            for x, y, *_ in ring:
                if not (-180 <= x <= 180 and -90 <= y <= 90):
                    raise ValueError("AOI coordinates must be lon/lat (EPSG:4326)")
    return obj


def bbox_geometry(west: float, south: float, east: float, north: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }


def utm_crs(lon: float, lat: float) -> str:
    zone = min(int((lon + 180) // 6) + 1, 60)
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    width: int
    height: int

    @property
    def res(self) -> float:
        return self.transform.a

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        t = self.transform
        return (t.c, t.f - self.height * self.res, t.c + self.width * self.res, t.f)

    @property
    def area_km2(self) -> float:
        return self.width * self.height * self.res**2 / 1e6

    def tiles(self, size: int, stride: int | None = None):
        """Yield (row, col) upper-left offsets covering the grid; the last tile is shifted inward, not padded."""
        stride = stride or size
        rows = _starts(self.height, size, stride)
        cols = _starts(self.width, size, stride)
        for r in rows:
            for c in cols:
                yield r, c

    def mask(self, geometry: dict) -> np.ndarray:
        """True for pixels whose centre falls inside the (EPSG:4326) geometry."""
        g = transform_geom("EPSG:4326", self.crs, geometry)
        return geometry_mask([g], (self.height, self.width), self.transform, invert=True)

    def to_dict(self) -> dict:
        return {"crs": self.crs, "transform": list(self.transform)[:6], "width": self.width, "height": self.height}


def _starts(n: int, size: int, stride: int) -> list[int]:
    if n <= size:
        return [0]
    starts = list(range(0, n - size, stride))
    return starts + [n - size]


def area_km2(geometry: dict) -> float:
    """Planar area of the AOI in its local UTM zone."""
    w, s, e, n = geom_bounds(geometry)
    g = transform_geom("EPSG:4326", utm_crs((w + e) / 2, (s + n) / 2), geometry)
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    total = 0.0
    for poly in polys:
        for k, ring in enumerate(poly):
            a = 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0, *_), (x1, y1, *_) in pairwise(ring)))
            total += a if k == 0 else -a
    return total / 1e6


def make_grid(geometry: dict, res: float = 10.0) -> Grid:
    """Grid covering the AOI in its UTM zone, with edges snapped to multiples of `res`.

    Snapping makes the grid deterministic: the same AOI always yields the same pixel edges, and two AOIs
    in the same zone share pixel boundaries, so cached rasters and labels line up without resampling.
    """
    w, s, e, n = geom_bounds(geometry)
    crs = utm_crs((w + e) / 2, (s + n) / 2)
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, w, s, e, n, densify_pts=21)
    minx, miny = math.floor(minx / res) * res, math.floor(miny / res) * res
    maxx, maxy = math.ceil(maxx / res) * res, math.ceil(maxy / res) * res
    return Grid(
        crs=crs,
        transform=Affine(res, 0.0, minx, 0.0, -res, maxy),
        width=round((maxx - minx) / res),
        height=round((maxy - miny) / res),
    )
