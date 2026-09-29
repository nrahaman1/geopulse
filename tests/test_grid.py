import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from geopulse.data import read_to_grid
from geopulse.grid import aoi_geometry, area_km2, bbox_geometry, make_grid, utm_crs


def test_utm_zone():
    assert utm_crs(-82.55, 35.6) == "EPSG:32617"
    assert utm_crs(151.2, -33.9) == "EPSG:32756"
    assert utm_crs(180.0, 10.0) == "EPSG:32660"


def test_grid_is_snapped_and_deterministic():
    g = bbox_geometry(-79.10, 34.58, -78.98, 34.68)
    a, b = make_grid(g), make_grid(g)
    assert a == b
    assert a.transform.c % 10 == 0 and a.transform.f % 10 == 0
    assert (a.transform.a, a.transform.e) == (10, -10)


def test_grid_covers_aoi():
    g = bbox_geometry(-79.10, 34.58, -78.98, 34.68)
    grid = make_grid(g)
    w, s, e, n = transform_bounds("EPSG:4326", grid.crs, -79.10, 34.58, -78.98, 34.68)
    gw, gs, ge, gn = grid.bounds
    assert gw <= w and gs <= s and ge >= e and gn >= n
    assert ge - e < 10 + 1e-6 or gw - w < 10  # at most one pixel of padding per edge


def test_tiles_cover_grid_without_padding():
    grid = make_grid(bbox_geometry(-79.10, 34.58, -78.98, 34.68))
    covered = np.zeros((grid.height, grid.width), bool)
    for r, c in grid.tiles(256):
        assert r + 256 <= grid.height and c + 256 <= grid.width
        covered[r : r + 256, c : c + 256] = True
    assert covered.all()


def test_area_km2():
    # 0.12° x 0.10° at 34.6°N ≈ 11.0 km x 11.1 km
    assert 115 < area_km2(bbox_geometry(-79.10, 34.58, -78.98, 34.68)) < 128


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "Point", "coordinates": [0, 0]},
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]},  # not closed
        {"type": "Polygon", "coordinates": [[[500000, 0], [500100, 0], [500100, 100], [500000, 0]]]},  # projected
    ],
)
def test_invalid_aoi_rejected(bad):
    with pytest.raises(ValueError):
        aoi_geometry(bad)


def test_feature_collection_aoi_unwrapped():
    geom = bbox_geometry(0, 0, 1, 1)
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": geom, "properties": {}}]}
    assert aoi_geometry(fc) == geom


@pytest.mark.parametrize("resampling", [Resampling.nearest, Resampling.bilinear])
def test_read_to_grid_is_pixel_exact(tmp_path, resampling):
    """Same 10 m lattice, offset by whole pixels: must come back as an exact shift (no half-pixel error)."""
    grid = make_grid(bbox_geometry(-79.10, 34.58, -79.08, 34.60))
    src = np.arange(60 * 70, dtype="float32").reshape(60, 70)
    path = tmp_path / "src.tif"
    transform = from_origin(grid.transform.c - 50, grid.transform.f + 30, 10, 10)  # 5 px left, 3 px up
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=70,
        height=60,
        count=1,
        dtype="float32",
        crs=grid.crs,
        transform=transform,
        nodata=-9999,
    ) as f:
        f.write(src, 1)
    out = read_to_grid(str(path), grid, resampling)
    np.testing.assert_array_equal(out[1:56, 1:64], src[4:59, 6:69])  # interior, away from kernel edge effects
    assert np.isnan(out[60:, :]).all() and np.isnan(out[:, 70:]).all()  # outside the source footprint -> NaN


def test_scene_selection():
    from geopulse.data import MAX_SCENES, select

    items = list(range(12))  # oldest first
    assert select(items, "pre") == items[-MAX_SCENES:] and select(items, "post") == items[:MAX_SCENES]
    spread = select(items, "post", "spread")
    assert spread[0] == 0 and spread[-1] == 11 and len(spread) == MAX_SCENES  # covers the whole window
    assert select(items[:3], "pre", "spread") == [0, 1, 2]
