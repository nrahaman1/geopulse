"""The browser engine (app/src/engine.js) must agree with the Python reference. Runs the engine under Node.js."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from rasterio.warp import transform

from geopulse import baseline, pipeline
from geopulse import model as models
from geopulse.data import select
from geopulse.grid import _starts, area_km2, bbox_geometry, make_grid
from geopulse.model import to_tensors
from geopulse.tasks import TASKS

from .conftest import AOI, synthetic_inputs

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

CROP = (slice(None), slice(40, 175), slice(10, 150))  # keeps the event and control boxes, stays small
TRIANGLE = {"type": "Polygon", "coordinates": [[[-79.10, 34.58], [-79.08, 34.583], [-79.093, 34.60], [-79.10, 34.58]]]}
AOIS = [
    bbox_geometry(11.78, 44.47, 11.92, 44.57),
    bbox_geometry(151.10, -33.95, 151.30, -33.80),
    bbox_geometry(-120.10, 38.90, -119.95, 39.05),  # straddles the UTM 10/11 boundary
    bbox_geometry(-82.60, 35.53, -82.50, 35.62),
    TRIANGLE,
]
POINTS = [
    (32632, 11.85, 44.52),
    (32756, 151.2, -33.9),
    (32610, -120.3, 39.0),
    (32611, -119.7, 39.0),
    (32633, 11.9, 44.5),
]
W1, W2 = "2024-09-01/2024-09-20", "2024-09-27/2024-10-05"
REQUESTS = [
    dict(aoi=AOI, before=W1, after=W2, task="flood", sensors=["sentinel-1", "S2"]),
    dict(aoi=AOI, before="2019-06-01/2019-08-31", after="2021-06-01/2021-08-31", task="vegetation", sensors=["s2"]),
    dict(
        aoi=AOI, before="2019-06-01/2019-08-31", after="2019-10-01/2019-12-31", task="vegetation", sensors=["s1", "s2"]
    ),
    dict(aoi=AOI, before="2019-12-20/2019-12-31", after="2021-01-01/2021-01-10", task="vegetation", sensors=["s1"]),
    dict(aoi=AOI, before=W1, after=W2, task="landslide", sensors=["s1"]),
    dict(aoi=AOI, before="2024-01-01/2024-12-31", after="2025-01-01/2025-01-02", task="flood", sensors=["s1"]),
    dict(aoi=AOI, before=W2, after=W1, task="flood", sensors=["s1"]),
    dict(aoi=bbox_geometry(-80, 34, -78, 36), before=W1, after=W2, task="flood", sensors=["s1"]),
    dict(aoi={"type": "Point", "coordinates": [0, 0]}, before=W1, after=W2, task="flood", sensors=["s1"]),
]


def enc(a):
    return [None if np.isnan(v) else float(v) for v in np.asarray(a, "float64").ravel()]


def dec(a):
    return np.array([np.nan if v is None else v for v in a], "float64")


def cases():
    out = []
    for task in TASKS:
        full = synthetic_inputs(task).arrays
        cloudy = {k: v.copy() for k, v in full.items()}
        cloudy["s2_post"][:, 60:80, 40:120] = np.nan  # an optical gap over part of the event
        for arrays in (full, cloudy, *({k: v for k, v in full.items() if k[:2] in (s, "de")} for s in ("s1", "s2"))):
            out.append((task, {k: v[CROP] for k, v in arrays.items()}))
    return out


def grid_dict(g):
    t = g.transform
    return {"epsg": int(g.crs.split(":")[1]), "res": t.a, "x0": t.c, "y0": t.f, "width": g.width, "height": g.height}


@pytest.fixture(scope="module")
def js(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    base = cases()
    payload = {
        "aois": AOIS,
        "points": POINTS,
        "requests": REQUESTS,
        "selection": [[n, p, h] for n in (2, 5, 6, 8, 11) for p in ("pre", "post") for h in ("closest", "spread")],
        "starts": [[n, 256, 192] for n in (100, 256, 300, 1151, 1506)],
        "masks": [{"grid": grid_dict(make_grid(g)), "geom": g} for g in (TRIANGLE, AOIS[3])],
        "cases": [
            {
                "task": t,
                "npx": int(np.prod(a["dem"].shape[1:])),
                "arrays": {k: [enc(b) for b in v] for k, v in a.items()},
            }
            for t, a in base
        ],
        "probe": [
            ["s2_post", [1, 4, 8, 8]],
            ["s1_pre_valid", [1, 1, 8, 8]],
            ["mask_e3", [1, 32, 1, 1]],
            ["dem", [1, 2, 64, 64]],
        ],
        "tensors": [
            {
                "case": i,
                "sensors": s,
                "npx": int(np.prod(base[i][1]["dem"].shape[1:])),
                "arrays": {k: [enc(b) for b in v] for k, v in base[i][1].items()},
            }
            for i, s in ((1, ["s1", "s2"]), (1, ["s2"]), (0, ["s1"]))  # case 1 = flood with an optical gap
        ],
    }
    (tmp / "in.json").write_text(json.dumps(payload))
    subprocess.run(
        [
            NODE,
            str(ROOT / "tests" / "js" / "parity.mjs"),
            str(ROOT / "app" / "src" / "engine.js"),
            str(tmp / "in.json"),
            str(tmp / "out.json"),
        ],
        check=True,
        timeout=120,
    )
    return payload, base, json.loads((tmp / "out.json").read_text())


def test_task_registry_matches(js):
    assert js[2]["tasks"] == json.loads(json.dumps(TASKS))


def test_grids_and_areas(js):
    for aoi, grid, km2 in zip(AOIS, js[2]["grids"], js[2]["areas"], strict=True):
        assert grid == grid_dict(make_grid(aoi))
        assert km2 == pytest.approx(area_km2(aoi), rel=1e-9)


def test_utm_matches_proj(js):
    for (epsg, lon, lat), xy, back in zip(POINTS, js[2]["utm"], js[2]["utm_inverse"], strict=True):
        x, y = transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
        assert xy == pytest.approx([x[0], y[0]], abs=1e-3)  # millimetres
        assert back == pytest.approx([lon, lat], abs=1e-9)


def test_requests_match(js):
    for req, got in zip(REQUESTS, js[2]["requests"], strict=True):
        try:
            want = pipeline.make_request(req["aoi"], req["before"], req["after"], req["task"], req["sensors"])
        except pipeline.RequestError:
            assert "error" in got, req
            continue
        assert "error" not in got, (req, got)
        for k in ("task", "before", "after", "sensors", "aoi_km2"):
            assert got[k] == pytest.approx(want[k]) if k == "aoi_km2" else got[k] == want[k]
        assert bool(got["warnings"]) == bool(want["warnings"])


def test_selection_and_tiling(js):
    payload, _, out = js
    for (n, period, how), got in zip(payload["selection"], out["selection"], strict=True):
        assert got == select(list(range(n)), period, how)
    for (n, size, stride), got in zip(payload["starts"], out["starts"], strict=True):
        assert got == _starts(n, size, stride)


def test_aoi_masks_match_rasterio(js):
    for g, got in zip((TRIANGLE, AOIS[3]), js[2]["masks"], strict=True):
        want = make_grid(g).mask(g).ravel()
        assert (np.array(got, bool) == want).mean() > 0.999


@pytest.mark.parametrize("key", ["target", "change", "uncertainty"])
def test_baseline_matches(js, key):
    _, base, out = js
    for (task, arrays), got in zip(base, out["baseline"], strict=True):
        want = baseline.predict(arrays, task)[key].ravel()
        np.testing.assert_allclose(dec(got[key]), want, atol=1e-4, err_msg=f"{task} {key} {sorted(arrays)}")


def test_severity_matches(js):
    _, base, out = js
    for (task, arrays), got in zip(base, out["baseline"], strict=True):
        want = baseline.predict(arrays, task).get("severity")
        assert (got["severity"] is None) == (want is None)
        if want is not None:
            assert got["severity"] == want.ravel().tolist()


def test_model_inputs_match(js):
    payload, base, out = js
    for spec, got in zip(payload["tensors"], out["tensors"], strict=True):
        want = to_tensors(base[spec["case"]][1], tuple(spec["sensors"]))
        for k, v in want.items():
            np.testing.assert_allclose(dec(got[k]), v.ravel(), atol=1e-5, err_msg=k)


def test_onnx_self_check_input_matches(js):
    payload, _, out = js
    for name, dims in payload["probe"]:
        np.testing.assert_array_equal(
            np.array(out["probe"][name], "float32"), models.probe_values(name, tuple(dims)).ravel()
        )
