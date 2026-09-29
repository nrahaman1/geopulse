"""Offline integration: synthetic inputs -> model -> COG / GeoJSON / summary / provenance / STAC item / web layers."""

import json

import numpy as np
import pytest
import rasterio

from geopulse import pipeline

from .conftest import AOI, synthetic_inputs

EVENT_KM2 = 60 * 80 * 1e-4  # the synthetic event box, 10 m pixels


def request(task="flood", before="2024-09-01/2024-09-20", after="2024-09-27/2024-10-05", **kw):
    return pipeline.make_request(AOI, before, after, task, **kw)


@pytest.mark.parametrize(("task", "target"), [("flood", "flood"), ("wildfire", "burn"), ("vegetation", "disturbance")])
def test_end_to_end_outputs(tmp_path, task, target):
    inp = synthetic_inputs(task)
    summary = pipeline.run(request(task), tmp_path, log=lambda m: None, inputs=inp)
    for name in (f"{target}_probability.tif", "change_probability.tif", "uncertainty.tif"):
        with rasterio.open(tmp_path / name) as src:
            assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
            assert src.crs.to_string() == inp.grid.crs and src.transform == inp.grid.transform
    assert summary["target"] == target
    assert summary["affected_km2"] == pytest.approx(EVENT_KM2, rel=0.1)
    fc = json.loads((tmp_path / f"{target}_extent.geojson").read_text())
    assert fc["features"] and fc["features"][0]["properties"]["area_km2"] == pytest.approx(EVENT_KM2, rel=0.1)
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["target_crs"] == inp.grid.crs and prov["sensors"]["s1_post"] == ["synthetic-s1_post"]
    item = json.loads((tmp_path / "item.json").read_text())
    assert {"classification", "uncertainty", "provenance"} <= set(item["assets"])
    layers = json.loads((tmp_path / "layers.json").read_text())
    assert layers["target"] == target and layers["extent"] == f"{target}_extent.geojson"
    assert all((tmp_path / "layers" / f"{k}.png").exists() for k in layers["layers"])


def test_wildfire_severity_output(tmp_path):
    summary = pipeline.run(request("wildfire"), tmp_path, log=lambda m: None, inputs=synthetic_inputs("wildfire"))
    assert summary["severity_km2"]["high"] == pytest.approx(EVENT_KM2, rel=0.1)
    with rasterio.open(tmp_path / "burn_severity.tif") as src:
        sev = src.read(1)
        assert src.nodata == 255 and set(np.unique(sev)) <= {0, 1, 2, 3, 4, 255}
    assert "severity" in json.loads((tmp_path / "layers.json").read_text())["layers"]


def test_missing_pre_event_is_warned_for_flood(tmp_path):
    inp = synthetic_inputs()
    for k in ("s1_pre", "s2_pre"):
        inp.arrays.pop(k)
    summary = pipeline.run(request(), tmp_path, log=lambda m: None, inputs=inp)
    assert any("permanent water" in w for w in summary["warnings"])


@pytest.mark.parametrize("task", ["wildfire", "vegetation"])
def test_missing_pre_event_is_an_error_for_change_tasks(tmp_path, task):
    inp = synthetic_inputs(task)
    for k in ("s1_pre", "s2_pre"):
        inp.arrays.pop(k)
    with pytest.raises(pipeline.RequestError):
        pipeline.run(request(task), tmp_path, log=lambda m: None, inputs=inp)


def test_no_post_event_data_is_an_error(tmp_path):
    inp = synthetic_inputs()
    for k in ("s1_post", "s2_post"):
        inp.arrays.pop(k)
    with pytest.raises(pipeline.RequestError):
        pipeline.run(request(), tmp_path, log=lambda m: None, inputs=inp)


def test_vegetation_warns_on_season_mismatch():
    same = request("vegetation", "2019-06-01/2019-08-31", "2021-06-01/2021-08-31")
    shifted = request("vegetation", "2019-06-01/2019-08-31", "2019-10-01/2019-12-31")
    assert same["warnings"] == [] and "seasonal cycle" in shifted["warnings"][0]
    assert request("wildfire", "2019-06-01/2019-08-31", "2019-10-01/2019-12-31")["warnings"] == []


def test_season_gap_wraps_the_year():
    d = pipeline.dt.date
    assert pipeline.season_gap((d(2019, 12, 20), d(2019, 12, 31)), (d(2021, 1, 1), d(2021, 1, 10))) < 20


@pytest.mark.parametrize("kw", [{"task": "landslide"}, {"sensors": ["landsat"]}, {"sensors": []}])
def test_request_validation(kw):
    with pytest.raises(pipeline.RequestError):
        request(**kw)


def test_windows_must_be_ordered():
    with pytest.raises(pipeline.RequestError):
        pipeline.make_request(AOI, "2024-09-27/2024-10-05", "2024-09-01/2024-09-20")
    with pytest.raises(pipeline.RequestError):
        pipeline.make_request(AOI, "2024-01-01/2024-12-31", "2025-01-01/2025-01-02")  # window too long


def test_sensor_aliases():
    assert request(sensors=["sentinel-1", "S2"])["sensors"] == ["s1", "s2"]


@pytest.mark.parametrize("task", ["wildfire", "vegetation"])
def test_open_water_is_never_burned_or_disturbed(tmp_path, task):
    """A model that fires everywhere must still be overruled on water that existed before the event."""
    import torch

    from geopulse import model as models

    from .conftest import CONTROL, SURFACES

    net = models.GPFT(channels=8, tasks=(task,))
    with torch.no_grad():
        net.head.bias[:] = torch.tensor([9.0, 0.0])  # "everything changed"
    models.save(
        net,
        models.models_dir() / "fires-everywhere.pt",
        {"model_id": "fires-everywhere", "version": "0", "tasks": [task]},
    )
    inp = synthetic_inputs(task)
    spectrum, backscatter = SURFACES["water"]
    for p in ("pre", "post"):
        inp.arrays[f"s2_{p}"][(slice(None), *CONTROL)] = np.array(spectrum)[:, None, None]
        inp.arrays[f"s1_{p}"][(slice(None), *CONTROL)] = np.array(backscatter)[:, None, None]
    pipeline.run(request(task, model="fires-everywhere"), tmp_path, log=lambda m: None, inputs=inp)
    target = pipeline.TASKS[task]["target"]
    with rasterio.open(tmp_path / f"{target}_probability.tif") as src:
        prob = src.read(1)
    assert (prob[CONTROL] == 0).all() and np.nanmean(prob[:40, :40]) > 0.9


def test_progress_is_monotonic_and_reaches_the_output_stage(tmp_path):
    calls = []
    pipeline.run(
        request(), tmp_path, log=lambda m: None, inputs=synthetic_inputs(), progress=lambda f, s: calls.append((f, s))
    )
    fracs = [f for f, _ in calls]
    assert fracs == sorted(fracs) and fracs[0] == 0.0
    assert calls[-1] == (pipeline.PROGRESS["model"], "Writing maps and downloads")


def test_acquisition_progress_advances_scene_by_scene(monkeypatch):
    """Each composite (sensor x period) and the DEM get an equal share, advanced as scenes load."""
    from types import SimpleNamespace

    from geopulse import data

    items = [SimpleNamespace(id=f"i{k}", properties={"sat:relative_orbit": 1}) for k in range(3)]
    monkeypatch.setattr(data.stac, "search", lambda sensor, *a, **kw: items)
    monkeypatch.setattr(data.stac, "describe", lambda item: {"id": item.id})

    def fake_composite(sensor, items, grid, method="median", done=None):
        for k in range(1, len(items) + 1):
            done(k)
        return np.ones((2, 4, 4), "float32")

    monkeypatch.setattr(data, "composite", fake_composite)
    monkeypatch.setattr(data, "dem", lambda items, grid: np.ones((2, 4, 4), "float32"))
    calls = []
    data.prepare({}, None, "a", "b", ("s1", "s2"), log=lambda m: None, progress=lambda f, s: calls.append((f, s)))
    fracs = [f for f, _ in calls]
    assert fracs == sorted(fracs) and fracs[-1] == 4 / 5  # 4 composites done, the DEM is last
    assert ("Reading Sentinel-2 after (3/3 scenes)") in [s for _, s in calls]
    assert calls[-1][1] == "Reading terrain (Copernicus DEM)"
