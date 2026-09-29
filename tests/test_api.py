import json
import time

import pytest
from fastapi.testclient import TestClient

from geopulse import api, pipeline

from .conftest import AOI

JOB = {"aoi": AOI, "before": ["2024-09-01", "2024-09-20"], "after": ["2024-09-27", "2024-10-05"]}


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/models").json()[0]["model_id"] == "threshold-baseline"
    assert client.get("/models/nope").status_code == 404


@pytest.mark.parametrize(
    "patch",
    [
        {"aoi": {"type": "Point", "coordinates": [0, 0]}},
        {"aoi": {"type": "Polygon", "coordinates": [[[-80, 34], [-78, 34], [-78, 36], [-80, 36], [-80, 34]]]}},
        {"before": ["2024-10-01", "2024-10-05"]},  # overlaps after window
        {"task": "landslide"},
    ],
)
def test_invalid_requests_rejected(client, patch):
    assert client.post("/jobs", json=JOB | patch).status_code == 422


def test_job_lifecycle(client, monkeypatch):
    def fake_run(request, out_dir, log, inputs=None):
        log("✓ fake")
        (out_dir / "layers").mkdir(parents=True, exist_ok=True)
        layers = {"bounds": [0, 0, 1, 1], "layers": ["target"], "target": "flood", "extent": "flood_extent.geojson"}
        (out_dir / "layers.json").write_text(json.dumps(layers))
        (out_dir / "layers" / "target.png").write_bytes(b"png")
        return {"affected_km2": 1.0}

    monkeypatch.setattr(pipeline, "run", fake_run)
    job = client.post("/jobs", json=JOB).json()
    for _ in range(100):
        state = client.get(f"/jobs/{job['id']}").json()
        if state["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert state["status"] == "succeeded" and state["log"] == ["✓ fake"]
    res = client.get(f"/jobs/{job['id']}/results").json()
    assert res["summary"] == {"affected_km2": 1.0}
    assert client.get(res["files"]["layers/target.png"]).content == b"png"
    assert client.get(f"/jobs/{job['id']}/files/../../etc/passwd").status_code == 404
    assert client.get(f"/jobs/{job['id']}/files/job.json").status_code == 404
    assert client.get("/jobs/zzz").status_code == 404
    assert any(j["id"] == job["id"] for j in client.get("/jobs").json())
    assert 'geopulse_jobs{status="succeeded"}' in client.get("/metrics").text


def test_failed_job_reports_error(client, monkeypatch):
    def boom(*a, **k):
        raise pipeline.RequestError("no usable post-event observations")

    monkeypatch.setattr(pipeline, "run", boom)
    r = client.post("/predict/sync", json=JOB)
    assert r.status_code == 422 and "post-event" in r.json()["detail"]


def test_web_ui_and_examples(client):
    page = client.get("/")  # the built web app (app/dist), or a pointer to how to build it
    assert page.status_code == 200 and "GeoPulse" in page.text
    ex = client.get("/examples").json()
    assert ex and {"id", "aoi", "before", "after"} <= set(ex[0])


def test_desktop_engine_requires_its_token(client, monkeypatch):
    monkeypatch.setattr(api, "TOKEN", "s3cret")
    assert client.get("/health").status_code == 200  # the app polls this before it has connected
    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/models?token=s3cret").status_code == 200  # file links (downloads, map layers)


def test_v01_flood_jobs_still_render(client):
    """Jobs written before multi-task support are served in the task-generic schema."""
    job = {
        "id": "0123456789ab",
        "status": "succeeded",
        "created": "2026-09-28T00:00:00+00:00",
        "log": [],
        "request": {"task": "flood", "aoi_km2": 1.0},
        "summary": {"flooded_km2": 2.0, "mean_confidence_flooded": 0.9, "review_km2": 0.1},
    }
    d = api.jobs_dir() / job["id"]
    (d / "layers").mkdir(parents=True)
    (d / "layers" / "flood.png").write_bytes(b"png")
    (d / "layers.json").write_text(json.dumps({"bounds": [0, 0, 1, 1], "layers": ["flood", "uncertainty"]}))
    api.JOBS[job["id"]] = job
    res = client.get(f"/jobs/{job['id']}/results").json()
    assert res["layers"]["target"] == "flood" and res["layers"]["layers"] == ["target", "uncertainty"]
    assert res["summary"]["affected_km2"] == 2.0 and res["files"]["layers/target.png"].endswith("flood.png")


def test_public_mode_hides_other_visitors_jobs(client, monkeypatch):
    monkeypatch.setattr(api, "PUBLIC", True)
    assert client.get("/jobs").status_code == 404
