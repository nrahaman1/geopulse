"""FastAPI service. Jobs are validated up front, run on a background worker, and persisted as files."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import __version__, pipeline, stac
from . import model as models
from .grid import aoi_geometry

ROOT = Path(__file__).resolve().parent
EXAMPLES = next(
    (p for p in (ROOT / "examples", ROOT.parent / "examples") if p.is_dir()), ROOT / "examples"
)  # wheel, repo
MAX_JOB_KM2 = float(os.environ.get("GEOPULSE_MAX_JOB_KM2", 500))
MAX_SYNC_KM2 = 25.0
PUBLIC = os.environ.get("GEOPULSE_PUBLIC") == "1"  # shared demo: never list other visitors' jobs
MAX_PENDING = 20


def jobs_dir() -> Path:
    return Path(os.environ.get("GEOPULSE_OUTPUTS", "outputs")) / "jobs"


class JobRequest(BaseModel):
    task: str = "flood"
    aoi: dict
    before: list[str] | str
    after: list[str] | str
    sensors: list[str] = ["sentinel-1", "sentinel-2"]
    model: str = "auto"


class SearchRequest(BaseModel):
    aoi: dict
    datetime: str
    sensor: str = "s1"
    max_cloud: float | None = None


app = FastAPI(title="GeoPulse", version=__version__, description="Multimodal Earth-change intelligence API")
# ponytail: one in-process worker thread; move to Redis + RQ/Celery when jobs must outlive the server process.
worker = ThreadPoolExecutor(1)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
STATS = {"requests": {}, "request_seconds": 0.0, "job_seconds": 0.0, "jobs_finished": 0}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _save(job: dict) -> None:
    d = jobs_dir() / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    with LOCK:
        (d / "job.json").write_text(json.dumps(job, indent=2))


def _load_jobs() -> None:
    for p in jobs_dir().glob("*/job.json"):
        job = json.loads(p.read_text(encoding="utf-8"))
        JOBS[job["id"]] = job
        if job["status"] in ("queued", "running"):
            job.update(status="failed", error="server restarted before the job finished")
            _save(job)


_load_jobs()


def _validate(req: JobRequest, max_km2: float) -> dict:
    try:
        return pipeline.make_request(req.aoi, req.before, req.after, req.task, req.sensors, req.model, max_km2=max_km2)
    except pipeline.RequestError as e:
        raise HTTPException(422, str(e)) from None


def _run(job_id: str) -> None:
    job = JOBS[job_id]
    job.update(status="running", started=_now())
    _save(job)
    t0 = time.time()

    def log(msg: str) -> None:
        job["log"].append(msg)
        _save(job)

    try:
        job["summary"] = pipeline.run(job["request"], jobs_dir() / job_id, log=log)
        job["status"] = "succeeded"
    except Exception as e:  # the job record is the error channel for async work
        job.update(status="failed", error=f"{type(e).__name__}: {e}")
    job["finished"] = _now()
    STATS["job_seconds"] += time.time() - t0
    STATS["jobs_finished"] += 1
    _save(job)


def _job(job_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id) or job_id not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[job_id]


@app.middleware("http")
async def count_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    key = (request.method, getattr(route, "path", "unmatched"), response.status_code)
    STATS["requests"][key] = STATS["requests"].get(key, 0) + 1
    STATS["request_seconds"] += time.perf_counter() - t0
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-cache"})  # revalidate on upgrade


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "device": str(models.device())}


@app.get("/models")
def list_models():
    return models.list_models()


@app.get("/models/{model_id}")
def get_model(model_id: str):
    try:
        return models.resolve(model_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None


@app.get("/examples")
def examples():
    out = []
    for p in sorted(EXAMPLES.glob("*/request.yaml")):
        req = yaml.safe_load(p.read_text(encoding="utf-8"))
        req["aoi"] = json.loads((p.parent / req.pop("aoi_file", "aoi.geojson")).read_text(encoding="utf-8"))
        out.append({"id": p.parent.name, **req})
    return out


@app.post("/search")
def search(req: SearchRequest):
    try:
        sensor = pipeline.SENSOR_ALIASES.get(req.sensor.lower(), req.sensor)
        items = stac.search(sensor, aoi_geometry(req.aoi), req.datetime, req.max_cloud)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(422, str(e)) from None
    return [stac.describe(i) for i in items]


@app.post("/jobs", status_code=202)
def create_job(req: JobRequest):
    request = _validate(req, MAX_JOB_KM2)
    if sum(j["status"] in ("queued", "running") for j in JOBS.values()) >= MAX_PENDING:
        raise HTTPException(429, "too many pending jobs")
    job = {"id": uuid.uuid4().hex[:12], "status": "queued", "created": _now(), "request": request, "log": []}
    JOBS[job["id"]] = job
    _save(job)
    worker.submit(_run, job["id"])
    return job


@app.get("/jobs")
def list_jobs():
    if PUBLIC:
        raise HTTPException(404, "job listing is disabled on public deployments")
    return sorted(
        (
            {k: j.get(k) for k in ("id", "status", "created", "finished", "summary")}
            | {"task": j["request"]["task"], "aoi_km2": j["request"].get("aoi_km2")}
            for j in JOBS.values()
        ),
        key=lambda j: j["created"],
        reverse=True,
    )


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _job(job_id)


@app.get("/jobs/{job_id}/results")
def job_results(job_id: str):
    job = _job(job_id)
    if job["status"] != "succeeded":
        raise HTTPException(409, f"job is {job['status']}")
    d = jobs_dir() / job_id
    files = sorted(
        str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*") if p.is_file() and p.name != "job.json"
    )
    files = {f: f"/jobs/{job_id}/files/{f}" for f in files}
    summary, layers = job["summary"], json.loads((d / "layers.json").read_text(encoding="utf-8"))
    if "target" not in layers:  # written by v0.1 (flood only): present it in the task-generic schema
        layers |= {
            "target": "flood",
            "extent": "flood_extent.geojson",
            "layers": ["target" if k == "flood" else k for k in layers["layers"]],
        }
        summary = summary | {
            "target": "flood",
            "affected_label": "flooded",
            "affected_km2": summary.get("flooded_km2"),
            "mean_confidence_affected": summary.get("mean_confidence_flooded"),
        }
        files["layers/target.png"] = files["layers/flood.png"]
    return {"summary": summary, "layers": layers, "files": files}


@app.get("/jobs/{job_id}/files/{path:path}")
def job_file(job_id: str, path: str):
    _job(job_id)
    base = (jobs_dir() / job_id).resolve()
    f = (base / path).resolve()
    if not f.is_relative_to(base) or not f.is_file() or f.name == "job.json":
        raise HTTPException(404, "file not found")
    return FileResponse(f)


@app.post("/predict/sync")
def predict_sync(req: JobRequest):
    """Small AOIs only; blocks until done and returns the summary."""
    request = _validate(req, MAX_SYNC_KM2)
    job = {"id": uuid.uuid4().hex[:12], "status": "queued", "created": _now(), "request": request, "log": []}
    JOBS[job["id"]] = job
    _run(job["id"])
    if job["status"] != "succeeded":
        raise HTTPException(422, job["error"])
    return job


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = ["# TYPE geopulse_requests_total counter"]
    for (method, path, code), n in sorted(STATS["requests"].items()):
        lines.append(f'geopulse_requests_total{{method="{method}",path="{path}",code="{code}"}} {n}')
    lines += [
        "# TYPE geopulse_request_seconds_total counter",
        f"geopulse_request_seconds_total {STATS['request_seconds']:.3f}",
        "# TYPE geopulse_job_seconds_total counter",
        f"geopulse_job_seconds_total {STATS['job_seconds']:.3f}",
        "# TYPE geopulse_jobs gauge",
    ]
    for status in ("queued", "running", "succeeded", "failed"):
        lines.append(f'geopulse_jobs{{status="{status}"}} {sum(j["status"] == status for j in JOBS.values())}')
    return "\n".join(lines) + "\n"
