"""`geopulse` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_aoi(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_search(a):
    from . import stac
    from .grid import aoi_geometry

    geometry = aoi_geometry(_load_aoi(a.aoi))
    for sensor in a.sensor or ["s1", "s2"]:
        items = stac.search(sensor, geometry, f"{a.start}/{a.end}", a.max_cloud)
        print(f"✓ {sensor.upper()} scenes found: {len(items)}")
        for i in items:
            d = stac.describe(i)
            cloud = d["cloud_cover"]
            extra = f"cloud {cloud:.0f}%" if cloud is not None else f"{d['orbit']} orbit {d['relative_orbit']}"
            print(f"    {d['datetime'][:19]}  {d['id']}  {extra}")


def cmd_infer(a):
    import yaml

    from . import pipeline

    if a.config:
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        aoi = _load_aoi(str(Path(a.config).parent / cfg.get("aoi_file", "aoi.geojson")))
        before, after = cfg["before"], cfg["after"]
        task, sensors = cfg.get("task", "flood"), cfg.get("sensors", ["s1", "s2"])
        output = a.output or f"outputs/{Path(a.config).parent.name}"
    else:
        if not (a.aoi and a.before and a.after):
            sys.exit("infer needs --config, or --aoi, --before and --after")
        aoi, before, after, task, sensors = _load_aoi(a.aoi), a.before, a.after, a.task, a.sensors
        output = a.output or "outputs/infer"
    request = pipeline.make_request(aoi, before, after, task, sensors, a.model)
    summary = pipeline.run(request, output)
    keys = ("task", "affected_label", "affected_km2", "review_km2", "observed_km2", "severity_km2", "warnings")
    print(json.dumps({k: summary[k] for k in keys if k in summary}, indent=2))


def cmd_dataset(a):
    from .bench import build

    build(a.manifest, a.out)


def cmd_dataset_pull(a):
    from .hub import pull_dataset

    pull_dataset(a.name, a.repo, a.out)


def cmd_dataset_push(a):
    from .hub import push_dataset

    push_dataset(a.repo, a.root, a.private)


def cmd_train(a):
    from .train import train

    train(a.config)


def cmd_evaluate(a):
    from .train import evaluate

    evaluate(a.model, a.dataset, a.split)


def cmd_models(a):
    from .model import list_models

    for card in list_models():
        val = card.get("metrics", {}).get("val", {})
        ious = "  ".join(f"{t} {m['s1+s2']['iou']}" for t, m in val.items() if "s1+s2" in m)
        print(
            f"{card['model_id']:30s} v{card['version']:7s} tasks {','.join(card['tasks']):26s}"
            + (f" val IoU  {ious}" if ious else "")
        )


def cmd_models_pull(a):
    from .hub import pull_models

    try:
        pulled = pull_models(a.repo, a.revision)
    except Exception as e:  # network, missing repo, auth: one readable line instead of a traceback
        sys.exit(f"error: could not pull models from {a.repo}: {type(e).__name__}: {str(e).splitlines()[0]}")
    if not pulled:
        sys.exit(f"error: no checkpoints found in {a.repo}")


def cmd_models_export(a):
    from .model import export_onnx, list_models

    cards = [c for c in list_models()[1:] if a.model in ("all", c["model_id"])]
    if not cards:
        sys.exit(f"error: no trained model matches {a.model!r}")
    for card in cards:
        card = export_onnx(card)
        print(f"✓ {card['onnx']['file']}  sha256 {card['onnx']['sha256'][:12]}")


def cmd_models_push(a):
    from .hub import push_models

    push_models(a.repo, a.private)


def cmd_serve(a):
    import uvicorn

    print(f"GeoPulse at http://{a.host}:{a.port}  (API docs: /docs)")
    uvicorn.run("geopulse.api:app", host=a.host, port=a.port)


def cmd_benchmark(a):
    from .model import benchmark

    print(json.dumps(benchmark(a.tile_size, a.batch_size, a.precision == "fp16"), indent=2))


def cmd_doctor(a):
    import platform

    checks = [("Python", lambda: platform.python_version())]

    def gdal():
        import rasterio

        return rasterio.__gdal_version__

    def proj():
        from rasterio.warp import transform

        x, _ = transform("EPSG:4326", "EPSG:32617", [-82.55], [35.6])
        assert abs(x[0] - 359_000) < 5_000
        return "ok"

    def torch_():
        import torch

        return torch.__version__

    def cuda():
        import torch

        return torch.cuda.get_device_name() if torch.cuda.is_available() else "not available (CPU inference)"

    def stac_():
        from pystac_client import Client

        from .stac import PROVIDER_URL

        Client.open(PROVIDER_URL, timeout=15)
        return PROVIDER_URL.split("/")[2]

    def api():
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
            return json.loads(r.read())["status"]

    def models_():
        from .model import list_models

        return f"{len(list_models())} registered"

    checks += [
        ("GDAL", gdal),
        ("PROJ", proj),
        ("PyTorch", torch_),
        ("CUDA", cuda),
        ("STAC connectivity", stac_),
        ("API (localhost:8000)", api),
        ("Models", models_),
    ]
    for name, fn in checks:
        try:
            print(f"{name:22s} OK   {fn()}")
        except Exception as e:
            print(f"{name:22s} --   {type(e).__name__}: {e}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="geopulse", description="Multimodal Earth-change intelligence")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="find scenes via STAC")
    s.add_argument("--aoi", required=True)
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--sensor", action="append", choices=["s1", "s2", "dem"])
    s.add_argument("--max-cloud", type=float)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("infer", help="map change for an AOI")
    s.add_argument("--config", help="request.yaml (aoi_file, before, after, task, sensors)")
    s.add_argument("--aoi")
    s.add_argument("--before", help="YYYY-MM-DD/YYYY-MM-DD")
    s.add_argument("--after", help="YYYY-MM-DD/YYYY-MM-DD")
    s.add_argument("--task", default="flood", choices=["flood", "wildfire", "vegetation"])
    s.add_argument("--sensors", default="s1,s2")
    s.add_argument("--model", default="auto")
    s.add_argument("--output")
    s.set_defaults(fn=cmd_infer)

    s = sub.add_parser("dataset", help="benchmark datasets")
    ds = s.add_subparsers(dest="action", required=True)
    b = ds.add_parser("build", help="build tiles from an event manifest")
    b.add_argument("manifest")
    b.add_argument("--out", default="data/bench")
    b.set_defaults(fn=cmd_dataset)
    b = ds.add_parser("pull", help="download prebuilt benchmark tiles from Hugging Face")
    b.add_argument("name", nargs="?", default="all", help="e.g. geopulse-bench-wildfire (default: all)")
    b.add_argument("--repo", default=None)
    b.add_argument("--out", default="data/bench")
    b.set_defaults(fn=cmd_dataset_pull)
    b = ds.add_parser("push", help="publish benchmark tiles to Hugging Face (maintainers)")
    b.add_argument("--repo", default=None)
    b.add_argument("--root", default="data/bench")
    b.add_argument("--private", action="store_true")
    b.set_defaults(fn=cmd_dataset_push)

    s = sub.add_parser("train", help="train from an experiment config")
    s.add_argument("config")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="robustness matrix of a model on a benchmark split")
    s.add_argument("--model", "--checkpoint", required=True, help="model id, checkpoint file, or threshold-baseline")
    s.add_argument("--dataset", default="data/bench/geopulse-bench-flood")
    s.add_argument("--split", default="test")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("models", help="list, pull or publish trained models")
    s.set_defaults(fn=cmd_models)
    ms = s.add_subparsers(dest="action")
    ms.add_parser("list", help="list registered models (default)").set_defaults(fn=cmd_models)
    m = ms.add_parser("pull", help="download trained checkpoints from Hugging Face")
    m.add_argument("--repo", default=None)
    m.add_argument("--revision", default=None)
    m.set_defaults(fn=cmd_models_pull)
    m = ms.add_parser("export-onnx", help="export checkpoints to ONNX for the in-browser engine")
    m.add_argument("--model", default="all")
    m.set_defaults(fn=cmd_models_export)
    m = ms.add_parser("push", help="publish trained checkpoints to Hugging Face (maintainers)")
    m.add_argument("--repo", default=None)
    m.add_argument("--private", action="store_true")
    m.set_defaults(fn=cmd_models_push)

    s = sub.add_parser("serve", help="run the API + web map")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("benchmark", help="model latency / throughput / memory")
    s.add_argument("--tile-size", type=int, default=256)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    s.set_defaults(fn=cmd_benchmark)

    sub.add_parser("doctor", help="check the local environment").set_defaults(fn=cmd_doctor)

    a = p.parse_args(argv)
    if getattr(a, "repo", "unset") is None:  # hub commands: default repos live in geopulse.hub
        from . import hub

        a.repo = hub.MODEL_REPO if a.cmd == "models" else hub.DATA_REPO
    sys.stdout.reconfigure(encoding="utf-8")  # progress lines use ✓ even on legacy Windows consoles
    try:
        a.fn(a)
    except ValueError as e:  # includes RequestError: bad input is a user error, not a traceback
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
