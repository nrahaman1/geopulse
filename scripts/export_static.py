"""Build the static online showcase: the web map plus precomputed results for every example event.

    python scripts/export_static.py site/            # runs the pipeline for each example (cached inputs are reused)
    hf upload nafizrahaman/geopulse site/ --repo-type space

The page detects static/site.json and switches to showcase mode (no API). Heavy GeoTIFFs are left out; the
map layers, polygons, summaries, provenance and STAC items are kept.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

from geopulse import __version__, pipeline
from geopulse import model as models
from geopulse.api import EXAMPLES, ROOT

CODESPACES = "https://codespaces.new/nrahaman1/geopulse?quickstart=1"
SPACE_README = Path(__file__).resolve().parent.parent / "deploy" / "huggingface-static" / "README.md"


def main(out: Path) -> None:
    static = out / "static"
    if out.exists():
        shutil.rmtree(out)
    (static / "results").mkdir(parents=True)
    shutil.copy2(ROOT / "web" / "index.html", out / "index.html")
    if SPACE_README.exists():
        shutil.copy2(SPACE_README, out / "README.md")

    examples, used = [], set()
    for p in sorted(EXAMPLES.glob("*/request.yaml")):
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        aoi = json.loads((p.parent / cfg.pop("aoi_file", "aoi.geojson")).read_text(encoding="utf-8"))
        examples.append({"id": p.parent.name, **cfg, "aoi": aoi})
        request = pipeline.make_request(aoi, cfg["before"], cfg["after"], cfg.get("task", "flood"), cfg.get("sensors"))
        d = static / "results" / p.parent.name
        log: list[str] = []
        summary = pipeline.run(request, d, log=log.append)
        for tif in d.glob("*.tif"):
            tif.unlink()
        used.add(summary["model"]["id"])
        files = sorted(str(f.relative_to(d)).replace("\\", "/") for f in d.rglob("*") if f.is_file())
        layers = json.loads((d / "layers.json").read_text(encoding="utf-8"))
        rel = f"static/results/{p.parent.name}"
        (d / "results.json").write_text(
            json.dumps({"summary": summary, "layers": layers, "files": {f: f"{rel}/{f}" for f in files}})
        )
        (d / "job.json").write_text(
            json.dumps({"id": p.parent.name, "status": "succeeded", "request": request, "log": log, "summary": summary})
        )
        print(
            f"✓ {p.parent.name}: {summary['affected_km2']} km² {summary['affected_label']} ({summary['model']['id']})"
        )

    cards = [c for c in models.list_models() if c["model_id"] in used or c["model_id"] == models.BASELINE_ID]
    (static / "examples.json").write_text(json.dumps(examples))
    (static / "models.json").write_text(json.dumps(cards))
    (static / "health.json").write_text(json.dumps({"status": "ok", "version": __version__, "device": "precomputed"}))
    (static / "site.json").write_text(
        json.dumps({"static": True, "codespaces": CODESPACES, "model": ", ".join(sorted(used)), "version": __version__})
    )
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**20
    print(f"✓ Showcase written to {out} ({size:.0f} MB)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "site"))
