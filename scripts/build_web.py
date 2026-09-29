"""Build the static GeoPulse web platform: the full app, computing in each visitor's browser (no server).

    python scripts/build_web.py site/

Output: index.html, web/ (engine + worker), examples.json and web/site.json. Deployed to GitHub Pages by
.github/workflows/pages.yml and mirrored to the Hugging Face static Space with
`hf upload nafizrahaman/geopulse site --repo-type space`.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "geopulse" / "web"


def version() -> str:
    for line in (ROOT / "geopulse" / "__init__.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    raise RuntimeError("no __version__")


def main(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "web").mkdir(parents=True)
    shutil.copy2(WEB / "index.html", out / "index.html")
    for f in ("engine.js", "worker.js"):
        shutil.copy2(WEB / f, out / "web" / f)
    examples = []
    for p in sorted((ROOT / "examples").glob("*/request.yaml")):
        req = yaml.safe_load(p.read_text(encoding="utf-8"))
        req["aoi"] = json.loads((p.parent / req.pop("aoi_file", "aoi.geojson")).read_text(encoding="utf-8"))
        examples.append({"id": p.parent.name, **req})
    (out / "examples.json").write_text(json.dumps(examples))
    (out / "web" / "site.json").write_text(json.dumps({"version": version()}))
    space = ROOT / "deploy" / "huggingface-static" / "README.md"
    if space.exists():
        shutil.copy2(space, out / "README.md")
    (out / ".nojekyll").write_text("")  # GitHub Pages: serve files as-is
    print(f"✓ {out}: {len(examples)} examples, GeoPulse {version()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "site"))
