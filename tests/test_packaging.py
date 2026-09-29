"""One version everywhere: the Python package, the web/desktop app (Tauri reads app/package.json) and CITATION.cff."""

import json
import re
from pathlib import Path

import geopulse

ROOT = Path(__file__).resolve().parents[1]


def test_versions_agree():
    app = json.loads((ROOT / "app" / "package.json").read_text(encoding="utf-8"))["version"]
    cff = re.search(r"^version: (.+)$", (ROOT / "CITATION.cff").read_text(encoding="utf-8"), re.M).group(1)
    assert app == cff == geopulse.__version__
