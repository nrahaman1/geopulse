"""GeoPulse: open-source multimodal geospatial AI for Earth-change intelligence."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

__version__ = "0.5.3"


@dataclass
class Result:
    path: Path
    summary: dict

    def save(self, dest: str | Path) -> Path:
        shutil.copytree(self.path, dest, dirs_exist_ok=True)
        return Path(dest)


class GeoPulse:
    """Python SDK.

    >>> model = GeoPulse.from_pretrained("auto")
    >>> result = model.predict(aoi="aoi.geojson", before=("2024-09-01", "2024-09-25"),
    ...                        after=("2024-09-27", "2024-10-10"))
    >>> result.save("outputs/")
    """

    def __init__(self, model_id: str = "auto"):
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, model_id: str = "auto") -> GeoPulse:
        from .model import resolve

        return cls(resolve(model_id)["model_id"])

    def predict(self, aoi, before, after, sensors=("s1", "s2"), task: str = "flood", output=None) -> Result:
        from . import pipeline

        if isinstance(aoi, (str, Path)):
            aoi = json.loads(Path(aoi).read_text(encoding="utf-8"))
        request = pipeline.make_request(aoi, before, after, task, sensors, self.model_id)
        out = Path(output or tempfile.mkdtemp(prefix="geopulse-"))
        return Result(out, pipeline.run(request, out))
