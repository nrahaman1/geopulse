"""Hugging Face Hub: fetch and publish trained checkpoints and GeoPulse-Bench tiles."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from . import __version__
from . import model as models
from .data import Log

MODEL_REPO = os.environ.get("GEOPULSE_HF_MODELS", "nafizrahaman/geopulse-gpft-mini")
DATA_REPO = os.environ.get("GEOPULSE_HF_DATA", "nafizrahaman/geopulse-bench")
DOCS = Path(__file__).resolve().parent.parent  # repo root when running from a checkout

MODEL_CARD_HEADER = """---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - remote-sensing
  - earth-observation
  - sentinel-1
  - sentinel-2
  - sar
  - change-detection
  - flood-mapping
  - wildfire
  - deforestation
  - geospatial
datasets:
  - {data_repo}
---

# GeoPulse GPFT-mini checkpoints

Weights for [GeoPulse](https://github.com/nrahaman1/geopulse), an open-source multimodal Earth-change mapper
(flood, wildfire/burn, vegetation disturbance) built on Sentinel-1, Sentinel-2 and the Copernicus DEM.

```bash
pip install https://github.com/nrahaman1/geopulse/archive/refs/heads/main.zip
geopulse models pull                      # downloads these checkpoints and verifies their SHA-256
geopulse infer --task wildfire --aoi aoi.geojson --before 2024-12-01/2025-01-06 --after 2025-02-01/2025-03-31
```

Each `<model>.pt` has a `<model>.json` card (tasks, training datasets and manifest hashes, label sources,
validation metrics, temperature, checkpoint SHA-256). Test-set reports are in `eval/`.

"""


def pull_models(repo: str = MODEL_REPO, revision: str | None = None, log: Log = print) -> list[str]:
    """Download every checkpoint + card into models_dir() and verify checksums (corrupt files are deleted)."""
    dest = models.models_dir()
    snapshot_download(
        repo, revision=revision, allow_patterns=["*.pt", "*.json"], ignore_patterns=["eval/*"], local_dir=dest
    )
    pulled = []
    for card_path in sorted(dest.glob("*.json")):
        card = json.loads(card_path.read_text(encoding="utf-8"))
        ckpt = dest / card.get("checkpoint", "")
        if not ckpt.is_file():
            continue
        if models.sha256(ckpt) != card.get("checkpoint_sha256"):
            ckpt.unlink()
            raise RuntimeError(f"checksum mismatch for {ckpt.name} from {repo}; the file was deleted")
        pulled.append(card["model_id"])
        log(f"✓ {card['model_id']} v{card['version']} ({', '.join(card['tasks'])})")
    return pulled


def stage_models(out: Path, runs: Path = Path("runs"), data_repo: str = DATA_REPO) -> list[str]:
    """Lay out a model repo: checkpoints + cards at the root, eval reports under eval/, README model card."""
    out.mkdir(parents=True, exist_ok=True)
    staged = []
    for card in models.list_models()[1:]:  # [0] is the training-free baseline
        for suffix in (".pt", ".json"):
            shutil.copy2(models.models_dir() / f"{Path(card['checkpoint']).stem}{suffix}", out)
        staged.append(card["model_id"])
    for report in runs.glob("*/eval_*.json"):
        (out / "eval" / report.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(report, out / "eval" / report.parent.name / report.name)
    body = (DOCS / "MODEL_CARD.md").read_text(encoding="utf-8") if (DOCS / "MODEL_CARD.md").exists() else ""
    (out / "README.md").write_text(MODEL_CARD_HEADER.format(data_repo=data_repo) + body, encoding="utf-8")
    return staged


def push_models(repo: str = MODEL_REPO, private: bool = False, log: Log = print) -> str:
    """Publish every registered checkpoint (uses the token from `hf auth login`)."""
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage_models(Path(tmp))
        if not staged:
            raise RuntimeError("no trained checkpoints in models/ to publish")
        api.upload_folder(repo_id=repo, folder_path=tmp, commit_message=f"GeoPulse v{__version__}: {', '.join(staged)}")
    url = f"https://huggingface.co/{repo}"
    log(f"✓ Published {len(staged)} models to {url}")
    return url


def pull_dataset(name: str = "all", repo: str = DATA_REPO, root: str | Path = "data/bench", log: Log = print) -> Path:
    """Download prebuilt benchmark tiles instead of rebuilding them from STAC (e.g. name=geopulse-bench-wildfire)."""
    patterns = ["*/index.json", "*/*/*.npz"] if name == "all" else [f"{name}/*"]
    snapshot_download(repo, repo_type="dataset", allow_patterns=patterns, local_dir=root)
    log(f"✓ {name} -> {root}")
    return Path(root)


def push_dataset(
    repo: str = DATA_REPO, root: str | Path = "data/bench", private: bool = False, log: Log = print
) -> str:
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(root),
        allow_patterns=["*/index.json", "*/*/*.npz"],
        commit_message=f"GeoPulse-Bench (GeoPulse v{__version__})",
    )
    card = DOCS / "docs" / "DATASET_CARD.md"
    if card.exists():
        api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(card), path_in_repo="README.md")
    url = f"https://huggingface.co/datasets/{repo}"
    log(f"✓ Published benchmark tiles to {url}")
    return url
