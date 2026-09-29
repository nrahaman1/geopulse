"""GitHub Releases: fetch and publish trained checkpoints and GeoPulse-Bench tiles. Everything lives on GitHub.

Models are the assets of one release (`models-v1`): `<model>.pt`, `<model>.onnx`, `<model>.json` cards and an
`index.json` of all cards (the in-browser engine picks models from it). Benchmarks are the assets of another
(`bench-v1`): one zip per task plus SHA256SUMS. Downloads need no account; publishing uses the `gh` CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from . import __version__
from . import model as models
from .data import Log

REPO = os.environ.get("GEOPULSE_REPO", "nrahaman1/geopulse")
MODELS_TAG = os.environ.get("GEOPULSE_MODELS_TAG", "models-v1")
BENCH_TAG = os.environ.get("GEOPULSE_BENCH_TAG", "bench-v1")
TASKS = ("flood", "wildfire", "vegetation")
DOCS = Path(__file__).resolve().parent.parent  # repo root when running from a checkout


def base_url(repo: str = REPO) -> str:
    return os.environ.get("GEOPULSE_RELEASES_URL", f"https://github.com/{repo}/releases/download")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str | None = None, tries: int = 4) -> Path:
    """Download to `dest` via a .part file; a checksum mismatch deletes the download and raises."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"geopulse/{__version__}"})
            with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if i == tries - 1 or getattr(e, "code", 500) < 500:  # 404 and friends will not heal by retrying
                part.unlink(missing_ok=True)
                raise
            time.sleep(2**i)
    if sha256 and _sha256(part) != sha256:
        part.unlink()
        raise RuntimeError(f"checksum mismatch for {dest.name} from {url}; the download was deleted")
    return part.replace(dest)


def _read_json(url: str):
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(_download(url, Path(tmp) / "f.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- models


def pull_models(tag: str = MODELS_TAG, repo: str = REPO, log: Log = print) -> list[str]:
    """Download every checkpoint (+ ONNX export) and card into models_dir(); files already present are verified."""
    dest = models.models_dir()
    url = f"{base_url(repo)}/{tag}"
    pulled = []
    for card in _read_json(f"{url}/index.json"):
        files = {card["checkpoint"]: card["checkpoint_sha256"]}
        if card.get("onnx"):
            files[card["onnx"]["file"]] = card["onnx"]["sha256"]
        for name, sha in files.items():
            if not ((dest / name).is_file() and _sha256(dest / name) == sha):
                _download(f"{url}/{name}", dest / name, sha)
        # The card goes last: a model only registers once all of its files are in place and verified.
        (dest / Path(card["checkpoint"]).with_suffix(".json")).write_text(json.dumps(card, indent=2), encoding="utf-8")
        pulled.append(card["model_id"])
        log(f"✓ {card['model_id']} v{card['version']} ({', '.join(card['tasks'])})")
    return pulled


def stage_models(out: Path, runs: Path = Path("runs")) -> list[Path]:
    """Release assets for every registered checkpoint: .pt/.onnx/.json, index.json and flattened eval reports."""
    out.mkdir(parents=True, exist_ok=True)
    cards = models.list_models()[1:]  # [0] is the training-free baseline
    files = []
    for card in cards:
        stem = Path(card["checkpoint"]).stem
        for suffix in (".pt", ".json", ".onnx"):
            if (models.models_dir() / f"{stem}{suffix}").exists():
                files.append(Path(shutil.copy2(models.models_dir() / f"{stem}{suffix}", out)))
    (out / "index.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")
    files.append(out / "index.json")
    for report in runs.glob("*/eval_*.json"):  # release assets are flat
        files.append(Path(shutil.copy2(report, out / f"eval__{report.parent.name}__{report.name}")))
    return files if cards else []


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True)


def _publish(tag: str, repo: str, title: str, notes: Path | None, files: list[Path], log: Log) -> str:
    try:
        _gh("release", "view", tag, "-R", repo)
    except subprocess.CalledProcessError:
        notes_args = ["--notes-file", str(notes)] if notes and notes.exists() else ["--notes", title]
        _gh("release", "create", tag, "-R", repo, "--title", title, "--latest=false", *notes_args)
    _gh("release", "upload", tag, "-R", repo, "--clobber", *map(str, files))
    url = f"https://github.com/{repo}/releases/tag/{tag}"
    log(f"✓ Published {len(files)} files to {url}")
    return url


def push_models(tag: str = MODELS_TAG, repo: str = REPO, log: Log = print) -> str:
    """Publish every registered checkpoint as release assets (maintainers; needs `gh auth login`)."""
    with tempfile.TemporaryDirectory() as tmp:
        files = stage_models(Path(tmp))
        if not files:
            raise RuntimeError("no trained checkpoints in models/ to publish")
        return _publish(tag, repo, "GeoPulse models", DOCS / "MODEL_CARD.md", files, log)


# --------------------------------------------------------------------------- benchmarks


def _tasks(name: str) -> list[str]:
    return list(TASKS) if name == "all" else [name.removeprefix("geopulse-bench-")]


def pull_dataset(
    name: str = "all", tag: str = BENCH_TAG, repo: str = REPO, root: str | Path = "data/bench", log: Log = print
) -> Path:
    """Download prebuilt benchmark tiles instead of rebuilding them from STAC (e.g. name=geopulse-bench-wildfire)."""
    root = Path(root)
    url = f"{base_url(repo)}/{tag}"
    with tempfile.TemporaryDirectory() as tmp:
        sums = dict(
            line.split()[::-1] for line in _download(f"{url}/SHA256SUMS", Path(tmp) / "SUMS").read_text().splitlines()
        )
        for task in _tasks(name):
            zname = f"geopulse-bench-{task}.zip"
            with zipfile.ZipFile(_download(f"{url}/{zname}", Path(tmp) / zname, sums[zname])) as z:
                if any(Path(n).is_absolute() or ".." in Path(n).parts for n in z.namelist()):
                    raise RuntimeError(f"{zname} contains unsafe paths")
                z.extractall(root)
            log(f"✓ geopulse-bench-{task} -> {root}")
    return root


def stage_dataset(root: str | Path, out: Path) -> list[Path]:
    """Release assets: one zip per benchmark (index.json + tiles) and their SHA256SUMS."""
    root = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for task in TASKS:
        src = root / f"geopulse-bench-{task}"
        if not (src / "index.json").exists():
            continue
        zpath = out / f"{src.name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # .npz tiles are already compressed
            for f in [src / "index.json", *sorted(src.glob("*/*.npz"))]:
                z.write(f, f.relative_to(root).as_posix())
        files.append(zpath)
    if not files:
        raise RuntimeError(f"no benchmarks under {root}")
    (out / "SHA256SUMS").write_text("".join(f"{_sha256(f)}  {f.name}\n" for f in files))
    return [*files, out / "SHA256SUMS"]


def push_dataset(tag: str = BENCH_TAG, repo: str = REPO, root: str | Path = "data/bench", log: Log = print) -> str:
    """Publish the benchmark zips (maintainers; needs `gh auth login`)."""
    with tempfile.TemporaryDirectory() as tmp:
        files = stage_dataset(root, Path(tmp))
        return _publish(tag, repo, "GeoPulse-Bench", DOCS / "docs" / "DATASET_CARD.md", files, log)
