"""GPFT-mini (GeoPulse Fusion Transformer, small edition), inference, checkpoints and the model registry.

Sensor adapters -> per-pixel quality-gated fusion (per time step) -> temporal change encoder
-> U-Net spatial context -> one head per task + a shared change head, temperature-scaled; MC dropout for uncertainty.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import baseline
from .grid import _starts
from .tasks import TASKS

SENSORS = {"s1": 2, "s2": 6}  # sensor -> input channels (see data.S1_BANDS / data.S2_BANDS)
TIMES = ("pre", "post")
BASELINE_ID = "threshold-baseline"

# Fixed, documented normalization so a checkpoint means the same thing on every AOI.
NORM = {
    "s1": (np.array([-12.0, -19.0]), np.array([5.0, 5.0])),  # VV, VH dB
    "s2": (
        np.array([0.08, 0.10, 0.10, 0.25, 0.20, 0.14]),  # B02 B03 B04 B08 B11 B12 reflectance
        np.array([0.06, 0.06, 0.08, 0.10, 0.10, 0.09]),
    ),
}


def to_tensors(arrays: dict, sensors=tuple(SENSORS)) -> dict[str, np.ndarray]:
    """Raw arrays -> normalized model inputs. Missing sensors become zeros with an all-zero validity mask."""
    h, w = next(iter(arrays.values())).shape[1:]
    out = {}
    for s, n in SENSORS.items():
        for t in TIMES:
            k = f"{s}_{t}"
            if k in arrays and s in sensors:
                a = arrays[k]
                v = np.isfinite(a).all(0)
                mean, std = NORM[s]
                x = np.where(v, (a - mean[:, None, None]) / std[:, None, None], 0.0)
            else:
                x, v = np.zeros((n, h, w)), np.zeros((h, w), bool)
            out[k] = x.astype("float32")
            out[f"{k}_valid"] = v[None].astype("float32")
    if "dem" in arrays:
        elev, slope = arrays["dem"]
        rel = (elev - np.nanmedian(elev)) / 20.0  # height above the AOI median: flood-relevant, AOI-independent
        out["dem"] = np.nan_to_num(np.stack([rel, slope / 10.0]), nan=0.0).astype("float32")
    else:
        out["dem"] = np.zeros((2, h, w), "float32")
    return out


def block(cin: int, cout: int) -> nn.Sequential:
    # GroupNorm, not BatchNorm: batch statistics are meaningless when modalities are randomly zeroed.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class GPFT(nn.Module):
    def __init__(self, channels: int = 32, dropout: float = 0.1, tasks: tuple[str, ...] = ("flood",)):
        super().__init__()
        c = channels
        self.tasks = list(tasks)
        self.config = {"channels": channels, "dropout": dropout, "tasks": self.tasks}
        self.adapters = nn.ModuleDict({s: block(n, c) for s, n in SENSORS.items()})
        self.gates = nn.ModuleDict({s: nn.Conv2d(c + 1, 1, 1) for s in SENSORS})
        self.terrain = block(2, c)
        self.enc1, self.enc2, self.enc3 = block(5 * c, c), block(c, 2 * c), block(2 * c, 4 * c)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), block(4 * c, 2 * c)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2), block(2 * c, c)
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c, len(self.tasks) + 1, 1)  # [task logits..., change logit]
        self.register_buffer("temperature", torch.ones(1))

    def fuse(self, batch: dict, t: str):
        """Per-pixel reliability weights across sensors; a missing/cloudy sensor gets ~zero weight."""
        feats, logits = [], []
        for s in SENSORS:
            v = batch[f"{s}_{t}_valid"]
            f = self.adapters[s](batch[f"{s}_{t}"]) * v
            feats.append(f)
            # Hard mask: an unobserved sensor gets exactly zero weight (all-missing pixels fall back to uniform
            # weights over zero features, i.e. "no information").
            logits.append(self.gates[s](torch.cat([f, v], 1)).masked_fill(v < 0.5, -1e4))
        w = torch.softmax(torch.stack(logits), 0)  # (S,B,1,H,W)
        return (w * torch.stack(feats)).sum(0), w

    def forward(self, batch: dict):
        pre, w_pre = self.fuse(batch, "pre")
        post, w_post = self.fuse(batch, "post")
        d = post - pre
        x = torch.cat([pre, post, d, d.abs(), self.terrain(batch["dem"])], 1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(self.drop(e3)), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(self.drop(d1)) / self.temperature, {"pre": w_pre, "post": w_post}


# --------------------------------------------------------------------------- inference


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_gpft(
    model: GPFT,
    arrays: dict,
    sensors=tuple(SENSORS),
    task: str = "flood",
    tile: int = 256,
    stride: int = 192,
    mc: int = 8,
    batch_size: int = 8,
    fp16: bool = False,
) -> dict[str, np.ndarray]:
    """Sliding-window inference with MC dropout; uncertainty = entropy of the mean task probability."""
    ti = model.tasks.index(task)
    stride = min(stride, tile)  # never leave gaps between windows
    t = to_tensors(arrays, sensors)
    h, w = t["dem"].shape[1:]
    ph, pw = max(h, tile), max(w, tile)  # tiny AOIs: zero-pad (validity 0 = "not observed")
    t = {k: np.pad(v, ((0, 0), (0, ph - h), (0, pw - w))) for k, v in t.items()}
    dev = next(model.parameters()).device
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()
    acc = np.zeros((len(model.tasks) + 1, ph, pw), "float32")
    wsum = np.zeros((len(SENSORS), ph, pw), "float32")
    cnt = np.zeros((ph, pw), "float32")
    offsets = [(r, c) for r in _starts(ph, tile, stride) for c in _starts(pw, tile, stride)]
    for i in range(0, len(offsets), batch_size):
        chunk = offsets[i : i + batch_size]
        batch = {
            k: torch.from_numpy(np.stack([v[:, r : r + tile, c : c + tile] for r, c in chunk])).to(dev)
            for k, v in t.items()
        }
        with torch.autocast(dev.type, dtype=torch.float16, enabled=fp16 and dev.type == "cuda"):
            runs = [model(batch) for _ in range(max(mc, 1))]
        probs = torch.stack([torch.sigmoid(r[0].float()) for r in runs]).mean(0).cpu().numpy()
        weights = runs[0][1]["post"].float().squeeze(2).permute(1, 0, 2, 3).cpu().numpy()  # (B,S,h,w)
        for j, (r, c) in enumerate(chunk):
            acc[:, r : r + tile, c : c + tile] += probs[j]
            wsum[:, r : r + tile, c : c + tile] += weights[j]
            cnt[r : r + tile, c : c + tile] += 1
    acc, wsum = (acc / cnt)[:, :h, :w], (wsum / cnt)[:, :h, :w]
    observed = (t["s1_post_valid"][0] + t["s2_post_valid"][0])[:h, :w] > 0
    out = {"target": acc[ti], "change": acc[-1], "uncertainty": baseline.binary_entropy(acc[ti])}
    for v in out.values():
        v[~observed] = np.nan
    out["weights"] = {s: float(wsum[k][observed].mean()) if observed.any() else None for k, s in enumerate(SENSORS)}
    return out


def predict(card: dict, arrays: dict, sensors=tuple(SENSORS), task: str = "flood", **kw) -> dict:
    if card["model_id"] == BASELINE_ID:
        return baseline.predict({k: v for k, v in arrays.items() if k == "dem" or k[:2] in sensors}, task)
    kw.setdefault("mc", int(os.environ.get("GEOPULSE_MC_PASSES", 8)))  # fewer passes on small CPU deployments
    out = predict_gpft(load(card["checkpoint"]), arrays, sensors, task, **kw)
    if task == "wildfire" and "s2_pre" in arrays and "s2_post" in arrays and "s2" in sensors:
        out["severity"] = baseline.severity(arrays)  # severity classes stay physically defined (dNBR)
    return out


# --------------------------------------------------------------------------- checkpoints + registry


def models_dir() -> Path:
    return Path(os.environ.get("GEOPULSE_MODELS", "models"))


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unversioned"
    except OSError:
        return "unversioned"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(model: GPFT, path: Path, card: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config}, path)
    card = card | {"checkpoint": path.name, "checkpoint_sha256": sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(card, indent=2))
    return card


_LOADED: dict[str, GPFT] = {}


def load(path: str | Path) -> GPFT:
    path = Path(path)
    if not path.is_absolute():
        path = models_dir() / path
    key = f"{path}:{path.stat().st_mtime}"
    if key not in _LOADED:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = GPFT(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        _LOADED[key] = model.to(device()).eval()
    return _LOADED[key]


BASELINE_CARD = {
    "model_id": BASELINE_ID,
    "version": "1.1.0",
    "tasks": list(TASKS),
    "architecture": "Training-free physics: SAR VV + MNDWI (flood), dNBR (burn), NDVI + VH drop (vegetation)",
    "modalities": ["sentinel1", "sentinel2"],
    "license": "Apache-2.0",
}


def list_models() -> list[dict]:
    cards = [BASELINE_CARD]
    for p in sorted(models_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        card = json.loads(p.read_text(encoding="utf-8"))
        if (p.parent / card.get("checkpoint", "")).is_file():
            cards.append(card)
    return cards


def resolve(model_id: str = "auto", task: str | None = None) -> dict:
    """'auto' = newest trained checkpoint that supports `task`, else the training-free baseline."""
    cards = [c for c in list_models() if task is None or task in c["tasks"]]
    if model_id == "auto":
        return cards[1] if len(cards) > 1 else cards[0]
    for card in cards:
        if model_id in (card["model_id"], f"{card['model_id']}-v{card['version']}"):
            return card
    raise KeyError(f"unknown model {model_id!r} for task {task!r}; available: {[c['model_id'] for c in cards]}")


def benchmark(tile: int = 256, batch_size: int = 16, fp16: bool = False, iters: int = 20) -> dict:
    """Forward-pass latency/throughput of an untrained GPFT-mini on synthetic tiles (architecture cost only)."""
    dev = device()
    model = GPFT().to(dev).eval()
    batch = {
        k: torch.randn(batch_size, n, tile, tile, device=dev)
        for s, n in SENSORS.items()
        for k in (f"{s}_pre", f"{s}_post")
    }
    batch |= {f"{s}_{t}_valid": torch.ones(batch_size, 1, tile, tile, device=dev) for s in SENSORS for t in TIMES}
    batch["dem"] = torch.randn(batch_size, 2, tile, tile, device=dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=fp16 and dev.type == "cuda"):
        for _ in range(3):
            model(batch)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(batch)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tile_km2 = (tile * 10 / 1000) ** 2
    return {
        "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
        "precision": "fp16" if fp16 and dev.type == "cuda" else "fp32",
        "tile": tile,
        "batch_size": batch_size,
        "params": sum(p.numel() for p in model.parameters()),
        "ms_per_tile": round(dt * 1000 / batch_size, 3),
        "km2_per_min_at_10m": round(batch_size * tile_km2 / dt * 60, 1),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == "cuda" else None,
        "torch": torch.__version__,
    }
