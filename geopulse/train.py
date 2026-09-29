"""Training (multi-task, sensor dropout, masked BCE+Dice, temperature scaling) and evaluation (robustness matrix)."""

from __future__ import annotations

import datetime as dt
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import baseline
from . import model as models
from .baseline import IGNORE
from .data import Log

SUBSETS = {"s1+s2": ("s1", "s2"), "s1 only": ("s1",), "s2 only": ("s2",)}


class Tiles(Dataset):
    """Tiles of one benchmark (one task). Labels are expanded to the model's channels: [tasks..., change];
    channels of other tasks are IGNORE, so one masked loss trains single- and multi-task models alike."""

    def __init__(self, root: str | Path, splits: tuple[str, ...], tasks: list[str], augment: bool = False):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.task = self.index.get("task", "flood")
        if self.task not in tasks:
            raise ValueError(f"{root} is a {self.task!r} dataset; the model only has {tasks}")
        self.ti, self.channels = tasks.index(self.task), len(tasks) + 1
        self.tiles = [t for t in self.index["tiles"] if t["split"] in splits]
        self.augment = augment

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        z = np.load(self.root / self.tiles[i]["path"])
        x = {k: torch.from_numpy(z[k].astype("float32")) for k in z.files if k != "labels"}
        raw = torch.from_numpy(z["labels"].astype("int64"))
        y = torch.full((self.channels, *raw.shape[1:]), IGNORE, dtype=torch.int64)
        y[self.ti], y[-1] = raw[0], raw[1]
        if self.augment:
            k, flip = random.randint(0, 3), random.random() < 0.5
            x = {n: _aug(v, k, flip) for n, v in x.items()}
            y = _aug(y, k, flip)
        return x, y


def _aug(t: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    t = torch.rot90(t, k, (-2, -1))
    return torch.flip(t, (-1,)) if flip else t


def restrict(batch: dict, sensors: tuple[str, ...]) -> dict:
    """Simulate sensors being unavailable at inference."""
    batch = dict(batch)
    for s in models.SENSORS:
        if s not in sensors:
            for t in models.TIMES:
                batch[f"{s}_{t}_valid"] = torch.zeros_like(batch[f"{s}_{t}_valid"])
    return batch


def sensor_dropout(batch: dict, probs: dict[str, float]) -> dict:
    """Randomly remove whole sensors per sample, never leaving a sample with no observed sensor."""
    batch = dict(batch)
    b = batch["dem"].shape[0]
    has = {s: batch[f"{s}_post_valid"].flatten(1).any(1).cpu() for s in models.SENSORS}
    drop = {s: (torch.rand(b) < probs.get(s, 0.0)) & has[s] for s in models.SENSORS}
    left = torch.stack([has[s] & ~drop[s] for s in models.SENSORS]).any(0)
    for s in models.SENSORS:
        keep = (~(drop[s] & left)).float().view(b, 1, 1, 1).to(batch["dem"].device)
        for t in models.TIMES:
            batch[f"{s}_{t}_valid"] = batch[f"{s}_{t}_valid"] * keep
    return batch


def loss_fn(logits: torch.Tensor, y: torch.Tensor, change_weight: float = 0.5, dice_weight: float = 1.0):
    """L = Σ_task (BCE + λ_dice·Dice) + λ_change·BCE_change, each over its labeled pixels only."""
    total = logits.new_zeros(())
    last = logits.shape[1] - 1
    for ch in range(logits.shape[1]):
        m = y[:, ch] != IGNORE
        if not m.any():
            continue
        z, t = logits[:, ch][m], y[:, ch][m].float()
        w, dice = (change_weight, 0.0) if ch == last else (1.0, dice_weight)
        total = total + w * F.binary_cross_entropy_with_logits(z, t)
        if dice:
            p = torch.sigmoid(z)
            total = total + dice * (1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1))
    return total


def metrics(p: np.ndarray, y: np.ndarray, threshold: float = 0.5, bins: int = 15) -> dict:
    """Pixel metrics on labeled pixels: IoU, F1, precision, recall, Brier, ECE."""
    pred, y = p >= threshold, y.astype(bool)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    edges = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(
        abs(p[edges == b].mean() - y[edges == b].mean()) * (edges == b).mean()
        for b in range(bins)
        if (edges == b).any()
    )
    return {
        "iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else None,
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "brier": round(float(((p - y) ** 2).mean()), 4),
        "ece": round(float(ece), 4),
        "pixels": int(y.size),
        "positive_fraction": round(float(y.mean()), 4),
    }


# --------------------------------------------------------------------------- predictors over benchmark tiles


def gpft_predictor(model, task: str, raw: bool = False):
    """x (one tile), sensors -> probability map (or temperature-free logits when raw, for fitting T)."""
    dev = next(model.parameters()).device
    ti = model.tasks.index(task)

    @torch.no_grad()
    def predict(x: dict, sensors: tuple[str, ...]) -> np.ndarray:
        model.eval()
        z = model(restrict({k: v[None].to(dev) for k, v in x.items()}, sensors))[0][0, ti].float()
        return (z * model.temperature).cpu().numpy() if raw else torch.sigmoid(z).cpu().numpy()

    return predict


def _raw_arrays(x: dict) -> dict:
    """Undo to_tensors normalisation so the physics baseline sees reflectance and dB again."""
    out = {}
    for s in models.SENSORS:
        mean, std = models.NORM[s]
        for t in models.TIMES:
            valid = x[f"{s}_{t}_valid"][0].numpy() > 0.5
            if valid.any():
                a = x[f"{s}_{t}"].numpy() * std[:, None, None] + mean[:, None, None]
                a[:, ~valid] = np.nan
                out[f"{s}_{t}"] = a
    return out


def baseline_predictor(task: str):
    def predict(x: dict, sensors: tuple[str, ...]) -> np.ndarray:
        arrays = {k: v for k, v in _raw_arrays(x).items() if k[:2] in sensors}
        if not arrays:
            return np.zeros(x["dem"].shape[1:], "float32")
        return np.nan_to_num(baseline.predict(arrays, task)["target"])  # unobserved -> "no change"

    return predict


def collect(predict, ds: Tiles, sensors: tuple[str, ...]) -> dict[str, tuple]:
    """Predictions and labels per event, labeled pixels only."""
    per_event: dict[str, list] = {}
    for i in range(len(ds)):
        x, y = ds[i]
        p, y0 = predict(x, sensors), y[ds.ti].numpy()
        m = y0 != IGNORE
        per_event.setdefault(ds.tiles[i]["event"], []).append((p[m], y0[m]))
    return {e: tuple(np.concatenate(a) for a in zip(*v, strict=True)) for e, v in per_event.items()}


def fit_temperature(model, val_sets: list[Tiles]) -> float:
    """Post-hoc temperature scaling on validation logits of every task (1-D grid search on NLL)."""
    zs, ys = [], []
    for ds in val_sets:
        for z, y in collect(gpft_predictor(model, ds.task, raw=True), ds, SUBSETS["s1+s2"]).values():
            zs.append(z)
            ys.append(y)
    if not zs:
        return 1.0
    z, y = torch.from_numpy(np.concatenate(zs)), torch.from_numpy(np.concatenate(ys)).float()
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 61))
    nll = [F.binary_cross_entropy_with_logits(z / t, y).item() for t in grid]
    return float(grid[int(np.argmin(nll))])


def evaluate_model(predict, ds: Tiles) -> dict:
    """Robustness matrix: pooled + per-event metrics for every sensor subset."""
    report = {}
    for name, sensors in SUBSETS.items():
        data = collect(predict, ds, sensors)
        if not data:
            continue
        per_event = {e: metrics(p, y) for e, (p, y) in data.items()}
        pooled = metrics(np.concatenate([d[0] for d in data.values()]), np.concatenate([d[1] for d in data.values()]))
        ious = [m["iou"] for m in per_event.values() if m["iou"] is not None]
        report[name] = {
            "pooled": pooled,
            "event_mean_iou": round(float(np.mean(ious)), 4) if ious else None,
            "per_event": per_event,
        }
    return report


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config_path: str | Path, log: Log = print) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))
    tc = cfg["train"]
    roots = [Path(r) for r in (cfg["dataset"] if isinstance(cfg["dataset"], list) else [cfg["dataset"]])]
    for r in roots:
        if not (r / "index.json").exists():
            raise RuntimeError(f"no dataset at {r}; run `geopulse dataset build` first")
    tasks = cfg.get("tasks") or list(
        dict.fromkeys(json.loads((r / "index.json").read_text(encoding="utf-8")).get("task", "flood") for r in roots)
    )
    train_sets = [Tiles(r, ("train",), tasks, augment=True) for r in roots]
    val_sets = [ds for ds in (Tiles(r, ("val",), tasks) for r in roots) if len(ds)]
    ds_train = ConcatDataset(train_sets)
    dev = models.device()
    model = models.GPFT(**cfg.get("model", {}), tasks=tasks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-4))
    loader = DataLoader(ds_train, batch_size=tc["batch_size"], shuffle=True, drop_last=len(ds_train) > tc["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=tc["epochs"] * len(loader))
    dropout = cfg.get("sensor_dropout") or {}
    n_val = sum(len(ds) for ds in val_sets)
    log(f"✓ tasks {tasks}: {len(ds_train)} train / {n_val} val tiles on {dev}; sensor dropout {dropout or 'off'}")
    out = models.models_dir() / f"{cfg['name']}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    def validate() -> dict:
        return {ds.task: evaluate_model(gpft_predictor(model, ds.task), ds) for ds in val_sets}

    best, history, t0 = -1.0, [], time.time()
    for epoch in range(tc["epochs"]):
        model.train()
        losses = []
        for x, y in loader:
            x, y = {k: v.to(dev) for k, v in x.items()}, y.to(dev)
            if dropout:
                x = sensor_dropout(x, dropout)
            loss = loss_fn(model(x)[0], y, cfg.get("loss", {}).get("change_weight", 0.5))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = validate()
        per_task = {t: v.get("s1+s2", {}).get("event_mean_iou") or 0.0 for t, v in val.items()}
        score = float(np.mean(list(per_task.values()))) if per_task else 0.0
        history.append({"epoch": epoch + 1, "loss": round(float(np.mean(losses)), 4), "val_iou": per_task})
        log(
            f"  epoch {epoch + 1:3d}  loss {history[-1]['loss']:.4f}  val IoU "
            + "  ".join(f"{t} {v:.4f}" for t, v in per_task.items())
        )
        if score >= best:
            best = score
            torch.save(model.state_dict(), out.with_suffix(".tmp"))
    model.load_state_dict(torch.load(out.with_suffix(".tmp"), weights_only=True))
    out.with_suffix(".tmp").unlink()
    if val_sets:
        t = fit_temperature(model, val_sets)
        model.temperature.fill_(t)
        log(f"✓ Temperature scaling: T = {t:.3f}")
    val = validate()
    card = {
        "model_id": cfg["name"],
        "version": cfg.get("version", "0.1.0"),
        "tasks": tasks,
        "architecture": "GPFT-mini: sensor adapters + quality-gated fusion + change encoder + U-Net",
        "backbone": "none (trained from scratch)",
        "modalities": ["sentinel1", "sentinel2", "dem"],
        "datasets": [f"{ds.index['dataset']}-{ds.index['version']}" for ds in train_sets],
        "dataset_manifest_sha256": {ds.index["dataset"]: ds.index["manifest_sha256"] for ds in train_sets},
        "label_source": sorted({s for ds in train_sets for s in ds.index["label_source"].values()}),
        "git_sha": models.git_sha(),
        "temperature": round(model.temperature.item(), 4),
        "metrics": {
            "val": {
                task: {k: v["pooled"] | {"event_mean_iou": v["event_mean_iou"]} for k, v in rep.items()}
                for task, rep in val.items()
            }
        },
        "license": "Apache-2.0",
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    card = models.save(model, out, card)
    run = {
        "config": cfg,
        "history": history,
        "card": card,
        "runtime_s": round(time.time() - t0, 1),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if dev.type == "cuda" else platform.processor(),
            "platform": platform.platform(),
        },
    }
    run_dir = Path("runs") / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(run, indent=2))
    log(f"✓ Checkpoint {out} (sha256 {card['checkpoint_sha256'][:12]}), run record {run_dir / 'run.json'}")
    return card


def evaluate(model_ref: str, dataset: str | Path, split: str = "test", log: Log = print) -> dict:
    """Robustness matrix of a model id / checkpoint file (or the physics baseline) on one benchmark split."""
    task = json.loads((Path(dataset) / "index.json").read_text(encoding="utf-8")).get("task", "flood")
    if model_ref == models.BASELINE_ID:
        name, ds = models.BASELINE_ID, Tiles(dataset, (split,), [task])
        predict = baseline_predictor(task)
    else:
        ckpt = model_ref if model_ref.endswith(".pt") else models.resolve(model_ref, task)["checkpoint"]
        model = models.load(ckpt)
        name, ds = Path(ckpt).stem, Tiles(dataset, (split,), model.tasks)
        predict = gpft_predictor(model, task)
    if not len(ds):
        raise RuntimeError(f"no {split!r} tiles in {dataset}")
    report = {
        "model": name,
        "task": task,
        "split": split,
        "tiles": len(ds),
        "label_source": ds.index["label_source"],
        "results": evaluate_model(predict, ds),
    }
    rows = [
        "| Sensors | IoU | F1 | Precision | Recall | ECE | Brier | Event-mean IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sub, r in report["results"].items():
        p = r["pooled"]
        rows.append(
            f"| {sub} | {p['iou']} | {p['f1']} | {p['precision']} | {p['recall']} | {p['ece']} | "
            f"{p['brier']} | {r['event_mean_iou']} |"
        )
    table = "\n".join(rows)
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{task}_{split}.json").write_text(json.dumps(report, indent=2))
    (out / f"eval_{task}_{split}.md").write_text(f"# {name} — {task} / {split}\n\n{table}\n", encoding="utf-8")
    log(table)
    log(f"✓ Report: {out / f'eval_{task}_{split}.json'}")
    return report
