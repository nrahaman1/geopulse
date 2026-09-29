import itertools
import json

import numpy as np
import pytest
import torch

from geopulse import model as models
from geopulse.baseline import IGNORE
from geopulse.train import Tiles, _raw_arrays, baseline_predictor, loss_fn, metrics, sensor_dropout

from .conftest import EVENT, synthetic_inputs

TEMPORAL = ("s1_pre", "s1_post", "s2_pre", "s2_post")
COMBOS = [c for n in range(1, 5) for c in itertools.combinations(TEMPORAL, n) if any(k.endswith("post") for k in c)]


def batch_from(arrays: dict, b: int = 2) -> dict:
    return {k: torch.from_numpy(np.stack([v[:, :64, :64]] * b)) for k, v in models.to_tensors(arrays).items()}


@pytest.mark.parametrize("keys", COMBOS, ids=["+".join(c) for c in COMBOS])
def test_forward_every_modality_combination(keys):
    arrays = synthetic_inputs().arrays
    arrays = {k: v for k, v in arrays.items() if k in keys or k == "dem"}
    model = models.GPFT(channels=16).eval()
    logits, w = model(batch_from(arrays))
    assert logits.shape == (2, 2, 64, 64) and torch.isfinite(logits).all()
    assert torch.allclose(w["post"].sum(0), torch.ones_like(w["post"][0]))


def test_missing_sensor_gets_no_weight():
    arrays = synthetic_inputs(sensors=("s1",)).arrays
    _, w = models.GPFT(channels=16).eval()(batch_from(arrays))
    assert w["post"][1].max() == 0  # s2 absent -> zero fusion weight


def test_sensor_dropout_never_removes_every_sensor():
    batch = batch_from(synthetic_inputs().arrays, b=16)
    out = sensor_dropout(batch, {"s1": 1.0, "s2": 1.0})
    left = sum(out[f"{s}_post_valid"].flatten(1).any(1) for s in models.SENSORS)
    assert (left >= 1).all()


def test_sensor_dropout_zeroes_both_periods():
    torch.manual_seed(0)
    out = sensor_dropout(batch_from(synthetic_inputs().arrays, b=8), {"s2": 1.0})
    assert out["s2_pre_valid"].sum() == 0 and out["s2_post_valid"].sum() == 0 and out["s1_post_valid"].sum() > 0


def test_checkpoint_roundtrip(tmp_path):
    model = models.GPFT(channels=16).eval()
    card = models.save(model, tmp_path / "m.pt", {"model_id": "m", "version": "0.0.1"})
    assert len(card["checkpoint_sha256"]) == 64
    batch = batch_from(synthetic_inputs().arrays)
    again = models.load(tmp_path / "m.pt").cpu().eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch)[0], again(batch)[0])


@pytest.mark.parametrize("size", [(50, 70), (300, 280)])
def test_predict_gpft_tiles_and_masks(size):
    inp = synthetic_inputs()
    h, w = size
    arrays = {k: np.resize(v, (v.shape[0], h, w)).copy() for k, v in inp.arrays.items()}
    for k in ("s1_post", "s2_post"):
        arrays[k][:, :5, :5] = np.nan  # unobserved corner
    out = models.predict_gpft(models.GPFT(channels=16), arrays, tile=64, stride=48, mc=2)
    assert out["target"].shape == (h, w)
    assert np.isnan(out["target"][:5, :5]).all() and np.isfinite(out["target"][10:, 10:]).all()
    assert np.nanmax(out["uncertainty"]) <= 1.0


def test_loss_ignores_unlabeled_pixels():
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    y = torch.full((2, 2, 8, 8), IGNORE)
    assert loss_fn(logits, y).item() == 0
    y[:, 0, :4] = 1
    y[:, 0, 4:] = 0
    loss = loss_fn(logits, y)
    loss.backward()
    assert loss.item() > 0 and logits.grad[:, 1].abs().sum() == 0  # change head unlabeled -> no gradient


def test_metrics_known_values():
    m = metrics(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 0, 1, 0]))
    assert m["iou"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["f1"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["brier"] == pytest.approx(0.325)


def test_registry_falls_back_to_baseline():
    assert models.resolve("auto")["model_id"] == models.BASELINE_ID
    with pytest.raises(KeyError):
        models.resolve("nope")


def test_registry_filters_by_task(tmp_path):
    models.save(
        models.GPFT(channels=16, tasks=("wildfire",)),
        models.models_dir() / "burn.pt",
        {"model_id": "burn", "version": "0.0.1", "tasks": ["wildfire"]},
    )
    assert models.resolve("auto", "wildfire")["model_id"] == "burn"
    assert models.resolve("auto", "flood")["model_id"] == models.BASELINE_ID  # the burn model can't map floods
    with pytest.raises(KeyError):
        models.resolve("burn", "vegetation")


def test_multitask_heads_and_channel_selection():
    model = models.GPFT(channels=16, tasks=("flood", "wildfire", "vegetation"))
    logits, _ = model.eval()(batch_from(synthetic_inputs().arrays))
    assert logits.shape[1] == 4  # three task heads + shared change head
    with torch.no_grad():
        model.head.bias[:] = torch.tensor([-9.0, 9.0, -9.0, 0.0])  # only the wildfire head fires
    arrays = synthetic_inputs().arrays
    assert np.nanmean(models.predict_gpft(model, arrays, task="wildfire", tile=64, mc=1)["target"]) > 0.5
    assert np.nanmean(models.predict_gpft(model, arrays, task="flood", tile=64, mc=1)["target"]) < 0.5


def test_flood_checkpoints_without_tasks_still_load(tmp_path):
    torch.save(
        {"state_dict": models.GPFT(channels=16).state_dict(), "config": {"channels": 16, "dropout": 0.1}},
        tmp_path / "old.pt",
    )  # v0.1 checkpoints predate the `tasks` config key
    assert models.load(tmp_path / "old.pt").tasks == ["flood"]


def test_loss_covers_every_task_channel():
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    y = torch.full((2, 4, 8, 8), IGNORE)
    y[:, 1] = 1  # only the wildfire channel is labeled
    loss_fn(logits, y).backward()
    assert logits.grad[:, 1].abs().sum() > 0 and logits.grad[:, [0, 2, 3]].abs().sum() == 0


def test_tiles_expand_labels_to_model_channels(tmp_path):
    t = models.to_tensors({k: v[:, :64, :64] for k, v in synthetic_inputs("wildfire").arrays.items()})
    np.savez(tmp_path / "a.npz", labels=np.stack([np.ones((64, 64)), np.zeros((64, 64))]).astype("uint8"), **t)
    (tmp_path / "index.json").write_text(
        json.dumps({"task": "wildfire", "tiles": [{"event": "e", "split": "train", "path": "a.npz"}]})
    )
    x, y = Tiles(tmp_path, ("train",), ["flood", "wildfire", "vegetation"])[0]
    assert y.shape == (4, 64, 64)
    assert (y[1] == 1).all() and (y[3] == 0).all() and (y[[0, 2]] == IGNORE).all()
    with pytest.raises(ValueError):
        Tiles(tmp_path, ("train",), ["flood"])  # a flood-only model can't train on burn tiles


def test_baseline_sees_the_same_physics_through_tiles():
    arrays = synthetic_inputs("vegetation").arrays
    x = {k: torch.from_numpy(v) for k, v in models.to_tensors(arrays).items()}
    raw = _raw_arrays(x)
    np.testing.assert_allclose(raw["s2_pre"], arrays["s2_pre"], atol=1e-5)
    p = baseline_predictor("vegetation")(x, ("s1", "s2"))
    assert p[EVENT].mean() > 0.9


def test_checkpoints_load_by_name_from_a_relative_models_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEOPULSE_MODELS", "models")  # the default; a card's checkpoint is a bare file name
    card = models.save(models.GPFT(channels=8), tmp_path / "models" / "m.pt", {"model_id": "m"})
    assert models.load(card["checkpoint"]).tasks == ["flood"]


def test_inference_reports_progress_per_batch():
    calls = []
    arrays = synthetic_inputs().arrays
    models.predict_gpft(
        models.GPFT(channels=8),
        arrays,
        tile=64,
        stride=64,
        mc=1,
        batch_size=4,
        progress=lambda f, s: calls.append((f, s)),
    )
    fracs = [f for f, _ in calls]
    assert fracs == sorted(fracs) and fracs[-1] == 1.0 and len(calls) > 1
    assert calls[-1][1].startswith("Running the model (") and calls[-1][1].endswith(" tiles)")
