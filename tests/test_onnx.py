"""The ONNX export (used by the in-browser engine) must reproduce the PyTorch model, MC-dropout masks included."""

import json

import numpy as np
import pytest
import torch

from geopulse import model as models

ort = pytest.importorskip("onnxruntime")


def test_onnx_matches_torch_with_explicit_dropout_masks():
    net = models.GPFT(channels=8, tasks=("flood", "wildfire"))
    with torch.no_grad():
        net.temperature.fill_(1.3)
    card = models.save(net, models.models_dir() / "tiny.pt", {"model_id": "tiny", "version": "0", "tasks": net.tasks})
    card = models.export_onnx(card)
    assert json.loads((models.models_dir() / "tiny.json").read_text())["onnx"]["sha256"] == card["onnx"]["sha256"]

    rng = np.random.default_rng(1)
    h = w = 64
    feeds = {}
    for s, n in models.SENSORS.items():
        for t in models.TIMES:
            feeds[f"{s}_{t}"] = rng.normal(size=(2, n, h, w)).astype("float32")
            feeds[f"{s}_{t}_valid"] = (rng.random((2, 1, h, w)) > 0.3).astype("float32")
    feeds["dem"] = rng.normal(size=(2, 2, h, w)).astype("float32")
    p = card["onnx"]["dropout"]
    for k, c in card["onnx"]["masks"].items():
        feeds[k] = ((rng.random((2, c, 1, 1)) >= p) / (1 - p)).astype("float32")

    sess = ort.InferenceSession(str(models.models_dir() / card["onnx"]["file"]), providers=["CPUExecutionProvider"])
    logits, weights = sess.run(None, feeds)
    t = {k: torch.from_numpy(v) for k, v in feeds.items()}
    with torch.no_grad():
        want, w_want = models.read("tiny.pt")(t, masks=(t["mask_e3"], t["mask_d1"]))
    np.testing.assert_allclose(logits, want.numpy(), atol=1e-4)
    np.testing.assert_allclose(weights, w_want["post"].squeeze(2).transpose(0, 1).numpy(), atol=1e-5)
    assert logits.shape == (2, 3, h, w)  # two task heads + change head
    # The browser's self-check: ONNX on the probe input reproduces the card's expected logits.
    probe = card["onnx"]["probe"]
    feeds = {k: models.probe_values(k, tuple(v)) for k, v in probe["shapes"].items()}
    logits = sess.run(["logits"], feeds)[0].astype("float64")
    assert logits.mean() == pytest.approx(probe["mean"], abs=1e-4)
    assert np.abs(logits).mean() == pytest.approx(probe["absmean"], rel=1e-4)
