"""GitHub Releases publishing/fetching, offline: a local directory stands in for the release download URL."""

import json
import shutil

import numpy as np
import pytest

from geopulse import model as models
from geopulse import releases


@pytest.fixture
def release_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOPULSE_RELEASES_URL", (tmp_path / "releases").as_uri())
    return tmp_path / "releases"


def _publish_model(tmp_path, release_dir):
    models.save(
        models.GPFT(channels=8, tasks=("wildfire",)),
        models.models_dir() / "burn.pt",
        {"model_id": "burn", "version": "0.0.1", "tasks": ["wildfire"]},
    )
    (tmp_path / "runs" / "burn").mkdir(parents=True)
    (tmp_path / "runs" / "burn" / "eval_wildfire_test.json").write_text("{}")
    files = releases.stage_models(release_dir / releases.MODELS_TAG, runs=tmp_path / "runs")
    shutil.rmtree(models.models_dir())  # a fresh machine
    return {f.name for f in files}


def test_stage_models_makes_flat_release_assets(tmp_path, release_dir):
    names = _publish_model(tmp_path, release_dir)
    assert names == {"burn.pt", "burn.json", "index.json", "eval__burn__eval_wildfire_test.json"}
    index = json.loads((release_dir / releases.MODELS_TAG / "index.json").read_text())
    assert [c["model_id"] for c in index] == ["burn"]


def test_pull_models_registers_verified_checkpoints(tmp_path, release_dir):
    _publish_model(tmp_path, release_dir)
    assert releases.pull_models(log=lambda m: None) == ["burn"]
    assert models.resolve("auto", "wildfire")["model_id"] == "burn"
    assert releases.pull_models(log=lambda m: None) == ["burn"]  # idempotent: verified files are kept


def test_pull_models_rejects_a_corrupted_checkpoint(tmp_path, release_dir):
    _publish_model(tmp_path, release_dir)
    (release_dir / releases.MODELS_TAG / "burn.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        releases.pull_models(log=lambda m: None)
    assert not (models.models_dir() / "burn.pt").exists()
    assert models.resolve("auto", "wildfire")["model_id"] == "threshold-baseline"  # the card was never written


def test_benchmarks_round_trip_through_release_zips(tmp_path, release_dir):
    bench = tmp_path / "bench" / "geopulse-bench-flood"
    (bench / "train").mkdir(parents=True)
    (bench / "index.json").write_text(json.dumps({"tiles": 1}))
    np.savez_compressed(bench / "train" / "t0.npz", x=np.arange(4))
    releases.stage_dataset(tmp_path / "bench", release_dir / releases.BENCH_TAG)
    out = releases.pull_dataset("geopulse-bench-flood", root=tmp_path / "pulled", log=lambda m: None)
    assert json.loads((out / "geopulse-bench-flood" / "index.json").read_text()) == {"tiles": 1}
    assert np.load(out / "geopulse-bench-flood" / "train" / "t0.npz")["x"].tolist() == [0, 1, 2, 3]
