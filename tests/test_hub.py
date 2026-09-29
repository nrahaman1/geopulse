"""Hugging Face publishing/fetching, offline: the Hub download is replaced by a local copy."""

import json
import shutil

import pytest

from geopulse import hub
from geopulse import model as models


def _publish_fake_repo(tmp_path):
    remote = tmp_path / "remote"
    models.save(
        models.GPFT(channels=8, tasks=("wildfire",)),
        remote / "burn.pt",
        {"model_id": "burn", "version": "0.0.1", "tasks": ["wildfire"]},
    )
    (remote / "eval").mkdir()
    (remote / "eval" / "report.json").write_text("{}")
    return remote


def _fake_snapshot(remote):
    def snapshot_download(repo, revision=None, allow_patterns=None, ignore_patterns=None, local_dir=None, **kw):
        for f in remote.glob("*"):
            if f.is_file() and f.suffix in (".pt", ".json"):
                shutil.copy2(f, local_dir)

    return snapshot_download


def test_pull_models_registers_verified_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "snapshot_download", _fake_snapshot(_publish_fake_repo(tmp_path)))
    models.models_dir().mkdir(parents=True, exist_ok=True)
    assert hub.pull_models("someone/repo", log=lambda m: None) == ["burn"]
    assert models.resolve("auto", "wildfire")["model_id"] == "burn"
    assert not (models.models_dir() / "report.json").exists()  # eval reports stay on the Hub


def test_pull_models_rejects_corrupted_checkpoint(tmp_path, monkeypatch):
    remote = _publish_fake_repo(tmp_path)
    (remote / "burn.pt").write_bytes(b"tampered")
    monkeypatch.setattr(hub, "snapshot_download", _fake_snapshot(remote))
    models.models_dir().mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="checksum"):
        hub.pull_models("someone/repo", log=lambda m: None)
    assert not (models.models_dir() / "burn.pt").exists()


def test_stage_models_builds_a_hub_repo(tmp_path):
    models.save(
        models.GPFT(channels=8),
        models.models_dir() / "flood.pt",
        {"model_id": "flood", "version": "0.0.1", "tasks": ["flood"]},
    )
    runs = tmp_path / "runs" / "flood"
    runs.mkdir(parents=True)
    (runs / "eval_flood_test.json").write_text(json.dumps({"results": {}}))
    out = tmp_path / "staged"
    assert hub.stage_models(out, runs=tmp_path / "runs") == ["flood"]
    assert {p.name for p in out.iterdir()} >= {"flood.pt", "flood.json", "README.md", "eval"}
    card = (out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\nlicense: apache-2.0") and "geopulse models pull" in card
    assert (out / "eval" / "flood" / "eval_flood_test.json").exists()
