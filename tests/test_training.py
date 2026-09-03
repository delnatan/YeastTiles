"""End-to-end test of tileclass.training.supervised.train_classifier on
tiny synthetic crops -- exercises the real training loop (no mocking,
matching this project's convention) at a scale that runs in a few seconds
on CPU: 1 epoch per stage, ~20 crops across 2 categories.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("torch")

from tileclass.training.supervised import (
    TrainingCancelled,
    TrainingParams,
    resolve_target_categories,
    train_classifier,
)


def _write_synthetic_crop(path, seed, size=64):
    rng = np.random.default_rng(seed)
    brightfield = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    target = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    mask = np.full((size, size), 255, dtype=np.uint8)
    crop = np.stack([brightfield, target, mask], axis=0)
    tifffile.imwrite(path, crop, photometric="minisblack", metadata={"axes": "CYX"})


def _make_records(tmp_path, n_per_class=6):
    records = []
    seed = 0
    for label in ("single", "junk"):
        for i in range(n_per_class):
            path = tmp_path / f"{label}_{i}.tif"
            _write_synthetic_crop(path, seed)
            seed += 1
            records.append((str(path), label))
    return records


def _tiny_params():
    return TrainingParams(
        val_frac=0.3,
        batch_size=4,
        probe_epochs=1,
        finetune_epochs=1,
    )


def test_resolve_target_categories_returns_sorted_record_labels():
    records = [("a.tif", "single"), ("b.tif", "junk"), ("c.tif", "single")]
    assert resolve_target_categories(records) == ["junk", "single"]


def test_train_classifier_runs_and_saves_weights(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    weights_path = weights_dir / "weights.pth"
    meta_path = weights_dir / "meta.json"
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_PATH", weights_path)

    records = _make_records(tmp_path)
    progress_events = []

    result = train_classifier(
        records, params=_tiny_params(), progress_callback=progress_events.append
    )

    assert weights_path.exists()
    assert meta_path.exists()
    assert set(result.categories) == {"single", "junk"}
    assert result.train_count > 0
    assert result.val_count > 0
    assert 0.0 <= result.val_accuracy <= 1.0
    assert set(result.per_class_accuracy) == set(result.categories)
    # probe (1 epoch) + finetune (1 epoch)
    assert len(progress_events) == 2
    assert {e.stage for e in progress_events} == {"probe", "finetune"}


def test_train_classifier_backs_up_existing_weights_on_rerun(tmp_path, monkeypatch):
    """Every run trains from scratch (no warm start -- see module
    docstring), but a rerun that lands in the same live slot should still
    back up whatever was there before overwriting it."""
    weights_dir = tmp_path / "weights"
    weights_path = weights_dir / "weights.pth"
    meta_path = weights_dir / "meta.json"
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_PATH", weights_path)

    records = _make_records(tmp_path)

    train_classifier(records, params=_tiny_params())
    assert weights_path.exists()
    assert not list(weights_dir.glob("weights.pth.bak-*"))

    # A second run should back up the first run's weights before overwriting.
    train_classifier(records, params=_tiny_params())
    backups = list(weights_dir.glob("weights.pth.bak-*"))
    assert len(backups) == 1


def test_train_classifier_records_trained_on_paths(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_PATH", weights_dir / "weights.pth")

    records = _make_records(tmp_path)
    result = train_classifier(records, params=_tiny_params())

    meta = json.loads((weights_dir / "meta.json").read_text())
    trained_paths = set(meta["trained_on_paths"])
    assert trained_paths
    assert trained_paths <= {path for path, _ in records}
    assert len(trained_paths) == result.train_count


def test_train_classifier_warm_starts_from_vicreg_backbone(tmp_path, monkeypatch):
    import torch

    from tileclass.training.model import build_yeast_efficientnet

    weights_dir = tmp_path / "weights"
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_PATH", weights_dir / "weights.pth")

    backbone_path = tmp_path / "vicreg_backbone.pth"
    headless_backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
    torch.save(headless_backbone.state_dict(), backbone_path)

    records = _make_records(tmp_path)
    result = train_classifier(
        records, params=_tiny_params(), backbone_weights_path=backbone_path
    )

    assert result.train_count > 0
    assert (weights_dir / "weights.pth").exists()


def test_train_classifier_rejects_mismatched_backbone(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_PATH", weights_dir / "weights.pth")

    import torch

    bogus_path = tmp_path / "bogus_backbone.pth"
    torch.save({"not": "a real state dict"}, bogus_path)

    records = _make_records(tmp_path)
    with pytest.raises(ValueError, match="doesn't match"):
        train_classifier(records, params=_tiny_params(), backbone_weights_path=bogus_path)


def test_train_classifier_rejects_unrecognized_category(tmp_path, monkeypatch):
    """With an explicit `categories` override narrower than what's in
    `records` -- the only way an "unrecognized category" can arise now
    that vocabulary is never inferred from a deployed classifier (see
    module docstring)."""
    weights_dir = tmp_path / "weights"
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_PATH", weights_dir / "weights.pth")

    records = _make_records(tmp_path)
    records.append((str(tmp_path / "mystery.tif"), "not_a_real_category"))
    _write_synthetic_crop(tmp_path / "mystery.tif", seed=999)

    with pytest.raises(ValueError, match="not_a_real_category"):
        train_classifier(
            records, params=_tiny_params(), categories=["single", "junk"]
        )


def test_train_classifier_rejects_too_little_data(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_PATH", weights_dir / "weights.pth")

    path = tmp_path / "only_one.tif"
    _write_synthetic_crop(path, seed=0)

    with pytest.raises(ValueError):
        train_classifier([(str(path), "single")], params=_tiny_params())


def test_train_classifier_output_dir_does_not_touch_live_slot(tmp_path, monkeypatch):
    live_dir = tmp_path / "live_weights"
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_DIR", live_dir)
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_PATH", live_dir / "weights.pth")

    session_dir = tmp_path / "session_output"
    records = _make_records(tmp_path)

    result = train_classifier(records, params=_tiny_params(), output_dir=session_dir)

    assert (session_dir / "weights.pth").exists()
    assert (session_dir / "meta.json").exists()
    assert result.weights_path == session_dir / "weights.pth"
    assert not live_dir.exists()  # yeastprep-driven runs never touch the live slot


def test_train_classifier_honors_cancel_check(tmp_path, monkeypatch):
    weights_dir = tmp_path / "weights"
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.supervised.WEIGHTS_PATH", weights_dir / "weights.pth")

    records = _make_records(tmp_path)

    with pytest.raises(TrainingCancelled):
        train_classifier(
            records,
            params=_tiny_params(),
            cancel_check=lambda: True,
        )
    assert not (weights_dir / "weights.pth").exists()
