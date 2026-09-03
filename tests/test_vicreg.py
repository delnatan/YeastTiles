"""Tests for the class-conditioned VICReg pretraining
(tileclass.training.vicreg) and its embedding-separability diagnostics
(tileclass.training.linear_probe) -- same "real training loop, tiny
synthetic data" convention as test_training.py.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("torch")

from tileclass.training.linear_probe import (
    LinearProbeParams,
    extract_embeddings,
    knn_accuracy,
    pca_2d,
    train_linear_probe,
    tsne_2d,
)
from tileclass.training.model import build_yeast_efficientnet
from tileclass.training.supervised import TrainingCancelled
from tileclass.training.vicreg import (
    ClassPairDataset,
    VICRegAugmentation,
    VICRegParams,
    pretrain_vicreg,
    warm_start_overlap,
)


def _write_synthetic_crop(path, seed, size=64):
    rng = np.random.default_rng(seed)
    brightfield = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    target = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    mask = np.full((size, size), 255, dtype=np.uint8)
    crop = np.stack([brightfield, target, mask], axis=0)
    tifffile.imwrite(path, crop, photometric="minisblack", metadata={"axes": "CYX"})


def _make_records(tmp_path, n_per_class=6, classes=("single", "junk")):
    records = []
    seed = 0
    for label in classes:
        for i in range(n_per_class):
            path = tmp_path / f"{label}_{i}.tif"
            _write_synthetic_crop(path, seed)
            seed += 1
            records.append((str(path), label))
    return records


# --- ClassPairDataset ---------------------------------------------------


def test_class_pair_dataset_draws_two_different_crops_of_same_class(tmp_path):
    records = _make_records(tmp_path, n_per_class=6)
    paths = [p for p, _ in records]
    labels = [label for _, label in records]

    dataset = ClassPairDataset(
        paths, labels, transform=lambda x: x, epoch_length=200, seed=0
    )
    assert dataset.singleton_classes == []

    for _ in range(len(dataset)):
        i, j = dataset._sample_pair_indices()
        assert labels[i] == labels[j]
        assert i != j  # every class here has >=2 members


def test_class_pair_dataset_falls_back_for_singleton_class(tmp_path):
    records = _make_records(tmp_path, n_per_class=6, classes=("single",))
    records.append((str(tmp_path / "rare.tif"), "rare"))
    _write_synthetic_crop(tmp_path / "rare.tif", seed=999)

    paths = [p for p, _ in records]
    labels = [label for _, label in records]
    dataset = ClassPairDataset(paths, labels, transform=lambda x: x, seed=0)
    assert dataset.singleton_classes == ["rare"]

    rare_idx = labels.index("rare")
    seen_rare_pair = False
    for _ in range(50):
        i, j = dataset._sample_pair_indices()
        if labels[i] == "rare":
            assert i == j == rare_idx
            seen_rare_pair = True
    assert seen_rare_pair


def test_class_pair_dataset_getitem_returns_augmented_pair(tmp_path):
    records = _make_records(tmp_path, n_per_class=4)
    paths = [p for p, _ in records]
    labels = [label for _, label in records]

    dataset = ClassPairDataset(paths, labels, transform=VICRegAugmentation())
    x1, x2 = dataset[0]
    assert x1.shape == (2, 64, 64)
    assert x2.shape == (2, 64, 64)
    assert x1.min() >= 0.0 and x1.max() <= 1.0


def test_class_pair_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ClassPairDataset(["a.tif", "b.tif"], ["single"], transform=lambda x: x)


def test_class_pair_dataset_rejects_empty_input():
    with pytest.raises(ValueError):
        ClassPairDataset([], [], transform=lambda x: x)


# --- pretrain_vicreg ------------------------------------------------------


def _tiny_vicreg_params():
    return VICRegParams(epochs=1, batch_size=4, num_workers=0)


def test_pretrain_vicreg_runs_and_saves_backbone(tmp_path, monkeypatch):
    weights_dir = tmp_path / "vicreg_weights"
    weights_path = weights_dir / "backbone.pth"
    meta_path = weights_dir / "meta.json"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_path)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_META_PATH", meta_path)

    records = _make_records(tmp_path, n_per_class=8)
    progress_events = []

    result = pretrain_vicreg(
        records,
        params=_tiny_vicreg_params(),
        progress_callback=progress_events.append,
    )

    assert weights_path.exists()
    assert meta_path.exists()
    assert set(result.categories) == {"single", "junk"}
    assert len(progress_events) == 1

    # The saved state dict should load straight into a headless backbone.
    import torch

    backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
    backbone.load_state_dict(torch.load(weights_path, map_location="cpu"))


def test_pretrain_vicreg_records_trained_on_paths(tmp_path, monkeypatch):
    weights_dir = tmp_path / "vicreg_weights"
    weights_path = weights_dir / "backbone.pth"
    meta_path = weights_dir / "meta.json"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_path)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_META_PATH", meta_path)

    records = _make_records(tmp_path, n_per_class=8)
    pretrain_vicreg(records, params=_tiny_vicreg_params())

    meta = json.loads(meta_path.read_text())
    assert set(meta["trained_on_paths"]) == {p for p, _ in records}
    assert meta["params"]["warm_start"] is True


def test_pretrain_vicreg_warm_starts_by_default(tmp_path, monkeypatch):
    """With `epochs=0` no gradient step ever runs, so whatever the
    backbone was initialized from is exactly what gets saved -- letting
    this assert warm-starting happened (or didn't) by comparing state
    dicts directly, rather than relying on it changing the loss."""
    import torch

    weights_dir = tmp_path / "vicreg_weights"
    weights_path = weights_dir / "backbone.pth"
    meta_path = weights_dir / "meta.json"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_path)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_META_PATH", meta_path)

    weights_dir.mkdir()
    seed_backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
    torch.save(seed_backbone.state_dict(), weights_path)

    records = _make_records(tmp_path, n_per_class=8)
    pretrain_vicreg(records, params=VICRegParams(epochs=0, batch_size=4, num_workers=0))

    saved = torch.load(weights_path, map_location="cpu")
    for key, value in seed_backbone.state_dict().items():
        assert torch.equal(value, saved[key])


def test_pretrain_vicreg_warm_start_false_ignores_live_slot(tmp_path, monkeypatch):
    import torch

    weights_dir = tmp_path / "vicreg_weights"
    weights_path = weights_dir / "backbone.pth"
    meta_path = weights_dir / "meta.json"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_path)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_META_PATH", meta_path)

    weights_dir.mkdir()
    seed_backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
    torch.save(seed_backbone.state_dict(), weights_path)

    records = _make_records(tmp_path, n_per_class=8)
    pretrain_vicreg(
        records,
        params=VICRegParams(epochs=0, batch_size=4, num_workers=0, warm_start=False),
    )

    saved = torch.load(weights_path, map_location="cpu")
    assert any(
        not torch.equal(value, saved[key])
        for key, value in seed_backbone.state_dict().items()
    )


def test_warm_start_overlap(tmp_path, monkeypatch):
    weights_dir = tmp_path / "vicreg_weights"
    weights_path = weights_dir / "backbone.pth"
    meta_path = weights_dir / "meta.json"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_path)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_META_PATH", meta_path)

    records = _make_records(tmp_path, n_per_class=8)
    pretrain_vicreg(records, params=_tiny_vicreg_params())

    trained_paths = [p for p, _ in records]
    probe_paths = trained_paths[:5] + ["/not/really/a/crop.tif"] * 3
    already_seen, total = warm_start_overlap(probe_paths, meta_path=meta_path)
    assert already_seen == 5
    assert total == 8


def test_warm_start_overlap_returns_none_without_recorded_provenance(tmp_path):
    missing = tmp_path / "missing.json"
    assert warm_start_overlap(["a.tif"], meta_path=missing) is None

    no_field = tmp_path / "old_style_meta.json"
    no_field.write_text('{"categories": ["single"]}')
    assert warm_start_overlap(["a.tif"], meta_path=no_field) is None


def test_pretrain_vicreg_output_dir_does_not_touch_live_slot(tmp_path, monkeypatch):
    live_dir = tmp_path / "vicreg_live"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", live_dir)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_PATH", live_dir / "backbone.pth")
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_META_PATH", live_dir / "meta.json")

    session_dir = tmp_path / "session_output"
    records = _make_records(tmp_path, n_per_class=8)

    result = pretrain_vicreg(records, params=_tiny_vicreg_params(), output_dir=session_dir)

    assert (session_dir / "backbone.pth").exists()
    assert (session_dir / "meta.json").exists()
    assert result.weights_path == session_dir / "backbone.pth"
    assert not live_dir.exists()  # yeastprep-driven runs never touch the live slot


def test_pretrain_vicreg_rejects_too_little_data(tmp_path):
    path = tmp_path / "only_one.tif"
    _write_synthetic_crop(path, seed=0)

    with pytest.raises(ValueError):
        pretrain_vicreg([(str(path), "single")], params=_tiny_vicreg_params())


def test_pretrain_vicreg_honors_cancel_check(tmp_path, monkeypatch):
    weights_dir = tmp_path / "vicreg_weights"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(
        "tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_dir / "backbone.pth"
    )
    monkeypatch.setattr(
        "tileclass.training.vicreg.VICREG_META_PATH", weights_dir / "meta.json"
    )

    records = _make_records(tmp_path, n_per_class=8)
    with pytest.raises(TrainingCancelled):
        pretrain_vicreg(
            records,
            params=_tiny_vicreg_params(),
            cancel_check=lambda: True,
        )
    assert not (weights_dir / "backbone.pth").exists()


# --- linear_probe ----------------------------------------------------------


def test_extract_embeddings_shape(tmp_path):
    import torch

    records = _make_records(tmp_path, n_per_class=3)
    paths = [p for p, _ in records]
    backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
    device = torch.device("cpu")
    embeddings = extract_embeddings(paths, backbone, device, batch_size=4)
    assert embeddings.shape == (len(paths), 1280)


def _separable_embeddings(n_per_class=20, num_classes=3, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    categories = [f"class{i}" for i in range(num_classes)]
    embeddings, labels = [], []
    for i, category in enumerate(categories):
        center = np.zeros(dim)
        center[i] = 10.0  # widely separated clusters
        cluster = rng.normal(loc=center, scale=0.1, size=(n_per_class, dim))
        embeddings.append(cluster)
        labels.extend([category] * n_per_class)
    return np.concatenate(embeddings).astype(np.float32), labels, categories


def test_linear_probe_separates_well_separated_clusters():
    embeddings, labels, categories = _separable_embeddings()
    result = train_linear_probe(
        embeddings, labels, categories, params=LinearProbeParams(epochs=200)
    )
    assert result.val_accuracy > 0.95
    assert result.train_count > 0 and result.val_count > 0


def test_knn_accuracy_separates_well_separated_clusters():
    embeddings, labels, _ = _separable_embeddings()
    acc = knn_accuracy(embeddings, labels, k=5)
    assert acc > 0.95


def test_linear_probe_rejects_unrecognized_category():
    embeddings, labels, categories = _separable_embeddings(n_per_class=5)
    labels[0] = "not_a_real_category"
    with pytest.raises(ValueError, match="not_a_real_category"):
        train_linear_probe(embeddings, labels, categories)


def test_pca_2d_shape():
    embeddings, _, _ = _separable_embeddings(n_per_class=5)
    coords = pca_2d(embeddings)
    assert coords.shape == (embeddings.shape[0], 2)


def test_tsne_2d_shape():
    embeddings, _, _ = _separable_embeddings(n_per_class=20)
    coords = tsne_2d(embeddings)
    assert coords.shape == (embeddings.shape[0], 2)


def test_tsne_2d_reproducible_with_same_seed():
    embeddings, _, _ = _separable_embeddings(n_per_class=20)
    first = tsne_2d(embeddings, seed=0)
    second = tsne_2d(embeddings, seed=0)
    np.testing.assert_array_equal(first, second)


def test_tsne_2d_handles_tiny_sample_count():
    """A handful of points, well below the requested perplexity -- the
    perplexity clamp in `tsne_2d` should keep sklearn from raising rather
    than requiring the caller to pick a perplexity that fits."""
    embeddings, _, _ = _separable_embeddings(n_per_class=1, num_classes=4)
    coords = tsne_2d(embeddings, perplexity=30.0)
    assert coords.shape == (embeddings.shape[0], 2)
