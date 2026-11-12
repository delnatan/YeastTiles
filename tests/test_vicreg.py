"""Tests for the self-supervised VICReg pretraining
(tileclass.training.vicreg) and its embedding-separability diagnostics
(tileclass.training.embeddings) -- same "real training loop, tiny
synthetic data" convention as test_training.py.
"""

import json

import numpy as np
import pytest
import tifffile

pytest.importorskip("torch")

from tileclass.tile_container import write_container
from tileclass.training.embeddings import extract_embeddings, knn_accuracy, umap_2d
from tileclass.training.model import build_yeast_efficientnet
from tileclass.training.supervised import TrainingCancelled
from tileclass.training.vicreg import (
    RandomChannelIntensity,
    RandomMaskJitter,
    RandomRot90,
    TileViewPairDataset,
    VICRegAugmentation,
    VICRegParams,
    pretrain_vicreg,
    warm_start_overlap,
)


def _synthetic_crop(seed, size=64):
    rng = np.random.default_rng(seed)
    brightfield = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    target = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    mask = np.full((size, size), 255, dtype=np.uint8)
    return np.stack([brightfield, target, mask], axis=0)


def _write_synthetic_crop(path, seed, size=64):
    tifffile.imwrite(
        path, _synthetic_crop(seed, size), photometric="minisblack", metadata={"axes": "CYX"}
    )


def _make_records(tmp_path, n_per_class=6, classes=("single", "junk")):
    """Packs every crop into one `fov.tiles` container (matching real
    usage -- see tile_container.py's module docstring) rather than one
    loose tif per crop, so records here exercise the same container-ref
    path format `_save_backbone`'s `trained_on` provenance expects."""
    container_path = tmp_path / "fov.tiles"
    records = []
    cells = []
    seed = 0
    for label in classes:
        for i in range(n_per_class):
            cell_id = f"fov_cell{seed:05d}"
            cells.append((cell_id, seed, _synthetic_crop(seed)))
            records.append((str(container_path / cell_id), label))
            seed += 1
    write_container(container_path, cells)
    return records


# --- TileViewPairDataset / augmentation -----------------------------------


def _make_paths(tmp_path, n=12):
    return [p for p, _ in _make_records(tmp_path, n_per_class=n, classes=("unlabeled",))]


def test_tile_view_pair_dataset_returns_two_views_of_the_same_tile(tmp_path):
    paths = _make_paths(tmp_path, n=4)
    calls = []

    def recording_transform(x):
        calls.append(x)
        return x

    dataset = TileViewPairDataset(paths, transform=recording_transform)
    assert len(dataset) == 4
    x1, x2 = dataset[2]
    # Both views come from tile 2's own pixels, each through its own
    # transform call.
    assert len(calls) == 2 and calls[0] is calls[1]
    # The transform gets [brightfield, fluorescence, mask] unmasked, and is
    # responsible for applying the mask itself.
    assert x1.shape == (3, 64, 64)
    assert x1.equal(x2)


def test_tile_view_pair_dataset_with_augmentation(tmp_path):
    dataset = TileViewPairDataset(_make_paths(tmp_path, n=2), transform=VICRegAugmentation())
    x1, x2 = dataset[0]
    assert x1.shape == x2.shape == (2, 64, 64)
    assert x1.min() >= 0.0 and x1.max() <= 1.0
    assert not x1.equal(x2)  # independent draws


def test_tile_view_pair_dataset_rejects_empty_input():
    with pytest.raises(ValueError):
        TileViewPairDataset([], transform=lambda x: x)


def test_random_rot90_is_an_exact_rotation():
    import torch

    x = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    y = RandomRot90()(x)
    assert any(y.equal(torch.rot90(x, k, dims=(-2, -1))) for k in range(4))


def test_random_channel_intensity_keeps_masked_out_pixels_at_zero():
    import torch

    x = torch.rand(2, 8, 8)
    x[:, :4] = 0.0  # outside the cell mask
    y = RandomChannelIntensity(p=1.0)(x)
    assert torch.all(y[:, :4] == 0)
    assert y.shape == x.shape


def _crop_with_disk_mask(size=64, radius=20):
    import torch

    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    mask = (((yy - size / 2) ** 2 + (xx - size / 2) ** 2) <= radius**2).float()
    image = torch.rand(2, size, size) * 0.5 + 0.25  # unmasked: nonzero everywhere
    return torch.cat([image, mask[None]], dim=0), mask


def test_vicreg_augmentation_returns_masked_two_channel_view():
    import torch

    x, mask = _crop_with_disk_mask()
    torch.manual_seed(0)
    for _ in range(20):
        y = VICRegAugmentation()(x)
        assert y.shape == (2, 64, 64)
        assert y.min() >= 0.0 and y.max() <= 1.0
        # Re-masked after noise: the far corners (outside any jittered,
        # rotated or zoomed disk) are exact zeros, as at inference.
        assert torch.all(y[:, :4, :4] == 0)
        assert torch.all(y[:, -4:, -4:] == 0)


def test_vicreg_augmentation_rotation_is_bilinear():
    import torchvision.transforms.v2 as T

    aug = VICRegAugmentation()
    rotations = [t for t in aug._spatial.transforms if isinstance(t, T.RandomRotation)]
    assert rotations and all(
        r.interpolation == T.InterpolationMode.BILINEAR for r in rotations
    )


def test_classification_transform_rotation_is_bilinear():
    import torchvision.transforms.v2 as T

    from tileclass.training.dataset import ClassificationTransform

    transform = ClassificationTransform(train=True)._build()
    rotations = [t for t in transform.transforms if isinstance(t, T.RandomRotation)]
    assert rotations and all(
        r.interpolation == T.InterpolationMode.BILINEAR for r in rotations
    )


def test_random_mask_jitter_dilates_or_erodes_by_at_most_max_px():
    import random

    import torch

    _, mask = _crop_with_disk_mask()
    mask = mask[None]
    random.seed(0)
    changed = set()
    for _ in range(40):
        out = RandomMaskJitter(max_px=2, p=1.0)(mask)
        assert out.shape == mask.shape
        assert set(out.unique().tolist()) <= {0.0, 1.0}
        area_delta = out.sum().item() - mask.sum().item()
        changed.add(area_delta > 0)
        # 2 px of dilation/erosion on a radius-20 disk: well under 2 rings
        # of 2*pi*22 pixels each.
        assert abs(area_delta) < 2 * 2 * 3.1416 * 22 * 2
    assert changed == {True, False}  # both dilation and erosion occur


def test_load_masked_crop_matches_crop_with_mask(tmp_path):
    from tileclass.training.dataset import load_crop_with_mask, load_masked_crop

    path = _make_paths(tmp_path, n=1)[0]
    full = load_crop_with_mask(path)
    assert full.shape == (3, 64, 64)
    assert set(np.unique(full[2]).tolist()) <= {0.0, 1.0}
    np.testing.assert_array_equal(load_masked_crop(path), full[:2] * full[2:3])


def test_lr_schedule_warms_up_then_decays_to_zero():
    from tileclass.training.vicreg import _lr_lambda

    f = _lr_lambda(warmup_steps=10, total_steps=100)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)
    assert f(55) == pytest.approx(0.5)
    assert f(100) == pytest.approx(0.0, abs=1e-9)
    assert all(f(s) >= f(s + 1) for s in range(10, 100))


def test_param_groups_exclude_norm_and_bias_from_weight_decay():
    import torch.nn as nn

    from tileclass.training.vicreg import _param_groups

    model = nn.Sequential(nn.Conv2d(2, 4, 3), nn.BatchNorm2d(4), nn.Linear(4, 2))
    decay, no_decay = _param_groups(model, 0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


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

    paths = _make_paths(tmp_path, n=16)
    progress_events = []

    result = pretrain_vicreg(
        paths,
        params=_tiny_vicreg_params(),
        progress_callback=progress_events.append,
    )

    assert weights_path.exists()
    assert meta_path.exists()
    assert result.tile_count == len(paths)
    assert len(progress_events) == 1
    meta = json.loads(meta_path.read_text())
    assert "categories" not in meta
    assert meta["tile_count"] == len(paths)
    assert "pairing" in meta  # what marks it as a backbone checkpoint

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

    paths = _make_paths(tmp_path, n=16)
    pretrain_vicreg(paths, params=_tiny_vicreg_params())

    meta = json.loads(meta_path.read_text())
    # Grouped by container path -> sorted cell labels -- see
    # tile_container.group_training_provenance.
    trained_on = meta["trained_on"]
    assert set(trained_on) == {str(tmp_path / "fov.tiles")}
    assert len(trained_on[str(tmp_path / "fov.tiles")]) == len(paths)
    assert meta["started_from"] == "imagenet"


def test_pretrain_vicreg_continues_from_given_backbone(tmp_path, monkeypatch):
    """With `epochs=0` no gradient step ever runs, so whatever the
    backbone was initialized from is exactly what gets saved -- letting
    this compare state dicts directly."""
    import torch

    weights_dir = tmp_path / "vicreg_weights"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_dir / "backbone.pth")
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_META_PATH", weights_dir / "meta.json")

    seed_path = tmp_path / "seed_backbone.pth"
    seed_backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
    torch.save(seed_backbone.state_dict(), seed_path)

    pretrain_vicreg(
        _make_paths(tmp_path, n=16),
        params=VICRegParams(epochs=0, batch_size=4, num_workers=0),
        backbone_weights_path=seed_path,
    )

    saved = torch.load(weights_dir / "backbone.pth", map_location="cpu")
    for key, value in seed_backbone.state_dict().items():
        assert torch.equal(value, saved[key])
    meta = json.loads((weights_dir / "meta.json").read_text())
    assert meta["started_from"] == str(seed_path)


def test_pretrain_vicreg_without_backbone_ignores_live_slot(tmp_path, monkeypatch):
    import torch

    weights_dir = tmp_path / "vicreg_weights"
    weights_path = weights_dir / "backbone.pth"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_path)
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_META_PATH", weights_dir / "meta.json")

    weights_dir.mkdir()
    seed_backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
    torch.save(seed_backbone.state_dict(), weights_path)

    pretrain_vicreg(
        _make_paths(tmp_path, n=16), params=VICRegParams(epochs=0, batch_size=4, num_workers=0)
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

    trained_paths = _make_paths(tmp_path, n=16)
    pretrain_vicreg(trained_paths, params=_tiny_vicreg_params())

    # A second, never-trained-on container stands in for "newly pooled
    # crops this backbone hasn't seen" -- a real probe is always some
    # other container ref, never a path outside the container format.
    unseen_container = tmp_path / "unseen.tiles"
    write_container(
        unseen_container,
        [(f"unseen_cell{i:05d}", i, _synthetic_crop(100 + i)) for i in range(3)],
    )
    unseen_paths = [f"{unseen_container}/unseen_cell{i:05d}" for i in range(3)]

    probe_paths = trained_paths[:5] + unseen_paths
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
    paths = _make_paths(tmp_path, n=16)

    result = pretrain_vicreg(paths, params=_tiny_vicreg_params(), output_dir=session_dir)

    assert (session_dir / "backbone.pth").exists()
    assert (session_dir / "meta.json").exists()
    assert result.weights_path == session_dir / "backbone.pth"
    assert not live_dir.exists()  # yeastprep-driven runs never touch the live slot


def test_pretrain_vicreg_rejects_too_little_data(tmp_path):
    path = tmp_path / "only_one.tif"
    _write_synthetic_crop(path, seed=0)

    with pytest.raises(ValueError):
        pretrain_vicreg([str(path)], params=_tiny_vicreg_params())


def test_pretrain_vicreg_honors_cancel_check(tmp_path, monkeypatch):
    weights_dir = tmp_path / "vicreg_weights"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(
        "tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_dir / "backbone.pth"
    )
    monkeypatch.setattr(
        "tileclass.training.vicreg.VICREG_META_PATH", weights_dir / "meta.json"
    )

    with pytest.raises(TrainingCancelled):
        pretrain_vicreg(
            _make_paths(tmp_path, n=16),
            params=_tiny_vicreg_params(),
            cancel_check=lambda: True,
        )
    assert not (weights_dir / "backbone.pth").exists()


def test_pretrain_vicreg_cancel_after_progress_saves_best_weights(tmp_path, monkeypatch):
    """Cancelling after at least one epoch has finished shouldn't discard
    the run -- it should save the lowest-loss epoch reached so far, same as
    a completed run, and flag the result as cancelled."""
    weights_dir = tmp_path / "vicreg_weights"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(
        "tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_dir / "backbone.pth"
    )
    monkeypatch.setattr(
        "tileclass.training.vicreg.VICREG_META_PATH", weights_dir / "meta.json"
    )

    paths = _make_paths(tmp_path, n=16)

    calls = {"n": 0}

    def cancel_after_a_few():
        calls["n"] += 1
        return calls["n"] > 2

    result = pretrain_vicreg(
        paths,
        params=VICRegParams(epochs=5, batch_size=4, num_workers=0),
        cancel_check=cancel_after_a_few,
    )

    assert result.cancelled is True
    assert (weights_dir / "backbone.pth").exists()
    assert (weights_dir / "meta.json").exists()


def _redirect_live_slot(tmp_path, monkeypatch):
    weights_dir = tmp_path / "vicreg_weights"
    monkeypatch.setattr("tileclass.training.vicreg.VICREG_WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(
        "tileclass.training.vicreg.VICREG_WEIGHTS_PATH", weights_dir / "backbone.pth"
    )
    monkeypatch.setattr(
        "tileclass.training.vicreg.VICREG_META_PATH", weights_dir / "meta.json"
    )
    return weights_dir


def test_pretrain_vicreg_selects_epoch_by_knn(tmp_path, monkeypatch):
    """With annotated tiles given, the kNN score is reported on every
    `eval_every`-th epoch and the last, and the best-scoring epoch is the
    one saved -- recorded in the result and meta.json."""
    weights_dir = _redirect_live_slot(tmp_path, monkeypatch)
    records = _make_records(tmp_path, n_per_class=8)
    paths = [p for p, _ in records]
    # Deterministic scores so the test pins which epoch is chosen.
    scores = iter([0.5, 0.9, 0.7])
    monkeypatch.setattr(
        "tileclass.training.vicreg.knn_accuracy", lambda emb, labels: next(scores)
    )
    progress = []

    result = pretrain_vicreg(
        paths,
        params=VICRegParams(epochs=5, batch_size=4, num_workers=0, eval_every=2),
        progress_callback=progress.append,
        eval_records=records,
    )

    assert [p.knn_accuracy for p in progress] == [None, 0.5, None, 0.9, 0.7]
    assert all(p.lr > 0 for p in progress[:-1])
    assert result.selected_by == "knn"
    assert result.selected_epoch == 4
    assert result.knn_accuracy == 0.9
    meta = json.loads((weights_dir / "meta.json").read_text())
    assert meta["selection"]["selected_epoch"] == 4
    assert meta["selection"]["knn_history"] == [[2, 0.5], [4, 0.9], [5, 0.7]]
    assert meta["params"]["warmup_epochs"] == VICRegParams().warmup_epochs


def test_pretrain_vicreg_without_eval_records_saves_last_epoch(tmp_path, monkeypatch):
    weights_dir = _redirect_live_slot(tmp_path, monkeypatch)
    progress = []

    result = pretrain_vicreg(
        _make_paths(tmp_path, n=8),
        params=VICRegParams(epochs=2, batch_size=4, num_workers=0),
        progress_callback=progress.append,
    )

    assert all(p.knn_accuracy is None for p in progress)
    assert result.selected_by == "last" and result.selected_epoch == 2
    meta = json.loads((weights_dir / "meta.json").read_text())
    assert meta["selection"]["knn_history"] == []


def test_pretrain_vicreg_eval_every_zero_disables_knn(tmp_path, monkeypatch):
    _redirect_live_slot(tmp_path, monkeypatch)
    records = _make_records(tmp_path, n_per_class=4)

    def fail(*_a, **_k):
        raise AssertionError("kNN should not run with eval_every=0")

    monkeypatch.setattr("tileclass.training.vicreg.knn_accuracy", fail)
    result = pretrain_vicreg(
        [p for p, _ in records],
        params=VICRegParams(epochs=1, batch_size=4, num_workers=0, eval_every=0),
        eval_records=records,
    )
    assert result.selected_by == "last"


# --- embeddings ------------------------------------------------------------


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


def test_knn_accuracy_separates_well_separated_clusters():
    embeddings, labels, _ = _separable_embeddings()
    acc = knn_accuracy(embeddings, labels, k=5)
    assert acc > 0.95


def test_umap_2d_shape():
    embeddings, _, _ = _separable_embeddings(n_per_class=20)
    coords = umap_2d(embeddings)
    assert coords.shape == (embeddings.shape[0], 2)


def test_umap_2d_reproducible_with_same_seed():
    embeddings, _, _ = _separable_embeddings(n_per_class=20)
    first = umap_2d(embeddings, seed=0)
    second = umap_2d(embeddings, seed=0)
    np.testing.assert_array_equal(first, second)


def test_umap_2d_handles_tiny_sample_count():
    """A handful of points, well below the requested n_neighbors -- the
    clamp in `umap_2d` should keep UMAP running rather than requiring the
    caller to pick a value that fits."""
    embeddings, _, _ = _separable_embeddings(n_per_class=1, num_classes=4)
    coords = umap_2d(embeddings, n_neighbors=15, metric="cosine")
    assert coords.shape == (embeddings.shape[0], 2)
