"""Tests for tileclass.checkpoint_import.import_checkpoint -- pure
filesystem logic, no Qt/torch needed. Matches the project's convention of
exercising real files on tmp_path rather than mocking."""

from pathlib import Path

import pytest

from tileclass.checkpoint_import import import_checkpoint


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_import_checkpoint_copies_weights_and_meta(tmp_path):
    weights_src = tmp_path / "src" / "backbone.pth"
    meta_src = tmp_path / "src" / "meta.json"
    _write(weights_src, "weights-v1")
    _write(meta_src, '{"categories": ["a", "b"]}')

    weights_dest = tmp_path / "dest" / "weights.pth"
    meta_dest = tmp_path / "dest" / "meta.json"

    import_checkpoint(weights_src, meta_src, weights_dest, meta_dest)

    assert weights_dest.read_text() == "weights-v1"
    assert meta_dest.read_text() == '{"categories": ["a", "b"]}'


def test_import_checkpoint_backs_up_existing_dest_before_overwriting(tmp_path):
    weights_src = tmp_path / "src" / "backbone.pth"
    meta_src = tmp_path / "src" / "meta.json"
    _write(weights_src, "weights-v2")
    _write(meta_src, '{"categories": ["c"]}')

    weights_dest = tmp_path / "dest" / "weights.pth"
    meta_dest = tmp_path / "dest" / "meta.json"
    _write(weights_dest, "weights-v1")
    _write(meta_dest, '{"categories": ["a", "b"]}')

    import_checkpoint(weights_src, meta_src, weights_dest, meta_dest)

    assert weights_dest.read_text() == "weights-v2"
    assert meta_dest.read_text() == '{"categories": ["c"]}'

    weights_backups = list(weights_dest.parent.glob("weights.pth.bak-*"))
    meta_backups = list(meta_dest.parent.glob("meta.json.bak-*"))
    assert len(weights_backups) == 1
    assert len(meta_backups) == 1
    assert weights_backups[0].read_text() == "weights-v1"
    assert meta_backups[0].read_text() == '{"categories": ["a", "b"]}'


def test_import_checkpoint_creates_dest_dir_if_missing(tmp_path):
    weights_src = tmp_path / "src" / "backbone.pth"
    meta_src = tmp_path / "src" / "meta.json"
    _write(weights_src, "weights")
    _write(meta_src, "{}")

    weights_dest = tmp_path / "nested" / "dest" / "weights.pth"
    meta_dest = tmp_path / "nested" / "dest" / "meta.json"

    import_checkpoint(weights_src, meta_src, weights_dest, meta_dest)

    assert weights_dest.exists()
    assert meta_dest.exists()


def test_import_checkpoint_raises_for_missing_weights_source(tmp_path):
    meta_src = tmp_path / "src" / "meta.json"
    _write(meta_src, "{}")

    with pytest.raises(FileNotFoundError):
        import_checkpoint(
            tmp_path / "src" / "missing.pth",
            meta_src,
            tmp_path / "dest" / "weights.pth",
            tmp_path / "dest" / "meta.json",
        )


def test_import_checkpoint_raises_for_missing_meta_source(tmp_path):
    weights_src = tmp_path / "src" / "backbone.pth"
    _write(weights_src, "weights")

    with pytest.raises(FileNotFoundError):
        import_checkpoint(
            weights_src,
            tmp_path / "src" / "missing.json",
            tmp_path / "dest" / "weights.pth",
            tmp_path / "dest" / "meta.json",
        )
