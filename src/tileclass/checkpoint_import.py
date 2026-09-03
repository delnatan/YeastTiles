"""Shared "import an externally-produced checkpoint into the app's one
active slot" logic, for both the deployed classifier and the VICReg
backbone. Not a multi-model registry -- there is exactly one active slot per
kind, and this only ever overwrites it, after a timestamped backup (mirrors
training/supervised.py's _save_weights and training/vicreg.py's
_save_backbone, which do the same backup-before-overwrite on every training
run; this is the equivalent for manually restoring/promoting a checkpoint
that wasn't just produced by a training run in this session).
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path


def _backup(dest: Path):
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(dest, dest.with_name(f"{dest.name}.bak-{stamp}"))


def import_checkpoint(weights_src, meta_src, weights_dest: Path, meta_dest: Path):
    """Copy (weights_src, meta_src) onto (weights_dest, meta_dest), backing
    up whatever's currently at each dest path first. Raises
    FileNotFoundError if either source is missing."""
    weights_src, meta_src = Path(weights_src), Path(meta_src)
    if not weights_src.exists():
        raise FileNotFoundError(weights_src)
    if not meta_src.exists():
        raise FileNotFoundError(meta_src)

    weights_dest.parent.mkdir(parents=True, exist_ok=True)
    _backup(weights_dest)
    _backup(meta_dest)
    shutil.copy2(weights_src, weights_dest)
    shutil.copy2(meta_src, meta_dest)
