"""One-off migration CLI: pack an already-exported project's loose
`05_tiles/<fov_id>/*.tif` crops into single `05_tiles/<fov_id>.tiles`
containers (see `tileclass.tile_container`).

    uv run yeastprep-pack-tiles <project_root> [--delete-originals]

Follows `core/cli.py`'s shape (argparse, per-item PASS/FAIL printing, exit
code). Every packed container is read back and verified cell-for-cell
against the original files before anything is touched -- the original
`<fov_id>/` folder is only ever removed if `--delete-originals` is passed
and every one of its cells verified, matching this codebase's general
preference for leaving source data alone unless explicitly told otherwise
(see e.g. core/stages.py's "archived", never deleted, status).
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import tifffile
from tileclass.tile_container import TileContainer, write_container

from .fs_status import list_visible
from .project import ProjectPaths
from .project_scan import natsort_key
from .tiles import tile_index_path


def _cell_label(cell_id: str) -> int:
    """Recover the cellpose mask label from a `export_tiles`-produced
    `<fov_id>_cell<label:05d>` cell_id, without needing tile_index.csv."""
    return int(cell_id.rsplit("_cell", 1)[1])


def pack_fov_dir(fov_dir: Path) -> tuple[Path, int, str | None]:
    """Pack one loose-tif `fov_dir` into a sibling `<fov_id>.tiles`
    container, verify every cell round-trips byte-for-byte, then return
    `(container_path, n_cells, error)` -- `error` is None on success, and
    on failure the container (if partially written) is removed so a
    retried run doesn't mistake it for a good one."""
    fov_id = fov_dir.name
    container_path = fov_dir.parent / f"{fov_id}.tiles"
    tif_paths = sorted(list_visible(fov_dir, "*.tif"), key=natsort_key)
    if not tif_paths:
        return container_path, 0, "no .tif files found"

    originals = {}
    cells = []
    for tif_path in tif_paths:
        cell_id = tif_path.stem
        array = tifffile.imread(tif_path)
        originals[cell_id] = array
        cells.append((cell_id, _cell_label(cell_id), array))

    write_container(container_path, cells)

    container = TileContainer(container_path)
    for cell_id, original in originals.items():
        repacked = container.read(cell_id)
        if not np.array_equal(original, repacked):
            container_path.unlink(missing_ok=True)
            return container_path, 0, f"verification mismatch on {cell_id}"

    return container_path, len(originals), None


def _rewrite_crop_paths(project_root: Path, packed_fov_ids: set[str]) -> None:
    """Point `tile_index.csv`'s `crop_path` column at the new virtual
    container reference for every row belonging to a freshly packed FOV --
    keeps the CSV consistent with `export_tiles`'s convention for anything
    still reading it (e.g. external analysis scripts)."""
    index_path = tile_index_path(ProjectPaths(project_root).tiles)
    if not index_path.exists():
        return
    df = pl.read_csv(index_path)
    if "fov_id" not in df.columns or "cell_id" not in df.columns:
        return
    tiles_dir = ProjectPaths(project_root).tiles
    new_crop_path = pl.when(pl.col("fov_id").is_in(list(packed_fov_ids))).then(
        pl.col("fov_id").map_elements(
            lambda fov_id: str(tiles_dir / f"{fov_id}.tiles"), return_dtype=pl.Utf8
        )
        + "/"
        + pl.col("cell_id")
        + ".tif"
    ).otherwise(pl.col("crop_path"))
    df = df.with_columns(new_crop_path.alias("crop_path"))
    df.write_csv(index_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="yeastprep-pack-tiles",
        description="Pack an already-exported project's loose 05_tiles/<fov_id>/ "
        "tif crops into single <fov_id>.tiles containers.",
    )
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Remove each <fov_id>/ folder after its container verifies "
        "successfully (default: leave originals in place).",
    )
    args = parser.parse_args(argv)

    tiles_dir = ProjectPaths(args.project_root).tiles
    fov_dirs = sorted(p for p in list_visible(tiles_dir) if p.is_dir())
    if not fov_dirs:
        print(f"No 05_tiles/<fov_id>/ folders found under {tiles_dir}")
        return 1

    n_failed = 0
    packed_fov_ids = set()
    bytes_before = bytes_after = 0
    for fov_dir in fov_dirs:
        before = sum(p.stat().st_size for p in list_visible(fov_dir, "*.tif"))
        container_path, n_cells, error = pack_fov_dir(fov_dir)
        if error:
            n_failed += 1
            print(f"FAIL  {fov_dir.name}: {error}")
            continue
        after = container_path.stat().st_size
        bytes_before += before
        bytes_after += after
        packed_fov_ids.add(fov_dir.name)
        print(f"PASS  {fov_dir.name}: {n_cells} cell(s) verified, {before} -> {after} bytes")
        if args.delete_originals:
            shutil.rmtree(fov_dir)

    if packed_fov_ids:
        _rewrite_crop_paths(args.project_root, packed_fov_ids)

    n_ok = len(fov_dirs) - n_failed
    print(f"\n{n_ok}/{len(fov_dirs)} FOV(s) packed.")
    if bytes_before:
        print(f"Total: {bytes_before} -> {bytes_after} bytes ({bytes_after / bytes_before:.1%})")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
