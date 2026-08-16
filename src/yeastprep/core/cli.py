"""Batch CLI: flatten every raw stack in a folder, non-interactively.

    uv run yeastprep-batch <input_folder> <output_folder> [options]

Restructures notebooks/01_flatten_field.ipynb's batch loop into a proper,
reusable, error-tolerant script (the original loop has no per-file error
handling -- one bad file aborts the whole run).
"""

import argparse
import sys
from pathlib import Path

from .channels import ChannelSelection
from .pipeline import DEFAULT_CHANNELS, FlattenFieldParams, process_and_save
from .project import ProjectPaths


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="yeastprep-batch",
        description="Batch field-flattening: extract a focal-plane brightfield "
        "image and a sum-projected target-channel image for every raw stack "
        "in a folder, saved together as one combined-channel tiff per FOV.",
    )
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument(
        "--pattern", default="*.ims", help="Glob pattern for input files (default: *.ims)"
    )
    parser.add_argument("--num-tiles-y", type=int, default=FlattenFieldParams().num_tiles_y)
    parser.add_argument("--num-tiles-x", type=int, default=FlattenFieldParams().num_tiles_x)
    parser.add_argument(
        "--prominence", type=float, default=FlattenFieldParams().inverted_variance_prominence
    )
    parser.add_argument("--poly-degree", type=int, nargs=2, metavar=("DX", "DY"), default=(2, 2))
    parser.add_argument("--offset-um", type=float, default=FlattenFieldParams().offset_um)
    parser.add_argument("--brightfield-channel", type=int, default=DEFAULT_CHANNELS.brightfield)
    parser.add_argument("--target-channel", type=int, default=DEFAULT_CHANNELS.projection)
    args = parser.parse_args(argv)

    params = FlattenFieldParams(
        num_tiles_y=args.num_tiles_y,
        num_tiles_x=args.num_tiles_x,
        inverted_variance_prominence=args.prominence,
        poly_degree=tuple(args.poly_degree),
        offset_um=args.offset_um,
    )
    channels = ChannelSelection(
        brightfield=args.brightfield_channel, projection=args.target_channel
    )

    files = sorted(args.input_folder.glob(args.pattern))
    if not files:
        print(f"No files matching '{args.pattern}' found under {args.input_folder}")
        return 1

    outdir = ProjectPaths(args.output_folder).reduced

    n_failed = 0
    for path in files:
        result = process_and_save(path, outdir, params, channels)
        if result.success:
            print(f"PASS  {path.name}")
        else:
            n_failed += 1
            print(f"FAIL  {path.name}: {result.error}")

    print(f"\n{len(files) - n_failed}/{len(files)} succeeded.")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
