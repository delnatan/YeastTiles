"""Launch the tile viewer (`tiled_viewer`) on some FOV containers as a
separate process, so it runs independently of yeastprep."""

import subprocess
import sys
from pathlib import Path


def open_tile_viewer(container_files: list[Path]) -> None:
    cmd = [sys.executable, "-m", "tileclass", *[str(f) for f in container_files]]
    subprocess.Popen(cmd, start_new_session=True)
