#!/usr/bin/env python3
"""Entry point: python tiled_viewer.py <input_folder>

Works without installing the package first -- falls back to putting
src/ on sys.path if tileclass isn't already importable.
"""

import sys
from pathlib import Path

try:
    import tileclass  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / "src"))

from tileclass.__main__ import main

if __name__ == "__main__":
    main()
