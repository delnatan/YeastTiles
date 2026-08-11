#!/usr/bin/env python3
"""Entry point: python yeastprep_app.py [input_folder]

Works without installing the package first -- falls back to putting
src/ on sys.path if yeastprep isn't already importable.
"""

import sys
from pathlib import Path

try:
    import yeastprep  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / "src"))

from yeastprep.__main__ import main

if __name__ == "__main__":
    main()
