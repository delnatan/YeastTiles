import argparse
import sys

from qtpy.QtWidgets import QApplication, QMessageBox

from .main_window import MainWindow
from .scan import filter_by_fov, scan_container
from .theme import apply_dark_theme


def main():
    parser = argparse.ArgumentParser(
        prog="tileclass",
        description="Tile-grid viewer for classifying packed .tiles containers "
        "(see core/tiles.py's export_tiles). Pass multiple containers to pool "
        "them into one browse/annotate/train session -- each keeps its own "
        "annotation sidecar file.",
    )
    parser.add_argument(
        "input_folders",
        nargs="+",
        help="Packed .tiles container(s) to browse/annotate",
    )
    parser.add_argument(
        "--tiles-per-page", type=int, default=100, help="Tiles per page (default: 100)"
    )
    parser.add_argument(
        "--fov",
        action="append",
        default=None,
        metavar="NAME",
        help="Only show tiles from this FOV (filename prefix before '_cell'). "
        "Repeatable to scope the session to several FOVs at once, e.g. for "
        "splitting annotation work by source file.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    try:
        image_paths = [p for folder in args.input_folders for p in scan_container(folder)]
    except (OSError, ValueError) as exc:
        QMessageBox.warning(None, "tileclass", f"Could not open a tile container: {exc}")
        sys.exit(1)
    if args.fov:
        image_paths = filter_by_fov(image_paths, args.fov)
    if not image_paths:
        QMessageBox.warning(
            None,
            "tileclass",
            "No cells found in:\n" + "\n".join(args.input_folders),
        )
        sys.exit(1)

    window = MainWindow(args.input_folders, image_paths, tiles_per_page=args.tiles_per_page)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
