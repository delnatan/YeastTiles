import argparse
import sys

from qtpy.QtWidgets import QApplication, QMessageBox

from .main_window import MainWindow
from .scan import filter_by_fov, scan_folder
from .theme import apply_dark_theme


def main():
    parser = argparse.ArgumentParser(
        prog="tileclass",
        description="Tile-grid viewer for classifying folders of images. Pass "
        "multiple folders to pool them into one browse/annotate/train session -- "
        "each folder keeps its own annotation sidecar file.",
    )
    parser.add_argument(
        "input_folders", nargs="+", help="Folder(s) of images to browse/annotate"
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

    image_paths = [p for folder in args.input_folders for p in scan_folder(folder)]
    if args.fov:
        image_paths = filter_by_fov(image_paths, args.fov)
    if not image_paths:
        QMessageBox.warning(
            None,
            "tileclass",
            "No supported images found under:\n" + "\n".join(args.input_folders),
        )
        sys.exit(1)

    window = MainWindow(args.input_folders, image_paths, tiles_per_page=args.tiles_per_page)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
