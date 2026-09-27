import argparse
import sys

from qtpy.QtWidgets import QApplication, QMessageBox

from tileclass.theme import apply_dark_theme


def main():
    parser = argparse.ArgumentParser(
        prog="yeastprep",
        description="Yeast image-processing pipeline (Data Reduction, Denoise, "
        "Deconvolve, Segmentation, Tile Generation) plus classifier training "
        "and pooled tile classification.",
    )
    parser.add_argument(
        "input_folder", nargs="?", default=None, help="Folder of raw stacks to open"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    try:
        from .ui.main_window import YeastPrepWindow
    except ImportError as exc:
        # A missing `classification` package (torch/pyvistra/...): the
        # window needs that extra to launch at all. The `prep` extra's
        # cellpose/jssl-denoise aren't needed here -- without them the
        # Segmentation/Denoise pages just show an "install prep"
        # placeholder (see yeastprep/optional_deps.py).
        QMessageBox.critical(
            None,
            "yeastprep",
            "yeastprep needs extra packages that aren't installed:\n"
            f"{exc}\n\nInstall with:\n  uv sync --extra lite\n"
            "or, on a GPU workstation:\n  uv sync --extra gpu",
        )
        sys.exit(1)

    window = YeastPrepWindow(initial_input_folder=args.input_folder)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
