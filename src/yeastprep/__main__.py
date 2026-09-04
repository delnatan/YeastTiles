import argparse
import sys

from qtpy.QtWidgets import QApplication, QMessageBox

from tileclass.theme import apply_dark_theme


def main():
    parser = argparse.ArgumentParser(
        prog="yeastprep",
        description="Yeast image-processing pipeline: Data Reduction, Denoise, "
        "Deconvolve, Segmentation, Tile Generation, and Train Denoiser pages.",
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
        # Could be a missing `prep` package (cellpose/pyvistra/jssl-denoise)
        # or a missing `classification` one (torch/scikit-learn) -- the
        # Classifier Training page (batch auto-annotate/fine-tune across a
        # pooled project) is wired into this window unconditionally, so
        # yeastprep needs both extras to launch at all, unlike `tiled_viewer`
        # which only needs `classification` when a classifier is selected.
        QMessageBox.critical(
            None,
            "yeastprep",
            "yeastprep needs extra packages that aren't installed:\n"
            f"{exc}\n\nInstall with:\n  pip install -e '.[classification,prep]'",
        )
        sys.exit(1)

    window = YeastPrepWindow(initial_input_folder=args.input_folder)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
