import argparse
import sys
import traceback

from qtpy.QtWidgets import QApplication, QMessageBox

from tileclass.theme import apply_dark_theme


def _install_error_dialog():
    """Show an exception that no handler caught in a dialog instead of
    losing the whole session to it: under PyQt6 one escaping from any
    button/signal handler calls qFatal() and the app just vanishes, unless
    `sys.excepthook` has been replaced. The traceback still goes to the
    terminal as before."""
    previous_hook = sys.excepthook
    showing = False

    def hook(exc_type, exc, tb):
        nonlocal showing
        previous_hook(exc_type, exc, tb)
        if showing or issubclass(exc_type, KeyboardInterrupt) or QApplication.instance() is None:
            return
        showing = True  # one dialog at a time, even if showing it raises again
        try:
            box = QMessageBox(
                QMessageBox.Critical,
                "yeastprep",
                f"Unexpected error -- {exc_type.__name__}: {exc}\n\n"
                "yeastprep is still running, but whatever you just did may not "
                "have finished.",
            )
            box.setDetailedText("".join(traceback.format_exception(exc_type, exc, tb)))
            box.exec_()
        finally:
            showing = False

    sys.excepthook = hook


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
    _install_error_dialog()

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
