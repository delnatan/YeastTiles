import argparse
import sys

from qtpy.QtWidgets import QApplication

from tileclass.theme import apply_dark_theme

from .ui.main_window import YeastPrepWindow


def main():
    parser = argparse.ArgumentParser(
        prog="yeastprep",
        description="Yeast image-processing pipeline: Data Reduction (raw stacks -> "
        "flattened focal slices), Segmentation, and Tile Generation pages.",
    )
    parser.add_argument(
        "input_folder", nargs="?", default=None, help="Folder of raw stacks to open"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = YeastPrepWindow(initial_input_folder=args.input_folder)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
