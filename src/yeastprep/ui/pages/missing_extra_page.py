"""Stand-in for a page whose packages aren't installed (e.g. Denoise or
Segmentation on a `classification`-only install -- see
yeastprep/optional_deps.py). Keeps the page's slot in the stage list and
the `progress_changed`/`load_selection`/`shutdown` shape the shell expects,
and just says which extra to install.
"""

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from .page_progress import PageProgress  # noqa: F401 -- shape of progress_changed, never emitted here


class MissingExtraPage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, title: str, missing_packages: list[str], extra: str, parent=None):
        super().__init__(parent)
        label = QLabel(
            f"<h3>{title} isn't available</h3>"
            f"<p>It needs {', '.join(missing_packages)}, which "
            f"{'is' if len(missing_packages) == 1 else 'are'} not installed.</p>"
            f"<p>Install the <b>{extra}</b> extra to enable it:<br>"
            f"<code>pip install 'tileclass[{extra}]'</code></p>"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.addWidget(label)

    def load_selection(self, stage: str, path: str, mode: str = "live"):
        pass

    def shutdown(self):
        pass
