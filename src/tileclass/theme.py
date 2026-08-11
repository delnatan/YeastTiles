"""App-wide dark theme.

The thumbnail grid canvas already paints itself dark unconditionally
(``widgets/thumbnail_grid.py``'s ``paintEvent``) -- everything else
(menus, dialogs, docks, buttons) otherwise falls back to the OS default
palette. ``Fusion`` is used because it's the one Qt style that fully
honors a custom ``QPalette`` cross-platform; native styles (macOS's in
particular) ignore most palette roles.
"""

from qtpy.QtGui import QColor, QPalette


def apply_dark_theme(app):
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#252526"))
    palette.setColor(QPalette.WindowText, QColor("#dcdcdc"))
    palette.setColor(QPalette.Base, QColor("#1e1e1e"))
    palette.setColor(QPalette.AlternateBase, QColor("#2d2d2d"))
    palette.setColor(QPalette.ToolTipBase, QColor("#2d2d2d"))
    palette.setColor(QPalette.ToolTipText, QColor("#dcdcdc"))
    palette.setColor(QPalette.Text, QColor("#dcdcdc"))
    palette.setColor(QPalette.Button, QColor("#2d2d2d"))
    palette.setColor(QPalette.ButtonText, QColor("#dcdcdc"))
    palette.setColor(QPalette.BrightText, QColor("#ff5c5c"))
    palette.setColor(QPalette.Link, QColor("#5aa0ff"))
    palette.setColor(QPalette.Highlight, QColor("#3d82c8"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Light, QColor("#3c3c3c"))
    palette.setColor(QPalette.Midlight, QColor("#323232"))
    palette.setColor(QPalette.Mid, QColor("#232323"))
    palette.setColor(QPalette.Dark, QColor("#141414"))
    palette.setColor(QPalette.Shadow, QColor("#000000"))
    palette.setColor(QPalette.PlaceholderText, QColor("#8c8c8c"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#777777"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#777777"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#777777"))
    app.setPalette(palette)

    # Fusion honors the palette almost everywhere, but a few widgets need
    # an explicit nudge to stop looking like a seam in an otherwise flat
    # theme (menu/dock borders, scrollbar handles).
    app.setStyleSheet(
        """
        QToolTip {
            color: #dcdcdc;
            background-color: #2d2d2d;
            border: 1px solid #555555;
        }
        QMenu {
            background-color: #2d2d2d;
            border: 1px solid #454545;
        }
        QMenu::item:selected {
            background-color: #3d82c8;
        }
        QMenuBar::item:selected {
            background-color: #3d82c8;
        }
        QDockWidget::title {
            background-color: #2d2d2d;
            padding: 4px;
            border-bottom: 1px solid #454545;
        }
        QScrollBar:vertical {
            background: #1e1e1e;
            width: 12px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #4a4a4a;
            min-height: 24px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #5a5a5a;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        """
    )
