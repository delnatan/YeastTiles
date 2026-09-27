"""Main window: a paged, annotatable grid of image tiles.

Every command is a QAction with its shortcut, listed in the View/Annotate
menus -- there's no separate keyPressEvent handling. Number keys 1-9 apply
the first nine categories (in vocabulary order, shown in the legend under
the toolbar) to the selected tiles.
"""


from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QImage, QKeySequence, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .classifiers import CLASSIFIERS
from .data.channel_state import ChannelStateList, default_channel_state
from .data.palette import category_color, readable_text_color
from .data.pooled_annotations import PooledAnnotations
from .tile_container import tile_display_name
from .widgets.annotation_stats_panel import AnnotationStatsPanel
from .widgets.manage_categories_dialog import ManageCategoriesDialog
from .widgets.thumbnail_colors_panel import ThumbnailColorsPanel
from .widgets.thumbnail_grid import ThumbnailGridWidget

TILE_SIZE_RANGE = (50, 400)
TILE_SIZE_STEP = 25
TILES_PER_PAGE_MAX = 300
NUMBER_KEY_CATEGORIES = 9


class MainWindow(QMainWindow):
    def __init__(self, folders, image_paths, tiles_per_page=96, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.image_paths = image_paths
        self.tiles_per_page = tiles_per_page
        self.current_page = 0
        self.tile_size = 80

        # None = show all, "" = un-annotated only, else exact category name.
        self._category_filter = None

        self.max_C = 1

        # One TileAnnotations per pooled container (see
        # data/pooled_annotations.py) -- each keeps its own sidecar file.
        self.annotations = PooledAnnotations(folders)
        self._category_colors = {}
        self._recompute_category_colors()

        self._classifier_instances = {}  # name -> loaded TileClassifier

        # Per-channel display state. Its colors are persisted to the
        # annotation sidecar on every change, and seeded from it in
        # _update_channel_state as channels are discovered (incrementally,
        # as tiles decode) -- from a frozen snapshot, since seeding itself
        # writes back through _on_channel_state_changed.
        self._pending_channel_colors = dict(self.annotations.channel_colors)
        self.channel_state = ChannelStateList(0)
        self.channel_state.subscribe(self._on_channel_state_changed)
        self._colors_seeded_channels = set()

        self.colors_panel = None
        self._colors_dock = None
        self.annotation_stats_panel = None
        self._annotation_stats_dock = None

        self._setup_ui()
        self._setup_menu()
        self._refresh_categories()
        self._load_current_page()

    # ------------------------------------------------------------------
    # Annotation badges / categories
    # ------------------------------------------------------------------

    def _category_color(self, category):
        """Hex color for category, assigned in first-seen order and
        kept stable until the vocabulary changes."""
        if category not in self._category_colors:
            index = len(self._category_colors)
            self._category_colors[category] = category_color(index)
        return self._category_colors[category]

    def _recompute_category_colors(self):
        self._category_colors = {}
        for category in self.annotations.categories():
            self._category_color(category)

    def _refresh_categories(self):
        """Vocabulary may have changed: recolor, and rebuild the legend and
        the filter dropdown."""
        self._recompute_category_colors()
        self._refresh_legend()
        self._refresh_category_filter_combo()

    def _refresh_legend(self):
        chips = []
        for i, name in enumerate(self.annotations.categories()):
            color = self._category_color(name)
            key = f"{i + 1} " if i < NUMBER_KEY_CATEGORIES else ""
            chips.append(
                f'<span style="background-color:{color}; color:{readable_text_color(color)};">'
                f"&nbsp;{key}{name}&nbsp;</span>"
            )
        if chips:
            text = "&nbsp; ".join(chips)
            text += (
                '&nbsp;&nbsp; <span style="color:#888;">T: pick &nbsp; '
                "A: accept AI &nbsp; Del: clear</span>"
            )
        else:
            text = '<span style="color:#888;">No categories yet -- Annotate &gt; Manage Categories...</span>'
        self.legend_label.setText(text)

    def _refresh_badges(self):
        """Rebuild the path -> (category, color, confidence) map and push
        it to the grid. Cheap: dict lookups only, no file I/O."""
        badges = {}
        for path in self.image_paths:
            relpath = self.annotations.relpath(path)
            category = self.annotations.get(relpath)
            if category:
                confidence = self.annotations.confidence(relpath)
                badges[path] = (category, self._category_color(category), confidence)
        self.thumbnail_grid.set_annotations(badges)

    def _after_annotations_changed(self):
        if self._category_filter is not None:
            # Relabeled tiles may no longer match the filter.
            self._load_current_page()
        else:
            self._refresh_badges()
        self._update_status()
        self._refresh_annotation_stats()
        self._refresh_categories()

    def _update_channel_state(self):
        """Channel count can only grow as more images finish decoding in
        the background."""
        self.max_C = max(self.max_C, self.thumbnail_grid.max_channels())
        self.channel_state.resize(self.max_C)
        self._seed_channel_colors()

        if self.colors_panel is not None and self._colors_dock.isVisible():
            self.colors_panel.refresh_ui()

    def _seed_channel_colors(self):
        """Apply any persisted per-channel color/blend-mode/opacity
        overrides (from the annotation sidecar's #channel_colors line) to
        newly-available channel indices, once each."""
        for idx, (color_hex, blend_mode, opacity) in self._pending_channel_colors.items():
            if idx in self._colors_seeded_channels or idx >= len(self.channel_state):
                continue
            self._colors_seeded_channels.add(idx)
            self.channel_state.set(
                idx, color_hex=color_hex, blend_mode=blend_mode, opacity=opacity
            )

    def _on_channel_state_changed(self, _channel_idx):
        # Sparse: only channels whose colors differ from their default get
        # an entry, so merely opening a folder never touches the sidecar
        # file unless the user actually changes a color.
        # Channels not decoded yet keep their persisted entry.
        n_channels = len(self.channel_state)
        colors = {
            idx: value for idx, value in self._pending_channel_colors.items() if idx >= n_channels
        }
        colors.update(
            (c, self.channel_state[c].colors)
            for c in range(n_channels)
            if self.channel_state[c].colors != default_channel_state(c).colors
        )
        if colors == self.annotations.channel_colors:
            return
        self.annotations.set_channel_colors(colors)

    def show_colors_panel(self):
        if self.colors_panel is None:
            panel = ThumbnailColorsPanel(self)
            dock = QDockWidget("Tile Colors", self)
            dock.setObjectName("colors_dock")
            dock.setWidget(panel)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.colors_panel = panel
            self._colors_dock = dock
        self._colors_dock.show()
        self._colors_dock.raise_()
        self.colors_panel.refresh_ui()

    def _on_selection_changed(self, paths):
        self._update_status()

    def _on_thumbnail_decoded(self, path):
        self._update_channel_state()

    def _view_full_size(self, path):
        """Full-resolution zoom popup for per-pixel inspection."""
        rgb = self.thumbnail_grid.full_resolution_rgb(path)
        if rgb is None:
            return
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()

        dlg = QDialog(self)
        dlg.setWindowTitle(tile_display_name(path))
        layout = QVBoxLayout(dlg)
        label = QLabel()
        label.setPixmap(QPixmap.fromImage(qimg))
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(w < 800 and h < 800)
        layout.addWidget(scroll)
        dlg.resize(min(w, 900), min(h, 900))
        dlg.exec_()

    def show_manage_categories_dialog(self):
        filter_before = self._category_filter
        dlg = ManageCategoriesDialog(self.annotations, parent=self)
        dlg.exec_()
        # Colors are assigned in vocabulary order, so a rename/delete/add
        # can shift them.
        self._refresh_categories()
        self._refresh_badges()
        self._refresh_annotation_stats()
        if filter_before is not None and self._category_filter is None:
            self._load_current_page()

    def show_annotation_stats(self):
        if self.annotation_stats_panel is None:
            panel = AnnotationStatsPanel(self)
            dock = QDockWidget("Annotation Stats", self)
            dock.setObjectName("annotation_stats_dock")
            dock.setWidget(panel)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.annotation_stats_panel = panel
            self._annotation_stats_dock = dock
        self._annotation_stats_dock.show()
        self._annotation_stats_dock.raise_()
        self._refresh_annotation_stats()

    def _refresh_annotation_stats(self):
        if (
            self.annotation_stats_panel is not None
            and self._annotation_stats_dock.isVisible()
        ):
            self.annotation_stats_panel.refresh(self.annotations, self.image_paths)

    # ------------------------------------------------------------------
    # Manual annotation
    # ------------------------------------------------------------------

    def _annotate_selection(self, category):
        """Tag every selected tile with `category` ("" clears). The single
        path every manual labeling action goes through -- it always drops
        any AI confidence, marking the tag as human-set."""
        paths = self.thumbnail_grid.selected_paths
        if not paths:
            return
        self.annotations.update((self.annotations.relpath(p), category) for p in paths)
        self._after_annotations_changed()

    def annotate_with_number(self, number):
        """Number key N applies the N-th vocabulary category."""
        categories = self.annotations.categories()
        if number <= len(categories):
            self._annotate_selection(categories[number - 1])

    def annotate_selected_tiles(self):
        """Pick a category from the vocabulary for the selected tiles --
        no free text, so a typo can't create a near-duplicate category."""
        paths = self.thumbnail_grid.selected_paths
        if not paths:
            return
        if not self.annotations.categories():
            QMessageBox.information(
                self,
                "Annotate",
                "No categories defined yet. Use Annotate > Manage "
                "Categories... to add some first.",
            )
            return

        items = [""] + self.annotations.categories()
        current = (
            self.annotations.get(self.annotations.relpath(paths[0])) or ""
            if len(paths) == 1
            else ""
        )
        if current not in items:
            items.append(current)
        if len(paths) == 1:
            prompt = f"Category for {tile_display_name(paths[0])}:"
        else:
            prompt = f"Category for {len(paths)} tiles:"

        text, ok = QInputDialog.getItem(
            self, "Annotate", prompt, items, items.index(current), editable=False
        )
        if ok:
            self._annotate_selection(text.strip())

    def clear_selected_annotations(self):
        self._annotate_selection("")

    def accept_predictions(self):
        """Confirm the AI prediction on every selected tile that has one,
        keeping its category but marking it human-reviewed."""
        updates = []
        for path in self.thumbnail_grid.selected_paths:
            relpath = self.annotations.relpath(path)
            if self.annotations.confidence(relpath) is not None:
                updates.append((relpath, self.annotations.get(relpath)))
        if not updates:
            return
        self.annotations.update(updates)
        self._after_annotations_changed()

    def select_all(self):
        self.thumbnail_grid.select_all()

    # ------------------------------------------------------------------
    # AI predictions
    # ------------------------------------------------------------------

    def _get_classifier(self, name):
        """Lazily instantiate (and cache) the named registered classifier.

        Instantiating is what actually imports torch/torchvision (see
        ``classifiers/yeast_efficientnet.py``) -- so this is also where a
        missing ``classification`` extra surfaces, as a dialog rather than
        a crash.
        """
        if name not in self._classifier_instances:
            try:
                self._classifier_instances[name] = CLASSIFIERS[name]()
            except ImportError as exc:
                QMessageBox.warning(
                    self,
                    "Auto-Annotate",
                    "This classifier needs extra packages that aren't "
                    f"installed:\n{exc}\n\nInstall with:\n"
                    "  uv sync --extra classification",
                )
                return None
        return self._classifier_instances[name]

    def auto_annotate_page(self):
        """Run a classifier over the *current page only* and tag every
        tile that isn't already annotated (existing tags -- manual or a
        prior AI pass -- are left untouched; use Clear AI Predictions
        first to re-predict). Tags carry the model's confidence, so the
        grid badges them as unconfirmed until a human accepts or relabels
        them."""
        paths = self.thumbnail_grid.paths
        if not paths:
            return

        if not CLASSIFIERS:
            QMessageBox.information(
                self, "Auto-Annotate", "No classifiers are registered."
            )
            return

        if len(CLASSIFIERS) == 1:
            name = next(iter(CLASSIFIERS))
        else:
            name, ok = QInputDialog.getItem(
                self,
                "Auto-Annotate",
                "Classifier:",
                list(CLASSIFIERS),
                0,
                editable=False,
            )
            if not ok:
                return

        classifier = self._get_classifier(name)
        if classifier is None:
            return

        for category in classifier.categories:
            self.annotations.add_category(category)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            predictions = classifier.predict(paths)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Auto-Annotate", f"Classification failed:\n{exc}")
            return
        QApplication.restoreOverrideCursor()

        updates = []
        skipped = 0
        for path, (label, confidence) in zip(paths, predictions):
            relpath = self.annotations.relpath(path)
            if self.annotations.get(relpath):
                skipped += 1
                continue
            updates.append((relpath, label, confidence))

        self.annotations.update_with_confidence(updates)
        self._after_annotations_changed()

        message = f"Tagged {len(updates)} of {len(paths)} tiles on this page"
        message += f" ({skipped} already annotated, left untouched)." if skipped else "."
        QMessageBox.information(self, "Auto-Annotate", message)

    def clear_ai_predictions(self):
        """Drop every unconfirmed AI prediction across this window's
        containers (human-set tags are kept), so a newer model can
        re-predict those tiles."""
        n_predictions = sum(
            1 for _key, _category, confidence in self.annotations.tagged_items()
            if confidence is not None
        )
        if not n_predictions:
            QMessageBox.information(
                self, "Clear AI Predictions", "There are no unconfirmed AI predictions."
            )
            return
        answer = QMessageBox.question(
            self,
            "Clear AI Predictions",
            f"Remove {n_predictions} unconfirmed AI prediction(s) from "
            f"{self.annotations.label()}? Human-set and accepted annotations are kept.",
        )
        if answer != QMessageBox.Yes:
            return
        self.annotations.clear_unconfirmed()
        self._after_annotations_changed()

    # ------------------------------------------------------------------

    def _refresh_category_filter_combo(self):
        current = self._category_filter
        combo = self.category_filter_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All", None)
        combo.addItem("Un-annotated", "")
        names = list(self.annotations.categories())
        for name in sorted(set(self.annotations.values())):
            if name not in names:
                names.append(name)
        for name in names:
            combo.addItem(name, name)
        idx = combo.findData(current)
        if idx < 0:
            idx = 0
            self._category_filter = None
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _on_category_filter_changed(self, index):
        self._category_filter = self.category_filter_combo.itemData(index)
        self.current_page = 0
        self._load_current_page()

    def sort_by_annotation(self):
        """Rearrange the current page's tiles so tiles sharing the same
        category -- including un-annotated, grouped first -- sit in one
        contiguous block, and within each block AI-predicted tiles sort by
        ascending confidence with human-set tiles pushed to the end -- so
        the network's least confident calls for a given category surface
        first and its human-reviewed ground truth is easy to compare
        against at a glance."""
        paths = self.thumbnail_grid.paths
        if not paths:
            return

        def path_sort_key(path):
            relpath = self.annotations.relpath(path)
            category = self.annotations.get(relpath) or ""
            confidence = self.annotations.confidence(relpath)
            confidence_key = (1, 0.0) if confidence is None else (0, confidence)
            return (category, confidence_key)

        new_order = sorted(paths, key=path_sort_key)
        if new_order != paths:
            self.thumbnail_grid.set_order(new_order)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        self.setWindowTitle(
            f"tileclass - {self.annotations.label()} - {len(self.image_paths)} tiles"
        )
        self.resize(1000, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedWidth(30)
        self.prev_btn.clicked.connect(self._prev_page)
        toolbar_layout.addWidget(self.prev_btn)

        self.page_label = QLabel("Page 1 / 1")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setFixedWidth(100)
        toolbar_layout.addWidget(self.page_label)

        self.next_btn = QPushButton(">")
        self.next_btn.setFixedWidth(30)
        self.next_btn.clicked.connect(self._next_page)
        toolbar_layout.addWidget(self.next_btn)

        toolbar_layout.addSpacing(20)

        toolbar_layout.addWidget(QLabel("Tiles/page:"))
        self.per_page_spin = QSpinBox()
        # Callers like the embedding-lasso viewer pass one page of however
        # many tiles were selected.
        self.per_page_spin.setRange(1, max(TILES_PER_PAGE_MAX, self.tiles_per_page))
        self.per_page_spin.setValue(self.tiles_per_page)
        self.per_page_spin.setFixedWidth(70)
        self.per_page_spin.setKeyboardTracking(False)
        self.per_page_spin.valueChanged.connect(self._on_per_page_changed)
        toolbar_layout.addWidget(self.per_page_spin)

        toolbar_layout.addSpacing(20)

        toolbar_layout.addWidget(QLabel("Tile size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(*TILE_SIZE_RANGE)
        self.size_slider.setValue(self.tile_size)
        self.size_slider.setFixedWidth(150)
        self.size_slider.valueChanged.connect(self._on_tile_size_changed)
        toolbar_layout.addWidget(self.size_slider)

        self.size_label = QLabel(f"{self.tile_size}px")
        self.size_label.setFixedWidth(50)
        toolbar_layout.addWidget(self.size_label)

        toolbar_layout.addSpacing(20)

        toolbar_layout.addWidget(QLabel("Filter:"))
        self.category_filter_combo = QComboBox()
        self.category_filter_combo.setMinimumWidth(120)
        self.category_filter_combo.currentIndexChanged.connect(
            self._on_category_filter_changed
        )
        toolbar_layout.addWidget(self.category_filter_combo)

        toolbar_layout.addStretch()
        main_layout.addWidget(toolbar)

        self.legend_label = QLabel()
        self.legend_label.setTextFormat(Qt.RichText)
        self.legend_label.setWordWrap(True)
        self.legend_label.setContentsMargins(5, 0, 5, 0)
        main_layout.addWidget(self.legend_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.thumbnail_grid = ThumbnailGridWidget()
        self.thumbnail_grid.set_channel_state(self.channel_state)
        self.thumbnail_grid.set_tile_size(self.tile_size)
        self.thumbnail_grid.set_show_info(False)
        self.thumbnail_grid.selectionChanged.connect(self._on_selection_changed)
        self.thumbnail_grid.annotateRequested.connect(self.annotate_selected_tiles)
        self.thumbnail_grid.viewFullSizeRequested.connect(self._view_full_size)
        self.thumbnail_grid.decoded.connect(self._on_thumbnail_decoded)
        self.scroll_area.setWidget(self.thumbnail_grid)
        main_layout.addWidget(self.scroll_area, 1)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        self.status_label.setToolTip(self.annotations.root_dir)
        main_layout.addWidget(self.status_label)

        self._update_page_controls()

    def _add_action(self, menu, text, slot, shortcuts=()):
        action = QAction(text, self)
        action.setShortcuts([QKeySequence(s) for s in shortcuts])
        action.triggered.connect(slot)
        if menu is None:
            self.addAction(action)
        else:
            menu.addAction(action)
        return action

    def _setup_menu(self):
        view_menu = self.menuBar().addMenu("&View")
        self._add_action(view_menu, "Previous Page", self._prev_page, ["Left", "PgUp"])
        self._add_action(view_menu, "Next Page", self._next_page, ["Right", "PgDown"])
        self._add_action(view_menu, "First Page", self._first_page, ["Home"])
        self._add_action(view_menu, "Last Page", self._last_page, ["End"])
        view_menu.addSeparator()
        self._add_action(
            view_menu, "Larger Tiles", lambda: self._step_tile_size(+1), ["+", "="]
        )
        self._add_action(view_menu, "Smaller Tiles", lambda: self._step_tile_size(-1), ["-"])
        show_names = self._add_action(
            view_menu, "Show Cell Numbers", self.thumbnail_grid.set_show_info, ["I"]
        )
        show_names.setCheckable(True)
        view_menu.addSeparator()
        self._add_action(view_menu, "Colors...", self.show_colors_panel)
        self._add_action(view_menu, "Auto Contrast All", self._auto_contrast_all, ["C"])

        annotate_menu = self.menuBar().addMenu("&Annotate")
        self._add_action(annotate_menu, "Annotate Selected...", self.annotate_selected_tiles, ["T"])
        self._add_action(annotate_menu, "Accept AI Prediction", self.accept_predictions, ["A"])
        self._add_action(
            annotate_menu,
            "Clear Annotation",
            self.clear_selected_annotations,
            ["Delete", "Backspace"],
        )
        self._add_action(
            annotate_menu, "Select All on Page", self.select_all, [QKeySequence.SelectAll]
        )
        annotate_menu.addSeparator()
        self._add_action(
            annotate_menu, "Auto-Annotate Page (AI)", self.auto_annotate_page, ["Shift+T"]
        )
        self._add_action(annotate_menu, "Clear AI Predictions...", self.clear_ai_predictions)
        self._add_action(annotate_menu, "Group by Category", self.sort_by_annotation, ["G"])
        annotate_menu.addSeparator()
        self._add_action(
            annotate_menu, "Manage Categories...", self.show_manage_categories_dialog
        )
        self._add_action(annotate_menu, "Annotation Stats...", self.show_annotation_stats)

        # Not in a menu: the legend under the toolbar lists them.
        for n in range(1, NUMBER_KEY_CATEGORIES + 1):
            self._add_action(None, f"Category {n}", lambda _=False, n=n: self.annotate_with_number(n), [str(n)])

    # ------------------------------------------------------------------
    # Paging
    # ------------------------------------------------------------------

    def _visible_paths(self):
        """image_paths after the active category filter (None = show all,
        "" = un-annotated only, else the exact category name)."""
        if self._category_filter is None:
            return self.image_paths
        target = self._category_filter
        return [
            p
            for p in self.image_paths
            if (self.annotations.get(self.annotations.relpath(p)) or "") == target
        ]

    def _total_pages(self):
        return max(
            1,
            (len(self._visible_paths()) + self.tiles_per_page - 1)
            // self.tiles_per_page,
        )

    def _update_page_controls(self):
        total = self._total_pages()
        self.page_label.setText(f"Page {self.current_page + 1} / {total}")
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < total - 1)

    def _update_status(self):
        visible = self._visible_paths()
        start = self.current_page * self.tiles_per_page + 1
        filter_suffix = ""
        if self._category_filter is not None:
            label = "Un-annotated" if self._category_filter == "" else self._category_filter
            filter_suffix = f"   |   Filter: {label}"

        n_shown = len(self.thumbnail_grid.paths)
        end = min(start + n_shown - 1, len(visible))
        text = f"Showing {start}-{end} of {len(visible)} tiles{filter_suffix}"
        selected = self.thumbnail_grid.selected_paths
        if len(selected) == 1:
            text += f"   |   Selected: {tile_display_name(selected[0])}"
        elif len(selected) > 1:
            text += f"   |   Selected: {len(selected)} tiles"
        self.status_label.setText(text)

    def _load_current_page(self):
        visible = self._visible_paths()
        total_pages = max(
            1, (len(visible) + self.tiles_per_page - 1) // self.tiles_per_page
        )
        if self.current_page >= total_pages:
            self.current_page = total_pages - 1

        start = self.current_page * self.tiles_per_page
        end = min(start + self.tiles_per_page, len(visible))
        page_paths = visible[start:end]

        self.thumbnail_grid.set_items(page_paths)

        # Decode the current page first, then the rest of the visible
        # (filtered) set in the background so paging around later tends
        # to already be warm. Filtered-out images are never decoded here.
        page_set = set(page_paths)
        priority = page_paths + [p for p in visible if p not in page_set]
        self.thumbnail_grid.set_priority(priority)

        self._refresh_badges()
        self._update_channel_state()
        self._update_page_controls()
        self._update_status()

    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._load_current_page()

    def _next_page(self):
        if self.current_page < self._total_pages() - 1:
            self.current_page += 1
            self._load_current_page()

    def _first_page(self):
        self.current_page = 0
        self._load_current_page()

    def _last_page(self):
        self.current_page = self._total_pages() - 1
        self._load_current_page()

    def _on_per_page_changed(self, value):
        self.tiles_per_page = value
        self.current_page = 0
        self._load_current_page()

    def _on_tile_size_changed(self, value):
        self.tile_size = value
        self.size_label.setText(f"{value}px")
        self.thumbnail_grid.set_tile_size(value)

    def _step_tile_size(self, direction):
        self.size_slider.setValue(self.tile_size + direction * TILE_SIZE_STEP)

    def _auto_contrast_all(self):
        """Revert any Colors-panel contrast overrides to each tile's own
        auto-contrast (computed at decode time)."""
        self.channel_state.reset_clims()

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.thumbnail_grid.close()
        super().closeEvent(event)
