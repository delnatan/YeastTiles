"""Main window: a paged, annotatable grid of image tiles.

Distilled from pyvistra's ``TiledViewer`` -- specifically just its
"fast mode" path (the pure-Qt ``ThumbnailGridWidget`` compositor for
folders of small images). The vispy per-tile viewer, T/Z/C navigation,
and axis-reordering machinery that ``TiledViewer`` also carries for
large multi-dimensional images don't apply here and aren't ported.
"""

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
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
from .data.overlay_state import OverlayStateList, default_channel_state
from .data.palette import category_color
from .data.pooled_annotations import PooledAnnotations
from .data.visual_state import VisualState
from .load_thumbnail import load_plane
from .widgets.annotation_stats_panel import AnnotationStatsPanel
from .widgets.manage_categories_dialog import ManageCategoriesDialog
from .widgets.thumbnail_colors_panel import ThumbnailColorsPanel
from .widgets.thumbnail_grid import ThumbnailGridWidget
from .widgets.tiled_display_settings_dialog import TiledDisplaySettingsDialog


class MainWindow(QMainWindow):
    def __init__(self, folders, image_paths, tiles_per_page=100, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.image_paths = image_paths
        self.tiles_per_page = tiles_per_page
        self.current_page = 0
        self.tile_size = 200

        # Per-window (not persisted) limits for the tile-size slider and
        # tiles-per-page spinbox -- see show_display_limits_dialog.
        self._tile_size_min = 100
        self._tile_size_max = 400
        self._tiles_per_page_min = 1
        self._tiles_per_page_max = 300

        # None = show all, "" = un-annotated only, else exact category name.
        self._category_filter = None

        self.max_C = 1
        self.show_info = False
        self._current_dims = None

        # One TileAnnotations per pooled folder (see data/pooled_annotations.py) --
        # each keeps its own sidecar file, dispatched to by absolute path.
        # Colors assigned to category names in first-seen order and kept
        # stable for the session.
        self.annotations = PooledAnnotations(folders)
        self._category_colors = {}
        for category in self.annotations.categories():
            self._category_color(category)

        self._classifier_instances = {}  # name -> loaded TileClassifier

        if self.annotations.dims:
            self._current_dims = self.annotations.dims

        self.visual_proxy = VisualState(self)

        # Per-channel color/blend-mode/opacity overlay state, persisted
        # to the annotation sidecar file on every change; seeded from it
        # in _update_channel_state once channels are known. Channels are
        # discovered incrementally as background decoding progresses, so
        # seed from a frozen snapshot rather than the live, mutating
        # self.annotations.channel_colors to avoid a partial-write race.
        self._pending_channel_colors = dict(self.annotations.channel_colors)
        self.overlay_state = OverlayStateList(0)
        self.overlay_state.subscribe(self._on_overlay_state_changed)
        self._colors_seeded_channels = set()

        self.colors_panel = None
        self._colors_dock = None
        self.annotation_stats_panel = None
        self._annotation_stats_dock = None

        self._setup_ui()
        self._setup_menu()
        self._load_current_page()

    # ------------------------------------------------------------------
    # Annotation badges / categories
    # ------------------------------------------------------------------

    def _category_color(self, category):
        """Hex color for category, assigned in first-seen order and
        kept stable for the life of this window."""
        if category not in self._category_colors:
            index = len(self._category_colors)
            self._category_colors[category] = category_color(index)
        return self._category_colors[category]

    def _refresh_badges(self):
        """Rebuild the whole folder's path -> (category, color) map and
        push it to the thumbnail grid. Cheap: just dict lookups over
        already-loaded annotations, no file I/O."""
        badges = {}
        for path in self.image_paths:
            relpath = self.annotations.relpath(path)
            category = self.annotations.get(relpath)
            if category:
                confidence = self.annotations.confidence(relpath)
                badges[path] = (category, self._category_color(category), confidence)
        self.thumbnail_grid.set_annotations(badges)

    def _update_channel_state(self):
        """Channel count can only grow as more images finish decoding in
        the background."""
        max_c = self.thumbnail_grid.max_channels()
        if max_c > self.max_C:
            self.max_C = max_c
        self.visual_proxy.update_max_channels(self.max_C)
        self.thumbnail_grid.set_channel_display(
            self.visual_proxy.display, self.visual_proxy.custom_clim
        )

        self.overlay_state.resize(self.max_C)
        self._seed_channel_colors()
        self.thumbnail_grid.set_overlay_state(self.overlay_state)

        if self.colors_panel is not None and self._colors_dock.isVisible():
            self.colors_panel.refresh_ui()

    def _seed_channel_colors(self):
        """Apply any persisted per-channel color/blend-mode/opacity
        overrides (from the annotation sidecar's #channel_colors line) to
        newly-available channel indices, once each."""
        for idx, (color_hex, blend_mode, opacity) in self._pending_channel_colors.items():
            if idx in self._colors_seeded_channels or idx >= len(self.overlay_state):
                continue
            self.overlay_state.set_color(idx, color_hex)
            self.overlay_state.set_blend_mode(idx, blend_mode)
            self.overlay_state.set_opacity(idx, opacity)
            self._colors_seeded_channels.add(idx)

    def _on_overlay_state_changed(self, channel_idx, field):
        # Sparse: only channels that differ from their default get an
        # entry, so merely opening a folder never touches the sidecar
        # file unless the user actually changes a color.
        colors = {}
        for c in range(len(self.overlay_state)):
            state = self.overlay_state[c]
            default_state = default_channel_state(c)
            if (state.color_hex, state.blend_mode, state.opacity) != (
                default_state.color_hex,
                default_state.blend_mode,
                default_state.opacity,
            ):
                colors[c] = (state.color_hex, state.blend_mode, state.opacity)
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

    def _open_in_viewer(self, path):
        """"Open in Viewer": a full-resolution zoom popup for per-pixel
        inspection -- there's no full ImageWindow in this project."""
        rgb = self.thumbnail_grid.full_resolution_rgb(path)
        if rgb is None:
            return
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()

        dlg = QDialog(self)
        dlg.setWindowTitle(Path(path).name)
        layout = QVBoxLayout(dlg)
        label = QLabel()
        label.setPixmap(QPixmap.fromImage(qimg))
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(w < 800 and h < 800)
        layout.addWidget(scroll)
        dlg.resize(min(w, 900), min(h, 900))
        dlg.exec_()

    def _show_metadata(self, path):
        try:
            plane = load_plane(path, dims=self._current_dims)
        except Exception as exc:
            QMessageBox.warning(self, "Show Info", f"Could not read {path}:\n{exc}")
            return
        C, H, W = plane.shape
        QMessageBox.information(
            self,
            "Image Info",
            f"{path}\n\nChannels: {C}\nHeight: {H}\nWidth: {W}\nDtype: {plane.dtype}",
        )

    def show_manage_categories_dialog(self):
        dlg = ManageCategoriesDialog(self.annotations, parent=self)
        dlg.exec_()
        # Colors are assigned in first-seen order over the vocabulary too,
        # so a rename/delete/add can shift them -- recompute from scratch.
        self._category_colors = {}
        for category in self.annotations.categories():
            self._category_color(category)
        self._refresh_badges()
        self._refresh_annotation_stats()

        filter_before = self._category_filter
        self._refresh_category_filter_combo()
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
            self.annotation_stats_panel.refresh(self.annotations, len(self.image_paths))

    def add_project_folders(self):
        """Pool in more already-annotated project folders at runtime, so
        VICReg pretraining / classifier training / stats can see them
        without relaunching with different command-line arguments.

        Deliberately annotations-pool-only: newly-added folders' tiles are
        NOT added to this window's browsable thumbnail grid. `image_paths`
        is built once in `__main__.py` before this window exists, and
        `PooledAnnotations.dims` is a single value shared by the whole grid
        -- extending the grid at runtime would need per-tile axis-order
        handling this app doesn't have. Open a folder directly
        (`tiled_viewer <folder>`) to browse/annotate it.
        """
        dirs = []
        start_dir = self.annotations.folders[-1] if self.annotations.folders else ""
        while True:
            d = QFileDialog.getExistingDirectory(self, "Add Project Folder", start_dir)
            if not d:
                break
            dirs.append(d)
            start_dir = d
            if (
                QMessageBox.question(self, "Add Project Folder(s)", "Add another folder?")
                != QMessageBox.Yes
            ):
                break
        if not dirs:
            return

        added = self.annotations.add_folders(dirs)
        if not added:
            QMessageBox.information(
                self, "Add Project Folder(s)", "Already pooled -- nothing to add."
            )
            return

        self._category_colors = {}
        for category in self.annotations.categories():
            self._category_color(category)
        self._refresh_badges()
        self._refresh_annotation_stats()
        self._refresh_category_filter_combo()

        QMessageBox.information(
            self,
            "Add Project Folder(s)",
            f"Added {len(added)} folder(s) to the annotation pool for stats, "
            "category management, and training. Their tiles are NOT shown in "
            "this window's grid -- open them directly (tiled_viewer <folder>) "
            "to browse/annotate them.",
        )

    def annotate_selected_tiles(self):
        """Prompt for a category and apply it to every selected tile.

        The picker only offers this folder's predefined category
        vocabulary (see ManageCategoriesDialog) -- no free text entry --
        so a typo can't silently create a near-duplicate category.
        """
        paths = self.thumbnail_grid.selected_paths
        if not paths:
            return

        items = [""] + self.annotations.categories()
        current = (
            self.annotations.get(self.annotations.relpath(paths[0])) or ""
            if len(paths) == 1
            else ""
        )
        if current and current not in items:
            items.append(current)
        start_idx = items.index(current) if current in items else 0

        if not self.annotations.categories():
            QMessageBox.information(
                self,
                "Annotate",
                "No categories defined yet. Use Annotate > Manage "
                "Categories... to add some first.",
            )
            return

        if len(paths) == 1:
            prompt = f"Category for {Path(paths[0]).name}:"
        else:
            prompt = f"Category for {len(paths)} images:"

        text, ok = QInputDialog.getItem(
            self, "Annotate", prompt, items, start_idx, editable=False
        )
        if not ok:
            return

        category = text.strip()
        self.annotations.update(
            (self.annotations.relpath(p), category) for p in paths
        )
        if self._category_filter is not None:
            self._load_current_page()
        else:
            self._refresh_badges()
        self._update_status()
        self._refresh_annotation_stats()
        self._refresh_category_filter_combo()

    def _get_classifier(self, name):
        """Lazily instantiate (and cache) the named registered classifier.

        Instantiating is what actually imports torch/torchvision (see
        ``classifiers/yeast_efficientnet.py``) -- so this is also where a
        missing ``[ai]`` extra install surfaces, as a friendly dialog
        rather than a crash.
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
                    "  pip install -e '.[ai]'",
                )
                return None
        return self._classifier_instances[name]

    def auto_annotate_page(self):
        """Run a classifier over the *current page only* and tag every
        tile that isn't already annotated (existing tags -- manual or a
        prior AI pass -- are left untouched; re-run after correcting a
        few tiles and only the rest get (re-)predicted).

        Applied tags are also recorded in ``_tile_confidence`` so the
        grid can badge them as "unconfirmed" until a human reviews them
        via ``annotate_selected_tiles``.
        """
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
        self._refresh_badges()
        self._update_status()
        self._refresh_annotation_stats()
        self._refresh_category_filter_combo()

        message = f"Tagged {len(updates)} of {len(paths)} tiles on this page"
        message += f" ({skipped} already annotated, left untouched)." if skipped else "."
        QMessageBox.information(self, "Auto-Annotate", message)

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
        ascending confidence with manually-annotated/confirmed tiles
        (no confidence score) pushed to the end -- so the network's least
        confident calls for a given category surface first and its
        human-reviewed ground truth for that category is easy to compare
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
            f"tileclass - {self.annotations.label()} - {len(self.image_paths)} images"
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
        self.per_page_spin.setRange(self._tiles_per_page_min, self._tiles_per_page_max)
        self.per_page_spin.setValue(self.tiles_per_page)
        self.per_page_spin.setFixedWidth(70)
        self.per_page_spin.setKeyboardTracking(False)
        self.per_page_spin.valueChanged.connect(self._on_per_page_changed)
        toolbar_layout.addWidget(self.per_page_spin)

        toolbar_layout.addSpacing(20)

        toolbar_layout.addWidget(QLabel("Tile size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(self._tile_size_min, self._tile_size_max)
        self.size_slider.setValue(self.tile_size)
        self.size_slider.setFixedWidth(150)
        self.size_slider.valueChanged.connect(self._on_tile_size_changed)
        toolbar_layout.addWidget(self.size_slider)

        self.size_label = QLabel(f"{self.tile_size}px")
        self.size_label.setFixedWidth(50)
        toolbar_layout.addWidget(self.size_label)

        toolbar_layout.addSpacing(20)

        self.show_info_check = QCheckBox("Show Info (I)")
        self.show_info_check.setChecked(False)
        self.show_info_check.toggled.connect(self._on_show_info_toggled)
        toolbar_layout.addWidget(self.show_info_check)

        toolbar_layout.addSpacing(20)

        toolbar_layout.addWidget(QLabel("Filter:"))
        self.category_filter_combo = QComboBox()
        self.category_filter_combo.setMinimumWidth(120)
        self._refresh_category_filter_combo()
        self.category_filter_combo.currentIndexChanged.connect(
            self._on_category_filter_changed
        )
        toolbar_layout.addWidget(self.category_filter_combo)

        toolbar_layout.addStretch()
        main_layout.addWidget(toolbar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.thumbnail_grid = ThumbnailGridWidget()
        self.thumbnail_grid.set_tile_size(self.tile_size)
        self.thumbnail_grid.set_show_info(self.show_info)
        self.thumbnail_grid.selectionChanged.connect(self._on_selection_changed)
        self.thumbnail_grid.annotateRequested.connect(self.annotate_selected_tiles)
        self.thumbnail_grid.openInViewerRequested.connect(self._open_in_viewer)
        self.thumbnail_grid.showInfoRequested.connect(self._show_metadata)
        self.thumbnail_grid.decoded.connect(self._on_thumbnail_decoded)
        self.scroll_area.setWidget(self.thumbnail_grid)
        main_layout.addWidget(self.scroll_area, 1)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        self.status_label.setToolTip(self.annotations.root_dir)
        main_layout.addWidget(self.status_label)

        self._update_page_controls()

    def _setup_menu(self):
        adjust_menu = self.menuBar().addMenu("&Adjust")

        colors_action = adjust_menu.addAction("Colors...")
        colors_action.triggered.connect(self.show_colors_panel)

        auto_action = adjust_menu.addAction("Auto Contrast All")
        auto_action.setShortcut("C")
        auto_action.triggered.connect(self._auto_contrast_all)

        adjust_menu.addSeparator()

        limits_action = adjust_menu.addAction("Tile Size / Page Limits...")
        limits_action.triggered.connect(self.show_display_limits_dialog)

        annotate_menu = self.menuBar().addMenu("&Annotate")

        annotate_action = annotate_menu.addAction("Annotate Selected...")
        annotate_action.setShortcut("T")
        annotate_action.triggered.connect(self.annotate_selected_tiles)

        auto_annotate_action = annotate_menu.addAction("Auto-Annotate Page (AI)...")
        auto_annotate_action.setShortcut("Shift+T")
        auto_annotate_action.triggered.connect(self.auto_annotate_page)

        group_action = annotate_menu.addAction("Group by Category")
        group_action.setShortcut("G")
        group_action.triggered.connect(self.sort_by_annotation)

        annotate_menu.addSeparator()

        manage_action = annotate_menu.addAction("Manage Categories...")
        manage_action.triggered.connect(self.show_manage_categories_dialog)

        stats_action = annotate_menu.addAction("Annotation Stats...")
        stats_action.triggered.connect(self.show_annotation_stats)

        add_folders_action = annotate_menu.addAction("Add Project Folder(s)...")
        add_folders_action.triggered.connect(self.add_project_folders)

    def show_display_limits_dialog(self):
        dlg = TiledDisplaySettingsDialog(
            {
                "tile_size_min": self._tile_size_min,
                "tile_size_max": self._tile_size_max,
                "tiles_per_page_min": self._tiles_per_page_min,
                "tiles_per_page_max": self._tiles_per_page_max,
            },
            parent=self,
        )
        if not dlg.exec_():
            return

        cfg = dlg.get_config()
        self._tile_size_min = cfg["tile_size_min"]
        self._tile_size_max = cfg["tile_size_max"]
        self._tiles_per_page_min = cfg["tiles_per_page_min"]
        self._tiles_per_page_max = cfg["tiles_per_page_max"]

        self.size_slider.setRange(self._tile_size_min, self._tile_size_max)
        self.per_page_spin.setRange(
            self._tiles_per_page_min, self._tiles_per_page_max
        )

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
        text = f"Showing {start}-{end} of {len(visible)} images{filter_suffix}"
        selected = self.thumbnail_grid.selected_paths
        if len(selected) == 1:
            text += f"   |   Selected: {Path(selected[0]).name}"
        elif len(selected) > 1:
            text += f"   |   Selected: {len(selected)} images"
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

        self.thumbnail_grid.set_items(page_paths, dims=self._current_dims)

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

    def _on_per_page_changed(self, value):
        self.tiles_per_page = value
        self.current_page = 0
        self._load_current_page()

    def _on_tile_size_changed(self, value):
        self.tile_size = value
        self.size_label.setText(f"{value}px")
        self.thumbnail_grid.set_tile_size(value)

    def _auto_contrast_all(self):
        """Every image's own per-image auto-contrast is already its
        default clim (baked in at decode time) -- this just reverts any
        explicit Colors-panel overrides back to that default."""
        self.visual_proxy.custom_clim.clear()
        self.thumbnail_grid.invalidate_pixmaps()
        if self.colors_panel is not None and self._colors_dock.isVisible():
            self.colors_panel.refresh_ui()

    def _on_show_info_toggled(self, checked):
        self.show_info = checked
        self.thumbnail_grid.set_show_info(checked)

    def _toggle_show_info(self):
        self.show_info_check.setChecked(not self.show_info_check.isChecked())

    # ------------------------------------------------------------------
    # Keyboard / lifecycle
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Left or key == Qt.Key_PageUp:
            self._prev_page()
        elif key == Qt.Key_Right or key == Qt.Key_PageDown:
            self._next_page()
        elif key == Qt.Key_Home:
            self.current_page = 0
            self._load_current_page()
        elif key == Qt.Key_End:
            self.current_page = self._total_pages() - 1
            self._load_current_page()
        elif key == Qt.Key_C:
            self._auto_contrast_all()
        elif key == Qt.Key_I:
            self._toggle_show_info()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            new_size = min(self._tile_size_max, self.tile_size + 25)
            self.size_slider.setValue(new_size)
        elif key == Qt.Key_Minus:
            new_size = max(self._tile_size_min, self.tile_size - 25)
            self.size_slider.setValue(new_size)
        elif key == Qt.Key_T:
            self.annotate_selected_tiles()
        elif key == Qt.Key_G:
            self.sort_by_annotation()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.thumbnail_grid.close()
        super().closeEvent(event)
