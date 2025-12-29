from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..constants import format_camera_label
from ..scaling import UiScale, add_drop_shadow
from .video_tile import VideoTile


class VehicleCameraSection(QFrame):
    """Vehicle selector with accompanying camera thumbnails."""

    def __init__(self, title_text: str, combo_placeholder: str, camera_slots, parent: QWidget | None = None):
        super().__init__(parent)
        self.camera_slots = list(camera_slots)
        self.setObjectName("Card")
        add_drop_shadow(self, radius=18, alpha=70)

        root = QVBoxLayout(self)
        root.setContentsMargins(UiScale.dp(18), UiScale.dp(18), UiScale.dp(18), UiScale.dp(18))
        root.setSpacing(UiScale.dp(18))

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(UiScale.dp(10))

        self.lbl = QLabel(title_text, self)
        self.lbl.setObjectName("SectionLabel")

        self.cmb = QComboBox(self)
        self.cmb.setEditable(False)
        self.cmb.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.cmb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cmb.setMinimumContentsLength(1)
        self.cmb.setPlaceholderText(combo_placeholder)

        header.addWidget(self.lbl)
        header.addWidget(self.cmb, 1)
        header.addStretch(1)

        self.grid_host = QWidget(self)
        self.grid_host.setObjectName("TileGridHost")
        self.grid_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(UiScale.dp(12))
        self.grid.setVerticalSpacing(UiScale.dp(12))

        self.tiles: dict[str, VideoTile] = {}
        self._tile_order: list[VideoTile] = []
        self._max_grid_columns = 2
        self._current_grid_columns = 0

        for slot in self.camera_slots:
            camera_key = slot['name']
            caption = slot.get('label') or format_camera_label(camera_key)
            tile = VideoTile(camera_key, caption, self.grid_host)
            self._tile_order.append(tile)
            self.tiles[camera_key] = tile

        initial_cols = min(self._max_grid_columns, max(1, len(self._tile_order)))
        self._reflow_tiles(cols=initial_cols)

        root.addLayout(header)
        root.addWidget(self.grid_host, 1)
        root.setStretch(1, 1)

    def _reflow_tiles(self, width: int | None = None, cols: int | None = None) -> None:
        if not self._tile_order:
            return
        if cols is None:
            if width is None or width <= 0:
                cols = min(self._max_grid_columns, len(self._tile_order))
            else:
                margins = self.grid.contentsMargins()
                available = width - (margins.left() + margins.right())
                spacing = self.grid.horizontalSpacing() or 0
                base_tile = self._tile_order[0]
                tile_width = max(base_tile.sizeHint().width(), UiScale.dp(160))
                effective = tile_width + spacing
                if effective <= 0:
                    cols = 1
                else:
                    usable = max(available, effective)
                    cols = max(1, min(self._max_grid_columns, (usable + spacing) // effective))
        cols = max(1, min(cols, len(self._tile_order)))
        if self._current_grid_columns == cols:
            return
        self._current_grid_columns = cols

        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.grid_host)

        for index, tile in enumerate(self._tile_order):
            row = index // cols
            col = index % cols
            self.grid.addWidget(tile, row, col)

        for col_index in range(cols):
            self.grid.setColumnStretch(col_index, 1)
        row_count = (len(self._tile_order) + cols - 1) // cols
        for row_index in range(row_count):
            self.grid.setRowStretch(row_index, 1)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._reflow_tiles(width=event.size().width())

    def set_title_label(self, text: str) -> None:
        self.lbl.setText(text)

    def set_combo_placeholder(self, text: str) -> None:
        self.cmb.setPlaceholderText(text)
