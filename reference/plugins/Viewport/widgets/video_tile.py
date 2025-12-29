from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..scaling import UiScale
from .aspect_box import AspectBox


class VideoTile(QWidget):
    """Single camera thumbnail tile used inside the multi-camera grid."""

    clicked = Signal(str)

    def __init__(self, camera_name: str, caption: str = "Front View", parent: QWidget | None = None):
        super().__init__(parent)
        self.camera_name = camera_name

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(UiScale.dp(6))

        self.aspect = AspectBox(self, 16, 9, min_w=180, max_w=None)
        self.aspect.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.caption = QLabel(caption, self)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.caption.setObjectName("TileCaption")
        self.caption.setWordWrap(True)

        layout.addWidget(self.aspect, 1)
        layout.addWidget(self.caption, 0)

    def set_caption(self, text: str) -> None:
        self.caption.setText(text)

    def set_image_bytes(self, raw: bytes | None) -> None:
        self.aspect.set_image_bytes(raw)

    def set_selected(self, selected: bool) -> None:
        self.aspect.set_selected(selected)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.camera_name)
        super().mousePressEvent(event)
