from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QFrame, QSizePolicy, QWidget

from ..scaling import UiScale, add_drop_shadow


class AspectBox(QWidget):
    """Viewport surface that preserves a fixed aspect ratio for preview frames."""

    def __init__(self, parent: QWidget | None = None, ratio_w: int = 16, ratio_h: int = 9, min_w: int = 240, max_w: int | None = None):
        super().__init__(parent)
        self.rw = ratio_w
        self.rh = ratio_h
        self._min_content = UiScale.dp(min_w) if min_w else 0
        self._max_content = UiScale.dp(max_w) if max_w else None

        self._inner = QFrame(self)
        self._inner.setObjectName("VideoInner")
        self._inner.setFrameStyle(QFrame.Shape.NoFrame)
        add_drop_shadow(self._inner, radius=18, alpha=70)

        self._img = QLabel(self._inner)
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._last_qimage: QImage | None = None
        self._last_pixmap: QPixmap | None = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        min_height = UiScale.dp(120)
        if self._min_content and self.rw:
            min_height = max(min_height, int(self._min_content * self.rh / max(1, self.rw)))
        self.setMinimumHeight(min_height)

    def set_selected(self, selected: bool) -> None:
        self._inner.setProperty("selected", "true" if selected else "false")
        style = self._inner.style()
        style.unpolish(self._inner)
        style.polish(self._inner)
        self._inner.update()

    def setMinimumContentWidth(self, width: int) -> None:  # noqa: N802 - Qt naming compatibility
        self._min_content = UiScale.dp(width)
        self.updateGeometry()

    def setMaximumContentWidth(self, width: int | None) -> None:  # noqa: N802
        self._max_content = UiScale.dp(width) if width else None
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        width = max(self._min_content, UiScale.dp(120))
        height = int(width * self.rh / self.rw) if self.rw else width
        return QSize(width, height)

    def _rescale(self) -> None:
        if self._last_qimage is None:
            return
        target = self._inner.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        pixmap = QPixmap.fromImage(self._last_qimage).scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._last_pixmap = pixmap
        self._img.resize(pixmap.size())
        self._img.move(
            (self._inner.width() - pixmap.width()) // 2,
            (self._inner.height() - pixmap.height()) // 2,
        )
        self._img.setPixmap(pixmap)

    def set_qimage(self, image: QImage) -> None:
        self._last_qimage = image
        self._rescale()

    def set_image_bytes(self, raw: bytes | None) -> None:
        if not raw:
            return
        image = QImage.fromData(raw)
        if image.isNull():
            return
        self.set_qimage(image)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        available_w = self.width()
        available_h = self.height()
        content_w = available_w
        if self._max_content is not None:
            content_w = min(content_w, self._max_content)
        if self._min_content:
            content_w = max(content_w, self._min_content)
        content_w = min(content_w, available_w)
        content_w = max(1, content_w)

        content_h = int(content_w * self.rh / self.rw) if self.rw else available_h
        if content_h > available_h and self.rh:
            content_h = available_h
            content_w = max(1, int(content_h * self.rw / self.rh))

        x = (available_w - content_w) // 2
        y = (available_h - content_h) // 2
        self._inner.setGeometry(QRect(x, y, content_w, content_h))
        self._img.raise_()
        self._rescale()
        super().resizeEvent(event)
