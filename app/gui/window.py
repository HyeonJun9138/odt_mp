from __future__ import annotations

from typing import Optional

from app.gui.qt import QUrl, QtCore, QtWidgets, QtWebEngineWidgets


class MapView(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, url: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setUrl(QUrl(url))


class MapWindow(QtWidgets.QMainWindow):
    def __init__(self, url: str, title: Optional[str] = None, aspect_ratio: float = 4 / 3) -> None:
        super().__init__()
        self._aspect_ratio = aspect_ratio
        self._adjusting = False
        if title:
            self.setWindowTitle(title)

        self._view = MapView(url, parent=self)
        self.setCentralWidget(self._view)
        self.setMinimumSize(800, 600)

    def resizeEvent(self, event: QtCore.QResizeEvent) -> None:
        if not self._adjusting:
            old = event.oldSize()
            if old.width() > 0 and old.height() > 0:
                new = event.size()
                width_delta = abs(new.width() - old.width())
                height_delta = abs(new.height() - old.height())
                if width_delta >= height_delta:
                    target_height = int(new.width() / self._aspect_ratio)
                    target = QtCore.QSize(new.width(), target_height)
                else:
                    target_width = int(new.height() * self._aspect_ratio)
                    target = QtCore.QSize(target_width, new.height())
                if target != new:
                    self._adjusting = True
                    self.resize(target)
                    self._adjusting = False
        super().resizeEvent(event)
