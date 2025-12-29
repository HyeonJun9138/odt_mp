from __future__ import annotations

from typing import Optional

from app.gui.qt import QUrl, QtCore, QtWidgets, QtWebEngineWidgets


class MapView(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, url: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setUrl(QUrl(url))


class MapWindow(QtWidgets.QMainWindow):
    def __init__(self, url: str, title: Optional[str] = None) -> None:
        super().__init__()
        if title:
            self.setWindowTitle(title)

        self._view = MapView(url, parent=self)
        self.setCentralWidget(self._view)
        self.setMinimumSize(800, 600)
