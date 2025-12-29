from __future__ import annotations

import sys
from typing import Optional

from app.gui.qt import QtWidgets
from app.gui.window import MapWindow


def run_app(url: str, title: Optional[str] = None) -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MapWindow(url, title=title)
    window.resize(1200, 900)
    window.show()
    return app.exec()
