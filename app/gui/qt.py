from __future__ import annotations

try:
    from PyQt6 import QtCore, QtWidgets, QtWebEngineWidgets
    from PyQt6.QtCore import QUrl
except ImportError:
    from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
    from PyQt5.QtCore import QUrl

__all__ = ["QtCore", "QtWidgets", "QtWebEngineWidgets", "QUrl"]
