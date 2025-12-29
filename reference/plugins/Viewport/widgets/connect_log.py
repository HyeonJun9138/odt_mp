from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QPlainTextEdit,
    QLabel,
    QVBoxLayout,
)

from ..scaling import UiScale, add_drop_shadow


class ConnectPanel(QFrame):
    """Displays contextual guidance while waiting for a vehicle selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        add_drop_shadow(self, radius=14, alpha=60)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(UiScale.dp(16), UiScale.dp(16), UiScale.dp(16), UiScale.dp(16))
        layout.setSpacing(UiScale.dp(8))

        self.text = QLabel("Select a vehicle on the left to establish an AirSim connection.", self)
        self.text.setWordWrap(True)
        self.text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text.setStyleSheet(f"font-size:{UiScale.dp(16)}px;")

        layout.addStretch(1)
        layout.addWidget(self.text)
        layout.addStretch(1)


class LogPanel(QFrame):
    """Simple container wrapping a log view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        add_drop_shadow(self, radius=14, alpha=60)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(UiScale.dp(16), UiScale.dp(16), UiScale.dp(16), UiScale.dp(16))
        layout.setSpacing(UiScale.dp(8))

        self.log = QPlainTextEdit(self)
        self.log.setObjectName("LogEdit")
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Log")
        layout.addWidget(self.log, 1)
