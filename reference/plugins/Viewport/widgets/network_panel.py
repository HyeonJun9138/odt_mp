from __future__ import annotations

import socket

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..scaling import UiScale, add_drop_shadow


class NetworkPanel(QFrame):
    """Network configuration form for AirSim connectivity."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        add_drop_shadow(self, radius=14, alpha=60)

        root = QVBoxLayout(self)
        root.setContentsMargins(UiScale.dp(16), UiScale.dp(14), UiScale.dp(16), UiScale.dp(14))
        root.setSpacing(UiScale.dp(10))

        title = QLabel("Connection", self)
        title.setObjectName("SectionLabel")
        root.addWidget(title)

        form_host = QWidget(self)
        form = QGridLayout(form_host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(UiScale.dp(8))
        form.setVerticalSpacing(UiScale.dp(8))

        def make_label(text: str) -> QLabel:
            label = QLabel(text, self)
            label.setObjectName("FormLabel")
            label.setMinimumWidth(UiScale.dp(120))
            label.setMaximumWidth(UiScale.dp(160))
            return label

        def detect_local_ip() -> str:
            ip = "127.0.0.1"
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
            except Exception:
                try:
                    ip = socket.gethostbyname(socket.gethostname())
                except Exception:
                    pass
            return ip

        host_ip = detect_local_ip()

        self.edtHost = QLineEdit(host_ip, self)
        self.edtClient = QLineEdit("", self)
        self.edtClient.setPlaceholderText("auto")
        self.spinPort = QSpinBox(self)
        self.spinPort.setRange(1, 65535)
        self.spinPort.setValue(41451)
        self.spinPort.hide()
        self.pluginPortValue = QLabel("N/A", self)
        self.pluginPortValue.setObjectName("FormLabel")
        self.pluginPortValue.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.edtHost.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.edtClient.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        form.setColumnStretch(0, 0)
        form.setColumnStretch(1, 3)
        form.setColumnStretch(2, 1)
        form.setColumnStretch(3, 1)

        form.addWidget(make_label("Host IP :"), 0, 0)
        form.addWidget(self.edtHost, 0, 1, 1, 3)
        form.addWidget(make_label("Client IP :"), 1, 0)
        form.addWidget(self.edtClient, 1, 1, 1, 3)
        form.addWidget(make_label("Plugin Port :"), 2, 0)
        form.addWidget(self.pluginPortValue, 2, 1, 1, 3)

        button_row = QWidget(self)
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(UiScale.dp(8))
        self.btnApply = QPushButton("Apply", self)
        self.btnApply.setObjectName("AccentButton")
        button_layout.addStretch(1)
        button_layout.addWidget(self.btnApply)

        root.addWidget(form_host, 1)
        root.addWidget(button_row, 0)


    def apply_defaults(self, host: str | None = None, port: int | None = None) -> None:
        if host:
            self.edtHost.setText(str(host))
        if port is not None:
            try:
                self.spinPort.setValue(int(port))
            except Exception:
                pass


    def set_plugin_port_display(self, port: int | None) -> None:
        if port is None:
            self.pluginPortValue.setText("N/A")
        else:
            self.pluginPortValue.setText(str(port))

