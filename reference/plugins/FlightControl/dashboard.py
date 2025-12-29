from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from plugins.Viewport.scaling import UiScale
from plugins.Viewport.theming import apply_common_qss


class AspectRatioFrame(QFrame):
    """Frame that maintains a fixed aspect ratio for its inner canvas."""

    def __init__(self, *, aspect_ratio: float = 16 / 9, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._aspect_ratio = aspect_ratio
        self.setObjectName("FlightControlCanvasContainer")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._canvas = QFrame(self)
        self._canvas.setObjectName("FlightControlCanvas")
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._canvas_layout = QVBoxLayout(self._canvas)
        self._canvas_layout.setContentsMargins(UiScale.dp(32), UiScale.dp(32), UiScale.dp(32), UiScale.dp(32))
        self._canvas_layout.setSpacing(UiScale.dp(12))

        self._placeholder = QLabel("Mission feed will appear here.")
        self._placeholder.setObjectName("FlightControlCanvasLabel")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._canvas_layout.addStretch()
        self._canvas_layout.addWidget(self._placeholder, 0, Qt.AlignCenter)
        self._canvas_layout.addStretch()

        self._update_canvas_geometry()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_canvas_geometry()

    def _update_canvas_geometry(self) -> None:
        area_w = max(0, self.width())
        area_h = max(0, self.height())
        if not area_w or not area_h:
            return

        target_h = int(round(area_w / self._aspect_ratio))
        target_w = int(round(area_h * self._aspect_ratio))
        if target_h <= area_h:
            width = area_w
            height = target_h
        else:
            width = target_w
            height = area_h

        x = (area_w - width) // 2
        y = (area_h - height) // 2
        self._canvas.setGeometry(int(x), int(y), int(width), int(height))

    def set_placeholder_text(self, text: str) -> None:
        self._placeholder.setText(text)

    @property
    def canvas(self) -> QFrame:
        return self._canvas


class FlightControlDashboard(QWidget):
    """Flight control hub window with viewport-inspired styling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._exit_callback: Callable[[], None] | None = None
        self._plugin_name: str | None = None
        self._plugin_port: int | None = None

        self._vehicle_combo: QComboBox | None = None
        self._port_edit: QLineEdit | None = None
        self._status_label: QLabel | None = None
        self._mode_checks: list[QCheckBox] = []
        self._canvas: AspectRatioFrame | None = None
        self._control_height = UiScale.dp(34)
        self._panel_padding_v = UiScale.dp(8)
        self._panel_padding_h = UiScale.dp(18)

        self._build_ui()

    def _build_ui(self) -> None:
        app = QApplication.instance()
        if app is not None:
            UiScale.init_from_screen(app)
            apply_common_qss(app)
        else:
            apply_common_qss(self)

        self._control_height = UiScale.dp(34)
        self._panel_padding_v = UiScale.dp(8)
        self._panel_padding_h = UiScale.dp(18)

        self.setObjectName("FlightControlRoot")
        self.setWindowTitle("Flight Control Hub")
        self.resize(1625, 900)
        self.setMinimumSize(1300, 760)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(UiScale.dp(32), UiScale.dp(32), UiScale.dp(32), UiScale.dp(32))
        root_layout.setSpacing(UiScale.dp(24))

        title = QLabel("Flight Control Hub", self)
        title.setObjectName("FlightControlTitle")
        title_font = title.font()
        title_font.setPointSizeF(max(28.0, 24.0 * UiScale.scale))
        title_font.setBold(True)
        title.setFont(title_font)
        root_layout.addWidget(title, 0, Qt.AlignLeft)

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(UiScale.dp(18))
        root_layout.addLayout(controls_row, 0)

        vehicle_panel = self._build_vehicle_selector()
        mode_panel = self._build_mode_selector()
        port_panel = self._build_port_panel()

        controls_row.addWidget(vehicle_panel)
        controls_row.addWidget(mode_panel)
        controls_row.addWidget(port_panel)
        controls_row.setStretch(0, 1)
        controls_row.setStretch(1, 2)
        controls_row.setStretch(2, 1)

        status_label = QLabel("Select a vehicle to begin configuring controls.", self)
        status_label.setObjectName("FlightControlStatus")
        status_label.setWordWrap(True)
        root_layout.addWidget(status_label, 0)
        self._status_label = status_label

        canvas = AspectRatioFrame(parent=self)
        canvas.set_placeholder_text("Awaiting telemetry or video feeds.")
        root_layout.addWidget(canvas, 1)
        self._canvas = canvas

        self._apply_local_styles()

    def _build_vehicle_selector(self) -> QFrame:
        combo = QComboBox(self)
        combo.setObjectName("VehicleCombo")
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo.setMinimumHeight(self._control_height)
        combo.addItems(
            [
                "-- Select Vehicle --",
                "PX4 Iris (Multi-Rotor)",
                "PX4 VTOL",
                "ArduPilot QuadPlane",
                "Simulation Dummy",
            ]
        )
        combo.setCurrentIndex(0)
        self._vehicle_combo = combo
        return self._wrap_panel("Selected Vehicle :", combo)

    def _build_mode_selector(self) -> QFrame:
        container = QFrame(self)
        container.setObjectName("ModeContainer")
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        container.setMinimumHeight(self._control_height)

        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(UiScale.dp(16))
        layout.setAlignment(Qt.AlignVCenter)

        modes = [
            ("Joystick Mode", False),
            ("Mission Flight Mode", False),
            ("Autonomous Mode", False),
        ]
        self._mode_checks.clear()
        for text, checked in modes:
            checkbox = QCheckBox(text, container)
            checkbox.setChecked(checked)
            checkbox.setMinimumHeight(self._control_height)
            layout.addWidget(checkbox)
            self._mode_checks.append(checkbox)

        layout.addStretch(1)
        return self._wrap_panel("Mode Selection :", container)

    def _build_port_panel(self) -> QFrame:
        port_edit = QLineEdit(self)
        port_edit.setObjectName("PortEdit")
        port_edit.setPlaceholderText("auto")
        port_edit.setReadOnly(True)
        port_edit.setAlignment(Qt.AlignCenter)
        port_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        port_edit.setMinimumHeight(self._control_height)
        self._port_edit = port_edit
        return self._wrap_panel("Port :", port_edit)

    def _wrap_panel(self, title: str, content: QWidget) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("ControlPanel")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        padding_h = getattr(self, "_panel_padding_h", UiScale.dp(18))
        padding_v = getattr(self, "_panel_padding_v", UiScale.dp(8))
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(padding_h, padding_v, padding_h, padding_v)
        layout.setSpacing(UiScale.dp(12))

        label = QLabel(title, frame)
        label.setObjectName("ControlLabel")
        label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        layout.addWidget(label, 0)

        content.setParent(frame)
        layout.addWidget(content, 1)

        control_height = getattr(self, "_control_height", UiScale.dp(34))
        frame.setMinimumHeight(control_height + (padding_v * 2))

        return frame

    def _apply_local_styles(self) -> None:
        pad_panel = getattr(self, "_panel_padding_h", UiScale.dp(18))
        pad_small = getattr(self, "_panel_padding_v", UiScale.dp(8))
        radius_panel = UiScale.dp(12)
        radius_canvas = UiScale.dp(18)
        control_height = getattr(self, "_control_height", UiScale.dp(34))

        stylesheet = f"""
        QWidget#FlightControlRoot {{
            background-color: #0b2f3f;
            color: #e2e8f0;
        }}
        QLabel#FlightControlTitle {{
            font-size: {UiScale.dp(30)}px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        QFrame#ControlPanel {{
            background-color: rgba(6, 43, 58, 0.82);
            border: 1px solid rgba(226, 232, 240, 0.38);
            border-radius: {radius_panel}px;
            padding: {pad_small}px {pad_panel}px;
        }}
        QLabel#ControlLabel {{
            font-size: {UiScale.dp(16)}px;
            font-weight: 600;
            padding-right: {UiScale.dp(8)}px;
        }}
        QLabel#FlightControlStatus {{
            padding: {pad_small}px;
            border-radius: {UiScale.dp(10)}px;
            background-color: rgba(15, 31, 45, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.35);
            color: rgba(226, 232, 240, 0.85);
            font-size: {UiScale.dp(13)}px;
        }}
        QComboBox, QLineEdit {{
            background-color: #f8fafc;
            color: #0f172a;
            border: 1px solid rgba(148, 163, 184, 0.55);
            border-radius: {UiScale.dp(8)}px;
            padding: {UiScale.dp(4)}px {UiScale.dp(12)}px;
            min-height: {control_height}px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: {UiScale.dp(30)}px;
            background-color: transparent;
        }}
        QComboBox QAbstractItemView {{
            background-color: #f8fafc;
            color: #0f172a;
            selection-background-color: rgba(56, 189, 248, 0.15);
            selection-color: #0f172a;
        }}
        QLineEdit:read-only {{
            background-color: rgba(248, 250, 252, 0.8);
        }}
        QCheckBox {{
            spacing: {UiScale.dp(12)}px;
            font-size: {UiScale.dp(14)}px;
            min-height: {control_height}px;
        }}
        QCheckBox::indicator {{
            width: {UiScale.dp(18)}px;
            height: {UiScale.dp(18)}px;
            border-radius: {UiScale.dp(4)}px;
            border: 2px solid rgba(226, 232, 240, 0.7);
            background-color: transparent;
        }}
        QCheckBox::indicator:checked {{
            background-color: #38bdf8;
            border-color: #38bdf8;
        }}
        QCheckBox::indicator:hover {{
            border-color: #38bdf8;
        }}
        QFrame#FlightControlCanvasContainer {{
            background-color: rgba(9, 34, 46, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: {radius_canvas}px;
        }}
        QFrame#FlightControlCanvas {{
            background-color: #f8fafc;
            border-radius: {UiScale.dp(14)}px;
        }}
        QLabel#FlightControlCanvasLabel {{
            color: #1e293b;
            font-size: {UiScale.dp(16)}px;
            font-weight: 500;
        }}
        """
        current = self.styleSheet()
        if current:
            self.setStyleSheet(current + stylesheet)
        else:
            self.setStyleSheet(stylesheet)

    def set_exit_callback(self, callback: Callable[[], None] | None) -> None:
        self._exit_callback = callback

    def set_connection_info(
        self,
        *,
        plugin_name: str | None = None,
        plugin_port: int | None = None,
    ) -> None:
        if plugin_name:
            self._plugin_name = plugin_name
            self.setWindowTitle(f"Flight Control Hub - {plugin_name}")
        if plugin_port is not None:
            self._plugin_port = plugin_port
            if self._port_edit is not None:
                self._port_edit.setText(str(plugin_port))
            self.set_status_message(f"Listening for clients on port {plugin_port}.")
        else:
            if self._port_edit is not None:
                self._port_edit.clear()

        context_parts = []
        if plugin_name:
            context_parts.append(plugin_name)
        if plugin_port is not None:
            context_parts.append(f"port {plugin_port}")
        if self._canvas is not None and context_parts:
            info = " • ".join(context_parts)
            self._canvas.set_placeholder_text(f"Awaiting telemetry from {info}.")
        elif self._canvas is not None:
            self._canvas.set_placeholder_text("Awaiting telemetry or video feeds.")

    def set_status_message(self, message: str) -> None:
        if self._status_label is not None:
            self._status_label.setText(message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if callable(self._exit_callback):
            try:
                self._exit_callback()
            except Exception:
                pass
        super().closeEvent(event)


def create_flight_control_dashboard(
    *, plugin_name: str | None = None, plugin_port: int | None = None
) -> FlightControlDashboard:
    window = FlightControlDashboard()
    window.set_connection_info(plugin_name=plugin_name, plugin_port=plugin_port)
    return window
