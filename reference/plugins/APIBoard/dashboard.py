from __future__ import annotations

import json
import socket
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from PySide6.QtCore import Qt, QSignalBlocker, QTimer, Signal, QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from plugins.Viewport.integration import HAS_AIRSIM, airsim


def _determine_local_ip() -> str:
    """Return the best-effort non-loopback IP for the current host."""
    try:
        hostname = socket.gethostname()
        candidate = socket.gethostbyname(hostname)
        if candidate and not candidate.startswith("127."):
            return candidate
    except Exception:
        candidate = ""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidate = sock.getsockname()[0]
    except Exception:
        candidate = "127.0.0.1"
    return candidate or "127.0.0.1"


def _to_serializable(value: Any) -> Any:
    """Convert AirSim return values into JSON-serialisable structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if len(items) > 120:
            trimmed = items[:120]
            trimmed.append(f"... ({len(items) - 120} more)")
            items = trimmed
        return [_to_serializable(v) for v in items]
    if hasattr(value, "_asdict"):
        return _to_serializable(value._asdict())
    if hasattr(value, "__dict__"):
        data = {
            key: _to_serializable(val)
            for key, val in value.__dict__.items()
            if not key.startswith("_")
        }
        if data:
            return data
    if hasattr(value, "to_dict"):
        try:
            return _to_serializable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


if HAS_AIRSIM:
    _SENSOR_TYPE_ENUM = getattr(airsim, "SensorType", None)
else:
    _SENSOR_TYPE_ENUM = None

_SENSOR_ENUM_NAMES = {
    "getImuData": "IMU",
    "getBarometerData": "Barometer",
    "getMagnetometerData": "Magnetometer",
    "getGpsData": "Gps",
    "getDistanceSensorData": "Distance",
    "getLidarData": "Lidar",
}

SENSOR_API_TYPES: Dict[str, int] = {}
if _SENSOR_TYPE_ENUM is not None:
    for api_name, enum_attr in _SENSOR_ENUM_NAMES.items():
        value = getattr(_SENSOR_TYPE_ENUM, enum_attr, None)
        if value is None:
            continue
        try:
            SENSOR_API_TYPES[api_name] = int(value)
        except (TypeError, ValueError):
            continue
else:
    SENSOR_API_TYPES = {}


class SelectableListWidget(QListWidget):
    """List widget that tracks manual selection and pin states."""

    selectionChanged = Signal(list)
    primaryItemChanged = Signal(object)

    STATE_ROLE = Qt.UserRole + 1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.NoSelection)
        self.setAlternatingRowColors(False)
        self.setFocusPolicy(Qt.NoFocus)
        self.setUniformItemSizes(True)
        self._last_active: Optional[QListWidgetItem] = None
        self.itemClicked.connect(self._handle_item_clicked)
        self.itemDoubleClicked.connect(self._handle_item_double_clicked)

    # Public API ---------------------------------------------------------
    def set_items(self, entries: Iterable[str]) -> None:
        self.clear()
        for text in entries:
            item = QListWidgetItem(text)
            item.setData(self.STATE_ROLE, {"selected": False, "pinned": False})
            item.setSizeHint(QSize(200, 34))
            self._apply_visual(item)
            self.addItem(item)
        self._last_active = None
        self.selectionChanged.emit([])
        self.primaryItemChanged.emit(None)

    def active_items(self) -> List[str]:
        return [
            item.text()
            for item in self._iter_items()
            if self._is_active(item)
        ]

    def primary_active_text(self) -> Optional[str]:
        if self._last_active and self._is_active(self._last_active):
            return self._last_active.text()
        for item in self._iter_items():
            if self._is_active(item):
                self._last_active = item
                return item.text()
        return None

    def set_all_selected(self, active: bool) -> None:
        for item in self._iter_items():
            state = self._state(item)
            if state["pinned"] and not active:
                continue
            self._set_state(item, selected=active or state["pinned"])
        if not active:
            self._last_active = None
        self._emit_state_change()

    def clear_states(self) -> None:
        for item in self._iter_items():
            state = self._state(item)
            state["selected"] = False
            state["pinned"] = False
            item.setData(self.STATE_ROLE, state)
            self._apply_visual(item)
        self._last_active = None
        self._emit_state_change()

    def all_active(self) -> bool:
        count = self.count()
        if count == 0:
            return False
        for item in self._iter_items():
            if not self._is_active(item):
                return False
        return True

    def has_items(self) -> bool:
        return self.count() > 0

    # Internal helpers ---------------------------------------------------
    def _iter_items(self):
        for idx in range(self.count()):
            yield self.item(idx)

    def _state(self, item: QListWidgetItem) -> Dict[str, bool]:
        state = item.data(self.STATE_ROLE)
        if not isinstance(state, dict):
            state = {"selected": False, "pinned": False}
        state.setdefault("selected", False)
        state.setdefault("pinned", False)
        return state

    def _set_state(
        self,
        item: QListWidgetItem,
        *,
        selected: Optional[bool] = None,
        pinned: Optional[bool] = None,
    ) -> None:
        state = self._state(item)
        if selected is not None:
            state["selected"] = selected
        if pinned is not None:
            state["pinned"] = pinned
            if pinned:
                state["selected"] = True
        item.setData(self.STATE_ROLE, state)
        self._apply_visual(item)

    def _apply_visual(self, item: QListWidgetItem) -> None:
        state = self._state(item)
        base = QColor(22, 26, 36)
        selected = QColor(36, 70, 107)
        pinned = QColor(53, 104, 83)
        color = (
            pinned if state["pinned"] else selected if state["selected"] else base
        )
        item.setBackground(color)
        item.setForeground(QColor(233, 239, 255))

    def _is_active(self, item: QListWidgetItem) -> bool:
        state = self._state(item)
        return state["selected"] or state["pinned"]

    def _handle_item_clicked(self, item: QListWidgetItem) -> None:
        # Single-click acts as a temporary focus; only pinned items persist across clicks.
        for other in self._iter_items():
            if other is item:
                continue
            other_state = self._state(other)
            if not other_state["pinned"]:
                self._set_state(other, selected=False)
        state = self._state(item)
        if not state["pinned"]:
            self._set_state(item, selected=True)
        self._last_active = item if self._is_active(item) else None
        self._emit_state_change()

    def _handle_item_double_clicked(self, item: QListWidgetItem) -> None:
        state = self._state(item)
        new_pinned = not state["pinned"]
        self._set_state(item, pinned=new_pinned)
        if new_pinned:
            self._set_state(item, selected=True)
        else:
            self._set_state(item, selected=False)
            if item is self._last_active:
                self._last_active = None
        self._last_active = item if self._is_active(item) else None
        self._emit_state_change()

    def _emit_state_change(self) -> None:
        active = self.active_items()
        self.selectionChanged.emit(active)
        self.primaryItemChanged.emit(
            self._last_active.text() if self._last_active and self._is_active(self._last_active) else None
        )


class ApiPanel(QFrame):
    """One half of the API Board layout."""

    vehicleSelectionChanged = Signal(list)
    dataSelectionChanged = Signal(list)
    activeDataChanged = Signal(object)
    frequencyChanged = Signal(int)
    generateRequested = Signal(object)
    codePreviewRequested = Signal(str)

    def __init__(
        self,
        title: str,
        button_text: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ApiPanel")
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._title_text = title
        self._button_text = button_text

        self.host_input = QLineEdit()
        self.port_input = QLineEdit()
        self.vehicle_list = SelectableListWidget()
        self.data_list = SelectableListWidget()
        self.detail_view = QPlainTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setObjectName("DetailView")
        self.detail_view.setWordWrapMode(self.detail_view.wordWrapMode())
        self.detail_view.setPlainText("Select a data item to preview.")

        self.vehicle_select_all = QCheckBox("Select All Vehicles")
        self.data_select_all = QCheckBox("Select All Data")
        self.frequency_spin = QSpinBox()
        self.frequency_spin.setRange(1, 30)
        self.frequency_spin.setValue(10)
        self.frequency_spin.setSuffix(" Hz")
        self.frequency_spin.setAccelerated(True)

        self.generate_button = QPushButton(button_text)
        self.generate_button.setObjectName("GenerateButton")
        self.generate_button.setMinimumWidth(200)
        self.generate_button.setCursor(Qt.PointingHandCursor)

        self.code_indicator = QPushButton()
        self.code_indicator.setFixedSize(24, 24)
        self.code_indicator.setCursor(Qt.PointingHandCursor)
        self.code_indicator.setFocusPolicy(Qt.NoFocus)
        self.code_indicator.setToolTip("No code generated yet")
        self._generated_code: Optional[str] = None
        self._update_code_indicator()

        self._build_ui()
        self._wire_signals()

    # Building and wiring ------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 24, 26, 24)
        outer.setSpacing(20)

        title_row = QHBoxLayout()
        title_label = QLabel(self._title_text)
        title_font = title_label.font()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_row.addStretch()
        title_row.addWidget(title_label, alignment=Qt.AlignCenter)
        title_row.addStretch()
        outer.addLayout(title_row)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        outer.addWidget(divider)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        grid.addWidget(QLabel("Host IP"), 0, 0)
        grid.addWidget(QLabel("TCP Port"), 0, 1)

        for line_edit in (self.host_input, self.port_input):
            line_edit.setReadOnly(True)
            line_edit.setAlignment(Qt.AlignCenter)
            line_edit.setStyleSheet("background: #2a2f3a; color: #f0f3ff; border: 1px solid #3f4656; padding: 6px;")

        grid.addWidget(self.host_input, 1, 0)
        grid.addWidget(self.port_input, 1, 1)
        outer.addLayout(grid)

        lists_layout = QHBoxLayout()
        lists_layout.setSpacing(18)

        lists_layout.addLayout(self._make_column_layout("Vehicle List", self.vehicle_select_all, self.vehicle_list))
        lists_layout.addLayout(self._make_column_layout("Data List", self.data_select_all, self.data_list))
        lists_layout.addLayout(self._make_detail_column())

        outer.addLayout(lists_layout, stretch=1)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(self.generate_button)
        controls.addWidget(self.code_indicator)
        bottom_row.addLayout(controls)
        outer.addLayout(bottom_row)

    def _make_column_layout(
        self,
        heading: str,
        checkbox: QCheckBox,
        widget: SelectableListWidget,
    ) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(10)

        header = QHBoxLayout()
        label = QLabel(heading)
        label_font = label.font()
        label_font.setPointSize(12)
        label.setFont(label_font)

        header.addWidget(label)
        header.addStretch()
        header.addWidget(checkbox)
        column.addLayout(header)

        widget.setStyleSheet("QListWidget { background: #1a1f2b; border: 1px solid #2f3746; }")
        widget.setMinimumWidth(200)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        column.addWidget(widget, stretch=1)
        return column

    def _make_detail_column(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(10)

        header = QHBoxLayout()
        label = QLabel("Detail Data")
        label_font = label.font()
        label_font.setPointSize(12)
        label.setFont(label_font)
        header.addWidget(label)
        header.addStretch()

        freq_label = QLabel("Update Frequency")
        header.addWidget(freq_label)
        header.addWidget(self.frequency_spin)
        column.addLayout(header)

        self.detail_view.setStyleSheet("QPlainTextEdit#DetailView { background: #1a1f2b; border: 1px solid #2f3746; color: #f0f3ff; }")
        column.addWidget(self.detail_view, stretch=1)
        return column

    def _wire_signals(self) -> None:
        self.vehicle_select_all.stateChanged.connect(self._handle_vehicle_select_all)
        self.data_select_all.stateChanged.connect(self._handle_data_select_all)
        self.vehicle_list.selectionChanged.connect(self._handle_vehicle_selection_change)
        self.data_list.selectionChanged.connect(self._handle_data_selection_change)
        self.data_list.primaryItemChanged.connect(self._handle_active_data_change)
        self.frequency_spin.valueChanged.connect(lambda _: self.set_generated_code(None))
        self.frequency_spin.valueChanged.connect(self.frequencyChanged.emit)
        self.generate_button.clicked.connect(self._on_generate_clicked)
        self.code_indicator.clicked.connect(self._on_code_indicator_clicked)

    # Public interface ---------------------------------------------------
    def set_connection_info(self, host: str, port: Optional[int]) -> None:
        self.host_input.setText(host or "--")
        self.port_input.setText(str(port) if port else "--")

    def set_vehicle_list(self, vehicles: Iterable[str]) -> None:
        self.vehicle_list.set_items(sorted({v for v in vehicles if v}))
        self._sync_checkbox(self.vehicle_list, self.vehicle_select_all)

    def set_data_options(self, options: Iterable[str]) -> None:
        options = list(dict.fromkeys(options))
        self.data_list.set_items(options)
        self._sync_checkbox(self.data_list, self.data_select_all)

    def active_vehicles(self) -> List[str]:
        return self.vehicle_list.active_items()

    def active_data(self) -> List[str]:
        return self.data_list.active_items()

    def primary_data(self) -> Optional[str]:
        return self.data_list.primary_active_text()

    def current_frequency(self) -> int:
        return max(1, self.frequency_spin.value())

    def set_detail_text(self, payload: str) -> None:
        self.detail_view.setPlainText(payload)

    def show_placeholder(self, message: str) -> None:
        self.detail_view.setPlainText(message)

    def disable_interaction(self, reason: Optional[str] = None) -> None:
        for widget in (self.vehicle_list, self.data_list, self.frequency_spin, self.generate_button, self.code_indicator):
            widget.setEnabled(False)
        self.set_generated_code(None)
        if reason:
            self.detail_view.setPlainText(reason)

    def set_generated_code(self, code: Optional[str]) -> None:
        self._generated_code = code
        self._update_code_indicator()

    # Internal signal handlers ------------------------------------------
    def _on_generate_clicked(self) -> None:
        self.set_generated_code(None)
        self.generateRequested.emit(self)

    def _on_code_indicator_clicked(self) -> None:
        if self._generated_code:
            self.codePreviewRequested.emit(self._generated_code)

    # Internal signal handlers ------------------------------------------
    def _handle_vehicle_select_all(self, state: int) -> None:
        self.set_generated_code(None)
        self.vehicle_list.set_all_selected(state == Qt.Checked)
        self._sync_checkbox(self.vehicle_list, self.vehicle_select_all)

    def _handle_data_select_all(self, state: int) -> None:
        self.set_generated_code(None)
        self.data_list.set_all_selected(state == Qt.Checked)
        self._sync_checkbox(self.data_list, self.data_select_all)

    def _handle_vehicle_selection_change(self, selection: list) -> None:
        self.set_generated_code(None)
        self._sync_checkbox(self.vehicle_list, self.vehicle_select_all)
        self.vehicleSelectionChanged.emit(selection)

    def _handle_data_selection_change(self, selection: list) -> None:
        self.set_generated_code(None)
        self._sync_checkbox(self.data_list, self.data_select_all)
        self.dataSelectionChanged.emit(selection)

    def _handle_active_data_change(self, active: object) -> None:
        self.set_generated_code(None)
        self.activeDataChanged.emit(active)

    def _sync_checkbox(self, widget: SelectableListWidget, checkbox: QCheckBox) -> None:
        with QSignalBlocker(checkbox):
            checkbox.setCheckState(Qt.Checked if widget.has_items() and widget.all_active() else Qt.Unchecked)

    def _update_code_indicator(self) -> None:
        active = bool(self._generated_code)
        palette_color = "#3fbf6b" if active else "#2f3746"
        border = "#54d98a" if active else "#465066"
        self.code_indicator.setEnabled(active)
        self.code_indicator.setStyleSheet(
            "QPushButton {"
            f" background-color: {palette_color};"
            f" border: 2px solid {border};"
            " border-radius: 4px;"
            "}"
            "QPushButton:pressed { background-color: #2d9b56; }"
        )
        self.code_indicator.setToolTip("Open generated code" if active else "No code generated yet")


class ApiBoardDashboard(QWidget):
    """Full API Board window combining Get/Control panels."""

    DEFAULT_AIRSIM_PORT = 41451
    SETTINGS_PATH = Path.home() / "Documents" / "AirSim" / "settings.json"
    DATA_APIS = [
        "simGetVehiclePose",
        "simGetGroundTruthKinematics",
        "simGetPhysicsRawKinematics",
        "simGetGroundTruthEnvironment",
        "getImuData",
        "getBarometerData",
        "getMagnetometerData",
        "getGpsData",
        "getDistanceSensorData",
        "getLidarData",
    ]
    SENSOR_DEFAULTS: Dict[str, str] = {
        "getImuData": "",
        "getBarometerData": "",
        "getMagnetometerData": "",
        "getGpsData": "",
        "getDistanceSensorData": "",
        "getLidarData": "",
    }

    def __init__(
        self,
        *,
        plugin_name: Optional[str] = None,
        plugin_port: Optional[int] = None,
        plugin_id: Optional[int] = None,
        comm_sender: Optional[Callable[[object], Optional[int]]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._plugin_name = plugin_name or "API Board"
        self._plugin_port = plugin_port
        self._plugin_id = plugin_id
        self._comm_sender = comm_sender
        self._last_stream_clients = 0
        self._host_ip = _determine_local_ip()
        self._settings_host: Optional[str] = None
        self._settings_ports: List[int] = []
        self._airsim_client = None
        self._connected_port: Optional[int] = None
        self._sensor_cache: Dict[str, Dict[int, List[str]]] = {}

        self._load_airsim_settings()
        if self._settings_host:
            self._host_ip = self._settings_host

        self.left_panel = ApiPanel("API To Get", 'Generate "Get" Code')
        self.right_panel = ApiPanel("API To Control", 'Generate "Control" Code')
        self.status_label = QLabel("")

        self._timers: Dict[str, QTimer] = {
            "get": QTimer(self),
            "control": QTimer(self),
        }
        for key, timer in self._timers.items():
            timer.setTimerType(Qt.CoarseTimer)
            timer.timeout.connect(lambda key=key: self._poll_data(key))

        self._active_data: Dict[str, Optional[str]] = {"get": None, "control": None}

        self._build_ui()
        self._connect_signals()

        self.left_panel.set_connection_info(self._host_ip, self._plugin_port)
        self.right_panel.set_connection_info(self._host_ip, self._plugin_port)
        self._update_status("Initialising AirSim connection...")
        QTimer.singleShot(200, self._initialize_connection)

    # UI construction ----------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("API Board" if not self._plugin_name else f"API Board - {self._plugin_name}")
        self.resize(1625, 900)
        self.setMinimumSize(1300, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        header = QLabel("API Board")
        header_font = header.font()
        header_font.setPointSize(24)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([1, 1])
        layout.addWidget(splitter, stretch=1)

        self.status_label.setObjectName("StatusLabel")
        status_font = self.status_label.font()
        status_font.setPointSize(10)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("color: #9fb2ff;")
        layout.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        self.left_panel.vehicleSelectionChanged.connect(lambda names: self._handle_vehicle_selection("get", names))
        self.right_panel.vehicleSelectionChanged.connect(lambda names: self._handle_vehicle_selection("control", names))
        self.left_panel.dataSelectionChanged.connect(lambda _: self._handle_data_selection("get"))
        self.right_panel.dataSelectionChanged.connect(lambda _: self._handle_data_selection("control"))
        self.left_panel.activeDataChanged.connect(lambda name: self._handle_active_data("get", name))
        self.right_panel.activeDataChanged.connect(lambda name: self._handle_active_data("control", name))
        self.left_panel.frequencyChanged.connect(lambda hz: self._handle_frequency_change("get", hz))
        self.right_panel.frequencyChanged.connect(lambda hz: self._handle_frequency_change("control", hz))
        self.left_panel.generateRequested.connect(lambda panel: self._handle_generate_code("get", panel))
        self.right_panel.generateRequested.connect(lambda panel: self._handle_generate_code("control", panel))
        self.left_panel.codePreviewRequested.connect(lambda code: self._show_generated_code_dialog(code, self.left_panel))
        self.right_panel.codePreviewRequested.connect(lambda code: self._show_generated_code_dialog(code, self.right_panel))

    # Configuration helpers ----------------------------------------------
    def _load_airsim_settings(self) -> None:
        self._settings_ports = []
        self._settings_host = None
        settings_path = self.SETTINGS_PATH
        if not settings_path.exists():
            return
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            return

        host = data.get("LocalHostIp") or data.get("local_host_ip")
        if isinstance(host, str) and host.strip():
            self._settings_host = host.strip()

        ports: List[int] = []
        root_port = data.get("RpcPort") or data.get("rpc_port")
        port = self._coerce_port(root_port)
        if port:
            ports.append(port)

        vehicles = data.get("Vehicles")
        if isinstance(vehicles, dict):
            for cfg in vehicles.values():
                if not isinstance(cfg, dict):
                    continue
                port = self._coerce_port(cfg.get("RpcPort") or cfg.get("rpc_port"))
                if port:
                    ports.append(port)

        unique_ports: List[int] = []
        for port in ports:
            if port not in unique_ports:
                unique_ports.append(port)
        self._settings_ports = unique_ports

    def _coerce_port(self, value: Any) -> Optional[int]:
        try:
            port = int(value)
        except (TypeError, ValueError):
            return None
        return port if port > 0 else None

    def _candidate_airsim_ports(self) -> List[int]:
        ports: List[int] = []

        def add(port: Optional[int], *, front: bool = False) -> None:
            if port is None:
                return
            if port <= 0:
                return
            if port in ports:
                return
            if front:
                ports.insert(0, port)
            else:
                ports.append(port)

        for port in self._settings_ports:
            add(port)

        add(self.DEFAULT_AIRSIM_PORT)
        add(self.DEFAULT_AIRSIM_PORT + 1)
        add(self.DEFAULT_AIRSIM_PORT + 2)

        if isinstance(self._plugin_port, int) and abs(int(self._plugin_port) - self.DEFAULT_AIRSIM_PORT) <= 10:
            add(int(self._plugin_port), front=True)

        return ports

    def _vehicle_sensors(self, vehicle: str) -> Dict[int | None, List[str]]:
        cached = self._sensor_cache.get(vehicle)
        if cached is not None:
            return cached
        sensors_by_type: Dict[int | None, List[str]] = {}
        client = self._airsim_client
        if client is None:
            self._sensor_cache[vehicle] = sensors_by_type
            return sensors_by_type
        try:
            sensor_names = client.simListSensors(vehicle)
        except Exception:
            self._sensor_cache[vehicle] = sensors_by_type
            return sensors_by_type
        for name in sensor_names or []:
            sensor_type = None
            try:
                info = client.simGetSensorInfo(name, vehicle)
            except Exception:
                info = None
            if info is not None:
                sensor_type = getattr(info, "sensor_type", None)
                if sensor_type is None:
                    sensor_type = getattr(info, "sensorType", None)
            key = sensor_type if isinstance(sensor_type, int) else None
            sensors_by_type.setdefault(key, []).append(str(name))
        self._sensor_cache[vehicle] = sensors_by_type
        return sensors_by_type

    def _resolve_sensor_name(self, vehicle: str, api_name: str) -> Optional[str]:
        sensors = self._vehicle_sensors(vehicle)
        sensor_type = SENSOR_API_TYPES.get(api_name)
        if sensor_type is not None:
            matches = sensors.get(sensor_type)
            if matches:
                return matches[0]
        all_names = [name for values in sensors.values() for name in values]
        fallback = self.SENSOR_DEFAULTS.get(api_name)
        if fallback and fallback in all_names:
            return fallback
        if sensor_type is not None and vehicle in self._sensor_cache:
            self._sensor_cache.pop(vehicle, None)
            sensors = self._vehicle_sensors(vehicle)
            matches = sensors.get(sensor_type)
            if matches:
                return matches[0]
            all_names = [name for values in sensors.values() for name in values]
            if fallback and fallback in all_names:
                return fallback
        # 최종 fallback: 기본 센서명을 그대로 쓰거나 빈 문자열 반환
        return fallback if fallback is not None else ""

    # Connection management ---------------------------------------------
    def _initialize_connection(self) -> None:
        if not HAS_AIRSIM:
            self.left_panel.disable_interaction("AirSim Python package is not available.")
            self.right_panel.disable_interaction("AirSim Python package is not available.")
            self._update_status("AirSim Python package not installed. Install `airsim` to enable live data.")
            return

        ports_to_try = self._candidate_airsim_ports()
        errors: List[str] = []
        client = None
        for port in ports_to_try:
            candidate = None
            try:
                candidate = airsim.MultirotorClient(ip=self._host_ip, port=port)
                candidate.confirmConnection()
                client = candidate
                self._connected_port = port
                break
            except Exception as exc:
                errors.append(f"{self._host_ip}:{port} -> {exc}")
                if candidate is not None:
                    with suppress(Exception):
                        candidate.close()

        if client is None:
            message = "Failed to connect to AirSim. " + "; ".join(errors) if errors else "Unknown error."
            self._update_status(message)
            self.left_panel.disable_interaction("Unable to connect to AirSim.")
            self.right_panel.disable_interaction("Unable to connect to AirSim.")
            self._last_stream_clients = 0
            return

        self._airsim_client = client
        self._last_stream_clients = 0
        self.left_panel.set_connection_info(self._host_ip, self._plugin_port)
        self.right_panel.set_connection_info(self._host_ip, self._plugin_port)
        self._update_status(f"Connected to AirSim at {self._host_ip}:{self._connected_port}")
        self._populate_vehicle_lists()

    def _populate_vehicle_lists(self) -> None:
        vehicles: List[str] = []
        if not self._airsim_client:
            return
        try:
            if hasattr(self._airsim_client, "listVehicles"):
                vehicles = list(self._airsim_client.listVehicles())
            elif hasattr(self._airsim_client, "simListVehicles"):
                vehicles = list(self._airsim_client.simListVehicles())
        except Exception as exc:
            self._update_status(f"Unable to list vehicles: {exc}")
            vehicles = []

        if not vehicles:
            self._update_status("No vehicles reported by AirSim.")
        self._sensor_cache.clear()
        self.left_panel.set_vehicle_list(vehicles)
        self.right_panel.set_vehicle_list(vehicles)

    # Event handlers -----------------------------------------------------
    def _handle_vehicle_selection(self, panel_key: str, vehicles: List[str]) -> None:
        panel = self._panel(panel_key)
        panel.set_generated_code(None)
        if vehicles:
            panel.set_data_options(self.DATA_APIS)
        else:
            panel.set_data_options([])
            panel.show_placeholder("Select a vehicle to inspect available APIs.")
        self._handle_data_selection(panel_key)

    def _handle_data_selection(self, panel_key: str) -> None:
        panel = self._panel(panel_key)
        panel.set_generated_code(None)
        if panel.active_data():
            panel.show_placeholder("Waiting to fetch data...")
        else:
            panel.show_placeholder("Select a data item to preview.")

    def _handle_active_data(self, panel_key: str, data_name: object) -> None:
        self._active_data[panel_key] = data_name if isinstance(data_name, str) else None
        timer = self._timers[panel_key]
        panel = self._panel(panel_key)
        panel.set_generated_code(None)
        if self._active_data[panel_key]:
            self._poll_data(panel_key)
            timer.start(self._interval_ms(panel_key))
        else:
            timer.stop()
            panel.show_placeholder("Select a data item to preview.")

    def _handle_frequency_change(self, panel_key: str, _: int) -> None:
        panel = self._panel(panel_key)
        panel.set_generated_code(None)
        timer = self._timers[panel_key]
        if timer.isActive():
            timer.start(self._interval_ms(panel_key))

    def _handle_generate_code(self, panel_key: str, panel: ApiPanel) -> None:
        vehicles = panel.active_vehicles()
        data_items = panel.active_data()
        if not vehicles:
            self._update_status("Select at least one vehicle before generating code.")
            panel.show_placeholder("Select vehicles before generating code.")
            panel.set_generated_code(None)
            return
        if not data_items:
            self._update_status("Select at least one data item before generating code.")
            panel.show_placeholder("Select data items before generating code.")
            panel.set_generated_code(None)
            return
        code = self._build_client_code(panel_key, vehicles, data_items)
        panel.set_generated_code(code)
        display_name = getattr(panel, "_title_text", "API")
        self._update_status(f"Generated client code for {display_name}. Click the green indicator to copy.")

    def _show_generated_code_dialog(self, code: str, panel: ApiPanel) -> None:
        dialog = QDialog(self)
        display_name = getattr(panel, "_title_text", "API")
        dialog.setWindowTitle(f"{display_name} Code Preview")
        layout = QVBoxLayout(dialog)
        editor = QPlainTextEdit(dialog)
        editor.setPlainText(code)
        editor.setReadOnly(True)
        editor.setMinimumSize(QSize(680, 420))
        layout.addWidget(editor)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        copy_button = button_box.addButton("Copy to Clipboard", QDialogButtonBox.ActionRole)
        button_box.rejected.connect(dialog.reject)

        def _copy_to_clipboard() -> None:
            app = QApplication.instance()
            if app is None:
                return
            clipboard = app.clipboard()
            clipboard.setText(code)

        copy_button.clicked.connect(_copy_to_clipboard)
        layout.addWidget(button_box)
        dialog.exec()

    def _build_client_code(self, panel_key: str, vehicles: List[str], data_items: List[str]) -> str:
        host = self._host_ip or "127.0.0.1"
        port = self._plugin_port or self._connected_port or self.DEFAULT_AIRSIM_PORT
        vehicles_literal = json.dumps(sorted(vehicles), ensure_ascii=False)
        data_literal = json.dumps(sorted(data_items), ensure_ascii=False)
        code_lines = [
            "# -*- coding: utf-8 -*-",
            '"""Client template for the ODT API Board stream."""',
            "import json",
            "import socket",
            "import time",
            "from datetime import datetime",
            "from pathlib import Path",
            "",
            f'STREAM_HOST = "{host}"',
            f"STREAM_PORT = {port}",
            f"SUBSCRIBED_VEHICLES = {vehicles_literal}",
            f"SUBSCRIBED_DATA = {data_literal}",
            f'EXPECTED_PANEL = "{panel_key}"',
            "",
            "OUTPUT_ROOT = Path(\"api_board_exports\")",
            "OUTPUT_DIR = OUTPUT_ROOT / datetime.now().strftime(\"%Y%m%d_%H%M%S\")",
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            "def connect():",
            "    sock = socket.create_connection((STREAM_HOST, STREAM_PORT), timeout=10)",
            "    sock.sendall(b\"SUBSCRIBE\\n\")",
            "    return sock, sock.makefile(\"r\", encoding=\"utf-8\")",
            "",
            "def ensure_handle(handles: dict, vehicle: str):",
            "    if vehicle not in handles:",
            "        path = OUTPUT_DIR / f\"{vehicle}.txt\"",
            "        handles[vehicle] = path.open(\"a\", encoding=\"utf-8\")",
            "    return handles[vehicle]",
            "",
            "def should_keep(record: dict) -> bool:",
            "    if SUBSCRIBED_VEHICLES and record.get(\"vehicle\") not in SUBSCRIBED_VEHICLES:",
            "        return False",
            "    if SUBSCRIBED_DATA and record.get(\"data_name\") not in SUBSCRIBED_DATA:",
            "        return False",
            "    if EXPECTED_PANEL and record.get(\"panel\") != EXPECTED_PANEL:",
            "        return False",
            "    return True",
            "",
            "def main():",
            "    files = {}",
            "    sock = None",
            "    reader = None",
            "    try:",
            "        while True:",
            "            try:",
            "                if sock is None:",
            "                    sock, reader = connect()",
            "                raw_line = reader.readline()",
            "                if not raw_line:",
            "                    raise ConnectionError(\"stream closed\")",
            "                payload = raw_line.strip()",
            "                if not payload:",
            "                    continue",
            "                try:",
            "                    record = json.loads(payload)",
            "                except json.JSONDecodeError:",
            "                    continue",
            "                if not should_keep(record):",
            "                    continue",
            "                vehicle = record.get(\"vehicle\", \"unknown\")",
            "                handle = ensure_handle(files, vehicle)",
            "                handle.write(json.dumps(record, ensure_ascii=False) + \"\\n\")",
            "                handle.flush()",
            "            except KeyboardInterrupt:",
            "                break",
            "            except Exception as exc:",
            "                print(f\"Stream error: {exc}. Retrying in 2 seconds...\")",
            "                time.sleep(2.0)",
            "                if reader:",
            "                    try:",
            "                        reader.close()",
            "                    except Exception:",
            "                        pass",
            "                    reader = None",
            "                if sock:",
            "                    try:",
            "                        sock.close()",
            "                    except Exception:",
            "                        pass",
            "                    sock = None",
            "    finally:",
            "        for handle in files.values():",
            "            try:",
            "                handle.close()",
            "            except Exception:",
            "                pass",
            "        if reader:",
            "            try:",
            "                reader.close()",
            "            except Exception:",
            "                pass",
            "        if sock:",
            "            try:",
            "                sock.close()",
            "            except Exception:",
            "                pass",
            "",
            "if __name__ == \"__main__\":",
            "    main()",
        ]
        return "\n".join(code_lines)
    # Data polling -------------------------------------------------------
    def _poll_data(self, panel_key: str) -> None:
        panel = self._panel(panel_key)
        if not self._airsim_client:
            panel.show_placeholder("AirSim connection is not available.")
            self._timers[panel_key].stop()
            return
        data_name = self._active_data.get(panel_key)
        if not data_name:
            panel.show_placeholder("Select a data item to preview.")
            self._timers[panel_key].stop()
            return
        vehicles = panel.active_vehicles()
        if not vehicles:
            panel.show_placeholder("Select at least one vehicle.")
            return
        payload: Dict[str, Any] = {}
        for vehicle in vehicles:
            payload[vehicle] = self._fetch_data_point(data_name, vehicle)
        self._push_network_payload(panel_key, data_name, payload)
        if len(payload) == 1:
            value = next(iter(payload.values()))
        else:
            value = payload
        try:
            text = json.dumps(_to_serializable(value), indent=2, default=str)
        except Exception:
            text = str(value)
        panel.set_detail_text(text)

    def _push_network_payload(self, panel_key: str, data_name: str, payload_map: Dict[str, Any]) -> None:
        if not self._comm_sender or not payload_map:
            return
        panel = self._panel(panel_key)
        frequency = panel.current_frequency()
        timestamp = time.time()
        latest_delivered: Optional[int] = None
        for vehicle, data in payload_map.items():
            record = {
                "plugin": self._plugin_name,
                "plugin_id": self._plugin_id,
                "host": self._host_ip,
                "port": self._plugin_port or self._connected_port,
                "panel": panel_key,
                "data_name": data_name,
                "vehicle": vehicle,
                "timestamp": timestamp,
                "frequency_hz": frequency,
                "payload": _to_serializable(data),
            }
            try:
                delivered = self._comm_sender(json.dumps(record, ensure_ascii=False))
            except Exception as exc:
                self._update_status(f"Failed to stream {data_name}: {exc}")
                return
            if isinstance(delivered, int):
                latest_delivered = delivered if latest_delivered is None else max(latest_delivered, delivered)
        if isinstance(latest_delivered, int):
            if latest_delivered > 0 and self._last_stream_clients == 0:
                self._update_status(f"Streaming {data_name} updates to {latest_delivered} client(s).")
            elif latest_delivered == 0 and self._last_stream_clients > 0:
                self._update_status("API Board stream idle - waiting for clients.")
            self._last_stream_clients = latest_delivered
    def _fetch_data_point(self, api_name: str, vehicle: str) -> Any:
        client = self._airsim_client
        if client is None:
            return {"error": "No AirSim client"}
        try:
            if api_name == "simGetVehiclePose":
                return client.simGetVehiclePose(vehicle_name=vehicle)
            if api_name == "simGetGroundTruthKinematics":
                return client.simGetGroundTruthKinematics(vehicle_name=vehicle)
            if api_name == "simGetPhysicsRawKinematics":
                return client.simGetPhysicsRawKinematics(vehicle_name=vehicle)
            if api_name == "simGetGroundTruthEnvironment":
                return client.simGetGroundTruthEnvironment()
            if api_name in self.SENSOR_DEFAULTS or api_name in SENSOR_API_TYPES:
                sensor = self._resolve_sensor_name(vehicle, api_name)
                if sensor is None:
                    return {"error": f"No sensor available for {api_name} on {vehicle}"}
                getter = getattr(client, api_name)
                return getter(sensor, vehicle_name=vehicle)
            getter = getattr(client, api_name)
            return getter(vehicle)
        except Exception as exc:
            return {"error": str(exc)}

    # Helpers ------------------------------------------------------------
    def _panel(self, key: str) -> ApiPanel:
        return self.left_panel if key == "get" else self.right_panel

    def _interval_ms(self, panel_key: str) -> int:
        hz = self._panel(panel_key).current_frequency()
        interval = max(1, int(1000 / max(1, hz)))
        return interval

    def _update_status(self, message: str) -> None:
        self.status_label.setText(message)


def create_api_board_dashboard(
    *,
    plugin_name: Optional[str] = None,
    plugin_port: Optional[int] = None,
    plugin_id: Optional[int] = None,
    comm_sender: Optional[Callable[[object], Optional[int]]] = None,
) -> ApiBoardDashboard:
    return ApiBoardDashboard(
        plugin_name=plugin_name,
        plugin_port=plugin_port,
        plugin_id=plugin_id,
        comm_sender=comm_sender,
    )
















