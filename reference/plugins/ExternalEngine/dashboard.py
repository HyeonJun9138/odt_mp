from __future__ import annotations

import json
import math
import socket
from typing import Optional, Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import airsim  # type: ignore

    HAS_AIRSIM = True
except Exception:  # pragma: no cover - optional dependency
    airsim = None  # type: ignore
    HAS_AIRSIM = False


def _determine_local_ip() -> str:
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


class AirSimAPIWrapper:
    def __init__(self) -> None:
        self.client = None
        self.connected = False

    def connect(self, host: str, port: int) -> tuple[bool, Optional[str]]:
        if not HAS_AIRSIM:
            return False, "AirSim Python package is not installed."
        try:
            self.client = airsim.MultirotorClient(ip=host or "127.0.0.1", port=int(port))
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.connected = True
            return True, None
        except Exception as exc:  # pragma: no cover - network
            self.connected = False
            self.client = None
            return False, str(exc)

    def disconnect(self) -> None:
        self.connected = False
        self.client = None

    def send_state(
        self,
        position: tuple[float, float, float],
        attitude_rad: tuple[float, float, float],
        vehicle: str,
    ) -> Optional[str]:
        if not HAS_AIRSIM:
            return "AirSim Python package is missing."
        if not self.connected or self.client is None:
            return "A connection to AirSim is not established."
        try:
            pose = airsim.Pose(
                airsim.Vector3r(position[0], position[1], position[2]),
                airsim.to_quaternion(attitude_rad[0], attitude_rad[1], attitude_rad[2]),
            )
            self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle or "UAM1")
            return None
        except Exception as exc:  # pragma: no cover - network
            return str(exc)


class ExternalEngineDashboard(QWidget):
    """Dashboard that bridges external dynamics engines to AirSim."""

    def __init__(
        self,
        *,
        plugin_name: Optional[str] = None,
        plugin_port: Optional[int] = None,
        plugin_id: Optional[int] = None,
        comm_sender=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._plugin_name = plugin_name or "External Engine Bridge"
        self._plugin_port = plugin_port
        self._plugin_id = plugin_id
        self._comm_sender = comm_sender
        self._local_ip = _determine_local_ip()
        self._api = AirSimAPIWrapper()
        self._pending_buffer = ""

        self._state_labels: Dict[str, QLabel] = {}
        self._state_label_formats: Dict[str, str] = {}
        
        self._ingest_count = 0          # 1초 동안 들어온 JSON 라인 수
        self._applied_count = 0         # 1초 동안 AirSim에 실제 적용한 횟수
        self._rpc_durations_ms = []     # RPC 소요시간(ms) 리스트
        self._last_stats_ts = 0.0       # 마지막 통계 출력 시각 (perf_counter 기준)

        self._build_ui()
        self._refresh_plugin_info()
        self._update_status("Awaiting AirSim connection.")

        if not HAS_AIRSIM:
            self._update_status("AirSim Python package not detected. Install `airsim` to enable control.")
            self.connect_button.setEnabled(False)
            self.send_button.setEnabled(False)

    # --- UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle(self._plugin_name)
        self.resize(1200, 850)
        self.setMinimumSize(1024, 850)

        self._state_labels = {}
        self._state_label_formats = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        title = QLabel("External Engine Bridge")
        title_font = title.font()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        info_frame = QFrame()
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setHorizontalSpacing(18)
        info_layout.setVerticalSpacing(8)

        info_layout.addWidget(QLabel("Plugin Host"), 0, 0)
        self.plugin_host_label = QLabel("")
        self.plugin_host_label.setObjectName("PluginInfoLabel")
        info_layout.addWidget(self.plugin_host_label, 0, 1)

        info_layout.addWidget(QLabel("Plugin Port"), 0, 2)
        self.plugin_port_label = QLabel("Pending")
        self.plugin_port_label.setObjectName("PluginInfoLabel")
        info_layout.addWidget(self.plugin_port_label, 0, 3)

        info_layout.addWidget(QLabel("Vehicle Name"), 1, 0)
        self.vehicle_input = QLineEdit("UAM1")
        info_layout.addWidget(self.vehicle_input, 1, 1)

        info_layout.addWidget(QLabel("Units"), 1, 2)
        self.deg_checkbox = QCheckBox("Angles in degrees")
        self.deg_checkbox.setChecked(True)
        info_layout.addWidget(self.deg_checkbox, 1, 3)

        layout.addWidget(info_frame)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        connection_group = QGroupBox("AirSim Connection")
        connection_layout = QGridLayout(connection_group)
        connection_layout.setContentsMargins(12, 12, 12, 12)
        connection_layout.setHorizontalSpacing(12)

        connection_layout.addWidget(QLabel("RPC Host"), 0, 0)
        self.airsim_host_input = QLineEdit("127.0.0.1")
        connection_layout.addWidget(self.airsim_host_input, 0, 1)

        connection_layout.addWidget(QLabel("RPC Port"), 0, 2)
        self.airsim_port_spin = QSpinBox()
        self.airsim_port_spin.setRange(1, 65535)
        self.airsim_port_spin.setValue(41451)
        connection_layout.addWidget(self.airsim_port_spin, 0, 3)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._on_connect_clicked)
        connection_layout.addWidget(self.connect_button, 1, 0, 1, 4)

        layout.addWidget(connection_group)

        state_group = QGroupBox("Last External State")
        state_layout = QGridLayout(state_group)
        state_layout.setContentsMargins(12, 12, 12, 12)
        state_layout.setHorizontalSpacing(18)
        state_layout.setVerticalSpacing(6)

        def add_state(row: int, col: int, label_text: str, key: str, fmt: str | None = None) -> None:
            title = QLabel(label_text)
            fmt_str = fmt or "{:.3f}"
            self._state_label_formats[key] = fmt_str
            value_label = self._make_state_value_label()
            self._state_labels[key] = value_label
            state_layout.addWidget(title, row, col * 2)
            state_layout.addWidget(value_label, row, col * 2 + 1)

        add_state(0, 0, "North (m)", "north")
        add_state(1, 0, "East (m)", "east")
        add_state(2, 0, "Down (m)", "down")
        add_state(3, 0, "Altitude (m)", "alt", "{:.2f}")
        add_state(0, 1, "Roll", "roll")
        add_state(1, 1, "Pitch", "pitch")
        add_state(2, 1, "Yaw", "yaw")
        add_state(3, 1, "Latitude", "lat", "{:.6f}")
        add_state(4, 1, "Longitude", "lon", "{:.6f}")

        layout.addWidget(state_group)
        self._reset_state_display()

        pose_group = QGroupBox("Manual Pose Injection")
        pose_layout = QGridLayout(pose_group)
        pose_layout.setContentsMargins(12, 12, 12, 12)
        pose_layout.setHorizontalSpacing(12)
        pose_layout.setVerticalSpacing(10)

        self.position_boxes: Dict[str, QDoubleSpinBox] = {}
        self.attitude_boxes: Dict[str, QDoubleSpinBox] = {}

        for idx, axis in enumerate(("X (North)", "Y (East)", "Z (Down)")):
            box = self._make_spinbox()
            pose_layout.addWidget(QLabel(axis), 0, idx)
            pose_layout.addWidget(box, 1, idx)
            self.position_boxes[axis[0].lower()] = box

        for idx, axis in enumerate(("Pitch", "Roll", "Yaw")):
            box = self._make_spinbox()
            pose_layout.addWidget(QLabel(axis), 2, idx)
            pose_layout.addWidget(box, 3, idx)
            self.attitude_boxes[axis.lower()] = box

        controls_row = QHBoxLayout()
        self.send_button = QPushButton("Send State")
        self.send_button.clicked.connect(self._on_send_clicked)
        self.send_button.setEnabled(False)
        controls_row.addWidget(self.send_button)

        self.reset_button = QPushButton("Reset Inputs")
        self.reset_button.clicked.connect(self._reset_inputs)
        controls_row.addWidget(self.reset_button)
        controls_row.addStretch()

        pose_layout.addLayout(controls_row, 4, 0, 1, 3)
        layout.addWidget(pose_group)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Connection and command logs will appear here.")
        self.log_view.setMinimumHeight(140)
        layout.addWidget(self.log_view, stretch=1)

        instructions_group = QGroupBox("Integration Snippet")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_layout.setContentsMargins(12, 12, 12, 12)
        instructions_layout.setSpacing(8)

        self.instructions_view = QPlainTextEdit("")
        self.instructions_view.setReadOnly(True)
        self.instructions_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.instructions_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        instructions_layout.addWidget(self.instructions_view, stretch=1)

        instruction_buttons = QHBoxLayout()
        self.copy_code_button = QPushButton("Copy to Clipboard")
        self.copy_code_button.clicked.connect(self._copy_instructions)
        instruction_buttons.addWidget(self.copy_code_button)
        instruction_buttons.addStretch()
        instructions_layout.addLayout(instruction_buttons)

        layout.addWidget(instructions_group, stretch=1)

    def _make_spinbox(self) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-100000.0, 100000.0)
        box.setDecimals(4)
        box.setSingleStep(0.1)
        box.setValue(0.0)
        return box

    def _make_state_value_label(self) -> QLabel:
        label = QLabel("--")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setObjectName("StateValueLabel")
        return label

    def _set_state_label(self, key: str, value: Optional[float]) -> None:
        label = self._state_labels.get(key)
        if not label:
            return
        fmt = self._state_label_formats.get(key, "{:.3f}")
        if value is None:
            label.setText("--")
            return
        try:
            label.setText(fmt.format(value))
        except Exception:
            label.setText(str(value))

    def _reset_state_display(self) -> None:
        for key in self._state_labels.keys():
            self._set_state_label(key, None)

    def _format_angle(self, value_rad: Optional[float]) -> str:
        if value_rad is None:
            return "--"
        try:
            deg = math.degrees(value_rad)
            return f"{value_rad:.4f} rad / {deg:.2f}°"
        except Exception:
            return str(value_rad)

    def _update_state_display(
        self,
        pos_meta: Dict[str, Optional[float]],
        att_rad_tuple: Optional[tuple[float, float, float]],
    ) -> None:
        self._set_state_label("north", pos_meta.get("north"))
        self._set_state_label("east", pos_meta.get("east"))
        self._set_state_label("down", pos_meta.get("down"))
        self._set_state_label("alt", pos_meta.get("alt"))
        self._set_state_label("lat", pos_meta.get("lat"))
        self._set_state_label("lon", pos_meta.get("lon"))

        roll_label = self._state_labels.get("roll")
        pitch_label = self._state_labels.get("pitch")
        yaw_label = self._state_labels.get("yaw")

        if att_rad_tuple is None:
            if roll_label:
                roll_label.setText("--")
            if pitch_label:
                pitch_label.setText("--")
            if yaw_label:
                yaw_label.setText("--")
            return

        pitch_rad, roll_rad, yaw_rad = att_rad_tuple
        if roll_label:
            roll_label.setText(self._format_angle(roll_rad))
        if pitch_label:
            pitch_label.setText(self._format_angle(pitch_rad))
        if yaw_label:
            yaw_label.setText(self._format_angle(yaw_rad))

    def _normalize_position(self, payload: object) -> tuple[Optional[tuple[float, float, float]], Dict[str, Optional[float]]]:
        meta: Dict[str, Optional[float]] = {
            "north": None,
            "east": None,
            "down": None,
            "lat": None,
            "lon": None,
            "alt": None,
        }
        if payload is None:
            return None, meta

        if isinstance(payload, (list, tuple)):
            try:
                values = tuple(float(v) for v in payload)
            except (TypeError, ValueError):
                return None, meta
            if len(values) != 3:
                return None, meta
            meta["north"], meta["east"], meta["down"] = values
            return values, meta

        if isinstance(payload, dict):
            lower: Dict[str, object] = {}
            for key, value in payload.items():
                if isinstance(key, str):
                    lower[key.lower()] = value
                else:
                    lower[str(key).lower()] = value

            mapping = {
                "north": "north",
                "n": "north",
                "x": "north",
                "east": "east",
                "e": "east",
                "y": "east",
                "down": "down",
                "d": "down",
                "z": "down",
            }
            for src, dst in mapping.items():
                if src in lower and meta[dst] is None:
                    try:
                        meta[dst] = float(lower[src])  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue

            for key in ("lat", "latitude"):
                if key in lower and meta["lat"] is None:
                    try:
                        meta["lat"] = float(lower[key])  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue
            for key in ("lon", "lng", "longitude"):
                if key in lower and meta["lon"] is None:
                    try:
                        meta["lon"] = float(lower[key])  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue
            for key in ("alt", "altitude"):
                if key in lower and meta["alt"] is None:
                    try:
                        meta["alt"] = float(lower[key])  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue

            if all(meta[k] is not None for k in ("north", "east", "down")):
                return (meta["north"], meta["east"], meta["down"]), meta  # type: ignore[return-value]
            return None, meta

        return None, meta

    def _normalize_attitude(
        self,
        payload: object,
        degrees_hint: bool,
        message: Dict[str, object],
    ) -> tuple[Optional[tuple[float, float, float]], bool]:
        deg_flag = bool(degrees_hint)

        def unit_hint_to_deg(value: object | None) -> Optional[bool]:
            if value is None:
                return None
            text = str(value).strip().lower()
            if text.startswith("deg") or text in ("degree", "degrees"):
                return True
            if text.startswith("rad") or text in ("radian", "radians"):
                return False
            return None

        for key in ("attitude_units", "att_units", "attitude_unit", "attitudeunits", "angle_units"):
            flag = unit_hint_to_deg(message.get(key))
            if flag is not None:
                deg_flag = flag
                break

        if payload is None:
            return None, deg_flag

        if isinstance(payload, (list, tuple)):
            try:
                values = tuple(float(v) for v in payload)
            except (TypeError, ValueError):
                return None, deg_flag
            if len(values) != 3:
                return None, deg_flag
            return values, deg_flag

        if isinstance(payload, dict):
            lower: Dict[str, object] = {}
            for key, value in payload.items():
                if isinstance(key, str):
                    lower[key.lower()] = value
                else:
                    lower[str(key).lower()] = value

            payload_flag = unit_hint_to_deg(
                lower.get("units") or lower.get("unit") or lower.get("angle_units")
            )
            if payload_flag is not None:
                deg_flag = payload_flag

            aliases = {
                "roll": ("roll", "phi"),
                "pitch": ("pitch", "theta"),
                "yaw": ("yaw", "psi", "heading"),
            }
            result: Dict[str, float] = {}
            for component, names in aliases.items():
                value = None
                comp_flag: Optional[bool] = None
                for name in names:
                    key = name.lower()
                    if key in lower:
                        try:
                            value = float(lower[key])  # type: ignore[arg-type]
                            break
                        except (TypeError, ValueError):
                            continue
                    key_deg = f"{key}_deg"
                    if key_deg in lower:
                        try:
                            value = float(lower[key_deg])  # type: ignore[arg-type]
                            comp_flag = True
                            break
                        except (TypeError, ValueError):
                            continue
                    key_rad = f"{key}_rad"
                    if key_rad in lower:
                        try:
                            value = float(lower[key_rad])  # type: ignore[arg-type]
                            comp_flag = False
                            break
                        except (TypeError, ValueError):
                            continue
                if value is not None:
                    result[component] = value
                    if comp_flag is True:
                        deg_flag = True
                    elif comp_flag is False and comp_flag is not None:
                        deg_flag = False

            if len(result) == 3:
                return (result["pitch"], result["roll"], result["yaw"]), deg_flag
            return None, deg_flag

        return None, deg_flag
    # --- External interface --------------------------------------------------
    def set_connection_info(
        self,
        *,
        plugin_name: str | None = None,
        plugin_port: int | None = None,
    ) -> None:
        if plugin_name:
            self._plugin_name = plugin_name
            self.setWindowTitle(self._plugin_name)
        if plugin_port is not None:
            self._plugin_port = plugin_port
        self._refresh_plugin_info()

    def handle_comm_message(self, payload: bytes | None, *, text: Optional[str] = None) -> None:
        message_text = text
        if message_text is None and payload is not None:
            try:
                message_text = payload.decode("utf-8")
            except UnicodeDecodeError:
                message_text = payload.decode("utf-8", "ignore")
        if not message_text:
            return

        self._pending_buffer += message_text

        # 1초 통계 틱 체크
        self._maybe_emit_stats()

        while True:
            if "\n" not in self._pending_buffer:
                break
            line, self._pending_buffer = self._pending_buffer.split("\n", 1)
            packet = line.strip()
            if not packet:
                continue

            # (추가) 수신 라인 카운트
            self._ingest_count += 1

            try:
                message = json.loads(packet)
            except json.JSONDecodeError as exc:
                self._append_log(f"Invalid JSON payload: {exc}", level="error")
                continue

            # 기존 동작: 즉시 처리 → 적용 횟수/시간은 아래 메서드에서 집계
            self._process_external_message(message)

    # --- Actions -------------------------------------------------------------
    def _on_connect_clicked(self) -> None:
        if self._api.connected:
            self._api.disconnect()
            self._update_status("Disconnected from AirSim.")
            self.connect_button.setText("Connect")
            self.send_button.setEnabled(False)
            self._append_log("Disconnected from AirSim.", level="info")
            return

        host = self.airsim_host_input.text().strip() or "127.0.0.1"
        port = self.airsim_port_spin.value()
        ok, error = self._api.connect(host, port)
        if ok:
            self._update_status(f"Connected to AirSim at {host}:{port}.")
            self.connect_button.setText("Disconnect")
            self.send_button.setEnabled(True)
            self._append_log(f"Connected to AirSim at {host}:{port}", level="success")
        else:
            self._update_status(f"Connection failed: {error}")
            self._append_log(f"Failed to connect to AirSim: {error}", level="error")

    def _on_send_clicked(self) -> None:
        if not self._api.connected:
            self._append_log("Connect to AirSim before sending states.", level="warning")
            return
        position = (
            self.position_boxes["x"].value(),
            self.position_boxes["y"].value(),
            self.position_boxes["z"].value(),
        )
        attitude = (
            self.attitude_boxes["pitch"].value(),
            self.attitude_boxes["roll"].value(),
            self.attitude_boxes["yaw"].value(),
        )
        if self.deg_checkbox.isChecked():
            attitude = tuple(math.radians(v) for v in attitude)

        vehicle = self.vehicle_input.text().strip() or "UAM1"
        error = self._api.send_state(position, attitude, vehicle)
        if error:
            self._append_log(f"Failed to send state: {error}", level="error")
            self._update_status(f"Send failed: {error}")
        else:
            self._append_log(
                f"Pose applied to {vehicle}: pos={tuple(round(v, 3) for v in position)}, "
                f"att(rad)={tuple(round(v, 3) for v in attitude)}",
                level="success",
            )
            self._update_status("Last command delivered to AirSim.")

    def _reset_inputs(self) -> None:
        for box in self.position_boxes.values():
            box.setValue(0.0)
        for box in self.attitude_boxes.values():
            box.setValue(0.0)
        self._append_log("Inputs reset to zero.", level="info")

    def _copy_instructions(self) -> None:
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.instructions_view.toPlainText())
        self._append_log("Instruction snippet copied to clipboard.", level="info")

    # --- Helpers -------------------------------------------------------------
    def _refresh_plugin_info(self) -> None:
        self.plugin_host_label.setText(self._local_ip)
        port_text = str(self._plugin_port) if self._plugin_port is not None else "Pending"
        self.plugin_port_label.setText(port_text)
        self.instructions_view.setPlainText(self._build_instruction_text())

    def _process_external_message(self, message: dict) -> None:
        if not self._api.connected:
            self._append_log("Pose received but AirSim is not connected.", level="warning")
            return

        vehicle = message.get("vehicle") or self.vehicle_input.text().strip() or "UAM1"
        pos_payload = message.get("pos") or message.get("position")
        att_payload = None
        degrees_flag = False

        if "att_deg" in message:
            att_payload = message.get("att_deg")
            degrees_flag = True
        elif "att_rad" in message:
            att_payload = message.get("att_rad")
        else:
            att_payload = message.get("att") or message.get("attitude")
            degrees_flag = bool(
                message.get("degrees") or message.get("deg") or str(message.get("units", "")).lower() == "deg"
            )

        pos_tuple, pos_meta = self._normalize_position(pos_payload)
        att_raw_tuple, degrees_flag = self._normalize_attitude(att_payload, degrees_flag, message)

        att_rad_tuple: Optional[tuple[float, float, float]] = None
        if att_raw_tuple is not None:
            att_rad_tuple = tuple(math.radians(v) for v in att_raw_tuple) if degrees_flag else att_raw_tuple

        self._update_state_display(pos_meta, att_rad_tuple)

        if pos_tuple is None:
            self._append_log("Message missing 'pos'/'position' north/east/down values.", level="warning")
            return

        if att_rad_tuple is None:
            self._append_log("Message missing roll/pitch/yaw fields.", level="warning")
            return

        att_tuple = att_rad_tuple

        # --- (추가) RPC 호출 시간 측정 ---
        import time
        t0 = time.perf_counter()
        error = self._api.send_state(pos_tuple, att_tuple, vehicle)
        t1 = time.perf_counter()
        self._rpc_durations_ms.append((t1 - t0) * 1000.0)  # ms

        if error:
            self._append_log(f"Failed to apply external pose: {error}", level="error")
            self._update_status(f"External command failed: {error}")
        else:
            # (추가) 적용 카운트
            self._applied_count += 1

            pos_log = tuple(round(v, 3) for v in pos_tuple)
            att_log = tuple(round(v, 3) for v in att_tuple)
            att_log_deg = tuple(round(math.degrees(v), 2) for v in att_tuple)
            extras = []
            lat = pos_meta.get("lat")
            lon = pos_meta.get("lon")
            alt = pos_meta.get("alt")
            if lat is not None and lon is not None:
                extras.append(f"lat/lon=({lat:.6f}, {lon:.6f})")
            if alt is not None:
                extras.append(f"alt={alt:.2f}")
            extra_str = (" " + " ".join(extras)) if extras else ""

            self._append_log(
                f"External pose applied to {vehicle}: pos={pos_log}, "
                f"att(rad)={att_log}, att(deg)={att_log_deg}{extra_str}",
                level="success",
            )
            self._update_status("External command delivered to AirSim.")

        if "lqr" in message:
            self._append_log(f"LQR payload received for {vehicle}: {repr(message['lqr'])}", level="info")

        # 1초 통계 틱 체크
        self._maybe_emit_stats()

    def _maybe_emit_stats(self) -> None:
        """1초마다 ingest/apply Hz와 RPC 시간 통계를 로그로 출력."""
        import time, statistics

        now = time.perf_counter()
        if self._last_stats_ts == 0.0:
            self._last_stats_ts = now
            return

        elapsed = now - self._last_stats_ts
        if elapsed < 1.0:
            return

        ingest_hz = self._ingest_count / elapsed if elapsed > 0 else 0.0
        applied_hz = self._applied_count / elapsed if elapsed > 0 else 0.0
        backlog = self._ingest_count - self._applied_count

        rpc_mean = rpc_p95 = rpc_max = 0.0
        if self._rpc_durations_ms:
            vals = list(self._rpc_durations_ms)
            vals.sort()
            rpc_mean = statistics.fmean(vals)
            rpc_max = max(vals)
            # p95
            idx95 = max(0, int(round(0.95 * (len(vals) - 1))))
            rpc_p95 = vals[idx95]

        self._append_log(
            f"[STATS] ingest={ingest_hz:.1f} Hz, applied={applied_hz:.1f} Hz, "
            f"rpc_mean={rpc_mean:.1f} ms, rpc_p95={rpc_p95:.1f} ms, backlog={backlog:+d}",
            level="info",
        )

        # 윈도우 리셋
        self._ingest_count = 0
        self._applied_count = 0
        self._rpc_durations_ms.clear()
        self._last_stats_ts = now

    def _update_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _append_log(self, message: str, *, level: str = "info") -> None:
        prefix = {
            "success": "[OK] ",
            "error": "[ERR] ",
            "warning": "[WARN] ",
            "info": "",
        }.get(level, "")
        self.log_view.appendPlainText(prefix + message)

    def _build_instruction_text(self) -> str:
        port_line = (
            str(self._plugin_port)
            if self._plugin_port is not None
            else "50000  # replace with External Engine plugin port"
        )
        lines = [
            "# External engine -> ODT plugin (JSON over TCP)",
            "import json",
            "import socket",
            "",
            f'PLUGIN_HOST = "{self._local_ip}"',
            f"PLUGIN_PORT = {port_line}",
            "",
            "# Minimal payload using lists",
            "payload = {",
            '    "vehicle": "UAM1",',
            '    "pos": [0.0, 0.0, 0.0],',
            '    "att_deg": [0.0, 0.0, 0.0],',
            '    "degrees": True,',
            "}",
            "",
            "# Nested dictionaries are also accepted",
            "nested_payload = {",
            '    "vehicle": "UAM1",',
            '    "position": {',
            '        "north": 0.0,',
            '        "east": 0.0,',
            '        "down": -10.0,',
            '        "lat": 37.5441926,',
            '        "lon": 127.0775195,',
            '        "alt": 20.4,',
            "    },",
            '    "attitude": {',
            '        "roll": 0.0,',
            '        "pitch": 0.0,',
            '        "yaw": 0.0,',
            '        "units": "rad",',
            "    },",
            "}",
            "",
            "with socket.create_connection((PLUGIN_HOST, PLUGIN_PORT)) as sock:",
            "    sock.sendall(json.dumps(payload).encode(\"utf-8\"))",
            "    sock.sendall(b\"\\n\")",
            "    # sock.sendall(json.dumps(nested_payload).encode(\"utf-8\"))",
            "    # sock.sendall(b\"\\n\")",
            "# Send newline-delimited JSON for each update.",
            "# Optional fields: att_rad, attitude.units (\"rad\"|\"deg\"), lqr, timestamp, etc.",
        ]
        return "\n".join(lines)


def create_external_engine_dashboard(
    *,
    plugin_name: Optional[str] = None,
    plugin_port: Optional[int] = None,
    plugin_id: Optional[int] = None,
    comm_sender=None,
) -> ExternalEngineDashboard:
    return ExternalEngineDashboard(
        plugin_name=plugin_name,
        plugin_port=plugin_port,
        plugin_id=plugin_id,
        comm_sender=comm_sender,
    )
