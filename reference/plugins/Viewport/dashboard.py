
from __future__ import annotations

import math
import queue
import time
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .constants import CANONICAL_CAMERAS, load_camera_slots_from_settings
from .integration import HAS_AIRSIM, airsim
from .scaling import UiScale
from .theming import apply_common_qss
from .threads import AirSimCaptureThread, CameraCommandWorker
from .widgets import (
    CameraControlPanel,
    ConnectPanel,
    LogPanel,
    NetworkPanel,
    VehicleCameraSection,
)

class ViewportDashboard(QWidget):
    CAMERA_MOVE_STEP = 0.25
    CAMERA_ALT_STEP = 0.25
    CAMERA_ROT_STEP_DEG = 5.0

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def __init__(
        self,
        settings_path: Optional[Path] = None,
        airsim_host: Optional[str] = None,
        airsim_port: Optional[int] = None,
        plugin_port: Optional[int] = None,
        parent: Optional[QWidget] = None,
        *,
        plugin_id: Optional[int] = None,
        apply_callback: Optional[Callable[[dict], None]] = None,
        close_callback: Optional[Callable[[Optional[int], bool], None]] = None,
        initial_config: Optional[dict] = None,
    ):
        super().__init__(parent)
        self._plugin_id = plugin_id
        self._apply_callback = apply_callback
        self._close_callback = close_callback
        self._apply_started = False
        self._initial_config: dict[str, Any] = dict(initial_config or {})
        self._initial_vehicle_name: Optional[str] = None
        self._initial_camera_name: Optional[str] = None

        self._plugin_port = self._coerce_int(plugin_port)
        if self._plugin_port is None:
            self._plugin_port = self._coerce_int(self._initial_config.get('plugin_port'))

        if airsim_host is None:
            cfg_host = self._initial_config.get('airsim_host')
            if isinstance(cfg_host, str) and cfg_host.strip():
                airsim_host = cfg_host.strip()

        if airsim_port is None:
            cfg_port = self._coerce_int(self._initial_config.get('airsim_port'))
            if cfg_port is not None:
                airsim_port = cfg_port

        vehicle_pref = self._initial_config.get('vehicle')
        if isinstance(vehicle_pref, str) and vehicle_pref.strip():
            self._initial_vehicle_name = vehicle_pref.strip()

        camera_pref = self._initial_config.get('camera') or self._initial_config.get('request_key')
        if isinstance(camera_pref, str) and camera_pref.strip():
            self._initial_camera_name = camera_pref.strip()

        app = QApplication.instance()
        if app is not None:
            UiScale.init_from_screen(app)
            apply_common_qss(app)
        else:
            apply_common_qss(self)
        self.setWindowTitle('ODT Viewport')
        self.resize(1625, 900)
        self.setMinimumSize(1300, 760)

        self.camera_slots = load_camera_slots_from_settings(settings_path)
        if not self.camera_slots:
            self.camera_slots = CANONICAL_CAMERAS
        self.camera_defaults = {}
        self.camera_request_map = {}
        for slot in self.camera_slots:
            if isinstance(slot, dict):
                name = slot.get('name')
                if name:
                    if slot.get('default'):
                        self.camera_defaults[name] = slot['default']
                    self.camera_request_map[name] = slot.get('request_key', name)
            elif isinstance(slot, (list, tuple)) and slot:
                name = slot[0]
                if isinstance(name, str):
                    self.camera_request_map.setdefault(name, str(name))
            elif isinstance(slot, str):
                self.camera_request_map.setdefault(slot, slot)
        self.capture_thread = None
        self.command_client = None
        self.selected_camera = 'front'
        if self.camera_slots:
            first_slot = self.camera_slots[0]
            if isinstance(first_slot, (list, tuple)) and first_slot:
                self.selected_camera = first_slot[0]
            elif isinstance(first_slot, dict):
                self.selected_camera = first_slot.get('name', self.selected_camera)
            elif isinstance(first_slot, str):
                self.selected_camera = first_slot
        self._initial_camera_poses = {}
        self._current_camera_poses: dict[tuple[str, str], object] = {}
        self._command_worker: CameraCommandWorker | None = None
        self._controls_wired = False
        self._pending_pose_prime: set[str] = set()

        central = self
        central.setObjectName('RootArea')

        root_v = QVBoxLayout(central)
        root_v.setContentsMargins(UiScale.dp(16), UiScale.dp(16), UiScale.dp(16), UiScale.dp(16))
        root_v.setSpacing(UiScale.dp(14))

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(0)
        header_row.addWidget(self._append_title(central))
        header_row.addStretch(1)
        root_v.addLayout(header_row, 0)

        split = QSplitter(Qt.Horizontal, central)
        split.setObjectName('RootSplit')
        split.setHandleWidth(UiScale.dp(6))
        split.setChildrenCollapsible(False)

        left = self._build_left_pane(split)
        right = self._build_right_pane(split, airsim_host, airsim_port, self._plugin_port)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        root_v.addWidget(split, 1)

        # Wire up connection buttons
        self.net.btnApply.clicked.connect(self._on_apply_clicked)

        # Tile click selects a camera and updates the preview
        for name, tile in self.sec_vehicle.tiles.items():
            tile.clicked.connect(self._on_tile_clicked)

        self._apply_initial_form_values()
        resolved_camera = self._resolve_camera_name(self._initial_camera_name)
        if resolved_camera:
            self.selected_camera = resolved_camera
        self._mark_selected_camera(self.selected_camera)
        self._initial_camera_name = None

        # Attempt an initial connection
        self._connect_airsim()


    # UI helpers
    def _append_title(self, parent: QWidget):
        title = QLabel("ODT Viewport", parent); title.setObjectName("TitleLabel"); return title

    def _build_left_pane(self, parent: QWidget) -> QFrame:
        host = QFrame(parent)
        host.setObjectName("LeftPane")
        host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        host.setMaximumWidth(UiScale.dp(720))

        layout = QVBoxLayout(host)
        layout.setContentsMargins(UiScale.dp(20), UiScale.dp(20), UiScale.dp(20), UiScale.dp(20))
        layout.setSpacing(UiScale.dp(18))
        layout.addSpacing(UiScale.dp(4))

        split = QSplitter(Qt.Vertical, host)
        split.setObjectName("LeftSplit")
        split.setHandleWidth(UiScale.dp(4))
        split.setChildrenCollapsible(False)

        vehicle_wrap = QWidget(split)
        vehicle_wrap.setObjectName("LeftUpper")
        veh_layout = QVBoxLayout(vehicle_wrap)
        veh_layout.setContentsMargins(0, 0, 0, 0)
        veh_layout.setSpacing(UiScale.dp(12))

        self.sec_vehicle = VehicleCameraSection("Selected Vehicle", "Select vehicle", self.camera_slots, vehicle_wrap)
        self.sec_vehicle.cmb.currentTextChanged.connect(self._on_vehicle_changed)
        veh_layout.addWidget(self.sec_vehicle, 1)

        bottom_wrap = QFrame(split)
        bottom_wrap.setObjectName("BottomStrip")
        bottom_layout = QHBoxLayout(bottom_wrap)
        bottom_layout.setContentsMargins(UiScale.dp(18), UiScale.dp(18), UiScale.dp(18), UiScale.dp(18))
        bottom_layout.setSpacing(UiScale.dp(16))

        self.connect_panel = ConnectPanel(bottom_wrap)
        self.connect_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_panel = LogPanel(bottom_wrap)
        self.log_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        bottom_layout.addWidget(self.connect_panel, 1)
        bottom_layout.addWidget(self.log_panel, 2)

        split.addWidget(vehicle_wrap)
        split.addWidget(bottom_wrap)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        layout.addWidget(split, 1)
        return host




    def _build_right_pane(self, parent: QWidget, default_host=None, default_port=None, plugin_port=None) -> QFrame:
        host = QFrame(parent)
        host.setObjectName("RightPane")
        host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(host)
        layout.setContentsMargins(UiScale.dp(20), UiScale.dp(20), UiScale.dp(20), UiScale.dp(20))
        layout.setSpacing(UiScale.dp(18))

        split = QSplitter(Qt.Vertical, host)
        split.setObjectName("RightSplit")
        split.setHandleWidth(UiScale.dp(4))
        split.setChildrenCollapsible(False)

        self.camera_panel = CameraControlPanel(host)
        self.camera_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.net = NetworkPanel(host)
        self.net.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        split.addWidget(self.camera_panel)
        split.addWidget(self.net)
        if default_host is not None or default_port is not None:
            self.net.apply_defaults(host=default_host, port=default_port)
        if plugin_port is not None:
            self.net.set_plugin_port_display(plugin_port)
        else:
            self.net.set_plugin_port_display(None)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        layout.addWidget(split)
        self._wire_camera_controls()
        return host

    def _apply_initial_form_values(self) -> None:
        if not hasattr(self, 'net'):
            return
        cfg = self._initial_config
        if self._plugin_port is not None:
            self.net.set_plugin_port_display(self._plugin_port)
        else:
            fallback_port = self._coerce_int(cfg.get('plugin_port') if cfg else None)
            if fallback_port is not None:
                self._plugin_port = fallback_port
                self.net.set_plugin_port_display(fallback_port)
            else:
                self.net.set_plugin_port_display(None)
        if not cfg:
            return
        host = cfg.get('airsim_host')
        if isinstance(host, str) and host.strip():
            self.net.edtHost.setText(host.strip())
        client_ip = cfg.get('client_ip')
        if isinstance(client_ip, str) and client_ip.strip():
            self.net.edtClient.setText(client_ip.strip())
        port_value = self._coerce_int(cfg.get('airsim_port'))
        if port_value is not None:
            self.net.spinPort.setValue(port_value)

    def _resolve_camera_name(self, candidate: Optional[str]) -> Optional[str]:
        if not candidate:
            return None
        candidate = candidate.strip()
        if not candidate:
            return None
        tiles = getattr(self, 'sec_vehicle', None)
        tile_map = getattr(tiles, 'tiles', {}) if tiles else {}
        if candidate in tile_map:
            return candidate
        for name, request in self.camera_request_map.items():
            if candidate == name or candidate == request:
                return name
        if candidate in tile_map:
            return candidate
        return None

    def _wire_camera_controls(self):
        if getattr(self, '_controls_wired', False):
            return
        panel = getattr(self, 'camera_panel', None)
        if panel is None:
            return
        move = self.CAMERA_MOVE_STEP
        vert = self.CAMERA_ALT_STEP
        rot = self.CAMERA_ROT_STEP_DEG
        panel.btnMoveUp.clicked.connect(lambda: self._adjust_camera_position(action='Move Up', local_up=vert))
        panel.btnMoveDown.clicked.connect(lambda: self._adjust_camera_position(action='Move Down', local_up=-vert))
        panel.btnMoveLeft.clicked.connect(lambda: self._adjust_camera_position(action='Move Left', local_right=-move))
        panel.btnMoveRight.clicked.connect(lambda: self._adjust_camera_position(action='Move Right', local_right=move))
        panel.btnPitchUp.clicked.connect(lambda: self._adjust_camera_orientation(action='Rotate Up', pitch_deg=rot))
        panel.btnPitchDown.clicked.connect(lambda: self._adjust_camera_orientation(action='Rotate Down', pitch_deg=-rot))
        panel.btnYawLeft.clicked.connect(lambda: self._adjust_camera_orientation(action='Rotate Left', yaw_deg=-rot))
        panel.btnYawRight.clicked.connect(lambda: self._adjust_camera_orientation(action='Rotate Right', yaw_deg=rot))
        panel.btnResetCamera.clicked.connect(lambda: self._reset_camera_pose(action='Reset Camera'))
        self._controls_wired = True

    def _camera_request_key(self, camera_name: str) -> str:
        return self.camera_request_map.get(camera_name, camera_name)

    def _current_vehicle_name(self) -> str:
        if not hasattr(self, 'sec_vehicle'):
            return ''
        return (self.sec_vehicle.cmb.currentText() or '').strip()

    def _ensure_command_client(self):
        if not HAS_AIRSIM:
            self._log('Camera control requires AirSim (pip install airsim).')
            return None
        if self.command_client is None:
            self._log('Camera control is unavailable until a connection is established.')
            return None
        return self.command_client
    def _get_vehicle_pose(self, vehicle: str):
        if not HAS_AIRSIM or self.command_client is None or not vehicle:
            return None
        try:
            return self.command_client.simGetVehiclePose(vehicle)
        except Exception as exc:
            self._log(f"Failed to read vehicle pose '{vehicle}': {exc}")
            return None


    def _cache_initial_pose(self, vehicle: str, camera: str, pose):
        if not HAS_AIRSIM:
            return
        key = (vehicle, camera)
        if key not in self._initial_camera_poses:
            self._initial_camera_poses[key] = self._clone_pose(pose)
            summary = self._summarize_pose(pose)
            self._log(f"Camera '{camera}' initial world pose -> {summary}")
            relative = self._summarize_relative_pose(vehicle, pose)
            if relative:
                self._log(f"Camera '{camera}' initial body pose -> {relative}")

    def _set_current_pose(self, vehicle: str, camera: str, pose):
        if not HAS_AIRSIM or pose is None:
            return
        key = (vehicle, camera)
        self._current_camera_poses[key] = self._clone_pose(pose)

    def _get_control_pose(self, client, vehicle: str, camera: str, *, fallback_to_default: bool = True, force_refresh: bool = False):
        if not HAS_AIRSIM:
            return None
        key = (vehicle, camera)
        if not force_refresh:
            cached = self._current_camera_poses.get(key)
            if cached is not None:
                return self._clone_pose(cached)
        request_key = self._camera_request_key(camera)
        info = None
        try:
            info = client.simGetCameraInfo(request_key, vehicle)
        except Exception as exc:
            self._log(f"Failed to read camera pose '{camera}' on {vehicle}: {exc}")
        if info and getattr(info, 'pose', None):
            pose = info.pose
            self._cache_initial_pose(vehicle, camera, pose)
            self._set_current_pose(vehicle, camera, pose)
            return self._clone_pose(pose)
        if not force_refresh:
            cached = self._current_camera_poses.get(key)
            if cached is not None:
                return self._clone_pose(cached)
        if not fallback_to_default:
            return None
        pose = self._pose_from_default(camera)
        if pose is None:
            return None
        try:
            client.simSetCameraPose(request_key, pose, vehicle)
        except Exception as exc:
            self._log(f"Failed to prime default pose '{camera}' on {vehicle}: {exc}")
            return None
        self._cache_initial_pose(vehicle, camera, pose)
        self._set_current_pose(vehicle, camera, pose)
        return self._clone_pose(pose)
    def _ensure_initial_pose_cached(self, vehicle: str, camera: str):
        if not HAS_AIRSIM or self.command_client is None or not vehicle or not camera:
            return
        key = (vehicle, camera)
        if key in self._initial_camera_poses:
            return
        request_key = self._camera_request_key(camera)
        try:
            info = self.command_client.simGetCameraInfo(request_key, vehicle)
        except Exception:
            return
        self._cache_initial_pose(vehicle, camera, info.pose)
        self._set_current_pose(vehicle, camera, info.pose)

    def _summarize_pose(self, pose) -> str:
        if not HAS_AIRSIM or pose is None:
            return 'pos=(0,0,0) rot_deg=(0,0,0)'
        px = pose.position.x_val
        py = pose.position.y_val
        pz = pose.position.z_val
        pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
        return (
            f"pos=({px:.3f}, {py:.3f}, {pz:.3f}) "
            f"rot_deg=({math.degrees(pitch):.2f}, {math.degrees(roll):.2f}, {math.degrees(yaw):.2f})"
        )

    def _summarize_relative_pose(self, vehicle: str, pose):
        if not HAS_AIRSIM or pose is None or not vehicle or self.command_client is None:
            return None
        try:
            vehicle_pose = self.command_client.simGetVehiclePose(vehicle)
        except Exception:
            return None
        if not getattr(vehicle_pose, 'orientation', None):
            return None
        cam_q = pose.orientation
        veh_q = vehicle_pose.orientation
        try:
            veh_inv = veh_q.inverse()
        except Exception:
            return None
        try:
            rel_q = veh_inv * cam_q
        except Exception:
            return None
        try:
            pitch, roll, yaw = airsim.to_eularian_angles(rel_q)
        except Exception:
            return None
        cam_pos = pose.position
        veh_pos = vehicle_pose.position
        dx = cam_pos.x_val - veh_pos.x_val
        dy = cam_pos.y_val - veh_pos.y_val
        dz = cam_pos.z_val - veh_pos.z_val
        rel_pos = None
        try:
            vq = airsim.Quaternionr(dx, dy, dz, 0.0)
            rotated = veh_inv * vq * veh_q
            rel_pos = (rotated.x_val, rotated.y_val, rotated.z_val)
        except Exception:
            rel_pos = (dx, dy, dz)
        return (
            f"pos_body=({rel_pos[0]:.3f}, {rel_pos[1]:.3f}, {rel_pos[2]:.3f}) "
            f"rot_deg_body=({math.degrees(pitch):.2f}, {math.degrees(roll):.2f}, {math.degrees(yaw):.2f})"
        )

    def _prime_initial_camera_poses(self, vehicle: str):
        if not HAS_AIRSIM or self.command_client is None or not vehicle:
            return
        for slot in self.camera_slots:
            camera_name = None
            if isinstance(slot, dict):
                camera_name = slot.get('name')
            elif isinstance(slot, (list, tuple)) and slot:
                camera_name = slot[0]
            elif isinstance(slot, str):
                camera_name = slot
            if not camera_name:
                continue
            self._ensure_initial_pose_cached(vehicle, camera_name)

    @staticmethod
    def _clone_pose(pose):
        return airsim.Pose(
            airsim.Vector3r(pose.position.x_val, pose.position.y_val, pose.position.z_val),
            airsim.Quaternionr(
                pose.orientation.x_val,
                pose.orientation.y_val,
                pose.orientation.z_val,
                pose.orientation.w_val,
            ),
        )

    @staticmethod
    def _axes_from_orientation(orientation):
        if orientation is None:
            return None
        w = orientation.w_val
        x = orientation.x_val
        y = orientation.y_val
        z = orientation.z_val
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm == 0:
            return None
        w /= norm
        x /= norm
        y /= norm
        z /= norm
        forward = (
            1 - 2 * (y * y + z * z),
            2 * (x * y + z * w),
            2 * (x * z - y * w),
        )
        right = (
            2 * (x * y - z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z + x * w),
        )
        up = (
            2 * (x * z + y * w),
            2 * (y * z - x * w),
            1 - 2 * (x * x + y * y),
        )
        return forward, right, up

    def _basis_from_pose(self, pose):
        if not HAS_AIRSIM or pose is None or getattr(pose, 'orientation', None) is None:
            raise ValueError('Pose is unavailable for basis computation')
        axes = self._axes_from_orientation(pose.orientation)
        if not axes:
            raise ValueError('Orientation has zero length')
        return axes

    def _pose_from_default(self, camera_name: str):
        if not HAS_AIRSIM:
            return None
        data = self.camera_defaults.get(camera_name)
        if not data:
            return None
        pos = data.get('position')
        rot = data.get('rotation_deg')
        if pos is None or rot is None:
            return None
        try:
            px, py, pz = map(float, pos)
            pitch_deg, roll_deg, yaw_deg = map(float, rot)
        except Exception:
            return None
        orientation = airsim.to_quaternion(
            math.radians(pitch_deg),
            math.radians(roll_deg),
            math.radians(yaw_deg),
        )
        return airsim.Pose(airsim.Vector3r(px, py, pz), orientation)

    def _flush_pending_pose_primes(self, prefer_vehicle: str | None = None):
        if not HAS_AIRSIM:
            return
        if self.command_client is None:
            if prefer_vehicle:
                self._pending_pose_prime.add(prefer_vehicle)
            return
        vehicles = set(self._pending_pose_prime)
        self._pending_pose_prime.clear()
        if prefer_vehicle:
            vehicles.add(prefer_vehicle)
        if not vehicles:
            return
        for vehicle in vehicles:
            if vehicle:
                self._prime_initial_camera_poses(vehicle)

    def _adjust_camera_position(self, local_forward=0.0, local_right=0.0, local_up=0.0, action: str | None = None):
        if not any((local_forward, local_right, local_up)):
            return
        action_info = f'[{action}] ' if action else ''
        client = self._ensure_command_client()
        if not client:
            return
        vehicle = self._current_vehicle_name()
        if not vehicle:
            self._log(f"{action_info}Select a vehicle before adjusting the camera.")
            return
        camera = self.selected_camera
        if not camera:
            self._log(f"{action_info}Select a camera tile before adjusting the camera.")
            return
        if self.capture_thread and camera in getattr(self.capture_thread, 'invalid_cameras', set()):
            self._log(f"{action_info}Camera '{camera}' is not available on vehicle {vehicle}.")
            return
        pose = self._get_control_pose(client, vehicle, camera, fallback_to_default=False)
        if pose is None:
            return
        dx = local_forward
        dy = local_right
        dz = -local_up
        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dz) < 1e-6:
            return
        target_pose = self._clone_pose(pose)
        target_pose.position.x_val += dx
        target_pose.position.y_val += dy
        target_pose.position.z_val += dz
        try:
            request_key = self._camera_request_key(camera)
            client.simSetCameraPose(request_key, target_pose, vehicle)
        except Exception as exc:
            self._log(f"{action_info}Failed to translate camera '{camera}': {exc}")
            return
        self._set_current_pose(vehicle, camera, target_pose)
        summary_world = self._summarize_pose(target_pose)
        summary_body = self._summarize_relative_pose(vehicle, target_pose)
        body_text = f" | body -> {summary_body}" if summary_body else ''
        self._log(
            f"{action_info}Camera '{camera}' translated body_delta({dx:.2f}, {dy:.2f}, {dz:.2f}) "
            f"from vehicle_local(f={local_forward:.2f}, r={local_right:.2f}, u={local_up:.2f}). -> world -> {summary_world}{body_text}"
        )

    def _adjust_camera_orientation(self, pitch_deg=0.0, yaw_deg=0.0, action: str | None = None):
        client = self._ensure_command_client()
        if not client:
            return
        action_info = f'[{action}] ' if action else ''
        vehicle = self._current_vehicle_name()
        if not vehicle:
            self._log(f"{action_info}Select a vehicle before adjusting the camera.")
            return
        camera = self.selected_camera
        if not camera:
            self._log(f"{action_info}Select a camera tile before adjusting the camera.")
            return
        if self.capture_thread and camera in getattr(self.capture_thread, 'invalid_cameras', set()):
            self._log(f"{action_info}Camera '{camera}' is not available on vehicle {vehicle}.")
            return
        request_key = self._camera_request_key(camera)
        pose = self._get_control_pose(client, vehicle, camera, fallback_to_default=False)
        if pose is None:
            return
        try:
            pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
            pitch += math.radians(pitch_deg)
            yaw += math.radians(yaw_deg)
            pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
            client.simSetCameraPose(request_key, pose, vehicle)
            self._set_current_pose(vehicle, camera, pose)
            summary_world = self._summarize_pose(pose)
            summary_body = self._summarize_relative_pose(vehicle, pose)
            body_text = f" | body -> {summary_body}" if summary_body else ''
            self._log(
                f"{action_info}Camera '{camera}' rotated (pitch {pitch_deg:.1f}deg, yaw {yaw_deg:.1f}deg). -> world -> {summary_world}{body_text}"
            )
        except Exception as exc:
            self._log(f"{action_info}Failed to rotate camera '{camera}': {exc}")
    def _reset_camera_pose(self, action: str | None = None):
        if not HAS_AIRSIM:
            self._log('Camera reset requires AirSim (pip install airsim).')
            return
        action_info = f'[{action}] ' if action else ''
        client = self._ensure_command_client()
        if not client:
            return
        vehicle = self._current_vehicle_name()
        if not vehicle:
            self._log(f"{action_info}Select a vehicle before resetting the camera.")
            return
        camera = self.selected_camera
        if not camera:
            self._log(f"{action_info}Select a camera tile before resetting the camera.")
            return
        if self.capture_thread and camera in getattr(self.capture_thread, 'invalid_cameras', set()):
            self._log(f"{action_info}Camera '{camera}' is not available on vehicle {vehicle}.")
            return
        request_key = self._camera_request_key(camera)
        key = (vehicle, camera)
        pose = None
        pose_source = "default"
        cached = self._initial_camera_poses.get(key)
        if cached:
            pose = self._clone_pose(cached)
            pose_source = "initial"
        else:
            pose = self._pose_from_default(camera)
        if pose is None:
            self._log(f"{action_info}No baseline pose recorded for camera '{camera}'.")
            return
        try:
            client.simSetCameraPose(request_key, pose, vehicle)
            self._initial_camera_poses[key] = self._clone_pose(pose)
            self._set_current_pose(vehicle, camera, pose)
            summary = self._summarize_pose(pose)
            self._log(f"{action_info}Camera '{camera}' reset to {pose_source} pose. -> {summary}")
        except Exception as exc:
            self._log(f"{action_info}Failed to reset camera '{camera}': {exc}")

    # Logging utility
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_panel.log.appendPlainText(f"[{ts}] {msg}")

    def _mark_selected_camera(self, name: str):
        self.selected_camera = name
        for key, tile in self.sec_vehicle.tiles.items():
            tile.set_selected(key == name)
        self.camera_panel.set_selected_camera(name)
        vehicle = self._current_vehicle_name()
        if vehicle:
            self._ensure_initial_pose_cached(vehicle, name)
        if self.capture_thread and name in getattr(self.capture_thread, "invalid_cameras", set()):
            self._log(f"Camera '{name}' is not available on this vehicle.")

    def _collect_stream_configuration(self) -> Optional[dict]:
        plugin_port = self._coerce_int(self._plugin_port)
        if plugin_port is None:
            label_text = ''
            try:
                label_text = self.net.pluginPortValue.text().strip()
            except Exception:
                label_text = ''
            plugin_port = self._coerce_int(label_text)
        if plugin_port is None:
            self._log('Plugin port is not available; cannot apply settings.')
            return None

        host = (self.net.edtHost.text().strip() or '127.0.0.1') if hasattr(self.net, 'edtHost') else '127.0.0.1'
        port_value = self._coerce_int(self.net.spinPort.value() if hasattr(self.net, 'spinPort') else None)
        if port_value is None:
            port_value = 41451

        camera_name = self.selected_camera or ''
        request_key = self._camera_request_key(camera_name) if camera_name else ''
        vehicle = self._current_vehicle_name()

        config: dict[str, Any] = {
            'airsim_host': host,
            'airsim_port': port_value,
            'plugin_port': plugin_port,
            'stream_port': plugin_port,
            'bind_host': '0.0.0.0',
            'fps': 60,
            'vehicle': vehicle,
            'camera': camera_name,
            'request_key': request_key,
            'use_compression': False,
            'jpeg_quality': 70,
        }
        client_ip = self.net.edtClient.text().strip() if hasattr(self.net, 'edtClient') else ''
        if client_ip:
            config['client_ip'] = client_ip
        if self._plugin_id is not None:
            config['plugin_id'] = self._plugin_id
        return config

    def _on_apply_clicked(self) -> None:
        config = self._collect_stream_configuration()
        if not config:
            return
        if hasattr(self.net, 'btnApply'):
            self.net.btnApply.setEnabled(False)
        try:
            if self._apply_callback:
                self._apply_callback(dict(config))
        except Exception as exc:
            self._log(f'Apply handler failed: {exc}')
            if hasattr(self.net, 'btnApply'):
                self.net.btnApply.setEnabled(True)
            return
        self._apply_started = True
        self._plugin_port = config.get('plugin_port', self._plugin_port)
        self._initial_config = dict(config)
        self._stop_capture_thread()
        self._log('Viewport configuration applied; streaming continues in background.')
        self.close()

    # AirSim connect/disconnect helpers
    def _stop_capture_thread(self):
        if self.capture_thread is not None:
            try:
                self.capture_thread.stop()
                self.capture_thread.wait(1500)
            except Exception:
                pass
            self.capture_thread = None
        self._initial_camera_poses.clear()
        self._current_camera_poses.clear()
        self._pending_pose_prime.clear()
        self.command_client = None

    def _connect_airsim(self):
        self._stop_capture_thread()
        host = self.net.edtHost.text().strip() or "127.0.0.1"
        port = int(self.net.spinPort.value())
        if not HAS_AIRSIM:
            self._log("AirSim package is not installed. Run 'pip install airsim'.")
            return
        self._log(f"Attempting AirSim connection: {host}:{port}")
        th = AirSimCaptureThread(host, port, self.camera_slots)
        th.vehiclesFound.connect(self._on_vehicles_found)
        th.framesReady.connect(self._on_frames_ready)
        th.previewReady.connect(self._on_preview_ready)
        th.logMsg.connect(self._log)
        th.errorMsg.connect(lambda e: self._log(f"ERROR: {e}"))
        self.capture_thread = th
        th.start()

        self.command_client = None
        if HAS_AIRSIM:
            try:
                cmd_client = airsim.MultirotorClient(ip=host, port=port)
                cmd_client.confirmConnection()
                self.command_client = cmd_client
                current_vehicle = self._current_vehicle_name()
                self._flush_pending_pose_primes(current_vehicle)
            except Exception as exc:
                self._log(f"Camera control client failed: {exc}")

    def _on_vehicles_found(self, names: list):
        self.sec_vehicle.cmb.blockSignals(True)
        self.sec_vehicle.cmb.clear()
        for n in names:
            self.sec_vehicle.cmb.addItem(n)
        self.sec_vehicle.cmb.blockSignals(False)

        selected_name = None
        if names:
            target = self._initial_vehicle_name
            if target and target in names:
                selected_name = target
            else:
                selected_name = names[0]
        if selected_name is not None:
            try:
                index = names.index(selected_name)
            except ValueError:
                index = 0
            self.sec_vehicle.cmb.setCurrentIndex(index)
        self._initial_vehicle_name = None
        self._log(f"Vehicles: {names}")

    # Callback: vehicle selection changed
    def _on_vehicle_changed(self, name: str):
        if not name:
            return
        self._log(f"Selected Vehicle = {name}")
        if self.capture_thread:
            self.capture_thread.set_selected_vehicle(name)
        self._flush_pending_pose_primes(name)
        if self.selected_camera:
            self._ensure_initial_pose_cached(name, self.selected_camera)

    # Callback: camera tile clicked
    def _on_tile_clicked(self, camera_name: str):
        self._mark_selected_camera(camera_name)
        if self.capture_thread:
            self.capture_thread.set_selected_camera(camera_name)
        self._log(f"Preview Camera = {camera_name}")

    # Update thumbnail grid when frames arrive
    def _on_frames_ready(self, frames: dict):
        # frames: {"front": bytes, ...}
        for key, raw in frames.items():
            tile = self.sec_vehicle.tiles.get(key)
            if tile and raw:
                tile.set_image_bytes(raw)

    # Update preview when the high-rate feed arrives
    def _on_preview_ready(self, raw: bytes):
        if raw:
            self.camera_panel.preview.set_image_bytes(raw)

    # Ensure the capture thread stops on close
    def closeEvent(self, e):
        self._stop_capture_thread()
        if self._close_callback:
            try:
                self._close_callback(self._plugin_id, bool(self._apply_started))
            except Exception as exc:
                self._log(f'Close callback failed: {exc}')
        return super().closeEvent(e)


# ---------------------------------------------------------------------
