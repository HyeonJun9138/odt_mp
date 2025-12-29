from __future__ import annotations

import math
import queue
import threading
import time
from typing import Callable

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    np = None  # type: ignore

from PySide6.QtCore import QThread, Signal

from .constants import CANONICAL_CAMERA_MAP, CANONICAL_CAMERA_ORDER, normalize_camera_key
from .integration import HAS_AIRSIM, airsim


class AirSimCaptureThread(QThread):
    """Background worker that polls AirSim for vehicle and camera imagery."""

    MAIN_INTERVAL = 0.1
    PREVIEW_INTERVAL = 0.1
    THUMB_TARGET_WIDTH = 512
    PREVIEW_TARGET_WIDTH = 960
    THUMB_JPEG_QUALITY = 75
    PREVIEW_JPEG_QUALITY = 82

    vehiclesFound = Signal(list)
    framesReady = Signal(dict)
    previewReady = Signal(bytes)
    logMsg = Signal(str)
    errorMsg = Signal(str)

    def __init__(self, host: str, port: int, camera_names):
        super().__init__()
        self.host = host
        self.port = port
        self.client = None
        self.selected_vehicle: str | None = None
        self.selected_camera: str = "front"
        self.running = True
        self.camera_names, self.camera_request_map = self._coerce_camera_names(camera_names)
        if not self.camera_names:
            self.camera_names = ["front"]
            self.camera_request_map = {"front": "front"}
        self.valid_cameras: set[str] = set()
        self.invalid_cameras: set[str] = set()
        self._nudge_lock = threading.Lock()
        self._nudge_preview = False
        self._nudge_main = False

    @staticmethod
    def _coerce_camera_names(camera_entries) -> tuple[list[str], dict[str, str]]:
        request_map: dict[str, str] = {
            name: CANONICAL_CAMERA_MAP[name]['request_key']
            for name in CANONICAL_CAMERA_ORDER
        }
        names = list(CANONICAL_CAMERA_ORDER)
        updated: set[str] = set()
        for entry in (camera_entries or []):
            raw_name = None
            request_key = None
            if isinstance(entry, str):
                raw_name = entry
            elif isinstance(entry, dict):
                raw_name = entry.get('name') or entry.get('key') or entry.get('id')
                request_key = entry.get('request_key') or entry.get('source') or entry.get('id')
            elif isinstance(entry, (list, tuple)) and entry:
                raw_name = entry[0]
                if len(entry) > 1:
                    request_key = entry[1]
            canonical = normalize_camera_key(raw_name)
            if not canonical or canonical not in CANONICAL_CAMERA_MAP or canonical in updated:
                continue
            if request_key is None:
                if isinstance(entry, dict):
                    request_key = entry.get('name') or raw_name or canonical
                else:
                    request_key = raw_name or canonical
            request_map[canonical] = str(request_key).strip()
            updated.add(canonical)
        return names, request_map

    def _prepare_display_payload(self, data: bytes | None, target_width: int, jpeg_quality: int) -> bytes | None:
        if not data:
            return None
        if cv2 is None or np is None:
            return data
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return data
        if frame is None:
            return data
        height, width = frame.shape[:2]
        need_resize = bool(target_width) and width > target_width
        if not need_resize:
            return data
        scale = target_width / float(width)
        target_height = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        quality = max(30, min(int(jpeg_quality), 95)) if jpeg_quality else 80
        try:
            ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        except Exception:
            return data
        if not ok:
            return data
        return encoded.tobytes()

    def _request_key(self, camera_key: str) -> str:
        return self.camera_request_map.get(camera_key, camera_key)

    def nudge(self, preview: bool = True, main: bool = False) -> None:
        with self._nudge_lock:
            if preview:
                self._nudge_preview = True
            if main:
                self._nudge_main = True

    def set_selected_vehicle(self, name: str) -> None:
        if self.selected_vehicle != name:
            self.selected_vehicle = name
            self.valid_cameras.clear()
            self.invalid_cameras.clear()

    def set_selected_camera(self, camera: str) -> None:
        self.selected_camera = camera
        if camera in self.invalid_cameras:
            self.logMsg.emit(f"[AirSim] camera unavailable: {camera}")

    def stop(self) -> None:
        self.running = False

    def _validate_camera(self, camera_key: str) -> bool:
        if camera_key in self.valid_cameras:
            return True
        if camera_key in self.invalid_cameras:
            return False
        if not self.selected_vehicle or not self.client:
            return False
        try:
            request_key = self._request_key(camera_key)
            resp = self.client.simGetImages([
                airsim.ImageRequest(request_key, airsim.ImageType.Scene, pixels_as_float=False, compress=True)
            ], vehicle_name=self.selected_vehicle)
            if resp and len(resp) > 0 and getattr(resp[0], 'image_data_uint8', None):
                self.valid_cameras.add(camera_key)
                self.logMsg.emit(f"[AirSim] camera available: {camera_key}")
                return True
            self.invalid_cameras.add(camera_key)
            self.logMsg.emit(f"[AirSim] camera produced no data: {camera_key}")
            return False
        except Exception as exc:  # pragma: no cover - defensive
            self.invalid_cameras.add(camera_key)
            self.errorMsg.emit(f"[AirSim] camera unavailable ({camera_key}): {exc}")
            return False

    def _validate_next_camera(self) -> None:
        if not self.selected_vehicle or not self.client:
            return
        for camera_key in self.camera_names:
            if camera_key in self.valid_cameras or camera_key in self.invalid_cameras:
                continue
            self._validate_camera(camera_key)
            break

    def _connect(self) -> None:
        if not HAS_AIRSIM:
            raise RuntimeError("AirSim package is not installed. Run 'pip install airsim'.")
        self.client = airsim.MultirotorClient(ip=self.host, port=self.port)
        self.client.confirmConnection()
        self.logMsg.emit(f"[AirSim] Connected to {self.host}:{self.port}")

    def _list_vehicles(self) -> None:
        names = []
        try:
            names = self.client.listVehicles()
        except Exception:
            pass
        if not names:
            names = ["UAM1", "UAM2"]
        self.vehiclesFound.emit(names)

    def _fetch_main_frames(self) -> None:
        if not self.selected_vehicle:
            return
        self._validate_next_camera()
        active_cameras = [camera_key for camera_key in self.camera_names if camera_key in self.valid_cameras]
        if not active_cameras:
            return
        requests = [
            airsim.ImageRequest(self._request_key(camera_key), airsim.ImageType.Scene, pixels_as_float=False, compress=True)
            for camera_key in active_cameras
        ]
        try:
            resp = self.client.simGetImages(requests, vehicle_name=self.selected_vehicle)
            if not resp:
                return
            out: dict[str, bytes | None] = {}
            for camera_key, image in zip(active_cameras, resp):
                payload = image.image_data_uint8 if image else None
                if payload:
                    payload = self._prepare_display_payload(payload, self.THUMB_TARGET_WIDTH, self.THUMB_JPEG_QUALITY)
                out[camera_key] = payload
            self.framesReady.emit(out)
        except Exception as exc:  # pragma: no cover - defensive
            self.errorMsg.emit(f"[AirSim] main frames error: {exc}")
            for camera_key in list(active_cameras):
                self.valid_cameras.discard(camera_key)
                if not self._validate_camera(camera_key):
                    self.valid_cameras.discard(camera_key)

    def _fetch_preview(self) -> None:
        if not self.selected_vehicle or not self.selected_camera:
            return
        if not self._validate_camera(self.selected_camera):
            return
        try:
            request_key = self._request_key(self.selected_camera)
            resp = self.client.simGetImages([
                airsim.ImageRequest(request_key, airsim.ImageType.Scene, pixels_as_float=False, compress=True)
            ], vehicle_name=self.selected_vehicle)
            if resp and len(resp) > 0:
                payload = resp[0].image_data_uint8
                if payload:
                    payload = self._prepare_display_payload(payload, self.PREVIEW_TARGET_WIDTH, self.PREVIEW_JPEG_QUALITY)
                if payload:
                    self.previewReady.emit(payload)
        except Exception as exc:  # pragma: no cover - defensive
            self.errorMsg.emit(f"[AirSim] preview error: {exc}")
            self.invalid_cameras.add(self.selected_camera)
            self.valid_cameras.discard(self.selected_camera)

    def run(self) -> None:  # type: ignore[override]
        try:
            self._connect()
            self._list_vehicles()
        except Exception as exc:  # pragma: no cover - defensive
            self.errorMsg.emit(f"[AirSim] connect/list error: {exc}")
            return

        last_main = 0.0
        last_prev = 0.0
        while self.running:
            now = time.time()
            with self._nudge_lock:
                preview_nudge = self._nudge_preview
                main_nudge = self._nudge_main
                if preview_nudge:
                    self._nudge_preview = False
                if main_nudge:
                    self._nudge_main = False
            if main_nudge or now - last_main >= self.MAIN_INTERVAL:
                self._fetch_main_frames()
                last_main = now
            if preview_nudge or now - last_prev >= self.PREVIEW_INTERVAL:
                self._fetch_preview()
                last_prev = now
            self.msleep(20)


class CameraCommandWorker(QThread):
    """Executes blocking AirSim camera commands on a background thread."""

    command_done = Signal(str, str)  # vehicle, camera
    command_error = Signal(str)

    def __init__(self, client_supplier: Callable[[], object | None], parent=None):
        super().__init__(parent)
        self._client_supplier = client_supplier
        self._queue: queue.Queue[tuple[str, dict] | None] = queue.Queue()
        self._running = True

    def enqueue_pose(self, vehicle: str, camera: str, request_key: str, pose) -> None:
        self._queue.put(("pose", {
            "vehicle": vehicle,
            "camera": camera,
            "request_key": request_key,
            "pose": pose,
        }))

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)

    def run(self) -> None:  # type: ignore[override]
        while self._running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                continue
            kind, payload = item
            try:
                client = self._client_supplier()
                if client is None:
                    raise RuntimeError("Camera control unavailable")
                if kind == "pose":
                    client.simSetCameraPose(payload["request_key"], payload["pose"], payload["vehicle"])
                self.command_done.emit(payload["vehicle"], payload["camera"])
            except Exception as exc:  # pragma: no cover - defensive
                self.command_error.emit(str(exc))
