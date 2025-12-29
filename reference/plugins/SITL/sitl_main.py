#!/usr/bin/env python
#python SITL/receiver_test.py --port 50051

# -*- coding: utf-8 -*-
"""
sitl_main.py  –  Stand-alone GUI shell for SITL Simulation
────────────────────────────────────────────────────────────
• 좌측  : 운영/시뮬레이션 설정 테이블 + Log + CSV Browse
• 중앙  : Simulation Play Log
• 우측  : 2-D Top View  + 3-D View
"""
from __future__ import annotations
import sys, datetime as dt
from pathlib import Path
from sitl_sim import SitlSim
import time
import socket
import threading
import json
import queue
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (3-D 등록용)
import matplotlib.pyplot as plt
import math
import re
from typing import Any, Callable, Dict, Optional, Set

# ===== [ADD] xlsx→csv와 동일한 좌표 변환을 위해 import =====
from sitl_coord_transform import (
    AffineGeoToUEMapper,
    GCP_LIST,
    CITY_HALL_UE,
    JSON_XY_DIV,
    JSON_Z_DIV,
    load_resources_vp, 
    lookup_vp_label_alt_m
)
# ===========================================================

try:
    import airsim
except ImportError:
    airsim = None

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui  import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QTextEdit,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QSplitter,
    QTableWidget, QTableWidgetItem, QPushButton, QCheckBox, QFileDialog, QDialog, QHeaderView, QDateEdit
)

from Monitoring.Functions.AircraftAgent import xy_to_lonlat

# ── Heading arrow scale (조절용) ───────────────────────────────
ARROW_LEN_LL  = 0.01000   # 2-D: degrees  (경도·위도), ≈15 m
ARROW_LEN_ENU = 1000         # 3-D: meters

# ── 2-D / 3-D 고정 범위 (경도·위도) ─────────────────────────────
LON_MIN, LON_MAX = 126.78, 127.15
LAT_MIN, LAT_MAX =  37.43,  37.65

# ── AirSim integration helpers ───────────────────────────────────
if airsim is not None:
    airsim.VehicleClient.getClientVersion = lambda self: 4

STATICMESH_CANDIDATES = [
    "KP2StaticMesh",
    "/Game/KP2A/KP2StaticMesh.KP2StaticMesh",
    "StaticMesh'/Game/KP2A/KP2StaticMesh.KP2StaticMesh'",
]
BLUEPRINT_CANDIDATES = [
    "KP2_Blueprint",
    "Blueprint'/Game/KP2/KP2_Blueprint.KP2_Blueprint'",
    "/Game/KP2/KP2_Blueprint.KP2_Blueprint_C",
]

PHYSICS_ENABLED = False
FALLBACK_TO_BP = True

WORLD_OFFSET_N = 0.0
WORLD_OFFSET_E = 0.0
WORLD_OFFSET_D = -2.5
YAW_BIAS_DEG = -90.0

YAW_TAU = 0.50
YAW_RATE_MAX = math.radians(60.0)
BANK_ENABLE = False
BANK_MAX_DEG = 8.0
BANK_TAU = 0.50
G = 9.80665

def sanitize_id(value: object) -> str:
    s2 = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    if not s2:
        s2 = "AC"
    if s2[0].isdigit():
        s2 = "AC_" + s2
    return s2[:60]

def angle_wrap_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi

def to_quat(pitch: float, roll: float, yaw: float):
    if airsim is None:
        raise RuntimeError('AirSim module not available')
    if hasattr(airsim, "to_quaternion"):
        return airsim.to_quaternion(pitch, roll, yaw)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return airsim.Quaternionr(x, y, z, w)

def is_blueprint_ref(asset_name: str) -> bool:
    return ("Blueprint'" in asset_name) or asset_name.endswith("_C")


class ActorState:
    def __init__(self, name: str, last_pos: tuple[float, float, float], last_yaw: float, last_t: float):
        self.name = name
        self.last_pos = last_pos
        self.last_yaw = last_yaw
        self.last_t = last_t
        self.speed_mps = 0.0
        self.yawrate_rps = 0.0
        self.roll = 0.0


class AirSimFleetBridge:
    def __init__(self, client: Any, log_cb: Optional[Callable[[str], None]] = None):
        self.client = client
        self._log = log_cb or (lambda msg: None)
        self.id_to_actor: Dict[str, ActorState] = {}
        self.used_names: Set[str] = set()
        try:
            self._assets = self.client.simListAssets()
        except Exception:
            self._assets = []
            
            
                    # ▼▼▼ 추가: resources_vp.csv 프리로드(있으면 캐시에 적재)
        resources_csv = Path(__file__).resolve().parent / "resource" / "resources_vp.csv"
        if resources_csv.exists():
            try:
                load_resources_vp(str(resources_csv))
            except Exception:
                pass


    def _dep_label_from_pkt(self, pkt: Dict[str, Any]) -> tuple[str, str] | None:
        """
        스냅샷 pkt에서 출발 버티포트/라벨(GATE n 또는 FATO n) 추출.
        라벨 입력이 '8'처럼 숫자일 경우 'GATE 8'로 정규화.
        """
        origin = str(pkt.get("From") or "").strip()
        if not origin:
            return None
        gate = pkt.get("DepGate_No")
        fato = pkt.get("DepFATO_No")

        # Gate 우선
        if gate not in (None, "", "None"):
            s = str(gate).strip().upper()
            if s.startswith("GATE"):
                label = s
            else:
                try:
                    label = f"GATE {int(float(s))}"
                except Exception:
                    label = None
            if label:
                return (origin, label)

        # FATO 대안
        if fato not in (None, "", "None"):
            s = str(fato).strip().upper()
            if s.startswith("FATO"):
                label = s
            else:
                try:
                    label = f"FATO {int(float(s))}"
                except Exception:
                    label = None
            if label:
                return (origin, label)

        return None

    def _spawn_alt_override_from_pkt(self, pkt: Dict[str, Any]) -> float | None:
        pair = self._dep_label_from_pkt(pkt)
        if not pair:
            return None
        origin, label = pair
        alt_up_m = lookup_vp_label_alt_m(origin, label)  # CSV의 Z(위로 +)
        if alt_up_m is None:
            return None
        return -float(alt_up_m)  # ★ AirSim NED: D = -Z
            

    def _log_msg(self, msg: str) -> None:
        try:
            self._log(msg)
        except Exception:
            pass

    def _pick_asset(self) -> tuple[str, bool]:
        for cand in STATICMESH_CANDIDATES:
            for asset in self._assets:
                if isinstance(asset, str) and (cand in asset):
                    return cand, False
        if STATICMESH_CANDIDATES:
            return STATICMESH_CANDIDATES[0], False
        if FALLBACK_TO_BP and BLUEPRINT_CANDIDATES:
            return BLUEPRINT_CANDIDATES[0], True
        return "KP2StaticMesh", False

    def _unique_name(self, base: str) -> str:
        name = base
        i = 2
        while name in self.used_names:
            name = f"{base}_{i}"
            i += 1
        self.used_names.add(name)
        return name

    def ensure_actor(self, acid: str, init_pose: "airsim.Pose") -> ActorState:
        if acid in self.id_to_actor:
            return self.id_to_actor[acid]

        base = f"AC_{sanitize_id(acid)}"
        name = self._unique_name(base)

        asset, bp = self._pick_asset()
        try:
            spawned_name = self.client.simSpawnObject(
                object_name=name,
                asset_name=asset,
                pose=init_pose,
                scale=airsim.Vector3r(1.0, 1.0, 1.0),
                physics_enabled=PHYSICS_ENABLED,
                is_blueprint=bp
            )
            name = spawned_name
        except Exception as exc:
            self._log_msg(f"[ERR] AirSim spawn 실패: {name} / {asset} → {exc}")
            raise

        st = ActorState(
            name=name,
            last_pos=(init_pose.position.x_val, init_pose.position.y_val, init_pose.position.z_val),
            last_yaw=0.0,
            last_t=time.perf_counter()
        )
        self.id_to_actor[acid] = st
        self._log_msg(f"[OK] AirSim actor 생성: {acid} → {name} (asset={asset}, blueprint={bp}, physics={PHYSICS_ENABLED})")
        return st

    def destroy_actor(self, acid: str) -> None:
        st = self.id_to_actor.pop(acid, None)
        if not st:
            return
        try:
            self.client.simDestroyObject(st.name)
        except Exception as exc:
            self._log_msg(f"[WARN] AirSim destroy 실패: {st.name} → {exc}")
        if st.name in self.used_names:
            self.used_names.remove(st.name)
        self._log_msg(f"[DEL] AirSim actor 제거: {acid} ({st.name})")

    def prune_actors(self, alive_ids: set[str]) -> None:
        for acid in list(self.id_to_actor.keys()):
            if acid not in alive_ids:
                self.destroy_actor(acid)

    def reset(self) -> None:
        for acid in list(self.id_to_actor.keys()):
            self.destroy_actor(acid)
        self.id_to_actor.clear()
        self.used_names.clear()

    def update_actor(self, acid: str, pos_nez: tuple[float, float, float], heading_deg: float,
                 now_t: float, use_bank: bool, alt_override_z: float | None = None) -> None:
        st = self.id_to_actor.get(acid)
        
        if not st:
            yaw0 = math.radians(heading_deg + YAW_BIAS_DEG)
            z_init = alt_override_z if (alt_override_z is not None) else pos_nez[2]
            pose0 = airsim.Pose(
                airsim.Vector3r(pos_nez[0] + WORLD_OFFSET_N, pos_nez[1] + WORLD_OFFSET_E, z_init + WORLD_OFFSET_D),
                to_quat(0.0, 0.0, yaw0)
            )
            st = self.ensure_actor(acid, pose0)
            st.last_yaw = yaw0
            st.last_pos = (pose0.position.x_val, pose0.position.y_val, pose0.position.z_val)
            st.last_t = now_t
            return

        dt_s = max(1e-6, now_t - st.last_t)
        x0, y0, z0 = st.last_pos
        x1 = pos_nez[0] + WORLD_OFFSET_N
        y1 = pos_nez[1] + WORLD_OFFSET_E
        z1 = pos_nez[2] + WORLD_OFFSET_D

        dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)
        spd = math.sqrt(dx * dx + dy * dy + dz * dz) / dt_s

        yaw_tgt = math.radians(heading_deg + YAW_BIAS_DEG)
        dyaw = angle_wrap_pi(yaw_tgt - st.last_yaw)
        alpha_yaw = max(0.0, min(1.0, dt_s / max(YAW_TAU, 1e-3)))
        yaw_cmd = st.last_yaw + alpha_yaw * dyaw
        yaw_rate = angle_wrap_pi(yaw_cmd - st.last_yaw) / dt_s
        if abs(yaw_rate) > YAW_RATE_MAX:
            yaw_rate = math.copysign(YAW_RATE_MAX, yaw_rate)
            yaw_cmd = angle_wrap_pi(st.last_yaw + yaw_rate * dt_s)

        roll = st.roll
        if use_bank and BANK_ENABLE:
            phi_tgt = math.atan(max(0.0, spd) * abs(yaw_rate) / G)
            if yaw_rate < 0:
                phi_tgt = -phi_tgt
            phi_tgt = max(-math.radians(BANK_MAX_DEG), min(math.radians(BANK_MAX_DEG), phi_tgt))
            a_bank = max(0.0, min(1.0, dt_s / max(BANK_TAU, 1e-3)))
            roll = roll + a_bank * (phi_tgt - roll)

        pose = airsim.Pose(airsim.Vector3r(x1, y1, z1), to_quat(0.0, roll, yaw_cmd))
        try:
            self.client.simSetObjectPose(st.name, pose, teleport=False)
        except Exception as exc:
            self._log_msg(f"[WARN] AirSim pose 업데이트 실패: {st.name} → {exc}")

        st.last_pos = (x1, y1, z1)
        st.last_yaw = yaw_cmd
        st.last_t = now_t
        st.speed_mps = spd
        st.yawrate_rps = yaw_rate
        st.roll = roll

    def sync_snapshot(self, snapshot: Dict[str, Dict[str, Any]], now_t: float, use_bank: bool = False) -> None:
        alive_ids: set[str] = set()
        for acid, pkt in snapshot.items():
            try:
                x = float(pkt['x']); y = float(pkt['y']); z = float(pkt['z'])
                hdg_deg = float(pkt['heading_deg'])
            except Exception:
                continue
            alive_ids.add(acid)

            # ▼ 스폰 시에만 적용할 Z override 계산 (resources_vp.csv)
            alt_override = self._spawn_alt_override_from_pkt(pkt)

            self.update_actor(
                acid,
                (x, y, z),
                hdg_deg,
                now_t,
                use_bank=use_bank,
                alt_override_z=alt_override
            )

        self.prune_actors(alive_ids)



class FlightPlanDialog(QDialog):
    """날짜별 FPL 폴더를 등록/정렬해서 돌려주는 간단한 설정창"""
    def __init__(self, parent=None, preset: list[tuple[dt.date, str]] | None = None):
        super().__init__(parent)
        self.setWindowTitle("비행계획 설정")
        self.resize(700, 360)

        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QTableWidget, QPushButton, QLineEdit, QLabel
        from PyQt5.QtCore import QDate

        v = QVBoxLayout(self)
        self.tbl = QTableWidget(0, 3, self)
        self.tbl.setHorizontalHeaderLabels(["날짜 (yyyy-MM-dd)", "FPL 폴더", ""])
        self.tbl.horizontalHeader().setStretchLastSection(False)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl.setColumnWidth(2, 60)
        v.addWidget(self.tbl)

        h = QHBoxLayout()
        btn_add    = QPushButton("행 추가")
        btn_del    = QPushButton("선택 삭제")
        h.addWidget(btn_add); h.addWidget(btn_del); h.addStretch()
        v.addLayout(h)

        h2 = QHBoxLayout()
        btn_ok     = QPushButton("확인")
        btn_cancel = QPushButton("취소")
        h2.addStretch(); h2.addWidget(btn_ok); h2.addWidget(btn_cancel)
        v.addLayout(h2)

        def _add_row(qdate: QDate | None = None, folder: str = ""):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            de = QDateEdit()
            de.setCalendarPopup(True)
            de.setDisplayFormat("yyyy-MM-dd")
            if qdate is None:
                # 직전 행 다음날을 기본값으로
                if r > 0:
                    prev_de: QDateEdit = self.tbl.cellWidget(r-1, 0)  # type: ignore
                    de.setDate(prev_de.date().addDays(1))
                else:
                    de.setDate(QDate.currentDate())
            else:
                de.setDate(qdate)
            self.tbl.setCellWidget(r, 0, de)

            le = QLineEdit(); le.setReadOnly(True); le.setText(folder)
            self.tbl.setCellWidget(r, 1, le)

            def _browse():
                fld = QFileDialog.getExistingDirectory(self, "FPL 폴더 선택", folder or str(Path.cwd()))
                if fld: le.setText(fld)
            b = QPushButton("...")
            b.clicked.connect(_browse)
            self.tbl.setCellWidget(r, 2, b)

        def _del_selected():
            rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
            for r in rows:
                self.tbl.removeRow(r)

        btn_add.clicked.connect(lambda: _add_row())
        btn_del.clicked.connect(_del_selected)
        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self.accept)

        # 미리 값 채우기(있다면)
        if preset:
            from PyQt5.QtCore import QDate
            for d, f in sorted(preset):
                _add_row(QDate(d.year, d.month, d.day), f)

    def get_plan(self) -> list[tuple[dt.date, str]]:
        """유효한 (날짜, 폴더) 목록을 날짜 오름차순으로 반환"""
        out: list[tuple[dt.date, str]] = []
        for r in range(self.tbl.rowCount()):
            de: QDateEdit = self.tbl.cellWidget(r, 0)   # type: ignore
            le: QLineEdit = self.tbl.cellWidget(r, 1)   # type: ignore
            d = de.date().toPyDate()
            f = (le.text() or "").strip()
            if f and Path(f).is_dir():
                out.append((d, f))
        out.sort(key=lambda x: x[0])
        return out


# ───────────────────────────────────────────────────────────────
class SitlMainWindow(QMainWindow):
    log_msg  = pyqtSignal(str)
    play_msg = pyqtSignal(str)
    def __init__(self, plan_days: list[tuple[dt.date, str]] | None = None,
                comm_config: dict | None = None):
        super().__init__()
        self.setWindowTitle("UAM SITL – Simulation Console")
        self.resize(1500, 900)

        self._sim_running = False
        self._sim_time    = None
        self._sim_speed   = 1.0  # 1x speed

        # 비행계획(홈에서 전달)
        self._plan_days: list[tuple[dt.date, str]] = plan_days or []
        self._plan_idx: int = 0 if self._plan_days else -1
        self._current_folder: str | None = None

        self._airsim_client: Any | None = None
        self._airsim_bridge: AirSimFleetBridge | None = None
        self._airsim_enabled = False
        self._airsim_thread: Optional[threading.Thread] = None
        self._airsim_queue: Optional[queue.Queue] = None
        self._airsim_stop_event: Optional[threading.Event] = None
        self._airsim_error_flag = False

        # [ADD] AirSim 활성화 시 좌표 변환 로그 1회 제어
        self._airsim_logged_ids: Set[str] = set()

        self._init_ui()
        self.sim: SitlSim | None = None
        self.udp_socks: dict[tuple[str,int], socket.socket] = {}

        self.log_msg.connect(self.te_log.append)
        self.play_msg.connect(self.te_play.append)

        self._tx_interval_ms = 1000
        self._ui_interval_ms = self._tx_interval_ms
        self._sim_step_accum = 0.0
        self._last_tick_ts = time.monotonic()
        self.udp_socks: dict[tuple[str,int], socket.socket] = {}
        
        # ── Play Log 필터/상태 (중복 억제 & 링버퍼)
        self._last_play = {}             # {acid: {'x','y','z','heading_deg','phase','lane'}}
        self._play_line_limit = 500      # 플레이 로그 최대 라인수(링버퍼)
        self._log_pos_eps_m = 25.0       # 위치 변화 임계(미터) - 이 이상 움직일 때만 찍음
        self._log_hdg_eps_deg = 10.0     # 헤딩 변화 임계(도)

        # [ADD] 변환 매퍼 지연 준비
        self._geo_mapper: Optional[AffineGeoToUEMapper] = None

        if not self._plan_days:
            self._auto_pick_latest_plan()

        if comm_config:
            self._apply_comm_config(comm_config)

        if self._plan_days:
            _, first_folder = self._plan_days[self._plan_idx]
            self._load_fpl_folder(first_folder)
            self._update_plan_label()
        if not self._plan_days:
            self._auto_pick_latest_plan()

        # 통신 설정 적용 (있으면 자동 연결)
        if comm_config:
            self._apply_comm_config(comm_config)

        # 계획이 있으면 1일차 로딩만(자동 시작은 Start 버튼에서)
        if self._plan_days:
            _, first_folder = self._plan_days[self._plan_idx]
            self._load_fpl_folder(first_folder)
            self._update_plan_label()

        # ── Haversine (거리[km])
        def _hav(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) *
                 math.cos(math.radians(lat2)) *
                 math.sin(dlon / 2) ** 2)
            return 2 * R * math.asin(math.sqrt(a))
        self._hav = _hav
        # ▼▼ [ADD] 고정 매퍼(NumPy float64) 준비 – 한 번만 피팅 ▼▼
        self._ue_mapper = AffineGeoToUEMapper(GCP_LIST)
        self._cityhall_ue = CITY_HALL_UE
        self._div_xy = float(JSON_XY_DIV)  # 100.0
        self._div_z  = float(JSON_Z_DIV)   # 100.0
# ▲▲ [ADD END] ▲▲
        # ▼▼ [ADD] AirSim 프리‑스폰 및 로그 관리 ▼▼
        self._vp_layout = None            # {(PortName, Label) -> {'N_m','E_m','D_m','yaw_deg'}}
        self._airsim_logged_ids = set()   # 활성화 1회 로그 방지
        # ▲▲ [ADD END] ▲▲

    # ----------------------------------------------------------
    # [ADD] 좌표 변환 준비/헬퍼 (xlsx→csv와 동일 스케일/오프셋)
    def _ensure_geo_mapper(self):
        if self._geo_mapper is None:
            self._geo_mapper = AffineGeoToUEMapper(GCP_LIST)
            self._cityhall_ue = CITY_HALL_UE
            self._div_xy = float(JSON_XY_DIV)  # 100.0
            self._div_z  = float(JSON_Z_DIV)   # 100.0

    def _latlon_to_ch_m(self, lat: float, lon: float, alt_m: float = 0.0) -> tuple[float,float,float]:
        """
        WGS84(lat,lon,alt) → UE 절대(cm) → (시청 원점 오프셋, cm) → (m)
        반환: (X_m, Y_m, Z_m)  ※ xlsx→csv의 X_m,Y_m,Z_m과 동일
        """
        self._ensure_geo_mapper()
        ue_abs_cm = self._geo_mapper.geodetic_to_ue(lat, lon, alt_m)  # [X,Y,Z] cm
        x_cm = float(ue_abs_cm[0]) - float(self._cityhall_ue[0])
        y_cm = float(ue_abs_cm[1]) - float(self._cityhall_ue[1])
        z_cm = float(ue_abs_cm[2]) - float(self._cityhall_ue[2])
        return (x_cm / self._div_xy, y_cm / self._div_xy, z_cm / self._div_z)


    # sitl_main.py 내
    def _snapshot_for_airsim(self, snap: dict) -> dict:
        if not snap:
            return {}
        out = {}
        from sitl_coord_transform import wgs84_to_airsim_ned, load_resources_vp
        load_resources_vp()  # 1회 로딩 보장

        def _lane_alt_m(lane: str) -> float | None:
            if not lane or lane == '-':
                return None
            m = re.search(r'(\d+)', str(lane))
            return (float(m.group(1)) * 0.3048) if m else None  # 'L-1000' → 304.8

        for acid, pkt in snap.items():
            q = dict(pkt)
            try:
                lat = float(pkt.get("lat")); lon = float(pkt.get("lon"))
                z_agl = float(pkt.get("alt_m", 0.0))        # ← 시뮬 내부 AGL 그대로
                phase = str(pkt.get("phase", "")).upper()[:1]
                dep = str(pkt.get("From", "") or "")
                arr = str(pkt.get("To", "") or "")
                dep_gate = str(pkt.get("DepGate_No", "") or "")
                dep_fato = str(pkt.get("DepFATO_No", "") or "")
                arr_gate = str(pkt.get("ArrGate_No", "") or "")
                arr_fato = str(pkt.get("ArrFATO_No", "") or "")
                lane_alt = _lane_alt_m(str(pkt.get("lane", "")))

                # 기준 고도 조회 (없으면 None)
                dep_gate_alt = lookup_vp_label_alt_m(dep, f"GATE {dep_gate}") if dep_gate else None
                dep_fato_alt = lookup_vp_label_alt_m(dep, f"FATO {dep_fato}") if dep_fato else None
                arr_gate_alt = lookup_vp_label_alt_m(arr, f"GATE {arr_gate}") if arr_gate else None
                arr_fato_alt = lookup_vp_label_alt_m(arr, f"FATO {arr_fato}") if arr_fato else None

                def _nz(*vals):
                    for v in vals:
                        if v is not None:
                            return float(v)
                    return 0.0

                if phase == 'A':          # Gate taxi
                    alt_abs = _nz(dep_fato_alt, dep_gate_alt)
                elif phase in ('B','C','D'):
                    base = _nz(dep_fato_alt, dep_gate_alt)
                    alt_abs = base + z_agl
                elif phase == 'E':        # 회랑 진입
                    alt_abs = lane_alt if lane_alt is not None else _nz(dep_fato_alt, dep_gate_alt) + z_agl
                elif phase == 'F':        # 크루즈(절대)
                    alt_abs = lane_alt if lane_alt is not None else z_agl  # lane 미지정이면 기존 1000ft 기본 유지
                elif phase in ('G','H','I'):
                    base = _nz(arr_fato_alt, arr_gate_alt)
                    alt_abs = base + z_agl
                elif phase == 'J':        # 수직착륙
                    alt_abs = _nz(arr_fato_alt, arr_gate_alt)
                elif phase == 'K':        # FATO→Gate taxi
                    alt_abs = _nz(arr_gate_alt, arr_fato_alt)
                else:
                    alt_abs = _nz(dep_fato_alt, dep_gate_alt) + z_agl

                # LLA(절대고도) → NED
                n_m, e_m, _ = wgs84_to_airsim_ned(lat, lon, alt_abs)
                down_m = -float(alt_abs)
                q["x"], q["y"], q["z"] = n_m, e_m, down_m
                q["alt_m_agl"] = z_agl         # (옵션) 디버그용
                q["alt_m_abs"] = alt_abs       # (옵션) 디버그용

            except Exception:
                pass
            out[acid] = q
        return out


    # ----------------------------------------------------------

    @staticmethod
    def _format_bytes(value: float) -> str:
        units = ["B", "KB", "MB", "GB"]
        amount = float(value)
        for unit in units:
            if amount < 1024.0 or unit == units[-1]:
                if amount >= 100 or unit == "B":
                    return f"{amount:.0f} {unit}"
                return f"{amount:.1f} {unit}"
            amount /= 1024.0
        return f"{amount:.1f} GB"

    def _zoom2d_reset(self) -> None:
        self._zoom2d_xlim = (LON_MIN, LON_MAX)
        self._zoom2d_ylim = (LAT_MIN, LAT_MAX)

    def _apply_2d_limits(self) -> None:
        x0, x1 = self._zoom2d_xlim
        y0, y1 = self._zoom2d_ylim
        base_w = LON_MAX - LON_MIN
        base_h = LAT_MAX - LAT_MIN
        min_w = base_w / 1_000.0
        min_h = base_h / 1_000.0
        if (x1 - x0) < min_w:
            cx = (x0 + x1) / 2.0
            half = min_w / 2.0
            x0, x1 = cx - half, cx + half
        if (y1 - y0) < min_h:
            cy = (y0 + y1) / 2.0
            half = min_h / 2.0
            y0, y1 = cy - half, cy + half
        x0 = max(LON_MIN, x0)
        x1 = min(LON_MAX, x1)
        y0 = max(LAT_MIN, y0)
        y1 = min(LAT_MAX, y1)
        self._zoom2d_xlim = (x0, x1)
        self._zoom2d_ylim = (y0, y1)
        if hasattr(self, 'ax2d'):
            self.ax2d.set_xlim(self._zoom2d_xlim)
            self.ax2d.set_ylim(self._zoom2d_ylim)

    def _on_2d_scroll(self, event) -> None:
        if event.inaxes is not self.ax2d or event.xdata is None or event.ydata is None:
            return
        button = getattr(event, 'button', None)
        if button == 'up':
            scale = 0.85
        elif button == 'down':
            scale = 1.0 / 0.85
        else:
            return
        x0, x1 = self._zoom2d_xlim
        y0, y1 = self._zoom2d_ylim
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            return
        rel_x = (event.xdata - x0) / width
        rel_y = (event.ydata - y0) / height
        rel_x = min(max(rel_x, 0.0), 1.0)
        rel_y = min(max(rel_y, 0.0), 1.0)
        new_width = width * scale
        new_height = height * scale
        new_x0 = event.xdata - rel_x * new_width
        new_y0 = event.ydata - rel_y * new_height
        new_x1 = new_x0 + new_width
        new_y1 = new_y0 + new_height
        base_w = LON_MAX - LON_MIN
        base_h = LAT_MAX - LAT_MIN
        if new_width > base_w:
            new_x0, new_x1 = LON_MIN, LON_MAX
        if new_height > base_h:
            new_y0, new_y1 = LAT_MIN, LAT_MAX
        self._zoom2d_xlim = (new_x0, new_x1)
        self._zoom2d_ylim = (new_y0, new_y1)
        self._apply_2d_limits()
        self.canvas2d.draw_idle()

    def _on_2d_key_press(self, event) -> None:
        if event.key and event.key.lower() == 'r':
            self._zoom2d_reset(); self._apply_2d_limits()
            self._apply_2d_limits()
            self.canvas2d.draw_idle()
    def _update_data_rate_label(self, payload_bytes: int, bytes_per_sec: float) -> None:
        if not hasattr(self, "lbl_data_rate"):
            return
        if payload_bytes > 0:
            text = f"Data ≈ {self._format_bytes(payload_bytes)} · {self._format_bytes(bytes_per_sec)}/s"
        else:
            text = "Data ≈ 0 B · 0 B/s"
        self.lbl_data_rate.setText(text)
        self.lbl_data_rate.setToolTip(
            f"Payload: {payload_bytes} bytes\nApprox rate: {bytes_per_sec:.1f} bytes/s"
        )

    def _init_ui(self):
        """UI 구성 + 기본값 세팅 (IP/Port 3쌍 입력 지원)"""
        root = QWidget(); self.setCentralWidget(root)
        h_main = QHBoxLayout(root); h_main.setContentsMargins(6, 6, 6, 6)

        # ─── 좌측 패널 ──────────────────────────────────────────
        left = QWidget(); v_left = QVBoxLayout(left); v_left.setSpacing(12)

        plan_box = QFrame(); plan_box.setStyleSheet("background:#f7f7ff; border:1px solid #ccd; border-radius:6px;")
        pb = QHBoxLayout(plan_box); pb.setContentsMargins(10, 8, 10, 8)
        self.lbl_plan_info = QLabel("운영 계획: (미설정)")
        pb.addWidget(self.lbl_plan_info, 1)
        v_left.addWidget(plan_box)

        # 1) Operation Info 테이블
        self.tbl_oper = QTableWidget(4, 2, selectionMode=QTableWidget.NoSelection)
        self.tbl_oper.setHorizontalHeaderLabels(["항목", "값"]); self.tbl_oper.verticalHeader().setVisible(False)
        self.tbl_oper.horizontalHeader().setStretchLastSection(True); self.tbl_oper.setColumnWidth(0, 150)
        for r, (k, v) in enumerate([
            ("Operation Start", "06:30"), ("Operation End", "21:30"),
            ("Actual Passengers", "-"), ("Throughput a day", "-")
        ]):
            self.tbl_oper.setItem(r, 0, QTableWidgetItem(k)); self.tbl_oper.setItem(r, 1, QTableWidgetItem(v))
        self.tbl_oper.setFixedHeight(4 * 30 + 28); v_left.addWidget(self.tbl_oper, stretch=0)

        # 2) Simulation Setting 테이블  -----------------------------------
        self.tbl_sim = QTableWidget(5, 2, selectionMode=QTableWidget.NoSelection)
        self.tbl_sim.setHorizontalHeaderLabels(["설정", "값"])
        self.tbl_sim.verticalHeader().setVisible(False)
        self.tbl_sim.horizontalHeader().setStretchLastSection(True)
        self.tbl_sim.setColumnWidth(0, 150)

        params = [
            ("Simulation Speed",      "1"),
            ("데이터 전송 주기",       "1 s"),
            ("IP/Port1",              "127.0.0.1:50051"),
            ("IP/Port2",              "127.0.0.1:50052"),
            ("IP/Port3",              "127.0.0.1:50053")
        ]
        for r, (k, v) in enumerate(params):
            self.tbl_sim.setItem(r, 0, QTableWidgetItem(k))
            self.tbl_sim.setItem(r, 1, QTableWidgetItem(v))

        self.tbl_sim.setFixedHeight(5 * 30 + 28)
        v_left.addWidget(self.tbl_sim, stretch=0)

        # ─── 통신·시뮬 버튼/로그 등 이하 내용은 그대로 ───────────────
        self.btn_comm = QPushButton("Open Communication"); self.btn_comm.clicked.connect(self._open_comm); v_left.addWidget(self.btn_comm)
        self.btn_start_sim = QPushButton("Start Simulation"); self.btn_start_sim.clicked.connect(self._toggle_sim); v_left.addWidget(self.btn_start_sim)
        self.btn_reset = QPushButton("Reset"); self.btn_reset.clicked.connect(self._reset_all); v_left.addWidget(self.btn_reset)

        self.chk_airsim = QCheckBox("AirSim Mode")
        self.chk_airsim.setToolTip("체크 시 AirSim과 연동합니다.")
        self.chk_airsim.setChecked(self._airsim_enabled)
        self.chk_airsim.stateChanged.connect(self._airsim_toggled)
        v_left.addWidget(self.chk_airsim)

        self.le_sim_time = QLineEdit(readOnly=True, alignment=Qt.AlignCenter); self.le_sim_time.setPlaceholderText("Simulation Time"); v_left.addWidget(self.le_sim_time)
        self.te_log = QTextEdit(readOnly=True); self.te_log.setPlaceholderText("Simulation Setting Log Only …"); v_left.addWidget(self.te_log, stretch=1)
        # ───────────────────────────────────────────────────────────────


        # == 중앙 ­log 창 ================================================
        self.te_play = QTextEdit(readOnly=True)
        self.te_play.setPlaceholderText("Simulation Play Log (Flight only)")
        self.te_play.setStyleSheet("background:#ffffff; color:#000000;")
        f = self.te_play.font(); f.setPointSize(12); self.te_play.setFont(f)

        # == 우측 2-D / 3-D 뷰 ==========================================
        self.fig2d, self.ax2d = plt.subplots(figsize=(5, 4))
        self.canvas2d = FigureCanvas(self.fig2d)
        self._zoom2d_reset(); self._apply_2d_limits()
        self.canvas2d.mpl_connect('scroll_event', self._on_2d_scroll)
        self.canvas2d.mpl_connect('key_press_event', self._on_2d_key_press)

        self.fig3d = plt.figure(figsize=(5, 4))
        self.ax3d  = self.fig3d.add_subplot(111, projection='3d')
        self.canvas3d = FigureCanvas(self.fig3d)

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(self.canvas2d)
        right_split.addWidget(self.canvas3d)
        right_split.setSizes([450, 450])

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(right_split, 1)
        self.lbl_data_rate = QLabel("Data ≈ 0 B · 0 B/s")
        self.lbl_data_rate.setAlignment(Qt.AlignRight)
        f_lbl = self.lbl_data_rate.font(); f_lbl.setPointSize(9); self.lbl_data_rate.setFont(f_lbl)
        self.lbl_data_rate.setStyleSheet("color:#555;")
        right_layout.addWidget(self.lbl_data_rate, 0, Qt.AlignRight)

        # == 레이아웃 배치 (폭 비율 조정) ================================
        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(left)
        main_split.addWidget(self.te_play)
        main_split.addWidget(right_panel)
        main_split.setSizes([350, 650, 500])
        h_main.addWidget(main_split)

        # 타이머 --------------------------------------------------------
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        # 초기 주기 = 테이블 값
        self._timer.start(self._get_tx_interval_ms())

    def set_plan_days(self, plan_days: list[tuple[dt.date, str]]):
        """런타임에 홈 없이도 주입 가능하도록 공개 메서드"""
        self._plan_days = plan_days or []
        self._plan_idx  = 0 if self._plan_days else -1
        if self._plan_idx >= 0:
            _, folder = self._plan_days[self._plan_idx]
            self._load_fpl_folder(folder)
        self._update_plan_label()

    def _apply_comm_config(self, cfg: dict):
        """SitlHome에서 전달된 통신 설정으로 소켓 열고 스레드 기동"""
        # 주기
        self._tx_interval_ms = int(cfg.get("tx_interval_ms", 1000))

        # 기존 소켓 정리
        for s in self.udp_socks.values():
            try: s.close()
            except: pass
        self.udp_socks.clear()

        # TX endpoints 오픈 (PING/PONG 확인)
        def _open_udp(ip: str, port: int, timeout: float = 1.0) -> bool:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.settimeout(timeout)
            try:
                sock.connect((ip, port))
                sock.send(b"PING")
                resp = sock.recv(16)
                if resp.strip().upper() == b"PONG":
                    self.udp_socks[(ip, port)] = sock
                    self.te_log.append(f"[OK] TX {ip}:{port}")
                    return True
                raise TimeoutError("bad reply")
            except Exception:
                self.te_log.append(f"[TIMEOUT] TX {ip}:{port}")
                sock.close()
                return False

        ok = 0
        for ip, port in cfg.get("tx_endpoints", []):
            if _open_udp(ip, int(port)): ok += 1
        self.te_log.append(f"[INFO] TX opened: {ok}/{len(cfg.get('tx_endpoints', []))}")

        # UDP 송신 스레드
        if self.udp_socks and not hasattr(self, "_udp_thread_started"):
            self._start_udp_thread()
            self._udp_thread_started = True

        self.te_log.append(f"[INFO] Tx interval = {self._tx_interval_ms} ms")

    def _auto_pick_latest_plan(self) -> bool:
        """Attempt to auto-fill plan_days with the most recent Scheduler FPL output."""
        base_dir = Path(__file__).resolve().parent.parent / "Scheduler" / "FPL_Result"
        try:
            candidates = [d for d in base_dir.iterdir() if d.is_dir()]
        except FileNotFoundError:
            return False

        dated: list[tuple[dt.date, Path]] = []
        for d in candidates:
            try:
                day = dt.datetime.strptime(d.name, "%Y%m%d").date()
            except ValueError:
                continue
            dated.append((day, d))

        if not dated:
            return False

        latest_day, latest_dir = max(dated, key=lambda item: item[0])
        self._plan_days = [(latest_day, str(latest_dir))]
        self._plan_idx = 0
        self._current_folder = None

        try:
            self.te_log.append(f"[INFO] Auto-selected FPL folder → {latest_dir}")
        except Exception:
            pass
        return True

    def _update_plan_label(self):
        if self._plan_idx >= 0 and self._plan_days:
            d, f = self._plan_days[self._plan_idx]
            self.lbl_plan_info.setText(f"운영 계획: {self._plan_idx+1}/{len(self._plan_days)}일차  {d.isoformat()}  ·  {Path(f).name}")
        else:
            self.lbl_plan_info.setText("운영 계획: (미설정)")

    def _open_plan_dialog(self):
        """비행계획(날짜→폴더) 등록 창을 띄우고, 저장 시 1일차 로딩"""
        preset = self._plan_days[:] if self._plan_days else None
        dlg = FlightPlanDialog(self, preset=preset)
        if dlg.exec_() != QDialog.Accepted:
            return
        plan = dlg.get_plan()
        if not plan:
            self.te_log.append("[WARN] 비행계획이 비었습니다.")
            return

        self._plan_days = plan
        self._plan_idx  = 0
        first_date, first_folder = self._plan_days[0]
        self.te_log.append(f"[INFO] 계획 등록: {len(self._plan_days)}일 / 1일차 {first_date}")
        self._load_fpl_folder(first_folder)

    def _load_fpl_folder(self, folder: str):
        try:
            self.sim = SitlSim(folder)
            self._current_folder = folder
            self.te_log.append(f"[INFO] FPL CSV folder → {folder}")
        except Exception as e:
            self.te_log.append(f"[ERROR] SitlSim 생성 실패: {e}")
            return

        try:
            std_times  = [fp.std for fp in getattr(self.sim, "_pending", [])]
            pax_total  = sum(getattr(fp, "pax", 0) for fp in getattr(self.sim, "_pending", []))
            n_flights  = len(getattr(self.sim, "_pending", []))
            if std_times:
                op_start = min(std_times).strftime("%H:%M")
                op_end   = max(std_times).strftime("%H:%M")
            else:
                op_start, op_end = "-", "-"

            self.tbl_oper.item(0,1).setText(op_start)
            self.tbl_oper.item(1,1).setText(op_end)
            self.tbl_oper.item(2,1).setText(f"{pax_total:,}")
            self.tbl_oper.item(3,1).setText(f"{n_flights:,}")
        except Exception:
            pass
        self._update_plan_label()

    def _check_autoroll(self, sim_snap: dict):
        """
        ① 당일 시뮬이 끝났는지 판단하고, ② 계획이 있고 ③ 다음 날짜가 '연속'이면
        다음 일자의 폴더로 자동 전환 후 즉시 시뮬 시작.
        """
        if not (self._sim_running and self.sim):
            return

        pending_left = len(getattr(self.sim, "_pending", []))
        if len(sim_snap) > 0 or pending_left > 0:
            return

        if not self._plan_days or self._plan_idx < 0 or self._plan_idx >= len(self._plan_days) - 1:
            self._sim_running = False
            self.btn_start_sim.setText("Start Simulation")
            self.te_log.append("[INFO] 당일 시뮬 종료 (계획 끝)")
            return

        cur_date, _  = self._plan_days[self._plan_idx]
        nxt_date, nxt_folder = self._plan_days[self._plan_idx + 1]
        if nxt_date != cur_date + dt.timedelta(days=1):
            self._sim_running = False
            self.btn_start_sim.setText("Start Simulation")
            self.te_log.append(f"[INFO] 다음 항목({nxt_date})은 연속 날짜가 아니므로 자동 전환 안 함")
            return

        self._plan_idx += 1
        self._load_fpl_folder(nxt_folder)
        self._update_plan_label()

        try:
            op_start = self.tbl_oper.item(0, 1).text() or "06:30"
            speed = float(self.tbl_sim.item(0, 1).text().rstrip('x') or 1)
        except Exception:
            op_start, speed = "06:30", 1.0

        self.sim.start(op_start, sim_speed=speed)
        self._sim_speed   = speed
        self._sim_time    = self.sim.sim_time
        self._sim_running = True
        self.btn_start_sim.setText("Stop Simulation")
        self.te_log.append(f"[INFO] Day rollover → {nxt_date} ({nxt_folder}) / {op_start}, {speed}x")

    def _browse_csv(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose FPL CSV Folder", str(Path.cwd()))
        if not folder:
            return
        self._load_fpl_folder(folder)

    def _lla_to_uech_m_fullprec(self, lat: float, lon: float, alt_m: float = 0.0):
        """
        1) geodetic_to_ue(lat,lon,0.0) → 절대 UE(cm) (NumPy float64)
        2) CITY_HALL_UE(cm) 빼서 CH-origin(cm)
        3) cm → m (JSON_XY_DIV/JSON_Z_DIV)
        ※ z는 ‘그대로’(뒤집지 않음), alt 더하지 않음 → 사용자가 보는 UE_CH와 동일 부호/단위
        """
        abs_cm = self._ue_mapper.geodetic_to_ue(lat, lon, float(alt_m))
        x_cm = abs_cm[0] - float(self._cityhall_ue[0])
        y_cm = abs_cm[1] - float(self._cityhall_ue[1])
        z_cm = abs_cm[2] - float(self._cityhall_ue[2])
        return (x_cm / self._div_xy, y_cm / self._div_xy, z_cm / self._div_z)


    def _lazy_load_vp_resources(self):
        """resources_vp.csv를 1회 로드하여
        (PortName, Label) 및 (정규화포트명, Label) → {'UE_X_m','UE_Y_m','UE_Z_m','N_m','E_m','D_m','yaw_deg'} 맵을 구축한다.
        - 파일 위치 우선순위:
        1) <SITL>/resource/resources_vp.csv
        2) <SITL>/resources_vp.csv
        - 열 이름은 X_m/Y_m/Z_m 우선, 없으면 X_cm/Y_cm/Z_cm → /100 스케일.
        - Z는 '그대로' 사용(뒤집지 않음); NED용 D는 -Z로 별도 저장.
        """
        if getattr(self, "_vp_layout", None) is not None:
            return
        self._vp_layout = {}
        self._vp_layout_norm = {}
        try:
            import csv, unicodedata, re
            base = Path(__file__).resolve().parent
            candidates = [
                base / "resource" / "resources_vp.csv",
                base / "resources_vp.csv",
            ]
            csv_path = None
            for c in candidates:
                if c.exists():
                    csv_path = c
                    break
            if csv_path is None:
                raise FileNotFoundError("resources_vp.csv not found in ./resource/ or ./.")

            def _num(row, *keys):
                for k in keys:
                    v = row.get(k)
                    if v is None: continue
                    s = str(v).strip()
                    if s == "":  continue
                    try: return float(s)
                    except: pass
                return None

            def _norm_port(s: str) -> str:
                s = unicodedata.normalize("NFKC", str(s))
                s = s.replace("\u00A0"," ").replace("\u2011","-").strip()
                return re.sub(r"\s+", "", s)   # 모든 공백 제거

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                r = csv.DictReader(f)
                
                import unicodedata, re, string
                # ★ 모든 공백류/제어문자/일부 구두점(.,·,-,_) 제거 + NFKC 정규화
                def _norm_port(s: str) -> str:
                    s = unicodedata.normalize("NFKC", str(s or ""))
                    s = s.replace("\u00A0"," ").strip()
                    # 공백류 제거
                    s = re.sub(r"\s+", "", s)
                    # 흔한 구분 기호 제거(가급적 폭넓게)
                    s = s.replace("·","").replace("-", "").replace("_","").replace(".", "")
                    return s

                for row in r:
                    # ★ 포트명 컬럼 우선순위에 Vertiport 추가(최우선)
                    port = (row.get("Vertiport") or row.get("PortName") or
                            row.get("port_name_raw") or row.get("port_name") or
                            row.get("VertiPort") or row.get("Name") or "").strip()
                    label = (row.get("Label") or row.get("label") or "").strip().upper()
                    if not port or not label:
                        continue

                    # m 우선, 없으면 cm → m
                    X_m = _num(row, "X_m","x_m","X_M")
                    Y_m = _num(row, "Y_m","y_m","Y_M")
                    Z_m = _num(row, "Z_m","z_m","Z_M")
                    if X_m is None or Y_m is None or Z_m is None:
                        X_cm = _num(row, "X_cm","x_cm","U_X","U_Xcm","U_XCM")
                        Y_cm = _num(row, "Y_cm","y_cm","U_Y","U_Ycm","U_YCM")
                        Z_cm = _num(row, "Z_cm","z_cm","U_Z","U_Zcm","U_ZCM")
                        if X_cm is None or Y_cm is None or Z_cm is None:
                            continue
                        X_m, Y_m, Z_m = X_cm/100.0, Y_cm/100.0, Z_cm/100.0

                    yaw_deg = _num(row, "yaw_deg","Yaw","YAW","angle","Angle") or 0.0

                    rec = {
                        "UE_X_m": X_m, "UE_Y_m": Y_m, "UE_Z_m": Z_m,  # ← UE_CH(m) 그대로
                        "N_m": X_m, "E_m": Y_m, "D_m": -Z_m,          # 참고용: NED (D=−Z)
                        "yaw_deg": yaw_deg,
                    }
                    self._vp_layout[(port, label)] = rec
                    self._vp_layout_norm[(_norm_port(port), label)] = rec

        except Exception as exc:
            self.te_log.append(f"[WARN] resources_vp.csv 로드 실패: {exc}")
            self._vp_layout = {}
            self._vp_layout_norm = {}



    def _ensure_prespawn_records(self, snap_for_airsim: dict) -> dict:
        """
        STD 1분 전인 항공편(아직 활성 스냅샷에 없는 ID)에 대해,
        resources_vp.csv의 'UE_CH(m)' 좌표(부호/스케일 그대로)를 사용해 미리 스폰 레코드를 추가한다.
        - 좌표: x=UE_X_m, y=UE_Y_m, z=UE_Z_m  (※ NED 변환/부호 반전 없음)
        - 헤딩: yaw_deg
        """
        try:
            if not self.sim or self.sim.sim_time is None:
                return snap_for_airsim

            self._lazy_load_vp_resources()
            now = self.sim.sim_time
            out = dict(snap_for_airsim)

            pending = getattr(self.sim, "_pending", []) or []
            for fp in list(pending):
                try:
                    dt_sec = (fp.std - now).total_seconds()
                except Exception:
                    continue
                # 정확히 1분 이내(초과 미포함) 창에서만 스폰
                if not (0.0 < dt_sec <= 30.0 + 1e-6):
                    continue
                acid = fp.id
                if acid in out:
                    continue  # 이미 활성화되어 있으면 스킵

                port = getattr(fp, "origin", None)
                gate_no = getattr(fp, "dep_gate_no", None)
                fato_no = getattr(fp, "dep_fato_no", None)

                label = None
                if gate_no:
                    label = f"GATE {gate_no}".upper()
                elif fato_no:
                    label = f"FATO {fato_no}".upper()
                if not (port and label):
                    continue

                # 맵 검색: 원문 키 → 없으면 정규화 키
                rec = self._vp_layout.get((port, label))
                if not rec:
                    from unicodedata import normalize
                    def _norm(s: str) -> str:
                        s = normalize("NFKC", str(s or ""))
                        s = s.replace("\u00A0"," ").strip()
                        s = re.sub(r"\s+", "", s)
                        s = s.replace("·","").replace("-", "").replace("_","").replace(".", "")
                        return s
                    rec = self._vp_layout_norm.get((_norm(port), label))
                if not rec:
                    self.te_log.append(f"[SPAWN-1min][MISS] {acid} key=({port},{label}) 매칭 불가")
                    continue

                # === UE_CH(m) 그대로 사용 ===
                x = rec["N_m"]      # (= UE_X_m)
                y = rec["E_m"]      # (= UE_Y_m)
                z = rec["D_m"] 

                # 로그(정밀)
                self.te_log.append(
                    f"[SPAWN-1min] {acid} @ {port} / {label}  "
                    f"UE_CH=({x:.10f},{y:.10f},{z:.10f})  yaw={rec.get('yaw_deg',0.0):.5f}°"
                )

                out[acid] = {
                    "id": acid,
                    # LLA는 표시용(옵션). 역변환은 생략
                    "lat": float("nan"), "lon": float("nan"), "alt_m": float("nan"),
                    "x": x, "y": y, "z": z,
                    "heading_deg": float(rec.get("yaw_deg", 0.0)),
                    "progress_m": 0.0,
                    "remain_m": 0.0,
                    "phase": "SPAWN",
                    "lane": None,
                    "pax": getattr(fp, "pax", 0),
                    "local_id": getattr(fp, "local_id", ""),
                    "atd": "-", "eta": "-",
                    "From": getattr(fp, "origin", None),
                    "To": getattr(fp, "dest", None),
                    "DepFATO_No": getattr(fp, "dep_fato_no", None),
                    "DepGate_No": getattr(fp, "dep_gate_no", None),
                }
            return out
        except Exception as exc:
            self.te_log.append(f"[SPAWN-1min][ERR] {exc}")
            return snap_for_airsim


    def _start_udp_thread(self):
        """
        현재 시뮬레이션 스냅샷을 JSON 으로 묶어 모든 UDP 소켓에 송신한다.
        (백그라운드 스레드 → GUI 로그는 pyqtSignal 로 전달해야 안전)
        """
        if not self.udp_socks:
            return

        def _udp_tx_loop():
            while self.udp_socks:
                fleet: dict = {}

                if self._sim_running and self.sim:
                    fleet = self.sim.snapshot()      # {id: {...}}

                if not fleet:
                    time.sleep(self._get_tx_interval_ms() / 1000)
                    continue

                now_str = (self.sim.sim_time.strftime("%H:%M:%S")
                           if self._sim_running and self.sim
                           else time.strftime("%H:%M:%S"))
                blob = json.dumps({"time": now_str, "fleet": fleet}).encode()

                for s in list(self.udp_socks.values()):
                    try:
                        s.send(blob)
                    except Exception:
                        pass

                self.log_msg.emit(
                    f"[TX] {now_str}  "
                    f"sim:{len(fleet):>2}  "
                    f"→ {len(self.udp_socks)} sock"
                )
                time.sleep(self._get_tx_interval_ms() / 1000)

        threading.Thread(target=_udp_tx_loop, daemon=True).start()

    def _get_tx_interval_ms(self) -> int:
        return int(getattr(self, "_tx_interval_ms", 1000))

    def _reset_all(self):
        """시뮬·통신·UI 상태를 모두 초기화한다."""
        if self._sim_running and self.sim:
            self.sim.stop()
        self._sim_running = False
        self.btn_start_sim.setText("Start Simulation")

        if self._airsim_enabled or self._airsim_bridge:
            self._disable_airsim()
        else:
            self._stop_airsim_worker()
            if self._airsim_bridge:
                try:
                    self._airsim_bridge.reset()
                except Exception as exc:
                    self.te_log.append(f"[WARN] AirSim bridge reset 실패: {exc}")
                self._airsim_bridge = None
                self._airsim_client = None

        for s in self.udp_socks.values():
            s.close()
        self.udp_socks.clear()

        self.le_fpl.clear() if hasattr(self, "le_fpl") else None
        self.le_sim_time.clear()
        for i in range(4):
            self.tbl_oper.item(i, 1).setText("-")
        self.tbl_sim.item(0, 1).setText("1")
        self.tbl_sim.item(1, 1).setText("1 s")

        self.te_log.clear()
        self.te_play.clear()

        self.sim = None
        self._sim_time = None
        self.te_log.append("[INFO] Reset completed")
        if hasattr(self, "_udp_thread_started"):
            del self._udp_thread_started

    def _set_airsim_checkbox(self, checked: bool) -> None:
        if hasattr(self, "chk_airsim"):
            self.chk_airsim.blockSignals(True)
            self.chk_airsim.setChecked(checked)
            self.chk_airsim.blockSignals(False)

    def _airsim_toggled(self, state: int) -> None:
        if state == Qt.Checked:
            self._enable_airsim()
        else:
            self._disable_airsim()

    def _enable_airsim(self) -> None:
        if self._airsim_enabled:
            return
        if airsim is None:
            self.te_log.append("[ERROR] AirSim Python API를 찾을 수 없습니다. (pip install airsim)")
            self._set_airsim_checkbox(False)
            return
        try:
            client = airsim.VehicleClient()
            client.confirmConnection()
        except Exception as exc:
            self.te_log.append(f"[ERROR] AirSim 연결 실패: {exc}")
            self._set_airsim_checkbox(False)
            return
        self._airsim_client = client
        self._airsim_bridge = AirSimFleetBridge(client, log_cb=self.log_msg.emit)

        self._airsim_enabled = True
        self._airsim_error_flag = False
        self._start_airsim_worker()
        self._set_airsim_checkbox(True)
        self.te_log.append("[OK] AirSim bridge 활성화")

    def _disable_airsim(self) -> None:
        if not self._airsim_enabled:
            return
        self._stop_airsim_worker()
        if self._airsim_bridge:
            try:
                self._airsim_bridge.reset()
            except Exception as exc:
                self.te_log.append(f"[WARN] AirSim bridge reset 실패: {exc}")
        self._airsim_bridge = None
        self._airsim_client = None
        self._airsim_enabled = False
        self._set_airsim_checkbox(False)
        self.te_log.append("[INFO] AirSim bridge 비활성화 완료")

    def _start_airsim_worker(self) -> None:
        if self._airsim_thread and self._airsim_thread.is_alive():
            return
        self._airsim_stop_event = threading.Event()
        self._airsim_queue = queue.Queue(maxsize=1)
        self._airsim_thread = threading.Thread(target=self._airsim_worker_loop, daemon=True)
        self._airsim_thread.start()

    def _stop_airsim_worker(self) -> None:
        stop_event = self._airsim_stop_event
        if stop_event:
            stop_event.set()
        if self._airsim_queue:
            try:
                self._airsim_queue.put_nowait(None)
            except queue.Full:
                try:
                    self._airsim_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._airsim_queue.put_nowait(None)
                except queue.Full:
                    pass
        thread = self._airsim_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._airsim_thread = None
        self._airsim_queue = None
        self._airsim_stop_event = None

    def _airsim_worker_loop(self) -> None:
        while self._airsim_stop_event and not self._airsim_stop_event.is_set():
            if not self._airsim_queue:
                break
            try:
                payload = self._airsim_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is None:
                continue
            snapshot, now_t = payload
            bridge = self._airsim_bridge
            if not bridge:
                continue
            try:
                bridge.sync_snapshot(snapshot, now_t, use_bank=False)
            except Exception as exc:
                self._airsim_error_flag = True
                self.log_msg.emit(f"[ERROR] AirSim 업데이트 실패: {exc}")

    def _open_comm(self):
        """
        IP/Port1‥3 셀에 적힌 모든 “ip:port” 엔드포인트를 시도해
        UDP 소켓을 만들고 self.udp_socks에 등록한다.
        · 성공/실패 및 전체 요약을 Log 창에 남김.
        """
        for s in self.udp_socks.values():
            s.close()
        self.udp_socks.clear()

        entries: list[str] = []
        for row in range(2, 5):
            txt = (self.tbl_sim.item(row, 1).text() or "").strip()
            if not txt:
                continue
            entries += [ep.strip() for ep in txt.split(",") if ep.strip()]

        if not entries:
            self.te_log.append("[WARN] IP/Port 목록이 비어 있습니다.")
            return

        def _open_udp(ip: str, port: int, timeout: float = 1.0) -> bool:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            try:
                sock.connect((ip, port))
                sock.send(b"PING")
                resp = sock.recv(16)
                if resp.strip().upper() == b"PONG":
                    self.udp_socks[(ip, port)] = sock
                    self.te_log.append(f"[OK]      {ip}:{port} ")
                    return True
                raise TimeoutError("bad reply")
            except Exception:
                self.te_log.append(f"[TIMEOUT] {ip}:{port})")
                sock.close()
                return False

        ok_cnt = 0
        for ep in entries:
            if ":" not in ep:
                self.te_log.append(f"[WARN] '{ep}' → ip:port 형식 아님")
                continue
            ip, port_s = ep.split(":", 1)
            try:
                port = int(port_s)
            except ValueError:
                self.te_log.append(f"[WARN] '{ep}' → 잘못된 port")
                continue
            if _open_udp(ip, port):
                ok_cnt += 1

        if ok_cnt:
            self.te_log.append(f"[INFO] 총 {ok_cnt}/{len(entries)}개 소켓 연결 완료")
        else:
            self.te_log.append("[ERROR] 소켓을 하나도 열지 못했습니다.")
        if ok_cnt == 0:
            self.te_log.append("[WARN] TX 연결없음")

        if self.udp_socks and not hasattr(self, "_udp_thread_started"):
            self._start_udp_thread()
            self._udp_thread_started = True

    def _toggle_sim(self):
        if not self.sim:
            self.te_log.append("[WARN] 먼저 FPL 폴더를 선택하세요."); return
        if not self.udp_socks:
            self.te_log.append("[INFO] UDP 연결 없이 시뮬레이션을 시작합니다.")

        try:
            speed = float(self.tbl_sim.item(0, 1).text().rstrip('x') or 1)
        except Exception:
            speed = 1.0

        if self._sim_running:
            self.sim.stop()
            self._sim_running = False
            self.btn_start_sim.setText("Start Simulation")
            self.te_log.append("[INFO] Simulation paused")
            return

        if self.sim is None:
            if self._plan_days and 0 <= self._plan_idx < len(self._plan_days):
                _, folder = self._plan_days[self._plan_idx]
                if folder != self._current_folder:
                    self._load_fpl_folder(folder)
            if self.sim is None:
                self.te_log.append("[WARN] 홈 화면에서 비행계획을 설정하거나 FPL을 로드해 주세요.")
                return

        if self.sim.sim_time is None:
            op_start = self.tbl_oper.item(0, 1).text() or "06:30"
            self.sim.start(op_start, sim_speed=speed)
            self.te_log.append(f"[INFO] Simulation started ({op_start}, {speed}x)")
        else:
            self.sim.sim_speed = speed
            self.sim.running   = True
            self.te_log.append(f"[INFO] Simulation resumed ({self.sim.sim_time.strftime('%H:%M:%S')}, {speed}x)")

        self._sim_speed   = speed
        self._sim_time    = self.sim.sim_time
        self._sim_running = True
        self.btn_start_sim.setText("Stop Simulation")

        if self.udp_socks and not hasattr(self, "_udp_thread_started"):
            self._start_udp_thread()
            self._udp_thread_started = True

    def _tick(self):
        """GUI 주기적으로 호출되는 하트비트 + 화면 갱신 루틴."""
        base_tx_ms = self._get_tx_interval_ms()
        airsim_locked = self._airsim_enabled and (self._airsim_bridge is not None)
        if self._airsim_error_flag:
            self._airsim_error_flag = False
            if self._airsim_enabled:
                self._disable_airsim()
            airsim_locked = False
            base_tx_ms = self._get_tx_interval_ms()

        now_monotonic = time.monotonic()
        last_ts = getattr(self, "_last_tick_ts", None)
        if last_ts is None:
            real_elapsed = base_tx_ms / 1000.0
        else:
            real_elapsed = now_monotonic - last_ts
            if real_elapsed <= 0:
                real_elapsed = base_tx_ms / 1000.0
        self._last_tick_ts = now_monotonic

        current_speed = getattr(self, "_sim_speed", 1.0)

        simulate_now = self._sim_running and (self.sim is not None)

        snap = {}
        if simulate_now and self.sim is not None:
            try:
                new_speed = float(self.tbl_sim.item(0, 1).text().rstrip('x') or 1)
            except Exception:
                new_speed = 1.0
            orig_speed = new_speed
            if new_speed < 0.0:
                new_speed = 0.0
            max_speed = 30.0
            if new_speed > max_speed:
                new_speed = max_speed
            if new_speed != orig_speed:
                item = self.tbl_sim.item(0, 1)
                if item is not None:
                    item.setText(f"{new_speed:g}")
                self.te_log.append(f"[INFO] Simulation speed limited to {new_speed:g}x")
            self.sim.sim_speed = new_speed
            self._sim_speed = new_speed
            current_speed = new_speed

            dt_sim = getattr(self.sim, "dt_sim", 1.0) or 1.0
            sim_accum = getattr(self, "_sim_step_accum", 0.0)
            sim_target_sec = (real_elapsed * new_speed) + sim_accum if new_speed > 0 else sim_accum
            remaining = sim_target_sec
            took_snapshot = False
            if new_speed > 0:
                while remaining > 1e-6:
                    sim_chunk = min(dt_sim, remaining)
                    self.sim.step(sim_chunk / new_speed)
                    snap = self.sim.snapshot()
                    took_snapshot = True
                    remaining -= sim_chunk
            else:
                snap = self.sim.snapshot()
            self._sim_step_accum = remaining if new_speed > 0 else sim_target_sec
            if not took_snapshot and self.sim is not None:
                snap = self.sim.snapshot()
        elif self.sim is not None:
            snap = self.sim.snapshot()
            self._sim_step_accum = 0.0
        else:
            self._sim_step_accum = 0.0

        speed_for_interval = current_speed if simulate_now else 1.0
        speed_for_interval = max(speed_for_interval, 1.0)
        effective_ms = max(33, int(round(base_tx_ms / speed_for_interval)))
            
        self._ui_interval_ms = effective_ms
        if effective_ms != self._timer.interval():
            self._timer.setInterval(effective_ms)
            self.te_log.append(f"[INFO] UI 주기 → {effective_ms} ms")

        payload_bytes = 0
        bytes_per_sec = 0.0
        if snap:
            try:
                serialized = json.dumps(snap, ensure_ascii=False, separators=(',', ':'))
                payload_bytes = len(serialized.encode("utf-8"))
                interval_ms = self._ui_interval_ms or self._timer.interval() or self._tx_interval_ms or 1000
                if interval_ms <= 0:
                    interval_ms = 1
                bytes_per_sec = payload_bytes * 1000.0 / interval_ms
            except Exception:
                payload_bytes = 0
                bytes_per_sec = 0.0
        self._update_data_rate_label(payload_bytes, bytes_per_sec)

        # ===== [MOD] AirSim 모드 전용: LLA→(UE_CH m)→NED로 x,y,z 치환 후 전송 =====
        if airsim_locked and snap and self._airsim_queue:
            snap_for_airsim = self._snapshot_for_airsim(snap)
            snap_for_airsim = self._ensure_prespawn_records(snap_for_airsim)
            payload = (snap_for_airsim, time.perf_counter())
            try:
                self._airsim_queue.put_nowait(payload)
            except queue.Full:
                try:
                    self._airsim_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._airsim_queue.put_nowait(payload)
                except queue.Full:
                    pass
        # =====================================================================

        if not snap:
            return

        if simulate_now:
            self._sim_time = self.sim.sim_time
            self.le_sim_time.setText(self._sim_time.strftime("%H:%M:%S"))
        else:
            self.le_sim_time.setText("—:—:—")

        self._check_autoroll(snap)

        self._update_play_log(snap)

        # ---- 2‑D ----
        self.ax2d.clear()
        self.ax2d.set_title("2‑D Top View (lat, lon)")
        self.ax2d.set_xlabel("Longitude"); self.ax2d.set_ylabel("Latitude")
        for v in snap.values():
            self._draw_2d_point(v, color='k')
        self._apply_2d_limits()
        self.ax2d.set_aspect('equal', 'box')
        self.canvas2d.draw_idle()

        # ---- 3‑D ----
        self.ax3d.clear()
        self.ax3d.set_title("3‑D View (lat, lon, alt)")
        self.ax3d.set_xlabel("Longitude"); self.ax3d.set_ylabel("Latitude"); self.ax3d.set_zlabel("Alt [m]")
        for v in snap.values():
            self._draw_3d_point(v, color='r')
        self.ax3d.set_xlim(LON_MIN, LON_MAX)
        self.ax3d.set_ylim(LAT_MIN, LAT_MAX)
        self.canvas3d.draw_idle()


    def _should_log_play(self, acid: str, pkt: dict) -> bool:
        """스팸 방지: phase/lane 변경, 충분히 이동/회전했을 때만 true."""
        last = self._last_play.get(acid)
        if not last:
            return True
        # phase/lane 변화면 즉시 로깅
        if pkt.get('phase') != last.get('phase') or (pkt.get('lane') or '-') != (last.get('lane') or '-'):
            return True
        # 충분히 이동?
        try:
            dx = float(pkt['x']) - float(last['x'])
            dy = float(pkt['y']) - float(last['y'])
            dz = float(pkt['z']) - float(last['z'])
        except Exception:
            dx = dy = dz = 0.0
        moved = (dx*dx + dy*dy + dz*dz) ** 0.5 >= self._log_pos_eps_m
        if moved:
            return True
        # 충분히 회전?
        try:
            dh = float(pkt['heading_deg']) - float(last['heading_deg'])
            dh = (dh + 180.0) % 360.0 - 180.0
        except Exception:
            dh = 0.0
        turned = abs(dh) >= self._log_hdg_eps_deg
        return turned

    # SitlMainWindow 내부
    def _update_play_log(self, snap: dict) -> None:
        """중앙 'Simulation Play Log'를 간결하게 갱신한다.
        - lane 필드 제거
        - x,y,z, heading 중심
        - ETA, remain_m 간단 표시
        - 과부하 방지: 최소 출력 간격 + 최대 라인 수 제한
        """
        if not snap:
            return

        # 과부하 방지: 최소 간격(ms)
        min_interval_ms = getattr(self, "_play_log_min_interval_ms", 300)
        now_mono = time.monotonic()
        last = getattr(self, "_play_log_last_emit", 0.0)
        if (now_mono - last) * 1000.0 < min_interval_ms:
            return
        self._play_log_last_emit = now_mono

        now_str = (self.sim.sim_time.strftime("%H:%M:%S")
                if (self._sim_running and self.sim and self.sim.sim_time)
                else time.strftime("%H:%M:%S"))

        lines = []
        # 보기 좋게 ID/LocalID/ID 중 하나 우선
        for _, pkt in sorted(snap.items()):
            reg  = pkt.get("reg") or pkt.get("local_id") or pkt.get("id") or "AC"
            ph   = (pkt.get("phase") or "-")[:1]           # 한 글자만
            x    = float(pkt.get("x") if pkt.get("x") is not None else float('nan'))
            y    = float(pkt.get("y") if pkt.get("y") is not None else float('nan'))
            z    = float(pkt.get("z") if pkt.get("z") is not None else float('nan'))
            hdg  = float(pkt.get("heading_deg") if pkt.get("heading_deg") is not None else float('nan'))
            eta  = str(pkt.get("eta") or "-")
            rem  = int(pkt.get("remain_m") or 0)

            lines.append(
                f"{now_str}  {reg:<10s}  ph={ph}  "
                f"x={x:8.1f} y={y:8.1f} z={z:6.1f}  hdg={hdg:6.1f}°  "
                f"ETA={eta:>8s} rem={rem:,}m"
            )

        if lines:
            self.te_play.append("\n".join(lines))
            self._trim_play_log()

    def _trim_play_log(self, max_rows: int = 400) -> None:
        """플레이 로그 라인 수 상한 유지(과부하 방지)."""
        doc = self.te_play.document()
        extra = doc.blockCount() - max_rows
        if extra <= 0:
            return
        cur = QTextCursor(doc)
        cur.movePosition(QTextCursor.Start)
        for _ in range(extra):
            cur.select(QTextCursor.LineUnderCursor)
            cur.removeSelectedText()
            cur.deleteChar()




    def _draw_2d_point(self, v: dict, color: str):
        """2‑D 점 + heading 화살표 한 기 그리기"""
        self.ax2d.plot(v["lon"], v["lat"], marker='o', ms=6, color=color)
        hdg = math.radians(v["heading_deg"])
        dx  = ARROW_LEN_LL * math.sin(hdg)
        dy  = ARROW_LEN_LL * math.cos(hdg)
        self.ax2d.arrow(v["lon"], v["lat"], dx, dy,
                         head_width=ARROW_LEN_LL*0.6,
                         head_length=ARROW_LEN_LL*0.6,
                         fc=color, ec=color)

    def _draw_3d_point(self, v: dict, color: str):
        """3‑D 점 + heading 선 한 기 그리기"""
        self.ax3d.scatter(v["lon"], v["lat"], v["alt_m"], c=color, s=20)
        hdg = math.radians(v["heading_deg"])
        dx  = ARROW_LEN_LL * math.sin(hdg)
        dy  = ARROW_LEN_LL * math.cos(hdg)
        self.ax3d.plot([v["lon"], v["lon"] + dx],
                        [v["lat"], v["lat"] + dy],
                        [v["alt_m"], v["alt_m"]], c=color)



# ─── entrypoint ─────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    win = SitlMainWindow(); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
