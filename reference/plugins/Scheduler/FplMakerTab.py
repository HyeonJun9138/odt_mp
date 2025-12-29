# FplMakerTab.py (수정본)
# 참고: RUNWAY_COUNT/TAKEOFF_MIN 등 스케줄링 상수는 Scheduling_Optimized.py에서 가져옵니다. :contentReference[oaicite:0]{index=0}
# 참고: 게이트/패드 타임라인 엔진과 NEW_DEP_PREOCCUPY_MIN, GATE_EXIT_LINGER_SEC 등은 GateResources.py를 참조합니다. :contentReference[oaicite:1]{index=1}

from __future__ import annotations

from PyQt5.QtCore    import Qt,pyqtSignal, QTime
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QComboBox, QTableWidget, QTableWidgetItem, QTextEdit, QGroupBox,
    QSizePolicy, QFrame, QHeaderView, QApplication, QProgressBar, QComboBox ,QAbstractItemView,QSpinBox, QTimeEdit, QCheckBox, QToolTip
)
from .Functions.Scheduling_Optimized import *

import math, os, tempfile, re
from pathlib import Path
import datetime as dt

import folium
from PyQt5.QtCore       import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas   # (추후 Node-Link용)
# Project-local
import sys
from pathlib import Path
from datetime import timedelta
from math import ceil
from math import isinf
from collections import defaultdict, deque
import datetime as dt, math
from PyQt5.QtGui       import QColor, QCursor  # ★ Delay 행 색상용
from matplotlib.figure  import Figure
import matplotlib.dates as mdates
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]   #  . (Scheduling 의 한 단계 위)
sys.path.append(str(ROOT_DIR))                   #  PYTHONPATH에 추가

# -------------------------------------------------------------------
from Scheduler.Functions.PathPlanning import PathPlanner, PathVisualizerGeo, rebuild_route, _km_to_dlon_dlat
from Scheduler.Functions.ETAComputer import compute_eta
from Scheduler.Functions.Convert_FPL import csvs_to_fpl_json

# ★ Gate 리소스 + 좌표 변환(UE→lon/lat, lon/lat→내부 xy[m])
from Scheduler.Functions.GateResources import (
    NetworkState, LOCKED_GATES_ALL_PORTS, GATE_COUNT_DEFAULT,
    load_resources, find_lonlat, lonlat_to_xy_m,
    NEW_DEP_PREOCCUPY_MIN,    # ← 추가
    GATE_EXIT_LINGER_SEC      # (폴백 계산시 사용할 수 있음)
)

# ★ A/K 삽입 + B/I/J 보정 유틸
from Scheduler.Functions.UAM_Path2Sim import path_to_profile, inject_ground_and_fato

class FplMakerTab(QWidget):

    generation_done = pyqtSignal(object)
    
    # ★ 표 헤더: T-Gate / L-Gate 복구
    COL_HEADERS = ["callSign", "regNum", "Type", "Pax",
                   "From", "STD", "ETOT","ATOT","T-Gate","T-Pad","To",
                   "ELDT", "ALDT", "L-Gate","L-Pad"]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.save_root: Path | None = None
        self._build_ui()
        self._flights: dict[str, list[dict]] = {}
        self._uam_hold_until = defaultdict(lambda: dt.datetime.min)
        self._uam_gate_pref = defaultdict(dict)
        self._uam_counter = 1                     # ★ UAM ID 카운터 영속화
        self.combo_vert.currentTextChanged.connect(self._on_vert_select)

        self.tbl_flights.itemSelectionChanged.connect(self._on_flight_select)
        self._last_std_by_origin = {}            # ★ 동일 STD 오프셋용(Origin별 마지막 STD)

        # ─────────────────────────────────────────────────────────
        # ★ 시간 파라미터(게이트 시퀀스 구성요소) — Prep(min) 없음
        #    READY(min) = LANDING_MIN(2) + TAXI_IN(5) + GATE_SERVICE(6) = 13
        self._taxi_out_min: int     = 5   # GATE → FATO
        self._taxi_in_min: int      = 5   # FATO → GATE
        self._gate_service_min: int = 6   # 하차/정리/탑승
        # ─────────────────────────────────────────────────────────
        self._gate_net = None
        self._gate_base0 = None
        self._dump_gate_csv = False
        self._integrated_scheduled = False

        # resources_vp.csv 경로(설정 가능)
        self._resources_csv: Path | None = None

    def set_save_root(self, path: Path):
        self.save_root = Path(path)

    def set_resources_csv(self, path: str | Path) -> None:
        """resources_vp.csv 절대/상대경로를 지정합니다."""
        p = Path(path)
        self._resources_csv = p if p.exists() else None

    # (호환 유지) main.py 가 부를 수 있으므로 남겨두되, 의미는 taxi_out으로 매핑
    def set_prep_min(self, v: int) -> None:
        try:
            self._taxi_out_min = int(v)  # 더 이상 'Prep' 로직은 없음
        except Exception:
            pass

    def reset_state_for_new_demands(self):
        """메인에서 Demand를 새로 만들 때 FPL 탭 내부 상태 초기화(호출될 수 있음)."""
        self._gate_net = None
        self._gate_base0 = None
        self._integrated_scheduled = False
        self._uam_hold_until.clear()
        self._last_std_by_origin.clear()
        self._uam_gate_pref.clear()

    # ------------------------------------------------------------------
    # UI builder (생략 없이 원문 유지)
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # … (원문 그대로) …
        root = QHBoxLayout(self)
        # =================================================================
        # ❶ LEFT PANE – Map & Operation-info
        # =================================================================
        left_widget = QWidget(); v_left = QVBoxLayout(left_widget)
        self._planner: PathPlanner | None = None
        self.map_view = MapView(None, zoom=11, center=(37.5665, 126.9780))
        self.map_view.setSizePolicy(
            QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        )
        v_left.addWidget(self.map_view, 1)

        gb_ops = QGroupBox("Operation Info"); grid = QGridLayout(gb_ops)
        self.tbl_ops = QTableWidget(4, 2)
        self.tbl_ops.verticalHeader().setVisible(False)
        self.tbl_ops.horizontalHeader().setVisible(False)
        self.tbl_ops.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_ops.setFrameShape(QFrame.Box)
        self.tbl_ops.horizontalHeader().setStretchLastSection(True)
        self.tbl_ops.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_ops.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for r, (k, v) in enumerate([
            ("Operation Start",  "06:30"),
            ("Operation End",    "21:30"),
            ("Number of Demand", "0"),
            ("Throughput a day", "0"),
        ]):
            self.tbl_ops.setItem(r, 0, QTableWidgetItem(k))
            self.tbl_ops.setItem(r, 1, QTableWidgetItem(v))
        grid.addWidget(self.tbl_ops, 0, 0)
        v_left.addWidget(gb_ops, 0)

        root.addWidget(left_widget, 1)
        # =================================================================
        # ❷ RIGHT PANE – Controls / Status / Flights / Logs
        # =================================================================
        right_widget = QWidget(); v_right = QVBoxLayout(right_widget)
        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(self._generate)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save)
        self.btn_gatewin = QPushButton("Open Gate Timeline Window")
        self.btn_gatewin.clicked.connect(self._open_gate_window)

        self.spin_maxsorties = QSpinBox()
        self.spin_maxsorties.setRange(1, 50)
        self.spin_maxsorties.setValue(30)
        self.spin_maxsorties.setSuffix(" flights/UAM")

        self.spin_extrawait = QSpinBox()
        self.spin_extrawait.setRange(0, 60)
        self.spin_extrawait.setValue(0)
        self.spin_extrawait.setSuffix(" min")

        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("Max Sorties per UAM"), 0, 0)
        settings_layout.addWidget(self.spin_maxsorties,           0, 1)
        settings_layout.addWidget(QLabel("Extra wait for UAM"),   1, 0)
        settings_layout.addWidget(self.spin_extrawait,            1, 1)
        settings_group = QGroupBox("Settings")
        settings_group.setLayout(settings_layout)
        v_right.addWidget(settings_group)

        btn_row.addWidget(self.btn_generate, 1)
        btn_row.addWidget(self.btn_save,     1)
        btn_row.addWidget(self.btn_gatewin,  1)
        v_right.addLayout(btn_row)

        self.combo_vert = QComboBox()
        self.combo_vert.addItem("(버티포트)")
        v_right.addWidget(self.combo_vert, 0)

        self.pbar = QProgressBar(); self.pbar.setRange(0, 100); self.pbar.setValue(0)
        v_right.addWidget(self.pbar)

        self.lbl_status = QLabel("대기 중", alignment=Qt.AlignCenter)
        self.lbl_status.setFrameShape(QFrame.Box)
        v_right.addWidget(self.lbl_status)

        self.tbl_flights = QTableWidget(8, len(self.COL_HEADERS))
        self.tbl_flights.setHorizontalHeaderLabels(self.COL_HEADERS)
        self.tbl_flights.verticalHeader().setVisible(False)
        self.tbl_flights.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_flights.setFrameShape(QFrame.Box)
        self.tbl_flights.horizontalHeader().setStretchLastSection(True)
        self.tbl_flights.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v_right.addWidget(self.tbl_flights, 2)
        hdr = self.tbl_flights.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        fm = self.tbl_flights.fontMetrics()
        for col, label in enumerate(self.COL_HEADERS):
            w = fm.horizontalAdvance(label) + 20
            hdr.resizeSection(col, w)
        self.tbl_flights.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)

        self.txt_logs = QTextEdit(); self.txt_logs.setReadOnly(True)
        self.txt_logs.setFrameShape(QFrame.Box)
        self.txt_logs.setPlaceholderText("All Logs")
        v_right.addWidget(self.txt_logs, 1)

        gb_gate = QGroupBox("Gate Timeline (Selected Port)"); v_gate = QVBoxLayout(gb_gate)
        self.gate_canvas = FigureCanvas(Figure(figsize=(6,2)))
        self.gate_ax = self.gate_canvas.figure.add_subplot(111)
        v_gate.addWidget(self.gate_canvas, 1)
        v_right.addWidget(gb_gate, 1)
        gb_gate.setVisible(False)
        root.addWidget(right_widget, 1)
        self.combo_vert.currentTextChanged.connect(self._on_vert_select)

    def set_planner(self, vert_csv: str | Path, wp_csv: str | Path) -> None:
        from Scheduler.Functions.PathPlanning import PathPlanner
        self._planner = PathPlanner(vert_csv, wp_csv)
        self.map_view.set_planner(self._planner)
        self.set_status("네트워크 로드 완료")

    def _on_flight_select(self):
        row = self.tbl_flights.currentRow()
        if row < 0:
            return
        fn = self.tbl_flights.item(row, 0).text()
        origin = self.combo_vert.currentText()
        for f in self._flights.get(origin, []):
            if f["flight_number"] == fn:
                self._visualize_route(f)
                break

    # ─────────────────────────────────────────────────────────────
    # Save: A/K 삽입 + B/I/J 보정 반영 + ★ FATO/GATE 실제 진입·진출 시각 저장
    # ─────────────────────────────────────────────────────────────
    def _save(self):
        if not self._flights:
            self.log("⚠ 저장할 비행계획이 없습니다.")
            return
        if self._planner is None:
            self.log("⚠ 네트워크(플래너)를 먼저 로드하세요")
            return

        import pandas as pd, datetime as dt, shutil, os, math, csv, pathlib
        from Scheduler.Functions.PathPlanning  import rebuild_route, _km_to_dlon_dlat
        # ground 삽입 유틸
        from Scheduler.Functions.UAM_Path2Sim  import path_to_profile, inject_ground_and_fato

        planner = self.map_view._planner

        # 0) Vertiport 인덱스 매핑
        port_idx = {v["name"]: i + 1 for i, v in enumerate(planner.iport_list)}

        # 0-1) resources_vp.csv 경로 해석
        def _resolve_resources_csv() -> Path | None:
            if self._resources_csv and self._resources_csv.exists():
                return self._resources_csv

            env_override = os.getenv("SCHEDULER_RESOURCES_VP")
            if env_override:
                env_path = Path(env_override)
                if env_path.exists():
                    return env_path

            scheduler_dir = Path(__file__).resolve().parent
            candidates = [
                scheduler_dir / "Sources" / "resources_vp.csv",
                scheduler_dir / "Sources" / "resource_vp.csv",
                ROOT_DIR / "Scheduler" / "Sources" / "resources_vp.csv",
                ROOT_DIR / "SITL" / "resource" / "resources_vp.csv",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return None

        res_csv = _resolve_resources_csv()
        if not res_csv:
            self.log("❌ resources_vp.csv 파일을 찾을 수 없습니다. (A/K 미삽입)")
            return
        df_res = load_resources(str(res_csv))

        # 1) 날짜 기반 폴더
        today_tag = dt.datetime.now().strftime("%Y%m%d")
        base_dir = self.save_root if self.save_root else Path(__file__).resolve().parents[1]
        root_dir = base_dir / "FPL_Result"
        root_dir.mkdir(parents=True, exist_ok=True)
        out_dir = root_dir / today_tag
        i = 1
        while out_dir.exists():
            out_dir = root_dir / f"{today_tag}_{i:02d}"
            i += 1
        out_dir.mkdir(parents=True, exist_ok=False)
        self.log(f"📁 Save folder: {out_dir}")

        # 2) 숫자 파싱("+범용") — "G03","T1","L2","1" 모두 허용
        def _num_any(s: str | None) -> int | None:
            if not s: return None
            m = re.search(r"\d+", str(s))
            return int(m.group()) if m else None

        resource_cache: dict[tuple[str, str, int], bool] = {}
        missing_resource_keys: set[str] = set()

        def _resource_has(port: str, kind: str, num: int | None) -> bool:
            if num is None:
                return False
            key = (port, kind.upper(), int(num))
            cached = resource_cache.get(key)
            if cached is not None:
                return cached
            lab = f"{key[1]} {key[2]}"
            mask = (df_res["Vertiport"] == port) & (df_res["Label"] == lab)
            exists = bool(mask.any())
            resource_cache[key] = exists
            return exists

        def _log_missing_resource(port: str, kind: str, num: int | None) -> None:
            if num is None:
                return
            msg_key = f"{port}:{kind.upper()} {int(num)}"
            if msg_key not in missing_resource_keys:
                missing_resource_keys.add(msg_key)
                self.log(f"[경고] resources_vp.csv에서 {msg_key} 정보가 없어 지상 경로 보정을 생략합니다.")

        # 3) 세그먼트 문자열 생성 (A/K 포함)
        def seg_strings(flt: dict) -> list[str]:
            origin = flt["origin"]; dest = flt["destination"]

            dist, prev = planner.dijkstra(origin, dest)
            if math.isinf(dist.get(dest, math.inf)):
                return []

            raw  = planner.reconstruct(prev, origin, dest)
            full = rebuild_route(planner, raw)

            # --- 게이트/패드 번호
            dep_gate_no = _num_any(flt.get("takeoff_gate"))
            dep_fato_no = _num_any(flt.get("takeoff_pad"))
            arr_fato_no = _num_any(flt.get("landing_pad"))
            arr_gate_no = _num_any(flt.get("landing_gate"))

            # --- 리소스 체크 (A/K 정확 좌표를 위해)
            numbers_missing = None in (dep_gate_no, dep_fato_no, arr_fato_no, arr_gate_no)
            required_pairs = [
                (origin, "GATE", dep_gate_no),
                (origin, "FATO", dep_fato_no),
                (dest,   "FATO", arr_fato_no),
                (dest,   "GATE", arr_gate_no),
            ]
            resources_available = all(_resource_has(p, k, n) for p, k, n in required_pairs)

            if numbers_missing or not resources_available:
                prof = path_to_profile(full, planner.nodes)
                if not resources_available:
                    for p, k, n in required_pairs:
                        if not _resource_has(p, k, n):
                            _log_missing_resource(p, k, n)
            else:
                prof0 = path_to_profile(full, planner.nodes)
                try:
                    prof  = inject_ground_and_fato(
                        prof0, planner, origin, dest,
                        dep_gate_no, dep_fato_no, arr_fato_no, arr_gate_no,
                        str(res_csv),
                        taxi_minutes=self._taxi_out_min
                    )
                except Exception as exc:
                    self.log(f"[경고] {origin}->{dest} 지상 경로 보정 실패: {exc}")
                    prof = prof0
                    resources_available = False

            def _safe_lonlat(port: str, kind: str, num: int | None):
                if not resources_available or not _resource_has(port, kind, num):
                    return None
                try:
                    return find_lonlat(df_res, port, kind, num)
                except Exception as exc:
                    self.log(f"[경고] {port} {kind} {num} 좌표 조회 실패: {exc}")
                    return None

            dep_fato_ll = _safe_lonlat(origin, "FATO", dep_fato_no)
            dep_gate_ll = _safe_lonlat(origin, "GATE", dep_gate_no)
            arr_fato_ll = _safe_lonlat(dest,   "FATO", arr_fato_no)
            arr_gate_ll = _safe_lonlat(dest,   "GATE", arr_gate_no)

            seg_out = []

            # 기준 점(위경도 변환용): km→deg 변환
            ref_lon, ref_lat = planner.nodes_geo[origin]
            ref_x_km, ref_y_km = planner.nodes[origin]

            for seg in prof.get_segments():
                # 1) 기본 변환 (km→deg)
                if getattr(seg, "start_point", None):
                    sx_km = seg.start_point["x"] / 1000.0
                    sy_km = seg.start_point["y"] / 1000.0
                    dlon_s, dlat_s = _km_to_dlon_dlat(sx_km - ref_x_km, sy_km - ref_y_km, ref_lat)
                    lon_sta = ref_lon + dlon_s
                    lat_sta = ref_lat + dlat_s
                    tmp_ref_lon, tmp_ref_lat = lon_sta, lat_sta
                    tmp_ref_xkm, tmp_ref_ykm = sx_km, sy_km
                else:
                    lon_sta, lat_sta = ref_lon, ref_lat
                    tmp_ref_lon, tmp_ref_lat = ref_lon, ref_lat
                    tmp_ref_xkm, tmp_ref_ykm = ref_x_km, ref_y_km

                ex_km = seg.end_point["x"] / 1000.0
                ey_km = seg.end_point["y"] / 1000.0
                dlon_e, dlat_e = _km_to_dlon_dlat(ex_km - tmp_ref_xkm, ey_km - tmp_ref_ykm, tmp_ref_lat)
                lon_end = tmp_ref_lon + dlon_e
                lat_end = tmp_ref_lat + dlat_e

                # 2) A/K 등 지상 세그먼트는 리소스 좌표로 덮어쓰기
                sid = seg.segment_id.upper()
                if   sid == "A" and dep_gate_ll and dep_fato_ll:
                    lon_sta, lat_sta = dep_gate_ll
                    lon_end, lat_end = dep_fato_ll
                elif sid == "B" and dep_fato_ll:
                    lon_sta, lat_sta = dep_fato_ll
                    lon_end, lat_end = dep_fato_ll
                elif sid == "C" and dep_fato_ll:
                    lon_sta, lat_sta = dep_fato_ll
                elif sid == "I" and arr_fato_ll:
                    lon_end, lat_end = arr_fato_ll
                elif sid == "J" and arr_fato_ll:
                    lon_sta, lat_sta = arr_fato_ll
                    lon_end, lat_end = arr_fato_ll
                elif sid == "K" and arr_fato_ll and arr_gate_ll:
                    lon_sta, lat_sta = arr_fato_ll
                    lon_end, lat_end = arr_gate_ll

                lane_tag = f" {seg.lane_type}" if sid == "F" and seg.lane_type else ""
                seg_out.append(
                    f"{sid} : {lon_sta:.6f} {lat_sta:.6f}{lane_tag}; "
                    f"{sid} : {lon_end:.6f} {lat_end:.6f}{lane_tag}"
                )

                # 3) 다음 세그먼트 기준 갱신
                ref_lon, ref_lat = lon_end, lat_end
                ref_x_km, ref_y_km = ex_km, ey_km

                ref_lon, ref_lat = lon_end, lat_end
                ref_x_km, ref_y_km = ex_km, ey_km

            return seg_out


        # 4) Vertiport별 CSV 작성 (★ FATO/GATE in/out 컬럼 추가)
        import pandas as pd
        def _fmt(t: dt.datetime | None) -> str:
            return t.strftime("%H:%M:%S") if isinstance(t, dt.datetime) else ""

        for vert, fls in self._flights.items():
            fls_sorted = sorted(fls, key=lambda f: f["scheduled_time"])
            rows = []; max_seg = 0

            for seq, f in enumerate(fls_sorted, 1):
                seg_list = seg_strings(f)
                max_seg = max(max_seg, len(seg_list))

                local_id = f"{port_idx.get(vert, 0)}-{seq}"

                # 번호 파싱(범용화)
                dep_gate_no = _num_any(f.get("takeoff_gate"))
                dep_fato_no = _num_any(f.get("takeoff_pad"))
                arr_fato_no = _num_any(f.get("landing_pad"))
                arr_gate_no = _num_any(f.get("landing_gate"))

                # ── ★ 실제 진입/진출 시각 계산 (fallback 포함)
                dep_fato_in  = f.get("dep_fato_in")  or f.get("actual_takeoff_start")
                dep_fato_out = f.get("dep_fato_out") or f.get("actual_takeoff_finish")

                # dep_gate_in/out: 통합배정이면 값 존재. 아니면 taxi_out_from_gate 기준으로 추정
                tg = f.get("taxi_out_from_gate")
                dep_gate_in  = f.get("dep_gate_in")  or (tg - dt.timedelta(minutes=NEW_DEP_PREOCCUPY_MIN) if tg else None)
                dep_gate_out = f.get("dep_gate_out") or tg

                arr_fato_in  = f.get("arr_fato_in")  or f.get("actual_touch")
                arr_fato_out = f.get("arr_fato_out") or f.get("actual_shutdown")
                arr_gate_in  = f.get("arr_gate_in")  or f.get("gate_in")
                arr_gate_out = f.get("arr_gate_out") or f.get("gate_out")

                base = {
                    "LocalID": local_id,
                    "ID":      f["uam_id"],
                    "Type":    f["aircraft_type"],
                    "Pax":     f["passengers"],
                    "From":    f["origin"],
                    "STD":     f["scheduled_time"].strftime("%H:%M:%S"),
                    "To":      f["destination"],
                    "STA":     f["actual_touch"].strftime("%H:%M:%S") if f.get("actual_touch") else "",
                    # 정수 번호 4개
                    "DepFATO_No": dep_fato_no,
                    "DepGate_No": dep_gate_no,
                    "ArrFATO_No": arr_fato_no,
                    "ArrGate_No": arr_gate_no,
                    # ── ★ 추가: in/out 타임스탬프 8개
                    "DepGateIn":  _fmt(dep_gate_in),
                    "DepGateOut": _fmt(dep_gate_out),
                    "DepFATOIn":  _fmt(dep_fato_in),
                    "DepFATOOut": _fmt(dep_fato_out),
                    "ArrFATOIn":  _fmt(arr_fato_in),
                    "ArrFATOOut": _fmt(arr_fato_out),
                    "ArrGateIn":  _fmt(arr_gate_in),
                    "ArrGateOut": _fmt(arr_gate_out),
                }
                seg_cols = {f"Seg{i+1}": seg_list[i] if i < len(seg_list) else ""
                            for i in range(max_seg)}
                rows.append({**base, **seg_cols})

            if not rows:
                continue

            cols = ["LocalID","ID","Type","Pax",
                    "From","DepFATO_No","DepGate_No","STD",
                    "To","ArrFATO_No","ArrGate_No","STA",
                    # ── ★ 새 컬럼들
                    "DepGateIn","DepGateOut","DepFATOIn","DepFATOOut",
                    "ArrFATOIn","ArrFATOOut","ArrGateIn","ArrGateOut"] \
                + [f"Seg{i+1}" for i in range(max_seg)]
            pd.DataFrame(rows, columns=cols).to_csv(
                out_dir / f"{vert}.csv",
                index=False,
                encoding="utf-8-sig"
            )

        self.log(f"✅ CSV 저장 완료 → {out_dir}")

    # ─────────────────────────────────────────────────────────────
    def _plot_uam_dist(self):
        # … (원문 그대로) …
        if not self._flights:
            self.log("⚠ 먼저 Generate를 완료하세요"); return
        import pandas as pd, matplotlib.pyplot as plt
        from collections import defaultdict

        cnt_by_uam = defaultdict(int)
        for fl in [f for lst in self._flights.values() for f in lst]:
            cnt_by_uam[fl["uam_id"]] += 1

        df = (pd.Series(cnt_by_uam)
                .value_counts()
                .sort_index()
                .rename_axis("NumFlights")
                .reset_index(name="NumUAM"))

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(df["NumFlights"], df["NumUAM"], marker="o", linewidth=2)
        ax.set_xlabel("운항 횟수 (편)")
        ax.set_ylabel("기체 수 (대)")
        ax.set_title("UAM 운항 횟수 분포")
        ax.set_xticks(df["NumFlights"])
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.show()
        plt.close()
        
    # ------------------------------------------------------------------
    def update_operation_info(self,
                            op_start: str,
                            op_end: str,
                            num_demand: int,
                            throughput: int) -> None:
        data = [op_start, op_end, f"{num_demand:,}", f"{throughput:,}"]
        for r, val in enumerate(data):
            self.tbl_ops.setItem(r, 1, QTableWidgetItem(val))

    def log(self, msg: str) -> None:
        self.txt_logs.append(msg)

    def set_flights(self, flights_by_origin: dict[str, list[dict]]) -> None:
        """Demand-based FPL 수신"""
        if not flights_by_origin:
            self.set_status("비행계획 없음")
            self.log("수요를 생성해주세요")
            return

        # ★★★ 초기화 + 사전 표시값(ETOT/ATOT/T-Pad 등) 제거 후 보관
        drop_keys = {
            "etot_plan","actual_takeoff_start","actual_takeoff_finish","takeoff_pad",
            "landing_pad","actual_touch","actual_shutdown","landing_ready","landing_ready_s",
            "std_delay_sec","delay_sec","t_wait_sec","_fixed",
            "takeoff_gate","landing_gate","gate_in","gate_out","gate_delay_sec",
            "dep_gate_in","dep_gate_out","dep_fato_in","dep_fato_out",
            "arr_fato_in","arr_fato_out","arr_gate_in","arr_gate_out",
            "taxi_out_from_gate"
        }
        cleaned: dict[str, list[dict]] = {}
        for k, lst in flights_by_origin.items():
            new_list = []
            for f in lst:
                g = f.copy()
                for dk in drop_keys:
                    g.pop(dk, None)
                new_list.append(g)
            cleaned[k] = new_list

        self._flights = cleaned
        self.populate_vertiports(sorted(cleaned.keys()))
        self.set_status("비행계획 생성 준비 완료")
        total = sum(len(v) for v in cleaned.values())
        self.log(f"{total} flights loaded")

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    # ─────────────────────────────────────────────────────────────

    def _assign_uam_ids(
        self,
        max_sorties: int = 50,      # 하루 한 기체 최대 운항 편수
        extra_wait_min: int = 5,    # STD 뒤에 최대 ‘N’분까지 승객이 기다려 줄 여유
        w_n:      float = 1.0,      # 운항 편수 벌점
        w_idle:   float = 0.1,      # (ready~DEP 사이) idle 보너스
        *,                         # ← 키워드 전용
        time_cut: dt.datetime      # ★ 이 시각 이후 편만 배정
    ) -> tuple[bool, str | None, str | None, dt.datetime | None, dt.datetime | None]:
        """
        UAM 배정(단일 패스):
        • _fixed=True 편은 제외
        • 원점(ori) 큐에서 ready ≤ STD 즉시 후보, STD < ready ≤ STD+Δt 지연 후보를 전수 스캔
        • 즉시 후보 없으면 지연 후보 중 가장 빠른 UAM, 그마저 없으면 신규 UAM 발급
        • (중요) 모든 배정편을 _fixed=True로 잠금
        • (중요) 큐는 사용한 항목을 제거하고, 남은 항목은 ready 기준으로 재정렬
        """

        # ── 준비 ──────────────────────────────────────────────────
        queues: dict[str, deque] = defaultdict(deque)   # {vert: deque[(ready,uam)]}
        usage:  dict[tuple[str, dt.date], int] = defaultdict(int)
        uam_counter = getattr(self, "_uam_counter", 1)

        delay_happened = False
        trigger_flt = trigger_uam = None
        t_old: dt.datetime | None = None
        t_new: dt.datetime | None = None

        # seed (이미 확정된 편의 도착→다음 출발 준비)
        queues.update({k: deque(v) for k, v in getattr(self, "_seed_queues", {}).items()})
        usage.update(getattr(self, "_seed_usage", {}))

        # 대상 flight (시간순·확정편 제외)
        flights_all = sorted(
            (f for lst in self._flights.values() for f in lst
            if not f.get("_fixed") and f["scheduled_time"] >= time_cut),
            key=lambda f: f["scheduled_time"]
        )

        if not flights_all:
            return False, None, None, None, None

        # ★★★ 통합 배정용 NetworkState 생성 (하루 공통) ─────────────────
        base0 = dt.datetime.combine(flights_all[0]["scheduled_time"].date(), dt.time(0,0,0))
        def m(t: dt.datetime) -> float:
            return (t - base0).total_seconds() / 60.0
        # 포트 집합 → FATO 수를 일괄 지정(RUNWAY_COUNT/NUM_ARR_RUNWAYS)
        ports = sorted({f["origin"] for f in flights_all} | {f["destination"] for f in flights_all})
        gate_by = {p: GATE_COUNT_DEFAULT for p in ports}
        tko_by  = {p: RUNWAY_COUNT for p in ports}
        ldg_by  = {p: NUM_ARR_RUNWAYS for p in ports}
        net = NetworkState(
            gate_count_by_port=gate_by,
            taxi_in_min=self._taxi_in_min,
            taxi_out_min=self._taxi_out_min,
            takeoff_min=TAKEOFF_MIN,
            landing_min=LANDING_MIN,
            prep_time_min=self._gate_service_min,
            tko_count_by_port=tko_by,
            ldg_count_by_port=ldg_by,
            locked_gates_by_port={"*": LOCKED_GATES_ALL_PORTS}
        )

        for f in flights_all:
            ori, dest = f["origin"], f["destination"]
            std_orig  = f["scheduled_time"]
            today     = std_orig.date()

            q_list = list(queues[ori])  # 정렬 보장 없음 → 전수 스캔

            # (a) 즉시 후보: ready ≤ STD
            imm_all = [(r,u) for (r,u) in q_list
                    if r <= std_orig and usage[(u, today)] < max_sorties]

            # (b) 지연 후보: STD < ready ≤ STD+Δt  (Δt=0이면 비어야 함)
            latest = std_orig + dt.timedelta(minutes=extra_wait_min)
            if extra_wait_min > 0:
                delay_all = [(r,u) for (r,u) in q_list
                            if std_orig < r <= latest and usage[(u, today)] < max_sorties]
            else:
                delay_all = []

            best_uam   = None
            best_ready = None
            dep_final  = std_orig
            std_delay  = 0
            source     = "new"   # imm | delay | new

            # 즉시 후보가 있으면 비용식으로 선택
            if imm_all:
                best_cost = float("inf")
                for ready, uam in imm_all:
                    dep     = std_orig
                    idle_s  = max(0, (dep - ready).total_seconds())
                    n_after = usage[(uam, today)] + 1
                    cost    = w_n * n_after - w_idle * (idle_s**0.5)
                    if cost < best_cost:
                        best_cost, best_uam, best_ready = cost, uam, ready
                dep_final = std_orig
                std_delay = 0
                source    = "imm"

            # 즉시 후보가 없으면 지연 후보 중 가장 빠른 것
            elif delay_all:
                delay_all.sort(key=lambda x: x[0])   # ready 이른 순
                best_ready, best_uam = delay_all[0]
                dep_final = best_ready
                std_delay = int((dep_final - std_orig).total_seconds())
                source    = "delay"
                delay_happened = True
                trigger_flt = f["flight_number"]
                trigger_uam = best_uam
                t_old, t_new = std_orig, dep_final

            # 둘 다 없으면 신규 기체 부여
            else:
                best_uam = f"UAM{uam_counter:04d}"
                uam_counter += 1
                dep_final = std_orig
                std_delay = 0
                source    = "new"

            # ★ 동일 Origin에서 동일 STD(초 단위까지)가 중복될 경우 1초 오프셋
            last_std = self._last_std_by_origin.get(ori)
            if last_std and abs((dep_final - last_std).total_seconds()) < 1:
                dep_final = last_std + dt.timedelta(seconds=1)
            self._last_std_by_origin[ori] = dep_final

            # ── 큐 갱신: 사용한 항목 제거 + 남은 항목 정렬 ───────────────
            if source == "imm":
                # imm 중 선택한 1개를 제외하고 모두 되돌림, >STD 항목도 포함
                others = [(r,u) for (r,u) in imm_all if not (r == best_ready and u == best_uam)]
                later  = [(r,u) for (r,u) in q_list if r > std_orig]
                queues[ori] = deque(sorted(others + later, key=lambda x: x[0]))
            elif source == "delay":
                rest = []
                used = False
                for r,u in q_list:
                    if not used and r == best_ready and u == best_uam:
                        used = True
                        continue
                    rest.append((r,u))
                queues[ori] = deque(sorted(rest, key=lambda x: x[0]))
            else:
                # 신규 기체 사용 → ori 큐 변화 없음
                queues[ori] = deque(sorted(q_list, key=lambda x: x[0]))

            # ── flight 확정/잠금 ───────────────────────────────────────
            f["uam_id"]        = best_uam
            f["scheduled_time"]= dep_final
            f["std_delay_sec"] = std_delay
            f["_fixed"]        = True

            usage[(best_uam, today)] += 1

            # =========================================================
            # ★★★ 여기서 자원 엔진 호출 → 실제 시간 확정(통합 배정)
            #  출발(HOLD) → 이륙 FATO → 항로 → 착륙 FATO → Taxi-in → GATE 6분
            # =========================================================
            # 출발
            std_min = m(dep_final)
            gate_pref = self._uam_gate_pref.get(best_uam, {}).get(ori)
            r_dep = net.departure_flow(ori, etot=std_min + self._taxi_out_min,
                                    flight_id=best_uam, std_min=std_min,
                                    departure_policy="HOLD",
                                    preferred_gate=gate_pref)
            if gate_pref is not None:
                self._uam_gate_pref[best_uam].pop(ori, None)

            # 1) TKO → 실제 ATOT 먼저 기록
            f["takeoff_gate"] = f"G{r_dep['gate_id']+1:02d}"
            f["takeoff_pad"]  = str(r_dep['fato_tko_id'] + 1)
            f["actual_takeoff_start"]  = base0 + dt.timedelta(minutes=r_dep["fato_tko_start"])
            f["actual_takeoff_finish"] = base0 + dt.timedelta(minutes=r_dep["fato_tko_end"])
            f["etot_plan"] = dep_final + dt.timedelta(minutes=TAKEOFF_MIN + self._taxi_out_min)

            # 2) (출발) FATO/GATE 실제 진입·진출
            f["dep_fato_in"]  = f["actual_takeoff_start"]
            f["dep_fato_out"] = f["actual_takeoff_finish"]
            f["dep_gate_in"]  = base0 + dt.timedelta(minutes=r_dep["taxi_out_start"] - NEW_DEP_PREOCCUPY_MIN)
            f["dep_gate_out"] = base0 + dt.timedelta(minutes=r_dep["taxi_out_start"])

            # 항로 → 착륙 준비
            trip = f.get("trip_time", dt.timedelta())
            f["landing_ready"] = f["actual_takeoff_finish"] + trip
            touch_min = m(f["landing_ready"])

            # 착륙 ~ GATE
            r_arr = net.arrival_flow(dest, touchdown_time=touch_min, flight_id=best_uam)
            self._uam_gate_pref[best_uam][dest] = r_arr["gate_id"]

            # 1) 실제 착륙/셔트다운·게이트 in/out
            f["landing_pad"]     = str(RUNWAY_COUNT + r_arr['fato_ldg_id'] + 1)
            f["actual_touch"]    = base0 + dt.timedelta(minutes=r_arr["fato_ldg_start"])
            f["actual_shutdown"] = base0 + dt.timedelta(minutes=r_arr["fato_ldg_end"])
            f["landing_gate"]    = f"G{r_arr['gate_id']+1:02d}"
            f["gate_in"]         = base0 + dt.timedelta(minutes=r_arr["gate_start"])
            f["gate_out"]        = base0 + dt.timedelta(minutes=r_arr["gate_end"])
            baseline = f["actual_touch"] + dt.timedelta(minutes=(LANDING_MIN + self._taxi_in_min))
            f["gate_delay_sec"]  = max(0, int((f["gate_in"] - baseline).total_seconds()))

            # 2) (도착) FATO/GATE 실제 진입·진출
            f["arr_fato_in"]  = f["actual_touch"]
            f["arr_fato_out"] = f["actual_shutdown"]
            f["arr_gate_in"]  = f["gate_in"]
            f["arr_gate_out"] = f["gate_out"]

            # 도착지 큐(실제 Ready=gate_out) 등록
            ready_real = f["gate_out"]
            self._uam_hold_until[best_uam] = ready_real
            queues[dest].append((ready_real, best_uam))

        # 카운터 보존 + 네트워크 보관(게이트 타임라인 표시용)
        self._uam_counter = uam_counter
        self._gate_net = net
        self._gate_base0 = base0
        self._integrated_scheduled = True
        return delay_happened, trigger_flt, trigger_uam, t_old, t_new





    def set_operation_info(self, start: str, end: str,
                           demand: int, throughput: int) -> None:
        vals = [start, end, f"{demand:06d}", f"{throughput:05d}"]
        for r, v in enumerate(vals):
            self.tbl_ops.setItem(r, 1, QTableWidgetItem(v))

    def populate_vertiports(self, names: list[str]) -> None:
        self.combo_vert.clear()
        self.combo_vert.addItems(names)

    def update_flight_table(self, rows: list[tuple]) -> None:
        """rows – iterable of row tuples"""
        self.tbl_flights.setRowCount(max(8, len(rows)))
        for r, data in enumerate(rows):
            for c, val in enumerate(data):
                item = QTableWidgetItem(str(val))
                self.tbl_flights.setItem(r, c, item)

    def _on_vert_select(self, vert_name: str):
        fls = sorted(self._flights.get(vert_name, []),
                 key=lambda f: f["scheduled_time"])

        rows = []
        for f in fls:
            # ② 착륙 시각 표기(키 수정)  ELDT=landing_ready, ALDT=actual_touch
            eldt = f.get("landing_ready")
            aldt = f.get("actual_touch")
            eldt_str = eldt.strftime("%H:%M:%S") if eldt else ""
            aldt_str = aldt.strftime("%H:%M:%S") if aldt else ""

            rows.append((
                f["flight_number"],                            # callSign
                f.get("uam_id", ""),                           # regNum
                f["aircraft_type"],                            # Type
                f["passengers"],                               # Pax
                f["origin"],                                   # From
                f["scheduled_time"].strftime("%H:%M:%S"),      # STD
                f.get("etot_plan","").strftime("%H:%M:%S")     # ETOT
                if f.get("etot_plan") else "",
                f.get("actual_takeoff_finish","").strftime("%H:%M:%S")  # ATOT
                if f.get("actual_takeoff_finish") else "",
                f.get("takeoff_gate",""),                      # T-Gate
                f.get("takeoff_pad",""),                       # T-Pad
                f["destination"],                              # To
                eldt_str,                                      # ELDT
                aldt_str,                                      # ALDT
                f.get("landing_gate",""),                      # L-Gate
                f.get("landing_pad",""),                       # L-Pad
            ))
        self.update_flight_table(rows)
        self._render_gate_timeline(vert_name)

    def _visualize_route(self, flight: dict):
        o, d = flight["origin"], flight["destination"]
        try:
            dist, prev = self._planner.dijkstra(o, d)
            if math.isinf(dist.get(d, math.inf)):
                self.log(f"❌ 경로 없음: {o}→{d}"); return
            raw  = self._planner.reconstruct(prev, o, d)
            full = rebuild_route(self._planner, raw)

            lonlat = []
            ref_lon, ref_lat = None, None
            prev_x,  prev_y  = None, None

            for p in full:
                if isinstance(p, str):
                    lon, lat = self._planner.nodes_geo[p]
                    ref_lon, ref_lat = lon, lat
                    prev_x,  prev_y  = self._planner.nodes[p]
                else:  # (x_km, y_km)
                    x_km, y_km = p
                    dlon, dlat = _km_to_dlon_dlat(x_km - prev_x,
                                                   y_km - prev_y, ref_lat)
                    lon = ref_lon + dlon; lat = ref_lat + dlat
                    ref_lon, ref_lat = lon, lat
                    prev_x,  prev_y  = x_km, y_km
                lonlat.append((lon, lat))

            self.map_view.draw_route(lonlat)
            self.set_status(f"Route: {o}→{d}  {dist[d]:.1f} km")
        except Exception as e:
            self.log(f"경로 시각화 실패: {e}")

    # ----------------------------------------------------------------
    def _generate(self):
        """
        Demand-based FPL 생성 루틴  — 재계산 루프 비활성화(단일 패스)
        1) 경로/소요시간(ETA)/거리 계산 (필요 시)
        2) UAM 배정 1회 수행 (extra_wait_min 반영)
        3) 지상/이륙 시뮬레이션(ETOT/ATOT) 최종 1회
        4) 착륙 시뮬레이션(ELDT/ALDT, delay_sec) 최종 1회
        5) Gate 배정(도착/출발) — T-Gate/L-Gate 표출
        6) Turn-around(T-wait) 계산 (게이트 배정 반영)
        """

        # ── (기존 유지) seed_queues/seed_usage: 이미 확정된(_fixed) 편의 UAM 점유 이월 ──
        seed_queues = defaultdict(deque)          # {dest → deque[(ready,uam)]}
        seed_usage  = defaultdict(int)            # {(uam,date) → sorties}
        for fls in self._flights.values():
            for f in fls:
                if not f.get("_fixed"):
                    continue
                touch = f["actual_touch"]
                # READY = (실제 접지 이후) 착륙패드 점유 + FATO→GATE + GATE서비스
                shutdown = f.get("actual_shutdown", touch + dt.timedelta(minutes=LANDING_MIN))
                ready = shutdown + dt.timedelta(minutes=(self._taxi_in_min + self._gate_service_min))
                seed_queues[f["destination"]].append((ready, f["uam_id"]))
                seed_usage[(f["uam_id"], touch.date())] += 1

        self._seed_queues = seed_queues
        self._seed_usage  = seed_usage

        # ── 사전 체크 ───────────────────────────────────────────
        if not self._flights:
            self.log("⚠ 수요를 먼저 불러오세요")
            return
        planner = self.map_view._planner
        if planner is None:
            self.log("⚠ 네트워크(플래너)를 먼저 로드하세요")
            return

        # 상태
        self._uam_hold_until = defaultdict(lambda: dt.datetime.min)
        self.set_status("시뮬레이션 중… (loop OFF)")
        QApplication.processEvents()

        # ==================================================
        # 1) Trip-time Δt + 거리(km)  (필요한 항목만 갱신)
        # ==================================================
        total = sum(len(v) for v in self._flights.values())
        done  = 0
        self.pbar.setValue(0)

        for origin, fls in self._flights.items():
            for f in fls:
                if f.get("trip_time") and f.get("dist_km") is not None:
                    done += 1
                else:
                    try:
                        delta = compute_eta(f["origin"], f["destination"],
                                            planner=planner)
                        f["trip_time"] = delta
                        dist, _ = planner.dijkstra(f["origin"], f["destination"])
                        f["dist_km"] = dist[f["destination"]]
                    except Exception as e:
                        self.log(f"ETA 실패 {f['flight_number']}: {e}")
                        f["trip_time"] = None
                        f["dist_km"]   = None
                    finally:
                        done += 1
                self.pbar.setValue(int(done/total*100))
                QApplication.processEvents()

        # ==================================================
        # 2) UAM 배정 (통합 배정으로 실제 시간까지 확정)
        # ==================================================
        max_sorties    = self.spin_maxsorties.value()
        extra_wait_min = self.spin_extrawait.value()
        _delay_flag, _trig_flt, _trig_uam, _t_old, _t_new = self._assign_uam_ids(
            max_sorties, extra_wait_min, time_cut=dt.datetime.min
        )

        # ==================================================
        # 3)~5) 사후 시뮬/게이트: 통합 배정이 끝났으면 생략
        # ==================================================
        if not self._integrated_scheduled:
            for vert, fls in self._flights.items():
                simulate_ground_operations(
                    fls,
                    num_runways=RUNWAY_COUNT
                )
            for f in (fl for lst in self._flights.values() for fl in lst):
                if f.get("trip_time") and f.get("actual_takeoff_finish"):
                    f["landing_ready"] = f["actual_takeoff_finish"] + f["trip_time"]

            dest_map = defaultdict(list)
            for fls in self._flights.values():
                for f in fls:
                    if f.get("landing_ready"):                 # 방어적 가드
                        dest_map[f["destination"]].append(f)

            for flist in dest_map.values():
                simulate_landing_ops(flist, num_runways=NUM_ARR_RUNWAYS)
                for f in flist:
                    f["delay_sec"] = max(
                        0,
                        int((f["actual_touch"] - f["landing_ready"]).total_seconds())
                    )

            def m(t: dt.datetime) -> float:
                base0 = dt.datetime.combine(t.date(), dt.time(0,0,0))
                return (t - base0).total_seconds() / 60.0

            all_fl = [f for lst in self._flights.values() for f in lst]
            if not all_fl:
                self.set_status("완료 (No flights)"); return

            base0 = dt.datetime.combine(all_fl[0]["scheduled_time"].date(), dt.time(0,0,0))

            net = NetworkState(
                takeoff_min=TAKEOFF_MIN,
                landing_min=LANDING_MIN,
                taxi_in_min=self._taxi_in_min,
                taxi_out_min=self._taxi_out_min,
                prep_time_min=self._gate_service_min,
                locked_gates_by_port={"*": LOCKED_GATES_ALL_PORTS}
            )

            events = []
            for f in all_fl:
                if f.get("actual_touch"):
                    events.append(("ARR", f["destination"], m(f["actual_touch"]), f))
                if f.get("actual_takeoff_finish"):   # ★ TKO(패드 입장) 기준으로 이벤트 생성
                    events.append(("DEP", f["origin"], m(f["actual_takeoff_finish"]), f))
            events.sort(key=lambda x: x[2])

            for kind, port, tmin, f in events:
                try:
                    if kind == "ARR":
                        r = net.arrival_flow(port, tmin, flight_id=f.get("uam_id"))
                        gate_no = r["gate_id"] + 1
                        f["landing_gate"] = f"G{gate_no:02d}"
                        f["gate_in"]  = base0 + dt.timedelta(minutes=r["gate_start"])
                        f["gate_out"] = base0 + dt.timedelta(minutes=r["gate_end"])
                        baseline = f["actual_touch"] + dt.timedelta(minutes=(LANDING_MIN + self._taxi_in_min))
                        f["gate_delay_sec"] = max(0, int((f["gate_in"] - baseline).total_seconds()))
                        # 도착 세그 진입/이탈도 저장
                        f["arr_fato_in"]  = base0 + dt.timedelta(minutes=r["fato_ldg_start"])
                        f["arr_fato_out"] = base0 + dt.timedelta(minutes=r["fato_ldg_end"])
                        f["arr_gate_in"]  = f["gate_in"]
                        f["arr_gate_out"] = f["gate_out"]
                        if f.get("uam_id"):
                            self._uam_gate_pref[f["uam_id"]][port] = r["gate_id"]
                    else:  # DEP
                        std_min = (f["scheduled_time"] - base0).total_seconds()/60.0
                        gate_pref = None
                        if f.get("uam_id"):
                            gate_pref = self._uam_gate_pref.get(f["uam_id"], {}).get(port)
                        r = net.departure_flow(port, tmin, flight_id=f.get("uam_id"),
                                               std_min=std_min, departure_policy="HOLD",
                                               preferred_gate=gate_pref)
                        if gate_pref is not None and f.get("uam_id"):
                            self._uam_gate_pref[f["uam_id"]].pop(port, None)
                        gate_no = r["gate_id"] + 1
                        f["takeoff_gate"] = f"G{gate_no:02d}"
                        # taxi_out_from_gate 기준으로 출발 게이트 in/out 추정 저장
                        f["taxi_out_from_gate"] = base0 + dt.timedelta(minutes=r["taxi_out_start"])
                        f["dep_gate_in"]  = f["taxi_out_from_gate"] - dt.timedelta(minutes=NEW_DEP_PREOCCUPY_MIN)
                        f["dep_gate_out"] = f["taxi_out_from_gate"]
                        f["dep_fato_in"]  = base0 + dt.timedelta(minutes=r["fato_tko_start"])
                        f["dep_fato_out"] = base0 + dt.timedelta(minutes=r["fato_tko_end"])
                except Exception as e:
                    self.log(f"Gate assign 실패 {f.get('flight_number','?')}@{port}/{kind}: {e}")

            self._gate_net = net
            self._gate_base0 = base0

        # ==================================================
        # 6) Turn-around(T-wait) 계산  (게이트 배정 반영)
        # ==================================================
        flights_by_uam = defaultdict(list)
        for f in (fl for lst in self._flights.values() for fl in lst):
            flights_by_uam[f["uam_id"]].append(f)

        for flist in flights_by_uam.values():
            flist.sort(key=lambda x: x["scheduled_time"])
            for i, f in enumerate(flist):
                touch = f["actual_touch"]
                shutdown = f.get("actual_shutdown", touch + dt.timedelta(minutes=LANDING_MIN))
                ready_gate = shutdown + dt.timedelta(minutes=(self._taxi_in_min + self._gate_service_min))
                if i + 1 < len(flist):
                    next_dep = flist[i + 1]["scheduled_time"]
                    f["t_wait_sec"] = max(0, int((next_dep - ready_gate).total_seconds()))
                else:
                    f["t_wait_sec"] = None

        # ==================================================
        # 7) UI 갱신 · 통계 로그 · 분포 차트
        # ==================================================
        self._on_vert_select(self.combo_vert.currentText())
        # dest_map: Arrivals 탭용 (landing_ready가 없을 일은 통합 배정에서도 없음)
        dest_map = defaultdict(list)
        for fls in self._flights.values():
            for f in fls:
                if f.get("landing_ready"):
                    dest_map[f["destination"]].append(f)
        self.generation_done.emit(dest_map)
        self.set_status("완료 (loop OFF)")

        flights = [f for lst in self._flights.values() for f in lst]
        sorties = len(flights)

        cnt_by_uam = {}
        for f in flights:
            cnt_by_uam[f["uam_id"]] = cnt_by_uam.get(f["uam_id"], 0) + 1
        num_uam     = len(cnt_by_uam)
        max_per_uam = max(cnt_by_uam.values()) if cnt_by_uam else 0
        one_off_uam = sum(1 for v in cnt_by_uam.values() if v == 1)

        dists   = [f["dist_km"] for f in flights if f.get("dist_km")]
        avg_leg = sum(dists) / len(dists) if dists else 0.0

        dist_by_uam = {}
        for f in flights:
            if not f.get("dist_km"): continue
            uid = f["uam_id"]
            dist_by_uam[uid] = dist_by_uam.get(uid, 0) + f["dist_km"]
        avg_uam = sum(dist_by_uam.values()) / len(dist_by_uam) if dist_by_uam else 0.0

        self.log("\n정밀 비행계획 생성 완료")
        self.log(f"1. 시간내 운항 총 량   : {sorties:,} sorties")
        self.log(f"2. 사용한 기체 수      : {num_uam:,} 대")
        self.log(f"3. 최대 운항 수        : {max_per_uam} 회 운항")
        self.log(f"4. 1회 운항 기체 수    : {one_off_uam} 대")
        self.log(f"5. 편당 평균 거리       : {avg_leg:6.1f} km")
        self.log(f"6. 기체별 총거리 평균  : {avg_uam:6.1f} km\n")

        self._plot_uam_dist()


    # ───────────────────────────────────────────────────────────────
    # Gate Timeline Renderer
    # ───────────────────────────────────────────────────────────────
    def _render_gate_timeline(self, vert_name: str) -> None:
        ax = self.gate_ax
        ax.clear()
        if not self._gate_net or not vert_name:
            self.gate_canvas.draw(); return

        rows = self._gate_net.get_gate_schedule(vert_name)
        if not rows:
            ax.set_title(f"{vert_name} – no gate usage")
            self.gate_canvas.draw(); return

        base0 = self._gate_base0 or dt.datetime.combine(dt.date.today(), dt.time(0,0,0))
        gates = sorted({r["gate"] for r in rows})
        ymap  = {g: i for i, g in enumerate(gates)}

        xs = []; xe = []
        for r in rows:
            s = base0 + dt.timedelta(minutes=r["start"])
            e = base0 + dt.timedelta(minutes=r["end"])
            left  = mdates.date2num(s)
            width = (e - s).total_seconds() / 86400.0
            ax.barh(ymap[r["gate"]], width, left=left, height=0.6, edgecolor="k")
            if r.get("flight_id"):
                ax.text(left + width/2, ymap[r["gate"]], str(r["flight_id"]), ha="center", va="center", fontsize=7)
            xs.append(left); xe.append(left + width)

        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels([f"G{g:02d}" for g in gates])
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelrotation=45)
        if xs and xe:
            ax.set_xlim(min(xs) - 1/24, max(xe) + 1/24)
        ax.set_title(f"{vert_name} – Gate occupancy")
        self.gate_canvas.draw()

    # ───────────────────────────────────────────────────────────────
    # Gate Timeline 별도 창
    # ───────────────────────────────────────────────────────────────
    def _open_gate_window(self):
        if not self._gate_net or not self._gate_base0:
            self.log("⚠ Gate timeline is not ready yet."); return
        ports = sorted(self._gate_net.ports.keys())
        if not ports:
            self.log("⚠ No gate schedule to show."); return
        # ★ 부모 없는 Top-level 창으로 생성 + 창 버튼(- ㅁ ×) 활성화
        #    + 모든 편(전체 리스트)을 전달해 클릭 시 원/도착 게이트 조회 가능
        flights_flat = [f for lst in self._flights.values() for f in lst]
        dlg = GateTimelineWindow(self._gate_net, self._gate_base0, ports, None, flights=flights_flat)
        dlg.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        cur = self.combo_vert.currentText()
        if cur in ports:
            dlg.cbo_port.setCurrentText(cur)
        self._gate_window = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()



# ───────────────────────────────────────────────────────────────
# Gate Timeline Window
# ───────────────────────────────────────────────────────────────
class GateTimelineWindow(QWidget):
    def __init__(self, net: NetworkState, base0: dt.datetime, ports: list[str], parent=None, flights=None):
        # ★ 부모를 None으로 고정 → 완전한 OS 상단 창 생성
        super().__init__(None)
        self.setWindowTitle("Gate Timeline")
        self.resize(1200, 600)
        self._net = net
        self._base0 = base0
        self._fl_all = flights or []
        self._by_uam = defaultdict(list)
        for f in self._fl_all:
            uid = f.get("uam_id")
            if uid:
                self._by_uam[uid].append(f)
        for uid in self._by_uam:
            self._by_uam[uid].sort(key=lambda x: x.get("scheduled_time", dt.datetime.min))

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Port:"))
        self.cbo_port = QComboBox(); self.cbo_port.addItems(ports)
        self.cbo_port.currentTextChanged.connect(self._render)
        top.addWidget(self.cbo_port, 1)

        top.addWidget(QLabel("Mode:"))
        self.cbo_mode = QComboBox(); self.cbo_mode.addItems(["Gantt","Heatmap"])
        self.cbo_mode.currentTextChanged.connect(self._render)
        top.addWidget(self.cbo_mode)

        top.addWidget(QLabel("From:"))
        self.t_from = QTimeEdit(); self.t_from.setDisplayFormat("HH:mm")
        top.addWidget(self.t_from)

        top.addWidget(QLabel("To:"))
        self.t_to = QTimeEdit(); self.t_to.setDisplayFormat("HH:mm")
        top.addWidget(self.t_to)

        top.addWidget(QLabel("Bin(min):"))
        self.spin_bin = QSpinBox(); self.spin_bin.setRange(1, 60); self.spin_bin.setValue(5)
        self.spin_bin.valueChanged.connect(self._render)
        top.addWidget(self.spin_bin)

        self.chk_label = QCheckBox("Labels"); self.chk_label.setChecked(False)
        self.chk_label.stateChanged.connect(self._render)
        top.addWidget(self.chk_label)

        lay.addLayout(top)

        self.canvas = FigureCanvas(Figure(figsize=(10,4)))
        self.ax = self.canvas.figure.add_subplot(111)
        lay.addWidget(self.canvas, 1)

        self.lab_info = QLabel()
        lay.addWidget(self.lab_info)

        # 클릭 이벤트 연결(간트 전용)
        self._rects_info = []
        self.canvas.mpl_connect("button_press_event", self._on_click)

        # 초기 시간 범위를 rows로부터 설정
        rows0 = self._rows(self.cbo_port.currentText())
        if rows0:
            s0 = min(r["start"] for r in rows0); e0 = max(r["end"] for r in rows0)
            t0 = (self._base0 + dt.timedelta(minutes=s0)).time().replace(second=0, microsecond=0)
            t1_dt = self._base0 + dt.timedelta(minutes=min(e0, s0 + 6*60))  # 기본 6시간 뷰
            t1 = t1_dt.time().replace(second=0, microsecond=0)
            self.t_from.setTime(QTime(t0.hour, t0.minute))
            self.t_to.setTime(QTime(t1.hour, t1.minute))
        else:
            self.t_from.setTime(QTime(6,0)); self.t_to.setTime(QTime(12,0))

        self._render(self.cbo_port.currentText())

    def _rows(self, port_name: str):
        """GateResources.get_gate_schedule 시그니처가 (port) 또는 (port, labels=…)일 수 있음."""
        try:
            # 라벨 분리 버전이면 PRE/ARR/LOCK 모두 요청
            return self._net.get_gate_schedule(port_name, labels=("GATE_DEP_PRE","GATE_ARR_SVC","GATE_LOCK"))
        except TypeError:
            # 구버전(라벨 미지원) 호환
            return self._net.get_gate_schedule(port_name)

    def _minutes_window(self):
        t0 = self.t_from.time(); t1 = self.t_to.time()
        m0 = t0.hour()*60 + t0.minute()
        m1 = t1.hour()*60 + t1.minute()
        if m1 <= m0: m1 = m0 + 60  # 최소 1시간
        return m0, m1

    def _render(self, _):
        self.ax.clear()
        self._rects_info = []
        port_name = self.cbo_port.currentText()
        rows = self._rows(port_name)
        if not rows:
            self.ax.set_title(f"{port_name} – no gate usage")
            self.canvas.draw(); return

        m0, m1 = self._minutes_window()

        # 선택 구간으로 필터/클립
        rows_clip = []
        for r in rows:
            s, e = r["start"], r["end"]
            if e <= m0 or s >= m1: 
                continue
            rows_clip.append({"gate":r["gate"], "start":max(s,m0), "end":min(e,m1), "flight_id":r.get("flight_id","")})

        if self.cbo_mode.currentText() == "Heatmap":
            self._render_heatmap(rows_clip, port_name, m0, m1)
        else:
            self._render_gantt(rows_clip, port_name, m0, m1)

        self._update_footer_stats(rows_clip)

    def _update_footer_stats(self, rows):
        """
        상태바(하단 텍스트)에 PRE/LOCK/ARR 카운트와 Overlap/중앙값을 표기.
        rows: {'gate','start','end','flight_id', ['label']} 리스트
        """
        import numpy as np
        # 길이(초)
        durs = [int(round((r['end'] - r['start']) * 60)) for r in rows]
        n = len(rows)
        med = int(np.median(durs)) if durs else 0

        # 기준값(현재 설정 반영)
        pre_sec  = int(NEW_DEP_PREOCCUPY_MIN * 60)                # 사전점유(분→초)  ex) 60
        lock_sec = int(GATE_EXIT_LINGER_SEC)                      # 릴린저(초)       ex) 10
        arr_sec  = int(getattr(self._net, 'prep_time_min', 6.0) * 60)  # 도착 서비스 ex) 360
        tol = 3  # 허용 오차(초)

        # 라벨이 있으면 라벨 기준, 없으면 길이로 추정
        lab  = lambda r: r.get('label', '')
        dur  = lambda r: int(round((r['end'] - r['start']) * 60))
        is_pre  = lambda r: (lab(r) == 'GATE_DEP_PRE') or (abs(dur(r) - pre_sec)  <= tol)
        is_lock = lambda r: (lab(r) == 'GATE_LOCK')    or (abs(dur(r) - lock_sec) <= tol)
        is_arr  = lambda r: (lab(r) == 'GATE_ARR_SVC') or (abs(dur(r) - arr_sec)  <= tol)

        c_pre  = sum(1 for r in rows if is_pre(r))
        c_lock = sum(1 for r in rows if is_lock(r))
        c_arr  = sum(1 for r in rows if is_arr(r))

        # 게이트별 Overlap(정상은 0)
        overlaps = 0
        by_gate = {}
        for r in rows:
            by_gate.setdefault(r['gate'], []).append((r['start'], r['end']))
        for _, segs in by_gate.items():
            segs.sort()
            end_prev = -1e9
            for s, e in segs:
                if s < end_prev - 1e-9:
                    overlaps += 1
                end_prev = max(end_prev, e)

        self.lab_info.setText(
            f"Intervals: {n}  |  Overlaps: {overlaps}  |  "
            f"PRE({pre_sec}s): {c_pre}  |  LOCK({lock_sec}s): {c_lock}  |  "
            f"ARR({arr_sec//60}m): {c_arr}  |  median: {med}s"
        )


    def _render_gantt(self, rows, port_name, m0, m1):
        gates = sorted({r["gate"] for r in rows})
        ymap = {g:i for i,g in enumerate(gates)}
        xs = []; xe = []
        for r in rows:
            s = self._base0 + dt.timedelta(minutes=r["start"])
            e = self._base0 + dt.timedelta(minutes=r["end"])
            left  = mdates.date2num(s)
            width = (e - s).total_seconds() / 86400.0
            bar = self.ax.barh(ymap[r["gate"]], width, left=left, height=0.6, edgecolor="k")
            patch = bar.patches[0] if hasattr(bar, "patches") and bar.patches else None
            if self.chk_label.isChecked() and r.get("flight_id"):
                self.ax.text(left + width/2, ymap[r["gate"]], str(r["flight_id"]), ha="center", va="center", fontsize=8)
            xs.append(left); xe.append(left + width)
            # 클릭 매핑용 정보 저장
            if patch is not None:
                self._rects_info.append({
                    "patch": patch,
                    "port": port_name,
                    "gate": r["gate"],
                    "start": r["start"],
                    "end": r["end"],
                    "uam": r.get("flight_id","")
                })

        self.ax.set_yticks(list(ymap.values()))
        self.ax.set_yticklabels([f"G{g:02d}" for g in gates])
        self.ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=list(range(0,60,30))))
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.tick_params(axis="x", labelrotation=45)
        if xs and xe:
            # xlim은 선택 범위에 맞춤
            base_day = self._base0.replace(hour=0, minute=0, second=0, microsecond=0)
            t0 = base_day + dt.timedelta(minutes=m0); t1 = base_day + dt.timedelta(minutes=m1)
            self.ax.set_xlim(mdates.date2num(t0), mdates.date2num(t1))
        self.ax.set_title(f"{port_name} – Gate occupancy (Gantt)")
        self.canvas.draw()

    def _render_heatmap(self, rows, port_name, m0, m1):
        gates = sorted({r["gate"] for r in rows})
        bin_min = max(1, self.spin_bin.value())
        nbin = math.ceil((m1 - m0)/bin_min)
        mat = np.zeros((len(gates), nbin), dtype=float)
        gmap = {g:i for i,g in enumerate(gates)}

        # 각 bin별 점유율(0~1) 누적
        for r in rows:
            s, e = r["start"], r["end"]
            gs = gmap[r["gate"]]
            b0 = int((s - m0)//bin_min); b1 = int(math.ceil((e - m0)/bin_min))
            for b in range(max(0,b0), min(nbin,b1)):
                bin_s = m0 + b*bin_min
                bin_e = bin_s + bin_min
                occ = max(0.0, min(e, bin_e) - max(s, bin_s))
                mat[gs, b] += occ / bin_min  # 비율

        extent=(m0, m1, -0.5, len(gates)-0.5)
        im = self.ax.imshow(mat, aspect="auto", origin="lower", extent=extent, vmin=0, vmax=1)
        self.ax.set_yticks(range(len(gates)))
        self.ax.set_yticklabels([f"G{g:02d}" for g in gates])
        # x축 눈금(시:분)
        base_day = self._base0.replace(hour=0, minute=0, second=0, microsecond=0)
        ticks = []
        labels = []
        for hm in range((m1-m0)//60 + 1):
            mm = m0 + hm*60
            tt = base_day + dt.timedelta(minutes=mm)
            ticks.append(mm)
            labels.append(tt.strftime("%H:%M"))
        self.ax.set_xticks(ticks); self.ax.set_xticklabels(labels, rotation=45)
        self.ax.set_xlim(m0, m1)
        self.ax.set_title(f"{port_name} – Gate occupancy (Heatmap, {bin_min}m)")
        self.canvas.draw()

    # ───────────────────────────────────────────────────────────────
    # 클릭 → 툴팁 표시: 어디서 와서 주기, 어디로 떠나는지
    # ───────────────────────────────────────────────────────────────
    def _on_click(self, event):
        if self.cbo_mode.currentText() != "Gantt":
            return
        if event.inaxes != self.ax:
            return
        for info in self._rects_info:
            patch = info["patch"]
            ok, _ = patch.contains(event)
            if not ok:
                continue
            port = info["port"]; gate = info["gate"]; uam = info["uam"]
            smin = info["start"]; emin = info["end"]
            tooltip = self._compose_tooltip(uam, port, gate, smin, emin)
            QToolTip.showText(QCursor.pos(), tooltip, self)
            break

    def _m(self, t: dt.datetime) -> float:
        return (t - self._base0).total_seconds() / 60.0


    def _compose_tooltip(self, uam: str, port: str, gate: int, smin: float, emin: float) -> str:
        """게이트 바 클릭 시 보여줄 툴팁을 보기 좋게 구성."""
        gate_label = f"G{int(gate):02d}"
        sdt = self._base0 + dt.timedelta(minutes=smin)
        edt = self._base0 + dt.timedelta(minutes=emin)
        dur_min = emin - smin
        dur_sec = int(round(dur_min * 60))
        def _fmt(t): 
            return t.strftime("%H:%M:%S") if isinstance(t, dt.datetime) else "?"
        def _fmt_pair(a,b,unit="s"):
            if isinstance(a, dt.datetime) and isinstance(b, dt.datetime):
                return f"{_fmt(a)} → {_fmt(b)} ({int((b-a).total_seconds())}{unit})"
            return f"{_fmt(a)} → {_fmt(b)}"

        # 0) 헤더
        lines: list[str] = []
        lines.append(f"<b>{port}</b> / <b>{gate_label}</b>  "
                    f"<span style='color:#555'>{_fmt(sdt)} ~ {_fmt(edt)}  "
                    f"({dur_sec}s)</span>")
        if uam:
            lines.append(f"UAM: <b>{uam}</b>")

        # 이 UAM의 모든 운항
        fls = self._by_uam.get(uam, [])

        # 1) 도착편(ARRIVAL) 매칭
        arr = None
        for f in fls:
            if f.get("destination") != port: 
                continue
            if f.get("landing_gate") != gate_label:
                continue
            gi, go = f.get("gate_in"), f.get("gate_out")
            if gi and go:
                # 클릭 범위와 게이트 in/out이 겹치면 도착편으로 간주
                if not (edt < gi or sdt > go):
                    arr = f
                    break

        if arr:
            lines.append("<hr/><b>ARRIVAL SERVICE</b>")
            origin = arr.get("origin","")
            toff_gate = arr.get("takeoff_gate") or "-"
            aldt = arr.get("actual_touch")
            gi, go = arr.get("gate_in"), arr.get("gate_out")
            gate_delay = arr.get("gate_delay_sec")
            lines.append(f"From <b>{origin}</b> / {toff_gate}")
            lines.append(f"ALDT {_fmt(aldt)}")
            if gi and go:
                lines.append(f"Gate-in/out: {_fmt_pair(gi, go)}")
            if isinstance(gate_delay, (int, float)):
                lines.append(f"Gate delay: <b>{int(gate_delay)}s</b>")

        # 2) 출발편(DEPARTURE) 매칭
        dep = None
        for f in fls:
            if f.get("origin") != port:
                continue
            if f.get("takeoff_gate") != gate_label:
                continue
            t0 = f.get("actual_takeoff_start")   # FATO TKO 시작
            if t0:
                t0m = self._m(t0)
                # 클릭구간과 TKO 시간이 ±2분 안에 있으면 출발편으로 간주
                if smin - 2 <= t0m <= emin + 2:
                    dep = f
                    break

        if dep:
            lines.append("<hr/><b>DEPARTURE</b>")
            dest = dep.get("destination","")
            next_gate = dep.get("landing_gate") or "-"
            std  = dep.get("scheduled_time")
            t0   = dep.get("actual_takeoff_start")   # FATO TKO start
            t1   = dep.get("actual_takeoff_finish")  # FATO TKO end

            taxi_out_min = getattr(self._net, "taxi_out_min", 5.0)
            pre_min = float(NEW_DEP_PREOCCUPY_MIN)      # 출발전 게이트 사전점유(분)
            linger = int(GATE_EXIT_LINGER_SEC)          # 출발 후 잠금(초)

            # Taxi-out 시작 = TKO 시작 - taxi_out_min
            to_start = (t0 - dt.timedelta(minutes=taxi_out_min)) if t0 else None
            pre_start = (to_start - dt.timedelta(minutes=pre_min)) if to_start else None
            lock_end  = (to_start + dt.timedelta(seconds=linger)) if to_start else None

            lines.append(f"To <b>{dest}</b> / next gate {next_gate}")
            lines.append(f"STD {_fmt(std)}")
            if pre_start and to_start:
                lines.append(f"Pre-occupy: {_fmt_pair(pre_start, to_start)}")
            if to_start:
                lines.append(f"Taxi-out: {_fmt(to_start)}  |  Lock: {_fmt_pair(to_start, lock_end)}")
            if t0 and t1:
                lines.append(f"ATOT: {_fmt_pair(t0, t1)}")

        return "<br/>".join(lines)

    
# ───────────────────────────────────────────────────────────────
# MapView : OpenStreetMap + 전체 네트워크 시각화
# ───────────────────────────────────────────────────────────────
class MapView(QWebEngineView):
    def __init__(self, planner: PathPlanner | None,
                 zoom: int = 11,
                 center: tuple[float, float] | None = None,
                 parent=None):
        super().__init__(parent)
        self._zoom = zoom
        self._center = center
        self._planner = planner
        self._tmp_path = None
        if planner:
            self._create_map(planner, zoom, center)

    def set_planner(self, planner: PathPlanner) -> None:
        self._planner = planner
        self._create_map(planner, self._zoom, self._center)

    @staticmethod
    def _sector(lon0: float, lat0: float,
                radius_m: float,
                bearing_deg: float,
                half_angle_deg: float,
                n_pts: int = 30) -> list[tuple[float, float]]:
        import math
        R = 6_371_000.0
        lat0_rad = math.radians(lat0)
        lon0_rad = math.radians(lon0)
        brg_rad  = math.radians(bearing_deg)
        start = brg_rad - math.radians(half_angle_deg)
        end   = brg_rad + math.radians(half_angle_deg)
        step  = (end - start) / n_pts
        poly = [(lat0, lon0)]
        for i in range(n_pts + 1):
            θ = start + i * step
            lat = math.asin(
                math.sin(lat0_rad) * math.cos(radius_m / R) +
                math.cos(lat0_rad) * math.sin(radius_m / R) * math.cos(θ)
            )
            lon = lon0_rad + math.atan2(
                math.sin(θ) * math.sin(radius_m / R) * math.cos(lat0_rad),
                math.cos(radius_m / R) - math.sin(lat0_rad) * math.sin(lat)
            )
            poly.append((math.degrees(lat), math.degrees(lon)))
        return poly

    def _create_map(self,
                    planner: PathPlanner,
                    zoom: int,
                    center: tuple[float, float] | None = None,
                    route: list[tuple[float, float]] | None = None) -> None:
        
        def _valid_latlon(lat, lon):
            return (
                lat is not None and lon is not None and
                not (isinstance(lat, float) and math.isnan(lat)) and
                not (isinstance(lon, float) and math.isnan(lon))
            )

        if center is not None:
            lat0, lon0 = center
        else:
            # [NEW] 첫 번째 '유효한' 버티포트 좌표를 중심으로 사용
            lat0 = lon0 = None
            for v in planner.iport_list:
                lon_c, lat_c = planner.nodes_geo.get(v["name"], (None, None))
                if _valid_latlon(lat_c, lon_c):
                    lon0, lat0 = lon_c, lat_c
                    break
            if lat0 is None:  # 모두 무효면 안전한 기본값(서울 시청 부근)
                lat0, lon0 = 37.5665, 126.9780

        fmap = folium.Map(location=[lat0, lon0], zoom_start=zoom, tiles=None)
        folium.TileLayer("OpenStreetMap", opacity=0.4,
                        control=False, name="Base").add_to(fmap)

        for v in planner.iport_list:
            lon, lat = planner.nodes_geo[v["name"]]
            if not _valid_latlon(lat, lon):      # [NEW] NaN 좌표 스킵
                continue
            folium.CircleMarker([lat, lon], radius=6,
                                color="blue", fill=True,
                                tooltip=v["name"]).add_to(fmap)
            for key, col, ls in (("INR", "green", 2),
                                ("OTR", "red",   2),
                                ("MTR", "purple",1)):
                r_km = v.get(key, 0)
                if r_km <= 0: continue
                folium.Circle([lat, lon], radius=r_km*1000,
                            color=col, weight=ls,
                            fill=False,
                            opacity=0.5 if key != "MTR" else 0.3).add_to(fmap)
            for deg_key, col in (("INR_Deg","green"), ("OTR_Deg","red")):
                b = v.get(deg_key)
                if b is None: continue
                poly = self._sector(lon, lat, v["MTR"]*1000, b, 10)
                folium.Polygon(poly, color=None, fill=True,
                            fill_color=col, fill_opacity=0.25).add_to(fmap)

        for w in planner.waypoint_list:
            lon, lat = planner.nodes_geo[w["name"]]
            if not _valid_latlon(lat, lon):      # [NEW] NaN 좌표 스킵
                continue
            folium.CircleMarker([lat, lon], radius=4,
                                color="green", fill=True,
                                tooltip=w["name"]).add_to(fmap)

        for u, nbrs in planner.vp_graph.items():
            lon1, lat1 = planner.nodes_geo[u]
            if not _valid_latlon(lat1, lon1):    # [NEW]
                continue
            for v in nbrs:
                if isinstance(v, tuple):
                    v = v[0]
                lon2, lat2 = planner.nodes_geo[v]
                if not _valid_latlon(lat2, lon2):  # [NEW]
                    continue
                folium.PolyLine([(lat1, lon1), (lat2, lon2)],
                                color="blue", weight=2,
                                opacity=0.4).add_to(fmap)
        for u, nbrs in planner.wp_graph.items():
            lon1, lat1 = planner.nodes_geo[u]
            if not _valid_latlon(lat1, lon1):    # [NEW]
                continue
            for v in nbrs:
                if isinstance(v, tuple):
                    v = v[0]
                lon2, lat2 = planner.nodes_geo[v]
                if not _valid_latlon(lat2, lon2):  # [NEW]
                    continue
                folium.PolyLine([(lat1, lon1), (lat2, lon2)],
                                color="red", weight=2,
                                opacity=0.9).add_to(fmap)

        if route:
            folium.PolyLine([(lat, lon) for lon, lat in route],
                            color="yellow", weight=4, opacity=0.85).add_to(fmap)
            folium.CircleMarker(route[0][::-1], radius=6,
                                color="green", fill=True,
                                tooltip="Origin").add_to(fmap)
            folium.CircleMarker(route[-1][::-1], radius=6,
                                color="red", fill=True,
                                tooltip="Destination").add_to(fmap)

        if hasattr(self, "_tmp_path") and self._tmp_path and os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        fmap.save(tmp.name); tmp.close()
        self._tmp_path = tmp.name
        self.load(QUrl.fromLocalFile(str(Path(tmp.name).resolve())))

        self._planner = planner
        self._zoom    = zoom
        self._center  = (lat0, lon0)

    def draw_route(self, lonlat_path: list[tuple[float, float]]):
        self._create_map(self._planner, self._zoom,
                         center=lonlat_path[0][::-1] if lonlat_path else None,
                         route=lonlat_path)
        
    def closeEvent(self, event):
        if self._tmp_path and os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)
        super().closeEvent(event)
