from __future__ import annotations
import sys, datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore    import Qt, QTime, QSize, pyqtSignal, QUrl
from PyQt5.QtGui import QCursor, QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QTextEdit, QProgressBar,
    QComboBox, QSizePolicy, QMessageBox, QLineEdit,
    QTimeEdit, QDoubleSpinBox, QToolTip, QSpinBox, QFrame, QTabWidget, QStackedWidget, QScrollArea                      # ← 추가
)

from PyQt5.QtGui import QPixmap, QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure  import Figure
from matplotlib.patches import Rectangle
import matplotlib.dates  as mdates
import mplcursors
import datetime
import math
from collections import defaultdict

# -------- runtime-path patch -------------------------------------------------
CUR  = Path(__file__).resolve(); ROOT = CUR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Monitoring.Functions.PathPlanning import PathPlanner
# ---- Project imports -------------------------------------------------------
from Scheduler.ArrivalsTab import ArrivalsTab 
from Scheduler.Functions.AssignmentPassenger    import AssignmentPassenger
from Scheduler.Functions.PassengerTimeScheduler import DemandProfile, PassengerTimeScheduler
from Scheduler.FplMakerTab import FplMakerTab
from Scheduler.Functions.Scheduling_Optimized import (
    RegularFlightScheduler, GRID_WAITS,
    PREP_TIME_MIN, RUNWAY_COUNT,
    simulate_ground_operations,       # 출발
    simulate_landing_ops,             # ← 새로 가져오기
    NUM_ARR_RUNWAYS,                   # 착륙-패드 개수 상수
    SEATS, MAX_CAP                              
)

from Scheduler.Functions.LocationDistanceCalculator import update_from_csv
from Scheduler.Functions.ETAComputer import compute_eta


# ── 상단 --------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "Sources"
SIM_DIR   = SOURCE_DIR
NET_FILES = {
    "vertiport": SIM_DIR / "vertiport.csv",
    "waypoint" : SIM_DIR / "waypoint.csv",
}

CSV_FILES: Dict[str, Path] = {               # 기존 5종
    "arrival"  : SOURCE_DIR / "arrival_ratios.csv",
    "departure": SOURCE_DIR / "departure_ratios.csv",
    "day"      : SOURCE_DIR / "TrafficDemand_DayofWork.csv",
    "month"    : SOURCE_DIR / "TrafficDemand_Monthly.csv",
    "timeline" : SOURCE_DIR / "TrafficDemand_Timeline.csv",
}
NET_FILES: Dict[str, Path] = {               # ★ 새로 추가
    "vertiport": SIM_DIR / "vertiport.csv",
    "waypoint" : SIM_DIR / "waypoint.csv",
}

BASE_TRAFFIC = 13_521_125


# ──────────────────────────────────────────────────────────────────────────
#  GanttCanvas (드래그-&-드롭으로 전체 교체)
# ──────────────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import QToolTip
from PyQt5 import QtCore                      # QPoint 변환용

class GanttCanvas(FigureCanvas):
    """단일 라인-Gantt  +  hover 테두리 강조 + QToolTip"""

    def __init__(self, parent: Optional[QWidget] = None):
        self.fig = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

        self._patches: list[Rectangle] = []        # 모든 막대를 저장
        self._highlighted: Optional[Rectangle] = None

        # ── 마우스 이벤트 연결 ─────────────────────────────────────────
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("figure_leave_event",  self._on_leave)

        self.prep_min = 10

    # ------------------------------------------------------------------
    def plot_flights(self, flights: List[dict], vert_name: str = "") -> None:
        title = f"{vert_name} – {len(flights)}편" if vert_name else "Gantt Chart"
        self.ax.clear()
        self.ax.set_title(title)
        self._patches.clear(); self._highlighted = None
        QToolTip.hideText()

        if not flights:
            self.draw(); return

        # 1) 막대 생성 -------------------------------------------------
        dests = sorted({f["destination"] for f in flights})
        ymap  = {d: i for i, d in enumerate(dests)}
        xs, xe = [], []

        for f in flights:
            y  = ymap[f["destination"]]
            x0 = mdates.date2num(f["scheduled_time"])
            x1 = mdates.date2num(
                    f.get("actual_takeoff_start",
                          f["scheduled_time"] + dt.timedelta(minutes=self.prep_min))
                    # f["scheduled_time"] + dt.timedelta(minutes=self.prep_min)
                    )
            xs.append(x0); xe.append(x1)

            bar = Rectangle((x0, y-0.3), x1-x0, 0.6,
                            facecolor="tab:blue", edgecolor="k", linewidth=.4)
            bar.flight = f
            self.ax.add_patch(bar)
            self._patches.append(bar)

        # 2) 축 포맷 ---------------------------------------------------
        self.ax.set_yticks(list(ymap.values()))
        self.ax.set_yticklabels(dests)
        self.ax.set_ylim(-0.5, len(dests)-0.5)
        self.ax.xaxis.set_major_locator(mdates.HourLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.tick_params(axis="x", rotation=45)
        if xs:
            self.ax.set_xlim(min(xs)-1/24, max(xe)+1/24)

        self.draw()                               # 즉시 렌더
    def set_prep(self, v:int):          # 간단 setter
        self.prep_min = v

    # =============== Hover Helpers ====================================
    def _highlight(self, bar: Rectangle, on: bool) -> None:
        if on:
            bar._old = (bar.get_edgecolor(), bar.get_linewidth())
            bar.set_edgecolor("yellow"); bar.set_linewidth(2.0)

            # ── 툴팁 표시 (★ 위치 수정) ─────────────────────────────
            info = (f"{bar.flight['flight_number']}\n"
                    f"출발 {bar.flight['scheduled_time']:%H:%M}\n"
                    f"→ {bar.flight['destination']}")
            QToolTip.showText(QCursor.pos(), info, self)   # ← ❶ 여기!
        else:
            if hasattr(bar, "_old"):
                ec, lw = bar._old
                bar.set_edgecolor(ec); bar.set_linewidth(lw)
            QToolTip.hideText()

    def _pick_bar(self, event) -> Optional[Rectangle]:
        """마우스 위치에 hit 된 막대 반환 (없으면 None)"""
        if event.inaxes is not self.ax:
            return None
        for bar in self._patches:
            contains, _ = bar.contains(event)
            if contains:
                return bar
        return None

    # -------- Mouse callbacks ----------------------------------------
    def _on_move(self, event):
        bar = self._pick_bar(event)
        if bar is self._highlighted:              # 상태 변화 X
            return

        # 이전 강조 해제
        if self._highlighted:
            self._highlight(self._highlighted, False)
            self._highlighted = None

        # 새 막대 강조
        if bar:
            self._highlight(bar, True)
            self._highlighted = bar
            self.draw_idle()
        else:
            self.draw_idle()

    def _on_leave(self, _event):
        if self._highlighted:
            self._highlight(self._highlighted, False)
            self._highlighted = None
            self.draw_idle()
# ──────────────────────────────────────────────────────────────────────────



# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, save_dir: Optional[Path] = None):
        super().__init__()
        # 전달받은 경로가 있으면 그걸 쓰고, 없으면 기본값
        self.save_dir = Path(save_dir) if save_dir else Path(__file__).resolve().parent
        self.setWindowTitle("UAM ScheduleMaker"); self.resize(1800, 1000)

        self.flights_by_origin: Dict[str, List[dict]] = {}
        self._file_paths: Dict[str, Optional[Path]] = {
            **{k: (p if p.is_file() else None) for k, p in CSV_FILES.items()},
            **{k: (p if p.is_file() else None) for k, p in NET_FILES.items()},
        }

        self._build_ui()
        self._populate_paths()

    def _window_ratio(self) -> float:
        """
        운항 시작·종료 시각( self.t_start , self.t_end )에 맞춰
        시간대 가중치의 합을 구해 반환한다.
        """
        profile = DemandProfile(excel_dir=self._file_paths["timeline"].parent)
        w       = profile.timeline_weights          # np.ndarray, len == 24, sum == 1

        s = self.t_start.time()
        e = self.t_end.time()

        start_h, start_m = s.hour(), s.minute()
        end_h,   end_m   = e.hour(), e.minute()

        # --- 범위 내 '완전 포함' 시간대 (시 단위) -------------------
        def hour_range(h0, h1):
            h = (h0 + 1) % 24
            while h != h1:
                yield h
                h = (h + 1) % 24

        ratio = sum(w[h] for h in hour_range(start_h, end_h))

        # --- 부분 시간 보정 ---------------------------------------
        ratio += w[start_h] * ((60 - start_m) / 60)   # 출발 시간대 남은 비율
        ratio += w[end_h]   * (end_m / 60)            # 종료 시간대 사용 비율

        return ratio

    def _preview_demand(self):
        """
        • est_pax  : base × ratio
        • flt_min  : 전부 9-석 기체 사용 → ceil(pax / 9)
        • flt_max  : 전부 4-석 기체 사용 → ceil(pax / 4)
        """
        est_pax_raw = int(self.spin_base.value() * self.spin_ratio.value() / 100)
        window_ratio = self._window_ratio()
        est_pax     = int(est_pax_raw * window_ratio)      # ← 운항시간 반영
        flt_min = math.ceil(est_pax / MAX_CAP)   # 최소 = 전부 MAX_CAP 기종
        flt_max = math.ceil(est_pax / 4)         # 최대 = 전부 4석
        self.lbl_prog.setText(
            f"Estimated demand: {est_pax:,} pax  |  flights {flt_min:,} ~ {flt_max:,}"
        )

    # ---------------- UI --------------------------------------------------
    def _build_ui(self):
        if not hasattr(self, "save_dir"):
            self.save_dir = Path(__file__).resolve().parent
        # 공통 탭 위젯
        tabs = QTabWidget(self)
        self.setCentralWidget(tabs)

        # ❶ Demands Generation 탭
        tab_fpl = QWidget();  tabs.addTab(tab_fpl, "Demands Generation")
        h_root  = QHBoxLayout(tab_fpl)

        # 왼쪽: 파일 + 옵션 + 로그
        left = QVBoxLayout(); h_root.addLayout(left, 0)

        # Demand CSVs
        gb_csv  = QGroupBox("Demand CSVs"); grid_csv = QGridLayout(gb_csv)
        self._edits_path: Dict[str, QLineEdit] = {}
        for r,(k,l) in enumerate([("arrival","Arrival"),("departure","Departure"),
                                ("day","Day"),("month","Month"),("timeline","Timeline")]):
            btn = QPushButton(f"Open-{l}")
            btn.clicked.connect(lambda _, key=k: self._browse(key))
            edt = QLineEdit(); edt.setReadOnly(True); edt.setPlaceholderText("(not loaded)")
            grid_csv.addWidget(btn, r, 0); grid_csv.addWidget(edt, r, 1)
            self._edits_path[k] = edt
        left.addWidget(gb_csv)

        # Network CSVs
        gb_net = QGroupBox("Network CSVs"); grid_net = QGridLayout(gb_net)
        for r, (k, l) in enumerate([("vertiport", "Vertiport"), ("waypoint",  "Waypoint")]):
            btn = QPushButton(f"Open-{l}")
            btn.clicked.connect(lambda _, key=k: self._browse(key))
            edt = QLineEdit(); edt.setReadOnly(True); edt.setPlaceholderText("(not loaded)")
            grid_net.addWidget(btn, r, 0); grid_net.addWidget(edt, r, 1)
            self._edits_path[k] = edt
        left.addWidget(gb_net)

        # Operation Settings
        gb_ops = QGroupBox("Operation Settings"); grid_ops = QGridLayout(gb_ops)

        self.t_start  = QTimeEdit(QTime(6,30));  self.t_start.setDisplayFormat("HH:mm")
        self.t_end    = QTimeEdit(QTime(21,30)); self.t_end.setDisplayFormat("HH:mm")

        self.spin_base = QSpinBox()
        self.spin_base.setRange(0, 50_000_000)
        self.spin_base.setSingleStep(10_000)
        self.spin_base.setValue(13_521_125)
        self.spin_base.setSuffix(" pax")

        self.spin_ratio = QDoubleSpinBox();  self.spin_ratio.setRange(0,100)
        self.spin_ratio.setDecimals(1);      self.spin_ratio.setSuffix(" %")
        self.spin_ratio.setSingleStep(0.1);  self.spin_ratio.setValue(0.8)
        self.spin_ratio.setToolTip("0.3 % – 중밀도\n0.8 % – 고밀도")

        self.spin_apart = QDoubleSpinBox();  self.spin_apart.setRange(0,50)
        self.spin_apart.setDecimals(1);      self.spin_apart.setSuffix(" km")
        self.spin_apart.setValue(7.0)

        # 저장 폴더 선택 UI 추가
        self.le_save_dir = QLineEdit(str(self.save_dir))
        self.btn_browse_save = QPushButton("Browse")

        self.le_save_dir.editingFinished.connect(self._apply_save_dir_from_lineedit)

        w_save = QWidget(); hb_save = QHBoxLayout(w_save); hb_save.setContentsMargins(0,0,0,0)
        hb_save.addWidget(self.le_save_dir, 1); hb_save.addWidget(self.btn_browse_save)

        rows = [
            ("Operation Start",          self.t_start),
            ("Operation End",            self.t_end),
            ("Base Traffic",             self.spin_base),
            ("Ratio of Demand Transfer", self.spin_ratio),
            ("Ground Cut-off (km)",      self.spin_apart),
            ("Save Folder",              w_save),          # ← 추가
        ]
        for r,(txt,w) in enumerate(rows):
            grid_ops.addWidget(QLabel(txt), r, 0)
            grid_ops.addWidget(w,        r, 1)
        left.addWidget(gb_ops)

        # 로그 + 버튼
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(220)
        left.addWidget(self.log, 1)
        btn_gen = QPushButton("Generate Demands"); btn_gen.clicked.connect(self._generate)
        btn_save= QPushButton("Save as");           btn_save.clicked.connect(self._save)
        hb = QHBoxLayout(); hb.addWidget(btn_gen); hb.addWidget(btn_save)
        left.addLayout(hb)

        # 오른쪽: 진행 + Gantt
        right = QVBoxLayout(); h_root.addLayout(right, 1)
        top   = QHBoxLayout()
        self.progress   = QProgressBar(); self.progress.setRange(0,100)
        self.combo_vert = QComboBox();    self.combo_vert.setFixedWidth(200)
        self.combo_vert.currentTextChanged.connect(self._on_combo)
        top.addWidget(self.progress,1); top.addWidget(self.combo_vert)
        right.addLayout(top)
        self.lbl_prog = QLabel("Generating 0%", alignment=Qt.AlignCenter)
        right.addWidget(self.lbl_prog)
        self.canvas = GanttCanvas(); self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.canvas,1)

        # ❷ FPL Maker placeholder 탭
        self.tab_opt = FplMakerTab(self)
        self.tab_opt.set_save_root(self.save_dir)           # ← 저장 루트 주입
        tabs.addTab(self.tab_opt, "FPL Maker")

        # Arrivals 탭
        self.tab_arr = ArrivalsTab()
        tabs.addTab(self.tab_arr, "Arrivals")
        self.tab_opt.generation_done.connect(self.tab_arr.set_flights)

        # 시그널: 수요 프리뷰 즉시 갱신
        self.btn_browse_save.clicked.connect(self._choose_save_dir)
        self.spin_ratio.valueChanged.connect(lambda _=None: self._preview_demand())
        self.spin_base .valueChanged.connect(lambda _=None: self._preview_demand())
        self.t_start.timeChanged.connect(  lambda _=None: self._preview_demand())
        self.t_end.timeChanged.connect(    lambda _=None: self._preview_demand())

        # 최초 프리뷰
        self._preview_demand()

    def _apply_save_dir_from_lineedit(self):
        from pathlib import Path
        import os
        text = self.le_save_dir.text().strip()
        if not text:
            return
        p = Path(os.path.expanduser(os.path.expandvars(text))).resolve()
        # 폴더가 없어도 괜찮게, 저장 시 만들 거라 여기선 세팅만
        self.save_dir = p
        if hasattr(self.tab_opt, "set_save_root"):
            self.tab_opt.set_save_root(self.save_dir)

    def _choose_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select save folder", str(self.save_dir))
        if d:
            from pathlib import Path
            self.save_dir = Path(d)
            self.le_save_dir.setText(d)
            if hasattr(self.tab_opt, "set_save_root"):
                self.tab_opt.set_save_root(self.save_dir)

    # -------------- 파일 경로 초기화 -----------------------------------
    def _populate_paths(self):
        for k, p in self._file_paths.items():
            if p.is_file():
                self._edits_path[k].setText(p.name); self._log(f"✔ {p} found")
            else:
                self._log(f"✘ missing {p}")

    # -------------- 파일 브라우저 -------------------------------------
    def _browse(self,key:str):
        fn,_ = QFileDialog.getOpenFileName(self,"CSV",str(SOURCE_DIR),"CSV (*.csv)")
        if fn:
            self._file_paths[key]=Path(fn)
            self._edits_path[key].setText(Path(fn).name)   # ← NameError 수정 (k → key)
            self._log(f"file set: {fn}")

    # ---------------------------------------------------------------------------
    def _generate(self):
        """
        1) Demand 계산 → Origin별 Flight 스케줄
        2) Progress bar 갱신
        3) FplMakerTab에 flight·Operation‑Info 전달
        """
        # 0) CSV 입력 검증 -----------------------------------------------------
        if any(v is None for v in self._file_paths.values()):
            self._log("⚠ CSV 파일을 모두 지정하세요."); return

        self.progress.setValue(0); self.lbl_prog.setText("Generating 0%")
        QApplication.processEvents()
        update_from_csv(self._file_paths["vertiport"])     # ★ 추가

        path_planner = PathPlanner(                 # ← NEW
            self._file_paths["vertiport"],
            self._file_paths["waypoint"]
        )
        
        # 1) “Ratio of Demand” → 총 승객 수 계산 ------------------------------
        base_traffic = self.spin_base.value()          # ★ GUI 입력 사용
        ratio_pct    = self.spin_ratio.value()
        total_people = int(base_traffic * ratio_pct / 100 * self._window_ratio())

        # 2) 승객 수요·도착시각 생성 -----------------------------------------
        planner = AssignmentPassenger(
            dep_csv = self._file_paths["departure"],
            arr_csv = self._file_paths["arrival"],
            vert_csv= self._file_paths["vertiport"],
        )

        plan = planner.plan_traffic(total_people)

        # DemandProfile : 3종 CSV가 **같은 폴더**에 있다는 전제
        base_dir = self._file_paths["day"].parent     # day·month·timeline csv 가 같이 있나 확인
        profile = DemandProfile(excel_dir=self._file_paths["day"].parent)

        pts      = PassengerTimeScheduler(plan, planner.locations, profile)
        p_info   = pts.assign_arrival_times(dt.date.today())

        # 3) 파라미터 준비 ----------------------------------------------------
        total_origins      = len(planner.locations)
        self.flights_by_origin.clear()
        cut_km   = self.spin_apart.value()
        # prep_min = self.spin_prep.value()
        from Scheduler.Functions.Scheduling_Optimized import TAKEOFF_MIN  # 이미 import 되어있다면 생략 가능
        TAXI_OUT_MIN = 5
        
        # 4) Origin 루프 ------------------------------------------------------
        for idx, origin in enumerate(planner.locations, 1):
            sched = RegularFlightScheduler(p_info)

            # 4‑1) 빈좌석 최소 wait* 탐색
            best_wait, best_empty, best_fl = None, float("inf"), []
            for w in GRID_WAITS:
                fl = sched.schedule_flights_for_origin(origin, w)
                empty = sum(SEATS[f['aircraft_type']] - f['passengers']   # ★ 수정
                            for f in fl)
                if empty < best_empty:
                    best_wait, best_empty, best_fl = w, empty, fl

            # 4‑2) 운영시간 윈도우 필터 ------------------------------------
            op_start = self.t_start.time()
            op_end   = self.t_end.time()
            best_fl  = [f for f in best_fl
                        if op_start <= f["scheduled_time"].time() <= op_end]

            # 4‑3) 지상 시뮬 -------------------------------------------------
            best_fl = simulate_ground_operations(
                [f.copy() for f in best_fl],
                num_runways=RUNWAY_COUNT,
                cut_km=cut_km,
                taxi_out_min=TAXI_OUT_MIN  
            )
            self.flights_by_origin[origin] = best_fl

            # --- 진행률 -----------------------------------------------------
            pct = int(idx / total_origins * 100)
            self.progress.setValue(pct)
            self.lbl_prog.setText(f"Generating {pct}%")
            QApplication.processEvents()
            
        # 5) 콤보박스·Gantt 초기화 ------------------------------------------
        origins = sorted(self.flights_by_origin)
        self.combo_vert.clear()
        self.combo_vert.addItems(origins)
        if origins:
            self.canvas.set_prep(TAXI_OUT_MIN)
            self.combo_vert.setCurrentIndex(0)

        # -------------------------------------------------------------------
        # ### NEW : Operation‑Info 패널에 값 전달 ############################
        total_flights = sum(len(lst) for lst in self.flights_by_origin.values())
        op_start_txt  = self.t_start.time().toString("HH:mm")
        op_end_txt    = self.t_end.time().toString("HH:mm")

        # 7) FPL Maker 탭 갱신 ---------------------------------------
        if hasattr(self.tab_opt, "reset_state_for_new_demands"):
            self.tab_opt.reset_state_for_new_demands()   # ★ Demand 생성마다 FPL 탭 초기화
        self.tab_opt.set_planner(
            self._file_paths["vertiport"], self._file_paths["waypoint"]
        )
        self.tab_opt.set_flights(self.flights_by_origin)

        # -------------------------------------------------------------------

        self._log(f"Generation complete – {total_people:,} Demand "
                f"({ratio_pct} %).  Flights: {total_flights:,}")

    # -------------- Save (stub) ---------------------------------------
    def _save(self):
        QMessageBox.information(self,"Save","Save not implemented yet.")

    # -------------- Combo 변경 ----------------------------------------
    def _on_combo(self, name: str):
        flights = self.flights_by_origin.get(name, [])
        self.canvas.plot_flights(flights, vert_name=name)   # ← 이름 전달

    # -------------- 로그 헬퍼 ----------------------------------------
    def _log(self,msg:str): self.log.append(msg)

# ───────────────────────── StartWindow (런처) ─────────────────────────
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout,
    QFrame, QSizePolicy, QMessageBox, QToolButton
)
from PyQt5.QtGui import QPixmap, QIcon
from pathlib import Path

class BlueCard(QFrame):
    def __init__(self, text="Blank", fixed_h=None):
        super().__init__()
        self.setObjectName("BlueCard")
        lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16)
        lab = QLabel(text); lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("font-size:18px; color:#e5ecff;")
        lay.addWidget(lab, 1)
        if fixed_h: self.setFixedHeight(fixed_h)

class InfoTable(QFrame):
    def __init__(self, ver:str, updated:str, expire:str):
        super().__init__()
        self.setObjectName("InfoTable")
        g = QGridLayout(self); g.setContentsMargins(8,8,8,8); g.setSpacing(2)
        rows = [("프로그램 버전", ver), ("업데이트 날짜", updated), ("사용 기한", expire)]
        for r,(k,v) in enumerate(rows):
            kL = QLabel(k); vL = QLabel(v)
            for w in (kL, vL):
                w.setAlignment(Qt.AlignCenter); w.setStyleSheet("color:#e5ecff; font-size:16px; padding:6px 10px;")
            kL.setObjectName("CellH"); vL.setObjectName("CellB")
            g.addWidget(kL, r, 0); g.addWidget(vL, r, 1)

class ImageButton(QToolButton):
    def __init__(self, title:str, imgname:str):
        super().__init__()
        self.setObjectName("BigButton")
        self.setText(title)
        self._imgname = imgname
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setIconSize(QSize(300, 300))        # 아이콘 크기
        self.setMinimumHeight(400)               # 버튼 자체 높이
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def load_icon(self, search_dirs):
        for d in search_dirs:
            p = (Path(d) / self._imgname)
            if p.exists():
                pm = QPixmap(str(p))
                if not pm.isNull():
                    self.setIcon(QIcon(pm))
                    return
        # 이미지 없으면 플레이스홀더 텍스트 유지
        self.setText(self.text())

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)

class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAM Flight Scheduler")
        self.resize(1800, 1000)

        # 중앙은 스택 위젯
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)

        # ── 0) 공통 리소스 경로
        here = Path(__file__).resolve().parent
        img_dirs = [here/"Assets", here/"Sources", here.parent/"Assets", here.parent/"Sources"]

        # ── 1) 홈 페이지 (지금까지 만들던 그리드)
        self.page_home = QWidget()
        self.stack.addWidget(self.page_home)
        grid = QGridLayout(self.page_home)
        grid.setContentsMargins(24,24,24,24)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(16)

        title = QLabel("UAM Flight Scheduler")
        f = QFont(); f.setPointSize(40); f.setWeight(QFont.Black)
        title.setFont(f); title.setStyleSheet("color:#e5ecff;")
        grid.addWidget(title, 0, 0, 1, 2, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        # 상단 우측 로고
        logo_top = ClickableLabel()
        logo_top.setObjectName("LogoTop")
        logo_top.setCursor(Qt.PointingHandCursor)   # 마우스 오버 시 손가락 커서
        logo_top.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        logo_top.setToolTip("KADA 홈페이지 열기")
        
        kada_path = None
        for d in img_dirs:
            p = (d / "KADA.png")
            if p.exists(): kada_path = str(p); break
        if kada_path:
            pm = QPixmap(kada_path)
            logo_top.setPixmap(pm.scaledToHeight(64, Qt.SmoothTransformation))
        else:
            logo_top.setText("KADA.png"); logo_top.setStyleSheet("color:#e5e7eb; font-size:18px;")
        grid.addWidget(logo_top, 0, 2)

        # kada_path 세팅 이후 공통으로
        logo_top.clicked.connect(self._open_kada_site)

        # 정보 표 + 공백(투명) 4칸
        info = InfoTable("v0.0.1", "25년 08월 12일", "~25년 08월 13일")
        grid.addWidget(info, 1, 0)

        # 투명 공백 위젯 (배경색과 동일하게 보이도록)
        class ClearSpace(QFrame):
            def __init__(self, h): super().__init__(); self.setObjectName("ClearSpace"); self.setFixedHeight(h)

        grid.addWidget(ClearSpace(110), 1, 1)
        grid.addWidget(ClearSpace(110), 1, 2)
        grid.addWidget(ClearSpace(110), 2, 1)
        grid.addWidget(ClearSpace(110), 2, 2)

        # 큰 버튼 3개
        self.btn_start = ImageButton("시작하기", "Start.png")
        self.btn_info  = ImageButton("프로그램 정보 및 사용방법", "Info.png")
        self.btn_set   = ImageButton("설정", "Setting.png")
        for b in (self.btn_start, self.btn_info, self.btn_set):
            ff = b.font(); ff.setPointSize(18); ff.setWeight(QFont.Black); b.setFont(ff)
            b.load_icon(img_dirs)

        grid.addWidget(self.btn_start, 3, 0)
        grid.addWidget(self.btn_info,  3, 1)
        grid.addWidget(self.btn_set,   3, 2)

        # 하단 푸터
        footer = QFrame(); footer.setObjectName("Footer"); footer.setFixedHeight(60)
        hb = QHBoxLayout(footer); hb.setContentsMargins(0,0,0,0); hb.setSpacing(0)
        hb.addStretch(1)
        copy = QLabel("This software is developed by KADA © 2025 All rights reserved")
        copy.setObjectName("FooterText")
        hb.addWidget(copy, 0, Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(footer, 4, 0, 1, 3)

        grid.setRowStretch(0, 0); grid.setRowStretch(1, 0); grid.setRowStretch(2, 0)
        grid.setRowStretch(3, 2); grid.setRowStretch(4, 1)
        for c in range(3): grid.setColumnStretch(c, 1)

        # ── 2) 도움말 페이지
        self.page_help = HelpPage(img_dirs, back_cb=self._back_home)
        self.stack.addWidget(self.page_help)

        # ── 3) 설정 페이지
        self.data_dir = None
        self.page_settings = SettingsPage(back_cb=self._back_home,
                                          save_cb=self._set_data_dir)
        self.stack.addWidget(self.page_settings)

        # 버튼 동작
        self.btn_start.clicked.connect(self._open_scheduler)           # 1. 시작하기
        self.btn_info.clicked.connect(lambda: self._show_page("help")) # 2. 도움말
        self.btn_set.clicked.connect(lambda: self._show_page("settings")) # 3. 설정

        # 스타일 (추가 셀렉터 포함)
        self.setStyleSheet("""
            QMainWindow { background:#163255; }

            QFrame#BlueCard { background:#446ccf; border-radius:12px; }
            QFrame#InfoTable { background:transparent; }
            QLabel#CellH, QLabel#CellB { background:#0f2444; border:1px solid #2b4479; }

            QToolButton#BigButton {
                background:#f3f4f6; border:none; border-radius:16px; padding:24px; color:#111827; font-size:30px; font-weight:900;
            }
            QToolButton#BigButton:hover { background:#e5e7eb; }

            QPushButton { background:#e5e7eb; border:none; border-radius:10px; padding:10px 18px; }
            QPushButton:hover { background:#d1d5db; }

            QFrame#ClearSpace { background:transparent; }
            QFrame#Footer { background:transparent; }
            QLabel#FooterText { color:#cbd5e1; font-size:12px; padding-right:8px; }

            /* 도움말/설정 공용 */
            QLabel#PageTitle { color:#e5ecff; font-size:36px; font-weight:900; }
            QLabel#SectionTitle { color:#e5ecff; font-size:20px; font-weight:700; }
            QLabel#BodyText { color:#e5e7eb; font-size:15px; }
            QFrame#InnerCard { background:#0f2444; border-radius:16px; }
        """)

    def _show_page(self, name: str):
        if name == "help":
            self.stack.setCurrentWidget(self.page_help)
        elif name == "settings":
            self.stack.setCurrentWidget(self.page_settings)

    def _back_home(self):
        self.stack.setCurrentWidget(self.page_home)

    def _set_data_dir(self, path: str):
        self.data_dir = Path(path)  # 필요하면 QSettings로 영구 저장 가능

    def _open_scheduler(self):
        # 설정에서 저장해 둔 경로를 MainWindow로 넘긴다
        self.app_win = MainWindow(save_dir=self.data_dir)
        self.app_win.show()
        self.close()

    def _show_info(self):
        QMessageBox.information(self, "프로그램 정보", "UAM Flight Scheduler")

    def _open_settings(self):
        QMessageBox.information(self, "설정", "설정 메뉴 준비중")

    def _open_kada_site(self):
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://kada.konkuk.ac.kr"))

class ClearSpace(QFrame):
    def __init__(self, fixed_h=None):
        super().__init__()
        self.setObjectName("ClearSpace")
        if fixed_h: self.setFixedHeight(fixed_h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

class HelpPage(QWidget):
    def __init__(self, img_dirs, back_cb):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(24,24,24,24); lay.setSpacing(16)

        # 헤더
        hdr = QHBoxLayout()
        t = QLabel("프로그램 정보 및 사용방법"); t.setObjectName("PageTitle")
        btn_back = QPushButton("뒤로가기"); btn_back.clicked.connect(back_cb)
        hdr.addWidget(t); hdr.addStretch(1); hdr.addWidget(btn_back)
        lay.addLayout(hdr)

        # 스크롤 본문
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        cont = QWidget(); scroll_lay = QVBoxLayout(cont); scroll_lay.setSpacing(16)

        # 섹션 1
        sec1 = QFrame(); sec1.setObjectName("InnerCard")
        s1 = QVBoxLayout(sec1); s1.setContentsMargins(16,16,16,16)
        s1.addWidget(self._img_label("Help_01.png", img_dirs))
        txt1 = QLabel("개요: 이 프로그램은 수요 기반 비행 스케줄 생성 경로 계산 지상 시뮬레이션을 한 번에 지원합니다")
        txt1.setObjectName("BodyText"); txt1.setWordWrap(True)
        s1.addWidget(txt1)
        scroll_lay.addWidget(sec1)

        # 섹션 2
        sec2 = QFrame(); sec2.setObjectName("InnerCard")
        s2 = QVBoxLayout(sec2); s2.setContentsMargins(16,16,16,16)
        st = QLabel("기본 흐름"); st.setObjectName("SectionTitle")
        s2.addWidget(st)
        flow = QLabel("1. CSV 경로를 지정합니다\n2. 운항 시간과 지상 파라미터를 설정합니다\n3. 수요를 생성하고 스케줄을 확인합니다\n4. 필요 시 Gantt에서 확인하고 저장합니다")
        flow.setObjectName("BodyText"); flow.setWordWrap(True)
        s2.addWidget(flow)
        s2.addWidget(self._img_label("Help_02.png", img_dirs))
        scroll_lay.addWidget(sec2)

        scroll_lay.addStretch(1)
        scroll.setWidget(cont)
        lay.addWidget(scroll, 1)

    def _img_label(self, name, img_dirs):
        lab = QLabel(); lab.setAlignment(Qt.AlignCenter)
        for d in img_dirs:
            p = (Path(d)/name)
            if p.exists():
                pm = QPixmap(str(p))
                if not pm.isNull():
                    lab.setPixmap(pm.scaledToWidth(980, Qt.SmoothTransformation))
                    return lab
        # 이미지 없으면 자리표시
        ph = QLabel("이미지 파일을 넣어주세요: " + name)
        ph.setAlignment(Qt.AlignCenter)
        ph.setStyleSheet("color:#9ca3af; font-size:14px;")
        return ph

class SettingsPage(QWidget):
    def __init__(self, back_cb, save_cb):
        super().__init__()
        self._save_cb = save_cb

        lay = QVBoxLayout(self); lay.setContentsMargins(24,24,24,24); lay.setSpacing(16)

        # 헤더
        hdr = QHBoxLayout()
        t = QLabel("설정"); t.setObjectName("PageTitle")
        btn_back = QPushButton("뒤로가기"); btn_back.clicked.connect(back_cb)
        hdr.addWidget(t); hdr.addStretch(1); hdr.addWidget(btn_back)
        lay.addLayout(hdr)

        # 카드
        card = QFrame(); card.setObjectName("InnerCard")
        g = QGridLayout(card); g.setContentsMargins(16,16,16,16); g.setSpacing(10)

        g.addWidget(QLabel("데이터 저장 폴더"), 0, 0)
        self.ed_path = QLineEdit(); self.ed_path.setReadOnly(True)
        g.addWidget(self.ed_path, 0, 1)
        btn_browse = QPushButton("찾아보기")
        btn_browse.clicked.connect(self._choose_dir)
        g.addWidget(btn_browse, 0, 2)

        self.lbl_msg = QLabel(""); self.lbl_msg.setObjectName("BodyText")
        g.addWidget(self.lbl_msg, 1, 1, 1, 2)

        btn_save = QPushButton("저장")
        btn_save.clicked.connect(self._save)
        g.addWidget(btn_save, 2, 2, alignment=Qt.AlignRight)

        lay.addWidget(card)
        lay.addStretch(1)

    def _choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "데이터 저장 폴더 선택", str(Path.home()))
        if path:
            self.ed_path.setText(path)
            self.lbl_msg.setText("폴더가 선택되었습니다")

    def _save(self):
        p = self.ed_path.text().strip()
        if not p:
            self.lbl_msg.setText("먼저 폴더를 선택하세요")
            return
        self._save_cb(p)   # StartWindow에 저장 콜백
        self.lbl_msg.setText("저장되었습니다")

# ---------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = StartWindow()
    win.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
