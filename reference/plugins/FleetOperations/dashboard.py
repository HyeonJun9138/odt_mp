# plugins/FleetOperations/dashboard.py
# Fleet Operation Center (AirSim 전용)
# - Selected Vertiport별로 Aircraft/Gate/FATO 간트 렌더
# - QScrollArea를 사용해 가로-세로 스크롤바 제공 (콘텐츠 크기=시간×행수)
# - Play/xN/Pause/Reset: SITL step → snapshot → AirSim update_actor + prune
# ------------------------------------------------------------
from __future__ import annotations
import errno
import json
import os
import re
import socket
import subprocess
import sys
import time
import threading
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QSize, QPoint, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QPainter, QPixmap, QColor, QBrush, QPolygon, QCursor
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QLabel, QPushButton, QLineEdit, QComboBox,
    QCheckBox, QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QScrollArea,
    QSizePolicy, QToolButton, QFileDialog, QToolButton, QFileDialog, QToolTip,
    QButtonGroup
)

# ----- 공통 테마/스케일링 -----
try:
    from plugins.Viewport.scaling import UiScale, add_drop_shadow
    from plugins.Viewport.theming import apply_common_qss
except Exception:
    from ..Viewport.scaling import UiScale, add_drop_shadow  # type: ignore
    from ..Viewport.theming import apply_common_qss  # type: ignore

# ----- FPL 유틸 -----
from .fpl_loader import (
    resolve_fpl_container_path, read_fpl_stats_from_path, build_gantt_from_folder,
    GanttBundle, Bar
)

# ----- SITL / AirSim 브릿지 (프로젝트 구조별 폴백) -----
try:
    from SITL.sitl_sim import SitlSim
except Exception:
    try:
        from plugins.SITL.sitl_sim import SitlSim  # type: ignore
    except Exception:
        SitlSim = None  # type: ignore

try:
    from SITL.airsim_bridge import AirSimFleetBridge, LLAConverter, GateSpawnResolver  # type: ignore
    import SITL.airsim_bridge as _airsim_bridge_mod  # type: ignore
except Exception:
    try:
        from plugins.SITL.airsim_bridge import AirSimFleetBridge, LLAConverter, GateSpawnResolver  # type: ignore
        import plugins.SITL.airsim_bridge as _airsim_bridge_mod  # type: ignore
    except Exception:
        try:
            from airsim_bridge import AirSimFleetBridge, LLAConverter, GateSpawnResolver  # type: ignore
            import airsim_bridge as _airsim_bridge_mod  # type: ignore
        except Exception:
            AirSimFleetBridge = None
            LLAConverter = None
            GateSpawnResolver = None
            _airsim_bridge_mod = None

if _airsim_bridge_mod is not None:
    try:
        import math as _math
        overrides = {
            "WORLD_OFFSET_N": 0.0,
            "WORLD_OFFSET_E": 0.0,
            "WORLD_OFFSET_D": -2.5,
            "YAW_BIAS_DEG": -90.0,
            "YAW_TAU": 0.50,
            "YAW_RATE_MAX": _math.radians(60.0),
            "BANK_ENABLE": False,
            "BANK_MAX_DEG": 8.0,
            "BANK_TAU": 0.50,
        }
        for key, val in overrides.items():
            if hasattr(_airsim_bridge_mod, key):
                setattr(_airsim_bridge_mod, key, val)
    except Exception:
        pass

if AirSimFleetBridge is None:
    try:
        from SITL.sitl_main import AirSimFleetBridge as _InlineBridge  # type: ignore
        AirSimFleetBridge = _InlineBridge
        LLAConverter = None
        GateSpawnResolver = None
    except Exception:
        try:
            from plugins.SITL.sitl_main import AirSimFleetBridge as _InlineBridge  # type: ignore
            AirSimFleetBridge = _InlineBridge
            LLAConverter = None
            GateSpawnResolver = None
        except Exception:
            pass

try:
    import airsim  # type: ignore
except Exception:
    airsim = None


# --- REPLACE: GanttCanvas ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import datetime as dt

BASE_DIR = Path(__file__).resolve().parent
SOURCES_DIR = BASE_DIR / "Sources"
DEFAULT_UDP_CONFIG_PATH = Path(
    os.environ.get("FLEETOPS_UDP_CONFIG", SOURCES_DIR / "fleet_udp.json")
)


class SimulationThread(QThread):
    """Background worker that advances the SITL simulation and produces snapshots."""

    snapshot_ready = Signal(object, object, float, float, float)

    def __init__(self, parent: Optional[QWidget] = None, interval_ms: int = 100) -> None:
        super().__init__(parent)
        self._interval_ms = max(5, int(interval_ms))
        self._sim_lock = threading.Lock()
        self._sim = None
        self._running = False
        self._terminate = False
        self._speed = 1.0

    def set_sim(self, sim) -> None:
        with self._sim_lock:
            self._sim = sim
        self._running = False

    def clear_sim(self) -> None:
        self.stop_loop()
        with self._sim_lock:
            self._sim = None

    def set_speed(self, speed: float) -> None:
        self._speed = float(speed)

    def start_loop(self) -> None:
        self._running = True

    def stop_loop(self) -> None:
        self._running = False

    def shutdown(self) -> None:
        self._terminate = True
        self.wait(500)

    def run(self) -> None:  # type: ignore[override]
        last = None
        while not self._terminate:
            with self._sim_lock:
                sim = self._sim
            if not self._running or sim is None:
                last = None
                self.msleep(10)
                continue

            now = time.perf_counter()
            if last is None:
                last = now
                self.msleep(self._interval_ms)
                continue

            real_dt = max(0.0, now - last)
            last = now
            speed = float(self._speed)

            try:
                if hasattr(sim, "sim_speed"):
                    sim.sim_speed = speed
                if hasattr(sim, "step"):
                    sim.step(real_dt)

                raw_snap = None
                snap = None
                if hasattr(sim, "snapshot"):
                    try:
                        raw_snap = sim.snapshot()
                        snap = raw_snap
                    except Exception:
                        raw_snap = None
                if snap is None and hasattr(sim, "snapshot_for_airsim"):
                    try:
                        snap = sim.snapshot_for_airsim()
                    except Exception:
                        snap = None

                now_t = time.perf_counter()
                self.snapshot_ready.emit(raw_snap, snap, speed, now_t, real_dt)
            except Exception:
                traceback.print_exc()

            self.msleep(self._interval_ms)


class SnapshotProcessor(QThread):
    """Worker that applies heavy post-processing (AirSim updates) off the UI thread."""

    def __init__(self, owner: "FleetOperationDashboard") -> None:
        super().__init__(owner)
        self._owner = owner
        self._queue: deque[tuple[object, object, float, float]] = deque()
        self._queue_lock = threading.Lock()
        self._wake = threading.Event()
        self._terminate = False

    def enqueue(self, raw_snap: object, snap: object, speed: float, now_t: float) -> None:
        with self._queue_lock:
            if raw_snap is None and snap is None:
                return
            self._queue.append((raw_snap, snap, speed, now_t))
            # 너무 오래된 스냅샷이 쌓이면 버리되, 최근 몇 개는 유지해 보간 단절을 줄인다
            while len(self._queue) > 3:
                self._queue.popleft()
            self._wake.set()

    def flush(self) -> None:
        with self._queue_lock:
            self._queue.clear()
        self._wake.clear()

    def shutdown(self) -> None:
        self._terminate = True
        self._wake.set()
        self.wait(500)

    def run(self) -> None:  # type: ignore[override]
        while not self._terminate:
            self._wake.wait(0.1)
            if self._terminate:
                break
            payload: tuple[object, object, float, float] | None = None
            with self._queue_lock:
                if self._queue:
                    payload = self._queue.pop()
                    self._queue.clear()
                self._wake.clear()
            if payload is None:
                continue
            raw_snap, snap, speed, now_t = payload
            try:
                self._owner._process_snapshot_async(raw_snap, snap, speed, now_t)
            except Exception:
                traceback.print_exc()
class GanttCanvas(FigureCanvas):
    """QScrollArea와 함께 쓰는 가변 크기 간트 캔버스 (여백 최소/좌측 앵커/직관적 줌)."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 4), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._hits: list[tuple[object, tuple]] = []   # (patch, (row_key, smin, emin, label, meta))
        self._time_line = None
        self._marker_dt: dt.datetime | None = None
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._press = None
        self._scroll_area = None
        self._base0: dt.datetime | None = None
        self._data_xlim: tuple[float, float] | None = None  # (min_x, max_x)

        # 테마
        self.fig.patch.set_facecolor("#0f172a")
        self.ax.set_facecolor("#0f172a")
        self._style_axes()

        # 인터랙션
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # 초기 빈 화면
        self.clear_blank()

    def set_scroll_area(self, scroll_area):
        self._scroll_area = scroll_area

    def clear_blank(self, bg: str = "#0f172a"):
        """축/라벨 없이 완전 빈 배경."""
        self.ax.clear()
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        self.ax.axis("off")
        w_px, h_px = 900, 360
        self.setMinimumSize(w_px, h_px)
        self.fig.set_size_inches(w_px / self.fig.dpi, h_px / self.fig.dpi)
        self.draw()

    # ---------- 내부 스타일 ----------
    def _style_axes(self):
        ax = self.ax
        # 스파인 최소화(그래프 느낌 제거)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.grid(True, axis="x", color="#334155", alpha=0.55, linestyle="--", linewidth=0.6)
        ax.margins(x=0, y=0)                 # ★ 좌/우 여백 0 → 왼쪽에 딱 붙음
        self.fig.subplots_adjust(            # ★ 바깥 여백 최소화(제목 없음)
            left=0.045, right=0.995, top=0.995, bottom=0.18
        )

    def _apply_extent(self, rows_cnt: int, time_span_min: float):
   
        hours = max(1e-3, time_span_min / 60.0)
        boost = max(1.0, min(6.0, 4.5 / max(hours, 0.25)))
        px_per_hour = 130.0 * boost
        row_h = 24.0

        # ?? ?????? ???
        w_px = int(hours * px_per_hour + 120)             # ????? ??? ??????
        h_content = int(max(200.0, rows_cnt * row_h + 110.0))  # ????? ??? ???

        # ??????? ?? ??? ?? ??????? ????? '???? ????'
        vp_h = None
        vp_w = None
        if self._scroll_area is not None:
            try:
                viewport = self._scroll_area.viewport()
                vp_h = int(viewport.height())
                vp_w = int(viewport.width())
            except Exception:
                vp_h = None
                vp_w = None
        h_px = max(h_content, (vp_h - 2) if vp_h else h_content)
        if vp_w:
            w_px = max(w_px, vp_w - 8)

        # ?? ????? ?????? '?????' ?? ???? ????
        self.setMinimumSize(w_px, h_px)
        self.setMaximumSize(w_px, h_px)
        self.resize(w_px, h_px)
        self.updateGeometry()

        # matplotlib figure?? ??????? ????
        self.fig.set_size_inches(w_px / self.fig.dpi, h_px / self.fig.dpi)


    def draw_bars(self, title, base0, rows, time_span_min):
        ax = self.ax
        ax.clear(); self._style_axes(); self._base0 = base0
        self._hits.clear()
        if not rows:
            self.clear_blank(); return

        keys = sorted({r[0] for r in rows})
        ymap = {k: i for i, k in enumerate(keys)}
        self._apply_extent(len(keys), time_span_min)

        xs, xe = [], []
        palette = {
            "GATE_DEP_PRE": "#38BDF8", "GATE_LOCK": "#38BDF8",
            "GATE_ARR_SVC": "#86EFAC", "FATO_TKO": "#60A5FA",
            "FATO_LDG": "#F59E0B", "FLIGHT": "#A78BFA",
        }

        for item in rows:
            # item: (row_key, smin, emin, label[, meta])
            if len(item) == 4:
                rk, smin, emin, lab = item; meta = {}
            else:
                rk, smin, emin, lab, meta = item
            s = base0 + dt.timedelta(minutes=float(smin))
            e = base0 + dt.timedelta(minutes=float(emin))
            left = mdates.date2num(s); width = (e - s).total_seconds() / 86400.0
            rect = ax.barh(ymap[rk], width, left=left, height=0.62,
                           edgecolor="#0b1220", linewidth=0.4,
                           color=palette.get(lab, "#64748b"))[0]
            self._hits.append((rect, (rk, smin, emin, lab, meta)))
            xs.append(left); xe.append(left + width)

        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels(keys, color="#cbd5e1")
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelrotation=45, colors="#cbd5e1")

        left, right = min(xs), max(xe)
        self._data_xlim = (left, right)
        ax.set_xlim(left, right); ax.set_ylim(-0.5, len(keys) - 0.5)
        self._refresh_time_marker()
        self.draw()

        if self._scroll_area is not None:
            self._scroll_area.horizontalScrollBar().setValue(self._scroll_area.horizontalScrollBar().minimum())
            self._scroll_area.verticalScrollBar().setValue(self._scroll_area.verticalScrollBar().minimum())

    def _on_hover(self, ev):
        # 축 밖이거나 hit-test 대상이 없으면 툴팁 숨김
        if ev.inaxes is not self.ax or not self._hits:
            QToolTip.hideText()
            return
        for patch, info in reversed(self._hits):
            try:
                ok, _ = patch.contains(ev)
            except Exception:
                ok = False
            if ok:
                QToolTip.showText(QCursor.pos(), self._format_tip(info), self)
                return
        QToolTip.hideText()

    def _format_tip(self, info: tuple) -> str:
        rk, smin, emin, lab, meta = info
        # 시간 문자열
        def _fmt(dtobj):
            if not dtobj: return ""
            return dtobj.strftime("%H:%M:%S" if dtobj.second else "%H:%M")
        base = self._base0 or dt.datetime.combine(dt.date.today(), dt.time())
        sdt = meta.get("start_dt") or (base + dt.timedelta(minutes=float(smin)))
        edt = meta.get("end_dt")   or (base + dt.timedelta(minutes=float(emin)))
        s_txt, e_txt = _fmt(sdt), _fmt(edt)

        acid = meta.get("acid") or ""
        o = meta.get("from") or ""; d = meta.get("to") or ""
        if lab.startswith("GATE"):
            gate = meta.get("gate") or rk
            phase = "Arr Service" if lab == "GATE_ARR_SVC" else ("Pre-occupy" if lab == "GATE_DEP_PRE" else "Lock")
            return f"<b>Gate {gate}</b> <span style='color:#94a3b8'>({acid})</span><br>{phase}: {s_txt}–{e_txt}<br>{o} → {d}"
        elif lab.startswith("FATO"):
            fato = meta.get("fato") or rk
            phase = "Takeoff" if lab == "FATO_TKO" else "Landing"
            return f"<b>FATO {fato}</b> <span style='color:#94a3b8'>({acid})</span><br>{phase}: {s_txt}–{e_txt}<br>{o} → {d}"
        else:  # FLIGHT
            return f"<b>{acid}</b><br>{s_txt} → {e_txt}<br>{o} → {d}"

    # ---------- 줌/이동 ----------
    def _zoom(self, scale: float, center: float | None = None):
        """가로 줌(scale<1 확대, >1 축소), 데이터 범위 내로 클램프."""
        if self._data_xlim is None:
            return
        xmin, xmax = self.ax.get_xlim()
        c = center if center is not None else (xmin + xmax) / 2.0
        new_w = (xmax - xmin) * scale
        l = c - new_w / 2.0
        r = c + new_w / 2.0
        dmin, dmax = self._data_xlim
        if l < dmin: r += (dmin - l); l = dmin
        if r > dmax: l -= (r - dmax); r = dmax
        if r - l < 1e-8:
            r = l + 1e-8
        self.ax.set_xlim(l, r)
        self.draw()

    def _fit(self):
        if self._data_xlim:
            self.ax.set_xlim(*self._data_xlim)
            self.draw()

    def _anchor_left(self):
        """현재 보이는 폭 유지한 채 왼쪽으로 딱 붙이기."""
        if self._data_xlim:
            cur = self.ax.get_xlim()
            width = cur[1] - cur[0]
            l, r = self._data_xlim
            self.ax.set_xlim(l, min(l + width, r))
            self.draw()

    # ---------- 마우스/키 이벤트 ----------
    def _on_press(self, ev): self._press = (ev.xdata, ev.ydata)
    def _on_release(self, ev): self._press = None
    def _on_move(self, ev):
        if self._press and ev.xdata:
            x0, _ = self._press
            dx = x0 - ev.xdata
            xmin, xmax = self.ax.get_xlim()
            self.ax.set_xlim(xmin + dx, xmax + dx)
            self.draw()
            self._press = (ev.xdata, ev.ydata)

    def _on_scroll(self, ev):
        # 기본: 수직 스크롤, Shift+휠: 수평 스크롤, Ctrl+휠: 가로 줌
        if self._scroll_area is not None and ev.key not in ("control", "ctrl", "cmd"):
            steps = getattr(ev, "step", 1)
            if ev.key in ("shift",):
                sb = self._scroll_area.horizontalScrollBar()
                sb.setValue(sb.value() - steps * 80)
            else:
                sb = self._scroll_area.verticalScrollBar()
                sb.setValue(sb.value() - steps * 80)
            return
        # Ctrl+휠 → 줌
        xmin, xmax = self.ax.get_xlim()
        c = ev.xdata if ev.xdata is not None else (xmin + xmax) / 2
        self._zoom(0.9 if ev.button == "up" else 1.1, c)

    def _on_key(self, ev):
        # + / - : 줌, 0: 전체 표시, f: fit, Home: 좌측 앵커
        if ev.key in ("+", "="):
            self._zoom(0.9)
        elif ev.key in ("-", "_"):
            self._zoom(1.1)
        elif ev.key == "0":
            self._fit()
        elif ev.key in ("f", "F"):
            self._fit()
        elif ev.key.lower() in ("home",):
            self._anchor_left()

    def _refresh_time_marker(self) -> None:
        if self._time_line is None and self._marker_dt is None:
            return
        if self._marker_dt is None or self._base0 is None:
            if self._time_line is not None:
                self._time_line.set_visible(False)
                self.ax.figure.canvas.draw_idle()
            return
        x = mdates.date2num(self._marker_dt)
        if self._time_line is None:
            self._time_line = self.ax.axvline(
                x, color="#f87171", linewidth=1.6, alpha=0.9, zorder=25
            )
        else:
            self._time_line.set_xdata([x, x])
            self._time_line.set_visible(True)
        self.ax.figure.canvas.draw_idle()

    def update_time_marker(self, current_dt: dt.datetime | None) -> None:
        self._marker_dt = current_dt
        self._refresh_time_marker()

# ============================================================
# 메인 대시보드
# ============================================================
class FleetOperationDashboard(QMainWindow):
    _COL_BG = "#0b1220"
    _COL_PANEL = "#0f172a"
    _COL_BORDER = "rgba(148,163,184,0.28)"
    _COL_TEXT = "#e2e8f0"
    _COL_SUBTLE = "#94a3b8"
    _COL_ACCENT = "#0ea5b7"
    _COL_ACCENT_HOV = "#14b8a6"
    _COL_ACCENT2 = "#60a5fa"

    _SPEED_STEPS = [1, 2, 5, 10]

    def __init__(self, plugin_name: str, plugin_port: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._plugin_name = plugin_name
        self._plugin_port = plugin_port
        self._exit_cb: Optional[Callable[[], None]] = None

        # --- Sim/AirSim 상태
        self._fpl_folder: Optional[Path] = None
        self._sim = None  # type: ignore
        self._airsim_client = None
        self._bridge = None
        self._running = False
        self._speed_idx = 0
        self._sim_seconds = 0.0
        self._vp_layout: Optional[dict] = None
        self._vp_layout_norm: Optional[dict] = None
        self._vp_logged_ids: set[str] = set()
        self._airsim_logged_ids: set[str] = set()
        self._monitoring_proc = None
        self._udp_socks: dict[tuple[str, int], socket.socket] = {}
        self._udp_interval_ms = 1000
        self._udp_last_tx = 0.0
        self._udp_config_path = DEFAULT_UDP_CONFIG_PATH
        self._udp_waiting_endpoints: dict[tuple[str, int], int] = {}
        self._udp_retry_timer = QTimer(self)
        self._udp_retry_timer.setSingleShot(True)
        self._udp_retry_timer.timeout.connect(self._retry_udp_config)
        self._udp_retry_attempts = 0
        self._udp_retry_delay_ms = 1200
        self._udp_last_packet: bytes | None = None
        # AirSim/Gantt 스냅샷 업데이트 주기를 약 60Hz(16ms)로 높여 위치 전송을 더 촘촘히 함
        self._sim_thread = SimulationThread(self, interval_ms=16)
        self._sim_thread.snapshot_ready.connect(self._on_sim_snapshot_ready)
        self._sim_thread.start()
        self._sim_thread.set_speed(float(self._SPEED_STEPS[self._speed_idx]))
        self._snapshot_worker = SnapshotProcessor(self)
        self._snapshot_worker.start()
        # --- Gantt 위젯 참조 (대시보드 런치 시 안전)
        self.scroll = None
        self.gantt_canvas = None
        # --- Gantt 데이터 캐시
        self._gantt_bundle: Optional[GanttBundle] = None

        # --- UI 기본
        self.setObjectName("FleetOpsWindow")
        self.setWindowTitle(f"Fleet Operations Center — {plugin_name} :{plugin_port}")
        self.resize(UiScale.dp(1625), UiScale.dp(900))

        central = QWidget(self); central.setObjectName("RootArea"); self.setCentralWidget(central)
        root_h = QHBoxLayout(central); root_h.setContentsMargins(UiScale.dp(16), UiScale.dp(12), UiScale.dp(16), UiScale.dp(12)); root_h.setSpacing(UiScale.dp(16))

        # ----- Left
        left = QWidget(central); left.setObjectName("MainColumn")
        left_v = QVBoxLayout(left); left_v.setContentsMargins(0, 0, 0, 0); left_v.setSpacing(UiScale.dp(10))

        # Header
        title_row = QHBoxLayout(); title_row.setSpacing(UiScale.dp(8))
        title_label = QLabel("Fleet Operation Center", left); title_label.setObjectName("TitleLabel")
        tfont = QFont(); tfont.setPointSize(int(16 * UiScale.scale)); tfont.setBold(True); title_label.setFont(tfont)

        self.btn_open = QPushButton("Open Schedules", left); self.btn_open.setObjectName("OpenSchedulesButton"); self.btn_open.setFixedHeight(UiScale.dp(30))
        self.btn_open.clicked.connect(self._on_open_schedules_clicked)
        self.btn_monitoring = QPushButton("Open Monitoring", left); self.btn_monitoring.setObjectName("OpenMonitoringButton"); self.btn_monitoring.setFixedHeight(UiScale.dp(30)); self.btn_monitoring.setEnabled(False)
        self.btn_monitoring.clicked.connect(self._on_open_monitoring_clicked)
        self.path_entry = QLineEdit(left); self.path_entry.setObjectName("PathEntry"); self.path_entry.setPlaceholderText("FPL folder ..."); self.path_entry.setReadOnly(True); self.path_entry.setFixedHeight(UiScale.dp(30))

        title_row.addWidget(title_label, 0); title_row.addSpacing(UiScale.dp(8)); title_row.addWidget(self.btn_open, 0); title_row.addSpacing(UiScale.dp(8)); title_row.addWidget(self.btn_monitoring, 0); title_row.addSpacing(UiScale.dp(8)); title_row.addWidget(self.path_entry, 1)

        # KPI
        kpi_row = QHBoxLayout(); kpi_row.setSpacing(UiScale.dp(10))
        def _kpi(title: str) -> QFrame:
            card = QFrame(left); card.setObjectName("KpiCard"); card.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); card.setMinimumHeight(UiScale.dp(66)); add_drop_shadow(card, radius=22, alpha=40)
            v = QVBoxLayout(card); v.setContentsMargins(UiScale.dp(10), UiScale.dp(8), UiScale.dp(10), UiScale.dp(8)); v.setSpacing(UiScale.dp(6))
            lbl = QLabel(title, card); lbl.setObjectName("KpiTitle"); sfont = QFont(); sfont.setPointSize(int(9 * UiScale.scale)); sfont.setBold(True); lbl.setFont(sfont)
            val = QLineEdit(card); val.setObjectName("KpiValue"); val.setReadOnly(True); val.setFixedHeight(UiScale.dp(28))
            v.addWidget(lbl); v.addWidget(val); return card
        self.kpi_op_time = _kpi("Operation Time")
        self.kpi_aircraft = _kpi("Expected Number of Operational Aircraft")
        self.kpi_pax = _kpi("Expected Number of Passengers")
        self.kpi_ports = _kpi("Number of Operational Vertiports")
        kpi_row.addWidget(self.kpi_op_time); kpi_row.addWidget(self.kpi_aircraft); kpi_row.addWidget(self.kpi_pax); kpi_row.addWidget(self.kpi_ports)

        hline1 = QFrame(left); hline1.setFrameShape(QFrame.HLine); hline1.setObjectName("Divider"); hline1.setFixedHeight(UiScale.dp(1))

        # Filter
        filter_row = QHBoxLayout(); filter_row.setSpacing(UiScale.dp(10))
        self.cbo_port = QComboBox(left); self.cbo_port.setObjectName("VertiportCombo"); self.cbo_port.addItem("(All)")
        
        # ▼ 아래 3줄 추가
        self.cbo_port.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cbo_port.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.cbo_port.setMinimumContentsLength(16)          # 긴 포트명 표시
        self.cbo_port.setMinimumWidth(UiScale.dp(240))      # 기본 폭 확보
                
        
        
        self.chk_aircraft = QCheckBox("Aircraft", left); self.chk_gate = QCheckBox("Gate", left); self.chk_fato = QCheckBox("FATO", left)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for btn in (self.chk_aircraft, self.chk_gate, self.chk_fato):
            self.mode_group.addButton(btn)
        self.chk_gate.setChecked(True)
        self.cbo_port.activated.connect(self._refresh_gantt)
        self.chk_aircraft.toggled.connect(self._on_mode_changed)
        self.chk_gate.toggled.connect(self._on_mode_changed)
        self.chk_fato.toggled.connect(self._on_mode_changed)

        filter_row.addWidget(QLabel("Selected Vertiport", left)); filter_row.addWidget(self.cbo_port, 0); filter_row.addSpacing(UiScale.dp(12))
        filter_row.addWidget(self.chk_aircraft, 0); filter_row.addWidget(self.chk_gate, 0); filter_row.addWidget(self.chk_fato, 0); filter_row.addStretch(1)

        # Chart (QScrollArea + Caption + GanttCanvas)
        chart_frame = QFrame(left); chart_frame.setObjectName("ChartFrame")
        chart_v = QVBoxLayout(chart_frame); chart_v.setContentsMargins(UiScale.dp(10), UiScale.dp(10), UiScale.dp(10), UiScale.dp(10))

        # 캡션 라벨 (오른쪽 정렬)
        caption_row = QHBoxLayout()
        self.chart_caption = QLabel("", chart_frame)
        self.chart_caption.setObjectName("ChartCaption")
        self.chart_caption.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        caption_row.addStretch(1)
        caption_row.addWidget(self.chart_caption)
        chart_v.addLayout(caption_row)

        # 스크롤 + 캔버스
        self.scroll = QScrollArea(chart_frame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("ChartScroll")
        self.scroll.setStyleSheet(
            "QScrollArea{background:transparent;} "
            "QScrollArea > QWidget > QWidget{background:#0f172a;}"
        )
        self.gantt_canvas = GanttCanvas(self.scroll)
        self.gantt_canvas.set_scroll_area(self.scroll)
        self.gantt_canvas.clear_blank()
        self.scroll.setWidget(self.gantt_canvas)

        chart_v.addWidget(self.scroll, 1)
        left_v.addLayout(title_row); left_v.addLayout(kpi_row); left_v.addWidget(hline1); left_v.addLayout(filter_row); left_v.addWidget(chart_frame, 1)

        # ----- Side
        side = QFrame(central); side.setObjectName("SidePanel"); side.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        side.setMinimumWidth(UiScale.dp(270)); side.setMaximumWidth(UiScale.dp(330))
        side_v = QVBoxLayout(side); side_v.setContentsMargins(UiScale.dp(12), UiScale.dp(12), UiScale.dp(12), UiScale.dp(12)); side_v.setSpacing(UiScale.dp(12))

        side_title = QLabel("Simulation Setting", side); sfont = QFont(); sfont.setPointSize(int(14 * UiScale.scale)); sfont.setBold(True); side_title.setFont(sfont)

        ctrl_layout = QGridLayout()
        ctrl_layout.setSpacing(UiScale.dp(10))
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        def make_icon_btn(text: str, icon: QIcon) -> QToolButton:
            b = QToolButton(side)
            b.setObjectName("CtrlButton")
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setText(text)
            b.setIcon(icon)
            b.setIconSize(QSize(UiScale.dp(36), UiScale.dp(36)))
            b.setMinimumSize(UiScale.dp(110), UiScale.dp(104))
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            font = QFont()
            font.setPointSize(int(14 * UiScale.scale))
            font.setBold(True)
            b.setFont(font)
            return b
        self.btn_play = make_icon_btn("Play", self._icon_play()); self.btn_play.clicked.connect(self._on_play)
        self.btn_speed = make_icon_btn("x1", self._icon_ff()); self.btn_speed.clicked.connect(self._on_speed)
        self.btn_pause = make_icon_btn("Pause", self._icon_pause()); self.btn_pause.clicked.connect(self._on_pause)
        self.btn_reset = make_icon_btn("Reset", self._icon_stop()); self.btn_reset.clicked.connect(self._on_reset)
        ctrl_layout.addWidget(self.btn_play, 0, 0)
        ctrl_layout.addWidget(self.btn_speed, 0, 1)
        ctrl_layout.addWidget(self.btn_pause, 1, 0)
        ctrl_layout.addWidget(self.btn_reset, 1, 1)
        ctrl_layout.setColumnStretch(0, 1)
        ctrl_layout.setColumnStretch(1, 1)
        ctrl_layout.setRowStretch(0, 1)
        ctrl_layout.setRowStretch(1, 1)

        time_lbl = QLabel("Sim Time", side)
        self.sim_time_edit = QLineEdit(side); self.sim_time_edit.setPlaceholderText("00:00:00"); self.sim_time_edit.setFixedHeight(UiScale.dp(30)); self.sim_time_edit.setReadOnly(True); self.sim_time_edit.setObjectName("SimTimeEdit")
        mono = QFont("Consolas"); self.sim_time_edit.setFont(mono)

        hline2 = QFrame(side); hline2.setFrameShape(QFrame.HLine); hline2.setObjectName("Divider")

        grid = QGridLayout(); grid.setSpacing(UiScale.dp(12))
        def side_btn(wide=False, text="BT 1"):
            b = QPushButton(text, side); b.setCheckable(False); b.setMinimumHeight(UiScale.dp(64 if not wide else 54))
            b.setObjectName("SideActionWideButton" if wide else "SideActionButton"); add_drop_shadow(b, radius=28 if not wide else 22, alpha=60); return b
        grid.addWidget(side_btn(False, "BT 1"), 0, 0); grid.addWidget(side_btn(False, "BT 1"), 0, 1)
        grid.addWidget(side_btn(False, "BT 1"), 1, 0); grid.addWidget(side_btn(False, "BT 1"), 1, 1)
        side_v.addWidget(side_title)
        side_v.addLayout(ctrl_layout)
        side_v.addWidget(time_lbl)
        side_v.addWidget(self.sim_time_edit)
        side_v.addWidget(hline2)
        side_v.addLayout(grid)
        side_v.addWidget(side_btn(True, "BT 1"))
        side_v.addWidget(side_btn(True, "BT 1"))
        side_v.addStretch(1)

        root_h.addWidget(left, 1); root_h.addWidget(side)
        self._load_udp_config()
        self._apply_styles()

        # 타이머

    def _resolve_airsim_endpoint(self) -> tuple[str, int]:
        """Resolve target IP/port for AirSim RPC using current host configuration."""
        env_ip = os.environ.get("FLEETOPS_AIRSIM_IP", "").strip()
        env_port = os.environ.get("FLEETOPS_AIRSIM_PORT", "").strip()

        def _safe_ip_candidate() -> str:
            # Prefer outbound interface; fallback to hostname resolution; default to loopback.
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    candidate = s.getsockname()[0]
                    if candidate and not candidate.startswith("127."):
                        return candidate
            except Exception:
                pass
            try:
                candidate = socket.gethostbyname(socket.gethostname())
                if candidate and not candidate.startswith("127."):
                    return candidate
            except Exception:
                pass
            return "127.0.0.1"

        ip = env_ip or _safe_ip_candidate()
        try:
            port = int(env_port) if env_port else 41451
        except ValueError:
            port = 41451
        return ip, port

    def _log(self, message: str) -> None:
        try:
            print(f"[FleetOps] {message}")
        except Exception:
            pass

    def _close_udp_sockets(self) -> None:
        if not self._udp_socks:
            return
        for sock in list(self._udp_socks.values()):
            try:
                sock.close()
            except Exception:
                pass
        self._udp_socks.clear()

    def _load_udp_config(self, *, silent: bool = False) -> None:
        path = self._udp_config_path
        cfg: dict | None = None
        if path:
            try:
                path = Path(path)
                if path.is_file():
                    with path.open("r", encoding="utf-8-sig") as f:
                        cfg = json.load(f)
                    if not silent:
                        self._log(f"[udp] config loaded: {path}")
                else:
                    self._log(f"[udp] config not found: {path}")
            except Exception as exc:
                self._log(f"[udp] failed to load config ({exc})")
        if not isinstance(cfg, dict):
            self._close_udp_sockets()
            return
        self._apply_udp_config(cfg)

    def _apply_udp_config(self, cfg: dict) -> None:
        self._udp_interval_ms = max(50, int(cfg.get("tx_interval_ms", 1000)))
        self._close_udp_sockets()

        endpoints = cfg.get("tx_endpoints", [])
        if not isinstance(endpoints, (list, tuple)) or not endpoints:
            self._log("[udp] no tx_endpoints configured")
            return

        opened = 0
        for entry in endpoints:
            host = None
            port = None
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                host, port = entry[0], entry[1]
            elif isinstance(entry, dict):
                host = entry.get("host") or entry.get("ip")
                port = entry.get("port")
            if not host or port is None:
                continue
            try:
                ip = str(host)
                port_i = int(port)
            except Exception:
                continue
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(1.0)
                sock.connect((ip, port_i))
                try:
                    sock.send(b"PING")
                    resp = sock.recv(32)
                    if resp.strip().upper() != b"PONG":
                        raise TimeoutError("unexpected reply")
                except Exception as exc:
                    wait_key = (ip, port_i)
                    is_reset = isinstance(exc, ConnectionResetError) or getattr(exc, "errno", None) == errno.ECONNRESET or getattr(exc, "winerror", None) == 10054
                    if is_reset:
                        if wait_key not in self._udp_waiting_endpoints:
                            self._log(f"[udp] waiting for monitoring listener at {ip}:{port_i}")
                        self._udp_waiting_endpoints[wait_key] = self._udp_waiting_endpoints.get(wait_key, 0) + 1
                    else:
                        self._log(f"[udp] {ip}:{port_i} handshake failed ({exc})")
                    sock.close()
                    continue
                sock.settimeout(None)
                self._udp_socks[(ip, port_i)] = sock
                self._udp_waiting_endpoints.pop((ip, port_i), None)
                opened += 1
                self._log(f"[udp] ready {ip}:{port_i}")
            except Exception as exc:
                self._log(f"[udp] failed to open {host}:{port} ({exc})")
        if opened == 0:
            if not self._udp_waiting_endpoints:
                self._log("[udp] no endpoints available")
            if self._udp_retry_attempts == 0 and not self._udp_retry_timer.isActive():
                self._schedule_udp_retry()
        self._udp_last_tx = 0.0

    def _schedule_udp_retry(self, attempts: int = 5, delay_ms: int = 1200) -> None:
        if self._udp_socks:
            return
        attempts = max(0, attempts)
        if attempts <= 0:
            return
        self._udp_retry_attempts = max(self._udp_retry_attempts, attempts)
        self._udp_retry_delay_ms = max(200, delay_ms)
        if not self._udp_retry_timer.isActive():
            self._udp_retry_timer.start(self._udp_retry_delay_ms)

    def _retry_udp_config(self) -> None:
        if self._udp_retry_attempts <= 0:
            return
        self._udp_retry_attempts -= 1
        self._load_udp_config(silent=True)
        if self._udp_socks:
            if self._udp_last_packet:
                self._send_udp_payload(self._udp_last_packet)
            self._udp_retry_attempts = 0
            return
        if self._udp_retry_attempts > 0:
            self._udp_retry_timer.start(self._udp_retry_delay_ms)

    def _send_udp_payload(self, data: bytes) -> None:
        if not self._udp_socks:
            return
        dead: list[tuple[str, int]] = []
        for key, sock in self._udp_socks.items():
            try:
                sock.send(data)
            except Exception:
                dead.append(key)
        for key in dead:
            sock = self._udp_socks.pop(key, None)
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
                self._log(f"[udp] disconnected {key[0]}:{key[1]}")

    def _maybe_send_udp(self, snapshot: dict[str, dict] | None) -> None:
        if not snapshot:
            return
        interval = max(0.05, self._udp_interval_ms / 1000.0)
        now = time.monotonic()
        if now - self._udp_last_tx < interval:
            return
        self._udp_last_tx = now
        sim_dt = getattr(self._sim, "sim_time", None)
        if sim_dt is not None and hasattr(sim_dt, "strftime"):
            time_label = sim_dt.strftime("%H:%M:%S")
        else:
            time_label = time.strftime("%H:%M:%S")
        payload = {"time": time_label, "fleet": snapshot}
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except Exception as exc:
            self._log(f"[udp] serialization failed ({exc})")
            return
        self._udp_last_packet = data
        if not self._udp_socks:
            if self._udp_retry_attempts == 0 and not self._udp_retry_timer.isActive():
                self._schedule_udp_retry()
            return
        self._send_udp_payload(data)

    def _kpi_set(self, card: QFrame | None, value: str | int | None) -> None:
        if card is None:
            return
        editor = card.findChild(QLineEdit, "KpiValue")
        if editor is None:
            editors = card.findChildren(QLineEdit)
            editor = editors[0] if editors else None
        if editor is not None:
            text = "-" if value is None else str(value)
            editor.setText(text)

    def _refresh_port_filter(self) -> None:
        if not hasattr(self, "cbo_port") or self.cbo_port is None:
            return
        current = self.cbo_port.currentText().strip() if self.cbo_port.currentText() else ""
        ports = sorted(self._gantt_bundle.ports) if self._gantt_bundle else []
        self.cbo_port.blockSignals(True)
        self.cbo_port.clear()
        for port in ports:
            self.cbo_port.addItem(port)
        idx = self.cbo_port.findText(current) if current else -1
        if idx < 0:
            idx = 0 if ports else -1
        self.cbo_port.setCurrentIndex(idx)
        self.cbo_port.blockSignals(False)

    def _rows_from_bundle(self, bundle: Optional[GanttBundle]):
        if bundle is None:
            now = dt.datetime.combine(dt.date.today(), dt.time(0, 0, 0))
            return "", now, [], 60.0

        base0 = bundle.base0
        time_span = max(1.0, float(bundle.time_span_min or 0.0))
        ports = sorted(bundle.ports)

        port_sel = None
        if hasattr(self, "cbo_port") and self.cbo_port:
            current = self.cbo_port.currentText()
            if current:
                port_sel = current.strip()
        if not port_sel:
            port_sel = ports[0] if ports else None
        if not port_sel:
            return "", base0, [], time_span

        show_gate = bool(getattr(self, "chk_gate", None) and self.chk_gate.isChecked())
        show_fato = bool(getattr(self, "chk_fato", None) and self.chk_fato.isChecked())
        show_ac = bool(getattr(self, "chk_aircraft", None) and self.chk_aircraft.isChecked())
        if not any([show_gate, show_fato, show_ac]):
            show_gate = True

        rows: list[tuple] = []
        title = ""

        def _extend_rows(prefix: str, bars: list[Bar]) -> None:
            for bar in bars:
                row_key = f"{prefix}{bar.row_key}"
                rows.append((row_key, float(bar.start_min), float(bar.end_min), bar.label, bar.meta or {}))

        if show_gate:
            bars = bundle.gate_bars_by_port.get(port_sel, [])
            _extend_rows("", bars)
            title = f"Gate Utilization - {port_sel}"
        elif show_fato:
            bars = bundle.fato_bars_by_port.get(port_sel, [])
            _extend_rows("", bars)
            title = f"FATO Utilization - {port_sel}"
        else:
            per_port = bundle.ac_bars_by_port.get(port_sel, {})
            for acid in sorted(per_port.keys()):
                bars = per_port.get(acid, [])
                for bar in bars:
                    rows.append((acid, float(bar.start_min), float(bar.end_min), bar.label, bar.meta or {}))
            title = f"Aircraft Activity - {port_sel}"

        return title, base0, rows, time_span

    def _refresh_gantt(self, *_args) -> None:
        if not getattr(self, "gantt_canvas", None):
            return
        if not self._gantt_bundle:
            self.gantt_canvas.clear_blank()
            if hasattr(self, "chart_caption") and self.chart_caption:
                self.chart_caption.setText("")
            return

        title, base0, rows, span = self._rows_from_bundle(self._gantt_bundle)
        self.gantt_canvas.draw_bars(title, base0, rows, span)
        if hasattr(self, "chart_caption") and self.chart_caption:
            self.chart_caption.setText(title)

    def _on_mode_changed(self, checked: bool) -> None:
        if checked:
            self._refresh_gantt()

    def _on_open_schedules_clicked(self) -> None:
        try:
            project_root = Path(__file__).resolve().parents[2]
            default_dir = self._fpl_folder if self._fpl_folder else project_root / "plugins" / "Scheduler" / "FPL_Result"
            default_dir_str = str(default_dir if default_dir else "")
            fname, _ = QFileDialog.getOpenFileName(
                self,
                "Open CSV (pick one in the date folder)",
                default_dir_str,
                "CSV Files (*.csv);;All Files (*.*)",
            )
            resolved = resolve_fpl_container_path(fname) if fname else resolve_fpl_container_path(None)
            if not resolved:
                self._log("No schedule folder selected.")
                return

            self._running = False
            self._sim_thread.stop_loop()
            self._sim_thread.clear_sim()
            if hasattr(self, '_snapshot_worker') and self._snapshot_worker is not None:
                self._snapshot_worker.flush()
            self._sim_seconds = 0.0
            self._sim = None  # type: ignore
            self._airsim_client = None
            self._bridge = None
            self._vp_layout = None
            self._vp_layout_norm = None

            self._fpl_folder = Path(resolved)
            if hasattr(self, "path_entry") and self.path_entry:
                self.path_entry.setText(str(self._fpl_folder))

            stats = read_fpl_stats_from_path(self._fpl_folder)
            if stats:
                # ★★★ 여기 두 줄 추가 ★★★
                self._op_start_dt = stats.op_start
                self._op_end_dt   = stats.op_end
                # ★★★★★★★★★★★★★★★★★★★★
                if stats.op_start and stats.op_end:
                    fmt = "%H:%M:%S" if (stats.op_start.second or stats.op_end.second) else "%H:%M"
                    op_window = f"{stats.op_start.strftime(fmt)} ~ {stats.op_end.strftime(fmt)}"
                else:
                    op_window = "-"
                self._kpi_set(self.kpi_op_time, op_window)
                self._kpi_set(self.kpi_pax, f"{stats.pax_total:,}")
                self._kpi_set(self.kpi_ports, str(stats.ports_count))
                self._kpi_set(self.kpi_aircraft, str(stats.aircraft_estimate))
            
            else:
                for card in (self.kpi_op_time, self.kpi_pax, self.kpi_ports, self.kpi_aircraft):
                    self._kpi_set(card, "-")

            self._gantt_bundle = build_gantt_from_folder(self._fpl_folder)
            self._refresh_port_filter()
            self._refresh_gantt()
            self._sync_sim_time_text()
            if self.btn_monitoring:
                self.btn_monitoring.setEnabled(True)
        except Exception as exc:
            self._log(f"[error] failed to open schedules: {exc}")

    def _cleanup_monitoring_proc(self) -> None:
        proc = getattr(self, "_monitoring_proc", None)
        if proc is not None and proc.poll() is not None:
            self._monitoring_proc = None

    def _stop_monitoring_proc(self) -> None:
        proc = getattr(self, "_monitoring_proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        finally:
            self._monitoring_proc = None

    def _on_open_monitoring_clicked(self) -> None:
        try:
            self._cleanup_monitoring_proc()
            if self._monitoring_proc is not None:
                self._log("Monitoring UI already running.")
                return

            script_path = Path(__file__).resolve().parents[1] / "Monitoring" / "main.py"
            if not script_path.exists():
                self._log(f"[warn] monitoring entry not found: {script_path}")
                return

            env = os.environ.copy()
            fpl_dir = getattr(self, "_fpl_folder", None)
            if fpl_dir:
                try:
                    fpl_path = Path(fpl_dir)
                    if fpl_path.is_file():
                        fpl_path = fpl_path.parent
                    if fpl_path.exists():
                        env["FLEETOPS_FPL_DIR"] = str(fpl_path.resolve())
                    else:
                        env.pop("FLEETOPS_FPL_DIR", None)
                        self._log(f"[warn] selected schedule folder no longer exists: {fpl_path}")
                except Exception:
                    env.pop("FLEETOPS_FPL_DIR", None)
            else:
                env.pop("FLEETOPS_FPL_DIR", None)
                self._log("Launching monitoring without a selected schedule folder.")

            cwd_base = script_path.parents[2] if len(script_path.parents) >= 3 else script_path.parent
            self._monitoring_proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(cwd_base),
                env=env,
            )
            self._log("Monitoring UI launched.")
            self._schedule_udp_retry(attempts=8, delay_ms=1000)
        except Exception as exc:
            self._monitoring_proc = None
            self._log(f"[error] failed to launch monitoring: {exc}")

    def set_connection_info(self, *, plugin_name: str, plugin_port: int) -> None:
        self._plugin_name = plugin_name
        self._plugin_port = plugin_port
        self.setWindowTitle(f"Fleet Operations Center - {plugin_name} :{plugin_port}")

    def set_exit_callback(self, callback: Callable[[], None]) -> None:
        self._exit_cb = callback

    def _lazy_load_vp_resources(self) -> None:
        if getattr(self, "_vp_layout", None) is not None:
            return

        self._vp_layout = {}
        self._vp_layout_norm = {}
        try:
            import csv
            import unicodedata

            candidates = []
            if self._fpl_folder:
                candidates.extend([
                    self._fpl_folder / "resources_vp.csv",
                    self._fpl_folder / "resource" / "resources_vp.csv",
                    self._fpl_folder.parent / "resources_vp.csv",
                ])
            sitl_root = Path(__file__).resolve().parents[2] / "plugins" / "SITL"
            candidates.extend([
                Path(__file__).resolve().parent / "Sources" / "resources_vp.csv",
                Path(__file__).resolve().parent / "resources_vp.csv",
                sitl_root / "resource" / "resources_vp.csv",
                sitl_root / "resources_vp.csv",
            ])

            csv_path = next((p for p in candidates if p is not None and p.exists()), None)
            if csv_path is None:
                self._log("resources_vp.csv not found for spawn layout lookup.")
                return

            def _numeric(row, *keys):
                for key in keys:
                    value = row.get(key)
                    if value is None:
                        continue
                    s = str(value).strip()
                    if not s:
                        continue
                    try:
                        return float(s)
                    except Exception:
                        continue
                return None

            def _normalize_port(name: str) -> str:
                name = unicodedata.normalize("NFKC", str(name or ""))
                name = name.replace(" ", " ").strip()
                name = re.sub(r"\s+", "", name)
                name = name.replace("-", "").replace("-", "").replace("_", "").replace(".", "")
                return name

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    port = (row.get("Vertiport") or row.get("PortName") or row.get("port_name_raw") or row.get("port_name") or row.get("VertiPort") or row.get("Name") or "").strip()
                    label = (row.get("Label") or row.get("label") or "").strip().upper()
                    if not port or not label:
                        continue

                    x_m = _numeric(row, "X_m", "x_m", "X_M")
                    y_m = _numeric(row, "Y_m", "y_m", "Y_M")
                    z_m = _numeric(row, "Z_m", "z_m", "Z_M")
                    if x_m is None or y_m is None or z_m is None:
                        x_cm = _numeric(row, "X_cm", "x_cm", "U_X", "U_Xcm", "U_XCM")
                        y_cm = _numeric(row, "Y_cm", "y_cm", "U_Y", "U_Ycm", "U_YCM")
                        z_cm = _numeric(row, "Z_cm", "z_cm", "U_Z", "U_Zcm", "U_ZCM")
                        if x_cm is None or y_cm is None or z_cm is None:
                            continue
                        x_m, y_m, z_m = x_cm / 100.0, y_cm / 100.0, z_cm / 100.0

                    yaw_deg = _numeric(row, "yaw_deg", "Yaw", "YAW", "angle", "Angle") or 0.0

                    rec = {
                        "UE_X_m": x_m,
                        "UE_Y_m": y_m,
                        "UE_Z_m": z_m,
                        "N_m": x_m,
                        "E_m": y_m,
                        "D_m": -z_m,
                        "yaw_deg": yaw_deg,
                    }
                    self._vp_layout[(port, label)] = rec
                    self._vp_layout_norm[(_normalize_port(port), label)] = rec
        except Exception as exc:
            self._log(f"[resources] failed to load resources_vp.csv: {exc}")
            self._vp_layout = {}
            self._vp_layout_norm = {}

    def _ensure_prespawn_records(self, snap_for_airsim: dict) -> dict:
        try:
            if self._sim is None or getattr(self._sim, "sim_time", None) is None:
                return snap_for_airsim

            self._lazy_load_vp_resources()
            if not self._vp_layout:
                return snap_for_airsim

            now = self._sim.sim_time
            out = dict(snap_for_airsim)
            pending = getattr(self._sim, "_pending", []) or []
            for fp in list(pending):
                std = getattr(fp, "std", None)
                if std is None:
                    continue
                try:
                    dt_sec = (std - now).total_seconds()
                except Exception:
                    continue
                if not (0.0 < dt_sec <= 30.0 + 1e-6):
                    continue

                acid = getattr(fp, "id", None)
                if not acid or acid in out:
                    continue
                port = getattr(fp, "origin", None)
                gate_no = getattr(fp, "dep_gate_no", None)
                fato_no = getattr(fp, "dep_fato_no", None)

                label = None
                if gate_no:
                    label = f"GATE {gate_no}".upper()
                elif fato_no:
                    label = f"FATO {fato_no}".upper()
                if not port or not label:
                    continue

                rec = self._vp_layout.get((port, label))
                if rec is None and self._vp_layout_norm:
                    import unicodedata

                    def _normalize_port(name: str) -> str:
                        name = unicodedata.normalize("NFKC", str(name or ""))
                        name = name.replace(" ", " ").strip()
                        name = re.sub(r"\s+", "", name)
                        name = name.replace("-", "").replace("-", "").replace("_", "").replace(".", "")
                        return name

                    rec = self._vp_layout_norm.get((_normalize_port(port), label))
                if rec is None:
                    self._log(f"[prespawn] missing layout entry for {port} / {label}")
                    continue

                x = float(rec.get("N_m", 0.0))
                y = float(rec.get("E_m", 0.0))
                z = float(rec.get("D_m", 0.0))
                if acid not in self._vp_logged_ids:
                    self._vp_logged_ids.add(acid)
                    self._log(f"[prespawn] {acid} port={port} label={label} N={x:.3f} E={y:.3f} D={z:.3f}")

                out[acid] = {
                    "id": acid,
                    "lat": float('nan'),
                    "lon": float('nan'),
                    "alt_m": float('nan'),
                    "x": x,
                    "y": y,
                    "z": z,
                    "heading_deg": float(rec.get("yaw_deg", 0.0)),
                    "progress_m": 0.0,
                    "remain_m": 0.0,
                    "phase": "SPAWN",
                    "lane": None,
                    "pax": getattr(fp, "pax", 0),
                    "local_id": getattr(fp, "local_id", ""),
                    "atd": "-",
                    "eta": "-",
                    "From": getattr(fp, "origin", None),
                    "To": getattr(fp, "dest", None),
                    "DepFATO_No": getattr(fp, "dep_fato_no", None),
                    "DepGate_No": getattr(fp, "dep_gate_no", None),
                }
            return out
        except Exception as exc:
            self._log(f"[prespawn] error: {exc}")
            return snap_for_airsim

    def _ensure_sim_ready(self) -> bool:
        if not self._fpl_folder:
            return False
        # SITL
        if self._sim is None and SitlSim is not None:
            self._sim = SitlSim(str(self._fpl_folder), dt_sim=1.0)
            self._sim_thread.set_sim(self._sim)
        # AirSim
        if self._airsim_client is None and airsim is not None:
            try:
                ip, port = self._resolve_airsim_endpoint()
                self._log(f"Connecting to AirSim RPC at {ip}:{port}")
                self._airsim_client = airsim.VehicleClient(ip=ip, port=port)
                self._airsim_client.confirmConnection()
            except Exception:
                self._log("AirSim RPC connection failed; keeping SITL-only mode.")
                self._airsim_client = None
        # Bridge
        # Bridge
        if self._bridge is None and self._airsim_client is not None and AirSimFleetBridge is not None:
            # resources_vp.csv 후보 경로
            candidates = [
                self._fpl_folder / "resources_vp.csv" if self._fpl_folder else None,
                (self._fpl_folder.parent / "resources_vp.csv") if self._fpl_folder else None,
                Path(__file__).resolve().parents[2] / "plugins" / "SITL" / "resources_vp.csv",
                Path(__file__).resolve().parent / "Sources" / "resources_vp.csv",
                Path(r"C:\Users\Junyong\ODT_PF\plugins\FleetOperations\Sources\resources_vp.csv"),
            ]
            candidates = [p for p in candidates if p is not None]
            gate_csv = next((p for p in candidates if p.exists()), None)

            # 선택적으로 생성 (없어도 동작하도록)
            try:
                conv = LLAConverter() if LLAConverter else None
            except Exception:
                conv = None
            try:
                spawner = GateSpawnResolver(gate_csv=gate_csv) if GateSpawnResolver else None
            except Exception:
                spawner = None

            # 생성자 시그니처 2종 모두 시도
            try:
                self._bridge = AirSimFleetBridge(self._airsim_client, lla_conv=conv, spawner=spawner)
            except TypeError:
                self._bridge = AirSimFleetBridge(self._airsim_client)

        if self._sim is None:
            return False
        if self._bridge is None:
            self._log("AirSim bridge unavailable; running SITL only.")
        return True

    def _on_play(self) -> None:
        if not self._ensure_sim_ready():
            return

        already_started = getattr(self._sim, "sim_time", None) is not None
        speed = float(self._SPEED_STEPS[self._speed_idx])
        self._sim_thread.set_speed(speed)

        if already_started:
            if hasattr(self._sim, "sim_speed"):
                self._sim.sim_speed = speed
            if hasattr(self._sim, "running"):
                self._sim.running = True
            self._running = True
            self._sim_thread.start_loop()
            return

        # 1) 운영 시작 시간 (기본값: self._op_start_dt 또는 KPI 값 -> 기본 06:30)
        start_raw = "06:30"
        try:
            if getattr(self, "_op_start_dt", None):
                start_raw = self._op_start_dt.strftime("%H:%M:%S")
            else:
                kv = self.kpi_op_time.findChild(QLineEdit, "KpiValue") if self.kpi_op_time else None
                raw = (kv.text().split("~")[0].strip() if kv and kv.text() else "")
                if raw:
                    start_raw = raw
        except Exception:
            pass

        # 2) "HH:MM[:SS]" -> ("HH:MM", sec_offset)
        import re
        hhmm = "06:30"
        sec_offset = 0
        m = re.match(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$", start_raw)
        if m:
            h = int(m.group(1)); mm = int(m.group(2)); s = int(m.group(3)) if m.group(3) else 0
            hhmm = f"{h:02d}:{mm:02d}"
            sec_offset = s

        # 3) SITL 시작 (SITL은 HH:MM만 받음)
        if hasattr(self._sim, "start"):
            try:
                self._sim.start(hhmm, sim_speed=speed)
            except TypeError:
                self._sim.start(hhmm)

        # 4) 초 단위 오프셋이 있으면, 정확한 HH:MM:SS로 맞추기
        if sec_offset > 0:
            try:
                eff_speed = float(getattr(self._sim, "sim_speed", speed) or speed)
                self._sim.step(sec_offset / max(1e-6, eff_speed))
            except Exception:
                pass

        # 5) 타이머 가동
        self._running = True
        self._sim_thread.start_loop()


    def _on_pause(self) -> None:
        self._running = False
        self._sim_thread.stop_loop()
        if hasattr(self, '_snapshot_worker') and self._snapshot_worker is not None:
            self._snapshot_worker.flush()
        if self._sim and hasattr(self._sim, "stop"):
            try: self._sim.stop()
            except Exception: pass

    def _on_reset(self) -> None:
        self._running = False
        self._sim_thread.stop_loop()
        if hasattr(self, '_snapshot_worker') and self._snapshot_worker is not None:
            self._snapshot_worker.flush()
        self._sim_seconds = 0.0
        self._sync_sim_time_text()
        self._sim_thread.clear_sim()
        if self._bridge:
            try:
                if hasattr(self._bridge, "reset"):
                    self._bridge.reset()                         # ← 전체 리셋(있으면)
                elif hasattr(self._bridge, "prune_actors"):
                    self._bridge.prune_actors(set())             # ← 폴백
            except Exception:
                pass
        self._sim = None

    def _on_speed(self) -> None:
        self._speed_idx = (self._speed_idx + 1) % len(self._SPEED_STEPS)
        new_speed = float(self._SPEED_STEPS[self._speed_idx])
        self.btn_speed.setText(f"x{self._SPEED_STEPS[self._speed_idx]}")
        self._sim_thread.set_speed(new_speed)

    def _snapshot_for_airsim(self, snap: dict) -> dict:
        """
        SITL snapshot(dict) → AirSim용 snapshot(dict)
        - lat/lon + '절대고도'로 변환해 (x,y,z)=N,E,D [m]을 채워준다.
        - 게이트/패드/Lane 고도 규칙은 sitl_main.py와 동일.
        """
        if not snap:
            return {}

        # 동적 import (프로젝트 경로 차이에 대응)
        try:
            from SITL.sitl_coord_transform import wgs84_to_airsim_ned, load_resources_vp, lookup_vp_label_alt_m
        except Exception:
            try:
                from plugins.SITL.sitl_coord_transform import wgs84_to_airsim_ned, load_resources_vp, lookup_vp_label_alt_m  # type: ignore
            except Exception:
                # 마지막 폴백 (동일 모듈 루트에 있는 경우)
                from sitl_coord_transform import wgs84_to_airsim_ned, load_resources_vp, lookup_vp_label_alt_m  # type: ignore

        # resources_vp.csv 1회 로딩 (게이트/패드 기준 Z[m] 조회용)
        try:
            load_resources_vp()
        except Exception:
            pass

        def _lane_alt_m(lane: str) -> float | None:
            if not lane or lane == "-":
                return None
            m = re.search(r"(\d+)", str(lane))
            return (float(m.group(1)) * 0.3048) if m else None  # 1000 ft → 304.8 m


        out: dict = {}
        for acid, pkt in snap.items():
            q = dict(pkt)  # 원본 보존
            try:
                lat = float(pkt.get("lat"))
                lon = float(pkt.get("lon"))
            except Exception:
                # LLA 없으면 변환 불가 → 있는 값 그대로 사용
                out[acid] = q
                continue

            # 시뮬 내부 AGL(상대) 고도
            z_agl_raw = float(pkt.get("alt_m", 0.0))
            profile_alt_m = z_agl_raw
            phase = str(pkt.get("phase", "")).upper()[:1]

            dep = str(pkt.get("From", "") or "")
            arr = str(pkt.get("To", "") or "")
            dep_gate = str(pkt.get("DepGate_No", "") or "")
            dep_fato = str(pkt.get("DepFATO_No", "") or "")
            arr_gate = str(pkt.get("ArrGate_No", "") or "")
            arr_fato = str(pkt.get("ArrFATO_No", "") or "")
            lane_alt = _lane_alt_m(str(pkt.get("lane", "")))

            # 게이트/패드 기준 고도 Z[m] 조회 (없으면 None)
            def _nz(*vals):
                for v in vals:
                    if v is not None:
                        return float(v)
                return 0.0

            try:
                dep_gate_alt = lookup_vp_label_alt_m(dep, f"GATE {dep_gate}") if dep_gate else None
                dep_fato_alt = lookup_vp_label_alt_m(dep, f"FATO {dep_fato}") if dep_fato else None
                arr_gate_alt = lookup_vp_label_alt_m(arr, f"GATE {arr_gate}") if arr_gate else None
                arr_fato_alt = lookup_vp_label_alt_m(arr, f"FATO {arr_fato}") if arr_fato else None
            except Exception:
                dep_gate_alt = dep_fato_alt = arr_gate_alt = arr_fato_alt = None

            # === sitl_main.py 의 고도 규칙 복제 ===
            # A: Gate taxi / B,C,D: Dep 상승 / E: 회랑 진입 / F: 크루즈(절대)
            # G,H,I: Arr 하강 / J: 수직착륙 / K: FATO→Gate taxi / 그 외: dep 기준 + z_agl
            dep_base = _nz(dep_fato_alt, dep_gate_alt)
            arr_base = _nz(arr_fato_alt, arr_gate_alt)

            if phase == "A":          # Gate taxi
                alt_abs = dep_base
                alt_agl = 0.0
            elif phase in ("B", "C", "D"):
                alt_abs = dep_base + profile_alt_m
                alt_agl = profile_alt_m
            elif phase == "E":        # 회랑 진입
                if lane_alt is not None:
                    alt_abs = lane_alt
                    alt_agl = max(0.0, lane_alt - dep_base)
                else:
                    alt_abs = profile_alt_m
                    alt_agl = profile_alt_m
            elif phase == "F":        # 크루즈(절대)
                alt_abs = lane_alt if lane_alt is not None else profile_alt_m
                alt_agl = max(0.0, alt_abs - dep_base)
            elif phase in ("G", "H", "I"):
                alt_abs = arr_base + profile_alt_m
                alt_agl = profile_alt_m
            elif phase == "J":        # 수직착륙
                alt_abs = arr_base
                alt_agl = 0.0
            elif phase == "K":        # FATO→Gate taxi
                alt_abs = arr_base
                alt_agl = 0.0
            else:
                alt_abs = dep_base + profile_alt_m
                alt_agl = profile_alt_m
                
            # --- (추가) A/J/K 구간에서 XY를 게이트/패드 중심으로 스냅 ---
            snap_xy = None
            snap_d  = None
            try:
                # resources_vp 로드(1회)
                if hasattr(self, "_lazy_load_vp_resources"):
                    self._lazy_load_vp_resources()

                rec_dep_gate = None
                rec_dep_fato = None
                rec_arr_gate = None
                rec_arr_fato = None

                if getattr(self, "_vp_layout", None):
                    if dep and dep_gate:
                        rec_dep_gate = self._vp_layout.get((dep, f"GATE {dep_gate}".upper()))
                    if dep and dep_fato:
                        rec_dep_fato = self._vp_layout.get((dep, f"FATO {dep_fato}".upper()))
                    if arr and arr_gate:
                        rec_arr_gate = self._vp_layout.get((arr, f"GATE {arr_gate}".upper()))
                    if arr and arr_fato:
                        rec_arr_fato = self._vp_layout.get((arr, f"FATO {arr_fato}".upper()))

                def _xy_from(rec):
                    if rec is None: return (None, None)
                    return (float(rec.get("N_m")), float(rec.get("E_m")))

                if phase == "A":  # Gate taxi 시작 → 출발 게이트 중심
                    nx, ex = _xy_from(rec_dep_gate)
                    if nx is not None and ex is not None:
                        snap_xy = (nx, ex)
                        snap_d  = -float(dep_base)
                elif phase == "K":  # FATO→Gate taxi 종료 → 도착 게이트 중심
                    nx, ex = _xy_from(rec_arr_gate)
                    if nx is not None and ex is not None:
                        snap_xy = (nx, ex)
                        snap_d  = -float(arr_base)
                elif phase == "J":  # 수직착륙 → 도착 FATO 중심
                    nx, ex = _xy_from(rec_arr_fato)
                    if nx is not None and ex is not None:
                        snap_xy = (nx, ex)
                        snap_d  = -float(arr_base)
            except Exception:
                snap_xy = None
                snap_d  = None   

            n_m, e_m, _ = wgs84_to_airsim_ned(lat, lon, alt_abs)
            d_m = -float(alt_abs)

            if snap_xy is not None:
                n_m, e_m = snap_xy
            if snap_d is not None:
                d_m = snap_d
                

            q["x"], q["y"], q["z"] = n_m, e_m, d_m
            
            out[acid] = q

        return out


    def _on_sim_snapshot_ready(self, raw_snap_obj, snap_obj, speed_val, now_t, real_dt) -> None:
        if not (self._running and self._sim):
            return

        try:
            speed = float(speed_val)
        except Exception:
            speed = float(self._SPEED_STEPS[self._speed_idx])

        try:
            delta = max(0.0, float(real_dt))
        except Exception:
            delta = 0.0

        raw_snap = raw_snap_obj if isinstance(raw_snap_obj, dict) else None
        snap = snap_obj if isinstance(snap_obj, dict) else None
        payload = raw_snap if isinstance(raw_snap, dict) and raw_snap else snap

        if isinstance(payload, dict) and payload:
            self._maybe_send_udp(payload)

        self._sim_seconds += delta * speed
        self._sync_sim_time_text()

        if self._bridge is None or not isinstance(payload, dict) or not payload:
            return

        try:
            now_val = float(now_t)
        except Exception:
            now_val = time.perf_counter()

        if hasattr(self, '_snapshot_worker') and self._snapshot_worker is not None:
            self._snapshot_worker.enqueue(raw_snap, snap, speed, now_val)


    def _process_snapshot_async(self, raw_snap_obj, snap_obj, speed_val, now_val) -> None:
        if not self._running or self._bridge is None or self._sim is None:
            return

        raw_snap = raw_snap_obj if isinstance(raw_snap_obj, dict) else None
        snap = snap_obj if isinstance(snap_obj, dict) else None

        if not isinstance(snap, dict) or not snap:
            if isinstance(raw_snap, dict) and raw_snap:
                try:
                    snap = self._snapshot_for_airsim(raw_snap)
                except Exception:
                    return
            else:
                return
        elif isinstance(raw_snap, dict) and raw_snap is snap:
            try:
                snap = self._snapshot_for_airsim(raw_snap)
            except Exception:
                pass

        try:
            now_float = float(now_val)
        except Exception:
            now_float = time.perf_counter()

        used_sync = False
        if hasattr(self._bridge, "sync_snapshot"):
            try:
                self._bridge.sync_snapshot(snap, now_t=now_float, use_bank=False)
                used_sync = True
            except Exception:
                used_sync = False

        if not used_sync:
            for acid, pkt in snap.items():
                x = float(pkt.get("x", 0.0))
                y = float(pkt.get("y", 0.0))
                z = float(pkt.get("z", 0.0))
                hdg = float(pkt.get("heading_deg", 0.0))
                try:
                    self._bridge.update_actor(str(acid), (x, y, z), now_t=now_float, heading_deg=hdg, use_bank=False)
                except Exception:
                    if all(k in pkt for k in ("lon", "lat", "alt")):
                        continue

        try:
            alive = set(map(str, snap.keys()))
            self._bridge.prune_actors(alive)
        except Exception:
            pass


    def _sync_sim_time_text(self) -> None:
        # 1) SitlSim이 가상시각(sim_time)을 제공하면 그걸 우선 표시
        marker_dt: dt.datetime | None = None
        label = None
        try:
            if self._sim is not None:
                sim_dt = getattr(self._sim, "sim_time", None)
                if sim_dt:
                    if hasattr(sim_dt, "strftime"):
                        label = sim_dt.strftime("%H:%M:%S")
                    else:
                        label = str(sim_dt)
                    if isinstance(sim_dt, dt.datetime):
                        marker_dt = dt.datetime.fromtimestamp(sim_dt.timestamp())
        except Exception:
            pass
        if label is None:
            # 2) 폴백: 내부 카운터(HH:MM:SS)
            s = max(0, int(self._sim_seconds))
            hh, rem = divmod(s, 3600); mm, ss = divmod(rem, 60)
            label = f"{hh:02d}:{mm:02d}:{ss:02d}"
            if self._op_start_dt is not None:
                try:
                    marker_dt = self._op_start_dt + dt.timedelta(seconds=self._sim_seconds)
                except Exception:
                    marker_dt = None
        self.sim_time_edit.setText(label)
        if hasattr(self, "gantt_canvas") and self.gantt_canvas:
            self.gantt_canvas.update_time_marker(marker_dt)

    # ====== Qt overrides ======
    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            try:
                self._stop_monitoring_proc()
            except Exception:
                pass
            try:
                self._close_udp_sockets()
            except Exception:
                pass
            try:
                self._sim_thread.stop_loop()
                self._sim_thread.shutdown()
            except Exception:
                pass
            try:
                if hasattr(self, '_snapshot_worker') and self._snapshot_worker is not None:
                    self._snapshot_worker.flush()
                    self._snapshot_worker.shutdown()
            except Exception:
                pass
            if hasattr(self, "_exit_cb") and callable(self._exit_cb):
                self._exit_cb()
        finally:
            super().closeEvent(event)

    # ====== 아이콘/QSS ======
    def _icon_play(self) -> QIcon:  return QIcon(self._icon_pm(self._COL_ACCENT2, "play"))
    def _icon_ff(self)   -> QIcon:  return QIcon(self._icon_pm(self._COL_ACCENT2, "ff"))
    def _icon_pause(self)-> QIcon:  return QIcon(self._icon_pm(self._COL_ACCENT2, "pause"))
    def _icon_stop(self) -> QIcon:  return QIcon(self._icon_pm(self._COL_ACCENT2, "stop"))
    def _icon_pm(self, color_hex: str, kind: str, size: int = 22) -> QPixmap:
        pm = QPixmap(size, size); pm.fill(Qt.transparent); p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
        col = QColor(color_hex); p.setPen(Qt.NoPen); p.setBrush(QBrush(col))
        if kind == "play":
            poly = QPolygon([QPoint(int(size*0.32), int(size*0.18)), QPoint(int(size*0.32), int(size*0.82)), QPoint(int(size*0.82), int(size*0.50))]); p.drawPolygon(poly)
        elif kind == "ff":
            p.drawPolygon(QPolygon([QPoint(int(size*0.10), int(size*0.18)), QPoint(int(size*0.10), int(size*0.82)), QPoint(int(size*0.50), int(size*0.50))]))
            p.drawPolygon(QPolygon([QPoint(int(size*0.50), int(size*0.18)), QPoint(int(size*0.50), int(size*0.82)), QPoint(int(size*0.90), int(size*0.50))]))
        elif kind == "pause":
            w = int(size*0.22); g = int(size*0.10)
            p.drawRoundedRect(int(size*0.16), int(size*0.16), w, int(size*0.68), 3, 3)
            p.drawRoundedRect(int(size*0.16)+w+g, int(size*0.16), w, int(size*0.68), 3, 3)
        elif kind == "stop":
            p.drawRoundedRect(int(size*0.20), int(size*0.20), int(size*0.60), int(size*0.60), 4, 4)
        p.end(); return pm

    def _apply_styles(self) -> None:
        base_qss = ""
        try:
            apply_common_qss(self)
            base_qss = self.styleSheet()
        except Exception:
            base_qss = ""

        radius_panel = UiScale.dp(10); radius_card = UiScale.dp(8); radius_btn = UiScale.dp(10)
        qss = f"""
        QLabel#ChartCaption {{ color: #cbd5e1; padding-right: 4px; font-size: 11px; }}
        QWidget#RootArea {{ background-color: {self._COL_BG}; }}
        QLabel#TitleLabel {{ color: {self._COL_TEXT}; letter-spacing: 0.5px; }}
        QPushButton#OpenSchedulesButton, QPushButton#OpenMonitoringButton {{
            background-color: {self._COL_ACCENT}; color: white; border: 1px solid {self._COL_ACCENT};
            border-radius: {radius_btn}px; padding: {UiScale.dp(6)}px {UiScale.dp(12)}px; font-weight: 600;
        }}
        QPushButton#OpenSchedulesButton:hover, QPushButton#OpenMonitoringButton:hover {{ background-color: {self._COL_ACCENT_HOV}; border-color: {self._COL_ACCENT_HOV}; }}
        QLineEdit#PathEntry {{
            background-color: #0e1728; border: 1px solid {self._COL_BORDER}; border-radius: {UiScale.dp(6)}px; color: {self._COL_TEXT}; padding-left: {UiScale.dp(8)}px;
        }}
        QFrame#KpiCard {{ background-color: {self._COL_PANEL}; border: 1px solid {self._COL_BORDER}; border-radius: {radius_card}px; }}
        QLabel#KpiTitle {{ color: {self._COL_SUBTLE}; }}
        QLineEdit#KpiValue {{ background-color: #1e293b; border: 1px solid {self._COL_BORDER}; border-radius: {UiScale.dp(6)}px; color: {self._COL_TEXT}; padding-left: {UiScale.dp(8)}px; }}
        QFrame#Divider {{ background-color: {self._COL_BORDER}; min-height: {UiScale.dp(1)}px; max-height: {UiScale.dp(1)}px; }}
        QFrame#ChartFrame {{ background-color: {self._COL_PANEL}; border: 1px solid {self._COL_BORDER}; border-radius: {radius_card}px; }}
        QScrollArea#ChartScroll {{ border: none; }}
        QComboBox#VertiportCombo {{
            background-color: #0e1728; border: 1px solid {self._COL_BORDER};
            border-radius: {UiScale.dp(6)}px; color: {self._COL_TEXT};
            padding: 0 {UiScale.dp(28)}px 0 {UiScale.dp(10)}px; min-height: {UiScale.dp(30)}px;
        }}
        QComboBox#VertiportCombo::drop-down {{
            border: none; width: {UiScale.dp(24)}px; background: transparent;
        }}
        QComboBox#VertiportCombo QAbstractItemView {{
            background-color: #0f172a; color: {self._COL_TEXT};
            selection-background-color: {self._COL_ACCENT}; selection-color: white;
            border: 1px solid {self._COL_BORDER};
        }}
        QFrame#SidePanel {{ background-color: {self._COL_PANEL}; border: 1px solid {self._COL_BORDER}; border-radius: {radius_panel}px; }}
        QToolButton#CtrlButton {{
            background-color: rgba(2,6,23,0.0); border: 1px solid {self._COL_BORDER}; border-radius: {UiScale.dp(10)}px; color: {self._COL_TEXT}; padding: {UiScale.dp(12)}px {UiScale.dp(8)}px; font-size: {int(13 * UiScale.scale)}px;
        }}
        QToolButton#CtrlButton:hover {{ background-color: rgba(96,165,250,0.10); border-color: rgba(96,165,250,0.35); }}
        QLineEdit#SimTimeEdit {{ background-color: #0e1728; border: 1px solid {self._COL_BORDER}; border-radius: {UiScale.dp(6)}px; color: {self._COL_TEXT}; padding-left: {UiScale.dp(8)}px; }}
        QPushButton#SideActionButton, QPushButton#SideActionWideButton {{
            background-color: {self._COL_ACCENT}; color: white; border: 1px solid {self._COL_ACCENT}; border-radius: {UiScale.dp(14)}px; font-weight: 600;
        }}
        QPushButton#SideActionButton:hover, QPushButton#SideActionWideButton:hover {{
            background-color: {self._COL_ACCENT_HOV}; border-color: {self._COL_ACCENT_HOV};
        }}
        """
        if base_qss:
            self.setStyleSheet(base_qss + "\n" + qss)
        else:
            self.setStyleSheet(qss)


# ===== factory (build/main.py 에서 사용) =====
def create_fleet_operations_dashboard(*, plugin_name: str, plugin_port: int) -> FleetOperationDashboard:
    window = FleetOperationDashboard(plugin_name=plugin_name, plugin_port=plugin_port)
    window.set_connection_info(plugin_name=plugin_name, plugin_port=plugin_port)
    return window
