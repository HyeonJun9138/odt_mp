# Scheduling/ArrivalsTab.py
import datetime as dt
from datetime import timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView
)




class ArrivalsTab(QWidget):
    """
    도착 버티포트 기준 목록만 보여 주는 아주 단순한 탭
      · FPL Maker의 Generate/Map 등은 제거
      · 콤보박스로 도착 VP 선택 → 우측 Table 갱신
    """
    COLS = ["regNum", "ETOT", "ATOT","T-Delay","ELDT", "ALDT",
        "L-Delay", "Standby Time", "From"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._flights = {}             # {dest : [flight dict]}
        self._build_ui()

    # ────────────────────────────────────────────────────────────
    def refresh_current(self):          # ← 새 메서드
        self._refresh_table(self.cbo_vp.currentText())

    def _build_ui(self):
        lay = QVBoxLayout(self)

        # ── 상단: 도착 VP 콤보박스 ───────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel("Destination VP:"))
        self.cbo_vp = QComboBox()
        self.cbo_vp.currentTextChanged.connect(self._refresh_table)
        top.addWidget(self.cbo_vp, stretch=1)
        lay.addLayout(top)

        # ── 중앙: Flight Table ─────────────────────────────
        self.tbl = QTableWidget(0, len(self.COLS))
        self.tbl.setHorizontalHeaderLabels(self.COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.tbl)

    # ────────────────────────────────────────────────────────────
    # 외부에서 호출
    def set_planner(self, vp_csv: str, wp_csv: str):
        """(인터페이스 맞추기용, 여기선 아무 일 안 함)"""
        return

    def set_flights(self, flights_by_dest: dict):
        """dest_map = {destination: [flight dict]} 그대로 주입"""
        self._flights = flights_by_dest
        self.cbo_vp.clear()
        self.cbo_vp.addItems(sorted(self._flights.keys()))
        self._refresh_table(self.cbo_vp.currentText())

    # ────────────────────────────────────────────────────────────
    # 내부
    
    def _refresh_table(self, dest_name: str):
        fls = self._flights.get(dest_name, [])
        self.tbl.setRowCount(len(fls))

        fls_sorted = sorted(
            fls, key=lambda x: x.get("actual_touch", x["landing_ready"])
        )

        for r, f in enumerate(fls_sorted):
            # ── ETOT / ATOT ─────────────────────────────────────────
            etot = f.get("etot_plan")
            etot_str = etot.strftime("%H:%M:%S") if isinstance(etot, dt.datetime) else "-"
            atot = f.get("actual_takeoff_finish")
            atot_str = atot.strftime("%H:%M:%S") if isinstance(atot, dt.datetime) else ""

            # ── Takeoff Delay ───────────────────────────────────────
            if isinstance(etot, dt.datetime) and isinstance(atot, dt.datetime):
                takeoff_delay_s = max(0, int((atot - etot).total_seconds()))
                to_delay_str = str(timedelta(seconds=takeoff_delay_s))
            else:
                to_delay_str = "-"

            landing_ready   = f["landing_ready"]
            actual_touch = f.get("actual_touch")

            # ── Delay Time ───────────────────────────────────────────
            if actual_touch:
                delay_sec = max(0, int((actual_touch - landing_ready).total_seconds()))
                delay_str = str(timedelta(seconds=delay_sec))    # HH:MM:SS
                actual_touch_str = actual_touch.strftime("%H:%M:%S")
            else:                        # 아직 계산 전
                delay_str = "-"
                actual_touch_str = "-"

            # ── Standby Time ────────────────────────────────────────
            idle_sec = f.get("t_wait_sec")
            idle_str = str(timedelta(seconds=idle_sec)) if idle_sec is not None else "-"

            # ── Row 채우기 ───────────────────────────────────────────
            row_vals = [
                f.get("uam_id", ""),
                etot_str,
                atot_str,
                to_delay_str,
                landing_ready.strftime("%H:%M:%S"),
                actual_touch_str,
                delay_str,
                idle_str,
                f["origin"],
            ]

            for c, val in enumerate(row_vals):
                self.tbl.setItem(r, c, QTableWidgetItem(val))