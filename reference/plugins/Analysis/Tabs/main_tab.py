from typing import Dict, List
from PyQt5.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QTextEdit,
    QLabel, QSplitter, QSizePolicy, QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # vid -> raw phase 코드 저장
        self._phase_data: Dict[str, str] = {}
        # lv3 혼잡 구간 저장
        self._congestion_lv3: List[str] = []
        # 최대 Flights In Progress 기록
        self._max_in_progress: int = 0
        self._init_ui()

    def _init_ui(self):
        # ── Header: Exact Time only
        header = QFrame(self)
        header.setFrameShape(QFrame.NoFrame)
        h_header = QHBoxLayout(header)
        h_header.setContentsMargins(10, 10, 10, 10)

        self.time_label = QLabel("Exact Time : --:--:--", self)
        font = QFont()
        font.setPointSize(32)
        font.setBold(True)
        self.time_label.setFont(font)
        self.time_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        h_header.addWidget(self.time_label, 1)

        # ── Left: Summary, Phase counts, Log
        self.summary_tbl = QTableWidget(4, 2, self)
        self.summary_tbl.setHorizontalHeaderLabels(["Metric", "Value"])
        metrics = ["Scheduled Flights", "Flights In Progress", "Completed Flights", "Progress (%)"]
        for i, m in enumerate(metrics):
            mi = QTableWidgetItem(m)
            mi.setTextAlignment(Qt.AlignCenter)
            vi = QTableWidgetItem("0")
            vi.setTextAlignment(Qt.AlignCenter)
            self.summary_tbl.setItem(i, 0, mi)
            self.summary_tbl.setItem(i, 1, vi)
        self._configure_table(self.summary_tbl)

        self.phase_tbl = QTableWidget(5, 2, self)
        self.phase_tbl.setHorizontalHeaderLabels(["Phase", "Count"])
        phases = ["Takeoff", "Climb", "Cruise", "Approach", "Landing"]
        for i, p in enumerate(phases):
            pi = QTableWidgetItem(p)
            pi.setTextAlignment(Qt.AlignCenter)
            ci = QTableWidgetItem("0")
            ci.setTextAlignment(Qt.AlignCenter)
            self.phase_tbl.setItem(i, 0, pi)
            self.phase_tbl.setItem(i, 1, ci)
        self._configure_table(self.phase_tbl)

        self.log_box = QTextEdit(self)
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Log / Text")

        left_container = QWidget(self)
        left_v = QVBoxLayout(left_container)
        left_v.setSpacing(8)
        left_v.setContentsMargins(5, 5, 5, 5)
        left_v.addWidget(self.summary_tbl, 1)
        left_v.addWidget(self.phase_tbl, 1)
        left_v.addWidget(self.log_box, 1)

        # ── Right: Schedule & Vis placeholders
        schedule_frame = QFrame(self)
        schedule_frame.setFrameShape(QFrame.Box)
        sf_layout = QVBoxLayout(schedule_frame)
        lbl_sched = QLabel("TBD", schedule_frame)
        lbl_sched.setAlignment(Qt.AlignCenter)
        sf_layout.addWidget(lbl_sched)

        vis_frame = QFrame(self)
        vis_frame.setFrameShape(QFrame.Box)
        vf_layout = QVBoxLayout(vis_frame)
        lbl_vis = QLabel("TBD", vis_frame)
        lbl_vis.setAlignment(Qt.AlignCenter)
        vf_layout.addWidget(lbl_vis)

        right_container = QWidget(self)
        right_v = QVBoxLayout(right_container)
        right_v.setSpacing(8)
        right_v.setContentsMargins(5, 5, 5, 5)
        right_v.addWidget(schedule_frame, 1)
        right_v.addWidget(vis_frame, 1)

        # ── Splitter
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([200, 200])

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(header)
        root.addWidget(splitter, 1)
        self.setLayout(root)

    def _configure_table(self, tbl: QTableWidget):
        tbl.verticalHeader().setVisible(False)
        tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def process_new_data_packet(self, vid: str, ac: dict):
        # 시뮬레이터에서 보내는 time 필드로 Exact Time 갱신
        ts = ac.get("time") or ac.get("timestamp")
        if ts:
            self.time_label.setText(f"Exact Time : {ts}")
        # phase 코드 저장 및 테이블 업데이트
        self._phase_data[vid] = str(ac.get("phase", "")).upper()
        self._update_phase_counts()

    def update_congestion_lv3(self, lv3_list):
        # 외부에서 congestion tab으로부터 Lv3 구간 전달
        self._congestion_lv3 = lv3_list
        self._update_log()

    def remove_vehicle(self, vid: str):
        self._phase_data.pop(vid, None)
        self._update_phase_counts()

    def _update_phase_counts(self):
        mapping = {
            'B': 'Takeoff',
            'C': 'Climb', 'D': 'Climb', 'E': 'Climb',
            'F': 'Cruise',
            'G': 'Approach', 'H': 'Approach', 'I': 'Approach',
            'J': 'Landing'
        }
        counts = {cat: 0 for cat in ["Takeoff", "Climb", "Cruise", "Approach", "Landing"]}
        for ph in self._phase_data.values():
            cat = mapping.get(ph)
            if cat:
                counts[cat] += 1
        # 테이블 반영
        for row, cat in enumerate(["Takeoff", "Climb", "Cruise", "Approach", "Landing"]):
            self.phase_tbl.item(row, 1).setText(str(counts[cat]))
        # Flights In Progress 업데이트 (Takeoff~Landing 총합)
        total = sum(counts.values())
        self.summary_tbl.item(1, 1).setText(str(total))
        # 최대값 갱신
        if total > self._max_in_progress:
            self._max_in_progress = total
        # Log 업데이트
        self._update_log()

    def _update_log(self):
        lines = ["[운항정보 요약]"]
        lines.append(f"- Maximum Flight Num : {self._max_in_progress}")
        lines.append("- 혼잡 Lv3 :")
        if self._congestion_lv3:
            for name in self._congestion_lv3:
                lines.append(f"    • {name}")
        else:
            lines.append("    • None")

        # 한 번에 세팅 (줄바꿈 유지)
        self.log_box.setPlainText("\n".join(lines))

    def _update_progress(self, completed: int = 0):
        scheduled = len(self._phase_data)
        in_progress = sum(1 for ph in self._phase_data.values() if ph in ['B','C','D','E','F','G','H','I','J'])
        self.summary_tbl.item(2, 1).setText(str(completed))
        self.summary_tbl.item(3, 1).setText(
            f"{(in_progress / scheduled * 100) if scheduled else 0:.1f}%"
        )