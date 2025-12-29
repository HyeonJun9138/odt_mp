"""Tabs/base_tab.py

공통 Tab 베이스 클래스
───────────────────────
다른 모든 *_tab.py 모듈은 여기서 정의한 `Tab` 을 상속합니다.
(Analytic/Monitoring 등 메인 윈도우 파일에는 Tab 정의가 없어야 함)
"""
from __future__ import annotations

from typing import List

from PyQt5.QtCore    import Qt
from PyQt5.QtWidgets import (
    QWidget, QSplitter, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QTextEdit,
)

# ──────────────────────────────────────────────────────────
# Tab Base‑class
# ──────────────────────────────────────────────────────────
class Tab(QWidget):
    """공통 레이아웃 ( Map | Table / Msg ) 탭 베이스 클래스"""

    MAP_WIDTH = 1200

    # --------------------------------------------------------
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._title = title
        self._build_ui()

    # --------------------------------------------------------
    def create_map_widget(self) -> QWidget:
        """좌측 Map 영역 위젯 생성 (서브클래스에서 오버라이드 가능)"""
        lbl = QLabel(f"{self._title}\nMap", alignment=Qt.AlignCenter)
        lbl.setStyleSheet("background:#4066C7; color:#ffffff; "
                          "font:600 24px 'Segoe UI';")
        return lbl

    # --------------------------------------------------------
    def _build_ui(self) -> None:
        # (L) Map placeholder
        self.map_view = self.create_map_widget()
        self.map_view.setFixedWidth(self.MAP_WIDTH)

        # (R‑Top) Table
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Col‑1", "Col‑2", "Col‑3"])

        # (R‑Bottom) GPT‑based message
        self.msg_box = QTextEdit(self)
        self.msg_box.setReadOnly(True)
        self.msg_box.setPlaceholderText("Decision Support Msg (GPT‑based)")

        # Splitters 구성
        v_split = QSplitter(Qt.Vertical, self)
        v_split.addWidget(self.table)
        v_split.addWidget(self.msg_box)
        v_split.setStretchFactor(0, 3)   # Table : Msg = 3 : 2
        v_split.setStretchFactor(1, 2)

        h_split = QSplitter(Qt.Horizontal, self)
        h_split.addWidget(self.map_view)
        h_split.addWidget(v_split)
        h_split.setStretchFactor(0, 0)   # Map 고정폭
        h_split.setStretchFactor(1, 1)
        h_split.setCollapsible(0, False)
        h_split.setCollapsible(1, False)

        self.h_split = h_split   # 자식 탭에서 좌우 폭 조절 등에 사용

        lay = QHBoxLayout(self)
        lay.addWidget(h_split)
        self.setLayout(lay)

    # -------------------------- helper ---------------------
    def populate_table(self, rows: List[List[str]]) -> None:
        """테이블 전체 덮어쓰기"""
        if not rows:
            self.table.clearContents()
            self.table.setRowCount(0)
            return
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(rows[0]))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))

    def update_msg(self, text: str) -> None:
        self.msg_box.setPlainText(text)

    # ----------------------- UDP hooks ---------------------
    def process_new_data_packet(self, vid: str, data: dict):
        """UDP 패킷 수신 시 호출 (서브클래스에서 오버라이드)"""
        pass

    def remove_vehicle(self, vid: str):
        pass
