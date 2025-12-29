# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import datetime as dt

from PyQt5.QtCore import Qt ,pyqtSignal,QUrl 
from PyQt5.QtGui import QPixmap, QFont, QFontMetrics,QDesktopServices
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel
)

# Data_analysis.py에 있던 스타일을 그대로 옮김.
# 단, 루트 선택자만 #ImpactViewer -> #BaseTab 으로 치환(배경색 적용 대상만 바뀜).
THEME_QSS = """
#BaseTab { background:#2b2f36; }
QLabel { color:#e9eef7; }
QFrame#Header {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #2f343f, stop:1 #262a33);
    border:0; border-radius:12px;
}
QLabel#MetaKey { color:#cfd6e4; font-size:12px; }
QLabel#MetaVal { color:#e9eef7; font-size:12px; font-family: Consolas, "Courier New", monospace; }
QFrame#Panel {
    background:#1f232b;
    border:1px solid #131722;
    border-radius:12px;
}
QTableWidget#DataTable {
    background:#1f232b;
    border:1px solid #131722;
    border-radius:12px;
    gridline-color:#2a3040;
    selection-background-color:#3158ff;
    selection-color:#ffffff;
    alternate-background-color:#1b1f27;
    font-size:12px;
}
QTableWidget#DataTable::item { padding:6px 10px; color:#ffffff;}
QTableWidget#DataTable::item:hover { background:#2b3242; color:#ffffff; }
QTableWidget#DataTable::item:selected { background:#3158ff; color:#ffffff; }
QHeaderView::section {
    background:#20252e; color:#cfd6e4; border:0;
    padding:6px 8px; font-weight:600; font-size:12px;
}
QScrollBar:vertical {
    background:#1f232b; width:12px; margin:0;
}
QScrollBar::handle:vertical {
    background:#3b4252; min-height:30px; border-radius:6px;
}
QScrollBar::handle:vertical:hover { background:#5b8cff; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }

QScrollBar:horizontal { background:#1f232b; height:12px; margin:0; }
QScrollBar::handle:horizontal { background:#3b4252; min-width:30px; border-radius:6px; }
QScrollBar::handle:horizontal:hover { background:#5b8cff; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width:0; }

QSlider#TimeSlider { border:0; }
QSlider#TimeSlider::groove:horizontal {
    height:8px; background:#3b4252; border-radius:4px; margin:10px 14px;
}
QSlider#TimeSlider::sub-page:horizontal { background:#5b8cff; border-radius:4px; }
QSlider#TimeSlider::handle:horizontal {
    background:#3573ff; border:1px solid #274bdb;
    width:18px; height:18px; margin:-6px 0; border-radius:9px;
}
QPushButton#ActionBtn {
    background:#274bdb; color:#ffffff;
    border:0; border-radius:10px; padding:10px 18px;
}
QPushButton#ActionBtn:hover { background:#3158ff; }
QPushButton#TopBtn {
    background:#ff6d00; color:#ffffff;
    border:0; border-radius:10px; padding:10px 18px;
}
QPushButton#TopBtn:hover { background:#ff9100; }
QPushButton#TopBtn:checked { background:#ff9100; }
QSpinBox#HSpin, QSpinBox#MSpin, QSpinBox#SSpin {
    background:#20252e; color:#f3f6fb;
    border:1px solid #131722; border-radius:8px; padding:6px 10px; font-size:14px;
}
QSpinBox#HSpin::up-button, QSpinBox#MSpin::up-button, QSpinBox#SSpin::up-button {
    subcontrol-origin: border; subcontrol-position: top right; width:18px; border:0;
}
QSpinBox#HSpin::down-button, QSpinBox#MSpin::down-button, QSpinBox#SSpin::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right; width:18px; border:0;
}
"""

class BaseTab(QWidget):
    """
    - 상단 헤더(로고 + 메타 정보)
    - 전체 테마(QSS)
    - 본문을 추가할 수 있는 body 레이아웃
    """
    backRequested = pyqtSignal()
    def __init__(
        self,
        parent=None,
        *,
        logo_path: Path | None = None,
        version_text: str = "v0.0.1",
        updated_date: str | None = None,
        usable_text: str = "-",
        show_back: bool = False
    ):
        super().__init__(parent)
        self.setObjectName("BaseTab")
        self.setStyleSheet(THEME_QSS)

        # 최상위 레이아웃
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── 헤더
        header = QFrame(objectName="Header")
        header.setFixedHeight(96)
        hb = QHBoxLayout(header)
        hb.setContentsMargins(16, 10, 16, 10)
        hb.setSpacing(12)

        # 로고
        self.logo_label = QLabel()
        if logo_path and Path(logo_path).exists():
            pix = QPixmap(str(logo_path)).scaledToHeight(56, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pix)
        else:
            self.logo_label.setText("KADA")
            self.logo_label.setStyleSheet("font-weight:700; font-size:26px;")

        # 클릭 시 KADA 홈페이지 열기  ← 여기만 이렇게 정리
        self.logo_label.setCursor(Qt.PointingHandCursor)
        def _open_kada(e):
            QDesktopServices.openUrl(QUrl("https://kada.konkuk.ac.kr"))
            e.accept()
        self.logo_label.mousePressEvent = _open_kada

        hb.addWidget(self.logo_label, 0, Qt.AlignVCenter)
        hb.addStretch(1)
        
        # (우측) 뒤로가기 버튼 (옵션)
        if show_back:
            from PyQt5.QtWidgets import QPushButton
            self.btn_back = QPushButton("뒤로가기")
            self.btn_back.setObjectName("TopBtn")
            self.btn_back.setCheckable(False)
            self.btn_back.clicked.connect(self.backRequested.emit)
            hb.addWidget(self.btn_back, 0, Qt.AlignVCenter)

        # 메타 정보 (버전/업데이트/사용가능일자)
        metaGrid = QGridLayout()
        metaGrid.setContentsMargins(0, 0, 0, 0)
        metaGrid.setHorizontalSpacing(12)
        metaGrid.setVerticalSpacing(2)

        key_font = QFont(); key_font.setPointSize(12)
        val_font = QFont("Consolas"); val_font.setPointSize(12); val_font.setStyleHint(QFont.Monospace)
        keys = ["프로그램 버전", "업데이트 날짜", "사용가능 일자"]
        max_w = max(QFontMetrics(key_font).width(k) for k in keys) + 8

        def add_row(row: int, key_text: str, attr: str, initial_val: str):
            k = QLabel(key_text); k.setObjectName("MetaKey"); k.setFont(key_font)
            v = QLabel(initial_val); v.setObjectName("MetaVal"); v.setFont(val_font)
            setattr(self, attr, v)
            k.setFixedWidth(max_w)
            k.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            v.setAlignment(Qt.AlignLeft  | Qt.AlignVCenter)
            metaGrid.addWidget(k, row, 0); metaGrid.addWidget(v, row, 1)

        add_row(0, "프로그램 버전", "meta_version", version_text)
        add_row(1, "업데이트 날짜", "meta_updated", updated_date or dt.date.today().strftime("%Y.%m.%d"))
        add_row(2, "사용가능 일자", "meta_usable", usable_text)

        metaGrid.setColumnStretch(0, 0)
        metaGrid.setColumnStretch(1, 1)
        hb.addLayout(metaGrid)

        root.addWidget(header)

        # ── 본문: 상속 측에서 이 레이아웃에 원하는 위젯을 추가하면 됨
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        root.addLayout(self.body, 1)
