#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel, QPushButton,
    QComboBox, QRadioButton, QButtonGroup, QSizePolicy, QTableWidget, QTableWidgetItem, QTextEdit, QSpacerItem
)

PRICING = {
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-4o": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
    "gpt-4o-mini": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o3-pro": {"input": 20.00, "cached_input": None, "output": 80.00},
    "o4-mini": {"input": 1.10, "cached_input": 0.28, "output": 4.40},
}

# 업로드된 파일들을 주입받아 단일 모드 콤보박스 목록을 갱신하기 위한 키별 필터(예시)
CATEGORY_HINTS = {
    "항로 설계": ("route", "항로", "path"),
    "버티포트": ("vertiport", "vp", "버티포트"),
    "항적 정보": ("track", "traj", "항적"),
    "소음": ("noise", "소음", "sound"),
}

class AnalyzePage(QWidget):
    closed = pyqtSignal()
    startRequested = pyqtSignal(dict)   # {'model':..., 'mode':'single'|'compare', 'inputs':{...}}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registered_files = []   # 메인에서 주입 가능
        self._build_ui()
        self._connect()

    # ───────────────────────────────────────────────────────────
    # 외부 주입: 업로드 등록 목록을 전달 받아 콤보 갱신
    # ───────────────────────────────────────────────────────────
    def set_registered_files(self, files: list):
        self._registered_files = list(files or [])
        self._refresh_single_mode_inputs()

    # ───────────────────────────────────────────────────────────
    # UI
    # ───────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        card = QFrame(); card.setObjectName("RightCard")
        lay = QVBoxLayout(card); lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(16)

        # 헤더
        hdr = QHBoxLayout(); hdr.setSpacing(10)
        title = QLabel("종합 분석")
        title.setStyleSheet("color:#ffffff; font-size:28px; font-weight:800; background:transparent;")
        hdr.addWidget(title); hdr.addStretch(1)

        self.btn_close = QPushButton("닫기")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setStyleSheet(
            "QPushButton { background:#5a6274; color:#fff; border:none; border-radius:8px; padding:10px 14px; }"
        )
        hdr.addWidget(self.btn_close)
        lay.addLayout(hdr)

        # 본문: 좌측 설정 / 우측 액션
        body = QHBoxLayout(); body.setSpacing(20)

        # 좌: 설정 패널
        left = QFrame(); left.setObjectName("RightCard")
        left.setStyleSheet("QFrame#RightCard{background:#3a404f; border-radius:12px;}")
        leftL = QVBoxLayout(left); leftL.setContentsMargins(8,8,8,8); leftL.setSpacing(6)

        # 1) 모델 선택
        sec1_title = QLabel("모델 선택"); sec1_title.setStyleSheet("color:#fff; font-size:20px; font-weight:300;"); sec1_title.setFixedHeight(24)
        leftL.addWidget(sec1_title)

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(list(PRICING.keys()))
        self.cmb_model.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cmb_model.setStyleSheet("""
            QComboBox { background:#ffffff; color:#222; border:0; border-radius:8px; padding:8px 10px; min-height:15px; }
            QComboBox QAbstractItemView { background:#ffffff; color:#222; }
        """)
        leftL.addWidget(self.cmb_model)

        # 2) 모드 선택 (단일/다중)

        leftL.addWidget(self.cmb_model)
        leftL.addSpacing(20)  # 간격 추가

        # 2) 모드 선택
        sec2_title = QLabel("모드 선택"); sec2_title.setStyleSheet("color:#fff; font-size:20px; font-weight:700;"); sec2_title.setFixedHeight(24)
        leftL.addWidget(sec2_title)

        self.rb_single = QRadioButton("단일 시나리오 분석 (Analyzer\\prompt\\single.txt)")
        self.rb_multi  = QRadioButton("다중 시나리오 비교 (Analyzer\\prompt\\compare.txt)")
        self.rb_single.setChecked(True)

        # 라디오를 묶어 상호배타
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_single)
        self.mode_group.addButton(self.rb_multi)

        for rb in (self.rb_single, self.rb_multi):
            rb.setStyleSheet("color:#fff; font-size:14px;")
            leftL.addWidget(rb)

        # 2-1) 단일 모드 입력(라벨-콤보 균등 격자)
        self.single_wrap = QFrame()
        g = QGridLayout(self.single_wrap)
        g.setContentsMargins(0, 3, 0, 0)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(20)

        def make_row(row, label_text):
            lab = QLabel(label_text); 
            lab.setStyleSheet("color:#fff; font-size:14px;")
            lab.setMinimumWidth(50)     # 글자수와 무관하게 여유/균등
            lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            cmb = QComboBox()
            cmb.setStyleSheet("""
                QComboBox { background:#ffffff; color:#222; border:0; border-radius:8px; padding:8px 10px; min-height:15px; }
                QComboBox QAbstractItemView { background:#ffffff; color:#222; }
            """)
            cmb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            g.addWidget(lab, row, 0)
            g.addWidget(cmb, row, 1)
            return cmb

        self.cmb_route    = make_row(0, "항로정보")
        self.cmb_verti    = make_row(1, "버티포트")
        self.cmb_track    = make_row(2, "항적정보")
        self.cmb_noise    = make_row(3, "소음정보")
        g.setRowStretch(5, 1)  
        
        leftL.addWidget(self.single_wrap)

        # 단일 모드 입력 폼 아래에 구분선 + 여백
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color:#4a4f5e;")
        leftL.addWidget(sep)
        leftL.addSpacing(10)  # ← 폼과 로그창 사이 바깥 간격

        # 2-2) 다중 모드 입력(TBD 표)
        self.multi_wrap = QFrame(); self.multi_wrap.setVisible(False)
        mv = QVBoxLayout(self.multi_wrap); mv.setContentsMargins(0,6,0,0); mv.setSpacing(8)

        hint = QLabel("비교 데이터(TBD): 업로드/비교 데이터 관리에서 올라간 파일명 나열 예정")
        hint.setStyleSheet("color:#d9d9d9; font-size:13px;")
        mv.addWidget(hint)

        self.tbl_compare = QTableWidget(0, 2)
        self.tbl_compare.setHorizontalHeaderLabels(["파일명", "유형(자동 추정/TBD)"])
        self.tbl_compare.horizontalHeader().setStretchLastSection(True)
        self.tbl_compare.setStyleSheet("""
            QTableWidget { background:#2f3542; color:#fff; border:0; border-radius:8px; }
            QHeaderView::section { background:#586073; color:#fff; border:0; padding:6px; }
        """)
        mv.addWidget(self.tbl_compare, 1)

        leftL.addWidget(self.multi_wrap, 1)

        # ── 로그창 (아래 남는 공간)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("""
            QTextEdit {
                background: #ffffff;
                color: #000000;
                border-radius: 6px;
                font-size: 13px;
                padding: 8px;          /* 내부 패딩(텍스트와 박스 간격) */
            }
        """)

        # 로그 래퍼에 top-margin을 주어 '폼과의 거리'를 더 확보
        log_wrap = QFrame()
        log_lay = QVBoxLayout(log_wrap)
        log_lay.setContentsMargins(0, 6, 0, 0)  # ← 폼과 로그 사이 바깥 간격(상단 6px)
        log_lay.setSpacing(0)
        log_lay.addWidget(self.log_view)

        leftL.addWidget(log_wrap, 1)  # ← 남는 공간은 로그가 채움

        body.addWidget(left, 3)

        # 우: 액션 패널
        right = QFrame(); right.setObjectName("RightCard")
        rr = QVBoxLayout(right); rr.setContentsMargins(16,16,16,16); rr.setSpacing(12)

        rr.addStretch(1)
        self.btn_start = QPushButton("분석 시작")
        self.btn_save  = QPushButton("Save as")
        for b in (self.btn_start, self.btn_save):
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(42)
            b.setStyleSheet(
                "QPushButton { background:#2e8b3d; color:#fff; border:none; border-radius:10px; padding:10px 18px; font-weight:700; }"
            )
        self.btn_save.setStyleSheet(
            "QPushButton { background:#5a6274; color:#fff; border:none; border-radius:10px; padding:10px 18px; font-weight:700; }"
        )
        rr.addWidget(self.btn_start)
        rr.addWidget(self.btn_save)
        body.addWidget(right, 1)

        lay.addLayout(body, 1)
        outer.addWidget(card)

    # ───────────────────────────────────────────────────────────
    # 시그널 연결
    # ───────────────────────────────────────────────────────────
    def _connect(self):
        self.btn_close.clicked.connect(self.closed.emit)
        self.rb_single.toggled.connect(self._on_mode_changed)
        self.btn_start.clicked.connect(self._on_start_clicked)

    # ───────────────────────────────────────────────────────────
    # 동작
    # ───────────────────────────────────────────────────────────
    def _on_mode_changed(self, checked: bool):
        single = self.rb_single.isChecked()
        self.single_wrap.setVisible(single)
        self.multi_wrap.setVisible(not single)

    def _on_start_clicked(self):
        mode = "single" if self.rb_single.isChecked() else "compare"
        model = self.cmb_model.currentText().strip()

        if mode == "single":
            payload = {
                "model": model,
                "mode": "single",
                "inputs": {
                    "항로 설계": self.cmb_route.currentText().strip(),
                    "버티포트": self.cmb_verti.currentText().strip(),
                    "항적 정보": self.cmb_track.currentText().strip(),
                    "소음": self.cmb_noise.currentText().strip(),
                }
            }
        else:
            # 비교 표는 향후 구현(TBD)
            rows = []
            for r in range(self.tbl_compare.rowCount()):
                fn = self.tbl_compare.item(r, 0).text() if self.tbl_compare.item(r, 0) else ""
                ty = self.tbl_compare.item(r, 1).text() if self.tbl_compare.item(r, 1) else ""
                rows.append((fn, ty))
            payload = {"model": model, "mode": "compare", "inputs": {"files": rows}}

        self.startRequested.emit(payload)

    # 업로드 목록을 바탕으로, 항목별 콤보 채우기 (간단한 휴리스틱)
    def _refresh_single_mode_inputs(self):
        def pick(caption):
            hints = CATEGORY_HINTS.get(caption, ())
            cand = []
            for f in self._registered_files:
                name = str(f).lower()
                if any(h in name for h in hints):
                    cand.append(f)
            # 비어있으면 전체에서 csv/엑셀 위주 노출
            if not cand:
                cand = [f for f in self._registered_files if any(f.lower().endswith(ext) for ext in (".csv",".xlsx",".xls"))]
            return cand

        def refill(combo: QComboBox, items):
            combo.clear()
            for it in items:
                combo.addItem(it)

        refill(self.cmb_route, pick("항로 설계"))
        refill(self.cmb_verti, pick("버티포트"))
        refill(self.cmb_track, pick("항적 정보"))
        refill(self.cmb_noise, pick("소음"))

    # 다중 모드 테이블 채우기(지금은 파일명만 나열, TBD)
    def populate_compare_table(self, files: list):
        self.tbl_compare.setRowCount(0)
        for f in files or []:
            r = self.tbl_compare.rowCount()
            self.tbl_compare.insertRow(r)
            self.tbl_compare.setItem(r, 0, QTableWidgetItem(f))
            self.tbl_compare.setItem(r, 1, QTableWidgetItem("TBD"))
