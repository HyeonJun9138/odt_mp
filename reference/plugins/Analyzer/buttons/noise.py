#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,QInputDialog, QFileDialog, QMessageBox, QApplication
from pathlib import Path
from Functions.io import pick_noise_from_registry_csv, validate_noise_schema
from Tabs.Noise_tab import NoiseTab

try:
    from Analyzer.Functions.core import get_db_dir
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from Functions.core import get_db_dir

def launch_noise_view(main):
    sel_path = None

    # (1) 업로드에서 등록한 파일이 있으면 그중 선택
    files = [p for p in getattr(main, "registered_files", []) if str(p).lower().endswith(".csv")]
    if files:
        labels = [Path(p).name for p in files]
        label, ok = QInputDialog.getItem(main, "소음 데이터 선택", "등록된 파일:", labels, 0, False)
        if ok:
            sel_path = files[labels.index(label)]
    else:
        # (2) 레지스트리 CSV가 지정돼 있으면 그 목록에서 선택 시도
        reg = getattr(main, "noise_registry_csv", None)
        if reg:
            try:
                sel_path = pick_noise_from_registry_csv(main, reg)
            except Exception as e:
                print(f"[registry error] {e}")

    # (3) 그래도 못 정했으면 일반 파일 선택
    if not sel_path:
        sel_path, _ = QFileDialog.getOpenFileName(
            main, "소음 CSV 선택", str(get_db_dir()), "CSV Files (*.csv)"
        )
        if not sel_path:
            QMessageBox.information(main, "안내", "소음 CSV가 선택되지 않았습니다.")
            return

    # (4) 스키마 검사
    ok, msg = validate_noise_schema(sel_path)  # gid + (Lmax_w|Lmax_1s) + (tsec|time|sim_time)
    if not ok:
        QMessageBox.warning(main, "양식 오류", f"선택한 파일의 양식이 올바르지 않습니다.\n\n{msg}")
        return

    # (5) 창 전환 (메인과 동일한 위치/크기/상태로 열기)
    geom  = main.frameGeometry()
    state = main.windowState()

    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        if getattr(main, "_noise_win", None):
            try: main._noise_win.close()
            except: pass

        main._noise_win = NoiseTab(parent=None, noise_csv=sel_path)
        main._noise_win.backRequested.connect(lambda: _back_to_main(main))
        main._noise_win.destroyed.connect(lambda: setattr(main, "_noise_win", None))

        main._noise_win.setGeometry(geom)
        if state & Qt.WindowMaximized:
            main._noise_win.showMaximized()
        else:
            main._noise_win.show()

        QApplication.processEvents()
        main.hide()
    finally:
        QApplication.restoreOverrideCursor()

def _back_to_main(main):
    if getattr(main, "_noise_win", None):
        try: main._noise_win.close()
        except: pass
        main._noise_win = None
    main.show()



class NoisePage(QWidget):
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._connect()

    def _build_ui(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        card = QFrame(); card.setObjectName("RightCard")
        lay = QVBoxLayout(card); lay.setContentsMargins(20,20,20,20); lay.setSpacing(12)

        hdr = QHBoxLayout()
        title = QLabel("소음 해석"); title.setStyleSheet("color:#ffffff; font-size:28px; font-weight:800; background:transparent;")
        hdr.addWidget(title); hdr.addStretch(1)
        self.btn_close = QPushButton("닫기")
        self.btn_close.setStyleSheet("QPushButton { background:#5a6274; color:#fff; border:none; border-radius:8px; padding:10px 14px; }")
        hdr.addWidget(self.btn_close)
        lay.addLayout(hdr)

        body = QLabel("준비 중입니다."); body.setAlignment(Qt.AlignCenter)
        body.setStyleSheet("color:#ffffff; font-size:20px; background:transparent;")
        lay.addWidget(body, 1)

        outer.addWidget(card)

    def _connect(self):
        self.btn_close.clicked.connect(self.closed.emit)

