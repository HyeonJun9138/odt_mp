#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton
from PyQt5.QtWidgets import QInputDialog, QFileDialog, QMessageBox, QApplication
from pathlib import Path

from Tabs.Traffic_tab import TrafficTab
from Functions.core import RES, get_db_dir


def _pick_from_registered(main, suffix: str):
    files = [Path(p) for p in getattr(main, "registered_files", []) if str(p).lower().endswith(suffix)]
    if not files: return None
    labels = [p.name for p in files]
    label, ok = QInputDialog.getItem(main, "파일 선택", f"등록된 {suffix}:", labels, 0, False)
    return (files[labels.index(label)] if ok else None)

def launch_traffic_view(main):
    """Noise와 같은 전체 전환(별도 창) 런처"""
    # 1) 트랙 CSV 선택(등록 목록 → 기본값 → 파일 다이얼로그)
    trk = _pick_from_registered(main, ".csv")
    if not trk or "track" not in trk.name.lower():
        cand = RES / "track_log.csv"
        if cand.exists():
            trk = cand
        else:
            p, _ = QFileDialog.getOpenFileName(
                main,
                "트랙 데이터(트랙 로그)",
                str(Path.cwd()),       # ← 메인에서 작업 디렉터리를 Analyzer/database로 바꿔뒀다면 여기서 그 폴더가 열립니다.
                "CSV (*.csv)"
            )
            if not p:
                QMessageBox.information(main, "안내", "트랙 CSV가 선택되지 않았습니다.")
                return
            trk = Path(p)

    # 2) VP/WP 기본값 먼저 설정 → 없으면 파일 선택 받기
    vp = RES / "vertiport.csv"
    wp = (RES / "waypoint_vipp.csv") if (RES / "waypoint_vipp.csv").exists() else (RES / "waypoint.csv")
    if not (vp.exists() and wp.exists()):
        QMessageBox.information(main, "참고", "vertiport/waypoint CSV를 지정해 주세요.")
        vp_p, _ = QFileDialog.getOpenFileName(main, "Vertiport CSV", str(Path.cwd()), "CSV (*.csv)")
        wp_p, _ = QFileDialog.getOpenFileName(main, "Waypoint CSV",  str(Path.cwd()), "CSV (*.csv)")
        if not (vp_p and wp_p):
            return
        vp, wp = Path(vp_p), Path(wp_p)

    # 3) 메인과 동일한 위치/크기로 트래픽 뷰어 열기
    geom  = main.geometry()
    state = main.windowState()

    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        # 중복 방지
        if getattr(main, "_traffic_win", None):
            try:
                main._traffic_win.close()
            except:
                pass

        win = TrafficTab(parent=None)
        # 파일 경로 주입 후 로드/렌더
        win.track_csv = trk
        win.vp_csv    = vp
        win.wp_csv    = wp
        try:
            win._load_all()
            win._build_map_base()
        except Exception as e:
            print("TrafficTab reload warning:", e)

        # back → 메인 복귀
        def _back():
            try:
                win.close()
            except:
                pass
            setattr(main, "_traffic_win", None)
            main.show()

        win.backRequested.connect(_back)
        win.destroyed.connect(lambda: setattr(main, "_traffic_win", None))

        win.setGeometry(geom)
        if state & Qt.WindowMaximized:
            win.showMaximized()
        else:
            win.show()

        main._traffic_win = win
        main.hide()
    finally:
        QApplication.restoreOverrideCursor()

        
# ---- 이하 기존 placeholder 유지(스택용 작은 페이지) ----
class TrafficPage(QWidget):
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
        title = QLabel("혼잡 해석"); title.setStyleSheet("color:#ffffff; font-size:28px; font-weight:800; background:transparent;")
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
