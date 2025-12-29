#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
from pathlib import Path
from PyQt5.QtCore import Qt, QSize,QUrl          
from PyQt5.QtGui import QPixmap, QIcon,QDesktopServices  
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QToolButton, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QMessageBox, QSizePolicy, QStackedLayout,
     QInputDialog, QFileDialog
)

from buttons.upload import UploadPage
from buttons.noise import NoisePage
from buttons.analyze import AnalyzePage
from buttons.trash import request_delete_all
from buttons.restart import request_restart

from buttons.traffic import TrafficPage, launch_traffic_view
from Tabs.Noise_tab import NoiseTab
from buttons.noise import launch_noise_view
from pathlib import Path

from Tabs.Traffic_tab import TrafficTab
import os
from Analyzer.Functions.config_models import set_openai_api_key


class DashboardWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KADA – Analysis Dashboard")
        self.resize(1500, 900)
        self._setup_ui()
        # 1) 앱 구동 시 초기값(환경변수 우선) 반영
        _init_api = os.environ.get("OPENAI_API_KEY", "")
        try:
            # API Key 입력 위젯에 초기값 세팅(있다면)
            self.api_key_edit.setText(_init_api)
        except Exception:
            # 위젯이 없다면 건너뜀
            pass
        set_openai_api_key(_init_api)

        # 2) 사용자가 입력을 바꾸면 전역에 즉시 반영
        def _on_api_key_changed(text: str):
            set_openai_api_key(text)

        # QLineEdit가 존재한다면 textChanged 시그널 연결
        try:
            self.api_key_edit.textChanged.connect(_on_api_key_changed)
        except Exception:
            pass
        self._connect_signals()
        self._noise_win = None
        self.registered_files = []   # 업로드 페이지에서 채워줌 (on_files_registered)
    # ─────────────────────────────────────────────────────────────────────
    # 아이콘(위) + 텍스트(아래) 버튼
    # ─────────────────────────────────────────────────────────────────────
    def _make_tile_button(self, text: str, bg: str, icon_name: str, icon_size: QSize = QSize(170, 170)) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setIconSize(icon_size)
        btn.setMinimumSize(300, 250)
        btn.setCursor(Qt.PointingHandCursor)

        icon_path = Path(__file__).parent / "Sources" / icon_name
        if icon_path.exists():
            btn.setIcon(QIcon(str(icon_path)))

        btn.setStyleSheet(f"""
            QToolButton {{
                background-color:{bg};
                color:#ffffff; font-size:24px; font-weight:700;
                border:none; border-radius:12px;
                padding:18px 12px 16px 12px;
            }}
            QToolButton::menu-indicator {{ image: none; }}
        """)
        return btn

    # ─────────────────────────────────────────────────────────────────────
    # UI 구축
    # ─────────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        self.resize(1500, 900)

        root = QWidget(self)
        self.setCentralWidget(root)

        # 다크 톤 + 흰글씨 스타일
        root.setStyleSheet("""
            QWidget { background-color:#2f3440; }
            QFrame#TitleCard   { background-color:#2f3440; border-radius:10px; }
            QFrame#RightCard   { background-color:#3a404f; border-radius:12px; }
            QFrame#NoticeCard  { background-color:#3a404f; border-radius:12px; }
            QFrame#ButtonsWrap { background-color:transparent; }
            QFrame#APIBar      { background-color:#3a404f; border-radius:12px; }

            QLabel#SectionTitle{ color:#ffffff; font-size:42px; font-weight:800; background-color:transparent; }
            QLabel#NoticeText  { color:#ffffff; font-size:28px; font-weight:700; background-color:transparent; }
            QLabel#InfoKey     { color:#ffffff; font-size:18px; background-color:transparent; }
            QLabel#InfoVal     { color:#ffffff; font-size:20px; font-weight:600; background-color:transparent; }

            QListWidget, QTableWidget {
                color:#ffffff; background-color:#3a404f;
                border:1px solid rgba(255,255,255,0.12);
            }
            QHeaderView::section {
                color:#ffffff; background-color:#586073;
                border:none; padding:6px;
            }
            QTableWidget::item:selected, QListWidget::item:selected {
                background:#586073; color:#ffffff;
            }

            QLineEdit#ApiEdit  { background:#2f3440; color:#ffffff; border:1px solid rgba(255,255,255,0.2); border-radius:8px; padding:12px; font-size:18px; }
        """)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(14)

        # 상단 로고
        topbar = QHBoxLayout()
        self.logo = QLabel(); self.logo.setFixedHeight(64)
        self._load_logo(self.logo)
        topbar.addWidget(self.logo, 0, Qt.AlignLeft | Qt.AlignVCenter)
        topbar.addStretch(1)
        outer.addLayout(topbar)

        # 본문 그리드
        grid = QGridLayout(); grid.setHorizontalSpacing(14); grid.setVerticalSpacing(14)
        outer.addLayout(grid, 1)

        grid.setColumnStretch(0, 5)   # 좌(Notice)
        grid.setColumnStretch(1, 8)   # 중(스택: 메인/업로드)
        grid.setColumnStretch(2, 4)   # 우(정보)
        grid.setRowStretch(0, 0)      # Title
        grid.setRowStretch(1, 0)      # 헤더
        grid.setRowStretch(2, 1)      # 본문

        # Row 0: Title(좌+중)
        self.title_card = QFrame(); self.title_card.setObjectName("TitleCard")
        tl = QHBoxLayout(self.title_card); tl.setContentsMargins(0, 0, 0, 0)
        tl.addStretch(1)  # 내용 없음(영역만 유지)
        self.title_card.setFixedHeight(68)
        grid.addWidget(self.title_card, 0, 0, 1, 2)

        # Row 0~1: 우측 정보 패널
        self.info_card = QFrame(); self.info_card.setObjectName("RightCard")
        il = QVBoxLayout(self.info_card); il.setContentsMargins(24, 24, 24, 24); il.setSpacing(18)
        def info_pair(key, val):
            kh = QLabel(key); kh.setObjectName("InfoKey")
            vh = QLabel(val); vh.setObjectName("InfoVal")
            line = QHBoxLayout(); line.addWidget(kh); line.addStretch(1); line.addWidget(vh, 0, Qt.AlignRight)
            il.addLayout(line); return vh
        self.ver_val = info_pair("프로그램버전", "V0.0.1")
        self.upd_val = info_pair("업데이트 날짜", "")
        self.exp_val = info_pair("사용가능일자", "~2025.08.20")
        il.addStretch(1)
        grid.addWidget(self.info_card, 0, 2, 2, 1)

        # Row 1: 좌측 섹션 타이틀
        notice_title = QLabel("Notice"); notice_title.setObjectName("SectionTitle")
        grid.addWidget(notice_title, 1, 0, 1, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        # Row 2: 좌측 Notice 카드
        self.notice_panel = QFrame(); self.notice_panel.setObjectName("NoticeCard")
        nlay = QVBoxLayout(self.notice_panel); nlay.setContentsMargins(24, 24, 24, 24)
        nlay.addStretch(1)
        nlabel = QLabel("공지사항 및 업데이트 내용"); nlabel.setObjectName("NoticeText"); nlabel.setAlignment(Qt.AlignCenter)
        nlay.addWidget(nlabel)
        nlay.addStretch(1)
        self.notice_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.notice_panel, 2, 0, 1, 1)

        # Row 2: 중앙(스택 레이아웃: 메인/업로드/트래픽/노이즈/분석/삭제)
        self.center = QFrame(); self.center.setObjectName("ButtonsWrap")
        self.center_stack = QStackedLayout(self.center)

        self.page_main    = self._create_center_main_page()
        self.page_upload  = UploadPage(self)
        
        self.page_traffic = TrafficTab(self)
        
        self.page_noise   = NoisePage(self)
        self.page_analyze = AnalyzePage(self)

        self.center_stack.addWidget(self.page_main)
        self.center_stack.addWidget(self.page_upload)
        self.center_stack.addWidget(self.page_traffic)
        self.center_stack.addWidget(self.page_noise)
        self.center_stack.addWidget(self.page_analyze)

        self.center_stack.setCurrentWidget(self.page_main)
        grid.addWidget(self.center, 2, 1, 1, 2)
        
        self.page_traffic.backRequested.connect(lambda: self.center_stack.setCurrentWidget(self.page_main))

    # 중앙: 버튼 그리드 + API 바
    def _create_center_main_page(self) -> QWidget:
        page = QWidget()
        cl = QVBoxLayout(page); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(12)

        btn_grid = QGridLayout(); btn_grid.setHorizontalSpacing(16); btn_grid.setVerticalSpacing(16)
        btn_grid.setRowStretch(0, 1); btn_grid.setRowStretch(1, 1)

        self.btn_upload      = self._make_tile_button("분석 데이터 관리", "#e4572e", "upload.png")
        self.btn_load_prev   = self._make_tile_button("비교 데이터 관리", "#b35a00", "history.png")
        self.btn_delete_up   = self._make_tile_button("업로드 데이터 삭제", "#39a845", "trash.png")
        self.btn_request_all = self._make_tile_button("종합 분석",   "#2e8b3d", "analyze.png")
        self.btn_congestion  = self._make_tile_button("혼잡 해석",   "#7057a3", "traffic.png")
        self.btn_noise       = self._make_tile_button("소음 해석",   "#6250b8", "noise.png")
        self.btn_restart     = self._make_tile_button("프로그램 재시작",   "#1e88e5", "restart.png")
        self.btn_quit        = self._make_tile_button("프로그램 종료",     "#4a70c2", "power.png")

        btn_grid.addWidget(self.btn_upload,      0, 0)
        btn_grid.addWidget(self.btn_load_prev,   0, 1)
        btn_grid.addWidget(self.btn_delete_up,   0, 2)
        btn_grid.addWidget(self.btn_request_all, 0, 3)
        btn_grid.addWidget(self.btn_congestion,  1, 0)
        btn_grid.addWidget(self.btn_noise,       1, 1)
        btn_grid.addWidget(self.btn_restart,     1, 2)
        btn_grid.addWidget(self.btn_quit,        1, 3)
        cl.addLayout(btn_grid, 1)

        # API 바
        self.api_bar = QFrame(); self.api_bar.setObjectName("APIBar"); self.api_bar.setFixedHeight(72)
        apil = QHBoxLayout(self.api_bar); apil.setContentsMargins(20, 12, 20, 12); apil.setSpacing(12)
        api_label = QLabel("GPT API"); api_label.setStyleSheet("color:#ffffff; font-size:28px; font-weight:800; background-color:transparent;")
        self.api_edit = QLineEdit(); self.api_edit.setObjectName("ApiEdit"); self.api_edit.setPlaceholderText("API Key를 입력하세요")
        apil.addWidget(api_label, 0, Qt.AlignVCenter); apil.addWidget(self.api_edit, 1)
        cl.addWidget(self.api_bar, 0)

        return page

    # ─────────────────────────────────────────────────────────────────────
    # 시그널 연결
    # ─────────────────────────────────────────────────────────────────────
    def _connect_signals(self):
        # 메인 타일 버튼
        pairs = [
            ("btn_upload",      self.on_upload_clicked),
            ("btn_load_prev",   self.on_load_prev_clicked),
            ("btn_delete_up",   self.on_delete_upload_clicked),
            ("btn_request_all", self.on_request_all_clicked),
            ("btn_congestion",  self.on_congestion_clicked),
            ("btn_noise",       self.on_noise_clicked),
            ("btn_restart",     self.on_restart_clicked),
            ("btn_quit",        self.on_quit_clicked),
        ]
        for name, slot in pairs:
            btn = getattr(self, name, None)
            if btn is not None:
                btn.clicked.connect(slot)

        # ★ API 키 실시간 저장
        if hasattr(self, "api_edit"):
            self.api_edit.textChanged.connect(set_openai_api_key)

        # 업로드 페이지
        self.page_upload.closed.connect(lambda: self.center_stack.setCurrentWidget(self.page_main))
        self.page_upload.registered.connect(self.on_files_registered)

        # 트래픽/노이즈/분석/삭제 페이지
        self.page_traffic.backRequested.connect(lambda: self.center_stack.setCurrentWidget(self.page_main))
        self.page_noise.closed.connect(lambda: self.center_stack.setCurrentWidget(self.page_main))
        self.page_analyze.closed.connect(lambda: self.center_stack.setCurrentWidget(self.page_main))

    # 업로드 완료 콜백
    def on_files_registered(self, files: list):
        self.registered_files = files
        QMessageBox.information(self, "등록", f"{len(files)}개 파일 등록 완료.")
        # ▼ 추가: AnalyzePage에 목록 주입
        try:
            self.page_analyze.set_registered_files(files)
        except Exception:
            pass
        self.center_stack.setCurrentWidget(self.page_main)

        # 로고 로드 (Analyzer/Sources/kada.png)
    def _load_logo(self, label: QLabel):
        logo_path = Path(__file__).parent / "Sources" / "kada.png"
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                label.setPixmap(pix.scaledToHeight(64, Qt.SmoothTransformation))
    
        else:
            label.setText("KADA")
            label.setStyleSheet("color:#ffffff; font-size: 28px; font-weight: 900;")
            

        # ★ 공통 클릭 핸들러 (반환값 없음!)
        label.setCursor(Qt.PointingHandCursor)
        def _open_kada(e):
            QDesktopServices.openUrl(QUrl("https://kada.konkuk.ac.kr"))
            e.accept()
            # return 값 없음 (None)

        label.mousePressEvent = _open_kada
    # ─────────────────────────────────────────────────────────────────────
    # 슬롯: 버튼 동작
    # ─────────────────────────────────────────────────────────────────────
    def on_upload_clicked(self):
        self.center_stack.setCurrentWidget(self.page_upload)

    def on_load_prev_clicked(self):
        QMessageBox.information(self, "지난 분석 불러오기", "이전 분석 기록을 로드합니다.")

    def on_delete_upload_clicked(self):
        files = getattr(self, "registered_files", [])
        if request_delete_all(self, len(files)):
            # 실제 물리 파일을 지우고 싶으면 여기서 처리 (예: os.remove)
            
            self.registered_files = []
            QMessageBox.information(self, "삭제", "등록된 업로드 데이터를 모두 삭제했습니다.")


    def on_request_all_clicked(self):
        # 종합 분석 (빈 페이지로 전환)
        self.center_stack.setCurrentWidget(self.page_analyze)
        
    def on_congestion_clicked(self):
        launch_traffic_view(self)

    def on_noise_clicked(self):
        launch_noise_view(self)   
        
    def on_restart_clicked(self):
        # 별도 모듈로 분리된 재시작 로직
        request_restart(self)

    def on_quit_clicked(self):
        QApplication.quit()
        
    def _on_noise_back(self):
        if not getattr(self, "_noise_win", None):
            self.show(); return

        # NoiseTab의 상태/크기 가져와서 메인에 반영
        n_state = self._noise_win.windowState()
        n_geom  = self._noise_win.frameGeometry()

        try:
            self._noise_win.close()
        finally:
            self._noise_win = None

        if n_state & Qt.WindowMaximized:
            self.showMaximized()
        else:
            self.setGeometry(n_geom)
            self.show()

    # main.py 클래스 내부에 추가 (Noise용 _on_noise_back와 아주 유사)
    def _on_congestion_back(self):
        if not getattr(self, "_traffic_win", None):
            self.show(); return

        n_state = self._traffic_win.windowState()
        n_geom  = self._traffic_win.frameGeometry()

        try:
            self._traffic_win.close()
        finally:
            self._traffic_win = None

        if n_state & Qt.WindowMaximized:
            self.showMaximized()
        else:
            self.setGeometry(n_geom)
            self.show()


def main():
    app = QApplication(sys.argv)

    # --- [추가] 모든 파일 열기/저장 대화상자의 기본 폴더를 Analyzer\database 로 고정 ---
    from PyQt5.QtCore import QDir
    from pathlib import Path
    import os
    base_dir = Path(__file__).resolve().parent      # .../Analyzer
    db_dir   = base_dir / "database"                # .../Analyzer/database
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    QDir.setCurrent(str(db_dir))    # Qt 기준 현재 디렉터리
    os.chdir(str(db_dir))           # 네이티브 대화상자 대비(보조)
    # ---------------------------------------------------------------------

    w = DashboardWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
