# -*- coding: utf-8 -*-
"""
종합 분석 내부창(메인 위젯)
- 모델 선택 콤보
- 모드 선택(체크 2개, 서로 배타)
- 우측 버튼 영역: [분석 시작], [Save as] (후속 구현 연결 예정)
- 신호: start_requested, save_requested
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QComboBox, QCheckBox, QButtonGroup, QSpacerItem, QSizePolicy
)

from Analyzer.Functions.config_models import MODEL_IDS
from Analyzer.prompt_paths import prompt_for_mode
import os

class AnalyzePanel(QWidget):
    """
    종합 분석 내부창 위젯
    - 외부창(main)에서 addWidget하여 사용
    - 스타일은 기존 QSS(#RightCard 등)에 자연스레 녹도록 objectName 지정
    """
    start_requested = pyqtSignal(str, str)   # (model, mode)
    save_requested  = pyqtSignal()          # Save as 버튼

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._connect()

    # ─────────────────────────────────────────
    # UI 구성
    # ─────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        card = QFrame()
        card.setObjectName("RightCard")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(16)

        # 좌측: 설정 컬럼
        left = QFrame()
        left.setObjectName("LeftColumn")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(12)

        # 제목
        title = QLabel("종합 분석")
        title.setStyleSheet("color:#ffffff; font-size:24px; font-weight:700;")
        left_lay.addWidget(title)

        # 구분선
        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine); sep1.setStyleSheet("color:#3c4250;")
        left_lay.addWidget(sep1)

        # 섹션: 모델 선택
        lbl_model = QLabel("모델 선택")
        lbl_model.setStyleSheet("color:#cfd6e4; font-size:12px;")
        self.cb_model = QComboBox()
        self.cb_model.setObjectName("ModelCombo")
        self.cb_model.addItems(MODEL_IDS)
        self.cb_model.setCurrentIndex(0)
        self.cb_model.setStyleSheet("QComboBox { padding:6px 8px; }")
        left_lay.addWidget(lbl_model)
        left_lay.addWidget(self.cb_model)

        # 섹션: 모드 선택(체크 2개, 서로 배타)
        lbl_mode = QLabel("모드 선택")
        lbl_mode.setStyleSheet("color:#cfd6e4; font-size:12px;")
        left_lay.addWidget(lbl_mode)

        self.chk_single = QCheckBox("단일 시나리오 분석")
        self.chk_compare = QCheckBox("다중 시나리오 비교")
        for c in (self.chk_single, self.chk_compare):
            c.setCursor(Qt.PointingHandCursor)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)   # 체크박스를 라디오처럼 배타 선택
        self._mode_group.addButton(self.chk_single)
        self._mode_group.addButton(self.chk_compare)
        self.chk_single.setChecked(True)      # 기본값: 단일

        left_lay.addWidget(self.chk_single)
        left_lay.addWidget(self.chk_compare)

        left_lay.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 우측: 버튼 컬럼
        right = QFrame()
        right.setObjectName("RightColumn")
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        self.btn_start = QPushButton("분석 시작")
        self.btn_save  = QPushButton("Save as")
        for b in (self.btn_start, self.btn_save):
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(40)
            b.setStyleSheet("""
                QPushButton {
                    background:#55607a; color:#e9eef7; border:0; border-radius:10px;
                    font-weight:600;
                }
                QPushButton:hover { opacity:.95; }
                QPushButton:pressed { opacity:.90; }
            """)

        right_lay.addWidget(self.btn_start)
        right_lay.addWidget(self.btn_save)
        right_lay.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        lay.addWidget(left, 1)
        lay.addWidget(right, 0)
        outer.addWidget(card)

    # ─────────────────────────────────────────
    # 시그널 연결
    # ─────────────────────────────────────────
    def _connect(self):
        self.btn_start.clicked.connect(self._on_click_start)
        self.btn_save.clicked.connect(self.save_requested.emit)

    # ─────────────────────────────────────────
    # 헬퍼
    # ─────────────────────────────────────────
    def selected_model(self) -> str:
        return self.cb_model.currentText()

    def selected_mode(self) -> str:
        return "compare" if self.chk_compare.isChecked() else "single"

    def current_prompt_path(self):
        return prompt_for_mode(self.selected_mode())

    # ─────────────────────────────────────────
    # 이벤트 핸들러
    # ─────────────────────────────────────────
    def _on_click_start(self):
        model = self.selected_model()
        mode  = self.selected_mode()
        self.start_requested.emit(model, mode)

        # ↓↓↓ 여기를 실제 호출로 교체 (예시: 더미 상태 변수에서 값 취득)
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            # 앱 전역에 보관된 키를 쓰는 경우: self._api_key 등으로 교체
            # api_key = self._api_key
            pass

        # 모드별 프롬프트 경로
        prompt_path = self.current_prompt_path()

        # 프롬프트 치환 변수 구성(앱의 실제 상태 값을 넣어주세요)
        if mode == "single":
            vars_dict = {
                "analysis_window": getattr(self, "_analysis_window", ""),
                "region_scope": getattr(self, "_region_text", ""),
                "audience": getattr(self, "_audience_choice", "임원"),
                "objectives": "\n".join(getattr(self, "_objectives_list", [])),
                "source_text": getattr(self, "_merged_source_text", ""),
                "report_title": getattr(self, "_report_title", "UAM 통합 분석 보고서"),
                "hotspot_min": "6",
                "hotspot_max": "10",
            }
        else:  # compare
            vars_dict = {
                "analysis_window": getattr(self, "_analysis_window", ""),
                "region_scope": getattr(self, "_region_text", ""),
                "audience": getattr(self, "_audience_choice", "임원"),
                "report_title": getattr(self, "_report_title", "UAM 다중 시나리오 비교 보고서"),
                "scenario_summaries": getattr(self, "_scenario_summaries_text", ""),
                "key_metrics_table_like_text": getattr(self, "_key_metrics_lines_text", ""),
            }

        try:
            result_text = self._runner.run(cfg)
        except Exception as e:
            result_text = f"[LLM 호출 실패] {e}"

        # 결과는 상위 출력 뷰로 올리거나, 임시로 콘솔에 출력
        # 상위 창 신호/슬롯 설계에 맞게 아래 라인을 바꾸세요.
        print("\n=== LLM RESULT ===\n", result_text)
