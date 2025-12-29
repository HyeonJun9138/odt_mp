# widgets/display_panel.py
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QGraphicsDropShadowEffect, QScrollArea, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

ACCENT = "#2f77ff"
TEXT_SUB = "#3a4151"

class DisplayPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame { background-color: rgba(255, 255, 255, 217); border-radius: 12px; }
            QLabel { background: transparent; }
        """)
        shadow = QGraphicsDropShadowEffect(self); shadow.setBlurRadius(20); shadow.setOffset(0,2); shadow.setColor(Qt.gray)
        self.setGraphicsEffect(shadow)

        self.root = QVBoxLayout(self); self.root.setContentsMargins(14,14,14,14); self.root.setSpacing(8)
        self.title = QLabel("Display Panel")
        self.title.setStyleSheet("color:#0e1726; font-size:16px; font-weight:600;")
        self.root.addWidget(self.title)

        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea{border:0;background:transparent;} QWidget#qt_scrollarea_viewport{background:transparent;}")
        self.root.addWidget(self.scroll, 1)

        self.wrap = QWidget(); self.scroll.setWidget(self.wrap)
        self.vbox = QVBoxLayout(self.wrap); self.vbox.setContentsMargins(0,0,0,0); self.vbox.setSpacing(8); self.vbox.addStretch(1)

    def _clear(self):
        while self.vbox.count() > 1:
            it = self.vbox.takeAt(0)
            if it.widget(): it.widget().deleteLater()
            elif it.layout(): it.layout().deleteLater()

    def visualize_action(self, action: dict, focus_token: str | None = None):
        """세부정보에서 선택된 action을 시각화(요약 표시, 향후 지도/경로 등으로 확장)"""
        self._clear()

        def add_label(text, style=""):
            lab = QLabel(text); 
            if style: lab.setStyleSheet(style)
            lab.setWordWrap(True); 
            self.vbox.insertWidget(self.vbox.count()-1, lab)
            return lab

        act = str(action.get("action","")).strip()
        why = str(action.get("why","")).strip()

        # 제목
        t = QLabel("선택한 조치")
        f = QFont(); f.setBold(True); t.setFont(f)
        self.vbox.insertWidget(self.vbox.count()-1, t)

        # 액션 본문
        if focus_token:
            # 포커스 토큰 강조
            highlighted = []
            for seg in [s.strip() for s in act.split(",") if s.strip()]:
                if seg == focus_token:
                    highlighted.append(f'<span style="background:#fff3cd;border-radius:6px;padding:2px 4px;">{seg}</span>')
                else:
                    highlighted.append(seg)
            act_html = ", ".join(highlighted)
            add_label(act_html, "color:#0e1726;")
        else:
            add_label(act, "color:#0e1726;")

        # why
        add_label(why, f"color:{TEXT_SUB};")

        # 원문 미니카드
        raw = QLabel(f"<pre style='white-space:pre-wrap; font-family:Consolas;'>{action}</pre>")
        raw.setStyleSheet("color:#0e1726;")
        raw.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.vbox.insertWidget(self.vbox.count()-1, raw)

        # TODO: 여기에서 실제 시각화(TBD: 경로/기동 벡터, 분리거리 오버레이 등)를 얹습니다.
