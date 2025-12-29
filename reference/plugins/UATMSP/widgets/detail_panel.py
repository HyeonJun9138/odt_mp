# widgets/detail_panel.py
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QGraphicsDropShadowEffect, \
    QScrollArea, QWidget, QHBoxLayout, QToolButton, QLayout, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPoint, QSize
from PyQt5.QtGui import QFont
import sip  # ✅ 삭제된 QObject 방어용

ACCENT = "#2f77ff"   # styles.py와 톤 맞춤(하드코딩)
PILL_BG = "#e8f1ff"  # 옅은 파랑
PILL_BG_H = "#dfeaff"
TEXT_SUB = "#3a4151"

# ─────────────────────────────────────────
# FlowLayout: 칩이 줄바꿈되며 흐르는 레이아웃
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing

    def addItem(self, item): self._items.append(item)
    def count(self): return len(self._items)
    def itemAt(self, index): return self._items[index] if 0 <= index < len(self._items) else None
    def takeAt(self, index): return self._items.pop(index) if 0 <= index < len(self._items) else None
    def expandingDirections(self): return Qt.Orientations(Qt.Orientation(0))
    def hasHeightForWidth(self): return True
    def heightForWidth(self, width): return self._do_layout(QRect(0,0,width,0), test_only=True)
    def setGeometry(self, rect): super().setGeometry(rect); self._do_layout(rect, test_only=False)
    def sizeHint(self): return self.minimumSize()
    def minimumSize(self):
        s = QSize(0,0)
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        s += QSize(m.left()+m.right(), m.top()+m.bottom())
        return s

    def _do_layout(self, rect, test_only):
        x, y = rect.x(), rect.y()
        line_height = 0
        m = self.contentsMargins()
        x += m.left(); y += m.top()
        max_w = rect.width() - m.left() - m.right()
        for it in self._items:
            w = it.sizeHint().width()
            h = it.sizeHint().height()
            if (x - rect.x() + w) > max_w and line_height > 0:
                x = rect.x() + m.left()
                y += line_height + self._spacing
                line_height = 0
            if not test_only:
                it.setGeometry(QRect(QPoint(x, y), it.sizeHint()))
            x += w + self._spacing
            line_height = max(line_height, h)
        return y + line_height + m.bottom() - rect.y()

ACCENT = "#2f77ff"
PILL_BG = "#e8f1ff"
PILL_BG_H = "#dfeaff"
TEXT_SUB = "#3a4151"

class _Pill(QToolButton):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)  # ✅ 포커스 링 제거
        self.setStyleSheet(f"""
            QToolButton {{
                background: {PILL_BG}; color: {ACCENT};
                border: 0; border-radius: 10px;
                padding: 2px 8px; font-weight: 600;
            }}
            QToolButton:hover {{ background: {PILL_BG_H}; }}
        """)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        
class _ActionItem(QFrame):
    clicked = pyqtSignal(dict)
    tokenClicked = pyqtSignal(dict, str)

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.data = data
        self._selected = False

        self.setObjectName("ActionCard")          # ✅ 스코프용 이름
        self.setFocusPolicy(Qt.NoFocus)           # ✅ 카드 자체 포커스 제거
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            #ActionCard { background: #ffffff; border: 2px solid transparent; border-radius: 10px; }
            QLabel { color:#0e1726; background: transparent; }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10,8,10,8)
        lay.setSpacing(6)

        # 1) 액션 라인 (굵게)
        act = QLabel(str(data.get("action","")).strip())
        f = QFont(); f.setBold(True)
        act.setFont(f)
        act.setWordWrap(True)
        lay.addWidget(act)

        # 2) 칩(쉼표 분해)
        chips_wrap = QWidget(self)
        chips = FlowLayout(chips_wrap, spacing=6)
        act_text = act.text()
        tokens = [t.strip() for t in act_text.split(",") if t.strip()]
        for t in tokens:
            pill = _Pill(t, chips_wrap)
            pill.clicked.connect(lambda _, tok=t: self.tokenClicked.emit(self.data, tok))
            chips.addWidget(pill)
        lay.addWidget(chips_wrap)

        # 3) why (서브텍스트)
        why = QLabel(str(data.get("why","")).strip())
        why.setStyleSheet(f"color:{TEXT_SUB};")
        why.setWordWrap(True)
        lay.addWidget(why)

        # hover 그림자
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(18); self._shadow.setOffset(0, 2)
        self._shadow.setColor(Qt.gray)
        self.setGraphicsEffect(self._shadow)
        self._shadow.setEnabled(False)

    def enterEvent(self, e):
        self._shadow.setEnabled(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._shadow.setEnabled(False)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self.clicked.emit(self.data)
        super().mousePressEvent(e)

    def setSelected(self, on: bool):
        self._selected = on
        border = f"2px solid {ACCENT}" if on else "2px solid transparent"
        self.setStyleSheet(f"""
            #ActionCard {{ background: #ffffff; border: {border}; border-radius: 10px; }}
            QLabel {{ color:#0e1726; background: transparent; }}
        """)

class DetailPanel(QFrame):
    # ✅ 액션 전체 카드 선택 + 특정 토큰 클릭 신호
    actionSelected = pyqtSignal(dict)
    tokenClicked = pyqtSignal(dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame { background-color: rgba(255, 255, 255, 217); border-radius: 12px; }
            QLabel { background: transparent; }
        """)
        shadow = QGraphicsDropShadowEffect(self); shadow.setBlurRadius(20); shadow.setOffset(0,2); shadow.setColor(Qt.gray)
        self.setGraphicsEffect(shadow)

        self._cards = []
        self._sel = None

        self.root = QVBoxLayout(self); self.root.setContentsMargins(14,14,14,14); self.root.setSpacing(8)
        self.title = QLabel("세부 정보"); self.title.setStyleSheet("color:#0e1726; font-size:16px; font-weight:600;")
        self.root.addWidget(self.title)

        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea{border:0;background:transparent;} QWidget#qt_scrollarea_viewport{background:transparent;}")
        self.root.addWidget(self.scroll, 1)

        self.wrap = QWidget(); self.scroll.setWidget(self.wrap)
        self.vbox = QVBoxLayout(self.wrap); self.vbox.setContentsMargins(0,0,0,0); self.vbox.setSpacing(8); self.vbox.addStretch(1)

    def _clear_cards(self):
        self._cards.clear()
        self._sel = None            # ✅ 삭제된 포인터 참조 방지
        while self.vbox.count() > 1:
            it = self.vbox.takeAt(0)
            if it.widget(): it.widget().deleteLater()
            elif it.layout(): it.layout().deleteLater()

    def set_actions(self, actions: list[dict]):
        self._clear_cards()
        if not actions:
            empty = QLabel("표시할 조치가 없습니다.")
            empty.setStyleSheet("color:#3a4151;")
            self.vbox.insertWidget(0, empty)
            return

        for a in actions:
            card = _ActionItem(a, self)
            card.clicked.connect(self._on_card_clicked)
            card.tokenClicked.connect(self.tokenClicked.emit)
            self._cards.append(card)
            self.vbox.insertWidget(self.vbox.count()-1, card)

    def _on_card_clicked(self, data: dict):
        sender = self.sender()
        # 스톨 클릭(이미 지워진 카드 신호) 무시
        if sender not in self._cards:
            return

        # 이전 선택 해제 (살아있고, 현재 카드목록에 있을 때만)
        if self._sel and self._sel is not sender:
            try:
                if (self._sel in self._cards) and (not sip.isdeleted(self._sel)):
                    self._sel.setSelected(False)
            except RuntimeError:
                pass

        self._sel = sender
        sender.setSelected(True)
        self.actionSelected.emit(data)
