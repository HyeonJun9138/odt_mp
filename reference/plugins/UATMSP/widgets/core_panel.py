# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QGraphicsDropShadowEffect, \
    QScrollArea, QWidget, QHBoxLayout, QToolButton
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPainter, QColor, QPen
from PyQt5.QtCore import QRectF

class BusyIndicator(QWidget):
    def __init__(self, size=50, line=3, parent=None):
        super().__init__(parent)
        self._ang = 0
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self._size, self._line = size, line
        self.setFixedSize(size, size)
        self.hide()

    def start(self):
        self.show(); self.raise_(); self._timer.start()

    def stop(self):
        self._timer.stop(); self.hide()

    def _tick(self):
        self._ang = (self._ang + 30) % 360
        self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = min(self.width(), self.height())/2 - 2
        p.translate(self.rect().center())
        p.rotate(self._ang)
        p.setPen(QPen(QColor(47,119,255), self._line, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(-r, -r, 2*r, 2*r), 0*16, 100*16)

# ─────────────────────────────────────────
# 로딩 오버레이 (패널 위를 덮어씌움)
class _LoadingOverlay(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background: rgba(43,47,54,160); border-radius:12px;")
        self._spin = BusyIndicator(parent=self)
        self._label = QLabel("분석중", self)
        self._label.setStyleSheet("color:#e9eef7; font-weight:600;")
        self._label.setAlignment(Qt.AlignCenter)
        self.hide()

    def start(self, text="분석중"):
        self._label.setText(text)
        self._spin.start()
        self.show(); self.raise_(); self._recenter()

    def stop(self):
        self._spin.stop(); self.hide()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._recenter()

    def _recenter(self):
        # 부모 전체 크기
        pr = self.parent().rect()
        margin_x = 500  # 좌우 여백(px)
        margin_y = 0   # 상하 여백(px)

        # 여백 적용한 영역으로 overlay 크기 설정
        self.setGeometry(
            pr.x() + margin_x,
            pr.y() + margin_y,
            pr.width() - margin_x * 2,
            pr.height() - margin_y * 2
        )

        # 중앙 정렬
        c = self.rect().center()
        self._spin.move(c.x() - self._spin.width()//2, c.y() - 26)
        self._label.move(0, c.y() + 30)
        self._label.resize(self.width(), 24)

class _WarnItem(QFrame):
    clicked = pyqtSignal(dict)

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.data = data
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QFrame { background: rgba(255,255,255,0.96); border-radius: 10px; border:0; }
            QLabel { background: transparent; color:#0e1726; }
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10,8,10,8)
        lay.setSpacing(10)

        pri = (data.get("priority") or "").strip()
        level = 3 if "3" in pri else 2 if "2" in pri else 1 if "1" in pri else 0
        bg = {3:"#ffe6e6", 2:"#fff4e0", 1:"#eefbe7"}.get(level, "#eef1ff")
        fg = {3:"#c53d3d", 2:"#b57400", 1:"#2a7b2a"}.get(level, "#2f77ff")
        badge = QLabel(pri or "Level ?")
        badge.setStyleSheet(f"QLabel{{padding:4px 8px;border-radius:8px;background:{bg};color:{fg};font-weight:700;}}")
        lay.addWidget(badge, 0, Qt.AlignVCenter)

        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0,0,0,0)
        v.setSpacing(2)

        title = QLabel(str(data.get("warning","")).strip() or "경고")
        f = QFont(); f.setBold(True)
        title.setFont(f)
        title.setWordWrap(False)
        title.setToolTip(title.text())
        title.setMaximumHeight(title.fontMetrics().height()+4)

        reason = QLabel(str(data.get("reason","")).strip())
        reason.setStyleSheet("color:#3a4151;")
        reason.setWordWrap(False)
        reason.setToolTip(reason.text())
        reason.setMaximumHeight(reason.fontMetrics().height()+4)

        v.addWidget(title)
        v.addWidget(reason)
        lay.addWidget(wrap, 1)

        # hover 그림자
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(18)
        self._shadow.setOffset(0, 2)
        self._shadow.setColor(Qt.gray)

        # ✅ 효과를 한 번 붙여두고, 기본은 비활성화
        self.setGraphicsEffect(self._shadow)
        self._shadow.setEnabled(False)

    def enterEvent(self, e):
        # self.setGraphicsEffect(self._shadow)  # ❌ 지우기
        self._shadow.setEnabled(True)            # ✅ 켜기
        super().enterEvent(e)

    def leaveEvent(self, e):
        # self.setGraphicsEffect(None)           # ❌ 지우기 (이게 삭제를 유발)
        self._shadow.setEnabled(False)           # ✅ 끄기
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self.clicked.emit(self.data); super().mousePressEvent(e)

class CorePanel(QFrame):
    warningSelected = pyqtSignal(dict)  # 클릭된 경고 dict
    rawRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame { background-color: rgba(255, 255, 255, 217); border-radius: 12px; }
            QLabel { background: transparent; }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20); shadow.setOffset(0,2); shadow.setColor(Qt.gray)
        self.setGraphicsEffect(shadow)
        
        # ▼ 로딩 오버레이
        self._loading = _LoadingOverlay(self)

        self._shadow = shadow
        self._shadow_orig = shadow.color()
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(450)            # 깜빡임 주기(ms)
        self._blink_timer.timeout.connect(self._tick_blink)
        self._blink_on = False

        root = QVBoxLayout(self); root.setContentsMargins(14,14,14,14); root.setSpacing(8)

        # ✅ 헤더: 제목(좌) + 원문 버튼(우)
        header = QHBoxLayout(); header.setContentsMargins(0,0,0,0); header.setSpacing(6)
        title = QLabel("핵심 정보")
        title.setStyleSheet("color:#0e1726; font-size:18px; font-weight:600;")

        self.btnRaw = QToolButton(self)
        self.btnRaw.setText("원문")
        self.btnRaw.setCursor(Qt.PointingHandCursor)
        self.btnRaw.setToolTip("UFTM 추론 원문 보기")
        self.btnRaw.setStyleSheet("""
            QToolButton {
                background:#f0f2f5; color:#0e1726;
                border:1px solid #e3e6ed; border-radius:8px;
                padding:2px 8px; font-weight:600;
            }
            QToolButton:hover { background:#e9ecf2; }
            QToolButton:pressed { opacity:.9; }
        """)
        self.btnRaw.clicked.connect(self.rawRequested.emit)  # ✅ 클릭 → 신호

        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.btnRaw)
        root.addLayout(header)

        # 아래는 기존 스크롤/리스트 구성 그대로
        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea{border:0;background:transparent;} QWidget#qt_scrollarea_viewport{background:transparent;}")
        root.addWidget(self.scroll, 1)

        self.container = QWidget(); self.container.setStyleSheet("background: transparent;")
        self.vbox = QVBoxLayout(self.container); self.vbox.setContentsMargins(0,0,0,0); self.vbox.setSpacing(8); self.vbox.addStretch(1)
        self.scroll.setWidget(self.container)


    # ▼ 외부에서 호출할 대기 시작/종료 API
    def begin_wait(self, text="분석중"):
        self._loading.start(text)

    def end_wait(self):
        self._loading.stop()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._loading:
            self._loading.setGeometry(self.rect())

    # ▼ 추가: 깜빡임 시작
    def start_alert_blink(self):
        if self._blink_timer.isActive():
            return
        self._shadow_orig = self._shadow.color()
        self._blink_on = False
        self._blink_timer.start()

    # ▼ 추가: 깜빡임 중지 및 원복
    def stop_alert_blink(self):
        if self._blink_timer.isActive():
            self._blink_timer.stop()
        self._shadow.setColor(self._shadow_orig)

    # ▼ 추가: 타이머 틱
    def _tick_blink(self):
        self._blink_on = not self._blink_on
        self._shadow.setColor(QColor(220, 0, 0, 200) if self._blink_on else self._shadow_orig)

    def set_warnings(self, items: list[dict]):
        while self.vbox.count() > 1:
            it = self.vbox.takeAt(0)
            if it.widget(): it.widget().deleteLater()
            elif it.layout(): it.layout().deleteLater()

        for d in items:
            w = _WarnItem(d, self)
            w.clicked.connect(self.warningSelected.emit)
            self.vbox.insertWidget(self.vbox.count()-1, w)

    # 하위호환: 예전 window.py가 set_output(text)를 호출하는 경우 대비
    def set_output(self, text: str):
        self.begin_wait("분석중")
