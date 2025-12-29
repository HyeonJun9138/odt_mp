# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QFrame, QLabel, QHBoxLayout, QSizePolicy, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt
from styles import BUBBLE_MINE, BUBBLE_PEER

class ChatPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ✅ 반투명 흰색 + 둥근 모서리
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 150);
                border-radius: 12px;
            }
        """)

        # 그림자 효과
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 2)
        shadow.setColor(Qt.gray)
        self.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("""
        QScrollArea { border: none; background: transparent; }
        QWidget#qt_scrollarea_viewport { background: transparent; }  /* ✅ 뷰포트 배경 제거 */
        """)

        root.addWidget(self.scroll)

        self.wrap = QFrame()
        self.wrap.setStyleSheet("background: transparent;")
        self.scroll.setWidget(self.wrap)

        self.vbox = QVBoxLayout(self.wrap)
        self.vbox.setContentsMargins(10,10,10,10)
        self.vbox.setSpacing(8)
        self.vbox.addStretch(1)

    def add_message(self, text: str, mine: bool):
        line = QHBoxLayout()
        line.setContentsMargins(0,0,0,0)
        line.setSpacing(0)

        bubble = QFrame()
        lay = QVBoxLayout(bubble)
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(0)
        lbl = QLabel(str(text))
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(lbl)

        if mine:
            bubble.setStyleSheet(BUBBLE_MINE)
            line.addStretch(1)
            line.addWidget(bubble, 0, Qt.AlignRight)
        else:
            bubble.setStyleSheet(BUBBLE_PEER)
            line.addWidget(bubble, 0, Qt.AlignLeft)
            line.addStretch(1)

        bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        bubble.setMaximumWidth(int(self.width()*0.9))

        idx = self.vbox.count()-1
        self.vbox.insertLayout(idx, line)
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())
        return idx  # 새 말풍선의 위치 인덱스 반환

    def replace_message(self, index: int, new_text: str, mine: bool):
        # 기존 말풍선 삭제 후 새 말풍선 추가
        item = self.vbox.itemAt(index)
        if not item or not item.layout():
            return
        lay = item.layout()
        # 말풍선(QFrame) 위젯 찾아 교체
        for j in range(lay.count()):
            w = lay.itemAt(j).widget()
            if isinstance(w, QFrame):
                for c in w.children():
                    if isinstance(c, QLabel):
                        c.setText(new_text)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        for i in range(self.vbox.count()-1):
            item = self.vbox.itemAt(i)
            if item and item.layout():
                lay = item.layout()
                for j in range(lay.count()):
                    w = lay.itemAt(j).widget()
                    if isinstance(w, QFrame):
                        w.setMaximumWidth(int(self.width()*0.9))
