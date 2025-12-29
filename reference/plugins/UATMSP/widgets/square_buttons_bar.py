# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget, QPushButton
from PyQt5.QtCore import Qt, QRect, pyqtSignal

class SquareButtonsBar(QWidget):
    clicked_index = pyqtSignal(int)  # 1-base 인덱스 방출

    def __init__(self, parent=None, count=10):
        super().__init__(parent)
        self.count = count
        self.setContentsMargins(0, 10, 0, 0)
        self.buttons = [QPushButton(f"{i+1}", self) for i in range(self.count)]
        for i, b in enumerate(self.buttons, start=1):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("""
                QPushButton {
                    background:#475066; color:#e9eef7;
                    border:0; border-radius:10px; font-weight:600;
                }
                QPushButton:pressed { opacity:.9; }
            """)
            b.clicked.connect(lambda _, idx=i: self.clicked_index.emit(idx))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        W, H = self.width(), self.height()
        if W <= 0 or H <= 0 or self.count <= 0: return
        m = self.contentsMargins()
        left_pad, top_pad, right_pad, bottom_pad = m.left(), m.top(), m.right(), m.bottom()
        pad = 10
        usable_w = max(0, W - left_pad - right_pad - pad*2)
        usable_h = max(0, H - top_pad - bottom_pad - pad*2)
        side = usable_h
        if self.count * side > usable_w:
            side = usable_w / self.count
        side = max(0, side)
        gap = (usable_w - side * self.count) / (self.count - 1) if self.count > 1 else 0.0
        x = left_pad + pad
        y = top_pad + (usable_h - side) / 2 + 20
        for b in self.buttons:
            b.setGeometry(QRect(int(round(x)), int(round(y)), int(round(side)), int(round(side))))
            x += side + gap
