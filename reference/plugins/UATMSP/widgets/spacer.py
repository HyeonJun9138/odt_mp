# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget

class VSpacer(QWidget):
    """테이블 한 행을 차지하는 간격 위젯"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
