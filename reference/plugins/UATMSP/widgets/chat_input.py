# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLineEdit, QToolButton, QPushButton
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from styles import INPUT_ROUND

class ChatInput(QFrame):
    """필 형태 입력창: 좌(+), 가운데 입력, 우(마이크), 맨오른쪽 원형 전송"""
    submitted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("InputWrap")
        self.setStyleSheet(INPUT_ROUND)

        # 살짝 그림자(선택)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 2)
        shadow.setColor(Qt.black if False else Qt.gray)  # 너무 진하면 어색 → 기본 회색
        self.setGraphicsEffect(shadow)

        lay = QHBoxLayout(self)
        # 높이를 낮추고 안쪽 여백만 살짝
        lay.setContentsMargins(12, 6, 8, 6)  # L T R B
        lay.setSpacing(6)

        # 좌측 플러스
        self.plus = QToolButton(self)
        self.plus.setObjectName("IconBtn")
        self.plus.setText("+")                     # 심플하게 텍스트 (+)
        self.plus.setFixedSize(32, 32)

        # 입력창
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText("무엇이든 물어보세요")
        self.edit.setMinimumHeight(32)

        # 우측 마이크
        self.mic = QPushButton(self)
        self.mic.setObjectName("MicBtn")
        self.mic.setText("🎤")                     # 유니코드 마이크
        self.mic.setFixedSize(32, 32)
        f = QFont(); f.setPointSize(11); self.mic.setFont(f)

        # 맨 오른쪽 원형 전송 버튼
        self.send = QPushButton(self)
        self.send.setObjectName("SendCircle")
        self.send.setText("➤")                     # 화살표 느낌
        self.send.setFixedSize(36, 36)             # 원형(지름 36)

        lay.addWidget(self.plus, 0, Qt.AlignVCenter)
        lay.addWidget(self.edit, 1, Qt.AlignVCenter)
        lay.addWidget(self.mic, 0, Qt.AlignVCenter)
        lay.addWidget(self.send, 0, Qt.AlignVCenter)

        # 동작
        self.edit.returnPressed.connect(self._emit)
        self.send.clicked.connect(self._emit)

    def _emit(self):
        txt = self.edit.text().strip()
        if not txt:
            return
        self.submitted.emit(txt)
        self.edit.clear()
